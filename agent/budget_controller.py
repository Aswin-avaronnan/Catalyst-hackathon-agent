from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BudgetControllerError(Exception):
    """Base exception for budget controller failures."""


class BudgetExceededError(BudgetControllerError):
    """Raised when a new spend would exceed configured budget."""


class BudgetPersistenceError(BudgetControllerError):
    """Raised when budget state cannot be loaded or saved."""


@dataclass
class BudgetStatus:
    budget_limit_usd: float
    spent_usd: float
    remaining_usd: float
    usage_ratio: float
    last_updated: str


class BudgetController:
    """
    Enforces budget limits and persists spend state to JSON.

    File format:
    {
      "budget_limit_usd": 15.0,
      "spent_usd": 0.0,
      "last_updated": "..."
    }
    """

    def __init__(
        self,
        budget_limit_usd: float = 15.0,
        state_file: str = "data/budget_state.json",
    ) -> None:
        if budget_limit_usd <= 0:
            raise ValueError("budget_limit_usd must be greater than zero.")

        self._budget_limit_usd = float(budget_limit_usd)
        self._state_path = Path(state_file)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Ensure budget state file exists and is valid."""
        async with self._lock:
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._state_path.exists():
                    payload = self._default_state()
                    self._write_json(payload)
                    logger.info("Initialized budget state at %s", self._state_path)
                else:
                    payload = self._read_json()
                    self._validate_payload(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.exception("Failed to initialize budget controller: %s", exc)
                raise BudgetPersistenceError(
                    f"Failed to initialize budget controller: {exc}"
                ) from exc

    async def ensure_can_spend(self, estimated_cost_usd: float) -> None:
        """
        Raise BudgetExceededError if spend would exceed budget.
        """
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative.")

        async with self._lock:
            try:
                state = self._safe_load_state()
                projected = float(state["spent_usd"]) + float(estimated_cost_usd)
                if projected > self._budget_limit_usd:
                    remaining = max(self._budget_limit_usd - float(state["spent_usd"]), 0.0)
                    message = (
                        f"Budget exceeded: attempted additional ${estimated_cost_usd:.6f}, "
                        f"remaining ${remaining:.6f}, limit ${self._budget_limit_usd:.2f}."
                    )
                    raise BudgetExceededError(message)
            except BudgetExceededError:
                raise
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.exception("Failed budget pre-check: %s", exc)
                raise BudgetPersistenceError(f"Failed budget pre-check: {exc}") from exc

    async def register_spend(self, actual_cost_usd: float) -> BudgetStatus:
        """
        Register actual spend after a successful LLM call.
        """
        if actual_cost_usd < 0:
            raise ValueError("actual_cost_usd cannot be negative.")

        async with self._lock:
            try:
                state = self._safe_load_state()
                new_spent = float(state["spent_usd"]) + float(actual_cost_usd)
                if new_spent > self._budget_limit_usd:
                    raise BudgetExceededError(
                        f"Cannot register spend of ${actual_cost_usd:.6f}; "
                        f"would exceed limit ${self._budget_limit_usd:.2f}."
                    )

                updated = {
                    "budget_limit_usd": self._budget_limit_usd,
                    "spent_usd": round(new_spent, 8),
                    "last_updated": self._utc_now(),
                }
                self._write_json(updated)
                return self._to_status(updated)
            except BudgetExceededError:
                raise
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.exception("Failed to register spend: %s", exc)
                raise BudgetPersistenceError(
                    f"Failed to register spend: {exc}"
                ) from exc

    async def get_status(self) -> BudgetStatus:
        """
        Return current budget status from persisted state.
        """
        async with self._lock:
            try:
                state = self._safe_load_state()
                return self._to_status(state)
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.exception("Failed to get budget status: %s", exc)
                raise BudgetPersistenceError(
                    f"Failed to get budget status: {exc}"
                ) from exc

    async def reset(self) -> BudgetStatus:
        """
        Reset spent amount back to zero while keeping budget limit.
        """
        async with self._lock:
            try:
                payload = self._default_state()
                self._write_json(payload)
                logger.info("Budget state reset at %s", self._state_path)
                return self._to_status(payload)
            except OSError as exc:
                logger.exception("Failed to reset budget state: %s", exc)
                raise BudgetPersistenceError(
                    f"Failed to reset budget state: {exc}"
                ) from exc

    def _safe_load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            payload = self._default_state()
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(payload)
            return payload

        payload = self._read_json()
        self._validate_payload(payload)
        return payload

    def _to_status(self, state: Dict[str, Any]) -> BudgetStatus:
        spent = float(state["spent_usd"])
        limit = float(state["budget_limit_usd"])
        remaining = max(limit - spent, 0.0)
        ratio = 0.0 if limit <= 0 else min(spent / limit, 1.0)
        return BudgetStatus(
            budget_limit_usd=round(limit, 8),
            spent_usd=round(spent, 8),
            remaining_usd=round(remaining, 8),
            usage_ratio=round(ratio, 8),
            last_updated=str(state["last_updated"]),
        )

    def _default_state(self) -> Dict[str, Any]:
        return {
            "budget_limit_usd": self._budget_limit_usd,
            "spent_usd": 0.0,
            "last_updated": self._utc_now(),
        }

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Budget state must be a JSON object.")
        required = {"budget_limit_usd", "spent_usd", "last_updated"}
        if not required.issubset(payload.keys()):
            raise ValueError(
                "Budget state must contain budget_limit_usd, spent_usd, last_updated."
            )

        limit = float(payload["budget_limit_usd"])
        spent = float(payload["spent_usd"])
        if limit <= 0:
            raise ValueError("budget_limit_usd must be greater than zero.")
        if spent < 0:
            raise ValueError("spent_usd cannot be negative.")

    def _read_json(self) -> Dict[str, Any]:
        with self._state_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            if not isinstance(data, dict):
                raise ValueError("Budget state root must be an object.")
            return data

    def _write_json(self, payload: Dict[str, Any]) -> None:
        temp_path = self._state_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=True, indent=2)
        temp_path.replace(self._state_path)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
