from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenTrackerError(Exception):
    """Base exception for token tracker failures."""


class TokenTrackerPersistenceError(TokenTrackerError):
    """Raised when usage data cannot be persisted."""


@dataclass
class UsageRecord:
    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    metadata: Dict[str, Any]


@dataclass
class UsageSummary:
    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float
    by_provider: Dict[str, Dict[str, float]]
    by_model: Dict[str, Dict[str, float]]
    last_updated: str


class TokenTracker:
    """
    Tracks LLM token usage and estimated cost, then persists it to JSON.

    File format:
    {
      "summary": {...},
      "records": [...]
    }
    """

    def __init__(self, usage_file: str = "data/usage.json") -> None:
        self._usage_path = Path(usage_file)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Ensure usage file exists with a valid initial structure."""
        async with self._lock:
            try:
                self._usage_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._usage_path.exists():
                    initial_payload = {
                        "summary": self._empty_summary_dict(),
                        "records": [],
                    }
                    self._write_json(initial_payload)
                    logger.info("Initialized usage file at %s", self._usage_path)
                else:
                    payload = self._read_json()
                    self._validate_payload(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.exception("Failed to initialize token tracker: %s", exc)
                raise TokenTrackerPersistenceError(
                    f"Failed to initialize token tracker: {exc}"
                ) from exc

    async def record_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Record one LLM call usage item and persist immediately.
        """
        if not provider.strip():
            raise ValueError("provider cannot be empty.")
        if not model.strip():
            raise ValueError("model cannot be empty.")
        if prompt_tokens < 0 or completion_tokens < 0 or total_tokens < 0:
            raise ValueError("Token counts cannot be negative.")
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative.")

        record = UsageRecord(
            timestamp=self._utc_now(),
            provider=provider.strip(),
            model=model.strip(),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            estimated_cost_usd=float(estimated_cost_usd),
            metadata=metadata or {},
        )

        async with self._lock:
            try:
                payload = self._safe_load_payload()
                payload["records"].append(asdict(record))
                payload["summary"] = self._recompute_summary(payload["records"])
                self._write_json(payload)
                return record
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.exception("Failed to record usage call: %s", exc)
                raise TokenTrackerPersistenceError(
                    f"Failed to record usage call: {exc}"
                ) from exc

    async def get_summary(self) -> UsageSummary:
        """
        Return current usage summary from persisted data.
        """
        async with self._lock:
            try:
                payload = self._safe_load_payload()
                summary_raw = payload["summary"]
                return UsageSummary(
                    total_calls=int(summary_raw["total_calls"]),
                    total_prompt_tokens=int(summary_raw["total_prompt_tokens"]),
                    total_completion_tokens=int(summary_raw["total_completion_tokens"]),
                    total_tokens=int(summary_raw["total_tokens"]),
                    total_estimated_cost_usd=float(
                        summary_raw["total_estimated_cost_usd"]
                    ),
                    by_provider=dict(summary_raw["by_provider"]),
                    by_model=dict(summary_raw["by_model"]),
                    last_updated=str(summary_raw["last_updated"]),
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.exception("Failed to read usage summary: %s", exc)
                raise TokenTrackerPersistenceError(
                    f"Failed to read usage summary: {exc}"
                ) from exc

    async def get_recent_records(self, limit: int = 20) -> List[UsageRecord]:
        """
        Return most recent usage records, newest first.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        async with self._lock:
            try:
                payload = self._safe_load_payload()
                records = payload["records"][-limit:]
                records.reverse()
                return [
                    UsageRecord(
                        timestamp=str(item["timestamp"]),
                        provider=str(item["provider"]),
                        model=str(item["model"]),
                        prompt_tokens=int(item["prompt_tokens"]),
                        completion_tokens=int(item["completion_tokens"]),
                        total_tokens=int(item["total_tokens"]),
                        estimated_cost_usd=float(item["estimated_cost_usd"]),
                        metadata=dict(item.get("metadata", {})),
                    )
                    for item in records
                ]
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.exception("Failed to read recent usage records: %s", exc)
                raise TokenTrackerPersistenceError(
                    f"Failed to read recent usage records: {exc}"
                ) from exc

    async def reset(self) -> None:
        """
        Clear all tracked usage and persist empty state.
        """
        async with self._lock:
            try:
                payload = {
                    "summary": self._empty_summary_dict(),
                    "records": [],
                }
                self._write_json(payload)
                logger.info("Usage tracker reset completed at %s", self._usage_path)
            except OSError as exc:
                logger.exception("Failed to reset usage tracker: %s", exc)
                raise TokenTrackerPersistenceError(
                    f"Failed to reset usage tracker: {exc}"
                ) from exc

    def _safe_load_payload(self) -> Dict[str, Any]:
        if not self._usage_path.exists():
            # Recover by creating the file on-demand if initialize() was skipped.
            payload = {"summary": self._empty_summary_dict(), "records": []}
            self._usage_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(payload)
            return payload

        payload = self._read_json()
        self._validate_payload(payload)
        return payload

    def _recompute_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_calls = len(records)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        by_provider: Dict[str, Dict[str, float]] = {}
        by_model: Dict[str, Dict[str, float]] = {}

        for item in records:
            prompt = int(item["prompt_tokens"])
            completion = int(item["completion_tokens"])
            total = int(item["total_tokens"])
            cost = float(item["estimated_cost_usd"])
            provider = str(item["provider"])
            model = str(item["model"])

            total_prompt_tokens += prompt
            total_completion_tokens += completion
            total_tokens += total
            total_cost += cost

            if provider not in by_provider:
                by_provider[provider] = {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0}
            by_provider[provider]["calls"] += 1.0
            by_provider[provider]["tokens"] += float(total)
            by_provider[provider]["cost_usd"] += float(cost)

            if model not in by_model:
                by_model[model] = {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0}
            by_model[model]["calls"] += 1.0
            by_model[model]["tokens"] += float(total)
            by_model[model]["cost_usd"] += float(cost)

        # Round only for display/persistence stability.
        for bucket in by_provider.values():
            bucket["calls"] = int(bucket["calls"])
            bucket["tokens"] = int(bucket["tokens"])
            bucket["cost_usd"] = round(float(bucket["cost_usd"]), 8)

        for bucket in by_model.values():
            bucket["calls"] = int(bucket["calls"])
            bucket["tokens"] = int(bucket["tokens"])
            bucket["cost_usd"] = round(float(bucket["cost_usd"]), 8)

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": round(total_cost, 8),
            "by_provider": by_provider,
            "by_model": by_model,
            "last_updated": self._utc_now(),
        }

    def _empty_summary_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_estimated_cost_usd": 0.0,
            "by_provider": {},
            "by_model": {},
            "last_updated": self._utc_now(),
        }

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Usage payload must be a JSON object.")
        if "summary" not in payload or "records" not in payload:
            raise ValueError("Usage payload must contain 'summary' and 'records'.")
        if not isinstance(payload["records"], list):
            raise ValueError("'records' must be a list.")
        if not isinstance(payload["summary"], dict):
            raise ValueError("'summary' must be an object.")

    def _read_json(self) -> Dict[str, Any]:
        with self._usage_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            if not isinstance(data, dict):
                raise ValueError("Usage file root must be an object.")
            return data

    def _write_json(self, payload: Dict[str, Any]) -> None:
        temp_path = self._usage_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=True, indent=2)
        temp_path.replace(self._usage_path)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
