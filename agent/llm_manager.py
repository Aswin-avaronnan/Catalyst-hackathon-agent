from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import httpx
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM-related errors."""


class LLMConfigurationError(LLMError):
    """Raised when provider configuration is invalid."""


class LLMRequestError(LLMError):
    """Raised when all provider requests fail."""


class BudgetExceededError(LLMError):
    """Raised when budget checks fail before an LLM call."""


class TokenTrackerProtocol(Protocol):
    async def record_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist token/cost usage for an LLM call."""


class BudgetControllerProtocol(Protocol):
    async def ensure_can_spend(self, estimated_cost_usd: float) -> None:
        """Raise if estimated spend would exceed budget."""

    async def register_spend(self, actual_cost_usd: float) -> None:
        """Register actual spend after successful call."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    raw_response: Dict[str, Any]


class LLMManager:
    """
    Multi-provider LLM manager with fallback routing:
    OpenRouter (primary) -> Anthropic (fallback) -> OpenAI (final fallback).

    Supports dry-run mode for offline/non-billable testing.
    """

    def __init__(
        self,
        token_tracker: Optional[TokenTrackerProtocol] = None,
        budget_controller: Optional[BudgetControllerProtocol] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        # Load .env values when running scripts directly.
        load_dotenv()

        self._token_tracker = token_tracker
        self._budget_controller = budget_controller

        self._openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self._openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

        self._dry_run = os.getenv("LLM_DRY_RUN", "false").strip().lower() == "true"

        self._openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self._gemini_base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")

        self._simple_model = os.getenv(
            "OPENROUTER_SIMPLE_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
        )
        self._gemini_simple_model = os.getenv(
            "GEMINI_SIMPLE_MODEL", "gemini-2.5-flash"
        )
        self._gemini_medium_model = os.getenv(
            "GEMINI_MEDIUM_MODEL", "gemini-2.5-flash"
        )
        self._gemini_complex_model = os.getenv(
            "GEMINI_COMPLEX_MODEL", "gemini-2.5-flash"
        )
        self._openrouter_complex_model = os.getenv(
            "OPENROUTER_COMPLEX_MODEL", "anthropic/claude-3.5-sonnet"
        )
        self._anthropic_complex_model = os.getenv(
            "ANTHROPIC_COMPLEX_MODEL", "claude-3-5-sonnet-20240620"
        )
        self._openai_complex_model = os.getenv("OPENAI_COMPLEX_MODEL", "gpt-4o-mini")

        self._provider_order = self._resolve_provider_order()

        # In dry-run mode, allow zero keys so development can proceed.
        if not self._dry_run and not self._provider_order:
            raise LLMConfigurationError(
                "No LLM API key configured. Set OPENROUTER_API_KEY at minimum, "
                "or enable LLM_DRY_RUN=true."
            )

        # Estimated costs per 1K tokens (input/output). Env-overridable.
        self._pricing_per_1k: Dict[str, Dict[str, float]] = {
            self._simple_model: {
                "input": float(os.getenv("PRICE_SIMPLE_INPUT", "0.0")),
                "output": float(os.getenv("PRICE_SIMPLE_OUTPUT", "0.0")),
            },
            # Gemini 2.5 Flash is currently configured as free.
            self._gemini_simple_model: {
                "input": 0.0,
                "output": 0.0,
            },
            self._gemini_medium_model: {
                "input": 0.0,
                "output": 0.0,
            },
            self._gemini_complex_model: {
                "input": 0.0,
                "output": 0.0,
            },
            self._openrouter_complex_model: {
                "input": float(os.getenv("PRICE_OPENROUTER_COMPLEX_INPUT", "0.003")),
                "output": float(os.getenv("PRICE_OPENROUTER_COMPLEX_OUTPUT", "0.015")),
            },
            self._anthropic_complex_model: {
                "input": float(os.getenv("PRICE_ANTHROPIC_COMPLEX_INPUT", "0.003")),
                "output": float(os.getenv("PRICE_ANTHROPIC_COMPLEX_OUTPUT", "0.015")),
            },
            self._openai_complex_model: {
                "input": float(os.getenv("PRICE_OPENAI_COMPLEX_INPUT", "0.00015")),
                "output": float(os.getenv("PRICE_OPENAI_COMPLEX_OUTPUT", "0.0006")),
            },
            "__dry_run_simple__": {
                "input": float(os.getenv("PRICE_DRY_RUN_SIMPLE_INPUT", "0.0")),
                "output": float(os.getenv("PRICE_DRY_RUN_SIMPLE_OUTPUT", "0.0")),
            },
            "__dry_run_complex__": {
                "input": float(os.getenv("PRICE_DRY_RUN_COMPLEX_INPUT", "0.0001")),
                "output": float(os.getenv("PRICE_DRY_RUN_COMPLEX_OUTPUT", "0.0002")),
            },
        }

        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        """Close underlying HTTP client."""
        try:
            await self._client.aclose()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.exception("Failed to close HTTP client: %s", exc)

    async def __aenter__(self) -> "LLMManager":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a precise recruitment assistant.",
        complexity: str = "simple",
        temperature: float = 0.2,
        max_tokens: int = 700,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate response with fallback across providers.
        complexity: simple | medium | complex
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        normalized_complexity = complexity.strip().lower()
        if normalized_complexity not in {"simple", "medium", "complex"}:
            raise ValueError("complexity must be one of: simple, medium, complex.")

        if self._dry_run:
            response = await self._generate_dry_run_response(
                prompt=prompt,
                complexity=normalized_complexity,
                max_tokens=max_tokens,
                metadata=metadata or {},
            )
            return response

        provider_attempts = self._build_attempt_plan(normalized_complexity)
        if not provider_attempts:
            raise LLMConfigurationError(
                "No providers available for the requested complexity."
            )

        last_error: Optional[Exception] = None
        for provider_name, model_name in provider_attempts:
            try:
                estimated_pre_call_cost = self._estimate_cost(model_name, 500, max_tokens)
                await self._ensure_budget(estimated_pre_call_cost)

                response = await self._retry_provider_call(
                    provider=provider_name,
                    model=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                await self._post_success_accounting(
                    response=response,
                    metadata=metadata or {},
                )
                return response
            except BudgetExceededError:
                raise
            except (RetryError, httpx.HTTPError, LLMError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Provider attempt failed provider=%s model=%s error=%s",
                    provider_name,
                    model_name,
                    str(exc),
                )
                continue

        raise LLMRequestError(
            f"All provider attempts failed. Last error: {last_error}"
        ) from last_error

    async def _generate_dry_run_response(
        self,
        prompt: str,
        complexity: str,
        max_tokens: int,
        metadata: Dict[str, Any],
    ) -> LLMResponse:
        try:
            prompt_tokens = self._estimate_prompt_tokens(prompt)
            completion_tokens = min(
                max_tokens,
                120 if complexity == "simple" else 220,
            )
            total_tokens = prompt_tokens + completion_tokens
            pricing_model = (
                "__dry_run_simple__" if complexity == "simple" else "__dry_run_complex__"
            )
            estimated_cost = self._estimate_cost(
                pricing_model,
                prompt_tokens,
                completion_tokens,
            )

            await self._ensure_budget(estimated_cost)

            response = LLMResponse(
                text=self._build_dry_run_text(prompt=prompt, complexity=complexity),
                provider="dry_run",
                model=pricing_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
                raw_response={
                    "mode": "dry_run",
                    "complexity": complexity,
                    "metadata": metadata,
                },
            )

            await self._post_success_accounting(response=response, metadata=metadata)
            return response
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.exception("Dry-run generation failed: %s", exc)
            raise LLMRequestError(f"Dry-run generation failed: {exc}") from exc

    def _build_dry_run_text(self, prompt: str, complexity: str) -> str:
        preview = prompt.strip().replace("\n", " ")
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        return (
            f"[DRY-RUN:{complexity}] Simulated LLM output for prompt: {preview}. "
            "No external API call was made."
        )

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        # Approximation: ~4 chars/token, minimum token floor for short prompts.
        estimated = max(20, len(prompt) // 4)
        return estimated

    def _resolve_provider_order(self) -> list[str]:
        order: list[str] = []
        if self._gemini_api_key:
            order.append("gemini")
        if self._openrouter_api_key:
            order.append("openrouter")
        if self._anthropic_api_key:
            order.append("anthropic")
        if self._openai_api_key:
            order.append("openai")
        return order

    def _build_attempt_plan(self, complexity: str) -> list[tuple[str, str]]:
        plan: list[tuple[str, str]] = []
        for provider in self._provider_order:
            if complexity == "simple":
                if provider == "gemini":
                    plan.append((provider, self._gemini_simple_model))
                if provider == "openrouter":
                    plan.append((provider, self._simple_model))
            elif complexity == "medium":
                if provider == "gemini":
                    plan.append((provider, self._gemini_medium_model))
                elif provider == "openrouter":
                    plan.append((provider, self._openrouter_complex_model))
                elif provider == "anthropic":
                    plan.append((provider, self._anthropic_complex_model))
                elif provider == "openai":
                    plan.append((provider, self._openai_complex_model))
            else:
                if provider == "gemini":
                    plan.append((provider, self._gemini_complex_model))
                elif provider == "openrouter":
                    plan.append((provider, self._openrouter_complex_model))
                elif provider == "anthropic":
                    plan.append((provider, self._anthropic_complex_model))
                elif provider == "openai":
                    plan.append((provider, self._openai_complex_model))

        if complexity == "simple" and not plan:
            for provider in self._provider_order:
                if provider == "gemini":
                    plan.append((provider, self._gemini_simple_model))
                elif provider == "anthropic":
                    plan.append((provider, self._anthropic_complex_model))
                elif provider == "openai":
                    plan.append((provider, self._openai_complex_model))
        return plan

    async def _retry_provider_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                retry=retry_if_exception_type((httpx.HTTPError, LLMRequestError)),
                reraise=True,
            ):
                with attempt:
                    return await self._call_provider(
                        provider=provider,
                        model=model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
        except Exception as exc:
            raise LLMRequestError(
                f"Provider call failed after retries provider={provider} model={model}: {exc}"
            ) from exc

    async def _call_provider(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if provider == "gemini":
            return await self._call_gemini(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openrouter":
            return await self._call_openrouter(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "anthropic":
            return await self._call_anthropic(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openai":
            return await self._call_openai(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise LLMConfigurationError(f"Unknown provider '{provider}'.")

    async def _call_gemini(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._gemini_api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not configured.")

        # Gemini 2.5 Flash is used for all complexity levels.
        gemini_model = "gemini-2.5-flash"
        url = f"{self._gemini_base_url}/models/{gemini_model}:generateContent"
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        headers = {"Content-Type": "application/json"}
        params = {"key": self._gemini_api_key}

        response = await self._client.post(
            url,
            json=payload,
            headers=headers,
            params=params,
        )
        if response.status_code >= 400:
            raise LLMRequestError(
                f"Gemini request failed status={response.status_code} body={response.text}"
            )

        data = response.json()
        text = self._extract_gemini_text(data)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(data)

        # Gemini 2.5 Flash is currently treated as free.
        cost = 0.0

        return LLMResponse(
            text=text,
            provider="gemini",
            model=gemini_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            raw_response=data,
        )

    async def _call_openrouter(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._openrouter_api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is not configured.")

        url = f"{self._openrouter_base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise LLMRequestError(
                f"OpenRouter request failed status={response.status_code} body={response.text}"
            )

        data = response.json()
        text = self._extract_chat_text(data)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(data)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            provider="openrouter",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            raw_response=data,
        )

    async def _call_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._anthropic_api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is not configured.")

        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "x-api-key": self._anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = await self._client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise LLMRequestError(
                f"Anthropic request failed status={response.status_code} body={response.text}"
            )

        data = response.json()
        text = self._extract_anthropic_text(data)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(data)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            provider="anthropic",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            raw_response=data,
        )

    async def _call_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._openai_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not configured.")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._openai_api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise LLMRequestError(
                f"OpenAI request failed status={response.status_code} body={response.text}"
            )

        data = response.json()
        text = self._extract_chat_text(data)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(data)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            provider="openai",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            raw_response=data,
        )

    async def _ensure_budget(self, estimated_cost_usd: float) -> None:
        if self._budget_controller is None:
            return
        try:
            await self._budget_controller.ensure_can_spend(estimated_cost_usd)
        except Exception as exc:
            logger.error("Budget check failed: %s", exc)
            raise BudgetExceededError(str(exc)) from exc

    async def _post_success_accounting(
        self, response: LLMResponse, metadata: Dict[str, Any]
    ) -> None:
        if self._token_tracker is not None:
            try:
                await self._token_tracker.record_call(
                    provider=response.provider,
                    model=response.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    estimated_cost_usd=response.estimated_cost_usd,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.exception("Token tracking failed: %s", exc)

        if self._budget_controller is not None:
            try:
                await self._budget_controller.register_spend(
                    response.estimated_cost_usd
                )
            except Exception as exc:
                logger.exception("Budget spend registration failed: %s", exc)

    def _extract_chat_text(self, payload: Dict[str, Any]) -> str:
        try:
            choices = payload.get("choices", [])
            if not choices:
                raise ValueError("No choices in response.")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [
                    chunk.get("text", "")
                    for chunk in content
                    if isinstance(chunk, dict)
                ]
                return " ".join(t.strip() for t in texts if t).strip()
            return str(content).strip()
        except Exception as exc:
            raise LLMRequestError(f"Failed to parse chat text: {exc}") from exc

    def _extract_anthropic_text(self, payload: Dict[str, Any]) -> str:
        try:
            content = payload.get("content", [])
            text_blocks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_blocks.append(str(item.get("text", "")).strip())
            text = " ".join(part for part in text_blocks if part).strip()
            if not text:
                raise ValueError("Anthropic response missing text content.")
            return text
        except Exception as exc:
            raise LLMRequestError(f"Failed to parse Anthropic text: {exc}") from exc

    def _extract_gemini_text(self, payload: Dict[str, Any]) -> str:
        try:
            candidates = payload.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini response has no candidates.")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            texts: list[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part["text"]).strip())
            text = " ".join(chunk for chunk in texts if chunk).strip()
            if not text:
                raise ValueError("Gemini response missing text output.")
            return text
        except Exception as exc:
            raise LLMRequestError(f"Failed to parse Gemini text: {exc}") from exc

    def _extract_usage(self, payload: Dict[str, Any]) -> tuple[int, int, int]:
        usage = payload.get("usage", payload.get("usageMetadata", {}))
        prompt_tokens = int(
            usage.get(
                "prompt_tokens",
                usage.get("input_tokens", usage.get("promptTokenCount", 0)),
            )
            or 0
        )
        completion_tokens = int(
            usage.get(
                "completion_tokens",
                usage.get("output_tokens", usage.get("candidatesTokenCount", 0)),
            )
            or 0
        )
        total_tokens = int(
            usage.get(
                "total_tokens",
                usage.get("totalTokenCount", prompt_tokens + completion_tokens),
            )
        )
        return prompt_tokens, completion_tokens, total_tokens

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        pricing = self._pricing_per_1k.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (prompt_tokens / 1000.0) * float(pricing["input"])
        output_cost = (completion_tokens / 1000.0) * float(pricing["output"])
        return round(input_cost + output_cost, 8)

    def debug_provider_config(self) -> str:
        """
        Non-sensitive diagnostics for health/stats endpoints.
        Does not expose keys.
        """
        payload = {
            "dry_run": self._dry_run,
            "available_providers": self._provider_order,
            "simple_model": self._simple_model,
            "gemini_simple_model": self._gemini_simple_model,
            "gemini_medium_model": self._gemini_medium_model,
            "gemini_complex_model": self._gemini_complex_model,
            "openrouter_complex_model": self._openrouter_complex_model,
            "anthropic_complex_model": self._anthropic_complex_model,
            "openai_complex_model": self._openai_complex_model,
        }
        return json.dumps(payload, indent=2)
