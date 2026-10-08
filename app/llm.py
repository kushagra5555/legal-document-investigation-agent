"""Gemini model construction behind a LangChain chat-model interface."""

from __future__ import annotations

import os
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic.v1 import Field, PrivateAttr, SecretStr


_OBSERVABILITY = {"llm_calls": 0, "input_chars": 0, "output_chars": 0}
_OBSERVABILITY_LOCK = threading.Lock()
_REQUEST_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "2"))))
_RATE_LOCK = threading.Lock()
_REQUEST_TIMES: list[float] = []
LOGGER = logging.getLogger("document_agent.llm")


def reset_observability() -> None:
    with _OBSERVABILITY_LOCK:
        for key in _OBSERVABILITY:
            _OBSERVABILITY[key] = 0


def get_observability() -> dict[str, int | None]:
    with _OBSERVABILITY_LOCK:
        snapshot = dict(_OBSERVABILITY)
    snapshot["approx_tokens"] = round((snapshot["input_chars"] + snapshot["output_chars"]) / 4)
    return snapshot


class ConfigurationError(RuntimeError):
    """Raised when required, non-secret configuration is missing."""


class DependencyError(RuntimeError):
    """Raised when the Gemini SDK cannot load."""


@dataclass
class LLMCallFailure(RuntimeError):
    """A redacted, structured failure from an external model call."""

    node: str
    attempt: int
    status: str
    error_type: str
    message: str
    http_status: int | None = None
    raw_snippet: str | None = None
    latency_ms: int = 0
    retry_after: float = 0.0

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "attempt": self.attempt,
            "status": self.status,
            "error_type": self.error_type,
            "http_status": self.http_status,
            "message": self.message,
            "raw_snippet": self.raw_snippet,
            "latency_ms": self.latency_ms,
            "retry_after": self.retry_after,
        }


def _config_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _config_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _http_status(exc: BaseException) -> int | None:
    for value in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _failure_status(exc: BaseException, http_status: int | None) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if http_status == 429 or "rate" in name or "rate limit" in message:
        return "rate_limited"
    if isinstance(exc, (TimeoutError, FutureTimeoutError)) or "timeout" in name or "timed out" in message:
        return "timeout"
    if http_status in {500, 502, 503, 504}:
        return "server_error"
    if "blocked" in message or "safety" in message:
        return "blocked"
    return "server_error" if http_status and http_status >= 500 else "error"


def _retry_after_seconds(exc: BaseException) -> float:
    headers = getattr(exc, "headers", None) or getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return max(0.0, float(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _wait_for_rate_limit() -> None:
    rpm = _config_int("LLM_MAX_REQUESTS_PER_MINUTE", 30, 0)
    if rpm <= 0:
        return
    while True:
        now = time.monotonic()
        with _RATE_LOCK:
            _REQUEST_TIMES[:] = [stamp for stamp in _REQUEST_TIMES if now - stamp < 60]
            if len(_REQUEST_TIMES) < rpm:
                _REQUEST_TIMES.append(now)
                return
            wait = max(0.01, 60 - (now - _REQUEST_TIMES[0]))
        time.sleep(wait)


def invoke_with_resilience(llm: Any, messages: list[Any], *, node: str, max_attempts: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Invoke a LangChain model with bounded retries and redacted diagnostics."""

    attempts = max_attempts or (_config_int("LLM_MAX_RETRIES", 2, 0) + 1)
    timeout = _config_float("LLM_REQUEST_TIMEOUT_SECONDS", 45.0, 0.1)
    base_delay = _config_float("LLM_BACKOFF_BASE_SECONDS", 0.5, 0.0)
    last_failure: LLMCallFailure | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            _wait_for_rate_limit()
            with _REQUEST_SEMAPHORE:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm.invoke, messages)
                    response = future.result(timeout=timeout)
            latency_ms = round((time.monotonic() - started) * 1000)
            content = str(getattr(response, "content", "") or "")
            if not content.strip():
                raise LLMCallFailure(node, attempt, "blocked", "EmptyResponse", "The model returned an empty or blocked response.", latency_ms=latency_ms)
            metadata = {"node": node, "attempt": attempt, "status": "ok", "latency_ms": latency_ms}
            LOGGER.info("llm_call node=%s attempt=%s status=ok latency_ms=%s", node, attempt, latency_ms)
            return response, metadata
        except LLMCallFailure as failure:
            failure.latency_ms = round((time.monotonic() - started) * 1000)
            last_failure = failure
        except FutureTimeoutError as exc:
            last_failure = LLMCallFailure(node, attempt, "timeout", type(exc).__name__, f"Model request exceeded {timeout:.1f}s.", latency_ms=round((time.monotonic() - started) * 1000))
        except Exception as exc:
            status_code = _http_status(exc)
            last_failure = LLMCallFailure(node, attempt, _failure_status(exc, status_code), type(exc).__name__, str(exc)[:500] or "Model request failed.", status_code, latency_ms=round((time.monotonic() - started) * 1000), retry_after=_retry_after_seconds(exc))
        LOGGER.error("llm_call node=%s attempt=%s status=%s type=%s http_status=%s message=%s", node, attempt, last_failure.status, last_failure.error_type, last_failure.http_status, last_failure.message)
        if last_failure.status not in {"rate_limited", "timeout", "server_error", "blocked"} or attempt >= attempts:
            break
        retry_after = last_failure.retry_after or _config_float("LLM_RETRY_AFTER_SECONDS", 0.0, 0.0)
        delay = retry_after or base_delay * (2 ** (attempt - 1)) + random.uniform(0, max(0.0, base_delay))
        time.sleep(delay)
    assert last_failure is not None
    raise last_failure


class GeminiChatModel(BaseChatModel):
    """LangChain adapter around Google's current Python Gemini SDK."""

    model: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    temperature: float = 0
    max_output_tokens: int = 4096
    _client: Any = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            from google import genai
        except Exception as exc:
            raise DependencyError("The Google GenAI SDK could not load.") from exc
        self._client = genai.Client(api_key=self.api_key.get_secret_value())

    @property
    def _llm_type(self) -> str:
        return "gemini_chat_model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        from google.genai import types

        system_parts = [str(message.content) for message in messages if message.type == "system"]
        conversation_parts = [
            f"{message.type.upper()}: {message.content}"
            for message in messages
            if message.type != "system"
        ]
        input_text = "\n\n".join(conversation_parts)
        response = self._client.models.generate_content(
            model=self.model,
            contents=input_text,
            config=types.GenerateContentConfig(
                system_instruction="\n\n".join(system_parts) or None,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        content = response.text
        if not content:
            raise RuntimeError("Gemini returned an empty response.")
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
        with _OBSERVABILITY_LOCK:
            _OBSERVABILITY["llm_calls"] += 1
            _OBSERVABILITY["input_chars"] += len(input_text) + sum(len(part) for part in system_parts)
            _OBSERVABILITY["output_chars"] += len(content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=content,
            response_metadata={"finish_reason": finish_reason} if finish_reason else {},
        ))])


def build_llm() -> GeminiChatModel:
    """Build the configured Gemini model without logging the secret."""

    load_dotenv()
    api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is missing. Add your Gemini key to the local .env file."
        )
    if not model:
        raise ConfigurationError(
            "GEMINI_MODEL is missing. Set it to an available Gemini model name."
        )

    max_output_tokens = _config_int("LLM_MAX_OUTPUT_TOKENS", 4096, 128)
    return GeminiChatModel(model=model, api_key=api_key, max_output_tokens=max_output_tokens)
