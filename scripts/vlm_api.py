#!/usr/bin/env python3
"""VLM API helper for OpenAI-compatible and Anthropic Messages transports."""

from __future__ import annotations

import base64
from email.utils import parsedate_to_datetime
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_TRANSPORT_ATTEMPTS = 3
TRANSPORT_POLICY_VERSION = "generic_three_attempts_v1"
JSON_PARSE_POLICY_VERSION = "first_complete_json_object_v1"


@dataclass
class VLMConfig:
    api_key: str
    base_url: str
    model: str
    api_format: str = "chat_completions"
    provider: str = ""
    auth_mode: str = "bearer"
    anthropic_version: str = "2023-06-01"
    thinking_type: str = ""
    reasoning_effort: str = ""
    disable_response_storage: bool = False
    temperature: float | None = 0.0
    top_p: float | None = None
    top_k: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 2.0
    min_request_interval_sec: float = 0.0
    max_tokens: int = 1200
    config_path: str = ""


class VLMRetryExhaustedError(RuntimeError):
    def __init__(
        self,
        *,
        request_id: str,
        attempts: int,
        last_error: Exception | None,
        request_attempts: list[dict[str, Any]],
    ):
        self.request_id = request_id
        self.attempts = attempts
        self.last_error = last_error
        self.request_attempts = request_attempts
        message = (
            f"VLM transport failed after {attempts} attempts: "
            f"{request_id or 'request'}"
        )
        if last_error is not None:
            message += f"; last_error={type(last_error).__name__}: {last_error}"
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempts": self.attempts,
            "last_error_type": type(self.last_error).__name__ if self.last_error is not None else "",
            "last_error": str(self.last_error) if self.last_error is not None else "",
            "request_attempts": self.request_attempts,
        }


class VLMFormatError(RuntimeError):
    """A successful provider response that does not contain a usable JSON action."""

    def __init__(
        self,
        *,
        request_id: str,
        raw_response: str,
        reasoning_content: str,
        usage: dict[str, Any],
        attempts: int,
        request_attempts: list[dict[str, Any]],
        error: Exception,
    ):
        self.request_id = request_id
        self.raw_response = raw_response
        self.reasoning_content = reasoning_content
        self.usage = usage
        self.attempts = attempts
        self.request_attempts = request_attempts
        self.error = error
        super().__init__(
            f"planner response is not a valid JSON action: "
            f"{type(error).__name__}: {error}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempts": self.attempts,
            "error_type": type(self.error).__name__,
            "error": str(self.error),
            "request_attempts": self.request_attempts,
        }


class VLMProviderError(RuntimeError):
    """A non-retryable provider or request error."""

    def __init__(
        self,
        *,
        request_id: str,
        message: str,
        attempts: int,
        request_attempts: list[dict[str, Any]],
        status_code: int | None = None,
        response_body: str = "",
    ):
        self.request_id = request_id
        self.attempts = attempts
        self.request_attempts = request_attempts
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "response_body": self.response_body,
            "request_attempts": self.request_attempts,
        }


class VLMContentFilterError(VLMProviderError):
    """A structured provider response reports that the request was filtered."""


_REQUEST_SLOT_LOCK = threading.Lock()
_NEXT_REQUEST_AT: dict[str, float] = {}


def load_api_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_vlm_config(
    config_path: Path,
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    temperature: float | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    retry_backoff: float | None = None,
    max_tokens: int | None = None,
) -> VLMConfig:
    file_config = load_api_config(config_path)
    api_format = str(
        file_config.get(
            "api_format",
            os.environ.get("OPENAI_API_FORMAT", "chat_completions"),
        )
        or "chat_completions"
    )
    is_anthropic = api_format.strip().lower() in {
        "anthropic",
        "anthropic_messages",
        "messages",
    }
    thinking_config = file_config.get("thinking", {})
    if not isinstance(thinking_config, dict):
        raise TypeError("thinking must be a JSON object or null")
    output_config = file_config.get("output_config", {})
    if not isinstance(output_config, dict):
        raise TypeError("output_config must be a JSON object or null")
    default_temperature = None if is_anthropic else 0.0
    file_temperature = file_config.get("temperature", default_temperature)
    file_top_p = file_config.get("top_p")
    file_top_k = file_config.get("top_k")
    chat_template_kwargs = file_config.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise TypeError("chat_template_kwargs must be a JSON object or null")
    api_key_env = "CLAUDE_API_KEY" if is_anthropic else "OPENAI_API_KEY"
    base_url_env = "CLAUDE_BASE_URL" if is_anthropic else "OPENAI_BASE_URL"
    model_env = "CLAUDE_MODEL" if is_anthropic else "OPENAI_MODEL"
    resolved = VLMConfig(
        api_key=api_key or file_config.get("api_key", "") or os.environ.get(api_key_env, ""),
        base_url=(base_url or file_config.get("base_url", "") or os.environ.get(base_url_env, "")).rstrip("/"),
        model=model or file_config.get("model", "") or os.environ.get(model_env, ""),
        api_format=api_format,
        provider=str(file_config.get("provider", os.environ.get("OPENAI_PROVIDER", "")) or ""),
        auth_mode=str(
            file_config.get("auth_mode", os.environ.get("CLAUDE_AUTH_MODE", "bearer"))
            or "bearer"
        ),
        anthropic_version=str(
            file_config.get(
                "anthropic_version",
                os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            )
            or "2023-06-01"
        ),
        thinking_type=str(
            thinking_config.get("type", "")
            or file_config.get("thinking_type", "")
            or ""
        ),
        reasoning_effort=str(
            output_config.get("effort", "")
            or file_config.get("reasoning_effort", "")
            or file_config.get(
                "model_reasoning_effort",
                os.environ.get("OPENAI_REASONING_EFFORT", ""),
            )
            or ""
        ),
        disable_response_storage=bool(file_config.get("disable_response_storage", False)),
        temperature=(
            temperature
            if temperature is not None
            else (None if file_temperature is None else float(file_temperature))
        ),
        top_p=None if file_top_p is None else float(file_top_p),
        top_k=None if file_top_k is None else int(file_top_k),
        chat_template_kwargs=chat_template_kwargs,
        timeout=timeout if timeout is not None else float(file_config.get("timeout", file_config.get("timeout_sec", 60.0))),
        max_retries=max_retries if max_retries is not None else int(file_config.get("max_retries", 2)),
        retry_backoff=retry_backoff if retry_backoff is not None else float(file_config.get("retry_backoff", 2.0)),
        min_request_interval_sec=float(
            file_config.get(
                "min_request_interval_sec",
                os.environ.get("VLM_MIN_REQUEST_INTERVAL_SEC", 0.0),
            )
            or 0.0
        ),
        max_tokens=max_tokens if max_tokens is not None else int(file_config.get("max_tokens", 1200)),
        config_path=str(config_path),
    )
    return resolved


def require_complete_config(config: VLMConfig) -> None:
    missing = []
    if not config.api_key:
        missing.append("api_key")
    if not config.base_url:
        missing.append("base_url")
    if not config.model:
        missing.append("model")
    if missing:
        is_anthropic = config.api_format.strip().lower() in {
            "anthropic",
            "anthropic_messages",
            "messages",
        }
        config_hint = (
            "Fill the Anthropic config or set CLAUDE_API_KEY/CLAUDE_BASE_URL/CLAUDE_MODEL."
            if is_anthropic
            else "Fill the Responses config or set OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL."
        )
        raise RuntimeError(
            "Missing VLM config fields: "
            + ", ".join(missing)
            + ". "
            + config_hint
        )
    if config.min_request_interval_sec < 0:
        raise ValueError("min_request_interval_sec must be >= 0")
    if config.timeout <= 0:
        raise ValueError("timeout must be > 0")
    if config.max_retries < 0 or config.max_retries >= MAX_TRANSPORT_ATTEMPTS:
        raise ValueError(
            f"max_retries must be between 0 and {MAX_TRANSPORT_ATTEMPTS - 1}; "
            f"the public transport policy allows at most {MAX_TRANSPORT_ATTEMPTS} "
            "physical attempts"
        )
    if config.retry_backoff < 0:
        raise ValueError("retry_backoff must be >= 0")


def wait_for_request_slot(config: VLMConfig, request_id: str = "") -> float:
    """Apply an optional process-wide transport throttle without consuming a step."""
    interval = float(config.min_request_interval_sec)
    if interval <= 0:
        return 0.0
    slot_key = f"{config.base_url}\n{config.model}"
    with _REQUEST_SLOT_LOCK:
        now = time.monotonic()
        request_at = max(now, _NEXT_REQUEST_AT.get(slot_key, now))
        _NEXT_REQUEST_AT[slot_key] = request_at + interval
    wait_sec = max(0.0, request_at - time.monotonic())
    if wait_sec > 0:
        prefix = f"[rate_limit] {request_id}: " if request_id else "[rate_limit] "
        print(
            f"{prefix}sleeping {wait_sec:.3f}s before the next provider request "
            f"(minimum interval {interval:.3f}s)"
        )
        time.sleep(wait_sec)
    return wait_sec


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    max_side_text = os.environ.get("VLM_IMAGE_MAX_SIDE", "").strip()
    max_side = int(max_side_text) if max_side_text.isdigit() and int(max_side_text) > 0 else 0
    if max_side > 0:
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                scale = min(1.0, float(max_side) / float(max(width, height)))
                if scale < 1.0:
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                data = base64.b64encode(buffer.getvalue()).decode("ascii")
                return f"data:image/png;base64,{data}"
        except Exception:
            pass

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    if start < 0:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("top-level JSON value is not an object", text, 0)
        return parsed
    decoder = json.JSONDecoder()
    parsed, end = decoder.raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", text, start)
    trailing = text[start + end :].strip()
    if trailing:
        parsed["_extra_json_ignored"] = trailing[:1000]
    return parsed


def is_retryable_http_response(response: requests.Response) -> bool:
    return response.status_code in RETRYABLE_HTTP_STATUS


def contains_content_filter_marker(text: str) -> bool:
    lowered = (text or "").lower()
    return "content_filter" in lowered or "content filtering policy" in lowered


def retry_after_seconds(response: requests.Response) -> float | None:
    value = str(response.headers.get("Retry-After", "")).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                return None
            return max(0.0, target.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def response_reports_content_filter(data: Any) -> bool:
    """Recognize standard structured safety signals without provider-specific text."""
    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {"finish_reason", "stop_reason", "code", "type"}:
                normalized_value = str(value).strip().lower()
                if normalized_value in {
                    "blocked",
                    "content_filter",
                    "content_filtered",
                    "refusal",
                    "safety",
                }:
                    return True
            if response_reports_content_filter(value):
                return True
    elif isinstance(data, list):
        return any(response_reports_content_filter(value) for value in data)
    return False


def provider_error_envelope(data: Any) -> str:
    """Return a compact top-level provider error without inspecting model text."""
    if not isinstance(data, dict) or not data.get("error"):
        return ""
    error = data["error"]
    if isinstance(error, str):
        return error[:1000]
    try:
        return json.dumps(error, ensure_ascii=False, sort_keys=True)[:1000]
    except (TypeError, ValueError):
        return str(error)[:1000]


def _response_body(response: requests.Response) -> str:
    return response.text[:1000].replace("\n", "\\n")


def responses_content_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    texts: list[str] = []
    for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if isinstance(item, dict):
            for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if not isinstance(content, dict):
                    continue
                if isinstance(content.get("text"), str):
                    texts.append(content["text"])
                elif isinstance(content.get("content"), str):
                    texts.append(content["content"])
    if texts:
        return "\n".join(texts)
    choice_content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        if isinstance(data.get("choices"), list) and data.get("choices")
        else ""
    )
    if isinstance(choice_content, str):
        return choice_content
    raise KeyError("could not find response text in Responses API payload")


def anthropic_content_text(data: dict[str, Any]) -> str:
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
            and block.get("type") in {"text", "output_text"}
        ]
        if texts:
            return "\n".join(text for text in texts if text)
    raise KeyError("could not find response text in Anthropic Messages payload")


def response_content_and_reasoning(data: dict[str, Any], api_format: str) -> tuple[str, str]:
    if api_format in {"anthropic", "anthropic_messages", "messages"}:
        return anthropic_content_text(data), ""
    if api_format in {"responses", "response", "gpt_pro_input", "pro_input", "chat_input"}:
        return responses_content_text(data), ""
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    if not isinstance(content, str):
        raise KeyError("could not find string content in Chat Completions payload")
    reasoning = message.get("reasoning_content", "")
    return content, reasoning if isinstance(reasoning, str) else ""


def anthropic_messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/messages") else f"{base}/messages"


def build_anthropic_payload(
    config: VLMConfig,
    text_prompt: str,
    image_paths: list[Path],
    max_tokens: int | None,
) -> tuple[str, dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for image_path in image_paths:
        data_url = image_data_url(image_path)
        header, encoded = data_url.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded,
                },
            }
        )
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if config.thinking_type:
        payload["thinking"] = {"type": config.thinking_type}
    if config.reasoning_effort:
        payload["output_config"] = {"effort": config.reasoning_effort}
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    return anthropic_messages_url(config.base_url), payload


def anthropic_headers(config: VLMConfig) -> dict[str, str]:
    mode = config.auth_mode.strip().lower()
    if mode not in {"bearer", "x-api-key", "both"}:
        raise ValueError(f"Unsupported Anthropic auth_mode: {config.auth_mode!r}")
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": config.anthropic_version,
    }
    if mode in {"x-api-key", "both"}:
        headers["x-api-key"] = config.api_key
    if mode in {"bearer", "both"}:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def build_chat_payload(config: VLMConfig, text_prompt: str, image_paths: list[Path], max_tokens: int | None) -> tuple[str, dict[str, Any]]:
    content: str | list[dict[str, Any]]
    if image_paths:
        content = [{"type": "text", "text": text_prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
    else:
        # Some OpenAI-compatible routers reject vision-style text blocks for
        # text-only chat requests, while accepting them when images are present.
        content = text_prompt
    payload = {
        "model": config.model,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.top_k is not None:
        payload["top_k"] = config.top_k
    if config.chat_template_kwargs:
        payload["chat_template_kwargs"] = config.chat_template_kwargs
    return f"{config.base_url}/chat/completions", payload


def build_responses_payload(config: VLMConfig, text_prompt: str, image_paths: list[Path], max_tokens: int | None) -> tuple[str, dict[str, Any]]:
    output_tokens = max_tokens if max_tokens is not None else config.max_tokens
    if image_paths:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text_prompt}]
        for image_path in image_paths:
            content.append({"type": "input_image", "image_url": image_data_url(image_path)})
        input_value: str | list[dict[str, Any]] = [{"role": "user", "content": content}]
    else:
        input_value = text_prompt
    payload: dict[str, Any] = {
        "model": config.model,
        "input": input_value,
        "max_output_tokens": output_tokens,
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    if config.disable_response_storage:
        payload["store"] = False
    if config.provider:
        payload["provider"] = config.provider
    return f"{config.base_url}/responses", payload


def build_gpt_pro_input_payload(config: VLMConfig, text_prompt: str, image_paths: list[Path], max_tokens: int | None) -> tuple[str, dict[str, Any]]:
    output_tokens = max_tokens if max_tokens is not None else config.max_tokens
    if image_paths:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text_prompt}]
        for image_path in image_paths:
            content.append({"type": "input_image", "image_url": image_data_url(image_path)})
        input_value: str | list[dict[str, Any]] = [{"role": "user", "content": content}]
    else:
        input_value = text_prompt
    payload: dict[str, Any] = {
        "model": config.model,
        "input": input_value,
        "reasoning": {"effort": config.reasoning_effort or "low"},
        "stream": False,
        "max_output_tokens": output_tokens,
    }
    return f"{config.base_url}/v1/responses", payload


def chat_completion_json(
    config: VLMConfig,
    *,
    text_prompt: str,
    image_paths: list[Path],
    max_tokens: int | None = None,
    request_id: str = "",
) -> dict[str, Any]:
    require_complete_config(config)
    api_format = (config.api_format or "chat_completions").strip().lower()
    if api_format in {"anthropic", "anthropic_messages", "messages"}:
        url, payload = build_anthropic_payload(
            config, text_prompt, image_paths, max_tokens
        )
        headers = anthropic_headers(config)
    elif api_format in {"responses", "response"}:
        url, payload = build_responses_payload(config, text_prompt, image_paths, max_tokens)
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
    elif api_format in {"gpt_pro_input", "pro_input", "chat_input"}:
        url, payload = build_gpt_pro_input_payload(config, text_prompt, image_paths, max_tokens)
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
    else:
        url, payload = build_chat_payload(config, text_prompt, image_paths, max_tokens)
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
    max_attempts = int(config.max_retries) + 1
    request_attempts: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        wait_for_request_slot(config, request_id)
        attempt_started = time.monotonic()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            elapsed_sec = time.monotonic() - attempt_started
            should_retry = attempt < max_attempts
            sleep_sec = (
                float(config.retry_backoff) * (2 ** (attempt - 1))
                if should_retry
                else 0.0
            )
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "transport_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": should_retry,
                    "retry_delay_sec": sleep_sec,
                }
            )
            if not should_retry:
                raise VLMRetryExhaustedError(
                    request_id=request_id,
                    attempts=attempt,
                    last_error=exc,
                    request_attempts=request_attempts,
                ) from exc
            prefix = f"[transport_retry] {request_id}: " if request_id else "[transport_retry] "
            print(
                f"{prefix}{type(exc).__name__} on attempt {attempt}/{max_attempts}; "
                f"sleeping {sleep_sec:.1f}s",
                flush=True,
            )
            time.sleep(sleep_sec)
            continue

        elapsed_sec = time.monotonic() - attempt_started
        if response.status_code >= 400:
            body = _response_body(response)
            retryable = is_retryable_http_response(response)
            should_retry = retryable and attempt < max_attempts
            retry_after = retry_after_seconds(response) if should_retry else None
            sleep_sec = (
                max(
                    float(config.retry_backoff) * (2 ** (attempt - 1)),
                    retry_after or 0.0,
                )
                if should_retry
                else 0.0
            )
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "http_error",
                    "status_code": response.status_code,
                    "error": body,
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": should_retry,
                    "retry_delay_sec": sleep_sec,
                }
            )
            error = requests.HTTPError(
                f"HTTP {response.status_code}: {body}", response=response
            )
            if should_retry:
                prefix = f"[transport_retry] {request_id}: " if request_id else "[transport_retry] "
                print(
                    f"{prefix}HTTP {response.status_code} on attempt "
                    f"{attempt}/{max_attempts}; sleeping {sleep_sec:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_sec)
                continue
            if retryable:
                raise VLMRetryExhaustedError(
                    request_id=request_id,
                    attempts=attempt,
                    last_error=error,
                    request_attempts=request_attempts,
                ) from error
            error_type = (
                VLMContentFilterError
                if contains_content_filter_marker(response.text)
                else VLMProviderError
            )
            raise error_type(
                request_id=request_id,
                message=f"non-retryable HTTP {response.status_code}: {body}",
                attempts=attempt,
                request_attempts=request_attempts,
                status_code=response.status_code,
                response_body=body,
            )

        try:
            data = response.json()
        except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as exc:
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "provider_payload_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMProviderError(
                request_id=request_id,
                message="provider returned a non-JSON response envelope",
                attempts=attempt,
                request_attempts=request_attempts,
                status_code=response.status_code,
                response_body=response.text,
            ) from exc

        if response_reports_content_filter(data):
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "provider_content_filter",
                    "status_code": response.status_code,
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMContentFilterError(
                request_id=request_id,
                message="provider response reports content filtering",
                attempts=attempt,
                request_attempts=request_attempts,
                status_code=response.status_code,
                response_body=response.text,
            )

        provider_error = provider_error_envelope(data)
        if provider_error:
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "provider_error",
                    "status_code": response.status_code,
                    "error": provider_error,
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMProviderError(
                request_id=request_id,
                message=f"provider returned an error envelope: {provider_error}",
                attempts=attempt,
                request_attempts=request_attempts,
                status_code=response.status_code,
                response_body=response.text,
            )

        reasoning_text = ""
        try:
            content_text, reasoning_text = response_content_and_reasoning(data, api_format)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "format_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMFormatError(
                request_id=request_id,
                raw_response=response.text,
                reasoning_content=reasoning_text,
                usage=data.get("usage", {}) if isinstance(data, dict) else {},
                attempts=attempt,
                request_attempts=request_attempts,
                error=exc,
            ) from exc

        if "{" not in content_text and contains_content_filter_marker(content_text):
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "provider_content_filter",
                    "status_code": response.status_code,
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMContentFilterError(
                request_id=request_id,
                message="provider response reports content filtering",
                attempts=attempt,
                request_attempts=request_attempts,
                status_code=response.status_code,
                response_body=content_text,
            )

        try:
            parsed = parse_json_object(content_text)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            request_attempts.append(
                {
                    "attempt": attempt,
                    "timeout_sec": config.timeout,
                    "outcome": "format_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_sec": round(elapsed_sec, 3),
                    "retry_scheduled": False,
                    "retry_delay_sec": 0.0,
                }
            )
            raise VLMFormatError(
                request_id=request_id,
                raw_response=content_text,
                reasoning_content=reasoning_text,
                usage=data.get("usage", {}) if isinstance(data, dict) else {},
                attempts=attempt,
                request_attempts=request_attempts,
                error=exc,
            ) from exc

        request_attempts.append(
            {
                "attempt": attempt,
                "timeout_sec": config.timeout,
                "outcome": "accepted",
                "status_code": response.status_code,
                "latency_sec": round(elapsed_sec, 3),
                "retry_scheduled": False,
                "retry_delay_sec": 0.0,
            }
        )
        parsed["_raw_response"] = content_text
        parsed["_reasoning_content"] = reasoning_text
        parsed["_usage"] = data.get("usage", {}) if isinstance(data, dict) else {}
        parsed["_attempts"] = attempt
        parsed["_attempt_latency_sec"] = round(elapsed_sec, 3)
        parsed["_request_attempts"] = request_attempts
        if parsed.get("_extra_json_ignored"):
            parsed["_parse_warning"] = "trailing content after the first JSON object was ignored"
        prefix = f"[accepted] {request_id}: " if request_id else "[accepted] "
        print(
            f"{prefix}physical_attempts={attempt} latency={elapsed_sec:.3f}s",
            flush=True,
        )
        return parsed

    raise AssertionError("unreachable transport retry loop")


def public_config_summary(config: VLMConfig) -> dict[str, Any]:
    summary = {
        "base_url": config.base_url,
        "model": config.model,
        "api_format": config.api_format,
        "provider": config.provider,
        "reasoning_effort": config.reasoning_effort,
        "disable_response_storage": config.disable_response_storage,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "chat_template_kwargs": config.chat_template_kwargs,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "retry_backoff": config.retry_backoff,
        "maximum_transport_attempts": config.max_retries + 1,
        "transport_policy_version": TRANSPORT_POLICY_VERSION,
        "json_parse_policy_version": JSON_PARSE_POLICY_VERSION,
        "min_request_interval_sec": config.min_request_interval_sec,
        "max_tokens": config.max_tokens,
        "config_path": config.config_path,
        "has_api_key": bool(config.api_key),
    }
    if config.api_format.strip().lower() in {
        "anthropic",
        "anthropic_messages",
        "messages",
    }:
        summary.update(
            {
                "auth_mode": config.auth_mode,
                "anthropic_version": config.anthropic_version,
                "thinking_type": config.thinking_type,
            }
        )
    return summary
