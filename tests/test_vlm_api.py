from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_api import (  # noqa: E402
    VLMConfig,
    VLMContentFilterError,
    VLMFormatError,
    VLMProviderError,
    VLMRetryExhaustedError,
    chat_completion_json,
    parse_json_object,
)


def response(status: int, payload: object, **headers: str) -> requests.Response:
    value = requests.Response()
    value.status_code = status
    value.headers.update(headers)
    if isinstance(payload, str):
        value._content = payload.encode("utf-8")
    else:
        value._content = json.dumps(payload).encode("utf-8")
    return value


def config() -> VLMConfig:
    return VLMConfig(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        api_format="responses",
        timeout=1.0,
        max_retries=2,
        retry_backoff=0.0,
    )


def test_tolerant_parser_uses_first_complete_object() -> None:
    parsed = parse_json_object(
        "preface\n```json\n{\"action\": \"first\"}\n```\n{\"action\": \"second\"}"
    )
    assert parsed["action"] == "first"
    assert "second" in parsed["_extra_json_ignored"]


def test_tolerant_parser_does_not_skip_a_malformed_first_object() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_object('prefix {not valid} then {"action": "second"}')


def test_malformed_model_output_is_not_retried() -> None:
    provider_response = response(200, {"output_text": "not json", "usage": {}})
    with patch("vlm_api.requests.post", return_value=provider_response) as post:
        with pytest.raises(VLMFormatError) as captured:
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="format"
            )
    assert post.call_count == 1
    assert captured.value.raw_response == "not json"
    assert captured.value.request_attempts[0]["retry_scheduled"] is False


def test_connection_errors_use_at_most_three_physical_attempts() -> None:
    provider_response = response(
        200, {"output_text": '{"action":"submit_answer"}', "usage": {}}
    )
    with (
        patch(
            "vlm_api.requests.post",
            side_effect=[requests.ConnectionError("down"), requests.Timeout("slow"), provider_response],
        ) as post,
        patch("vlm_api.time.sleep"),
    ):
        parsed = chat_completion_json(
            config(), text_prompt="prompt", image_paths=[], request_id="retry"
        )
    assert post.call_count == 3
    assert parsed["_attempts"] == 3
    assert [item["outcome"] for item in parsed["_request_attempts"]] == [
        "transport_error",
        "transport_error",
        "accepted",
    ]


def test_nonretryable_http_error_fails_immediately() -> None:
    with patch(
        "vlm_api.requests.post", return_value=response(401, {"error": "unauthorized"})
    ) as post:
        with pytest.raises(VLMProviderError):
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="401"
            )
    assert post.call_count == 1


def test_retryable_http_error_honors_three_attempt_cap() -> None:
    provider_response = response(429, {"error": "busy"}, **{"Retry-After": "0"})
    with (
        patch("vlm_api.requests.post", return_value=provider_response) as post,
        patch("vlm_api.time.sleep"),
    ):
        with pytest.raises(VLMRetryExhaustedError) as captured:
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="429"
            )
    assert post.call_count == 3
    assert captured.value.attempts == 3


def test_retry_after_is_honored_before_a_successful_retry() -> None:
    busy = response(429, {"error": "busy"}, **{"Retry-After": "7"})
    accepted = response(200, {"output_text": '{"action":"submit_answer"}'})
    with (
        patch("vlm_api.requests.post", side_effect=[busy, accepted]) as post,
        patch("vlm_api.time.sleep") as sleep,
    ):
        parsed = chat_completion_json(
            config(), text_prompt="prompt", image_paths=[], request_id="retry-after"
        )
    assert post.call_count == 2
    sleep.assert_called_once_with(7.0)
    assert parsed["_attempts"] == 2


def test_content_filter_http_error_fails_without_retry() -> None:
    filtered = response(400, {"error": {"type": "content_filter"}})
    with patch("vlm_api.requests.post", return_value=filtered) as post:
        with pytest.raises(VLMContentFilterError):
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="filtered"
            )
    assert post.call_count == 1


def test_structured_content_filter_response_fails_without_retry() -> None:
    filtered = response(
        200,
        {"output": [{"finish_reason": "content_filter", "content": []}]},
    )
    with patch("vlm_api.requests.post", return_value=filtered) as post:
        with pytest.raises(VLMContentFilterError):
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="filtered"
            )
    assert post.call_count == 1


def test_success_status_provider_error_envelope_fails_without_retry() -> None:
    invalid = response(200, {"error": {"type": "invalid_request", "message": "bad input"}})
    with patch("vlm_api.requests.post", return_value=invalid) as post:
        with pytest.raises(VLMProviderError):
            chat_completion_json(
                config(), text_prompt="prompt", image_paths=[], request_id="invalid"
            )
    assert post.call_count == 1


def test_valid_json_text_may_mention_content_filtering() -> None:
    accepted = response(
        200,
        {
            "output_text": (
                '{"action":"submit_answer",'
                '"rationale":"content filtering policy is unrelated"}'
            )
        },
    )
    with patch("vlm_api.requests.post", return_value=accepted) as post:
        parsed = chat_completion_json(
            config(), text_prompt="prompt", image_paths=[], request_id="valid-marker"
        )
    assert post.call_count == 1
    assert parsed["action"] == "submit_answer"
