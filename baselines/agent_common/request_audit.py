#!/usr/bin/env python3
"""Summaries for logical planner turns and physical provider requests."""

from __future__ import annotations

from typing import Any, Iterable


def summarize_model_outputs(model_outputs: Iterable[dict[str, Any]]) -> dict[str, int]:
    outputs = [value for value in model_outputs if isinstance(value, dict)]
    logical_turns = {
        (str(output.get("phase", "planner")), int(output.get("step", 0)))
        for output in outputs
        if int(output.get("step", 0)) > 0
        and str(output.get("phase", "planner")) != "worker"
    }
    request_attempts = [
        attempt
        for output in outputs
        for attempt in output.get("request_attempts", [])
        if isinstance(attempt, dict)
    ]
    return {
        "logical_planner_turns": len(logical_turns),
        "planner_call_events": len(outputs),
        "physical_request_attempts": sum(
            int(output.get("attempts", 0)) for output in outputs
        ),
        "transport_retries": sum(
            bool(attempt.get("retry_scheduled")) for attempt in request_attempts
        ),
        "format_failures": sum(
            output.get("event_type") == "format_error" for output in outputs
        ),
        "provider_failures": sum(
            output.get("event_type") == "provider_error" for output in outputs
        ),
        "accepted_responses": sum(
            output.get("response_accepted") is True for output in outputs
        ),
        "rejected_responses": sum(
            output.get("response_accepted") is False for output in outputs
        ),
    }


def aggregate_request_audits(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        audit = row.get("request_audit", {})
        if not isinstance(audit, dict):
            continue
        for key, value in audit.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals
