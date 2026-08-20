#!/usr/bin/env python3
"""Inject complete annotated support context before an agent's first action."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


INTERVENTION_VERSION = "support_provided_initial_observation_v1"
INITIAL_EVENT_NAME = "initial_page_observations"
PROMPT_APPENDIX = (
    "An initial_page_observations environment event, when present, displays corpus "
    "pages before the first action. Those opaque page handles may subsequently be "
    "used anywhere that an already discovered and inspected page handle is accepted."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_id_for_row(row: dict[str, Any]) -> str:
    query_id = str(row.get("query_id") or row.get("path_id") or "").strip()
    if not query_id:
        raise ValueError("query row has no query_id")
    return query_id


def support_page_ids(row: dict[str, Any]) -> list[str]:
    query_id = query_id_for_row(row)
    level = int(row.get("level"))
    if level not in {2, 3}:
        raise ValueError(f"support-provided evaluation accepts only L2/L3, got L{level}")
    values = row.get("support_unit_ids")
    if not isinstance(values, list):
        raise ValueError(f"{query_id} has no evaluator support_unit_ids")
    supports = [str(value).strip() for value in values]
    if len(supports) != level - 1 or len(set(supports)) != len(supports):
        raise ValueError(f"{query_id} has an invalid support path: {supports}")
    if str(row.get("gold_page_id") or "") in supports:
        raise ValueError(f"{query_id} support path contains the answer")
    return supports


def visible_support_page_ids(row: dict[str, Any], shuffle_seed: int) -> list[str]:
    values = support_page_ids(row)
    digest = hashlib.sha256(
        f"{shuffle_seed}:{query_id_for_row(row)}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    return values


def intervention_fingerprint(
    row: dict[str, Any], *, route: str, shuffle_seed: int
) -> dict[str, Any]:
    annotated = support_page_ids(row)
    visible = visible_support_page_ids(row, shuffle_seed)
    return {
        "version": INTERVENTION_VERSION,
        "route": route,
        "condition": "support_provided",
        "level": int(row["level"]),
        "shuffle_seed": shuffle_seed,
        "annotated_support_page_ids": annotated,
        "injected_page_ids": visible,
        "free_initial_observation": True,
        "agent_steps_consumed": 0,
        "information_tool_calls_consumed": 0,
        "role_labels_visible_to_model": False,
        "path_order_visible_to_model": False,
    }


def _initial_trace_items(runtime: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in runtime.trace
        if isinstance(item, dict) and item.get("tool_name") == INITIAL_EVENT_NAME
    ]


def inject_initial_page_observations(
    runtime: Any,
    row: dict[str, Any],
    *,
    route: str,
    shuffle_seed: int,
) -> dict[str, Any]:
    metadata = intervention_fingerprint(row, route=route, shuffle_seed=shuffle_seed)
    existing = _initial_trace_items(runtime)
    if existing:
        if len(existing) != 1:
            raise ValueError("runtime contains multiple initial page observations")
        stored = existing[0].get("evaluator_only_intervention")
        if not isinstance(stored, dict):
            raise ValueError("checkpoint initial-page metadata is invalid")
        comparable = {key: value for key, value in stored.items() if key != "visible_page_handles"}
        if comparable != metadata:
            raise ValueError("checkpoint initial-page metadata does not match")
        return copy.deepcopy(stored)

    usage_before = copy.deepcopy(runtime.public_usage())
    loaded_pages: list[dict[str, Any]] = []
    page_handles: list[str] = []
    for page_id in metadata["injected_page_ids"]:
        page_handle = runtime.handle_space.page_id_to_handle(page_id)
        page = runtime.handle_space.require_page(page_handle)
        if not page.image_path.is_file():
            raise FileNotFoundError(f"Initial page image is missing for {page_id}")
        page_handles.append(page_handle)
        loaded_pages.append(
            {
                "page_handle": page_handle,
                "page_width": page.page_width,
                "page_height": page.page_height,
            }
        )
        runtime.discovered_page_handles.add(page_handle)
        runtime.inspected_page_handles.add(page_handle)
        runtime.seen_page_handles.add(page_handle)
        runtime.seen_image_handles.add(page_handle)

    observation = {"ok": True, "loaded_pages": loaded_pages, "failed_pages": []}
    runtime._record(INITIAL_EVENT_NAME, {}, observation, page_handles)
    event = runtime.trace[-1]
    event["agent_action"] = False
    event["evaluator_only_intervention"] = metadata
    if runtime.public_usage() != usage_before:
        raise AssertionError("Initial observations changed public tool usage")

    visible_payload = json.dumps(
        {
            "tool": INITIAL_EVENT_NAME,
            "arguments": event.get("arguments", {}),
            "observation": event.get("observation", {}),
            "artifacts": event.get("artifacts", []),
        },
        ensure_ascii=False,
    ).lower()
    forbidden = ("gold", "support", "first hop", "second hop", "path order")
    leaked = [token for token in forbidden if token in visible_payload]
    if leaked:
        raise AssertionError(f"Model-visible initial observation leaks roles: {leaked}")

    metadata = dict(metadata)
    metadata["visible_page_handles"] = page_handles
    event["evaluator_only_intervention"] = metadata
    return metadata


def _decorate_prediction(
    row: dict[str, Any], runtime: Any, metadata: dict[str, Any]
) -> dict[str, Any]:
    initial_events = _initial_trace_items(runtime)
    if len(initial_events) != 1:
        raise ValueError("completed episode must contain one initial observation")
    event = initial_events[0]
    row["initial_page_observations"] = {
        "loaded_pages": copy.deepcopy(event["observation"]["loaded_pages"]),
        "visible_image_artifacts": list(event.get("artifacts", [])),
        "agent_action": False,
    }
    row["intervention"] = copy.deepcopy(metadata)
    row["tool_trace"] = [
        copy.deepcopy(item)
        for item in row.get("tool_trace", [])
        if item.get("tool_name") != INITIAL_EVENT_NAME
    ]
    return row


def install_support_provided_adapter(
    runner: Any, *, route: str, shuffle_seed: int = 20260815
) -> None:
    """Patch the imported runner only inside this dedicated process."""

    if route not in {"visual", "ocr_text"}:
        raise ValueError(f"invalid route: {route}")
    base_harness_version = str(runner.HARNESS_VERSION)
    base_schema_version = str(runner.SCHEMA_VERSION)
    original_loader = runner.load_benchmark_rows
    original_system_prompt = runner.system_prompt_for_tools
    original_finalization_prompt = runner.finalization_prompt_for_tools
    original_run_episode = runner.run_episode
    original_fingerprint = runner.episode_checkpoint_fingerprint
    original_worker_exception = runner.worker_exception_row

    runner.HARNESS_VERSION = f"{base_harness_version}+{INTERVENTION_VERSION}"
    runner.SCHEMA_VERSION = f"{base_schema_version}.SupportProvidedV1"

    def load_benchmark_rows(queries_path: Path, evaluator_path: Path) -> list[dict[str, Any]]:
        rows = original_loader(queries_path, evaluator_path)
        evaluator = {
            str(item["query_id"]): item for item in _read_jsonl(evaluator_path)
        }
        selected: list[dict[str, Any]] = []
        for row in rows:
            if int(row["level"]) not in {2, 3}:
                continue
            item = evaluator[str(row["query_id"])]
            row = dict(row)
            row["support_unit_ids"] = list(item.get("support_page_ids") or [])
            selected.append(row)
        return selected

    def append_protocol(prompt: str) -> str:
        return f"{prompt}\n\nInitial observation protocol:\n- {PROMPT_APPENDIX}"

    def system_prompt_for_tools(enabled_information_tools: Any = None) -> str:
        return append_protocol(original_system_prompt(enabled_information_tools))

    def finalization_prompt_for_tools(enabled_information_tools: Any = None) -> str:
        return append_protocol(original_finalization_prompt(enabled_information_tools))

    def episode_checkpoint_fingerprint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_fingerprint(*args, **kwargs)
        query_row = kwargs.get("query_row") or (args[0] if args else None)
        if not isinstance(query_row, dict):
            raise ValueError("checkpoint fingerprint has no query_row")
        result["support_provided_initial_observation"] = intervention_fingerprint(
            query_row, route=route, shuffle_seed=shuffle_seed
        )
        return result

    def run_episode(*args: Any, **kwargs: Any) -> dict[str, Any]:
        query_row = kwargs.get("query_row") or (args[0] if args else None)
        runtime = kwargs.get("runtime") or (args[1] if len(args) > 1 else None)
        if not isinstance(query_row, dict) or runtime is None:
            raise ValueError("support-provided run_episode requires query_row and runtime")
        metadata = inject_initial_page_observations(
            runtime, query_row, route=route, shuffle_seed=shuffle_seed
        )
        return _decorate_prediction(original_run_episode(*args, **kwargs), runtime, metadata)

    def worker_exception_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_worker_exception(*args, **kwargs)
        query_row = kwargs.get("query_row")
        runtime = kwargs.get("runtime")
        if isinstance(query_row, dict) and runtime is not None:
            metadata = inject_initial_page_observations(
                runtime, query_row, route=route, shuffle_seed=shuffle_seed
            )
            return _decorate_prediction(result, runtime, metadata)
        return result

    runner.load_benchmark_rows = load_benchmark_rows
    runner.system_prompt_for_tools = system_prompt_for_tools
    runner.finalization_prompt_for_tools = finalization_prompt_for_tools
    runner.episode_checkpoint_fingerprint = episode_checkpoint_fingerprint
    runner.run_episode = run_episode
    runner.worker_exception_row = worker_exception_row
