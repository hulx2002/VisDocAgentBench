#!/usr/bin/env python3
"""Run the VisDocAgentBench visual-retrieval agent."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
import threading
import traceback
from typing import Any, Callable, TextIO

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
AGENT_COMMON_DIR = Path(__file__).resolve().parents[1] / "agent_common"
if str(AGENT_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_COMMON_DIR))

from common import (  # noqa: E402
    PROJECT_ROOT,
    ensure_dir,
    read_jsonl,
    short_json,
    write_json,
    write_jsonl,
)
from corpus import load_handle_space  # noqa: E402
from ocr_backends import build_ocr_backend  # noqa: E402
from tools import (  # noqa: E402
    INFORMATION_TOOLS,
    MAX_SEARCH_RESULTS,
    MAX_VIEW_IMAGES_PER_CALL,
    AgentToolRuntime,
)
from vl_retriever import VisualRetriever  # noqa: E402
from resume_scope import assert_inplace_resume_scope  # noqa: E402
from episode_checkpoint import (  # noqa: E402
    RECOVERABLE_EPISODE_STATUSES,
    EpisodeCheckpointError,
    EpisodeProgress,
    canonical_digest,
    checkpoint_path,
    directory_identity,
    file_collection_identity,
    file_identity,
    load_episode_checkpoint,
    observations_from_tool_trace,
    recoverable_progress_from_prediction,
    remove_episode_checkpoint,
    write_episode_checkpoint,
)
from request_audit import aggregate_request_audits, summarize_model_outputs  # noqa: E402
from benchmark_io import load_benchmark_rows  # noqa: E402


SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vlm_api import (  # noqa: E402
    JSON_PARSE_POLICY_VERSION,
    TRANSPORT_POLICY_VERSION,
    VLMFormatError,
    VLMProviderError,
    VLMRetryExhaustedError,
    chat_completion_json,
    public_config_summary,
    resolve_vlm_config,
)


SCHEMA_VERSION = "VisDocAgentBenchVisualAgentV1"
HARNESS_VERSION = "visdocagentbench_agent_harness_v2"
CROP_PROTOCOL_VERSION = "batch_purpose_soft_partial_no_area_cap_v1"


class StepWallTimeoutError(TimeoutError):
    pass


CROP_TOOL_DESCRIPTION = (
    "- crop_pages(crops): after inspect_pages, displays 1 to 10 cropped regions to "
    "resolve specific details that remain unreadable or ambiguous in the full-page "
    'views. Each item should include {"page_handle": ..., "region": [x1, y1, x2, y2], '
    '"purpose": "brief region-specific inspection purpose"}; coordinates are '
    "relative to the source page and normalized to [0, 1], with (0, 0) at the "
    "top-left and (1, 1) at the bottom-right. Use the top-level reason for the "
    "batch-level decision and each purpose for its region-specific target. A missing "
    "purpose is recorded but does not prevent the crop. Items are processed "
    "independently, so valid crops are returned even if other items fail."
)


SYSTEM_PROMPT = """You are an agent for closed-corpus rich-document page image retrieval.

You must find answer pages from a fixed corpus of page images. You do not have web access. You cannot access paper titles, document ids, page numbers, topics, source metadata, or extracted page text unless a tool explicitly returns it.

The page handles are opaque identifiers. Never infer document order, page order, or source identity from the handle strings.

Available information tools:
- visual_search(query, top_k): searches page-image embeddings only and returns page handles plus scores. It does not show page images.
- inspect_pages(page_handles): displays 1 to 10 full page images previously returned by visual_search. Every requested and successfully loaded page is shown in the next turn.
__CROP_TOOL_DESCRIPTION__
- ocr_image(target_handle): OCRs a full page or crop image. This is the only way to obtain text from images.

All information tools share the single episode step limit. There are no per-tool cumulative call limits.

Terminal action:
- submit_answer(ranked_answers, overall_rationale): submit a ranked top-10 answer list and end the episode.

At each step, output exactly one JSON object and no extra text. Do not output multiple JSON objects, multiple tool calls, or a planned sequence of actions. If you need another tool after seeing the result, wait for the next step.

For an information tool call:
{
  "action": "visual_search" | "inspect_pages" | "crop_pages" | "ocr_image",
  "arguments": {...},
  "reason": "brief reason"
}

For final submission:
{
  "action": "submit_answer",
  "ranked_answers": [
    {
      "page_handle": "page_000123",
      "evidence_page_handles": ["page_000045", "page_000123"],
      "rationale": "one concise sentence explaining why these evidence pages support this answer page"
    }
  ],
  "overall_rationale": "brief strategy summary"
}

Submission ranking policy:
- Your primary objective is to put the single best answer page at rank 1.
- Do not sacrifice rank-1 precision to diversify the list.
- After the best answer, include the next most plausible candidate pages until the ranked list has 10 distinct pages.
- Before submitting, make sure at least 10 candidate page handles are available. If fewer than 10 page handles have been found, call visual_search again with top_k large enough to obtain at least 10 candidates.
- submit_answer must contain exactly 10 ranked_answers whenever 10 or more candidate page handles have been found in the episode.

Only include page handles that exist in this episode. evidence_page_handles must be pages you actually found or inspected during the tool trace.
For a candidate that was returned by visual_search but was not inspected or OCRed, use an empty evidence_page_handles list.
Keep each per-rank rationale to one short sentence so the complete top-10 JSON fits in the response budget.
""".replace("__CROP_TOOL_DESCRIPTION__", CROP_TOOL_DESCRIPTION)


ANSWER_ONLY_FINALIZATION_PROMPT = """You are in the answer-only finalization stage for closed-corpus rich-document page image retrieval.

The shared episode step limit has been exhausted. You cannot call visual_search, inspect_pages, crop_pages, ocr_image, or any other information tool. You must submit the best ranked answer pages using only the evidence already collected in this episode.

The page handles are opaque identifiers. Never infer document order, page order, or source identity from the handle strings.

Output exactly one JSON object and no extra text:
{
  "action": "submit_answer",
  "ranked_answers": [
    {
      "page_handle": "page_000123",
      "evidence_page_handles": ["page_000045", "page_000123"],
      "rationale": "one concise sentence explaining why the collected evidence supports this answer page"
    }
  ],
  "overall_rationale": "brief strategy summary"
}

Rules:
- Your primary objective is to put the single best answer page at rank 1.
- Do not sacrifice rank-1 precision to diversify the list.
- Submit exactly 10 ranked_answers if eligible_answer_page_handles contains 10 or more handles.
- If fewer than 10 eligible answer handles are available, submit all available eligible answer handles.
- page_handle values must come from eligible_answer_page_handles.
- evidence_page_handles must come from eligible_evidence_page_handles. If no inspected/OCRed evidence page is suitable for a rank, use an empty evidence_page_handles list for that rank.
- Do not invent page handles.
"""


TOOL_DESCRIPTION_LINES = {
    "visual_search": "- visual_search(query, top_k): searches page-image embeddings only and returns page handles plus scores. It does not show page images.",
    "inspect_pages": "- inspect_pages(page_handles): displays 1 to 10 full page images previously returned by visual_search. Every requested and successfully loaded page is shown in the next turn.",
    "crop_pages": CROP_TOOL_DESCRIPTION,
    "ocr_image": "- ocr_image(target_handle): OCRs a full page or crop image. This is the only way to obtain text from images.",
}
DEFAULT_ACTION_ENUM = '"visual_search" | "inspect_pages" | "crop_pages" | "ocr_image"'


def system_prompt_for_tools(enabled_tools: frozenset[str] | None = None) -> str:
    """Return the frozen full prompt or its mechanically tool-restricted ablation form."""
    enabled = INFORMATION_TOOLS if enabled_tools is None else frozenset(enabled_tools)
    unknown_tools = enabled - INFORMATION_TOOLS
    if unknown_tools:
        raise ValueError(f"Unknown enabled information tools: {sorted(unknown_tools)}")
    if enabled == INFORMATION_TOOLS:
        return SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT
    for tool_name in sorted(INFORMATION_TOOLS - enabled):
        prompt = prompt.replace(TOOL_DESCRIPTION_LINES[tool_name] + "\n", "")

    if "crop_pages" not in enabled and "ocr_image" in enabled:
        prompt = prompt.replace(
            TOOL_DESCRIPTION_LINES["ocr_image"],
            "- ocr_image(target_handle): OCRs a full page image. This is the only way to obtain text from images.",
        )
    if "ocr_image" not in enabled:
        prompt = prompt.replace(
            "but was not inspected or OCRed", "but was not inspected"
        )

    action_enum = " | ".join(
        f'"{name}"'
        for name in ("visual_search", "inspect_pages", "crop_pages", "ocr_image")
        if name in enabled
    )
    prompt = prompt.replace(DEFAULT_ACTION_ENUM, action_enum)
    return prompt


def finalization_prompt_for_tools(enabled_tools: frozenset[str] | None = None) -> str:
    """Return the frozen finalization prompt with unavailable tool names removed for ablations."""
    enabled = INFORMATION_TOOLS if enabled_tools is None else frozenset(enabled_tools)
    unknown_tools = enabled - INFORMATION_TOOLS
    if unknown_tools:
        raise ValueError(f"Unknown enabled information tools: {sorted(unknown_tools)}")
    if enabled == INFORMATION_TOOLS:
        return ANSWER_ONLY_FINALIZATION_PROMPT

    prompt = ANSWER_ONLY_FINALIZATION_PROMPT
    enabled_list = ", ".join(
        name
        for name in ("visual_search", "inspect_pages", "crop_pages", "ocr_image")
        if name in enabled
    )
    prompt = prompt.replace(
        "You cannot call visual_search, inspect_pages, crop_pages, ocr_image, or any other information tool.",
        f"You cannot call {enabled_list}, or any other information tool.",
    )
    if "ocr_image" not in enabled:
        prompt = prompt.replace(
            "inspected/OCRed evidence page", "inspected evidence page"
        )
    return prompt


class TeeStream:
    """Mirror stdout/stderr to the terminal and a per-run log file."""

    def __init__(self, primary: TextIO, secondary: TextIO):
        self.primary = primary
        self.secondary = secondary
        self.lock = threading.Lock()

    def write(self, text: str) -> int:
        with self.lock:
            self.primary.write(text)
            self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.primary.flush()
            self.secondary.flush()


def install_run_log(run_dir: Path) -> Path:
    log_path = run_dir / "run.log"
    log_fh = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_fh)  # type: ignore[assignment]
    sys.stderr = TeeStream(sys.__stderr__, log_fh)  # type: ignore[assignment]

    def _close_log() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            try:
                log_fh.close()
            except Exception:
                pass

    atexit.register(_close_log)
    print(f"[log] teeing stdout/stderr to {log_path}", flush=True)
    return log_path


@contextmanager
def step_wall_timeout(seconds: float, label: str):
    if seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        # SIGALRM cannot be configured from ThreadPoolExecutor workers. The
        # request-level HTTP timeout remains active for concurrent runs.
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise StepWallTimeoutError(f"{label} exceeded {seconds:.1f}s wall timeout")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def build_episode_prompt(
    *,
    query: str,
    runtime: AgentToolRuntime,
    observations: list[dict[str, Any]],
    current_step: int,
    max_steps: int,
    format_repair: str = "",
    context_artifact_handles: list[str] | None = None,
    enabled_information_tools: frozenset[str] | None = None,
) -> str:
    state = {
        "task_query": query,
        "step_usage": {
            "current_step": current_step,
            "max_steps": max_steps,
            "steps_remaining_after_this_action": max_steps - current_step,
        },
        "tool_usage": runtime.public_usage(),
        "per_call_limits": runtime.public_per_call_limits(),
        "visible_image_artifacts_in_this_message": context_artifact_handles or [],
        "tool_observations": observations[-12:],
    }
    prompt = (
        system_prompt_for_tools(enabled_information_tools)
        + "\n\nCurrent episode state:\n"
        + json.dumps(state, ensure_ascii=False, indent=2)
    )
    if format_repair:
        prompt += (
            "\n\nYour previous output was invalid:\n"
            + format_repair
            + "\nReturn a valid JSON action now."
        )
    return prompt


def build_finalization_prompt(
    *,
    query: str,
    runtime: AgentToolRuntime,
    observations: list[dict[str, Any]],
    max_steps: int,
    format_repair: str = "",
    context_artifact_handles: list[str] | None = None,
    enabled_information_tools: frozenset[str] | None = None,
) -> str:
    eligible_answer_handles = sorted(
        runtime.discovered_page_handles | runtime.seen_page_handles
    )
    eligible_evidence_handles = sorted(runtime.seen_page_handles)
    state = {
        "task_query": query,
        "step_usage": {
            "steps_used": max_steps,
            "max_steps": max_steps,
            "steps_remaining": 0,
        },
        "tool_usage": runtime.public_usage(),
        "per_call_limits": runtime.public_per_call_limits(),
        "visible_image_artifacts_in_this_message": context_artifact_handles or [],
        "eligible_answer_page_handles": eligible_answer_handles,
        "eligible_evidence_page_handles": eligible_evidence_handles,
        "tool_observations": observations[-12:],
    }
    prompt = (
        finalization_prompt_for_tools(enabled_information_tools)
        + "\n\nFinal episode state:\n"
        + json.dumps(state, ensure_ascii=False, indent=2)
    )
    if format_repair:
        prompt += (
            "\n\nThe final in-budget planner response could not be parsed as one "
            "JSON object. Return exactly one valid submit_answer JSON object."
        )
    return prompt


def normalize_submit(
    action: dict[str, Any],
    runtime: AgentToolRuntime,
    *,
    allow_fewer_than_ten: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    ranked = action.get("ranked_answers")
    if not isinstance(ranked, list):
        return None, "submit_answer requires ranked_answers list"
    eligible_answer_handles = (
        runtime.discovered_page_handles | runtime.seen_page_handles
    )
    if not allow_fewer_than_ten and len(eligible_answer_handles) < 10:
        return None, (
            "submit_answer requires at least 10 discovered candidate page handles during the normal episode; "
            f"only {len(eligible_answer_handles)} are available, so call visual_search before submitting"
        )
    target_count = min(10, len(eligible_answer_handles))
    if len(ranked) > 10:
        return (
            None,
            f"submit_answer must contain at most 10 ranked_answers, got {len(ranked)}",
        )
    if target_count > 0 and len(ranked) != target_count:
        return None, (
            f"submit_answer must contain exactly {target_count} ranked_answers "
            f"because {len(eligible_answer_handles)} candidate page handles are available; got {len(ranked)}"
        )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    for index, item in enumerate(ranked[:10], start=1):
        if not isinstance(item, dict):
            errors.append(f"ranked_answers[{index}] is not an object")
            continue
        page_handle = str(item.get("page_handle", "")).strip()
        if page_handle not in runtime.handle_space.by_handle:
            errors.append(f"unknown page_handle at rank {index}: {page_handle}")
            continue
        if (
            page_handle not in runtime.discovered_page_handles
            and page_handle not in runtime.seen_page_handles
        ):
            errors.append(
                f"rank {index} page_handle was not returned or observed in this episode: {page_handle}"
            )
            continue
        if page_handle in seen:
            errors.append(f"duplicate page_handle: {page_handle}")
            continue
        seen.add(page_handle)
        evidence_raw = item.get("evidence_page_handles", [])
        if isinstance(evidence_raw, str):
            evidence = [evidence_raw]
        elif isinstance(evidence_raw, list):
            evidence = [str(handle) for handle in evidence_raw]
        else:
            evidence = []
        invalid_evidence = runtime.validate_evidence_handles(evidence)
        if invalid_evidence:
            invalid_set = set(invalid_evidence)
            evidence = [handle for handle in evidence if handle not in invalid_set]
            warnings.append(
                f"rank {index} removed invalid/unseen evidence_page_handles: {invalid_evidence}"
            )
        evidence = list(dict.fromkeys(evidence))
        out.append(
            {
                "page_handle": page_handle,
                "evidence_page_handles": evidence,
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    if errors:
        return None, "; ".join(errors)
    if not out:
        return None, "submit_answer contained no valid ranked answers"
    normalized = {
        "ranked_answers": out,
        "overall_rationale": str(action.get("overall_rationale", "")).strip(),
    }
    if warnings:
        normalized["normalization_warnings"] = warnings
    return normalized, ""


def score_submission(
    submission: dict[str, Any], gold_page_handle: str
) -> dict[str, Any]:
    ranked = [item["page_handle"] for item in submission.get("ranked_answers", [])]
    valid_ranking = len(ranked) == 10 and len(set(ranked)) == 10
    rank = (
        ranked.index(gold_page_handle) + 1
        if valid_ranking and gold_page_handle in ranked
        else None
    )
    return {
        "gold_rank": rank,
        "recall_at_1": 1.0 if rank is not None and rank <= 1 else 0.0,
        "recall_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
        "recall_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "recall_at_10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr_at_10": (1.0 / float(rank)) if rank is not None and rank <= 10 else 0.0,
    }


def build_metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, float | int]:
        n = len(subset)
        if not n:
            return {"count": 0}
        return {
            "count": n,
            "recall_at_1": sum(row["metrics"]["recall_at_1"] for row in subset) / n,
            "recall_at_3": sum(row["metrics"]["recall_at_3"] for row in subset) / n,
            "recall_at_5": sum(row["metrics"]["recall_at_5"] for row in subset) / n,
            "recall_at_10": sum(row["metrics"]["recall_at_10"] for row in subset) / n,
            "mrr_at_10": sum(row["metrics"]["mrr_at_10"] for row in subset) / n,
        }

    by_level: dict[str, Any] = {}
    by_topic: dict[str, Any] = {}
    for level in sorted({str(row.get("level", "")) for row in rows}):
        by_level[level] = summarize(
            [row for row in rows if str(row.get("level", "")) == level]
        )
    for topic in sorted({str(row.get("topic_id", "")) for row in rows}):
        by_topic[topic] = summarize(
            [row for row in rows if str(row.get("topic_id", "")) == topic]
        )
    return {"overall": summarize(rows), "by_level": by_level, "by_topic": by_topic}


def query_id_for_row(query_row: dict[str, Any], index: int) -> str:
    return str(
        query_row.get("query_id") or query_row.get("path_id") or f"query_{index:04d}"
    )


def get_query_text(query_row: dict[str, Any]) -> str:
    return str(
        query_row.get("manual_query")
        or query_row.get("query")
        or query_row.get("original_query")
        or ""
    )


def get_gold_page_id(query_row: dict[str, Any]) -> str:
    gold_page_id = str(
        query_row.get("gold_page_id") or query_row.get("gold_unit_id") or ""
    )
    if (
        gold_page_id
        and "_p" not in gold_page_id
        and query_row.get("gold_page_image_path")
    ):
        gold_page_id = Path(str(query_row["gold_page_image_path"])).stem
    if not gold_page_id and query_row.get("gold_page_image_path"):
        gold_page_id = Path(str(query_row["gold_page_image_path"])).stem
    return gold_page_id


def episode_checkpoint_fingerprint(
    *,
    query_row: dict[str, Any],
    config: Any,
    max_steps: int,
    allow_repair_turn: bool,
    force_final_answer_after_max_steps: bool,
    enabled_information_tools: frozenset[str],
    handle_seed: int,
    page_manifest: Path,
    page_embeddings: Path,
    embedding_base_url: str,
    embedding_model: str,
    embedding_service_revision: str,
    page_images_identity: dict[str, Any],
    ocr_backend: str,
    paddleocr_model_dir: str,
    paddleocr_layout_model_dir: str,
    ocr_model_identity: dict[str, Any],
    ocr_layout_model_identity: dict[str, Any],
    paddleocr_pipeline_version: str,
    paddleocr_vl_rec_server_url: str,
    step_wall_timeout_sec: float,
    dry_run_tools_only: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_contract": "resume_same_unobserved_planner_step_v2",
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "route": "visual",
        "query_id": str(query_row.get("query_id") or query_row.get("path_id") or ""),
        "query": get_query_text(query_row),
        "gold_page_id": get_gold_page_id(query_row),
        "max_steps": max_steps,
        "allow_repair_turn": allow_repair_turn,
        "force_final_answer_after_max_steps": force_final_answer_after_max_steps,
        "step_wall_timeout_sec": step_wall_timeout_sec,
        "dry_run_tools_only": dry_run_tools_only,
        "enabled_information_tools": sorted(enabled_information_tools),
        "prompt_sha256": canonical_digest(
            {
                "episode": system_prompt_for_tools(enabled_information_tools),
                "finalization": finalization_prompt_for_tools(
                    enabled_information_tools
                ),
            }
        ),
        "crop_protocol_version": CROP_PROTOCOL_VERSION,
        "handle_seed": handle_seed,
        "inputs": {
            "page_manifest": file_identity(page_manifest),
            "page_embeddings": file_identity(page_embeddings),
            "page_images": page_images_identity,
        },
        "retriever": {
            "base_url": embedding_base_url,
            "model": embedding_model,
            "service_revision": embedding_service_revision,
        },
        "ocr": {
            "backend": ocr_backend,
            "backend_factory": (
                f"{build_ocr_backend.__module__}."
                f"{getattr(build_ocr_backend, '__qualname__', build_ocr_backend.__name__)}"
            ),
            "model_dir": str(Path(paddleocr_model_dir).resolve()),
            "layout_model_dir": str(Path(paddleocr_layout_model_dir).resolve()),
            "model_identity": ocr_model_identity,
            "layout_model_identity": ocr_layout_model_identity,
            "pipeline_version": paddleocr_pipeline_version,
            "recognition_server_url": paddleocr_vl_rec_server_url,
            "sidecar_url": os.environ.get("QWEN35_OCR_SIDECAR_URL", ""),
        },
        "planner": {
            "base_url": config.base_url,
            "model": config.model,
            "api_format": config.api_format,
            "provider": config.provider,
            "auth_mode": config.auth_mode,
            "anthropic_version": config.anthropic_version,
            "thinking_type": config.thinking_type,
            "reasoning_effort": config.reasoning_effort,
            "disable_response_storage": config.disable_response_storage,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "chat_template_kwargs": config.chat_template_kwargs,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "retry_backoff": config.retry_backoff,
            "min_request_interval_sec": config.min_request_interval_sec,
            "image_max_side": os.environ.get("VLM_IMAGE_MAX_SIDE", ""),
        },
        "response_protocol": {
            "transport_policy_version": TRANSPORT_POLICY_VERSION,
            "json_parse_policy_version": JSON_PARSE_POLICY_VERSION,
        },
    }


def is_reusable_resume_row(
    row: dict[str, Any], *, valid_page_handles: set[str]
) -> bool:
    submission = row.get("submission")
    if not isinstance(submission, dict):
        return False
    ranked = submission.get("ranked_answers")
    if not isinstance(ranked, list) or len(ranked) != 10:
        return False
    handles: list[str] = []
    for item in ranked:
        if not isinstance(item, dict):
            return False
        page_handle = item.get("page_handle")
        if not isinstance(page_handle, str) or page_handle not in valid_page_handles:
            return False
        handles.append(page_handle)
    status = row.get("status")
    return (
        status in {"submitted", "forced_submit_after_max_steps", "dry_run_submitted"}
        and len(set(handles)) == 10
    )


def load_resume_rows(
    path: Path,
    query_ids: set[str],
    *,
    current_query_rows: dict[str, dict[str, Any]],
    expected_gold_page_handles: dict[str, str],
    valid_page_handles: set[str],
    required_harness_version: str | None = None,
    expected_fingerprints: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if (
            required_harness_version is not None
            and row.get("harness_version") != required_harness_version
        ):
            continue
        query_id = str(row.get("query_id") or "")
        if (
            query_id in query_ids
            and query_id in current_query_rows
            and query_id in expected_gold_page_handles
            and is_reusable_resume_row(
                row, valid_page_handles=valid_page_handles
            )
            and (
                expected_fingerprints is None
                or str(row.get("episode_fingerprint_sha256") or "")
                == expected_fingerprints.get(query_id, "")
            )
        ):
            current_row = dict(row)
            current_row["level"] = current_query_rows[query_id].get("level", "")
            current_row["topic_id"] = current_query_rows[query_id].get(
                "topic_id", ""
            )
            gold_page_handle = expected_gold_page_handles[query_id]
            current_row["gold_page_handle"] = gold_page_handle
            current_row["metrics"] = score_submission(
                current_row["submission"], gold_page_handle
            )
            rows[query_id] = current_row
    return rows


def worker_exception_row(
    *,
    query_row: dict[str, Any],
    handle_space: Any,
    exc: Exception,
    trace_text: str,
    runtime: AgentToolRuntime | None = None,
) -> dict[str, Any]:
    query_id = str(query_row.get("query_id") or query_row.get("path_id") or "")
    gold_page_id = get_gold_page_id(query_row)
    try:
        gold_page_handle = handle_space.page_id_to_handle(gold_page_id)
    except Exception:
        gold_page_handle = ""
    submission = {
        "ranked_answers": [],
        "overall_rationale": f"worker exception: {type(exc).__name__}: {exc}",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "query_id": query_id,
        "level": query_row.get("level", ""),
        "topic_id": query_row.get("topic_id", ""),
        "query": get_query_text(query_row),
        "gold_page_id": gold_page_id,
        "gold_page_handle": gold_page_handle,
        "submission": submission,
        "metrics": score_submission(submission, gold_page_handle),
        "tool_trace": runtime.trace if runtime is not None else [],
        "model_outputs": [
            {
                "step": 0,
                "phase": "worker",
                "action": {"action": "worker_exception"},
                "raw_response": "",
                "usage": {},
                "attempts": 0,
                "request_attempts": [],
                "response_accepted": False,
                "event_type": "worker_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": trace_text,
            }
        ],
        "status": "worker_exception",
        "tool_usage": runtime.public_usage() if runtime is not None else {},
    }


def retry_exhausted_output(
    *, exc: VLMRetryExhaustedError, step: int, phase: str = ""
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": phase,
        "action": {"action": "planner_retry_exhausted"},
        "raw_response": "",
        "usage": {},
        "attempts": exc.attempts,
        "request_attempts": exc.request_attempts,
        "response_accepted": False,
        "event_type": "transport_failure",
        "error": str(exc),
        "retry_exhausted": exc.to_dict(),
    }


def format_error_output(
    *, exc: VLMFormatError, step: int, phase: str = ""
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": phase,
        "action": {"action": "planner_format_error"},
        "raw_response": exc.raw_response,
        "reasoning_content": exc.reasoning_content,
        "usage": exc.usage,
        "attempts": exc.attempts,
        "request_attempts": exc.request_attempts,
        "response_accepted": False,
        "event_type": "format_error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def provider_error_output(
    *, exc: VLMProviderError, step: int, phase: str = ""
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": phase,
        "action": {"action": "provider_error"},
        "raw_response": exc.response_body,
        "usage": {},
        "attempts": exc.attempts,
        "request_attempts": exc.request_attempts,
        "response_accepted": False,
        "event_type": "provider_error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "provider_error": exc.to_dict(),
    }


def accepted_response_output(
    *, response: dict[str, Any], action: dict[str, Any], step: int, phase: str = ""
) -> dict[str, Any]:
    output = {
        "step": step,
        "phase": phase,
        "action": action,
        "raw_response": response.get("_raw_response", ""),
        "reasoning_content": response.get("_reasoning_content", ""),
        "usage": response.get("_usage", {}),
        "attempts": response.get("_attempts", 1),
        "request_attempts": response.get("_request_attempts", []),
        "attempt_latency_sec": response.get("_attempt_latency_sec"),
        "response_accepted": True,
        "event_type": "accepted_response",
    }
    if response.get("_parse_warning"):
        output["parse_warning"] = response["_parse_warning"]
        output["ignored_trailing_content"] = response.get(
            "_extra_json_ignored", ""
        )
    return output


def run_episode(
    *,
    query_row: dict[str, Any],
    runtime: AgentToolRuntime,
    config: Any,
    max_steps: int,
    allow_repair_turn: bool,
    force_final_answer_after_max_steps: bool,
    step_wall_timeout_sec: float = 0.0,
    dry_run_tools_only: bool = False,
    enabled_information_tools: frozenset[str] | None = None,
    episode_progress: EpisodeProgress | None = None,
    checkpoint_callback: Callable[[EpisodeProgress], None] | None = None,
) -> dict[str, Any]:
    query = get_query_text(query_row)
    query_id = str(query_row.get("query_id") or query_row.get("path_id") or "")
    gold_page_id = get_gold_page_id(query_row)
    gold_page_handle = runtime.handle_space.page_id_to_handle(gold_page_id)
    enabled_tools = (
        INFORMATION_TOOLS
        if enabled_information_tools is None
        else frozenset(enabled_information_tools)
    )

    progress = episode_progress or EpisodeProgress()
    observations = observations_from_tool_trace(runtime.trace)
    model_outputs = list(progress.model_outputs)
    repair_used = progress.repair_used
    pending_repair = progress.pending_repair
    transport_failures = list(progress.transport_failures)

    def persist_progress(
        next_step: int, transport_failure: dict[str, Any] | None = None
    ) -> None:
        if checkpoint_callback is None:
            return
        if transport_failure is not None:
            transport_failures.append(transport_failure)
        checkpoint_callback(
            EpisodeProgress(
                next_step=next_step,
                model_outputs=list(model_outputs),
                repair_used=repair_used,
                pending_repair=pending_repair,
                transport_failures=list(transport_failures),
                resume_count=progress.resume_count,
            )
        )

    if dry_run_tools_only:
        if progress.next_step != 1 or runtime.trace:
            raise ValueError("dry-run tools-only mode cannot resume a partial episode")
        if "visual_search" not in enabled_tools:
            raise ValueError(
                "dry-run tools-only mode requires visual_search to be enabled"
            )
        observation, _ = runtime.visual_search({"query": query, "top_k": 10})
        observations.append({"tool": "visual_search", "observation": observation})
        ranked_answers = [
            {
                "page_handle": item["page_handle"],
                "evidence_page_handles": [],
                "rationale": "dry-run visual-search rank",
            }
            for item in observation.get("results", [])[:10]
        ]
        submission = {
            "ranked_answers": ranked_answers,
            "overall_rationale": "dry-run visual_search only",
        }
        metrics = score_submission(submission, gold_page_handle)
        return {
            "schema_version": SCHEMA_VERSION,
            "harness_version": HARNESS_VERSION,
            "query_id": query_id,
            "level": query_row.get("level", ""),
            "topic_id": query_row.get("topic_id", ""),
            "query": query,
            "gold_page_id": gold_page_id,
            "gold_page_handle": gold_page_handle,
            "submission": submission,
            "metrics": metrics,
            "tool_trace": runtime.trace,
            "model_outputs": model_outputs,
            "status": "dry_run_submitted",
        }

    submission: dict[str, Any] | None = None
    status = "max_steps_exceeded"
    for step in range(progress.next_step, max_steps + 1):
        context_artifact_handles = runtime.context_artifact_handles(
            max_images=MAX_VIEW_IMAGES_PER_CALL
        )
        prompt = build_episode_prompt(
            query=query,
            runtime=runtime,
            observations=observations,
            current_step=step,
            max_steps=max_steps,
            format_repair=pending_repair,
            context_artifact_handles=context_artifact_handles,
            enabled_information_tools=enabled_tools,
        )
        image_paths = runtime.artifact_paths(context_artifact_handles)
        request_id = f"agent:{query_id}:step{step}"
        try:
            with step_wall_timeout(step_wall_timeout_sec, request_id):
                response = chat_completion_json(
                    config,
                    text_prompt=prompt,
                    image_paths=image_paths,
                    max_tokens=config.max_tokens,
                    request_id=request_id,
                )
        except StepWallTimeoutError as exc:
            status = "step_wall_timeout"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(
                {
                    "step": step,
                    "phase": "planner",
                    "action": {"action": "step_wall_timeout"},
                    "raw_response": "",
                    "usage": {},
                    "attempts": 0,
                    "request_attempts": [],
                    "response_accepted": False,
                    "event_type": "step_wall_timeout",
                    "error": str(exc),
                }
            )
            persist_progress(
                step,
                {
                    "status": status,
                    "step": step,
                    "phase": "planner",
                    "attempts": 0,
                    "error": str(exc),
                },
            )
            print(f"[timeout] {exc}", flush=True)
            break
        except VLMFormatError as exc:
            model_outputs.append(format_error_output(exc=exc, step=step))
            pending_repair = (
                "The previous planner response could not be parsed as one JSON "
                "object. Return exactly one valid JSON action."
            )
            persist_progress(step + 1)
            print(f"[format_error] {request_id}: {exc}", flush=True)
            continue
        except VLMProviderError as exc:
            status = "provider_error"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(provider_error_output(exc=exc, step=step))
            print(f"[provider_error] {request_id}: {exc}", flush=True)
            break
        except VLMRetryExhaustedError as exc:
            status = "planner_retry_exhausted"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(retry_exhausted_output(exc=exc, step=step))
            persist_progress(
                step,
                {
                    "status": status,
                    "step": step,
                    "phase": "planner",
                    "attempts": exc.attempts,
                    "request_attempts": exc.request_attempts,
                    "error": str(exc),
                },
            )
            print(f"[retry_exhausted] {exc}", flush=True)
            break
        action = {
            key: value for key, value in response.items() if not key.startswith("_")
        }
        model_outputs.append(
            accepted_response_output(response=response, action=action, step=step)
        )
        pending_repair = ""
        action_name = str(action.get("action", "")).strip()
        if action_name == "submit_answer":
            normalized, error = normalize_submit(action, runtime)
            if normalized is None:
                if allow_repair_turn and not repair_used:
                    repair_used = True
                    pending_repair = error
                    persist_progress(step + 1)
                    continue
                status = "invalid_submission"
                submission = {
                    "ranked_answers": [],
                    "overall_rationale": f"invalid submission: {error}",
                }
            else:
                status = "submitted"
                submission = normalized
            break
        if action_name not in enabled_tools:
            if allow_repair_turn and not repair_used:
                repair_used = True
                pending_repair = f"Unknown or missing action: {action_name}"
                persist_progress(step + 1)
                continue
            status = "invalid_action"
            submission = {
                "ranked_answers": [],
                "overall_rationale": f"invalid action: {action_name}",
            }
            break

        arguments = action.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        observation, artifacts = runtime.call_tool(action_name, arguments)
        observations.append(
            {
                "tool": action_name,
                "arguments": arguments,
                "observation": observation,
                "artifacts": artifacts,
            }
        )
        persist_progress(step + 1)

    if (
        submission is None
        and status == "max_steps_exceeded"
        and force_final_answer_after_max_steps
    ):
        context_artifact_handles = runtime.context_artifact_handles(
            max_images=MAX_VIEW_IMAGES_PER_CALL
        )
        prompt = build_finalization_prompt(
            query=query,
            runtime=runtime,
            observations=observations,
            max_steps=max_steps,
            format_repair=(
                pending_repair
                if model_outputs
                and model_outputs[-1].get("event_type") == "format_error"
                else ""
            ),
            context_artifact_handles=context_artifact_handles,
            enabled_information_tools=enabled_tools,
        )
        image_paths = runtime.artifact_paths(context_artifact_handles)
        request_id = f"agent:{query_id}:finalize"
        try:
            with step_wall_timeout(step_wall_timeout_sec, request_id):
                response = chat_completion_json(
                    config,
                    text_prompt=prompt,
                    image_paths=image_paths,
                    max_tokens=config.max_tokens,
                    request_id=request_id,
                )
        except StepWallTimeoutError as exc:
            status = "forced_submit_timeout"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(
                {
                    "step": max_steps + 1,
                    "phase": "finalization",
                    "action": {"action": "step_wall_timeout"},
                    "raw_response": "",
                    "usage": {},
                    "attempts": 0,
                    "request_attempts": [],
                    "response_accepted": False,
                    "event_type": "step_wall_timeout",
                    "error": str(exc),
                }
            )
            persist_progress(
                max_steps + 1,
                {
                    "status": status,
                    "step": max_steps + 1,
                    "phase": "finalization",
                    "attempts": 0,
                    "error": str(exc),
                },
            )
            print(f"[timeout] {exc}", flush=True)
        except VLMFormatError as exc:
            status = "forced_submit_format_error"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(
                format_error_output(
                    exc=exc, step=max_steps + 1, phase="finalization"
                )
            )
            print(f"[format_error] {request_id}: {exc}", flush=True)
        except VLMProviderError as exc:
            status = "provider_error"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(
                provider_error_output(
                    exc=exc, step=max_steps + 1, phase="finalization"
                )
            )
            print(f"[provider_error] {request_id}: {exc}", flush=True)
        except VLMRetryExhaustedError as exc:
            status = "forced_submit_retry_exhausted"
            submission = {"ranked_answers": [], "overall_rationale": str(exc)}
            model_outputs.append(
                retry_exhausted_output(
                    exc=exc, step=max_steps + 1, phase="finalization"
                )
            )
            persist_progress(
                max_steps + 1,
                {
                    "status": status,
                    "step": max_steps + 1,
                    "phase": "finalization",
                    "attempts": exc.attempts,
                    "request_attempts": exc.request_attempts,
                    "error": str(exc),
                },
            )
            print(f"[retry_exhausted] {exc}", flush=True)
        else:
            action = {
                key: value for key, value in response.items() if not key.startswith("_")
            }
            model_outputs.append(
                accepted_response_output(
                    response=response,
                    action=action,
                    step=max_steps + 1,
                    phase="finalization",
                )
            )
            action_name = str(action.get("action", "")).strip()
            if action_name != "submit_answer":
                status = "forced_submit_invalid_action"
                submission = {
                    "ranked_answers": [],
                    "overall_rationale": f"finalization returned invalid action: {action_name}",
                }
            else:
                normalized, error = normalize_submit(
                    action, runtime, allow_fewer_than_ten=True
                )
                if normalized is None:
                    status = "forced_submit_invalid_submission"
                    submission = {
                        "ranked_answers": [],
                        "overall_rationale": f"invalid forced submission: {error}",
                    }
                else:
                    status = "forced_submit_after_max_steps"
                    submission = normalized

    if submission is None:
        submission = {"ranked_answers": [], "overall_rationale": status}
    metrics = score_submission(submission, gold_page_handle)
    result = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "query_id": query_id,
        "level": query_row.get("level", ""),
        "topic_id": query_row.get("topic_id", ""),
        "query": query,
        "gold_page_id": gold_page_id,
        "gold_page_handle": gold_page_handle,
        "submission": submission,
        "metrics": metrics,
        "tool_trace": runtime.trace,
        "model_outputs": model_outputs,
        "status": status,
        "tool_usage": runtime.public_usage(),
        "request_audit": summarize_model_outputs(model_outputs),
    }
    if progress.resume_count or transport_failures:
        result["transport_recovery"] = {
            "checkpoint_resumes": progress.resume_count,
            "failures": transport_failures,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "data/benchmark/queries.jsonl",
    )
    parser.add_argument(
        "--evaluator-annotations",
        type=Path,
        default=PROJECT_ROOT / "data/benchmark/evaluator_annotations.jsonl",
        help="Gold annotations used only for scoring and never exposed to the planner.",
    )
    parser.add_argument(
        "--page-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/corpus/pages.jsonl",
    )
    parser.add_argument(
        "--page-embeddings",
        type=Path,
        default=PROJECT_ROOT / "data/derived/qwen3_vl_8b_page_embeddings.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/agent_visual",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Append to an existing run_* directory and skip query_ids already present in predictions.jsonl.",
    )
    parser.add_argument(
        "--resume-predictions",
        type=Path,
        default=None,
        help="Seed this run with completed rows from an existing predictions.jsonl file.",
    )
    parser.add_argument(
        "--resume-partial-episodes",
        action="store_true",
        default=False,
        help=(
            "Opt in to atomic per-step checkpoints. Transport retry exhaustion "
            "resumes the same unobserved step without changing the agent step budget."
        ),
    )
    parser.add_argument(
        "--episode-checkpoint-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory; defaults to RUN_DIR/episode_checkpoints.",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/api.json"
    )
    parser.add_argument(
        "--planner-model",
        default="",
        help="Planner model override; pass an empty value to use the model from --config.",
    )
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--embedding-model", default="Qwen3-VL-Embedding-8B")
    parser.add_argument(
        "--embedding-service-revision",
        default=os.environ.get("EMBEDDING_SERVICE_REVISION", ""),
        help="Optional immutable revision label for a mutable embedding endpoint.",
    )
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-timeout", type=float, default=300.0)
    parser.add_argument("--embedding-max-retries", type=int, default=20)
    parser.add_argument("--embedding-retry-backoff", type=float, default=5.0)
    parser.add_argument("--page-matrix-cache", type=Path, default=None)
    parser.add_argument("--handle-seed", type=int, default=20260707)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-id", default="")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of query episodes to run concurrently.",
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--step-wall-timeout-sec",
        type=float,
        default=0.0,
        help="Reserved compatibility option; nonzero values are rejected. Use --timeout for planner requests.",
    )
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff", type=float, default=None)
    parser.add_argument("--allow-repair-turn", action="store_true", default=True)
    parser.add_argument(
        "--no-allow-repair-turn", action="store_false", dest="allow_repair_turn"
    )
    parser.add_argument(
        "--force-final-answer-after-max-steps", action="store_true", default=True
    )
    parser.add_argument(
        "--no-force-final-answer-after-max-steps",
        action="store_false",
        dest="force_final_answer_after_max_steps",
    )
    parser.add_argument("--dry-run-tools-only", action="store_true", default=False)
    parser.add_argument("--ocr-backend", default="paddleocr_vl")
    parser.add_argument(
        "--paddleocr-model-dir",
        default=str(PROJECT_ROOT / "models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument(
        "--paddleocr-layout-model-dir",
        default=str(PROJECT_ROOT / "models/PP-DocLayoutV3"),
    )
    parser.add_argument("--paddleocr-pipeline-version", default="v1.6")
    parser.add_argument("--paddleocr-vl-rec-server-url", default="")
    parser.add_argument(
        "--ablation-without-ocr",
        action="store_true",
        default=False,
        help="Remove ocr_image from the planner prompt, legal action set, and runtime for a controlled ablation.",
    )
    parser.add_argument(
        "--ablation-without-crop",
        action="store_true",
        default=False,
        help="Remove crop_pages from the planner prompt, legal action set, and runtime for a controlled ablation.",
    )
    parser.add_argument(
        "--ablation-without-inspect",
        action="store_true",
        default=False,
        help="Remove inspect_pages and its dependent crop_pages tool while retaining search and OCR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.step_wall_timeout_sec != 0:
        raise ValueError(
            "--step-wall-timeout-sec is unsupported because query episodes run in "
            "worker threads; use --timeout for planner requests"
        )
    disabled_information_tools: set[str] = set()
    if args.ablation_without_ocr:
        disabled_information_tools.add("ocr_image")
        args.ocr_backend = "disabled"
    if args.ablation_without_crop:
        disabled_information_tools.add("crop_pages")
    if args.ablation_without_inspect:
        disabled_information_tools.update({"inspect_pages", "crop_pages"})
    enabled_information_tools = frozenset(
        INFORMATION_TOOLS - disabled_information_tools
    )
    ensure_dir(args.out_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        ensure_dir(args.resume_run_dir)
        if args.resume_run_dir is not None
        else ensure_dir(args.out_dir / f"run_{run_id}")
    )
    log_path = install_run_log(run_dir)
    print(
        f"[tools] enabled={sorted(enabled_information_tools)} "
        f"disabled={sorted(disabled_information_tools)}",
        flush=True,
    )
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.json"
    episode_checkpoint_dir = None
    if args.resume_partial_episodes:
        episode_checkpoint_dir = ensure_dir(
            args.episode_checkpoint_dir or (run_dir / "episode_checkpoints")
        )
    query_cache_path = (
        ensure_dir(args.out_dir / "cache") / "visual_search_query_embeddings.jsonl"
    )
    page_matrix_cache_path = args.page_matrix_cache or (
        ensure_dir(args.out_dir / "cache") / "page_image_embedding_matrix.npz"
    )

    handle_space = load_handle_space(args.page_manifest, args.handle_seed)
    page_images_identity = file_collection_identity(
        (page.page_id, page.image_path) for page in handle_space.pages
    )
    ocr_uses_local_models = (
        "ocr_image" in enabled_information_tools
        and args.ocr_backend.strip().lower() not in {"none", "disabled"}
    )
    ocr_model_identity = (
        directory_identity(Path(args.paddleocr_model_dir))
        if ocr_uses_local_models
        else {"used": False}
    )
    ocr_layout_model_identity = (
        directory_identity(Path(args.paddleocr_layout_model_dir))
        if ocr_uses_local_models
        else {"used": False}
    )
    write_jsonl(
        run_dir / "page_handle_mapping.internal.jsonl",
        handle_space.public_mapping_rows(),
    )

    retriever = VisualRetriever(
        page_embeddings_path=args.page_embeddings,
        handle_space=handle_space,
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        cache_path=query_cache_path,
        timeout=args.embedding_timeout,
        max_retries=args.embedding_max_retries,
        retry_backoff=args.embedding_retry_backoff,
        matrix_cache_path=page_matrix_cache_path,
        service_revision=args.embedding_service_revision,
    )
    ocr_backend = build_ocr_backend(args)
    config = resolve_vlm_config(
        args.config,
        model=args.planner_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        max_tokens=args.max_tokens,
    )

    queries = load_benchmark_rows(args.queries, args.evaluator_annotations)
    if args.query_id:
        queries = [
            row
            for row in queries
            if str(row.get("query_id") or row.get("path_id")) == args.query_id
        ]
        if not queries:
            raise ValueError(f"No query matched --query-id {args.query_id}")
    if args.limit > 0:
        queries = queries[: args.limit]

    def checkpoint_fingerprint_for(query_row: dict[str, Any]) -> dict[str, Any]:
        return episode_checkpoint_fingerprint(
            query_row=query_row,
            config=config,
            max_steps=args.max_steps,
            allow_repair_turn=args.allow_repair_turn,
            force_final_answer_after_max_steps=args.force_final_answer_after_max_steps,
            enabled_information_tools=enabled_information_tools,
            handle_seed=args.handle_seed,
            page_manifest=args.page_manifest,
            page_embeddings=args.page_embeddings,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_service_revision=args.embedding_service_revision,
            page_images_identity=page_images_identity,
            ocr_backend=args.ocr_backend,
            paddleocr_model_dir=args.paddleocr_model_dir,
            paddleocr_layout_model_dir=args.paddleocr_layout_model_dir,
            ocr_model_identity=ocr_model_identity,
            ocr_layout_model_identity=ocr_layout_model_identity,
            paddleocr_pipeline_version=args.paddleocr_pipeline_version,
            paddleocr_vl_rec_server_url=args.paddleocr_vl_rec_server_url,
            step_wall_timeout_sec=args.step_wall_timeout_sec,
            dry_run_tools_only=args.dry_run_tools_only,
        )

    query_row_by_id = {
        query_id_for_row(query_row, index): query_row
        for index, query_row in enumerate(queries, start=1)
    }
    query_ids = {
        query_id_for_row(query_row, index)
        for index, query_row in enumerate(queries, start=1)
    }
    episode_fingerprints = {
        query_id_for_row(query_row, index): canonical_digest(
            checkpoint_fingerprint_for(query_row)
        )
        for index, query_row in enumerate(queries, start=1)
    }
    resume_predictions_path = args.resume_predictions
    if resume_predictions_path is None and args.resume_run_dir is not None:
        resume_predictions_path = run_dir / "predictions.jsonl"
    gold_page_handles: dict[str, str] = {}
    valid_page_handles: set[str] = set()
    if resume_predictions_path is not None:
        valid_page_handles = set(handle_space.by_handle)
        for query_id, query_row in query_row_by_id.items():
            try:
                gold_page_handles[query_id] = handle_space.page_id_to_handle(
                    get_gold_page_id(query_row)
                )
            except KeyError:
                pass
    if (
        resume_predictions_path is not None
        and resume_predictions_path.resolve() == predictions_path.resolve()
    ):
        assert_inplace_resume_scope(
            resume_predictions_path,
            queries,
            expected_fingerprints=episode_fingerprints,
        )
    resumed_by_query_id = (
        load_resume_rows(
            resume_predictions_path,
            query_ids,
            current_query_rows=query_row_by_id,
            expected_gold_page_handles=gold_page_handles,
            valid_page_handles=valid_page_handles,
            required_harness_version=HARNESS_VERSION,
            expected_fingerprints=episode_fingerprints,
        )
        if resume_predictions_path is not None
        else {}
    )

    if (
        args.resume_partial_episodes
        and episode_checkpoint_dir is not None
        and resume_predictions_path is not None
        and resume_predictions_path.exists()
    ):
        bootstrapped = 0
        for row in read_jsonl(resume_predictions_path):
            query_id = str(row.get("query_id") or "")
            query_row = query_row_by_id.get(query_id)
            if query_row is None or query_id in resumed_by_query_id:
                continue
            if str(row.get("status") or "") not in RECOVERABLE_EPISODE_STATUSES:
                continue
            if (
                row.get("harness_version") != HARNESS_VERSION
                or str(row.get("query") or "") != get_query_text(query_row)
                or str(row.get("gold_page_id") or "") != get_gold_page_id(query_row)
                or str(row.get("episode_fingerprint_sha256") or "")
                != episode_fingerprints[query_id]
            ):
                print(
                    f"[checkpoint] skip incompatible legacy partial row {query_id}",
                    flush=True,
                )
                continue
            row_ablation = row.get("ablation")
            if row_ablation is None:
                row_enabled_tools = INFORMATION_TOOLS
            elif isinstance(row_ablation, dict):
                row_enabled_tools = frozenset(
                    str(value)
                    for value in row_ablation.get("enabled_information_tools", [])
                )
            else:
                row_enabled_tools = frozenset()
            if row_enabled_tools != enabled_information_tools:
                print(
                    f"[checkpoint] skip legacy partial row from a different ablation {query_id}",
                    flush=True,
                )
                continue
            path = checkpoint_path(episode_checkpoint_dir, query_id)
            fingerprint = checkpoint_fingerprint_for(query_row)
            if path.exists():
                load_episode_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    max_steps=args.max_steps,
                )
                continue
            try:
                progress, legacy_runtime_state = recoverable_progress_from_prediction(
                    row,
                    max_steps=args.max_steps,
                    enabled_tools=enabled_information_tools,
                )
            except EpisodeCheckpointError as exc:
                print(
                    f"[checkpoint] cannot bootstrap {query_id}: {exc}", flush=True
                )
                continue
            write_episode_checkpoint(
                path,
                fingerprint=fingerprint,
                progress=progress,
                runtime_state=None,
                legacy_runtime_state=legacy_runtime_state,
            )
            bootstrapped += 1
            print(
                f"[checkpoint] bootstrapped {query_id} at step {progress.next_step}",
                flush=True,
            )
        if bootstrapped:
            print(
                f"[checkpoint] bootstrapped {bootstrapped} partial episodes from "
                f"{resume_predictions_path}",
                flush=True,
            )
    if resumed_by_query_id:
        print(
            f"[resume] loaded {len(resumed_by_query_id)} completed rows from {resume_predictions_path}",
            flush=True,
        )

    query_order = [
        query_id_for_row(query_row, index)
        for index, query_row in enumerate(queries, start=1)
    ]
    row_by_query_id: dict[str, dict[str, Any]] = dict(resumed_by_query_id)

    def ordered_completed_rows() -> list[dict[str, Any]]:
        return [
            row_by_query_id[query_id]
            for query_id in query_order
            if query_id in row_by_query_id
        ]

    if row_by_query_id:
        write_jsonl(predictions_path, ordered_completed_rows())

    def run_one(index: int, query_row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        query_id = query_id_for_row(query_row, index)
        print(f"[agent] start {index}/{len(queries)} {query_id}", flush=True)
        checkpoint_file = (
            checkpoint_path(episode_checkpoint_dir, query_id)
            if episode_checkpoint_dir is not None
            else None
        )
        fingerprint = checkpoint_fingerprint_for(query_row)
        progress = EpisodeProgress()
        restored_runtime_state: dict[str, Any] | None = None
        restored_legacy_state: dict[str, Any] | None = None
        if checkpoint_file is not None and checkpoint_file.exists():
            progress, restored_runtime_state, restored_legacy_state = (
                load_episode_checkpoint(
                    checkpoint_file,
                    expected_fingerprint=fingerprint,
                    max_steps=args.max_steps,
                )
            )
            progress.resume_count += 1
            print(
                f"[checkpoint] resume {query_id} from step {progress.next_step} "
                f"resume_count={progress.resume_count}",
                flush=True,
            )

        runtime = AgentToolRuntime(
            handle_space=handle_space,
            retriever=retriever,
            ocr_backend=ocr_backend,
            artifact_dir=run_dir / "artifacts" / query_id / f"invocation_{run_id}",
            enabled_tools=enabled_information_tools,
        )
        if restored_runtime_state is not None:
            runtime.restore_checkpoint_state(restored_runtime_state)
        elif restored_legacy_state is not None:
            runtime.restore_legacy_tool_trace(
                restored_legacy_state.get("tool_trace", []),
                restored_legacy_state.get("tool_usage", {}),
            )

        checkpoint_callback: Callable[[EpisodeProgress], None] | None = None
        if checkpoint_file is not None:

            def save_progress(value: EpisodeProgress) -> None:
                write_episode_checkpoint(
                    checkpoint_file,
                    fingerprint=fingerprint,
                    progress=value,
                    runtime_state=runtime.checkpoint_state(),
                )

            checkpoint_callback = save_progress
            checkpoint_callback(progress)
        try:
            row = run_episode(
                query_row=query_row,
                runtime=runtime,
                config=config,
                max_steps=args.max_steps,
                allow_repair_turn=args.allow_repair_turn,
                force_final_answer_after_max_steps=args.force_final_answer_after_max_steps,
                step_wall_timeout_sec=args.step_wall_timeout_sec,
                dry_run_tools_only=args.dry_run_tools_only,
                enabled_information_tools=enabled_information_tools,
                episode_progress=progress if checkpoint_file is not None else None,
                checkpoint_callback=checkpoint_callback,
            )
        except Exception as exc:
            trace_text = traceback.format_exc()
            print(
                f"[worker_exception] agent:{query_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(trace_text, flush=True)
            row = worker_exception_row(
                query_row=query_row,
                handle_space=handle_space,
                exc=exc,
                trace_text=trace_text,
                runtime=runtime,
            )

        if (
            checkpoint_file is not None
            and row.get("status") not in RECOVERABLE_EPISODE_STATUSES
        ):
            remove_episode_checkpoint(checkpoint_file)
        row["harness_version"] = HARNESS_VERSION
        row["episode_fingerprint_sha256"] = canonical_digest(fingerprint)
        row.setdefault(
            "request_audit", summarize_model_outputs(row.get("model_outputs", []))
        )
        if disabled_information_tools:
            row["ablation"] = {
                "enabled_information_tools": sorted(enabled_information_tools),
                "disabled_information_tools": sorted(disabled_information_tools),
            }
        return query_id, row

    pending: list[tuple[int, dict[str, Any]]] = []
    for index, query_row in enumerate(queries, start=1):
        query_id = query_id_for_row(query_row, index)
        if query_id in row_by_query_id:
            print(f"[resume] skip {index}/{len(queries)} {query_id}", flush=True)
        else:
            pending.append((index, query_row))

    print(
        f"[parallel] num_workers={args.num_workers} pending={len(pending)} resumed={len(row_by_query_id)} total={len(queries)}",
        flush=True,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_meta = {
                executor.submit(run_one, index, query_row): (index, query_row)
                for index, query_row in pending
            }
            for future in as_completed(future_to_meta):
                index, query_row = future_to_meta[future]
                try:
                    query_id, row = future.result()
                except Exception as exc:
                    query_id = query_id_for_row(query_row, index)
                    trace_text = traceback.format_exc()
                    print(
                        f"[future_exception] agent:{query_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(trace_text, flush=True)
                    row = worker_exception_row(
                        query_row=query_row,
                        handle_space=handle_space,
                        exc=exc,
                        trace_text=trace_text,
                    )
                row.setdefault("harness_version", HARNESS_VERSION)
                row.setdefault(
                    "episode_fingerprint_sha256", episode_fingerprints[query_id]
                )
                row.setdefault(
                    "request_audit",
                    summarize_model_outputs(row.get("model_outputs", [])),
                )
                row_by_query_id[query_id] = row
                rows_now = ordered_completed_rows()
                write_jsonl(predictions_path, rows_now)
                print(
                    f"[agent] done {index}/{len(queries)} {query_id} status={row.get('status')} "
                    f"completed={len(rows_now)}/{len(queries)}",
                    flush=True,
                )

    rows = ordered_completed_rows()
    metrics = build_metric_report(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "input_files": {
            "queries": str(args.queries),
            "evaluator_annotations": str(args.evaluator_annotations),
            "page_manifest": str(args.page_manifest),
            "page_embeddings": str(args.page_embeddings),
            "config": str(args.config),
        },
        "output_files": {
            "predictions": str(predictions_path),
            "report": str(report_path),
        },
        "log_file": str(log_path),
        "num_queries": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "valid_top10_submissions": sum(
            len((row.get("submission") or {}).get("ranked_answers") or []) == 10
            for row in rows
        ),
        "num_workers": args.num_workers,
        "handle_seed": args.handle_seed,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_service_revision": args.embedding_service_revision,
        "page_matrix_cache": str(page_matrix_cache_path),
        "ocr_backend": args.ocr_backend,
        "paddleocr_model_dir": args.paddleocr_model_dir,
        "paddleocr_layout_model_dir": args.paddleocr_layout_model_dir,
        "paddleocr_pipeline_version": args.paddleocr_pipeline_version,
        "vlm_config": public_config_summary(config),
        "force_final_answer_after_max_steps": args.force_final_answer_after_max_steps,
        "step_wall_timeout_sec": args.step_wall_timeout_sec,
        "response_policy": {
            "transport_policy_version": TRANSPORT_POLICY_VERSION,
            "json_parse_policy_version": JSON_PARSE_POLICY_VERSION,
            "format_errors_consume_agent_steps": True,
            "whole_query_restarts": False,
        },
        "request_audit": aggregate_request_audits(rows),
        "resumed_rows": len(resumed_by_query_id),
        "resume_predictions": str(resume_predictions_path)
        if resume_predictions_path is not None
        else "",
        "shared_step_policy": {
            "max_steps": args.max_steps,
            "per_tool_cumulative_call_limits": False,
            "forced_answer_only_finalization": args.force_final_answer_after_max_steps,
        },
        "per_call_limits": {
            "max_visual_search_top_k": MAX_SEARCH_RESULTS,
            "max_inspect_pages_per_call": MAX_VIEW_IMAGES_PER_CALL,
            "max_crop_regions_per_call": MAX_VIEW_IMAGES_PER_CALL,
        },
        "crop_pages_protocol": {
            "version": CROP_PROTOCOL_VERSION,
            "enabled": "crop_pages" in enabled_information_tools,
            "per_region_purpose_prompted": True,
            "missing_purpose_blocks_execution": False,
            "partial_batch_success": True,
        },
        "metrics": metrics,
    }
    if disabled_information_tools:
        report["ablation"] = {
            "enabled_information_tools": sorted(enabled_information_tools),
            "disabled_information_tools": sorted(disabled_information_tools),
            "without_ocr": args.ablation_without_ocr,
            "without_crop": args.ablation_without_crop,
            "without_inspect": args.ablation_without_inspect,
        }
    if episode_checkpoint_dir is not None:
        report["partial_episode_recovery"] = {
            "enabled": True,
            "checkpoint_dir": str(episode_checkpoint_dir),
            "recoverable_statuses": sorted(RECOVERABLE_EPISODE_STATUSES),
            "active_checkpoints": len(list(episode_checkpoint_dir.glob("*.json"))),
            "contract": "resume_same_unobserved_planner_step_v2",
        }
    write_json(report_path, report)
    print(f"[done] wrote predictions to {predictions_path}")
    print(f"[done] wrote report to {report_path}")
    print(short_json(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
