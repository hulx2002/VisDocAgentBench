#!/usr/bin/env python3
"""Fixed top-30 retrieval followed by non-iterative VLM assessment and reranking."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    file_collection_identity,
    file_identity,
    gold_page_id,
    opaque_handle_maps,
    query_text,
    read_jsonl,
    resolve_page_image,
    safe_float,
    sha256_json,
    summarize_predictions,
    write_json,
    write_jsonl,
)


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
    resolve_vlm_config,
)


SCHEMA_VERSION = "VisDocAgentBenchFixedTwoStageRerankerV2"
ACCEPTED_STATUS = "submitted"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def effective_image_max_side() -> int:
    value = os.environ.get("VLM_IMAGE_MAX_SIDE", "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else 0


def usage_counts(usage: dict[str, Any]) -> tuple[int, int, int]:
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens


def public_response(response: dict[str, Any]) -> dict[str, Any]:
    parsed = {key: value for key, value in response.items() if not key.startswith("_")}
    result = {
        "parsed": parsed,
        "raw_response": response.get("_raw_response", ""),
        "reasoning_content": response.get("_reasoning_content", ""),
        "usage": response.get("_usage", {}),
        "attempts": response.get("_attempts", 1),
        "request_attempts": response.get("_request_attempts", []),
        "response_accepted": True,
        "event_type": "accepted_response",
    }
    if response.get("_parse_warning"):
        result["parse_warning"] = response["_parse_warning"]
        result["ignored_trailing_content"] = response.get(
            "_extra_json_ignored", ""
        )
    return result


def rejected_response(error: Exception) -> dict[str, Any]:
    if isinstance(error, VLMFormatError):
        return {
            "parsed": {},
            "raw_response": error.raw_response,
            "reasoning_content": error.reasoning_content,
            "usage": error.usage,
            "attempts": error.attempts,
            "request_attempts": error.request_attempts,
            "response_accepted": False,
            "event_type": "format_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if isinstance(error, VLMProviderError):
        return {
            "parsed": {},
            "raw_response": error.response_body,
            "reasoning_content": "",
            "usage": {},
            "attempts": error.attempts,
            "request_attempts": error.request_attempts,
            "response_accepted": False,
            "event_type": "provider_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if isinstance(error, VLMRetryExhaustedError):
        return {
            "parsed": {},
            "raw_response": "",
            "reasoning_content": "",
            "usage": {},
            "attempts": error.attempts,
            "request_attempts": error.request_attempts,
            "response_accepted": False,
            "event_type": "transport_failure",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "parsed": {},
        "raw_response": "",
        "reasoning_content": "",
        "usage": {},
        "attempts": 0,
        "request_attempts": [],
        "response_accepted": False,
        "event_type": "local_validation_failure",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def summarize_call_trace(call_trace: list[dict[str, Any]]) -> dict[str, int]:
    responses = [
        call.get("response", {})
        for call in call_trace
        if isinstance(call.get("response"), dict)
    ]
    attempts = [
        attempt
        for response in responses
        for attempt in response.get("request_attempts", [])
        if isinstance(attempt, dict)
    ]
    return {
        "logical_vlm_calls": len(call_trace),
        "physical_request_attempts": sum(
            int(response.get("attempts", 0)) for response in responses
        ),
        "transport_retries": sum(
            bool(attempt.get("retry_scheduled")) for attempt in attempts
        ),
        "format_failures": sum(
            response.get("event_type") in {"format_error", "schema_error"}
            for response in responses
        ),
        "schema_failures": sum(
            response.get("event_type") == "schema_error" for response in responses
        ),
        "provider_failures": sum(
            response.get("event_type") == "provider_error" for response in responses
        ),
        "accepted_responses": sum(
            response.get("response_accepted") is True for response in responses
        ),
        "rejected_responses": sum(
            response.get("response_accepted") is False for response in responses
        ),
    }


def mark_schema_rejection(response: dict[str, Any], error: ValueError) -> None:
    response.update(
        {
            "response_accepted": False,
            "event_type": "schema_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def cached_ocr_text(row: dict[str, Any]) -> str:
    """Read OCR text from either the raw cache or its prepared corpus view."""
    return str(
        row.get("ocr_text") or row.get("embedding_text") or row.get("text") or ""
    )


def validate_batch_assessment(
    response: dict[str, Any], expected_handles: list[str]
) -> list[dict[str, Any]]:
    assessments = response.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(expected_handles):
        raise ValueError(
            f"expected {len(expected_handles)} assessments, got "
            f"{len(assessments) if isinstance(assessments, list) else type(assessments).__name__}"
        )
    by_handle: dict[str, dict[str, Any]] = {}
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("assessment is not an object")
        handle = str(assessment.get("page_handle") or "")
        if handle not in expected_handles or handle in by_handle:
            raise ValueError(f"unexpected or duplicate assessed page handle: {handle}")
        score = safe_float(assessment.get("relevance_score"), float("nan"))
        if not 0.0 <= score <= 100.0:
            raise ValueError(f"invalid relevance_score for {handle}: {assessment.get('relevance_score')}")
        by_handle[handle] = {
            "page_handle": handle,
            "relevance_score": score,
            "visual_evidence": str(assessment.get("visual_evidence") or ""),
            "textual_evidence": str(assessment.get("textual_evidence") or ""),
            "constraint_analysis": str(assessment.get("constraint_analysis") or ""),
            "rationale": str(assessment.get("rationale") or ""),
        }
    return [by_handle[handle] for handle in expected_handles]


def validate_aggregate(
    response: dict[str, Any], allowed_handles: set[str]
) -> tuple[list[str], list[dict[str, str]], str]:
    raw_answers = response.get("ranked_answers")
    if not isinstance(raw_answers, list) or len(raw_answers) != 10:
        raise ValueError("aggregate response must contain exactly 10 ranked_answers")
    handles: list[str] = []
    answers: list[dict[str, str]] = []
    for item in raw_answers:
        if isinstance(item, str):
            handle = item
            rationale = ""
        elif isinstance(item, dict):
            handle = str(item.get("page_handle") or "")
            rationale = str(item.get("rationale") or "")
        else:
            raise ValueError("ranked_answers entries must be strings or objects")
        if handle not in allowed_handles:
            raise ValueError(f"aggregate response contains an unknown handle: {handle}")
        if handle in handles:
            raise ValueError(f"aggregate response contains a duplicate handle: {handle}")
        handles.append(handle)
        answers.append({"page_handle": handle, "rationale": rationale})
    return handles, answers, str(response.get("overall_rationale") or "")


def batch_prompt(query: str, batch: list[dict[str, Any]]) -> str:
    candidate_sections: list[str] = []
    for index, candidate in enumerate(batch, start=1):
        candidate_sections.append(
            "\n".join(
                [
                    f"Candidate {index}",
                    f"page_handle: {candidate['page_handle']}",
                    f"first_stage_rank: {candidate['first_stage_rank']}",
                    f"first_stage_score: {candidate['first_stage_score']:.8f}",
                    "cached_page_ocr:",
                    "<ocr>",
                    candidate["ocr_text"],
                    "</ocr>",
                ]
            )
        )
    return f"""You are assessing one fixed batch in a visually rich page-retrieval pipeline.
This is a non-interactive comparison: assess only the three supplied candidates. Do not search,
reformulate the query, request another page, or infer source-document identity from handles.

Retrieval query:
{query}

The three complete page images are attached after this text in the same order as the candidate
sections below. Use both the rendered image and its cached OCR. Judge whether each page satisfies
the complete query, including semantic, visual, layout, and relational constraints.

{chr(10).join(candidate_sections)}

Return one JSON object with exactly this structure:
{{
  "assessments": [
    {{
      "page_handle": "page_000000",
      "relevance_score": 0,
      "visual_evidence": "brief image-grounded evidence",
      "textual_evidence": "brief OCR-grounded evidence",
      "constraint_analysis": "which query constraints match or fail",
      "rationale": "concise overall assessment"
    }}
  ]
}}
Include each supplied page_handle exactly once. relevance_score must be a number from 0 to 100.
Return JSON only."""


def aggregate_prompt(query: str, candidates: list[dict[str, Any]]) -> str:
    payload = [
        {
            "page_handle": candidate["page_handle"],
            "first_stage_rank": candidate["first_stage_rank"],
            "first_stage_score": candidate["first_stage_score"],
            "assessment": candidate["assessment"],
        }
        for candidate in candidates
    ]
    return f"""Produce the final ranking for a fixed two-stage page-retrieval pipeline.
No further search or page inspection is allowed. Rank only the 30 candidates below using the
original query, first-stage evidence, and independent image-plus-OCR assessments.

Retrieval query:
{query}

Candidate assessments:
{json.dumps(payload, ensure_ascii=False)}

Return JSON only:
{{
  "ranked_answers": [
    {{"page_handle": "page_000000", "rationale": "brief rank-specific reason"}}
  ],
  "overall_rationale": "brief explanation of the ordering"
}}
ranked_answers must contain exactly 10 distinct page_handle values selected from the 30 candidates.
Prioritize rank-1 precision; do not diversify at the expense of the best answer."""


def fingerprint_for_query(
    query: dict[str, Any], first_stage: dict[str, Any], settings: dict[str, Any]
) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": query.get("query_id"),
            "query": query_text(query),
            "gold": gold_page_id(query),
            "first_stage_pages": first_stage.get("ranked_page_ids", []),
            "first_stage_scores": first_stage.get("ranked_scores", []),
            "settings": settings,
        }
    )


def process_query(
    *,
    query: dict[str, Any],
    first_stage: dict[str, Any],
    page_by_id: dict[str, dict[str, Any]],
    ocr_by_id: dict[str, str],
    page_to_handle: dict[str, str],
    handle_to_page: dict[str, str],
    page_manifest_path: Path,
    config: Any,
    settings: dict[str, Any],
    episode_path: Path,
) -> dict[str, Any]:
    query_id = str(query["query_id"])
    fingerprint = fingerprint_for_query(query, first_stage, settings)
    if episode_path.is_file():
        existing = json.loads(episode_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("status") == ACCEPTED_STATUS
            and len(existing.get("ranked_page_ids", [])) == 10
        ):
            print(f"[resume] {query_id}", flush=True)
            return existing

    first_stage_pages = [str(value) for value in first_stage.get("ranked_page_ids", [])]
    first_stage_scores = [safe_float(value) for value in first_stage.get("ranked_scores", [])]
    top_n = int(settings["candidate_count"])
    if len(first_stage_pages) < top_n or len(first_stage_scores) < top_n:
        raise ValueError(f"{query_id} has fewer than {top_n} first-stage candidates")
    first_stage_pages = first_stage_pages[:top_n]
    first_stage_scores = first_stage_scores[:top_n]
    if len(first_stage_pages) != len(set(first_stage_pages)):
        raise ValueError(f"{query_id} first-stage ranking contains duplicates")

    candidates: list[dict[str, Any]] = []
    for rank, (page_id, score) in enumerate(zip(first_stage_pages, first_stage_scores), start=1):
        page = page_by_id.get(page_id)
        if page is None or page_id not in ocr_by_id:
            raise ValueError(f"missing page or OCR data for {page_id}")
        candidates.append(
            {
                "page_id": page_id,
                "page_handle": page_to_handle[page_id],
                "first_stage_rank": rank,
                "first_stage_score": score,
                "image_path": resolve_page_image(page, page_manifest_path),
                "ocr_text": ocr_by_id[page_id],
            }
        )

    gold = gold_page_id(query)
    first_stage_gold_rank = (
        first_stage_pages.index(gold) + 1 if gold in first_stage_pages else None
    )
    call_trace: list[dict[str, Any]] = []
    active_call: dict[str, Any] | None = None
    error_trace = ""
    try:
        for batch_start in range(0, top_n, int(settings["batch_size"])):
            batch = candidates[batch_start : batch_start + int(settings["batch_size"])]
            handles = [candidate["page_handle"] for candidate in batch]
            active_call = {
                "phase": "candidate_assessment",
                "batch_index": batch_start // int(settings["batch_size"]) + 1,
                "page_handles": handles,
            }
            response = chat_completion_json(
                config,
                text_prompt=batch_prompt(query_text(query), batch),
                image_paths=[candidate["image_path"] for candidate in batch],
                max_tokens=int(settings["max_tokens"]),
                request_id=f"fixed_rerank:{settings['route']}:{query_id}:batch{batch_start // int(settings['batch_size']) + 1}",
            )
            active_call["response"] = public_response(response)
            try:
                assessments = validate_batch_assessment(response, handles)
            except ValueError as exc:
                mark_schema_rejection(active_call["response"], exc)
                call_trace.append(active_call)
                active_call = None
                raise
            call_trace.append(active_call)
            active_call = None
            for candidate, assessment in zip(batch, assessments):
                candidate["assessment"] = assessment

        active_call = {
            "phase": "fixed_aggregation",
            "page_handles": [],
        }
        response = chat_completion_json(
            config,
            text_prompt=aggregate_prompt(query_text(query), candidates),
            image_paths=[],
            max_tokens=int(settings["max_tokens"]),
            request_id=f"fixed_rerank:{settings['route']}:{query_id}:aggregate",
        )
        active_call["response"] = public_response(response)
        try:
            ranked_handles, ranked_answers, overall_rationale = validate_aggregate(
                response, {candidate["page_handle"] for candidate in candidates}
            )
        except ValueError as exc:
            mark_schema_rejection(active_call["response"], exc)
            call_trace.append(active_call)
            active_call = None
            raise
        call_trace.append(active_call)
        active_call = None
        ranked_page_ids = [handle_to_page[handle] for handle in ranked_handles]
        total_input = total_output = total_tokens = transport_attempts = 0
        for call in call_trace:
            recorded_response = call["response"]
            input_tokens, output_tokens, call_total = usage_counts(
                recorded_response.get("usage", {})
            )
            total_input += input_tokens
            total_output += output_tokens
            total_tokens += call_total
            transport_attempts += int(recorded_response.get("attempts", 0))
        row = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "query_id": query_id,
            "query": query_text(query),
            "gold_page_id": gold,
            "level": query.get("level"),
            "topic_id": query.get("topic_id"),
            "route": settings["route"],
            "status": ACCEPTED_STATUS,
            "first_stage_gold_rank": first_stage_gold_rank,
            "first_stage_candidates": [
                {
                    "page_handle": candidate["page_handle"],
                    "first_stage_rank": candidate["first_stage_rank"],
                    "first_stage_score": candidate["first_stage_score"],
                    "assessment": candidate["assessment"],
                }
                for candidate in candidates
            ],
            "ranked_page_ids": ranked_page_ids,
            "ranked_answers": ranked_answers,
            "overall_rationale": overall_rationale,
            "logical_vlm_calls": len(call_trace),
            "pages_examined": top_n,
            "usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_tokens,
                "transport_attempts": transport_attempts,
            },
            "request_audit": summarize_call_trace(call_trace),
            "call_trace": call_trace,
        }
        write_json(episode_path, row)
        print(f"[submitted] {settings['route']} {query_id}", flush=True)
        return row
    except VLMFormatError as exc:
        error_trace = traceback.format_exc()
        status = "invalid_model_output"
        error: Exception = exc
    except VLMProviderError as exc:
        error_trace = traceback.format_exc()
        status = "provider_error"
        error = exc
    except VLMRetryExhaustedError as exc:
        error_trace = traceback.format_exc()
        status = "planner_retry_exhausted"
        error = exc
    except ValueError as exc:
        error_trace = traceback.format_exc()
        status = "invalid_model_output"
        error = exc
    except Exception as exc:  # isolate every query from worker failures
        error_trace = traceback.format_exc()
        status = "worker_exception"
        error = exc

    if active_call is not None:
        active_call["response"] = rejected_response(error)
        call_trace.append(active_call)
    row = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "query_id": query_id,
        "query": query_text(query),
        "gold_page_id": gold,
        "level": query.get("level"),
        "topic_id": query.get("topic_id"),
        "route": settings["route"],
        "status": status,
        "first_stage_gold_rank": first_stage_gold_rank,
        "ranked_page_ids": [],
        "ranked_answers": [],
        "logical_vlm_calls": len(call_trace),
        "pages_examined": sum(
            len(call.get("page_handles", []))
            for call in call_trace
            if call.get("phase") == "candidate_assessment"
        ),
        "request_audit": summarize_call_trace(call_trace),
        "call_trace": call_trace,
        "last_error_type": type(error).__name__,
        "last_error": str(error),
        "traceback": error_trace,
    }
    write_json(episode_path, row)
    print(f"[failed] {settings['route']} {query_id}: {status}: {error}", flush=True)
    return row


def extended_metrics(
    queries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    valid_page_ids: set[str] | None = None,
) -> dict[str, Any]:
    report = summarize_predictions(
        queries, predictions, valid_page_ids=valid_page_ids
    )
    query_by_id = {str(row["query_id"]): row for row in queries}

    def augment(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        hits = [row for row in rows if row.get("first_stage_gold_rank") is not None]
        rank1_hits = sum(
            bool(row.get("ranked_page_ids"))
            and row["ranked_page_ids"][0] == gold_page_id(query_by_id[str(row["query_id"])])
            for row in hits
        )
        summary["first_stage_recall@30"] = len(hits) / len(rows) if rows else 0.0
        summary["final_recall@1_given_first_stage_hit"] = (
            rank1_hits / len(hits) if hits else 0.0
        )
        valid = [row for row in rows if row.get("status") == ACCEPTED_STATUS]
        summary["mean_logical_vlm_calls"] = (
            sum(int(row.get("logical_vlm_calls", 0)) for row in valid) / len(valid)
            if valid
            else 0.0
        )
        summary["mean_pages_examined"] = (
            sum(int(row.get("pages_examined", 0)) for row in valid) / len(valid)
            if valid
            else 0.0
        )
        for key in ("input_tokens", "output_tokens", "total_tokens", "transport_attempts"):
            summary[f"mean_{key}"] = (
                sum(int(row.get("usage", {}).get(key, 0)) for row in valid) / len(valid)
                if valid
                else 0.0
            )

    augment(report["overall"], predictions)
    for level, summary in report["by_level"].items():
        augment(summary, [row for row in predictions if str(row.get("level")) == level])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--first-stage-predictions", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--route", choices=("visual", "ocr_text"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--handle-seed", type=int, default=20260707)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix; 0 runs all queries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_count != 30 or args.batch_size != 3:
        raise ValueError("the frozen fixed-reranker protocol requires 30 candidates in batches of 3")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    queries = read_jsonl(args.queries)
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.limit:
        queries = queries[: args.limit]
    first_stage_rows = read_jsonl(args.first_stage_predictions)
    page_rows = read_jsonl(args.page_manifest)
    ocr_rows = read_jsonl(args.ocr_cache)
    expected_queries = args.limit if args.limit else 120
    if len(queries) != expected_queries or len(page_rows) != 2375 or len(ocr_rows) != 2375:
        raise ValueError(
            f"expected {expected_queries} queries and 2375 page/OCR rows, found "
            f"{len(queries)}, {len(page_rows)}, {len(ocr_rows)}"
        )
    if not args.limit and {str(row.get("level")) for row in queries} != {"1", "2", "3"}:
        raise ValueError("query set does not contain all three levels")

    first_stage_by_id = {str(row.get("query_id")): row for row in first_stage_rows}
    page_by_id = {str(row.get("page_id")): row for row in page_rows}
    ocr_by_id = {str(row.get("page_id")): cached_ocr_text(row) for row in ocr_rows}
    if len(first_stage_by_id) != len(first_stage_rows):
        raise ValueError("duplicate query rows in first-stage predictions")
    if any(str(row.get("query_id")) not in first_stage_by_id for row in queries):
        raise ValueError("first-stage predictions are missing requested query rows")
    if len(page_by_id) != 2375 or len(ocr_by_id) != 2375:
        raise ValueError("duplicate or missing page rows in fixed-reranker inputs")
    page_image_identity = file_collection_identity(
        (
            str(row["page_id"]),
            resolve_page_image(row, args.page_manifest),
        )
        for row in page_rows
    )
    page_to_handle, handle_to_page = opaque_handle_maps(page_rows, args.handle_seed)

    config = resolve_vlm_config(
        args.config,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        max_tokens=args.max_tokens,
    )
    if config.api_format.strip().lower() not in {"responses", "response"}:
        raise ValueError("the frozen GPT-5.6-sol run requires the Responses API")
    if config.reasoning_effort != "medium":
        raise ValueError(f"expected reasoning effort medium, got {config.reasoning_effort!r}")

    settings = {
        "route": args.route,
        "base_url": config.base_url,
        "model": config.model,
        "api_format": config.api_format,
        "provider": config.provider,
        "reasoning_effort": config.reasoning_effort,
        "temperature": config.temperature,
        "candidate_count": args.candidate_count,
        "batch_size": args.batch_size,
        "candidate_assessment_calls": args.candidate_count // args.batch_size,
        "aggregation_calls": 1,
        "max_tokens": args.max_tokens,
        "timeout_sec": config.timeout,
        "transport_policy_version": TRANSPORT_POLICY_VERSION,
        "json_parse_policy_version": JSON_PARSE_POLICY_VERSION,
        "max_transport_attempts": config.max_retries + 1,
        "retry_backoff": config.retry_backoff,
        "handle_seed": args.handle_seed,
        "candidate_inputs": ["complete_page_image", "full_cached_ocr_text"],
        "image_max_side": effective_image_max_side(),
        "adaptive_search": False,
        "input_identity": {
            "page_manifest": file_identity(args.page_manifest),
            "ocr_cache": file_identity(args.ocr_cache),
            "first_stage_predictions": file_identity(args.first_stage_predictions),
            "page_images": page_image_identity,
        },
    }
    episodes_dir = args.out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    output_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for query in queries:
            query_id = str(query["query_id"])
            first_stage = first_stage_by_id.get(query_id)
            if first_stage is None:
                raise ValueError(f"missing first-stage prediction for {query_id}")
            future = executor.submit(
                process_query,
                query=query,
                first_stage=first_stage,
                page_by_id=page_by_id,
                ocr_by_id=ocr_by_id,
                page_to_handle=page_to_handle,
                handle_to_page=handle_to_page,
                page_manifest_path=args.page_manifest,
                config=config,
                settings=settings,
                episode_path=episodes_dir / f"{safe_filename(query_id)}.json",
            )
            futures[future] = query_id
        for future in as_completed(futures):
            query_id = futures[future]
            try:
                output_by_id[query_id] = future.result()
            except Exception as exc:
                query = next(row for row in queries if str(row["query_id"]) == query_id)
                output_by_id[query_id] = {
                    "schema_version": SCHEMA_VERSION,
                    "query_id": query_id,
                    "query": query_text(query),
                    "gold_page_id": gold_page_id(query),
                    "level": query.get("level"),
                    "topic_id": query.get("topic_id"),
                    "route": args.route,
                    "status": "worker_exception",
                    "ranked_page_ids": [],
                    "last_error_type": type(exc).__name__,
                    "last_error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(episodes_dir / f"{safe_filename(query_id)}.json", output_by_id[query_id])

    predictions = [output_by_id[str(query["query_id"])] for query in queries]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "predictions.jsonl", predictions)
    metrics = extended_metrics(
        queries, predictions, valid_page_ids=set(page_by_id)
    )
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "inputs": {
                "queries": str(args.queries.resolve()),
                "first_stage_predictions": str(args.first_stage_predictions.resolve()),
                "page_manifest": str(args.page_manifest.resolve()),
                "ocr_cache": str(args.ocr_cache.resolve()),
                "config_path": str(args.config.resolve()),
            },
            "num_queries": len(queries),
            "num_pages": len(page_rows),
            "workers": args.workers,
            "smoke_limit": args.limit,
            "status_counts": {
                status: sum(row.get("status") == status for row in predictions)
                for status in sorted({str(row.get("status")) for row in predictions})
            },
        },
    )
    print(json.dumps(metrics["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
