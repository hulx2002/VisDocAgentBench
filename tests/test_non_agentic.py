from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baselines" / "non_agentic"
sys.path.insert(0, str(BASELINE_DIR))

from run_bm25_dense_rrf import (  # noqa: E402
    bm25_ranking,
    build_bm25_index,
    reciprocal_rank_fusion,
)
from common import rank_metrics, summarize_predictions  # noqa: E402
from run_fixed_reranker import (  # noqa: E402
    effective_image_max_side,
    fingerprint_for_query,
    mark_schema_rejection,
    summarize_call_trace,
    validate_aggregate,
    validate_batch_assessment,
)


def test_bm25_and_rrf_are_deterministic() -> None:
    pages = [
        {"page_id": "p1", "ocr_text": "alpha beta"},
        {"page_id": "p2", "ocr_text": "gamma delta"},
    ]
    page_ids, lengths, postings, average = build_bm25_index(pages, k1=1.2, b=0.75)
    ranking, scores = bm25_ranking(
        "alpha",
        page_ids=page_ids,
        document_lengths=lengths,
        postings=postings,
        average_length=average,
        k1=1.2,
        b=0.75,
        depth=2,
    )
    assert ranking[0] == "p1"
    assert scores[0] > scores[1]
    assert lengths.dtype == np.float32
    fused, _ = reciprocal_rank_fusion(
        ["a", "b", "c"], ["b", "c", "d"], rrf_k=60, output_depth=4
    )
    assert fused[:2] == ["b", "c"]


def test_fixed_reranker_requires_complete_batches_and_top10() -> None:
    response = {
        "assessments": [
            {"page_handle": "p2", "relevance_score": 20},
            {"page_handle": "p1", "relevance_score": 90},
            {"page_handle": "p3", "relevance_score": 10},
        ]
    }
    rows = validate_batch_assessment(response, ["p1", "p2", "p3"])
    assert [row["page_handle"] for row in rows] == ["p1", "p2", "p3"]
    handles = [f"p{index}" for index in range(30)]
    aggregate = {
        "ranked_answers": [
            {"page_handle": handle, "rationale": "candidate"}
            for handle in handles[:10]
        ],
        "overall_rationale": "ordered",
    }
    ranked, answers, rationale = validate_aggregate(aggregate, set(handles))
    assert ranked == handles[:10]
    assert len(answers) == 10
    assert rationale == "ordered"


def test_fixed_reranker_logs_schema_rejection_without_resampling() -> None:
    recorded_response = {
        "attempts": 1,
        "request_attempts": [
            {"attempt": 1, "outcome": "accepted", "retry_scheduled": False}
        ],
        "response_accepted": True,
        "event_type": "accepted_response",
        "raw_response": '{"assessments": []}',
    }
    mark_schema_rejection(recorded_response, ValueError("expected three assessments"))
    audit = summarize_call_trace(
        [{"phase": "candidate_assessment", "response": recorded_response}]
    )
    assert recorded_response["response_accepted"] is False
    assert recorded_response["event_type"] == "schema_error"
    assert audit["logical_vlm_calls"] == 1
    assert audit["physical_request_attempts"] == 1
    assert audit["format_failures"] == 1
    assert audit["schema_failures"] == 1
    assert audit["rejected_responses"] == 1


def test_fixed_reranker_resume_identity_includes_page_images() -> None:
    query = {"query_id": "q1", "query": "query", "gold_page_id": "p1"}
    first_stage = {"ranked_page_ids": ["p1"], "ranked_scores": [1.0]}
    settings = {"input_identity": {"page_images": {"sha256": "image-set-a"}}}
    first = fingerprint_for_query(query, first_stage, settings)
    settings["input_identity"]["page_images"]["sha256"] = "image-set-b"
    second = fingerprint_for_query(query, first_stage, settings)
    assert first != second


def test_fixed_reranker_resume_identity_includes_effective_image_size(
    monkeypatch,
) -> None:
    query = {"query_id": "q1", "query": "query", "gold_page_id": "p1"}
    first_stage = {"ranked_page_ids": ["p1"], "ranked_scores": [1.0]}

    monkeypatch.delenv("VLM_IMAGE_MAX_SIDE", raising=False)
    default_size = effective_image_max_side()
    monkeypatch.setenv("VLM_IMAGE_MAX_SIDE", "1024")
    resized = effective_image_max_side()

    assert default_size == 0
    assert resized == 1024
    assert fingerprint_for_query(
        query, first_stage, {"image_max_side": default_size}
    ) != fingerprint_for_query(query, first_stage, {"image_max_side": resized})


def test_non_agentic_summary_uses_the_official_top10_contract() -> None:
    query = {"query_id": "q1", "gold_page_id": "p1", "level": 1, "topic_id": "t1"}
    valid_page_ids = {f"p{index}" for index in range(1, 11)}

    def summarize(ranking: list[str], *, status: str = "submitted") -> dict:
        report = summarize_predictions(
            [query],
            [
                {
                    "query_id": "q1",
                    "status": status,
                    "ranked_page_ids": ranking,
                }
            ],
            valid_page_ids=valid_page_ids,
        )
        return report["overall"]

    for short_ranking in ([], ["p1"], ["p1", *[f"p{i}" for i in range(2, 10)]]):
        summary = summarize(short_ranking)
        assert summary["num_valid"] == 0
        assert summary["recall@1"] == 0.0
        assert summary["mrr@10"] == 0.0

    duplicate = ["p1", *[f"p{i}" for i in range(2, 10)], "p1"]
    summary = summarize(duplicate)
    assert summary["num_valid"] == 0
    assert summary["recall@1"] == 0.0

    unknown_in_top10 = ["p1", *[f"p{i}" for i in range(2, 10)], "unknown"]
    summary = summarize(unknown_in_top10)
    assert summary["num_valid"] == 0
    assert summary["recall@1"] == 0.0

    valid = [f"p{i}" for i in range(1, 11)]
    summary = summarize(valid)
    assert summary["num_valid"] == 1
    assert summary["recall@1"] == 1.0
    assert summary["mrr@10"] == 1.0

    for extra in ("p1", "unknown"):
        summary = summarize([*valid, extra])
        assert summary["num_valid"] == 1
        assert summary["recall@1"] == 1.0

    summary = summarize(valid, status="worker_exception")
    assert summary["num_valid"] == 0
    assert summary["recall@1"] == 0.0

    assert rank_metrics("p1", ["p1"])["recall@1"] == 0.0
