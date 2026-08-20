from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "baselines" / "agent_common"
sys.path.insert(0, str(COMMON))

from episode_checkpoint import (  # noqa: E402
    directory_identity,
    EpisodeCheckpointError,
    EpisodeProgress,
    file_collection_identity,
    load_episode_checkpoint,
    write_episode_checkpoint,
)
from resume_scope import assert_inplace_resume_scope, resume_scope_report  # noqa: E402
from request_audit import summarize_model_outputs  # noqa: E402


def test_resume_scope_rejects_a_shrinking_subset(tmp_path: Path) -> None:
    predictions = [
        {"query_id": "done", "query": "q", "gold_page_id": "p1"},
        {"query_id": "pending", "query": "q", "gold_page_id": "p1"},
    ]
    annotations = [{"query_id": "pending", "query": "q", "gold_page_id": "p1"}]
    report = resume_scope_report(predictions, annotations)
    assert not report["compatible"]
    path = tmp_path / "predictions.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in predictions)
    path.write_text(original, encoding="utf-8")
    try:
        assert_inplace_resume_scope(path, annotations)
    except ValueError as error:
        assert "Refusing destructive" in str(error)
    else:
        raise AssertionError("incompatible in-place resume was accepted")
    assert path.read_text(encoding="utf-8") == original


def test_resume_scope_rejects_changed_run_fingerprint(tmp_path: Path) -> None:
    predictions = [
        {
            "query_id": "q1",
            "query": "query",
            "gold_page_id": "p1",
            "episode_fingerprint_sha256": "old",
        }
    ]
    annotations = [{"query_id": "q1", "query": "query", "gold_page_id": "p1"}]
    report = resume_scope_report(
        predictions,
        annotations,
        expected_fingerprints={"q1": "new"},
    )
    assert report["fingerprint_mismatch_ids"] == ["q1"]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(predictions[0]) + "\n", encoding="utf-8")
    try:
        assert_inplace_resume_scope(
            path,
            annotations,
            expected_fingerprints={"q1": "new"},
        )
    except ValueError as error:
        assert "run configuration" in str(error)
    else:
        raise AssertionError("changed run fingerprint was accepted")


def test_checkpoint_round_trip_and_fingerprint_guard(tmp_path: Path) -> None:
    path = tmp_path / "episode.json"
    fingerprint = {"query": "q", "route": "visual", "max_steps": 12}
    progress = EpisodeProgress(next_step=8, model_outputs=[], transport_failures=[])
    write_episode_checkpoint(
        path,
        fingerprint=fingerprint,
        progress=progress,
        runtime_state={"trace": []},
    )
    loaded, state, legacy = load_episode_checkpoint(
        path, expected_fingerprint=fingerprint, max_steps=12
    )
    assert loaded.to_dict() == progress.to_dict()
    assert state == {"trace": []}
    assert legacy is None
    try:
        load_episode_checkpoint(
            path,
            expected_fingerprint={**fingerprint, "max_steps": 8},
            max_steps=8,
        )
    except EpisodeCheckpointError:
        pass
    else:
        raise AssertionError("changed checkpoint fingerprint was accepted")


def test_request_audit_separates_logical_turns_from_physical_attempts() -> None:
    audit = summarize_model_outputs(
        [
            {
                "step": 5,
                "phase": "planner",
                "attempts": 3,
                "event_type": "transport_failure",
                "response_accepted": False,
                "request_attempts": [
                    {"attempt": 1, "retry_scheduled": True},
                    {"attempt": 2, "retry_scheduled": True},
                    {"attempt": 3, "retry_scheduled": False},
                ],
            },
            {
                "step": 5,
                "phase": "planner",
                "attempts": 1,
                "event_type": "accepted_response",
                "response_accepted": True,
                "request_attempts": [
                    {"attempt": 1, "retry_scheduled": False},
                ],
            },
        ]
    )
    assert audit["logical_planner_turns"] == 1
    assert audit["planner_call_events"] == 2
    assert audit["physical_request_attempts"] == 4
    assert audit["transport_retries"] == 2
    assert audit["accepted_responses"] == 1
    assert audit["rejected_responses"] == 1


def test_episode_input_identities_track_page_and_model_bytes(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    model = tmp_path / "model"
    weight = model / "weights.bin"
    model.mkdir()
    page.write_bytes(b"page-a")
    weight.write_bytes(b"weight-a")
    page_before = file_collection_identity([("p1", page)])
    model_before = directory_identity(model)

    page.write_bytes(b"page-b-longer")
    weight.write_bytes(b"weight-b-longer")
    page_after = file_collection_identity([("p1", page)])
    model_after = directory_identity(model)

    assert page_before["sha256"] != page_after["sha256"]
    assert model_before["sha256"] != model_after["sha256"]
