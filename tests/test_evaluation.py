from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_missing_and_invalid_predictions_score_zero(tmp_path: Path) -> None:
    annotations = [
        {"query_id": "q1", "answer_page_id": "p1", "level": 1, "topic_id": "t1"},
        {"query_id": "q2", "answer_page_id": "p2", "level": 2, "topic_id": "t2"},
        {"query_id": "q3", "answer_page_id": "p3", "level": 3, "topic_id": "t3"},
        {"query_id": "q4", "answer_page_id": "p4", "level": 3, "topic_id": "t3"},
    ]
    predictions = [
        {"query_id": "q1", "ranked_page_ids": ["p1"] + [f"p{i}" for i in range(5, 14)]},
        {"query_id": "q2", "ranked_page_ids": ["p2"] * 10},
        {"query_id": "q3", "ranked_page_ids": ["p3"] + [f"x{i}" for i in range(9)]},
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    corpus_path = tmp_path / "pages.jsonl"
    write_jsonl(annotation_path, annotations)
    write_jsonl(prediction_path, predictions)
    write_jsonl(corpus_path, [{"page_id": f"p{i}"} for i in range(1, 14)])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluation" / "evaluate.py"),
            "--predictions",
            str(prediction_path),
            "--evaluator-annotations",
            str(annotation_path),
            "--corpus-pages",
            str(corpus_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["overall"]["valid_rankings"] == 1
    assert report["overall"]["recall@1"] == 25.0
    assert report["by_level"]["2"]["recall@10"] == 0.0
    assert report["by_level"]["3"]["mrr@10"] == 0.0


def test_only_the_first_ten_ranks_define_validity(tmp_path: Path) -> None:
    annotations = [
        {"query_id": "q1", "answer_page_id": "p1", "level": 1, "topic_id": "t1"}
    ]
    predictions = [
        {
            "query_id": "q1",
            "ranked_page_ids": [f"p{i}" for i in range(1, 11)] + ["p1", "not_in_corpus"],
        }
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    corpus_path = tmp_path / "pages.jsonl"
    write_jsonl(annotation_path, annotations)
    write_jsonl(prediction_path, predictions)
    write_jsonl(corpus_path, [{"page_id": f"p{i}"} for i in range(1, 11)])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluation" / "evaluate.py"),
            "--predictions",
            str(prediction_path),
            "--evaluator-annotations",
            str(annotation_path),
            "--corpus-pages",
            str(corpus_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["overall"]["valid_rankings"] == 1
    assert report["overall"]["recall@1"] == 100.0
