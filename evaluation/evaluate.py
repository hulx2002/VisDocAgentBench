#!/usr/bin/env python3
"""Evaluate ranked VisDocAgentBench predictions overall and by evidence level."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def handle_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        str(row["page_handle"]): str(row["page_id"])
        for row in read_jsonl(path)
    }


def corpus_page_ids(path: Path) -> set[str]:
    rows = read_jsonl(path)
    page_ids = [str(row.get("page_id") or "") for row in rows]
    if not page_ids or any(not page_id for page_id in page_ids):
        raise ValueError("Corpus pages must contain non-empty page_id values")
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Corpus pages contain duplicate page_ids")
    return set(page_ids)


def ranked_page_ids(row: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    direct = row.get("ranked_page_ids")
    if isinstance(direct, list):
        return [str(value) for value in direct]
    ranked_answers = (row.get("submission") or {}).get("ranked_answers") or []
    handles = [
        str(item.get("page_handle") if isinstance(item, dict) else item)
        for item in ranked_answers
    ]
    return [mapping[value] for value in handles if value in mapping]


def score(gold: str, ranking: list[str]) -> dict[str, float | int | None]:
    rank = ranking.index(gold) + 1 if gold in ranking[:10] else None
    return {
        "gold_rank": rank,
        "recall@1": float(rank == 1),
        "recall@3": float(rank is not None and rank <= 3),
        "recall@5": float(rank is not None and rank <= 5),
        "recall@10": float(rank is not None and rank <= 10),
        "mrr@10": 0.0 if rank is None else 1.0 / rank,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "valid_rankings": sum(bool(row["valid_ranking"]) for row in rows),
        **{
            metric: 100.0 * sum(float(row[metric]) for row in rows) / count
            if count
            else 0.0
            for metric in METRICS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--evaluator-annotations",
        type=Path,
        default=Path("data/benchmark/evaluator_annotations.jsonl"),
    )
    parser.add_argument(
        "--page-handle-mapping",
        type=Path,
        default=None,
        help="Required for agent outputs that rank opaque page handles.",
    )
    parser.add_argument(
        "--corpus-pages",
        type=Path,
        default=Path("data/corpus/pages.jsonl"),
        help="Corpus page manifest used to validate submitted page IDs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = read_jsonl(args.evaluator_annotations)
    predictions = read_jsonl(args.predictions)
    prediction_by_id = {str(row.get("query_id") or ""): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Predictions contain duplicate query_ids")
    mapping = handle_map(args.page_handle_mapping)
    valid_page_ids = corpus_page_ids(args.corpus_pages)

    unknown_gold = sorted(
        {
            str(annotation["answer_page_id"])
            for annotation in annotations
            if str(annotation["answer_page_id"]) not in valid_page_ids
        }
    )
    if unknown_gold:
        raise ValueError(f"Evaluator annotations contain unknown answer_page_ids: {unknown_gold}")

    scored: list[dict[str, Any]] = []
    for annotation in annotations:
        query_id = str(annotation["query_id"])
        ranking = ranked_page_ids(prediction_by_id.get(query_id, {}), mapping)[:10]
        valid = (
            len(ranking) == 10
            and len(set(ranking)) == 10
            and all(page_id in valid_page_ids for page_id in ranking)
        )
        if not valid:
            ranking = []
        metrics = score(str(annotation["answer_page_id"]), ranking)
        scored.append(
            {
                "query_id": query_id,
                "level": str(annotation["level"]),
                "topic_id": str(annotation["topic_id"]),
                "valid_ranking": valid,
                **metrics,
            }
        )

    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_level[row["level"]].append(row)
        by_topic[row["topic_id"]].append(row)
    report = {
        "metric_scale": "percentage",
        "overall": summarize(scored),
        "by_level": {key: summarize(value) for key, value in sorted(by_level.items())},
        "by_topic": {key: summarize(value) for key, value in sorted(by_topic.items())},
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
