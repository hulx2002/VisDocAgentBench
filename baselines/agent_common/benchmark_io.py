"""Load the public query file together with evaluator-only annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_benchmark_rows(
    queries_path: Path,
    evaluator_annotations_path: Path,
) -> list[dict[str, Any]]:
    """Join public queries with gold fields used only by the harness evaluator.

    The returned shape matches the frozen agent runners. Support-page annotations
    are deliberately not copied into an episode row, so the standard agent cannot
    access them through either its prompt or runtime state.
    """

    query_rows = _read_jsonl(queries_path)
    evaluator_rows = _read_jsonl(evaluator_annotations_path)

    evaluator_by_id: dict[str, dict[str, Any]] = {}
    for row in evaluator_rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError(f"Missing query_id in {evaluator_annotations_path}")
        if query_id in evaluator_by_id:
            raise ValueError(f"Duplicate evaluator query_id: {query_id}")
        evaluator_by_id[query_id] = row

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in query_rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError(f"Missing query_id in {queries_path}")
        if query_id in seen:
            raise ValueError(f"Duplicate query query_id: {query_id}")
        seen.add(query_id)
        evaluator = evaluator_by_id.get(query_id)
        if evaluator is None:
            raise ValueError(f"No evaluator annotation for query_id: {query_id}")

        query = str(row.get("query") or "").strip()
        answer_page_id = str(evaluator.get("answer_page_id") or "").strip()
        if not query:
            raise ValueError(f"Empty query text for query_id: {query_id}")
        if not answer_page_id:
            raise ValueError(f"Missing answer_page_id for query_id: {query_id}")

        query_topic = str(row.get("topic_id") or "")
        evaluator_topic = str(evaluator.get("topic_id") or "")
        if query_topic and evaluator_topic and query_topic != evaluator_topic:
            raise ValueError(f"Topic mismatch for query_id: {query_id}")

        merged.append(
            {
                "query_id": query_id,
                "query": query,
                "gold_page_id": answer_page_id,
                "level": evaluator.get("level", row.get("level", "")),
                "topic_id": evaluator_topic or query_topic,
            }
        )

    extra = sorted(set(evaluator_by_id) - seen)
    if extra:
        preview = ", ".join(extra[:5])
        raise ValueError(f"Evaluator annotations contain unknown query_ids: {preview}")
    return merged
