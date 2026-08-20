#!/usr/bin/env python3
"""Rank all corpus pages for each query from cached embeddings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import SCHEMA_VERSION, read_jsonl, write_json, write_jsonl


def load_embedding_map(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get(id_field)
        if not item_id:
            raise ValueError(f"Embedding row in {path} is missing {id_field}")
        if item_id in out:
            raise ValueError(f"Duplicate embedding id {item_id} in {path}")
        if "embedding" not in row:
            raise ValueError(f"Embedding row {item_id} in {path} has no embedding")
        out[str(item_id)] = row
    return out


def l2_normalize(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--page-embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--query-id-field", default="query_id")
    parser.add_argument("--page-id-field", default="page_id")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normalize-inputs", action="store_true", default=True)
    parser.add_argument("--no-normalize-inputs", action="store_false", dest="normalize_inputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    import numpy as np

    query_rows = read_jsonl(args.queries)
    query_embeddings = load_embedding_map(args.query_embeddings, args.query_id_field)
    page_embeddings = load_embedding_map(args.page_embeddings, args.page_id_field)

    page_ids = list(page_embeddings.keys())
    page_matrix = np.asarray([page_embeddings[page_id]["embedding"] for page_id in page_ids], dtype=np.float32)
    query_matrix = np.asarray(
        [query_embeddings[str(row[args.query_id_field])]["embedding"] for row in query_rows],
        dtype=np.float32,
    )
    if page_matrix.ndim != 2 or query_matrix.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays")
    if page_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError(f"Embedding dimensions differ: queries={query_matrix.shape[1]} pages={page_matrix.shape[1]}")
    if args.normalize_inputs:
        page_matrix = l2_normalize(page_matrix)
        query_matrix = l2_normalize(query_matrix)

    top_k = min(args.top_k, len(page_ids))
    predictions: list[dict[str, Any]] = []
    for row_index, query in enumerate(query_rows):
        query_id = str(query[args.query_id_field])
        if query_id not in query_embeddings:
            raise ValueError(f"Missing query embedding for {query_id}")
        scores = query_matrix[row_index] @ page_matrix.T
        order = np.argsort(-scores, kind="mergesort")
        top_indices = order[:top_k]
        ranked_page_ids = [page_ids[int(idx)] for idx in top_indices]
        ranked_scores = [float(scores[int(idx)]) for idx in top_indices]
        gold_page_id = str(query.get("gold_page_id", ""))
        gold_rank = None
        if gold_page_id:
            matches = np.where(order == page_ids.index(gold_page_id))[0] if gold_page_id in page_ids else []
            if len(matches):
                gold_rank = int(matches[0]) + 1
        predictions.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "gold_page_id": gold_page_id,
                "level": query.get("level", ""),
                "topic_id": query.get("topic_id", ""),
                "top_k": top_k,
                "ranked_page_ids": ranked_page_ids,
                "ranked_scores": ranked_scores,
                "gold_rank": gold_rank,
                "gold_in_top_k": bool(gold_page_id in ranked_page_ids),
            }
        )

    write_jsonl(args.out, predictions)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "input_files": {
            "queries": str(args.queries),
            "query_embeddings": str(args.query_embeddings),
            "page_embeddings": str(args.page_embeddings),
        },
        "output_files": {"predictions": str(args.out), "report": str(args.report)},
        "num_queries": len(query_rows),
        "num_pages": len(page_ids),
        "embedding_dim": int(page_matrix.shape[1]),
        "top_k": top_k,
        "normalize_inputs": args.normalize_inputs,
    }
    write_json(args.report, report)
    print(f"[done] wrote {len(predictions)} predictions to {args.out}")
    print(f"[done] wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
