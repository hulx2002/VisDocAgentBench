#!/usr/bin/env python3
"""Run BM25 and BM25+dense reciprocal-rank fusion on cached OCR text."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    PROJECT_ROOT,
    gold_page_id,
    query_text,
    read_jsonl,
    sha256_json,
    summarize_predictions,
    write_json,
    write_jsonl,
)


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
SCHEMA_VERSION = "VisDocAgentBenchBM25RRFV1"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(
    page_rows: list[dict[str, Any]], *, k1: float, b: float
) -> tuple[list[str], np.ndarray, dict[str, tuple[float, list[tuple[int, int]]]], float]:
    page_ids: list[str] = []
    document_lengths = np.zeros(len(page_rows), dtype=np.float32)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(page_rows):
        page_id = str(row.get("page_id") or "")
        if not page_id:
            raise ValueError(f"OCR row {index} has no page_id")
        text = str(row.get("ocr_text") or row.get("text") or "")
        counts = Counter(tokenize(text))
        page_ids.append(page_id)
        document_lengths[index] = sum(counts.values())
        for term, frequency in counts.items():
            postings[term].append((index, frequency))

    if len(page_ids) != len(set(page_ids)):
        raise ValueError("duplicate page ids in OCR corpus")
    average_length = float(document_lengths.mean())
    if average_length <= 0:
        raise ValueError("OCR corpus has zero average token length")

    total_documents = len(page_ids)
    weighted_postings: dict[str, tuple[float, list[tuple[int, int]]]] = {}
    for term, rows in postings.items():
        document_frequency = len(rows)
        idf = math.log(
            1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        weighted_postings[term] = (idf, rows)
    return page_ids, document_lengths, weighted_postings, average_length


def bm25_ranking(
    text: str,
    *,
    page_ids: list[str],
    document_lengths: np.ndarray,
    postings: dict[str, tuple[float, list[tuple[int, int]]]],
    average_length: float,
    k1: float,
    b: float,
    depth: int,
) -> tuple[list[str], list[float]]:
    scores = np.zeros(len(page_ids), dtype=np.float64)
    for term, query_frequency in Counter(tokenize(text)).items():
        entry = postings.get(term)
        if entry is None:
            continue
        idf, term_postings = entry
        for document_index, frequency in term_postings:
            normalization = k1 * (
                1.0 - b + b * float(document_lengths[document_index]) / average_length
            )
            scores[document_index] += query_frequency * idf * (
                frequency * (k1 + 1.0) / (frequency + normalization)
            )
    order = sorted(range(len(page_ids)), key=lambda index: (-scores[index], page_ids[index]))[:depth]
    return [page_ids[index] for index in order], [float(scores[index]) for index in order]


def load_dense_rankings(path: Path, expected_depth: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        ranking = [str(value) for value in row.get("ranked_page_ids", [])]
        if not query_id or len(ranking) < expected_depth:
            raise ValueError(
                f"dense ranking for {query_id or '<missing>'} has {len(ranking)} rows; "
                f"expected at least {expected_depth}"
            )
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"dense ranking has duplicates for {query_id}")
        if query_id in by_id:
            raise ValueError(f"duplicate dense ranking for {query_id}")
        by_id[query_id] = row
    return by_id


def reciprocal_rank_fusion(
    sparse: list[str], dense: list[str], *, rrf_k: int, output_depth: int
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking in (sparse, dense):
        for rank, page_id in enumerate(ranking, start=1):
            scores[page_id] += 1.0 / (rrf_k + rank)
            best_rank[page_id] = min(best_rank.get(page_id, rank), rank)
    order = sorted(scores, key=lambda page_id: (-scores[page_id], best_rank[page_id], page_id))
    selected = order[:output_depth]
    return selected, [scores[page_id] for page_id in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--dense-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--fusion-depth", type=int, default=1000)
    parser.add_argument("--output-depth", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k1 <= 0 or not 0 <= args.b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    if args.rrf_k < 0 or args.fusion_depth <= 0 or args.output_depth <= 0:
        raise ValueError("RRF and depth arguments are invalid")

    queries = read_jsonl(args.queries)
    pages = read_jsonl(args.ocr_cache)
    dense_by_id = load_dense_rankings(args.dense_predictions, args.fusion_depth)
    page_ids, lengths, postings, average_length = build_bm25_index(
        pages, k1=args.k1, b=args.b
    )
    if len(page_ids) != 2375:
        raise ValueError(f"expected 2375 OCR pages, found {len(page_ids)}")

    sparse_predictions: list[dict[str, Any]] = []
    hybrid_predictions: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        query_id = str(query.get("query_id") or "")
        dense = dense_by_id.get(query_id)
        if dense is None:
            raise ValueError(f"missing dense ranking for {query_id}")
        sparse_pages, sparse_scores = bm25_ranking(
            query_text(query),
            page_ids=page_ids,
            document_lengths=lengths,
            postings=postings,
            average_length=average_length,
            k1=args.k1,
            b=args.b,
            depth=args.fusion_depth,
        )
        dense_pages = [str(value) for value in dense["ranked_page_ids"][: args.fusion_depth]]
        hybrid_pages, hybrid_scores = reciprocal_rank_fusion(
            sparse_pages,
            dense_pages,
            rrf_k=args.rrf_k,
            output_depth=args.output_depth,
        )
        base = {
            "schema_version": SCHEMA_VERSION,
            "query_id": query_id,
            "gold_page_id": gold_page_id(query),
            "level": query.get("level"),
            "topic_id": query.get("topic_id"),
            "status": "submitted",
        }
        sparse_predictions.append(
            {
                **base,
                "ranked_page_ids": sparse_pages[: args.output_depth],
                "ranked_scores": sparse_scores[: args.output_depth],
            }
        )
        hybrid_predictions.append(
            {
                **base,
                "ranked_page_ids": hybrid_pages,
                "ranked_scores": hybrid_scores,
            }
        )
        print(f"[rank] {index}/{len(queries)} {query_id}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sparse_path = args.out_dir / "bm25_predictions.jsonl"
    hybrid_path = args.out_dir / "bm25_dense_rrf_predictions.jsonl"
    write_jsonl(sparse_path, sparse_predictions)
    write_jsonl(hybrid_path, hybrid_predictions)
    valid_page_ids = set(page_ids)
    sparse_metrics = summarize_predictions(
        queries, sparse_predictions, valid_page_ids=valid_page_ids
    )
    hybrid_metrics = summarize_predictions(
        queries, hybrid_predictions, valid_page_ids=valid_page_ids
    )
    write_json(args.out_dir / "bm25_metrics.json", sparse_metrics)
    write_json(args.out_dir / "bm25_dense_rrf_metrics.json", hybrid_metrics)
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "queries": str(args.queries.resolve()),
            "ocr_cache": str(args.ocr_cache.resolve()),
            "dense_predictions": str(args.dense_predictions.resolve()),
            "num_queries": len(queries),
            "num_pages": len(page_ids),
            "tokenizer": "Unicode word tokens lowercased with (?u)\\b\\w+\\b",
            "bm25": {"variant": "Okapi", "k1": args.k1, "b": args.b},
            "rrf": {"k": args.rrf_k, "fusion_depth": args.fusion_depth},
            "output_depth": args.output_depth,
            "input_fingerprint": sha256_json(
                {
                    "query_ids": [row.get("query_id") for row in queries],
                    "queries": [query_text(row) for row in queries],
                    "pages": page_ids,
                }
            ),
        },
    )
    print("[done] BM25 and BM25+dense RRF outputs are complete", flush=True)
    print(json.dumps({"bm25": sparse_metrics["overall"], "hybrid": hybrid_metrics["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
