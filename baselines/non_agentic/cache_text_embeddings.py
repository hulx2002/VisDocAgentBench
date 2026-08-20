#!/usr/bin/env python3
"""Cache text embeddings through an OpenAI-compatible /v1/embeddings endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from common import SCHEMA_VERSION, normalize_vector, read_jsonl, sha256_json, write_json, write_jsonl


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
EMBEDDING_CACHE_VERSION = "text_embedding_cache_v2"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def embedding_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/embeddings"


def call_embeddings(
    base_url: str,
    model: str,
    api_key: str,
    texts: list[str],
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> list[list[float]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    attempts = None if max_retries < 0 else max(1, max_retries + 1)
    attempt = 1
    last_error: Exception | None = None
    while attempts is None or attempt <= attempts:
        try:
            response = requests.post(embedding_url(base_url), headers=headers, json=payload, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}: {response.text[:300]}",
                    response=response,
                )
            response.raise_for_status()
            data = response.json().get("data", [])
            rows = sorted(data, key=lambda row: row.get("index", 0))
            vectors = [row["embedding"] for row in rows]
            if len(vectors) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
            return vectors
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                status = getattr(exc.response, "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS_CODES:
                    raise
            if attempts is not None and attempt >= attempts:
                break
            total = "inf" if attempts is None else str(attempts)
            print(f"[retry] text embedding batch: {type(exc).__name__}: {exc}; sleeping {retry_backoff:.1f}s before attempt {attempt + 1}/{total}")
            time.sleep(retry_backoff)
            attempt += 1
    raise RuntimeError("Embedding request failed") from last_error


def load_existing(path: Path, id_field: str, model: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = row.get(id_field)
        if item_id and row.get("model") == model:
            existing[str(item_id)] = row
    return existing


def input_identity(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "base_url": normalize_base_url(args.base_url),
        "model": args.model,
        "service_revision": args.service_revision,
        "normalized": bool(args.normalize),
        "id_field": args.id_field,
        "text_field": args.text_field,
        "include_text": bool(args.include_text),
        "payload_format": "openai_embeddings_input_list",
        "text": str(item.get(args.text_field, "")),
    }


def current_fingerprint(item: dict[str, Any], args: argparse.Namespace) -> str:
    return sha256_json(input_identity(item, args))


def make_embedding_row(
    item: dict[str, Any],
    id_field: str,
    text_field: str,
    model: str,
    embedding: list[float],
    normalize: bool,
    include_text: bool,
    input_sha256: str,
    base_url: str,
    service_revision: str,
) -> dict[str, Any]:
    if normalize:
        embedding = normalize_vector(embedding)
    text = str(item.get(text_field, ""))
    row = {
        "schema_version": SCHEMA_VERSION,
        id_field: str(item[id_field]),
        "cache_version": EMBEDDING_CACHE_VERSION,
        "base_url": normalize_base_url(base_url),
        "model": model,
        "service_revision": service_revision,
        "dim": len(embedding),
        "normalized": normalize,
        "input_sha256": input_sha256,
        "embedding": embedding,
    }
    for key in ("query_id", "page_id", "gold_page_id", "level", "topic_id", "doc_id", "arxiv_id", "page_index"):
        if key in item and key not in row:
            row[key] = item[key]
    if include_text:
        row[text_field] = text
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--id-field", required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--service-revision",
        default=os.environ.get("EMBEDDING_SERVICE_REVISION", ""),
        help="Optional immutable revision label for a mutable embedding endpoint.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum missing or stale items to process; current cache rows are preserved.",
    )
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--include-text", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    items = read_jsonl(args.items)
    for item in items:
        if item.get(args.id_field) in (None, ""):
            raise ValueError(f"Missing id field {args.id_field}: {item}")
        if args.text_field not in item:
            raise ValueError(f"Missing text field {args.text_field}: {item.get(args.id_field)}")

    existing = load_existing(args.out, args.id_field, args.model) if args.resume else {}
    invalidated_existing = 0
    if existing:
        item_by_id = {str(item[args.id_field]): item for item in items}
        filtered_existing: dict[str, dict[str, Any]] = {}
        for item_id, row in existing.items():
            item = item_by_id.get(item_id)
            if item is None:
                continue
            current_hash = current_fingerprint(item, args)
            if row.get("input_sha256") == current_hash:
                filtered_existing[item_id] = row
            else:
                invalidated_existing += 1
        existing = filtered_existing
    output_by_id: dict[str, dict[str, Any]] = dict(existing)
    all_pending = [
        item for item in items if str(item[args.id_field]) not in output_by_id
    ]
    pending = all_pending[: args.limit] if args.limit > 0 else all_pending

    print(
        f"[start] items={len(items)} existing={len(existing)} "
        f"stale_or_missing={len(all_pending)} processing={len(pending)} "
        f"model={args.model}"
    )
    processed = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        texts = [str(item.get(args.text_field, "")) for item in batch]
        vectors = call_embeddings(
            args.base_url,
            args.model,
            args.api_key,
            texts,
            args.timeout,
            args.max_retries,
            args.retry_backoff,
        )
        for item, embedding in zip(batch, vectors):
            row = make_embedding_row(
                item=item,
                id_field=args.id_field,
                text_field=args.text_field,
                model=args.model,
                embedding=embedding,
                normalize=args.normalize,
                include_text=args.include_text,
                input_sha256=current_fingerprint(item, args),
                base_url=args.base_url,
                service_revision=args.service_revision,
            )
            output_by_id[str(item[args.id_field])] = row
        processed += len(batch)
        print(f"[embed] {len(existing) + processed}/{len(items)} items")

    ordered_rows = [
        output_by_id[str(item[args.id_field])]
        for item in items
        if str(item[args.id_field]) in output_by_id
    ]
    write_jsonl(args.out, ordered_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "input_file": str(args.items),
        "output_file": str(args.out),
        "report_file": str(args.report),
        "id_field": args.id_field,
        "text_field": args.text_field,
        "base_url": args.base_url,
        "model": args.model,
        "service_revision": args.service_revision,
        "num_items": len(items),
        "num_existing_reused": len(existing),
        "num_existing_invalidated_by_input_hash": invalidated_existing,
        "num_new_embedded": len(pending),
        "num_pending_total": len(all_pending),
        "num_pending_deferred": len(all_pending) - len(pending),
        "num_output_embeddings": len(ordered_rows),
        "batch_size": args.batch_size,
        "normalized": args.normalize,
        "dim_counts": dict(sorted(Counter(row.get("dim", 0) for row in ordered_rows).items())),
    }
    write_json(args.report, report)
    print(f"[done] wrote {len(ordered_rows)} embeddings to {args.out}")
    print(f"[done] wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
