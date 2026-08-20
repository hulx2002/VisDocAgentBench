#!/usr/bin/env python3
"""Cache Qwen3-VL embeddings through a vLLM /v1/embeddings service."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from common import (
    PROJECT_ROOT,
    SCHEMA_VERSION,
    normalize_vector,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
EMBEDDING_CACHE_VERSION = "qwen3_vl_embedding_cache_v2"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def embedding_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/embeddings"


def image_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".png":
        return "png"
    if suffix == ".webp":
        return "webp"
    return "jpeg"


def image_to_data_url(path: Path, max_pixels: int, max_file_size_mb: float) -> str:
    raw = path.read_bytes()
    max_bytes = int(max_file_size_mb * 1024 * 1024)
    fmt = image_format(path)

    if len(raw) <= max_bytes:
        try:
            with Image.open(io.BytesIO(raw)) as img:
                if img.width * img.height <= max_pixels:
                    return f"data:image/{fmt};base64,{base64.b64encode(raw).decode('utf-8')}"
        except Exception:
            return f"data:image/{fmt};base64,{base64.b64encode(raw).decode('utf-8')}"

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB") if img.mode not in {"RGB", "L"} else img
        pixels = img.width * img.height
        if pixels > max_pixels:
            scale = (max_pixels / float(pixels)) ** 0.5
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        quality = 95
        while quality >= 60:
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            if out.tell() <= max_bytes or quality == 60:
                data = out.getvalue()
                return f"data:image/jpeg;base64,{base64.b64encode(data).decode('utf-8')}"
            quality -= 10 if quality > 85 else 5

    raise RuntimeError(f"Failed to encode image as data URL: {path}")


def input_identity(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "cache_version": EMBEDDING_CACHE_VERSION,
        "base_url": normalize_base_url(args.base_url),
        "model": args.model,
        "service_revision": args.service_revision,
        "normalized": bool(args.normalize),
        "id_field": args.id_field,
        "text_field": args.text_field,
        "image_field": args.image_field,
        "payload_format": "qwen3_vl_messages_empty_assistant",
        "max_pixels": int(args.max_pixels),
        "max_file_size_mb": float(args.max_file_size_mb),
    }
    if args.text_field:
        identity["text"] = str(item.get(args.text_field, ""))
    if args.image_field:
        image_value = str(item.get(args.image_field, ""))
        if not image_value:
            raise ValueError(
                f"Missing image field {args.image_field}: {item.get(args.id_field)}"
            )
        image_path = resolve_project_path(image_value, args.project_root).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image path for {item.get(args.id_field)}: {image_path}"
            )
        stat = image_path.stat()
        identity["image"] = {
            "declared_path": image_value,
            "resolved_path": str(image_path),
            "size": stat.st_size,
            "sha256": sha256_file(image_path),
        }
    return identity


def build_content_parts(item: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    parts: list[dict[str, Any]] = []
    identity = input_identity(item, args)

    if args.text_field:
        text = str(item.get(args.text_field, ""))
        parts.append({"type": "text", "text": text})

    if args.image_field:
        image_path = Path(identity["image"]["resolved_path"])
        data_url = image_to_data_url(image_path, args.max_pixels, args.max_file_size_mb)
        parts.append({"type": "image_url", "image_url": {"url": data_url}})

    if not parts:
        raise ValueError(f"Input row has neither text nor image: {item.get(args.id_field)}")

    return parts, sha256_json(identity)


def build_payload(model: str, content_parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": content_parts},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def parse_embedding(result: dict[str, Any]) -> list[float]:
    embedding = None
    data = result.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            embedding = first.get("embedding")
        elif isinstance(first, list):
            embedding = first
    elif isinstance(data, dict):
        embedding = data.get("embedding")
    if embedding is None:
        embedding = result.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"Unable to extract embedding from response keys: {list(result.keys())}")
    return [float(value) for value in embedding]


def call_embedding(
    base_url: str,
    model: str,
    api_key: str,
    content_parts: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> list[float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = build_payload(model, content_parts)
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
            return parse_embedding(response.json())
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                status = getattr(exc.response, "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS_CODES:
                    raise
            if attempts is not None and attempt >= attempts:
                break
            total = "inf" if attempts is None else str(attempts)
            print(f"[retry] vl embedding item: {type(exc).__name__}: {exc}; sleeping {retry_backoff:.1f}s before attempt {attempt + 1}/{total}")
            time.sleep(retry_backoff)
            attempt += 1
    raise RuntimeError("VL embedding request failed") from last_error


def load_existing(path: Path, id_field: str, model: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = row.get(id_field)
        if item_id and row.get("model") == model:
            existing[str(item_id)] = row
    return existing


def make_embedding_row(
    item: dict[str, Any],
    id_field: str,
    model: str,
    embedding: list[float],
    normalize: bool,
    input_sha256: str,
    base_url: str,
    service_revision: str,
) -> dict[str, Any]:
    if normalize:
        embedding = normalize_vector(embedding)
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
    for key in (
        "query_id",
        "page_id",
        "gold_page_id",
        "level",
        "topic_id",
        "doc_id",
        "arxiv_id",
        "page_index",
        "page_image_path",
    ):
        if key in item and key not in row:
            row[key] = item[key]
    return row


def encode_one(item: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    content_parts, fingerprint = build_content_parts(item, args)
    embedding = call_embedding(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        content_parts=content_parts,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    row = make_embedding_row(
        item=item,
        id_field=args.id_field,
        model=args.model,
        embedding=embedding,
        normalize=args.normalize,
        input_sha256=fingerprint,
        base_url=args.base_url,
        service_revision=args.service_revision,
    )
    return str(item[args.id_field]), row


def current_fingerprint(item: dict[str, Any], args: argparse.Namespace) -> str:
    return sha256_json(input_identity(item, args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--id-field", required=True)
    parser.add_argument("--text-field", default="")
    parser.add_argument("--image-field", default="")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--service-revision",
        default=os.environ.get("EMBEDDING_SERVICE_REVISION", ""),
        help="Optional immutable revision label for a mutable embedding endpoint.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum missing or stale items to process; current cache rows are preserved.",
    )
    parser.add_argument("--max-pixels", type=int, default=36_000_000)
    parser.add_argument("--max-file-size-mb", type=float, default=10.0)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.text_field and not args.image_field:
        raise ValueError("Provide --text-field and/or --image-field")

    items = read_jsonl(args.items)
    for item in items:
        if item.get(args.id_field) in (None, ""):
            raise ValueError(f"Missing id field {args.id_field}: {item}")

    existing = load_existing(args.out, args.id_field, args.model) if args.resume else {}
    invalidated_existing = 0
    if existing:
        item_by_id = {str(item[args.id_field]): item for item in items}
        filtered_existing: dict[str, dict[str, Any]] = {}
        for item_id, row in existing.items():
            item = item_by_id.get(item_id)
            if item is None:
                continue
            if row.get("input_sha256") == current_fingerprint(item, args):
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
    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(encode_one, item, args): item for item in pending}
            for future in as_completed(futures):
                item_id, row = future.result()
                output_by_id[item_id] = row
                completed += 1
                if completed == len(pending) or completed % max(1, args.workers) == 0:
                    print(f"[embed] {len(existing) + completed}/{len(items)} items")

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
        "image_field": args.image_field,
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
        "workers": args.workers,
        "timeout": args.timeout,
        "max_pixels": args.max_pixels,
        "max_file_size_mb": args.max_file_size_mb,
        "normalized": args.normalize,
        "prompting_mode": "zero_instruction",
        "payload_format": "qwen3-vl-embedding messages with empty assistant final message",
        "dim_counts": dict(sorted(Counter(row.get("dim", 0) for row in ordered_rows).items())),
    }
    write_json(args.report, report)
    print(f"[done] wrote {len(ordered_rows)} embeddings to {args.out}")
    print(f"[done] wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
