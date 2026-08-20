#!/usr/bin/env python3
"""Build the full-page PaddleOCR-VL cache used by the OCR-text route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIR = PROJECT_ROOT / "baselines/agent_visual"
if str(VISUAL_DIR) not in sys.path:
    sys.path.insert(0, str(VISUAL_DIR))

from common import sha256_file, sha256_text  # noqa: E402
from corpus import resolve_page_image_path  # noqa: E402
from ocr_backends import build_ocr_backend  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def index_unique_rows(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        page_id = str(row.get("page_id") or "").strip()
        if not page_id:
            raise ValueError(f"{label} row is missing page_id")
        if page_id in indexed:
            raise ValueError(f"Duplicate page_id in {label}: {page_id}")
        indexed[page_id] = row
    return indexed


def ocr_config_hash(args: argparse.Namespace) -> str:
    values = {
        "ocr_backend": str(args.ocr_backend),
        "paddleocr_pipeline_version": str(args.paddleocr_pipeline_version),
        "paddleocr_model_dir": str(Path(args.paddleocr_model_dir).expanduser().resolve()),
        "paddleocr_layout_model_dir": str(
            Path(args.paddleocr_layout_model_dir).expanduser().resolve()
        ),
        "paddleocr_vl_rec_server_url": str(args.paddleocr_vl_rec_server_url),
    }
    return sha256_text(json.dumps(values, ensure_ascii=False, sort_keys=True))


def reusable_existing_rows(
    *,
    existing_by_id: dict[str, dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    page_manifest: Path,
    config_hash: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    reusable: dict[str, dict[str, Any]] = {}
    invalidated = 0
    for page_id, row in existing_by_id.items():
        manifest_row = manifest_by_id.get(page_id)
        if manifest_row is None:
            continue
        if row.get("ocr_ok") is not True:
            invalidated += 1
            continue
        if str(row.get("ocr_config_sha256") or "") != config_hash:
            invalidated += 1
            continue
        image_path = resolve_page_image_path(manifest_row, page_manifest, PROJECT_ROOT)
        if str(row.get("image_sha256") or "") != sha256_file(image_path):
            invalidated += 1
            continue
        reusable[page_id] = row
    return reusable, invalidated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-manifest", type=Path, default=Path("data/corpus/pages.jsonl"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/derived/paddleocr_vl_1_6_page_ocr.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum missing or stale pages to process; current cache rows are preserved.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--ocr-backend", default="paddleocr_vl")
    parser.add_argument(
        "--paddleocr-model-dir",
        default=str(PROJECT_ROOT / "models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument(
        "--paddleocr-layout-model-dir",
        default=str(PROJECT_ROOT / "models/PP-DocLayoutV3"),
    )
    parser.add_argument("--paddleocr-pipeline-version", default="v1.6")
    parser.add_argument("--paddleocr-vl-rec-server-url", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    manifest = sorted(read_jsonl(args.page_manifest), key=lambda row: str(row["page_id"]))
    manifest_by_id = index_unique_rows(manifest, label="page manifest")
    existing_by_id = (
        index_unique_rows(read_jsonl(args.out), label="OCR cache")
        if args.resume and args.out.is_file()
        else {}
    )
    config_hash = ocr_config_hash(args)
    reusable, invalidated_existing = reusable_existing_rows(
        existing_by_id=existing_by_id,
        manifest_by_id=manifest_by_id,
        page_manifest=args.page_manifest,
        config_hash=config_hash,
    )
    all_pending = [row for row in manifest if str(row["page_id"]) not in reusable]
    pending = all_pending[: args.limit] if args.limit > 0 else all_pending
    print(
        f"[start] corpus_pages={len(manifest)} reusable={len(reusable)} "
        f"stale_or_missing={len(all_pending)} processing={len(pending)} "
        f"invalidated={invalidated_existing}",
        flush=True,
    )
    output = dict(reusable)
    current_page_ids = set(reusable)
    ordered = [
        output[str(item["page_id"])]
        for item in manifest
        if str(item["page_id"]) in output
    ]
    write_jsonl(args.out, ordered)
    backend = build_ocr_backend(args) if pending else None
    for index, page in enumerate(pending, start=1):
        page_id = str(page["page_id"])
        image_path = resolve_page_image_path(page, args.page_manifest, PROJECT_ROOT)
        image_hash = sha256_file(image_path)
        try:
            result = backend.recognize(image_path) if backend is not None else {}
            text = str(result.get("ocr_text") or "")
            blocks = result.get("blocks") if isinstance(result.get("blocks"), list) else []
            row = {
                "page_id": page_id,
                "document_id": str(page.get("document_id") or ""),
                "page_index": int(page["page_index"]),
                "image_sha256": image_hash,
                "ocr_config_sha256": config_hash,
                "ocr_ok": True,
                "ocr_engine": str(result.get("engine") or ""),
                "ocr_text": text,
                "blocks": blocks,
                "text_chars": len(text),
                "text_words": len(text.split()),
            }
        except Exception as exc:
            if args.fail_fast:
                raise
            row = {
                "page_id": page_id,
                "document_id": str(page.get("document_id") or ""),
                "page_index": int(page["page_index"]),
                "image_sha256": image_hash,
                "ocr_config_sha256": config_hash,
                "ocr_ok": False,
                "ocr_engine": "",
                "ocr_text": "",
                "blocks": [],
                "text_chars": 0,
                "text_words": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        output[page_id] = row
        if row["ocr_ok"]:
            current_page_ids.add(page_id)
        else:
            current_page_ids.discard(page_id)
        ordered = [
            output[str(item["page_id"])]
            for item in manifest
            if str(item["page_id"]) in output
        ]
        write_jsonl(args.out, ordered)
        print(
            f"[ocr] {index}/{len(pending)} {page_id} "
            f"words={row['text_words']} ok={row['ocr_ok']}",
            flush=True,
        )
    ordered = [
        output[str(item["page_id"])]
        for item in manifest
        if str(item["page_id"]) in output
    ]
    write_jsonl(args.out, ordered)
    remaining_page_ids = sorted(set(manifest_by_id) - current_page_ids)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_manifest_pages": len(manifest),
        "num_pages": len(ordered),
        "num_successful": sum(row.get("ocr_ok") is True for row in ordered),
        "num_failed": sum(row.get("ocr_ok") is not True for row in ordered),
        "num_reusable": len(reusable),
        "num_invalidated_existing": invalidated_existing,
        "num_processed_this_run": len(pending),
        "num_current_pages": len(current_page_ids),
        "num_remaining_pages": len(remaining_page_ids),
        "remaining_page_ids": remaining_page_ids,
        "cache_complete": not remaining_page_ids,
        "ocr_config_sha256": config_hash,
        "output": str(args.out),
    }
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[done] wrote {len(ordered)} OCR rows to {args.out}; "
        f"current={len(current_page_ids)}/{len(manifest)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
