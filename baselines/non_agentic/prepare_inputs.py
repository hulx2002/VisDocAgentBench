#!/usr/bin/env python3
"""Prepare query and corpus JSONL files for pure embedding baselines."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    read_jsonl,
    require_keys,
    resolve_project_path,
    sha256_text,
    write_json,
    write_jsonl,
)


DEFAULT_QUERIES = PROJECT_ROOT / "data/benchmark/queries.jsonl"
DEFAULT_EVALUATOR_ANNOTATIONS = (
    PROJECT_ROOT / "data/benchmark/evaluator_annotations.jsonl"
)
DEFAULT_PAGE_MANIFEST = PROJECT_ROOT / "data/corpus/pages.jsonl"
DEFAULT_PAGE_TEXT = PROJECT_ROOT / "data/derived/paddleocr_vl_1_6_page_ocr.jsonl"


def index_unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = row.get(key)
        if not value:
            raise ValueError(f"{label} row is missing {key}: {row}")
        if value in indexed:
            duplicates.append(str(value))
        indexed[str(value)] = row
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate {key} in {label}: {preview}")
    return indexed


def build_queries(
    input_query_rows: list[dict[str, Any]],
    evaluator_rows: list[dict[str, Any]],
    page_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    evaluator_by_id = index_unique(evaluator_rows, "query_id", "evaluator_annotations")
    query_rows: list[dict[str, Any]] = []
    missing_gold: list[str] = []
    for row in input_query_rows:
        require_keys(row, ["query_id", "query"], "query")
        evaluator = evaluator_by_id.get(str(row["query_id"]))
        if evaluator is None:
            raise ValueError(f"Missing evaluator annotation for {row['query_id']}")
        require_keys(
            evaluator,
            ["query_id", "answer_page_id", "level", "topic_id"],
            "evaluator annotation",
        )
        query_text = str(row["query"]).strip()
        if not query_text:
            raise ValueError(f"Query {row.get('query_id')} has empty text")
        gold_page_id = str(evaluator["answer_page_id"])
        if gold_page_id not in page_ids:
            missing_gold.append(gold_page_id)
        query_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": str(row["query_id"]),
                "level": evaluator.get("level"),
                "topic_id": evaluator.get("topic_id", row.get("topic_id", "")),
                "query_text": query_text,
                "text_embedding_input": query_text,
                "gold_page_id": gold_page_id,
                "query_sha256": sha256_text(query_text),
            }
        )
    return query_rows, missing_gold


def build_page_text_corpus(
    manifest_rows: list[dict[str, Any]],
    text_by_page: dict[str, dict[str, Any]],
    text_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in manifest_rows:
        require_keys(page, ["page_id", "document_id", "page_index", "image_path"], "page_manifest")
        page_id = str(page["page_id"])
        text_row = text_by_page[page_id]
        text = str(text_row.get(text_field, ""))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "page_id": page_id,
                "document_id": page.get("document_id", ""),
                "page_index": page.get("page_index", ""),
                "image_path": page.get("image_path", ""),
                "word_count": text_row.get("word_count", text_row.get("text_words", "")),
                "text": text,
                "embedding_text": text,
                "text_sha256": sha256_text(text),
            }
        )
    return rows


def validate_page_text_cache(
    *,
    page_by_id: dict[str, dict[str, Any]],
    text_by_page: dict[str, dict[str, Any]],
    text_field: str,
) -> None:
    page_ids = set(page_by_id)
    text_page_ids = set(text_by_page)
    missing = sorted(page_ids - text_page_ids)
    unexpected = sorted(text_page_ids - page_ids)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(
                f"missing {len(missing)} corpus pages (first: {', '.join(missing[:10])})"
            )
        if unexpected:
            details.append(
                f"contains {len(unexpected)} non-corpus pages "
                f"(first: {', '.join(unexpected[:10])})"
            )
        raise ValueError(
            "Page-text cache does not match the page manifest: " + "; ".join(details)
        )

    missing_field = [
        page_id for page_id, row in text_by_page.items() if text_field not in row
    ]
    if missing_field:
        preview = ", ".join(sorted(missing_field)[:10])
        raise ValueError(
            f"{len(missing_field)} page-text rows are missing field "
            f"{text_field!r}: {preview}"
        )

    failed_ocr = sorted(
        page_id
        for page_id, row in text_by_page.items()
        if "ocr_ok" in row and row.get("ocr_ok") is not True
    )
    if failed_ocr:
        preview = ", ".join(failed_ocr[:10])
        raise ValueError(
            f"Page-text cache contains {len(failed_ocr)} failed OCR rows; "
            f"rebuild them before preparing retrieval inputs (first: {preview})"
        )


def build_page_image_corpus(manifest_rows: list[dict[str, Any]], project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for page in manifest_rows:
        require_keys(page, ["page_id", "document_id", "page_index", "image_path"], "page_manifest")
        image_path = str(page.get("image_path", ""))
        if not resolve_project_path(image_path, project_root).exists():
            missing_images.append(image_path)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "page_id": str(page["page_id"]),
                "document_id": page.get("document_id", ""),
                "page_index": page.get("page_index", ""),
                "image_path": image_path,
                "width": page.get("width", ""),
                "height": page.get("height", ""),
            }
        )
    return rows, missing_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--evaluator-annotations",
        type=Path,
        default=DEFAULT_EVALUATOR_ANNOTATIONS,
    )
    parser.add_argument("--page-manifest", type=Path, default=DEFAULT_PAGE_MANIFEST)
    parser.add_argument("--page-text", type=Path, default=DEFAULT_PAGE_TEXT)
    parser.add_argument(
        "--page-text-field",
        default="ocr_text",
        help="Field containing page text in --page-text (use ocr_text for the PaddleOCR-VL cache).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--query-set-name", default="visdocagentbench")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_rows_input = read_jsonl(args.queries)
    evaluator_rows = read_jsonl(args.evaluator_annotations)
    manifest_rows = read_jsonl(args.page_manifest)
    text_rows = read_jsonl(args.page_text)

    if not args.page_text_field:
        raise ValueError("--page-text-field must be non-empty")

    page_by_id = index_unique(manifest_rows, "page_id", "page_manifest")
    text_by_page = index_unique(text_rows, "page_id", "page_text")
    validate_page_text_cache(
        page_by_id=page_by_id,
        text_by_page=text_by_page,
        text_field=args.page_text_field,
    )

    query_rows, missing_gold = build_queries(
        query_rows_input, evaluator_rows, set(page_by_id)
    )

    page_text_rows = build_page_text_corpus(manifest_rows, text_by_page, args.page_text_field)
    page_image_rows, missing_images = build_page_image_corpus(manifest_rows, args.project_root)

    query_path = args.out_dir / f"queries.{args.query_set_name}.jsonl"
    text_path = args.out_dir / "page_text_corpus.jsonl"
    image_path = args.out_dir / "page_image_corpus.jsonl"
    report_path = args.out_dir / "prepare_report.json"

    write_jsonl(query_path, query_rows)
    write_jsonl(text_path, page_text_rows)
    write_jsonl(image_path, page_image_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "input_files": {
            "queries": str(args.queries),
            "evaluator_annotations": str(args.evaluator_annotations),
            "page_manifest": str(args.page_manifest),
            "page_text": str(args.page_text),
            "page_text_field": args.page_text_field,
        },
        "output_files": {
            "queries": str(query_path),
            "page_text_corpus": str(text_path),
            "page_image_corpus": str(image_path),
            "report": str(report_path),
        },
        "num_queries": len(query_rows),
        "num_pages": len(page_by_id),
        "num_page_text_rows": len(page_text_rows),
        "num_page_image_rows": len(page_image_rows),
        "missing_gold_page_ids": sorted(set(missing_gold)),
        "missing_image_paths": sorted(set(missing_images)),
        "level_counts": dict(sorted(Counter(str(row.get("level")) for row in query_rows).items())),
        "topic_counts": dict(sorted(Counter(str(row.get("topic_id")) for row in query_rows).items())),
        "prompting_mode": "zero_instruction",
        "text_query_input": "raw query_text",
        "text_page_input": f"raw {args.page_text_field} page text",
        "vl_query_input": "raw query_text",
        "vl_page_input": "raw rendered page image",
    }
    write_json(report_path, report)

    if missing_gold:
        raise RuntimeError(f"{len(set(missing_gold))} gold page ids are missing from the corpus")
    if missing_images:
        raise RuntimeError(f"{len(set(missing_images))} page image paths are missing")

    print(f"[done] wrote {len(query_rows)} queries to {query_path}")
    print(f"[done] wrote {len(page_text_rows)} text pages to {text_path}")
    print(f"[done] wrote {len(page_image_rows)} image pages to {image_path}")
    print(f"[done] wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
