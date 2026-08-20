#!/usr/bin/env python3
"""Validate the public VisDocAgentBench data layout and benchmark invariants."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{label} row has no {key}")
        if value in result:
            raise ValueError(f"Duplicate {label} {key}: {value}")
        result[value] = row
    return result


def validate_page_images(rows: list[dict[str, Any]], data_root: Path) -> None:
    missing: list[str] = []
    unreadable: list[str] = []
    wrong_size: list[str] = []
    for row in rows:
        page_id = str(row["page_id"])
        image_path = data_root / str(row["image_path"])
        if not image_path.is_file():
            missing.append(page_id)
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                actual_size = image.size
                image_format = image.format
        except (OSError, ValueError) as exc:
            unreadable.append(f"{page_id} ({type(exc).__name__}: {exc})")
            continue
        if image_format != "PNG":
            unreadable.append(f"{page_id} (detected format: {image_format or 'unknown'})")
            continue

        expected_size = (int(row.get("width") or 0), int(row.get("height") or 0))
        if actual_size != expected_size:
            wrong_size.append(
                f"{page_id} (expected {expected_size[0]}x{expected_size[1]}, "
                f"found {actual_size[0]}x{actual_size[1]})"
            )

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required page images; first page_ids: "
            f"{', '.join(missing[:10])}"
        )
    if unreadable:
        raise ValueError(
            f"Could not decode {len(unreadable)} required PNG images; first: "
            f"{'; '.join(unreadable[:10])}"
        )
    if wrong_size:
        raise ValueError(
            f"Found {len(wrong_size)} page images whose dimensions differ from "
            f"the manifest; first: {'; '.join(wrong_size[:10])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--require-complete-corpus",
        action="store_true",
        help="Require all 2,375 page images, including locally reconstructed pages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries = read_jsonl(args.data_root / "benchmark/queries.jsonl")
    evaluator = read_jsonl(args.data_root / "benchmark/evaluator_annotations.jsonl")
    documents = read_jsonl(args.data_root / "corpus/documents.jsonl")
    pages = read_jsonl(args.data_root / "corpus/pages.jsonl")
    query_by_id = unique_index(queries, "query_id", "query")
    evaluator_by_id = unique_index(evaluator, "query_id", "evaluator annotation")
    page_by_id = unique_index(pages, "page_id", "page")
    unique_index(documents, "document_id", "document")

    if set(query_by_id) != set(evaluator_by_id):
        raise ValueError("Query and evaluator query_id sets differ")
    if len(queries) != 120 or len(documents) != 100 or len(pages) != 2375:
        raise ValueError(
            f"Expected 120 queries, 100 documents, and 2,375 pages; found "
            f"{len(queries)}, {len(documents)}, and {len(pages)}"
        )
    levels = Counter(int(row["level"]) for row in evaluator)
    if levels != Counter({1: 40, 2: 40, 3: 40}):
        raise ValueError(f"Unexpected level distribution: {levels}")

    answers: list[str] = []
    for row in evaluator:
        level = int(row["level"])
        answer = str(row["answer_page_id"])
        supports = [str(value) for value in row.get("support_page_ids", [])]
        if answer not in page_by_id or any(value not in page_by_id for value in supports):
            raise ValueError(f"Unknown answer/support page in {row['query_id']}")
        if len(supports) != level - 1 or len(set(supports)) != len(supports):
            raise ValueError(f"Invalid support path in {row['query_id']}")
        if answer in supports:
            raise ValueError(f"Answer is also a support page in {row['query_id']}")
        answers.append(answer)
    if len(set(answers)) != 120:
        raise ValueError("Answer pages are not unique")

    required_files = [
        row
        for row in pages
        if args.require_complete_corpus or bool(row.get("image_included"))
    ]
    validate_page_images(required_files, args.data_root)
    print(
        f"[ok] queries=120 levels=40/40/40 documents=100 pages=2375 "
        f"files_checked={len(required_files)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
