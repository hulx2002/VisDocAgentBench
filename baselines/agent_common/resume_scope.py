#!/usr/bin/env python3
"""Prevent an in-place resume from replacing rows from a different query set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("path_id") or "").strip()


def query_text(row: dict[str, Any]) -> str:
    return str(
        row.get("manual_query")
        or row.get("query")
        or row.get("original_query")
        or ""
    )


def gold_page_id(row: dict[str, Any]) -> str:
    value = str(row.get("gold_page_id") or row.get("gold_unit_id") or "")
    if not value and row.get("gold_page_image_path"):
        value = Path(str(row["gold_page_image_path"])).stem
    return value


def annotation_contract(
    annotations: Iterable[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    contract: dict[str, tuple[str, str]] = {}
    for row in annotations:
        identifier = query_id(row)
        if not identifier:
            raise ValueError("annotation row is missing query_id/path_id")
        if identifier in contract:
            raise ValueError(f"duplicate annotation query_id={identifier}")
        contract[identifier] = (query_text(row), gold_page_id(row))
    return contract


def resume_scope_report(
    predictions: Iterable[dict[str, Any]],
    annotations: Iterable[dict[str, Any]],
    *,
    expected_fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected = annotation_contract(annotations)
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    outside_ids: list[str] = []
    query_mismatch_ids: list[str] = []
    gold_mismatch_ids: list[str] = []
    fingerprint_mismatch_ids: list[str] = []

    row_count = 0
    for row in predictions:
        row_count += 1
        identifier = query_id(row)
        if not identifier or identifier in seen:
            duplicate_ids.append(identifier)
            continue
        seen.add(identifier)
        expected_values = expected.get(identifier)
        if expected_values is None:
            outside_ids.append(identifier)
            continue
        expected_query, expected_gold = expected_values
        if query_text(row) != expected_query:
            query_mismatch_ids.append(identifier)
        if gold_page_id(row) != expected_gold:
            gold_mismatch_ids.append(identifier)
        if (
            expected_fingerprints is not None
            and str(row.get("episode_fingerprint_sha256") or "")
            != expected_fingerprints.get(identifier, "")
        ):
            fingerprint_mismatch_ids.append(identifier)

    compatible = not (
        duplicate_ids
        or outside_ids
        or query_mismatch_ids
        or gold_mismatch_ids
        or fingerprint_mismatch_ids
    )
    return {
        "compatible": compatible,
        "prediction_rows": row_count,
        "annotation_rows": len(expected),
        "duplicate_query_ids": duplicate_ids,
        "outside_query_ids": outside_ids,
        "query_mismatch_ids": query_mismatch_ids,
        "gold_mismatch_ids": gold_mismatch_ids,
        "fingerprint_mismatch_ids": fingerprint_mismatch_ids,
    }


def assert_inplace_resume_scope(
    predictions_path: Path,
    annotations: Iterable[dict[str, Any]],
    *,
    expected_fingerprints: dict[str, str] | None = None,
) -> None:
    if not predictions_path.is_file():
        return
    report = resume_scope_report(
        read_jsonl(predictions_path),
        annotations,
        expected_fingerprints=expected_fingerprints,
    )
    if not report["compatible"]:
        raise ValueError(
            "Refusing destructive in-place resume because the existing predictions "
            "belong to a different query or run configuration: "
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = resume_scope_report(
        read_jsonl(args.predictions),
        read_jsonl(args.annotations),
    )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
