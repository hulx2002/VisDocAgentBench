#!/usr/bin/env python3
"""Minimal read-only page OCR cache used by the OCR-text agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_common import read_jsonl


@dataclass(frozen=True)
class CachedPageOCR:
    page_id: str
    ocr_text: str
    ocr_ok: bool
    error: str = ""


def load_page_ocr_cache(
    path: Path,
    *,
    expected_page_ids: set[str] | None = None,
) -> dict[str, CachedPageOCR]:
    """Load only plain OCR text and status, deliberately dropping blocks/bboxes."""
    if not path.is_file():
        raise FileNotFoundError(f"Page OCR cache does not exist: {path}")

    by_page_id: dict[str, CachedPageOCR] = {}
    for row in read_jsonl(path):
        page_id = str(row.get("page_id") or "").strip()
        if not page_id:
            raise ValueError(f"OCR cache row is missing page_id: {path}")
        if page_id in by_page_id:
            raise ValueError(f"Duplicate page_id in OCR cache: {page_id}")
        by_page_id[page_id] = CachedPageOCR(
            page_id=page_id,
            ocr_text=str(row.get("ocr_text") or ""),
            ocr_ok=bool(row.get("ocr_ok", True)),
            error=str(row.get("error") or ""),
        )

    if expected_page_ids is not None:
        missing = sorted(expected_page_ids - set(by_page_id))
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"OCR cache is missing {len(missing)} corpus pages; first missing page_ids: {preview}"
            )
    return by_page_id
