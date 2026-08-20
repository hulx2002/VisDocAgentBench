#!/usr/bin/env python3
"""Corpus loading and opaque handles for the VisDocAgentBench OCR-text agent."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_common import PROJECT_ROOT, read_jsonl


@dataclass(frozen=True)
class PageRecord:
    page_handle: str
    page_id: str
    image_path: Path
    page_width: int
    page_height: int


class PageHandleSpace:
    """Deterministically shuffled opaque handle mapping."""

    def __init__(self, pages: list[PageRecord]):
        self.pages = pages
        self.by_handle = {page.page_handle: page for page in pages}
        self.by_page_id = {page.page_id: page for page in pages}
        if len(self.by_handle) != len(pages):
            raise ValueError("Duplicate page handles")
        if len(self.by_page_id) != len(pages):
            raise ValueError("Duplicate page ids")

    def require_page(self, page_handle: str) -> PageRecord:
        page = self.by_handle.get(page_handle)
        if page is None:
            raise KeyError(f"Unknown page_handle: {page_handle}")
        return page

    def page_id_to_handle(self, page_id: str) -> str:
        page = self.by_page_id.get(page_id)
        if page is None:
            raise KeyError(f"Unknown page_id: {page_id}")
        return page.page_handle

    def page_handle_to_id(self, page_handle: str) -> str:
        return self.require_page(page_handle).page_id

    def public_mapping_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "page_handle": page.page_handle,
                "page_id": page.page_id,
                "page_width": page.page_width,
                "page_height": page.page_height,
            }
            for page in self.pages
        ]


def infer_dataset_root(page_manifest: Path) -> Path:
    if page_manifest.parent.name == "corpus":
        return page_manifest.parent.parent
    return page_manifest.parent


def resolve_page_image_path(row: dict[str, Any], page_manifest: Path, project_root: Path = PROJECT_ROOT) -> Path:
    dataset_root = infer_dataset_root(page_manifest)
    page_id = str(row["page_id"])
    doc_id = str(
        row.get("document_id") or row.get("doc_id") or page_id.rsplit("_p", 1)[0]
    )
    raw_path = str(row.get("image_path") or row.get("page_image_path") or "")

    candidates: list[Path] = [
        dataset_root / "corpus" / "pages" / doc_id / f"{page_id}.png",
        dataset_root / "pages" / doc_id / f"{page_id}.png",
    ]
    if raw_path:
        raw = Path(raw_path)
        candidates.append(raw if raw.is_absolute() else project_root / raw)
        candidates.append(page_manifest.parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve image for page_id={page_id}. Checked:\n{checked}")


def load_handle_space(page_manifest: Path, handle_seed: int, project_root: Path = PROJECT_ROOT) -> PageHandleSpace:
    rows = read_jsonl(page_manifest)
    rows = sorted(rows, key=lambda row: str(row["page_id"]))
    rng = random.Random(handle_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    pages: list[PageRecord] = []
    for index, row in enumerate(shuffled, start=1):
        page_id = str(row["page_id"])
        image_path = resolve_page_image_path(row, page_manifest, project_root)
        pages.append(
            PageRecord(
                page_handle=f"page_{index:06d}",
                page_id=page_id,
                image_path=image_path,
                page_width=int(row.get("width") or row.get("page_width") or 0),
                page_height=int(row.get("height") or row.get("page_height") or 0),
            )
        )
    return PageHandleSpace(pages)
