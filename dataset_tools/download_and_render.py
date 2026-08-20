#!/usr/bin/env python3
"""Download exact arXiv PDFs and render missing VisDocAgentBench pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import fitz
import requests


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_pdf(url: str, destination: Path, timeout: float, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pdf.part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
            temporary.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=Path("data/corpus/documents.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("data/corpus/pages.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pdf-cache", type=Path, default=Path("data/source_pdfs"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-documents", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = read_jsonl(args.documents)
    pages = read_jsonl(args.pages)
    pages_by_document: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        pages_by_document.setdefault(str(page["document_id"]), []).append(page)

    pending_documents: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document["document_id"])
        document_pages = pages_by_document.get(document_id, [])
        if args.overwrite or any(
            not (args.data_root / str(page["image_path"])).is_file()
            for page in document_pages
        ):
            pending_documents.append(document)
    if args.limit_documents > 0:
        pending_documents = pending_documents[: args.limit_documents]

    print(
        f"[start] documents={len(documents)} pending={len(pending_documents)} "
        f"pages={len(pages)}",
        flush=True,
    )
    for document_index, document in enumerate(pending_documents, start=1):
        document_id = str(document["document_id"])
        versioned_id = str(document["arxiv_versioned_id"])
        pdf_path = args.pdf_cache / f"{versioned_id}.pdf"
        if not pdf_path.is_file():
            print(f"[download] {document_index}/{len(pending_documents)} {versioned_id}", flush=True)
            download_pdf(str(document["pdf_url"]), pdf_path, args.timeout, args.retries)

        document_pages = sorted(
            pages_by_document[document_id], key=lambda page: int(page["page_index"])
        )
        with fitz.open(pdf_path) as pdf:
            if len(pdf) != int(document["page_count"]):
                raise ValueError(
                    f"{versioned_id}: expected {document['page_count']} pages, found {len(pdf)}"
                )
            for page_row in document_pages:
                page_index = int(page_row["page_index"])
                output_path = args.data_root / str(page_row["image_path"])
                if output_path.is_file() and not args.overwrite:
                    continue
                page = pdf.load_page(page_index - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                expected = (int(page_row["width"]), int(page_row["height"]))
                actual = (pixmap.width, pixmap.height)
                if actual != expected:
                    raise ValueError(
                        f"{page_row['page_id']}: expected dimensions {expected}, got {actual}"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pixmap.save(output_path)
        print(
            f"[render] {document_index}/{len(pending_documents)} {document_id} "
            f"pages={len(document_pages)}",
            flush=True,
        )
    missing = [
        str(page["page_id"])
        for page in pages
        if not (args.data_root / str(page["image_path"])).is_file()
    ]
    if missing:
        raise RuntimeError(f"Rendering finished with {len(missing)} missing pages")
    print(f"[done] all {len(pages)} page images are available", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
