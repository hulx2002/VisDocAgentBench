#!/usr/bin/env python3
"""Shared utilities for non-agentic VisDocAgentBench baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")
SCHEMA_VERSION = "VisDocAgentBenchEmbeddingRetrievalV1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embedding_retrieval"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    _atomic_replace(path, writer)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in materialized:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    _atomic_replace(path, writer)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Describe one file by its resolved location and content."""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def file_collection_identity(
    files: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    """Return a compact, deterministic identity for an ordered file collection."""
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for label, path in sorted(files, key=lambda item: item[0]):
        identity = file_identity(path)
        total_bytes += int(identity["size"])
        records.append({"label": str(label), **identity})
    return {
        "num_files": len(records),
        "total_bytes": total_bytes,
        "sha256": sha256_json(records),
    }


def directory_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for all regular files below a directory."""
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_file():
        return {"path": str(resolved), "exists": True, **file_identity(resolved)}
    files = [
        (child.relative_to(resolved).as_posix(), child)
        for child in resolved.rglob("*")
        if child.is_file()
    ]
    return {
        "path": str(resolved),
        "exists": True,
        **file_collection_identity(files),
    }


def normalize_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def resolve_project_path(
    path_value: str | Path, project_root: Path = PROJECT_ROOT
) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def require_keys(row: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{label} row is missing required keys: {missing}")


def query_text(row: dict[str, Any]) -> str:
    text = str(
        row.get("query_text")
        or row.get("manual_query")
        or row.get("query")
        or row.get("original_query")
        or ""
    ).strip()
    if not text:
        raise ValueError(f"query {row.get('query_id')} has no text")
    return text


def gold_page_id(row: dict[str, Any]) -> str:
    page_id = str(row.get("gold_page_id") or row.get("gold_unit_id") or "").strip()
    if not page_id and row.get("gold_page_image_path"):
        page_id = Path(str(row["gold_page_image_path"])).stem
    if not page_id:
        raise ValueError(f"query {row.get('query_id')} has no gold page id")
    return page_id


def valid_top10(
    ranking: list[str], *, valid_page_ids: set[str] | None = None
) -> list[str] | None:
    top10 = ranking[:10]
    if len(top10) != 10 or len(set(top10)) != 10:
        return None
    if valid_page_ids is not None and any(
        page_id not in valid_page_ids for page_id in top10
    ):
        return None
    return top10


def rank_metrics(
    gold: str,
    ranking: list[str],
    *,
    valid_page_ids: set[str] | None = None,
) -> dict[str, float]:
    top10 = valid_top10(ranking, valid_page_ids=valid_page_ids)
    rank: int | None = None
    if top10 is not None:
        for index, page_id in enumerate(top10, start=1):
            if page_id == gold:
                rank = index
                break
    return {
        "recall@1": float(rank == 1),
        "recall@3": float(rank is not None and rank <= 3),
        "recall@5": float(rank is not None and rank <= 5),
        "recall@10": float(rank is not None and rank <= 10),
        "mrr@10": 0.0 if rank is None else 1.0 / rank,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_predictions(
    queries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    valid_page_ids: set[str] | None = None,
) -> dict[str, Any]:
    by_prediction = {str(row.get("query_id")): row for row in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError("duplicate query ids in predictions")

    scored: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query.get("query_id") or "")
        prediction = by_prediction.get(query_id, {})
        ranking = [str(value) for value in prediction.get("ranked_page_ids", [])]
        top10 = valid_top10(ranking, valid_page_ids=valid_page_ids)
        valid = bool(
            prediction.get("status", "submitted") == "submitted"
            and top10 is not None
        )
        metrics = rank_metrics(
            gold_page_id(query),
            top10 if valid else [],
            valid_page_ids=valid_page_ids,
        )
        scored.append(
            {
                "query_id": query_id,
                "level": str(query.get("level", "")),
                "topic_id": str(query.get("topic_id", "")),
                "valid": valid,
                **metrics,
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_queries": len(rows),
            "num_valid": sum(bool(row["valid"]) for row in rows),
            **{metric: _mean([float(row[metric]) for row in rows]) for metric in METRIC_KEYS},
        }

    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_level[row["level"]].append(row)
        by_topic[row["topic_id"]].append(row)
    return {
        "metrics": list(METRIC_KEYS),
        "overall": summarize(scored),
        "by_level": {key: summarize(value) for key, value in sorted(by_level.items())},
        "by_topic": {key: summarize(value) for key, value in sorted(by_topic.items())},
    }


def resolve_page_image(row: dict[str, Any], page_manifest: Path) -> Path:
    page_id = str(row["page_id"])
    doc_id = str(
        row.get("document_id") or row.get("doc_id") or page_id.rsplit("_p", 1)[0]
    )
    dataset_root = page_manifest.parent.parent if page_manifest.parent.name == "corpus" else page_manifest.parent
    candidates = [
        dataset_root / "corpus" / "pages" / doc_id / f"{page_id}.png",
        dataset_root / "pages" / doc_id / f"{page_id}.png",
    ]
    raw_path = str(row.get("image_path") or row.get("page_image_path") or "")
    if raw_path:
        raw = Path(raw_path)
        candidates.extend(
            [raw if raw.is_absolute() else PROJECT_ROOT / raw, page_manifest.parent / raw]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve image for {page_id}: {candidates}")


def opaque_handle_maps(
    page_rows: list[dict[str, Any]], handle_seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(page_rows, key=lambda row: str(row["page_id"]))
    random.Random(handle_seed).shuffle(ordered)
    page_to_handle = {
        str(row["page_id"]): f"page_{index:06d}"
        for index, row in enumerate(ordered, start=1)
    }
    return page_to_handle, {handle: page for page, handle in page_to_handle.items()}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
