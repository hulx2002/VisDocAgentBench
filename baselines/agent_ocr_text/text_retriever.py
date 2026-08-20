#!/usr/bin/env python3
"""Qwen text retriever used by the VisDocAgentBench OCR-text agent."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from text_common import append_jsonl, read_jsonl, sha256_file, sha256_text
from text_corpus import PageHandleSpace


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
PAGE_MATRIX_CACHE_VERSION = "ocr_text_page_matrix_v2"
QUERY_EMBEDDING_CACHE_VERSION = "ocr_text_query_embedding_v2"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def embedding_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/embeddings"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def parse_embedding(result: dict[str, Any]) -> list[float]:
    embedding: Any = None
    data = result.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            embedding = first.get("embedding")
        else:
            embedding = first
    elif isinstance(data, dict):
        embedding = data.get("embedding")
    if embedding is None:
        embedding = result.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"Unable to extract embedding from response keys: {list(result.keys())}")
    return [float(value) for value in embedding]


def call_text_embedding(
    *,
    base_url: str,
    model: str,
    api_key: str,
    text: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> list[float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": [text]}
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
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                status = getattr(exc.response, "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS_CODES:
                    raise
            if attempts is not None and attempt >= attempts:
                break
            total = "inf" if attempts is None else str(attempts)
            print(
                f"[retry] text_search query embedding: {type(exc).__name__}: {exc}; "
                f"sleeping {retry_backoff:.1f}s before attempt {attempt + 1}/{total}",
                flush=True,
            )
            time.sleep(retry_backoff)
            attempt += 1
    raise RuntimeError("Text query embedding request failed") from last_error


class TextRetriever:
    def __init__(
        self,
        *,
        page_embeddings_path: Path,
        handle_space: PageHandleSpace,
        base_url: str,
        model: str,
        api_key: str,
        cache_path: Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 20,
        retry_backoff: float = 5.0,
        matrix_cache_path: Path | None = None,
        service_revision: str = "",
    ):
        self.page_embeddings_path = page_embeddings_path
        self.handle_space = handle_space
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.cache_path = cache_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.service_revision = service_revision
        self.query_cache = self._load_query_cache(cache_path)
        self.query_cache_lock = threading.Lock()
        self.page_handles, self.page_matrix = self._load_page_matrix(
            page_embeddings_path=page_embeddings_path,
            handle_space=handle_space,
            matrix_cache_path=matrix_cache_path,
        )

    def _load_query_cache(self, cache_path: Path | None) -> dict[str, list[float]]:
        if cache_path is None or not cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        for row in read_jsonl(cache_path):
            key = str(row.get("cache_key", ""))
            embedding = row.get("embedding")
            if key and isinstance(embedding, list):
                cache[key] = [float(value) for value in embedding]
        return cache

    def _load_page_matrix(
        self,
        *,
        page_embeddings_path: Path,
        handle_space: PageHandleSpace,
        matrix_cache_path: Path | None,
    ) -> tuple[list[str], np.ndarray]:
        expected_page_ids = [page.page_id for page in handle_space.pages]
        page_handles = [page.page_handle for page in handle_space.pages]
        resolved_embeddings = page_embeddings_path.resolve()
        source_stat = resolved_embeddings.stat()
        cache_identity = {
            "cache_version": PAGE_MATRIX_CACHE_VERSION,
            "source": {
                "path": str(resolved_embeddings),
                "size": source_stat.st_size,
                "sha256": sha256_file(resolved_embeddings),
            },
            "embedding_model": self.model,
            "normalization": "l2",
            "page_ids_sha256": sha256_text(
                json.dumps(expected_page_ids, ensure_ascii=False, separators=(",", ":"))
            ),
        }
        if matrix_cache_path is not None and matrix_cache_path.exists():
            try:
                with np.load(matrix_cache_path, allow_pickle=False) as data:
                    cached_page_ids = [str(value) for value in data["page_ids"].tolist()]
                    cached_matrix = data["page_matrix"].astype(np.float32)
                    metadata = json.loads(str(data["metadata_json"].item()))
                if (
                    cached_page_ids == expected_page_ids
                    and metadata.get("identity") == cache_identity
                    and cached_matrix.ndim == 2
                    and cached_matrix.shape[0] == len(expected_page_ids)
                    and int(metadata.get("embedding_dim", -1)) == cached_matrix.shape[1]
                ):
                    return page_handles, cached_matrix
            except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            print(f"[cache] ignoring stale page text matrix cache: {matrix_cache_path}", flush=True)

        print(f"[load] reading OCR text page embeddings: {page_embeddings_path}", flush=True)
        rows = read_jsonl(page_embeddings_path)
        print(f"[load] loaded {len(rows)} OCR text page embedding rows", flush=True)
        embedding_by_page_id: dict[str, list[float]] = {}
        for row in rows:
            page_id = str(row.get("page_id", ""))
            embedding = row.get("embedding")
            if not page_id or not isinstance(embedding, list):
                raise ValueError(f"Invalid page embedding row in {page_embeddings_path}")
            embedding_by_page_id[page_id] = [float(value) for value in embedding]

        vectors: list[list[float]] = []
        for page in handle_space.pages:
            if page.page_id not in embedding_by_page_id:
                raise ValueError(f"Missing OCR text page embedding for {page.page_id}")
            vectors.append(embedding_by_page_id[page.page_id])
        page_matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if page_matrix.ndim != 2 or page_matrix.shape[0] != len(expected_page_ids):
            raise ValueError(f"Invalid OCR text page embedding matrix shape: {page_matrix.shape}")
        if matrix_cache_path is not None:
            matrix_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = matrix_cache_path.with_name(f".{matrix_cache_path.name}.tmp.npz")
            metadata = {
                "identity": cache_identity,
                "embedding_dim": int(page_matrix.shape[1]),
            }
            np.savez_compressed(
                tmp_path,
                page_ids=np.asarray(expected_page_ids),
                page_matrix=page_matrix,
                metadata_json=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
            tmp_path.replace(matrix_cache_path)
            print(f"[cache] wrote page text matrix cache: {matrix_cache_path}", flush=True)
        return page_handles, page_matrix

    def embed_query(self, query: str) -> np.ndarray:
        cache_key = sha256_text(
            "\n".join(
                [
                    QUERY_EMBEDDING_CACHE_VERSION,
                    normalize_base_url(self.base_url),
                    self.model,
                    self.service_revision,
                    query,
                ]
            )
        )
        with self.query_cache_lock:
            embedding = self.query_cache.get(cache_key)
        if embedding is None:
            embedding = call_text_embedding(
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                text=query,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_backoff=self.retry_backoff,
            )
            with self.query_cache_lock:
                existing = self.query_cache.get(cache_key)
                if existing is None:
                    self.query_cache[cache_key] = embedding
                    if self.cache_path is not None:
                        append_jsonl(
                            self.cache_path,
                            {
                                "cache_key": cache_key,
                                "cache_version": QUERY_EMBEDDING_CACHE_VERSION,
                                "base_url": normalize_base_url(self.base_url),
                                "model": self.model,
                                "service_revision": self.service_revision,
                                "query": query,
                                "embedding": embedding,
                            },
                        )
                else:
                    embedding = existing
        vector = np.asarray([embedding], dtype=np.float32)
        return l2_normalize(vector)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_vector = self.embed_query(query)
        scores = query_vector @ self.page_matrix.T
        k = min(max(1, top_k), len(self.page_handles))
        top_indices = np.argpartition(-scores[0], kth=k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[0][top_indices])]
        return [
            {
                "page_handle": self.page_handles[int(index)],
                "score": float(scores[0][int(index)]),
            }
            for index in top_indices
        ]
