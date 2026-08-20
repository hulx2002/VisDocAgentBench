#!/usr/bin/env python3
"""Run Nemotron ColEmbed VL 8B V2 with streaming page-side MaxSim scoring."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    directory_identity,
    file_collection_identity,
    file_identity,
    gold_page_id,
    query_text,
    read_jsonl,
    resolve_page_image,
    sha256_json,
    summarize_predictions,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "VisDocAgentBenchNemotronColEmbedV2BaselineV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--page-corpus", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--page-chunk-size", type=int, default=32)
    parser.add_argument("--image-forward-batch-size", type=int, default=8)
    parser.add_argument("--score-query-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.query_batch_size <= 0
        or args.page_chunk_size <= 0
        or args.image_forward_batch_size <= 0
        or args.score_query_batch_size <= 0
    ):
        raise ValueError("all batch sizes must be positive")
    if args.top_k < 10:
        raise ValueError("top-k must be at least 10")

    queries = read_jsonl(args.queries)
    pages = read_jsonl(args.page_corpus)
    if args.limit_pages > 0:
        pages = pages[: args.limit_pages]
    if len(queries) != 120:
        raise ValueError(f"expected 120 queries, found {len(queries)}")
    if args.limit_pages <= 0 and len(pages) != 2375:
        raise ValueError(f"expected 2375 pages, found {len(pages)}")
    page_ids = [str(row.get("page_id") or "") for row in pages]
    if not all(page_ids) or len(page_ids) != len(set(page_ids)):
        raise ValueError("page corpus has missing or duplicate page ids")
    texts = [query_text(row) for row in queries]
    page_image_paths = [
        resolve_page_image(row, args.page_manifest) for row in pages
    ]
    page_image_identity = file_collection_identity(
        zip(page_ids, page_image_paths, strict=True)
    )

    expected_shards = [args.model / f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)]
    missing = [str(path) for path in expected_shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Nemotron model is incomplete; missing: {missing}")
    weight_index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    from safetensors import safe_open

    indexed_keys = set(weight_index["weight_map"])
    shard_keys: set[str] = set()
    for shard in expected_shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            shard_keys.update(tensors.keys())
    if shard_keys != indexed_keys:
        missing_keys = sorted(indexed_keys - shard_keys)[:10]
        unexpected_keys = sorted(shard_keys - indexed_keys)[:10]
        raise ValueError(
            "Nemotron shard key map does not match the weight index: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    tensor_weight_bytes = int(weight_index["metadata"]["total_size"])
    weight_file_bytes = sum(path.stat().st_size for path in expected_shards)

    processor_config_path = args.model / "processor_config.json"
    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    image_size = processor_config.get("image_processor", {}).get("size", {})
    native_processor_settings = {
        "q_max_length": int(processor_config["q_max_length"]),
        "p_max_length": int(processor_config["p_max_length"]),
        "query_prefix": str(processor_config.get("query_prefix") or ""),
        "passage_prefix": str(processor_config.get("passage_prefix") or ""),
        "min_pixels": int(image_size["shortest_edge"]),
        "max_pixels": int(image_size["longest_edge"]),
    }
    settings = {
        "model": str(args.model.resolve()),
        "model_identity": directory_identity(args.model),
        "dtype": "bfloat16",
        "attn_implementation": args.attn_implementation,
        "query_batch_size": args.query_batch_size,
        "page_chunk_size": args.page_chunk_size,
        "image_forward_batch_size": args.image_forward_batch_size,
        "score_query_batch_size": args.score_query_batch_size,
        "top_k": args.top_k,
        "query_instruction": None,
        "document_instruction": None,
        "scoring": "ColBERT MaxSim via model.get_scores",
        "preprocessing": "model-native processor defaults",
        "processor": native_processor_settings,
        "tensor_weight_bytes": tensor_weight_bytes,
        "weight_file_bytes": weight_file_bytes,
    }
    fingerprint = sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "queries": [
                {"id": row.get("query_id"), "text": text, "gold": gold_page_id(row)}
                for row, text in zip(queries, texts)
            ],
            "page_ids": page_ids,
            "inputs": {
                "queries": file_identity(args.queries),
                "page_corpus": file_identity(args.page_corpus),
                "page_manifest": file_identity(args.page_manifest),
                "page_images": page_image_identity,
            },
            "settings": settings,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    score_path = args.out_dir / "score_matrix.npy"
    next_page_index = 0
    if not args.no_resume and progress_path.is_file() and score_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") == fingerprint:
            next_page_index = int(progress.get("next_page_index", 0))
            score_matrix = np.lib.format.open_memmap(score_path, mode="r+")
            if score_matrix.shape != (len(queries), len(pages)):
                raise ValueError(f"resume score matrix has unexpected shape {score_matrix.shape}")
            print(f"[resume] continuing at page {next_page_index}/{len(pages)}", flush=True)
        else:
            raise ValueError("existing Nemotron progress belongs to different inputs or settings")
    else:
        score_matrix = np.lib.format.open_memmap(
            score_path, mode="w+", dtype=np.float32, shape=(len(queries), len(pages))
        )
        score_matrix[:] = np.nan
        score_matrix.flush()
        write_json(
            progress_path,
            {"fingerprint": fingerprint, "next_page_index": 0, "num_pages": len(pages)},
        )

    import torch
    from PIL import Image
    from transformers import AutoModel

    print(f"[load] {args.model}", flush=True)
    model = AutoModel.from_pretrained(
        str(args.model),
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).eval()
    processor = model._get_processor()
    actual_processor_settings = {
        "q_max_length": int(processor.q_max_length),
        "p_max_length": int(processor.p_max_length),
        "query_prefix": str(processor.query_prefix),
        "passage_prefix": str(processor.passage_prefix),
        "min_pixels": int(processor.min_pixels),
        "max_pixels": int(processor.max_pixels),
    }
    if actual_processor_settings != native_processor_settings:
        raise ValueError(
            "loaded Nemotron processor does not match processor_config.json: "
            f"expected {native_processor_settings}, got {actual_processor_settings}"
        )

    print(f"[encode] {len(queries)} raw queries", flush=True)
    query_embeddings = model.forward_queries(texts, batch_size=args.query_batch_size)
    if query_embeddings.shape[0] != len(queries):
        raise ValueError(f"expected {len(queries)} query embeddings, got {query_embeddings.shape}")

    for start in range(next_page_index, len(pages), args.page_chunk_size):
        batch_rows = pages[start : start + args.page_chunk_size]
        images = []
        try:
            for image_path in page_image_paths[start : start + len(batch_rows)]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            image_embeddings = model.forward_images(
                images,
                batch_size=min(args.image_forward_batch_size, len(images)),
            )
            with torch.inference_mode():
                scores = model.get_scores(
                    query_embeddings,
                    image_embeddings,
                    batch_size=args.score_query_batch_size,
                )
            score_array = scores.detach().float().cpu().numpy()
            if score_array.shape != (len(queries), len(batch_rows)):
                raise ValueError(
                    f"unexpected score shape at pages {start}:{start + len(batch_rows)}: {score_array.shape}"
                )
            if not np.isfinite(score_array).all():
                raise ValueError(f"non-finite scores at page batch starting {start}")
            score_matrix[:, start : start + len(batch_rows)] = score_array
            score_matrix.flush()
            write_json(
                progress_path,
                {
                    "fingerprint": fingerprint,
                    "next_page_index": start + len(batch_rows),
                    "num_pages": len(pages),
                    "last_page_id": page_ids[start + len(batch_rows) - 1],
                },
            )
            print(
                f"[pages] {start + len(batch_rows)}/{len(pages)} "
                f"{page_ids[start + len(batch_rows) - 1]}",
                flush=True,
            )
        finally:
            for image in images:
                image.close()
        del image_embeddings, scores, score_array
        gc.collect()
        torch.cuda.empty_cache()

    if np.isnan(np.asarray(score_matrix)).any():
        raise ValueError("score matrix remains incomplete after page encoding")
    predictions: list[dict[str, Any]] = []
    output_depth = min(args.top_k, len(page_ids))
    for query_index, query in enumerate(queries):
        scores = np.asarray(score_matrix[query_index])
        order = sorted(range(len(page_ids)), key=lambda index: (-float(scores[index]), page_ids[index]))
        selected = order[:output_depth]
        predictions.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": str(query["query_id"]),
                "gold_page_id": gold_page_id(query),
                "level": query.get("level"),
                "topic_id": query.get("topic_id"),
                "status": "submitted",
                "ranked_page_ids": [page_ids[index] for index in selected],
                "ranked_scores": [float(scores[index]) for index in selected],
            }
        )

    write_jsonl(args.out_dir / "predictions.jsonl", predictions)
    metrics = summarize_predictions(
        queries, predictions, valid_page_ids=set(page_ids)
    )
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "settings": settings,
            "num_queries": len(queries),
            "num_pages": len(pages),
            "inputs": {
                "queries": str(args.queries.resolve()),
                "page_corpus": str(args.page_corpus.resolve()),
                "page_manifest": str(args.page_manifest.resolve()),
            },
        },
    )
    print(json.dumps(metrics["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
