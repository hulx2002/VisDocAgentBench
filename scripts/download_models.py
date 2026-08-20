#!/usr/bin/env python3
"""Download the pinned open models used by the released baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "qwen-visual-embedding": (
        "Qwen/Qwen3-VL-Embedding-8B",
        "2c4565515e0f265c6511776e7193b22c0968ddc7",
        "Qwen3-VL-Embedding-8B",
    ),
    "qwen-text-embedding": (
        "Qwen/Qwen3-Embedding-8B",
        "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "Qwen3-Embedding-8B",
    ),
    "paddleocr": (
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "66317acc4c9fc17bd154591ce650735cd2855f3e",
        "PaddleOCR-VL-1.6",
    ),
    "paddle-layout": (
        "PaddlePaddle/PP-DocLayoutV3",
        "eca1b6d6bfcdf285322ba585366278a72bc8b102",
        "PP-DocLayoutV3",
    ),
    "nemotron": (
        "nvidia/nemotron-colembed-vl-8b-v2",
        "34b640612f311ed05a6c7c62c6564847ed555f5f",
        "nemotron-colembed-vl-8b-v2",
    ),
    "qwen35-planner": (
        "Qwen/Qwen3.5-397B-A17B",
        "243a5beb0f8ecc6b171907088d6ff96f812158d7",
        "Qwen3.5-397B-A17B",
    ),
}

DEFAULT_MODELS = tuple(name for name in MODELS if name != "qwen35-planner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(DEFAULT_MODELS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        repo_id, revision, directory = MODELS[name]
        destination = args.model_root / directory
        print(f"[download] {name}: {repo_id}@{revision}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
        )
        print(f"[done] {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
