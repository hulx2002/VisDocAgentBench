#!/usr/bin/env python3
"""Download the public VisDocAgentBench dataset snapshot from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hulx2002/VisDocAgentBench")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.data_root,
    )
    print(f"[done] dataset available at {args.data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
