#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

QUERY_FILE=data/derived/retrieval_inputs/queries.visdocagentbench.jsonl
OUT_ROOT="${OUT_ROOT:-outputs/nemotron_colembed}"
python baselines/non_agentic/run_nemotron_colembed.py \
  --queries "${QUERY_FILE}" \
  --page-corpus data/derived/retrieval_inputs/page_image_corpus.jsonl \
  --page-manifest data/corpus/pages.jsonl \
  --model models/nemotron-colembed-vl-8b-v2 \
  --out-dir "${OUT_ROOT}" --top-k 10
