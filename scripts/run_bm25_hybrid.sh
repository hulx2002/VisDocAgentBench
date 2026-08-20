#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

QUERY_FILE=data/derived/retrieval_inputs/queries.visdocagentbench.jsonl
OUT_ROOT="${OUT_ROOT:-outputs/bm25_hybrid}"
python baselines/non_agentic/run_bm25_dense_rrf.py \
  --queries "${QUERY_FILE}" \
  --ocr-cache data/derived/paddleocr_vl_1_6_page_ocr.jsonl \
  --dense-predictions outputs/embedding_ocr_text/predictions.jsonl \
  --out-dir "${OUT_ROOT}" \
  --k1 1.2 --b 0.75 --rrf-k 60 --fusion-depth 1000 --output-depth 10
