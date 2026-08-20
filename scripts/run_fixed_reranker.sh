#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROUTE="${ROUTE:-visual}"
CONFIG="${CONFIG:-configs/api.json}"
MODEL="${MODEL:-gpt-5.6-sol}"
QUERY_FILE=data/derived/retrieval_inputs/queries.visdocagentbench.jsonl
case "${ROUTE}" in
  visual) FIRST_STAGE=outputs/embedding_visual/predictions.jsonl ;;
  ocr_text) FIRST_STAGE=outputs/embedding_ocr_text/predictions.jsonl ;;
  *) echo "ROUTE must be visual or ocr_text" >&2; exit 2 ;;
esac

python baselines/non_agentic/run_fixed_reranker.py \
  --queries "${QUERY_FILE}" \
  --first-stage-predictions "${FIRST_STAGE}" \
  --page-manifest data/corpus/pages.jsonl \
  --ocr-cache data/derived/paddleocr_vl_1_6_page_ocr.jsonl \
  --out-dir "${OUT_DIR:-outputs/fixed_reranker_${ROUTE}}" \
  --config "${CONFIG}" --model "${MODEL}" --route "${ROUTE}" \
  --workers "${NUM_WORKERS:-4}" --candidate-count 30 --batch-size 3 \
  --max-tokens 3200
