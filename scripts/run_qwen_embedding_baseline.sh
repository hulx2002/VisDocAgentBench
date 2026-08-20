#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROUTE="${ROUTE:-visual}"
INPUT_ROOT="data/derived/retrieval_inputs"
QUERY_FILE="${INPUT_ROOT}/queries.visdocagentbench.jsonl"
OUT_ROOT="${OUT_ROOT:-outputs/embedding_${ROUTE}}"
mkdir -p "${OUT_ROOT}" data/derived

if [[ ! -f "${QUERY_FILE}" ]]; then
  bash scripts/prepare_baseline_inputs.sh
fi

case "${ROUTE}" in
  visual)
    python baselines/non_agentic/cache_visual_embeddings.py \
      --items "${QUERY_FILE}" \
      --out "${OUT_ROOT}/query_embeddings.jsonl" \
      --report "${OUT_ROOT}/query_embeddings.report.json" \
      --id-field query_id --text-field text_embedding_input \
      --model Qwen3-VL-Embedding-8B --base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8002/v1}" \
      --workers "${EMBEDDING_WORKERS:-8}"
    python baselines/non_agentic/cache_visual_embeddings.py \
      --items "${INPUT_ROOT}/page_image_corpus.jsonl" \
      --out "data/derived/qwen3_vl_8b_page_embeddings.jsonl" \
      --report "${OUT_ROOT}/page_embeddings.report.json" \
      --id-field page_id --image-field image_path --project-root data \
      --model Qwen3-VL-Embedding-8B --base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8002/v1}" \
      --workers "${EMBEDDING_WORKERS:-8}"
    python baselines/non_agentic/rank_pages.py \
      --queries "${QUERY_FILE}" \
      --query-embeddings "${OUT_ROOT}/query_embeddings.jsonl" \
      --page-embeddings data/derived/qwen3_vl_8b_page_embeddings.jsonl \
      --out "${OUT_ROOT}/predictions.jsonl" \
      --report "${OUT_ROOT}/run_manifest.json" --top-k 30
    ;;
  ocr_text)
    python baselines/non_agentic/cache_text_embeddings.py \
      --items "${QUERY_FILE}" \
      --out "${OUT_ROOT}/query_embeddings.jsonl" \
      --report "${OUT_ROOT}/query_embeddings.report.json" \
      --id-field query_id --text-field text_embedding_input \
      --model Qwen3-Embedding-8B --base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
    python baselines/non_agentic/cache_text_embeddings.py \
      --items "${INPUT_ROOT}/page_text_corpus.jsonl" \
      --out data/derived/qwen3_8b_ocr_page_embeddings.jsonl \
      --report "${OUT_ROOT}/page_embeddings.report.json" \
      --id-field page_id --text-field embedding_text \
      --model Qwen3-Embedding-8B --base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
    python baselines/non_agentic/rank_pages.py \
      --queries "${QUERY_FILE}" \
      --query-embeddings "${OUT_ROOT}/query_embeddings.jsonl" \
      --page-embeddings data/derived/qwen3_8b_ocr_page_embeddings.jsonl \
      --out "${OUT_ROOT}/predictions.jsonl" \
      --report "${OUT_ROOT}/run_manifest.json" --top-k 1000
    ;;
  *) echo "ROUTE must be visual or ocr_text" >&2; exit 2 ;;
esac

python evaluation/evaluate.py \
  --predictions "${OUT_ROOT}/predictions.jsonl" \
  --output "${OUT_ROOT}/metrics.json"
