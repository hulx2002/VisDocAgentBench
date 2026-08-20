#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/api.json}"
PLANNER_MODEL="${PLANNER_MODEL:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_DIR="${OUT_DIR:-outputs/agent_visual}"
ABLATION="${ABLATION:-full}"
EXTRA_ARGS=()
case "${ABLATION}" in
  full) ;;
  without_ocr) EXTRA_ARGS+=(--ablation-without-ocr) ;;
  without_crop) EXTRA_ARGS+=(--ablation-without-crop) ;;
  without_inspect) EXTRA_ARGS+=(--ablation-without-inspect) ;;
  *) echo "Unknown ABLATION=${ABLATION}" >&2; exit 2 ;;
esac

MODEL_ARGS=()
if [[ -n "${PLANNER_MODEL}" ]]; then MODEL_ARGS+=(--planner-model "${PLANNER_MODEL}"); fi

python baselines/agent_visual/run.py \
  --config "${CONFIG}" \
  "${MODEL_ARGS[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --out-dir "${OUT_DIR}" \
  --max-steps 12 \
  --max-tokens "${MAX_TOKENS:-3200}" \
  --resume-partial-episodes \
  "${EXTRA_ARGS[@]}" \
  "$@"
