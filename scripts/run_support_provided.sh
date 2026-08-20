#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROUTE="${ROUTE:-visual}"
CONFIG="${CONFIG:-configs/api.json}"
PLANNER_MODEL="${PLANNER_MODEL:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_ARGS=()
if [[ -n "${PLANNER_MODEL}" ]]; then MODEL_ARGS+=(--planner-model "${PLANNER_MODEL}"); fi

case "${ROUTE}" in
  visual)
    RUNNER=baselines/support_provided/run_visual.py
    OUT_DIR="${OUT_DIR:-outputs/support_provided_visual}"
    ;;
  ocr_text)
    RUNNER=baselines/support_provided/run_ocr_text.py
    OUT_DIR="${OUT_DIR:-outputs/support_provided_ocr_text}"
    ;;
  *) echo "ROUTE must be visual or ocr_text" >&2; exit 2 ;;
esac

python "${RUNNER}" \
  --config "${CONFIG}" \
  "${MODEL_ARGS[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --out-dir "${OUT_DIR}" \
  --max-steps 12 \
  --max-tokens "${MAX_TOKENS:-3200}" \
  --resume-partial-episodes \
  "$@"
