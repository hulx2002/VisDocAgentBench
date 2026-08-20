#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG=configs/qwen35.thinking.json \
OUT_DIR=outputs/qwen35_visual_thinking \
MAX_TOKENS=32768 \
NUM_WORKERS="${NUM_WORKERS:-1}" bash scripts/run_visual_agent.sh

CONFIG=configs/qwen35.no_thinking.json \
OUT_DIR=outputs/qwen35_visual_no_thinking \
MAX_TOKENS=3200 \
NUM_WORKERS="${NUM_WORKERS:-1}" bash scripts/run_visual_agent.sh

CONFIG=configs/qwen35.thinking.json \
OUT_DIR=outputs/qwen35_ocr_text_thinking \
MAX_TOKENS=32768 \
NUM_WORKERS="${NUM_WORKERS:-1}" bash scripts/run_ocr_text_agent.sh

CONFIG=configs/qwen35.no_thinking.json \
OUT_DIR=outputs/qwen35_ocr_text_no_thinking \
MAX_TOKENS=3200 \
NUM_WORKERS="${NUM_WORKERS:-1}" bash scripts/run_ocr_text_agent.sh
