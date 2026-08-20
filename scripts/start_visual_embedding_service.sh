#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the Qwen3-VL OpenAI-compatible vLLM embedding service.
# The client sends Qwen3-VL-Embedding's special /v1/embeddings messages payload.

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH##*/}}"
CONDA_PY="${CONDA_PY:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

exec "${CONDA_PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --runner pooling \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  ${VLLM_EXTRA_ARGS:-}
