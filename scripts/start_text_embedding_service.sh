#!/usr/bin/env bash
set -euo pipefail

# Wrapper for an OpenAI-compatible vLLM text embedding service.
# Keep this process running while cache_text_embeddings.py is running.

MODEL_PATH="${MODEL_PATH:-models/Qwen3-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH##*/}}"
CONDA_PY="${CONDA_PY:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

exec "${CONDA_PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  ${VLLM_EXTRA_ARGS:-}
