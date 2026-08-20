#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python baselines/non_agentic/prepare_inputs.py \
  --out-dir data/derived/retrieval_inputs \
  "$@"
