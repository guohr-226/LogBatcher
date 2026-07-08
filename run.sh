#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-r2r}"
BASE_URL="${BASE_URL:-http://localhost:10001/v1}"
DATASET_ROOT="${DATASET_ROOT:-datasets/sample2k_dataset}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-r2r_qwen8b}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ "${PYTHON_BIN}" = "python" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

"${PYTHON_BIN}" benchmark.py \
  --model "${MODEL_NAME}" \
  --base_url "${BASE_URL}" \
  --dataset_root "${DATASET_ROOT}" \
  --batch_size 10 \
  --chunk_size 1000 \
  --config "${OUTPUT_FOLDER}" \
  --force \
  > run.log 2>&1
