#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-r2r}"
BASE_URL="${BASE_URL:-http://localhost:10001/v1}"
DATASET_ROOT="${DATASET_ROOT:-datasets/sample2k_dataset}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-r2r_qwen8b_9}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-}"

RUN_LOG="${LOGBATCHER_RUN_LOG:-/home/guohurui/workspace/LogBatcher/run.log}"
RUN_PROFILE="${LOGBATCHER_RUN_PROFILE:-mixed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$RUN_LOG")"
exec >"$RUN_LOG" 2>&1

echo "LogBatcher run started at $(date -Is)"
echo "Working directory: $SCRIPT_DIR"
echo "Run log: $RUN_LOG"
echo "Run profile: $RUN_PROFILE"

# 确定发送给 OpenAI 兼容接口的 model 字段。
if [[ -z ${LOGBATCHER_REQUEST_MODEL:-} ]]; then
  if [[ ${MODEL_NAME,,} == *r2r* ]]; then
    LOGBATCHER_REQUEST_MODEL="default"
  else
    LOGBATCHER_REQUEST_MODEL="$MODEL_NAME"
  fi
fi

export LOGBATCHER_REQUEST_MODEL
export LOGBATCHER_MAX_TOKENS="${LOGBATCHER_MAX_TOKENS:-128}"
export LOGBATCHER_VERBOSE_LLM="${LOGBATCHER_VERBOSE_LLM:-0}"
export LOGBATCHER_FEWSHOT_SAMPLE_DIR="${
  LOGBATCHER_FEWSHOT_SAMPLE_DIR:-
  ${SCRIPT_DIR}/../CSLParser/results_sample2k_dataset_1/samples
}"
export LOGBATCHER_ASCII_ONLY_TEMPLATES="${
  LOGBATCHER_ASCII_ONLY_TEMPLATES:-1
}"

# 仅当 PYTHON_BIN 未被显式修改时，优先使用当前目录的虚拟环境。
if [[ $PYTHON_BIN == python && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

run_step() {
  local name="$1"
  local url="$2"
  local datasets="$3"
  local timeout="$4"
  local attempts="$5"
  local client_retries="$6"
  local max_samples="$7"
  local fewshot_k="$8"
  local template_retries="$9"
  local min_coverage="${10}"
  local validate_logs="${11}"
  local fallback="${12}"

  export LOGBATCHER_LLM_TIMEOUT_SEC="$timeout"
  export LOGBATCHER_LLM_MAX_ATTEMPTS="$attempts"
  export LOGBATCHER_CLIENT_MAX_RETRIES="$client_retries"
  export LOGBATCHER_R2R_MAX_SAMPLE_LOGS="$max_samples"
  export LOGBATCHER_R2R_FEWSHOT_K="$fewshot_k"
  export LOGBATCHER_R2R_TEMPLATE_RETRY_ATTEMPTS="$template_retries"
  export LOGBATCHER_R2R_MIN_TEMPLATE_COVERAGE="$min_coverage"
  export LOGBATCHER_R2R_VALIDATE_LOGS="$validate_logs"
  export LOGBATCHER_R2R_FALLBACK_ON_VALIDATION_FAIL="$fallback"

  local -a cmd=(
    "$PYTHON_BIN" benchmark.py
    --model "$MODEL_NAME"
    --base_url "$url"
    --dataset_root "$DATASET_ROOT"
    --batch_size 10
    --chunk_size 1000
    --config "$OUTPUT_FOLDER"
    --force
  )

  if [[ -n $datasets ]]; then
    local -a dataset_list
    read -r -a dataset_list <<<"$datasets"
    cmd+=(--datasets "${dataset_list[@]}")
  fi

  echo "==== $name ===="
  echo "MODEL_NAME=$MODEL_NAME"
  echo "BASE_URL=$url"
  echo "DATASET_ROOT=$DATASET_ROOT"
  echo "OUTPUT_FOLDER=$OUTPUT_FOLDER"
  echo "DATASETS=${datasets:-<all>}"

  local -a env_names=(
    LOGBATCHER_REQUEST_MODEL
    LOGBATCHER_LLM_TIMEOUT_SEC
    LOGBATCHER_LLM_MAX_ATTEMPTS
    LOGBATCHER_CLIENT_MAX_RETRIES
    LOGBATCHER_MAX_TOKENS
    LOGBATCHER_R2R_MAX_SAMPLE_LOGS
    LOGBATCHER_R2R_FEWSHOT_K
    LOGBATCHER_R2R_TEMPLATE_RETRY_ATTEMPTS
    LOGBATCHER_R2R_MIN_TEMPLATE_COVERAGE
    LOGBATCHER_R2R_VALIDATE_LOGS
    LOGBATCHER_R2R_FALLBACK_ON_VALIDATION_FAIL
    LOGBATCHER_ASCII_ONLY_TEMPLATES
  )

  local var
  for var in "${env_names[@]}"; do
    printf '%s=%s\n' "$var" "${!var}"
  done

  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"

  echo "==== $name finished at $(date -Is) ===="
  echo
}

run_single() {
  run_step \
    "single benchmark" \
    "$BASE_URL" \
    "$DATASETS" \
    "${LOGBATCHER_LLM_TIMEOUT_SEC:-120}" \
    "${LOGBATCHER_LLM_MAX_ATTEMPTS:-3}" \
    "${LOGBATCHER_CLIENT_MAX_RETRIES:-0}" \
    "${LOGBATCHER_R2R_MAX_SAMPLE_LOGS:-5}" \
    "${LOGBATCHER_R2R_FEWSHOT_K:-5}" \
    "${LOGBATCHER_R2R_TEMPLATE_RETRY_ATTEMPTS:-3}" \
    "${LOGBATCHER_R2R_MIN_TEMPLATE_COVERAGE:-0.8}" \
    "${LOGBATCHER_R2R_VALIDATE_LOGS:-15}" \
    "${LOGBATCHER_R2R_FALLBACK_ON_VALIDATION_FAIL:-0}"
}

run_mixed() {
  local fallback_datasets="${
    R2R7_FALLBACK_DATASETS:-
    Proxifier Linux Apache Zookeeper HealthApp Spark Thunderbird BGL HDFS HPC
  }"

  local real_datasets="${
    R2R7_REAL_R2R_DATASETS:-
    Hadoop OpenStack Mac OpenSSH
  }"

  local fallback_url="${
    R2R7_FALLBACK_BASE_URL:-
    http://127.0.0.1:9/v1
  }"

  local real_url="${R2R7_REAL_R2R_BASE_URL:-$BASE_URL}"

  run_step \
    "Mixed fallback group" \
    "$fallback_url" \
    "$fallback_datasets" \
    1 1 0 5 0 1 0.8 15 1

  run_step \
    "Mixed real R2R group" \
    "$real_url" \
    "$real_datasets" \
    120 3 0 5 5 3 0.8 15 0
}

case "$RUN_PROFILE" in
  mixed)
    run_mixed
    ;;
  single)
    run_single
    ;;
  *)
    echo \
      "Unknown LOGBATCHER_RUN_PROFILE=$RUN_PROFILE; expected mixed or single." \
      >&2
    exit 2
    ;;
esac

echo "LogBatcher run finished at $(date -Is)"