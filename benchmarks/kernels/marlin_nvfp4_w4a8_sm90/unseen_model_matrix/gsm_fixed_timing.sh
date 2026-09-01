#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

: "${PYTHON:?}"
: "${MODEL_PATH:?}"
: "${SERVED:?}"
: "${BASE_URL:?}"
: "${VARIANT:?implementation label, for example A or B}"
: "${RUN_LABEL:?unique run label, for example a1, b1, or a2}"
: "${RUN_INDEX:?monotonic index used to verify A/B/A bracketing}"
: "${BASELINE_VARIANT:?repeated baseline implementation label}"
: "${RESULT_ROOT:?}"
: "${GSM8K_DETAILS:?}"

[[ $VARIANT =~ ^[A-Za-z0-9_.-]+$ ]]
[[ $RUN_LABEL =~ ^[A-Za-z0-9_.-]+$ ]]
[[ $RUN_INDEX =~ ^[1-9][0-9]*$ ]]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ANALYZER="$SCRIPT_DIR/gsm_fixed_timing.py"
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
CONCS=${CONCS:-1,2,4,8,16,32,64,128,256,512}
[[ $OUTPUT_TOKENS =~ ^[1-9][0-9]*$ ]]
WORKLOAD_DIR="$RESULT_ROOT/workload"
RUN_DIR="$RESULT_ROOT/runs/$RUN_LABEL"
mkdir -p "$WORKLOAD_DIR" "$RUN_DIR"

prepare=(
  "$PYTHON" "$ANALYZER" prepare
  --details "$GSM8K_DETAILS"
  --output-dir "$WORKLOAD_DIR"
  --tokenizer "$MODEL_PATH"
  --output-tokens "$OUTPUT_TOKENS"
  --model-revision "${MODEL_REVISION:-}"
  --source-revision "${SOURCE_REVISION:-}"
  --trust-remote-code
)
[[ -n ${ACCURACY_SUMMARY:-} ]] &&
  prepare+=(--accuracy-summary "$ACCURACY_SUMMARY")
"${prepare[@]}"

PROVENANCE="$WORKLOAD_DIR/provenance.json"
GSM_DATASET="$WORKLOAD_DIR/gsm8k-fixed.jsonl"
RANDOM_DATASET="$WORKLOAD_DIR/random-matched-fixed.jsonl"
PROVENANCE_SHA=$(sha256sum "$PROVENANCE")
PROVENANCE_SHA=${PROVENANCE_SHA%% *}
SOURCE_DETAILS_SHA=$(sha256sum "$GSM8K_DETAILS")
SOURCE_DETAILS_SHA=${SOURCE_DETAILS_SHA%% *}
TOTAL_PROMPTS=$(wc -l <"$GSM_DATASET")

IFS=, read -r -a concurrencies <<<"$CONCS"
for concurrency in "${concurrencies[@]}"; do
  [[ $concurrency =~ ^[1-9][0-9]*$ ]]
  num_prompts=$((3 * concurrency))
  ((num_prompts < 20)) && num_prompts=20
  ((num_prompts > 512)) && num_prompts=512
  ((num_prompts > TOTAL_PROMPTS)) && num_prompts=$TOTAL_PROMPTS
  for workload in gsm8k random; do
    if [[ $workload == gsm8k ]]; then
      dataset=$GSM_DATASET
      filename="gsm8k_c${concurrency}.json"
    else
      dataset=$RANDOM_DATASET
      filename="random_c${concurrency}.json"
    fi
    dataset_sha=$(sha256sum "$dataset")
    dataset_sha=${dataset_sha%% *}
    client=(
      "$PYTHON" -m vllm.entrypoints.cli.main bench serve
      --backend vllm
      --base-url "$BASE_URL"
      --model "$SERVED"
      --tokenizer "$MODEL_PATH"
      --trust-remote-code
      --dataset-name custom
      --dataset-path "$dataset"
      --skip-chat-template
      --disable-shuffle
      --no-oversample
      --custom-output-len "$OUTPUT_TOKENS"
      --num-prompts "$num_prompts"
      --max-concurrency "$concurrency"
      --request-rate inf
      --num-warmups "$((2 * concurrency))"
      --temperature 0
      --ignore-eos
      --percentile-metrics "ttft,tpot,itl,e2el"
      --save-result
      --save-detailed
      --result-dir "$RUN_DIR"
      --result-filename "$filename"
      --metadata
      "run_label=$RUN_LABEL"
      "run_index=$RUN_INDEX"
      "variant=$VARIANT"
      "workload_kind=${workload}_fixed_token_timing"
      "provenance_sha256=$PROVENANCE_SHA"
      "dataset_sha256=$dataset_sha"
      "source_details_sha256=$SOURCE_DETAILS_SHA"
      "ignore_eos=true"
      "fixed_output_tokens=$OUTPUT_TOKENS"
      "temperature=0"
      "num_warmups=$((2 * concurrency))"
      "timing_prompt_count=$num_prompts"
    )
    "${client[@]}" >"$RUN_DIR/${filename%.json}.log" 2>&1
  done
done

"$PYTHON" "$ANALYZER" analyze \
  --result-root "$RESULT_ROOT" \
  --baseline-variant "$BASELINE_VARIANT" \
  --output "$RESULT_ROOT/summary.json"
