#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

: "${1:?generated environment file required}"
source "$1"

: "${MODEL_PATH:?}"
: "${SERVED:?}"
: "${TP:?}"
: "${EP:?}"
: "${ISL:?}"
: "${OSL:?}"
: "${RESULT_DIR:?}"
: "${CACHE_ROOT:?}"
: "${PYTHON:?}"

: "${SLURM_JOB_ID:?}"
PORT=${PORT:-$((10000 + SLURM_JOB_ID % 20000))}
MASTER_PORT=${MASTER_PORT:-$((30000 + SLURM_JOB_ID % 20000))}
GPU_UTIL=${GPU_UTIL:-0.9}
RANGE_RATIO=${RANGE_RATIO:-0.8}
MML=${MML_OVERRIDE:-$((ISL + OSL + 64))}
NODES=${NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
LOG="$RESULT_DIR/server-rank${NODE_RANK}.log"
STOP="$RESULT_DIR/.stop-$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR" "$CACHE_ROOT"
export VLLM_CACHE_ROOT="$CACHE_ROOT"

server=(
  "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --trust-remote-code
  --max-model-len "$MML"
  --gpu-memory-utilization "$GPU_UTIL"
  --port "$PORT"
)
[[ -n ${DP_SIZE:-} ]] && server+=(--data-parallel-size "$DP_SIZE")
[[ $EP == 1 ]] && server+=(--enable-expert-parallel)
[[ -n ${MAX_NUM_SEQS:-} ]] && server+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -n ${KV_DTYPE:-} ]] && server+=(--kv-cache-dtype "$KV_DTYPE")
if ((NODES > 1)); then
  : "${SLURM_LAUNCH_NODE_IPADDR:?}"
  server+=(
    --distributed-executor-backend mp
    --nnodes "$NODES"
    --node-rank "$NODE_RANK"
    --master-addr "$SLURM_LAUNCH_NODE_IPADDR"
    --master-port "$MASTER_PORT"
  )
  ((NODE_RANK > 0)) && server+=(--headless)
fi
if [[ -n ${EXTRA_SERVE:-} ]]; then
  read -r -a extra_serve <<<"$EXTRA_SERVE"
  server+=("${extra_serve[@]}")
fi

"${server[@]}" >"$LOG" 2>&1 &
server_pid=$!
cleanup() {
  if ((NODES > 1 && NODE_RANK == 0)); then
    touch "$STOP"
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

if ((NODE_RANK > 0)); then
  while [[ ! -e $STOP ]] && kill -0 "$server_pid" 2>/dev/null; do
    sleep 1
  done
  if [[ -e $STOP ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    exit
  fi
  wait "$server_pid"
  exit
fi

ready=0
for _ in {1..360}; do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
  fi
  sleep 10
done
if [[ $ready != 1 ]]; then
  tail -200 "$LOG"
  exit 1
fi

if [[ -n ${GSM8K_CLIENT:-} ]]; then
  gsm=(
    "$PYTHON" "$GSM8K_CLIENT"
    --url "http://127.0.0.1:$PORT"
    --model "$SERVED"
    --data-dir "$GSM8K_DATA"
    --output-dir "$RESULT_DIR"
    --variant "$GSM8K_VARIANT"
    --model-revision "${MODEL_REVISION:-}"
    --num-questions "${GSM8K_EXAMPLES:-1319}"
    --max-concurrency "${GSM8K_MAX_CONCURRENCY:-64}"
  )
  if [[ -n ${GSM8K_BASELINE_DETAILS:-} ]]; then
    gsm+=(--baseline-details "$GSM8K_BASELINE_DETAILS")
  fi
  "${gsm[@]}"
  exit
fi

: "${IXBENCH:?}"
: "${CONCS:?}"
IFS=, read -r -a concurrencies <<<"$CONCS"
for c in "${concurrencies[@]}"; do
  prompts=$((3 * c))
  ((prompts < 20)) && prompts=20
  ((prompts > 512)) && prompts=512
  client=(
    "$PYTHON" "$IXBENCH"
    --backend vllm
    --base-url "http://localhost:$PORT"
    --model "$SERVED"
    --tokenizer "$MODEL_PATH"
    --trust-remote-code
    --dataset-name random
    --random-input-len "$ISL"
    --random-output-len "$OSL"
    --random-range-ratio "$RANGE_RATIO"
    --max-concurrency "$c"
    --num-prompts "$prompts"
    --request-rate inf
    --ignore-eos
    --num-warmups "$((2 * c))"
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result
    --result-dir "$RESULT_DIR"
    --result-filename "bench_c${c}.json"
  )
  [[ -n ${CHAT_TEMPLATE:-} ]] &&
    client+=(--chat-template "$CHAT_TEMPLATE")
  "${client[@]}" >"$RESULT_DIR/bench_c${c}.log" 2>&1
done
