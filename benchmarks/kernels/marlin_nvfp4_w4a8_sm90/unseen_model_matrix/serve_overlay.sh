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
: "${PYTHON:?}"
: "${CUDA_HOME:?}"

[[ -f $CUDA_HOME/include/cuda/std/cstdint ]]
[[ -f $CUDA_HOME/include/cuda_bf16.h ]]
[[ -x $CUDA_HOME/bin/cuobjdump ]]

: "${SLURM_JOB_ID:?}"
MASTER_PORT=${MASTER_PORT:-$((30000 + SLURM_JOB_ID % 20000))}
GPU_UTIL=${GPU_UTIL:-0.9}
RANGE_RATIO=${RANGE_RATIO:-0.8}
MML=${MML_OVERRIDE:-$((ISL + OSL + 64))}
NODES=${NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
LOG="$RESULT_DIR/server-rank${NODE_RANK}.log"
STOP="$RESULT_DIR/.stop-$SLURM_JOB_ID"
SCRATCH=${SLURM_TMPDIR:-/tmp}
[[ -d $SCRATCH && -w $SCRATCH ]] || SCRATCH=/tmp
RUNTIME_CACHE=$(mktemp -d \
  "$SCRATCH/w4a8-${SLURM_JOB_ID}-${NODE_RANK}.XXXXXX")
mkdir -p "$RESULT_DIR"
export TMPDIR="$RUNTIME_CACHE/tmp"
export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg-cache"
export XDG_CONFIG_HOME="$RUNTIME_CACHE/xdg-config"
export HF_HOME="$RUNTIME_CACHE/huggingface"
export TRITON_CACHE_DIR="$RUNTIME_CACHE/triton"
export CUDA_CACHE_PATH="$RUNTIME_CACHE/cuda"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_CACHE/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$RUNTIME_CACHE/flashinfer"
export VLLM_CACHE_ROOT="$RUNTIME_CACHE/vllm"
export VLLM_CONFIG_ROOT="$RUNTIME_CACHE/vllm-config"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUNTIME_CACHE/flashinfer-autotune"
export DG_JIT_CACHE_DIR="$RUNTIME_CACHE/deep_gemm"
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
export VLLM_NO_USAGE_STATS=1
mkdir -p \
  "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" \
  "$VLLM_CONFIG_ROOT" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" \
  "$DG_JIT_CACHE_DIR"
if [[ -z ${PORT:-} ]]; then
  PORT=$("$PYTHON" -c \
    'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
fi
client_source=${GSM8K_CLIENT:-${GSM_FIXED_TIMING_CLIENT:-${IXBENCH:-}}}
: "${client_source:?benchmark client required}"
if ((NODE_RANK == 0)); then
  {
    printf 'schema_version 1\nmodel %s\nmodel_revision %s\ndisabled_kernels %s\n' \
      "$SERVED" "${MODEL_REVISION:-}" "${VLLM_DISABLED_KERNELS:-}"
    sha256sum "$1" "$SCRIPTS/$RUN_SCRIPT" "$client_source"
  } >"$RESULT_DIR/runtime.provenance"
fi

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

printf 'UNSEEN_MODEL_RUNTIME disabled_kernels=%s\n' \
  "${VLLM_DISABLED_KERNELS:-}" >"$LOG"
"${server[@]}" >>"$LOG" 2>&1 &
server_pid=$!
cleanup() {
  if ((NODES > 1 && NODE_RANK == 0)); then
    touch "$STOP"
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  rm -rf -- "$RUNTIME_CACHE"
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
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
  fi
  if curl --connect-timeout 2 --max-time 2 -fsS \
    "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
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

if [[ -n ${GSM_FIXED_TIMING_CLIENT:-} ]]; then
  export BASE_URL="http://127.0.0.1:$PORT"
  "$GSM_FIXED_TIMING_CLIENT"
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
    --save-detailed
    --save-result
    --result-dir "$RESULT_DIR"
    --result-filename "bench_c${c}.json"
  )
  [[ -n ${CHAT_TEMPLATE:-} ]] &&
    client+=(--chat-template "$CHAT_TEMPLATE")
  "${client[@]}" >"$RESULT_DIR/bench_c${c}.log" 2>&1
  cp "$RESULT_DIR/runtime.provenance" "$RESULT_DIR/runtime_c${c}.provenance"
  cp "$LOG" "$RESULT_DIR/server_c${c}.log"
done
