#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: $0 smoke|full}"
case "$mode" in
  smoke)
    variants=(adaptive)
    stop_index=1
    profile=1
    time_limit=00:45:00
    prefix=smoke
    ;;
  full)
    if [[ "${S39_FULL_CAMPAIGN_GO:-0}" != 1 ]]; then
      echo "full launch is gated; set S39_FULL_CAMPAIGN_GO=1 after final H100 gates pass" >&2
      exit 2
    fi
    variants=(marlin adaptive r1 sqrt6 r6 adaptive_prod)
    stop_index=1319
    profile=0
    time_limit=04:00:00
    prefix=full
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

host_campaign=/home/mgschwind/lustre/s39-quality-campaign
campaign=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/mgschwind/s39-quality-campaign
composite=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/mgschwind/composite-fix-eeaf7a59
bench=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/mgschwind/s39-block-bench
data=$composite/gsm8k-data
python=$bench/.venv-vllm024/bin/python
compat_root=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/mgschwind/cudagym/s39-marlin-nvfp4-to-fp8/block-scale-f9197bcc7074/prod-validation-current-1d217203
so=$composite/final-afe5b8aa-9491805d/artifacts/build-converter/s39_final_afe5b8aa_vllm024.so
so_sha=2609b524121436d9a44ea8d78faa7329172eb114a240df44662421671e236ffc
stable_so=$composite/final-afe5b8aa-9491805d/artifacts/build-stable-only-r3/_C_stable_libtorch.abi3.so
stable_target=/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so
image=/home/mgschwind/lustre/mmpareto/containers/vllm_v0.24.0.sqsh
source_tag=afe5b8aa-9491805d-3376f631-c6bf359b-bb42106a

mkdir -p "$host_campaign/logs" "$host_campaign/results"
run_id="${S39_CAMPAIGN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
submission="$host_campaign/submitted_${prefix}_${run_id}.tsv"
printf "job_id\tmodel\tvariant\tcache_root\tlog\titems\tsummary\n" > "$submission"

models=(llama q36d q36m q3m)
for model in "${models[@]}"; do
  case "$model" in
    llama)
      model_path=$composite/models/llama
      model_id=parasail-ai/Llama-3.1-8B-Instruct-NVFP4A16
      revision=166764ea0872b41d3253efae3639367edf055906
      gen_prefix=none
      ;;
    q36d)
      model_path=$composite/models/q36d
      model_id=nvidia/Qwen3.6-27B-NVFP4
      revision=0893e1606ff3d5f97a441f405d5fc541a6bdf404
      gen_prefix=qwen36
      ;;
    q36m)
      model_path=$composite/models/q36m
      model_id=nvidia/Qwen3.6-35B-A3B-NVFP4
      revision=491c2f1ea524c639598bf8fa787a93fed5a6fbce
      gen_prefix=qwen36
      ;;
    q3m)
      model_path=$composite/models/q3m
      model_id=Benasd/Qwen3-30B-A3B-Instruct-2507-NVFP4A16
      revision=d3273f65a140017763d977132f187cbb75984587
      gen_prefix=none
      ;;
  esac
  for variant in "${variants[@]}"; do
    if [[ $mode == full ]]; then
      stem=${model}__${variant}
    else
      stem=${prefix}__${model}__${variant}
    fi
    log=$host_campaign/logs/${stem}.log
    items=$campaign/results/${stem}.jsonl
    summary=$campaign/results/${stem}.summary.json
    cache_root=$campaign/cache/${source_tag}/${run_id}/${model}/${variant}
    host_cache_root=$host_campaign/cache/${source_tag}/${run_id}/${model}/${variant}
    if [[ -e $host_cache_root ]]; then
      echo "cache root already exists: $host_cache_root" >&2
      exit 2
    fi
    job_id=$(sbatch --parsable \
      --partition=interactive \
      --account=sw_aidot \
      --qos=normal \
      --constraint=H100 \
      --gres=gpu:1 \
      --cpus-per-task=16 \
      --time="$time_limit" \
      --job-name="s39-${prefix}-${model}-${variant}" \
      --output="$log" \
      --error="$log" \
      --open-mode=truncate \
      --wrap="srun --container-image=$image --container-mounts=/lustre:/lustre,$stable_so:$stable_target --no-container-entrypoint /usr/bin/env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$cache_root S39_RUN_ID=$run_id S39_SOURCE_TAG=$source_tag S39_ROOT=$compat_root S39_RUNNER=$campaign/s39_campaign_eval.py S39_SO=$so S39_SO_SHA256=$so_sha S39_VARIANT=$variant S39_MODEL=$model_path S39_MODEL_ID=$model_id S39_MODEL_REVISION=$revision S39_GEN_PREFIX=$gen_prefix S39_DATA=$data S39_START_INDEX=0 S39_STOP_INDEX=$stop_index S39_PROFILE=$profile S39_MAX_NUM_SEQS=64 S39_MAX_TOKENS=256 S39_MAX_MODEL_LEN=2048 S39_OUTPUT=$items S39_SUMMARY=$summary S39_MANIFEST=$campaign/manifest.json $python $campaign/s39_compat_bootstrap.py")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$job_id" "$model" "$variant" "$cache_root" "$log" "$items" "$summary" \
      >> "$submission"
    echo "$job_id $model $variant"
  done
done
echo "submission_manifest=$submission"
