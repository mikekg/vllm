#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#SBATCH --partition=backfill
#SBATCH --account=sw_aidot
#SBATCH --qos=normal
#SBATCH --constraint=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=04:00:00

set -euo pipefail

: "${1:?generated environment file required}"
config=$1
source "$config"

: "${IMAGE:?}"
: "${MOUNTS:?}"
: "${SCRIPTS:?}"
: "${RUN_SCRIPT:?}"

LD_PRELOAD= LD_LIBRARY_PATH= PYTHONPATH= \
  srun --container-image="$IMAGE" --container-mounts="$MOUNTS" \
  --no-container-entrypoint bash "$SCRIPTS/$RUN_SCRIPT" "$config"
