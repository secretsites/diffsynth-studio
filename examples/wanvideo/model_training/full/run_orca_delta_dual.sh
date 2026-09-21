#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yiwei/.conda/envs/wan-wm/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/wanvideo/model_training/train_orca_delta_distributed.py \
  --dataset "${WAN_DATASET_PATH:-/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta}" \
  --model "${WAN_MODEL_PATH:-/home/yiwei/workspace/model/Wan2.2-TI2V-5B}" \
  --output "${TRAIN_OUTPUT:-outputs/orca_delta_dual_full_epoch1}" \
  --global-batch "${GLOBAL_BATCH:-128}" --no-gradient-checkpointing \
  --workers "${DATA_WORKERS:-2}" --fused-adamw \
  --val-updates 100 --val-windows 120 --checkpoint-updates 100 "$@"
