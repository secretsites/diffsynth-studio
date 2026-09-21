#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yiwei/.conda/envs/wan-wm/bin/python}"
PROBE_OUTPUT="${PROBE_OUTPUT:-${REPO_ROOT}/outputs/orca_action_probe}"
PROBE_CACHE="${PROBE_CACHE:-${REPO_ROOT}/work/orca_probe_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" examples/wanvideo/model_training/probe_orca_action_conditioning.py \
  --model "${WAN_MODEL_PATH:-/home/yiwei/workspace/model/Wan2.2-TI2V-5B}" \
  --dataset "${WAN_DATASET_PATH:-/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-command}" \
  --output "${PROBE_OUTPUT}" --cache-dir "${PROBE_CACHE}" "$@"
