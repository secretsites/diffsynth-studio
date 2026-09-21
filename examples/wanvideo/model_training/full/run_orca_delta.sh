#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yiwei/.conda/envs/wan-wm/bin/python}"
WAN_MODEL_PATH="${WAN_MODEL_PATH:-/home/yiwei/workspace/model/Wan2.2-TI2V-5B}"
WAN_DATASET_PATH="${WAN_DATASET_PATH:-/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
MODEL_PATHS="$("${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.dumps([[str(p/f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors") for i in range(1,4)],str(p/"Wan2.2_VAE.pth")]))' "${WAN_MODEL_PATH}")"
TRAIN_PATHS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps([sys.argv[1]+"/train_data"]))' "${WAN_DATASET_PATH}")"
VAL_PATHS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps([sys.argv[1]+"/val_data"]))' "${WAN_DATASET_PATH}")"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file examples/wanvideo/model_training/full/accelerate_single_gpu.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --dataset RLinfDataset --action_dim 58 \
  --train_dataset_base_path "${TRAIN_PATHS}" --val_dataset_base_path "${VAL_PATHS}" \
  --model_paths "${MODEL_PATHS}" --trainable_models dit \
  --height 256 --width 256 --num_frames 13 --Ta 8 --To 4 \
  --action2obs_bias true --retain_actions true --stride 1 --max_finish_step 0 \
  --static_video_prob 0.15 --extra_inputs input_image,action \
  --use_gradient_checkpointing \
  --learning_rate 1e-5 --num_epochs "${NUM_EPOCHS:-1}" \
  --save_epochs "${SAVE_EPOCHS:-1}" --val_interval "${VAL_INTERVAL:-1}" \
  --output_path "${TRAIN_OUTPUT:-outputs/orca_delta}" "$@"
