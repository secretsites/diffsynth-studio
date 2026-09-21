# Wan World-Model Finetuning (DiffSynth-Studio)

This repository is a Wan2.2-TI2V-5B world-model training/inference fork based on DiffSynth-Studio.

## Training Entry

Main training entry:

```bash
python examples/wanvideo/model_training/train_rlinf.py ...
```

Common launch scripts are under:

- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_libero_10_posttrain.sh`
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf.sh`
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf_agx_realworld.sh`


## Dataset Construction (Most Important)

We currently support two dataset families for world-model training.

### 1) Simulation rollout dataset (`RLinfDataset`)

Example base path:

`/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/base_policy_rollout/train_data`

Expected directory shape under each base path:

```text
<base_path>/
  <sub_path>/
    <seed_name>/
      rgb.npy      # shape [T, N, 3, H, W]
      actions.npy  # shape [T, N, action_dim]
```

`RLinfDataset` will scan `step_name/seed_name` folders and build sliding windows.

### 2) Real-world trajectory dataset (`SimpleVLARealWorldRLinfDataset`)

Example base path:

`/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process`

Expected files:

```text
<base_path>/
  *.npy  # each file is one trajectory (length T), each element is a dict:
         # {"observations": (H, W, C), "actions": (action_dim,)}
```


## Dataset Parameters Explained

The dataset classes are implemented in:

- `diffsynth/trainers/dataset.py`

Key parameters:

- `Ta`: action prediction horizon (future action length).
- `To`: observation context window (history length).
- `retain_actions`:
  - `False`: action window is centered on future actions.
  - `True`: action window is aligned with observation window.
  - Both modes are padded to a fixed final length.
- `action2obs_bias`:
  - Set `True` when each row contains `o_t` and the subsequent command `a_t`, without a prior shift.
  - Set `False` when commands are already aligned to their output observations `o_{t+1}`; do not shift twice.
  - Internally applies right shift with leading zero action:
    - `a'[0] = 0`
    - `a'[t] = a[t-1]`
  - This aligns supervision to world-model pairing `(a_t, o_{t+1})`.
- `repeat`: dataset repeat factor.


## `action_dim` and Checkpoint Hash Mapping (Critical)

`action_dim` affects DiT action projection layers and must match checkpoint structure.

The model-config dispatch is implemented in:

- `diffsynth/models/wan_video_dit.py`

Inside `WanModelStateDictConverter.from_civitai()`, model structure is selected by:

- `hash_state_dict_keys(state_dict)`

Then a `config` dict is returned (including `action_dim`, `action_mode`, etc.).

### Existing TI2V hash examples

- `1f5ab7703c6fc803fdded85ff040c316` -> Wan2.2-TI2V-5B (`wo action`)
- `fcc43a93949201bafeb34aa1eb8bc50f` -> AGX default config (`action_dim: 10`)
- `bc4824aef7c3f23d3378cec6e2b1316c` -> LIBERO default config (`action_dim: 7`)

### How to add a new hash mapping

1. Run with your new checkpoint once and inspect the detected hash path (or add a temporary print around `hash_state_dict_keys`).
2. Open `diffsynth/models/wan_video_dit.py`.
3. In `from_civitai()`, add a new:
   - `elif hash_state_dict_keys(state_dict) == "<your_hash>":`
   - `config = {...}`
4. Make sure `config["action_dim"]` matches your dataset action dimension.
5. If you want runtime override, keep:
   - `action_dim_override = int(os.environ.get("WAN_ACTION_DIM", "7"))`
   - and set `config["action_dim"] = action_dim_override`.
6. Re-run training to verify no shape mismatch in action-related layers.

> Note:
> If your **first** training run starts from the original Wan base checkpoint, `shape mismatch` warnings are expected (especially around newly introduced action-related parameters).
> After you save a finetuned checkpoint and resume from that checkpoint, these warnings should typically disappear because model structure and checkpoint keys become aligned.


## Minimal Training Example

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --dataset RLinfDataset \
  --train_dataset_base_path '["/path/to/train_data"]' \
  --val_dataset_base_path '["/path/to/val_data"]' \
  --action_dim 7 \
  --Ta 8 \
  --To 4 \
  --action2obs_bias true \
  --retain_actions false
```
