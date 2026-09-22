# Wan 世界模型微调（DiffSynth-Studio）

本仓库是基于 DiffSynth-Studio 的 Wan2.2-TI2V-5B 世界模型训练/推理分支。

本分支提供 [ORCA H2 + Sharpa 58D 训练](examples/wanvideo/model_training/README_ORCA.md)：
统一入口 `examples/wanvideo/model_training/train.py` 直接读取原始数据，
支持 abs、delta、relative 动作空间的全参数/LoRA 训练、优化器续训、连续训练及 TensorBoard。
动作定义、归一化、静态增强和配置示例见该文档。

## 训练入口

其他 RLinf 数据集的原生训练入口：

```bash
python examples/wanvideo/model_training/train_rlinf.py ...
```

常用启动脚本位于：

- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_libero_10_posttrain.sh`
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_libero_10_posttrain_test.sh`
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf_agx_realworld.sh`

## 数据集构建（最重要）

当前世界模型训练支持两类数据集。

### 1）仿真 rollout 数据集（`RLinfDataset`）

示例路径：

`/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_10/base_policy_rollout/train_data`

每个 base path 下建议目录结构：

```text
<base_path>/
  <sub_path>/
    <seed_name>/
      rgb.npy      # shape [T, N, 3, H, W]
      actions.npy  # shape [T, N, action_dim]
```

`RLinfDataset` 会扫描 `step_name/seed_name` 并构建滑窗样本。

### 2）真实世界轨迹数据集（`SimpleVLARealWorldRLinfDataset`）

示例路径：

`/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process`

数据文件格式：

```text
<base_path>/
  *.npy  # 每个文件是一条轨迹（长度 T），每个元素是 dict：
         # {"observations": (H, W, C), "actions": (action_dim,)}
```


## 数据集参数说明

数据集实现位置：

- `diffsynth/trainers/dataset.py`

关键参数：

- `Ta`：动作预测窗口长度（未来动作长度）。
- `To`：观测上下文窗口长度（历史长度）。
- `retain_actions`：
  - `False`：动作窗口偏向未来动作；
  - `True`：动作窗口与观测窗口对齐；
  - 两种模式最终都会 padding 到固定长度。
- `action2obs_bias`：
  - 若同一行保存 `o_t` 和随后执行的 `a_t`，且尚未移位，设为 `True`；
  - 若动作已经与其对应的输出观测 `o_{t+1}` 位于同一行，设为 `False`，避免重复移位；
  - 内部会执行“右移一位 + 首位零动作”：
    - `a'[0] = 0`
    - `a'[t] = a[t-1]`
  - 目的是让数据对齐到世界模型的训练对为 `(a_t, o_{t+1})`。
- `repeat`：数据重复倍数。


## `action_dim` 与 Checkpoint Hash 映射（关键）

`action_dim` 会影响 DiT 的 action 投影层，必须与 checkpoint 结构匹配。

模型配置分发逻辑在：

- `diffsynth/models/wan_video_dit.py`

在 `WanModelStateDictConverter.from_civitai()` 中，模型结构由以下哈希决定：

- `hash_state_dict_keys(state_dict)`

然后返回 `config`（包含 `action_dim`、`action_mode` 等）。

### 已有 TI2V hash 示例

- `1f5ab7703c6fc803fdded85ff040c316` -> Wan2.2-TI2V-5B（`wo action`）
- `fcc43a93949201bafeb34aa1eb8bc50f` -> AGX 默认配置（`action_dim: 10`）
- `bc4824aef7c3f23d3378cec6e2b1316c` -> LIBERO 默认配置（`action_dim: 7`）

### 新增 hash 映射流程

1. 先用新 checkpoint 跑一遍，拿到实际 hash（或在 `hash_state_dict_keys` 附近临时打印）。
2. 打开 `diffsynth/models/wan_video_dit.py`。
3. 在 `from_civitai()` 中新增：
   - `elif hash_state_dict_keys(state_dict) == "<your_hash>":`
   - `config = {...}`
4. 确保 `config["action_dim"]` 与数据集动作维度一致。
5. 如果要支持运行时覆盖，保留：
   - `action_dim_override = int(os.environ.get("WAN_ACTION_DIM", "7"))`
   - 并设置 `config["action_dim"] = action_dim_override`。
6. 重新训练，确认 action 相关层无 shape mismatch。

> 说明：
> 如果你在**首次训练**时使用的是原生 Wan 的 base checkpoint，出现 `shape mismatch` warning 通常是预期现象（尤其是 action 相关新增参数）。
> 当你先保存一版训练后的 checkpoint，再基于该 checkpoint 继续训练时，这类 warning 通常会消失，因为模型结构与 checkpoint 键已对齐。


## 最小训练命令示例

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
