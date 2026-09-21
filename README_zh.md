# Wan 世界模型微调（DiffSynth-Studio）

本仓库是基于 DiffSynth-Studio 的 Wan2.2-TI2V-5B 世界模型训练/推理分支。

ORCA H2 + Sharpa 58D command 数据在全数据训练前，可运行
[单卡全参数 action-conditioning 快速验证](examples/wanvideo/model_training/ACTION_PROBE.md)：
包括小样本微调、独立轨迹上的正确/错配动作评估，以及相同噪声下的生成视频对照。

长训练入口 `examples/wanvideo/model_training/train_orca_sampled.py` 支持每个采样轮
从每条完整训练轨迹各随机抽取两个不同窗口，并逐轮重新采样。默认训练 1000 个采样轮，
保存第 100/200/500/1000 轮权重，另维护一份包含优化器的滚动续训文件；
这里的“采样轮”不同于全部滑窗遍历一次的 epoch。
该入口要求事先冻结的 `--validation-manifest`，并用 `--resume` 恢复同一配置。
训练和保存参数、动作时间编码、实际采样清单会记录到运行目录。

## 训练入口

主训练入口：

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
      rgb.npy      # shape [T, N, 3, H, W] 或 [T, N, H, W, 3]
      actions.npy  # shape [T, N, action_dim]
```

`RLinfDataset` 会扫描 `step_name/seed_name` 并构建滑窗样本。

ORCA command 数据使用 `train_data/orca/episode_NNNNNN/` 和
`val_data/orca/episode_NNNNNN/`；分别将 `train_data` / `val_data` 作为 base path。
额外的 `states.npy` 是归一化关节状态，用于构造保持动作。
该数据为 14 维双臂命令 + 44 维原生手部 desired 命令，需 `action_dim=58`。
转换时已经令 `actions[t]=source_action[t-1]`，因此必须设置
`action2obs_bias=False`，不能再次移位。ORCA 专用探针会保留真实的初始保持动作。

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
  - 若同一行保存观测 `o_t` 和随后执行的 `a_t`，且尚未转换到输出帧对齐，可设为 `True`；
  - 若动作已经与其导致的 `o_{t+1}` 位于同一行，应设为 `False`；
  - 内部会执行“右移一位 + 首位零动作”：
    - `a'[0] = 0`
    - `a'[t] = a[t-1]`
  - 目的是让数据对齐到世界模型的训练对为 `(a_t, o_{t+1})`。
- `repeat`：数据重复倍数。
- `stride`：滑窗步长，默认 1。增大步长会减少重叠窗口；不同 stride 的 epoch 不能直接比较。
- `max_finish_step`：轨迹截断位置，默认 0 使用整条轨迹。
- `max_train_steps_per_epoch`：仅用于显式限制调试轮数，默认不限制；一轮会完整遍历训练集。

ORCA command 数据全量训练应使用 `--action_dim 58 --Ta 8 --To 4
--retain_actions true --action2obs_bias false --max_finish_step 0`。
保留官方 `val_data` 作为验证集；这里的“全量训练集”指全部 136 个训练 episode。
绝对关节位置命令中的零向量并不代表保持当前姿态，因此该数据应设置
`--static_video_prob 0`，避免通用入口的“静态视频 + 零动作”增强。

## 动作时间编码

动作 cross-attention 现在默认使用固定正弦时间编码：
`action_mlp1(action[t]) + PE(t)`。训练与 rollout 共用
`WanModel.embed_action_context()`，原有按 4 帧动作拼接的 modulation 分支保留。
在 5 帧条件模式中，槽位 0 表示 episode 参考帧，1–4 表示最近历史，5 起为未来动作。
这是窗口内的位置编号，不是参考帧与当前帧的实际时间间隔；每个 rollout chunk 使用相同局部编号。

训练入口可用 `--action_time_encoding sinusoidal`（默认）或 `none`；其他脚本及推理可通过
`WAN_ACTION_TIME_ENCODING=sinusoidal` / `none` 设置，训练和推理必须一致。
时间编码不增加参数，checkpoint 张量形状和键名不变；旧无时间编码 checkpoint 的行为复现必须显式选 `none`。
该开关不存放在权重张量中，应与训练配置一起保存。补充顺序信息不等于已经通过动作控制验证。

吞吐测量入口 `examples/wanvideo/model_training/benchmark_orca_training.py` 会短暂进行全参数更新，
分别测量预缓存 latent 和在线 VAE 的每步时间，不保存模型权重。其数字适用于单卡、batch=1、
256×256、8 帧预测、BF16 骨干及 AdamW 状态、FP32 动作 MLP 的探针训练配置；
更改为 FP32 主权重、CPU offload 或更长预测窗口后需要重新测速。


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
