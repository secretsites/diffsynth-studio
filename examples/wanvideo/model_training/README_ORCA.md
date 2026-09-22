# ORCA 训练

统一入口是 [train.py](train.py)，通用配置是 [configs/orca_raw_train.yaml](configs/orca_raw_train.yaml)。支持 `abs`、`delta`、`relative` 动作空间的全参数和 LoRA 训练，以及单卡/多卡、梯度累积、优化器续训和 TensorBoard。直接读取原始 LeRobot 数据，无需离线转换。

## 启动与配置

在仓库根目录执行，先激活 `wan-wm` 环境。配置中的数据、基础模型路径和 GPU 编号应与实际环境一致。

```bash
conda activate wan-wm
python examples/wanvideo/model_training/train.py \
  --config examples/wanvideo/model_training/configs/orca_raw_train.yaml \
  --set dataset.action_mode=relative \
  --set output=outputs/orca_relative_train \
  --check
```

移除 `--check` 开始训练。`--check` 检查配置和输入路径；`--dry-run` 只打印解析后的配置及命令。两者均不启动训练。`--set key=value` 可重复使用，相对路径按仓库根目录解析，未知字段直接报错。每次实验保存完整 `resolved_config.yaml`，新实验使用新的输出目录。

| 配置 | 含义 |
| --- | --- |
| `dataset.action_mode` | `abs` / `delta` / `relative` |
| `training.mode` | `full` 训练整个 DiT；`lora` 训练 LoRA 和 action MLP |
| `training.epochs` | 本次执行的轮数；每轮遍历全部训练窗口 |
| `training.micro_batch_size` | 每张 GPU 单次前向/反向的实际窗口数 |
| `training.global_batch` | 多卡和梯度累积后的总窗口数 |
| `training.gradient_checkpointing` | 重算激活以减少显存，通常增加计算时间 |
| `checkpoint.every_updates` | 每多少次优化器更新写入最新恢复状态；0 关闭轮内保存 |
| `checkpoint.save_epochs` | 导出权重的累计 epoch 编号；`null` 每轮导出，`[]` 不导出 |
| `validation.every_updates` / `windows` | 固定验证子集的检查间隔和窗口数 |
| `validation.full_at_end` | 最后一个请求 epoch 完成后遍历全部验证窗口 |
| `logging.tensorboard` | 将训练、验证及性能指标写入 TensorBoard |

`global_batch = GPU 数 × micro_batch_size × 梯度累积步数`，必须整除。例如两卡 `global_batch=128`、`micro_batch_size=4` 对应每卡累积 16 次。配置默认 microbatch 为 1；当前机器已验证 relative 全参数训练可使用 4。最后不足一个 global batch 的窗口按实际样本数计算梯度，多卡补齐窗口权重为零。

## 动作空间

58 维动作由 14 维手臂目标和 44 维手部目标组成。设原始目标为 `target[i]`、实测状态为 `state[i]`，预测窗口起点为 `t`：

| 模式 | 归一化前的动作 |
| --- | --- |
| `abs` | `target[i]` |
| `delta` | `target[i] - state[i]` |
| `relative` | `target[i] - state[t]` |

每个样本是固定参考帧 + 4 帧历史 + 8 帧未来，图像为 256×256。动作 `i` 对应输出帧 `i+1`，加载器只移动一次；参考槽和缺失历史槽使用零值。relative 的参考是每个窗口起点的实测状态。

训练与评估共用 [orca_online.py](../../../diffsynth/trainers/orca_online.py) 和 [orca_source.py](../../../diffsynth/trainers/orca_source.py)。归一化统计只从训练 episode 拟合：abs 使用分位数中心和尺度，delta/relative 保持物理零映射为零；`normalization: none` 使用物理单位。每轮输出 `action_stats.json`，评估可通过 `dataset.action_stats` 固定使用它。

静态增强中，abs 动作保持参考帧姿态，delta/relative 使用零动作。验证时关闭静态增强。动作定义、归一化及 relative 参考约定会写入 checkpoint 并在恢复/评估时校验。已转换的 delta NPY 数据仍可用 `dataset.format: npy` 读取，不能重新解释成 abs 或 relative。

## LoRA

使用同一个配置覆盖训练模式，无需独立脚本：

```bash
python examples/wanvideo/model_training/train.py \
  --config examples/wanvideo/model_training/configs/orca_raw_train.yaml \
  --set dataset.action_mode=relative \
  --set training.mode=lora \
  --set training.learning_rate=1e-4 \
  --set lora.rank=32 \
  --set output=outputs/orca_relative_lora
```

LoRA 冻结基础 DiT，训练注意力/FFN 适配器及 action MLP；可训练参数保留 FP32。导出的 `.safetensors` 仅包含适配器和 action MLP，评估时需要原基础模型。全参数模式导出完整 DiT；两种模式都冻结 VAE。

## 恢复与增加轮数

从完整 epoch 继续时，使用原实验的解析后配置和 `resume_latest.pt`，设置累计已完成轮数，并选择新的输出目录：

```bash
python examples/wanvideo/model_training/train.py \
  --config outputs/orca_relative_train/resolved_config.yaml \
  --set initialization.mode=continue \
  --set initialization.checkpoint=outputs/orca_relative_train/checkpoints/resume_latest.pt \
  --set initialization.completed_epochs=2 \
  --set training.epochs=4 \
  --set 'checkpoint.save_epochs=[4,6]' \
  --set output=outputs/orca_relative_continue
```

此例执行累计第 3–6 轮并保留优化器状态。若恢复中断的 epoch，将模式设为 `resume`，保留原输出目录及该轮的覆盖日志，`completed_epochs` 填中断前完整结束的轮数。例如第 3 轮中断时填 2。恢复文件必须是包含模型和优化器的 `.pt`；导出的 `.safetensors` 用于推理。

## 连续全参数训练

[continue_orca_training.py](continue_orca_training.py) 接管已有全参数实验，适用于三种动作空间。它读取该实验的 `resolved_config.yaml`，等待已有训练结束，再从最新完整或部分 checkpoint 接着训练。先通过 `train.py` 建立实验，再保存如下策略为 `continuous.yaml`：

```yaml
version: 1
run: outputs/orca_relative_train
enabled: true
save_every_epochs: 2
keep_last_exports: 2
full_validation_every_epochs: 2
reserve_gib: 4
poll_seconds: 15
```

```bash
python examples/wanvideo/model_training/continue_orca_training.py --config continuous.yaml
```

该策略无 epoch 上限，每 2 轮导出一次权重并完整验证，滚动保留最近两份符合间隔的权重及最新恢复状态。删除前会核验新权重及窗口覆盖；只处理当前 run 的受管理权重。磁盘不足时等待。策略在轮间重新读取，改为 `enabled: false` 会在当前轮结束后停止。

当前 relative 实验的完整配置保存在 `outputs/orca_raw_relative_full_epoch1/resolved_config.yaml`。运行中的监督进程仍读取 [orca_raw_relative_full_continuous.yaml](configs/orca_raw_relative_full_continuous.yaml)，因此保留该策略文件。新实验使用通用模板；不要为同一 run 重复启动监督进程。

## 输出与 TensorBoard

```text
output/
  resolved_config.yaml
  job_status.json / progress.json
  checkpoints/
    resume_latest.pt          # 最新模型和优化器恢复状态
    epoch-N.safetensors       # 指向对应 epoch 导出权重的链接
  epochs/epoch-NNNN/
    training.log / metrics.jsonl / validation.jsonl
    action_stats.json / run_config.json
    source_snapshot/ / source_hashes.json
    processed_windows.rankN.jsonl / coverage_verification.json
    result.json
  tensorboard/
```

```bash
python -m tensorboard.main \
  --logdir outputs/orca_relative_train/tensorboard \
  --host 127.0.0.1 --port 6007
```

打开 <http://127.0.0.1:6007/#scalars>。查看 `Loss/train`、验证损失、训练进度和吞吐；跨 epoch 的 step 连续。当前实验的服务已经运行在该端口。

## 代码入口

- [train.py](train.py)：YAML 配置、多轮调度、续训和 TensorBoard 桥接。
- [train_orca.py](train_orca.py)：共用的单卡/DDP 后端、全参数/LoRA、microbatch、性能测量和 checkpoint。
- [continue_orca_training.py](continue_orca_training.py)：连续训练与滚动保留。
- [train_rlinf.py](train_rlinf.py)：共用的 `WanTrainingModule`，以及其他 RLinf 数据集的原生入口。
- [tensorboard_orca_live.py](tensorboard_orca_live.py)：日志转换为 TensorBoard events。
- [评估入口](../model_inference/README_ORCA.md)：与训练共用动作编码和 checkpoint 元数据。

`train_orca_delta_distributed.py` 仅转发到新后端，供旧命令和正在运行的监督进程兼容使用。旧 delta 转换、审计、smoke、shell 和续训脚本已由以上入口替代。
