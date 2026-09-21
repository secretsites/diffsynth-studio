# ORCA 64×4 全参数探针与 3-chunk 验证

`medium_orca_action_probe.py` 使用现有 `probe_orca_action_conditioning.py` 的全参数训练、动作干预与 VAE 编码。它先冻结清单/门槛，再训练、连续生成和汇总。没有 LoRA，也不会在未通过门槛时自动启动全量训练。

## 数据核查

`audit_orca_command_dataset.py` 全量检查结构、动作/状态原始记录重建、训练集归一化统计，并重新解码六个完整 episode。通过 `--converter` 指向用于解析 native sidecar 的已有转换器，报告保留该脚本路径和 SHA256。只读原始和转换数据。

```bash
python examples/wanvideo/model_training/audit_orca_command_dataset.py \
  --dataset /path/orca-template1-dev-wan-256-command \
  --source /path/orca-template1-dev \
  --converter /path/convert_orca_lerobot.py \
  --output /path/run/dataset_audit.json
```

## 固定实验

```bash
python examples/wanvideo/model_training/medium_orca_action_probe.py prepare \
  --output /path/run/medium --audit /path/run/dataset_audit.json \
  --init-checkpoint /path/previous/dit-final.safetensors
python examples/wanvideo/model_training/medium_orca_action_probe.py train \
  --output /path/run/medium --cache-dir /path/cache
python examples/wanvideo/model_training/medium_orca_action_probe.py rollout \
  --output /path/run/medium
python examples/wanvideo/model_training/medium_orca_action_probe.py summarize \
  --output /path/run/medium
```

prepare 接受 `--dataset` 和 `--model`；默认是当前工作站的 ORCA command 数据及 Wan2.2-TI2V-5B。后续阶段严格使用冻结协议中的路径和参数。

- 均匀选择 64 个训练 episode，每个 4 个相隔至少 28 帧的窗口，共 256 个；13 帧训练输入包含参考帧、4 帧历史、8 帧未来。
- 在初始图像和关节状态近邻内选择未来动作增量不同的成对样本，优先让两者都进入训练集。近邻不是完全相同的物理状态。
- 验证使用全部 15 个官方验证 episode，每个 2 个固定窗口。此前已使用这些 episode，因此不能称为全新盲测。
- 从给定完整 DiT 权重开始，重建 AdamW 优化器；8192 步，主干 lr=1e-5，动作分支 lr=1e-4。BF16 主干/梯度/AdamW moments，FP32 动作分支，梯度检查点；VAE 冻结。没有 FP32 master weights，单张 48GB GPU 的精度/显存折中须保留在实验记录中。
- 24 帧生成是 3 个连续 8 帧 chunk。每个条件在每个 chunk 使用相同噪声；后两段上下文来自该条件自己生成的最后 4 帧，不注入真实未来观测。参考帧始终固定。
- 保持动作是初始最后观测的真实关节状态；反转操作针对全部 24 个未来动作；增量错配是近邻 donor 的变化量接到当前最后动作并裁剪 [-1,1]。第二、三段的历史动作也使用该干预序列。
- 2 个固定噪声种子、50 步采样、30 fps 原速视频；保存无损生成数组、每窗口完整结果及噪声/上下文 hash。

协议将门槛写入 `protocol.json`，SHA256 保存在 `freeze.json`。核心门槛针对保持/反转/增量错配分别要求：纯噪声 sigma=1 验证误差及 24 帧运动区域像素误差都显著优于对照，按 episode bootstrap 95% CI 下界 >0，相对差至少 2%，至少 60% episode 获胜；各 chunk 与各噪声种子平均优势为正。另需优于静止预测至少 10%，并人工检查固定 episode 12/77/144 的全部两个窗口。

`summarize` 会校验 30 窗口 × 2 种子 × 全部条件/时段完整性。可视检查写入 `visual_review.json`，至少包含 `passed`、具体检查对象、观察与局限；缺少可视检查不会通过总门槛。训练集结果及离群跨组错配只作诊断，不计入核心门槛。

检查点仅保存权重。训练过程拒绝覆盖已完成实验；生成可根据每窗口完成标志续跑，结果仍须对应相同的冻结协议与最终检查点。
