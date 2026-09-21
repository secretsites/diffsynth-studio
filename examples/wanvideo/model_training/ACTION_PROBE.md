# ORCA command 数据：全参数 action-conditioning 快速验证

```bash
cd /home/yiwei/workspace/wm-test/diffsynth-studio-rlinf
bash examples/wanvideo/model_training/full/run_orca_action_probe.sh
```

默认使用已有 `wan-wm` 环境、Wan2.2-TI2V-5B 和
`orca-template1-dev-wan-256-command`，单卡，全 DiT + 两个 action MLP 训练；
不插入 LoRA，VAE 冻结。可用 `PROBE_OUTPUT`、`PROBE_CACHE`、
`WAN_MODEL_PATH`、`WAN_DATASET_PATH` 覆盖路径。已有最终权重时拒绝覆盖。

默认实验固定 32 个训练 episode × 2 个窗口 = 64 个训练样本，
8 个不重叠的验证 episode × 2 个窗口 = 16 个验证样本。
全数据只用于检查格式和从预先指定的 episode 中选择高运动片段，
验证 episode 不参与优化，也不与训练窗口重叠。
每个窗口是 episode 第 0 帧 reference + 连续 12 帧，其中前 4 帧为上下文，
后 8 帧为预测目标（30 fps，预测 0.267 秒）。

## 必须保持的数据约定

- 58 维顺序：左臂 7、右臂 7、左手 22、右手 22。
- command 数据已按输出图像对齐：`actions[0]=states[0]`；
  `actions[t]=source_action[t-1]`。不要再次平移。
- 已经按训练集统计量归一化到 [-1,1]，不要再次归一化。
- 保持当前姿态使用最后一帧上下文的 `states.npy`，不是全零。
- 所有干预只替换未来 8 个动作，reference、4 帧上下文及其动作均保持不变。
- VAE 编码前 5 帧与完整 13 帧编码的前两帧 latent 做一致性检查。
  自由生成重新仅编码 5 帧上下文，不接收真实未来画面。

## 训练与评估

先记录未训练 action 分支的基线，然后固定 800 步全参数短训（最多 3600 秒训练），
最后成对评估。训练和推理都使用干净上下文，只对未来两帧 latent 计算 flow-matching
MSE。使用 shift=5 的 sigma 采样；DiT 学习率 1e-5，新 action MLP 学习率 1e-4，
AdamW、gradient checkpointing、全局梯度裁剪 1.0。

为在 48GB 单卡上容纳全参数优化，预训练 DiT 权重、梯度和 AdamW moments 使用 BF16，
action MLP 使用 FP32，关闭 AdamW foreach。此配置没有 FP32 master weights；
它是单卡快速探针，正式长训应另外评估混合精度/优化器状态方案。

验证包括真实动作、保持姿态、跨轨迹错配全部未来动作、仅错配手臂、仅错配手指。
错配 donor 从同一 split 的不同 episode 中，按当前状态选最近 8 个候选，再取
未来动作差异最大的一个；剩余状态差异写入 manifest。跨轨迹错配不是有真实
反事实 ground truth 的物理实验。

各对照共享窗口、噪声 seed（2026、2027）和 sigma（0.5、0.9、1.0）。
`sigma=1.0` 时未来 latent 为纯噪声，是减少真实未来泄漏影响的额外检查。
另测 8 个训练窗口判断是否只在训练数据上有效。
按 episode 聚合差值并做 5000 次 episode bootstrap，避免把不同噪声重复计为独立轨迹。

`delta_mse = 对照 MSE − 正确动作 MSE`：正值支持正确动作更有用。
`prediction_rmse > 0` 只表示输出对动作敏感；未训练分支也可能满足，不能单独作为通过标准。
只有验证集正确动作优势、相比初始化基线的改善，以及自由生成的画面质量共同支持结论，
才能认为已得到短时间跨度的实用证据。分别报告 arm/hand，不把手臂结果当作手指验证。
bootstrap 区间是小样本诊断量，不代表长期 rollout 或所有任务上的泛化保证。

## 产物与重复运行

输出目录包含 `config.json`、`manifest.json`、`gradient_audit.json`、
`train.jsonl`、`eval-{initial,final}.{json,csv}`、`result.json`、
`dit-step-400.safetensors`、`dit-final.safetensors`、对照图/视频和像素误差。
最终权重是完整 DiT（约 10GB，包含 58D action MLP），不包含冻结的 VAE。
保存的是模型权重，不包含精确续训所需的 optimizer/RNG 状态。

```bash
# 仅检查数据并生成窗口清单
bash examples/wanvideo/model_training/full/run_orca_action_probe.sh --prepare-only

# 重新评估已经保存的完整 DiT；沿用原输出目录和缓存路径
bash examples/wanvideo/model_training/full/run_orca_action_probe.sh \
  --eval-checkpoint /absolute/path/to/dit-final.safetensors

# 不同输出目录另起一次预先设定的实验
PROBE_OUTPUT=/absolute/path/to/another-run \
  bash examples/wanvideo/model_training/full/run_orca_action_probe.sh --steps 1600

# 回归检查
/home/yiwei/.conda/envs/wan-wm/bin/python -m unittest discover -s tests -v
```

脚本通过本仓库的 `WanVideoPipeline` / `model_fn_wan_video` 执行模型，
为这个探针显式构造时序窗口和训练循环，不经过旧 `train_rlinf.py`。
旧入口的参数透传、边界补帧和静态增强语义需要在正式全数据训练前单独处理；
不要直接将旧的默认静态增强（action=0）用于这份绝对关节 command 数据。

## 独立复核与更强对照

800 步是初筛，可在另一输出目录预先设定 4000 步；例如 `--steps 4000 --save-every 2000`。
在查看较长实验结果前，固定未用于初筛的验证 episode，再做独立复核：

```bash
PROBE_OUTPUT=/absolute/path/to/confirmation \
  bash examples/wanvideo/model_training/full/run_orca_action_probe.sh \
  --prepare-only --val-episodes 7 --val-episode-ids 13 61 90 106 109 121 142

PROBE_OUTPUT=/absolute/path/to/confirmation \
  bash examples/wanvideo/model_training/full/run_orca_action_probe.sh \
  --eval-checkpoint /absolute/path/to/4000-step/dit-final.safetensors \
  --eval-initial --eval-label final --render-windows 14

/home/yiwei/.conda/envs/wan-wm/bin/python \
  examples/wanvideo/model_training/summarize_orca_action_probe.py \
  /absolute/path/to/confirmation
```

新评估入口还包括 `reverse`：仅反转未来 8 个动作，保持各维数值分布不变。
`motion_region_mse` 是将真实未来相对末帧上下文变化的区域映射到 latent 网格后计算的误差；
它是辅助局部指标，不是关节跟踪误差。`rollout_metrics.json` 中的 `persistence`
是一直重复最后一帧上下文的基线；视频统一以 10 fps 慢放，原数据为 30 fps。

汇总会输出 `paired_conclusion.json`，分别记录所有 sigma、纯噪声未来、
运动区域，以及训练前后动作优势的变化。五组基础对照的所有诊断条件通过，
仍只支持短片段、组级动作使用证据，不代表 58 个关节均得到精确控制，也不代表长时 rollout 可靠。
当前 58D 完整 checkpoint 还已注册到 `ModelManager`，可从文件形状直接识别动作维度。

`delta_swap` 将 donor 的未来指令相对其末帧上下文指令的变化量，接到当前窗口的
末帧上下文指令上，再裁剪到已有归一化范围 [-1,1]，减少绝对姿态突然跳变的影响。
它仍是离线合成的替代指令，并没有真实反事实视频标签。复核视频同时展示此对照；
仅显示末帧上下文和未来 8 帧，避免 episode reference 引入时间跳跃。
可用 `--inference-steps 50` 增加自由生成的去噪步数。
