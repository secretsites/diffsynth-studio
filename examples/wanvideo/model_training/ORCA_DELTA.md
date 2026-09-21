# ORCA H2 + Sharpa：58D delta action 适配

本分支从原 `main`（`6d9ce59`）建立，使用原生 `train_rlinf.py` 训练路径。动作定义为 **当前目标位置减当前实际位置**，不是相邻目标之差，也不是下一帧实际位置之差。

## 数据语义

```text
raw_delta[t] = source_target[t] - source_state[t]
actions.npy[t] = clip(raw_delta[t] / scale, -1, 1)
```

前 14 维目标来自主 parquet 的双臂 action，后 44 维来自原生手部 telemetry 中按采样时钟因果选取的 `desired` 命令。主表的手部反馈代理不用于动作目标。减法在原始关节位置数值上进行，不从已经裁剪、归一化、移位过的旧 action 倒推；因此保留了第一帧和最后一帧的真实原始命令。

仅用 136 个训练 episode，逐维取 `abs(raw_delta)` 的 99% 分位值作为 scale；接近常量的维度使用 scale=1。**不减均值、不减分位中心**，保证原始零增量在模型输入中严格为 0。超过 scale 的数值裁剪到 `[-1,1]`，裁剪率记录在转换审计中。缩放统计不使用验证集。

### 文件里的 action 不移位

磁盘第 t 行保存同一时刻的观测和 delta[t]；不在转换时插入初始零动作、不丢掉最后一条原始命令。元数据明确写入 `action2obs_bias_applied=false`。

训练必须启用 `--action2obs_bias true`，加载器执行一次：

```text
aligned[0] = 0
aligned[t] = actions.npy[t-1]   # t > 0
```

当前观测索引为 t 时，未来图像 `t+1…t+8` 对应源文件动作 `delta[t…t+7]`。原命令数据已经做过移位，不能把旧 `-command` 目录配合该选项再次移位；加载器会拒绝已声明的重复移位。未移位 delta 数据若关闭该选项也会报错。

### 其他数据保持原样

沿用 `-command` 数据的 136/15 划分和 RGB、states 数组。`states.npy` 仍是旧的**归一化绝对关节反馈**，其统计放在 `state_stats.json`；新的 `action_stats.json` 只描述 delta 的零中心缩放，两者不能混用。模型本身仍输入图像和 action，没有新增数值 state 分支。

转换结果为：

```text
orca-template1-dev-wan-256-delta/
  dataset_info.json
  action_stats.json
  state_stats.json
  conversion_audit.json
  adaptation_verification.json       # 独立审计后生成
  train_data/orca/episode_NNNNNN/{rgb,actions,states}.npy
  val_data/orca/episode_NNNNNN/{rgb,actions,states}.npy
```

RGB 为 uint8 `[T,1,256,256,3]`，actions/states 为 float32 `[T,1,58]`。同一文件系统优先硬链接 RGB/states，避免重复占用图像空间；均按只读数据使用。旧数据目录保持原状。

## 转换与独立核验

在仓库根目录执行，输出目录必须尚不存在：

```bash
python examples/wanvideo/model_training/convert_orca_delta.py \
  --source-root /home/yiwei/workspace/datasets/orca-template1-dev \
  --template-root /home/yiwei/workspace/datasets/orca-template1-dev-wan-256-command \
  --output-root /home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta

python examples/wanvideo/model_training/audit_orca_delta.py \
  --dataset /home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta
```

原生命令读取代码已包含在本仓库，不依赖另一个工作区中的转换脚本。转换会先核验模板的绝对 action/state 与原始记录一致；独立审计再重建全部 delta、重算训练统计、逐 episode 检查实际加载器的起始/内部/末尾窗口，并比对全部 RGB/states。

## 原生训练配置

```bash
bash examples/wanvideo/model_training/full/run_orca_delta.sh
```

| 项目 | 配置 |
|---|---|
| 初始化 | 原始 Wan2.2-TI2V-5B；新增动作参数按缺失键正确初始化 |
| 优化范围 | 全 DiT，包括两个 action MLP；VAE 冻结，无 LoRA |
| 动作维度 | 58；modulation 每组 4 帧，输入 232 |
| `action2obs_bias` | true，加载器移位一次 |
| `retain_actions` | true，保留四帧历史动作；参考动作补零 |
| `static_video_prob` | 0.15，沿用主训练入口默认比例 |
| 静态增强实现 | 训练时随机将 13 帧替换成参考帧，并将全部动作置零；验证模式不增强 |
| 图像窗口 | 参考帧＋4 帧历史＋8 帧未来，256×256 |
| 数据范围 | `max_finish_step=0` 保留完整 episode，stride=1 |
| 条件分支 | 保留原 cross-attention＋modulation，两者均开启 |
| 动作时间编码 | 未加入上一实验分支的 sinusoidal action PE |
| 扩散训练 | 沿用主分支噪声调度、参考/历史 latent 处理、未来 latent 损失掩码 |
| 运行配置 | 单 GPU，BF16，gradient checkpointing，学习率 1e-5 |

可以通过 `PYTHON_BIN`、`WAN_MODEL_PATH`、`WAN_DATASET_PATH`、`TRAIN_OUTPUT`、`NUM_EPOCHS`、`SAVE_EPOCHS`、`VAL_INTERVAL`、`CUDA_VISIBLE_DEVICES` 覆盖路径或运行参数；脚本末尾附加的参数也会传给训练入口。

**本启动脚本沿用主分支训练循环的每 epoch 最多 501 个 batch 上限。** `max_finish_step=0` 仅保证所有 episode 帧都进入可采样窗口集合，不代表每轮遍历全部窗口。本分支没有移植之前“每 episode 每轮随机抽 2 个窗口”的采样训练器；这两个 epoch/采样轮概念不能混用。脚本默认 1 个 epoch，执行脚本才会启动训练。

旧 abs 权重即使同为 58 维，也没有按新 delta 语义训练。本脚本默认从原始 base model 开始；此前 abs 模型的 rollout 和验收结果不能当成这个新表示的结果。归一化全零现在具有保持命令的正确数值含义，但模型能否学会停止仍需新训练后的独立生成验证。

## 适配修复与测试

- 修复 `action2obs_bias`、`retain_actions`、Ta/To 等参数仅被解析却未传给数据集的问题，以及 Python `bool('false')` 的解析问题。
- 修复起始窗口补帧长度，保留原零动作 reference/padding；uint8 暗帧不再被误按浮点 `[0,1]` 图像放大。
- 增加 58D 完整检查点识别和实际动作维度校验。
- 从原始 base 加载时，根据缺失键初始化 action MLP，避免 `to_empty` 的有限但未初始化内存进入训练；不覆盖已有动作权重。
- 修复五帧条件、2D 动作输入路径中使用未定义 batch 变量的问题；保持原条件计算方式。

```bash
python -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/model_training/smoke_orca_delta.py \
  --dataset /home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta \
  --model /home/yiwei/workspace/model/Wan2.2-TI2V-5B \
  --output /path/to/model_smoke.json
```

Smoke 检查使用真实 5B 模型与原生训练损失，各执行一次普通样本和强制静态样本的前向/反向，检查动作维度、输入、损失与代表性梯度。无优化器更新，不保存模型；它验证运行通路，不是训练质量评估。全零输入时第一层 action MLP 的 weight 梯度为零是正常乘法结果，下游 bias 和主干仍可有梯度。
