# ORCA：YAML 评估入口

统一入口 [eval.py](eval.py)，配置 [configs/orca_raw_eval.yaml](configs/orca_raw_eval.yaml)。模型、episode、起始帧、预测长度、采样参数和动作对照都通过配置选择，不再绑定第8轮或ep144。

## 启动

```bash
cd /home/yiwei/workspace/wm-test/diffsynth-studio-rlinf
conda activate wan-wm

# 仅显示配置和任务，不创建输出、不加载模型。
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml --dry-run

# 校验真实数据的native窗口、checkpoint结构/哈希，不做GPU推理、不写run输出；原始数据模式可能填充自动缓存。
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml --check

# 默认：第8轮模型、ep144、440帧、TF、真实/全零两组。
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml
```

首次验证先用24帧，同时做AR/TF：

```bash
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml \
  --set output=outputs/orca_eval_24 \
  --set rollout.frames=24 \
  --set 'rollout.modes=[ar,tf]'
```

直接编辑YAML，或用 `--set key=value` 覆盖。相对路径按仓库根目录解析，最终配置会写入输出目录的 `resolved_config.yaml`。

## 更换模型和数据

Relative LoRA训练导出的`epoch-1.safetensors`包含LoRA及action MLP，评估入口通过元数据自动识别适配器，并从`model.base_path`加载原基础DiT后应用它。请保留训练时使用的基础模型，并设置相同的relative动作定义和归一化：

```bash
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml \
  --set model.checkpoint=outputs/orca_relative_lora/checkpoints/epoch-1.safetensors \
  --set dataset.action_mode=relative \
  --set output=outputs/orca_relative_lora_eval
```

适配器会校验全部LoRA/action MLP键和形状；全量DiT checkpoint仍按原流程加载。

```yaml
model:
  base_path: /home/yiwei/workspace/model/Wan2.2-TI2V-5B
  checkpoint: /path/to/another-delta-epoch.safetensors
  checkpoint_sha256: null  # 不固定旧hash；可显式填写SHA256来固定某个文件
rollout:
  episode: val_data/orca/episode_000150
  start_frame: 16
  frames: 80
  modes: [tf]
  actions: [real, zero]
sampling:
  steps: 50
  sigma_shift: 5.0
  seed: 12345
  seed_stride: 100000
```

模型路径使用**完整DiT推理 `.safetensors`**，base_path提供VAE/tokenizer；支持当前原生58维、DiT action_mode=both、five-frame condition的Wan2.2-TI2V-5B checkpoint。dataset.action_mode是数据语义，和DiT的action_mode=both是不同概念。旧abs+额外时间编码模型及其他架构不适用。

## 原始数据在线评估

使用 [configs/orca_raw_eval.yaml](configs/orca_raw_eval.yaml) 直接读取 `/home/yiwei/workspace/datasets/orca-template1-dev`。默认delta可与现有第8轮模型配合：

```bash
python examples/wanvideo/model_inference/eval.py \
  --config examples/wanvideo/model_inference/configs/orca_raw_eval.yaml \
  --set output=outputs/orca_raw_eval_24 \
  --set rollout.frames=24
```

三种模式和训练共用同一个加载器：abs为物理目标，delta为目标减同一时刻实际state，relative为目标减**本chunk预测起点实际state**。详细归一化、训练/验证划分及时间对齐见 [训练说明](../model_training/README_ORCA.md#动作空间)。相应字段为：

```yaml
dataset:
  format: lerobot
  path: /home/yiwei/workspace/datasets/orca-template1-dev
  action_mode: relative
  action_stats: /path/to/training/epochs/epoch-0001/action_stats.json
model:
  checkpoint: /path/to/relative-epoch.safetensors
```

评估模式必须与checkpoint一致，不能把现有delta权重直接用于abs/relative。新checkpoint还会核对归一化尺度和relative参考定义。旧第8轮只有delta语义元数据，没有完整尺度；本仓库默认raw delta已与该模型原训练数据逐项核验一致，使用旧模型时保留默认训练尺度。

`rollout.episode`仍写成 `val_data/orca/episode_000144`，这是逻辑episode选择，不要求原始数据中存在该NPY目录；所选episode必须属于配置的train/val划分。原始格式仅支持 `environment_index: 0`。统计量只从训练episode拟合，评估可指定训练输出的 `action_stats.json` 以固定尺度。

AR中的delta和relative也使用**记录中的实际state**：delta逐动作时刻相减，relative每chunk按记录起点相减，不从生成图像估计关节状态。全零对照把归一化后的全部输入置零；在abs模式下这表示训练分布中心，不能解释为保持姿态。

## 参数与控制方式

| 字段 | 含义 |
| --- | --- |
| `output` | 新输出目录，拒绝覆盖已有数据 |
| `devices` | 一个或多个CUDA设备；每卡同时一个任务，任务多于卡数则分批 |
| `cpu_threads` | 每个推理进程CPU线程数 |
| `dataset.path`、`format` | 原始LeRobot目录（lerobot）或已有delta NPY目录（npy） |
| `dataset.action_mode`、`action_stats` | 动作定义及可选的训练统计文件，必须匹配checkpoint |
| `rollout.episode` | 根目录下的 `split/task/episode` 相对路径 |
| `rollout.environment_index` | N维的环境下标，当前数据为0 |
| `rollout.start_frame` | 已观测到的当前帧t；未来预测从t+1开始 |
| `rollout.frames` | 预测帧数，必须为8的倍数；null使用剩余完整chunk，末尾不足8帧不补齐 |
| `rollout.modes` | `[tf]`、`[ar]` 或 `[ar,tf]` |
| `rollout.actions` | `[real]`、`[zero]` 或 `[real,zero]` |
| `sampling.steps`、`sigma_shift` | 去噪步数及FlowMatchScheduler的shift |
| `sampling.seed`、`seed_stride` | 本次第k段噪声种子=seed+seed_stride×k；所有组对应段使用相同噪声 |
| `render.enabled` | 是否导出视频；false仍核验输入输出并计算指标 |
| `render.fps`、`difference_gain` | 视频帧率和像素差图放大倍数，不影响模型预测 |

开始时reference始终是真实第0帧，历史是t−3…t四帧，负下标重复第0帧。**非零start_frame不会重复起始帧来伪造静态历史**。

- TF每个chunk都接回相同真实历史图像，生成图像不反馈。
- AR在首段后使用固定第0帧+自身最后4个预测帧。
- 真实动作按所选abs/delta/relative定义在线编码，时间对齐仅在loader中做一次；不根据生成图像重算state。
- 全零动作把全部13×58输入置零，包括历史动作槽。图像保持原控制方式。
- 计算精度、空间大小和chunk布局保持已验证的BF16、256×256、4帧历史+8帧预测；当前不把这些架构限制伪装成可调超参数。

若修改start_frame后要复现旧长视频的某段，还需设置对应seed。例如原始seed=12345、stride=100000，从第16帧开始复现旧第3段时，首段seed应设212345。

## 结果

```text
output/
  resolved_config.yaml
  experiment.json
  job_status.json
  action_contract.json
  source_rgb.npy          # 本次episode的解码图像，供TF读取和核验
  initial_context.npy
  ground_truth.npy
  commands_real.npy / commands_zero.npy
  tf_real.npy / tf_zero.npy / ar_real.npy / ar_zero.npy  # 按所选组生成
  *_chunks.jsonl / *_runtime.json / *_status.json / *.log
  tf_comparison.mp4 / ar_comparison.mp4
  tf_chunks_slow.mp4
  metrics.json
  verification.json
  README.md
```

每段实际历史、动作、噪声与生成结果都有hash核验；同一组的模型参数逐张量核对checkpoint。视频完整解码核验帧数/帧率。TF分段慢放自动选取覆盖时域的最多6个chunk。

重新核验/导出已完成的run：

```bash
python examples/wanvideo/model_inference/eval.py \
  --config outputs/orca_delta_eval/resolved_config.yaml --render-only
```

此操作使用输出目录记录的配置，重写派生视频和报告，不重新推理；不会覆盖原始npy。

指标是RGB/255像素MSE，不是关节角度。真实录像只对应真实动作，没有全零的真实反事实录像；TF拼接视频每8帧接回真实历史，不能用于判断持续440帧自主静止。

## 实现与旧实验

采样和动作对齐在 [orca_rollout.py](orca_rollout.py)，核验与视频导出在 [orca_render.py](orca_render.py)。配置解析由 [orca_config.py](../orca_config.py) 共享。启动只需 `eval.py`。

旧评估代码原件仍保留在 `/home/yiwei/Documents/Codex/2026-09-18/wo/work/`，对应结果保留在 `/home/yiwei/workspace/wm-test/checkpoints/orca-delta-full-dataset/` 下的 `ep144_*` 目录。上一版临时 `scripts/orca/` 和固定ep144的仓库副本已被本入口替代；原实验数据与checkpoint不动。

训练入口见 [model_training/README_ORCA.md](../model_training/README_ORCA.md)。
