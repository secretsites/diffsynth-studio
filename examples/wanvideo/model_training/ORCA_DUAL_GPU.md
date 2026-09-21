# ORCA delta：双卡完整滑窗训练

目标是从 `/home/yiwei/workspace/model/Wan2.2-TI2V-5B` 开始，对新 delta 数据做一次完整训练集遍历。无 LoRA，全部 DiT 参数可训练，VAE 冻结；保持主训练入口的扩散损失、两条动作条件分支、58D 动作、加载时一次 action2obs_bias 和 15% 零动作静态增强。

## 数据与轮次

- 训练：136 个完整 episode，63,516 帧，stride=1、Ta=8，共 **62,428 个有效窗口**。
- 验证：15 个独立 episode，6,786 帧，共 6,666 个窗口；不加入训练。
- 每个训练窗口在本轮恰好贡献一次，不按 episode 截断，不使用 501 个 batch 上限，不采用每 episode 抽两窗口的旧轮次定义。
- 全局 batch=128，每卡 micro batch=1，各累积64次再同步和更新。完整一轮 **488 次 optimizer update**；末组92个窗口，按实际数量归一化，保留全部样本。
- 不足双卡整组时使用零权重占位，使每个 rank 的调用次数一致；真实样本不重复计入损失。
- 每次实际更新的窗口 ID 写入 `processed_windows.rank*.jsonl`，训练结束独立核验无遗漏、无重复。恢复时重放的更新以最后一次记录为准。

## 启动

```bash
TRAIN_OUTPUT=/path/to/new/run \
  bash examples/wanvideo/model_training/full/run_orca_delta_dual.sh
```

默认2个GPU进程、BF16、fused AdamW、学习率1e-5、weight decay=0.01、关闭梯度检查点、无CPU offload、每个进程2个数据加载worker。环境变量可覆盖 `CUDA_VISIBLE_DEVICES`、`PYTHON_BIN`、`WAN_MODEL_PATH`、`WAN_DATASET_PATH`、`TRAIN_OUTPUT`、`GLOBAL_BATCH`、`DATA_WORKERS`；末尾CLI参数可覆盖具体选项。

真实训练仍使用原生 `WanTrainingModule`。这里的batch128是梯度累积的有效batch，不会将128个视频同时放进显存；也没有通过只取batch中的第一个样本伪造批处理。

## 验证、保存与恢复

- 初始模型先测固定120个验证窗口。
- 每100次参数更新，检查同一份验证窗口清单并保存可恢复状态；每次验证对相同窗口使用相同随机种子，静态增强关闭。
- 完整训练结束后再次检查固定验证清单，并遍历全部6,666个验证窗口。
- `resume_latest.pt` 保存全部 DiT 权重、AdamW 状态、配置和已完成更新数，原子替换上一份。
- 最终 `epoch-1.safetensors` 是可独立加载的完整58D DiT；VAE继续使用原始base的VAE文件。
- 源码副本、源码哈希、配置、样本顺序及数据覆盖清单随run保存。

```bash
TRAIN_OUTPUT=/path/to/existing/run \
  bash examples/wanvideo/model_training/full/run_orca_delta_dual.sh \
  --resume /path/to/existing/run/resume_latest.pt
```

`--stop-after-updates N` 仅用于验证保存/恢复或主动暂停：到绝对更新N后保存并标记paused，**不会把部分训练标记为完整epoch**。随后恢复时不传该参数即可完成剩余全部窗口。每个样本的随机数由seed、epoch和窗口ID决定，恢复不会重新采样顺序。

## 性能基准

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/wanvideo/model_training/train_orca_delta_distributed.py \
  --benchmark --global-batch 128 --no-gradient-checkpointing --output /path/to/new/benchmark
```

benchmark会进行真实前向、反向和AdamW更新，但不会保存或将测试权重用于正式训练。默认预热1次更新、测量3次，记录全局窗口/秒、更新耗时和完整优化器显存。正式训练重新从base初始化。

当前主板 MPG Z690 EDGE TI WIFI DDR4 的GPU0为Gen5×16、GPU1为芯片组Gen3×4，卡间NCCL约2GB/s，故小batch双卡反而慢。实测的一个候选为全局batch128、关闭重算：双卡约8.57窗口/秒、峰值分配约50.67GiB（结果是短基准，不是收敛速度保证）。应与同全局batch、同精度的单卡配置比较。

## 正确性边界

- 全参数更新范围不因DDP改变；DDP归约bucket共享梯度存储，使用no_sync实现累积。
- 每次optimizer.step后才清零梯度；最后一组按真实样本数缩放。
- 优化器保持现有BF16参数对应的状态精度，没有改成LoRA、8-bit优化器或FP8训练。
- 较大的全局batch意味着每轮更新次数较少；一次完整数据遍历不保证模型已经学会动作控制，生成质量须单独验证。
- 现有原生入口的梯度累积清零顺序也已修复；双卡完整遍历使用本文件入口，不受旧入口的每轮501batch限制。

## TensorBoard 实时曲线

已经运行的训练无需重启，可以将 JSONL 记录同步到 TensorBoard：

```bash
CUDA_VISIBLE_DEVICES='' python examples/wanvideo/model_training/tensorboard_orca_live.py --run /path/to/run
tensorboard --logdir /path/to/run/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

访问 `http://127.0.0.1:6006/#scalars`。主要标签为 `Loss/train`、`Loss/validation_fixed`、`Loss/validation_full`、`Performance/update_seconds`、`Performance/windows_per_second_20_updates`、`Progress/epoch_fraction`。GPU0显存标签记录PyTorch allocator峰值，未包含驱动或桌面显存。

同步程序先导入已有历史，再每5秒跟进新记录，训练终止后完成最后一次刷新并退出。横轴step是优化器更新数；事件wall time是日志导入时间，实际每步耗时看Performance标签。训练loss随窗口与扩散时刻变化，收敛比较优先使用固定验证清单。
