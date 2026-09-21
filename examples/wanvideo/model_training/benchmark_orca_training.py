"""Bounded full-parameter throughput benchmark; never saves model weights.

Uses real ORCA train windows and the same optimizer/flow objective as the probe.
The cached and online-VAE estimates apply to this memory-conscious probe setup,
not to arbitrary Accelerate, FP32 optimizer, batch-size or horizon settings.
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import probe_orca_action_conditioning as probe
from diffsynth.trainers.dataset import RLinfDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(43)
    dataset = RLinfDataset(str(args.dataset / "train_data"), action_dim=58,
                           Ta=8, To=4, stride=1, max_finish_step=0,
                           action2obs_bias=False, retain_actions=True)
    # Shape-only scan: include all train episodes and keep validation excluded.
    counts = {}
    for stride in (1, 4, 8):
        counts[str(stride)] = sum(len(range(0, end - 8 + 1, stride)) * n
                                  for _, end, _, n in dataset.episode_info)
    picked = np.linspace(0, len(dataset) - 1, 16, dtype=int)
    pipe = probe.load_pipe(SimpleNamespace(model=str(args.model)))
    assert pipe.dit.action_time_encoding == "sinusoidal"

    @torch.no_grad()
    def encode(index):
        with contextlib.redirect_stdout(io.StringIO()):
            data = dataset[int(index)]
        rgb = np.stack([np.asarray(frame) for frame in data["video"]])
        v = torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].to("cuda", torch.bfloat16) / 127.5 - 1
        z = pipe.vae.encode(v, device="cuda")
        return z, data["action"].to("cuda")

    started = time.monotonic()
    cache = []
    for index in picked:
        z, action = encode(index)
        cache.append((z.cpu(), action.cpu()))
    torch.cuda.synchronize()
    cache_seconds = time.monotonic() - started
    trainable = probe.configure_trainable(pipe)
    params = [p for p in pipe.dit.parameters() if p.requires_grad]
    action_params = [p for n, p in pipe.dit.named_parameters() if n.startswith("action_mlp")]
    action_ids = {id(p) for p in action_params}
    optimizer = torch.optim.AdamW([
        {"params": action_params, "lr": 1e-4},
        {"params": [p for p in params if id(p) not in action_ids], "lr": 1e-5},
    ], weight_decay=.01, foreach=False)
    rng = np.random.default_rng(43)
    pipe.dit.train()
    runs = {}
    gradient_checked = False
    for mode in ("cached", "online_vae"):
        if mode == "online_vae":
            pipe.vae.to("cuda")
        timings = []
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize()
            start = time.monotonic()
            if mode == "cached":
                z, action = (x.to("cuda") for x in cache[step % len(cache)])
            else:
                z, action = encode(picked[step % len(picked)])
            u = float(rng.uniform(.001, 1))
            sigma = 5 * u / (1 + 4 * u)
            x, target = probe.noisy_input(z, sigma, int(rng.integers(0, 2**31 - 1)))
            optimizer.zero_grad(set_to_none=True)
            pred = probe.predict(pipe, x, action, sigma, checkpoint=True)
            loss = (pred[:, :, 2:].float() - target[:, :, 2:].float()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite benchmark loss")
            loss.backward()
            if not gradient_checked:
                missing = [n for n, p in pipe.dit.named_parameters() if p.requires_grad and p.grad is None]
                assert not missing, missing
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            if step >= args.warmup:
                timings.append(elapsed)
            if step % 25 == 0:
                print(f"BENCH {mode} {step + 1}/{args.warmup + args.steps} seconds={elapsed:.4f}", flush=True)
        runs[mode] = dict(steps=len(timings), mean_seconds=float(np.mean(timings)),
                          median_seconds=float(np.median(timings)),
                          p10_p90_seconds=np.quantile(timings, [.1, .9]).tolist(),
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        print("MEASURED", mode, json.dumps(runs[mode]), flush=True)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  train_episodes=len(dataset.episode_info),
                  train_frames=sum(t * n for _, _, t, n in dataset.episode_info),
                  windows_per_epoch_by_stride=counts,
                  trainable_parameters=trainable, action_time_encoding=pipe.dit.action_time_encoding,
                  frame_shape=[13, 256, 256], future_frames=8, batch_size=1,
                  gradient_accumulation=1, all_trainable_parameters_received_gradients=gradient_checked,
                  optimizer="AdamW foreach=False; BF16 backbone/moments, FP32 action MLP/moments; no FP32 master weights",
                  cache_preparation_seconds_per_window=cache_seconds / len(cache),
                  runs=runs, checkpoint_saved=False,
                  limitations="16 stratified real windows reused for throughput; not a quality experiment. Online VAE benchmark includes loader and encoder with warm filesystem cache. No validation, disk checkpoint, FP32-master/offload or long-horizon costs measured.")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
