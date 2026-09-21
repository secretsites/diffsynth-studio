#!/usr/bin/env python3
"""Evaluate the final sampled-training checkpoint with the frozen 24-frame test."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
import train_orca_sampled as sampled
import medium_orca_action_probe as medium
import probe_orca_action_conditioning as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    config = json.loads((run / "config.json").read_text())
    state = json.loads((run / "status.json").read_text())
    if state["state"] != "completed" or state["global_step"] != config["total_steps"]:
        print("Training has not completed; final evaluation deferred.", flush=True)
        return
    out = run / "final_evaluation"
    out.mkdir(exist_ok=True)
    checkpoint = run / f"checkpoints/epoch-{config['rounds']:04d}.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    link = out / "dit-final.safetensors"
    if not link.is_symlink():
        link.symlink_to(checkpoint.resolve())
    assert link.resolve() == checkpoint.resolve()
    # Protocol and manifest were frozen before the first training update.
    medium.frozen(SimpleNamespace(output=out))
    status_path = out / "status.json"
    sampled.json_write(status_path, dict(state="running", updated_utc=sampled.utc()))
    os.environ["WAN_ACTION_TIME_ENCODING"] = "sinusoidal"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    evaluation_args = SimpleNamespace(dataset=Path(config["dataset"]), model=config["model"],
                        output=out, eval_sigmas=[1.], eval_seeds=[2026, 2027], cache_dir=out / "unused_cache")
    try:
        if not (out / "eval-final.json").exists():
            manifest = json.loads((out / "manifest.json").read_text())
            pipe = probe.load_pipe(evaluation_args)
            probe.configure_trainable(pipe)
            medium.load_checkpoint(pipe, checkpoint)
            pipe.dit.requires_grad_(False).eval()
            pipe.vae.to("cuda")
            windows = sampled.Windows(evaluation_args.dataset)
            samples = []
            for row in manifest["rows"]:
                rgb, action, hold = windows.get("val_data", row["episode"], row["start"] + 3)
                samples.append(dict(z=sampled.encode(pipe, rgb).cpu(),
                                    action=torch.from_numpy(action), hold=torch.from_numpy(hold)))
            probe.evaluate(evaluation_args, pipe, samples, manifest, "final")
            del pipe, samples, windows
            gc.collect(); torch.cuda.empty_cache()
        medium.run_rollout(evaluation_args)
        medium.summarize(evaluation_args)
        gates = json.loads((out / "gate_result.json").read_text())
        lines = ["# ORCA 1000 个采样轮训练结果\n",
                 f"完成 {state['global_step']:,} 次全参数更新；每轮 136 个训练 episode 各随机抽取 2 个完整历史窗口，启用动作时间编码，无 LoRA。\n",
                 "| 保存轮数 | 更新步数 | 检查点 |\n|---:|---:|---|\n"]
        for epoch in config["save_rounds"]:
            path = run / f"checkpoints/epoch-{epoch:04d}.safetensors"
            lines.append(f"| {epoch} | {epoch * config['steps_per_round']:,} | [{path.name}]({path}) |\n")
        lines += ["\n最终验证沿用预先冻结的 15 条验证轨迹、每条 2 个窗口、2 个噪声种子、3 个自回归 chunk（24 帧）。\n",
                  "| 动作对照 | sigma=1 去噪误差增幅 | 24 帧运动区域误差增幅 | rollout CI 下界 |\n|---|---:|---:|---:|\n"]
        for variant, values in gates["controls"].items():
            f, m = values["flow_sigma1"], values["rollout_motion"]
            lines.append(f"| {variant} | {100*f['relative_delta']:.2f}% | {100*m['relative_delta']:.2f}% | {m['ci95'][0]:.8f} |\n")
        lines.append(f"\n数值门槛通过：{gates['quantitative_pass']}。可视审查：{gates['visual_review'].get('status', 'recorded')}。未经实际画面复核，不宣布可靠动作控制通过。\n")
        (run / "training_report.md").write_text("".join(lines))
        sampled.json_write(status_path, dict(state="completed", updated_utc=sampled.utc(),
                          quantitative_pass=gates["quantitative_pass"], visual_review=gates["visual_review"]))
    except Exception as error:
        sampled.json_write(status_path, dict(state="failed", updated_utc=sampled.utc(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
