#!/usr/bin/env python3
"""Full-parameter ORCA training with two fresh windows per episode per round.

No LoRA; 5 observed RGB frames, 8 predicted frames, aligned 58D commands.
Milestone weights are immutable training deliverables; latest.pt is a rolling
model + optimizer recovery point. Sampling and diffusion noise are keyed by
round/update, so restart does not silently switch to a new sampling schedule.
"""
import argparse
import collections
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import probe_orca_action_conditioning as probe


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", buffering=1) as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def discover(dataset):
    info = json.loads((dataset / "dataset_info.json").read_text())
    stats = json.loads((dataset / "action_stats.json").read_text())
    assert info["action_dim"] == stats["action_dim"] == 58
    assert stats["frame_alignment"] == "actions[0]=state[0]; actions[t]=source_action[t-1] for t>0"
    assert not set(info["train_episodes"]) & set(info["val_episodes"])
    records = []
    for ep in sorted(info["train_episodes"]):
        path = probe.episode_path(dataset, "train_data", ep)
        rgb = np.load(path / "rgb.npy", mmap_mode="r")
        action = np.load(path / "actions.npy", mmap_mode="r")
        assert rgb.shape[1:] == (1, 256, 256, 3) and rgb.dtype == np.uint8
        assert action.shape == (len(rgb), 1, 58) and action.dtype == np.float32
        assert len(rgb) >= 13
        files = {name: {"bytes": (path / name).stat().st_size,
                        "mtime_ns": (path / name).stat().st_mtime_ns}
                 for name in ("rgb.npy", "actions.npy", "states.npy")}
        records.append(dict(episode=int(ep), frames=len(rgb), files=files))
    return records, info


def sample_round(episodes, round_id, seed, windows_per_episode=2):
    """Uniform starts with four actual history frames; distinct within episode."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 701]))
    plan = []
    for ep in episodes:
        # current t has context t-3:t+1 and targets t+1:t+9.
        starts = rng.choice(np.arange(3, ep["frames"] - 8), windows_per_episode, replace=False)
        plan.extend(dict(episode=ep["episode"], current_frame=int(t)) for t in starts)
    rng.shuffle(plan)
    return plan


def noise_settings(seed, step):
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 1701]))
    u = float(rng.uniform(.001, 1))
    return 5 * u / (1 + 4 * u), int(rng.integers(0, 2**31 - 1))


class Windows:
    def __init__(self, dataset):
        self.dataset = dataset
        self.arrays = {}

    def get(self, split, episode, current_frame):
        key = (split, episode)
        if key not in self.arrays:
            p = probe.episode_path(self.dataset, split, episode)
            self.arrays[key] = tuple(np.load(p / name, mmap_mode="r")
                                     for name in ("rgb.npy", "actions.npy", "states.npy"))
        rgb, action, state = self.arrays[key]
        t = current_frame
        if not 3 <= t < len(rgb) - 8:
            raise ValueError(f"Invalid complete-context window: {key}, {t}")
        ids = [0] + list(range(t - 3, t + 9))
        a = action[ids, 0].copy()
        if not np.isfinite(a).all() or np.abs(a).max() > 1.0001:
            raise ValueError(f"Invalid action values: {key}, {t}")
        return rgb[ids, 0].copy(), a, state[t, 0].copy()


@torch.no_grad()
def encode(pipe, rgb):
    v = torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].to("cuda", torch.bfloat16) / 127.5 - 1
    z = pipe.vae.encode(v, device="cuda")
    assert tuple(z.shape) == (1, 48, 4, 16, 16) and torch.isfinite(z).all()
    return z


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_resume(path, model, optimizer, progress, config_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(version=1, config_sha256=config_sha256, progress=progress,
                 model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    json_write(path.with_suffix(".json"), dict(config_sha256=config_sha256,
               progress=progress, bytes=path.stat().st_size, updated_utc=utc()))
    del state
    gc.collect()


def restore_resume(path, model, optimizer, config_sha256):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config_sha256"] != config_sha256:
        raise ValueError("Resume configuration differs from this run")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    progress = state["progress"]
    del state
    gc.collect()
    return progress


def save_weights(path, model, round_id, step, config_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()}
    metadata = dict(action_time_encoding=model.action_time_encoding,
                    action_dim="58", round=str(round_id), global_step=str(step),
                    config_sha256=config_sha256)
    temporary = path.with_suffix(".safetensors.tmp")
    save_file(state, str(temporary), metadata=metadata)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        assert set(saved.keys()) == set(state)
        assert saved.metadata() == metadata
    json_write(path.with_suffix(".json"), dict(**metadata, bytes=path.stat().st_size,
               sha256=digest(path), saved_utc=utc(), trainable_parameters=sum(p.numel() for p in model.parameters())))
    del state
    gc.collect()


def trim_log(path, key, maximum):
    """Remove work beyond the last durable resume point after a crash."""
    path = Path(path)
    if not path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with path.open() as source, temporary.open("w") as dest:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row[key] <= maximum:
                dest.write(json.dumps(row) + "\n")
    temporary.replace(path)


def paired_summary(records):
    real = {(r["id"], r["seed"]): r["mse"] for r in records if r["variant"] == "real"}
    baseline = float(np.mean(list(real.values())))
    result = {"real_mse": baseline, "controls": {}}
    for variant in ("hold", "reverse", "delta_swap"):
        by_episode = collections.defaultdict(list)
        for r in records:
            if r["variant"] == variant:
                by_episode[r["episode"]].append(r["mse"] - real[r["id"], r["seed"]])
        values = np.array([np.mean(v) for _, v in sorted(by_episode.items())])
        rng = np.random.default_rng(20260918)
        boots = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        ci = np.quantile(boots, [.025, .975]).tolist()
        relative = float(values.mean() / baseline)
        wins = float((values > 0).mean())
        result["controls"][variant] = dict(relative_delta=relative, ci95=ci, episode_win_rate=wins,
                    pass_flow=bool(ci[0] > 0 and relative >= .02 and wins >= .60))
    return result


@torch.no_grad()
def validate(pipe, samples, manifest, path, step):
    pipe.dit.eval()
    records = []
    for i, row in enumerate(manifest["rows"]):
        sample = samples[i]
        z, action, hold = [sample[k].to("cuda") for k in ("z", "action", "hold")]
        donor = samples[row["donor_index"]]["action"].to("cuda")
        for seed in (2026, 2027):
            x, target = probe.noisy_input(z, 1., seed)
            for variant in ("real", "hold", "reverse", "delta_swap"):
                changed = probe.intervention(action, hold, donor, variant)
                pred = probe.predict(pipe, x, changed, 1.)[:, :, 2:].float()
                mse = float((pred - target[:, :, 2:].float()).square().mean())
                if not math.isfinite(mse):
                    raise ValueError("Non-finite validation result")
                records.append(dict(id=row["id"], episode=row["episode"], split="val_data",
                                    sigma=1., seed=seed, variant=variant, mse=mse))
    summary = paired_summary(records)
    json_write(path, dict(step=step, updated_utc=utc(), summary=summary, records=records))
    print("VALIDATION", step, json.dumps(summary), flush=True)
    pipe.dit.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--save-rounds", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--resume-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--action-lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.windows_per_episode, args.resume_every, args.validate_every) < 1:
        parser.error("rounds/windows/intervals must be positive")
    if any(r < 1 or r > args.rounds for r in args.save_rounds):
        parser.error("save rounds must lie within training")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    os.environ["WAN_ACTION_TIME_ENCODING"] = "sinusoidal"
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    episodes, info = discover(args.dataset)
    validation_manifest = json.loads(args.validation_manifest.read_text())
    assert all(r["split"] == "val_data" and r["episode"] in info["val_episodes"]
               for r in validation_manifest["rows"])
    steps_per_round = len(episodes) * args.windows_per_episode
    total_steps = steps_per_round * args.rounds
    config = dict(version=1, dataset=str(args.dataset.resolve()), model=str(args.model.resolve()),
                  episodes=episodes, action_stats_sha256=digest(args.dataset / "action_stats.json"),
                  validation_manifest_sha256=digest(args.validation_manifest),
                  rounds=args.rounds, save_rounds=sorted(set(args.save_rounds)),
                  steps_per_round=steps_per_round, total_steps=total_steps,
                  windows_per_episode=args.windows_per_episode, seed=args.seed,
                  sampling="Per episode uniform current_frame in [3,T-9], no replacement within round; reshuffle/resample every round",
                  action_time_encoding="sinusoidal", future_frames=8, history_frames=4, reference_frames=1,
                  batch_size=1, gradient_accumulation=1, action_dim=58, action2obs_bias=False,
                  lr=args.lr, action_lr=args.action_lr, warmup_steps=steps_per_round,
                  learning_rate_schedule="One sampling-round linear warmup, then constant",
                  optimizer="AdamW foreach=False, weight_decay=.01; BF16 backbone+moments, FP32 action MLP+moments; no FP32 master weights",
                  grad_clip=1., resume_every=args.resume_every, validate_every=args.validate_every,
                  vae="frozen, online encoding", trainable="full DiT and both action MLPs, no LoRA")
    config_path = args.output / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Existing run requires --resume with identical configuration")
    else:
        json_write(config_path, config)
    config_sha = digest(config_path)
    progress = dict(global_step=0, completed_rounds=0, round_loss_sum=0., round_loss_count=0,
                    training_seconds=0., started_utc=utc())
    latest = args.output / "resume/latest.pt"
    status_base = dict(pid=os.getpid(), target_rounds=args.rounds, total_steps=total_steps,
                       steps_per_round=steps_per_round, config_sha256=config_sha)

    def status(state, **extra):
        done = progress["global_step"]
        rate = progress["training_seconds"] / done if done else None
        json_write(args.output / "status.json", dict(**status_base, state=state,
                   updated_utc=utc(), **progress, seconds_per_update=rate,
                   remaining_training_hours=(total_steps - done) * rate / 3600 if rate else None, **extra))

    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
        print("STOP_REQUESTED", signum, flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        status("loading")
        pipe = probe.load_pipe(args)
        probe.configure_trainable(pipe)
        pipe.vae.to("cuda").eval()
        assert pipe.dit.action_time_encoding == "sinusoidal"
        params = list(pipe.dit.parameters())
        assert all(p.requires_grad for p in params)
        action_params = [p for n, p in pipe.dit.named_parameters() if n.startswith("action_mlp")]
        action_ids = {id(p) for p in action_params}
        optimizer = torch.optim.AdamW([
            {"params": action_params, "lr": args.action_lr},
            {"params": [p for p in params if id(p) not in action_ids], "lr": args.lr},
        ], weight_decay=.01, foreach=False)
        if latest.exists():
            if not args.resume:
                raise ValueError("Resume checkpoint exists")
            progress = restore_resume(latest, pipe.dit, optimizer, config_sha)
            print("RESUMED", json.dumps(progress), flush=True)
        elif progress["global_step"] == 0:
            weight_bytes = sum(p.numel() * p.element_size() for p in params)
            required = (len(config["save_rounds"]) + 6) * weight_bytes + 5 * 2**30
            if shutil.disk_usage(args.output).free < required:
                raise RuntimeError(f"Insufficient disk for milestones + atomic rolling resume: need {required} bytes")
        trim_log(args.output / "train.jsonl", "step", progress["global_step"])
        trim_log(args.output / "rounds.jsonl", "round", progress["completed_rounds"])
        trim_log(args.output / "sampling.jsonl", "round", progress["completed_rounds"])
        windows = Windows(args.dataset)
        validation = []
        status("preparing_validation")
        for row in validation_manifest["rows"]:
            rgb, action, hold = windows.get("val_data", row["episode"], row["start"] + 3)
            validation.append(dict(z=encode(pipe, rgb).cpu(), action=torch.from_numpy(action), hold=torch.from_numpy(hold)))
        if progress["global_step"] == 0:
            validate(pipe, validation, validation_manifest, args.output / "validation/initial.json", 0)
        elif progress["global_step"] % steps_per_round == 0:
            saved_round = progress["completed_rounds"]
            validation_path = args.output / f"validation/epoch-{saved_round:04d}.json"
            if (saved_round % args.validate_every == 0 or saved_round in config["save_rounds"] or saved_round == args.rounds) and not validation_path.exists():
                validate(pipe, validation, validation_manifest, validation_path, progress["global_step"])
        gradient_checked = False
        causal_checked = False
        pipe.dit.train()
        while progress["global_step"] < total_steps:
            round_id = progress["global_step"] // steps_per_round + 1
            offset = progress["global_step"] % steps_per_round
            plan = sample_round(episodes, round_id, args.seed, args.windows_per_episode)
            append_json(args.output / "sampling.jsonl", dict(round=round_id, windows=plan,
                sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()))
            for position in range(offset, steps_per_round):
                row = plan[position]
                step = progress["global_step"] + 1
                torch.cuda.synchronize()
                started = time.monotonic()
                rgb, action, _ = windows.get("train_data", row["episode"], row["current_frame"])
                z = encode(pipe, rgb)
                if not causal_checked:
                    with torch.no_grad():
                        v = torch.from_numpy(rgb[:5]).permute(3, 0, 1, 2)[None].to("cuda", torch.bfloat16) / 127.5 - 1
                        prefix = pipe.vae.encode(v, device="cuda")
                        error = float((prefix.float() - z[:, :, :2].float()).abs().max())
                        assert error < .03, error
                    json_write(args.output / "causal_prefix_audit.json", dict(max_error=error, passed=True))
                    causal_checked = True
                action = torch.from_numpy(action).to("cuda")
                sigma, noise_seed = noise_settings(args.seed, step)
                x, target = probe.noisy_input(z, sigma, noise_seed)
                warmup = min(1., step / steps_per_round)
                optimizer.param_groups[0]["lr"] = args.action_lr * warmup
                optimizer.param_groups[1]["lr"] = args.lr * warmup
                optimizer.zero_grad(set_to_none=True)
                pred = probe.predict(pipe, x, action, sigma, checkpoint=True)
                loss = (pred[:, :, 2:].float() - target[:, :, 2:].float()).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {step}")
                loss.backward()
                if not gradient_checked:
                    missing = [n for n, p in pipe.dit.named_parameters() if p.grad is None]
                    assert not missing, missing
                    json_write(args.output / "gradient_audit.json", dict(step=step,
                        trainable_parameters=sum(p.numel() for p in params), missing_gradients=missing,
                        action_time_encoding=pipe.dit.action_time_encoding))
                    gradient_checked = True
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                value = float(loss.detach())
                progress["global_step"] = step
                progress["training_seconds"] += elapsed
                progress["round_loss_sum"] += value
                progress["round_loss_count"] += 1
                record = dict(step=step, round=round_id, position=position + 1, **row,
                    sigma=sigma, noise_seed=noise_seed, loss=value, grad_norm=float(norm),
                    seconds=elapsed, backbone_lr=optimizer.param_groups[1]["lr"],
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                append_json(args.output / "train.jsonl", record)
                if step == 1 or step % 25 == 0:
                    status("training")
                    print("TRAIN", json.dumps(record), flush=True)
                if stop["requested"]:
                    break
            completed = progress["global_step"] % steps_per_round == 0
            if completed:
                progress["completed_rounds"] = round_id
                append_json(args.output / "rounds.jsonl", dict(round=round_id,
                    step=progress["global_step"], mean_loss=progress["round_loss_sum"] / progress["round_loss_count"],
                    updated_utc=utc()))
                progress["round_loss_sum"] = 0.; progress["round_loss_count"] = 0
            optimizer.zero_grad(set_to_none=True)
            milestone = completed and round_id in config["save_rounds"]
            if milestone:
                status("saving_checkpoint")
                save_weights(args.output / f"checkpoints/epoch-{round_id:04d}.safetensors",
                             pipe.dit, round_id, progress["global_step"], config_sha)
            if stop["requested"] or (completed and (round_id == 1 or round_id % args.resume_every == 0 or milestone or round_id == args.rounds)):
                status("saving_resume")
                save_resume(latest, pipe.dit, optimizer, progress, config_sha)
            if stop["requested"]:
                status("stopped", reason="Signal received; resumable state saved")
                return
            if completed and (round_id % args.validate_every == 0 or milestone or round_id == args.rounds):
                status("validating")
                validate(pipe, validation, validation_manifest,
                         args.output / f"validation/epoch-{round_id:04d}.json", progress["global_step"])
            status("training")
        status("completed")
        print("TRAINING_COMPLETE", json.dumps(progress), flush=True)
    except Exception as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
