"""Native Wan 5B ORCA abs/delta/relative rollout. Sampling matches the audited ep144 experiments."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from orca_config import ROOT, save_config, write_json

CHUNK = 8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def history_indices(t):
    return [0] + [max(0, f) for f in range(t - 3, t + 1)]


def action_window(actions, t):
    """Reference zero plus history/future actions, applying action2obs_bias exactly once."""
    result = np.zeros((13, 58), np.float32)
    for slot, frame in enumerate(range(t - 3, t + 9), start=1):
        if frame > 0: result[slot] = actions[frame - 1]
    return result


def frame_count(total, start, requested):
    available = total - 1 - start
    count = (available // CHUNK) * CHUNK if requested is None else requested
    if count < CHUNK or count % CHUNK or count > available:
        raise ValueError(f'Requested {count} predictions from frame {start}, but only {available} available; use complete 8-frame chunks')
    return count


def check_action_metadata(metadata, mode, contract=None):
    from diffsynth.trainers.orca_online import MODES
    # Older audited epoch-8 exports have representation but no explicit mode/normalization.
    declared = metadata.get('action_mode')
    representation = metadata.get('action_representation')
    if declared is None and representation is None:
        raise ValueError('Checkpoint has no action representation metadata; cannot infer action units from tensor shape')
    if declared is not None and declared != mode or representation is not None and representation != MODES[mode]:
        raise ValueError(f'Checkpoint action mode/representation differs from dataset.action_mode={mode}')
    if contract is not None and metadata.get('action_contract'):
        recorded = json.loads(metadata['action_contract'])
        if recorded and recorded != contract: raise ValueError('Checkpoint action normalization/anchor differs from dataset contract')


def load_episode(config):
    """Use the training loader for both original LeRobot and previously prepared NPY data."""
    from diffsynth.trainers.orca_online import build_dataset, action_contract_for, fingerprint
    options, roll = config['dataset'], config['rollout']
    split, task, name = Path(roll['episode']).parts
    if split not in ('train_data', 'val_data'): raise ValueError('Episode split must be train_data or val_data')
    ds = build_dataset(options, split)
    env = roll['environment_index']
    if options['format'] == 'lerobot':
        if task != 'orca' or not name.startswith('episode_') or not name[8:].isdigit() or env != 0:
            raise ValueError('Raw ORCA uses split/orca/episode_NNNNNN and environment_index=0')
        episode = int(name[8:])
        if episode not in ds.ids: raise ValueError(f'Episode {episode} is not in configured {split}')
        ep_id = ds.ids.index(episode)
        rgb = ds.reader.frames(episode)
        provenance = dict(format='lerobot', source_contract_sha256=fingerprint(ds.contract))
        source = str(ds.reader.root)
    else:
        ep = Path(options['path']) / roll['episode']
        full = np.load(ep / 'rgb.npy', mmap_mode='r')
        actions = np.load(ep / 'actions.npy', mmap_mode='r')
        if full.ndim != 5 or env >= full.shape[1] or actions.shape != (*full.shape[:2], 58) or not np.isfinite(actions).all():
            raise ValueError('Environment index or action array does not match RGB')
        rgb = full[:, env]
        ep_id = next(i for i, e in enumerate(ds.episode_info) if Path(e[0]).resolve() == ep.resolve())
        provenance = dict(format='npy', rgb_sha256=digest(ep / 'rgb.npy'), actions_sha256=digest(ep / 'actions.npy'))
        source = str(ep)
    if rgb.ndim != 4 or rgb.shape[1:] != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError('This native recipe requires RGB [T,256,256,3] uint8')
    n = frame_count(len(rgb), roll['start_frame'], roll['frames'])
    indices = {t: i for i, (e, a, t) in enumerate(ds.sample_indices) if e == ep_id and a == env}
    commands = []
    for t in range(roll['start_frame'], roll['start_frame'] + n, CHUNK):
        if t not in indices: raise ValueError(f'No complete native window at frame {t}')
        sample = ds[indices[t]]
        command = sample['action'].numpy()
        if options['format'] == 'npy' and not np.array_equal(command, action_window(actions[:, env], t)):
            raise ValueError(f'Native action alignment mismatch at {t}')
        if not np.array_equal(np.stack([np.asarray(f) for f in sample['video'][:5]]), rgb[history_indices(t)]):
            raise ValueError(f'Native image history mismatch at {t}')
        commands.append(command)
    return rgb, np.stack(commands), action_contract_for(options, getattr(ds, 'reader', None)), source, provenance


def prepare(config, write_outputs=True):
    from safetensors import safe_open
    out, roll = Path(config['output']), config['rollout']
    if write_outputs and out.exists() and any(out.iterdir()): raise FileExistsError(f'Refusing to overwrite {out}')
    checkpoint = Path(config['model']['checkpoint'])
    mode = config['dataset']['action_mode']
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        if f.get_slice('action_mlp1.0.weight').get_shape() != [3072, 58] or f.get_slice('action_mlp2.0.weight').get_shape() != [12288, 232]:
            raise ValueError('Checkpoint is not a native 58D Wan TI2V 5B action model')
        metadata = f.metadata() or {}
    check_action_metadata(metadata, mode)
    rgb, commands, contract, source, provenance = load_episode(config)
    check_action_metadata(metadata, mode, contract)
    n = len(commands) * CHUNK
    sha = digest(checkpoint)
    if config['model']['checkpoint_sha256'] is not None and sha != config['model']['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    base = Path(config['model']['base_path'])
    if not (base / 'Wan2.2_VAE.pth').is_file() or not (base / 'google/umt5-xxl').is_dir():
        raise FileNotFoundError('base_path must contain Wan2.2_VAE.pth and google/umt5-xxl')
    plan = dict(source=source, source_provenance=provenance, source_rgb_array_sha256=ah(rgb),
        checkpoint=str(checkpoint), checkpoint_sha256=sha, checkpoint_metadata=metadata,
        start_frame=roll['start_frame'], predicted_frames=n, total_source_frames=len(rgb), chunks=n // CHUNK,
        frames_per_chunk=CHUNK, environment_index=roll['environment_index'], modes=roll['modes'], actions=roll['actions'],
        fps=config['render']['fps'], inference_steps=config['sampling']['steps'], sigma_shift=config['sampling']['sigma_shift'],
        chunk_seeds=[config['sampling']['seed'] + config['sampling']['seed_stride'] * k for k in range(n // CHUNK)],
        action_mode=mode, action_representation=contract['representation'], action_contract=contract,
        action_dim=58, action2obs_bias_applied_once=True,
        all_windows_equal_native_loader=True, zero_policy='All 13 x 58 normalized action slots are zero, including history. In abs mode this is the normalization center, not a hold command.',
        noise_policy='Same raw latent noise for all variants of each chunk.',
        action_state_policy='Recorded measured states: delta uses each command timestamp; relative uses each chunk prediction-start state. AR does not estimate state from generated images.',
        history_policy='Fixed reference frame 0. TF uses true t-3..t every chunk; AR uses its own last four predictions after the initial real history.',
        command_array_sha256=ah(commands), runner_sha256=digest(__file__))
    if write_outputs:
        out.mkdir(parents=True, exist_ok=True)
        save_config(out / 'resolved_config.yaml', config)
        write_json(out / 'action_contract.json', contract)
        np.save(out / 'commands_real.npy', commands)
        np.save(out / 'commands_zero.npy', np.zeros_like(commands))
        np.save(out / 'source_rgb.npy', rgb)
        np.save(out / 'initial_context.npy', rgb[history_indices(roll['start_frame'])])
        np.save(out / 'ground_truth.npy', rgb[roll['start_frame'] + 1:roll['start_frame'] + n + 1])
        write_json(out / 'experiment.json', plan)
    return plan


def worker(config, mode, variant, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ['WAN_ACTION_DIM'] = '58'
    os.environ.pop('WAN_ACTION_TIME_ENCODING', None)
    import torch
    from safetensors import safe_open
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
    from diffsynth.schedulers.flow_match import FlowMatchScheduler
    out = Path(config['output']); tag = f'{mode}_{variant}'
    plan = json.loads((out / 'experiment.json').read_text())
    n, start, env = plan['predicted_frames'], plan['start_frame'], plan['environment_index']
    started = time.monotonic()
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config['sampling']['seed'])
    write_json(out / f'{tag}_status.json', dict(state='loading', pid=os.getpid(), device=device))
    base, ckpt = Path(config['model']['base_path']), config['model']['checkpoint']
    with safe_open(ckpt, framework='pt', device='cpu') as source:
        metadata = source.metadata() or {}
    is_lora = metadata.get('checkpoint_format') == 'orca_lora_adapter'
    dit_path = ([str(base / f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors') for i in range(1, 4)]
                if is_lora else ckpt)
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cpu',
        model_configs=[ModelConfig(path=dit_path), ModelConfig(path=str(base / 'Wan2.2_VAE.pth'))],
        tokenizer_config=ModelConfig(path=str(base / 'google/umt5-xxl')))
    if is_lora:
        from safetensors.torch import load_file
        from diffsynth.trainers.orca_lora import configure_lora, load_adapter_state, upcast_trainable_parameters
        adapter_config = json.loads(metadata['lora_config'])
        pipe.dit = configure_lora(pipe.dit, **adapter_config)
        if metadata.get('trainable_precision') == 'float32': upcast_trainable_parameters(pipe.dit)
        load_adapter_state(pipe.dit, load_file(ckpt))
    with safe_open(ckpt, framework='pt', device='cpu') as source:
        state = pipe.dit.state_dict()
        if not is_lora: assert set(state) == set(source.keys())
        for name in source.keys(): assert torch.equal(state[name], source.get_tensor(name)), name
    assert not hasattr(pipe.dit, 'action_time_encoding')
    assert pipe.dit.action_mode == 'both' and pipe.dit.five_frame_condition
    assert pipe.dit.action_mlp1[0].in_features == 58 and pipe.dit.action_mlp2[0].in_features == 232
    del state
    pipe.dit.requires_grad_(False).eval().to(device='cuda', dtype=torch.bfloat16)
    pipe.vae.requires_grad_(False).eval().to('cuda')
    pipe.device = torch.device('cuda')
    write_json(out / f'{tag}_runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), physical_gpu=device, actual_checkpoint_weights_exact=True,
        dtype='bfloat16', action_mode=pipe.dit.action_mode, action_time_encoding='none/native'))
    commands = np.load(out / f'commands_{variant}.npy')
    initial = np.load(out / 'initial_context.npy'); ref = initial[0].copy()
    if variant == 'zero': assert not np.count_nonzero(commands)
    rgb = np.load(out / 'source_rgb.npy', mmap_mode='r') if mode == 'tf' else None
    generated = np.lib.format.open_memmap(out / f'{tag}.npy', mode='w+', dtype=np.uint8, shape=(n, 256, 256, 3))
    context = initial.copy()
    with torch.inference_mode():
        for k in range(n // CHUNK):
            offset, t, tick = k * CHUNK, start + k * CHUNK, time.monotonic()
            if mode == 'tf': context = rgb[history_indices(t)].copy()
            elif k: context = np.concatenate([ref[None], generated[offset - 4:offset]])
            if not k: assert np.array_equal(context, initial)
            video = torch.from_numpy(context).permute(3, 0, 1, 2)[None].to('cuda', torch.bfloat16) / 127.5 - 1
            prefix = pipe.vae.encode(video, device='cuda')
            assert tuple(prefix.shape) == (1, 48, 2, 16, 16)
            scheduler = FlowMatchScheduler(shift=config['sampling']['sigma_shift'], sigma_min=0, extra_one_step=True)
            scheduler.set_timesteps(config['sampling']['steps'])
            x = torch.randn((1, 48, 4, 16, 16), generator=torch.Generator(device='cuda').manual_seed(plan['chunk_seeds'][k]),
                            device='cuda', dtype=torch.bfloat16)
            noise_hash = ah(x.cpu().view(torch.uint8).numpy())
            x[:, :, :2] = prefix
            action = torch.tensor(commands[k], device='cuda')
            assert ah(action.cpu().numpy()) == ah(commands[k])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for timestep in scheduler.timesteps:
                    prediction = pipe.model_fn(dit=pipe.dit, latents=x, action=action,
                        timestep=timestep.reshape(1).to('cuda', torch.bfloat16),
                        fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False)
                    x = scheduler.step(prediction, timestep, x); x[:, :, :2] = prefix
            assert torch.isfinite(x).all()
            decoded = pipe.vae.decode(x, device='cuda')[0].float().permute(1, 2, 3, 0).cpu().numpy()
            assert np.isfinite(decoded).all()
            prediction = np.uint8(np.clip((decoded[5:] + 1) * 127.5, 0, 255))
            assert prediction.shape == (8, 256, 256, 3)
            generated[offset:offset + CHUNK] = prediction; generated.flush()
            record = dict(mode=mode, variant=variant, chunk=k + 1, current_frame=t, future_indices=list(range(t + 1, t + 9)),
                history_indices=history_indices(t)[1:], context_sha256=ah(context), noise_sha256=noise_hash,
                commands_sha256=ah(commands[k]), prediction_sha256=ah(prediction), seconds=time.monotonic() - tick)
            with (out / f'{tag}_chunks.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            write_json(out / f'{tag}_status.json', dict(state='generating', pid=os.getpid(), chunks_completed=k + 1,
                total_chunks=n // CHUNK, predicted_frames_completed=offset + CHUNK))
            if not k or (k + 1) % 5 == 0: print('PROGRESS', tag, k + 1, '/', n // CHUNK, flush=True)
    write_json(out / f'{tag}_status.json', dict(state='completed', chunks_completed=n // CHUNK,
        predicted_frames_completed=n, elapsed_seconds=time.monotonic() - started, output_sha256=digest(out / f'{tag}.npy')))
