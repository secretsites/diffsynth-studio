#!/usr/bin/env python3
"""Full-window ORCA training and matched-batch single/DDP benchmarks.

Uses the native WanTrainingModule, loss, VAE, and augmentation, with full or LoRA
training. Each real window contributes exactly once per epoch. Padding
is zero-weighted and partial accumulation groups use their actual global size.
"""
import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from diffsynth.trainers.orca_online import build_dataset, MODES, fingerprint, action_contract_for
from diffsynth.trainers.orca_lora import (configure_lora, trainable_state_dict, load_adapter_state,
                                        check_training_mode, upcast_trainable_parameters)
from train_rlinf import WanTrainingModule


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sample_seed(seed, epoch, index):
    return (seed + 1_000_003 * epoch + 97 * index) % (2**32)


def seed_sample(seed, epoch, index):
    value = sample_seed(seed, epoch, index)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def make_order(length, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(length).astype(np.int64)


def shard_order(order, world_size, rank):
    positions = list(range(rank, math.ceil(len(order) / world_size) * world_size, world_size))
    return [(int(order[p]) if p < len(order) else int(order[0]), p < len(order)) for p in positions]


def group_real_count(total, update, accumulation, world_size):
    return min(accumulation * world_size, total - update * accumulation * world_size)


class IndexedWindows(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original, valid = self.indices[index]
        return original, valid, self.dataset[original]


def identity(item):
    return item


def create_loader(dataset, indices, workers, micro_batch_size=1):
    options = dict(batch_size=micro_batch_size, num_workers=workers, collate_fn=identity)
    if workers:
        # Avoid forking an initialized CUDA context. Workers receive only the dataset.
        options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    return DataLoader(IndexedWindows(dataset, indices), **options)


def synchronize():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def memory_record(device):
    return {'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30}


def snapshot_sources(output):
    files = ['examples/wanvideo/model_training/train_orca.py',
             'examples/wanvideo/model_training/train_orca_delta_distributed.py',
             'examples/wanvideo/model_training/train_rlinf.py',
             'diffsynth/trainers/dataset.py', 'diffsynth/trainers/utils.py',
             'diffsynth/models/wan_video_dit.py', 'diffsynth/models/model_manager.py',
             'diffsynth/pipelines/wan_video_new.py', 'diffsynth/configs/model_config.py',
             'diffsynth/trainers/orca_online.py', 'diffsynth/trainers/orca_source.py',
             'diffsynth/trainers/orca_lora.py', 'examples/wanvideo/orca_config.py',
             'examples/wanvideo/model_training/train.py']
    records = {}
    for name in files:
        source, target = ROOT / name, output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(output / 'source_hashes.json', records)
    diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)
    (output / 'source_snapshot' / 'tracked_changes.patch').write_bytes(diff)


def audit_processed_windows(output, total, updates, world):
    """Audit actual successfully updated samples, including checkpoint resumes."""
    counts = np.zeros(total, dtype=np.int32)
    for rank in range(world):
        # Replayed steps after an interruption replace the earlier attempt.
        rows = {}
        for line in (output / f'processed_windows.rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['update'] <= updates:
                rows[row['update']] = row['indices']
        if set(rows) != set(range(1, updates + 1)):
            raise RuntimeError(f'Missing processed-window records on rank {rank}')
        for indices in rows.values():
            np.add.at(counts, indices, 1)
    result = {'passed': bool(np.all(counts == 1)), 'total_windows': total,
              'processed_windows': int(counts.sum()), 'missing': int(np.count_nonzero(counts == 0)),
              'duplicated': int(np.count_nonzero(counts > 1)), 'optimizer_updates': updates}
    atomic_json(output / 'coverage_verification.json', result)
    if not result['passed']:
        raise RuntimeError(f'Actual training coverage failed: {result}')
    return result


def save_checkpoint(model, optimizer, output, epoch, completed_updates, completed_windows, config, rank, final=False,
                    checkpoint_dir=None, export_final=True):
    synchronize()
    if rank == 0:
        checkpoint_dir = checkpoint_dir or output
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        is_lora = config.get('training_mode', 'full') == 'lora'
        dit_state = trainable_state_dict(model.pipe.dit) if is_lora else model.pipe.dit.state_dict()
        tensor_bytes = sum(t.numel() * t.element_size() for t in dit_state.values())
        optimizer_bytes = sum(t.numel() * t.element_size() for state in optimizer.state.values()
                              for t in state.values() if isinstance(t, torch.Tensor))
        if shutil.disk_usage(checkpoint_dir).free < tensor_bytes + optimizer_bytes + 2 * 2**30:
            raise RuntimeError('Insufficient free space for an atomic resumable checkpoint plus 2 GiB reserve')
        started = time.monotonic()
        # torch.save streams CUDA storages to disk; no second full CPU optimizer copy.
        payload = {'dit': dit_state, 'optimizer': optimizer.state_dict(),
                   'epoch': epoch, 'completed_updates': completed_updates,
                   'completed_windows': completed_windows, 'config': config}
        tmp = checkpoint_dir / 'resume_latest.pt.tmp'
        torch.save(payload, tmp)
        tmp.replace(checkpoint_dir / 'resume_latest.pt')
        del payload
        export_path = None
        if final and export_final:
            if shutil.disk_usage(output).free < tensor_bytes + 2 * 2**30:
                raise RuntimeError('Insufficient free space for final DiT export plus 2 GiB reserve')
            state = {k: v.detach().cpu().contiguous() for k, v in dit_state.items()}
            path = output / f'epoch-{epoch}.safetensors'
            save_file(state, str(path.with_suffix('.safetensors.tmp')),
                      metadata={'checkpoint_format': 'orca_lora_adapter' if is_lora else 'full_dit',
                                'base_model': config['model'],
                                'lora_config': json.dumps({'rank': config.get('lora_rank', 32),
                                    'target_modules': config.get('lora_target_modules', '').split(',')}) if is_lora else '{}',
                                'trainable_precision': config.get('trainable_precision', 'bfloat16'),
                                'action_representation': config.get('action_representation', 'delta_joint_target_error'),
                                'action_mode': config.get('action_mode', 'delta'),
                                'action_contract': json.dumps(config.get('action_contract', {}), sort_keys=True), 'action_dim': '58',
                                'action2obs_bias_in_loader': 'true', 'epochs_completed': str(epoch)})
            path.with_suffix('.safetensors.tmp').replace(path)
            export_path = str(path)
            del state
        atomic_json(output / 'checkpoint_status.json', {'epoch': epoch, 'completed_updates': completed_updates,
                    'completed_windows': completed_windows, 'final': final, 'export': export_path,
                    'resume_checkpoint': str(checkpoint_dir / 'resume_latest.pt'),
                    'save_seconds': time.monotonic() - started})
    synchronize()


def check_initial_checkpoint(payload, config, epoch_index):
    """A new epoch starts only from a complete preceding epoch, keeping AdamW."""
    previous = payload['config']
    check_training_mode(previous, config)
    if previous.get('action_mode', 'delta') != config.get('action_mode', 'delta'):
        raise ValueError('Initial checkpoint action mode mismatch')
    for key in ['global_batch', 'learning_rate', 'weight_decay', 'static_probability',
                'world_size', 'dataset_contract_sha256', 'model', 'dataset', 'seed',
                'gradient_checkpointing', 'fused_adamw', 'action_dim', 'action2obs_bias']:
        if previous[key] != config[key]:
            raise ValueError(f'Initial checkpoint configuration mismatch: {key}')
    expected_updates = math.ceil(config['train_windows'] / config['global_batch'])
    if (payload['epoch'] != epoch_index or payload['completed_windows'] != config['train_windows']
            or payload['completed_updates'] != expected_updates):
        raise ValueError('Initial checkpoint must contain the complete preceding epoch')
    if not payload['optimizer']['state']:
        raise ValueError('Initial checkpoint has no optimizer state')


def validate(model, dataset, indices, args, rank, world, phase):
    was_training = model.training
    model.eval()
    total = torch.zeros(2, device=model.pipe.device, dtype=torch.float64)
    local_indices = indices[rank::world]
    for offset in range(0, len(local_indices), args.micro_batch_size):
        batch_indices = local_indices[offset:offset + args.micro_batch_size]
        samples = [dataset[int(index)] for index in batch_indices]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if args.micro_batch_size == 1:
                seed_sample(args.seed + 100_000_000, 0, int(batch_indices[0]))
                loss = model(samples[0]).reshape(1)
            else:
                loss = model(samples, sample_seeds=[sample_seed(args.seed + 100_000_000, 0, int(i)) for i in batch_indices])
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f'Non-finite validation loss at {batch_indices}')
        total[0] += loss.double().sum()
        total[1] += len(samples)
    reduce_sum(total)
    result = {'phase': phase, 'windows': int(total[1].item()), 'loss': float((total[0] / total[1]).item())}
    model.train(was_training)
    if rank == 0:
        with (args.output / 'validation.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        print('VALIDATION', json.dumps(result), flush=True)
    return result


def main(args):
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(args.cpu_threads)
    if world > 1:
        dist.init_process_group('nccl', device_id=device, timeout=datetime.timedelta(minutes=30))
    if args.global_batch % (world * args.micro_batch_size):
        raise ValueError('global_batch must be divisible by world_size * micro_batch_size')
    samples_per_rank = args.global_batch // world
    accumulation = samples_per_rank // args.micro_batch_size
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / 'run_config.json').exists() and not args.resume:
            raise FileExistsError(f'Refusing to overwrite existing run: {args.output}')
        if not args.resume:
            snapshot_sources(args.output)
    if world > 1: dist.barrier()
    options = json.loads(args.dataset_options) if args.dataset_options else dict(path=str(args.dataset), format='npy', action_mode='delta')
    if Path(options['path']).resolve() != args.dataset.resolve(): raise ValueError('dataset-options path differs from --dataset')
    train = build_dataset(options, 'train_data')
    val = build_dataset(options, 'val_data', shared=getattr(train, 'reader', None))
    action_mode = options.get('action_mode', 'delta')
    action_contract = action_contract_for(options, getattr(train, 'reader', None))
    contract_hashes = ({'online': fingerprint(train.contract)} if options.get('format') == 'lerobot' else
        {name: hashlib.sha256((args.dataset / name).read_bytes()).hexdigest()
         for name in ['dataset_info.json', 'action_stats.json', 'adaptation_verification.json']})
    order = make_order(len(train), args.seed, args.epoch_index)
    val_indices = np.sort(np.random.default_rng(args.seed + 77).choice(len(val), min(args.val_windows, len(val)), replace=False))
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'world_size': world, 'accumulation_per_rank': accumulation, 'micro_batch_per_rank': args.micro_batch_size,
              'train_windows': len(train), 'train_episodes': len(train.episode_info),
              'val_windows_total': len(val), 'val_episodes': len(val.episode_info),
              'sample_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
              'epochs': 1, 'cumulative_epoch': args.epoch_index + 1,
              'action_dim': 58, 'action2obs_bias': True, 'full_parameter_training': args.training_mode == 'full',
              'trainable_precision': 'float32' if args.training_mode == 'lora' else 'bfloat16',
              'action_mode': action_mode, 'action_representation': MODES[action_mode], 'action_contract': action_contract,
              'dataset_contract_sha256': contract_hashes,
              'torch': torch.__version__, 'optimizer_state_precision': 'float32' if args.training_mode == 'lora' else 'bfloat16',
              'gpu': torch.cuda.get_device_name(device)}
    if rank == 0:
        if args.resume:
            resume_record = args.output / 'resumes' / str(time.time_ns())
            resume_record.mkdir(parents=True)
            snapshot_sources(resume_record)
            atomic_json(resume_record / 'run_config.json', config)
            atomic_json(args.output / 'resume_config.json', config)
        if not args.resume:
            atomic_json(args.output / 'run_config.json', config)
            atomic_json(args.output / 'action_contract.json', action_contract)
            stats = (train.reader.stats if hasattr(train, 'reader') else
                     json.loads((args.dataset / 'action_stats.json').read_text()))
            atomic_json(args.output / 'action_stats.json', stats)
            np.save(args.output / 'sample_order.npy', order)
            atomic_json(args.output / 'validation_manifest.json', [
                {'index': int(i), 'episode': val.episode_info[val.sample_indices[i][0]][0],
                 'start': val.sample_indices[i][2]} for i in val_indices])
            counts = np.bincount([train.sample_indices[i][0] for i in order], minlength=len(train.episode_info))
            atomic_json(args.output / 'coverage_manifest.json', {'windows': len(order), 'unique_windows': len(np.unique(order)),
                        'episodes': [{'path': e[0], 'frames': e[2], 'windows': int(n)} for e, n in zip(train.episode_info, counts)]})
    seed_sample(args.seed, 0, 0)
    model_paths = [[str(args.model / f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors') for i in range(1, 4)], str(args.model / 'Wan2.2_VAE.pth')]
    model = WanTrainingModule(model_paths=json.dumps(model_paths), trainable_models='dit', action_dim=58,
                             extra_inputs='input_image,action', static_video_prob=args.static_probability,
                             use_gradient_checkpointing=args.gradient_checkpointing)
    model.to(device=device, dtype=torch.bfloat16)
    if args.training_mode == 'lora':
        model.pipe.dit = configure_lora(model.pipe.dit, args.lora_rank, args.lora_target_modules.split(','))
        upcast_trainable_parameters(model.pipe.dit)
    model.train()
    if args.training_mode == 'full':
        assert all(p.requires_grad for p in model.pipe.dit.parameters())
    else:
        for name, parameter in model.pipe.dit.named_parameters():
            expected = '.lora_' in name or name.startswith(('action_mlp1.', 'action_mlp2.'))
            assert parameter.requires_grad == expected, name
    assert not any(p.requires_grad for p in model.pipe.vae.parameters())
    assert not hasattr(model.pipe.dit, 'action_time_encoding')
    if rank == 0:
        summary = {'mode': args.training_mode,
                   'trainable_parameters': sum(p.numel() for p in model.pipe.dit.parameters() if p.requires_grad),
                   'total_parameters': sum(p.numel() for p in model.pipe.dit.parameters()),
                   'trainable_precision': config['trainable_precision'],
                   'trainable_names': [n for n, p in model.pipe.dit.named_parameters() if p.requires_grad]}
        atomic_json(args.output / 'trainable_parameters.json', summary)
        print('TRAINABLE', json.dumps({k: v for k, v in summary.items() if k != 'trainable_names'}), flush=True)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, fused=args.fused_adamw)
    start_update = 0
    if args.resume or args.initial_checkpoint:
        # Sequential CPU loads prevent two 30GB optimizer snapshots exhausting host RAM.
        for loading_rank in range(world):
            if loading_rank == rank:
                payload = torch.load(args.resume or args.initial_checkpoint, map_location='cpu', weights_only=False)
                if args.initial_checkpoint:
                    check_initial_checkpoint(payload, config, args.epoch_index)
                else:
                    check_training_mode(payload['config'], config)
                    if payload['config'].get('action_mode', 'delta') != config.get('action_mode', 'delta'):
                        raise ValueError('Resume action mode mismatch')
                    for key in ['sample_order_sha256', 'global_batch', 'learning_rate', 'weight_decay',
                                'static_probability', 'world_size', 'dataset_contract_sha256', 'model',
                                'seed', 'gradient_checkpointing', 'fused_adamw', 'action_dim', 'action2obs_bias']:
                        if payload['config'][key] != config[key]:
                            raise ValueError(f'Resume configuration mismatch: {key}')
                if args.training_mode == 'lora':
                    load_adapter_state(model.pipe.dit, payload['dit'])
                else:
                    model.pipe.dit.load_state_dict(payload['dit'], strict=True)
                optimizer.load_state_dict(payload['optimizer'])
                start_update = payload['completed_updates'] if args.resume else 0
                if rank == 0:
                    steps = [float(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s]
                    atomic_json(args.output / 'initialization.json', {
                        'mode': 'resume_epoch' if args.resume else 'continue_next_epoch',
                        'checkpoint': str(args.resume or args.initial_checkpoint),
                        'source_completed_epochs': payload['epoch'],
                        'source_run': payload['config']['output'],
                        'source_completed_updates': payload['completed_updates'],
                        'optimizer_state_count': len(payload['optimizer']['state']),
                        'optimizer_step_min': min(steps), 'optimizer_step_max': max(steps),
                        'source_micro_batch_size': payload['config'].get('micro_batch_per_rank', 1),
                        'micro_batch_size': args.micro_batch_size,
                        'start_update_in_epoch': start_update})
                del payload
                gc.collect()
            if world > 1: dist.barrier()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=True, gradient_as_bucket_view=True, broadcast_buffers=False,
                  bucket_cap_mb=64) if world > 1 else model
    total_updates = math.ceil(len(order) / args.global_batch)
    end_update = min(total_updates, args.warmup_updates + args.benchmark_updates) if args.benchmark else total_updates
    if args.stop_after_updates and not args.benchmark:
        end_update = min(end_update, args.stop_after_updates)
    if end_update < start_update:
        raise ValueError('Run limit must not precede the resumed optimizer update')
    selected_order = order[:min(len(order), end_update * args.global_batch)]
    indices = shard_order(selected_order, world, rank)[start_update * samples_per_rank:]
    loader = create_loader(train, indices, args.workers, args.micro_batch_size)
    iterator = iter(loader)
    if not args.benchmark and start_update == 0:
        validate(model, val, val_indices, args, rank, world, 'initial')
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.monotonic()
    timings, measured_windows = [], 0
    completed_windows = min(len(order), start_update * args.global_batch)
    probe = model.pipe.dit.action_mlp1[0].weight
    initial_probe = probe.detach().clone()
    lora_probe = (model.pipe.dit.blocks[0].self_attn.q.lora_B['default'].weight
                  if args.training_mode == 'lora' and 'q' in args.lora_target_modules.split(',') else None)
    if args.training_mode == 'lora' and lora_probe is None:
        lora_probe = next(p for n, p in model.pipe.dit.named_parameters() if '.lora_B.' in n)
    initial_lora = lora_probe.detach().clone() if lora_probe is not None else None
    for update in range(start_update, end_update):
        real_count = group_real_count(len(selected_order), update, samples_per_rank, world)
        local_micro_steps = math.ceil(math.ceil(real_count / world) / args.micro_batch_size)
        group_loss = torch.zeros((), device=device, dtype=torch.float64)
        processed_indices = []
        synchronize()
        step_started = time.monotonic()
        for micro in range(local_micro_steps):
            batch = next(iterator)
            batch_indices = [int(item[0]) for item in batch]
            valid = torch.tensor([item[1] for item in batch], device=device, dtype=torch.float32)
            context = wrapped.no_sync() if world > 1 and micro < local_micro_steps - 1 else contextlib.nullcontext()
            with context, torch.autocast('cuda', dtype=torch.bfloat16):
                if args.micro_batch_size == 1:
                    seed_sample(args.seed, args.epoch_index, batch_indices[0])
                    losses = wrapped(batch[0][2]).reshape(1)
                else:
                    losses = wrapped([item[2] for item in batch],
                        sample_seeds=[sample_seed(args.seed, args.epoch_index, i) for i in batch_indices])
                if not torch.isfinite(losses).all():
                    raise FloatingPointError(f'Non-finite training loss at windows {batch_indices}')
                # DDP divides by world_size. Undo that and divide by the exact
                # number of real samples, including the last partial group.
                (losses * valid * (world / real_count)).sum().backward()
            group_loss += (losses.detach().double() * valid).sum()
            processed_indices.extend(int(index) for index, is_valid, _ in batch if is_valid)
        if update == start_update:
            probe_names = ['action_mlp1.0.weight', 'action_mlp2.0.weight']
            probe_names += ([n for n, p in model.pipe.dit.named_parameters() if p is lora_probe]
                            if lora_probe is not None else ['blocks.0.self_attn.q.weight'])
            for name in probe_names:
                grad = dict(model.pipe.dit.named_parameters())[name].grad
                if grad is None or not torch.isfinite(grad).all():
                    raise FloatingPointError(f'Missing/nonfinite gradient: {name}')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        seconds = time.monotonic() - step_started
        mean_loss = float((reduce_sum(group_loss) / real_count).item())
        with (args.output / f'processed_windows.rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps({'update': update + 1, 'indices': processed_indices}) + '\n')
        completed_windows += real_count
        if not args.benchmark or update >= args.warmup_updates:
            timings.append(seconds)
            measured_windows += real_count
        metrics = {'status': 'benchmarking' if args.benchmark else 'training', 'optimizer_updates': update + 1,
                   'total_optimizer_updates': total_updates, 'completed_windows': completed_windows,
                   'total_windows': len(order), 'epoch_fraction': completed_windows / len(order),
                   'micro_batch_size': args.micro_batch_size, 'accumulation_per_rank': accumulation,
                   'loss': mean_loss, 'update_seconds': seconds, 'global_windows_per_second': real_count / seconds,
                   'elapsed_seconds': time.monotonic() - run_start, **memory_record(device)}
        if rank == 0:
            with (args.output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics) + '\n')
            atomic_json(args.output / 'progress.json', metrics)
            if (update + 1) % args.log_updates == 0 or update == start_update:
                print('PROGRESS', json.dumps(metrics), flush=True)
        if not args.benchmark:
            if args.val_updates and (update + 1) % args.val_updates == 0 and update + 1 < end_update:
                validate(model, val, val_indices, args, rank, world, f'update-{update + 1}')
            if args.checkpoint_updates and (update + 1) % args.checkpoint_updates == 0 and update + 1 < end_update:
                save_checkpoint(model, optimizer, args.output, args.epoch_index, update + 1, completed_windows,
                                config, rank, checkpoint_dir=args.checkpoint_dir)
    del iterator, loader
    change = float((probe.detach().float() - initial_probe.float()).abs().max().item())
    if change == 0 and end_update > start_update:
        raise RuntimeError('Action weights did not change after optimizer updates')
    lora_change = (float((lora_probe.detach().float() - initial_lora.float()).abs().max().item())
                   if lora_probe is not None else None)
    if lora_change == 0 and end_update > start_update:
        raise RuntimeError('LoRA weights did not change after optimizer updates')
    if world > 1:
        # Check representative action parameters agree after real optimizer updates.
        check = probe.detach().clone()
        dist.broadcast(check, src=0)
        if not torch.equal(check, probe): raise RuntimeError('DDP action weights differ between ranks')
        if lora_probe is not None:
            check = lora_probe.detach().clone()
            dist.broadcast(check, src=0)
            if not torch.equal(check, lora_probe): raise RuntimeError('DDP LoRA weights differ between ranks')
        dist.barrier()
    result = {'passed': True, 'mode': 'benchmark' if args.benchmark else 'full_epoch',
              'completed_windows': completed_windows, 'total_windows': len(order),
              'optimizer_updates': end_update, 'measured_updates': len(timings),
              'measured_windows': measured_windows, 'measured_seconds': sum(timings),
              'global_windows_per_second': measured_windows / sum(timings) if timings else None,
              'update_seconds': timings, 'action_weight_max_change': change,
              'training_mode': args.training_mode, 'lora_weight_max_change': lora_change,
              'world_size': world, 'global_batch': args.global_batch,
              'micro_batch_size': args.micro_batch_size, 'accumulation_per_rank': accumulation,
              'gradient_checkpointing': args.gradient_checkpointing, 'fused_adamw': args.fused_adamw,
              **memory_record(device)}
    epoch_complete = completed_windows == len(order)
    if not args.benchmark and not epoch_complete:
        result['mode'] = 'partial_epoch'
    if not args.benchmark and epoch_complete:
        synchronize()
        if rank == 0:
            audit_processed_windows(args.output, len(order), end_update, world)
        validate(model, val, val_indices, args, rank, world, 'final_fixed')
        if args.full_validation_at_end:
            validate(model, val, np.arange(len(val)), args, rank, world, 'final_full')
        save_checkpoint(model, optimizer, args.output, args.epoch_index + 1, end_update, completed_windows,
                        config, rank, final=True, checkpoint_dir=args.checkpoint_dir, export_final=args.export_final)
    elif not args.benchmark:
        save_checkpoint(model, optimizer, args.output, args.epoch_index, end_update, completed_windows,
                        config, rank, checkpoint_dir=args.checkpoint_dir)
    if rank == 0:
        finished = args.benchmark or epoch_complete
        result['epoch_complete'] = epoch_complete
        atomic_json(args.output / ('result.json' if finished else 'partial_result.json'), result)
        atomic_json(args.output / 'progress.json', {**result, 'status': 'completed' if finished else 'paused'})
        print('RESULT', json.dumps(result), flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta'))
    p.add_argument('--dataset-options', help='JSON dataset options from the YAML launcher; omitted = legacy prepared delta NPY')
    p.add_argument('--model', type=Path, default=Path('/home/yiwei/workspace/model/Wan2.2-TI2V-5B'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--global-batch', type=int, default=16)
    p.add_argument('--micro-batch-size', type=int, default=1, help='Actual examples per GPU forward/backward')
    p.add_argument('--training-mode', choices=['full', 'lora'], default='full')
    p.add_argument('--lora-rank', type=int, default=32)
    p.add_argument('--lora-target-modules', default='q,k,v,o,ffn.0,ffn.2')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fused-adamw', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--static-probability', type=float, default=.15)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--warmup-updates', type=int, default=1)
    p.add_argument('--benchmark-updates', type=int, default=3)
    p.add_argument('--log-updates', type=int, default=10)
    p.add_argument('--val-updates', type=int, default=250)
    p.add_argument('--val-windows', type=int, default=120)
    p.add_argument('--checkpoint-updates', type=int, default=500)
    restore = p.add_mutually_exclusive_group()
    restore.add_argument('--resume', type=Path)
    restore.add_argument('--initial-checkpoint', type=Path, help='Complete preceding epoch; retain model and optimizer, begin a new shuffled epoch')
    p.add_argument('--epoch-index', type=int, default=0, help='Zero-based cumulative epoch index, used for sample order and RNG')
    p.add_argument('--checkpoint-dir', type=Path, help='Optional shared directory for atomic resume_latest.pt across epoch runs')
    p.add_argument('--export-final', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--full-validation-at-end', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stop-after-updates', type=int, default=0, help='Save a resumable checkpoint at this absolute update; 0 completes the epoch')
    args = p.parse_args()
    if min(args.global_batch, args.micro_batch_size, args.cpu_threads, args.benchmark_updates, args.log_updates, args.val_windows) < 1:
        p.error('Batch, thread, benchmark, logging and validation sizes must be positive')
    if min(args.workers, args.warmup_updates, args.val_updates, args.checkpoint_updates, args.stop_after_updates, args.epoch_index) < 0:
        p.error('Counts cannot be negative')
    if not 0 <= args.static_probability <= 1: p.error('static-probability must be in [0,1]')
    if args.lora_rank < 1: p.error('lora-rank must be positive')
    if args.initial_checkpoint and args.epoch_index == 0:
        p.error('--initial-checkpoint requires --epoch-index >= 1')
    return args


if __name__ == '__main__':
    main(parse_args())
