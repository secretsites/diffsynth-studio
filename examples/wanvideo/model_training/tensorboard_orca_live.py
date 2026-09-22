#!/usr/bin/env python3
"""Mirror a running ORCA JSONL log into TensorBoard without touching training."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


def read_complete_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        rows.append(json.loads(line))
    return rows


def validation_step(phase, final_step):
    if phase == 'initial':
        return 0
    if phase.startswith('update-'):
        return int(phase.removeprefix('update-'))
    if phase in {'final_fixed', 'final_full'}:
        return final_step
    raise ValueError(f'Unknown validation phase: {phase}')


def main(args):
    run = args.run.resolve()
    config = json.loads((run / 'run_config.json').read_text())
    logdir = args.logdir or run / 'tensorboard'
    logdir.mkdir(parents=True, exist_ok=True)
    status_path = run / 'tensorboard_bridge_status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        proc = Path(f"/proc/{previous.get('pid', -1)}/cmdline")
        if proc.exists() and b'tensorboard_orca_live.py' in proc.read_bytes():
            raise RuntimeError('A TensorBoard bridge is already running for this run')
    # Rebuild the complete event history from authoritative JSONL on bridge restart.
    writer = SummaryWriter(str(logdir), purge_step=args.step_offset + 1 if args.step_offset else 0, flush_secs=5)
    writer.add_text('Run/configuration', '\n\n'.join([
        f"**Base model:** {config['model']}", f"**Dataset:** {config['dataset']}",
        f"**GPUs:** {config['world_size']}; **global batch:** {config['global_batch']}",
        f"**Training:** {config.get('training_mode', 'full')} DiT; VAE frozen; 58D {config.get('action_mode', 'delta')}; static augmentation {config['static_probability']:.0%}.",
        f"**Validation:** fixed windows at initialization and every {config['val_updates']} updates; full validation at the end.",
        '**Time axes:** steps are optimizer updates. Event wall time is ingestion time; Performance/update_seconds is measured training time.',
        '**Memory:** GPU0 allocator peak, excluding driver/display allocations.'
    ]), args.step_offset)
    train_seen = {}
    validation_seen = {}
    final_step = math.ceil(config['train_windows'] / config['global_batch'])
    while True:
        # A recovered update can appear again after a checkpoint resume. The
        # last occurrence is authoritative; don't alternate between attempts.
        canonical = {row['optimizer_updates']: row for row in read_complete_rows(run / 'metrics.jsonl')}
        records = [canonical[step] for step in sorted(canonical)]
        for position, row in enumerate(records):
            step = args.step_offset + row['optimizer_updates']
            encoded = json.dumps(row, sort_keys=True)
            if train_seen.get(step) == encoded:
                continue
            rolling = records[max(0, position - 19):position + 1]
            rolling_windows = sum(min(config['global_batch'], config['train_windows'] - (v['optimizer_updates'] - 1) * config['global_batch']) for v in rolling)
            values = {
                'Loss/train': row['loss'],
                'Performance/update_seconds': row['update_seconds'],
                'Performance/windows_per_second': row['global_windows_per_second'],
                'Performance/windows_per_second_20_updates': rolling_windows / sum(v['update_seconds'] for v in rolling),
                'Progress/epoch_fraction': row['epoch_fraction'],
                'Progress/epochs_completed': args.step_offset / final_step + row['epoch_fraction'],
                'Progress/cumulative_epochs_completed': config.get('cumulative_epoch', 1) - 1 + row['epoch_fraction'],
                'Progress/completed_windows': args.step_offset // final_step * config['train_windows'] + row['completed_windows'],
                'GPU0/peak_allocated_GiB': row['peak_allocated_gib'],
                'GPU0/peak_reserved_GiB': row['peak_reserved_gib'],
                'Train/learning_rate': config['learning_rate'],
            }
            if 'micro_batch_size' in row:
                values['Train/micro_batch_per_gpu'] = row['micro_batch_size']
                values['Train/gradient_accumulation'] = row['accumulation_per_rank']
            for tag, value in values.items():
                writer.add_scalar(tag, value, step)
            train_seen[step] = encoded
        for row in {v['phase']: v for v in read_complete_rows(run / 'validation.jsonl')}.values():
            phase = row['phase']
            encoded = json.dumps(row, sort_keys=True)
            if validation_seen.get(phase) == encoded:
                continue
            tag = 'Loss/validation_full' if phase == 'final_full' else 'Loss/validation_fixed'
            writer.add_scalar(tag, row['loss'], args.step_offset + validation_step(phase, final_step))
            validation_seen[phase] = encoded
        writer.flush()
        job = json.loads((run / 'job_status.json').read_text())
        status = {'pid': os.getpid(), 'logdir': str(logdir), 'last_train_step': max(train_seen, default=0),
                  'validation_phases': list(validation_seen), 'updated_unix_time': time.time(),
                  'training_status': job['status'], 'status': 'following' if job['status']=='running' else 'finished'}
        temp = status_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(status_path)
        if args.once or job['status'] in {'completed', 'failed', 'stopped'}:
            print(json.dumps(status), flush=True)
            break
        time.sleep(args.interval)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--logdir', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--step-offset', type=int, default=0, help='Prior updates in the same multi-epoch experiment')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    if args.step_offset < 0:
        parser.error('step-offset must be nonnegative')
    main(args)
