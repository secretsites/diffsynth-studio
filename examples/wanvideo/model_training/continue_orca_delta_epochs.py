#!/usr/bin/env python3
"""Continue complete ORCA epochs with the same model and AdamW state.

Each epoch has its own coverage audit and sample order. A single atomic optimizer
checkpoint is shared; only requested epochs retain standalone DiT exports.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open

from train_orca_delta_distributed import ROOT, atomic_json, audit_processed_windows
from diffsynth.models.utils import hash_state_dict_keys


def read(path):
    return json.loads(path.read_text())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def inspect_export(path, cumulative_epoch):
    state = {}
    elements = 0
    with safe_open(path, framework='pt', device='cpu') as source:
        metadata = source.metadata()
        assert metadata['action_representation'] == 'delta_joint_target_error'
        assert int(metadata['epochs_completed']) == cumulative_epoch
        for name in source.keys():
            tensor = source.get_tensor(name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Nonfinite checkpoint tensor: {name}')
            state[name] = torch.empty(tensor.shape, dtype=tensor.dtype, device='meta')
            elements += tensor.numel()
    assert state['action_mlp1.0.weight'].shape == (3072, 58)
    assert state['action_mlp2.0.weight'].shape == (12288, 232)
    structure = hash_state_dict_keys(state)
    assert structure == 'cd86b8137f89754c6c4557e285274a95'
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(passed=True, path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest(),
                structure_hash=structure, all_tensors_finite=True, tensor_count=len(state), elements=elements)


def main(args):
    torch.set_num_threads(2)
    out, parent = args.output.resolve(), args.parent_run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pid_file = out / 'supervisor.pid'
    if pid_file.exists():
        previous = int(pid_file.read_text())
        command_file = Path(f'/proc/{previous}/cmdline')
        if command_file.exists() and b'continue_orca_delta_epochs.py' in command_file.read_bytes():
            raise RuntimeError('This continuation already has a live supervisor')
    pid_file.write_text(str(os.getpid()))
    config = read(parent / 'run_config.json')
    count = config['train_windows']
    updates = (count + config['global_batch'] - 1) // config['global_batch']
    parent_epochs = config.get('cumulative_epoch', 1)
    plan = dict(parent_run=str(parent), additional_epochs=args.epochs, parent_epochs=parent_epochs,
                final_cumulative_epochs=parent_epochs + args.epochs, windows_per_epoch=count,
                updates_per_epoch=updates, total_windows=count * args.epochs,
                total_optimizer_updates=updates * args.epochs, save_relative_epochs=args.save_epochs,
                optimizer_state_preserved=True, full_validation='last additional epoch',
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if (out / 'plan.json').exists():
        old_plan = read(out / 'plan.json')
        for key in ['parent_run', 'additional_epochs', 'parent_epochs', 'save_relative_epochs']:
            if plan[key] != old_plan[key]:
                raise ValueError(f'Existing plan mismatch: {key}')
    else:
        atomic_json(out / 'plan.json', plan)
    shutil.copy2(__file__, out / 'supervisor_source.py')
    old_status = read(out / 'job_status.json') if (out / 'job_status.json').exists() else {}
    status = dict(status='running', stage='waiting_parent', supervisor_pid=os.getpid(),
                  started_at=old_status.get('started_at', now()), **plan)

    def write(**fields):
        status.update(fields)
        atomic_json(out / 'job_status.json', status)

    child = bridge = None
    try:
        write()
        while True:
            preceding = read(parent / 'job_status.json')
            if preceding['status'] == 'failed':
                raise RuntimeError('Parent training failed; continuation was not started')
            if preceding['status'] == 'completed':
                break
            time.sleep(30)
        write(stage='verifying_parent')
        previous_result = read(parent / 'result.json')
        assert previous_result['passed'] and previous_result['epoch_complete']
        assert previous_result['completed_windows'] == count
        audit_processed_windows(parent, count, updates, config['world_size'])
        parent_export = parent / f'epoch-{parent_epochs}.safetensors'
        # Independent file audit also makes the chain safe if the chat watcher exits.
        parent_audit = inspect_export(parent_export, parent_epochs)
        atomic_json(out / 'parent_checkpoint_verification.json', parent_audit)
        write(training_started_at=old_status.get('training_started_at', now()))
        exports = []
        for relative in range(1, args.epochs + 1):
            cumulative = parent_epochs + relative
            epoch_out = out / 'epochs' / f'epoch-{relative:02d}-total-{cumulative:02d}'
            result_path = epoch_out / 'continuation_verification.json'
            if result_path.exists() and read(result_path)['passed']:
                record = read(result_path)
                if record.get('export'):
                    exports.append(record['export'])
                continue
            latest = out / 'resume_latest.pt'
            initial = parent / 'resume_latest.pt' if relative == 1 else latest
            restore = ['--initial-checkpoint', str(initial)]
            if epoch_out.exists() and (epoch_out / 'run_config.json').exists():
                payload = torch.load(latest if latest.exists() else initial, mmap=True, map_location='cpu', weights_only=False)
                checkpoint_run = payload['config']['output']
                del payload
                if checkpoint_run == str(epoch_out):
                    restore = ['--resume', str(latest)]
                else:
                    # Preserve uncheckpointed attempts and restart this epoch from its predecessor.
                    epoch_out.rename(epoch_out.with_name(epoch_out.name + f'.incomplete-{time.time_ns()}'))
            epoch_out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                       str(ROOT / 'examples/wanvideo/model_training/train_orca_delta_distributed.py'),
                       '--dataset', config['dataset'], '--model', config['model'], '--output', str(epoch_out),
                       '--global-batch', str(config['global_batch']), '--no-gradient-checkpointing', '--fused-adamw',
                       '--learning-rate', str(config['learning_rate']), '--weight-decay', str(config['weight_decay']),
                       '--static-probability', str(config['static_probability']), '--seed', str(config['seed']),
                       '--workers', str(config['workers']), '--cpu-threads', str(config['cpu_threads']),
                       '--val-updates', str(config['val_updates']), '--val-windows', str(config['val_windows']),
                       '--checkpoint-updates', str(config['checkpoint_updates']),
                       '--epoch-index', str(cumulative - 1), '--checkpoint-dir', str(out), *restore]
            if relative not in args.save_epochs:
                command.append('--no-export-final')
            if relative < args.epochs:
                command.append('--no-full-validation-at-end')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS=str(config['cpu_threads']),
                       PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            with (epoch_out / 'training.log').open('a') as log, (epoch_out / 'tensorboard_bridge.log').open('a') as tb_log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                write(stage='training', relative_epoch=relative, cumulative_epoch=cumulative,
                      child_pid=child.pid, epoch_output=str(epoch_out), command=command)
                atomic_json(epoch_out / 'job_status.json', dict(status='running', child_pid=child.pid))
                bridge = None
                while child.poll() is None:
                    if (epoch_out / 'run_config.json').exists() and bridge is None:
                        bridge = subprocess.Popen([sys.executable, str(ROOT / 'examples/wanvideo/model_training/tensorboard_orca_live.py'),
                            '--run', str(epoch_out), '--logdir', str(out / 'tensorboard'),
                            '--step-offset', str((relative - 1) * updates)], cwd=ROOT,
                            env=dict(env, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'), stdout=tb_log, stderr=subprocess.STDOUT)
                    if (epoch_out / 'progress.json').exists():
                        progress = read(epoch_out / 'progress.json')
                        atomic_json(out / 'progress.json', dict(relative_epoch=relative, cumulative_epoch=cumulative,
                            optimizer_updates=(relative - 1) * updates + progress['optimizer_updates'],
                            total_optimizer_updates=args.epochs * updates,
                            completed_windows=(relative - 1) * count + progress['completed_windows'],
                            total_windows=args.epochs * count, epoch_progress=progress))
                    time.sleep(15)
                code = child.returncode
                atomic_json(epoch_out / 'job_status.json', dict(status='completed' if code == 0 else 'failed', returncode=code))
                if bridge is not None:
                    bridge.wait(timeout=30)
                if code:
                    raise RuntimeError(f'Additional epoch {relative} exited with status {code}')
            write(stage='verifying_epoch')
            result = read(epoch_out / 'result.json')
            assert result['passed'] and result['epoch_complete'] and result['completed_windows'] == count
            coverage = audit_processed_windows(epoch_out, count, updates, 2)
            initialization = read(epoch_out / 'initialization.json')
            assert initialization['optimizer_state_count'] > 0
            expected_initial_step = (cumulative - 1) * updates + initialization['start_update_in_epoch']
            assert initialization['optimizer_step_min'] == initialization['optimizer_step_max'] == expected_initial_step
            payload = torch.load(latest, mmap=True, map_location='cpu', weights_only=False)
            assert payload['epoch'] == cumulative and payload['completed_updates'] == updates
            steps = [float(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s]
            assert min(steps) == max(steps) == cumulative * updates
            del payload
            validations = {v['phase']: v for v in map(json.loads, (epoch_out / 'validation.jsonl').read_text().splitlines())}
            assert validations['final_fixed']['windows'] == config['val_windows']
            if relative == args.epochs:
                assert validations['final_full']['windows'] == config['val_windows_total']
            record = dict(passed=True, relative_epoch=relative, cumulative_epoch=cumulative, coverage=coverage,
                          optimizer_step=cumulative * updates, initialization=initialization, validations=validations)
            if relative in args.save_epochs:
                source = epoch_out / f'epoch-{cumulative}.safetensors'
                export = inspect_export(source, cumulative)
                alias = out / f'continuation-epoch-{relative}-total-{cumulative}.safetensors'
                if not alias.exists():
                    os.link(source, alias)
                elif not os.path.samefile(source, alias):
                    raise RuntimeError(f'Export alias points to a different file: {alias}')
                export['alias'] = str(alias)
                record['export'] = export
                exports.append(export)
            atomic_json(result_path, record)
            print('EPOCH_VERIFIED', json.dumps(record), flush=True)
        final = dict(passed=True, additional_epochs_completed=args.epochs,
                     cumulative_epochs_completed=parent_epochs + args.epochs,
                     completed_windows=count * args.epochs, optimizer_updates=updates * args.epochs,
                     checkpoints=exports, scope='Training coverage and checkpoint integrity; not action-following rollout validation.')
        atomic_json(out / 'result.json', final)
        write(status='completed', stage='completed', returncode=0, finished_at=now())
        print('CONTINUATION_COMPLETED', json.dumps(final), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        write(status='failed', error=repr(error), finished_at=now())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--save-epochs', type=int, nargs='+', default=[6, 7])
    args = parser.parse_args()
    if args.epochs < 1 or any(e < 1 or e > args.epochs for e in args.save_epochs):
        parser.error('Invalid additional epoch count or save epochs')
    main(args)
