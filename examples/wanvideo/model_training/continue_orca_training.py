#!/usr/bin/env python3
"""Follow an existing YAML run indefinitely, preserving model and AdamW state.

The policy file is reread between epochs. Set enabled=false to stop at an epoch
boundary. The active epoch is allowed to finish before this supervisor takes over.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

import yaml

from train import ROOT, epoch_plan, execute_epoch
from orca_config import load_config, save_config, write_json


def read(path):
    return json.loads(Path(path).read_text())


def policy_from(path):
    policy = yaml.safe_load(Path(path).read_text())
    required = {'version', 'run', 'enabled', 'save_every_epochs', 'keep_last_exports',
                'full_validation_every_epochs', 'reserve_gib', 'poll_seconds'}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError('Continuous policy fields differ from the documented schema')
    if type(policy['version']) is not int or policy['version'] != 1 or type(policy['enabled']) is not bool:
        raise ValueError('Expected version=1 and boolean enabled')
    for key in ['save_every_epochs', 'full_validation_every_epochs', 'reserve_gib', 'poll_seconds']:
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    keep = policy['keep_last_exports']
    if keep is not None and (type(keep) is not int or keep < 1):
        raise ValueError('keep_last_exports must be null (keep all) or a positive integer')
    if not isinstance(policy['run'], str) or not policy['run']:
        raise ValueError('run must name an existing training output')
    run = Path(policy['run']).expanduser()
    policy['run'] = str((run if run.is_absolute() else ROOT / run).resolve())
    return policy


def alive(pid):
    try:
        return bool(Path(f'/proc/{int(pid)}/cmdline').read_bytes())
    except (OSError, TypeError, ValueError):
        return False


def inspect_checkpoint(path):
    import torch
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    if config.get('training_mode', 'full') != 'full':
        raise ValueError('Continuous supervisor currently requires full-parameter training')
    updates = math.ceil(config['train_windows'] / config['global_batch'])
    steps = [float(state['step']) for state in payload['optimizer']['state'].values() if 'step' in state]
    expected = config['epoch_index'] * updates + payload['completed_updates']
    if not steps or min(steps) != max(steps) or min(steps) != expected:
        raise ValueError('Checkpoint AdamW step count is inconsistent with completed training')
    result = dict(epoch_index=config['epoch_index'], saved_epoch=payload['epoch'],
                  completed_updates=payload['completed_updates'], completed_windows=payload['completed_windows'],
                  train_windows=config['train_windows'], updates_per_epoch=updates,
                  output=config['output'], optimizer_step=expected,
                  optimizer_state_count=len(payload['optimizer']['state']),
                  checkpoint_bytes=path.stat().st_size,
                  model_bytes=sum(t.numel() * t.element_size() for t in payload['dit'].values()))
    del payload
    return result


def completed_epoch(directory, info):
    result_path, coverage_path = directory / 'result.json', directory / 'coverage_verification.json'
    if not result_path.exists() or not coverage_path.exists():
        return False
    result, coverage = read(result_path), read(coverage_path)
    return (info['completed_windows'] == info['train_windows']
            and info['completed_updates'] == info['updates_per_epoch']
            and info['saved_epoch'] == info['epoch_index'] + 1
            and result.get('passed') is True and result.get('epoch_complete') is True
            and result.get('completed_windows') == info['train_windows']
            and coverage.get('passed') is True
            and coverage.get('processed_windows') == info['train_windows']
            and coverage.get('missing') == 0 and coverage.get('duplicated') == 0)


def next_plan(config, policy, info):
    run = Path(policy['run'])
    directory = run / 'epochs' / f"epoch-{info['epoch_index'] + 1:04d}"
    if Path(info['output']).resolve() != directory.resolve():
        raise ValueError('Latest checkpoint belongs to another run or epoch')
    complete = completed_epoch(directory, info)
    epoch = info['epoch_index'] + (2 if complete else 1)
    updated = copy.deepcopy(config)
    updated['output'] = str(run)
    updated['training']['epochs'] = 1
    updated['initialization'] = dict(mode='continue' if complete else 'resume',
        checkpoint=str(run / 'checkpoints/resume_latest.pt'), completed_epochs=epoch - 1)
    updated['checkpoint']['save_epochs'] = [epoch] if epoch % policy['save_every_epochs'] == 0 else []
    updated['validation']['full_at_end'] = epoch % policy['full_validation_every_epochs'] == 0
    plan = epoch_plan(updated)[0]
    return updated, plan


def verify_export(path, epoch, action_mode):
    from safetensors import safe_open
    with safe_open(path, framework='pt', device='cpu') as file:
        metadata = file.metadata()
        if (metadata.get('checkpoint_format') != 'full_dit'
                or int(metadata.get('epochs_completed', -1)) != epoch
                or metadata.get('action_mode') != action_mode):
            raise ValueError(f'Unexpected checkpoint metadata: {path}')


def exports_in(run, action_mode):
    exports = []
    for directory in (run / 'epochs').glob('epoch-*'):
        if not directory.name.removeprefix('epoch-').isdigit():
            continue
        epoch = int(directory.name.removeprefix('epoch-'))
        path = directory / f'epoch-{epoch}.safetensors'
        if not path.exists():
            continue
        if path.is_symlink() or path.resolve().parent.parent.parent != run.resolve():
            raise ValueError(f'Export leaves the managed run: {path}')
        result_path = directory / 'result.json'
        if not result_path.exists() or not read(result_path).get('epoch_complete'):
            continue
        verify_export(path, epoch, action_mode)
        exports.append((epoch, path))
    return sorted(exports)


def retain_exports(run, policy, action_mode):
    """Only prune this run, after enough newer verified milestone exports exist."""
    keep = policy['keep_last_exports']
    exports = exports_in(run, action_mode)
    milestones = [(epoch, path) for epoch, path in exports if epoch % policy['save_every_epochs'] == 0]
    if keep is None or len(milestones) < keep:
        return []
    retained = {epoch for epoch, path in milestones[-keep:]}
    removed = []
    for epoch, path in exports:
        if epoch in retained:
            continue
        alias = run / 'checkpoints' / path.name
        if alias.is_symlink():
            if alias.resolve() != path.resolve():
                raise ValueError(f'Checkpoint alias points elsewhere: {alias}')
            alias.unlink()
        elif alias.exists():
            raise ValueError(f'Refusing to remove an unexpected regular checkpoint alias: {alias}')
        record = dict(epoch=epoch, path=str(path), bytes=path.stat().st_size,
                      retained_epochs=sorted(retained), time=time.time())
        path.unlink()
        with (run / 'continuous_retention.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        removed.append(record)
    return removed


def needed_free_bytes(info, plan, policy):
    # Space for a new atomic optimizer checkpoint, an optional export and reserve.
    return info['checkpoint_bytes'] + (info['model_bytes'] if plan['export'] else 0) + policy['reserve_gib'] * 2**30


def follow(config_path):
    policy = policy_from(config_path)
    run = Path(policy['run'])
    config = load_config(run / 'resolved_config.yaml', 'train')
    if Path(config['output']).resolve() != run or config['training']['mode'] != 'full':
        raise ValueError('Policy must follow its existing full-parameter training output')
    lock = (run / '.continuous.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', stage='waiting_existing_epoch', supervisor_pid=os.getpid(),
                 policy=str(config_path.resolve()), run=str(run), epoch_limit=None, started=time.time())
    shutil.copy2(__file__, run / 'continuous_supervisor_source.py')
    shutil.copy2(config_path, run / 'continuous_initial_policy.yaml')
    owns_training = False

    def status(**changes):
        state.update(changes, updated=time.time())
        write_json(run / 'continuous_status.json', state)
        if owns_training:
            write_json(run / 'job_status.json', state)

    try:
        status()
        while True:
            policy = policy_from(config_path)
            if Path(policy['run']) != run:
                raise ValueError('Cannot change the run directory while a supervisor is active')
            if not policy['enabled']:
                status(status='stopped', stage='stopped_at_epoch_boundary')
                return
            job = read(run / 'job_status.json')
            if not owns_training and alive(job.get('supervisor_pid')):
                status(stage='waiting_existing_epoch', current_epoch=job.get('current_epoch'),
                       parent_supervisor_pid=job.get('supervisor_pid'),
                       save_every_epochs=policy['save_every_epochs'], keep_last_exports=policy['keep_last_exports'])
                time.sleep(policy['poll_seconds'])
                continue
            # Also protect against an orphaned torchrun whose supervisor exited.
            for path in (run / 'epochs').glob('epoch-*/job_status.json'):
                child = read(path)
                if child.get('status') == 'running' and alive(child.get('child_pid')):
                    raise RuntimeError(f'Existing GPU worker still running: {path}')
            owns_training = True
            latest = run / 'checkpoints/resume_latest.pt'
            if not latest.exists():
                raise RuntimeError('No resumable checkpoint is available; refusing to restart from base')
            info = inspect_checkpoint(latest)
            active_config, plan = next_plan(config, policy, info)
            retain_exports(run, policy, config['dataset']['action_mode'])
            required = needed_free_bytes(info, plan, policy)
            free = shutil.disk_usage(run).free
            if free < required:
                status(status='paused', stage='waiting_disk_space', free_bytes=free, required_free_bytes=required,
                       next_epoch=plan['epoch'], completed_epochs=info['epoch_index'] + 1)
                time.sleep(policy['poll_seconds'])
                continue
            directory = Path(plan['output'])
            if active_config['initialization']['mode'] == 'continue' and directory.exists():
                # Preserve any uncheckpointed failed attempt before replaying it.
                directory.rename(directory.with_name(directory.name + f'.uncheckpointed-{time.time_ns()}'))
            directory.mkdir(parents=True, exist_ok=True)
            save_config(directory / 'continuous_config.yaml', active_config)
            write_json(directory / 'continuous_plan.json', dict(policy=policy, plan=plan, initial_checkpoint=info))
            status(status='running', stage='training', current_epoch=plan['epoch'], epoch_output=str(directory),
                   completed_epochs=plan['epoch'] - 1, optimizer_step_at_start=info['optimizer_step'],
                   save_every_epochs=policy['save_every_epochs'], keep_last_exports=policy['keep_last_exports'])
            print('CONTINUOUS_EPOCH_START', json.dumps(state), flush=True)
            result = execute_epoch(plan, active_config)
            after = inspect_checkpoint(latest)
            if not completed_epoch(directory, after) or after['epoch_index'] + 1 != plan['epoch']:
                raise RuntimeError('Epoch/checkpoint coverage verification failed; refusing to advance')
            if plan['export']:
                source = directory / f"epoch-{plan['epoch']}.safetensors"
                verify_export(source, plan['epoch'], config['dataset']['action_mode'])
                alias = run / 'checkpoints' / source.name
                if not alias.exists():
                    alias.symlink_to(os.path.relpath(source, alias.parent))
                elif alias.resolve() != source.resolve():
                    raise RuntimeError('Existing export alias differs from this epoch')
            write_json(directory / 'continuous_verification.json', dict(passed=True, checkpoint=after, result=result))
            write_json(run / 'progress.json', dict(epoch=plan['epoch'], result=result, epoch_limit=None))
            retain_exports(run, policy, config['dataset']['action_mode'])
            status(stage='epoch_complete', completed_epochs=plan['epoch'])
            print('CONTINUOUS_EPOCH_COMPLETE', plan['epoch'], flush=True)
    except BaseException as error:
        status(status='failed', stage='failed', error=repr(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Continuous policy YAML')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    policy = policy_from(args.config)
    config = load_config(Path(policy['run']) / 'resolved_config.yaml', 'train')
    if config['training']['mode'] != 'full':
        raise ValueError('Expected full-parameter training')
    if args.check:
        print(json.dumps(dict(policy=policy, training=config['training'], action_mode=config['dataset']['action_mode']), indent=2))
        print('CONTINUOUS_PREFLIGHT_PASSED')
    else:
        follow(args.config)


if __name__ == '__main__':
    main()
