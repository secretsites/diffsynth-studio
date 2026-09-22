#!/usr/bin/env python3
"""YAML entrypoint for full/LoRA ORCA training and optimizer-preserving continuation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orca_config import ROOT, load_config, save_config, write_json

HERE = Path(__file__).resolve().parent


def epoch_plan(config):
    output = Path(config['output'])
    train, init = config['training'], config['initialization']
    first = init['completed_epochs'] + 1
    saves = config['checkpoint']['save_epochs']
    plans = []
    for relative in range(train['epochs']):
        epoch = first + relative
        directory = output / 'epochs' / f'epoch-{epoch:04d}'
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(config["devices"])}', str(HERE / 'train_orca.py'),
            '--dataset', config['dataset']['path'], '--dataset-options', json.dumps(config['dataset']), '--model', config['model']['base_path'],
            '--output', str(directory), '--epoch-index', str(epoch - 1),
            '--checkpoint-dir', str(output / 'checkpoints')]
        for key in ['global_batch', 'micro_batch_size', 'learning_rate', 'weight_decay', 'static_probability', 'seed', 'workers']:
            cmd += ['--' + key.replace('_', '-'), str(train[key])]
        cmd += ['--training-mode', train['mode'], '--lora-rank', str(config['lora']['rank']),
                '--lora-target-modules', ','.join(config['lora']['target_modules'])]
        cmd += ['--cpu-threads', str(config['cpu_threads']),
                '--val-updates', str(config['validation']['every_updates']),
                '--val-windows', str(config['validation']['windows']),
                '--checkpoint-updates', str(config['checkpoint']['every_updates']),
                '--log-updates', str(config['logging']['every_updates'])]
        for key in ['gradient_checkpointing', 'fused_adamw']:
            cmd += [('--' if train[key] else '--no-') + key.replace('_', '-')]
        if saves is not None and epoch not in saves: cmd.append('--no-export-final')
        if not config['validation']['full_at_end'] or relative != train['epochs'] - 1:
            cmd.append('--no-full-validation-at-end')
        if relative:
            cmd += ['--initial-checkpoint', str(output / 'checkpoints/resume_latest.pt')]
        elif init['mode'] != 'base':
            cmd += ['--resume' if init['mode'] == 'resume' else '--initial-checkpoint', init['checkpoint']]
        plans.append(dict(epoch=epoch, output=str(directory), command=cmd,
                          export=saves is None or epoch in saves))
    return plans


def terminate(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def execute_epoch(plan, config):
    directory = Path(plan['output']); directory.mkdir(parents=True, exist_ok=True)
    root = Path(config['output'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(config['devices']),
               OMP_NUM_THREADS=str(config['cpu_threads']), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    env.pop('WAN_ACTION_TIME_ENCODING', None)
    env['WAN_ACTION_DIM'] = '58'
    child = bridge = None
    with (directory / 'training.log').open('a') as log, (directory / 'tensorboard_bridge.log').open('a') as tb:
        try:
            child = subprocess.Popen(plan['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(directory / 'job_status.json', dict(status='running', child_pid=child.pid))
            print(f'Epoch {plan["epoch"]}: {directory / "training.log"}', flush=True)
            while child.poll() is None:
                if config['logging']['tensorboard'] and bridge is None and (directory / 'run_config.json').exists():
                    run = json.loads((directory / 'run_config.json').read_text())
                    updates = math.ceil(run['train_windows'] / run['global_batch'])
                    bridge = subprocess.Popen([sys.executable, str(HERE / 'tensorboard_orca_live.py'),
                        '--run', str(directory), '--logdir', str(root / 'tensorboard'),
                        '--step-offset', str((plan['epoch'] - 1) * updates)], cwd=ROOT,
                        env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=tb, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(1)
            code = child.returncode
            write_json(directory / 'job_status.json', dict(status='completed' if code == 0 else 'failed', returncode=code))
            if bridge:
                try: bridge.wait(timeout=30)
                except subprocess.TimeoutExpired: terminate(bridge)
                if bridge.returncode: print('TensorBoard bridge error; see tensorboard_bridge.log', file=sys.stderr)
            if code: raise RuntimeError(f'Epoch {plan["epoch"]} failed with exit code {code}')
            result = json.loads((directory / 'result.json').read_text())
            if not result['passed'] or not result['epoch_complete']:
                raise RuntimeError('Epoch was not complete; refusing to advance to the next epoch')
            return result
        except BaseException:
            terminate(child); terminate(bridge)
            write_json(directory / 'job_status.json', dict(status='failed', error='Interrupted or failed; see training.log'))
            raise


def preflight(config, plans):
    base = Path(config['model']['base_path'])
    for name in ['Wan2.2_VAE.pth', *[f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors' for i in range(1, 4)]]:
        if not (base / name).is_file(): raise FileNotFoundError(base / name)
    if config['dataset']['format'] == 'lerobot':
        sys.path.insert(0, str(ROOT))
        from diffsynth.trainers.orca_source import resolve_dataset_root
        raw = resolve_dataset_root(Path(config['dataset']['path']))
        for name in ['meta/info.json', 'meta/episodes.jsonl', 'meta/conversion.json']:
            if not (raw / name).is_file(): raise FileNotFoundError(raw / name)
    else:
        for name in ['dataset_info.json', 'action_stats.json', 'adaptation_verification.json', 'train_data', 'val_data']:
            path = Path(config['dataset']['path']) / name
            if not path.exists(): raise FileNotFoundError(path)
    init, out = config['initialization'], Path(config['output'])
    if init['mode'] != 'base' and not Path(init['checkpoint']).is_file(): raise FileNotFoundError(init['checkpoint'])
    if init['mode'] == 'resume':
        # Coverage logs and sample order are needed to resume an epoch, not just the optimizer file.
        directory = Path(plans[0]['output'])
        if not (directory / 'run_config.json').is_file():
            raise ValueError('resume requires the existing output/epochs/epoch-NNNN directory and coverage logs')
        old = json.loads((directory / 'run_config.json').read_text())
        if old['epoch_index'] != init['completed_epochs']: raise ValueError('Resume epoch index differs from the existing run')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    status = out / 'job_status.json'
    if status.is_file():
        previous = json.loads(status.read_text())
        proc = Path(f'/proc/{previous.get("supervisor_pid", -1)}/cmdline')
        if previous.get('status') == 'running' and proc.exists() and str(Path(__file__).resolve()).encode() in proc.read_bytes():
            raise RuntimeError('A training supervisor is already running for this output')


def run(config):
    plans = epoch_plan(config)
    preflight(config, plans)
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    resolved = output / 'resolved_config.yaml'
    if resolved.exists(): resolved.rename(output / f'resolved_config.previous-{time.time_ns()}.yaml')
    save_config(resolved, config)
    write_json(output / 'epoch_plan.json', plans)
    state = dict(status='running', supervisor_pid=os.getpid(), epochs_requested=len(plans), completed_epochs=[])
    try:
        for plan in plans:
            state.update(current_epoch=plan['epoch'], epoch_output=plan['output'])
            write_json(output / 'job_status.json', state)
            result = execute_epoch(plan, config)
            state['completed_epochs'].append(plan['epoch'])
            if plan['export']:
                source = Path(plan['output']) / f'epoch-{plan["epoch"]}.safetensors'
                target = output / 'checkpoints' / source.name
                if not target.exists(): target.symlink_to(os.path.relpath(source, target.parent))
            write_json(output / 'progress.json', dict(epoch=plan['epoch'], result=result))
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(output / 'job_status.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override a YAML field; repeatable')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved config and commands, no jobs or output files')
    parser.add_argument('--check', action='store_true', help='Also check input paths and resume directory, without starting training')
    args = parser.parse_args()
    config = load_config(args.config, 'train', args.set)
    plans = epoch_plan(config)
    if args.dry_run or args.check:
        print(json.dumps(config, indent=2))
        for plan in plans: print(shlex.join(plan['command']))
        if args.check: preflight(config, plans); print('TRAIN_PREFLIGHT_PASSED')
    else:
        print(json.dumps(run(config), indent=2))


if __name__ == '__main__': main()
