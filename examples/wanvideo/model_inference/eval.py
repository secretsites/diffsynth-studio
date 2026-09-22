#!/usr/bin/env python3
"""YAML-configured ORCA TF/AR evaluation with paired action/noise controls."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orca_config import ROOT, load_config, write_json
sys.path.insert(0, str(ROOT))


def jobs(config):
    return [(mode, action) for action in config['rollout']['actions'] for mode in config['rollout']['modes']]


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()


def generate(config):
    from orca_rollout import prepare
    plan = prepare(config)
    out = Path(config['output']); tasks = jobs(config); devices = config['devices']
    codes, running, logs = {}, [], []
    started = time.monotonic()
    state = dict(status='running', supervisor_pid=os.getpid(), jobs=[f'{m}_{v}' for m, v in tasks])
    write_json(out / 'job_status.json', state)
    try:
        for offset in range(0, len(tasks), len(devices)):
            running, logs = [], []
            for (mode, action), device in zip(tasks[offset:offset + len(devices)], devices):
                tag = f'{mode}_{action}'
                log = (out / f'{tag}.log').open('w'); logs.append(log)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(out / 'resolved_config.yaml'),
                           '--worker', mode, action, device]
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running.append((tag, proc))
            state['children'] = {tag: proc.pid for tag, proc in running}
            write_json(out / 'job_status.json', state)
            # Poll all workers, so a failure terminates peers instead of waiting for a long rollout.
            while any(proc.poll() is None for _, proc in running):
                if any(proc.poll() not in (None, 0) for _, proc in running):
                    raise RuntimeError('A rollout worker failed; see per-variant log')
                time.sleep(.5)
            for tag, proc in running:
                codes[tag] = proc.returncode
                if proc.returncode: raise RuntimeError(f'{tag} failed: {proc.returncode}')
            for log in logs: log.close()
        state['status'] = 'completed'
    except BaseException as exc:
        for tag, proc in running: stop(proc); codes[tag] = proc.returncode
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        for log in logs: log.close()
        state.update(returncodes=codes, elapsed_seconds=time.monotonic() - started)
        write_json(out / 'job_status.json', state)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--dry-run', action='store_true', help='Print resolved configuration and jobs, without GPU or output writes')
    options.add_argument('--check', action='store_true', help='Check checkpoint and native windows without GPU inference or run output; raw data may populate the reusable cache')
    options.add_argument('--render-only', action='store_true', help='Audit/render completed output using its recorded configuration')
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config, 'eval', args.set)
    if args.worker:
        from orca_rollout import worker
        mode, action, device = args.worker
        if (mode, action) not in jobs(config): raise ValueError('Worker job is not in the resolved configuration')
        worker(config, mode, action, device)
        return
    if args.dry_run:
        print(json.dumps(dict(config=config, jobs=jobs(config)), indent=2)); return
    if args.check:
        from orca_rollout import prepare
        print(json.dumps(prepare(config, write_outputs=False), indent=2)); return
    if args.render_only:
        config = load_config(Path(config['output']) / 'resolved_config.yaml', 'eval')
    else:
        generate(config)
    from orca_render import audit_and_render
    print(json.dumps(audit_and_render(config, render=config['render']['enabled'] or args.render_only), indent=2))


if __name__ == '__main__': main()
