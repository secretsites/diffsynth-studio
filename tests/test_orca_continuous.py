"""Continuous scheduling, optimizer continuation and bounded checkpoint retention."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/wanvideo/model_training'))
import continue_orca_training as continuous


def policy(run):
    return dict(version=1, run=str(run), enabled=True, save_every_epochs=2, keep_last_exports=2,
                full_validation_every_epochs=2, reserve_gib=4, poll_seconds=1)


def checkpoint_info(run, epoch, updates=488):
    return dict(epoch_index=epoch - 1, saved_epoch=epoch if updates == 488 else epoch - 1,
                completed_updates=updates, completed_windows=min(62428, updates * 128),
                train_windows=62428, updates_per_epoch=488,
                output=str(run / 'epochs' / f'epoch-{epoch:04d}'),
                optimizer_step=(epoch - 1) * 488 + updates, optimizer_state_count=1,
                checkpoint_bytes=30 * 2**30, model_bytes=10 * 2**30)


def finish_epoch(run, epoch, export=False):
    directory = run / 'epochs' / f'epoch-{epoch:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, epoch_complete=True, completed_windows=62428)
    (directory / 'result.json').write_text(json.dumps(result))
    (directory / 'coverage_verification.json').write_text(json.dumps(
        dict(passed=True, processed_windows=62428, missing=0, duplicated=0)))
    if export:
        path = directory / f'epoch-{epoch}.safetensors'
        save_file({'weight': torch.ones(1)}, str(path), metadata={
            'checkpoint_format': 'full_dit', 'epochs_completed': str(epoch), 'action_mode': 'relative'})
        alias = run / 'checkpoints' / path.name
        alias.parent.mkdir(exist_ok=True)
        alias.symlink_to(path)
    return result


def config_for(run):
    config = continuous.load_config(ROOT / 'examples/wanvideo/model_training/configs/orca_raw_train.yaml',
                                    'train', ['output=' + str(run), 'dataset.action_mode=relative',
                                              'training.micro_batch_size=4'])
    continuous.save_config(run / 'resolved_config.yaml', config)
    return config


class ContinuousTests(unittest.TestCase):
    def test_complete_epochs_continue_with_optimizer_and_even_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            config = config_for(run)
            for previous in [1, 2, 3, 999]:
                finish_epoch(run, previous)
                updated, plan = continuous.next_plan(config, policy(run), checkpoint_info(run, previous))
                epoch = previous + 1
                self.assertEqual(plan['epoch'], epoch)
                self.assertEqual(plan['export'], epoch % 2 == 0)
                self.assertIn('--initial-checkpoint', plan['command'])
                self.assertNotIn('--resume', plan['command'])
                self.assertEqual(updated['initialization']['completed_epochs'], previous)
                self.assertEqual(updated['validation']['full_at_end'], epoch % 2 == 0)
                self.assertEqual(updated['training']['micro_batch_size'], 4)
                self.assertEqual(updated['training']['global_batch'], 128)

    def test_partial_checkpoint_resumes_same_epoch_and_foreign_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); config = config_for(run)
            info = checkpoint_info(run, 3, updates=100)
            updated, plan = continuous.next_plan(config, policy(run), info)
            self.assertEqual(plan['epoch'], 3)
            self.assertIn('--resume', plan['command'])
            self.assertEqual(updated['initialization']['completed_epochs'], 2)
            with self.assertRaisesRegex(ValueError, 'another run'):
                continuous.next_plan(config, policy(run), {**info, 'output': '/elsewhere'})

    def test_does_not_advance_unverified_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); config = config_for(run)
            _, plan = continuous.next_plan(config, policy(run), checkpoint_info(run, 1))
            self.assertEqual(plan['epoch'], 1)
            self.assertIn('--resume', plan['command'])

    def test_mmap_inspection_checks_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume_latest.pt'
            payload = dict(config=dict(training_mode='full', train_windows=62428,
                global_batch=128, epoch_index=2, output='/run/epochs/epoch-0003'),
                epoch=2, completed_updates=100, completed_windows=12800,
                dit={'weight': torch.ones(2)}, optimizer={'state': {0: {'step': torch.tensor(1076.)}}})
            torch.save(payload, path)
            self.assertEqual(continuous.inspect_checkpoint(path)['optimizer_step'], 1076)
            payload['optimizer']['state'][0]['step'] = torch.tensor(1.)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'AdamW'):
                continuous.inspect_checkpoint(path)

    def test_retention_waits_for_new_exports_and_preserves_other_runs_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'; run.mkdir()
            finish_epoch(run, 1, export=True); finish_epoch(run, 2, export=True)
            self.assertEqual(continuous.retain_exports(run, policy(run), 'relative'), [])
            checkpoint = run / 'checkpoints/resume_latest.pt'; checkpoint.write_text('optimizer')
            other = Path(tmp) / 'other.safetensors'; other.write_text('other experiment')
            finish_epoch(run, 4, export=True); finish_epoch(run, 6, export=True)
            removed = continuous.retain_exports(run, policy(run), 'relative')
            self.assertEqual([item['epoch'] for item in removed], [1, 2])
            self.assertEqual([epoch for epoch, _ in continuous.exports_in(run, 'relative')], [4, 6])
            self.assertEqual(checkpoint.read_text(), 'optimizer')
            self.assertEqual(other.read_text(), 'other experiment')
            self.assertFalse((run / 'checkpoints/epoch-2.safetensors').is_symlink())

    def test_keep_all_never_deletes_and_disk_budget_includes_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); finish_epoch(run, 1, export=True); finish_epoch(run, 2, export=True)
            keep_all = {**policy(run), 'keep_last_exports': None}
            self.assertEqual(continuous.retain_exports(run, keep_all, 'relative'), [])
            info = checkpoint_info(run, 1)
            self.assertEqual(continuous.needed_free_bytes(info, {'export': True}, policy(run)), 44 * 2**30)
            self.assertEqual(continuous.needed_free_bytes(info, {'export': False}, policy(run)), 34 * 2**30)

    def test_live_parent_then_multiple_epochs_then_user_boundary_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); config_for(run)
            policy_file = run / 'continuous.yaml'; policy_file.write_text(yaml.safe_dump(policy(run)))
            finish_epoch(run, 1, export=True)
            (run / 'checkpoints/resume_latest.pt').write_text('placeholder')
            (run / 'job_status.json').write_text(json.dumps(dict(status='running', supervisor_pid=98765, current_epoch=1)))
            current = checkpoint_info(run, 1); running = {'parent': True}; seen = []
            def wait(_):
                self.assertEqual(seen, [])
                running['parent'] = False
            def execute(plan, config):
                epoch = plan['epoch']; seen.append(epoch)
                self.assertEqual(config['initialization']['mode'], 'continue')
                self.assertEqual(current['optimizer_step'], (epoch - 1) * 488)
                result = finish_epoch(run, epoch, export=plan['export'])
                current.update(checkpoint_info(run, epoch))
                if epoch == 4:
                    policy_file.write_text(yaml.safe_dump({**policy(run), 'enabled': False}))
                return result
            with patch.object(continuous, 'alive', side_effect=lambda pid: running['parent']), \
                 patch.object(continuous.time, 'sleep', side_effect=wait), \
                 patch.object(continuous, 'inspect_checkpoint', side_effect=lambda _: copy.deepcopy(current)), \
                 patch.object(continuous.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(200 * 2**30, 0, 200 * 2**30)), \
                 patch.object(continuous, 'execute_epoch', side_effect=execute):
                continuous.follow(policy_file)
            self.assertEqual(seen, [2, 3, 4])
            self.assertEqual(continuous.read(run / 'continuous_status.json')['status'], 'stopped')
            self.assertEqual([epoch for epoch, _ in continuous.exports_in(run, 'relative')], [2, 4])
            self.assertEqual(continuous.read(run / 'progress.json')['epoch'], 4)


if __name__ == '__main__':
    unittest.main()
