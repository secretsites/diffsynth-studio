"""YAML configuration, optimizer continuation plans and native rollout alignment regressions."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/wanvideo'))
sys.path.insert(0, str(ROOT / 'examples/wanvideo/model_inference'))
from orca_config import load_config
from orca_rollout import action_window, history_indices, frame_count

spec = importlib.util.spec_from_file_location('orca_train_yaml', ROOT / 'examples/wanvideo/model_training/train.py')
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
TRAIN = ROOT / 'examples/wanvideo/model_training/configs/orca_raw_train.yaml'
EVAL = ROOT / 'examples/wanvideo/model_inference/configs/orca_raw_eval.yaml'
CONTINUE = ['initialization.mode=continue', 'initialization.checkpoint=/tmp/resume_latest.pt',
            'initialization.completed_epochs=2']


class ConfigTests(unittest.TestCase):
    def test_overrides_paths_and_scientific_learning_rate(self):
        c = load_config(TRAIN, 'train', ['training.learning_rate=5e-6', 'training.epochs=3', 'output=outputs/new trial'])
        self.assertEqual(c['training']['learning_rate'], 5e-6)
        self.assertEqual(c['output'], str(ROOT / 'outputs/new trial'))
        self.assertEqual(load_config(TRAIN, 'train')['training']['epochs'], 1)

    def test_misspelled_keys_are_not_silently_ignored(self):
        for key in ['training.learning_rates=.2', 'foo.bar=1', 'sampling.stpes=50']:
            with self.assertRaises(ValueError): load_config(TRAIN if key.startswith('training') else EVAL, 'train' if key.startswith('training') else 'eval', [key])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.yaml'; p.write_text('trainig: {}\n')
            with self.assertRaisesRegex(ValueError, 'Unknown'): load_config(p, 'train')

    def test_types_and_invalid_horizons_fail(self):
        for override in ['rollout.frames=25', 'rollout.frames=0', 'rollout.start_frame=-1', 'rollout.modes=[bad]',
                         'rollout.actions=[]', 'sampling.steps=0', 'sampling.sigma_shift=.nan', 'devices=[0,0]',
                         'render.enabled="false"', 'dataset.action_dim=7', 'rollout.episode=../outside/episode']:
            with self.subTest(override=override), self.assertRaises(ValueError): load_config(EVAL, 'eval', [override])
        with self.assertRaisesRegex(ValueError, 'divisible'): load_config(TRAIN, 'train', ['training.global_batch=3'])

    def test_model_episode_and_single_gpu_are_not_pinned(self):
        c = load_config(EVAL, 'eval', ['model.checkpoint=/tmp/new-epoch.safetensors',
            'rollout.episode=val_data/orca/episode_000150', 'rollout.start_frame=16', 'rollout.frames=null',
            'devices=[1]', 'sampling.steps=20'])
        self.assertEqual(c['model']['checkpoint'], '/tmp/new-epoch.safetensors')
        self.assertIsNone(c['rollout']['frames']); self.assertIsNone(c['model']['checkpoint_sha256'])
        self.assertEqual(c['devices'], ['1'])

    def test_dry_runs_do_not_create_output_or_require_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for entry, cfg in [('model_training/train.py', TRAIN), ('model_inference/eval.py', EVAL)]:
                out = Path(tmp) / entry.split('/')[0]
                result = subprocess.run([sys.executable, str(ROOT / 'examples/wanvideo' / entry), '--config', str(cfg),
                    '--set', 'output=' + str(out), '--set', 'model.base_path=/missing/base', '--dry-run'],
                    cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(out.exists())


class EpochPlanTests(unittest.TestCase):
    def test_full_epochs_preserve_optimizer_and_selected_cumulative_exports(self):
        c = load_config(TRAIN, 'train', ['training.epochs=8', 'checkpoint.save_epochs=[7,8]'])
        plans = train.epoch_plan(c)
        self.assertEqual([p['epoch'] for p in plans], list(range(1, 9)))
        self.assertNotIn('--initial-checkpoint', plans[0]['command'])
        self.assertEqual([p['epoch'] for p in plans if p['export']], [7, 8])
        for p in plans[1:]:
            cmd = p['command']
            self.assertEqual(cmd[cmd.index('--initial-checkpoint') + 1], str(Path(c['output']) / 'checkpoints/resume_latest.pt'))
            self.assertEqual(cmd[cmd.index('--epoch-index') + 1], str(p['epoch'] - 1))
        self.assertTrue(all('--no-full-validation-at-end' in p['command'] for p in plans[:-1]))
        self.assertNotIn('--no-full-validation-at-end', plans[-1]['command'])

    def test_continuation_and_partial_resume_are_distinct(self):
        c = load_config(TRAIN, 'train', CONTINUE)
        p = train.epoch_plan(c)[0]
        self.assertEqual(p['epoch'], 3); self.assertIn('--initial-checkpoint', p['command'])
        c['initialization']['mode'] = 'resume'
        p = train.epoch_plan(c)[0]
        self.assertIn('--resume', p['command']); self.assertNotIn('--initial-checkpoint', p['command'])
        self.assertIn('--no-gradient-checkpointing', p['command'])

    def test_save_epochs_outside_requested_range_rejected(self):
        for epoch in [2, 4]:
            with self.assertRaises(ValueError):
                load_config(TRAIN, 'train', CONTINUE + [f'checkpoint.save_epochs=[{epoch}]'])

    def test_supervisor_advances_only_after_complete_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load_config(TRAIN, 'train', ['output=' + tmp, 'training.epochs=2', 'checkpoint.save_epochs=[2]'])
            seen = []
            def execute(plan, cfg):
                seen.append(plan['epoch'])
                out = Path(plan['output']); out.mkdir(parents=True)
                (Path(tmp) / 'checkpoints').mkdir(exist_ok=True)
                (Path(tmp) / 'checkpoints/resume_latest.pt').write_text('fake optimizer')
                if plan['export']: (out / f'epoch-{plan["epoch"]}.safetensors').write_text('fake export')
                return {'passed': True, 'epoch_complete': True}
            with patch.object(train, 'preflight'), patch.object(train, 'execute_epoch', side_effect=execute):
                state = train.run(c)
            self.assertEqual(seen, [1, 2]); self.assertEqual(state['status'], 'completed')
            self.assertTrue((Path(tmp) / 'checkpoints/epoch-2.safetensors').is_file())
            self.assertFalse((Path(tmp) / 'checkpoints/epoch-1.safetensors').exists())
        with tempfile.TemporaryDirectory() as tmp:
            c['output'] = tmp
            with patch.object(train, 'preflight'), patch.object(train, 'execute_epoch', side_effect=RuntimeError('failed')) as execute:
                with self.assertRaises(RuntimeError): train.run(c)
                self.assertEqual(execute.call_count, 1)
            self.assertEqual(json.loads((Path(tmp) / 'job_status.json').read_text())['status'], 'failed')


class AlignmentTests(unittest.TestCase):
    def test_initial_history_padding_and_single_action_shift(self):
        actions = np.arange(40 * 58, dtype=np.float32).reshape(40, 58)
        w = action_window(actions, 0)
        self.assertTrue(np.array_equal(w[:5], np.zeros((5, 58))))
        np.testing.assert_array_equal(w[5:], actions[:8])
        self.assertEqual(history_indices(0), [0, 0, 0, 0, 0])

    def test_nonzero_start_preserves_reference_and_past_actions(self):
        actions = np.arange(40 * 58, dtype=np.float32).reshape(40, 58)
        w = action_window(actions, 16)
        np.testing.assert_array_equal(w[0], np.zeros(58))
        np.testing.assert_array_equal(w[1:5], actions[12:16])
        np.testing.assert_array_equal(w[5:], actions[16:24])
        self.assertEqual(history_indices(16), [0, 13, 14, 15, 16])

    def test_auto_horizon_uses_only_complete_chunks_and_explicit_overflow_fails(self):
        self.assertEqual(frame_count(441, 0, None), 440)
        self.assertEqual(frame_count(442, 0, None), 440)
        self.assertEqual(frame_count(441, 16, 24), 24)
        for total, start, n in [(441, 0, 448), (441, 16, 440), (441, 440, None), (441, 0, 7)]:
            with self.assertRaises(ValueError): frame_count(total, start, n)


if __name__ == '__main__': unittest.main()
