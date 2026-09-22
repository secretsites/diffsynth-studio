"""Action units, causal alignment, train-only statistics and mode-aware static supervision."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/wanvideo'),
               str(ROOT / 'examples/wanvideo/model_training'), str(ROOT / 'examples/wanvideo/model_inference')]
from diffsynth.trainers.orca_online import encode_window, fit_stats, normalize, OnlineOrcaDataset, build_dataset, MODES
from orca_config import load_config
from orca_rollout import check_action_metadata
from train_rlinf import WanTrainingModule
from train_orca import check_initial_checkpoint
spec = importlib.util.spec_from_file_location('orca_train_online_test', ROOT / 'examples/wanvideo/model_training/train.py')
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
RAW_TRAIN = ROOT / 'examples/wanvideo/model_training/configs/orca_raw_train.yaml'
RAW_EVAL = ROOT / 'examples/wanvideo/model_inference/configs/orca_raw_eval.yaml'


def physical():
    t = np.arange(24, dtype=np.float32)[:, None]
    d = np.arange(58, dtype=np.float32)[None] / 10
    return 100 + 3 * t + d, 40 + 2 * t - d


def identity_stats(): return dict(center=[0.] * 58, scale=[1.] * 58, clip=False)


class EncodingTests(unittest.TestCase):
    def test_three_definitions_and_exactly_one_shift(self):
        target, state = physical()
        for start in [0, 1, 2, 3, 7, 15]:
            for mode in MODES:
                with self.subTest(start=start, mode=mode):
                    before_target, before_state = target.copy(), state.copy()
                    out = encode_window(target, state, start, mode, identity_stats())
                    self.assertEqual(out.shape, (13, 58)); self.assertEqual(out.dtype, np.float32)
                    np.testing.assert_array_equal(out[0], 0)
                    for slot, frame in enumerate(range(start - 3, start + 9), 1):
                        if frame <= 0: expected = np.zeros(58)
                        elif mode == 'abs': expected = target[frame - 1]
                        elif mode == 'delta': expected = target[frame - 1] - state[frame - 1]
                        else: expected = target[frame - 1] - state[start]
                        np.testing.assert_array_equal(out[slot], expected)
                    np.testing.assert_array_equal(target, before_target)
                    np.testing.assert_array_equal(state, before_state)

    def test_relative_anchor_changes_for_same_target_in_overlapping_windows(self):
        target, state = physical()
        # target[8] generates output frame 9 in both windows; each has a different anchor.
        a = encode_window(target, state, 4, 'relative', identity_stats())[9]
        b = encode_window(target, state, 5, 'relative', identity_stats())[8]
        np.testing.assert_allclose(a - b, state[5] - state[4])
        np.testing.assert_array_equal(a, target[8] - state[4])
        da = encode_window(target, state, 4, 'delta', identity_stats())[9]
        db = encode_window(target, state, 5, 'delta', identity_stats())[8]
        np.testing.assert_array_equal(da, db)

    def test_transform_precedes_normalization_and_sentinel_stays_zero(self):
        target, state = physical()
        stats = dict(center=[0.] * 58, scale=[100.] * 58, clip=True)
        out = encode_window(target, state, 4, 'relative', stats)
        np.testing.assert_array_equal(out[5], np.clip((target[4] - state[4]) / 100, -1, 1))
        stats['center'] = [50.] * 58
        out = encode_window(target, state, 0, 'abs', stats)
        np.testing.assert_array_equal(out[:5], 0)
        np.testing.assert_array_equal(out[5], np.clip((target[0] - 50) / 100, -1, 1))

    def test_incomplete_or_wrong_shape_rejected(self):
        a, b = physical()
        for start, mode, x, y in [(-1, 'abs', a, b), (16, 'abs', a, b), (0, 'bad', a, b), (0, 'delta', a[:, :57], b[:, :57])]:
            with self.assertRaises(ValueError): encode_window(x, y, start, mode, identity_stats())


class NormalizationTests(unittest.TestCase):
    def test_fit_reads_only_training_episodes_in_all_modes(self):
        target, state = physical()
        calls = []
        def reader(episode):
            calls.append(episode)
            if episode not in [0, 2]: raise AssertionError('Validation leak')
            return target + episode, state
        for mode in MODES:
            calls.clear(); stats = fit_stats(reader, [0, 2], mode)
            self.assertEqual(calls, [0, 2]); self.assertEqual(stats['train_episode_ids'], [0, 2])
            if mode != 'abs': np.testing.assert_array_equal(normalize(np.zeros(58), stats), 0)
            self.assertTrue(np.isfinite(stats['scale']).all())

    def test_relative_scale_fits_window_future_differences(self):
        target, state = physical()
        pairs = np.stack([target[t:t+8] - state[t] for t in range(16)]).reshape(-1, 58)
        stats = fit_stats(lambda _: (target, state), [0], 'relative')
        np.testing.assert_array_equal(stats['scale'], np.quantile(abs(pairs), .99, axis=0).astype(np.float32))

    def test_constant_coordinates_and_none_normalization(self):
        zero = np.zeros((16, 58), np.float32)
        for mode in MODES:
            stats = fit_stats(lambda _: (zero, zero), [0], mode)
            np.testing.assert_array_equal(stats['scale'], 1)
            stats = fit_stats(lambda _: self.fail('none should not fit'), [0], mode, normalization='none')
            self.assertFalse(stats['clip']); np.testing.assert_array_equal(normalize(np.full(58, 10), stats), 10)


class FakeReader:
    def __init__(self, mode):
        self.root = Path('/fixture'); self.train_ids = [0]; self.val_ids = [1]
        self.lengths = {0: 24, 1: 24}; self.mode = mode; self.contract = {}
        self.stats = dict(center=[20.] * 58 if mode == 'abs' else [0.] * 58, scale=[100.] * 58, clip=True)
    def actions(self, episode): return physical()
    def frames(self, episode):
        return np.broadcast_to(np.arange(24, dtype=np.uint8)[:, None, None, None], (24, 2, 2, 3))
    def window(self, episode, start): return encode_window(*self.actions(episode), start, self.mode, self.stats)


class LoaderTests(unittest.TestCase):
    def module(self):
        m = WanTrainingModule.__new__(WanTrainingModule); torch.nn.Module.__init__(m)
        m.static_video_prob = 1.; m.pipe = SimpleNamespace(device='cpu', units=[])
        m.extra_inputs = ['input_image', 'action']; m.use_gradient_checkpointing = False
        m.use_gradient_checkpointing_offload = False; m.max_timestep_boundary = 1.; m.min_timestep_boundary = 0.
        return m

    def test_static_supervision_uses_reference_frame_state_in_abs(self):
        reader = FakeReader('abs'); ds = OnlineOrcaDataset({}, 'train_data', shared=reader)
        for start in [0, 7, 15]:
            item = ds[start]
            out = self.module().forward_preprocess(item)
            expected_hold = normalize(reader.actions(0)[1][0], reader.stats)
            self.assertTrue(np.any(expected_hold))
            for slot, frame in enumerate(range(start - 3, start + 9), 1):
                np.testing.assert_array_equal(out['action'][slot], expected_hold if frame > 0 else np.zeros(58))
            np.testing.assert_array_equal(out['action'][0], 0)
            for f in out['input_video']: np.testing.assert_array_equal(np.asarray(f), 0)

    def test_delta_relative_static_zero_and_validation_never_augmented(self):
        for mode in ['delta', 'relative']:
            ds = OnlineOrcaDataset({}, 'val_data', shared=FakeReader(mode)); self.assertEqual(len(ds), 16)
            out = self.module().forward_preprocess(ds[7])
            np.testing.assert_array_equal(out['action'], 0)
            m = self.module(); m.eval(); original = ds[7]; out = m.forward_preprocess(original)
            np.testing.assert_array_equal(out['action'], ds.reader.window(1, 7))
            self.assertEqual(np.asarray(out['input_video'][-1])[0, 0, 0], 15)

    def test_preencoded_npy_cannot_be_reinterpreted(self):
        for mode in ['abs', 'relative']:
            with self.assertRaisesRegex(ValueError, 'already encoded'):
                build_dataset(dict(path='/missing', format='npy', action_mode=mode), 'train_data')


class ConfigAndCheckpointTests(unittest.TestCase):
    def test_all_modes_forwarded_to_training_backend_and_eval(self):
        for mode in MODES:
            c = load_config(RAW_TRAIN, 'train', [f'dataset.action_mode={mode}'])
            cmd = train.epoch_plan(c)[0]['command']; opts = json.loads(cmd[cmd.index('--dataset-options') + 1])
            self.assertEqual(opts['action_mode'], mode); self.assertEqual(opts['format'], 'lerobot')
            self.assertEqual(load_config(RAW_EVAL, 'eval', [f'dataset.action_mode={mode}'])['dataset']['action_mode'], mode)

    def test_invalid_mode_split_and_normalization_rejected(self):
        for opt in ['dataset.action_mode=bad', 'dataset.format=raw', 'dataset.validation_episodes=[1,1]',
                    'dataset.validation_episodes=[]', 'dataset.scale_quantile=0.5', 'dataset.normalization=std']:
            with self.subTest(opt=opt), self.assertRaises(ValueError): load_config(RAW_TRAIN, 'train', [opt])

    def test_checkpoint_mode_scale_and_anchor_must_match(self):
        for mode in MODES:
            contract = dict(mode=mode, scale=[1]*58, relative_anchor='prediction_start_measured_state' if mode == 'relative' else None)
            meta = dict(action_mode=mode, action_representation=MODES[mode], action_contract=json.dumps(contract))
            check_action_metadata(meta, mode, contract)
            for other in MODES:
                if other != mode:
                    with self.assertRaisesRegex(ValueError, 'mode/representation'): check_action_metadata(meta, other, contract)
            with self.assertRaisesRegex(ValueError, 'normalization/anchor'): check_action_metadata(meta, mode, dict(contract, scale=[2]*58))
        check_action_metadata({'action_representation': MODES['delta']}, 'delta')
        with self.assertRaisesRegex(ValueError, 'no action representation'): check_action_metadata({}, 'delta')

    def test_optimizer_checkpoint_cannot_change_action_mode(self):
        with self.assertRaisesRegex(ValueError, 'action mode mismatch'):
            check_initial_checkpoint(dict(config=dict(action_mode='delta')), dict(action_mode='relative'), 1)


if __name__ == '__main__': unittest.main()
