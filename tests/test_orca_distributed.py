"""Coverage and gradient regressions for real full-window training."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/wanvideo/model_training'))
from train_orca_delta_distributed import make_order, shard_order, group_real_count, audit_processed_windows
from diffsynth.trainers.utils import DiffusionTrainingModule, launch_training_task


class CoverageTests(unittest.TestCase):
    def test_actual_processed_log_audit_handles_resume_replay_and_rejects_missing_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'processed_windows.rank0.jsonl').write_text(
                json.dumps({'update': 1, 'indices': [0, 2]}) + '\n' +
                json.dumps({'update': 1, 'indices': [0, 2]}) + '\n')
            (output / 'processed_windows.rank1.jsonl').write_text(
                json.dumps({'update': 1, 'indices': [1, 3]}) + '\n')
            self.assertTrue(audit_processed_windows(output, 4, 1, 2)['passed'])
            (output / 'processed_windows.rank1.jsonl').write_text(
                json.dumps({'update': 1, 'indices': [1, 2]}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'coverage failed'):
                audit_processed_windows(output, 4, 1, 2)

    def test_full_real_dataset_size_is_covered_once_with_partial_final_update(self):
        order = make_order(62428, 12345, 0)
        shards = [shard_order(order, 2, rank) for rank in range(2)]
        real = [index for shard in shards for index, valid in shard if valid]
        self.assertEqual(sorted(real), list(range(62428)))
        self.assertEqual(len(shards[0]), len(shards[1]))
        counts = [group_real_count(62428, update, 8, 2) for update in range(3902)]
        self.assertEqual(sum(counts), 62428)
        self.assertEqual(counts[-1], 12)

    def test_odd_length_padding_is_zero_weighted_without_missing_or_duplicate_real_windows(self):
        order = make_order(19, 11, 0)
        shards = [shard_order(order, 2, rank) for rank in range(2)]
        self.assertEqual([len(s) for s in shards], [10, 10])
        self.assertEqual(sum(not valid for s in shards for _, valid in s), 1)
        self.assertEqual(sorted(index for s in shards for index, valid in s if valid), list(range(19)))
        self.assertEqual(group_real_count(19, 2, 4, 2), 3)

    def test_epoch_order_reproducible_and_resume_prefix_disjoint(self):
        order = make_order(101, 19, 0)
        np.testing.assert_array_equal(order, make_order(101, 19, 0))
        self.assertFalse(np.array_equal(order, make_order(101, 19, 1)))
        done = set(order[:32])
        remaining = [index for rank in range(2) for index, valid in shard_order(order, 2, rank)[16:] if valid]
        self.assertFalse(done.intersection(remaining))
        self.assertEqual(done.union(remaining), set(range(101)))

    def test_partial_ddp_group_matches_unsharded_full_batch_gradient(self):
        # Unequal final sample counts per rank; compare against a single mean loss.
        inputs = torch.tensor([1., 4., -2.])
        weight = torch.tensor(.2, requires_grad=True)
        ((weight * inputs - 1) ** 2).mean().backward()
        expected = weight.grad.clone()
        rank_grads = []
        for rank in range(2):
            local_weight = torch.tensor(.2, requires_grad=True)
            for index, valid in shard_order(np.arange(3), 2, rank):
                loss = (local_weight * inputs[index] - 1) ** 2
                (loss * (2 / 3 if valid else 0)).backward()
            rank_grads.append(local_weight.grad)
        torch.testing.assert_close(torch.stack(rank_grads).mean(), expected)


class ToyDataset(torch.utils.data.Dataset):
    load_from_cache = False
    def __len__(self): return 4
    def __getitem__(self, index): return {'value': torch.tensor(float(index + 1))}


class ToyModel(DiffusionTrainingModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
    def to(self, *args, **kwargs):
        return torch.nn.Module.to(self, *args, **kwargs)
    def forward(self, data): return self.weight * data['value']


class NoopLogger:
    def on_step_end(self, *args): pass
    def on_epoch_end(self, *args): pass
    def on_training_end(self, *args): pass


class AccumulationTests(unittest.TestCase):
    def test_actual_native_training_loop_preserves_all_four_microstep_gradients(self):
        model = ToyModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(learning_rate=1., weight_decay=0., dataset_num_workers=0,
                save_steps=None, save_epochs=100, num_epochs=1, gradient_accumulation_steps=4,
                find_unused_parameters=False, val_interval=100, output_path=tmp)
            # Exercise the actual loop; SGD exposes the expected average gradient
            # directly, whereas Adam's first step can hide a scaling error.
            with patch('diffsynth.trainers.utils.Accelerator', lambda **kw: Accelerator(cpu=True, **kw)), \
                 patch('diffsynth.trainers.utils.torch.optim.AdamW', torch.optim.SGD), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                launch_training_task(ToyDataset(), ToyDataset(), model, NoopLogger(), args=args)
            # Native ConstantLR initially scales lr by 1/3.
            self.assertAlmostEqual(model.weight.item(), -2.5 / 3, places=5)


if __name__ == '__main__': unittest.main()
