"""Full epochs include trajectory tails and do not stop at 501 batches."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from diffsynth.trainers.dataset import RLinfDataset
from diffsynth.trainers.utils import launch_training_task


class DatasetWindowTests(unittest.TestCase):
    def test_full_tail_stride_and_early_window_alignment(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp) / "orca/episode_000001"
            folder.mkdir(parents=True)
            t = np.arange(450)
            rgb = np.broadcast_to((t % 200 + 2).astype(np.uint8)[:, None, None, None, None], (450, 1, 4, 4, 3)).copy()
            actions = np.broadcast_to(t.astype(np.float32)[:, None, None], (450, 1, 2)).copy()
            np.save(folder / "rgb.npy", rgb)
            np.save(folder / "actions.npy", actions)
            data = RLinfDataset(tmp, action_dim=2, retain_actions=True)
            self.assertEqual(len(data), 450 - 8)
            for start in [0, 1, 2, 3, 4, 441]:
                sample = data[start]
                ids = [0] + [max(i, 0) for i in range(start - 3, start + 9)]
                np.testing.assert_array_equal(sample["action"][:, 0].numpy(), ids)
                self.assertEqual([np.asarray(x)[0, 0, 0] for x in sample["video"]], [i % 200 + 2 for i in ids])
            self.assertEqual(len(RLinfDataset(tmp, action_dim=2, stride=8)), 56)
            self.assertEqual(len(RLinfDataset(tmp, action_dim=2, max_finish_step=440)), 433)


class FakeAccelerator:
    is_main_process = True
    is_local_main_process = True
    def __init__(self, **kwargs): pass
    def prepare(self, *objects): return objects
    def accumulate(self, model): return contextlib.nullcontext()
    def backward(self, loss): loss.backward()


class EpochTests(unittest.TestCase):
    def run_training(self, limit):
        class Data(torch.utils.data.Dataset):
            load_from_cache = False
            def __len__(self): return 505
            def __getitem__(self, idx): return torch.tensor(float(idx % 3))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.calls = 0
            def trainable_modules(self): return self.parameters()
            def forward(self, x):
                self.calls += 1
                return (self.weight - x).square()
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(learning_rate=1e-4, weight_decay=0., dataset_num_workers=0,
                save_steps=None, save_epochs=1, num_epochs=1, gradient_accumulation_steps=1,
                find_unused_parameters=False, val_interval=2, output_path=tmp,
                max_train_steps_per_epoch=limit)
            with patch("diffsynth.trainers.utils.Accelerator", FakeAccelerator), \
                 patch("diffsynth.trainers.utils.SummaryWriter", MagicMock()), \
                 patch("diffsynth.trainers.utils.tqdm", lambda x: x), \
                 contextlib.redirect_stdout(io.StringIO()):
                launch_training_task(Data(), Data(), model, MagicMock(), args=args)
        return model.calls

    def test_full_epoch_crosses_old_501_batch_limit(self):
        self.assertEqual(self.run_training(None), 505)

    def test_explicit_debug_limit_is_exact(self):
        self.assertEqual(self.run_training(3), 3)


if __name__ == "__main__":
    unittest.main()
