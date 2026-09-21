"""Sampling coverage and exact optimizer recovery for the long ORCA run."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/wanvideo/model_training"))
import train_orca_sampled as sampled


class SamplingTests(unittest.TestCase):
    def test_each_round_has_two_distinct_uniform_candidates_per_episode(self):
        episodes = [dict(episode=1, frames=30), dict(episode=7, frames=41), dict(episode=12, frames=22)]
        for round_id in range(1, 21):
            plan = sampled.sample_round(episodes, round_id, 42)
            self.assertEqual(Counter(r["episode"] for r in plan), {1: 2, 7: 2, 12: 2})
            self.assertEqual(len({(r["episode"], r["current_frame"]) for r in plan}), 6)
            lengths = {r["episode"]: r["frames"] for r in episodes}
            self.assertTrue(all(3 <= r["current_frame"] <= lengths[r["episode"]] - 9 for r in plan))
            self.assertEqual(plan, sampled.sample_round(episodes, round_id, 42))
        self.assertNotEqual(sampled.sample_round(episodes, 1, 42), sampled.sample_round(episodes, 2, 42))
        starts = {r["current_frame"] for e in range(1, 1001)
                  for r in sampled.sample_round(episodes, e, 42) if r["episode"] == 7}
        self.assertEqual(starts, set(range(3, 33)))

    def test_window_preserves_reference_and_exact_aligned_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "train_data/orca/episode_000003"
            folder.mkdir(parents=True)
            indices = np.arange(20)
            rgb = np.broadcast_to(indices[:, None, None, None, None], (20, 1, 2, 2, 3)).astype(np.uint8)
            actions = np.broadcast_to(indices[:, None, None] / 20, (20, 1, 58)).astype(np.float32)
            np.save(folder / "rgb.npy", rgb)
            np.save(folder / "actions.npy", actions)
            np.save(folder / "states.npy", actions + .1)
            windows = sampled.Windows(Path(tmp))
            for current in (3, 11):
                v, a, hold = windows.get("train_data", 3, current)
                ids = [0] + list(range(current - 3, current + 9))
                np.testing.assert_array_equal(v[:, 0, 0, 0], ids)
                np.testing.assert_allclose(a[:, 0], np.array(ids) / 20)
                np.testing.assert_allclose(hold, current / 20 + .1)
            with self.assertRaises(ValueError):
                windows.get("train_data", 3, 2)

    def test_diffusion_noise_is_update_keyed_and_reproducible(self):
        self.assertEqual(sampled.noise_settings(19, 102), sampled.noise_settings(19, 102))
        self.assertNotEqual(sampled.noise_settings(19, 102), sampled.noise_settings(19, 103))


class ResumeTests(unittest.TestCase):
    def test_optimizer_resume_reproduces_the_next_parameter_update(self):
        torch.manual_seed(13)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
        x = torch.tensor([[.2, .3, .7]])
        def update(m, opt):
            opt.zero_grad(set_to_none=True)
            m(x).square().mean().backward()
            opt.step()
        update(model, optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            sampled.save_resume(path, model, optimizer, {"global_step": 1}, "signature")
            update(model, optimizer)
            restored = torch.nn.Linear(3, 2)
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=9., foreach=False)
            progress = sampled.restore_resume(path, restored, restored_opt, "signature")
            self.assertEqual(progress["global_step"], 1)
            update(restored, restored_opt)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "configuration"):
                sampled.restore_resume(path, restored, restored_opt, "wrong")

    def test_milestone_retains_time_encoding_metadata(self):
        model = torch.nn.Linear(3, 2)
        model.action_time_encoding = "sinusoidal"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch-0100.safetensors"
            sampled.save_weights(path, model, 100, 27200, "signature")
            with safe_open(str(path), framework="pt") as saved:
                self.assertEqual(saved.metadata()["action_time_encoding"], "sinusoidal")
                self.assertEqual(saved.metadata()["global_step"], "27200")
                torch.testing.assert_close(saved.get_tensor("weight"), model.weight)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["sha256"], sampled.digest(path))

    def test_log_recovery_discards_unsaved_updates_and_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":')
            sampled.trim_log(path, "step", 2)
            self.assertEqual([json.loads(x)["step"] for x in path.read_text().splitlines()], [1, 2])


if __name__ == "__main__":
    unittest.main()
