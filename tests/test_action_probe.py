"""Regression tests for initialization and causal action interventions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import patch

import numpy as np
import torch
from diffsynth.models.model_manager import load_model_from_single_file

SPEC = importlib.util.spec_from_file_location('orca_probe', Path(__file__).resolve().parents[1] / 'examples/wanvideo/model_training/probe_orca_action_conditioning.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class TinyConverter:
    def from_civitai(self, state):
        return state, {'dim': 4}


class TinyModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = torch.nn.Linear(dim, dim)
        self.action_mlp1 = torch.nn.Sequential(torch.nn.Linear(2, dim), torch.nn.GELU(), torch.nn.Linear(dim, dim))
        self.action_mlp2 = torch.nn.Sequential(torch.nn.Linear(8, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim))

    @staticmethod
    def state_dict_converter():
        return TinyConverter()


def load(state):
    return load_model_from_single_file(state, ['tiny'], [TinyModel], 'civitai', torch.float32, 'cpu')[1][0]


class InitializationTests(unittest.TestCase):
    def test_missing_finite_action_memory_is_initialized(self):
        base = TinyModel(4).state_dict()
        base = {n:p for n,p in base.items() if n.startswith('backbone.')}
        original = torch.nn.Module.to_empty
        def finite_empty(module, **kwargs):
            original(module, **kwargs)
            with torch.no_grad():
                for p in module.parameters():
                    p.fill_(123)
            return module
        with patch.object(torch.nn.Module, 'to_empty', finite_empty):
            torch.manual_seed(77)
            model = load(base)
        self.assertTrue(torch.equal(model.backbone.weight, base['backbone.weight']))
        self.assertTrue((model.action_mlp1[0].weight.abs() < 2).all())
        self.assertTrue((model.action_mlp2[0].bias == 0).all())

    def test_supplied_action_weights_are_preserved(self):
        state = TinyModel(4).state_dict()
        loaded = load(state).state_dict()
        for name in state:
            self.assertTrue(torch.equal(state[name], loaded[name]), name)

    def test_corrupt_supplied_action_weights_fail(self):
        state = TinyModel(4).state_dict()
        state['action_mlp1.0.weight'][0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            load(state)

    def test_missing_backbone_weights_fail(self):
        state = TinyModel(4).state_dict()
        del state['backbone.weight']
        with self.assertRaisesRegex(ValueError, 'required parameter'):
            load(state)


class CausalityTests(unittest.TestCase):
    def test_all_controls_preserve_context_and_group_boundaries(self):
        a = torch.arange(13*58).reshape(13,58).float()
        donor = -a
        state = torch.ones(58)*77
        for variant in probe.VARIANTS:
            out = probe.intervention(a,state,donor,variant)
            self.assertTrue(torch.equal(out[:5],a[:5]))
            if variant == 'arm_swap':
                self.assertTrue(torch.equal(out[5:,14:],a[5:,14:]))
                self.assertTrue(torch.equal(out[5:,:14],donor[5:,:14]))
            if variant == 'hand_swap':
                self.assertTrue(torch.equal(out[5:,:14],a[5:,:14]))
                self.assertTrue(torch.equal(out[5:,14:],donor[5:,14:]))
            if variant == 'hold':
                self.assertTrue(torch.equal(out[5:],state.expand(8,58)))
            if variant == 'reverse':
                self.assertTrue(torch.equal(out[5:], a[5:].flip(0)))
        self.assertTrue(torch.equal(a,torch.arange(13*58).reshape(13,58).float()))

    def test_already_aligned_commands_are_not_shifted_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'train_data/orca/episode_000003';p.mkdir(parents=True)
            rgb = np.broadcast_to(np.arange(30,dtype=np.uint8)[:,None,None,None,None],(30,1,2,2,3)).copy()
            action = np.broadcast_to(np.arange(30,dtype=np.float32)[:,None,None],(30,1,58)).copy()
            np.save(p/'rgb.npy',rgb);np.save(p/'actions.npy',action);np.save(p/'states.npy',action+100)
            v,a,hold = probe.window_arrays(tmp,dict(split='train_data',episode=3,start=9))
            expected = np.array([0]+list(range(9,21)))
            np.testing.assert_equal(v[:,0,0,0],expected)
            np.testing.assert_equal(a[:,0],expected)
            np.testing.assert_equal(hold,np.full(58,112))

    def test_delta_swap_does_not_copy_donor_absolute_pose(self):
        action = torch.full((13,58), .2)
        donor = torch.full((13,58), -.7)
        donor[5:] += torch.arange(1,9).float()[:,None]*.01
        out = probe.intervention(action, torch.zeros(58), donor, 'delta_swap')
        torch.testing.assert_close(out[5:], .2+torch.arange(1,9).float()[:,None].expand(8,58)*.01)
        self.assertTrue(torch.equal(out[:5], action[:5]))


class FullCheckpointSchemaTests(unittest.TestCase):
    def test_full_58d_checkpoint_is_detected_without_action_environment(self):
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.model_manager import ModelDetectorFromSingleFile
        from diffsynth.configs.model_config import model_loader_configs
        with init_weights_on_device():
            model = WanModel(dim=3072, in_dim=48, out_dim=48, ffn_dim=14336,
                freq_dim=256, text_dim=4096, num_heads=24, num_layers=30,
                has_image_input=False, action_dim=58, eps=1e-6, patch_size=(1,2,2))
        state = model.state_dict()
        self.assertTrue(all(t.is_meta for t in state.values()))
        self.assertTrue(ModelDetectorFromSingleFile(model_loader_configs).match(state_dict=state))
        with patch.dict(os.environ, {'WAN_ACTION_DIM':'7'}):
            _, config = WanModel.state_dict_converter().from_civitai(state)
        self.assertEqual(config['action_dim'],58)
        self.assertEqual(config['length_conditonal_frames'],5)


if __name__ == '__main__':
    unittest.main()
