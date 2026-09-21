"""Regressions for zero-preserving deltas, temporal alignment and native training."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/wanvideo/model_training'))
import convert_orca_delta as convert
import train_rlinf as train
from diffsynth.trainers.dataset import RLinfDataset, validate_action_contract
from diffsynth.models.model_manager import load_model_from_single_file


class DeltaTests(unittest.TestCase):
    def test_delta_is_same_time_target_minus_state_including_first_and_last(self):
        state = np.tile(np.array([3, 8, 11], np.float32)[:, None], (1, 58))
        target = np.tile(np.array([4, 10, 15], np.float32)[:, None], (1, 58))
        np.testing.assert_array_equal(convert.target_error(target, state), np.tile([[1], [2], [4]], (1, 58)))
        np.testing.assert_array_equal(convert.target_error(state, state), np.zeros_like(state))

    def test_asymmetric_training_distribution_keeps_physical_zero_at_zero(self):
        deltas = np.tile(np.array([1, 2, 3, 4], np.float32)[:, None], (1, 58))
        scale = convert.fit_scale([deltas], 1.0)
        np.testing.assert_array_equal(scale, np.full(58, 4))
        normalized = convert.normalize_delta(np.tile([[-2], [0], [2], [8]], (1, 58)), scale)
        np.testing.assert_array_equal(normalized[:, 0], [-.5, 0, .5, 1])

    def test_constant_coordinates_and_nonfinite_inputs(self):
        zero = np.zeros((2, 58), np.float32)
        np.testing.assert_array_equal(convert.fit_scale([zero]), np.ones(58))
        with self.assertRaises(ValueError): convert.target_error(zero, np.full_like(zero, np.nan))
        with self.assertRaises(ValueError): convert.normalize_delta(zero, np.zeros(58))


class LoaderTests(unittest.TestCase):
    def make_dataset(self, root, retain=True):
        p = root / 'train_data/orca/episode_000000'
        p.mkdir(parents=True)
        rgb = np.broadcast_to(np.arange(24, dtype=np.uint8)[:, None, None, None, None], (24, 1, 2, 2, 3)).copy()
        action = np.broadcast_to(np.arange(1, 25, dtype=np.float32)[:, None, None], (24, 1, 58)).copy()
        np.save(p / 'rgb.npy', rgb); np.save(p / 'actions.npy', action)
        args = SimpleNamespace(Ta=8, To=4, stride=1, max_finish_step=0, action2obs_bias=train.parse_bool('true'), retain_actions=retain)
        with contextlib.redirect_stdout(io.StringIO()):
            ds = RLinfDataset(str(root / 'train_data'), action_dim=58, **train.dataset_options(args))
        return ds, p, action

    def test_one_shift_all_early_windows_reference_and_complete_tail(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ds, p, original = self.make_dataset(Path(tmp))
            self.assertEqual(ds.sample_indices[-1][-1], 15)
            for t in [0, 1, 2, 3, 7, 15]:
                item = ds[t]
                ids = [0] + [max(0, i) for i in range(t-3, t+9)]
                np.testing.assert_array_equal([np.asarray(f)[0, 0, 0] for f in item['video']], ids)
                # source action[i] = i+1; loader-aligned action[i] = i.
                np.testing.assert_array_equal(item['action'][:, 0], ids)
                np.testing.assert_array_equal(item['action'][5:, 0], np.arange(t+1, t+9))
                self.assertEqual(tuple(item['action'].shape), (13, 58))
            np.testing.assert_array_equal(np.load(p / 'actions.npy'), original)

    def test_future_only_layout_and_false_boolean(self):
        self.assertFalse(train.parse_bool('false'))
        self.assertFalse(train.parse_bool('0'))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ds, _, _ = self.make_dataset(Path(tmp), retain=False)
            a = ds[3]['action'].numpy()
            np.testing.assert_array_equal(a[:5], np.zeros((5,58)))
            np.testing.assert_array_equal(a[5:,0], np.arange(4,12))

    def test_declared_delta_rejects_missing_shift_double_shift_and_nonzero_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); split = root / 'train_data'; split.mkdir()
            info = {'action_dim':58, 'action_representation':'delta_joint_target_error', 'action2obs_bias_applied':False}
            stats = {'center':[0]*58, 'scale':[1]*58, 'action2obs_bias_applied':False}
            (root/'dataset_info.json').write_text(json.dumps(info)); (root/'action_stats.json').write_text(json.dumps(stats))
            self.assertEqual(len(validate_action_contract(str(split),58,True)),1)
            with self.assertRaisesRegex(ValueError,'requires action2obs_bias'): validate_action_contract(str(split),58,False)
            with self.assertRaisesRegex(ValueError,'action_dim'): validate_action_contract(str(split),7,True)
            info['action2obs_bias_applied']=True; (root/'dataset_info.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'unshifted'): validate_action_contract(str(split),58,True)
            info['action2obs_bias_applied']=False; (root/'dataset_info.json').write_text(json.dumps(info))
            stats['center'][0]=.1; (root/'action_stats.json').write_text(json.dumps(stats))
            with self.assertRaisesRegex(ValueError,'preserve zero'): validate_action_contract(str(split),58,True)


class StaticAugmentationTests(unittest.TestCase):
    def module(self):
        m = train.WanTrainingModule.__new__(train.WanTrainingModule)
        torch.nn.Module.__init__(m)
        m.static_video_prob = 1.0
        m.pipe = SimpleNamespace(device='cpu', units=[])
        m.extra_inputs = ['input_image','action']
        m.use_gradient_checkpointing = False; m.use_gradient_checkpointing_offload = False
        m.max_timestep_boundary = 1.; m.min_timestep_boundary = 0.
        return m

    def data(self):
        from PIL import Image
        return {'video':[Image.fromarray(np.full((8,8,3),i,np.uint8)) for i in range(13)],
                'action':torch.ones(13,58)}

    def test_training_static_sample_has_constant_rgb_and_zero_58d_actions(self):
        out = self.module().forward_preprocess(self.data())
        self.assertEqual(tuple(out['action'].shape),(13,58))
        self.assertEqual(int(torch.count_nonzero(out['action'])),0)
        for f in out['input_video']: np.testing.assert_array_equal(np.asarray(f),np.zeros((8,8,3)))

    def test_validation_never_augmented(self):
        m=self.module(); m.eval(); out=m.forward_preprocess(self.data())
        self.assertTrue(torch.equal(out['action'],torch.ones(13,58)))
        self.assertEqual(np.asarray(out['input_video'][-1])[0,0,0],12)


class TinyConverter:
    def from_civitai(self,state): return state, {'dim':4}


class TinyModel(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.backbone=torch.nn.Linear(dim,dim)
        self.action_mlp1=torch.nn.Sequential(torch.nn.Linear(58,dim),torch.nn.GELU(),torch.nn.Linear(dim,dim))
        self.action_mlp2=torch.nn.Sequential(torch.nn.Linear(232,dim),torch.nn.SiLU(),torch.nn.Linear(dim,dim))
    @staticmethod
    def state_dict_converter(): return TinyConverter()


def load_tiny(state):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_model_from_single_file(state,['tiny'],[TinyModel],'civitai',torch.float32,'cpu')[1][0]


class DimensionAndInitializationTests(unittest.TestCase):
    def test_missing_58d_branches_initialized_even_if_memory_was_finite(self):
        state={k:v for k,v in TinyModel(4).state_dict().items() if k.startswith('backbone.')}
        original=torch.nn.Module.to_empty
        def filled_empty(m,**kwargs):
            original(m,**kwargs)
            with torch.no_grad():
                for p in m.parameters(): p.fill_(123)
            return m
        with patch.object(torch.nn.Module,'to_empty',filled_empty): m=load_tiny(state)
        self.assertTrue(torch.equal(m.backbone.weight,state['backbone.weight']))
        self.assertTrue((m.action_mlp1[0].weight.abs()<2).all())
        self.assertTrue((m.action_mlp2[0].bias==0).all())
        self.assertEqual(m.action_mlp2[0].in_features,58*4)

    def test_supplied_weights_preserved_and_corruption_rejected(self):
        state=TinyModel(4).state_dict(); m=load_tiny(state)
        for k,v in state.items(): self.assertTrue(torch.equal(v,m.state_dict()[k]))
        state['action_mlp1.0.weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'): load_tiny(state)

    def test_complete_58d_checkpoint_schema_and_original_action_architecture(self):
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.model_manager import ModelDetectorFromSingleFile
        from diffsynth.configs.model_config import model_loader_configs
        with init_weights_on_device():
            m=WanModel(dim=3072,in_dim=48,out_dim=48,ffn_dim=14336,freq_dim=256,text_dim=4096,
                num_heads=24,num_layers=30,has_image_input=False,action_dim=58,eps=1e-6,patch_size=(1,2,2))
        self.assertFalse(hasattr(m,'action_time_encoding'))
        self.assertTrue(m.enable_action_crossattn and m.enable_action_modulation)
        self.assertTrue(ModelDetectorFromSingleFile(model_loader_configs).match(state_dict=m.state_dict()))
        with patch.dict(os.environ,{'WAN_ACTION_DIM':'7'}): _,config=WanModel.state_dict_converter().from_civitai(m.state_dict())
        self.assertEqual(config['action_dim'],58)


if __name__=='__main__': unittest.main()
