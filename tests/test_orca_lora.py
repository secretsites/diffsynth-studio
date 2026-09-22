"""Freeze boundaries, adapter round trips and optimizer continuation for ORCA LoRA."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/wanvideo'), str(ROOT / 'examples/wanvideo/model_training')]
from diffsynth.trainers.orca_lora import (configure_lora, trainable_state_dict, load_adapter_state,
                                        check_training_mode, upcast_trainable_parameters)
from orca_config import load_config
from train_orca import save_checkpoint
spec = importlib.util.spec_from_file_location('orca_lora_launcher', ROOT / 'examples/wanvideo/model_training/train.py')
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
CONFIG = ROOT / 'examples/wanvideo/model_training/configs/orca_raw_train.yaml'


class ToyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4)
        self.action_mlp1 = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.action_mlp2 = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.q(x) + self.action_mlp1(x) + self.action_mlp2(x))


class AdapterTests(unittest.TestCase):
    def test_freeze_update_export_reload_and_optimizer_continuation(self):
        torch.manual_seed(7)
        base = ToyDiT()
        model = configure_lora(copy.deepcopy(base), 2, ['q'])
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        x = torch.randn(8, 4)
        model(x).square().mean().backward(); optimizer.step(); optimizer.zero_grad()
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(frozen[n], p), n)
                self.assertIsNone(p.grad)
        self.assertFalse(torch.equal(before['q.lora_B.default.weight'], model.q.lora_B['default'].weight))
        for branch in ['action_mlp1', 'action_mlp2']:
            self.assertFalse(torch.equal(before[branch + '.0.weight'], getattr(model, branch)[0].weight))
        state = trainable_state_dict(model)
        self.assertFalse(any('base_layer' in n or n.startswith('head.') for n in state))
        restored = configure_lora(copy.deepcopy(base), 2, ['q'])
        load_adapter_state(restored, state)
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=.01)
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for m, opt in [(model, optimizer), (restored, restored_optimizer)]:
            m(x).square().mean().backward(); opt.step(); opt.zero_grad()
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_adapter_state(restored, {n: p for n, p in state.items() if not n.startswith('action_mlp2.')})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_adapter_state(restored, {**state, 'head.weight': base.head.weight})

    def test_real_checkpoint_writer_saves_compact_adapter_and_metadata(self):
        dit = configure_lora(ToyDiT(), 2, ['q'])
        optimizer = torch.optim.AdamW((p for p in dit.parameters() if p.requires_grad))
        dit(torch.ones(2, 4)).sum().backward(); optimizer.step()
        config = dict(training_mode='lora', model='base', lora_rank=2, lora_target_modules='q',
                      trainable_precision='float32', action_mode='relative', action_representation='relative_joint_target')
        with tempfile.TemporaryDirectory() as tmp, patch('train_orca.synchronize'):
            save_checkpoint(SimpleNamespace(pipe=SimpleNamespace(dit=dit)), optimizer, Path(tmp), 1, 1, 2,
                            config, 0, final=True)
            payload = torch.load(Path(tmp) / 'resume_latest.pt', weights_only=False)
            self.assertEqual(set(payload['dit']), set(trainable_state_dict(dit)))
            self.assertTrue(payload['optimizer']['state'])
            with safe_open(Path(tmp) / 'epoch-1.safetensors', framework='pt') as f:
                self.assertEqual(f.metadata()['checkpoint_format'], 'orca_lora_adapter')
                self.assertEqual(f.metadata()['action_mode'], 'relative')
                self.assertEqual(json.loads(f.metadata()['lora_config']), {'rank': 2, 'target_modules': ['q']})
            load_adapter_state(dit, load_file(Path(tmp) / 'epoch-1.safetensors'))

    def test_upcast_and_checkpoint_mode_guards(self):
        model = configure_lora(ToyDiT().bfloat16(), 2, ['q'])
        upcast_trainable_parameters(model)
        for p in model.parameters(): self.assertEqual(p.dtype, torch.float32 if p.requires_grad else torch.bfloat16)
        check_training_mode({}, {'training_mode': 'full'})
        config = dict(training_mode='lora', lora_rank=32, lora_target_modules='q', trainable_precision='float32')
        check_training_mode(config, config)
        for update in [dict(training_mode='full'), dict(lora_rank=16), dict(lora_target_modules='v'),
                       dict(trainable_precision='bfloat16')]:
            with self.assertRaisesRegex(ValueError, 'mismatch'): check_training_mode(config, {**config, **update})

    def test_yaml_plan_carries_lora_and_relative_configuration(self):
        config = load_config(CONFIG, 'train', ['training.mode=lora', 'dataset.action_mode=relative',
                                             'training.learning_rate=1e-4'])
        command = launcher.epoch_plan(config)[0]['command']
        for key, value in [('--training-mode', 'lora'), ('--lora-rank', '32'), ('--learning-rate', '0.0001')]:
            self.assertEqual(command[command.index(key) + 1], value)
        data = json.loads(command[command.index('--dataset-options') + 1])
        self.assertEqual(data['action_mode'], 'relative')
        for override in ['training.mode=unknown', 'lora.rank=0', 'lora.rank=true', 'lora.target_modules=[]',
                         'lora.target_modules=[action_mlp1]', 'lora.target_modules=[q,q]']:
            with self.subTest(override=override), self.assertRaises(ValueError): load_config(CONFIG, 'train', [override])


if __name__ == '__main__': unittest.main()
