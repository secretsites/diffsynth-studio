"""ORCA adapters include LoRA weights AND the newly initialized action branches."""
import torch
from peft import LoraConfig, inject_adapter_in_model

ACTION_BRANCHES = ('action_mlp1', 'action_mlp2')
DEFAULT_TARGETS = ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']


def configure_lora(dit, rank=32, target_modules=None):
    targets = DEFAULT_TARGETS if target_modules is None else target_modules
    if type(rank) is not int or rank < 1:
        raise ValueError('LoRA rank must be a positive integer')
    if not targets or len(set(targets)) != len(targets) or any(t not in DEFAULT_TARGETS for t in targets):
        raise ValueError(f'LoRA targets must be a unique nonempty subset of {DEFAULT_TARGETS}')
    if hasattr(dit, 'peft_config'):
        raise ValueError('LoRA is already configured')
    dit.requires_grad_(False)
    dit = inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, target_modules=targets), dit)
    # Base Wan has no ORCA action weights. These must learn alongside the adapters.
    for name in ACTION_BRANCHES:
        getattr(dit, name).requires_grad_(True)
    return dit


def trainable_state_dict(dit):
    names = {name for name, p in dit.named_parameters() if p.requires_grad}
    return {name: p for name, p in dit.state_dict().items() if name in names}


def load_adapter_state(dit, state):
    expected = set(trainable_state_dict(dit))
    if set(state) != expected:
        raise ValueError(f'Incomplete or incompatible ORCA adapter: missing={sorted(expected - set(state))}, '
                         f'unexpected={sorted(set(state) - expected)}')
    # Check all shapes before copying anything; frozen base tensors are deliberately absent.
    current = dit.state_dict()
    for name, tensor in state.items():
        if tensor.shape != current[name].shape:
            raise ValueError(f'Adapter shape mismatch: {name}')
    dit.load_state_dict(state, strict=False)


def check_training_mode(previous, current):
    before, after = previous.get('training_mode', 'full'), current.get('training_mode', 'full')
    if before != after:
        raise ValueError('Checkpoint training mode mismatch (full versus lora)')
    if after == 'lora':
        for name in ['lora_rank', 'lora_target_modules', 'trainable_precision']:
            if previous.get(name) != current.get(name):
                raise ValueError(f'Checkpoint LoRA configuration mismatch: {name}')


def upcast_trainable_parameters(dit):
    # Preserve small AdamW updates; frozen base weights remain BF16.
    for parameter in dit.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
