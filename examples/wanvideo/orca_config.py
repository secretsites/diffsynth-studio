"""Strict YAML configuration shared by the ORCA train/eval entrypoints (no torch import)."""
import copy
import json
import math
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = '/home/yiwei/workspace/model/Wan2.2-TI2V-5B'
DATA = '/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-delta'
COMMON = dict(version=1, output='outputs/orca', devices=[0, 1], cpu_threads=2,
              dataset=dict(path=DATA, action_dim=58, format='npy', action_mode='delta',
                  camera_key='observation.images.cam_left_high', cache_dir='outputs/.cache/orca_online',
                  validation_episodes=[12, 13, 29, 61, 77, 90, 100, 106, 108, 109, 114, 121, 140, 142, 144],
                  normalization='quantile', scale_quantile=.99, action_stats=None))
TRAIN_DEFAULTS = dict(**COMMON, model=dict(base_path=BASE),
    initialization=dict(mode='base', checkpoint=None, completed_epochs=0),
    training=dict(mode='full', epochs=1, global_batch=128, micro_batch_size=1, learning_rate=1e-5, weight_decay=.01,
                  static_probability=.15, gradient_checkpointing=False, fused_adamw=True, seed=12345, workers=2),
    lora=dict(rank=32, target_modules=['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']),
    checkpoint=dict(every_updates=100, save_epochs=None),
    validation=dict(every_updates=100, windows=120, full_at_end=True),
    logging=dict(every_updates=10, tensorboard=True))
EVAL_DEFAULTS = dict(**COMMON, model=dict(base_path=BASE, checkpoint=None, checkpoint_sha256=None),
    rollout=dict(episode='val_data/orca/episode_000144', environment_index=0, start_frame=0, frames=440,
                 modes=['tf'], actions=['real', 'zero']),
    sampling=dict(steps=50, sigma_shift=5.0, seed=12345, seed_stride=100000),
    render=dict(enabled=True, fps=30, difference_gain=4.0))


def merge(target, incoming, prefix=''):
    if not isinstance(incoming, dict): raise ValueError(f'{prefix or "config"} must be a mapping')
    for key, value in incoming.items():
        label = f'{prefix}{key}'
        if key not in target: raise ValueError(f'Unknown configuration key: {label}')
        if isinstance(target[key], dict): merge(target[key], value, label + '.')
        else: target[key] = value


def path_value(value):
    if not isinstance(value, str) or not value.strip(): raise ValueError('Path must be a nonempty string')
    expanded = os.path.expandvars(value)
    if '$' in expanded: raise ValueError(f'Unresolved environment variable in path: {value}')
    path = Path(expanded).expanduser()
    return str((path if path.is_absolute() else ROOT / path).resolve())


def integer(value, key, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError(f'{key} must be an integer >= {minimum}')


def number(section, key, minimum=0, maximum=None, positive=False):
    value = section[key]
    if isinstance(value, bool): raise ValueError(f'{key} must be numeric')
    try: value = float(value)
    except (TypeError, ValueError): raise ValueError(f'{key} must be numeric') from None
    if not math.isfinite(value) or value < minimum or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f'Invalid {key}: {value}')
    section[key] = value


def boolean(value, key):
    if type(value) is not bool: raise ValueError(f'{key} must be a YAML boolean')


def choices(values, allowed, name):
    if not isinstance(values, list) or not values or any(v not in allowed for v in values) or len(set(values)) != len(values):
        raise ValueError(f'{name} must be a nonempty unique list from {allowed}')


def load_config(path, kind, overrides=()):
    defaults = TRAIN_DEFAULTS if kind == 'train' else EVAL_DEFAULTS
    config = copy.deepcopy(defaults)
    incoming = yaml.safe_load(Path(path).read_text())
    merge(config, incoming)
    for override in overrides:
        key, sep, value = override.partition('=')
        if not sep: raise ValueError('--set requires key=value')
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict): raise ValueError(f'Unknown override: {key}')
            target = target[part]
        merge(target, {parts[-1]: yaml.safe_load(value)}, '.'.join(parts[:-1]) + '.')
    if type(config['version']) is not int or config['version'] != 1: raise ValueError('Only configuration version 1 is supported')
    devices = config['devices']
    if not isinstance(devices, list) or not devices or any(isinstance(x, bool) or not isinstance(x, (str, int)) or not str(x) or ',' in str(x) for x in devices):
        raise ValueError('devices must be a nonempty list of CUDA IDs')
    config['devices'] = [str(x) for x in devices]
    if len(set(config['devices'])) != len(devices): raise ValueError('Duplicate CUDA devices')
    integer(config['cpu_threads'], 'cpu_threads', 1)
    if config['dataset']['action_dim'] != 58: raise ValueError('Current ORCA backend requires action_dim=58')
    for section, key in [(config, 'output'), (config['dataset'], 'path'), (config['model'], 'base_path')]:
        section[key] = path_value(section[key])
    data = config['dataset']
    if data['format'] not in ['npy', 'lerobot']: raise ValueError('dataset.format must be npy or lerobot')
    if data['action_mode'] not in ['abs', 'delta', 'relative']: raise ValueError('dataset.action_mode must be abs, delta, or relative')
    if data['format'] == 'npy' and data['action_mode'] != 'delta':
        raise ValueError('Use format=lerobot for online abs/relative; existing NPY input is preencoded delta')
    if data['normalization'] not in ['quantile', 'none']: raise ValueError('normalization must be quantile or none')
    number(data, 'scale_quantile', minimum=.5, maximum=1)
    if data['scale_quantile'] == .5: raise ValueError('scale_quantile must be > .5')
    if not isinstance(data['validation_episodes'], list) or not data['validation_episodes']: raise ValueError('validation_episodes must be a nonempty list')
    for value in data['validation_episodes']: integer(value, 'validation episode')
    if len(set(data['validation_episodes'])) != len(data['validation_episodes']): raise ValueError('Duplicate validation episodes')
    if not isinstance(data['camera_key'], str) or not data['camera_key']: raise ValueError('camera_key must be a nonempty string')
    data['cache_dir'] = path_value(data['cache_dir'])
    if data['action_stats'] is not None: data['action_stats'] = path_value(data['action_stats'])
    if kind == 'train':
        train, init = config['training'], config['initialization']
        if train['mode'] not in ['full', 'lora']: raise ValueError('training.mode must be full or lora')
        integer(config['lora']['rank'], 'lora.rank', 1)
        choices(config['lora']['target_modules'], ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2'], 'lora.target_modules')
        for key in ['epochs', 'global_batch', 'micro_batch_size']: integer(train[key], key, 1)
        for key in ['workers', 'seed']: integer(train[key], key)
        if train['global_batch'] % len(devices): raise ValueError('global_batch must be divisible by number of devices')
        if train['global_batch'] % (len(devices) * train['micro_batch_size']):
            raise ValueError('global_batch must be divisible by devices * micro_batch_size')
        for key in ['gradient_checkpointing', 'fused_adamw']: boolean(train[key], key)
        number(train, 'learning_rate', positive=True); number(train, 'weight_decay')
        number(train, 'static_probability', maximum=1)
        if init['mode'] not in ['base', 'continue', 'resume']: raise ValueError('initialization.mode: base, continue, or resume')
        integer(init['completed_epochs'], 'completed_epochs')
        if init['mode'] == 'base':
            if init['checkpoint'] is not None or init['completed_epochs'] != 0:
                raise ValueError('Base initialization requires checkpoint=null and completed_epochs=0')
        else:
            init['checkpoint'] = path_value(init['checkpoint'])
            if init['mode'] == 'continue' and init['completed_epochs'] == 0:
                raise ValueError('continue requires at least one completed epoch')
        integer(config['checkpoint']['every_updates'], 'checkpoint.every_updates')
        integer(config['validation']['every_updates'], 'validation.every_updates')
        integer(config['validation']['windows'], 'validation.windows', 1)
        integer(config['logging']['every_updates'], 'logging.every_updates', 1)
        boolean(config['validation']['full_at_end'], 'validation.full_at_end')
        boolean(config['logging']['tensorboard'], 'logging.tensorboard')
        saves = config['checkpoint']['save_epochs']
        if saves is not None:
            if not isinstance(saves, list): raise ValueError('save_epochs must be null (all) or a list of cumulative epoch numbers')
            first, last = init['completed_epochs'] + 1, init['completed_epochs'] + train['epochs']
            for epoch in saves:
                integer(epoch, 'save_epochs entry', first)
                if epoch > last: raise ValueError('save_epochs lies outside this run')
            if len(set(saves)) != len(saves): raise ValueError('Duplicate save_epochs')
    else:
        config['model']['checkpoint'] = path_value(config['model']['checkpoint'])
        sha = config['model']['checkpoint_sha256']
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('checkpoint_sha256 must be a lowercase SHA256 or null')
        roll, sampling = config['rollout'], config['sampling']
        episode = roll['episode']
        if not isinstance(episode, str) or len(Path(episode).parts) != 3 or Path(episode).is_absolute() or '..' in Path(episode).parts:
            raise ValueError('rollout.episode must be relative split/task/episode under dataset.path')
        for key in ['environment_index', 'start_frame']: integer(roll[key], key)
        if roll['frames'] is not None:
            integer(roll['frames'], 'frames', 8)
            if roll['frames'] % 8: raise ValueError('frames must be a multiple of native chunk size 8; no padding')
        choices(roll['modes'], ['ar', 'tf'], 'rollout.modes')
        choices(roll['actions'], ['real', 'zero'], 'rollout.actions')
        integer(sampling['steps'], 'steps', 1)
        integer(sampling['seed'], 'seed'); integer(sampling['seed_stride'], 'seed_stride')
        number(sampling, 'sigma_shift', positive=True)
        integer(config['render']['fps'], 'fps', 1)
        number(config['render'], 'difference_gain', positive=True)
        boolean(config['render']['enabled'], 'render.enabled')
    return config


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_config(path, config):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
