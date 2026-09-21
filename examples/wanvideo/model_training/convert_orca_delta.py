#!/usr/bin/env python3
"""Convert raw target-minus-current-state actions without temporal shifting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import orca_command_source as source

REPRESENTATION = 'delta_joint_target_error'
FRAME_ALIGNMENT = 'actions[t]=normalize(source_target[t]-source_state[t]); no temporal shift'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def target_error(target, state):
    target, state = np.asarray(target, np.float32), np.asarray(state, np.float32)
    if target.shape != state.shape or target.ndim != 2 or target.shape[1] != 58:
        raise ValueError('Expected matching [T,58] raw absolute targets and states')
    if not np.isfinite(target).all() or not np.isfinite(state).all():
        raise ValueError('Non-finite raw target/state')
    return target - state


def fit_scale(training_deltas, quantile=0.99):
    if not 0 < quantile <= 1:
        raise ValueError('Scale quantile must be in (0,1]')
    joined = np.concatenate(training_deltas)
    if joined.ndim != 2 or joined.shape[1] != 58 or not np.isfinite(joined).all():
        raise ValueError('Expected finite training deltas [T,58]')
    magnitude = np.quantile(np.abs(joined), quantile, axis=0).astype(np.float32)
    # No centering: zero must stay zero, including nearly constant dimensions.
    return np.where(magnitude > 1e-6, magnitude, 1.0).astype(np.float32)


def normalize_delta(delta, scale):
    scale = np.asarray(scale, np.float32)
    if scale.shape != (58,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Expected 58 finite positive action scales')
    if not np.isfinite(delta).all():
        raise ValueError('Non-finite delta')
    return np.clip(np.asarray(delta, np.float32) / scale, -1, 1).astype(np.float32)


def reuse_array(src, dest):
    try:
        os.link(src, dest)
        return 'hardlink'
    except OSError as error:
        # Only cross-device/unsupported-link failures warrant a copy fallback.
        import errno
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        shutil.copy2(src, dest)
        return 'copy'


def convert(args):
    raw_root = source.resolve_dataset_root(args.source_root)
    template = args.template_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a new output directory')
    info = source.load_json(raw_root / 'meta/info.json')
    old_info = source.load_json(template / 'dataset_info.json')
    old_stats = source.load_json(template / 'action_stats.json')
    if Path(old_info['source_root']).resolve() != raw_root or old_info['action_dim'] != 58:
        raise ValueError('Template must be the matching ORCA 58D command dataset')
    if old_stats['frame_alignment'] != 'actions[0]=state[0]; actions[t]=source_action[t-1] for t>0':
        raise ValueError('Template must use the audited absolute-command convention')
    splits = {'train_data': old_info['train_episodes'], 'val_data': old_info['val_episodes']}
    available = {int(r['episode_index']): int(r['length']) for r in source.load_jsonl(raw_root / 'meta/episodes.jsonl')}
    train, val = set(splits['train_data']), set(splits['val_data'])
    if not train or train & val or train | val != set(available):
        raise ValueError('Template split must cover all source episodes exactly once')
    if sum(map(len, splits.values())) != len(available):
        raise ValueError('Duplicate episode IDs')
    sidecars = source.load_native_sidecars(raw_root)
    raw = {}
    for i, episode in enumerate(sorted(available)):
        target, state, diagnostics = source.read_action_and_state(raw_root, info, sidecars, episode, 58)
        delta = target_error(target, state)
        if len(delta) != available[episode]:
            raise ValueError(f'Source length mismatch for episode {episode}')
        raw[episode] = (target, state, delta, diagnostics)
        if (i + 1) % 25 == 0:
            print(f'READ_RAW {i+1}/{len(available)}', flush=True)
    scale = fit_scale([raw[e][2] for e in sorted(train)], args.scale_quantile)
    assert not np.any(normalize_delta(np.zeros((1, 58), np.float32), scale))
    output.mkdir(parents=True)
    (output / '.incomplete').write_text('Conversion is not complete. Do not train on this directory.\n')
    stats = {'action_dim': 58, 'names': old_stats['names'], 'representation': REPRESENTATION,
             'formula_raw': 'source_target[t] - source_state[t]', 'raw_units': 'source joint-position units',
             'normalization': 'clip(raw_delta / scale, -1, 1)', 'center': [0.0] * 58,
             'scale': scale.tolist(), 'scale_quantile': args.scale_quantile,
             'scale_fit_episode_ids': sorted(train), 'scale_fit_split': 'train_data',
             'normalized_range': [-1, 1], 'clip': True, 'zero_means': 'target equals current measured position',
             'frame_alignment': FRAME_ALIGNMENT, 'action2obs_bias_applied': False,
             'required_loader_action2obs_bias': True,
             'arm_dimensions': [0, 14], 'hand_dimensions': [14, 58],
             'arm_action_source': 'main parquet action[0:14]',
             'hand_action_source': 'causal native_hand_telemetry.desired; main hand feedback proxy ignored'}
    save_json(output / 'action_stats.json', stats)
    save_json(output / 'state_stats.json', {
        'representation': 'normalized_absolute_joint_position',
        'normalization': 'clip((source_state-center)/scale,-1,1)',
        **{k: old_stats[k] for k in ('names', 'center', 'scale', 'low', 'high')},
        'note': 'states.npy is preserved from the absolute dataset; these are not the delta action statistics.'})
    records = []
    abs_center = np.asarray(old_stats['center'], np.float32)
    abs_scale = np.asarray(old_stats['scale'], np.float32)
    for split, ids in splits.items():
        for episode in ids:
            target, state, delta, diagnostics = raw[episode]
            src = template / split / 'orca' / f'episode_{episode:06d}'
            dest = output / split / 'orca' / src.name
            dest.mkdir(parents=True)
            rgb = np.load(src / 'rgb.npy', mmap_mode='r', allow_pickle=False)
            old_state = np.load(src / 'states.npy', allow_pickle=False)
            old_action = np.load(src / 'actions.npy', allow_pickle=False)
            if rgb.shape != (len(delta), 1, 256, 256, 3) or rgb.dtype != np.uint8:
                raise ValueError(f'RGB format mismatch: {src}')
            expected_state = np.clip((state - abs_center) / abs_scale, -1, 1).astype(np.float32)
            expected_target = np.clip((target - abs_center) / abs_scale, -1, 1).astype(np.float32)
            expected_old = np.concatenate([expected_state[:1], expected_target[:-1]])
            if not np.array_equal(old_state[:, 0], expected_state) or not np.array_equal(old_action[:, 0], expected_old):
                raise ValueError(f'Template differs from reconstructed raw source: {src}')
            normalized = normalize_delta(delta, scale)
            np.save(dest / 'actions.npy', normalized[:, None], allow_pickle=False)
            rgb_mode = reuse_array(src / 'rgb.npy', dest / 'rgb.npy')
            state_mode = reuse_array(src / 'states.npy', dest / 'states.npy')
            saved = np.load(dest / 'actions.npy', allow_pickle=False)
            assert np.array_equal(saved[:, 0], normalized)
            clipped = np.abs(delta / scale) > 1
            records.append({'episode': episode, 'split': split, 'frames': len(delta),
                            'actions_sha256': digest(dest / 'actions.npy'),
                            'raw_delta_sha256': hashlib.sha256(delta.tobytes()).hexdigest(),
                            'unshifted_first_and_last_actions_verified': True,
                            'clipped_fraction': float(clipped.mean()),
                            'clipped_count_per_dimension': clipped.sum(0).tolist(),
                            'max_raw_reconstruction_error_without_clipping': float(np.max(np.abs((normalized * scale - delta)[~clipped]))),
                            'rgb_reuse': rgb_mode, 'state_reuse': state_mode, **diagnostics})
    result_info = {**old_info, 'source_root': str(raw_root), 'template_root': str(template),
                   'action_representation': REPRESENTATION, 'action2obs_bias_applied': False,
                   'required_loader_action2obs_bias': True, 'frame_alignment': FRAME_ALIGNMENT,
                   'rgb_reused_from': str(template), 'states_reused_from': str(template),
                   'states_representation': 'normalized_absolute_joint_position; see state_stats.json',
                   'action_normalization': 'zero-preserving per-dimension scale fitted on training episodes only',
                   'conversion_complete': True}
    save_json(output / 'dataset_info.json', result_info)
    save_json(output / 'conversion_audit.json', {
        'passed': True, 'episodes': len(records), 'frames': sum(r['frames'] for r in records),
        'train_episodes': len(train), 'val_episodes': len(val),
        'action2obs_bias_applied': False, 'raw_command_state_source_verified': True,
        'zero_preserved': True, 'stats_fit_training_only': True,
        'converter_sha256': digest(__file__), 'raw_reader_sha256': digest(source.__file__),
        'action_stats_sha256': digest(output / 'action_stats.json'), 'records': records})
    (output / '.incomplete').unlink()
    print(json.dumps({'output': str(output), 'episodes': len(records), 'frames': sum(r['frames'] for r in records),
                      'train_episodes': len(train), 'val_episodes': len(val), 'action2obs_bias_applied': False}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--template-root', type=Path, required=True, help='Audited absolute command dataset; reuse RGB/states and split')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--scale-quantile', type=float, default=0.99)
    convert(p.parse_args())
