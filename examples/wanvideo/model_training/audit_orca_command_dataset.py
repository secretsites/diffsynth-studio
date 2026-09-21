#!/usr/bin/env python3
"""Read-only structural and raw-source audit of the converted ORCA commands."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--converter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('orca_converter', args.converter)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    source = cv.resolve_dataset_root(args.source)
    raw_info = cv.load_json(source / 'meta/info.json')
    info = cv.load_json(args.dataset / 'dataset_info.json')
    stats = cv.load_json(args.dataset / 'action_stats.json')
    norm = {k: np.asarray(stats[k], dtype=np.float32) for k in ('low', 'high', 'center', 'scale')}
    sidecars = cv.load_native_sidecars(source)
    raw_episodes = cv.load_jsonl(source / 'meta/episodes.jsonl')
    raw_lengths = {int(r['episode_index']): int(r['length']) for r in raw_episodes}
    train_ids, val_ids = info['train_episodes'], info['val_episodes']
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids + val_ids) == set(raw_lengths)
    assert stats['frame_alignment'] == 'actions[0]=state[0]; actions[t]=source_action[t-1] for t>0'
    assert info['action_dim'] == stats['action_dim'] == 58
    assert raw_info['fps'] == info['fps'] == 30
    arrays, records, rgb_checks = [], [], []
    expected_paths = set()
    for split, episodes in [('train_data', train_ids), ('val_data', val_ids)]:
        for ep in episodes:
            path = args.dataset / split / 'orca' / f'episode_{ep:06d}'
            expected_paths.add(str(path.relative_to(args.dataset)))
            rgb = np.load(path / 'rgb.npy', mmap_mode='r', allow_pickle=False)
            act = np.load(path / 'actions.npy', allow_pickle=False)
            state = np.load(path / 'states.npy', allow_pickle=False)
            assert rgb.shape == (raw_lengths[ep], 1, 256, 256, 3) and rgb.dtype == np.uint8
            assert act.shape == state.shape == (len(rgb), 1, 58)
            assert act.dtype == state.dtype == np.float32
            assert np.isfinite(act).all() and np.isfinite(state).all()
            assert max(abs(act).max(), abs(state).max()) <= 1
            raw_action, raw_state, diagnostic = cv.read_action_and_state(source, raw_info, sidecars, ep, 58)
            expected_state = cv.normalize(raw_state, norm)
            expected_action = cv.align_actions_to_output_frames(cv.normalize(raw_action, norm), expected_state[0])
            a_error = float(abs(act[:, 0] - expected_action).max())
            s_error = float(abs(state[:, 0] - expected_state).max())
            assert a_error == s_error == 0, (ep, a_error, s_error)
            if split == 'train_data':
                arrays.append(raw_action)
            records.append(dict(episode=ep, split=split, frames=len(rgb), action_max_error=a_error,
                                state_max_error=s_error, **diagnostic))
            if ep in (0, 12, 61, 100, 144, 150):
                decoded = cv.decode_center_crop(cv.video_path(source, raw_info, ep, info['camera_key']), 256)
                error = int(abs(decoded.astype(np.int16) - rgb[:, 0].astype(np.int16)).max())
                assert decoded.shape == rgb[:, 0].shape and error == 0, (ep, error)
                rgb_checks.append(dict(episode=ep, frames=len(rgb), max_pixel_error=error))
            print(f'AUDITED {len(records)}/151 episode={ep} frames={len(rgb)} raw_action_error={a_error}', flush=True)
    actual_paths = {str(q.parent.relative_to(args.dataset)) for q in args.dataset.glob('*/*/*/rgb.npy')}
    assert actual_paths == expected_paths
    all_actions = np.concatenate(arrays)
    low = np.quantile(all_actions, stats['lower_quantile'], axis=0).astype(np.float32)
    high = np.quantile(all_actions, stats['upper_quantile'], axis=0).astype(np.float32)
    center = (low + high) / 2
    scale = np.where((high-low)/2 > 1e-6, (high-low)/2, 1).astype(np.float32)
    stats_errors = {k: float(abs(v - norm[k]).max()) for k, v in dict(low=low, high=high, center=center, scale=scale).items()}
    assert max(stats_errors.values()) == 0, stats_errors
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from diffsynth.trainers.dataset import RLinfDataset
    loader_checks = []
    for split, episodes in [('train_data', train_ids), ('val_data', val_ids)]:
        ds = RLinfDataset(base_path=str(args.dataset/split), action_dim=58, action2obs_bias=False,
                          retain_actions=True, max_finish_step=10**9)
        assert len(ds.data_paths) == len(episodes)
        # Interior window avoids the legacy loader's start-of-episode padding.
        item = ds[4]
        path = Path(ds.data_paths[0])
        original = np.load(path/'actions.npy', mmap_mode='r')
        assert len(item['video']) == 13 and tuple(item['action'].shape) == (13, 58)
        assert np.array_equal(item['action'][1:].numpy(), original[1:13, 0])
        assert np.array_equal(np.asarray(item['video'][1]), np.load(path/'rgb.npy', mmap_mode='r')[1, 0])
        loader_checks.append(dict(split=split, episodes=len(ds.data_paths), windows=len(ds), output_action_shape=[13,58]))
    result = dict(passed=True, dataset=str(args.dataset), source=str(source),
                  converter=str(args.converter), converter_sha256=hashlib.sha256(args.converter.read_bytes()).hexdigest(),
                  episodes=len(records), frames=sum(r['frames'] for r in records), train_episodes=len(train_ids),
                  val_episodes=len(val_ids), rgb_shape='[T,1,256,256,3]', actions_shape='[T,1,58]',
                  stats_sha256=hashlib.sha256((args.dataset/'action_stats.json').read_bytes()).hexdigest(),
                  stats_recomputed_from_training_only=stats_errors, raw_records=records,
                  rgb_redecoded=rgb_checks, loader_checks=loader_checks,
                  required_settings=dict(action_dim=58, action2obs_bias=False, retain_actions=True, Ta=8, To=4),
                  caveats=['README previously documented CHW only; loader supports HWC.',
                           'Pass train_data or val_data as the loader base path, not the dataset root.',
                           'Legacy generic loader uses zero reference/padding actions; probe uses actual initial hold.',
                           'Only six complete RGB episodes independently redecoded; all action/state frames reconstructed.',
                           'Causal hand alignment reconstructs converter logic; hardware latency calibration is not measured.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('AUDIT_PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
