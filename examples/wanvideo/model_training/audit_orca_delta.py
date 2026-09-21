#!/usr/bin/env python3
"""Independently recompute deltas/statistics and test loader alignment on every episode."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import orca_command_source as source
from diffsynth.trainers.dataset import RLinfDataset


def audit(dataset):
    info = source.load_json(dataset/'dataset_info.json')
    stats = source.load_json(dataset/'action_stats.json')
    raw_root = Path(info['source_root'])
    raw_info = source.load_json(raw_root/'meta/info.json')
    sidecars = source.load_native_sidecars(raw_root)
    scale = np.asarray(stats['scale'], np.float32)
    assert info['action2obs_bias_applied'] is False
    assert stats['action2obs_bias_applied'] is False
    assert np.array_equal(stats['center'], np.zeros(58))
    assert not (dataset/'.incomplete').exists()
    train_delta, records, checked_windows = [], [], 0
    for split, ids in [('train_data', info['train_episodes']), ('val_data', info['val_episodes'])]:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = RLinfDataset(str(dataset/split), action_dim=58, Ta=8, To=4,
                              retain_actions=True, action2obs_bias=True, max_finish_step=0)
        lookup = {(int(Path(ds.data_paths[ep]).name.split('_')[-1]), t): i for i,(ep,env,t) in enumerate(ds.sample_indices)}
        assert len(ds.data_paths) == len(ids)
        for episode in ids:
            p = dataset/split/'orca'/f'episode_{episode:06d}'
            target, state, _ = source.read_action_and_state(raw_root, raw_info, sidecars, episode, 58)
            delta = target.astype(np.float32) - state.astype(np.float32)
            expected = np.clip(delta / scale, -1, 1).astype(np.float32)
            a = np.load(p/'actions.npy')
            assert a.dtype == np.float32 and a.shape == (len(delta),1,58)
            assert np.array_equal(a[:,0], expected)
            if split == 'train_data': train_delta.append(delta)
            aligned = np.concatenate([np.zeros((1,58),np.float32), expected[:-1]])
            rgb = np.load(p/'rgb.npy', mmap_mode='r')
            template = Path(info['template_root'])/split/'orca'/p.name
            for filename in ['rgb.npy','states.npy']:
                assert np.array_equal(np.load(p/filename,mmap_mode='r'), np.load(template/filename,mmap_mode='r'))
            times = sorted({0,1,2,3,min(13,len(delta)-9),len(delta)-9})
            for t in times:
                with contextlib.redirect_stdout(io.StringIO()): item = ds[lookup[(episode,t)]]
                indices = [0] + [max(0,j) for j in range(t-3,t+9)]
                np.testing.assert_array_equal(item['action'].numpy(), aligned[indices])
                np.testing.assert_array_equal(item['action'][5:].numpy(), expected[t:t+8])
                for f, idx in zip(item['video'],indices): np.testing.assert_array_equal(np.asarray(f),rgb[idx,0])
                checked_windows += 1
            records.append({'episode':episode,'split':split,'frames':len(delta),'checked_current_frames':times,
                            'actions_sha256':hashlib.sha256((p/'actions.npy').read_bytes()).hexdigest()})
    magnitudes = np.quantile(np.abs(np.concatenate(train_delta)),stats['scale_quantile'],axis=0).astype(np.float32)
    recomputed = np.where(magnitudes>1e-6,magnitudes,1).astype(np.float32)
    np.testing.assert_array_equal(scale,recomputed)
    assert stats['scale_fit_episode_ids'] == sorted(info['train_episodes'])
    result = {'passed':True,'episodes':len(records),'frames':sum(r['frames'] for r in records),
              'loader_windows_checked':checked_windows,'raw_delta_reconstruction_exact':True,
              'first_and_last_raw_actions_retained':True,'scale_recomputed_from_training_only':True,
              'zero_preserved':True,'rgb_and_states_unchanged':True,
              'loader_future_actions_are_source_deltas_t_through_t_plus_7':True,
              'only_loader_applies_action2obs_bias':True,'records':records}
    (dataset/'adaptation_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',type=Path,required=True)
    audit(p.parse_args().dataset)
