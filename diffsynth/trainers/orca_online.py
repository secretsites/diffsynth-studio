"""Read original ORCA LeRobot data and encode abs/delta/window-relative actions on demand."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from . import orca_source as source

MODES = {'abs': 'absolute_joint_target', 'delta': 'delta_joint_target_error', 'relative': 'relative_joint_target'}
DEFAULT_VAL = [12, 13, 29, 61, 77, 90, 100, 106, 108, 109, 114, 121, 140, 142, 144]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path):
    p = Path(path); s = p.stat()
    return {'path': str(p.resolve()), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def atomic_json(path, data):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalize(values, stats):
    values = (np.asarray(values, np.float32) - np.asarray(stats['center'], np.float32)) / np.asarray(stats['scale'], np.float32)
    if stats['clip']: values = np.clip(values, -1, 1)
    return values.astype(np.float32)


def encode_window(target, state, start, mode, stats, history=4, future=8):
    if mode not in MODES: raise ValueError(f'Unknown action mode {mode}')
    if target.shape != state.shape or target.ndim != 2 or target.shape[1] != 58:
        raise ValueError('Expected matching physical target/state [T,58]')
    if start < 0 or start + future >= len(target): raise ValueError('Incomplete video prediction window')
    result = np.zeros((1 + history + future, 58), np.float32)
    for slot, frame in enumerate(range(start - history + 1, start + future + 1), 1):
        # The reference and unavailable history stay zero AFTER encoding. Exactly one temporal shift.
        if frame <= 0: continue
        index = frame - 1
        value = target[index]
        if mode == 'delta': value = value - state[index]
        elif mode == 'relative': value = value - state[start]
        result[slot] = normalize(value, stats)
    return result


def fit_stats(episodes, train_ids, mode, quantile=.99, normalization='quantile'):
    if not train_ids: raise ValueError('At least one training episode is required for normalization')
    stats = dict(mode=mode, representation=MODES[mode], train_episode_ids=list(train_ids),
                 normalization=normalization, clip=normalization != 'none', quantile=quantile)
    if normalization == 'none':
        stats.update(center=[0.] * 58, scale=[1.] * 58); return stats
    values = []
    for episode in train_ids:
        target, state = episodes(episode)
        if mode == 'abs': values.append(target)
        elif mode == 'delta': values.append(target - state)
        else:
            # Fit only training future commands under the same window-start anchor used at sampling.
            starts = np.arange(max(0, len(target) - 8))
            values.append((target[starts[:, None] + np.arange(8)] - state[starts, None]).reshape(-1, 58))
    joined = np.concatenate(values)
    if not len(joined) or not np.isfinite(joined).all(): raise ValueError('No finite training actions for fitting')
    if mode == 'abs':
        low = np.quantile(joined, 1 - quantile, axis=0).astype(np.float32)
        high = np.quantile(joined, quantile, axis=0).astype(np.float32)
        center, scale = (low + high) / 2, (high - low) / 2
    else:
        center = np.zeros(58, np.float32)
        scale = np.quantile(np.abs(joined), quantile, axis=0).astype(np.float32)
    stats.update(center=center.tolist(), scale=np.where(scale > 1e-6, scale, 1.).astype(np.float32).tolist())
    return stats


class RawOrcaSource:
    """Process-local readers backed by atomic, fingerprinted caches; never writes the source dataset."""
    def __init__(self, options):
        self.options = dict(options)
        self.root = source.resolve_dataset_root(Path(options['path']))
        self.info = source.load_json(self.root / 'meta/info.json')
        self.lengths = {int(e['episode_index']): int(e['length']) for e in source.load_jsonl(self.root / 'meta/episodes.jsonl')}
        self.sidecars = source.load_native_sidecars(self.root)
        self.val_ids = sorted(options['validation_episodes'])
        if len(self.val_ids) != len(set(self.val_ids)) or set(self.val_ids) - set(self.lengths):
            raise ValueError('Unknown or duplicate validation episode IDs')
        self.train_ids = sorted(set(self.lengths) - set(self.val_ids))
        if not self.train_ids or not self.val_ids: raise ValueError('Raw ORCA requires nonempty train and validation splits')
        self.mode = options['action_mode']
        self.cache = Path(options['cache_dir']) / fingerprint(str(self.root))[:16]
        self._actions, self._videos = OrderedDict(), OrderedDict()
        self.reader_version = hashlib.sha256((Path(__file__).read_bytes() + Path(source.__file__).read_bytes())).hexdigest()
        self.action_identities = {e: self.action_identity(e) for e in sorted(self.lengths)}
        stats_identity = dict(version=self.reader_version, mode=self.mode, train_ids=self.train_ids,
            sources=[self.action_identities[e] for e in self.train_ids], quantile=options['scale_quantile'], normalization=options['normalization'])
        if options.get('action_stats'):
            self.stats = json.loads(Path(options['action_stats']).read_text())
            expected = self.stats.get('mode', {'delta_joint_target_error': 'delta', 'absolute_joint_target': 'abs', 'relative_joint_target': 'relative'}.get(self.stats.get('representation')))
            ids = self.stats.get('train_episode_ids', self.stats.get('scale_fit_episode_ids'))
            if expected != self.mode or ids != self.train_ids: raise ValueError('Action statistics mode/training split mismatch')
            self.stats.setdefault('clip', True)
        else:
            path = self.cache / ('stats-' + fingerprint(stats_identity) + '.json')
            with locked(path.with_suffix('.lock')):
                if not path.is_file():
                    stats = fit_stats(self.actions, self.train_ids, self.mode, options['scale_quantile'], options['normalization'])
                    stats['fit_identity'] = fingerprint(stats_identity)
                    atomic_json(path, stats)
            self.stats = json.loads(path.read_text())
        if type(self.stats['clip']) is not bool: raise ValueError('Action statistics clip must be boolean')
        center, scale = np.asarray(self.stats['center'], np.float32), np.asarray(self.stats['scale'], np.float32)
        if center.shape != (58,) or scale.shape != (58,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid action normalization')
        if self.mode != 'abs' and np.any(center): raise ValueError('delta/relative normalization must preserve zero')
        # Compact semantic contract: loader storage path/cache location do not define action units.
        self.action_contract = dict(mode=self.mode, representation=MODES[self.mode], center=center.tolist(), scale=scale.tolist(),
            clip=bool(self.stats['clip']), relative_anchor='prediction_start_measured_state' if self.mode == 'relative' else None,
            action2obs_bias_applied_once=True, reference_and_negative_history_zero=True)
        self.contract = dict(format='lerobot', action_contract=self.action_contract,
            train_ids=self.train_ids, val_ids=self.val_ids, camera=options['camera_key'], image_size=256,
            source_identity=fingerprint(dict(actions=self.action_identities,
                videos={e: file_identity(self.video_path(e)) for e in sorted(self.lengths)})))

    def __getstate__(self):
        state = dict(self.__dict__); state['_actions'] = OrderedDict(); state['_videos'] = OrderedDict(); return state

    def action_identity(self, episode):
        sidecar = self.sidecars[episode]
        return [file_identity(source.parquet_path(self.root, self.info, episode)),
                file_identity(sidecar / 'hand_telemetry.parquet'), file_identity(sidecar / 'raw/clock_samples.jsonl')]

    def actions(self, episode):
        if episode in self._actions:
            self._actions.move_to_end(episode); return self._actions[episode]
        key = fingerprint(dict(version=self.reader_version, sources=self.action_identities[episode]))
        path = self.cache / 'actions' / f'{episode:06d}-{key}.npz'
        with locked(path.with_suffix('.lock')):
            if not path.is_file():
                target, state, _ = source.read_action_and_state(self.root, self.info, self.sidecars, episode, 58)
                if len(target) != self.lengths[episode]: raise ValueError('Raw action length differs from metadata')
                tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f: np.savez(f, target=target, state=state)
                tmp.replace(path)
        with np.load(path) as data: result = data['target'], data['state']
        self._actions[episode] = result
        while len(self._actions) > 8: self._actions.popitem(last=False)
        return result

    def video_path(self, episode):
        return self.root / self.info['video_path'].format(episode_chunk=episode // int(self.info['chunks_size']),
            episode_index=episode, video_key=self.options['camera_key'])

    def frames(self, episode):
        if episode in self._videos:
            self._videos.move_to_end(episode); return self._videos[episode]
        video = self.video_path(episode)
        key = fingerprint(dict(video=file_identity(video), transform='center-square-lanczos-256-v1'))
        path = self.cache / 'rgb' / f'{episode:06d}-{key}.npy'
        with locked(path.with_suffix('.lock')):
            if not path.is_file():
                stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height', '-of', 'json', str(video)]))['streams'][0]
                w, h = stream['width'], stream['height']; side = min(w, h)
                filt = f'crop={side}:{side}:{(w-side)//2}:{(h-side)//2},scale=256:256:flags=lanczos'
                decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
                    '-map', '0:v:0', '-vf', filt, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
                expected = self.lengths[episode] * 256 * 256 * 3
                if len(decoded) != expected: raise ValueError(f'Video length mismatch for episode {episode}')
                frames = np.frombuffer(decoded, np.uint8).reshape(-1, 256, 256, 3)
                tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f: np.save(f, frames)
                tmp.replace(path)
        result = np.load(path, mmap_mode='r')
        self._videos[episode] = result
        while len(self._videos) > 2: self._videos.popitem(last=False)
        return result

    def window(self, episode, start):
        return encode_window(*self.actions(episode), start, self.mode, self.stats)


class OnlineOrcaDataset(torch.utils.data.Dataset):
    def __init__(self, options, split, shared=None):
        self.reader = shared or RawOrcaSource(options)
        ids = self.reader.train_ids if split == 'train_data' else self.reader.val_ids
        self.ids = ids; self.load_from_cache = False
        self.episode_info = [(str(self.reader.root / split / 'orca' / f'episode_{e:06d}'), self.reader.lengths[e] - 1, self.reader.lengths[e], 1) for e in ids]
        self.sample_indices = [(i, 0, t) for i, e in enumerate(ids) for t in range(self.reader.lengths[e] - 8)]
        self.action_contracts = [dict(root=str(self.reader.root), representation=MODES[self.reader.mode], action2obs_bias=True, action_dim=58)]
        self.action_mode = self.reader.mode
        self.contract = self.reader.contract

    def __len__(self): return len(self.sample_indices)

    def __getitem__(self, index):
        ep_index, _, start = self.sample_indices[index]
        episode = self.ids[ep_index]
        frames = self.reader.frames(episode)
        ids = [0] + [max(0, t) for t in range(start - 3, start + 9)]
        video = [Image.fromarray(frames[t]) for t in ids]
        action = self.reader.window(episode, start)
        static = np.zeros_like(action)
        if self.action_mode == 'abs':
            # Static augmentation repeats reference frame 0, so ABS targets must hold measured state[0].
            hold = normalize(self.reader.actions(episode)[1][0], self.reader.stats)
            for slot, frame in enumerate(range(start - 3, start + 9), 1):
                if frame > 0: static[slot] = hold
        return dict(video=video, reference_image=[video[0]], action=torch.from_numpy(action),
                    static_action=torch.from_numpy(static), action_mode=self.action_mode)


def build_dataset(options, split, shared=None):
    if options.get('format', 'npy') == 'lerobot': return OnlineOrcaDataset(options, split, shared)
    from .dataset import RLinfDataset
    if options.get('action_mode', 'delta') != 'delta':
        raise ValueError('Online abs/relative processing requires format=lerobot and original raw data; converted NPY is already encoded')
    ds = RLinfDataset(str(Path(options['path']) / split), action_dim=58, action2obs_bias=True,
                     retain_actions=True, Ta=8, To=4, stride=1, max_finish_step=0)
    if not ds.action_contracts or any(c['representation'] != MODES['delta'] for c in ds.action_contracts):
        raise ValueError('NPY mode requires the audited unshifted delta dataset; use lerobot for raw abs/delta/relative')
    ds.action_mode = 'delta'
    return ds


def action_contract_for(options, reader=None):
    if options.get('format', 'npy') == 'lerobot':
        return (reader or RawOrcaSource(options)).action_contract
    stats = json.loads((Path(options['path']) / 'action_stats.json').read_text())
    return dict(mode='delta', representation=MODES['delta'], center=stats['center'], scale=stats['scale'],
                clip=bool(stats.get('clip', True)), relative_anchor=None,
                action2obs_bias_applied_once=True, reference_and_negative_history_zero=True)
