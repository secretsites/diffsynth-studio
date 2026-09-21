from __future__ import annotations

import imageio
import json
import os
from pathlib import Path
import warnings
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm


def validate_action_contract(base_path, action_dim, action2obs_bias):
    """Check declared ORCA semantics before the loader applies temporal alignment."""
    roots = base_path if isinstance(base_path, list) else [base_path]
    contracts = []
    for path in roots:
        root = Path(path).resolve().parent
        if (root / '.incomplete').exists():
            raise ValueError(f'Dataset conversion is incomplete: {root}')
        info_path, stats_path = root / 'dataset_info.json', root / 'action_stats.json'
        if not info_path.is_file() or not stats_path.is_file():
            continue
        info = json.loads(info_path.read_text())
        stats = json.loads(stats_path.read_text())
        if info.get('action_dim') != action_dim:
            raise ValueError(f'action_dim={action_dim} differs from dataset metadata: {root}')
        representation = info.get('action_representation')
        if representation == 'delta_joint_target_error':
            if info.get('action2obs_bias_applied') is not False or stats.get('action2obs_bias_applied') is not False:
                raise ValueError('Delta files must be unshifted; action2obs_bias belongs in the loader')
            if not action2obs_bias:
                raise ValueError('Unshifted delta data requires action2obs_bias=True')
            center = np.asarray(stats.get('center', []))
            scale = np.asarray(stats.get('scale', []))
            if center.shape != (action_dim,) or np.any(center != 0):
                raise ValueError('Delta normalization must preserve zero: center must be all zero')
            if scale.shape != (action_dim,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('Invalid delta action scales')
        elif action2obs_bias and stats.get('frame_alignment', '').startswith('actions[0]=state[0]'):
            raise ValueError('This absolute command dataset is already shifted; refusing double action2obs_bias')
        contracts.append({'root': str(root), 'representation': representation,
                          'action2obs_bias': action2obs_bias, 'action_dim': action_dim})
    return contracts


class RLinfDataset(torch.utils.data.Dataset):
    """
    dataset_base_path/
        rgb.npy      : [T, N, 3, H, W]
        actions.npy  : [T, N, action_dim]

    Args:
        base_path: Root dataset path; supports either a single path or a list of paths.
        Ta: Action prediction horizon length (number of future action steps).
        To: Observation context length (number of historical observation steps).
        retain_reference_image: Whether to keep frame 0 as the reference image (required by current training pipeline).
        retain_actions:
            If False, the action window uses [start+1, start+Ta+1).
            If True, the action window is aligned with the observation window [vs, ve).
            In both modes, the final action sequence is padded to length Ta + To + 1.
        stride: Sliding-window stride for sampling.
        action_dim: Action dimension (used for zero padding).
        max_finish_step: Maximum usable end step per trajectory; 0 uses the full trajectory.
        repeat: Dataset repetition factor.
        action2obs_bias:
            Whether to enable action->observation alignment bias.
            If True, the action sequence is shifted right by one step with a leading zero action:
                a'[0] = 0, a'[t] = a[t-1]
            This aligns datasets recorded as (o_t, a_t) with the world-model training pair format (a_t -> o_{t+1}).
    """
    def __init__(
        self,
        base_path: Optional[list[str]] = None,
        Ta: int = 8,
        To: int = 4, # 4
        retain_reference_image: bool = True,
        retain_actions: bool = True,
        stride: int = 1,
        action_dim: int = 7,
        max_finish_step: int = 440,
        repeat: int = 1,
        action2obs_bias: bool = False,
    ):
        super().__init__()
        self.load_from_cache = False
        self.retain_reference_image = retain_reference_image
        self.retain_actions = retain_actions

        if not self.retain_reference_image:
            raise ValueError("retain_reference_image must be True; we use first image as reference image")

        self.Ta, self.To, self.stride = Ta, To, stride
        self.action_dim = action_dim
        self.max_finish_step = max_finish_step
        self.repeat = repeat
        self.action2obs_bias = action2obs_bias
        if min(Ta, To, stride, action_dim) < 1 or max_finish_step < 0:
            raise ValueError('Ta, To, stride and action_dim must be positive; max_finish_step must be nonnegative')
        self.action_contracts = validate_action_contract(base_path, action_dim, action2obs_bias)

        # Scan trajectory directories containing rgb.npy and actions.npy.
        # 遵循与 MyNpyDatasetnew 相同的目录结构：base_path/step_name/video/eval/seed_name
        self.data_paths = []

        print(f"base_path: {base_path}")
        
        if isinstance(base_path, list):
            for base_path in base_path:
                for step_name in sorted(os.listdir(base_path)):
                    step_path = os.path.join(base_path, step_name)
                    if not os.path.isdir(step_path):
                        continue
                    for seed_name in sorted(os.listdir(step_path)):
                        seed_path = os.path.join(step_path, seed_name)
                        if os.path.isdir(seed_path):
                            self.data_paths.append(seed_path)
        else:
            for step_name in sorted(os.listdir(base_path)):
                # step_path = os.path.join(base_path, step_name, "video/eval")
                step_path = os.path.join(base_path, step_name)
                if not os.path.isdir(step_path):
                    continue
                for seed_name in sorted(os.listdir(step_path)):
                    seed_path = os.path.join(step_path, seed_name)
                    if os.path.isdir(seed_path):
                        self.data_paths.append(seed_path)
      
        if len(self.data_paths) == 0:
            raise FileNotFoundError(f"在 '{base_path}' 没找到任何包含 rgb.npy 和 actions.npy 的目录")
        
        print(f'[RLinfDataset] Found {len(self.data_paths)} data paths under {base_path}.')

        # 预存每个 episode 的信息
        # RGB accepts CHW or HWC; actions are [T, N, action_dim].
        self.episode_info = []  # 每个元素是 (data_path, finish_step, T, N)
        self.sample_indices = []  # 每个元素是 (episode_idx, env_id, start)
        
        for data_path in self.data_paths:
            # 加载 video 和 action 的形状信息（不加载完整数据）
            video_shape = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r').shape
            T, N = video_shape[0], video_shape[1]  # (T, N, C, H, W)

            action_array = np.load(os.path.join(data_path, 'actions.npy'), mmap_mode='r')
            if action_array.shape != (T, N, self.action_dim):
                raise ValueError(f'Action shape {action_array.shape} != {(T, N, self.action_dim)} in {data_path}')
            finish_step = min(T - 1, self.max_finish_step) if self.max_finish_step > 0 else T - 1
            
            episode_idx = len(self.episode_info)
            self.episode_info.append((data_path, finish_step, T, N))
            
            # 为每个环境和每个可能的窗口生成样本
            for env_id in range(N):
                for start in range(0, finish_step - self.Ta + 1, self.stride):
                    self.sample_indices.append((episode_idx, env_id, start))
        
        self.length = len(self.sample_indices) * repeat
        print(f"[RLinfDataset] 总共生成 {len(self.sample_indices)} 个样本，repeat={repeat}，总长度={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 计算实际的样本索引
        sample_idx = idx % len(self.sample_indices)
        episode_idx, env_id, start = self.sample_indices[sample_idx]
        data_path, finish_step, T, N = self.episode_info[episode_idx]
        # 懒加载当前 npy 文件
        video_np = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r')  # (T, N, C, H, W)
        action_np = np.load(os.path.join(data_path, "actions.npy"), mmap_mode='r')
        if self.action2obs_bias:
            # Align to (a_t -> o_{t+1}): shift actions right with a leading zero action.
            zero_action = np.zeros((1, action_np.shape[1], action_np.shape[2]), dtype=action_np.dtype)
            action_np = np.concatenate([zero_action, action_np[:-1]], axis=0)

        vs, ve = start - self.To + 1, start + self.Ta + 1

        if not self.retain_actions:
            action_s, action_e = start + 1, start + self.Ta + 1
        else:
            action_s, action_e = vs, ve

        if len(video_np) < ve:
            raise ValueError(f"video length is not right: {len(video_np)} < {ve}")
        
        if vs < 0:
            # padding the first image To 
            pad = np.repeat(video_np[0:1, env_id], -vs, axis=0)
            vid_win = np.concatenate([pad, video_np[:ve, env_id]], axis=0)
        else:
            vid_win = video_np[vs:ve, env_id]  # [ve-vs, C, H, W]

        if self.retain_reference_image:
            vid_win = np.concatenate([video_np[0:1, env_id], vid_win], axis=0)  # [1+ve-vs, C, H, W]

        if action_s < 0:
            # padding the zero action To times 
            pad = np.zeros((-action_s, self.action_dim), dtype=action_np.dtype)
            act_win = np.concatenate([pad, action_np[:action_e, env_id]], axis=0)
        else:
            act_win = action_np[action_s:action_e, env_id]  # [action_e-action_s, action_dim]
        
        left_padding_length = 1 if self.retain_actions else self.To + 1       
        act_win = np.concatenate([np.zeros((left_padding_length, self.action_dim), dtype=action_np.dtype), act_win], axis=0)

        action_tensor = torch.from_numpy(act_win).float()
        
        # 确保 vid_win 的形状正确
        assert vid_win.shape[0] == self.Ta + self.To + 1, f"video len is not right: {vid_win.shape[0]} != {self.Ta + self.To}"      
        # retain_actions True/False 最终长度都固定为 Ta + To + 1
        assert act_win.shape[0] == self.Ta + self.To + 1, f"action len is not right: {act_win.shape[0]} != {self.Ta + self.To + 1}"
        

        if os.environ.get('WAN_DEBUG', '0').lower() in {'1', 'true', 'yes', 'on'}:
            print("action start, action end is ", action_s, action_e)
            print("video start, video end is ", vs, ve)


        # 转换为与 MyNpyDatasetnew 对齐的格式
        # video: List[PIL.Image]，每个图像是 uint8
        video_list = []
        for frame in vid_win:
            # frame shape: (C, H, W) -> 转换为 (H, W, C)
            if frame.shape[0] == 3:  # CHW 格式
                frame = frame.transpose(1, 2, 0)  # CHW → HWC
            # 确保是 uint8
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255)
            img = frame.astype(np.uint8)
            video_list.append(Image.fromarray(img))

        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }

class SimpleVLARealWorldRLinfDataset(torch.utils.data.Dataset):
    """
    Real-world RL dataset wrapper aligned with RLinfDataset action/video formatting.

    Args:
        base_path: Root path(s) containing trajectory .npy files.
        Ta: Action prediction horizon length.
        To: Observation context length.
        reference_image: Whether to prepend frame 0 as reference image.
        stride: Sliding-window stride.
        action_dim: Target action dimension (used for zero padding/truncation if needed).
        max_finish_step: Reserved for compatibility.
        repeat: Dataset repetition factor.
        retain_actions:
            False -> use action window [start, start+Ta).
            True  -> use action window aligned to observation [start-To, start+Ta).
            In both modes, final action length is padded to Ta + To + 1.
        action2obs_bias:
            If True, shift action sequence right by one with leading zero action:
                a'[0] = 0, a'[t] = a[t-1]
            This aligns (o_t, a_t) recordings to (a_t -> o_{t+1}) supervision.
    """
    def __init__(
        self,
        base_path: str | list[str] | tuple[str, ...],
        Ta: int = 8,
        To: int = 4, # 4
        reference_image: bool = True,
        stride: int = 1,
        max_finish_step: int = 0,
        repeat: int = 1,
        retain_actions: bool = False,
        action2obs_bias: bool = True,
        retain_action: Optional[bool] = None,
    ):
        super().__init__()
        self.load_from_cache = False
        self.reference_image = reference_image
        self.retain_actions = retain_actions
        self.action2obs_bias = action2obs_bias

        if not self.reference_image:
            To = To + 1 # 不包含 reference_image 时，To 需要 +1, 即使用五帧

        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = self.Ta + self.To
        self.max_finish_step = max_finish_step
        self.repeat = repeat

        # 扫描 base_path 下的所有 .npy 文件
        # 每个 .npy 文件包含一个轨迹，格式为：数组[(T,)] 其中每个元素是 dict{'observations': (H,W,C), 'actions': (action_dim,)}
        self.data_paths = []
        
        base_paths = list(base_path) if isinstance(base_path, (list, tuple)) else [base_path]
        for current_base_path in base_paths:
            for filename in sorted(os.listdir(current_base_path)):
                if filename.endswith('.npy'):
                    file_path = os.path.join(current_base_path, filename)
                    self.data_paths.append(file_path)
      
        if len(self.data_paths) == 0:
            raise FileNotFoundError(f"在 '{base_paths}' 没找到任何 .npy 文件")
        
        print(f'[SimpleVLARealWorldRLinfDataset] Found {len(self.data_paths)} trajectory files under {base_paths}.')

        # 预存每个 episode 的信息
        # 数据格式：每个 .npy 文件是 [T] 个 dict，每个 dict 包含 'observations' 和 'actions'
        # 每个元素是 (file_path, finish_step, T, action_seq)
        self.sample_indices = []  # 每个元素是 (episode_idx, start)
        self.episode_info = []  # 每个元素是 (file_path, finish_step, T, action_seq)
        
        for file_path in self.data_paths:
            # 加载轨迹的长度信息（包含 Python 对象的数组不能使用 mmap）
            traj_data = np.load(file_path, allow_pickle=True)
            T = len(traj_data)  # 轨迹长度
            finish_step = T - 1
            
            # 在 init 阶段完成 action 预处理（含可选 action2obs_bias 移位），
            # __getitem__ 直接使用缓存，避免每个 sample 重复移位。
            action_seq = np.array([traj_data[i]['actions'] for i in range(T)])
            action_dim = action_seq.shape[1]
            # print("action dim is ", action_seq.shape)
            # if action_seq.ndim == 1:
            #     action_seq = action_seq[:, None]
            if self.action2obs_bias:
                zero_action = np.zeros((1, action_dim), dtype=action_seq.dtype)
                action_seq = np.concatenate([zero_action, action_seq[:-1]], axis=0)

            episode_idx = len(self.episode_info)
            self.episode_info.append((file_path, finish_step, T, action_seq))
            
            # 为每个可能的窗口生成样本
            for start in range(0, finish_step - self.Ta + 1, self.stride):
                self.sample_indices.append((episode_idx, start))
        
        self.length = len(self.sample_indices) * repeat
        print(f"[SimpleVLARealWorldRLinfDataset] 总共生成 {len(self.sample_indices)} 个样本，repeat={repeat}，总长度={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 计算实际的样本索引
        sample_idx = idx % len(self.sample_indices)
        episode_idx, start = self.sample_indices[sample_idx]
        file_path, finish_step, T, action_seq = self.episode_info[episode_idx]
        
        # 懒加载当前轨迹文件（包含 Python 对象的数组不能使用 mmap）
        traj_data = np.load(file_path, allow_pickle=True)  # [T] 个 dict

        # 采样逻辑：遵循 SimpleVLARLinfDataset 的采样逻辑
        vs, ve = start - self.To + 1, start + self.Ta + 1
        if not self.retain_actions:
            action_s, action_e = start + 1, start + self.Ta + 1
        else:
            action_s, action_e = vs, ve

        if len(traj_data) < ve:
            raise ValueError(f"trajectory length is not right: {len(traj_data)} < {ve}")
        
        # 提取视频帧和动作
        if vs < 0:
            # 直接使用前面的帧进行 padding
            pad_frames = [traj_data[0]['observations'] for _ in range(self.To)]
            vid_frames = [traj_data[i]['observations'] for i in range(self.Ta)]
            vid_win = pad_frames + vid_frames
        else:
            vid_win = [traj_data[i]['observations'] for i in range(vs, ve)]

        if self.reference_image:
            # 添加第0帧作为参考图像
            vid_win = [traj_data[0]['observations']] + vid_win
        
        action_dim = action_seq.shape[1]

        if action_s < 0:
            # padding the zero action To times
            pad = np.zeros((self.To, action_dim), dtype=action_seq.dtype)  # [To, action_dim]
            act_win = np.concatenate([pad, action_seq[:self.Ta]], axis=0)  # [action_e-action_s, action_dim]
        else:
            act_win = action_seq[action_s:action_e]  # [action_e-action_s, action_dim]

        left_padding_length = 1 if self.retain_actions else self.To + 1
        act_win = np.concatenate(
            [np.zeros((left_padding_length, action_dim), dtype=action_seq.dtype), act_win],
            axis=0
        )  # [Ta + To + 1, action_dim]
        action_tensor = torch.from_numpy(act_win).float()
        
        # 确保长度正确
        if self.reference_image:
            expected_len = self.Ta + self.To + 1
            assert len(vid_win) == expected_len, f"video len is not right: {len(vid_win)} != {expected_len}"
        else:
            expected_len = self.Ta + self.To
            assert len(vid_win) == expected_len, f"video len is not right: {len(vid_win)} != {expected_len}"
        assert act_win.shape[0] == self.Ta + self.To + 1, f"action len is not right: {act_win.shape[0]} != {self.Ta + self.To + 1}"
        
        # 转换为与 MyNpyDatasetnew 对齐的格式
        # video: List[PIL.Image]，每个图像是 uint8
        video_list = []
        for frame in vid_win:
            # frame shape: 应该已经是 (H, W, C)
            # 确保是 uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255)
            img = frame.astype(np.uint8)
            video_list.append(Image.fromarray(img))

        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }

# python -m diffsynth.trainers.dataset
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="SimpleVLARealWorldRLinfDataset", choices=["RLinfDataset", "SimpleVLARealWorldRLinfDataset"])
    parser.add_argument("--Ta", type=int, default=8)
    parser.add_argument("--To", type=int, default=4)
    # for real-world
    parser.add_argument("--base_path", type=str, default="/mnt/project_rlinf/jzn/dataset/agx_3task/agx_3tasks_base_policy_rollout/fold_towel_eef_infer_data_3task_fold_towel_clean_process")
    # for simulation
    # parser.add_argument("--base_path", type=str, default='/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_spatial/base_policy_rollout/train_data')
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    if args.dataset_type == "RLinfDataset":
        dataset = RLinfDataset(base_path=args.base_path, repeat=args.repeat)
    elif args.dataset_type == "SimpleVLARealWorldRLinfDataset":
        dataset = SimpleVLARealWorldRLinfDataset(base_path=args.base_path, repeat=args.repeat)

    print(f"dataset_type={args.dataset_type} len={len(dataset)}")
    probe_indices = [0, 1, 2, 10, 100][: args.num_samples]
    for i in probe_indices:
        sample = dataset[i]
        action = sample['action']
        print(
            f"idx={i} video_len={len(sample['video'])} action_shape={tuple(sample['action'].shape)} "
            f"ref_len={len(sample['reference_image'])}"
        )
        print("")
