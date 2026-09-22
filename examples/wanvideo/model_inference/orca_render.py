"""Audit any configured ORCA rollout and render synchronized comparison artifacts."""
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import numpy as np
from orca_config import write_json
from orca_rollout import ah, digest, history_indices, action_window, load_episode, CHUNK


def mse(a, b, mask=None):
    error = ((a.astype(np.float32) - b.astype(np.float32)) / 255) ** 2
    return float((error if mask is None else error[mask]).mean())


def audit_and_render(config, render=True):
    out = Path(config['output']); plan = json.loads((out / 'experiment.json').read_text())
    job = json.loads((out / 'job_status.json').read_text())
    modes, variants = plan['modes'], plan['actions']
    tags = [f'{mode}_{v}' for mode in modes for v in variants]
    assert job['status'] == 'completed' and set(job['returncodes']) == set(tags) and not any(job['returncodes'].values())
    n, start, env = plan['predicted_frames'], plan['start_frame'], plan['environment_index']
    if 'source_provenance' in plan:
        rgb, real_commands, contract, _, provenance = load_episode(config)
        assert contract == plan['action_contract'] and provenance == plan['source_provenance']
        assert ah(rgb) == plan['source_rgb_array_sha256']
        assert np.array_equal(rgb, np.load(out / 'source_rgb.npy', mmap_mode='r'))
    else:  # Re-render earlier YAML experiments without rewriting their recorded inputs.
        rgb = np.load(Path(plan['source']) / 'rgb.npy', mmap_mode='r')[:, env]
        raw_actions = np.load(Path(plan['source']) / 'actions.npy', mmap_mode='r')[:, env]
        real_commands = np.stack([action_window(raw_actions, t) for t in range(start, start + n, CHUNK)])
        assert digest(Path(plan['source']) / 'rgb.npy') == plan['rgb_sha256']
        assert digest(Path(plan['source']) / 'actions.npy') == plan['actions_sha256']
    gt = np.load(out / 'ground_truth.npy', mmap_mode='r')
    initial = np.load(out / 'initial_context.npy')
    commands = {v: np.load(out / f'commands_{v}.npy') for v in variants}
    arrays = {tag: np.load(out / f'{tag}.npy', mmap_mode='r') for tag in tags}
    assert np.array_equal(gt, rgb[start + 1:start + n + 1])
    assert np.array_equal(initial, rgb[history_indices(start)])
    assert digest(plan['checkpoint']) == plan['checkpoint_sha256']
    if 'real' in commands: assert ah(commands['real']) == plan['command_array_sha256']
    records = {}
    for tag, array in arrays.items():
        mode, v = tag.split('_')
        assert array.shape == (n, 256, 256, 3) and array.dtype == np.uint8
        state = json.loads((out / f'{tag}_status.json').read_text())
        assert state['state'] == 'completed' and digest(out / f'{tag}.npy') == state['output_sha256']
        assert json.loads((out / f'{tag}_runtime.json').read_text())['actual_checkpoint_weights_exact']
        rows = [json.loads(line) for line in (out / f'{tag}_chunks.jsonl').read_text().splitlines()]
        assert len(rows) == n // CHUNK
        for k, row in enumerate(rows):
            offset, t = k * CHUNK, start + k * CHUNK
            expected = real_commands[k] if v == 'real' else np.zeros((13, 58), np.float32)
            assert row['current_frame'] == t and row['future_indices'] == list(range(t + 1, t + 9))
            assert row['history_indices'] == history_indices(t)[1:]
            assert np.array_equal(expected, commands[v][k]) and row['commands_sha256'] == ah(expected)
            context = rgb[history_indices(t)] if mode == 'tf' else initial if not k else np.concatenate([initial[:1], array[offset - 4:offset]])
            assert ah(context) == row['context_sha256']
            assert ah(array[offset:offset + CHUNK]) == row['prediction_sha256']
        records[tag] = rows
    for k in range(n // CHUNK):
        assert len({records[tag][k]['noise_sha256'] for tag in tags}) == 1
        if 'tf' in modes: assert len({records[f'tf_{v}'][k]['context_sha256'] for v in variants}) == 1
    assert len({records[tag][0]['context_sha256'] for tag in tags}) == 1
    if set(modes) == {'ar', 'tf'}:
        for v in variants: assert np.array_equal(arrays[f'ar_{v}'][:CHUNK], arrays[f'tf_{v}'][:CHUNK])
    change = sum(np.abs(frame.astype(np.float32) - initial[-1].astype(np.float32)).mean(2) / (255 * n) for frame in gt)
    mask = change > .035
    if not mask.any(): mask[:] = True
    per_frame, summaries, pairs = {}, {}, {}
    for tag, frames in arrays.items():
        mode, variant = tag.split('_')
        series = dict(gt_rgb_mse=[], chunk_start_motion_mse=[], within_chunk_step_motion_mse=[])
        for i, pred in enumerate(frames):
            offset = (i // CHUNK) * CHUNK
            anchor = rgb[start + offset] if mode == 'tf' else initial[-1] if not offset else frames[offset - 1]
            prev = anchor if i == offset else frames[i - 1]
            series['gt_rgb_mse'].append(mse(pred, gt[i]))
            series['chunk_start_motion_mse'].append(mse(pred, anchor, mask))
            series['within_chunk_step_motion_mse'].append(mse(pred, prev, mask))
        per_frame[tag] = series
        summaries[tag] = {key: float(np.mean(value)) for key, value in series.items()}
    if set(variants) == {'real', 'zero'}:
        for mode in modes:
            values = [mse(a, b) for a, b in zip(arrays[f'{mode}_real'], arrays[f'{mode}_zero'])]
            pairs[mode] = dict(mean_rgb_mse=float(np.mean(values)), per_frame_rgb_mse=values)
    write_json(out / 'metrics.json', dict(summaries=summaries, per_frame=per_frame, pairs=pairs,
        motion_mask_fraction=float(mask.mean()), metric_scope='RGB/255 MSE, not joint angles. No zero-action counterfactual ground truth.',
        motion_mask='GT-only mean absolute change from initial current frame > .035; all pixels if mask empty'))
    checks = render_artifacts(config, plan, arrays, gt, initial, rgb) if render else []
    verification = dict(passed=True, chunks_verified=len(tags) * n // CHUNK, predicted_frames=n,
        same_initial_image_history=True, same_noise_every_matching_chunk=True,
        same_tf_image_history_every_matching_chunk='tf' in modes, model_weights_exact=True,
        no_end_padding=True, contexts_actions_predictions_hash_verified=True, video_checks=checks,
        checkpoint_sha256=plan['checkpoint_sha256'])
    write_json(out / 'verification.json', verification)
    table = '\n'.join(f'| {tag} | {s["gt_rgb_mse"]:.6f} | {s["chunk_start_motion_mse"]:.6f} |' for tag, s in summaries.items())
    links = '\n'.join(f'- [{v["file"]}]({out / v["file"]})' for v in checks)
    (out / 'README.md').write_text(f'''# ORCA rollout: {config['rollout']['episode']}

权重 `{plan['checkpoint']}`，SHA256 `{plan['checkpoint_sha256']}`。起始帧{start}，预测未来{n}帧，共{n//8}段×8帧。

TF每段用真实过去图像；AR首段之后反馈自身生成图像；两者固定reference第0帧。真实/全零组每段噪声相同。全零动作替换全部13×58输入，包括历史槽。真实组使用 `{plan.get('action_mode', 'delta')}` 编码：abs=目标，delta=目标−同时刻实际state，relative=目标−本段预测起点实际state。state均来自记录，不从生成图像重算。abs的归一化全零不是保持姿态命令。

## 输出

{links}

## 像素诊断

| 模式/动作 | 对真实录像RGB MSE | 运动区域相对每段起点MSE |
| --- | ---: | ---: |
{table}

只有真实动作对应的真实录像，没有全零动作的真实反事实结果。像素变化包括形变和光照，不能当作关节位移。TF拼接视频每8帧接回真实历史，不用于判定长程自主静止。动作敏感性也不能单独证明正确关节控制。

已核验{verification['chunks_verified']}个chunk及输出视频；配置见 [resolved_config.yaml]({out/'resolved_config.yaml'})，协议见 [experiment.json]({out/'experiment.json'})，逐帧指标见 [metrics.json]({out/'metrics.json'})，核验见 [verification.json]({out/'verification.json'})。
''')
    return verification


def render_artifacts(config, plan, arrays, gt, initial, rgb):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    out = Path(config['output']); n = plan['predicted_frames']; start = plan['start_frame']
    variants, fps = plan['actions'], config['render']['fps']
    @lru_cache(None)
    def font(size): return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size)
    def text(d, xy, s, size=20): d.text(xy, s, font=font(size), fill='#233348')
    def frame(tag, offset): return initial[-1] if offset == 0 else gt[offset - 1] if tag == 'gt' else arrays[tag][offset - 1]
    def composite(mode, offset, slow_start=None):
        paired = set(variants) == {'real', 'zero'}
        pictures = [('真实录像', frame('gt', offset))]
        for v in variants: pictures.append(('真实动作' if v == 'real' else '全零动作', frame(f'{mode}_{v}', offset)))
        if slow_start is not None:
            anchor = rgb[start + slow_start]
            if offset == slow_start: pictures = [(label, anchor) for label, _ in pictures]
            pictures.insert(0, ('本段共同起点', anchor))
        elif paired:
            diff = np.abs(frame(f'{mode}_real', offset).astype(np.float32) - frame(f'{mode}_zero', offset).astype(np.float32)).mean(2)
            diff = np.uint8(np.clip(diff * config['render']['difference_gain'], 0, 255))
            pictures.append((f'像素差 ×{config["render"]["difference_gain"]:g}', np.repeat(diff[:, :, None], 3, axis=2)))
        im = Image.new('RGB', (24 + 334 * len(pictures), 510), '#f7f9fc'); d = ImageDraw.Draw(im)
        text(d, (16, 10), f'{Path(config["rollout"]["episode"]).name} · {mode.upper()} · {Path(plan["checkpoint"]).name}', 26)
        text(d, (16, 51), f'源帧 {start+offset} · 预测进度 {offset}/{n} · 相同逐段噪声', 22)
        for j, (label, array) in enumerate(pictures):
            x = 16 + 334 * j; text(d, (x, 90), label, 23)
            im.paste(Image.fromarray(array).resize((320, 320), Image.Resampling.NEAREST), (x, 128))
        label = '每段接回同一组真实历史图像；比较段内8帧。' if mode == 'tf' else '相同初始历史；后续各组反馈自身生成图像。'
        text(d, (16, 455), label, 21)
        text(d, (16, 486), '全零组历史动作槽也置零；像素差异不是关节角度。', 17)
        return im
    videos = []
    def encode(name, frames, count, rate=fps):
        path = out / name
        with imageio.get_writer(path, fps=rate, codec='libx264', quality=9, macro_block_size=2,
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for f in frames: writer.append_data(np.asarray(f))
        videos.append((path, count, rate))
    for tag, frames in arrays.items(): encode(f'{tag}.mp4', frames, n)
    for mode in plan['modes']:
        encode(f'{mode}_comparison.mp4', (composite(mode, t) for t in range(n + 1)), n + 1)
        times = sorted(set([0, min(8, n), min(24, n), n]))
        for t in times: composite(mode, t).save(out / f'{mode}_f{start+t:06d}.png')
        if mode == 'tf':
            chunks = np.unique(np.linspace(0, n // 8 - 1, min(6, n // 8), dtype=int))
            def slowed():
                for k in chunks:
                    offset = int(k) * 8
                    for _ in range(8): yield composite(mode, offset, offset)
                    for step in range(1, 9): yield composite(mode, offset + step, offset)
                    for _ in range(8): yield composite(mode, offset + 8, offset)
            encode('tf_chunks_slow.mp4', slowed(), len(chunks) * 24, 8)
    checks = []
    for path, count, rate in videos:
        meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate', '-of', 'json', str(path)]))['streams'][0]
        assert int(meta['nb_frames']) == count and meta['avg_frame_rate'] == f'{rate}/1'
        decoded = 0
        with imageio.get_reader(path) as reader:
            for f in reader:
                assert f.shape == (meta['height'], meta['width'], 3); decoded += 1
        assert decoded == count
        checks.append(dict(file=path.name, **meta, decoded_frames=decoded, sha256=digest(path)))
    return checks
