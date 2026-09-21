#!/usr/bin/env python3
"""Bounded, paired action-conditioning probe for the ORCA 58D command dataset.

Uses the repository's Wan DiT/model_fn and VAE. Clean five-frame context, eight
future frames; no future RGB is used for generation. Actions are ALREADY aligned.
This is a small-data full-parameter experiment, not evidence of convergence.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

VARIANTS = ('real', 'hold', 'swap', 'arm_swap', 'hand_swap', 'reverse', 'delta_swap')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def episode_path(dataset, split, episode):
    return Path(dataset) / split / 'orca' / f'episode_{episode:06d}'


def window_arrays(dataset, row):
    p = episode_path(dataset, row['split'], row['episode'])
    s = row['start']
    ids = [0] + list(range(s, s + 12))
    rgb = np.load(p / 'rgb.npy', mmap_mode='r')[ids, 0].copy()
    action = np.load(p / 'actions.npy', mmap_mode='r')[ids, 0].copy()
    state = np.load(p / 'states.npy', mmap_mode='r')[s + 3, 0].copy()
    # Index 0 is a real episode reference with its actual initial hold command.
    # Four recent context frames occupy 1:5, prediction frames occupy 5:13.
    return rgb, action, state


def intervention(action, state, donor, variant):
    out = action.clone()
    if variant == 'hold':
        out[5:] = state
    elif variant == 'reverse':
        out[5:] = action[5:].flip(0)
    elif variant == 'delta_swap':
        # Transfer the donor's command increments, anchored at this window's
        # latest command, to reduce a cross-episode absolute-pose discontinuity.
        out[5:] = (action[4] + donor[5:] - donor[4]).clamp(-1,1)
    elif variant in ('swap', 'arm_swap', 'hand_swap'):
        dims = slice(None) if variant == 'swap' else (slice(0, 14) if variant == 'arm_swap' else slice(14, 58))
        out[5:, dims] = donor[5:, dims]
    elif variant != 'real':
        raise ValueError(variant)
    assert torch.equal(out[:5], action[:5]), 'Intervention changed context actions'
    return out


def prepare_manifest(args):
    root = Path(args.dataset)
    info = json.loads((root / 'dataset_info.json').read_text())
    stats = json.loads((root / 'action_stats.json').read_text())
    if info['action_dim'] != 58 or stats['action_dim'] != 58:
        raise ValueError('This probe expects ORCA 58D actions')
    if stats.get('frame_alignment') != 'actions[0]=state[0]; actions[t]=source_action[t-1] for t>0':
        raise ValueError('Action alignment must be explicitly verified before using this probe')
    train_ids, val_ids = info['train_episodes'], info['val_episodes']
    assert not set(train_ids) & set(val_ids)
    lengths = {}
    for split, episodes in [('train_data', train_ids), ('val_data', val_ids)]:
        for ep in episodes:
            p = episode_path(root, split, ep)
            rgb = np.load(p / 'rgb.npy', mmap_mode='r')
            a = np.load(p / 'actions.npy', mmap_mode='r')
            st = np.load(p / 'states.npy', mmap_mode='r')
            assert rgb.shape[1:] == (1, 256, 256, 3) and rgb.dtype == np.uint8, (p, rgb.shape)
            assert a.shape == st.shape == (len(rgb), 1, 58) and len(rgb) >= 14, p
            assert np.isfinite(a).all() and np.isfinite(st).all(), p
            assert max(float(abs(a).max()), float(abs(st).max())) <= 1.00001, p
            assert np.allclose(a[0], st[0]), p
            lengths[f'{split}/{ep}'] = len(rgb)
    rows = []
    selected_val_ids = val_ids if args.val_episode_ids is None else args.val_episode_ids
    if len(set(selected_val_ids)) != len(selected_val_ids) or not set(selected_val_ids) <= set(val_ids):
        raise ValueError('--val-episode-ids must be unique members of the official validation split')
    for split, episodes, count in [('train_data', train_ids, args.train_episodes), ('val_data', selected_val_ids, args.val_episodes)]:
        if count > len(episodes) or count < 2:
            raise ValueError('Require at least two and no more than available episodes per split')
        selected = [episodes[i] for i in np.linspace(0, len(episodes)-1, count, dtype=int)]
        for ep in selected:
            p = episode_path(root, split, ep)
            rgb = np.load(p / 'rgb.npy', mmap_mode='r')
            a = np.load(p / 'actions.npy', mmap_mode='r')[:, 0]
            st = np.load(p / 'states.npy', mmap_mode='r')[:, 0]
            small = np.asarray(rgb[:, 0, ::8, ::8], dtype=np.float32)
            candidates = []
            for s in range(1, len(rgb)-11, 4):
                motion = float(abs(small[s+4:s+12] - small[s+3]).mean()) / 255
                arm = float(np.sqrt(((a[s+4:s+12, :14] - st[s+3, :14])**2).mean()))
                hand = float(np.sqrt(((a[s+4:s+12, 14:] - st[s+3, 14:])**2).mean()))
                candidates.append(dict(split=split, episode=ep, start=s, motion=motion, arm_span=arm, hand_span=hand))
            scores = np.array([[r['motion'], r['arm_span'], r['hand_span']] for r in candidates])
            ranks = np.argsort(np.argsort(scores, axis=0), axis=0) / max(1, len(scores)-1)
            # Pixel and both action groups contribute, independent of model results.
            score = ranks @ np.array([1., .5, .5])
            chosen = []
            for i in np.argsort(-score):
                row = candidates[i]
                if all(abs(row['start'] - c['start']) >= 24 for c in chosen):
                    chosen.append(row)
                if len(chosen) == args.windows_per_episode:
                    break
            assert len(chosen) == args.windows_per_episode
            rows.extend(chosen)
    arrays = [window_arrays(root, r)[1:] for r in rows]
    for i, row in enumerate(rows):
        candidates = [j for j, other in enumerate(rows) if other['split'] == row['split'] and other['episode'] != row['episode']]
        # Pick a different command among the 8 closest starting states. Log the
        # remaining mismatch: this is observational ablation, not physical ground truth.
        candidates.sort(key=lambda j: float(((arrays[i][1]-arrays[j][1])**2).mean()))
        j = max(candidates[:8], key=lambda j: float(((arrays[i][0][5:]-arrays[j][0][5:])**2).mean()))
        row['donor_index'] = j
        row['donor_state_rmse'] = float(np.sqrt(((arrays[i][1]-arrays[j][1])**2).mean()))
        row['donor_future_action_rmse'] = float(np.sqrt(((arrays[i][0][5:]-arrays[j][0][5:])**2).mean()))
        row['id'] = f"{row['split']}_ep{row['episode']:06d}_s{row['start']}"
    manifest = dict(dataset=str(root.resolve()), stats_sha256=hashlib.sha256((root/'action_stats.json').read_bytes()).hexdigest(),
                    audit=dict(episodes=len(lengths), frames=sum(lengths.values()), action_dim=58, alignment=stats['frame_alignment'],
                               train_episodes=len(train_ids), val_episodes=len(val_ids),
                               selection='Uniformly spaced episode IDs; non-overlapping high-motion / arm-and-hand command-change windows'), rows=rows)
    write_json(args.output / 'manifest.json', manifest)
    print('DATA_AUDIT', json.dumps(manifest['audit']), flush=True)
    return manifest


def load_pipe(args):
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
    os.environ['WAN_ACTION_DIM'] = '58'
    model = Path(args.model)
    shards = sorted(str(p) for p in model.glob('diffusion_pytorch_model-*-of-*.safetensors'))
    if len(shards) != 3:
        raise ValueError('Expected the three original Wan2.2-TI2V-5B shards')
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cpu',
        model_configs=[ModelConfig(path=shards), ModelConfig(path=str(model/'Wan2.2_VAE.pth'))],
        tokenizer_config=ModelConfig(path=str(model/'google/umt5-xxl')))
    pipe.device = torch.device('cuda')
    pipe.dit.requires_grad_(False)
    pipe.vae.requires_grad_(False).eval().to('cuda')
    return pipe


def cache_latents(args, pipe, manifest):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode() + str(Path(args.model).resolve()).encode()).hexdigest()
    path = args.cache_dir / f'{signature}.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        return payload['samples']
    samples = []
    with torch.no_grad():
        for i, row in enumerate(manifest['rows']):
            rgb, act, state = window_arrays(args.dataset, row)
            v = torch.from_numpy(rgb).permute(3,0,1,2).unsqueeze(0).to('cuda', torch.bfloat16) / 127.5 - 1
            z = pipe.vae.encode(v, device='cuda').cpu()
            assert z.shape == (1,48,4,16,16) and torch.isfinite(z).all(), z.shape
            if i == 0:
                prefix = pipe.vae.encode(v[:,:,:5], device='cuda').cpu()
                error = float((prefix.float() - z[:,:,:2].float()).abs().max())
                print('VAE_CAUSAL_PREFIX_MAX_ERROR', error, flush=True)
                if error > .03:
                    raise ValueError('Full-video prefix differs from context-only encoding')
            samples.append(dict(z=z, action=torch.from_numpy(act), hold=torch.from_numpy(state)))
            if (i+1) % 8 == 0:
                print(f'CACHED {i+1}/{len(manifest["rows"])}', flush=True)
    torch.save(dict(samples=samples, signature=signature), path)
    return samples


def configure_trainable(pipe):
    pipe.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    pipe.dit.requires_grad_(True)
    # Full DiT in BF16, including AdamW moments; newly initialized action MLPs
    # use FP32 parameters/moments. foreach=False limits optimizer peak memory.
    for name, parameter in pipe.dit.named_parameters():
        if name.startswith(('action_mlp1.', 'action_mlp2.')):
            parameter.data = parameter.data.float()
    pipe.dit.to('cuda')
    n = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    assert all(p.requires_grad for p in pipe.dit.parameters())
    print('FULL_TRAINABLE_PARAMETERS', n, flush=True)
    return n


def predict(pipe, x, action, sigma, checkpoint=False):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return pipe.model_fn(dit=pipe.dit, latents=x, action=action,
            timestep=torch.tensor([sigma*1000], device='cuda', dtype=torch.bfloat16),
            fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=checkpoint)


def noisy_input(z, sigma, seed):
    noise = torch.randn(z.shape, generator=torch.Generator(device='cuda').manual_seed(seed), device='cuda', dtype=z.dtype)
    x = (1-sigma)*z + sigma*noise
    x[:,:,:2] = z[:,:,:2]
    return x, noise-z


def eval_summary(rows):
    summary = {}
    for split in sorted({r['split'] for r in rows}):
        subset = [r for r in rows if r['split']==split]
        real = {(r['id'],r['sigma'],r['seed']):r for r in subset if r['variant']=='real'}
        summary[split] = {'real_mse': float(np.mean([r['mse'] for r in real.values()]))}
        for variant in VARIANTS[1:]:
            vals = [r for r in subset if r['variant']==variant]
            deltas = np.array([r['mse']-real[(r['id'],r['sigma'],r['seed'])]['mse'] for r in vals])
            eps = sorted({r['episode'] for r in vals})
            grouped = np.array([np.mean([d for r,d in zip(vals,deltas) if r['episode']==ep]) for ep in eps])
            rng = np.random.default_rng(772)
            boots = grouped[rng.integers(0,len(grouped),size=(5000,len(grouped)))].mean(axis=1)
            summary[split][variant] = dict(delta_mse=float(deltas.mean()), relative_delta=float(deltas.mean()/summary[split]['real_mse']),
                episode_win_rate=float((grouped>0).mean()), episode_count=len(eps), episode_bootstrap_95ci=np.quantile(boots,[.025,.975]).tolist(),
                prediction_rmse=float(np.mean([r['prediction_rmse'] for r in vals])))
    return summary


@torch.no_grad()
def evaluate(args, pipe, samples, manifest, label):
    pipe.dit.eval()
    selected = [i for i,r in enumerate(manifest['rows']) if r['split']=='val_data']
    train = [i for i,r in enumerate(manifest['rows']) if r['split']=='train_data']
    selected += train[::max(1,len(train)//8)][:8]
    records = []
    started = time.monotonic()
    for k,i in enumerate(selected):
        row, sample = manifest['rows'][i], samples[i]
        rgb, _, _ = window_arrays(args.dataset, row)
        pixel_mask = np.abs(rgb[5:].astype(np.float32)-rgb[4].astype(np.float32)).mean(axis=(0,3)) / 255 > .035
        motion_mask = torch.nn.functional.avg_pool2d(torch.tensor(pixel_mask,device='cuda',dtype=torch.float32)[None,None],16)[0,0] > .1
        z = sample['z'].to('cuda')
        action = sample['action'].to('cuda')
        hold = sample['hold'].to('cuda')
        donor = samples[row['donor_index']]['action'].to('cuda')
        for sigma in args.eval_sigmas:
            for seed in args.eval_seeds:
                x,target = noisy_input(z, sigma, seed)
                ref = None
                for variant in VARIANTS:
                    a = intervention(action,hold,donor,variant)
                    pred = predict(pipe,x,a,sigma)[:,:,2:].float()
                    if not torch.isfinite(pred).all():
                        raise ValueError(f'Non-finite evaluation prediction: {row["id"]} / {variant}')
                    if ref is None:
                        ref = pred
                    mse = float((pred-target[:,:,2:].float()).square().mean())
                    records.append(dict(id=row['id'],split=row['split'],episode=row['episode'],sigma=sigma,seed=seed,variant=variant,mse=mse,
                                        prediction_rmse=float((pred-ref).square().mean().sqrt()),
                                        action_rmse=float((a[5:]-action[5:]).square().mean().sqrt()),
                                        motion_region_mse=float((pred-target[:,:,2:].float()).square()[...,motion_mask].mean()) if motion_mask.any() else None))
        print(f'EVAL {label} {k+1}/{len(selected)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    summary = eval_summary(records)
    write_json(args.output/f'eval-{label}.json',dict(summary=summary,records=records))
    with (args.output/f'eval-{label}.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print('EVAL_SUMMARY',label,json.dumps(summary),flush=True)
    return summary


def save_checkpoint(pipe, path):
    state = {n:p.detach().cpu().contiguous() for n,p in pipe.dit.named_parameters() if p.requires_grad}
    assert any(n.startswith('action_mlp1.') for n in state) and any(n.startswith('blocks.29.') for n in state)
    temporary = path.with_suffix(path.suffix + '.tmp')
    save_file(state, str(temporary))
    temporary.replace(path)


def train(args, pipe, samples, manifest):
    params = [p for p in pipe.dit.parameters() if p.requires_grad]
    action_params = [p for n,p in pipe.dit.named_parameters() if p.requires_grad and n.startswith('action_mlp')]
    action_ids = {id(p) for p in action_params}
    optimizer = torch.optim.AdamW([{'params':action_params,'lr':args.action_lr},
                                  {'params':[p for p in params if id(p) not in action_ids],'lr':args.lr}], weight_decay=0.01,foreach=False)
    train_ids = [i for i,r in enumerate(manifest['rows']) if r['split']=='train_data']
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    pipe.dit.train()
    log = (args.output/'train.jsonl').open('w')
    gradient_audit = {}
    for step in range(1,args.steps+1):
        if (step-1) % len(train_ids)==0:
            order = rng.permutation(train_ids)
        i = int(order[(step-1)%len(train_ids)])
        z = samples[i]['z'].to('cuda')
        action = samples[i]['action'].to('cuda')
        # Same shifted flow-matching sigma family as Wan; clean context is shared
        # with evaluation and generation. Only the two future latent frames score.
        u = float(rng.uniform(.001,1))
        sigma = 5*u/(1+4*u)
        x,target = noisy_input(z,sigma,int(rng.integers(0,2**31-1)))
        optimizer.zero_grad(set_to_none=True)
        pred = predict(pipe,x,action,sigma,checkpoint=True)
        loss = (pred[:,:,2:].float()-target[:,:,2:].float()).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Non-finite loss at {step}')
        loss.backward()
        if step == 1:
            missing_grad = [n for n,p in pipe.dit.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing_grad, f'Trainable parameters outside the graph: {missing_grad}'
            for group in ['action_mlp1','action_mlp2','blocks.0.','blocks.29.']:
                ps = [p for n,p in pipe.dit.named_parameters() if group in n and p.requires_grad]
                gradient_audit[group] = dict(parameters=sum(p.numel() for p in ps),with_grad=sum(p.grad is not None for p in ps),
                    grad_norm=float(torch.stack([p.grad.float().norm() for p in ps if p.grad is not None]).norm()))
            for group in ['action_mlp1','action_mlp2','blocks.0.','blocks.29.']:
                assert gradient_audit[group]['grad_norm']>0, gradient_audit
            write_json(args.output/'gradient_audit.json',gradient_audit)
            print('GRADIENT_AUDIT',json.dumps(gradient_audit),flush=True)
        norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        elapsed = time.monotonic()-started
        record = dict(step=step,sample=manifest['rows'][i]['id'],loss=float(loss.detach()),sigma=sigma,grad_norm=float(norm),
                      elapsed_seconds=elapsed,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        log.write(json.dumps(record)+'\n');log.flush()
        if step % 10==0 or step==1:
            print('TRAIN',json.dumps(record),flush=True)
        if step % args.save_every==0 and step < args.steps:
            save_checkpoint(pipe,args.output/f'dit-step-{step}.safetensors')
        if elapsed >= args.train_seconds:
            print('TRAIN_TIME_LIMIT',step,flush=True)
            break
    log.close()
    save_checkpoint(pipe,args.output/'dit-final.safetensors')
    del optimizer
    gc.collect();torch.cuda.empty_cache()
    return dict(steps=step,seconds=elapsed,gradient_audit=gradient_audit)


@torch.no_grad()
def render(args,pipe,samples,manifest):
    import imageio.v2 as imageio
    from diffsynth.schedulers.flow_match import FlowMatchScheduler
    pipe.dit.eval()
    pipe.vae.to('cuda')
    chosen = [i for i,r in enumerate(manifest['rows']) if r['split']=='val_data'][:args.render_windows]
    train_ids = [i for i,r in enumerate(manifest['rows']) if r['split']=='train_data']
    chosen += train_ids[:1]
    metrics = []
    for i in chosen:
        row,sample = manifest['rows'][i],samples[i]
        rgb,_,_ = window_arrays(args.dataset,row)
        # Re-encode ONLY reference+four context frames. No target latent input.
        v = torch.from_numpy(rgb[:5]).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)/127.5-1
        prefix = pipe.vae.encode(v,device='cuda')
        action,hold,donor = sample['action'].to('cuda'),sample['hold'].to('cuda'),samples[row['donor_index']]['action'].to('cuda')
        frames = {'ground_truth':rgb}
        target = rgb[5:].astype(np.float32)/255
        persistence = np.broadcast_to(rgb[4], rgb[5:].shape).astype(np.float32)/255
        mask = np.abs(target-rgb[4].astype(np.float32)/255).mean(axis=(0,3)) > .035
        metrics.append(dict(id=row['id'],split=row['split'],variant='persistence',mse=float(((persistence-target)**2).mean()),
                            motion_region_mse=float(((persistence-target)**2)[:,mask].mean()) if mask.any() else None,
                            motion_mask_fraction=float(mask.mean())))
        for variant in ('real','hold','swap','delta_swap'):
            scheduler = FlowMatchScheduler(shift=5,sigma_min=0,extra_one_step=True)
            scheduler.set_timesteps(args.inference_steps)
            x = torch.randn(sample['z'].shape,generator=torch.Generator(device='cuda').manual_seed(12345),device='cuda',dtype=torch.bfloat16)
            x[:,:,:2] = prefix
            a = intervention(action,hold,donor,variant)
            for t,sigma in zip(scheduler.timesteps,scheduler.sigmas):
                pred = predict(pipe,x,a,float(sigma))
                x = scheduler.step(pred,t,x)
                x[:,:,:2] = prefix
            decoded = pipe.vae.decode(x,device='cuda')[0].float().permute(1,2,3,0).cpu().numpy()
            frames[variant] = np.uint8(np.clip((decoded+1)*127.5,0,255))
            target = rgb[5:].astype(np.float32)/255
            gen = frames[variant][5:].astype(np.float32)/255
            mask = np.abs(target-rgb[4].astype(np.float32)/255).mean(axis=(0,3)) > .035
            metrics.append(dict(id=row['id'],split=row['split'],variant=variant,mse=float(((gen-target)**2).mean()),
                                motion_region_mse=float(((gen-target)**2)[:,mask].mean()) if mask.any() else None,
                                motion_mask_fraction=float(mask.mean())))
        labels = list(frames)
        sheet = Image.new('RGB',(4*256, len(labels)*282),'white')
        for r,label in enumerate(labels):
            ImageDraw.Draw(sheet).text((8,r*282+5),label,fill='black')
            for c,f in enumerate([4,5,8,12]):
                sheet.paste(Image.fromarray(frames[label][f]),(c*256,r*282+26))
        sheet.save(args.output/f'{row["id"]}-comparison.png')
        video=[]
        # Show the last observed frame followed by the eight predicted frames;
        # the episode reference is temporally distant and would create a jump.
        for f in range(4,13):
            canvas = Image.new('RGB',(len(labels)*256,282),'white')
            for c,label in enumerate(labels):
                ImageDraw.Draw(canvas).text((c*256+8,5),label,fill='black')
                canvas.paste(Image.fromarray(frames[label][f]),(c*256,26))
            video.append(np.array(canvas))
        imageio.mimwrite(args.output/f'{row["id"]}-comparison.mp4',video,fps=10,macro_block_size=2)
        print('RENDERED',row['id'],flush=True)
    write_json(args.output/'rollout_metrics.json',metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-command')
    p.add_argument('--model',default='/home/yiwei/workspace/model/Wan2.2-TI2V-5B')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--train-episodes',type=int,default=32)
    p.add_argument('--val-episodes',type=int,default=8)
    p.add_argument('--val-episode-ids',type=int,nargs='+',help='Optional explicit subset of the official validation split')
    p.add_argument('--windows-per-episode',type=int,default=2)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--train-seconds',type=float,default=3600)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--action-lr',type=float,default=1e-4)
    p.add_argument('--save-every',type=int,default=400)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-sigmas',type=float,nargs='+',default=[.5,.9,1.])
    p.add_argument('--eval-seeds',type=int,nargs='+',default=[2026,2027])
    p.add_argument('--render-windows',type=int,default=2)
    p.add_argument('--inference-steps',type=int,default=30)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--eval-checkpoint',type=Path)
    p.add_argument('--eval-initial',action='store_true',help='When evaluating a saved checkpoint, also evaluate the fresh base + random action branches first')
    p.add_argument('--eval-label',help='Output label for checkpoint evaluation; defaults to checkpoint filename stem')
    args=p.parse_args()
    for key in ('steps', 'train_seconds', 'save_every', 'windows_per_episode', 'inference_steps', 'lr', 'action_lr'):
        if getattr(args, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if not all(0 < sigma <= 1 for sigma in args.eval_sigmas):
        p.error('--eval-sigmas must lie in (0, 1]')
    if args.render_windows < 0:
        p.error('--render-windows must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'dit-final.safetensors').exists() and not args.eval_checkpoint:
        raise FileExistsError('Refusing to overwrite an existing experiment; choose a fresh output')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config['torch']=torch.__version__
    if args.eval_checkpoint:
        manifest=json.loads((args.output/'manifest.json').read_text())
        if Path(manifest['dataset']).resolve() != Path(args.dataset).resolve():
            raise ValueError('Evaluation dataset differs from saved manifest')
        write_json(args.output/f'eval-config-{args.eval_label or args.eval_checkpoint.stem}.json',config)
    else:
        write_json(args.output/'config.json',config)
        manifest=prepare_manifest(args)
    if args.prepare_only:
        return
    pipe=load_pipe(args)
    samples=cache_latents(args,pipe,manifest)
    n=configure_trainable(pipe)
    if args.eval_checkpoint:
        if args.eval_initial:
            evaluate(args,pipe,samples,manifest,'initial')
        state=load_file(str(args.eval_checkpoint))
        expected={n for n,p in pipe.dit.named_parameters() if p.requires_grad}
        assert set(state)==expected, 'Full DiT architecture does not match checkpoint'
        pipe.dit.load_state_dict(state,strict=True)
        evaluate(args,pipe,samples,manifest,args.eval_label or args.eval_checkpoint.stem)
    else:
        baseline=evaluate(args,pipe,samples,manifest,'initial')
        training=train(args,pipe,samples,manifest)
        final=evaluate(args,pipe,samples,manifest,'final')
        write_json(args.output/'result.json',dict(trainable_parameters=n,training=training,baseline=baseline,final=final))
    render(args,pipe,samples,manifest)
    print('PROBE_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
