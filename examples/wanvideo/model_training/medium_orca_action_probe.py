#!/usr/bin/env python3
"""Frozen 64 x 4 full-parameter probe and paired autoregressive 24-frame audit.

Stages: prepare -> train -> rollout -> summarize. Validation thresholds are saved
before training. A failed gate never starts a full-data run automatically.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

import probe_orca_action_conditioning as base

CONTROLS = ('real', 'hold', 'reverse', 'delta_swap')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def long_arrays(dataset, row):
    p = base.episode_path(dataset, row['split'], row['episode'])
    ids = [0] + list(range(row['start'], row['start'] + 28))
    rgb = np.load(p/'rgb.npy', mmap_mode='r')[ids, 0].copy()
    action = np.load(p/'actions.npy', mmap_mode='r')[ids, 0].copy()
    state = np.load(p/'states.npy', mmap_mode='r')[row['start']+3, 0].copy()
    assert len(rgb) == len(action) == 29
    return rgb, action, state


def chunk_actions(sequence, chunk):
    """Reference + four preceding commands + eight future commands.

    For chunks 2/3 the preceding commands belong to the intervention as well;
    substituting real command history would invalidate the counterfactual.
    """
    if chunk not in (0, 1, 2):
        raise ValueError(chunk)
    offset = chunk * 8
    return torch.cat([sequence[:1], sequence[1+offset:13+offset]])


def candidates(dataset, split, episodes):
    rows, images, states, increments = [], [], [], []
    for ep in episodes:
        path = base.episode_path(dataset, split, ep)
        rgb = np.load(path/'rgb.npy', mmap_mode='r')
        action = np.load(path/'actions.npy', mmap_mode='r')[:, 0]
        state = np.load(path/'states.npy', mmap_mode='r')[:, 0]
        small = np.asarray(rgb[:, 0, ::16, ::16], dtype=np.float32) / 255
        for s in range(1, len(rgb)-27, 8):
            change = action[s+4:s+28]-action[s+3]
            rows.append(dict(split=split, episode=ep, start=s,
                             motion=float(abs(small[s+4:s+28]-small[s+3]).mean()),
                             arm_span=float(np.sqrt(np.mean(change[:, :14]**2))),
                             hand_span=float(np.sqrt(np.mean(change[:, 14:]**2)))))
            images.append(small[s+3].flatten())
            states.append(state[s+3].copy())
            # First 8 frames determine the actual training-window contrast.
            increments.append(change[:8].flatten())
    return rows, np.stack(images), np.stack(states), np.stack(increments)


def pair_distances(images, states):
    def distances(values):
        t = torch.from_numpy(values)
        return (torch.cdist(t,t,compute_mode='use_mm_for_euclid_dist') / values.shape[1]**.5).numpy()
    image = distances(images)
    state = distances(states)
    # Equal contributions relative to typical separation within this split.
    combined = image / max(float(np.median(image)), 1e-6) + state / max(float(np.median(state)), 1e-6)
    return image, state, combined


def select_rows(dataset, split, episodes, count):
    rows, images, states, increments = candidates(dataset, split, episodes)
    di, ds, distance = pair_distances(images, states)
    eps = np.array([r['episode'] for r in rows])
    distance[eps[:, None] == eps[None, :]] = np.inf
    spans = np.array([[r['motion'], r['arm_span'], r['hand_span']] for r in rows])
    ranks = np.argsort(np.argsort(spans, axis=0), axis=0) / max(1, len(rows)-1)
    activity = ranks @ np.array([1., .5, .5])
    # Build pairs among the eight closest image + state contexts. Choose the
    # largest command-increment contrast only inside that local neighbourhood.
    pairs = []
    for i in range(len(rows)):
        near = np.argsort(distance[i])[:8]
        delta = np.sqrt(np.mean((increments[near]-increments[i])**2, axis=1))
        j = int(near[np.argmax(delta)])
        score = float(delta.max() * min(activity[i], activity[j]) / (.1 + distance[i,j]))
        pairs.append((score, i, j))
    selected, by_ep = [], {ep: [] for ep in episodes}
    def available(i):
        r = rows[i]
        return len(by_ep[r['episode']]) < count and all(abs(r['start']-rows[j]['start']) >= 28 for j in by_ep[r['episode']])
    # Most slots are filled in pairs, so matched alternatives are actually in
    # the training set, not just external donors used during evaluation.
    for score, i, j in sorted(pairs, reverse=True):
        if available(i) and available(j):
            for k in (i, j):
                selected.append(k)
                by_ep[rows[k]['episode']].append(k)
    for ep in episodes:
        for i in np.argsort(-activity):
            if rows[i]['episode'] == ep and available(int(i)):
                selected.append(int(i)); by_ep[ep].append(int(i))
            if len(by_ep[ep]) == count:
                break
        if len(by_ep[ep]) != count:
            raise ValueError(f'Cannot choose {count} nonoverlapping windows in episode {ep}')
    selected.sort(key=lambda i: (rows[i]['episode'], rows[i]['start']))
    chosen = [rows[i] for i in selected]
    for k, i in enumerate(selected):
        eligible = [n for n,j in enumerate(selected) if rows[j]['episode'] != rows[i]['episode']]
        near = sorted(eligible, key=lambda n: distance[i,selected[n]])[:4]
        n = max(near, key=lambda n: np.mean((increments[i]-increments[selected[n]])**2))
        j = selected[n]
        chosen[k].update(donor_local_index=n, donor_state_rmse=float(ds[i,j]), donor_image_rmse=float(di[i,j]),
                         donor_increment_rmse=float(np.sqrt(np.mean((increments[i]-increments[j])**2))),
                         id=f"{split}_ep{rows[i]['episode']:06d}_s{rows[i]['start']}")
    return chosen


def prepare(args):
    torch.set_num_threads(8)
    out = args.output
    if (out/'protocol.json').exists() or (out/'manifest.json').exists():
        raise FileExistsError('Frozen experiment exists; use another output directory')
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.audit.read_text())
    assert audit['passed'] and Path(audit['dataset']).resolve() == args.dataset.resolve()
    assert audit['stats_sha256'] == digest(args.dataset/'action_stats.json')
    info = json.loads((args.dataset/'dataset_info.json').read_text())
    episodes = [info['train_episodes'][i] for i in np.linspace(0,len(info['train_episodes'])-1,64,dtype=int)]
    rows = select_rows(args.dataset, 'train_data', episodes, 4)
    rows += select_rows(args.dataset, 'val_data', info['val_episodes'], 2)
    for split in ('train_data', 'val_data'):
        offset = next(i for i,r in enumerate(rows) if r['split'] == split)
        for row in rows:
            if row['split'] == split:
                row['donor_index'] = offset + row.pop('donor_local_index')
    coverage = {}
    for split in ('train_data', 'val_data'):
        part = [r for r in rows if r['split'] == split]
        coverage[split] = {key: dict(median=float(np.median([r[key] for r in part])),
                                   q10_q90=np.quantile([r[key] for r in part],[.1,.9]).tolist())
                           for key in ('motion', 'donor_state_rmse', 'donor_image_rmse', 'donor_increment_rmse')}
    manifest = dict(dataset=str(args.dataset.resolve()), stats_sha256=digest(args.dataset/'action_stats.json'),
                    audit=dict(episodes=151, frames=audit['frames'], action_dim=58,
                               selection='64 uniformly spaced train episodes x4 locally image/state-matched contrasting-command windows; all 15 val episodes x2; 28-frame spans; stride8 candidates'),
                    coverage=coverage, rows=rows)
    base.write_json(out/'manifest.json', manifest)
    protocol = dict(version=1, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    dataset=str(args.dataset.resolve()), model=str(args.model.resolve()),
                    initial_checkpoint=str(args.init_checkpoint.resolve()), optimizer='Fresh AdamW; warm-start weights only, no exact optimizer resume',
                    manifest_sha256=digest(out/'manifest.json'), dataset_audit_sha256=digest(args.audit),
                    training=dict(steps=8192, train_seconds=4500, lr=1e-5, action_lr=1e-4, seed=43, save_every=8192,
                                  train_episodes=64, windows_per_episode=4, trainable='All DiT + both action MLPs; VAE frozen; no LoRA'),
                    evaluation=dict(sigmas=[.5,.9,1.], flow_seeds=[2026,2027], rollout_seeds=[12345,54321],
                                    inference_steps=50, chunks=3, frames_per_chunk=8, fps=30,
                                    controls=list(CONTROLS), reverse='Reverse the complete 24-frame future command order',
                                    hold='Repeat measured state at last initial observed frame for all 24 future commands',
                                    delta_swap='Transfer donor 24-frame increments anchored at latest real command, clip [-1,1]',
                                    context='Fixed episode reference plus four initial observed frames; later context is generated RGB only',
                                    bootstrap='10000 resamples of episode means; windows/seeds remain inside episode; random seed 20260918',
                                    val_is_new_blind_holdout=False,
                                    val_caveat='Same 15 official validation episodes were seen in previous experiments; this protocol/long-horizon window list is frozen before this training.'),
                    gates=dict(all_required=True,
                               flow='For hold/reverse/delta_swap separately: sigma=1 val paired MSE delta lower 95% CI >0, relative improvement >=2%, episode win rate >=0.60',
                               rollout='For hold/reverse/delta_swap separately: aggregate 24-frame motion-region pixel MSE delta lower 95% CI >0, relative improvement >=2%, episode win rate >=0.60; full-frame mean delta >0',
                               sustained='For each control: motion-region delta >0 in each of chunks 1/2/3 and separately for both noise seeds',
                               persistence='True action rollout beats last-frame persistence in motion-region mean MSE by at least 10%',
                               visual='Review preselected episodes 12,77,144 (both windows, seed12345), all 3 chunks. Reject severe collapse, wrong arm/hand movement or repeated controls outperforming true actions. Human-readable review with evidence required.',
                               excluded='Train-only gains, swap/arm_swap/hand_swap outlier deterioration, and mere change of generated pixels never count as a pass'),
                    full_data_authorized_only_if_all_gates_pass=True)
    base.write_json(out/'protocol.json', protocol)
    base.write_json(out/'freeze.json', dict(protocol_sha256=digest(out/'protocol.json'), manifest_sha256=digest(out/'manifest.json')))
    print('FROZEN', json.dumps(dict(train_windows=256,val_windows=30,coverage=coverage)), flush=True)


def frozen(args):
    protocol = json.loads((args.output/'protocol.json').read_text())
    freeze = json.loads((args.output/'freeze.json').read_text())
    assert digest(args.output/'protocol.json') == freeze['protocol_sha256']
    assert digest(args.output/'manifest.json') == freeze['manifest_sha256'] == protocol['manifest_sha256']
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert digest(Path(protocol['dataset'])/'action_stats.json') == manifest['stats_sha256']
    return protocol, manifest


def run_args(args, protocol):
    return SimpleNamespace(**protocol['training'], dataset=protocol['dataset'], model=protocol['model'],
                           output=args.output, cache_dir=args.cache_dir,
                           eval_sigmas=protocol['evaluation']['sigmas'], eval_seeds=protocol['evaluation']['flow_seeds'])


def load_checkpoint(pipe, path):
    state = load_file(str(path))
    assert set(state) == {n for n,p in pipe.dit.named_parameters()}
    pipe.dit.load_state_dict(state, strict=True)
    del state
    gc.collect()


def run_train(args):
    protocol, manifest = frozen(args)
    if (args.output/'dit-final.safetensors').exists():
        raise FileExistsError('Final checkpoint already exists')
    a = run_args(args, protocol)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = True
    base.write_json(args.output/'runtime.json', dict(torch=torch.__version__, git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
                                                   medium_script_sha256=digest(__file__), base_script_sha256=digest(base.__file__),
                                                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    pipe = base.load_pipe(a)
    samples = base.cache_latents(a,pipe,manifest)
    n = base.configure_trainable(pipe)
    load_checkpoint(pipe,protocol['initial_checkpoint'])
    initial = base.evaluate(a,pipe,samples,manifest,'initial')
    training = base.train(a,pipe,samples,manifest)
    final = base.evaluate(a,pipe,samples,manifest,'final')
    base.write_json(args.output/'result.json',dict(trainable_parameters=n,training=training,baseline=initial,final=final))
    print('MEDIUM_TRAIN_COMPLETE',flush=True)


def metric_rows(row, seed, variant, generated, target, last_observed):
    target = target.astype(np.float32)/255
    gen = generated.astype(np.float32)/255
    # One GT-derived spatial mask, shared by all interventions/seeds/chunks.
    # Targets are used only for scoring, never as later generated context.
    mask = abs(target-last_observed.astype(np.float32)/255).mean(axis=(0,3)) > .035
    records = []
    for name,sl in [('all',slice(0,24)),('1',slice(0,8)),('2',slice(8,16)),('3',slice(16,24))]:
        err = (gen[sl]-target[sl])**2
        records.append(dict(id=row['id'],episode=row['episode'],split=row['split'],seed=seed,variant=variant,chunk=name,
                            mse=float(err.mean()),motion_region_mse=float(err[:,mask].mean()) if mask.any() else None,
                            motion_mask_fraction=float(mask.mean())))
    return records


def save_visual(path, row, seed, original, frames):
    import imageio.v2 as imageio
    labels = ['ground_truth'] + list(CONTROLS)
    values = {'ground_truth':original[5:], **frames}
    sheet = Image.new('RGB',(7*256,len(labels)*282),'white')
    for r,label in enumerate(labels):
        ImageDraw.Draw(sheet).text((8,r*282+5),f'{label} | observed, +4,+8,+12,+16,+20,+24',fill='black')
        for c,frame in enumerate([original[4]]+[values[label][i] for i in (3,7,11,15,19,23)]):
            sheet.paste(Image.fromarray(frame),(c*256,r*282+26))
    stem = f'{row["id"]}-seed{seed}'
    sheet.save(path/f'{stem}.png')
    video = []
    for t in range(-1,24):
        canvas = Image.new('RGB',(len(labels)*256,282),'white')
        for c,label in enumerate(labels):
            ImageDraw.Draw(canvas).text((c*256+5,5),f'{label}  +{t+1}/24',fill='black')
            canvas.paste(Image.fromarray(original[4] if t<0 else values[label][t]),(c*256,26))
        video.append(np.asarray(canvas))
    imageio.mimwrite(path/f'{stem}.mp4',video,fps=30,macro_block_size=2)
    # Lossless arrays preserve exact frames for audit and subsequent metrics.
    np.savez_compressed(path/f'{stem}.npz',ground_truth=original[5:],observed=original[:5],**frames)


@torch.no_grad()
def run_rollout(args):
    from diffsynth.schedulers.flow_match import FlowMatchScheduler
    protocol, manifest = frozen(args)
    checkpoint_sha256=digest(args.output/'dit-final.safetensors')
    base.write_json(args.output/'rollout_runtime.json',dict(checkpoint_sha256=checkpoint_sha256,
                    script_sha256=digest(__file__),base_script_sha256=digest(base.__file__),torch=torch.__version__))
    a = run_args(args,protocol)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    pipe = base.load_pipe(a)
    base.configure_trainable(pipe)
    load_checkpoint(pipe,args.output/'dit-final.safetensors')
    pipe.dit.requires_grad_(False).eval();pipe.vae.to('cuda')
    target_dir = args.output/'rollouts'
    target_dir.mkdir(exist_ok=True)
    e = protocol['evaluation']
    selected = [r for r in manifest['rows'] if r['split']=='val_data']
    records, causal_audit = [], []
    started = time.monotonic()
    for k,row in enumerate(selected):
        original, real_action, hold = long_arrays(a.dataset,row)
        _,donor,_ = long_arrays(a.dataset,manifest['rows'][row['donor_index']])
        real_action,hold,donor = [torch.tensor(v,device='cuda') for v in (real_action,hold,donor)]
        for seed in e['rollout_seeds']:
            done = target_dir/f'{row["id"]}-seed{seed}.json'
            if done.exists():
                previous=json.loads(done.read_text())
                assert previous['protocol_sha256']==digest(args.output/'protocol.json')
                assert previous['checkpoint_sha256']==checkpoint_sha256
                records.extend(previous['metrics']);causal_audit.extend(previous['causal_audit'])
                continue
            frames, local_audit, local_metrics = {}, [], []
            for variant in CONTROLS:
                command = base.intervention(real_action,hold,donor,variant)
                # Keep only observed RGB in the generation state. The original
                # future RGB array is never indexed by the context construction.
                context = original[:5].copy()
                generated = []
                for chunk in range(3):
                    v = torch.from_numpy(context).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)/127.5-1
                    prefix = pipe.vae.encode(v,device='cuda')
                    assert tuple(prefix.shape)==(1,48,2,16,16)
                    scheduler = FlowMatchScheduler(shift=5,sigma_min=0,extra_one_step=True)
                    scheduler.set_timesteps(e['inference_steps'])
                    chunk_seed=seed+100000*chunk
                    x = torch.randn((1,48,4,16,16),generator=torch.Generator(device='cuda').manual_seed(chunk_seed),device='cuda',dtype=torch.bfloat16)
                    noise_hash=hashlib.sha256(x.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    x[:,:,:2]=prefix
                    action=chunk_actions(command,chunk)
                    for t,sigma in zip(scheduler.timesteps,scheduler.sigmas):
                        pred=base.predict(pipe,x,action,float(sigma))
                        x=scheduler.step(pred,t,x);x[:,:,:2]=prefix
                    assert torch.isfinite(x).all()
                    decoded=pipe.vae.decode(x,device='cuda')[0].float().permute(1,2,3,0).cpu().numpy()
                    prediction=np.uint8(np.clip((decoded[5:]+1)*127.5,0,255))
                    assert prediction.shape==(8,256,256,3)
                    generated.append(prediction)
                    local_audit.append(dict(id=row['id'],seed=seed,variant=variant,chunk=chunk+1,noise_sha256=noise_hash,
                                            context_source='observed' if chunk==0 else 'generated_previous_chunk',
                                            context_sha256=hashlib.sha256(context.tobytes()).hexdigest()))
                    context=np.concatenate([original[:1],prediction[-4:]],axis=0)
                frames[variant]=np.concatenate(generated)
                local_metrics.extend(metric_rows(row,seed,variant,frames[variant],original[5:],original[4]))
            persistence=np.broadcast_to(original[4],original[5:].shape)
            local_metrics.extend(metric_rows(row,seed,'persistence',persistence,original[5:],original[4]))
            # Assert identical random draws and initial RGB across every control.
            for chunk in (1,2,3):
                assert len({r['noise_sha256'] for r in local_audit if r['chunk']==chunk})==1
            assert len({r['context_sha256'] for r in local_audit if r['chunk']==1})==1
            save_visual(target_dir,row,seed,original,frames)
            base.write_json(done,dict(protocol_sha256=digest(args.output/'protocol.json'),checkpoint_sha256=checkpoint_sha256,
                                     metrics=local_metrics,causal_audit=local_audit))
            records.extend(local_metrics);causal_audit.extend(local_audit)
            print(f'ROLLOUT {k+1}/{len(selected)} seed={seed} elapsed={time.monotonic()-started:.1f}s',flush=True)
    base.write_json(args.output/'rollout_metrics.json',records)
    base.write_json(args.output/'causal_rollout_audit.json',causal_audit)
    print('THREE_CHUNK_COMPLETE',flush=True)


def paired(records, metric, variant):
    def key(r):
        return (r['id'],r['seed'],r.get('sigma'),r.get('chunk'))
    real={key(r):r for r in records if r['variant']=='real'}
    grouped={}
    bases=[]
    for r in records:
        if r['variant']!=variant:
            continue
        ref=real[key(r)]
        if r[metric] is None or ref[metric] is None:
            raise ValueError('Empty motion mask invalidates this frozen evaluation')
        grouped.setdefault(r['episode'],[]).append(r[metric]-ref[metric])
        bases.append(ref[metric])
    if not grouped:
        raise ValueError(f'No paired data for {variant}')
    values=np.array([np.mean(v) for _,v in sorted(grouped.items())])
    rng=np.random.default_rng(20260918)
    bootstrap=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
    baseline=float(np.mean(bases))
    return dict(delta=float(values.mean()),relative_delta=float(values.mean()/baseline),real_mse=baseline,
                ci95=np.quantile(bootstrap,[.025,.975]).tolist(),episode_win_rate=float(np.mean(values>0)),
                episodes=len(values),episode_deltas={str(k):float(np.mean(v)) for k,v in sorted(grouped.items())})


def summarize(args):
    protocol,manifest=frozen(args)
    flow=json.loads((args.output/'eval-final.json').read_text())['records']
    rollout=json.loads((args.output/'rollout_metrics.json').read_text())
    expected_ids={r['id'] for r in manifest['rows'] if r['split']=='val_data'}
    expected={(i,seed,variant,chunk) for i in expected_ids for seed in protocol['evaluation']['rollout_seeds']
              for variant in (*CONTROLS,'persistence') for chunk in ('all','1','2','3')}
    actual=[(r['id'],r['seed'],r['variant'],r['chunk']) for r in rollout]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    flow=[r for r in flow if r['split']=='val_data' and r['sigma']==1]
    expected_flow={(i,seed,v) for i in expected_ids for seed in protocol['evaluation']['flow_seeds'] for v in base.VARIANTS}
    actual_flow=[(r['id'],r['seed'],r['variant']) for r in flow]
    assert len(actual_flow)==len(set(actual_flow)) and set(actual_flow)==expected_flow
    result=dict(protocol_sha256=digest(args.output/'protocol.json'),controls={},gates={})
    for variant in CONTROLS[1:]:
        f=paired(flow,'mse',variant)
        total=[r for r in rollout if r['chunk']=='all']
        m=paired(total,'motion_region_mse',variant)
        pixels=paired(total,'mse',variant)
        chunks={c:paired([r for r in rollout if r['chunk']==c],'motion_region_mse',variant) for c in ('1','2','3')}
        seeds={str(s):paired([r for r in total if r['seed']==s],'motion_region_mse',variant) for s in protocol['evaluation']['rollout_seeds']}
        result['controls'][variant]=dict(flow_sigma1=f,rollout_motion=m,rollout_pixels=pixels,chunk_motion=chunks,seed_motion=seeds)
        result['gates'][f'{variant}_flow']=f['ci95'][0]>0 and f['relative_delta']>=.02 and f['episode_win_rate']>=.6
        result['gates'][f'{variant}_rollout']=m['ci95'][0]>0 and m['relative_delta']>=.02 and m['episode_win_rate']>=.6 and pixels['delta']>0
        result['gates'][f'{variant}_sustained']=all(v['delta']>0 for v in [*chunks.values(),*seeds.values()])
    p=paired([r for r in rollout if r['chunk']=='all'],'motion_region_mse','persistence')
    result['persistence']=p
    # relative_delta uses true error as denominator; convert to reduction vs persistence.
    result['gates']['persistence']=p['delta']/(p['delta']+p['real_mse'])>=.1
    result['quantitative_pass']=all(result['gates'].values())
    visual=args.output/'visual_review.json'
    result['visual_review']=json.loads(visual.read_text()) if visual.exists() else dict(passed=False,status='pending')
    result['all_gates_pass']=result['quantitative_pass'] and result['visual_review'].get('passed') is True
    base.write_json(args.output/'gate_result.json',result)
    print('GATE_RESULT',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','rollout','summarize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=base.ROOT/'work/orca_probe_cache')
    p.add_argument('--dataset',type=Path,default=Path('/home/yiwei/workspace/datasets/orca-template1-dev-wan-256-command'))
    p.add_argument('--model',type=Path,default=Path('/home/yiwei/workspace/model/Wan2.2-TI2V-5B'))
    p.add_argument('--audit',type=Path)
    p.add_argument('--init-checkpoint',type=Path)
    args=p.parse_args()
    if args.stage=='prepare' and (args.audit is None or args.init_checkpoint is None):
        p.error('prepare requires --audit and --init-checkpoint')
    {'prepare':prepare,'train':run_train,'rollout':run_rollout,'summarize':summarize}[args.stage](args)


if __name__=='__main__':
    main()
