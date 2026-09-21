#!/usr/bin/env python3
"""Summarize paired future-action interventions without treating seeds as episodes."""
import argparse
import json
from pathlib import Path
import numpy as np


def paired_deltas(records, variant, sigma=None, metric='mse'):
    records = [r for r in records if r['split']=='val_data' and (sigma is None or r['sigma']==sigma) and r.get(metric) is not None]
    real={(r['id'],r['sigma'],r['seed']):r for r in records if r['variant']=='real'}
    return {(r['id'],r['sigma'],r['seed']):(r['episode'],r[metric]-real[(r['id'],r['sigma'],r['seed'])][metric])
            for r in records if r['variant']==variant}


def interval(values):
    eps=sorted({v[0] for v in values.values()})
    grouped=np.array([np.mean([v[1] for v in values.values() if v[0]==ep]) for ep in eps])
    rng=np.random.default_rng(20260918)
    bootstrap=grouped[rng.integers(0,len(grouped),(10000,len(grouped)))].mean(1)
    return dict(mean=float(grouped.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                episode_wins=int((grouped>0).sum()),episodes=len(eps))


def summarize(output):
    initial=json.loads((output/'eval-initial.json').read_text())
    final=json.loads((output/'eval-final.json').read_text())
    result={}
    variants=['hold','swap','arm_swap','hand_swap']
    for extra in ('reverse','delta_swap'):
        if all(any(r['variant']==extra for r in data['records']) for data in (initial,final)):
            variants.append(extra)
    for label,sigma in [('all_sigmas',None),('pure_noise_future',1.)]:
        result[label]={}
        for variant in variants:
            before=paired_deltas(initial['records'],variant,sigma)
            after=paired_deltas(final['records'],variant,sigma)
            assert set(before)==set(after), 'Baseline/final pairs do not match'
            change={k:(v[0],v[1]-before[k][1]) for k,v in after.items()}
            result[label][variant]=dict(before=interval(before),after=interval(after),gain=interval(change))
    if all(any(r.get('motion_region_mse') is not None for r in data['records']) for data in (initial,final)):
        result['motion_region']={}
        for variant in variants:
            before=paired_deltas(initial['records'],variant,metric='motion_region_mse')
            after=paired_deltas(final['records'],variant,metric='motion_region_mse')
            assert set(before)==set(after)
            change={k:(v[0],v[1]-before[k][1]) for k,v in after.items()}
            result['motion_region'][variant]=dict(before=interval(before),after=interval(after),gain=interval(change))
    # Operational gate fixed independently of results. Passing supports this
    # short-horizon subset; it does not prove interventions are physically right.
    check=result['all_sigmas']
    result['diagnostic_gate']={
        'correct_action_mse_improves_vs_initial': final['summary']['val_data']['real_mse'] < initial['summary']['val_data']['real_mse'],
        'correct_actions_help_vs_swap':check['swap']['after']['ci95'][0]>0,
        'correct_actions_help_vs_hold':check['hold']['after']['ci95'][0]>0,
        'swap_advantage_improves_vs_initial':check['swap']['gain']['ci95'][0]>0,
        'arm_specific_evidence':check['arm_swap']['after']['ci95'][0]>0,
        'hand_specific_evidence':check['hand_swap']['after']['ci95'][0]>0,
        'pure_noise_swap_evidence':result['pure_noise_future']['swap']['after']['ci95'][0]>0,
    }
    result['all_gates_pass']=all(result['diagnostic_gate'].values())
    (output/'paired_conclusion.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    result=summarize(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
