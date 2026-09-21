#!/usr/bin/env python3
"""Two real 58D training forward/backward checks; no optimizer step or checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from diffsynth.trainers.dataset import RLinfDataset
from train_rlinf import WanTrainingModule


def main(args):
    torch.set_num_threads(8)
    torch.manual_seed(12345)
    np.random.seed(12345)
    started=time.monotonic()
    ds=RLinfDataset(str(args.dataset/'train_data'),action_dim=58,action2obs_bias=True,
                    retain_actions=True,Ta=8,To=4,max_finish_step=0)
    model_paths=[[str(args.model/f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors') for i in range(1,4)],str(args.model/'Wan2.2_VAE.pth')]
    m=WanTrainingModule(model_paths=json.dumps(model_paths),trainable_models='dit',
        action_dim=58,extra_inputs='input_image,action',static_video_prob=.15,use_gradient_checkpointing=True)
    m.to(device='cuda',dtype=torch.bfloat16)
    m.train()
    assert m.pipe.dit.action_mlp1[0].in_features==58 and m.pipe.dit.action_mlp2[0].in_features==232
    assert all(p.requires_grad for p in m.pipe.dit.parameters())
    assert not any(p.requires_grad for p in m.pipe.vae.parameters())
    assert not hasattr(m.pipe.dit,'action_time_encoding')
    records=[]
    for name,prob in [('normal',0.0),('static',1.0)]:
        m.zero_grad(set_to_none=True)
        m.static_video_prob=prob
        data=m.transfer_data_to_device(ds[7],m.pipe.device,m.pipe.torch_dtype)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            inputs=m.forward_preprocess(data)
            assert tuple(inputs['action'].shape)==(13,58)
            nonzero=int(torch.count_nonzero(inputs['action']))
            assert (nonzero==0) if name=='static' else (nonzero>0)
            if name=='static':
                for f in data['video']: assert np.array_equal(np.asarray(f),np.asarray(data['video'][0]))
            loss=m({},inputs=inputs)
        assert torch.isfinite(loss)
        loss.backward()
        norms={}
        for key in ['action_mlp1.0.weight','action_mlp2.0.weight','blocks.0.self_attn.q.weight']:
            p=dict(m.pipe.dit.named_parameters()).get(key)
            if p is None: continue
            assert p.grad is not None and torch.isfinite(p.grad).all()
            norms[key]=float(p.grad.float().norm())
        # Zero action can give zero first-layer weight gradients; downstream biases must still train.
        bias=m.pipe.dit.action_mlp1[-1].bias.grad
        assert bias is not None and torch.isfinite(bias).all()
        norms['action_mlp1.final_bias']=float(bias.float().norm())
        records.append({'sample':name,'loss':float(loss.detach()),'action_nonzero_count':nonzero,'gradient_norms':norms})
        print('SMOKE',records[-1],flush=True)
    result={'passed':True,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'trainable_dit_parameters':sum(p.numel() for p in m.pipe.dit.parameters()),
        'action_input_dimensions':[58,232],'action_time_encoding_added':False,
        'original_training_loss_and_conditioning':True,'optimizer_updates':0,'checkpoints_saved':0,
        'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'elapsed_seconds':time.monotonic()-started,'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
