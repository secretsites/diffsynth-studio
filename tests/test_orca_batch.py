"""Real DiT batching, independent noise/timesteps and exact window accounting."""
import copy
from pathlib import Path
import sys
import types
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/wanvideo/model_training'), str(ROOT / 'examples/wanvideo')]
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, model_fn_wan_video
from train_rlinf import WanTrainingModule
from train_orca import seed_sample, sample_seed, shard_order, create_loader
from orca_config import load_config


def tiny_model():
    torch.manual_seed(11)
    model = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(model)
    pipe = WanVideoPipeline(device='cpu', torch_dtype=torch.float32)
    pipe.dit = WanModel(dim=24, in_dim=4, out_dim=4, ffn_dim=48, freq_dim=16, text_dim=24,
        num_heads=2, num_layers=1, eps=1e-6, patch_size=(1, 2, 2), has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True, action_dim=58)
    pipe.scheduler.set_timesteps(1000, training=True)
    model.pipe = pipe
    def preprocess(self, data):
        # Frozen preprocessing stand-in retains independent augmentation and noise.
        latents = data['latents'].clone()
        action = data['action'].clone()
        if np.random.rand() < .4:
            latents = latents[:, :, :1].expand_as(latents).clone(); action.zero_()
        return dict(input_latents=latents, noise=torch.randn_like(latents), action=action,
                    fuse_vae_embedding_in_latents=True)
    model.forward_preprocess = types.MethodType(preprocess, model)
    return model


class BatchTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_real_dit_per_sample_timestep_matches_single_sample(self):
        model = tiny_model().pipe.dit
        x, a, t = torch.randn(3, 4, 4, 4, 4), torch.randn(3, 13, 58), torch.tensor([950., 300., 70.])
        batch = model_fn_wan_video(dit=model, latents=x, timestep=t, action=a, fuse_vae_embedding_in_latents=True)
        single = torch.cat([model_fn_wan_video(dit=model, latents=x[i:i+1], timestep=t[i:i+1], action=a[i],
                                             fuse_vae_embedding_in_latents=True) for i in range(3)])
        torch.testing.assert_close(batch, single, atol=3e-6, rtol=3e-5)
        # Inference with a shared timestep must remain supported.
        shared = model_fn_wan_video(dit=model, latents=x, timestep=t[:1], action=a, fuse_vae_embedding_in_latents=True)
        per_sample = model_fn_wan_video(dit=model, latents=x, timestep=t[:1].expand(3), action=a, fuse_vae_embedding_in_latents=True)
        torch.testing.assert_close(shared, per_sample, atol=3e-6, rtol=3e-5)

    def test_batched_losses_gradients_and_rng_match_sequential_samples(self):
        model = tiny_model()
        samples = [dict(latents=torch.randn(1, 4, 4, 4, 4), action=torch.randn(13, 58)) for _ in range(3)]
        seeds = [sample_seed(12345, 0, i) for i in range(3)]
        separate = []
        for i, item in enumerate(samples):
            seed_sample(12345, 0, i)
            loss = model(copy.deepcopy(item)); separate.append(loss.detach())
            (loss / 3).backward()
        reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        losses = model(copy.deepcopy(samples), sample_seeds=seeds)
        torch.testing.assert_close(losses, torch.stack(separate), atol=3e-6, rtol=3e-5)
        losses.mean().backward()
        for name, p in model.named_parameters():
            if name in reference: torch.testing.assert_close(p.grad, reference[name], atol=3e-6, rtol=3e-4)

    def test_masked_per_sample_loss_preserves_weights_and_ignores_history(self):
        pipe = tiny_model().pipe
        pred = torch.zeros(2, 4, 4, 2, 2, requires_grad=True)
        target = torch.ones_like(pred); target[:, :, :2] = 100
        losses = pipe.training_loss_from_prediction(pred, target, torch.tensor([2., 3.]), reduction='none')
        torch.testing.assert_close(losses, torch.tensor([2., 3.]))
        (losses * torch.tensor([1., 0.])).sum().backward()
        self.assertEqual(pred.grad[1].count_nonzero().item(), 0)
        self.assertEqual(pred.grad[:, :, :2].count_nonzero().item(), 0)

    def test_loader_preserves_partial_batch_padding_and_resume_offset(self):
        class Data:
            def __getitem__(self, i): return i
        for size in [1, 2, 4, 8]:
            processed = []
            for rank in range(2):
                # First global batch (16 samples) was checkpointed with any micro batch size.
                remaining = shard_order(np.arange(35), 2, rank)[8:]
                batches = list(create_loader(Data(), remaining, 0, size))
                self.assertTrue(all(len(b) <= size for b in batches))
                processed.extend(i for b in batches for i, valid, _ in b if valid)
            self.assertEqual(sorted(processed), list(range(16, 35)))

    def test_global_batch_divisibility_checked(self):
        path = ROOT / 'examples/wanvideo/model_training/configs/orca_raw_train.yaml'
        for size in [1, 2, 4, 8, 16, 32, 64]:
            self.assertEqual(load_config(path, 'train', [f'training.micro_batch_size={size}'])['training']['micro_batch_size'], size)
        for size in [0, 3, 128]:
            with self.assertRaises(ValueError): load_config(path, 'train', [f'training.micro_batch_size={size}'])

    def test_partial_ddp_batch_gradient_equals_mean_over_real_windows(self):
        # Odd dataset lengths create a padded rank; its duplicate must have zero weight.
        for total in [19, 35]:
            for size in [1, 2, 4, 8]:
                for start in range(0, total, 16):
                    order = np.arange(start, min(start + 16, total))
                    reference = torch.tensor(.3, requires_grad=True)
                    ((reference * torch.tensor(order + 1.) - 2).square().mean()).backward()
                    gradients = []
                    for rank in range(2):
                        weight = torch.tensor(.3, requires_grad=True)
                        shard = shard_order(order, 2, rank)
                        for offset in range(0, len(shard), size):
                            group = shard[offset:offset + size]
                            x = torch.tensor([i + 1. for i, valid in group])
                            valid = torch.tensor([valid for i, valid in group])
                            (((weight * x - 2).square() * valid) * (2 / len(order))).sum().backward()
                        gradients.append(weight.grad)
                    torch.testing.assert_close(torch.stack(gradients).mean(), reference.grad)


if __name__ == '__main__': unittest.main()
