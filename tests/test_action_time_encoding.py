"""Action order must affect the actual video model's cross-attention path."""
import unittest
from unittest.mock import patch

import torch
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video_new import model_fn_wan_video


def tiny_model(encoding):
    return WanModel(dim=32, in_dim=4, out_dim=4, ffn_dim=64, freq_dim=16,
                    text_dim=16, num_heads=4, num_layers=2, eps=1e-6,
                    patch_size=(1, 2, 2), has_image_input=False,
                    require_vae_embedding=False, require_clip_embedding=False,
                    seperated_timestep=True, action_dim=3,
                    action_mode="crossattn", action_time_encoding=encoding).eval()


class ActionTimeEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.patch = patch.multiple("diffsynth.models.wan_video_dit",
                                    FLASH_ATTN_3_AVAILABLE=False,
                                    FLASH_ATTN_2_AVAILABLE=False,
                                    SAGE_ATTN_AVAILABLE=False)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_future_order_changes_video_prediction_only_with_time_encoding(self):
        model = tiny_model("none")
        latents = torch.randn(1, 4, 4, 4, 4)
        actions = torch.randn(13, 3)
        reversed_actions = actions.clone()
        reversed_actions[5:] = actions[5:].flip(0)

        def predict(a, checkpoint=False):
            return model_fn_wan_video(dit=model, latents=latents,
                                      timestep=torch.tensor([900.]), action=a,
                                      fuse_vae_embedding_in_latents=True,
                                      use_gradient_checkpointing=checkpoint)

        with torch.no_grad():
            torch.testing.assert_close(predict(actions), predict(reversed_actions), atol=1e-6, rtol=1e-5)
        model.action_time_encoding = "sinusoidal"
        actions.requires_grad_(True)
        actual = predict(actions)
        changed = predict(reversed_actions)
        self.assertGreater(float((actual[:, :, 2:] - changed[:, :, 2:]).detach().abs().max()), 1e-5)
        actual[:, :, 2:].square().mean().backward()
        self.assertGreater(float(actions.grad[5:].abs().sum()), 0)
        torch.testing.assert_close(actual, predict(actions, checkpoint=True))
        torch.testing.assert_close(actual, predict(actions[None]))

    def test_same_local_positions_across_horizons_batches_and_dtypes(self):
        model = tiny_model("sinusoidal")
        actions = torch.randn(2, 29, 3)
        encoded = model.embed_action_context(actions)
        torch.testing.assert_close(encoded[:, :13], model.embed_action_context(actions[:, :13]))
        torch.testing.assert_close(encoded[0], model.embed_action_context(actions[0]))
        self.assertEqual(encoded.dtype, actions.dtype)
        model = model.to(torch.bfloat16)
        self.assertEqual(model.embed_action_context(actions.bfloat16()).dtype, torch.bfloat16)

    def test_no_checkpoint_schema_change_and_legacy_switch(self):
        old, new = tiny_model("none"), tiny_model("sinusoidal")
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        a = torch.randn(13, 3)
        torch.testing.assert_close(old.embed_action_context(a), old.action_mlp1(a))
        with self.assertRaisesRegex(ValueError, "action_time_encoding"):
            tiny_model("invalid")


if __name__ == "__main__":
    unittest.main()
