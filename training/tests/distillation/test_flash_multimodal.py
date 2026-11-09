"""Multimodal Flash forward and learner scores on the two-layer B200 fixture."""

import os
import unittest

import torch
import torch.distributed as dist
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from tests.distillation.flash_test_model import build_test_model, snapshot_kernels
from tests.distillation.test_flash_ppo import make_trainer
from training.flash.flex_qsa import QUERY_TILE
from training.flash.input_layout import pack_padded_batch
from training.flash_model import FlashMultimodalDecoder


@unittest.skipUnless(torch.cuda.is_available(), "Flash requires CUDA/FLA")
class FlashMultimodalTests(unittest.TestCase):
    distributed = False

    @classmethod
    def setUpClass(cls):
        cls.trainer = make_trainer(build_test_model(
            fast=True, ple=True, distributed=cls.distributed,
            replicate_frozen_routers=cls.distributed,
        ))
        cls.trainer.config.multimodal = True
        config = qwen.Qwen4ExpVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                           num_heads=4, out_hidden_size=2560, patch_size=2,
                                           temporal_patch_size=1, spatial_merge_size=2,
                                           num_position_embeddings=64)
        torch.manual_seed(42)
        visual = qwen.Qwen4ExpVisionModel(config).to(device="cuda", dtype=torch.bfloat16)
        visual.requires_grad_(False).eval()
        cls.decoder = FlashMultimodalDecoder(visual, cls.trainer.model, image_token_id=255)
        cls.trainer._multimodal_decoder = cls.decoder

    def data(self, length=144):
        ids = torch.full((2, length), 4, dtype=torch.long, device="cuda")
        ids[0, 126:132] = 255  # An image crosses an attention tile, never a document boundary.
        ids[1, 3:7] = 255
        mask = torch.ones_like(ids)
        mask[1, 23:] = 0
        ids[1, 23:] = 0
        types = (ids == 255).long()
        grids = torch.tensor([[1, 4, 6], [1, 4, 4]], device="cuda")
        # Invoke HF's geometry oracle without constructing the full multimodal model.
        from types import SimpleNamespace
        oracle = qwen.Qwen4ExpModel.__new__(qwen.Qwen4ExpModel)
        torch.nn.Module.__init__(oracle)
        oracle.config = SimpleNamespace(vision_config=self.decoder.visual.config)
        positions = oracle.get_rope_index(ids, types, image_grid_thw=grids, attention_mask=mask)[0]
        labels = ids.clone()
        labels[(mask == 0) | (types == 1)] = -100
        labels[:, :8] = -100
        pixels = torch.randn(40, 12, device="cuda", dtype=torch.bfloat16)
        return dict(input_ids=ids, attention_mask=mask, labels=labels, pixel_values=pixels,
                    image_grid_thw=grids, position_ids=positions, mm_token_type_ids=types)

    def test_no_images_is_bitwise_text(self):
        ids = torch.full((2, 33), 4, dtype=torch.long, device="cuda")
        mask = torch.ones_like(ids)
        mask[1, 19:] = 0
        with snapshot_kernels(fixed_config=True), torch.no_grad():
            plain = self.trainer.model(input_ids=ids, attention_mask=mask).last_hidden_state
            wrapped = self.decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(wrapped, plain, rtol=0, atol=0)

    def test_ple_autocast_keeps_native_bf16_arithmetic(self):
        ple = self.trainer.model.layers[0].ple
        ids = torch.tensor([[4, 255, 255, 7]], device="cuda")
        hidden = torch.randn(1, 4, 4 * 2560, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            expected = ple(hidden, ids, None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual = ple(hidden, ids, None)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_padded_matches_explicit_packed_and_scorer(self):
        # Beyond the indexer's all-visible prefix: exercise M-RoPE in selection too.
        data = self.data(length=2176)
        inputs = {key: value for key, value in data.items() if key != "labels"}
        layout = pack_padded_batch(data["input_ids"], data["attention_mask"], data["position_ids"],
                                   eos_token_id=2, alignment=QUERY_TILE)
        with snapshot_kernels(fixed_config=True), torch.no_grad(), self.trainer._autocast_context():
            hidden = self.decoder(**inputs).last_hidden_state
            packed = self.decoder(input_ids=layout.input_ids, pixel_values=data["pixel_values"],
                                  image_grid_thw=data["image_grid_thw"],
                                  mm_token_type_ids=layout.pack(data["mm_token_type_ids"]),
                                  position_ids=layout.position_ids,
                                  packed_lengths=layout.packed_lengths).last_hidden_state
            torch.testing.assert_close(hidden, layout.unpack(packed), rtol=0, atol=0)
            actual, entropy = self.trainer._token_scores(data, compute_entropies=True)
            mask = data["labels"][:, 1:] != -100
            logits = self.trainer.bundle.head(hidden[:, :-1][mask]).float()
            expected = logits.log_softmax(-1).gather(-1, data["input_ids"][:, 1:][mask, None]).squeeze(-1)
            torch.testing.assert_close(actual[mask], expected, rtol=0, atol=1e-6)
            self.assertTrue(torch.isfinite(entropy).all())

    def test_ranks_with_and_without_images_complete_one_step(self):
        if not self.distributed:
            self.skipTest("Two ranks required")
        data = self.data()
        if dist.get_rank() == 1:  # this rank's batch carries no images at all
            ids = data["input_ids"].clone()
            ids[ids == 255] = 4
            data = dict(input_ids=ids, attention_mask=data["attention_mask"], labels=data["labels"])
        self.trainer.optimizer.zero_grad(set_to_none=True)
        try:
            with snapshot_kernels(fixed_config=True):
                scores, _ = self.trainer._token_scores(data, compute_entropies=False)
                scores.sum().backward()
            gradients = [p.grad for p in self.trainer._trainable_parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
        finally:
            self.trainer.optimizer.zero_grad(set_to_none=True)

    def test_backward_does_not_recompute_vision_and_ple_receives_raw_ids(self):
        data = self.data()
        vision_calls, ple_ids = [], []
        vision_hook = self.decoder.visual.register_forward_hook(lambda *args: vision_calls.append(1))
        ple = self.trainer.model.layers[0].ple.ple_embedding
        ple_hook = ple.register_forward_pre_hook(lambda module, args: ple_ids.append(args[0].detach().clone()))
        self.trainer.optimizer.zero_grad(set_to_none=True)
        try:
            with snapshot_kernels(fixed_config=True):
                scores, _ = self.trainer._token_scores(data, compute_entropies=False)
                scores.sum().backward()
            self.assertEqual(len(vision_calls), 1)
            self.assertTrue(ple_ids)
            self.assertEqual(int((ple_ids[0] == 255).sum()), 10)
            self.assertTrue(all(torch.equal(value, ple_ids[0]) for value in ple_ids))
            self.assertTrue(all(p.grad is None for p in self.decoder.visual.parameters()))
            gradients = [p.grad for p in self.trainer._trainable_parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
        finally:
            vision_hook.remove()
            ple_hook.remove()
            self.trainer.optimizer.zero_grad(set_to_none=True)


@unittest.skipUnless(torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 2,
                     "Two CUDA ranks required")
class FlashDistributedMultimodalTests(FlashMultimodalTests):
    distributed = True

    @classmethod
    def setUpClass(cls):
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    @unittest.skip("Direct PLE arithmetic is covered on the unsharded fixture")
    def test_ple_autocast_keeps_native_bf16_arithmetic(self):
        pass


if __name__ == "__main__":
    unittest.main()
