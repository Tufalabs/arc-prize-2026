import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from training.flash.options import FlashOptions
from training.flash.adapters import (
    FlashTextModel,
    PackedNGramEmbedding,
    PackedPLELayer,
    QualifiedGatedDeltaNet,
    _active_lengths,
    _packed_boundaries,
)
from training.flash.input_layout import pack_padded_batch


def public_model():
    model = FlashTextModel.__new__(FlashTextModel)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(pad_token_id=99, eos_token_id=2)
    model._flash_flex_qsa = True
    return model


class PackedModelBoundaryTest(unittest.TestCase):
    def test_embedded_public_batch_preserves_positions_raw_ids_and_eos(self):
        model = public_model()
        model.config.ple_layer_ids = [1]
        model.embed_tokens = nn.Embedding(100, 8)
        ids = torch.tensor([[4, 99, 99], [7, 8, 0]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        positions = torch.tensor([[[0, 1, 1], [0, 1, 0]],
                                  [[0, 1, 1], [0, 1, 0]],
                                  [[0, 1, 2], [0, 1, 0]]])
        embeds = model.embed_tokens(ids).detach().requires_grad_()
        layout = pack_padded_batch(ids, mask, positions, eos_token_id=2)

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            self.assertIsNone(input_ids)
            torch.testing.assert_close(kwargs["ple_input_ids"], layout.input_ids, rtol=0, atol=0)
            torch.testing.assert_close(kwargs["position_ids"], layout.position_ids, rtol=0, atol=0)
            torch.testing.assert_close(kwargs["inputs_embeds"][0, 3], model.embed_tokens.weight[2])
            return types.SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])

        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            output = model(inputs_embeds=embeds, ple_input_ids=ids,
                           position_ids=positions, attention_mask=mask)
        output.last_hidden_state.sum().backward()
        torch.testing.assert_close(embeds.grad, mask[..., None].expand_as(embeds).float())
        with self.assertRaisesRegex(ValueError, "raw ple_input_ids"):
            model(inputs_embeds=embeds, attention_mask=mask)

    def test_single_document_accepts_3d_positions_without_packed_metadata(self):
        model = public_model()
        ids = torch.tensor([[4, 99, 99]])
        positions = torch.tensor([[[0, 1, 1]], [[0, 1, 1]], [[0, 1, 2]]])

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            self.assertIs(kwargs["position_ids"], positions)
            self.assertNotIn("packed_lengths", kwargs)
            return types.SimpleNamespace(last_hidden_state=input_ids[..., None].float())

        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            model(input_ids=ids, position_ids=positions)

    def test_explicit_positions_without_packed_metadata_must_not_reset(self):
        model = public_model()
        ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
        with self.assertRaisesRegex(ValueError, "require packed document metadata"):
            model(input_ids=ids, position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]))
        with self.assertRaisesRegex(ValueError, "require packed document metadata"):
            model(input_ids=ids, position_ids=torch.arange(6)[None])
        temporal_reset = torch.tensor([[[0, 1, 2, 0, 1, 2]], [[0, 1, 2, 0, 1, 2]], [[0, 1, 2, 0, 1, 2]]])
        with self.assertRaisesRegex(ValueError, "reset without packed document metadata"):
            model(input_ids=ids, position_ids=temporal_reset)
        packed = torch.tensor([[[0, 1, 2, 3, 0, 1, 2, 3]]]).expand(3, 1, 8).clone()

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            return types.SimpleNamespace(last_hidden_state=input_ids[..., None].float())

        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward), \
                patch("training.flash.adapters.QUERY_TILE", 4):
            model(input_ids=torch.arange(8)[None] + 3, position_ids=packed, packed_lengths=(4, 4))
            with self.assertRaisesRegex(ValueError, "reset inside a packed document"):
                model(input_ids=torch.arange(8)[None] + 3, position_ids=packed, packed_lengths=(8,))

    def test_public_padded_batch_is_packed_and_restored(self):
        model = public_model()
        ids = torch.tensor([[4, 5, 0], [7, 8, 9]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            instance.call = (input_ids, attention_mask, kwargs)
            hidden = input_ids[..., None].float()
            return types.SimpleNamespace(
                last_hidden_state=hidden,
                hidden_states=(hidden + 100,),
            )

        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            output = model.forward(input_ids=ids, attention_mask=mask, use_cache=False)

        packed_ids, internal_mask, kwargs = model.call
        self.assertEqual(tuple(packed_ids.shape), (1, 256))
        self.assertEqual(packed_ids[0, :4].tolist(), [4, 5, 2, 2])
        self.assertEqual(packed_ids[0, 128:132].tolist(), [7, 8, 9, 2])
        self.assertEqual(
            internal_mask, {"full_attention": None, "linear_attention": None}
        )
        self.assertEqual(kwargs["packed_lengths"], (128, 128))
        self.assertEqual(kwargs["cu_seq_lens_q"].tolist(), [0, 128, 256])
        self.assertEqual(kwargs["position_ids"][0, :3].tolist(), [0, 1, 2])
        self.assertEqual(kwargs["position_ids"][0, 128:131].tolist(), [0, 1, 2])
        torch.testing.assert_close(
            output.last_hidden_state[..., 0],
            torch.tensor([[4.0, 5.0, 0.0], [7.0, 8.0, 9.0]]),
        )
        torch.testing.assert_close(
            output.hidden_states[0][..., 0],
            torch.tensor([[104.0, 105.0, 0.0], [107.0, 108.0, 109.0]]),
        )

    def test_public_batch_unpack_preserves_only_real_token_gradients(self):
        layout = pack_padded_batch(
            torch.tensor([[1, 2, 0], [3, 4, 5]]),
            torch.tensor([[1, 1, 0], [1, 1, 1]]),
            alignment=4,
        )
        hidden = torch.arange(8.0).reshape(1, 8, 1).requires_grad_()
        restored = layout.unpack(hidden)
        self.assertEqual(restored[..., 0].tolist(), [[0.0, 1.0, 0.0], [4.0, 5.0, 6.0]])
        restored.sum().backward()
        self.assertEqual(
            hidden.grad[..., 0].tolist(), [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
        )

    def test_public_batch_resets_real_ple_ngram_history_at_each_row(self):
        layout = pack_padded_batch(
            torch.tensor([[4, 5, 6, 7], [7, 8, 9, 0]]),
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            eos_token_id=2,
            alignment=4,
        )
        ple = types.SimpleNamespace(eos_token_id=2)
        shift = qwen.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos
        second = layout.spans[1].start
        self.assertEqual(shift(ple, layout.input_ids, 1)[0, second].item(), 2)
        self.assertEqual(shift(ple, layout.input_ids, 2)[0, second].item(), 2)

    def test_explicit_lengths_reset_real_ple_ngram_history_without_eos(self):
        config = qwen.Qwen4ExpTextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            ple_layer_ids=[1],
            ple_embed_dim=16,
            heads_per_ngram=2,
            ngram_size=3,
            ngram_vocab_size_base=1000,
            eos_token_id=2,
        )
        torch.manual_seed(0)
        ple = PackedPLELayer(config, 0, 0)
        self.assertIsInstance(ple.ple_embedding, PackedNGramEmbedding)
        first = torch.full((1, 128), 4)
        second = torch.full((1, 128), 7)
        packed_ids = torch.cat([first, second], dim=1)

        native = ple.ple_embedding(packed_ids, None)
        token = _active_lengths.set((128, 128))
        try:
            packed = ple.ple_embedding(packed_ids, None)
        finally:
            _active_lengths.reset(token)
        alone = torch.cat(
            [ple.ple_embedding(first, None), ple.ple_embedding(second, None)], dim=1
        )

        torch.testing.assert_close(packed, alone, rtol=0, atol=0)
        # Without the packed override, document two inherits token 4 history.
        self.assertFalse(torch.equal(native[:, 128:130], alone[:, 128:130]))
        torch.testing.assert_close(native[:, 130:], alone[:, 130:], rtol=0, atol=0)

    def test_public_batch_rejects_left_padding_and_explicit_packing_mix(self):
        with self.assertRaisesRegex(ValueError, "right-padded"):
            pack_padded_batch(
                torch.tensor([[0, 1], [2, 3]]),
                torch.tensor([[0, 1], [1, 1]]),
            )
        model = public_model()
        with self.assertRaisesRegex(ValueError, "combine"):
            model.forward(
                input_ids=torch.ones(2, 4, dtype=torch.long),
                packed_lengths=(4, 4),
            )

    def test_explicit_lengths_derive_and_validate_delta_net_boundaries(self):
        model = public_model()

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            instance.call = kwargs
            return types.SimpleNamespace(last_hidden_state=input_ids[..., None].float())

        ids = torch.arange(256)[None]
        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            model.forward(input_ids=ids, packed_lengths=(128, 128), use_cache=False)
        self.assertEqual(model.call["cu_seq_lens_q"].tolist(), [0, 128, 256])
        with (
            patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward),
            self.assertRaisesRegex(ValueError, "may not supply both"),
        ):
            model.forward(
                input_ids=ids,
                packed_lengths=(128, 128),
                cu_seq_lens_q=torch.tensor([0, 127, 256], dtype=torch.int32),
                use_cache=False,
            )

    def test_sdpa_path_derives_delta_net_boundaries_once(self):
        model = public_model()
        model._flash_flex_qsa = False

        def native_forward(instance, input_ids=None, attention_mask=None, **kwargs):
            instance.call = kwargs
            return types.SimpleNamespace(last_hidden_state=input_ids[..., None].float())

        ids = torch.arange(256)[None]
        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            model.forward(input_ids=ids, packed_lengths=(128, 128), use_cache=False)
        boundaries = model.call["cu_seq_lens_q"]
        self.assertEqual(boundaries.tolist(), [0, 128, 256])
        # GDN layers and checkpoint recomputes reuse the trusted tensor as is.
        self.assertIs(_packed_boundaries((128, 128), boundaries, 256, ids.device), boundaries)
        with patch.object(qwen.Qwen4ExpTextModel, "forward", native_forward):
            model.forward(input_ids=ids, use_cache=False)
        self.assertNotIn("cu_seq_lens_q", model.call)

    def test_gdn_convolution_is_segmented_and_default_is_unchanged(self):
        def causal(x, weight, bias=None, activation=None, **kwargs):
            return x.cumsum(-1)

        module = QualifiedGatedDeltaNet.__new__(QualifiedGatedDeltaNet)
        nn.Module.__init__(module)
        module.options = FlashOptions()
        module._flash_causal_conv = causal
        module.conv1d = nn.Conv1d(2, 2, 1, groups=2, bias=False)
        module.activation = None
        x = torch.ones(1, 2, 8)
        unpacked = module._convolution(x, None)
        packed = module._convolution(x, (4, 4))
        self.assertEqual(unpacked[0, 0].tolist(), list(range(1, 9)))
        self.assertEqual(packed[0, 0].tolist(), list(range(1, 5)) * 2)


if __name__ == "__main__":
    unittest.main()
