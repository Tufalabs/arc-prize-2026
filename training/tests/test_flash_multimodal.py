"""CPU image scatter, witnessed M-RoPE, and document packing contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from training.flash.input_layout import pack_padded_batch
from training.flash_model import FlashMultimodalDecoder, load_flash_vision_tower


def rope_reference(ids, types, grids, mask=None):
    model = qwen.Qwen4ExpModel.__new__(qwen.Qwen4ExpModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    return model.get_rope_index(ids, types, image_grid_thw=grids, attention_mask=mask)[0]


def test_witnessed_positions_equal_hf_before_and_after_packing():
    from test_multimodal_training import ImageProcessor, media
    from PIL import Image
    from training.multimodal import prepare_training_media

    processor = SimpleNamespace(image_processor=ImageProcessor(), image_token_id=99)
    identity = media.processing_identity(processor, model="flash", revision="sha",
                                         mm_processor_kwargs={},
                                         model_architecture="Qwen4ExpForConditionalGeneration")
    # The first image crosses QUERY_TILE=128, and the second has a different aspect ratio.
    prompt = [3] * 126 + [99] * 6 + [4, 5] + [99] * 4 + [6]
    payload = media.build_media(prompt, [Image.new("RGB", (6, 4)), Image.new("RGB", (4, 4))],
                                processor, identity)
    evidence = {key: value for key, value in payload.items() if key not in {"version", "images"}}
    rows = [prompt + [7, 8], [3, 4, 5]]
    result = prepare_training_media(rows, [payload, None], [evidence, None], processor)
    ids = torch.zeros(2, len(rows[0]), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(rows):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    expected = rope_reference(ids, result["mm_token_type_ids"], result["image_grid_thw"], mask)
    assert torch.equal(result["position_ids"], expected)
    layout = pack_padded_batch(ids, mask, result["position_ids"])
    packed_types = layout.pack(result["mm_token_type_ids"])
    for row, span in enumerate(layout.spans):
        grids = result["image_grid_thw"] if row == 0 else None
        packed_positions = rope_reference(layout.input_ids[:, span.start:span.padded_stop],
                                          packed_types[:, span.start:span.padded_stop], grids)
        assert torch.equal(layout.position_ids[:, :, span.start:span.padded_stop], packed_positions)
    assert layout.input_ids[0, 126:132].tolist() == [99] * 6
    assert torch.equal(layout.unpack(packed_types), result["mm_token_type_ids"])
    # Exercise the exact HF interleaving used by build_flash_model, including the indexer.
    config = qwen.Qwen4ExpTextConfig(head_dim=256, rope_parameters={
        "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    })
    rotary = qwen.Qwen4ExpTextRotaryEmbedding(config)
    hidden = torch.zeros(2, ids.shape[1], 1, dtype=torch.bfloat16)
    public_cos, public_sin = rotary(hidden, expected)
    packed_cos, packed_sin = rotary(hidden[:1], layout.position_ids)
    for public, packed in ((public_cos, packed_cos), (public_sin, packed_sin)):
        assert torch.equal(layout.unpack(packed)[mask.bool()], public[mask.bool()])


class ToyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8, dtype=torch.bfloat16)
        self.call = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.call = kwargs
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds)


class ToyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 8, dtype=torch.bfloat16))
        self.calls = 0

    def forward(self, pixels, *, grid_thw, return_dict):
        assert not torch.is_grad_enabled() and pixels.dtype == torch.bfloat16
        assert grid_thw.tolist() == [[1, 2, 4]] and return_dict
        self.calls += 1
        return SimpleNamespace(pooler_output=self.weight * 3)


def test_scatter_raw_ple_ids_types_and_no_image_text_identity():
    text, visual = ToyText(), ToyVision()
    decoder = FlashMultimodalDecoder(visual, text, image_token_id=15)
    ids = torch.tensor([[3, 15, 15, 4]])
    mask = torch.ones_like(ids)
    positions = torch.arange(4)[None, None].expand(3, 1, -1)
    media = dict(pixel_values=torch.zeros(8, 3), image_grid_thw=torch.tensor([[1, 2, 4]]),
                 position_ids=positions, mm_token_type_ids=(ids == 15).long())
    result = decoder(ids, mask, **media, packed_lengths=(4,)).last_hidden_state
    assert torch.equal(result[ids == 15], torch.full((2, 8), 3, dtype=torch.bfloat16))
    assert torch.equal(result[ids != 15], text.embed_tokens(ids)[ids != 15])
    assert text.call["ple_input_ids"] is ids and text.call["position_ids"] is positions
    assert text.call["packed_lengths"] == (4,)
    result.sum().backward()
    assert visual.weight.grad is None
    assert torch.count_nonzero(text.embed_tokens.weight.grad[15]) == 0
    ordinary = torch.tensor([[3, 4, 5]])
    assert torch.equal(decoder(ordinary).last_hidden_state, text(ordinary).last_hidden_state)
    assert visual.calls == 1
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        decoder(ids, **{**media, "mm_token_type_ids": torch.zeros_like(ids)})
    with pytest.raises(ValueError, match="placeholders"):
        decoder(torch.tensor([[3, 15, 4, 5]]), **{**media, "mm_token_type_ids": None})


def test_image_and_text_batches_share_one_embedding_call_sequence():
    # FSDP ranks may hold different batch types in one step; the sharded
    # embedding must see the same calls on every rank.
    text, visual = ToyText(), ToyVision()
    decoder = FlashMultimodalDecoder(visual, text, image_token_id=15)
    calls = []
    text.embed_tokens.register_forward_hook(lambda module, args, output: calls.append(tuple(args[0].shape)))
    ids = torch.tensor([[3, 15, 15, 4]])
    mask = torch.ones_like(ids)
    positions = torch.arange(4)[None, None].expand(3, 1, -1)
    decoder(ids, mask, pixel_values=torch.zeros(8, 3), image_grid_thw=torch.tensor([[1, 2, 4]]),
            position_ids=positions, mm_token_type_ids=(ids == 15).long())
    with_images, calls[:] = list(calls), []
    plain = torch.tensor([[3, 4, 5, 6]])
    decoder(plain, mask)
    assert calls == with_images == [(1, 4)]
    assert text.call["ple_input_ids"] is plain and text.call["position_ids"] is None
    assert text.call["attention_mask"] is mask


def test_positions_reset_inside_a_document_is_detected():
    from training.flash.input_layout import positions_reset

    text = torch.tensor([[0, 1, 2, 0, 1, 2]])
    assert positions_reset(text, [(0, 0, 6)])
    assert not positions_reset(text, [(0, 0, 3), (0, 3, 6)])
    # M-RoPE: image rows and columns step back inside a document, the temporal axis never does.
    mrope = torch.tensor([[[0, 1, 1, 1, 1, 2]], [[0, 1, 1, 2, 2, 3]], [[0, 1, 2, 1, 2, 3]]])
    assert not positions_reset(mrope, [(0, 0, 6)])
    assert positions_reset(torch.cat([mrope, mrope], dim=-1), [(0, 0, 12)])
    with pytest.raises(ValueError, match="reset inside a padded row"):
        pack_padded_batch(torch.ones(1, 6, dtype=torch.long), None, text, eos_token_id=2, alignment=4)


def test_vision_loader_freezes_bf16_weights_and_rebuilds_rotary(tmp_path):
    import json

    config = qwen.Qwen4ExpVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                       num_heads=4, out_hidden_size=8, patch_size=2,
                                       temporal_patch_size=1, spatial_merge_size=2,
                                       num_position_embeddings=64)
    config._attn_implementation = "sdpa"
    visual = qwen.Qwen4ExpVisionModel(config).bfloat16().eval()
    rotary = visual.rotary_pos_emb
    visual.rotary_pos_emb = type(rotary)(rotary.dim, theta=rotary.theta)
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": config.to_dict()}))
    weights = {"model.visual." + name: value for name, value in visual.state_dict().items()}
    loaded = load_flash_vision_tower(tmp_path, device="cpu",
                                   reader=SimpleNamespace(weight_map=weights, tensor=weights.__getitem__))
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())
    assert not any(b.is_meta for b in loaded.buffers())
    pixels, grids = torch.randn(8, 12).bfloat16(), torch.tensor([[1, 2, 4]])
    with torch.no_grad():
        assert torch.equal(loaded(pixels, grid_thw=grids).pooler_output,
                           visual(pixels, grid_thw=grids).pooler_output)


def test_embedding_pack_preserves_gradients_and_eos_fill():
    ids = torch.tensor([[3, 4, 0], [5, 6, 7]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    layout = pack_padded_batch(ids, mask, eos_token_id=2, alignment=4)
    values = torch.randn(2, 3, 8, requires_grad=True)
    eos = torch.randn(8)
    packed = layout.pack(values, eos)
    assert torch.equal(packed[0, 2:4], eos.expand(2, -1))
    layout.unpack(packed).sum().backward()
    assert torch.equal(values.grad, mask[..., None].expand_as(values).float())
