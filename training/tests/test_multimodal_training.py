"""Image bytes, native spans, processor evidence and policy batch invariants."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from training.dataset import TokenizedSample, collate_samples
from training.episode_packing import pack_episode_samples
from training.full_weight_ppo import FullWeightPPOTrainer
from training.multimodal import prepare_training_media
from training.ppo import PPOTrainer, _parity_prefix

# CPU CI reads the pure processing module without installing the GPU harness.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "platypus/packages/ullm/src")
)
media = pytest.importorskip("ullm.media", reason="Image transport tests require the pinned harness submodule")


class ImageProcessor:
    """Small deterministic processor whose tensors depend on every input pixel."""

    merge_size = 2

    def to_dict(self):
        """Expose stable resize and normalization settings."""
        return {"merge_size": self.merge_size, "normalization": "identity"}

    def __call__(self, *, images, return_tensors, **kwargs):
        """Represent each 4x4 image as sixteen RGB patch vectors."""
        assert return_tensors == "pt" and not kwargs
        return {
            "pixel_values": torch.cat(
                [
                    torch.from_numpy(np.array(image)).reshape(-1, 3).float()
                    for image in images
                ]
            ),
            "image_grid_thw": torch.tensor(
                [[1, image.height, image.width] for image in images]
            ),
        }


def fixture_media():
    """Build two ordered image spans, including a repeated-size image."""
    processor = SimpleNamespace(image_processor=ImageProcessor(), image_token_id=99)
    identity = media.processing_identity(
        processor,
        model="model",
        revision="sha",
        mm_processor_kwargs={},
        model_architecture="Qwen3VLForConditionalGeneration",
    )
    images = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    ids = [1, 99, 99, 99, 99, 2, 99, 99, 99, 99, 3]
    payload = media.build_media(ids, images, processor, identity)
    evidence = {
        key: value for key, value in payload.items() if key not in {"version", "images"}
    }
    return processor, ids, payload, evidence


def image_sample(prompt, completion, payload, evidence, *, step_index=0):
    """Keep sampled targets separate from the server's witnessed image prompt."""
    return TokenizedSample(
        trajectory_id="image-episode",
        step_index=step_index,
        input_ids=prompt + completion,
        labels=[-100] * len(prompt) + completion,
        attention_mask=[1] * (len(prompt) + len(completion)),
        completion_mask=[0] * len(prompt) + [1] * len(completion),
        old_token_logprobs=[0.0] * (len(prompt) - 1) + [-0.4] * len(completion),
        old_logprob=-0.4 * len(completion),
        prompt_token_count=len(prompt),
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
        media=payload,
        media_evidence=evidence,
    )


@pytest.mark.parametrize("trainer_class", [PPOTrainer, FullWeightPPOTrainer])
def test_image_trust_region_reproduces_media_and_restores_model_modes(trainer_class):
    processor, ids, payload, evidence = fixture_media()
    sample = image_sample(ids, [7, 8], payload, evidence)
    expected = prepare_training_media(
        [sample.input_ids], [payload], [evidence], processor
    )
    trainer = trainer_class.__new__(trainer_class)
    trainer.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    trainer.model[0].eval()
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.processor = processor
    forwards = []

    def score(batch):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in trainer.model.modules())
        for key, value in expected.items():
            assert torch.equal(batch[key], value)
        forwards.append(batch["input_ids"].tolist())
        return batch["old_token_logprobs"]

    trainer._token_logprobs = score
    actual = trainer._trust_region_token_logprobs([sample])
    assert actual.tolist() == pytest.approx([-0.4, -0.4])
    assert forwards == [[sample.input_ids]]
    assert [module.training for module in trainer.model.modules()] == [True, False, True]


@pytest.mark.parametrize("token_budget", [1, 128, 257, 300])
def test_packed_image_probe_preserves_witness_and_only_scores_selected_targets(
    token_budget,
):
    processor, ids, payload, evidence = fixture_media()
    first = image_sample(ids, [8] * 256, payload, evidence)
    # A later image belongs to the packed witness even when the probe only
    # scores targets from the first request, before that image was introduced.
    next_prompt = first.input_ids + [6, 99, 99, 99, 99, 7]
    next_payload = media.build_media(
        next_prompt,
        [media.decode_image(item) for item in payload["images"]]
        + [Image.new("RGB", (4, 4), "green")],
        processor,
        payload["processing"],
    )
    next_evidence = {
        key: value
        for key, value in next_payload.items()
        if key not in {"version", "images"}
    }
    last = image_sample(next_prompt, [9] * 4, next_payload, next_evidence, step_index=1)
    packed, stats = pack_episode_samples([first, last], max_sequence_tokens=1024)
    assert stats.packed_sequences == 1
    original = deepcopy(packed[0])
    prefix = _parity_prefix(packed[0], token_budget)
    batch = collate_samples([prefix], pad_token_id=0, processor=processor)

    targets = [i for i, mask in enumerate(original.completion_mask) if mask][:token_budget]
    assert [i for i, mask in enumerate(prefix.completion_mask) if mask] == targets
    assert len(prefix.input_ids) == max(targets[-1] + 1, len(next_prompt))
    assert prefix.input_ids == original.input_ids[: len(prefix.input_ids)]
    assert prefix.media is packed[0].media
    assert prefix.media_evidence is packed[0].media_evidence
    assert packed[0] == original
    assert batch["old_token_logprobs"][batch["completion_mask"][:, 1:]].tolist() == (
        pytest.approx([-0.4] * len(targets))
    )
    assert prefix.labels[targets[-1] + 1 :] == [-100] * (
        len(prefix.input_ids) - targets[-1] - 1
    )
    for values in (
        prefix.old_token_logprobs,
        prefix.span_ids,
        prefix.token_advantages,
        prefix.token_levels,
    ):
        assert len(values) == len(prefix.input_ids) - 1
        assert not any(values[targets[-1] :])
    assert prefix.old_logprob == pytest.approx(-0.4 * len(targets))
    first_media = prepare_training_media(
        [first.input_ids], [payload], [evidence], processor
    )
    assert torch.equal(batch["pixel_values"][:32], first_media["pixel_values"])
    assert torch.equal(
        batch["position_ids"][:, :, : len(ids) + min(token_budget, 256)],
        first_media["position_ids"][:, :, : len(ids) + min(token_budget, 256)],
    )
    assert batch["image_grid_thw"].shape == (3, 3)


def test_lossless_png_and_changed_pixel():
    """Same dimensions cannot conceal a changed image or corrupt encoded bytes."""
    image = Image.new("RGB", (4, 4), "red")
    payload = media.encode_image(image)
    assert media.pixel_hash(media.decode_image(payload)) == media.pixel_hash(image)
    image.putpixel((0, 0), (254, 0, 0))
    assert media.pixel_hash(image) != payload["pixel_sha256"]
    payload["png_base64"] = media.encode_image(image)["png_base64"]
    with pytest.raises(ValueError, match="PNG hash"):
        media.decode_image(payload)


def test_ordered_images_and_processed_tensors():
    """Both order and image-processing evidence are mandatory."""
    processor, ids, payload, evidence = fixture_media()
    tensors, actual = media.reproduce_media(
        ids, payload, processor, payload["processing"]
    )
    assert actual == evidence
    assert tensors["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 4]]
    assert actual["layout"] == [
        {"offset": 1, "length": 4, "grid_thw": [1, 4, 4]},
        {"offset": 6, "length": 4, "grid_thw": [1, 4, 4]},
    ]
    swapped = deepcopy(payload)
    swapped["images"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        media.reproduce_media(ids, swapped, processor, payload["processing"])
    changed = deepcopy(payload)
    changed["tensors"]["pixel_values"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tensors"):
        media.reproduce_media(ids, changed, processor, payload["processing"])


@pytest.mark.parametrize(
    "ids", [[1, 99, 99, 99, 2, 99, 99, 99, 99, 3], [1] + [99] * 10]
)
def test_placeholder_expansion_is_exact(ids):
    """Neither shortened nor doubly expanded image runs are accepted."""
    processor, _, payload, _ = fixture_media()
    payload["prompt_length"] = len(ids)
    with pytest.raises(ValueError, match="placeholder"):
        media.reproduce_media(ids, payload, processor, payload["processing"])


def test_qwen_positions_and_mixed_training_batch():
    """Positions compress image geometry and continue through sampled suffixes."""
    processor, ids, payload, evidence = fixture_media()
    result = prepare_training_media(
        [ids + [7, 8], [1, 2, 3]], [payload, None], [evidence, None], processor
    )
    assert result["position_ids"].shape == (3, 2, 13)
    assert result["position_ids"][:, 0, :6].tolist() == [
        [0, 1, 1, 1, 1, 3],
        [0, 1, 1, 2, 2, 3],
        [0, 1, 2, 1, 2, 3],
    ]
    assert result["position_ids"][:, 0, -3:].tolist() == [[6, 7, 8]] * 3
    assert result["position_ids"][:, 1, :3].tolist() == [[0, 1, 2]] * 3
    assert result["mm_token_type_ids"][0].tolist() == [
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    assert result["pixel_values"].shape == (32, 3)


def test_repeated_images_and_pruned_image_context():
    """Repeated bytes remain separate image slots and an explicit reset can evict one."""
    processor, ids, payload, _ = fixture_media()
    image = media.decode_image(payload["images"][0])
    repeated = media.build_media(ids, [image, image], processor, payload["processing"])
    assert repeated["image_hashes"][0] == repeated["image_hashes"][1]
    assert len(repeated["layout"]) == 2
    reset_ids = [5, 99, 99, 99, 99, 6]
    reset = media.build_media(reset_ids, [image], processor, payload["processing"])
    _, observed = media.reproduce_media(
        reset_ids, reset, processor, payload["processing"]
    )
    assert len(observed["layout"]) == 1


def test_packed_unpacked_processing_equivalence():
    """Keeping the final native continuation yields the same image tensor inputs."""
    processor, ids, payload, evidence = fixture_media()
    unpacked = prepare_training_media(
        [ids, ids + [7]], [payload, payload], [evidence, evidence], processor
    )
    packed = prepare_training_media([ids + [7]], [payload], [evidence], processor)
    assert torch.equal(unpacked["pixel_values"][:32], packed["pixel_values"])
    assert torch.equal(unpacked["position_ids"][:, 1], packed["position_ids"][:, 0])


def test_server_evidence_and_processing_must_match():
    """A changed processor or server witness cannot enter an optimization batch."""
    processor, ids, payload, evidence = fixture_media()
    changed = deepcopy(evidence)
    changed["image_hashes"].reverse()
    with pytest.raises(ValueError, match="server evidence"):
        prepare_training_media([ids], [payload], [changed], processor)
    payload["processing"]["pillow"] = "different"
    with pytest.raises(ValueError, match="identity"):
        prepare_training_media([ids], [payload], [evidence], processor)


def test_token_count_pickles_media():
    """Compaction proofs survive the worker proxy without losing media identity."""
    import pickle

    pytest.importorskip("nest_asyncio")
    pytest.importorskip("typeguard")
    pytest.importorskip("jsonschema")
    from ullm.backend import TokenCount

    _, ids, payload, _ = fixture_media()
    restored = pickle.loads(pickle.dumps(TokenCount(len(ids), ids, payload)))
    assert restored.media == payload
    assert restored.prompt_token_ids == tuple(ids)


@pytest.mark.parametrize("architecture", ["Qwen3VLForConditionalGeneration", "Qwen4ExpForConditionalGeneration"])
def test_native_builder_passes_processed_engine_input(monkeypatch, architecture):
    """The server dispatches a real multimodal shape with witnessed pixel tensors."""
    import sys
    import types
    from training.native_vllm_server import NativeService

    processor, ids, payload, evidence = fixture_media()
    processor_config = SimpleNamespace(_get_mm_fields_config=lambda values, kwargs: {})
    native = NativeService.__new__(NativeService)
    native._media_processor = processor
    native._processing_identity = payload["processing"]
    native.engine = SimpleNamespace(
        renderer=SimpleNamespace(mm_processor=processor_config)
    )
    native.model = SimpleNamespace(hf_config=object(), architectures=[architecture])

    class Items(dict):
        """Stand-in for the pinned vLLM field containers, without importing CUDA."""

        @staticmethod
        def from_hf_inputs(tensors, fields):
            """Retain actual processed tensors behind separate image slots."""
            result = Items(image=[{"index": 0}, {"index": 1}])
            result.tensors = tensors
            return result

        def get_data(self):
            """Recombine the exact tensors that will enter the engine."""
            return self.tensors

    def make_input(prompt, kwargs, hashes, placeholders):
        """Expose the exact native EngineInput schema used by vLLM0.25.1."""
        return {
            "type": "multimodal",
            "prompt_token_ids": prompt,
            "mm_kwargs": kwargs,
            "mm_hashes": hashes,
            "mm_placeholders": placeholders,
        }

    def native_positions(*, input_tokens, mm_features, config):
        """Use a separate analytically checked witness for the native position call."""
        assert input_tokens == ids and len(mm_features) == 2
        return torch.tensor(
            [
                [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6],
                [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
                [0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6],
            ]
        ), -4

    modules = {
        "vllm.inputs.engine": {"mm_input": make_input},
        "vllm.multimodal.inputs": {
            "MultiModalFeatureSpec": SimpleNamespace,
            "MultiModalKwargsItems": Items,
            "PlaceholderRange": SimpleNamespace,
        },
        ("vllm.models.qwen4_exp.nvidia.model" if architecture == "Qwen4ExpForConditionalGeneration"
         else "vllm.model_executor.models.qwen3_vl"): {
            architecture: SimpleNamespace(
                _get_mrope_input_positions=native_positions
            )
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    engine_input, observed = native._multimodal_input(ids, payload)
    assert engine_input["type"] == "multimodal"
    assert engine_input["prompt_token_ids"] == ids
    assert "multi_modal_data" not in engine_input
    assert "image" in engine_input["mm_kwargs"]
    assert len(engine_input["mm_placeholders"]["image"]) == 2
    assert observed == evidence


@pytest.mark.parametrize("changed_evidence", [False, True])
def test_engine_proxy_transports_lossless_media_once(changed_evidence):
    """A single remote draw preserves media or fails on a changed server witness."""
    import asyncio

    pytest.importorskip("nest_asyncio")
    pytest.importorskip("typeguard")
    from ullm.backends.engine_proxy import EngineProxy, output_to_wire

    _, ids, payload, evidence = fixture_media()
    requests = []

    class HTTP:
        """Nonretrying in-memory JSON boundary with an explicit consumed witness."""

        async def post(self, url, *, json):
            """Capture one engine envelope and return one finished sampled response."""
            requests.append(deepcopy(json))
            witness = deepcopy(evidence)
            if changed_evidence:
                witness["image_hashes"].reverse()
            body = {
                "request_id": "run#nonce",
                "prompt_token_ids": ids,
                "finished": True,
                "num_cached_tokens": 0,
                "lora_request": None,
                "adapter": None,
                "media_evidence": witness,
                "outputs": [
                    {
                        "index": 0,
                        "text": "answer",
                        "token_ids": [7],
                        "logprobs": [{"7": {"logprob": -0.2, "rank": 1}}],
                        "finish_reason": "length",
                        "stop_reason": None,
                    }
                ],
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    async def execute():
        """Run exactly one request without touching a socket or model tokenizer."""
        proxy = EngineProxy("http://engine", "base")
        proxy.metadata = {"adapter": None, "processing": payload["processing"]}
        proxy._http = HTTP()
        return [
            item
            async for item in proxy.generate(
                {"type": "token", "prompt_token_ids": ids, "media": payload},
                {"max_tokens": 1},
                "run#nonce",
            )
        ]

    if changed_evidence:
        with pytest.raises(ValueError, match="different image"):
            asyncio.run(execute())
    else:
        result = asyncio.run(execute())
        assert result[0].media_evidence == evidence
        assert output_to_wire(result[0], {})["media_evidence"] == evidence
    assert len(requests) == 1
    assert requests[0]["prompt"]["media"] == payload
