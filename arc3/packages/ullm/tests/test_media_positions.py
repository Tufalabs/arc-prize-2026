"""U5: image position evidence matches model positions across image boundaries."""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch
import ullm.media


@pytest.fixture
def position_case() -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    prompt_ids = [9, 2, *([1] * 6), 3, 9, 2, 1, 1, 3]
    processing = {
        "image_token_id": 1,
        "spatial_merge_size": 2,
        "model_architecture": "Qwen3_5ForConditionalGeneration",
        "position_algorithm": "qwen3-vl-image-mrope-v1",
    }
    layout = ullm.media.image_layout(prompt_ids, torch.tensor([[1, 4, 6], [1, 4, 2]]), processing)
    return prompt_ids, layout, processing


def test_rectangular_images_advance_text_positions_by_largest_spatial_axis(
    position_case: tuple[list[int], list[dict[str, Any]], dict[str, Any]],
) -> None:
    prompt_ids, layout, processing = position_case
    actual = ullm.media.position_inputs(len(prompt_ids), layout, processing)
    assert actual["position_ids"].tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 8, 8, 10],
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 8, 9, 10],
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 8, 10],
    ]
    assert actual["mm_token_type_ids"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]


@pytest.mark.parametrize("field,value", [("model_architecture", "UnknownModel"), ("position_algorithm", "flat")])
def test_position_inputs_refuses_unverified_algorithms(
    position_case: tuple[list[int], list[dict[str, Any]], dict[str, Any]], field: str, value: str
) -> None:
    prompt_ids, layout, processing = position_case
    with pytest.raises(ValueError, match="model architecture"):
        ullm.media.position_inputs(len(prompt_ids), layout, {**processing, field: value})


@pytest.mark.parametrize(
    "module_name,architecture",
    [
        ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
    ],
)
def test_position_evidence_matches_installed_vllm_model(
    position_case: tuple[list[int], list[dict[str, Any]], dict[str, Any]], module_name: str, architecture: str
) -> None:
    model_module = pytest.importorskip(f"vllm.model_executor.models.{module_name}")
    mm = pytest.importorskip("vllm.multimodal.inputs")
    prompt_ids, layout, processing = position_case
    model_class = getattr(model_module, architecture)
    model = model_class.__new__(model_class)
    model.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2),
        video_token_id=4,
        vision_start_token_id=2,
        vision_end_token_id=3,
    )
    features = [
        mm.MultiModalFeatureSpec(
            data=mm.MultiModalKwargsItem(
                image_grid_thw=mm.MultiModalFieldElem(
                    data=torch.tensor(span["grid_thw"]),
                    field=mm.MultiModalBatchedField(),
                )
            ),
            modality="image",
            identifier=str(index),
            mm_position=mm.PlaceholderRange(offset=span["offset"], length=span["length"]),
        )
        for index, span in enumerate(layout)
    ]
    expected, delta = model.get_mrope_input_positions(prompt_ids, features)
    actual = ullm.media.position_inputs(len(prompt_ids), layout, {**processing, "model_architecture": architecture})
    assert torch.equal(actual["position_ids"], expected)
    assert delta == int(actual["position_ids"].max()) + 1 - len(prompt_ids)


def test_flash_next_positions_match_transformers_rope_index(
    position_case: tuple[list[int], list[dict[str, Any]], dict[str, Any]],
) -> None:
    """Flash-Next (Qwen4Exp) walks text and image positions as Qwen3-VL does:
    ullm's evidence equals Transformers' own ``get_rope_index``."""
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    prompt_ids, layout, processing = position_case
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))
    )
    fake.get_vision_position_ids = types.MethodType(modeling.Qwen4ExpModel.get_vision_position_ids, fake)
    processing = {**processing, "model_architecture": "Qwen4ExpForConditionalGeneration"}
    actual = ullm.media.position_inputs(len(prompt_ids), layout, processing)
    expected, delta = modeling.Qwen4ExpModel.get_rope_index(
        fake,
        torch.tensor([prompt_ids]),
        mm_token_type_ids=actual["mm_token_type_ids"].unsqueeze(0),
        image_grid_thw=torch.tensor([span["grid_thw"] for span in layout]),
    )
    assert torch.equal(actual["position_ids"], expected[:, 0])
    assert int(delta) == int(actual["position_ids"].max()) + 1 - len(prompt_ids)
