"""The ``_qwen38_config`` builder: the card's numbers, and ``_qwen36_config``
left untouched.

The sibling exists so a column can select Qwen3.8 through one seam. What it
must get right is exactly what a reader cannot check at a glance: the thinking-mode sampling matches the model card, the effort opt-in
is on (this is the only config that sets it), and the two engine quirks the
3.6 config carries for other reasons did not tag along.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import ullm.kaggle

deployment = pytest.importorskip("ullm.backends.deployment", reason="needs the vllm extra (linux)")


def test_qwen38_matches_the_card_and_opts_into_effort() -> None:
    """Model id, thinking-mode sampling, and the opt-in flag."""
    config = deployment._qwen38_config("27B-FP8")

    assert config.engine_args.model == "Qwen/Qwen3.8-27B-FP8"
    assert config.engine_args.max_model_len == 262144
    assert config.engine_args.reasoning_parser == "qwen3"

    sp = config.sampling_params
    assert (sp.temperature, sp.top_p, sp.max_tokens) == (1.0, 0.95, 262144)
    assert sp.extra_params == {
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }

    assert config.backend_kwargs["native_reasoning_effort"] is True
    assert config.backend_kwargs["chat_template_kwargs"] == {"preserve_thinking": True}


def test_qwen38_does_not_inherit_the_36_specific_quirks() -> None:
    """Two engine args exist for reasons 3.8 does not have.

    ``max_num_batched_tokens=4096`` sizes around the 35B-A3B's mamba block,
    and the ``mm_processor_kwargs`` truncation override works around a
    Trainer artifact in the vrfai 3.6 checkpoint. Carrying either silently
    would be configuration nobody chose.
    """
    config = deployment._qwen38_config("27B-FP8")
    assert config.engine_args.max_num_batched_tokens is None
    assert not config.engine_args.mm_processor_kwargs


def test_a_path_shaped_model_is_taken_verbatim() -> None:
    """Same rule as every family here: a '/' means the caller chose."""
    config = deployment._qwen38_config("somewhere/custom-38")
    assert config.engine_args.model == "somewhere/custom-38"


def test_qwen36_is_byte_for_byte_what_it_was() -> None:
    """The sibling must not have moved the original.

    Every committed duck reference stands on ``_qwen36_config``'s exact
    output, so this snapshot is the cheap tripwire for an accidental edit.
    """
    config = deployment._qwen36_config("vrfai/Qwen3.6-27B-FP8")
    assert config.engine_args.model == "vrfai/Qwen3.6-27B-FP8"
    assert config.engine_args.max_num_batched_tokens == 4096
    sp = config.sampling_params
    assert (sp.temperature, sp.top_p, sp.max_tokens) == (1.0, 0.95, 262144)
    assert sp.extra_params == {"top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
    assert "native_reasoning_effort" not in config.backend_kwargs
    assert config.backend_kwargs["chat_template_kwargs"] == {"preserve_thinking": True}


@pytest.mark.parametrize("model_family", ["qwen36", "qwen38"])
def test_backend_kwargs_are_pure_and_match_local_family(monkeypatch: pytest.MonkeyPatch, model_family: str) -> None:
    def engine_args(**kwargs: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(deployment.vllm, "AsyncEngineArgs", engine_args)
    monkeypatch.setattr(deployment.ullm.backends.config_local, "set_env", lambda: None)
    builder = deployment._qwen38_config if model_family == "qwen38" else deployment._qwen36_config
    local = builder("test/model")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend defaults must not configure a local engine")

    monkeypatch.setattr(deployment.vllm, "AsyncEngineArgs", forbidden)
    monkeypatch.setattr(deployment.ullm.backends.config_local, "set_env", forbidden)
    kwargs = deployment._qwen_backend_kwargs(model_family)
    expected: dict[str, Any] = {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "chat_template_kwargs": {"preserve_thinking": True},
    }
    if model_family == "qwen38":
        expected["native_reasoning_effort"] = True
    assert kwargs == local.backend_kwargs == expected
    kwargs["chat_template_kwargs"]["preserve_thinking"] = False
    assert deployment._qwen_backend_kwargs(model_family) == local.backend_kwargs == expected


def test_backend_kwargs_reject_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported Qwen backend family"):
        deployment._qwen_backend_kwargs("qwen39")


def test_every_column_asks_for_dense_recurrent_retention() -> None:
    """The engine's default keeps only the boundaries it can prove reachable,
    and a continuation resumes at its own previous end, which is not one.
    Every engine here is a hybrid, so the correction is every column's, not
    the tier's."""
    for engine in deployment.ENGINES:
        engine_args = deployment.build(engine, "slurm-b200").factory.engine_args
        assert engine_args.prefix_cache_retention_interval is None, engine


def test_the_ple_table_is_served_from_the_host() -> None:
    """Flash-Next's cluster cell keeps the PLE table in pinned host memory
    through vLLM's Engram offload, and the generic weight offloader stays
    off; the other engines have no table and offload nothing."""
    engine_args = deployment.build("qwen3.8-flash-next", "slurm-b200").factory.engine_args
    assert engine_args.engram_config is not None and engine_args.engram_config.cpu_offload
    assert engine_args.cpu_offload_gb == 0, "the table is not the generic offloader's"
    for other in ("qwen3.8-27b", "qwen3.6-27b"):
        other_args = deployment.build(other, "slurm-b200").factory.engine_args
        assert other_args.cpu_offload_gb == 0, other
        assert other_args.engram_config is None or not other_args.engram_config.cpu_offload, other


def test_flash_next_drafts_with_its_head_where_it_is_measured() -> None:
    """Flash-Next drafts four tokens a step with its fine-tuned MTP head
    (RecoverSSM, drafts verified against their probabilities, no trailing
    block dropped), each cell naming where the head is; the four L4s, where
    drafting is descoped, do not draft."""
    b200 = deployment.build("qwen3.8-flash-next", "slurm-b200").factory.engine_args.speculative_config
    assert b200 is not None
    assert b200["method"] == "mtp" and b200["num_speculative_tokens"] == 4
    assert b200["draft_sample_method"] == "probabilistic"
    assert b200["gdn_recoverssm"] and b200["disable_eagle_block_drop"]
    assert b200["model"] == deployment.FLASHNEXT_E307_DRAFTER
    rtx = deployment.build("qwen3.8-flash-next", "kaggle-rtx")
    speculative = rtx.factory.engine_args.speculative_config
    assert speculative is not None
    assert speculative["model"] == ullm.kaggle.model_mount(deployment.FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL)
    assert deployment.FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL in rtx.kaggle_model_sources
    l4 = deployment.build("qwen3.8-flash-next", "kaggle-4xl4")
    assert l4.factory.engine_args.speculative_config == speculative
    assert deployment.FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL in l4.kaggle_model_sources
    for other in ("qwen3.8-27b", "qwen3.6-27b"):
        assert deployment.build(other, "slurm-b200").factory.engine_args.speculative_config is None, other


def test_flash_next_keeps_conversations_live_between_turns_on_every_cell() -> None:
    """Every Flash-Next cell keys the prefix cache every 80 tokens, keeps a
    decode tail and names each conversation's thread, which puts the host
    tier in park mode. 80 divides every cell's block (1,840 at four drafts,
    on one card or on two-rank stages); the other engines keep vLLM's
    defaults."""
    for environment in ("slurm-b200", "kaggle-rtx", "kaggle-4xl4"):
        engine_args = deployment.build("qwen3.8-flash-next", environment).factory.engine_args
        assert engine_args.prefix_match_unit == 80, environment
        assert engine_args.prefix_cache_decode_tail and engine_args.prefix_cache_context_threads, environment
    for other in ("qwen3.8-27b", "qwen3.6-27b"):
        assert not deployment.build(other, "slurm-b200").factory.engine_args.prefix_cache_context_threads, other


def test_a_task_serves_its_own_checkpoint_on_a_kaggle_cell() -> None:
    """The cell's pair replaced: the engine reads the new paths, the kernel
    mounts the new datasets and nothing of the old, and a split checkpoint is
    assembled in the setup -- on the RTX the read-ahead follows it."""
    model = ullm.kaggle.KaggleCheckpoint("owner/prune", shards_from=("owner/base",))
    drafter = ullm.kaggle.KaggleCheckpoint("owner/prune-drafter")
    for environment in ("kaggle-4xl4", "kaggle-rtx"):
        served = deployment.build("qwen3.8-flash-next", environment)
        deployment.serve_kaggle_checkpoint(served, model, drafter)
        engine_args = served.factory.engine_args
        assert engine_args.model == model.path, environment
        assert engine_args.speculative_config["model"] == drafter.path, environment
        assert served.kaggle_model_sources == (), environment
        assert served.kaggle_dataset_sources == ("owner/prune", "owner/base", "owner/prune-drafter"), environment
        assert served.kaggle_engine is not None
        assert served.kaggle_engine.assemble == (model,), environment
        assert served.kaggle_engine.setup_commands()[0] == model.assemble_command(), environment
    rtx = deployment.build("qwen3.8-flash-next", "kaggle-rtx")
    deployment.serve_kaggle_checkpoint(rtx, model, drafter)
    assert rtx.kaggle_engine is not None and rtx.kaggle_engine.prefetch == (model.path, drafter.path)


def test_a_split_checkpoint_taking_shards_from_a_model_mounts_that_model() -> None:
    """ARC2's prune once its base's files live in a Kaggle model: the kernel
    mounts the prune's dataset and the base's model version, and no dataset of
    the base's."""
    base = "owner/base/Transformers/default/1"
    model = ullm.kaggle.KaggleCheckpoint("owner/prune", shards_from=(base,))
    drafter = ullm.kaggle.KaggleCheckpoint("owner/prune-drafter")
    served = deployment.build("qwen3.8-flash-next", "kaggle-4xl4")
    deployment.serve_kaggle_checkpoint(served, model, drafter)
    assert served.kaggle_dataset_sources == ("owner/prune", "owner/prune-drafter")
    assert served.kaggle_model_sources == (base,)
    assert served.kaggle_engine is not None and served.kaggle_engine.assemble == (model,)


def test_serving_another_checkpoint_refuses_what_it_cannot_replace() -> None:
    """A drafting cell without a drafter, and a cell that is not on Kaggle."""
    model = ullm.kaggle.KaggleCheckpoint("owner/prune")
    with pytest.raises(ValueError, match="takes a drafter"):
        deployment.serve_kaggle_checkpoint(deployment.build("qwen3.8-flash-next", "kaggle-4xl4"), model)
    with pytest.raises(ValueError, match="not a Kaggle cell"):
        deployment.serve_kaggle_checkpoint(deployment.build("qwen3.8-flash-next", "slurm-b200"), model)
