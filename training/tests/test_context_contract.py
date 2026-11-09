"""Context budgets preserve the shipped recipes and legacy gateway policy."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from training.config import (
    ExperimentConfig,
    InferenceConfig,
    full_game_context_contract,
    max_served_sequence_tokens,
    resolve_context_contract,
)
from training.context_contract import ContextContract
from training.gateway_lineage import LineagePolicy
from training.official_eval import _collector_config
from training.rollout_gateway import GatewayError, SamplingPolicy


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# The recipes that keep the 32k-era derivation (4,096 summary cap and a
# 49,152 served ceiling for ARC-2), listed explicitly: a new recipe is not
# subject to these rules unless it is added here. The Flash recipes take the
# deployment harness's context and are checked by tests/test_flash_arc2_recipe.py.
FLASH_RECIPE = CONFIGS / "flash_8gpu_arc2_grpo_tokmean.json"
LEGACY_CONTEXT_CONFIG_NAMES = (
    "dppo_2gpu_arc3_ar25.json",
    "dppo_2gpu_arc3_ar25_hidden_critic.json",
    "dppo_2gpu_arc3_multimodal_smoke.json",
    "dppo_9gpu_arc3_official25_hidden_critic_seed61.json",
    "dr_grpo_6gpu_arc2_overfit.json",
    "eval_8gpu_arc3_ctx64k.json",
    "grpo_3gpu_arc2_heldout.json",
    "grpo_3gpu_arc2_overfit.json",
    "grpo_6gpu_arc2_overfit.json",
    "grpo_6gpu_arc2_overfit_score_centering.json",
    "grpo_6gpu_arc2_overfit_terminal_empty_episode_mean.json",
    "grpo_8gpu_arc2_full_weight_heldout.json",
    "grpo_8gpu_arc2_full_weight_overfit.json",
    "grpo_8gpu_arc2_full_weight_overfit_60k.json",
    "ppo_2gpu_arc3_games25_fix_control.json",
    "ppo_2gpu_arc3_games25_fix_noretry.json",
    "ppo_2gpu_arc3_games25_fix_tokmean.json",
    "ppo_2gpu_arc3_games25_fix_tokvoid.json",
    "ppo_2gpu_arc3_games25_fix_void.json",
    "ppo_2gpu_arc3_games25_lrhalf.json",
    "ppo_2gpu_arc3_games25_platypus.json",
    "ppo_2gpu_arc3_pathfinder_platypus_debug.json",
    "ppo_2gpu_arc3_platypus_lora_smoke.json",
    "ppo_3gpu_arc3_synthetic49_heldout_official25.json",
    "ppo_4gpu_arc3_synthetic49_heldout_official25.json",
    "ppo_5gpu_arc3_officials_platypus.json",
    "ppo_5gpu_arc3_synthetic49_heldout_official25.json",
    "ppo_8gpu_arc3_games25_hidden.json",
    "ppo_8gpu_arc3_synthetic49_heldout_official25.json",
)
CONTEXT_CONFIGS = [CONFIGS / name for name in LEGACY_CONTEXT_CONFIG_NAMES]
assert all(path.is_file() for path in CONTEXT_CONFIGS)


def _flash_recipe_at_32k() -> dict:
    """The Flash recipe with the 32k lengths it shipped before the deployment
    context (the rules below are stated at those lengths)."""
    payload = json.loads(FLASH_RECIPE.read_text())
    payload["inference"].update(
        agent_context_window=32768,
        agent_max_completion_tokens=4096,
        agent_safety_margin_tokens=0,
        agent_compact_to_tokens=28672,
        agent_max_summary_tokens=4096,
        agent_max_served_sequence_tokens=49152,
    )
    payload["training"]["max_sequence_tokens"] = 49152
    return payload


def _inference(**changes) -> InferenceConfig:
    return InferenceConfig(
        **{
            "base_url": "http://unused.invalid",
            "model_id": "model",
            "game": "ar25",
            "games": ["ar25"],
            "max_runtime_minutes": 15,
            "agent_context_window": 32768,
            "agent_max_completion_tokens": 4096,
            "agent_compact_to_tokens": 16384,
            **changes,
        }
    )


@pytest.mark.parametrize("path", CONTEXT_CONFIGS, ids=lambda path: path.name)
def test_shipped_context_contract_matches_legacy_derivation(path: Path) -> None:
    inference = ExperimentConfig.load(path).inference
    context = resolve_context_contract(inference)
    window = inference.agent_context_window
    reply = inference.agent_max_completion_tokens
    margin = inference.agent_safety_margin_tokens
    margin = 512 if margin is None else margin
    trigger = max(1024, window - reply - margin)
    target = inference.agent_compact_to_tokens
    assert context == ContextContract(
        context_window_tokens=window,
        max_completion_tokens=reply,
        max_summary_completion_tokens=4096 if "arc2" in path.name else 0,
        safety_margin_tokens=margin,
        compaction_trigger_tokens=trigger,
        reset_prompt_ceiling_tokens=target,
        max_served_sequence_tokens=49152 if "arc2" in path.name else window,
        max_compaction_proof_tokens=2 * window,
        summary_forbidden_token_ids=(248058,) if "arc2" in path.name else (),
    )
    assert context.lineage_policy() == LineagePolicy(trigger, target, 2 * window)
    assert full_game_context_contract(inference) == (trigger, target, margin)


@pytest.mark.parametrize(
    "name",
    [
        "grpo_6gpu_arc2_overfit.json",
        "grpo_6gpu_arc2_overfit_terminal_empty_episode_mean.json",
    ],
)
def test_six_gpu_arc2_context_numbers(name: str) -> None:
    assert resolve_context_contract(
        ExperimentConfig.load(CONFIGS / name).inference
    ) == (
        ContextContract(
            context_window_tokens=32768,
            max_completion_tokens=4096,
            max_summary_completion_tokens=4096,
            safety_margin_tokens=0,
            compaction_trigger_tokens=28672,
            reset_prompt_ceiling_tokens=28672,
            max_served_sequence_tokens=49152,
            max_compaction_proof_tokens=65536,
            summary_forbidden_token_ids=(248058,),
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_window_tokens", True),
        ("max_completion_tokens", 1.5),
        ("max_summary_completion_tokens", True),
        ("safety_margin_tokens", False),
        ("compaction_trigger_tokens", None),
        ("reset_prompt_ceiling_tokens", "16384"),
        ("max_served_sequence_tokens", 32768.0),
        ("max_compaction_proof_tokens", True),
    ],
)
def test_direct_context_contract_rejects_noninteger_budgets(field, value) -> None:
    budgets = vars(resolve_context_contract(_inference())).copy()
    budgets[field] = value
    with pytest.raises(ValueError, match="budgets must be integers"):
        ContextContract(**budgets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_window_tokens", 0),
        ("max_completion_tokens", -1),
        ("max_summary_completion_tokens", -1),
        ("max_completion_tokens", 32768),
        ("safety_margin_tokens", -1),
        ("compaction_trigger_tokens", 0),
        ("compaction_trigger_tokens", 65536),
        ("reset_prompt_ceiling_tokens", 0),
        ("reset_prompt_ceiling_tokens", 28161),
        ("max_served_sequence_tokens", 32767),
        ("max_compaction_proof_tokens", 32767),
    ],
)
def test_direct_context_contract_rejects_inconsistent_budgets(field, value) -> None:
    budgets = vars(resolve_context_contract(_inference())).copy()
    budgets[field] = value
    with pytest.raises(ValueError, match="positive and consistent"):
        ContextContract(**budgets)


def test_direct_context_contract_rejects_invalid_summary_fields() -> None:
    budgets = vars(resolve_context_contract(_inference())).copy()
    with pytest.raises(ValueError, match="positive and consistent"):
        ContextContract(**{**budgets, "max_summary_completion_tokens": 4096})
    with pytest.raises(ValueError, match="positive and consistent"):
        ContextContract(**{**budgets, "summary_forbidden_token_ids": (2,)})
    with pytest.raises(ValueError, match="nonnegative integers"):
        ContextContract(**{**budgets, "summary_forbidden_token_ids": (True,)})


def test_context_contract_allows_reset_ceiling_at_trigger() -> None:
    budgets = vars(resolve_context_contract(_inference())).copy()
    budgets["reset_prompt_ceiling_tokens"] = budgets["compaction_trigger_tokens"]
    budgets["max_served_sequence_tokens"] = 49152
    assert ContextContract(**budgets).reset_prompt_ceiling_tokens == 28160


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_context_window": None},
        {"agent_context_window": 0},
        {"agent_context_window": True},
        {"agent_max_completion_tokens": 0},
        {"agent_max_completion_tokens": 32768},
        {"agent_max_completion_tokens": True},
        {"agent_safety_margin_tokens": -1},
        {"agent_safety_margin_tokens": False},
        {"agent_compact_to_tokens": 0},
        {"agent_compact_to_tokens": True},
        {"agent_compact_to_tokens": 28160},
    ],
)
def test_full_game_validation_messages_are_unchanged(changes) -> None:
    message = (
        "agent_compact_to_tokens must be positive and below the context trigger"
        if "agent_compact_to_tokens" in changes
        else "full-game serving requires explicit positive context/reply budgets and a nonnegative margin"
    )
    for resolve in (resolve_context_contract, full_game_context_contract):
        with pytest.raises(ValueError) as error:
            resolve(_inference(**changes))
        assert str(error.value) == message


def test_trigger_floor_and_default_margin() -> None:
    context = resolve_context_contract(
        _inference(
            agent_context_window=2048,
            agent_max_completion_tokens=1024,
            agent_compact_to_tokens=512,
        )
    )
    assert context.compaction_trigger_tokens == 1024
    assert context.safety_margin_tokens == 512


def test_contract_sampling_policy_keeps_admission_boundaries() -> None:
    context = resolve_context_contract(_inference(agent_safety_margin_tokens=0))
    policy = SamplingPolicy("model", 1.0, 1.0, -1, context=context)
    assert policy.context.max_served_sequence_tokens == 32768
    assert policy.context.max_completion_tokens == 4096
    assert policy.context.lineage_policy() == LineagePolicy(28672, 16384, 65536)
    body = {
        "model": "model",
        "user": "run#nonce",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_p": 0.0,
        "min_tokens": 0,
        "prompt": [1] * (32768 - 4096),
        "max_tokens": 4096,
    }
    policy.validate(body)
    with pytest.raises(GatewayError, match="over-trigger requests"):
        policy.validate({**body, "prompt": body["prompt"] + [1]})
    with pytest.raises(GatewayError, match="explicit completion limit"):
        policy.validate({**body, "max_tokens": 4097})


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"agent_compact_to_tokens": 0}, "agent_compact_to_tokens"),
        ({"agent_max_completion_tokens": 0}, "positive context/reply budgets"),
        ({"agent_max_completion_tokens": -1}, "positive context/reply budgets"),
    ],
)
def test_loader_preserves_deferred_full_game_validation(
    tmp_path: Path, changes, message
) -> None:
    payload = json.loads((CONFIGS / "ppo_2gpu_arc3_games25_platypus.json").read_text())
    payload["inference"].update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = ExperimentConfig.load(path)
    with pytest.raises(ValueError, match=message):
        resolve_context_contract(config.inference)
    payload["training"]["max_sequence_tokens"] = 32767
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as error:
        ExperimentConfig.load(path)
    assert str(error.value) == (
        "training.max_sequence_tokens=32767 is below inference.agent_context_window=32768: "
        "a rollout segment may legally grow to the served ceiling, so a smaller training "
        "budget guarantees a fatal packing error mid-run. Use one max context for both sides."
    )


def test_official_eval_overrides_resolve_through_the_same_contract() -> None:
    config = ExperimentConfig.load(CONFIGS / "ppo_5gpu_arc3_officials_platypus.json")
    evaluation = _collector_config(config, base_url="http://unused.invalid")
    context = resolve_context_contract(evaluation.inference)
    assert context.context_window_tokens == context.max_served_sequence_tokens == 32768
    assert context.lineage_policy() == LineagePolicy(28160, 16384, 65536)
    assert resolve_context_contract(config.inference).context_window_tokens == 65536


@pytest.mark.parametrize("window", [None, 32768])
@pytest.mark.parametrize("reply", [-1, 0])
def test_sequence_ceiling_does_not_validate_full_game_budgets(window, reply) -> None:
    inference = _inference(
        agent_context_window=window,
        agent_max_completion_tokens=reply,
        agent_compact_to_tokens=0,
    )
    assert max_served_sequence_tokens(inference) == window
    with pytest.raises(ValueError, match="positive context/reply budgets"):
        resolve_context_contract(inference)


def test_context_policy_replacement_preserves_unrelated_fields_and_can_swap_context() -> (
    None
):
    context = resolve_context_contract(_inference())
    policy = SamplingPolicy("model", 1.0, 1.0, -1, context=context)
    updated = replace(policy, model="other", max_runtime_seconds=60)
    assert updated.model == "other"
    assert updated.max_runtime_seconds == 60
    assert updated.context is context
    assert updated.context.lineage_policy() == policy.context.lineage_policy()
    other = resolve_context_contract(
        _inference(
            agent_context_window=65536,
            agent_max_completion_tokens=8192,
            agent_compact_to_tokens=32768,
        )
    )
    swapped = replace(updated, context=other)
    assert swapped.context is other
    assert swapped.context.max_served_sequence_tokens == 65536
    assert swapped.context.max_completion_tokens == 8192
    assert swapped.context.lineage_policy() == other.lineage_policy()
    assert swapped.model == "other"
    assert swapped.max_runtime_seconds == 60


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"agent_max_summary_tokens": -1}, "nonnegative integer"),
        ({"agent_max_summary_tokens": True}, "nonnegative integer"),
        ({"agent_max_served_sequence_tokens": 32767}, "at least the context window"),
        ({"agent_max_served_sequence_tokens": True}, "at least the context window"),
        ({"agent_max_summary_tokens": 4096}, "above the window"),
    ],
)
def test_extended_contract_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        resolve_context_contract(_inference(**changes))


@pytest.mark.parametrize("ceiling", [49152, 98304])
def test_served_ceiling_and_proof_ceiling(ceiling):
    inference = _inference(
        agent_max_summary_tokens=4096, agent_max_served_sequence_tokens=ceiling
    )
    context = resolve_context_contract(inference)
    assert max_served_sequence_tokens(inference) == ceiling
    assert context.max_served_sequence_tokens == ceiling
    assert context.max_compaction_proof_tokens == max(65536, ceiling)
    assert full_game_context_contract(inference) == (28160, 16384, 512)


@pytest.mark.parametrize("explicit", [False, True])
def test_loader_resolves_optional_fields_and_checks_training_ceiling(
    tmp_path, explicit
):
    payload = json.loads((CONFIGS / "ppo_2gpu_arc3_games25_platypus.json").read_text())
    payload["inference"].update(
        agent_max_summary_tokens=4096 if explicit else None,
        agent_max_served_sequence_tokens=49152 if explicit else None,
    )
    if explicit:
        payload["inference"].update(
            agent_compact_to_tokens=28672,
            agent_summary_forbidden_token_ids=[248058],
        )
    payload["training"]["max_sequence_tokens"] = 49152
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    inference = ExperimentConfig.load(path).inference
    context = resolve_context_contract(inference)
    assert context.max_summary_completion_tokens == (4096 if explicit else 0)
    assert max_served_sequence_tokens(inference) == (49152 if explicit else 32768)
    if explicit:
        payload["training"]["max_sequence_tokens"] = 49151
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="agent_max_served_sequence_tokens=49152"):
            ExperimentConfig.load(path)


def test_over_trigger_sampling_bounds_and_unapproved_logit_bias():
    context = resolve_context_contract(
        _inference(
            agent_max_summary_tokens=8192, agent_max_served_sequence_tokens=49152
        )
    )
    policy = SamplingPolicy("model", 1.0, 1.0, -1, context=context)
    body = dict(
        model="model",
        user="run#nonce",
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_p=0.0,
        min_tokens=0,
        prompt=[1] * (49152 - 8192),
        max_tokens=8192,
    )
    policy.validate(body)
    with pytest.raises(GatewayError, match="completion limit"):
        policy.validate({**body, "max_tokens": 8193})
    with pytest.raises(GatewayError, match="prompt plus completion budget"):
        policy.validate({**body, "prompt": body["prompt"] + [1]})
    with pytest.raises(GatewayError, match="completion limit"):
        policy.validate({**body, "prompt": [1] * context.compaction_trigger_tokens})
    with pytest.raises(GatewayError, match="logit_bias"):
        policy.validate({**body, "logit_bias": {1: -100}})
    with pytest.raises(GatewayError, match="over-trigger"):
        replace(policy, context=resolve_context_contract(_inference())).validate(body)


@pytest.mark.parametrize(
    "name",
    ["grpo_6gpu_arc2_overfit.json", "ppo_5gpu_arc3_officials_platypus.json"],
)
def test_launcher_sizes_training_and_eval_engines_from_served_ceiling(
    name, monkeypatch, capsys
):
    import sys

    path = CONFIGS / name
    config = ExperimentConfig.load(path)
    monkeypatch.setattr(sys, "argv", ["launcher", str(path)])
    monkeypatch.delenv("ARC3_SERVER_GPUS", raising=False)
    monkeypatch.delenv("ARC3_SERVER_TENSOR_PARALLEL", raising=False)
    script = (CONFIGS.parent / "scripts/platypus_debug_smoke.sbatch").read_text()
    sizing = (
        script.split('SERVER_SHAPE="$(', 1)[1]
        .split("<<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    exec(compile(sizing, "server-shape", "exec"), {})
    ceilings = list(map(int, capsys.readouterr().out.splitlines()[5].split()))
    training = resolve_context_contract(config.inference).max_served_sequence_tokens
    assert ceilings[: len(config.inference.server_gpus)] == [training] * len(
        config.inference.server_gpus
    )
    expected_eval = max(
        config.official_eval.context_window_tokens or training,
        config.inference.agent_max_served_sequence_tokens or 0,
    )
    assert ceilings[len(config.inference.server_gpus) :] == [expected_eval] * len(
        config.official_eval.server_gpus
    )


@pytest.mark.parametrize("ids", [[-1], [True], [1.0], ["2"], None, "2"])
def test_config_rejects_non_integer_summary_token_ids(ids):
    payload = json.loads((CONFIGS / "grpo_6gpu_arc2_overfit.json").read_text())[
        "inference"
    ]
    payload["agent_summary_forbidden_token_ids"] = ids
    with pytest.raises(ValueError, match="nonnegative integers"):
        InferenceConfig.from_dict(payload)


def test_summary_token_ids_require_a_summary_cap():
    with pytest.raises(ValueError, match="positive summary cap"):
        resolve_context_contract(_inference(agent_summary_forbidden_token_ids=[2]))
    for ids in ([-1], [True], [1.5]):
        with pytest.raises(ValueError, match="nonnegative integers"):
            resolve_context_contract(_inference(agent_summary_forbidden_token_ids=ids))
    context = resolve_context_contract(
        _inference(
            agent_summary_forbidden_token_ids=[2, 4],
            agent_max_summary_tokens=4096,
            agent_max_served_sequence_tokens=49152,
        )
    )
    assert context.summary_forbidden_token_ids == (2, 4)


@pytest.mark.parametrize("window", [24576, 40960, 65535, 65536])
def test_arc2_eval_context_is_shared_by_worker_spec_and_launcher(
    window, tmp_path, monkeypatch, capsys
):
    import sys
    from training.config import evaluation_inference_config

    payload = json.loads((CONFIGS / "grpo_6gpu_arc2_overfit.json").read_text())
    payload["official_eval"]["context_window_tokens"] = window
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = ExperimentConfig.load(path)
    evaluation = _collector_config(config, base_url="http://unused.invalid")
    context = resolve_context_contract(evaluation.inference)
    assert context.max_summary_completion_tokens == 2 * (window // 16)
    assert context.reset_prompt_ceiling_tokens == window - 4096
    # The larger of the 12288 headroom rule and the Duck's summary bound.
    assert context.max_served_sequence_tokens == max(
        window + 2 * (window // 16) + 12288, window + 4096 + 6 * (window // 16)
    )
    assert context == resolve_context_contract(evaluation_inference_config(config))
    assert resolve_context_contract(config.inference).context_window_tokens == 32768
    monkeypatch.setattr(sys, "argv", ["launcher", str(path)])
    monkeypatch.delenv("ARC3_SERVER_GPUS", raising=False)
    monkeypatch.delenv("ARC3_SERVER_TENSOR_PARALLEL", raising=False)
    script = (CONFIGS.parent / "scripts/platypus_debug_smoke.sbatch").read_text()
    sizing = script.split('SERVER_SHAPE="$(')[1].split("<<'PY'\n")[1].split("\nPY\n")[0]
    exec(compile(sizing, "server-shape", "exec"), {})
    ceilings = list(map(int, capsys.readouterr().out.splitlines()[5].split()))
    assert ceilings[len(config.inference.server_gpus) :] == [
        context.max_served_sequence_tokens
    ] * len(config.official_eval.server_gpus)


@pytest.mark.parametrize("summary", [1536, 4096, 1535, None])
def test_arc2_load_requires_the_worker_summary_cap(tmp_path, summary):
    """A shorter ARC-2 window takes its summary cap with it. The worker refuses
    any cap but twice window/16 when the first window starts, so loading does."""
    payload = _flash_recipe_at_32k()
    # The Flash recipe's lengths shrunk as training/FLASH_RL.md describes.
    window, reply = 12288, 4096
    served = window + 2 * (window // 16) + 12288
    payload["inference"].update(
        agent_context_window=window,
        agent_compact_to_tokens=window - reply,
        agent_max_summary_tokens=summary,
        agent_max_served_sequence_tokens=served,
    )
    payload["training"]["max_sequence_tokens"] = served
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    if summary == 1536:
        context = resolve_context_contract(ExperimentConfig.load(path).inference)
        assert context.max_summary_completion_tokens == 1536
        assert context.compaction_trigger_tokens == context.reset_prompt_ceiling_tokens == 8192
        assert context.max_served_sequence_tokens == 26112
        return
    with pytest.raises(ValueError) as error:
        ExperimentConfig.load(path)
    assert str(error.value) == (
        "ARC-2 summaries are capped at twice the window/16 summary target: "
        "inference.agent_context_window=12288 requires "
        f"inference.agent_max_summary_tokens=1536, got {summary}."
    )


@pytest.mark.parametrize("served", [53248, 53247, 49152])
def test_arc2_load_requires_the_duck_summary_bound(tmp_path, served):
    """The ARC-2 Duck (arc2-platypus f380ffd) owns window + reply cap + 6 x its
    window/16 summary target; with an 8k reply cap that exceeds the 12288
    headroom rule (49152), so loading requires 53248."""
    payload = _flash_recipe_at_32k()
    window, reply = 32768, 8192
    payload["inference"].update(
        agent_context_window=window,
        agent_max_completion_tokens=reply,
        agent_compact_to_tokens=window - reply,
        agent_max_summary_tokens=2 * (window // 16),
        agent_max_served_sequence_tokens=served,
    )
    payload["training"]["max_sequence_tokens"] = served
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    if served >= 53248:
        context = resolve_context_contract(ExperimentConfig.load(path).inference)
        assert context.max_served_sequence_tokens == served
        return
    with pytest.raises(ValueError, match=r"agent_max_served_sequence_tokens>=53248, got "):
        ExperimentConfig.load(path)


@pytest.mark.parametrize("window,served", [(32768, None), (24576, 39936), (40960, 60416), (65536, 94208)])
def test_arc2_evaluation_ceiling_meets_the_duck(tmp_path, window, served):
    from types import SimpleNamespace

    from harness_support.arc2.trainer import evaluation_context_overrides

    path = tmp_path / "config.json"
    path.write_text(json.dumps(_flash_recipe_at_32k()))
    inference = ExperimentConfig.load(path).inference
    overrides = evaluation_context_overrides(inference, SimpleNamespace(context_window_tokens=window))
    assert overrides.get("agent_max_served_sequence_tokens") == served


@pytest.mark.parametrize("served", [26112, 49152, 26111, 16384, None])
def test_arc2_load_requires_the_served_overflow_headroom(tmp_path, served):
    """A summary request carries the whole live history, which can run past the
    window. The gateway refuses a request past the served ceiling mid-run, so
    loading requires window + summary cap + 12288 (training/FLASH_RL.md)."""
    payload = _flash_recipe_at_32k()
    window, reply = 12288, 4096
    payload["inference"].update(
        agent_context_window=window,
        agent_compact_to_tokens=window - reply,
        agent_max_summary_tokens=2 * (window // 16),
        agent_max_served_sequence_tokens=served,
    )
    # The learner ceiling matches, so only the served ceiling can fail.
    payload["training"]["max_sequence_tokens"] = 26112 if served is None else served
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    if served is not None and served >= 26112:
        context = resolve_context_contract(ExperimentConfig.load(path).inference)
        assert context.max_served_sequence_tokens == served
        return
    with pytest.raises(ValueError) as error:
        ExperimentConfig.load(path)
    assert str(error.value) == (
        "ARC-2 serves 12288 tokens above window plus summary cap, "
        "and at least the Duck's window + reply cap + 6 x summary target: "
        "inference.agent_context_window=12288 requires "
        f"inference.agent_max_served_sequence_tokens>=26112, got {served}."
    )
