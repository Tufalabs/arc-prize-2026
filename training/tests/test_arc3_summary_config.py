"""ARC-3 summary budgets agree across recipes, evaluation and the worker."""

import json
from pathlib import Path

import pytest

from training.config import (
    ExperimentConfig,
    evaluation_inference_config,
    resolve_context_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def summary_config(tmp_path, **changes):
    payload = json.loads(
        (ROOT / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json").read_text()
    )
    payload["inference"].update(
        dict(
            agent_context_window=32768,
            agent_max_completion_tokens=4096,
            agent_compact_to_tokens=28672,
            agent_max_summary_tokens=4096,
            agent_max_served_sequence_tokens=49152,
            agent_summary_forbidden_token_ids=[248058],
        )
    )
    payload["inference"].update(changes)
    payload["training"]["max_sequence_tokens"] = 49152
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("window", [24576, 40960, 65535, 65536])
def test_evaluation_resizes_summary_and_serving_limits(tmp_path, window):
    config = ExperimentConfig.load(summary_config(tmp_path))
    config.official_eval.context_window_tokens = window
    context = resolve_context_contract(evaluation_inference_config(config))
    assert context.max_summary_completion_tokens == 2 * (window // 16)
    assert context.reset_prompt_ceiling_tokens == window - 4096
    # The larger of the configured room above window + summary cap (12288)
    # and above window + reply reserve (49152 - 32768 - 4096 = 12288).
    assert context.max_served_sequence_tokens == max(
        window + 2 * (window // 16) + 12288, window + 4096 + 12288
    )
    assert resolve_context_contract(config.inference).context_window_tokens == 32768


def test_a_smaller_evaluation_window_keeps_the_arc3_serving_minimum(tmp_path):
    """Review finding: a 64k recipe served at the ARC-3 minimum (77824)
    evaluated at 32k must keep window + reply + 8192 = 45056, not ARC-2's
    40960, or validation refuses the recipe before evaluation starts."""
    payload = json.loads(
        summary_config(
            tmp_path,
            agent_context_window=65536,
            agent_compact_to_tokens=61440,
            agent_max_summary_tokens=8192,
            agent_max_served_sequence_tokens=77824,
        ).read_text()
    )
    payload["training"]["max_sequence_tokens"] = 77824
    payload["official_eval"]["context_window_tokens"] = 32768
    path = tmp_path / "evaluated.json"
    path.write_text(json.dumps(payload))
    config = ExperimentConfig.load(path)  # validation accepts the evaluation override
    context = resolve_context_contract(evaluation_inference_config(config))
    assert context.max_served_sequence_tokens == 32768 + 4096 + 8192
    assert context.max_summary_completion_tokens == 4096
    assert context.reset_prompt_ceiling_tokens == 32768 - 4096


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"agent_max_summary_tokens": 2048}, "twice the window/16"),
        ({"agent_compact_to_tokens": 16384}, "reset to the ordinary"),
        ({"agent_summary_forbidden_token_ids": []}, "tool-call token ban"),
        ({"agent_max_served_sequence_tokens": 40000}, "8192 summary tokens"),
    ],
)
def test_invalid_summary_contract_fails_before_worker_start(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.load(summary_config(tmp_path, **changes))


def test_pruning_evaluation_keeps_legacy_limits():
    config = ExperimentConfig.load(
        ROOT / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json"
    )
    config.official_eval.context_window_tokens = 16384
    context = resolve_context_contract(evaluation_inference_config(config))
    assert context.max_summary_completion_tokens == 0
    assert context.reset_prompt_ceiling_tokens == 4096
    assert context.max_served_sequence_tokens == 16384
