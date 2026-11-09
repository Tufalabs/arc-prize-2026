"""The actual ARC-2 worker uses upstream deployment and context injection hooks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


# A Flash checkpoint is a local directory without a revision: engine metadata
# owns it, and an empty training.model_revision must not be compared with it.
@pytest.mark.parametrize(("window", "model_revision"), [
    *((window, "pinned-revision") for window in (24576, 32768, 40960, 65535, 65536)),
    (32768, ""),
])
# Each engine serves the prompt injection its own Duck preset resolves.
@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.8-flash-next"])
def test_worker_builds_a_direct_engine_without_resolving_local_weights(
    tmp_path, window, model_revision, engine
):
    pytest.importorskip("vllm")
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from pathlib import Path
from ullm.backends import deployment
from ullm.backends.engine_proxy import EngineProxy, EngineProxyFactory
import harness_support.arc2.worker as worker
from harness_support.arc2.worker import build_benchmark
from duck import presets
from duck.agent import DuckAgent
from taaf.datasets import load
from training.config import ExperimentConfig, evaluation_inference_config, resolve_context_contract

def forbidden(*args, **kwargs):
    raise AssertionError('a supplied engine must not construct a local deployment')
deployment.build = forbidden
# No server answers here: the checkpoint's own processor ceiling.
worker.served_image_max_pixels = lambda spec: 16_777_216
puzzle = load('arc2-eval')[0].game_id
config = ExperimentConfig.load(Path(sys.argv[3]) / 'configs/grpo_6gpu_arc2_overfit.json')
config.official_eval.context_window_tokens = int(sys.argv[2])
context = resolve_context_contract(evaluation_inference_config(config))
spec = dict(
    dataset='arc2-eval', games=[puzzle]*8, slot_seeds=list(range(8)),
    max_concurrent=8, max_runtime_s_per_game=900, rng_seed=7,
    base_url='http://127.0.0.1:1', served_model='policy',
    tokenizer_id='Qwen/Qwen3.8-27B', model_revision=sys.argv[4],
    max_completion_tokens=context.max_completion_tokens, context_window_tokens=context.context_window_tokens,
    max_summary_tokens=context.max_summary_completion_tokens, max_served_sequence_tokens=context.max_served_sequence_tokens,
    summary_forbidden_token_ids=[248058],
    compact_to_tokens=context.reset_prompt_ceiling_tokens, safety_margin_tokens=context.safety_margin_tokens,
    max_episode_tokens=None, reasoning_effort='medium',
    compaction_trigger_tokens=context.compaction_trigger_tokens, terminate_on_empty_reply=True,
    sampling={'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}, job_dir=sys.argv[1],
    engine=sys.argv[5],
)
benchmark = build_benchmark(spec)
agent = benchmark.solver.agent
factory = agent.llm_config.backend_factory
assert isinstance(factory, EngineProxyFactory)
assert isinstance(factory.engine, EngineProxy)
assert factory.engine._http is None
assert factory.engine.tokenizer_revision == (sys.argv[4] or None)
# The engine's own parsers and template options (ullm), plus the training flags.
assert factory.backend_kwargs == {
    **deployment.backend_kwargs(sys.argv[5]),
    'classify_tool_calls': True,
    'accept_malformed_output': True,
    'require_explicit_sampling': True,
    'verify_engine_output': True,
    'token_count_evidence': True,
    'replay_on_link_loss': False,
}
assert agent.preserve_failed_draws and agent.classify_tool_calls
assert agent.malformed_retries == 0 and not agent.perturb_retry_prompt
assert getattr(agent, 'malformed_nudges', 0) == 0
assert agent.llm_config.exact_history
assert type(agent).__name__ == "DuckAgent"
assert tuple(agent.compaction_forbidden_token_ids) == (248058,)
assert agent.images == 'task+show' and agent.compaction_strategy == 'summarize'
assert agent.run_config.max_runtime_s_per_game == 900
assert agent.run_config.min_request_gap_s == 0
assert agent.max_tokens_per_game is None
assert agent.terminate_on_empty_reply is True
assert agent._reply_reserve_tokens() == context.max_completion_tokens
assert agent.engine_ceiling_tokens == context.max_served_sequence_tokens
# The prompt injection is the engine preset's, all three fields, and every
# run of the window reports it.
injection = ('inject_text_on_first_prompt', 'inject_text_on_later_prompts', 'prompt_injection_schedule')
preset = DuckAgent()
presets.configure(preset, engine=sys.argv[5], environment='slurm-b200', backend_factory=factory)
expected = {name: getattr(preset, name) for name in injection}
assert {name: getattr(agent, name) for name in injection} == expected, expected
import json
from harness_support.agent_settings import AGENT_SETTINGS_FILE
assert json.loads((Path(sys.argv[1]) / AGENT_SETTINGS_FILE).read_text()) == expected
assert len(agent.run_config.rng_seed_per_game) == 8
assert len(set(agent.run_config.rng_seed_per_game.values())) == 8
assert benchmark.raise_solver_errors
print('direct engine, retained draws, budgets, seeds and context injection verified')
"""
    source_paths = [
        root / "arc2-platypus/packages" / name / "src"
        for name in ("duck", "platypus", "taaf")
    ] + [root / "platypus/packages/ullm/src", root / "scripts", root]
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(window), str(root), model_revision, engine],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, source_paths))},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Every ARC-2 recipe, at its training window and at evaluation windows the
# trainer resizes the limits for (harness_support.arc2.trainer).
ARC2_RECIPES = sorted(
    path.name for path in (Path(__file__).resolve().parents[1] / "configs").glob("*arc2*.json")
)


@pytest.mark.parametrize("recipe", ARC2_RECIPES)
def test_trainer_limits_are_what_the_built_duck_requires(recipe):
    """The trainer re-derives the Duck's limits without importing it
    (docs/arc2_training.md, "Where the trainer can drift from the Duck"); here
    the Duck built with each recipe's window, reply cap and margin says what
    it needs, through its own (private) methods: the served ceiling meets its
    bound, the summary cap is what its compaction requests, the trainer's
    copy of the bound is window + reply reserve + its summary cap with
    reasoning, and the compaction trigger and reset target are its own."""
    pytest.importorskip("vllm")
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from pathlib import Path
from duck.agent import DuckAgent
from ullm.backends.engine_proxy import EngineProxy
from harness_support.arc2.trainer import duck_summary_ceiling
from harness_support.arc2.worker import duck_summary_request_caps
from training.config import ExperimentConfig, evaluation_inference_config, resolve_context_contract
from training.harness import harness_engine
from training_backend import (
    configure_training_agent, qwen_engine_factory, training_backend_factory, training_sampling,
)

config = ExperimentConfig.load(Path(sys.argv[1]) / 'configs' / sys.argv[2])
engine = harness_engine(config.training)
inferences = [config.inference]
for window in (24576, 40960, 65536, 98304):
    if window <= 2 * config.inference.agent_max_completion_tokens:
        continue  # the trainer refuses a window this close to the reply cap
    config.official_eval.context_window_tokens = window
    inferences.append(evaluation_inference_config(config))
for inference in inferences:
    context = resolve_context_contract(inference)
    factory = qwen_engine_factory(
        EngineProxy(base_url='http://127.0.0.1:1', model='policy'), engine=engine
    )
    agent = DuckAgent.build(
        engine=engine, environment='slurm-b200', backend_factory=factory,
        derive_max_model_len=False, derive_max_num_seqs=False, derive_max_pixels=False,
        reply_cap_tokens=context.max_completion_tokens,
        context_window_tokens=context.context_window_tokens,
        engine_ceiling_tokens=context.max_served_sequence_tokens,
        safety_margin_tokens=context.safety_margin_tokens,
        compact_to_tokens=context.reset_prompt_ceiling_tokens,
        summary_thinking=False, malformed_retries=0, perturb_retry_prompt=False,
        preserve_failed_draws=True, classify_tool_calls=True, malformed_nudges=0,
    )
    configure_training_agent(
        agent,
        backend_factory=training_backend_factory(source_factory=factory),
        sampling=training_sampling({'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}),
    )
    agent.prepare_for_backend()
    label = (sys.argv[2], context.context_window_tokens)
    # The Duck refuses (ValueError) a ceiling below its own bound.
    assert agent._derived_engine_window() == context.max_served_sequence_tokens, label
    caps = duck_summary_request_caps(agent)
    assert caps['cap'] == context.max_summary_completion_tokens, (label, caps)
    assert duck_summary_ceiling(
        context.context_window_tokens, inference.agent_max_completion_tokens
    ) == agent.context_window_tokens + agent._reply_reserve_tokens() + caps['reasoning_cap'], (
        label, caps)
    assert agent.context_budget_tokens() == context.compaction_trigger_tokens, label
    assert agent.compact_target_tokens() == context.reset_prompt_ceiling_tokens, label
print(sys.argv[2], 'matches the Duck at', len(inferences), 'windows')
"""
    source_paths = [
        root / "arc2-platypus/packages" / name / "src"
        for name in ("duck", "platypus", "taaf")
    ] + [root / "platypus/packages/ullm/src", root / "scripts", root]
    result = subprocess.run(
        [sys.executable, "-c", program, str(root), recipe],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, source_paths))},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_refuses_limits_the_duck_does_not_have(tmp_path):
    """At window start the worker refuses a served ceiling under the Duck's
    own bound and a summary cap other than the one the Duck requests."""
    pytest.importorskip("vllm")
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from pathlib import Path
import harness_support.arc2.worker as worker
from harness_support.arc2.worker import build_benchmark
from taaf.datasets import load
from training.config import ExperimentConfig, resolve_context_contract

config = ExperimentConfig.load(Path(sys.argv[2]) / 'configs/grpo_6gpu_arc2_overfit.json')
context = resolve_context_contract(config.inference)
spec = dict(
    dataset='arc2-eval', games=[load('arc2-eval')[0].game_id], slot_seeds=[0],
    max_concurrent=1, max_runtime_s_per_game=900, rng_seed=7,
    base_url='http://127.0.0.1:1', served_model='policy',
    tokenizer_id='Qwen/Qwen3.8-27B', model_revision='',
    max_completion_tokens=context.max_completion_tokens,
    context_window_tokens=context.context_window_tokens,
    max_summary_tokens=context.max_summary_completion_tokens,
    max_served_sequence_tokens=context.max_served_sequence_tokens,
    summary_forbidden_token_ids=[248058],
    compact_to_tokens=context.reset_prompt_ceiling_tokens,
    safety_margin_tokens=context.safety_margin_tokens, max_episode_tokens=None,
    reasoning_effort='xhigh', compaction_trigger_tokens=context.compaction_trigger_tokens,
    sampling={'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}, job_dir=sys.argv[1],
    engine='qwen3.8-27b',
)
worker.served_image_max_pixels = lambda spec: 16_777_216
build_benchmark(dict(spec))
# The recipe sits exactly on the Duck's bound: one token less is refused by the Duck.
try:
    build_benchmark({**spec, 'max_served_sequence_tokens': spec['max_served_sequence_tokens'] - 1})
except ValueError as error:
    assert 'ceiling' in str(error), error
else:
    raise AssertionError('a ceiling under the Duck bound was accepted')
try:
    build_benchmark({**spec, 'max_summary_tokens': spec['max_summary_tokens'] + 1})
except ValueError as error:
    assert 'summary cap differs' in str(error), error
else:
    raise AssertionError('a summary cap the Duck does not request was accepted')
print('refused')
"""
    source_paths = [
        root / "arc2-platypus/packages" / name / "src"
        for name in ("duck", "platypus", "taaf")
    ] + [root / "platypus/packages/ullm/src", root / "scripts", root]
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(root)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, source_paths))},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
