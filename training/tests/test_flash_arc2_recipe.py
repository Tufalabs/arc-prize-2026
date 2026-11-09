"""The shipped Flash ARC-2 recipe (configs/flash_8gpu_arc2_grpo_tokmean.json):
the 89 puzzles e307 did not complete on all ten TAAF passes, the deployment
Duck's context, a 40-minute game limit and drafting on; the training algorithm
as before (training/FLASH_RL.md, "The recipe and smoke configs")."""

import ast
import json

import pytest

from flash_configs import RECIPE, ROOT
from training.config import ExperimentConfig, resolve_context_contract
from training.context_contract import ContextContract

DUCK = ROOT / "arc2-platypus/packages/duck/src/duck"
DEPLOYMENT = ROOT / "platypus/packages/ullm/src/ullm/backends/deployment.py"
HAS_ARC2_DATA = (ROOT / "arc2-platypus/packages/taaf/src/taaf/data/arc2-train.npz").exists()


def _literals(nodes):
    """name -> value of every literal assignment among NODES."""
    values = {}
    for node in nodes:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        for target in targets:
            name = ast.unparse(target)
            try:
                values[name] = ast.literal_eval(node.value)
            except ValueError:
                values[name] = ast.unparse(node.value)
    return values


def _definition(path, kind, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, kind) and node.name == name)


def test_recipe_context_is_the_deployment_ducks():
    """What DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    plays with at the arc2-platypus pin, derived as the Duck derives it."""
    if not (DUCK / "agent.py").is_file() or not DEPLOYMENT.is_file():
        pytest.skip("the harness submodules are not checked out")
    duck = _literals(_definition(DUCK / "agent.py", ast.ClassDef, "DuckAgent").body)
    window, reply = duck["context_window_tokens"], duck["reply_cap_tokens"]
    margin, ratio = duck["safety_margin_tokens"], duck["compaction_ratio"]
    # presets.configure: every ARC-2 column summarises under the engine's own window
    # and leaves the window, reply cap, margin and summary ratio at the class defaults.
    configure = _literals(ast.walk(_definition(DUCK / "presets.py", ast.FunctionDef, "configure")))
    assert configure["agent.compaction_strategy"] == "summarize"
    assert configure["agent.engine_ceiling_tokens"] == "engine_args.max_model_len"
    for field in ("context_window_tokens", "reply_cap_tokens", "safety_margin_tokens", "compaction_ratio"):
        assert f"agent.{field}" not in configure
    engine_window = _literals(
        ast.walk(_definition(DEPLOYMENT, ast.FunctionDef, "_flashnext_config")))["max_model_len"]
    target = window // ratio  # DuckAgent._summary_target_tokens
    inference = json.loads(RECIPE.read_text())["inference"]
    assert (window, reply, margin, engine_window) == (73728, 24576, 512, 262144)
    assert inference["agent_context_window"] == window
    assert inference["agent_max_completion_tokens"] == reply
    assert inference["agent_safety_margin_tokens"] == margin
    # Under summarize the reset ceiling is the prompt budget (_observer_compaction_target).
    assert inference["agent_compact_to_tokens"] == window - reply - margin == 48640
    # A summary is capped at twice its target (_summarize_live_history: cap=2 * target).
    assert inference["agent_max_summary_tokens"] == 2 * target == 9216
    assert inference["agent_max_served_sequence_tokens"] == engine_window
    # The engine window holds the Duck's own bound (_derived_engine_window) and ours.
    assert engine_window >= window + reply + 6 * target == 125952
    assert engine_window >= window + 2 * target + 12288 == 95232


def test_recipe_context_contract():
    assert resolve_context_contract(ExperimentConfig.load(RECIPE).inference) == ContextContract(
        context_window_tokens=73728,
        max_completion_tokens=24576,
        max_summary_completion_tokens=9216,
        safety_margin_tokens=512,
        compaction_trigger_tokens=48640,
        reset_prompt_ceiling_tokens=48640,
        max_served_sequence_tokens=262144,
        max_compaction_proof_tokens=262144,
        summary_forbidden_token_ids=(248058,),
    )


def test_recipe_plays_the_89_puzzles_from_two_sets():
    cfg = ExperimentConfig.load(RECIPE)
    assert cfg.inference.dataset == ["arc2-eval", "arc2-train-hard"]
    games = cfg.inference.games
    assert len(games) == len(set(games)) == 89
    if not HAS_ARC2_DATA:
        pytest.skip("Requires the private ARC-2 submodule")
    from harness_support.arc2.dataset import dataset_manifest
    from harness_support.arc2.trainer import resolve_datasets

    held = {name: set(dataset_manifest(name)["puzzle_ids"]) for name in cfg.inference.dataset}
    assert [sum(game in held[name] for game in games) for name in held] == [52, 37]
    assert resolve_datasets(cfg)["training"]["games"] == games


def test_recipe_window_and_learner_lengths():
    cfg = ExperimentConfig.load(RECIPE)
    assert cfg.inference.max_runtime_minutes == 40.0
    assert cfg.inference.agent_max_episode_tokens is None
    # A segment reaches the window plus a reply; a micro-batch holds one such segment,
    # and the learner accepts anything the servers may serve.
    window = cfg.inference.agent_context_window + cfg.inference.agent_max_completion_tokens
    assert cfg.training.max_micro_batch_tokens == window == 98304
    assert cfg.training.max_sequence_tokens == cfg.inference.agent_max_served_sequence_tokens
    # Without activation offload a 98,304-token micro-batch runs out of device memory;
    # every fourth layer leaves 14.5 GiB (training/FLASH_RL.md, "Learner memory").
    assert cfg.training.flash.offload_every == 4
    assert cfg.inference.speculative_decoding.enabled is True
    augmentation = cfg.inference.augmentation
    assert (augmentation.enabled, augmentation.dihedral, augmentation.colors) == (True, True, True)


def test_recipe_nudges_past_an_empty_reply_and_charges_each_one():
    """An empty reply (no text, no tool call) no longer ends the episode: the
    Duck nudges as the deployment harness does and the game goes on, and every
    empty reply is charged as a malformed call (2026-10-10)."""
    cfg = ExperimentConfig.load(RECIPE)
    assert cfg.inference.agent_terminate_on_empty_reply is False
    assert cfg.training.malformed_call_penalty == 0.05
    # An episode's penalties take at most 0.5 (2026-10-10): a solved
    # puzzle with many penalties still beats an unsolved one.
    assert json.loads(RECIPE.read_text())["training"]["max_episode_penalty"] == 0.5
    assert cfg.training.max_episode_penalty == 0.5
    assert (cfg.inference.agent_malformed_retries, cfg.inference.agent_accept_malformed_output) == (
        0, True)


def test_recipe_keeps_the_training_algorithm():
    cfg = ExperimentConfig.load(RECIPE)
    training, inference = cfg.training, cfg.inference
    assert (training.advantage_estimator, training.loss_aggregation) == ("grpo", "token_mean")
    assert (training.learning_rate, training.lora.r, training.lora.alpha, training.lora.dropout) == (
        5e-06, 16, 32, 0)
    assert len(training.lora.target_modules) == 12
    assert (inference.agent_temperature, inference.agent_top_p, inference.agent_top_k) == (1.0, 1.0, 0)
    assert (cfg.parallel_rollouts, cfg.sampling.game_sampling_games_per_window) == (128, 16)
    assert not cfg.official_eval.enabled
    assert training.flash.train_routers is True
    assert training.flash.policy_weights_rounding == "dither"


@pytest.mark.parametrize("recipe_name", [
    "flash_8gpu_arc2_grpo_tokmean.json", "flash_8gpu_arc2_grpo_epmean.json",
])
def test_recipe_keeps_every_adapter_and_trainer_state(recipe_name):
    recipe = RECIPE.parent / recipe_name
    assert json.loads(recipe.read_text())["training"]["checkpoint_keep_last"] is None
    assert ExperimentConfig.load(recipe).training.checkpoint_keep_last is None


@pytest.mark.parametrize("recipe_name", [
    "flash_8gpu_arc2_grpo_tokmean.json", "flash_8gpu_arc2_grpo_epmean.json",
])
def test_recipe_sample_budget_starts_at_full_coverage(recipe_name):
    """The 0.3 start was a throughput guess that only dropped data during the
    ramp (2026-10-12): the budget starts at 1.0
    and the controller lowers it only if a trainer iteration exceeds 0.95 of
    a rollout iteration."""
    recipe = RECIPE.parent / recipe_name
    assert json.loads(recipe.read_text())["training"]["sample_budget"]["initial_fraction"] == 1.0
    budget = ExperimentConfig.load(recipe).training.sample_budget
    assert (budget.mode, budget.initial_fraction, budget.max_fraction) == ("auto", 1.0, 1.0)
    assert budget.target_train_to_collection_ratio == 0.95


def test_checkpoint_retention_stays_out_of_checkpoint_identity(monkeypatch):
    from harness_support.arc2 import trainer as support
    from training.multimodal import build_checkpoint_identity

    monkeypatch.setattr(
        "training.multimodal.resolve_processing_identity",
        lambda *_args, **_kwargs: {"model": "flash"},
    )
    cfg = ExperimentConfig.load(RECIPE)
    manifest = {"name": "arc2-eval", "sha256": "dataset", "puzzle_ids": ["p"]}
    expected = build_checkpoint_identity(cfg, support=support, manifest=manifest)
    assert expected is not None
    assert "checkpoint_keep_last" not in expected
    for keep in (6, 3, None):
        cfg.training.checkpoint_keep_last = keep
        assert build_checkpoint_identity(cfg, support=support, manifest=manifest) == expected


def test_epmean_recipe_differs_only_in_loss_aggregation():
    """configs/flash_8gpu_arc2_grpo_epmean.json is the token-mean recipe with
    episode-mean loss weighting (the 27B original run's choice,
    2026-10-11); every other value stays in step with the shipped recipe."""
    tokmean = json.loads(RECIPE.read_text())
    epmean = json.loads((RECIPE.parent / "flash_8gpu_arc2_grpo_epmean.json").read_text())
    assert tokmean["training"].pop("loss_aggregation") == "token_mean"
    assert epmean["training"].pop("loss_aggregation") == "episode_mean"
    assert epmean == tokmean
