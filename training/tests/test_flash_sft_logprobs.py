"""Native baseline CE is preserved; recorded KL and both validation losses are opt-in."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sft.logprob.targets import FIELD, prepare_row
from tests.logprob import conftest as teacher_fixtures
from tests.logprob.test_targets_and_loss import dense_reference, row_and_steps
from training.flash_sft import (
    FlashSFT,
    RowIndex,
    _Log,
    completion_spans,
    load_config,
    summarize,
    wandb_metrics,
)


@pytest.fixture
def vocabulary(tmp_path):
    return teacher_fixtures.vocab.__wrapped__(tmp_path)


def setup_row(vocabulary, packed):
    row, steps = row_and_steps()
    row.update(game_id="heldout", source_set="new")
    if packed:
        row.update(
            input_token_ids=[0, 1, 7, 2, 3, 4, 0, 1],
            steps=[9, 9],
            packed_completions=[[2, 4]],
        )
    return prepare_row(row, steps, vocabulary)


def trainer(row, vocabulary, chunk=2):
    result = object.__new__(FlashSFT)
    result.config = {"head_chunk": chunk}
    result.device = torch.device("cpu")
    result.first_nll = None
    result.recorded_logprobs = {
        "weight": 0.3,
        "tokenizer_sha256": vocabulary.sha256,
        "require_logprobs": True,
    }
    torch.manual_seed(123)
    ids = torch.tensor(row["input_token_ids"] + row["output_token_ids"])
    hidden = torch.randn(1, len(ids), 5, requires_grad=True)
    result.head = SimpleNamespace(weight=torch.randn(8, 5))
    result._hidden = lambda unused: (hidden, ids)
    return result, hidden, ids


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_native_combined_loss_and_gradients_match_independent_reference(
    vocabulary, packed, chunk
):
    row = setup_row(vocabulary, packed)
    run, hidden, ids = trainer(row, vocabulary, chunk)
    actual = run._row_loss(row)
    h = hidden.detach().clone().requires_grad_()
    head = run.head.weight
    ce = sum(
        F.cross_entropy(
            h[0, p - 1 : p + n - 1] @ head.T, ids[p : p + n], reduction="sum"
        )
        for p, n in completion_spans(row)
    )
    events = row[FIELD]["events"]
    kl = dense_reference(
        h[0, [e["position"] - 1 for e in events]],
        head,
        torch.tensor([e["ids"] for e in events]),
        torch.tensor([e["probabilities"] for e in events]),
    )
    expected = ce + 0.3 * kl
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, h.grad, atol=2e-6, rtol=2e-5)
    assert run._loss_components[2] == len(events)


@pytest.mark.parametrize("packed", [False, True])
def test_exclusive_ce_trains_only_the_tokens_without_a_recorded_target(
    vocabulary, packed
):
    row = setup_row(vocabulary, packed)
    run, hidden, ids = trainer(row, vocabulary, chunk=2)
    run.recorded_logprobs["ce_positions"] = "unaligned"
    actual = run._row_loss(row)
    h = hidden.detach().clone().requires_grad_()
    head = run.head.weight
    events = row[FIELD]["events"]
    aligned = {e["position"] for e in events}
    # Every supervised token is either CE-trained or KL-trained, never both.
    supervised = [t for p, n in completion_spans(row) for t in range(p, p + n)]
    unaligned = [t for t in supervised if t not in aligned]
    assert aligned <= set(supervised) and len(unaligned) + len(aligned) == len(supervised)
    assert unaligned, "the fixture must leave CE-only tokens"
    ce = F.cross_entropy(h[0, [t - 1 for t in unaligned]] @ head.T, ids[unaligned], reduction="sum")
    kl = dense_reference(
        h[0, [e["position"] - 1 for e in events]],
        head,
        torch.tensor([e["ids"] for e in events]),
        torch.tensor([e["probabilities"] for e in events]),
    )
    expected = ce + 0.3 * kl
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, h.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(run._loss_components[0], ce.detach())
    assert run._loss_components[2] == len(events)


def test_exclusive_ce_matches_the_mixed_objective_without_recorded_targets(vocabulary):
    row, steps = row_and_steps()
    steps[9]["extra"] = {}
    row = prepare_row(row, steps, vocabulary, allow_missing_logprobs=True)
    run, hidden, ids = trainer(row, vocabulary)
    run.recorded_logprobs.update(require_logprobs=False, ce_positions="unaligned")
    actual = run._row_loss(row)
    run.recorded_logprobs["ce_positions"] = "all"
    torch.testing.assert_close(actual, run._row_loss(row))


def test_absent_option_keeps_native_ce_loss_and_validation_shape(vocabulary):
    row = setup_row(vocabulary, True)
    run, hidden, ids = trainer(row, vocabulary)
    run.recorded_logprobs = None
    ce = run._row_loss(row)
    expected = sum(
        F.cross_entropy(
            hidden[0, p - 1 : p + n - 1] @ run.head.weight.T,
            ids[p : p + n],
            reduction="sum",
        )
        for p, n in completion_spans(row)
    )
    torch.testing.assert_close(ce, expected)
    assert len(run._row_scores(row)) == 4


def test_missing_logprobs_keep_exact_ce_value_and_gradient_in_mixed_mode(vocabulary):
    row, steps = row_and_steps()
    steps[9]["extra"] = {}
    row = prepare_row(row, steps, vocabulary, allow_missing_logprobs=True)
    run, hidden, ids = trainer(row, vocabulary)
    run.recorded_logprobs["require_logprobs"] = False
    actual = run._row_loss(row)
    clone = hidden.detach().clone().requires_grad_()
    expected = sum(F.cross_entropy(clone[0, p - 1:p + n - 1] @ run.head.weight.T,
                                 ids[p:p + n], reduction="sum") for p, n in completion_spans(row))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, clone.grad)
    assert row[FIELD]["coverage"]["steps_without_logprobs"] == 1
    assert run._loss_components[1].item() == 0 and run._loss_components[2] == 0


def test_reasoning_enabled_summary_retains_recorded_kl_targets(vocabulary):
    row, steps = row_and_steps()
    steps[9]["messages"] = [{"role":"user","content":"Your context window is nearly full. Write a summary."}]
    without_thinking = prepare_row(row, steps, vocabulary)
    with_thinking = prepare_row(dict(row, summary_thinking=True), steps, vocabulary)
    assert not without_thinking[FIELD]["events"]
    assert len(with_thinking[FIELD]["events"]) == 3


def test_validation_ce_and_combined_share_tokens_groups_and_logging(
    vocabulary, tmp_path, monkeypatch
):
    row = setup_row(vocabulary, True)
    run, hidden, _ids = trainer(row, vocabulary)
    scored = run._row_scores(row)
    positions = [p - 1 + i for p, n in completion_spans(row) for i in range(n)]
    logp = (hidden.detach()[0, positions] @ run.head.weight.T).log_softmax(-1)
    assert scored[3] == pytest.approx(float(-(logp.exp() * logp).sum()), rel=1e-6)
    expected = run._row_loss(row).detach()
    tokens = sum(n for _, n in completion_spans(row))
    assert scored[1] == tokens and scored[5] == len(row[FIELD]["events"])
    assert -scored[0] + 0.3 * scored[4] == pytest.approx(float(expected), rel=1e-6)
    assert hidden.grad is None
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    run.validation = RowIndex(
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        max_length=100,
        fields=("source_set",),
        recorded_logprobs=run.recorded_logprobs,
        vocab_size=8,
    )
    run.rank, run.world = 0, 1
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: None)
    report = run._validate(0)
    assert report["ce_loss"] == report["nll"]
    assert report["ce_plus_kl_loss"] == pytest.approx(float(expected) / tokens)
    assert (
        report["by_source_set"]["new"]["ce_plus_kl_loss"] == report["ce_plus_kl_loss"]
    )
    logger = object.__new__(_Log)
    logger.rank = 0
    logger.event = lambda **unused: None
    logged = []
    logger.wandb = SimpleNamespace(log=lambda payload, step: logged.append(payload))
    logger.metrics(0, validation=report)
    assert logged[0]["train/sft_validation/ce_loss"] == report["ce_loss"]
    assert (
        logged[0]["train/sft_validation/ce_plus_kl_loss"] == report["ce_plus_kl_loss"]
    )
    assert "train/sft_validation/source_set/new/ce_plus_kl_loss" in logged[0]


@pytest.mark.parametrize("telemetry", [False, True])
def test_native_update_keeps_baseline_stats_and_logs_separate_loss_components(
    vocabulary, monkeypatch, tmp_path, telemetry
):
    row = setup_row(vocabulary, False)
    run, hidden, _ids = trainer(row, vocabulary)
    run.config.update(
        passes=1, optimizer={"lr": 1e-4, "warmup_updates": 5, "max_grad_norm": 1}
    )
    run.schedule, run.rank, run.world = [[0]], 0, 1
    run.sharded = set()
    run.run_dir = tmp_path
    monkeypatch.setenv("FLASH_SFT_PINNED_TELEMETRY", "1" if telemetry else "0")
    samples = []
    monkeypatch.setattr(
        "sft.distillation.host_memory.record_pinned",
        lambda directory, rank, step, phase: samples.append((rank, step, phase)),
    )
    run.rows = SimpleNamespace(tokens=[4], row=lambda index: row)
    run.trainables = {"hidden": hidden}
    run.optimizer = torch.optim.AdamW([hidden], lr=1e-4)
    run._reduce_gradients = lambda: None
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda device: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, op: None)
    monkeypatch.setattr(
        torch.distributed, "all_gather", lambda output, value: output[0].copy_(value)
    )
    metrics = run._train_update(0)
    assert metrics["loss_per_token"] == pytest.approx(
        metrics["ce_loss"] + 0.3 * metrics["teacher_kl"]
    )
    assert metrics["aligned_positions"] == 3 and metrics["kl_coverage"] == 0.75
    assert metrics["lr_factor"] == 0.2 and metrics["per_rank"]["sequence_length"] == [6]
    payload = wandb_metrics(1, train=metrics)
    assert payload["train/ce_loss"] == metrics["ce_loss"]
    assert payload["train/teacher_kl"] == metrics["teacher_kl"]
    assert payload["train/loss_per_completion_token"] == metrics["loss_per_token"]
    assert samples == (
        [(0, 1, "before_forward"), (0, 1, "after_update")] if telemetry else []
    )


def test_index_rejects_missing_logprobs_stale_tokens_and_tokenizer(
    vocabulary, tmp_path
):
    row = setup_row(vocabulary, False)
    path = tmp_path / "rows.jsonl"
    spec = {
        "weight": 0.3,
        "tokenizer_sha256": vocabulary.sha256,
        "require_logprobs": True,
    }
    for mode in ("missing", "tokens", "tokenizer"):
        changed = copy.deepcopy(row)
        if mode == "missing":
            changed[FIELD]["coverage"]["steps_without_logprobs"] = 1
        elif mode == "tokens":
            changed["input_token_ids"][0] = 6
        else:
            changed[FIELD]["tokenizer_sha256"] = "0" * 64
        path.write_text(json.dumps(changed) + "\n")
        with pytest.raises(ValueError):
            RowIndex(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                max_length=100,
                recorded_logprobs=spec,
                vocab_size=8,
            )


@pytest.mark.parametrize("game_id", ["crane_dipper-0001", "cd82-fb555c5d"])
def test_recorded_logprob_training_retains_main_holdout_guard(vocabulary, tmp_path, game_id):
    row = setup_row(vocabulary, False)
    row["game_id"] = game_id
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    options = {
        "max_length": 100,
        "recorded_logprobs": {"weight": 0.3, "tokenizer_sha256": vocabulary.sha256,
                              "require_logprobs": True},
        "vocab_size": 8,
    }
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="holdout cd82-fb555c5d"):
        RowIndex(path, checksum, for_training=True, **options)
    assert len(RowIndex(path, checksum, **options)) == 1


def test_summary_uses_global_token_denominator():
    report = summarize(
        [[-4.0, 2.0, 1.0, 0.5, 2.0, 1.0], [-18.0, 6.0, 4.0, 0.8, 3.0, 2.0]],
        {"game": ["a", "b"]},
        0.3,
    )
    assert report["ce_loss"] == 22 / 8
    assert report["teacher_kl"] == 5 / 8
    assert report["ce_plus_kl_loss"] == pytest.approx((22 + 0.3 * 5) / 8)
    assert report["kl_coverage"] == 3 / 8


def test_recipe_preserves_native_cosine_hardware_and_optimizer_settings():
    root = Path(__file__).resolve().parents[1] / "configs/flash_sft"
    baseline = json.loads((root / "reap307-kimi-cosine.json").read_text())
    arm = json.loads((root / "reap307-kimi-mixed-cosine.json").read_text())
    assert {k for k in set(arm) | set(baseline) if arm.get(k) != baseline.get(k)} <= {
        "rows", "validation", "recorded_logprobs", "wandb", "save_every", "optimizer",
    }
    assert arm["passes"] == 2
    assert arm["save_every"] == 20
    assert arm["recorded_logprobs"]["weight"] == 0.3
    assert arm["recorded_logprobs"]["require_logprobs"] is False
    assert {k: v for k, v in arm["optimizer"].items() if k != "decay"} == {
        k: v for k, v in baseline["optimizer"].items() if k != "decay"
    }
    manifest = json.loads((root / "reap307-kimi-mixed-sources.json").read_text())
    assert len(manifest["collections"]) == 19
    assert not any("wm-mandatory" in source["name"] for source in manifest["collections"])
    assert all(len(source["archive_sha256"]) == 64 for source in manifest["collections"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight", 0),
        ("weight", float("nan")),
        ("tokenizer_sha256", "bad"),
        ("require_logprobs", 1),
        ("ce_positions", "aligned"),
        ("ce_positions", None),
    ],
)
def test_invalid_recorded_configuration_is_rejected_before_model_import(
    tmp_path, field, value
):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/flash_sft/reap307-kimi-mixed-cosine.json").read_text()
    )
    config["recorded_logprobs"][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="recorded_logprobs"):
        load_config(path)


@pytest.mark.parametrize("ce_positions", ["all", "unaligned"])
def test_explicit_ce_positions_pass_the_recorded_key_validation(tmp_path, ce_positions):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/flash_sft/reap307-kimi-mixed-cosine.json").read_text()
    )
    # The digest is checked after ce_positions: reaching its error proves the key passed.
    config["recorded_logprobs"].update(ce_positions=ce_positions, tokenizer_sha256="bad")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        load_config(path)


def test_resume_identity_allows_only_memory_placement_changes():
    from training.flash_sft import check_resume_identity, config_sha256

    root = Path(__file__).resolve().parents[1]
    written = json.loads((root / "configs/flash_sft/reap307-kimi-mixed-cosine.json").read_text())
    state = {"config_sha256": config_sha256(written)}
    check_resume_identity(state, written)
    moved = dict(written, offload_every=2, offload_chunk_mib=None)
    with pytest.raises(ValueError, match="another configuration"):
        check_resume_identity(state, moved)
    check_resume_identity(state, moved, identity=written)
    with pytest.raises(ValueError, match="another configuration"):
        check_resume_identity(state, moved, identity=moved)
    retuned = dict(moved, optimizer=dict(written["optimizer"], lr=2e-4))
    with pytest.raises(ValueError, match="beyond memory placement: \\['optimizer'\\]"):
        check_resume_identity(state, retuned, identity=written)


def test_placement_free_ignores_only_the_offload_keys():
    from training.flash_sft import placement_free

    root = Path(__file__).resolve().parents[1]
    written = json.loads((root / "configs/flash_sft/reap307-kimi-mixed-cosine.json").read_text())
    assert placement_free(dict(written, offload_every=4)) == placement_free(written)
    assert placement_free(dict(written, seed=1)) != placement_free(written)


@pytest.mark.parametrize(
    "mode,expected", [("bounded", 1), ("stop_file", 1), ("full", 8)]
)
def test_segment_stop_saves_and_validates_final_update_without_changing_schedule(
    tmp_path, monkeypatch, mode, expected
):
    run = object.__new__(FlashSFT)
    run.config = {"validation": {"every": 20}, "save_every": 40, "wandb": None}
    run.run_dir, run.rank, run.world, run.device = tmp_path, 0, 4, torch.device("cpu")
    run.rows, run.schedule, run.update = [None] * 7, [None] * 8, 0
    run.kernel_stack, run.trainables = {}, {}
    run.sharded = set()
    run._preflight = lambda: None
    evaluated, saved = [], []
    run._train_update = lambda step: {"update": step + 1}
    run._validate = lambda step: evaluated.append(step) or {"nll": 1.0}
    run._save = saved.append
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, op: None)
    options = {}
    if mode == "bounded":
        options["max_updates"] = 1
    elif mode == "stop_file":
        stop = tmp_path / "STOP"
        stop.touch()
        options["stop_file"] = stop
    run.run(**options)
    state = json.loads((tmp_path / "RUN_STATE.json").read_text())
    assert state["update"] == expected and state["total_updates"] == 8
    assert state["complete"] == (expected == 8)
    assert saved == [expected] and evaluated == [0, expected]
    assert len(run.schedule) == 8


@pytest.mark.parametrize("change_router", [False, True])
def test_parameter_qualification_checks_replica_updates_and_frozen_routers(
    tmp_path, monkeypatch, change_router
):
    run = object.__new__(FlashSFT)
    run.config = {
        "validation": None,
        "save_every": 40,
        "wandb": None,
        "train_routers": False,
        "expert_lora_sharded": False,
    }
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.gate = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    run.bundle = SimpleNamespace(model=model)
    run.head = SimpleNamespace(weight=torch.ones(2, 2))
    adapter = torch.nn.Parameter(torch.zeros(2, 2))
    run.trainables = {"layers.0.self_attn.q_proj.lora_B": adapter}
    run.kernel_stack = {}
    run.sharded = set()
    run._preflight = lambda: None
    run.run_dir, run.rank, run.world, run.device = tmp_path, 0, 2, torch.device("cpu")
    run.rows, run.schedule, run.update = [None] * 2, [None] * 8, 0

    def update(step):
        with torch.no_grad():
            adapter.add_(1)
            if change_router:
                model.mlp.gate.weight.add_(1)
        return {"update": step + 1}

    run._train_update, run._save = update, lambda step: None

    def gather(output, value):
        output[:] = [value] * 2

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if change_router:
        with pytest.raises(RuntimeError, match="frozen router"):
            run.run(max_updates=1, verify_parameters=True)
        assert not (tmp_path / "PARAMETERS_PASSED.json").exists()
    else:
        run.run(max_updates=1, verify_parameters=True)
        report = json.loads((tmp_path / "PARAMETERS_PASSED.json").read_text())
        assert report["trainable_replicas_equal"] and report["frozen_routers_unchanged"]
        assert report["updated_adapter_groups"] == ["attention"]
