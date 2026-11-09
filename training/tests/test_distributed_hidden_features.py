"""Hidden feature distribution preserves forward shapes and decision coverage."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from training.ppo import (
    batch_policy_hidden_segments,
    distribute_policy_hidden_batches,
    merge_policy_hidden_features,
)


def test_distribution_balances_whole_existing_batches():
    plans = [
        (i, [0], [length - 1], [i + 1] * length)
        for i, length in enumerate([16, 15, 14, 13, 8, 7, 4, 3, 2])
    ]
    batches = batch_policy_hidden_segments(plans, max_tokens=16, token_budget=32)
    shares = distribute_policy_hidden_batches(batches, 4)
    assert sorted(id(batch) for share in shares for batch in share) == sorted(
        map(id, batches)
    )
    costs = [sum(len(batch) * len(batch[0][3]) for batch in share) for share in shares]
    assert max(costs) - min(costs) <= max(
        len(batch) * len(batch[0][3]) for batch in batches
    )
    assert all(len(batch) * len(batch[0][3]) <= 32 for batch in batches)
    assert sorted(plan[0] for batch in batches for plan in batch) == list(
        range(len(plans))
    )


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unexpected", "nonfinite", "width", "rows"]
)
def test_merge_rejects_invalid_or_incomplete_decision_features(failure):
    blocks = [([(0, 0), (0, 1)], torch.ones(2, 4))]
    expected = {(0, 0), (0, 1)}
    if failure == "missing":
        expected.add((1, 0))
    elif failure == "duplicate":
        blocks.append(([(0, 0)], torch.ones(1, 4)))
    elif failure == "unexpected":
        blocks[0][0][1] = (1, 0)
    elif failure == "nonfinite":
        blocks[0][1][0, 0] = float("nan")
    elif failure == "width":
        blocks.append(([(1, 0)], torch.ones(1, 3)))
        expected.add((1, 0))
    else:
        blocks[0][0].pop()
    with pytest.raises(RuntimeError, match="hidden critic"):
        merge_policy_hidden_features(blocks, expected)


@pytest.mark.parametrize(
    "scenario", ["normal", "sigterm", "sigterm_all", "sigterm_worker"]
)
def test_real_three_rank_feature_protocol_preserves_policy_and_completes_before_shutdown(
    tmp_path, scenario
):
    # Run the real DDP worker command loop and real Gloo object collectives.
    # Only the expensive policy and CUDA tensor group are replaced with a
    # deterministic causal CPU decoder and Gloo. Rank zero and workers are
    # separate OS processes, including the signal/cleanup protocol.
    script = r"""
import datetime, errno, json, os, signal, socket, subprocess, sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import training.ddp_worker as ddp
import training.ppo as ppo
from training.config import TrainingConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.shutdown import TrainingShutdown, training_sigterm_handler

root, scenario, role = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
real_trainer_class = ppo.PPOTrainer

def toy_trainer(initial, rank):
    trainer = real_trainer_class.__new__(real_trainer_class)
    trainer.config = SimpleNamespace(max_sequence_tokens=16, max_micro_batch_tokens=16)
    trainer.device = torch.device('cpu')
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight.data.fill_(initial)
    trainer._set_policy_mode_for_logprob_computation = lambda: trainer.model.train()
    trainer._autocast_context = nullcontext
    trainer.calls = []
    class Decoder:
        def __call__(self, input_ids, attention_mask):
            assert not torch.is_grad_enabled()
            trainer.calls.append(input_ids.tolist())
            if scenario.startswith('sigterm') and rank == 1 and len(trainer.calls) == 1:
                if scenario != 'sigterm_worker':
                    os.kill(int(sys.argv[5]), signal.SIGTERM)
                if scenario == 'sigterm_all':
                    for child in root.glob('worker_*.pid'):
                        os.kill(int(child.read_text()), signal.SIGTERM)
                elif scenario == 'sigterm_worker':
                    os.kill(os.getpid(), signal.SIGTERM)
            values = (input_ids.cumsum(-1).float() + trainer.model.weight.item())
            return SimpleNamespace(last_hidden_state=values.unsqueeze(-1).expand(-1, -1, 4))
    trainer._decoder_and_lm_head = lambda: (Decoder(), None)
    return trainer

if role == 'worker':
    rank, parent = int(sys.argv[4]), int(sys.argv[5])
    spec = json.loads(Path(sys.argv[6]).read_text())
    (root / f'worker_{rank}.pid').write_text(str(os.getpid()))
    trainer = toy_trainer(-99, rank)
    ppo.PPOTrainer = lambda *args, **kwargs: trainer
    torch.cuda.set_device = lambda device: None
    real_new_group = dist.new_group
    dist.new_group = lambda *args, **kwargs: real_new_group(backend='gloo', timeout=datetime.timedelta(seconds=15))
    real_flatten = ddp.flatten_tensors
    ddp.flatten_tensors = lambda tensors, **kwargs: real_flatten(tensors, device=torch.device('cpu'), dtype=torch.float32)
    ddp.run_worker(spec)
    (root / f'worker_{rank}.json').write_text(json.dumps({
        'calls': trainer.calls, 'weight': trainer.model.weight.item(),
        'grad': trainer.model.weight.grad is not None}))
    raise SystemExit(0)

# Exercise the actual rank-zero constructor and worker command loop; replace
# only the expensive model, CUDA tensor group, and child Python entry point.
# Assert the rendezvous port is already owned whenever a child is spawned.
real_popen = subprocess.Popen
def start_worker(command, **kwargs):
    spec_path = Path(command[-1])
    spec = json.loads(spec_path.read_text())
    with socket.socket() as other:
        try:
            other.bind(('127.0.0.1', spec['port']))
        except OSError as error:
            assert error.errno == errno.EADDRINUSE
        else:
            raise AssertionError('Rendezvous port was not reserved before spawn')
    return real_popen([sys.executable, __file__, str(root), scenario, 'worker',
                       str(spec['rank']), str(os.getpid()), str(spec_path)], **kwargs)
subprocess.Popen = start_worker
real_new_group = dist.new_group
dist.new_group = lambda *args, **kwargs: real_new_group(backend='gloo', timeout=datetime.timedelta(seconds=15))
trainer = toy_trainer(2.75, 0)
config = TrainingConfig.from_dict(dict(model_name='unused',output_dir=str(root),
    device='cpu',worker_device_map_gpus=[[1],[2]]))
ppo._DDP_SHUTDOWN_TIMEOUT_SECONDS = 5
rollouts = []
for index, length in enumerate([13, 12, 11, 10, 9, 8]):
    prefix = list(range(1, length + 1))
    rollouts.append(RolloutTrajectory(str(index), 'game', [
        RolloutStep(0, [], [], {}, 0, False, prefix, [100]),
        RolloutStep(1, [], [], {}, 0, False, prefix + [100, 200], [300]),
        RolloutStep(2, [], [], {}, 1, True, [9, 8], [10], metadata={'segment_start': True}),
    ]))
with training_sigterm_handler():
    group = ppo._DdpWorkerGroup(config=config, parameters=list(trainer.model.parameters()),
                               resume_adapter_dir=None)
    reference = trainer._extract_policy_hidden_states(rollouts)
    reference_calls = trainer.calls[:]
    trainer.calls.clear()
    trainer._ddp_group = group
    interrupted = False
    try:
        actual = trainer._extract_policy_hidden_states(rollouts)
        assert actual == reference
    except TrainingShutdown:
        interrupted = True
    finally:
        group.shutdown()
    assert interrupted == scenario.startswith('sigterm')
    assert trainer.model.weight.item() == 2.75
    assert trainer.model.weight.grad is None
    worker_results = [json.loads((root / f'worker_{rank}.json').read_text()) for rank in (1, 2)]
    assert all(value['weight'] == 2.75 and not value['grad'] for value in worker_results)
    calls = trainer.calls + [call for value in worker_results for call in value['calls']]
    if scenario == 'normal':
        assert sorted(map(str, calls)) == sorted(map(str, reference_calls))
        assert len(reference) == 18
        assert all(value['calls'] for value in worker_results)
    else:
        # One bounded round finishes; the next is never dispatched.
        assert len(calls) == 3
"""
    script_path = tmp_path / "protocol.py"
    script_path.write_text(script)
    result = subprocess.run(
        [sys.executable, str(script_path), str(tmp_path), scenario, "root"],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OMP_NUM_THREADS": "1",
            "ARC3_RUN_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "run_failure.json").exists()
    assert not (tmp_path / "STOP_AUTO_RELAUNCH").exists()


def test_hidden_features_respect_token_budget_when_context_limit_is_larger():
    plans = [(i, [0], [15999], [1] * 16000) for i in range(4)]
    batches = batch_policy_hidden_segments(plans, max_tokens=65536, token_budget=40960)
    assert [len(batch) for batch in batches] == [2, 2]
    assert all(len(batch) * len(batch[0][3]) <= 40960 for batch in batches)


def test_hidden_features_allow_a_single_oversized_prompt_without_batching_it():
    plans = [
        (i, [0], [n - 1], [1] * n) for i, n in enumerate([45000, 16000, 16000, 16000])
    ]
    batches = batch_policy_hidden_segments(plans, max_tokens=65536, token_budget=40960)
    assert [len(batch) for batch in batches] == [1, 2, 1]
    assert batch_policy_hidden_segments([], max_tokens=65536, token_budget=40960) == []
