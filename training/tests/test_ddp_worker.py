import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from training.ddp_worker import (
    flatten_grads,
    flatten_tensors,
    length_balanced_split,
    round_robin_split,
    unflatten_into,
)


def test_round_robin_split_deals_in_order_and_balances() -> None:
    items = list(range(7))
    shares = round_robin_split(items, 3)
    assert shares == [[0, 3, 6], [1, 4], [2, 5]]
    assert max(len(s) for s in shares) - min(len(s) for s in shares) <= 1


def test_round_robin_split_world_one_is_identity() -> None:
    items = ["a", "b", "c"]
    assert round_robin_split(items, 1) == [items]


def _size(item: tuple[str, int]) -> int:
    return item[1]


def test_length_balanced_split_deals_slots_largest_first() -> None:
    items = [("a", 3), ("b", 9), ("c", 1), ("d", 9), ("e", 5), ("f", 7), ("g", 2)]
    shares = length_balanced_split(items, 3, size=_size)
    # Sorted: b9 d9 f7 | e5 a3 g2 | c1; ties keep their input order.
    assert shares == [
        [("b", 9), ("e", 5), ("c", 1)],
        [("d", 9), ("a", 3)],
        [("f", 7), ("g", 2)],
    ]
    assert length_balanced_split(items, 3, size=_size) == shares
    assert items[0] == ("a", 3)  # the input list is not reordered


@pytest.mark.parametrize("world_size", range(1, 9))
def test_length_balanced_split_slots_ranks_and_membership(world_size: int) -> None:
    generator = random.Random(world_size)
    for count in range(1, 3 * world_size + 2):
        items = [(f"m{i}", generator.randint(1, 6)) for i in range(count)]
        shares = length_balanced_split(items, world_size, size=_size)
        assert len(shares) == world_size
        assert sorted(item for share in shares for item in share) == sorted(items)
        base, extra = divmod(count, world_size)
        assert [len(share) for share in shares] == [
            base + (rank < extra) for rank in range(world_size)
        ]
        slots = [
            [share[slot] for share in shares if slot < len(share)]
            for slot in range(len(shares[0]))
        ]
        # Slot 0 holds the largest items; every slot is no smaller than the next.
        assert (
            sorted(map(_size, slots[0])) == sorted(map(_size, items))[-len(slots[0]) :]
        )
        for earlier, later in zip(slots, slots[1:]):
            assert min(map(_size, earlier)) >= max(map(_size, later))
        assert length_balanced_split(items, world_size, size=_size) == shares


def test_length_balanced_split_cycle_smaller_than_world() -> None:
    items = [("a", 2), ("b", 8), ("c", 5)]
    shares = length_balanced_split(items, 5, size=_size)
    assert shares == [[("b", 8)], [("c", 5)], [("a", 2)], [], []]
    assert length_balanced_split([], 3, size=_size) == [[], [], []]
    assert length_balanced_split(items, 1, size=_size) == [
        [("b", 8), ("c", 5), ("a", 2)]
    ]


def test_flatten_unflatten_roundtrip_preserves_values_and_shapes() -> None:
    tensors = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.float32).reshape(4) * -1.0,
    ]
    flat = flatten_tensors(tensors, device=torch.device("cpu"), dtype=torch.float32)
    assert flat.shape == (10,)

    targets = [torch.zeros(2, 3), torch.zeros(4)]
    unflatten_into(flat, targets)
    assert torch.equal(targets[0], tensors[0])
    assert torch.equal(targets[1], tensors[1])


def test_unflatten_rejects_size_mismatch() -> None:
    flat = torch.zeros(5)
    with pytest.raises(RuntimeError, match="targets consume"):
        unflatten_into(flat, [torch.zeros(2, 2)])


def test_flatten_grads_zero_fills_missing_grads() -> None:
    with_grad = torch.nn.Parameter(torch.ones(3))
    with_grad.grad = torch.full((3,), 2.0)
    without_grad = torch.nn.Parameter(torch.ones(2))

    flat = flatten_grads([with_grad, without_grad], device=torch.device("cpu"))
    assert torch.equal(flat, torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "scenario",
    ["clean", "broadcast", "destroy", "wait", "error", "worker_failure", "stderr"],
)
def test_ddp_shutdown_is_collective_before_wait_and_bounded(
    tmp_path: Path, scenario: str
) -> None:
    # A real child worker cannot exit cleanly until rank 0 destroys its groups.
    # Failure cases must kill/reap it and terminate the trainer process, even
    # when a collective never returns; an exception-only mock misses hangs in
    # interpreter finalization caused by stranded distributed operations.
    script = r'''
import os
from pathlib import Path
import subprocess
import sys
import time
import training.ppo as ppo

scenario, root = sys.argv[1], Path(sys.argv[2])
ppo._DDP_SHUTDOWN_TIMEOUT_SECONDS = float(sys.argv[3])
marker = root / "groups_destroyed"
worker_code = """
from pathlib import Path
import sys, time
if sys.argv[2] == 'worker_failure':
    sys.exit(7)
while not Path(sys.argv[1]).exists():
    time.sleep(0.01)
"""
worker = subprocess.Popen([sys.executable, "-c", worker_code, str(marker), scenario])
(root / "worker_pid").write_text(str(worker.pid))
group = ppo._DdpWorkerGroup.__new__(ppo._DdpWorkerGroup)
group._processes = [worker]

if scenario == "stderr":
    # Redirect stderr to a full, unread pipe so the emergency os.write really
    # blocks. Keep its read end open to avoid turning the stall into EPIPE.
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    while True:
        try:
            os.write(write_fd, b"x" * 65536)
        except BlockingIOError:
            break
    os.set_blocking(write_fd, True)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    (root / "stderr_pipe_blocked").touch()

def broadcast(*args, **kwargs):
    if scenario == "broadcast":
        time.sleep(60)
    if scenario in {"error", "stderr"}:
        raise RuntimeError("injected collective failure")

def destroy():
    if scenario == "destroy":
        time.sleep(60)
    if scenario == "clean":
        marker.touch()

ppo.dist.broadcast_object_list = broadcast
ppo.dist.destroy_process_group = destroy
group.shutdown()
assert scenario == "clean"
assert worker.returncode == 0
assert marker.exists()
# Repeated cleanup must not call into an already destroyed process group.
ppo.dist.destroy_process_group = lambda: (_ for _ in ()).throw(AssertionError())
group.shutdown()
'''
    # In the stalled scenarios the budget is the bound under test: it must
    # fire long before the 60 s stalls and the 30 s run timeout. Elsewhere a
    # correct shutdown has to finish within it, worker interpreter start-up
    # included, and 0.5 s did not cover that start-up on a loaded node.
    budget = 0.5 if scenario in {"broadcast", "destroy", "wait"} else 10.0
    env = os.environ.copy()
    env["ARC3_RUN_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script, scenario, str(tmp_path), str(budget)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    pid = int((tmp_path / "worker_pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    if scenario == "clean":
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "run_failure.json").exists()
    else:
        assert result.returncode == 1, result.stderr
        if scenario == "stderr":
            assert (tmp_path / "stderr_pipe_blocked").exists()
        else:
            assert "FATAL: DDP shutdown" in result.stderr
        failure = json.loads((tmp_path / "run_failure.json").read_text())
        assert "DDP shutdown" in failure["message"]
        assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()
        if scenario in {"error", "stderr"}:
            assert "injected collective failure" in failure["message"]
        elif scenario == "worker_failure":
            assert "exited with code 7" in failure["message"]


def test_dead_worker_interrupts_blocked_collective_and_publishes_failure(tmp_path):
    script = r"""
import os, subprocess, sys, threading, time
from pathlib import Path
import training.ppo as ppo
root = Path(sys.argv[1])
ppo._DDP_HEALTH_POLL_SECONDS = 0.02
worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.1); raise SystemExit(23)'])
group = ppo._DdpWorkerGroup.__new__(ppo._DdpWorkerGroup)
group._processes = [worker]
group._health_stopped = threading.Event()
threading.Thread(target=group._watch_worker_health, daemon=True).start()
# Model a main thread stranded inside a collective: no CLI health callback.
time.sleep(60)
raise AssertionError('watchdog did not interrupt the stalled trainer')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "ARC3_RUN_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    failure = json.loads((tmp_path / "run_failure.json").read_text())
    assert "worker rank 1" in failure["message"]
    assert "code 23" in failure["message"]
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()


@pytest.mark.parametrize("protocol", ["startup", "train_cycle", "sync_weights"])
def test_sigterm_finishes_real_collectives_before_exit_and_staging(tmp_path, protocol):
    # Real Gloo command + tensor groups exercise the production rank-0 methods
    # without GPUs. The worker sends an OS signal after accepting the command,
    # so an immediate exception would strand its next collective. Isolate rank
    # 0 too: the production fatal watchdog hard-exits on protocol violations.
    script = r'''
import datetime, os, signal, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import training.ppo as ppo
from training.memory_profile import NullMemoryLogger
from training.run_artifacts import stage_out_rollout_scratch
from training.shutdown import TrainingShutdown, raise_if_shutdown_requested, training_sigterm_handler

root, protocol = Path(sys.argv[1]), sys.argv[2]
method = (root / "rendezvous").as_uri()
worker_code = """
import datetime, json, os, signal, sys
from pathlib import Path
import torch
import torch.distributed as dist
root, method, parent = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
protocol = sys.argv[4]
dist.init_process_group('gloo', init_method=method, rank=1, world_size=2,
                        timeout=datetime.timedelta(seconds=15))
tensor_group = dist.new_group(backend='gloo')
weights = torch.zeros(1)
if protocol == 'startup':
    os.kill(parent, signal.SIGTERM)
dist.broadcast(weights, src=0, group=tensor_group)
if protocol == 'startup':
    (root / 'worker_completed_protocol').write_text('initialized')
commands = []
while True:
    command = [None]
    dist.broadcast_object_list(command, src=0)
    kind = command[0]['type']
    commands.append(kind)
    if kind == 'exit':
        break
    os.kill(parent, signal.SIGTERM)
    if kind == 'train_cycle':
        dist.reduce(torch.ones(1), dst=0, group=tensor_group)
        dist.gather_object({}, dst=0)
    else:
        assert kind == 'sync_weights'
    dist.broadcast(weights, src=0, group=tensor_group)
    (root / 'worker_completed_protocol').write_text(str(weights.item()))
dist.destroy_process_group()
(root / 'worker_commands.json').write_text(json.dumps(commands))
"""
worker = subprocess.Popen([sys.executable, '-c', worker_code, str(root), method, str(os.getpid()), protocol])
(root / 'worker_pid').write_text(str(worker.pid))
ppo._DDP_SHUTDOWN_TIMEOUT_SECONDS = 5.0
parameter = torch.nn.Parameter(torch.zeros(1))
parameter.grad = torch.ones(1)
group = ppo._DdpWorkerGroup.__new__(ppo._DdpWorkerGroup)
group.parameters = [parameter]
group.primary_device = torch.device('cpu')
group.world_size = 2
group._processes = [worker]
with training_sigterm_handler():
    dist.init_process_group('gloo', init_method=method, rank=0, world_size=2,
                            timeout=datetime.timedelta(seconds=15))
    group._nccl_group = dist.new_group(backend='gloo')
    group._flat_weights = torch.zeros(1)
    dist.broadcast(group._flat_weights, src=0, group=group._nccl_group)
    trainer = ppo.PPOTrainer.__new__(ppo.PPOTrainer)
    trainer._ddp_group = group
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(max_grad_norm=1.0)
    trainer.live_entropy_coef = 0.0
    trainer._optimizer_step_count = 0
    trainer._trainable_parameters = lambda: [parameter]
    trainer._set_optimizer_learning_rate = lambda step: 0.1
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer._micro_batch_forward_backward = lambda *a, **kw: dict(
        mean_ratio=1.0, clip_fraction=0.0, entropy=0.0, entropy_loss=0.0, loss=1.0)
    totals = dict.fromkeys(['mean_ratio_sum', 'clip_fraction_sum', 'entropy_sum',
        'entropy_loss_sum', 'ratio_updates', 'grad_norm_sum', 'max_grad_norm',
        'grad_norm_updates', 'final_loss', 'global_step'], 0)
    scratch = root / 'scratch' / 'training'
    scratch.mkdir(parents=True)
    (scratch / 'retained.json').write_text('{"completed": true}')
    run_root = root / 'durable'
    run_root.mkdir()
    interrupted = False
    try:
        if protocol == 'train_cycle':
            trainer._execute_train_cycle([[1], [2]], cycle_index=1, epoch_label='test',
                epoch_number=1, iteration_index=0, loss_denominator=1.0, world_size=2,
                mem=NullMemoryLogger(), accumulators=totals, expected_optimizer_steps_total=1)
        elif protocol == 'sync_weights':
            group.sync_weights_out_of_cycle()
        else:
            # As in CLI startup, wait until trainer ownership is assigned.
            raise_if_shutdown_requested()
    except TrainingShutdown:
        interrupted = True
    finally:
        trainer.shutdown_workers()
        assert worker.returncode == 0
        assert (root / 'worker_completed_protocol').exists()
        # A repeated signal during cleanup must not prevent the durable copy.
        signal.raise_signal(signal.SIGTERM)
        stage_out_rollout_scratch(run_root=run_root, rollout_roots=[scratch],
            server_log_paths=[], cleanup_rollout_artifacts=False)
    assert interrupted, 'signal was not honored after completing the protocol'
    assert trainer.optimizer_step_count == (1 if protocol == 'train_cycle' else 0)
    assert not scratch.exists()
    assert (run_root / 'training' / 'retained.json').read_text() == '{"completed": true}'

'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), protocol],
        env={**os.environ, "ARC3_RUN_DIR": str(tmp_path), "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "worker_commands.json").read_text()) == (
        ["exit"] if protocol == "startup" else [protocol, "exit"]
    )
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "worker_pid").read_text()), 0)
    assert not (tmp_path / "run_failure.json").exists()
    assert not (tmp_path / "STOP_AUTO_RELAUNCH").exists()


def test_worker_entry_handles_term_before_importing_torch_worker(monkeypatch):
    import builtins
    import signal
    from types import SimpleNamespace
    from training.ddp_worker_entry import main

    calls = []
    original_import = builtins.__import__
    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))

    def intercept(name, *args, **kwargs):
        if name == "training.ddp_worker":
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)
            return SimpleNamespace(main=lambda: calls.append("worker ran"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept)
    main()
    assert calls == [(os.getppid(), signal.SIGTERM), "worker ran"]
    assert signal.getsignal(signal.SIGTERM) is previous_handler
