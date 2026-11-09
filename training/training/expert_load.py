"""Expert load balance of the Flash learner's own routing (observation only).

Every router of the learner hands its top-k expert indices to the optional
``route_recorder`` hook (``training/flash/adapters.py::router_forward``).
``ExpertLoadRecorder`` adds them, on the GPU, into per-expert assignment counts
per layer for three token roles:

- ``train``: every real token of the sequences a training forward scores;
- ``rollout``: the positions whose next token the policy sampled (the tokens
  the PPO loss scores), a subset of ``train``;
- ``eval``: the same positions in a bounded no-grad pass over eval episodes.

The counts are the learner's routing of those tokens, not the servers'
(which differs only by numerical flips at the parity floor). Nothing here
feeds a loss: there is no balancing term and no optimization pressure.

Per window the counts are pooled across ranks with one ``all_reduce`` and
summarized per layer (``load_statistics``); ``training/metrics.py`` names the
TensorBoard tags.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

ROLES = ("train", "rollout", "eval")
# Role rows a capture fills: the training forward fills train and rollout
# together, the eval pass fills eval.
TRAINING_ROLES = slice(0, 2)
EVAL_ROLES = slice(2, 3)

# Official-eval cycles leave their episodes here (under the run root) for the
# learner's eval pass; the learner consumes and deletes them.
EVAL_HANDOFF_DIR = "expert_load_eval"
_EVAL_HANDOFF_VERSION = 1


class ExpertLoadRecorder:
    """Per role, layer and expert: top-k assignments of the learner's routers.

    Installed as every router's ``route_recorder``. A router calls ``record``
    on every forward; it adds only while a ``capture`` is armed with one
    weight row per role in the router's token order. Checkpoint recomputation
    runs after the capture closes and unarmed diagnostic forwards (parity,
    probes) never arm one, so neither adds. Counting is GPU-only (no host
    synchronization); ``pooled_counts`` is the one collective per window.
    """

    def __init__(self, layers: int, experts: int, top_k: int, device: torch.device | str) -> None:
        self.top_k = int(top_k)
        self.counts = torch.zeros(
            len(ROLES), layers, experts, dtype=torch.float64, device=device
        )
        self._roles: slice | None = None
        self._weights: torch.Tensor | None = None
        # The capture's weights repeated per top-k slot, shared by all layers.
        self._slot_weights: torch.Tensor | None = None

    @contextmanager
    def capture(self, roles: slice, weights: torch.Tensor):
        """Count the routing of the forwards inside, weighted per token.

        ``weights`` is ``[roles, tokens]`` in the order the routers see the
        tokens (``router_token_weights``): 1 for a token of that role, 0 for
        anything else (padding, packing filler, other roles)."""
        if self._weights is not None:
            raise RuntimeError("an expert load capture is already armed")
        if weights.ndim != 2 or weights.shape[0] != len(range(*roles.indices(len(ROLES)))):
            raise ValueError(f"expert load weights {tuple(weights.shape)} do not match roles {roles}")
        self._roles, self._weights = roles, weights.to(self.counts.device, torch.float64)
        try:
            yield
        finally:
            self._roles = self._weights = self._slot_weights = None

    @torch.no_grad()
    def record(self, layer: int, indices: torch.Tensor, values: torch.Tensor) -> None:
        del values  # assignment counts only; the routing weights are not used
        weights = self._weights
        if weights is None:
            return
        tokens, top_k = indices.shape
        if tokens != weights.shape[1]:
            raise ValueError(
                f"expert load: layer {layer} routed {tokens} tokens, the capture "
                f"describes {weights.shape[1]}; the router token layout changed"
            )
        # indices flatten token-major ([t0 e0..e9, t1 e0..e9, ...]), so each
        # token's weight repeats top_k times in the same order.
        slot_weights = self._slot_weights
        if slot_weights is None or slot_weights.shape[1] != tokens * top_k:
            slot_weights = self._slot_weights = weights.repeat_interleave(top_k, dim=1)
        self.counts[self._roles, layer].scatter_add_(
            1, indices.reshape(1, -1).expand(weights.shape[0], -1), slot_weights
        )

    def reset(self) -> None:
        self.counts.zero_()

    def pooled_counts(self) -> torch.Tensor:
        """Every rank's counts summed (one ``all_reduce``), on the CPU; resets."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.counts, op=dist.ReduceOp.SUM)
        pooled = self.counts.to("cpu", copy=True)
        self.counts.zero_()
        return pooled


def install_expert_load_recorder(
    model: torch.nn.Module, *, router_type: type, device: torch.device | str
) -> ExpertLoadRecorder | None:
    """Attach one recorder to every ``router_type`` module of ``model``.

    Routers carry their decoder layer as ``layer_idx`` and their shape as
    ``num_experts`` and ``top_k``. A router of any other class has no hook;
    None when the model has no ``router_type`` router at all."""
    routers = [module for module in model.modules() if isinstance(module, router_type)]
    if not routers:
        return None
    layers = sorted(int(router.layer_idx) for router in routers)
    if layers != list(range(len(routers))):
        raise ValueError(f"expert load metrics: router layers {layers} are not 0..{len(routers) - 1}")
    shapes = {(int(router.num_experts), int(router.top_k)) for router in routers}
    if len(shapes) != 1:
        raise ValueError(f"expert load metrics: routers disagree on (experts, top_k) {sorted(shapes)}")
    experts, top_k = shapes.pop()
    recorder = ExpertLoadRecorder(len(routers), experts, top_k, device)
    for router in routers:
        router.route_recorder, router.route_layer = recorder, int(router.layer_idx)
    return recorder


def router_token_weights(
    samples: Sequence[Any], roles: slice, *, packed_alignment: int | None
) -> torch.Tensor:
    """``[roles, tokens]`` capture weights of one micro-batch, on the CPU.

    The rows follow ``collate_samples`` (right padding to the longest sample).
    A padded batch reaches the routers row by row (``packed_alignment`` None,
    the SDPA learner); the flex-QSA learner first packs it into one stream
    with ``pack_padded_batch`` (each document padded to the alignment), so the
    weights are packed by that same layout."""
    names = ROLES[roles]
    width = max(len(sample.input_ids) for sample in samples)
    public = torch.zeros(len(names), len(samples), width, dtype=torch.float64)
    attention = torch.zeros(len(samples), width, dtype=torch.long)
    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        real = torch.tensor(sample.attention_mask, dtype=torch.float64)
        # Position i predicts token i + 1: it is a sampled decision when that
        # token is a completion token.
        decides = torch.zeros(length, dtype=torch.float64)
        decides[:-1] = torch.tensor(sample.completion_mask[1:], dtype=torch.float64)
        attention[row, :length] = real.long()
        for index, name in enumerate(names):
            public[index, row, :length] = real if name == "train" else decides * real
    if packed_alignment is None:
        return public.reshape(len(names), -1)
    from training.flash.input_layout import pack_padded_batch

    layout = pack_padded_batch(
        torch.zeros_like(attention), attention, alignment=packed_alignment
    )
    return layout.pack(public.permute(1, 2, 0)).squeeze(0).T.contiguous()


def load_statistics(counts: torch.Tensor) -> dict[str, float] | None:
    """Balance of one role's ``[layers, experts]`` assignment counts.

    Per layer: ``cv`` (population standard deviation of the per-expert counts
    over their mean; 0 = balanced), ``max_share`` (the busiest expert's count
    over the mean; 1 = balanced), ``entropy`` (of the load distribution, over
    log(experts); 1 = uniform) and the fraction of experts with no tokens.
    Returned averaged over layers, with the worst layer of the first three.
    Layers that routed nothing are skipped; None when none routed anything."""
    counts = counts.to(torch.float64)
    routed = counts.sum(-1) > 0
    if not bool(routed.any()):
        return None
    layer_ids = torch.arange(counts.shape[0])[routed]
    counts = counts[routed]
    experts = counts.shape[-1]
    mean = counts.mean(-1)
    cv = counts.std(-1, correction=0) / mean
    max_share = counts.amax(-1) / mean
    share = counts / counts.sum(-1, keepdim=True)
    entropy = -torch.special.xlogy(share, share).sum(-1) / math.log(experts)
    unused = (counts == 0).to(torch.float64).mean(-1)
    worst = int(max_share.argmax())
    return {
        "expert_load_cv": float(cv.mean()),
        "expert_load_max_share": float(max_share.mean()),
        "expert_load_entropy": float(entropy.mean()),
        "experts_unused_fraction": float(unused.mean()),
        "expert_load_cv_worst_layer": float(cv.max()),
        "expert_load_max_share_worst_layer": float(max_share[worst]),
        "expert_load_entropy_worst_layer": float(entropy.min()),
        "worst_layer": float(layer_ids[worst]),
        "assignments": float(counts.sum(-1).mean()),
    }


def summarize_counts(counts: torch.Tensor, *, top_k: int) -> dict[str, dict[str, float]]:
    """``load_statistics`` of every role that routed tokens, by role name,
    with the role's token count (each token makes ``top_k`` assignments)."""
    summary = {}
    for index, role in enumerate(ROLES):
        statistics = load_statistics(counts[index])
        if statistics is not None:
            statistics["tokens"] = statistics.pop("assignments") / top_k
            summary[role] = statistics
    return summary


def eval_handoff_episodes(config: Any) -> int:
    """Eval episodes per cycle the learner scores: 0 unless a Flash run has them on."""
    training = config.training
    if training.weight_mode != "flash" or not training.flash.load_balance_metrics:
        return 0
    return int(training.flash.load_balance_eval_episodes)


def _spread_by_game(rollouts: Sequence[Any], limit: int) -> list[Any]:
    """Up to ``limit`` rollouts, one per game in turn (slot order within a game)."""
    by_game: dict[str, list[Any]] = {}
    for rollout in rollouts:
        by_game.setdefault(str(rollout.task_id or ""), []).append(rollout)
    selected: list[Any] = []
    depth = 0
    while len(selected) < limit and any(depth < len(group) for group in by_game.values()):
        for group in by_game.values():
            if depth < len(group) and len(selected) < limit:
                selected.append(group[depth])
        depth += 1
    return selected


def write_eval_handoff(
    run_root: Path,
    *,
    cycle_index: int,
    policy_iteration: int,
    rollouts: Sequence[Any],
    episodes: int,
    max_sequence_tokens: int,
) -> Path | None:
    """Leave up to ``episodes`` of an eval cycle's episodes for the learner.

    Each episode is tokenized and packed as training packs it
    (``build_samples`` + ``pack_episode_samples``, no learning signal). An
    episode without captured token ids, or one that does not pack under the
    training budget, is skipped. Returns the file written, or None."""
    from training.dataset import build_samples
    from training.episode_packing import pack_episode_samples
    from training.rewards import StepSignal

    if episodes <= 0:
        return None
    samples, kept = [], 0
    for rollout in _spread_by_game(rollouts, len(rollouts)):
        if kept == episodes:
            break
        signals = [[StepSignal(reward=0.0, return_to_go=0.0, advantage=0.0)] * len(rollout.steps)]
        try:
            steps = build_samples([rollout], per_rollout_signals=signals)
            packed, _ = pack_episode_samples(steps, max_sequence_tokens=max_sequence_tokens)
        except (ValueError, RuntimeError) as error:
            log.info("expert load: eval episode %s not handed to the learner: %s", rollout.trajectory_id, error)
            continue
        if packed:
            samples.extend(packed)
            kept += 1
    if not samples:
        return None
    directory = Path(run_root) / EVAL_HANDOFF_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cycle_{int(cycle_index):05d}.pt"
    staging = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "version": _EVAL_HANDOFF_VERSION,
            "cycle_index": int(cycle_index),
            "policy_iteration": int(policy_iteration),
            "episodes": kept,
            "samples": samples,
        },
        staging,
    )
    staging.replace(path)
    return path


def take_eval_handoff(run_root: Path) -> dict[str, Any] | None:
    """The newest pending eval cycle, removing every pending file (older
    cycles are superseded). None when nothing is pending or it is unreadable."""
    directory = Path(run_root) / EVAL_HANDOFF_DIR
    pending = sorted(directory.glob("cycle_*.pt")) if directory.is_dir() else []
    if not pending:
        return None
    payload = None
    try:
        payload = torch.load(pending[-1], map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("version") != _EVAL_HANDOFF_VERSION:
            raise ValueError(f"unsupported eval handoff {pending[-1]}")
    except Exception as error:  # noqa: BLE001 - a diagnostic input never stops training
        log.warning("expert load: ignoring unreadable eval handoff %s: %s", pending[-1], error)
        payload = None
    for path in pending:
        path.unlink(missing_ok=True)
    return payload
