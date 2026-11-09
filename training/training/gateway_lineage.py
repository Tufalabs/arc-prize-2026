"""Admission ledger for ordinary full-game Duck context continuation.

A reset preserves Duck's existing eviction/journal behavior. Its proof is the
untrimmed exact prompt, measured or served, above the trainer's context
threshold before a smaller new-epoch prompt may be served.
No game-level sampling, forced summaries, or reward logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class LineageError(RuntimeError):
    """A generation would break the trainer-owned history contract."""


def _media_extends(
    previous: dict[str, Any] | None, current: dict[str, Any] | None
) -> bool:
    """Require an unchanged ordered media prefix until an authorized reset."""
    if previous is None:
        return True
    if current is None or previous["processing"] != current["processing"]:
        return False
    count = len(previous["images"])
    return (
        current["images"][:count] == previous["images"]
        and current["layout"][:count] == previous["layout"]
    )


@dataclass(frozen=True)
class LineagePolicy:
    compaction_trigger_tokens: int
    max_reset_prompt_tokens: int
    max_compaction_prompt_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.max_reset_prompt_tokens,
            self.compaction_trigger_tokens,
            self.max_compaction_prompt_tokens,
        )
        if (
            any(type(v) is not int for v in values)
            or not 0 < values[0] < values[2]
            or not 0 < values[1] < values[2]
        ):
            raise ValueError(
                "reset ceiling and compaction threshold must be below the proof ceiling"
            )


def _token_ids(value: Any, *, label: str, maximum: int) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > maximum
        or any(type(token) is not int or token < 0 for token in value)
    ):
        raise LineageError(f"{label}: expected a bounded nonempty integer token list")
    return value


@dataclass
class _RunChain:
    epoch: int = 0
    last_prompt: list[int] = field(default_factory=list)
    last_output: list[int] = field(default_factory=list)
    pending_nonce: str | None = None
    pending_prompt: list[int] = field(default_factory=list)
    pending_metadata: dict[str, Any] = field(default_factory=dict)
    nonces: list[str] = field(default_factory=list)
    reported: bool = False
    last_media: dict[str, Any] | None = None
    pending_media: dict[str, Any] | None = None
    last_summary_request: bool = False
    pending_summary_request: bool = False


class LineageLedger:
    """Methods run under the gateway's store lock, before generation dispatch."""

    def __init__(self) -> None:
        self.runs: dict[str, _RunChain] = {}
        self.seen: set[tuple[str, str]] = set()
        self.closed_runs: set[str] = set()

    def admit(
        self,
        run_id: str,
        nonce: str,
        prompt: list[int],
        metadata: Any,
        policy: LineagePolicy,
        *,
        native_token_prompt: bool,
        media: dict[str, Any] | None = None,
        summary_request: bool = False,
    ) -> None:
        label = f"run {run_id} nonce {nonce}"
        if (run_id, nonce) in self.seen:
            raise LineageError(f"{label}: generation retries are forbidden")
        if run_id in self.closed_runs:
            raise LineageError(f"{label}: a closed run cannot generate again")
        if native_token_prompt is not True:
            raise LineageError(f"{label}: every generation requires native token IDs")
        _token_ids(prompt, label=label, maximum=policy.max_compaction_prompt_tokens)
        if not isinstance(metadata, dict) or set(metadata) - {"compaction_media"} != {
            "version",
            "epoch",
            "kind",
            "compaction_prompt_token_ids",
        }:
            raise LineageError(f"{label}: explicit arc3_lineage metadata is required")
        epoch, kind, proof = (
            metadata[k] for k in ("epoch", "kind", "compaction_prompt_token_ids")
        )
        if type(metadata["version"]) is not int or metadata["version"] != 1:
            raise LineageError(f"{label}: unsupported lineage version")
        if (
            type(epoch) is not int
            or epoch < 0
            or kind not in ("start", "continue", "reset")
        ):
            raise LineageError(f"{label}: invalid generation epoch or kind")
        if kind != "reset" and proof is not None:
            raise LineageError(f"{label}: only a reset may carry compaction evidence")
        if kind != "reset" and "compaction_media" in metadata:
            raise LineageError(f"{label}: only a reset may carry compaction media")
        chain = self.runs.get(run_id)
        if chain is None:
            if kind != "start" or epoch != 0:
                raise LineageError(f"{label}: the first request must start epoch zero")
            chain = _RunChain()
        else:
            if chain.reported:
                raise LineageError(f"{label}: generation after the result callback")
            if chain.pending_nonce is not None:
                raise LineageError(
                    f"{label}: concurrent generations within one rollout"
                )
            prefix = chain.last_prompt + chain.last_output
            if kind == "reset":
                if epoch <= chain.epoch:
                    raise LineageError(f"{label}: compaction must advance the epoch")
                if proof is None:
                    if (
                        not chain.last_summary_request
                        or len(chain.last_prompt) <= policy.compaction_trigger_tokens
                    ):
                        raise LineageError(
                            f"{label}: compaction lacks a served over-trigger proof"
                        )
                    if "compaction_media" in metadata:
                        raise LineageError(
                            f"{label}: a served proof carries no compaction media"
                        )
                    proof = chain.last_prompt
                else:
                    if epoch != chain.epoch + 1:
                        raise LineageError(
                            f"{label}: measured compaction must advance exactly one epoch"
                        )
                    proof = _token_ids(
                        proof,
                        label=f"{label} compaction proof",
                        maximum=policy.max_compaction_prompt_tokens,
                    )
                    if len(proof) < len(prefix) or proof[: len(prefix)] != prefix:
                        raise LineageError(
                            f"{label}: compaction proof does not extend the latest sampled history"
                        )
                    proof_media = metadata.get("compaction_media")
                    if proof_media is not None:
                        from ullm.media import validate_media

                        validate_media(proof_media, proof)
                    if not _media_extends(chain.last_media, proof_media):
                        raise LineageError(
                            f"{label}: compaction proof changes preceding images"
                        )
                    if len(proof) <= policy.compaction_trigger_tokens:
                        raise LineageError(
                            f"{label}: compaction below the trainer token threshold"
                        )
                if len(prompt) > policy.max_reset_prompt_tokens or len(prompt) >= len(
                    proof
                ):
                    raise LineageError(
                        f"{label}: compacted prompt exceeds the reset ceiling or did not shrink"
                    )
            else:
                if kind != "continue" or epoch != chain.epoch:
                    raise LineageError(
                        f"{label}: undeclared history reset or changed epoch"
                    )
                if chain.last_summary_request:
                    raise LineageError(
                        f"{label}: continuation after a summary requires a reset"
                    )
                if len(prompt) <= len(prefix) or prompt[: len(prefix)] != prefix:
                    raise LineageError(
                        f"{label}: prompt must extend the immediately preceding prompt+output token-for-token"
                    )
                if not _media_extends(chain.last_media, media):
                    raise LineageError(
                        f"{label}: continuation changes preceding images"
                    )
        self.seen.add((run_id, nonce))
        chain.pending_nonce = nonce
        chain.pending_prompt = list(prompt)
        chain.pending_metadata = dict(metadata)
        chain.pending_media = media
        chain.pending_summary_request = summary_request
        self.runs[run_id] = chain

    def complete(
        self, run_id: str, nonce: str, prompt: list[int], output: list[int]
    ) -> dict[str, Any]:
        chain = self.runs.get(run_id)
        if chain is None or chain.pending_nonce != nonce:
            raise LineageError(
                f"run {run_id} nonce {nonce}: capture has no unique admission"
            )
        if prompt != chain.pending_prompt:
            raise LineageError(
                f"run {run_id} nonce {nonce}: served prompt differs from the admitted token prompt"
            )
        chain.last_prompt = list(prompt)
        chain.last_output = list(output)
        chain.last_media = chain.pending_media
        chain.last_summary_request = chain.pending_summary_request
        chain.pending_summary_request = False
        chain.pending_media = None
        chain.epoch = chain.pending_metadata["epoch"]
        chain.nonces.append(nonce)
        chain.pending_nonce = None
        chain.pending_prompt = []
        result, chain.pending_metadata = chain.pending_metadata, {}
        return result

    def report(
        self, run_id: str, nonces: list[str], *, deadline_reached: bool = False
    ) -> None:
        chain = self.runs.get(run_id)
        if chain is None:
            if nonces:
                raise LineageError(f"run {run_id}: reported steps without generations")
            self.closed_runs.add(run_id)
            return
        if nonces != chain.nonces[: len(nonces)]:
            raise LineageError(
                f"run {run_id}: callback reordered or omitted an interior generation"
            )
        omitted = len(chain.nonces) - len(nonces) + int(chain.pending_nonce is not None)
        if omitted > 1 or (omitted and not deadline_reached):
            raise LineageError(
                f"run {run_id}: unreported generations require exactly one terminal deadline draw"
            )
        chain.reported = True

    def drain(self) -> None:
        if any(chain.pending_nonce is not None for chain in self.runs.values()):
            raise LineageError("cannot drain unfinished generations")
        self.closed_runs.update(self.runs)
        self.runs.clear()
        # Closed run IDs prohibit all future nonces; per-draw tombstones need
        # only survive while the corresponding run is active.
        self.seen.clear()
