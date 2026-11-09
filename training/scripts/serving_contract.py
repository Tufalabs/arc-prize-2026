"""Training lineage metadata over public harness context observations."""

from dataclasses import dataclass
from typing import Any

from ullm.backend import TokenCount as CompactionTokenCount
from training_backend import (
    TrainingLLMAgentSolver as ContractLLMAgentSolver,
    validate_training_backend as install_contract_backend,
)

__all__ = [
    "CompactionTokenCount",
    "TrainingContextObserver",
    "ContractLLMAgentSolver",
    "install_contract_backend",
]


@dataclass
class TrainingContextObserver:
    """Collect exact compaction evidence without implementing agent behavior."""

    last_request_epoch: int | None = None
    candidate: Any = None
    reset_proof: Any = None
    served_proof: Any = None
    last_trigger: int | None = None
    measured_epoch: int | None = None
    last_rewrite_epoch: int | None = None
    reply_cap_tokens: int = 0

    def prepare_request(self, params: Any) -> None:
        """Measurements must not reuse the previous generation's metadata."""
        params.request_context.pop("arc3_lineage", None)
        self.candidate = None
        self.last_trigger = None
        self.measured_epoch = None

    def prompt_measured(
        self, count: int, *, epoch: int, trigger: int, compacting: bool
    ) -> None:
        """Keep pre-trim tokens when the current history exceeds its trigger."""
        self.last_trigger = trigger
        self.measured_epoch = epoch
        if compacting and epoch == self.last_request_epoch and count > trigger:
            if not isinstance(count, CompactionTokenCount):
                raise RuntimeError("compaction lacks the exact pre-trim token prompt")
            self.candidate = count

    def history_rewritten(self, *, epoch: int) -> None:
        """Require fresh evidence before the harness rewrites served history."""
        if self.reset_proof is not None:
            if (
                self.reset_proof == "served"
                and self.last_rewrite_epoch is not None
                and epoch == self.last_rewrite_epoch + 1
            ):
                self.last_rewrite_epoch = epoch
                self.candidate = None
                return
            raise RuntimeError("compaction cannot reuse an earlier history-reset proof")
        if self.candidate is None and self.served_proof is None:
            raise RuntimeError(
                "history may change only during a verified context compaction"
            )
        if epoch != self.last_request_epoch:
            raise RuntimeError("compaction cannot reuse an earlier history-reset proof")
        self.reset_proof = "served" if self.served_proof is not None else self.candidate
        self.last_rewrite_epoch = epoch
        self.candidate = None

    def request_prepared(
        self, params: Any, *, epoch: int, prompt_tokens: int, target: int
    ) -> None:
        """Stamp a generation after the harness completes context preparation."""
        previous = self.last_request_epoch
        if previous is None:
            if epoch != 0 or self.reset_proof is not None:
                raise RuntimeError(
                    "the first training generation must start at epoch zero"
                )
            kind = "start"
        elif epoch == previous:
            if self.reset_proof is not None:
                raise RuntimeError("a compaction proof must advance the history epoch")
            kind = "continue"
        elif epoch > previous and self.reset_proof is not None:
            if self.reset_proof != "served" and epoch != previous + 1:
                raise RuntimeError("measured compaction must advance exactly one epoch")
            if self.reset_proof == "served" and self.reply_cap_tokens:
                target += max(0, self.reply_cap_tokens - params.max_tokens)
            if prompt_tokens > target:
                raise RuntimeError(
                    "compaction did not meet the trainer's configured context target"
                )
            kind = "reset"
        else:
            raise RuntimeError("undeclared history rewrite before generation")
        # Only a measurement for this preparation may identify a served proof.
        served_proof = None
        if (
            self.measured_epoch == epoch
            and self.last_trigger is not None
            and prompt_tokens > self.last_trigger
        ):
            if not isinstance(prompt_tokens, CompactionTokenCount):
                raise RuntimeError("compaction lacks the exact pre-trim token prompt")
            served_proof = prompt_tokens
        proof = None if self.reset_proof == "served" else self.reset_proof
        media = getattr(proof, "media", None)
        params.request_context["arc3_lineage"] = {
            "version": 1,
            "epoch": epoch,
            "kind": kind,
            "compaction_prompt_token_ids": list(proof.prompt_token_ids)
            if proof is not None
            else None,
            **({"compaction_media": media} if media is not None else {}),
        }
        self.last_request_epoch = epoch
        self.reset_proof = None
        self.candidate = None
        self.served_proof = served_proof
        self.last_rewrite_epoch = None
        self.last_trigger = None
        self.measured_epoch = None
