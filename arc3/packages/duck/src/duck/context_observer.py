"""Public observation points around Duck's request-context lifecycle."""

from typing import Protocol

from ullm.backend import GenericSamplingParameters


class ContextObserver(Protocol):
    """Per-game, picklable observer; exceptions stop the request before generation.

    Observers may attach request metadata, but must not edit the conversation.
    The agent and its managers own measurement, compaction and history-epoch
    advancement. Prepared-request observations cover every backend attempt,
    including summary calls and malformed-output retries. Rewrite observations
    precede both agent compactions and manager rollbacks.
    """

    def prepare_request(self, params: GenericSamplingParameters) -> None:
        """Retire previous-request metadata before counting or compacting."""
        ...

    def prompt_measured(self, count: int, *, epoch: int, trigger: int, compacting: bool) -> None:
        """Observe a prompt count, including any evidence carried by an int subclass."""
        ...

    def history_rewritten(self, *, epoch: int) -> None:
        """Observe the old epoch immediately before a declared history rewrite."""
        ...

    def request_prepared(
        self, params: GenericSamplingParameters, *, epoch: int, prompt_tokens: int, target: int
    ) -> None:
        """Attach metadata immediately before each backend attempt, after its final count."""
        ...
