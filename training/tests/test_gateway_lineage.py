import pytest

from training.gateway_lineage import LineageError, LineageLedger, LineagePolicy


POLICY = LineagePolicy(8, 3, 20)


def metadata(kind="start", epoch=0, proof=None):
    return dict(version=1, epoch=epoch, kind=kind, compaction_prompt_token_ids=proof)


def started():
    ledger = LineageLedger()
    ledger.admit("run", "first", [1, 2], metadata(), POLICY, native_token_prompt=True)
    ledger.complete("run", "first", [1, 2], [3, 4])
    return ledger


def test_arbitrary_output_tokens_continue_exactly_and_cannot_be_omitted():
    ledger = started()
    prompt = [1, 2, 3, 4, 5]
    ledger.admit(
        "run", "second", prompt, metadata("continue"), POLICY, native_token_prompt=True
    )
    ledger.complete("run", "second", prompt, [17, 18])
    with pytest.raises(LineageError, match="token-for-token"):
        ledger.admit(
            "run",
            "third",
            [1, 2, 3, 4, 5, 19],
            metadata("continue"),
            POLICY,
            native_token_prompt=True,
        )
    ledger.admit(
        "run",
        "third",
        prompt + [17, 18, 19],
        metadata("continue"),
        POLICY,
        native_token_prompt=True,
    )


def test_verified_eviction_resets_once_without_generating_a_summary():
    ledger = started()
    proof = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ledger.admit(
        "run",
        "reset",
        [90, 91],
        metadata("reset", 1, proof),
        POLICY,
        native_token_prompt=True,
    )
    recorded = ledger.complete("run", "reset", [90, 91], [92])
    assert recorded == metadata("reset", 1, proof)
    ledger.admit(
        "run",
        "after",
        [90, 91, 92, 93],
        metadata("continue", 1),
        POLICY,
        native_token_prompt=True,
    )
    ledger.complete("run", "after", [90, 91, 92, 93], [94])
    with pytest.raises(LineageError, match="compaction proof"):
        ledger.admit(
            "run",
            "reuse-proof",
            [80],
            metadata("reset", 2, proof),
            POLICY,
            native_token_prompt=True,
        )


def test_measured_reset_cannot_skip_an_epoch():
    ledger = started()
    proof = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(LineageError, match="exactly one epoch"):
        ledger.admit(
            "run",
            "reset",
            [90, 91],
            metadata("reset", 2, proof),
            POLICY,
            native_token_prompt=True,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "prefix",
        "threshold",
        "oversized_proof",
        "oversized_reset",
        "epoch",
        "bool_epoch",
        "no_proof",
    ],
)
def test_invalid_eviction_proofs_are_rejected(damage):
    ledger = started()
    proof = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    prompt, epoch = [99], 1
    if damage == "prefix":
        proof[0] = 0
    if damage == "threshold":
        proof = proof[:8]
    if damage == "oversized_proof":
        proof += [0] * 20
    if damage == "oversized_reset":
        prompt = [99] * 4
    if damage == "epoch":
        epoch = 0
    if damage == "bool_epoch":
        epoch = True
    if damage == "no_proof":
        proof = None
    with pytest.raises(LineageError):
        ledger.admit(
            "run",
            "reset",
            prompt,
            metadata("reset", epoch, proof),
            POLICY,
            native_token_prompt=True,
        )


def test_same_nonce_fresh_nonce_chat_and_epoch_bypasses_fail():
    ledger = started()
    for nonce, prompt, info, native in [
        ("first", [1, 2], metadata(), False),
        ("new", [1, 2], metadata(), False),
        ("new", [1, 2, 3, 4, 5], metadata("continue"), False),
        ("new", [1, 2, 3, 4, 5], metadata("continue", 1), True),
    ]:
        with pytest.raises(LineageError):
            ledger.admit("run", nonce, prompt, info, POLICY, native_token_prompt=native)


def test_pending_draw_blocks_concurrency_and_requires_exact_served_echo():
    ledger = started()
    ledger.admit(
        "run",
        "second",
        [1, 2, 3, 4, 5],
        metadata("continue"),
        POLICY,
        native_token_prompt=True,
    )
    with pytest.raises(LineageError, match="concurrent"):
        ledger.admit(
            "run",
            "third",
            [1, 2, 3, 4, 5],
            metadata("continue"),
            POLICY,
            native_token_prompt=True,
        )
    with pytest.raises(LineageError, match="served prompt"):
        ledger.complete("run", "second", [88], [9])
    with pytest.raises(LineageError, match="unfinished"):
        ledger.drain()


def test_results_may_omit_only_one_terminal_deadline_draw():
    ledger = started()
    with pytest.raises(LineageError, match="unreported"):
        ledger.report("run", [])
    ledger.report("run", [], deadline_reached=True)
    with pytest.raises(LineageError, match="after the result"):
        ledger.admit(
            "run",
            "late",
            [1, 2, 3, 4, 5],
            metadata("continue"),
            POLICY,
            native_token_prompt=True,
        )
    ledger.drain()
    with pytest.raises(LineageError, match="closed run"):
        ledger.admit("run", "new", [1], metadata(), POLICY, native_token_prompt=True)


def test_results_cannot_omit_interior_or_multiple_generations():
    ledger = started()
    prompt = [1, 2, 3, 4, 5]
    ledger.admit(
        "run", "second", prompt, metadata("continue"), POLICY, native_token_prompt=True
    )
    ledger.complete("run", "second", prompt, [6])
    with pytest.raises(LineageError, match="interior"):
        ledger.report("run", ["second"], deadline_reached=True)
    with pytest.raises(LineageError, match="unreported"):
        ledger.report("run", [], deadline_reached=True)
    ledger.report("run", ["first", "second"])


@pytest.mark.parametrize(
    "metadata_value",
    [None, {}, metadata(epoch=True), {**metadata(), "extra": 1}, metadata(proof=[1])],
)
def test_missing_or_invalid_lineage_metadata_fails(metadata_value):
    with pytest.raises(LineageError):
        LineageLedger().admit(
            "r", "n", [1], metadata_value, POLICY, native_token_prompt=True
        )


@pytest.mark.parametrize("epoch", [1, 2, 4])
def test_served_over_trigger_proof_resets_at_any_later_epoch(epoch):
    ledger = started()
    prompt = list(range(1, 10))
    ledger.admit(
        "run",
        "summary",
        prompt,
        metadata("continue"),
        POLICY,
        native_token_prompt=True,
        summary_request=True,
    )
    ledger.complete("run", "summary", prompt, [10])
    ledger.admit(
        "run",
        "reset",
        [90, 91],
        metadata("reset", epoch),
        POLICY,
        native_token_prompt=True,
    )
    assert ledger.complete("run", "reset", [90, 91], [92]) == metadata("reset", epoch)
    with pytest.raises(LineageError, match="served over-trigger proof"):
        ledger.admit(
            "run",
            "stale",
            [80],
            metadata("reset", epoch + 1),
            POLICY,
            native_token_prompt=True,
        )


def test_consecutive_over_trigger_continuations_are_refused():
    ledger = started()
    prompt = list(range(1, 10))
    ledger.admit(
        "run",
        "summary",
        prompt,
        metadata("continue"),
        POLICY,
        native_token_prompt=True,
        summary_request=True,
    )
    ledger.complete("run", "summary", prompt, [10])
    with pytest.raises(LineageError, match="continuation after a summary"):
        ledger.admit(
            "run",
            "retry",
            prompt + [10, 11],
            metadata("continue"),
            POLICY,
            native_token_prompt=True,
        )


def test_served_proof_needs_no_media_but_cannot_carry_fresh_media():
    ledger = started()
    prompt = list(range(1, 10))
    ledger.admit(
        "run",
        "summary",
        prompt,
        metadata("continue"),
        POLICY,
        native_token_prompt=True,
        media={"images": ["witness"]},
        summary_request=True,
    )
    ledger.complete("run", "summary", prompt, [10])
    with pytest.raises(LineageError, match="no compaction media"):
        ledger.admit(
            "run",
            "reset",
            [90],
            {**metadata("reset", 2), "compaction_media": {}},
            POLICY,
            native_token_prompt=True,
        )
    ledger.admit(
        "run", "reset", [90], metadata("reset", 2), POLICY, native_token_prompt=True
    )
