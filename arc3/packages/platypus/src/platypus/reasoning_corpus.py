"""What the model wrote while thinking, read back out of a game's diagnostics.

Every ``llm/turn`` span closes with a ``conversation`` artifact -- the
surviving prompt, in capture's canonical serialization with large text held
in the block store -- and, when compaction removed rounds before the turn
closed, a ``discarded_rounds`` artifact holding what left. Reasoning lives
in both: a round's ``reasoning`` content is in whichever of the two still
has the round. Reading only the surviving conversation loses every round
that compaction evicted, which in a long game is most of them.

A move's conversation carries the whole history, so the same round appears
in many artifacts; texts are deduplicated within a game and returned in
generation order. Neither artifact carries that order on its own: a round
compaction evicted is older than everything the turn kept, a response a
lifted per-turn cap discarded sits between kept ones, and the span stores
both kinds after the surviving conversation. What does carry it is the
span's ``llm/response`` events -- one per response in the order they were
generated, each saying whether it was discarded -- so a turn's kept and
discarded responses are merged along those. Keyed on the artifact and
event names, not the span names, so any agent that runs through
``DiagnosticTurnManager`` is read the same way.

The scorer at the bottom is the register classifier applied to that text:
one number per game, 1 for the blanked register and 0 for control, from
``ullm.reasoning_register``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Union, cast

import ullm.reasoning_register

from platypus.diagnostics import (
    BlockStore,
    SpanNode,
    build_span_tree,
    game_run_ids,
    job_block_store,
    read_game_events,
    resolve_blocks,
)

#: Artifacts a turn's reasoning can be found in: the surviving prompt, and
#: the rounds that left it before the turn closed.
REASONING_ARTIFACTS = ("conversation", "discarded_rounds")


def _spans(root: SpanNode) -> Iterator[SpanNode]:
    """Every span under and including ``root``, depth first."""
    yield root
    for child in root.children:
        yield from _spans(child)


def _responses(node: Any) -> list[dict[str, Any]]:
    """Every response node inside a resolved capture structure, in document order.

    Walks the structure rather than a known layout: ``conversation`` holds
    turns holding rounds, ``discarded_rounds`` holds a ``conversation``
    under ``records``' metadata, and both nest the same round shape.
    """
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        mapping = cast("dict[str, Any]", node)
        if mapping.get("kind") == "response":
            found.append(mapping)
        else:
            for value in mapping.values():
                found.extend(_responses(value))
    elif isinstance(node, list):
        for value in cast("list[Any]", node):
            found.extend(_responses(value))
    return found


def _reasoning_of(response: dict[str, Any], strip_injected: bool) -> str:
    """A response's reasoning text, "" when it has none.

    ``strip_injected`` drops the served prefix (``response_injection``),
    leaving only what the model itself wrote.
    """
    text = _content_text(response.get("reasoning"))
    if strip_injected:
        injected = response.get("response_injection")
        if isinstance(injected, str) and injected and text.startswith(injected.strip()):
            text = text[len(injected.strip()) :].strip()
    return text


def _content_text(content: Any) -> str:
    """The text of a resolved capture content node: a string, or a list of them joined."""
    if not isinstance(content, dict):
        return ""
    mapping = cast("dict[str, Any]", content)
    if mapping.get("c") == "str":
        return str(mapping.get("text") or "").strip()
    if mapping.get("c") == "list":
        return "\n".join(t for t in (_content_text(e) for e in cast("list[Any]", mapping.get("elements", []))) if t)
    return ""


def _artifact(span: SpanNode, store: BlockStore, name: str) -> Any:
    """The span's named JSON artifact, resolved; None when absent or its block is gone."""
    for artifact in span.artifacts:
        if artifact.get("name") == name and artifact.get("codec") == "json":
            try:
                return resolve_blocks(store, store.load(artifact["block"], "json"))
            except Exception:  # noqa: BLE001 - one lost block must not hide the rest of the game
                return None
    return None


def _turn_reasoning(span: SpanNode, store: BlockStore, strip_injected: bool) -> Iterator[str]:
    """One span's reasoning texts in generation order, empty ones included.

    Emitted with the empties so positions line up with the events; the
    caller drops them. The history before this turn comes first (normally
    all seen already), then this turn's kept and discarded responses merged
    along the ``llm/response`` events, then discarded rounds this turn did
    not generate -- older rounds compaction took out of the history, which
    an earlier span already carried.
    """
    conversation = _artifact(span, store, "conversation")
    discarded = _artifact(span, store, "discarded_rounds")
    elements = cast("list[Any]", conversation.get("elements", [])) if isinstance(conversation, dict) else []
    turns = [
        i for i, e in enumerate(elements) if isinstance(e, dict) and cast("dict[str, Any]", e).get("kind") == "turn"
    ]
    current = turns[-1] if turns else len(elements)
    earlier = [_reasoning_of(r, strip_injected) for r in _responses(elements[:current])]
    kept = [_reasoning_of(r, strip_injected) for r in _responses(elements[current:])]
    gone = [
        _reasoning_of(r, strip_injected)
        for r in _responses(
            cast("dict[str, Any]", discarded).get("conversation") if isinstance(discarded, dict) else None
        )
    ]
    events = [cast("dict[str, Any]", e.get("values") or {}) for e in span.events if e.get("kind") == "llm/response"]
    generated_here = sum(1 for v in events if v.get("discarded_reason"))
    yield from earlier
    if events and len(events) == len(kept) + generated_here and generated_here <= len(gone):
        kept_iter, gone_iter = iter(kept), iter(gone)
        for values in events:
            yield next(gone_iter) if values.get("discarded_reason") else next(kept_iter)
        yield from gone_iter
    else:
        # The bookkeeping does not line up (a span closed without its events,
        # a block gone): what the artifacts alone can say, the discarded
        # rounds first since compaction takes the oldest.
        yield from gone
        yield from kept


def _reasoning_texts(game_dir: Path, store: BlockStore, strip_injected: bool = False) -> Iterator[str]:
    """Distinct reasoning texts, in generation order, lazily.

    Generated rather than returned as a list so a caller that wants only the
    opening pays for the first turn's artifacts instead of the game's several
    hundred.
    """
    # A game dir is ``<job_dir>/games/<run_id>``, whether or not its stream is
    # still loose there or has been archived.
    events = read_game_events(game_dir.parent.parent, game_dir.name)
    seen: set[str] = set()
    for root in build_span_tree(events):
        for span in _spans(root):
            if not any(a.get("name") in REASONING_ARTIFACTS for a in span.artifacts):
                continue
            for text in _turn_reasoning(span, store, strip_injected):
                if text and text not in seen:
                    seen.add(text)
                    yield text


def game_reasoning(
    game_dir: Union[str, Path], blocks: BlockStore | None = None, strip_injected: bool = False
) -> list[str]:
    """Every distinct reasoning text in one game, in order of first appearance.

    ``game_dir`` is a ``games/<run_id>/`` directory; its block store is the
    job's ``diag_blocks/`` two levels up unless one is passed.
    ``strip_injected`` leaves only what the model wrote, dropping the text
    the harness served it.
    """
    game_dir = Path(game_dir)
    store = blocks or job_block_store(game_dir.parent.parent)
    return list(_reasoning_texts(game_dir, store, strip_injected))


def first_reasoning(game_dir: Union[str, Path], blocks: BlockStore | None = None, strip_injected: bool = False) -> str:
    """The game's OPENING reasoning text, or "" if it recorded none.

    The register is readable from this alone with the opening model, and the
    opening is where the two registers are furthest apart -- so this is the
    cheap read: the first turn's artifacts, not the whole game's.
    """
    game_dir = Path(game_dir)
    store = blocks or job_block_store(game_dir.parent.parent)
    return next(_reasoning_texts(game_dir, store, strip_injected), "")


def run_reasoning(job_dir: Union[str, Path]) -> dict[str, list[str]]:
    """``game_reasoning`` for every game of a job, keyed by the game's run id."""
    job_dir = Path(job_dir)
    store = job_block_store(job_dir)
    return {run_id: game_reasoning(job_dir / "games" / run_id, store) for run_id in game_run_ids(job_dir)}


def game_register(game_dir: Union[str, Path], blocks: BlockStore | None = None) -> float:
    """The game's reasoning scored as one text: 1 is the blanked register, 0 control.

    0.5 for a game with no reasoning recorded, which is what the classifier
    returns for empty input. Served text is stripped before scoring: the
    shipped duck opens every response with register words of our choosing,
    and counting those would read our own prompt back as the model's voice.
    """
    return ullm.reasoning_register.blanked_score(game_reasoning(game_dir, blocks, strip_injected=True))


def game_opening_register(game_dir: Union[str, Path], blocks: BlockStore | None = None) -> float:
    """The game's OPENING reasoning scored: 1 is the blanked register, 0 control.

    One block read rather than the whole game, and scored with the model
    fitted on openings. Served text is stripped first, as in
    :func:`game_register`.
    """
    return ullm.reasoning_register.opening_blanked_score(first_reasoning(game_dir, blocks, strip_injected=True))
