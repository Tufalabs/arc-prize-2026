"""Token continuation: extend the served history by the tokens the model
sampled instead of re-rendering it (requirements U5, the nesting invariant).

The tokens a model sampled are the truth of what happened; re-deriving them
from parsed messages is lossy (tool-call whitespace is normalised on
re-serialisation, sampled tokenisations are not always canonical). A
continuation-capable backend therefore leaves a ``ServedTurn`` witness on
every processed answer and, on the next call, sends ``previous prompt ids
+ sampled ids + the tokens of what the template appends after the answer``
— tokenising only that delta, standalone, which is exact because the delta
starts at a special token (or the newline that follows one).

Backends share this module; what differs per backend is only where the ids
and the template come from. ``Conversation.history_epoch`` is the agent's
declaration that it rewrote sent history: a call whose epoch still matches
the last witness MUST continue, and a backend that cannot raises
``HistoryDriftError`` when the caller asked for exact history.

Injected text (``GenericSamplingParameters.response_injection``) is served
as tokens after the generation prompt: it sits in the witness's prompt ids
and, since the answer's text starts with it, in the template's render of the
answer — the witness carries it so the sampled text is matched with it.
"""

import dataclasses
import hashlib
from typing import Any, Callable, Optional

import ullm.conv

#: What a chat template places between an answer's terminator and the next
#: element: the newline the sampled ids do not include.
FRAMING = "\n"
THINK_OPEN = "<think>"


class HistoryDriftError(RuntimeError):
    """The served history could not be extended by tokens although the agent
    declared no rewrite — the backend would have to re-tokenise what the
    model already sampled, which the caller asked never to happen."""


@dataclasses.dataclass(frozen=True)
class ServedTurn:
    """Witness a backend leaves on a processed ``LLMResponse``.

    ``prefix_ids`` are the prompt ids this answer was sampled after that
    the previous witness did not already cover — the whole prompt when the
    history was rendered afresh, the delta when it was continued — so a
    conversation holds every served token exactly once."""

    epoch: int  # Conversation.history_epoch the prompt was built under
    prefix_ids: tuple[int, ...]
    output_ids: tuple[int, ...]  # sampled ids verbatim (terminator included when sampled)
    continued: bool  # prefix_ids extend the previous witness rather than replace it
    head_len: int  # length of the template's render of the history the answer was sampled after
    answered_len: int  # length of the template's render of history-through-this-answer
    answered_sha: str  # sha256 of that render
    ended_with_eos: bool
    # Fingerprints of the images in the prompt the answer was sampled after,
    # in message order: the render and the placeholder count cannot tell a
    # served image from one swapped in with the same size, but the engine
    # gets whatever pixels ride the request, so the served pixels are part
    # of what a continuation must reproduce.
    image_fingerprints: tuple[str, ...] = ()
    # Text served after the generation prompt that the answer's render
    # starts with but the sampled ids do not: the injection of this call.
    injected: str = ""


@dataclasses.dataclass(frozen=True)
class ContinuationPrompt:
    """The next call's id prompt: what was served, plus the tokenised delta."""

    prompt_ids: list[int]
    delta_ids: list[int]
    think_seeded: bool  # the generation prompt ends inside an opened think block


def _sha(text: str) -> str:
    """Content hash of a rendered history (what the witness compares against)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _responses(
    conversation: ullm.conv.Conversation, until: Optional[ullm.conv.LLMResponse] = None
) -> list[ullm.conv.LLMResponse]:
    """Processed answers the model was sent, in order — stopping before ``until``."""
    out: list[ullm.conv.LLMResponse] = []
    for element in conversation.elements:
        candidates = element.elements if isinstance(element, ullm.conv.ResponseTurn) else [element]
        for candidate in candidates:
            if candidate is until:
                return out
            if (
                isinstance(candidate, ullm.conv.LLMResponse)
                and candidate.status == "processed"
                and not candidate.do_not_send_to_agent
            ):
                out.append(candidate)
    return out


def served_history(
    conversation: ullm.conv.Conversation, until: Optional[ullm.conv.LLMResponse] = None
) -> Optional[tuple[list[int], ServedTurn]]:
    """The ids the model has been served so far under the current epoch —
    every witness back to the last fresh render, concatenated — with the
    last witness; None when the last answer left none or a rewrite has
    happened since (``history_epoch`` moved on). ``until`` excludes an
    answer being processed right now."""
    responses = _responses(conversation, until)
    witnesses: list[ServedTurn] = []
    for response in reversed(responses):
        witness = response._served
        if not isinstance(witness, ServedTurn) or witness.epoch != conversation.history_epoch:
            return None
        witnesses.append(witness)
        if not witness.continued:
            break
    else:
        return None  # no fresh base under this epoch: nothing served yet
    if not witnesses:
        return None
    ids: list[int] = []
    for witness in reversed(witnesses):
        ids.extend(witness.prefix_ids)
        ids.extend(witness.output_ids)
    return ids, witnesses[0]


def continue_prompt(
    tokenizer: Any,
    served: list[int],
    last: ServedTurn,
    new_render: str,
    *,
    encode_delta: Optional[Callable[[str], list[int]]] = None,
) -> Optional[ContinuationPrompt]:
    """``served`` + the tokens of what ``new_render`` (the template's render
    of the whole history with the generation prompt) adds after the last
    answer — or None when the history before the answer no longer renders
    the same, or the delta does not tokenise standalone as it would inside
    the full render.

    ``encode_delta`` replaces the tokenizer for the delta when the text
    alone cannot produce the served ids — a delta carrying an image
    placeholder the engine's processor expands per image. Its ids are then
    taken as the engine would compute them (the decode self-check does not
    apply: an expanded placeholder does not decode to its text); the
    engine's echo of the sent ids is the witness on that path."""
    if len(new_render) < last.answered_len or _sha(new_render[: last.answered_len]) != last.answered_sha:
        return None
    eos = str(tokenizer.eos_token or "")
    sampled = last.injected + str(tokenizer.decode(list(last.output_ids)))
    rendered_answer = new_render[last.head_len : last.answered_len]
    # Parsers and templates trim the answer's whitespace at both ends before
    # rendering it; the sampled whitespace is nonetheless in the served ids,
    # so the match is made without it and the close follows the ids as
    # sampled.
    core = sampled.strip()
    if rendered_answer.startswith(core):
        # The template rendered the answer as sampled, then whatever closes
        # it: the framing newline for a complete answer, or the structure a
        # budget-stopped answer never reached (a closing think tag, the
        # terminator). All of that is delta.
        delta = new_render[last.head_len + len(core) :]
    else:
        # The template re-serialised the answer (normalised whitespace): the
        # answered render still ends with eos + FRAMING, and the sampled ids
        # carry the eos (unless the budget stopped them) but never the
        # framing newline.
        delta = new_render[last.answered_len - len(FRAMING) :]
        if not last.ended_with_eos:
            delta = eos + delta
    if encode_delta is not None:
        delta_ids = [int(t) for t in encode_delta(delta)]
    else:
        delta_ids = [int(t) for t in tokenizer.encode(delta, add_special_tokens=False)]
        if tokenizer.decode(delta_ids) != delta:
            return None
    return ContinuationPrompt(
        prompt_ids=[*served, *delta_ids],
        delta_ids=delta_ids,
        think_seeded=new_render.endswith(THINK_OPEN + FRAMING),
    )


def witness(
    tokenizer: Any,
    *,
    epoch: int,
    prompt_ids: list[int],
    output_ids: list[int],
    served_before: Optional[list[int]],
    head_render: str,
    answered_render: str,
    image_fingerprints: tuple[str, ...] = (),
    injected: str = "",
) -> Optional[ServedTurn]:
    """The witness for an answer sampled after ``prompt_ids``: None when the
    template does not render the answer as a pure suffix of the prompt it
    was sampled after (then the next call cannot locate the delta)."""
    if not answered_render.startswith(head_render):
        return None
    eos = str(tokenizer.eos_token or "")
    if not eos or not answered_render.endswith(eos + FRAMING):
        return None
    continued = (
        served_before is not None
        and len(prompt_ids) >= len(served_before)
        and prompt_ids[: len(served_before)] == served_before
    )
    prefix = prompt_ids[len(served_before) :] if continued and served_before is not None else prompt_ids
    ended_with_eos = bool(output_ids) and int(output_ids[-1]) == int(tokenizer.eos_token_id)
    return ServedTurn(
        epoch=epoch,
        prefix_ids=tuple(int(t) for t in prefix),
        output_ids=tuple(int(t) for t in output_ids),
        continued=continued,
        head_len=len(head_render),
        answered_len=len(answered_render),
        answered_sha=_sha(answered_render),
        ended_with_eos=ended_with_eos,
        image_fingerprints=image_fingerprints,
        injected=injected,
    )


def has_image(messages: list[dict[str, Any]]) -> bool:
    """Whether any message carries an image part (a delta a text tokenizer cannot stand in for)."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("image", "image_url") for part in content
        ):
            return True
    return False
