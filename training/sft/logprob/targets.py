"""Prepare sparse API-teacher targets without changing Flash's tokenizer.

Reasoning text, final content and tool-call argument values are aligned at
byte boundaries shared by both tokenizers. The teacher's own tool-call markup
and Flash's chat-template wrapper are never aligned to each other: the student
is supervised on the same semantic completion (same reasoning, same answer,
same tool call with the same argument values) rendered through Flash's
template. A teacher token longer than the student token at its boundary is
attached to that student token (the rest of its bytes stay CE-only), teacher
tokens that together spell one student token are merged along the sampled
path, and a teacher alternative that is no student token is projected onto
the first token of Flash's own BPE spelling of it; mass that cannot be placed
stays in the residual bucket. Missing token bytes and teacher tokens that
cross a student token's end are excluded. The API distribution is used as
returned, without temperature reconstruction.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

SCHEMA = "flash-api-teacher-topk-v1"
SCOPE = "reasoning, content, tool names, parameter keys and argument values at shared byte boundaries; a teacher token longer than the student token there is attached to it, teacher tokens that together spell the student token are merged along the sampled path; alternatives without a student token are projected onto the first token of Flash's BPE spelling; the teacher's control-token decisions (end of reasoning, of content, of a value; another argument or close; another call or end) are attached to the Flash token that begins each branch, as are control-token alternatives at text positions; the rest of both templates' wrappers CE-only; remaining mass stays in the other bucket"
FIELD = "teacher_logprob_targets"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ByteVocabulary:
    """Recover raw bytes, including partial UTF-8 tokens, from ByteLevel BPE."""

    def __init__(self, path):
        config = json.loads(Path(path).read_text())
        if config["decoder"]["type"] != "ByteLevel":
            raise ValueError("prototype requires a ByteLevel tokenizer")
        base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = base[:]
        extra = 0
        for byte in range(256):
            if byte not in base:
                base.append(byte)
                chars.append(256 + extra)
                extra += 1
        inverse = {chr(char): byte for byte, char in zip(base, chars, strict=True)}
        self.by_id = {
            token_id: bytes(inverse[c] for c in token)
            for token, token_id in config["model"]["vocab"].items()
        }
        self.by_bytes = {value: key for key, value in self.by_id.items()}
        if len(self.by_bytes) != len(self.by_id):
            raise ValueError("ambiguous byte vocabulary")
        self.added = {}
        for token in config.get("added_tokens", []):
            self.by_id[token["id"]] = token["content"].encode()
            self.added[token["content"].encode()] = token["id"]
            # Added/control tokens are never API alternative targets.
            self.by_bytes.pop(token["content"].encode(), None)
        self.ranks = {}
        for rank, merge in enumerate(config["model"].get("merges") or []):
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            pair = tuple(bytes(inverse[c] for c in part) for part in (left, right))
            if pair in self.ranks:
                raise ValueError("duplicate BPE merge")
            self.ranks[pair] = rank
        self.sha256 = file_sha(path)
        self._first = {}

    def token_id(self, data):
        """The id of a whole token, added tokens included."""
        found = self.by_bytes.get(data)
        return self.added.get(data) if found is None else found

    def leading(self, literal):
        """The token Flash emits first when it writes ``literal``: the literal
        itself when it is one token, else the first token of the BPE spelling
        of what precedes the first added token in it."""
        found = self.token_id(literal)
        if found is not None:
            return found
        cut = min(
            (literal.index(added) for added in self.added if added in literal),
            default=len(literal),
        )
        if cut == 0:
            return next(i for a, i in self.added.items() if literal.startswith(a))
        return self.first_token(literal[:cut])

    def first_token(self, data):
        """The first token of this tokenizer's BPE spelling of ``data`` on its own.

        Byte-level BPE as the ``tokenizers`` model applies it to one
        pre-tokenized chunk: merge the lowest-ranked adjacent pair until none
        applies. Returns None when the spelling starts with no vocabulary token.
        """
        cached = self._first.get(data)
        if cached is not None or data in self._first:
            return cached
        word = [bytes([b]) for b in data]
        while len(word) > 1:
            best = None
            for pair in pairwise(word):
                rank = self.ranks.get(pair)
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, pair)
            if best is None:
                break
            left, right = best[1]
            merged, index = [], 0
            while index < len(word):
                if word[index : index + 2] == [left, right]:
                    merged.append(left + right)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = merged
        self._first[data] = result = self.by_bytes.get(word[0])
        return result

    def render(self, ids):
        pieces = [self.by_id[i] for i in ids]
        offsets = {}
        cursor = 0
        for index, piece in enumerate(pieces):
            offsets[cursor] = (index, len(piece))
            cursor += len(piece)
        return b"".join(pieces), offsets


def token_bytes(item):
    value = item.get("bytes")
    if value is None:
        return None
    if not isinstance(value, list) or any(
        type(b) is not int or not 0 <= b < 256 for b in value
    ):
        raise ValueError("invalid teacher token bytes")
    return bytes(value) or None


def logprob(item):
    value = item.get("logprob")
    if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-6:
        raise ValueError("invalid teacher logprob")
    return min(float(value), 0.0)


def reasoning_tokens(step):
    """Require an exact contiguous byte prefix; never decode/re-encode the teacher."""
    text = (step["output_message"].get("reasoning_content") or "").encode()
    if not text:
        return text, []
    tokens = (step.get("extra") or {}).get("teacher_logprobs")
    if not tokens:
        raise ValueError("reasoning has no teacher_logprobs")
    pieces, selected, size = [], [], 0
    for item in tokens:
        data = token_bytes(item)
        if data is None:
            break
        pieces.append(data)
        selected.append(item)
        size += len(data)
        if size >= len(text):
            break
    if b"".join(pieces) != text:
        raise ValueError("teacher token bytes do not exactly reproduce reasoning")
    return text, selected


def projectable(data):
    """Whole characters only: a partial UTF-8 sequence is a fragment of a
    longer student token, not bytes the student would start with its own."""
    try:
        data.decode()
    except UnicodeDecodeError:
        return False
    return True


def unique_alternatives(item):
    """Observed alternatives at one teacher position as {bytes: logprob}.

    The provider sometimes appends the sampled token outside the top-k;
    it is kept once, with its probability rather than added twice.
    """
    unique = {}
    for alternative in list(item.get("top_logprobs") or []) + [item]:
        lp = logprob(alternative)
        if lp <= -999:  # API's -9999 placeholder is not an observed alternative.
            continue
        data = token_bytes(alternative)
        if data is None:
            continue
        if data in unique and abs(unique[data] - lp) > 1e-5:
            raise ValueError("conflicting duplicate teacher alternatives")
        unique[data] = lp
    if sum(math.exp(lp) for lp in unique.values()) > 1 + 1e-5:
        raise ValueError("teacher alternatives have probability mass above one")
    return unique


def special_mass(item, *, sampled_logprob=None):
    """Mass and count of the byte-less (control token) alternatives at a position.

    A sampled control token also appears among its own alternatives; the one
    entry at ``sampled_logprob`` is left out.
    """
    mass, count, skipped = 0.0, 0, sampled_logprob is None
    for alternative in item.get("top_logprobs") or []:
        lp = logprob(alternative)
        if lp <= -999 or token_bytes(alternative) is not None:
            continue
        if not skipped and abs(lp - sampled_logprob) < 1e-5:
            skipped = True
            continue
        mass += math.exp(lp)
        count += 1
    return mass, count


def place(mass, data, probability, vocab, counts, *, sampled_id=None, sampled=None):
    """Add teacher mass for the byte string ``data`` to the student token it maps to.

    The sampled teacher token maps to ``sampled_id`` when given (the student
    token at its boundary, a prefix of it), before any vocabulary lookup: Flash
    spelled those bytes with that token in context, so its mass belongs there
    even when the whole byte string is a token too (``a`` + ``b`` rendered
    where ``ab`` exists). Other exact bytes map to their token; anything else
    is projected onto the first token of Flash's BPE spelling of its bytes.
    What maps nowhere belongs to the residual bucket.
    """
    if sampled_id is not None and data == sampled:
        token = sampled_id
    else:
        token = vocab.by_bytes.get(data)
    if token is None and projectable(data):
        token = vocab.first_token(data)
        if token is not None and counts is not None:
            counts["projected_alternatives"] += 1
            counts["projected_mass"] += probability
    if token is not None:
        mass[token] = mass.get(token, 0.0) + probability


def finish(mass):
    if not mass:
        return [], []
    ids = sorted(mass)
    probabilities = [mass[i] for i in ids]
    # Correct only floating point overshoot, never renormalize a truncated top-k.
    total = sum(probabilities)
    if total > 1:
        probabilities = [p / total for p in probabilities]
    return ids, probabilities


def stop_here(mass, stop, probability, count, counts):
    """Control-token alternatives mean "stop this region here": their mass goes
    to the Flash token that begins the wrapper, ``stop``, when it is known."""
    if stop is None or probability <= 0:
        return
    mass[stop] = mass.get(stop, 0.0) + probability
    if counts is not None:
        counts["special_alternatives"] += count
        counts["special_alternative_mass"] += probability


def mapped_distribution(item, vocab, counts=None, *, sampled_id=None, stop=None):
    """Teacher alternatives at one position as student ids; mass on one id adds up."""
    mass = {}
    sampled = token_bytes(item)
    for data, lp in unique_alternatives(item).items():
        probability = math.exp(lp)
        if probability > 0:
            place(
                mass,
                data,
                probability,
                vocab,
                counts,
                sampled_id=sampled_id,
                sampled=sampled,
            )
    stop_here(mass, stop, *special_mass(item), counts)
    return finish(mass)


def merged_distribution(path, sampled_id, vocab, counts=None, *, stop=None):
    """Several teacher tokens spell the one student token ``sampled_id``.

    The sampled path carries the product of its probabilities to that token.
    Each observed alternative along the path carries the product of the
    sampled probabilities before it times its own, as the bytes it would have
    spelled from the student token's start; a later alternative has no second
    step of its own, so only the sampled path is followed past it.
    """
    mass = {}
    prefix, prefix_probability = b"", 1.0
    for item in path:
        sampled = token_bytes(item)
        for data, lp in unique_alternatives(item).items():
            probability = prefix_probability * math.exp(lp)
            if data != sampled and probability > 0:
                place(mass, prefix + data, probability, vocab, counts)
        special, count = special_mass(item)
        if prefix:
            # Stopping after the bytes spelled so far: the student would
            # write them as a token of their own, then the wrapper.
            if special > 0:
                place(mass, prefix, prefix_probability * special, vocab, counts)
        else:
            stop_here(mass, stop, special, count, counts)
        prefix += sampled
        prefix_probability *= math.exp(logprob(item))
    if prefix != vocab.by_id[sampled_id]:
        raise ValueError("teacher path does not spell the student token")
    if prefix_probability > 0:
        mass[sampled_id] = mass.get(sampled_id, 0.0) + prefix_probability
    return finish(mass)


THINK_CLOSE = b"\n</think>\n\n"
TOOL_CALL_OPEN = b"<tool_call>\n<function="
TOOL_CALL_CLOSE = b"</function>\n</tool_call>"
PARAMETER_OPEN = b"<parameter="
PARAMETER_CLOSE = b"\n</parameter>\n"
IM_END = b"<|im_end|>"
COMPLETION_TAILS = (b"", IM_END, IM_END + b"\n")
CALL_TAG = re.compile(rb'^call tool="([^"]*)"(?: index="[^"]*")?$')
ARGUMENT_TAG = re.compile(rb'^argument key="([^"]*)" type="([^"]*)"$')
CALL_NAME_OFFSET = len(b'call tool="')
ARGUMENT_KEY_OFFSET = len(b'argument key="')


def is_special(item):
    """Provider control tokens carry no bytes; they delimit the teacher markup."""
    return item.get("bytes") is None


def run_bytes(items):
    return b"".join(token_bytes(item) or b"" for item in items)


def teacher_markup(items):
    """Parse the teacher's post-reasoning stream.

    Kimi K3.1 closes the reasoning and wraps the answer and the tool calls in
    tags whose brackets are control tokens and whose names are text::

        </think><response>CONTENT</response><tools>
          <call tool="NAME" index="N"><argument key="KEY" type="TYPE">VALUE</argument>...</call>...
        </tools>

    A final answer ends inside ``<response>``. Returns ``None`` when the stream
    stops before any tag, otherwise ``{"content": items, "calls": [...]}`` with
    the token items of the content and of every argument value, plus the tag
    items that carry each tool name and parameter key. Anything else raises.
    """
    runs, current, specials = [], [], []
    for item in items:
        if is_special(item):
            runs.append(current)
            specials.append(item)
            current = []
        else:
            current.append(item)
    runs.append(current)
    if runs[0]:
        raise ValueError(
            "teacher text continues past the reasoning without a delimiter"
        )
    tags, bodies = runs[1::2], runs[2::2]
    if not any(tags) and not any(bodies):
        return None
    pairs = list(zip(tags, bodies + [[]] * (len(tags) - len(bodies)), strict=True))

    def expect(index, name):
        if index >= len(pairs) or run_bytes(pairs[index][0]) != name:
            raise ValueError(f"unexpected teacher markup: expected {name!r} tag")
        return pairs[index][1]

    def opening(index):
        """The control token that opens tag ``index``: the teacher's decision
        to write that tag after what came before."""
        return specials[2 * index] if 2 * index < len(specials) else None

    if expect(0, b"think"):
        raise ValueError("unexpected teacher text between </think> and <response>")
    content = expect(1, b"response")
    parsed = {
        "content": content,
        "calls": [],
        "think_close": opening(0),
        "content_close": None,
    }
    if len(pairs) == 2:
        return parsed
    if expect(2, b"response") or expect(3, b"tools"):
        raise ValueError("unexpected teacher text outside <response> and <call>")
    parsed["content_close"] = opening(2)
    index = 4
    while index < len(pairs):
        tag, body = pairs[index]
        name = run_bytes(tag)
        if name == b"tools":
            if body or index + 1 != len(pairs):
                raise ValueError("unexpected teacher markup after </tools>")
            break
        match = CALL_TAG.match(name)
        if match is None or body:
            raise ValueError("unexpected teacher markup: expected <call> tag")
        call = {"name": match.group(1), "tag": tag, "arguments": []}
        index += 1
        while True:
            if index >= len(pairs):
                raise ValueError("teacher stream ends inside a tool call")
            tag, body = pairs[index]
            name = run_bytes(tag)
            if name == b"call":
                if body:
                    raise ValueError("unexpected teacher text after </call>")
                call["close"] = opening(index)
                index += 1
                # Another call, the end of the calls, or a stream that stops.
                if index < len(pairs):
                    call["next"] = opening(index)
                    call["next_is_end"] = run_bytes(pairs[index][0]) == b"tools"
                else:
                    call["next"] = None
                    call["next_is_end"] = False
                break
            match = ARGUMENT_TAG.match(name)
            if match is None:
                raise ValueError("unexpected teacher markup: expected <argument> tag")
            if expect(index + 1, b"argument"):
                raise ValueError("unexpected teacher text after </argument>")
            call["arguments"].append(
                {
                    "key": match.group(1),
                    "type": match.group(2),
                    "tag": tag,
                    "value": body,
                    "open": opening(index),
                    "close": opening(index + 1),
                }
            )
            index += 2
        parsed["calls"].append(call)
    return parsed


def tool_arguments(call):
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments else {}
    if not isinstance(arguments, dict):
        raise TypeError("tool-call arguments must be a JSON object")
    return str(function.get("name") or "").encode(), arguments


class Walk:
    """Cursor over Flash's rendered completion bytes; every literal is verified."""

    def __init__(self, rendered, cursor):
        self.rendered = rendered
        self.cursor = cursor

    def expect(self, literal):
        if not self.rendered.startswith(literal, self.cursor):
            found = self.rendered[self.cursor : self.cursor + len(literal) + 16]
            raise ValueError(
                f"Flash rendering mismatch: expected {literal!r}, found {found!r}"
            )
        start = self.cursor
        self.cursor += len(literal)
        return start

    def until(self, literal):
        end = self.rendered.find(literal, self.cursor)
        if end < 0:
            raise ValueError(f"Flash rendering mismatch: no {literal!r}")
        start, self.cursor = self.cursor, end
        return start, end


class Aligner:
    """Attach teacher items to Flash positions at shared byte boundaries."""

    def __init__(self, sequence, prompt, boundaries, vocab, events, counts):
        self.sequence = sequence
        self.prompt = prompt
        self.boundaries = boundaries
        self.vocab = vocab
        self.events = events
        self.counts = counts

    def record(self, position, sampled, ids, probabilities, kind):
        self.events.append(
            {
                "position": position,
                "sampled_id": sampled,
                "ids": ids,
                "probabilities": probabilities,
            }
        )
        self.counts["aligned_positions"] += 1
        self.counts[f"{kind}_positions"] += 1
        self.counts["mapped_alternatives"] += len(ids)
        self.counts["mapped_mass"] += sum(probabilities)

    @staticmethod
    def spelling(items, start, length):
        """The teacher items from ``start`` whose bytes spell exactly ``length``
        bytes, or None when a control token or a crossing token intervenes."""
        path, size = [], 0
        for item in items[start:]:
            data = token_bytes(item)
            if data is None:
                return None
            path.append(item)
            size += len(data)
            if size >= length:
                return path if size == length else None
        return None

    def decide(self, offset, item, taken, other, kind):
        """A teacher decision made by a control token, at byte ``offset`` of the
        rendering where Flash's wrapper ``taken`` begins.

        The Flash token there carries the control token's probability. Other
        control tokens among the alternatives are the other branch: the token
        beginning the wrapper ``other``, or, when the other branch is text
        (``other`` None), the same token, since any control token there means
        stopping. Text alternatives are projected as anywhere else.
        """
        if item is None:
            return
        boundary = self.boundaries.get(offset)
        if boundary is None:
            self.counts["decisions_unplaced"] += 1
            return
        position = self.prompt + boundary[0]
        sampled = self.sequence[position]
        if not taken.startswith(self.vocab.by_id[sampled]):
            self.counts["decisions_unplaced"] += 1
            return
        other_id = sampled if other is None else self.vocab.leading(other)
        lp = logprob(item)
        mass = {sampled: math.exp(lp)}
        special, _ = special_mass(item, sampled_logprob=lp)
        if special > 0:
            if other_id is None:
                self.counts["decisions_other_branch_unplaced"] += 1
            else:
                mass[other_id] = mass.get(other_id, 0.0) + special
        for alternative in item.get("top_logprobs") or []:
            alp = logprob(alternative)
            data = token_bytes(alternative)
            if alp > -999 and data is not None:
                place(mass, data, math.exp(alp), self.vocab, self.counts)
        ids, probabilities = finish(mass)
        self.record(position, sampled, ids, probabilities, kind)
        self.counts["decision_positions"] += 1

    def align(
        self, items, offset, lo, hi, kind, *, count_tokens=True, stop=None, outside=None
    ):
        """``items`` render from byte ``offset``; only tokens inside [lo, hi) align.

        At a shared boundary a teacher token as long as the student token maps
        one-to-one; a longer one is attached to the student token (the student
        tokens that spell its rest stay CE-only); a shorter one is merged with
        the teacher tokens that spell the student token. Tag runs (tool names,
        parameter keys) are not counted as teacher tokens of their kind: they
        remain part of ``markup_teacher_tokens``.
        """
        cursor, index = offset, 0
        while index < len(items):
            item = items[index]
            index += 1
            data = token_bytes(item)
            if count_tokens:
                self.counts[f"{kind}_teacher_tokens"] += 1
            if data is None:
                continue
            boundary = self.boundaries.get(cursor)
            inside = lo <= cursor and cursor + len(data) <= hi
            if not inside:
                if count_tokens:
                    self.counts[outside or f"{kind}_teacher_tokens_outside"] += 1
            elif boundary is None:
                if count_tokens:
                    self.counts[f"{kind}_teacher_tokens_no_boundary"] += 1
            elif boundary[1] > len(data):
                path = self.spelling(items, index - 1, boundary[1])
                if path is not None and cursor + boundary[1] <= hi:
                    position = self.prompt + boundary[0]
                    sampled = self.sequence[position]
                    ids, probabilities = merged_distribution(
                        path, sampled, self.vocab, self.counts, stop=stop
                    )
                    if ids:
                        self.record(position, sampled, ids, probabilities, kind)
                        self.counts["merged_positions"] += 1
                        self.counts[f"{kind}_merged_positions"] += 1
                        self.counts["merged_teacher_tokens"] += len(path)
                    if count_tokens:
                        self.counts[f"{kind}_teacher_tokens"] += len(path) - 1
                    index += len(path) - 1
                    cursor += boundary[1]
                    continue
                if count_tokens:
                    self.counts[f"{kind}_teacher_tokens_shorter_than_student"] += 1
            else:
                position = self.prompt + boundary[0]
                sampled = self.sequence[position]
                if self.vocab.by_id[sampled] != data[: boundary[1]]:
                    raise ValueError(
                        "student token is not a prefix of the teacher token"
                    )
                exact = boundary[1] == len(data)
                ids, probabilities = mapped_distribution(
                    item,
                    self.vocab,
                    self.counts,
                    sampled_id=None if exact else sampled,
                    stop=stop,
                )
                if ids:
                    self.record(position, sampled, ids, probabilities, kind)
                    if not exact:
                        self.counts["prefix_positions"] += 1
                        self.counts[f"{kind}_prefix_positions"] += 1
                        # How often the standalone projection would have picked
                        # another token than Flash did in context.
                        if self.vocab.first_token(data) != sampled:
                            self.counts["prefix_positions_projection_differs"] += 1
            cursor += len(data)


def align_step(step, rendered, boundaries, aligner):
    """Align one teacher step to its Flash completion rendering.

    Flash's template renders ``<think>\\nREASONING\\n</think>\\n\\nCONTENT`` and then
    each call as ``<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n
    ...</function>\\n</tool_call>``; string values verbatim, others as JSON.
    Only the teacher's reasoning, content, tool names, parameter keys and
    argument values are aligned; both templates' wrappers stay CE-only.
    """
    counts = aligner.counts
    message = step["output_message"]
    text, reasoning = reasoning_tokens(step)
    tokens = (step.get("extra") or {}).get("teacher_logprobs") or []
    markup = teacher_markup(tokens[len(reasoning) :])
    counts["markup_teacher_tokens"] += len(tokens) - len(reasoning)
    vocab = aligner.vocab
    # The chat converter strips the reasoning at its edges (str.strip), so Flash
    # renders only the trimmed bytes. Teacher tokens are never split or
    # re-encoded: those not entirely inside the surviving window stay CE-only.
    original = message.get("reasoning_content") or ""
    trimmed = original.strip().encode()
    left = len(text) - len(original.lstrip().encode())
    if text and not trimmed:
        counts["whitespace_only_reasoning_steps"] += 1
    elif trimmed != text:
        counts["trimmed_reasoning_steps"] += 1
    if trimmed:
        # The reasoning starts the completion; prefer that anchored match over
        # the same text occurring again later (e.g. quoted inside a tool call).
        start = 0 if rendered.startswith(trimmed) else rendered.find(trimmed)
        if start < 0 or (start > 0 and rendered.find(trimmed, start + 1) >= 0):
            raise ValueError(
                "reasoning must occur exactly once in its converted completion"
            )
        aligner.align(
            reasoning,
            start - left,
            start,
            start + len(trimmed),
            "reasoning",
            stop=vocab.leading(THINK_CLOSE),
            outside="trimmed_teacher_tokens",
        )
    else:
        start = 0
    if markup is None:
        counts["steps_without_markup"] += 1
        return
    walk = Walk(rendered, start + len(trimmed))
    if trimmed:
        if text.endswith(trimmed):
            aligner.decide(
                walk.cursor, markup["think_close"], THINK_CLOSE, None, "think_close"
            )
        else:
            # Flash trimmed the trailing whitespace the teacher stopped after.
            counts["decisions_unplaced"] += 1
    walk.expect(THINK_CLOSE)
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise TypeError("assistant content must be text")
    if run_bytes(markup["content"]) != content.encode():
        raise ValueError("teacher content tokens do not reproduce the content")
    trimmed = content.strip()
    lead = len(content[: len(content) - len(content.lstrip())].encode())
    calls = message.get("tool_calls") or []
    if len(calls) != len(markup["calls"]):
        raise ValueError(
            "teacher markup and tool_calls disagree on the number of calls"
        )
    # What follows the content: the first call's wrapper or the end of the turn.
    after_content = (b"\n\n" if trimmed else b"") + TOOL_CALL_OPEN if calls else IM_END
    lo = walk.expect(trimmed.encode())
    counts["markup_teacher_tokens"] -= len(markup["content"])
    aligner.align(
        markup["content"],
        lo - lead,
        lo,
        walk.cursor,
        "content",
        stop=vocab.leading(after_content),
    )
    if markup["content_close"] is not None:
        if content == content.rstrip():
            aligner.decide(
                walk.cursor,
                markup["content_close"],
                after_content,
                IM_END,
                "content_close",
            )
        else:
            # Flash trimmed the trailing whitespace the teacher stopped after.
            counts["decisions_unplaced"] += 1
    for index, (call, parsed) in enumerate(zip(calls, markup["calls"], strict=True)):
        name, arguments = tool_arguments(call)
        if name != parsed["name"]:
            raise ValueError("teacher markup and tool_calls disagree on a tool name")
        if list(arguments) != [a["key"].decode() for a in parsed["arguments"]]:
            raise ValueError("teacher markup and tool_calls disagree on argument keys")
        counts["tool_calls"] += 1
        if index == 0:
            walk.expect((b"\n\n" if trimmed else b"") + TOOL_CALL_OPEN)
        else:
            walk.expect(b"\n" + TOOL_CALL_OPEN)
        lo = walk.expect(name)
        aligner.align(
            parsed["tag"],
            lo - CALL_NAME_OFFSET,
            lo,
            walk.cursor,
            "tool_name",
            count_tokens=False,
        )
        walk.expect(b">\n")
        for argument in parsed["arguments"]:
            aligner.decide(
                walk.cursor,
                argument["open"],
                PARAMETER_OPEN,
                TOOL_CALL_CLOSE,
                "argument_or_close",
            )
            walk.expect(PARAMETER_OPEN)
            lo = walk.expect(argument["key"])
            aligner.align(
                argument["tag"],
                lo - ARGUMENT_KEY_OFFSET,
                lo,
                walk.cursor,
                "parameter_key",
                count_tokens=False,
            )
            walk.expect(b">\n")
            value = arguments[argument["key"].decode()]
            counts["argument_values"] += 1
            if isinstance(value, str):
                lo = walk.expect(value.encode())
                flash_value = value.encode()
            else:
                # Flash renders non-string values with Jinja's tojson; the
                # teacher may have spaced its JSON differently.
                lo, end = walk.until(PARAMETER_CLOSE)
                flash_value = rendered[lo:end]
                if json.loads(flash_value) != value:
                    raise ValueError(
                        "Flash rendering of a non-string argument differs from tool_calls"
                    )
            counts["markup_teacher_tokens"] -= len(argument["value"])
            if run_bytes(argument["value"]) == flash_value:
                aligner.align(
                    argument["value"],
                    lo,
                    lo,
                    walk.cursor,
                    "argument_value",
                    stop=vocab.leading(PARAMETER_CLOSE),
                )
            else:
                counts["argument_values_unaligned"] += 1
                counts["argument_value_teacher_tokens"] += len(argument["value"])
            aligner.decide(
                walk.cursor, argument["close"], PARAMETER_CLOSE, None, "value_close"
            )
            walk.expect(PARAMETER_CLOSE)
        aligner.decide(
            walk.cursor,
            parsed["close"],
            TOOL_CALL_CLOSE,
            PARAMETER_OPEN,
            "argument_or_close",
        )
        walk.expect(TOOL_CALL_CLOSE)
        if parsed["next"] is not None:
            another = b"\n" + TOOL_CALL_OPEN
            if parsed["next_is_end"]:
                aligner.decide(
                    walk.cursor, parsed["next"], IM_END, another, "call_or_end"
                )
            else:
                aligner.decide(
                    walk.cursor, parsed["next"], another, IM_END, "call_or_end"
                )
    if rendered[walk.cursor :] not in COMPLETION_TAILS:
        raise ValueError("Flash completion continues past the rendered tool calls")


def spans_for_row(row):
    spans = [tuple(s) for s in row.get("packed_completions", [])]
    spans.append((len(row["input_token_ids"]), len(row["output_token_ids"])))
    previous = 1
    for prompt, count in spans:
        if (
            type(prompt) is not int
            or type(count) is not int
            or prompt < previous
            or count < 1
        ):
            raise ValueError("invalid packed completion spans")
        previous = prompt + count
    if previous != len(row["input_token_ids"]) + len(row["output_token_ids"]):
        raise ValueError("completion spans exceed row")
    return spans


def prepare_row(row, steps, vocab, *, allow_missing_logprobs=False):
    spans = spans_for_row(row)
    step_ids = row.get("steps")
    if not isinstance(step_ids, list) or len(step_ids) != len(spans):
        raise ValueError("steps must identify every packed completion in order")
    sequence = row["input_token_ids"] + row["output_token_ids"]
    events, counts = [], Counter()
    for step_id, (prompt, count) in zip(step_ids, spans, strict=True):
        step = steps[step_id]
        if step["run_id"] != row["run_id"]:
            raise ValueError("raw teacher run does not match converted row")
        counts["steps"] += 1
        if (
            not (step.get("extra") or {}).get("teacher_logprobs")
            and allow_missing_logprobs
        ):
            counts["steps_without_logprobs"] += 1
            counts["ce_only_completion_tokens"] += count
            continue
        # Reasoning-free summary prefills have no matching reasoning targets.
        last = (step.get("messages") or [{}])[-1]
        if last.get(
            "role"
        ) == "user" and not row.get("summary_thinking", False) and "Your context window is nearly full. Write a " in str(
            last.get("content", "")
        ):
            counts["summary_steps_excluded"] += 1
            continue
        rendered, boundaries = vocab.render(sequence[prompt : prompt + count])
        align_step(
            step,
            rendered,
            boundaries,
            Aligner(sequence, prompt, boundaries, vocab, events, counts),
        )
    result = dict(row)
    result[FIELD] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "tokenizer_sha256": vocab.sha256,
        "token_ids_sha256": token_identity(row),
        "events": events,
        "coverage": dict(counts),
    }
    return result


def token_identity(row):
    payload = [
        row["input_token_ids"],
        row["output_token_ids"],
        row.get("packed_completions", []),
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def validate_targets(row, vocab_size=None, tokenizer_sha256=None):
    target = row.get(FIELD)
    if not isinstance(target, dict) or target.get("schema") != SCHEMA:
        raise ValueError("missing or unsupported API teacher targets")
    identity = target.get("tokenizer_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("missing tokenizer identity")
    if tokenizer_sha256 is not None and identity != tokenizer_sha256:
        raise ValueError("API teacher targets use a different student tokenizer")
    if target.get("token_ids_sha256") != token_identity(row):
        raise ValueError("API teacher targets use different prompt/completion tokens")
    events = target.get("events")
    if not isinstance(events, list):
        raise TypeError("teacher events must be a list")
    sequence = row["input_token_ids"] + row["output_token_ids"]
    spans = spans_for_row(row)
    previous = -1
    for event in events:
        position = event["position"]
        if (
            type(position) is not int
            or position <= previous
            or not any(p <= position < p + n for p, n in spans)
        ):
            raise ValueError("teacher position must be unique, ordered and supervised")
        previous = position
        if event["sampled_id"] != sequence[position]:
            raise ValueError("teacher target is attached to different student tokens")
        ids, probs = event["ids"], event["probabilities"]
        if not ids or len(ids) != len(probs) or len(set(ids)) != len(ids):
            raise ValueError("invalid sparse teacher target")
        if any(
            type(i) is not int or i < 0 or (vocab_size is not None and i >= vocab_size)
            for i in ids
        ):
            raise ValueError("teacher alternative outside student vocabulary")
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1
            for p in probs
        ):
            raise ValueError("invalid teacher probability")
        if sum(probs) > 1 + 1e-6:
            raise ValueError("mapped teacher mass exceeds one")
    return events


def validate_weight(weight, teacher_mode, kl_weight, temperature, policy_gradient=None):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("teacher_logprob_weight must be finite and positive")
    if (
        teacher_mode != "none"
        or kl_weight != 0
        or temperature != 1
        or policy_gradient is not None
    ):
        raise ValueError(
            "API teacher loss needs teacher_mode none, kl_weight 0, temperature 1 and no policy_gradient"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=Path, required=True, help="existing Flash serving-format rows"
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--traces",
        type=Path,
        help="raw Kimi traces/<run_id>/steps.jsonl",
    )
    sources.add_argument(
        "--sources", type=Path, help="JSON mapping source_set to original trace roots"
    )
    parser.add_argument(
        "--allow-missing-logprobs",
        action="store_true",
        help="preserve missing-logprob completions as CE only; never drop rows",
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Flash tokenizer.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vocab = ByteVocabulary(args.tokenizer)
    source_roots = json.loads(args.sources.read_text()) if args.sources else None

    @lru_cache(maxsize=2)
    def raw(source, run):
        if not isinstance(run, str) or Path(run).name != run or run in (".", ".."):
            raise ValueError("invalid run_id")
        records = {}
        root = args.traces if source_roots is None else Path(source_roots[source])
        with (root / run / "steps.jsonl").open() as f:
            for line in f:
                step = json.loads(line)
                index = step["step_index"]
                if index in records:
                    raise ValueError("duplicate raw step_index")
                records[index] = step
        return records

    if args.out.resolve() == args.rows.resolve():
        raise ValueError("write to a separate output file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    counts = Counter()
    by_source = {}
    try:
        with args.rows.open() as src, temp.open("w") as dst:
            for line in src:
                row = json.loads(line)
                source = row.get("source_set", "")
                prepared = prepare_row(
                    row,
                    raw(source, row["run_id"]),
                    vocab,
                    allow_missing_logprobs=args.allow_missing_logprobs,
                )
                validate_targets(prepared)
                if {k: v for k, v in prepared.items() if k != FIELD} != row:
                    raise ValueError("target preparation changed baseline fields")
                counts.update(prepared[FIELD]["coverage"])
                counts["rows"] += 1
                counts["completion_tokens"] += sum(n for _, n in spans_for_row(row))
                local = by_source.setdefault(source, Counter())
                local.update(prepared[FIELD]["coverage"])
                local["rows"] += 1
                local["completion_tokens"] += sum(n for _, n in spans_for_row(row))
                dst.write(json.dumps(prepared, separators=(",", ":")) + "\n")
        if not counts["aligned_positions"]:
            raise ValueError("no usable teacher targets; output was not published")
        temp.replace(args.out)
    finally:
        temp.unlink(missing_ok=True)
    report = {
        "schema": SCHEMA,
        "counts": dict(counts),
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "tokenizer_sha256": vocab.sha256,
        "input_rows_sha256": file_sha(args.rows),
        "output_sha256": file_sha(args.out),
        "source_traces": source_roots
        if source_roots is not None
        else str(args.traces.resolve()),
        "missing_logprob_policy": "ce_only" if args.allow_missing_logprobs else "error",
        "baseline_fields_and_row_order_preserved": True,
        "probability_semantics": "provider generation distribution as returned; temperature unspecified",
        "scope": SCOPE,
    }
    args.out.with_suffix(args.out.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
