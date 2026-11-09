"""Token continuation core (ullm.continuation): the next prompt is the served
ids + the sampled ids + the tokens of what the template appends after the
answer; a rewritten history is detected; the witness chain stores every
served token once (requirements U5, the nesting invariant)."""

from typing import Any

import pytest
import ullm.continuation as cont
import ullm.conv

EOS = "<|im_end|>"


class _StubTokenizer:
    """A Qwen-shaped template over a character tokenizer: special tokens are
    single ids, everything else one id per character."""

    specials = {EOS: 1000, "<|im_start|>": 1001, "<think>": 1002, "</think>": 1003}
    eos_token = EOS
    eos_token_id = 1000

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, **kwargs: Any) -> str:
        out = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                body = "<think>\n" + (message.get("reasoning_content") or "") + "\n</think>\n\n"
                body += message.get("content") or ""
                out += "<|im_start|>assistant\n" + body + EOS + "\n"
            else:
                out += "<|im_start|>" + role + "\n" + str(message.get("content")) + EOS + "\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n"
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special, sid in self.specials.items():
                if text.startswith(special, i):
                    ids.append(sid)
                    i += len(special)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse.get(i, chr(i)) for i in ids)


TOK = _StubTokenizer()
PREV = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
ASSISTANT = {"role": "assistant", "reasoning_content": "hm", "content": "answer"}


def _render(messages: list[dict[str, Any]], *, generation: bool) -> str:
    return TOK.apply_chat_template(messages, add_generation_prompt=generation)


def _witness(
    *, epoch: int = 0, sampled: str, served_before: list[int] | None = None, prompt_ids: list[int] | None = None
) -> cont.ServedTurn:
    head = _render(PREV, generation=True)
    ids = prompt_ids if prompt_ids is not None else TOK.encode(head)
    witness = cont.witness(
        TOK,
        epoch=epoch,
        prompt_ids=ids,
        output_ids=TOK.encode(sampled),
        served_before=served_before,
        head_render=head,
        answered_render=_render(PREV + [ASSISTANT], generation=False),
    )
    assert witness is not None
    return witness


def _conversation_with(witness: cont.ServedTurn | None, epoch: int = 0) -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation(history_epoch=epoch)
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q")))
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="answer"))
    response.status = "processed"
    response._served = witness
    turn = ullm.conv.ResponseTurn()
    turn.elements.append(response)
    conversation.elements.append(turn)
    return conversation


def test_next_prompt_is_served_ids_plus_the_tokenised_delta() -> None:
    # The sampled ids are NOT what the template re-renders (an extra newline).
    witness = _witness(sampled="hm\n</think>\n\n\nanswer" + EOS)
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    assert last is witness
    new = _render(PREV + [ASSISTANT, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None
    prompt_len = len(witness.prefix_ids)
    assert continued.prompt_ids[:prompt_len] == list(witness.prefix_ids)
    assert continued.prompt_ids[prompt_len : prompt_len + len(witness.output_ids)] == list(witness.output_ids)
    answered = _render(PREV + [ASSISTANT], generation=False)
    assert TOK.decode(continued.delta_ids) == new[len(answered) - 1 :]
    assert continued.think_seeded is True


def test_length_stopped_answer_gets_the_terminator_the_template_adds() -> None:
    witness = _witness(sampled="hm\n</think>\n\nanswer")  # no EOS sampled
    assert witness.ended_with_eos is False
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    new = _render(PREV + [ASSISTANT, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None and continued.delta_ids[0] == TOK.eos_token_id


def test_rewritten_history_is_not_continued() -> None:
    witness = _witness(sampled="hm\n</think>\n\nanswer" + EOS)
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    rewritten = [{"role": "system", "content": "S2"}, PREV[1], ASSISTANT, {"role": "user", "content": "r"}]
    assert cont.continue_prompt(TOK, served, last, _render(rewritten, generation=True)) is None


def test_epoch_bump_invalidates_the_witness() -> None:
    witness = _witness(sampled="hm\n</think>\n\nanswer" + EOS)
    conversation = _conversation_with(witness)
    assert cont.served_history(conversation) is not None
    conversation.mark_history_rewritten()
    assert cont.served_history(conversation) is None


def test_witness_chain_stores_each_served_token_once() -> None:
    base = _witness(sampled="a" + EOS)
    served_before = [*base.prefix_ids, *base.output_ids]
    delta = TOK.encode("\n<|im_start|>user\nr" + EOS + "\n<|im_start|>assistant\n<think>\n")
    second = _witness(sampled="b" + EOS, served_before=served_before, prompt_ids=served_before + delta)
    assert second.continued and list(second.prefix_ids) == delta
    conversation = _conversation_with(base)
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="b"))
    response.status = "processed"
    response._served = second
    conversation.last_turn.elements.append(response)
    history = cont.served_history(conversation)
    assert history is not None
    served, last = history
    assert last is second
    assert served == served_before + delta + list(second.output_ids)


def test_answer_the_template_cannot_render_as_a_suffix_leaves_no_witness() -> None:
    head = _render(PREV, generation=True)
    assert (
        cont.witness(
            TOK,
            epoch=0,
            prompt_ids=TOK.encode(head),
            output_ids=TOK.encode("x" + EOS),
            served_before=None,
            head_render=head,
            answered_render="<|im_start|>assistant\nsomething else",
        )
        is None
    )


def test_image_parts_are_detected() -> None:
    assert cont.has_image(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}]}]
    )
    assert not cont.has_image([{"role": "user", "content": "plain"}])


def test_a_budget_stopped_think_block_gets_the_structure_it_never_sampled() -> None:
    """A length stop inside the seeded think block sampled no ``</think>``:
    the delta carries the template's whole unsampled close (closing tag,
    terminator, framing), not just the terminator."""
    unfinished = {"role": "assistant", "reasoning_content": "unfinished thought", "content": ""}
    head = _render(PREV, generation=True)
    witness = cont.witness(
        TOK,
        epoch=0,
        prompt_ids=TOK.encode(head),
        output_ids=TOK.encode("unfinished thought"),  # the budget stopped it here
        served_before=None,
        head_render=head,
        answered_render=_render(PREV + [unfinished], generation=False),
    )
    assert witness is not None and not witness.ended_with_eos
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    new = _render(PREV + [unfinished, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None
    delta = TOK.decode(continued.delta_ids)
    assert delta.startswith("\n</think>\n\n" + EOS + "\n"), delta
    assert TOK.decode(continued.prompt_ids) == new


def test_a_budget_stop_on_trailing_whitespace_still_closes_the_think_block() -> None:
    """The cap lands after newlines the template trims away: the served ids
    keep them (they were sampled) and the template's close still follows,
    so the model is never served an unclosed reasoning block."""
    unfinished = {"role": "assistant", "reasoning_content": "unfinished thought", "content": ""}
    head = _render(PREV, generation=True)
    witness = cont.witness(
        TOK,
        epoch=0,
        prompt_ids=TOK.encode(head),
        output_ids=TOK.encode("unfinished thought\n\n"),  # stopped right after blank lines
        served_before=None,
        head_render=head,
        answered_render=_render(PREV + [unfinished], generation=False),
    )
    assert witness is not None
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    new = _render(PREV + [unfinished, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None
    prompt = TOK.decode(continued.prompt_ids)
    assert "unfinished thought\n\n\n</think>\n\n" + EOS + "\n" in prompt, prompt
    assert prompt.count("</think>") == prompt.count("<think>") - 1  # the generation prompt opens the next one


def test_a_budget_stop_after_leading_whitespace_still_closes_the_think_block() -> None:
    """The seeded think block sampled a newline first and then hit the cap:
    the parser strips it from the reasoning, the template renders the
    trimmed text, and the served ids keep the newline — the close must
    still follow."""
    unfinished = {"role": "assistant", "reasoning_content": "unfinished thought", "content": ""}
    head = _render(PREV, generation=True)
    witness = cont.witness(
        TOK,
        epoch=0,
        prompt_ids=TOK.encode(head),
        output_ids=TOK.encode("\nunfinished thought"),
        served_before=None,
        head_render=head,
        answered_render=_render(PREV + [unfinished], generation=False),
    )
    assert witness is not None
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    new = _render(PREV + [unfinished, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None
    prompt = TOK.decode(continued.prompt_ids)
    assert "<think>\n\nunfinished thought\n</think>\n\n" + EOS + "\n" in prompt, prompt
    assert prompt.count("</think>") == prompt.count("<think>") - 1


IMG = "<img>"


class _ImageTokenizer(_StubTokenizer):
    """The stub template with an image part rendered as one placeholder token."""

    specials = {**_StubTokenizer.specials, IMG: 1004}

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, **kwargs: Any) -> str:
        flat = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(IMG if part.get("type") == "image" else part["text"] for part in content)
                flat.append({**message, "content": text})
            else:
                flat.append(message)
        return super().apply_chat_template(flat, add_generation_prompt=add_generation_prompt, **kwargs)


def test_an_image_delta_is_encoded_by_the_caller_and_expanded_per_image() -> None:
    """A delta carrying an image placeholder cannot be tokenised by text
    alone: the caller's encoder (the engine's processor) expands each
    placeholder per image; the served ids and the rest of the delta are
    untouched, and the witness knows how many images came before."""
    tok = _ImageTokenizer()
    prev = [{"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "image"}]}]
    assistant = {"role": "assistant", "reasoning_content": "hm", "content": "answer"}
    head = tok.apply_chat_template(prev, add_generation_prompt=True)
    witness = cont.witness(
        tok,
        epoch=0,
        prompt_ids=[7, 7, 7] + tok.encode(head)[1:],  # the engine served an expanded first image
        output_ids=tok.encode("hm\n</think>\n\nanswer" + EOS),
        served_before=None,
        head_render=head,
        answered_render=tok.apply_chat_template(prev + [assistant], add_generation_prompt=False),
        image_fingerprints=("img-a",),
    )
    assert witness is not None and witness.image_fingerprints == ("img-a",)
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    new = tok.apply_chat_template(
        prev + [assistant, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "r"}]}],
        add_generation_prompt=True,
    )

    def expand(delta: str) -> list[int]:
        return [t for t in tok.encode(delta) for t in ([8, 8, 8, 8] if t == 1004 else [t])]

    continued = cont.continue_prompt(tok, served, last, new, encode_delta=expand)
    assert continued is not None
    assert continued.prompt_ids[: len(served)] == served
    assert continued.delta_ids.count(8) == 4 and 1004 not in continued.delta_ids
    assert tok.decode([t for t in continued.delta_ids if t != 8]).startswith("\n<|im_start|>user\n")


def test_continuation_memo_keys_on_served_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-slot memo must key on the served ids' CONTENT, not their
    length: noncanonical tokenizations let two conversations agree on
    epoch, render, answer sha, and ledger length while holding different
    ids, and reusing the other conversation's ContinuationPrompt ships
    the wrong history (the served-echo check passes -- the engine echoes
    exactly the wrong ids it was sent)."""
    import ullm.continuation
    import ullm.conv
    from ullm.backends.oai import OpenAIServer

    backend = OpenAIServer(base_url="http://localhost:0/v1", api_key="k", model="m")
    encodes: list[list[int]] = []

    def fake_continue(
        tokenizer: Any, served: list[int], last: cont.ServedTurn, new_render: str
    ) -> cont.ContinuationPrompt:
        encodes.append(list(served))
        return cont.ContinuationPrompt(prompt_ids=[*served, 9], delta_ids=[9], think_seeded=False)

    monkeypatch.setattr(ullm.continuation, "continue_prompt", fake_continue)
    conversation = ullm.conv.Conversation(elements=[], history_epoch=3)
    last = cont.ServedTurn(
        epoch=3,
        prefix_ids=(),
        output_ids=(),
        continued=False,
        head_len=0,
        answered_len=0,
        answered_sha="abc",
        ended_with_eos=True,
    )

    first = backend._continue_prompt_cached(object(), [1, 2, 3], last, "render", conversation)
    second = backend._continue_prompt_cached(object(), [1, 2, 4], last, "render", conversation)
    assert len(encodes) == 2, "same-length different ids must re-encode, not reuse"
    assert second is not first
    third = backend._continue_prompt_cached(object(), [1, 2, 4], last, "render", conversation)
    assert len(encodes) == 2 and third is second, "identical ledger reuses the memo"
