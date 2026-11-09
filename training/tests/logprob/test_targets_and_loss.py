"""API-teacher alignment and a dense reference for sparse loss/backward."""

import math
from collections import Counter
from copy import deepcopy

import pytest
import torch

from sft.logprob.sparse_loss import (
    SparseTeacherKL,
    completion_teacher_losses,
)
from sft.logprob.targets import (
    FIELD,
    IM_END,
    THINK_CLOSE,
    TOOL_CALL_OPEN,
    Aligner,
    mapped_distribution,
    prepare_row,
    reasoning_tokens,
    validate_targets,
)
from tests.logprob.conftest import write_tokenizer


def item(data, p=0.6, alternatives=None):
    return {
        "token": "unused",
        "bytes": list(data) if data is not None else None,
        "logprob": math.log(p),
        "top_logprobs": alternatives or [],
    }


def row_and_steps():
    tokens = [item(b"a"), item(b"b"), item(b" "), item(b"c")]
    step = {
        "run_id": "run",
        "step_index": 9,
        "output_message": {"reasoning_content": "ab c"},
        "extra": {"teacher_logprobs": tokens + [item(None)]},
    }
    return (
        {
            "run_id": "run",
            "steps": [9],
            "input_token_ids": [0, 1],
            "output_token_ids": [7, 2, 3, 4],
        },
        {9: step},
    )


def test_teacher_tokens_spelling_one_student_token_merge_along_the_sampled_path(
    vocab,
):
    row, steps = row_and_steps()
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    assert [e["position"] for e in events] == [3, 4, 5]
    assert [e["sampled_id"] for e in events] == [2, 3, 4]
    # "a" then "b" spell Flash's "ab": the path product is its probability.
    assert events[0]["ids"] == [2] and events[0]["probabilities"] == [
        pytest.approx(0.36)
    ]
    coverage = prepared[FIELD]["coverage"]
    assert coverage["reasoning_teacher_tokens"] == 4
    assert coverage["aligned_positions"] == 3
    assert coverage["merged_positions"] == 1
    assert coverage["reasoning_merged_positions"] == 1
    assert coverage["merged_teacher_tokens"] == 2
    # No control token is a probability target.
    assert all(7 not in e["ids"] for e in events)


def test_teacher_tokens_crossing_the_student_token_end_stay_unaligned(vocab):
    # "a" then "b " cannot spell Flash's "ab": the second token crosses its end.
    row, steps = row_and_steps()
    steps[9]["extra"]["teacher_logprobs"] = [
        item(b"a"),
        item(b"b "),
        item(b"c"),
        item(None),
    ]
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    assert [(e["position"], e["sampled_id"]) for e in events] == [(5, 4)]
    coverage = prepared[FIELD]["coverage"]
    assert coverage["reasoning_teacher_tokens_shorter_than_student"] == 1
    assert coverage["reasoning_teacher_tokens_no_boundary"] == 1
    assert "merged_positions" not in coverage


def test_merged_path_carries_alternatives_as_the_bytes_they_would_spell(vocab):
    row, steps = row_and_steps()
    first = item(b"a", 0.6)
    first["top_logprobs"] = [item(b"a", 0.6), item(b"ab", 0.2), item(b"c", 0.1)]
    second = item(b"b", 0.5)
    second["top_logprobs"] = [item(b"b", 0.5), item(b" ", 0.3)]
    steps[9]["extra"]["teacher_logprobs"] = [first, second, item(b" "), item(b"c")]
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    merged = events[0]
    # ab: the path 0.6 * 0.5 plus the first step's own "ab"; c: a first-step
    # alternative; a: the second step's " " spells "a ", projected onto "a".
    assert merged["ids"] == [0, 2, 4]
    assert merged["probabilities"] == pytest.approx([0.18, 0.5, 0.1])
    coverage = prepared[FIELD]["coverage"]
    assert coverage["projected_alternatives"] == 1
    assert coverage["projected_mass"] == pytest.approx(0.18)


def test_partial_utf8_bytes_survive_without_lossy_decoding(vocab):
    step = {
        "run_id": "r",
        "output_message": {"reasoning_content": "é"},
        "extra": {"teacher_logprobs": [item(b"\xc3"), item(b"\xa9")]},
    }
    row = {
        "run_id": "r",
        "steps": [0],
        "input_token_ids": [7],
        "output_token_ids": [5, 6],
    }
    assert len(prepare_row(row, {0: step}, vocab)[FIELD]["events"]) == 2


def test_missing_or_modified_reasoning_is_rejected(vocab):
    row, steps = row_and_steps()
    bad = deepcopy(steps[9])
    bad["extra"]["teacher_logprobs"][1]["bytes"] = None
    with pytest.raises(ValueError, match="exactly reproduce"):
        reasoning_tokens(bad)
    bad = deepcopy(steps)
    bad[9]["output_message"]["reasoning_content"] = "ac"
    with pytest.raises(ValueError, match="exactly reproduce"):
        prepare_row(row, bad, vocab)
    with pytest.raises(ValueError, match="run does not match"):
        prepare_row(dict(row, run_id="wrong"), steps, vocab)


def test_tail_mass_is_preserved_and_sampled_token_is_not_counted_twice(vocab):
    selected = item(b"a", 0.6)
    selected["top_logprobs"] = [
        item(b"a", 0.6),
        item(b"b", 0.2),
        item(b"unmapped", 0.1),
        {"token": "placeholder", "bytes": [99], "logprob": -9999},
    ]
    ids, probabilities = mapped_distribution(selected, vocab)
    assert ids == [0, 1]
    assert probabilities == pytest.approx([0.6, 0.2])
    assert 1 - sum(probabilities) == pytest.approx(0.2)
    selected["top_logprobs"].append(item(b"a", 0.3))
    with pytest.raises(ValueError, match="conflicting"):
        mapped_distribution(selected, vocab)


@pytest.mark.parametrize("pieces", [[b" ", b"a", b"b", b" "], [b" a", b"b "]])
def test_stripped_reasoning_keeps_only_whole_original_teacher_tokens(vocab, pieces):
    row = {"run_id": "r", "steps": [0], "input_token_ids": [7], "output_token_ids": [0, 1]}
    step = {"run_id": "r", "output_message": {"reasoning_content": " ab "},
            "extra": {"teacher_logprobs": [item(piece) for piece in pieces]}}
    target = prepare_row(row, {0: step}, vocab)[FIELD]
    expected = [1, 2] if len(pieces) == 4 else []
    assert [event["position"] for event in target["events"]] == expected
    assert target["coverage"]["trimmed_reasoning_steps"] == 1
    assert target["coverage"]["trimmed_teacher_tokens"] == 2


def test_whitespace_only_reasoning_is_ce_only_but_corrupt_bytes_still_fail(vocab):
    row = {"run_id": "r", "steps": [0], "input_token_ids": [7], "output_token_ids": [0, 3, 1, 3]}
    step = {"run_id": "r", "output_message": {"reasoning_content": " "},
            "extra": {"teacher_logprobs": [item(b" ")]}}
    assert prepare_row(row, {0: step}, vocab)[FIELD]["events"] == []
    step["extra"]["teacher_logprobs"] = [item(b"b")]
    with pytest.raises(ValueError, match="exactly reproduce"):
        prepare_row(row, {0: step}, vocab)


def test_trimmed_reasoning_does_not_match_the_same_text_inside_tool_output(vocab):
    row = {"run_id": "r", "steps": [0], "input_token_ids": [7], "output_token_ids": [0, 3, 0, 3]}
    step = {"run_id": "r", "output_message": {"reasoning_content": " a "},
            "extra": {"teacher_logprobs": [item(b" "), item(b"a"), item(b" ")]}}
    events = prepare_row(row, {0: step}, vocab)[FIELD]["events"]
    assert [event["position"] for event in events] == [1]


def test_targets_are_bound_to_prompt_tokens_and_tokenizer(vocab):
    row, steps = row_and_steps()
    prepared = prepare_row(row, steps, vocab)
    bad = deepcopy(prepared)
    bad["input_token_ids"][0] = 6
    with pytest.raises(ValueError, match="different prompt"):
        validate_targets(bad)
    with pytest.raises(ValueError, match="different student tokenizer"):
        validate_targets(prepared, tokenizer_sha256="0" * 64)
    bad = deepcopy(prepared)
    bad[FIELD]["events"][0]["position"] = 1
    with pytest.raises(ValueError, match="supervised"):
        validate_targets(bad)


def dense_reference(hidden, head, indices, q):
    logp = (hidden.float() @ head.float().t()).log_softmax(-1)
    result = logp.new_zeros(())
    for i in range(len(hidden)):
        selected = set(indices[i].tolist())
        for token, probability in zip(indices[i], q[i], strict=True):
            if probability > 0:
                result += probability * (probability.log() - logp[i, token])
        tail = (1 - q[i].sum()).clamp_min(0)
        if tail > 0:
            other = torch.tensor([j for j in range(head.shape[0]) if j not in selected])
            result += tail * (tail.log() - logp[i, other].logsumexp(0))
    return result


@pytest.mark.parametrize("chunk", [1, 2, 17])
def test_sparse_loss_and_gradient_match_independent_dense_reference(chunk):
    torch.manual_seed(8)
    head = torch.randn(9, 4)
    h = torch.randn(3, 4, requires_grad=True)
    # Last row includes repeated zero-mass padding, not a new bucket member.
    ix = torch.tensor([[1, 3, 4], [0, 2, 6], [2, 2, 2]])
    q = torch.tensor([[0.3, 0.2, 0.1], [0.6, 0.2, 0.2], [0.7, 0, 0]])
    actual = SparseTeacherKL.apply(h, head, ix, q, chunk)
    reference_h = h.detach().clone().requires_grad_()
    reference = dense_reference(reference_h, head, ix, q)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(h.grad, reference_h.grad, atol=2e-6, rtol=2e-5)


def test_full_vocabulary_and_extreme_student_tail_have_finite_gradients():
    head = torch.eye(3)
    h = torch.tensor([[40.0, 39.0, -60.0], [0.0, 0.0, 0.0]], requires_grad=True)
    ix = torch.tensor([[0, 1, 0], [0, 1, 2]])
    q = torch.tensor([[0.2, 0.3, 0.0], [0.2, 0.3, 0.5]])
    actual = SparseTeacherKL.apply(h, head, ix, q, 1)
    reference_h = h.detach().clone().requires_grad_()
    reference = dense_reference(reference_h, head, ix, q)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    reference.backward()
    assert torch.isfinite(h.grad).all()
    torch.testing.assert_close(h.grad, reference_h.grad, atol=2e-6, rtol=2e-5)


def test_packed_spans_and_off_by_one_gradient_placement(vocab):
    row, steps = row_and_steps()
    row = dict(
        row,
        input_token_ids=[0, 1, 7, 2, 3, 4, 0, 1],
        steps=[9, 9],
        packed_completions=[[2, 4]],
    )
    prepared = prepare_row(row, steps, vocab)
    torch.manual_seed(4)
    head = torch.randn(8, 5)
    hidden = torch.randn(1, 12, 5, requires_grad=True)
    parts = completion_teacher_losses(hidden, head, prepared, [(2, 4), (8, 4)], 1)
    objective = 0.2 * parts[0] + 0.7 * parts[1]
    objective.backward()
    positions = (hidden.grad.abs().sum(-1)[0] > 0).nonzero().flatten().tolist()
    assert positions == [2, 3, 4, 8, 9, 10]
    before = objective.detach()
    updated = (hidden.detach() - 0.01 * hidden.grad).requires_grad_()
    after = completion_teacher_losses(updated, head, prepared, [(2, 4), (8, 4)], 2)
    assert 0.2 * after[0] + 0.7 * after[1] < before


# --- Content and tool-call alignment through Flash's chat template -----------

S = {"token": "", "bytes": None, "logprob": 0.0, "top_logprobs": []}  # control token


def items(*pieces, p=0.6):
    return [item(piece, p) for piece in pieces]


def tag(text):
    """A teacher tag body split into its text tokens, bracketed by control tokens."""
    return [S, *items(*text), S]


def markup_steps():
    """Step 9: content plus two calls. Step 10: a final answer ending inside <response>."""
    reasoning = items(b"a", b"b", b" ", b"c")
    value = items(b"x", b"=", b"1", b"\n", b"f", b"()")
    # A control-token alternative (close the argument) and a mappable one.
    value[0]["top_logprobs"] = [
        item(b"x", 0.6),
        item(b"f", 0.2),
        dict(S, logprob=math.log(0.2)),
    ]
    call_one = [
        *tag([b"call", b' tool="', b"python", b'"', b' index="', b"1", b'"']),
        *tag([b"argument", b' key="', b"code", b'"', b' type="', b"string", b'"']),
        *value,
        *tag([b"argument"]),
        *tag([b"call"]),
    ]
    call_two = [
        *tag([b"call", b' tool="', b"view", b"_file", b'"', b' index="', b"2", b'"']),
        *tag([b"argument", b' key="', b"path", b'"', b' type="', b"string", b'"']),
        *items(b"p"),
        *tag([b"argument"]),
        *tag(
            [
                b"argument",
                b' key="',
                b"view",
                b"_range",
                b'"',
                b' type="',
                b"array",
                b'"',
            ]
        ),
        *items(b"[", b"1", b",", b"240", b"]"),
        *tag([b"argument"]),
        *tag([b"call"]),
    ]
    stream = [
        *reasoning,
        *tag([b"think"]),
        *tag([b"response"]),
        *items(b" hi", b"!"),
        *tag([b"response"]),
        *tag([b"tools"]),
        *call_one,
        *call_two,
        *tag([b"tools"]),
    ]
    final = [*reasoning, *tag([b"think"]), *tag([b"response"]), *items(b"done")]
    calls = [
        {"function": {"name": "python", "arguments": '{"code":"x=1\\nf()"}'}},
        {
            "function": {
                "name": "view_file",
                "arguments": '{"path":"p","view_range":[1,240]}',
            }
        },
    ]
    return {
        9: {
            "run_id": "run",
            "step_index": 9,
            "output_message": {
                "reasoning_content": "ab c",
                "content": " hi!",
                "tool_calls": calls,
            },
            "extra": {"teacher_logprobs": stream},
        },
        10: {
            "run_id": "run",
            "step_index": 10,
            "output_message": {"reasoning_content": "ab c", "content": "done"},
            "extra": {"teacher_logprobs": final},
        },
    }


# Flash renders step 9 as
#   ab c\n</think>\n\nhi!\n\n<tool_call>\n<function=python>\n<parameter=code>\nx=1\nf()\n
#   </parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=view_file>\n
#   <parameter=path>\np\n</parameter>\n<parameter=view_range>\n[1, 240]\n</parameter>\n
#   </function>\n</tool_call><|im_end|>
STEP_NINE = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    10,
    23,
    12,
    24,
    25,
    26,
    27,
    12,
    28,
    15,
    29,
    30,
    31,
    32,
    33,
    36,
]
STEP_TEN = [2, 3, 4, 5, 34, 36]


def markup_row():
    return {
        "run_id": "run",
        "steps": [9, 10],
        "input_token_ids": [37, *STEP_NINE, 37],
        "output_token_ids": STEP_TEN,
        "packed_completions": [[1, len(STEP_NINE)]],
    }


def test_reasoning_only_stream_has_no_markup(vocab):
    row, steps = row_and_steps()
    coverage = prepare_row(row, steps, vocab)[FIELD]["coverage"]
    assert coverage["steps_without_markup"] == 1 and coverage["aligned_positions"] == 3


def test_content_and_argument_values_align_through_the_flash_template(markup_vocab):
    row, steps = markup_row(), markup_steps()
    prepared = prepare_row(row, steps, markup_vocab)
    events = validate_targets(prepared, 38, markup_vocab.sha256)
    sequence = row["input_token_ids"] + row["output_token_ids"]
    aligned = [markup_vocab.by_id[sequence[e["position"]]] for e in events]
    assert aligned == [
        b"ab",  # reasoning: the teacher's "a" + "b" merged
        b" ",
        b"c",
        b"\n</think>\n\n",  # the decision to stop reasoning
        b"!",  # content: the leading space is trimmed by the template
        b"\n\n<tool_call>\n<function=",  # the decision to stop the content and call
        b"python",
        b"code",
        b"x",
        b"=",
        b"1",
        b"\n",
        b"f",  # "()" merges with "\n"
        b"\n<tool_call>\n<function",  # the decision to make another call
        b"_file",  # "view" sits inside Flash's "=view"
        b"path",
        b"p",
        b"view",
        b"_range",  # "[1,240]" is spaced differently by Flash
        b"\n</parameter>\n",  # the decision to end the value
        b"</function>\n</tool_call>",  # the decision to close the call
        b"<|im_end|>",  # the decision to end the calls
        b"ab",
        b" ",
        b"c",
        b"\n</think>\n\n",
        b"done",  # the final answer
    ]
    coverage = prepared[FIELD]["coverage"]
    assert coverage == {
        "steps": 2,
        "tool_calls": 2,
        "aligned_positions": 27,
        "decision_positions": 7,
        "decisions_unplaced": 6,  # wrappers the fixture merges with a neighbour
        "think_close_positions": 2,
        "content_close_positions": 1,
        "value_close_positions": 1,
        "argument_or_close_positions": 1,
        "call_or_end_positions": 2,
        "reasoning_teacher_tokens": 8,
        "reasoning_positions": 6,
        "reasoning_merged_positions": 2,  # "a" + "b" spell "ab"
        "merged_positions": 2,
        "merged_teacher_tokens": 4,
        "content_teacher_tokens": 3,
        "content_positions": 2,
        "content_teacher_tokens_outside": 1,  # the trimmed leading space
        "tool_name_positions": 2,
        "parameter_key_positions": 4,
        "argument_values": 3,
        "argument_values_unaligned": 1,
        "argument_value_teacher_tokens": 6 + 1 + 5,
        "argument_value_positions": 6,
        "argument_value_teacher_tokens_shorter_than_student": 1,  # "()" in "()\n"
        "special_alternatives": 1,  # "close the argument" considered at "x"
        "special_alternative_mass": pytest.approx(0.2),
        "markup_teacher_tokens": len(steps[9]["extra"]["teacher_logprobs"])
        - 4
        - 2
        - 12
        + len(steps[10]["extra"]["teacher_logprobs"])
        - 4
        - 1,
        "mapped_alternatives": 29,
        "mapped_mass": pytest.approx(0.6 * 18 + 0.2 + 0.36 * 2 + 7 + 0.2),
    }
    # The control-token alternative (close the argument here) lands on the
    # token that begins Flash's parameter close.
    first_value = next(e for e in events if markup_vocab.by_id[e["sampled_id"]] == b"x")
    assert first_value["ids"] == [13, 17, 32]
    assert sum(first_value["probabilities"]) == pytest.approx(1.0)
    assert prepared[FIELD]["scope"]


def test_teacher_markup_is_parsed_strictly():
    from sft.logprob.targets import teacher_markup

    assert teacher_markup([]) is None and teacher_markup([S]) is None
    parsed = teacher_markup(markup_steps()[9]["extra"]["teacher_logprobs"][4:])
    assert [c["name"] for c in parsed["calls"]] == [b"python", b"view_file"]
    assert [a["key"] for a in parsed["calls"][1]["arguments"]] == [
        b"path",
        b"view_range",
    ]
    assert parsed["calls"][1]["arguments"][1]["type"] == b"array"
    for broken, message in [
        ([*items(b"x"), S, *items(b"think"), S], "without a delimiter"),
        ([*tag([b"reply"])], "expected b'think'"),
        (
            [
                *tag([b"think"]),
                *tag([b"response"]),
                *tag([b"response"]),
                *tag([b"tools"]),
                *tag([b"call", b' tool="', b"python", b'"']),
            ],
            "ends inside",
        ),
        (
            [
                *tag([b"think"]),
                *tag([b"response"]),
                *tag([b"response"]),
                *tag([b"tools"]),
                *tag([b"argument"]),
            ],
            "expected <call>",
        ),
        (
            [
                *tag([b"think"]),
                *tag([b"response"]),
                *tag([b"response"]),
                *tag([b"tools"]),
                *tag([b"tools"]),
                *tag([b"x"]),
            ],
            "after </tools>",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            teacher_markup(broken)


def test_tool_calls_and_flash_rendering_are_cross_checked(markup_vocab):
    row, steps = markup_row(), markup_steps()
    message = steps[9]["output_message"]
    bad = deepcopy(steps)
    bad[9]["output_message"]["tool_calls"] = message["tool_calls"][:1]
    with pytest.raises(ValueError, match="number of calls"):
        prepare_row(row, bad, markup_vocab)
    bad = deepcopy(steps)
    bad[9]["output_message"]["tool_calls"][1]["function"]["arguments"] = (
        '{"view_range":[1,240],"path":"p"}'
    )
    with pytest.raises(ValueError, match="argument keys"):
        prepare_row(row, bad, markup_vocab)
    bad = deepcopy(steps)
    bad[9]["output_message"]["tool_calls"][1]["function"]["arguments"] = (
        '{"path":"p","view_range":[1,241]}'
    )
    with pytest.raises(ValueError, match="non-string argument"):
        prepare_row(row, bad, markup_vocab)
    bad = deepcopy(steps)
    bad[9]["output_message"]["content"] = "hi!"
    with pytest.raises(ValueError, match="reproduce the content"):
        prepare_row(row, bad, markup_vocab)
    # Flash's rendering must follow the template exactly: no blank line before
    # the first call when there is content, or a completion that runs on.
    broken = deepcopy(row)
    broken["input_token_ids"][1 + STEP_NINE.index(8)] = 35
    with pytest.raises(ValueError, match="Flash rendering mismatch"):
        prepare_row(broken, steps, markup_vocab)
    broken = deepcopy(row)
    broken["output_token_ids"] = [*STEP_TEN, 34]
    with pytest.raises(ValueError, match="continues past"):
        prepare_row(broken, steps, markup_vocab)


def test_alternatives_without_a_student_token_project_onto_flashs_first_token(
    vocab,
):
    selected = item(b"a", 0.5)
    selected["top_logprobs"] = [
        item(b"a", 0.5),
        item(b"ab", 0.1),  # exact
        item(b"abc", 0.2),  # Flash spells it ab, c: lands on ab
        item(b"a\xc3", 0.05),  # partial UTF-8: residual
        item(b"zz", 0.05),  # no student token starts it: residual
    ]
    counts = Counter()
    ids, probabilities = mapped_distribution(selected, vocab, counts)
    assert ids == [0, 2]
    assert probabilities == pytest.approx([0.5, 0.3])
    assert counts["projected_alternatives"] == 1
    assert counts["projected_mass"] == pytest.approx(0.2)
    assert vocab.first_token(b"abc") == 2 and vocab.first_token(b"zz") is None


def test_longer_teacher_token_is_attached_to_the_student_token_at_its_boundary(
    vocab,
):
    # The teacher spells "ab c" as "ab " + "c"; Flash as ab, " ", c.
    row, steps = row_and_steps()
    first = item(b"ab ", 0.7)
    first["top_logprobs"] = [item(b"ab ", 0.7), item(b"ab", 0.1), item(b"a", 0.1)]
    steps[9]["extra"]["teacher_logprobs"] = [first, item(b"c"), item(None)]
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    # ab carries the teacher's "ab " mass; the " " that spells its rest is CE-only.
    assert [(e["position"], e["sampled_id"]) for e in events] == [(3, 2), (5, 4)]
    assert events[0]["ids"] == [0, 2]
    assert events[0]["probabilities"] == pytest.approx([0.1, 0.8])
    coverage = prepared[FIELD]["coverage"]
    assert coverage["aligned_positions"] == 2
    assert coverage["prefix_positions"] == 1
    assert coverage["reasoning_prefix_positions"] == 1
    assert "prefix_positions_projection_differs" not in coverage


def test_projection_that_differs_from_flashs_context_choice_is_counted(vocab):
    row, steps = row_and_steps()
    row["output_token_ids"] = [7, 0, 1, 4]  # Flash spelled "abc" as a, b, c here
    steps[9]["output_message"]["reasoning_content"] = "abc"
    steps[9]["extra"]["teacher_logprobs"] = [item(b"abc", 0.9), item(None)]
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    assert [(e["position"], e["sampled_id"], e["ids"]) for e in events] == [(3, 0, [0])]
    assert prepared[FIELD]["coverage"]["prefix_positions_projection_differs"] == 1


def test_sampled_token_maps_to_flashs_context_choice_even_when_its_bytes_are_a_token(vocab):
    # Flash spelled "ab" as a, b although ab is a token; the teacher's sampled "ab" (and its
    # alternative "a") belong to the a Flash chose, not to the ab the vocabulary also holds.
    row, steps = row_and_steps()
    row["output_token_ids"] = [7, 0, 1]
    steps[9]["output_message"]["reasoning_content"] = "ab"
    first = item(b"ab", 0.9)
    first["top_logprobs"] = [item(b"ab", 0.9), item(b"a", 0.05), item(b"b", 0.02)]
    steps[9]["extra"]["teacher_logprobs"] = [first, item(None)]
    prepared = prepare_row(row, steps, vocab)
    events = validate_targets(prepared, 8, vocab.sha256)
    assert [(e["position"], e["sampled_id"], e["ids"]) for e in events] == [(3, 0, [0, 1])]
    assert events[0]["probabilities"] == pytest.approx([0.95, 0.02])
    coverage = prepared[FIELD]["coverage"]
    assert coverage["prefix_positions"] == 1
    assert coverage["prefix_positions_projection_differs"] == 1


def special(p, alternatives):
    return {
        "token": "",
        "bytes": None,
        "logprob": math.log(p),
        "top_logprobs": alternatives,
    }


def test_control_token_decisions_land_on_the_wrapper_branches(tmp_path):
    pieces = [b"x", b"\n", b"\n\n", b"</think>", b" Could", b"<", b"</"]
    vocab = write_tokenizer(tmp_path / "t.json", pieces, ("<|im_end|>",))
    # Stop reasoning (0.9, its own entry repeated among the alternatives) or
    # go on with " Could"; any other control token there also means stopping.
    stop = special(0.9, [special(0.9, []), item(b" Could", 0.05), special(0.03, [])])
    sequence = [0, 1]
    aligner = Aligner(sequence, 0, vocab.render(sequence)[1], vocab, [], Counter())
    aligner.decide(1, stop, THINK_CLOSE, None, "think_close")
    assert aligner.events == [
        {"position": 1, "sampled_id": 1, "ids": [1, 4], "probabilities": [0.93, 0.05]}
    ] or aligner.events[0]["probabilities"] == pytest.approx([0.93, 0.05])
    # Stop the content and call a tool (0.7) or end the turn (0.3).
    calls = special(0.7, [special(0.7, []), special(0.3, [])])
    sequence = [0, 2]
    aligner = Aligner(sequence, 0, vocab.render(sequence)[1], vocab, [], Counter())
    aligner.decide(1, calls, b"\n\n" + TOOL_CALL_OPEN, IM_END, "content_close")
    event = aligner.events[0]
    assert (event["sampled_id"], event["ids"]) == (2, [2, 7])
    assert event["probabilities"] == pytest.approx([0.7, 0.3])
    assert aligner.counts["decision_positions"] == 1
    # Where Flash's token does not begin the wrapper, nothing is placed.
    aligner.decide(0, stop, THINK_CLOSE, None, "think_close")
    assert aligner.counts["decisions_unplaced"] == 1 and len(aligner.events) == 1


def test_control_token_alternatives_at_text_positions_mean_stop(vocab):
    selected = item(b"a", 0.6)
    selected["top_logprobs"] = [item(b"a", 0.6), special(0.3, [])]
    assert mapped_distribution(selected, vocab) == ([0], [0.6])
    counts = Counter()
    ids, probabilities = mapped_distribution(selected, vocab, counts, stop=4)
    assert ids == [0, 4] and probabilities == pytest.approx([0.6, 0.3])
    assert counts["special_alternatives"] == 1
    assert counts["special_alternative_mass"] == pytest.approx(0.3)
