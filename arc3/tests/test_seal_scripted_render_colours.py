"""A render outside the colours: a world model's board that holds a value no frame can.

One scripted reference over the delta game (``test_seal_scripted_frame_delta``)
with ``use_world_model="optional"`` and the frame delta on. Every python cell
opens with comments saying what it exercises.

The model's ``render`` draws the scramble's checkerboard from a table of
``(row, column, colour)`` and reads one colour from the wrong field: a column,
40, where a colour index goes. This is the bug a trained checkpoint wrote on
cd82 (sft_eval, 2026-10-30), and it hides until the scramble is first drawn,
so ``test_world_model()`` passes on a history without one.

- The checked SPACE renders the board with 40 in it. The render contract
  refuses it at the check: the batch stops on a render error the model can act
  on, the move goes unpredicted, and the frame is described from the previous
  one. Before the contract held colours, the out-of-range render reached the
  frame delta as the prediction and ended the game.
- ``test_world_model()`` on the history that now holds the scramble fails on
  that turn with the same contract message.
- The model fixes the field and plays checked to the end of both levels.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_render_colours'
"""

from __future__ import annotations

from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    descendants,
    regenerate_case,
    replay_case,
    replay_spans,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from test_seal_scripted_frame_delta import MODEL, DeltaGame

RNG_SEED = 20261031

# The delta game's model, with the scramble drawn from a table whose colour
# is read off the wrong field, and the step of each level from its background.
WRONG_FIELD = MODEL.replace(
    "STEP = 4\n",
    "STEP = 4 if BG == 5 else 8\n"
    "# (row, column, colour) of the checkerboard's first two cells, read off\n"
    "# the scramble screen.\n"
    "CHECKER = ((8, 40, 3), (8, 41, 4))\n",
).replace(
    "np.where((rows + cols) % 2 == 0, 3, 4)",
    "np.where((rows + cols) % 2 == 0, CHECKER[0][1], CHECKER[1][2])",
)
assert WRONG_FIELD.count("CHECKER[0][1]") == 1 and "STEP = 4 if BG == 5 else 8" in WRONG_FIELD


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model editing a file in place."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call. A decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1; history: the seed turn) ===================
        _python(
            "# No model of its own yet: the example fails here, so the move runs\n"
            "# unchecked.\n"
            "print(submit_action([Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _create("world_model.py", WRONG_FIELD),
        _python(
            "# The model passes: no scramble has been drawn yet, so the wrong field\n"
            "# has not been read. SPACE is checked, and its render holds 40, a\n"
            "# column, where the checkerboard's colour goes. The render contract\n"
            "# refuses it: the batch stops at SPACE on a render error.\n"
            "print(test_world_model())\n"
            "submit_action([Action('SPACE'), Action('RIGHT')])\n"
            "print('not reached')\n"
        ),
        # === decision 3 (history: seed, RIGHT, SPACE) =======================
        _python(
            "# The scramble is in history now: test_world_model() fails on it, on\n"
            "# the same contract.\n"
            "print(test_world_model())\n"
        ),
        _replace("world_model.py", "CHECKER[0][1]", "CHECKER[0][2]"),
        _python(
            "# Fixed: the model passes on the scramble, and four checked RIGHTs win\n"
            "# the level.\n"
            "print(test_world_model())\n"
            "r = submit_action([Action('RIGHT')] * 4)\n"
            "print(r['executed_actions'], len(history))\n"
        ),
        # === decision 4 (level 2, a fresh kernel) ===========================
        _python(
            "# Level 2 steps eight cells, which the model reads off the background.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('RIGHT')] * 3)['executed_actions'])\n"
        ),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the delta game, with an optional world model and the default frame delta."""
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        # No side turns: a reflection or an audit would take the script's
        # next replies, and none of the corners here is theirs.
        level_reflection="none",
        strategy_audit="none",
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
        use_world_model="optional",
        # Decision 3 tests, edits and tests again before it acts.
        max_llm_calls_per_move=8,
    )
    return taaf.benchmark.Benchmark(
        games=[DeltaGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="seal_scripted_render_colours",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_render_colours_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    text = ref.all_request_text()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == [
        "ACTION4",  # unchecked
        "ACTION5",  # checked: the render holds 40, the batch stops
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION4",  # level 1 won, checked
        "ACTION4",
        "ACTION4",
        "ACTION4",  # the game won
    ]
    contract = (
        "render(state) must return colour indices 0-15, or -1 for a cell it does not predict; "
        "got 40 at row 8, column 0."
    )
    # At the check: the batch stops on the contract, as on any render the check cannot use.
    assert (
        "runtime.game.WorldModelStop: Your world model mispredicted history[-1]: render raised after Action('SPACE'):\n"
        "Traceback (most recent call last):\n"
        '  File "<harness>", in world model\n'
        f"runtime.test._RenderContractError: {contract} The batch stopped after 1 of 2 actions; "
        "the rest were not executed."
    ) in text
    # At the test, on the turn the check refused.
    assert (
        f"TestResult: FAIL at history[2]\n  {contract}\n"
        "  State at failure, move was Action('SPACE'): history[1].state (input) → history[2].state (output)"
    ) in text
    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text and "frame_delta" not in text
    assert "framediff" not in text and "must hold colours" not in text

    # The refused move has no prediction, so its frame is described from the previous one (the
    # scramble, too costly to show on an understood level), and no delta is ever built from a
    # render with 40 in it. The failing test has no turn to explain: the contract names the cell.
    # Each later call ends on a win, whose frame gets no delta.
    records = [span.kv for span in descendants(replay_spans(tmp_path), "seal/frame_delta")]
    assert [(r["event"], r["reason"], r["turn_index"], r.get("chosen"), r["shown"]) for r in records] == [
        ("frame", "unpredicted", 1, "previous", True),
        ("frame", "unpredicted", 2, "previous", False),
    ]


async def test_regenerate_seal_scripted_render_colours(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path / CASE.name)
