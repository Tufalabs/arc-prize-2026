"""The optional world model: playing without one, choosing one, and dropping it.

One scripted reference over the meter game (``test_seal_scripted_world_model``)
with ``use_world_model="optional"``. Every python cell opens with comments
saying what it exercises.

- No exploration: with the example model failing, a batch of any length runs
  unchecked from the first move, and the next prompt says so.
- A model written and passing: its batch runs checked and stops at the first
  move it gets wrong. A ``WorldModelStop`` caught in the cell does not let the
  cell play on: the stop is final for the call.
- After the stop, the prompt offers the choice, and the model plays on
  without its model: the batch runs unchecked and wins the level.
- The next level: a fixed model passes again, so its batch runs checked, the
  next prompt says so, and a checked move wins the game. Its energy runs
  down twice as fast as the game's, which the check allows: the frame line
  says the frame is as predicted once energy is reconciled, and shows no
  block of the bar, which a model is better off leaving rough.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_optional'
"""

from __future__ import annotations

from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from platypus import LLMAgentSolver, RunConfig
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration, tool_call
from seal import SealAgent
from seal.prompts import SealPrompts
from test_seal_scripted_world_model import FIXED_MODEL, NAIVE_MODEL, RULES, MeterGame

RNG_SEED = 20260924


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call. A decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1; history: the seed turn) ====================
        _python(
            "# The optional mode preloads no WorldModelRequired: nothing refuses a move.\n"
            "print('WorldModelRequired' in globals(), 'WorldModelStop' in globals())\n"
            "# world_model.py is still the example, which fails here, so the batch runs\n"
            "# unchecked and whole: no exploration budget, no one-action rule.\n"
            "r = submit_action([Action('DOWN'), Action('DOWN')])\n"
            "print(r['executed_actions'], len(history))\n"
        ),
        # === decision 2 (the prompt said the batch ran unchecked) ============
        _create("rules.py", RULES),
        _create("world_model.py", NAIVE_MODEL),
        _python(
            "# A passing model: the batch runs checked. The model believes a non-target\n"
            "# move leaves progress alone, but DOWN at progress 1 resets it, so the\n"
            "# batch stops at the DOWN and the last UP never runs.\n"
            "print(test_world_model())\n"
            "try:\n"
            "    submit_action([Action('UP'), Action('DOWN'), Action('UP')])\n"
            "except WorldModelStop as stop:\n"
            "    print('caught:', str(stop).splitlines()[0])\n"
            "# Caught or not, the stop is final for this python call.\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 3 (the prompt offered the choice) ======================
        _python(
            "# Playing on without the model: it fails on the mispredicted turn now,\n"
            "# so the batch runs unchecked, and wins the level.\n"
            "print(bool(test_world_model()))\n"
            "print(submit_action([Action('UP'), Action('UP')])['executed_actions'])\n"
        ),
        # === decision 4 (level 2; the last batch ran unchecked) ==============
        _create("world_model.py", FIXED_MODEL.replace("state['energy'] - 1", "state['energy'] - 2")),
        _python(
            "# The fixed model, calibrated from this level's first frame, passes: the\n"
            "# move runs checked. Its energy falls by 2 where the game's falls by 1,\n"
            "# so the new frame matches only under another energy value.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 5 (the prompt said the batch ran checked) ==============
        _python("# A checked move wins the game.\nprint(submit_action([Action('DOWN')])['executed_actions'])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the meter game, with an optional world model."""
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
    )
    return taaf.benchmark.Benchmark(
        games=[MeterGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="seal_scripted_optional",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_optional_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    text = ref.all_request_text()
    surface = SealPrompts()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == [
        "ACTION2",  # unchecked, under the failing example
        "ACTION2",
        "ACTION1",  # checked
        "ACTION2",  # checked, mispredicted: the batch stops here
        "ACTION1",  # unchecked, after the stop
        "ACTION1",  # level 1 won
        "ACTION2",  # checked, under the fixed model
        "ACTION2",  # the game won, checked
    ]
    # No gate refuses a move, and nothing about exploration is said.
    assert "False True\n[Action('DOWN'), Action('DOWN')] 3" in text
    for mandatory_only in (
        surface.world_model_required_line,
        surface.exploration_status_line.split("{")[0],
        surface.world_model_check_lines,
        surface.world_model_batch_stop_line,
        surface.checklist_act_world_model,
        surface.checklist_act_world_model_without_journal,
    ):
        assert mandatory_only.strip() not in text, mandatory_only[:80]
    # What each batch ran under, said in the next prompt.
    assert surface.summary_unchecked_line in text
    assert surface.summary_checked_line in text
    # The checked batch stopped at the move the model got wrong; the caught stop
    # did not let the cell play on.
    assert (
        "caught: Your world model mispredicted history[-1]: render does not match the new frame after "
        "Action('DOWN') (4086 pixels differ). The batch stopped after 2 of 3 actions; the rest were not executed."
    ) in text
    assert (
        "runtime.game.WorldModelStop: Your world model mispredicted an earlier move in this cell, so no further "
        "action runs here. Nothing was executed."
    ) in text
    # A render the check accepted only under another energy value: as predicted, no block of the bar.
    assert (
        surface.summary_checked_line + "\nCurrent state: step 8, level 2.\nThe current frame is as predicted by "
        "your world model, once its energy is reconciled with the frame's."
    ) in text
    # The choice after a stop, and playing on without the model.
    assert surface.summary_world_model_mismatch + "\n" + surface.summary_world_model_stop_choice in text
    assert "True\n[Action('UP'), Action('UP')]" in text
    # The optional surface, every sentence of it.
    for fragment in (
        surface.world_model_intro_line_optional,
        surface.world_model_seed_line_optional,
        surface.world_model_check_lines_optional,
        surface.world_model_pays_lines.split("{gains}")[0],
        surface.world_model_gain_check,
        surface.world_model_gain_search,
        surface.world_model_gain_frame,
        surface.world_model_workflow_line_optional,
        surface.tool_world_model_sentence_optional,
        surface.new_level_world_model_lines_optional,
        surface.checklist_decide_world_model_optional + surface.checklist_decide_world_model_optional_game_start,
    ):
        assert fragment.strip() in text, fragment[:80]

    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text


async def test_regenerate_seal_scripted_optional(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)
