"""Scripted-LLM integration references for the seal (I13 applied to
authored corner cases — the whale's stub-golden pattern on platypus rails).

Four scenarios, each a committed reference generated from a scripted
``StubBackend`` (deterministic, zero API spend) and replayed bit-exact in
the cheap tier. On top of the replay, the tests parse the COMMITTED
reference and assert every steered corner actually fired — so a future
regeneration cannot silently lose coverage.

Regenerate (rewrites the committed files; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted'

Scenario "corners" (CornerGame: 2 levels of [UP,UP] / [DOWN,DOWN]; LEFT =
instant GAME_OVER): nudge path, parallel tool calls with skipped results,
level-transition absorption, persistent memory, in-snippet batch
rejection, a file written and read back from the working directory,
mid-batch game-over shown to the
model with its non-RESET follow-up refused and its own RESET reviving the
game, and a game win.

Scenario "eviction" (ExampleGame, one level of 16 UPs, small context
window, image off): context eviction actually fires while the
explicit memory survives it.

Scenario "hostile" (CornerGame): host-call frames the model can author are
refused without executing code in the agent process or ending the game.

Scenario "summarize" (ExampleGame, two levels of 8 UPs, small context window,
otherwise the shipped seal without a world model): the summary the seal
compacts with fires more than once, on both levels, and the rebuilt history
carries the summary, the status, the kept tool call and the closing listing.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import arcengine
import numpy as np
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game import Game, GameState, RunSession
from taaf.game_examples import ExampleGame
from ullm.tools.python import HOST_CALL_REFUSED_NOTICE

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    parallel_tool_calls,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from seal import SealAgent

RNG_SEED = 20260722


@dataclass
class CornerGame(Game):
    """Two levels ([UP,UP], [DOWN,DOWN]); ACTION3 (LEFT) = instant
    GAME_OVER; RESET clears in-level progress and revives from game over.
    Frame fill encodes (level, progress) so board_changed is realistic."""

    label: str = field(default="corner_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _progress: int = field(default=0, init=False, repr=False)

    _TARGETS = ([arcengine.GameAction.ACTION1] * 2, [arcengine.GameAction.ACTION2] * 2)

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = len(self._TARGETS)
        self.base_actions_per_level = [len(seq) for seq in self._TARGETS]
        self.hint = "scripted corner exercise"
        self.grid_size = (64, 64)
        self._level = 0
        self._progress = 0
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            self._progress = 0
            return self._build(arcengine.GameState.NOT_FINISHED)
        if action.id == arcengine.GameAction.ACTION3:
            return self._build(arcengine.GameState.GAME_OVER)
        target = self._TARGETS[self._level]
        if action.id == target[self._progress]:
            self._progress += 1
            if self._progress >= len(target):
                self._level += 1
                self._progress = 0
                if self._level >= len(self._TARGETS):
                    return self._build(arcengine.GameState.WIN)
        else:
            self._progress = 0
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        if state == arcengine.GameState.WIN:
            fill, levels_completed = 14, len(self._TARGETS)
        elif state == arcengine.GameState.GAME_OVER:
            fill, levels_completed = 9, self._level
        else:
            fill, levels_completed = (self._level * 4 + self._progress) % 16, self._level
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=levels_completed,
            win_levels=len(self._TARGETS),
            available_actions=[1, 2, 3],
        )
        raw.frame = [np.full((64, 64), fill, dtype=np.int8)]
        return GameState(raw=raw)


def _python(code: str, *notes: str) -> ullm.conv.LLMResponse:
    return tool_call("python", {"code": code}, *notes)


def _corners_script() -> list[ullm.conv.LLMResponse]:
    return [
        # decision 1, call 1: text-only -> the nudge path fires.
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="I should inspect the two-lane board.")),
        # call 2: PARALLEL tool calls; memory appends, Python executes both
        # UPs (winning level 1 via the split-absorbed transition), and the
        # final Python call must be skipped explicitly (dangling tool_use fix).
        parallel_tool_calls(
            (
                "memory",
                {"text": "Two-lane board. UP advances level 1; test DOWN on level 2."},
            ),
            (
                "python",
                {"code": "submit_action([Action('UP')] * 2)\n"},
            ),
            ("python", {"code": "print('should be skipped')\n"}),
        ),
        # decision 2, call 3: in-snippet rejection - DOWN is valid but
        # TELEPORT is unknown; the whole call rejects, nothing executes.
        _python("submit_action([Action('DOWN'), Action('TELEPORT')])\n"),
        # call 4: real Python -- a file written with open(), listed, and read
        # back, where the duck's kernel refused the import.
        _python("open('notes.txt', 'w').write('kept\\n')\nprint(sorted(os.listdir('.')), open('notes.txt').read())\n"),
        # call 5: execute DOWN, LEFT (=lose), DOWN: the game-over stops the
        # batch early and the trailing DOWN never executes.
        _python("submit_action([Action('DOWN'), Action('LEFT'), Action('DOWN')])\n"),
        # decision 3 is built on the LOST board. call 6: a non-RESET action
        # is refused while the game is over.
        _python("submit_action([Action('DOWN')])\n"),
        # call 7: the model's own RESET revives the game.
        _python("submit_action([Action('RESET')])\n"),
        # decision 4, call 8: finish level 2 -> game WIN (no further call).
        _python("submit_action([Action('DOWN')] * 2)\n"),
    ]


def _eviction_script(n_moves: int) -> list[ullm.conv.LLMResponse]:
    responses = [
        parallel_tool_calls(
            ("memory", {"text": "EVICTION-SURVIVOR marker: keep pressing UP."}),
            ("python", {"code": "submit_action([Action('UP')])\n"}),
        )
    ]
    for i in range(1, n_moves):
        responses.append(_python(f"print('padding block {i} ' * 200)\nsubmit_action([Action('UP')])\n"))
    return responses


# A host-call payload is authored in the model's own cell, and the trusted
# adapter is reachable from that cell by object traversal: the private store
# is importable under its presented name, and its ``_submit`` adapter's
# closure cell 0 is the primitive. So each cell below is a host call the
# model can legitimately reach; the property under test is that none of them
# executes code in the agent process or ends the game, and each fails for its
# own reason (a refusal on one route is no evidence about another).
_REACH = "import runtime._repl_state as _rs\nhc = _rs._submit.__closure__[0].cell_contents\n"

# Where the A1 reduce bomb writes IFF it executes. A file (not a raise) is the
# probe: an exception would be caught by the refusal path and prove nothing,
# but a file on disk is a durable side effect only a real execution can leave,
# and it survives across a process boundary if the job runs out-of-process.
# The path is resolved from `tempfile.gettempdir()` at runtime, not written as
# a literal: the payload text rides into the committed reference as the model's
# own source, and a machine-specific temp root there would fail the replay
# guard and differ across machines. The test computes the same location.
_A1_MARKER_NAME = "seal_scripted_hostile_reduce_executed.marker"
_A1_EXEC_MARKER = Path(tempfile.gettempdir()) / _A1_MARKER_NAME

# A payload object whose ``__reduce__`` executes on load. A full unpickle would
# run it here, in the agent process, outside the kernel's limits — and would
# leave `_A1_EXEC_MARKER` behind. The restricted decoder refuses it at
# ``builtins.exec`` before any of that runs.
_HOSTILE_A1 = (
    _REACH
    + "_exec = hc.__globals__['__builtins__'].exec\n"
    + '_code = "import tempfile, pathlib; '
    + f"pathlib.Path(tempfile.gettempdir(), {_A1_MARKER_NAME!r}).write_text('executed')\"\n"
    + "Boom = type('Boom', (), {'__reduce__': lambda self: (_exec, (_code,))})\n"
    + "print('A1', hc('action', (Boom(),)))\n"
)
# The shared ``pickle.dumps`` rebound, so the primitive stamps a valid token
# onto bytes that are not a pickle.
_HOSTILE_A2 = (
    "import pickle\n"
    "_orig = pickle.dumps\n"
    "pickle.dumps = lambda *a, **k: b'not a pickle at all'\n"
    "try:\n"
    "    print('A2', submit_action([Action('UP')]))\n"
    "finally:\n"
    "    pickle.dumps = _orig\n"
)
# A host-function name the harness does not register.
_HOSTILE_A3 = _REACH + "print('A3', hc('definitely_not_a_host_function'))\n"
# A well-formed frame whose arguments a real host function rejects. The seal
# returns ``{'error': ...}`` rather than raising, so this must stay recoverable.
_HOSTILE_A4 = _REACH + "print('A4', hc('action', 'not a list of action dicts'))\n"
# A cell that stops itself with SystemExit, as a notebook cell may: it ends
# that cell, not the kernel, so the next cell still has what this one bound.
_HOSTILE_EXIT = "kept_across_exit = 41\nprint('before the exit')\nraise SystemExit\nprint('after the exit')\n"
_HOSTILE_AFTER_EXIT = "print('session kept', kept_across_exit + 1)\n"


def _hostile_script() -> list[ullm.conv.LLMResponse]:
    """One attack per call, then win the game — the point is that the run
    reaches the win, so a crash anywhere is a failure the outcome records."""
    return [
        _python(_HOSTILE_A1),
        _python(_HOSTILE_A3),
        _python(_HOSTILE_A4),
        _python(_HOSTILE_A2),
        _python(_HOSTILE_EXIT),
        _python(_HOSTILE_AFTER_EXIT),
        _python("submit_action([Action('UP')] * 2)\n"),
        _python("submit_action([Action('DOWN')] * 2)\n"),
    ]


#: The summary the stub writes, numbered by the order it was asked for.
SUMMARY_TEXT = "SUMMARY {n}: UP advances every level; keep pressing UP."


@dataclass
class _SummaryStub(ullm.backends.stub.StubBackend):
    """Plays the script, and answers each summary request with a numbered
    summary of its own, without advancing the script: where a summary fires
    follows from the token counts, which a script cannot name in advance."""

    summaries_written: int = 0

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        users = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        if users and users[-1].content.to_string().startswith("Your context window is nearly full"):
            self.summaries_written += 1
            text = SUMMARY_TEXT.format(n=self.summaries_written)
            return ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=text))
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        if llm_output.final_content.to_string().startswith("SUMMARY "):
            self._fill_into(llm_output, element, conversation, self.resolve_response_injection(sampling_params))
            return
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


#: The world model the summarize scenario's model writes over the seeded
#: example: what a rebuilt conversation then shows in full.
SUMMARY_WORLD_MODEL = '"""WORLD MODEL MARKER: UP advances."""\n'


def _summarize_script(n_moves: int, *, world_model: str | None = None) -> list[ullm.conv.LLMResponse]:
    """Three rounds a move: two that pad the history and bind session
    objects, then the acting one, so a summary can fire between a turn's
    rounds as well as at its start. The first move also writes a file, for
    the listing's files half, and ``world_model`` over the seeded example."""
    responses: list[ullm.conv.LLMResponse] = []
    for move in range(n_moves):
        first = f"trace_{move} = list(range({move + 1}))\nprint('padding {move}a ' * 1500)\n"
        if move == 0:
            first = "open('notes.txt', 'w').write('UP advances')\n" + first
            if world_model is not None:
                first = f"open('world_model.py', 'w').write({world_model!r})\n" + first
        responses.append(_python(first))
        responses.append(_python(f"def probe_{move}():\n    return {move}\n\n\nprint('padding {move}b ' * 1500)\n"))
        responses.append(_python("submit_action([Action('UP')])\n"))
    return responses


def _case(name: str, game: Callable[[], Game], agent_kwargs: dict[str, Any]) -> ReferenceCase:
    def build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
        # Without a world model unless a case asks for one: the corners are
        # the game's and the harness's, not the gate's.
        kwargs: dict[str, Any] = {"use_world_model": "none", **agent_kwargs}
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
            **kwargs,
        )
        return taaf.benchmark.Benchmark(
            games=[game()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
        )

    return ReferenceCase(
        name=name,
        build=build,
        profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
    )


CORNERS = _case(
    "seal_scripted_corners",
    CornerGame,
    # A per-move call cap that fits the widest scripted decision (rejection,
    # sandbox refusal, then the acting call) without ever re-entering. The
    # journal on: a memory append from the nudged response is a corner here.
    {"max_llm_calls_per_move": 4, "journal": True},
)

EVICTION = _case(
    "seal_scripted_eviction",
    lambda: ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 16], label="evict_game"),
    # Budget sized so the run overflows partway: system prompt + a few
    # blocks fit, later blocks force eviction (but never so tight that only
    # the newest block survives from the start). Two rulers meet here — the
    # trigger runs on the stub's billed usage (chars/4 of the sent
    # conversation) while the post-trim measurement is the provable bound
    # (character count + framing constants, ~4x the billed reading of the
    # same text) — so the budget must sit ABOVE the bound's system-prompt
    # floor plus one retained block (or a fired trim could never land) and
    # BELOW the billed peak the padded script reaches (or nothing fires),
    # which is why the script runs sixteen moves: billed usage climbs a
    # quarter as fast as the bound's floor stands tall.
    {
        "compaction_strategy": "prune",
        "journal": True,
        "multimodal_context": "",
        "context_window_tokens": 21504,
        "compact_to_tokens": 17000,
        "max_llm_calls_per_move": 2,
    },
)

HOSTILE = _case(
    "seal_scripted_hostile",
    CornerGame,
    # One attack per decision, so each gets its own LLM call and its own
    # refusal text in the next request.
    {"max_llm_calls_per_move": 1},
)

SUMMARIZE = _case(
    "seal_scripted_summarize",
    lambda: ExampleGame(
        target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 8, [arcengine.GameAction.ACTION1] * 8],
        label="summarize_game",
    ),
    # The shipped seal, summarize and all, in a window small enough that the
    # padded rounds overflow it several times over the game, and that a move's
    # third round overflows it, so a summary also replaces rounds of the turn
    # it fires in. The engine window is stated, as a hand-built seal that
    # summarizes must. Tool output capped below the default, as the window
    # was sized for, so the latest call fits beside the summary.
    {"context_window_tokens": 36000, "engine_ceiling_tokens": 65536, "tool_output_tokens": 2425},
)

SUMMARIZE_WORLD_MODEL = _case(
    "seal_scripted_summarize_world_model",
    lambda: ExampleGame(
        target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 4],
        label="summarize_game",
    ),
    # The summarize scenario with the shipped optional world model, whose file
    # the model replaces on its first call.
    {"context_window_tokens": 36000, "engine_ceiling_tokens": 65536, "use_world_model": "optional"},
)

SCRIPTS = {
    CORNERS.name: _corners_script(),
    EVICTION.name: _eviction_script(16),
    HOSTILE.name: _hostile_script(),
    SUMMARIZE.name: _summarize_script(16),
    SUMMARIZE_WORLD_MODEL.name: _summarize_script(4, world_model=SUMMARY_WORLD_MODEL),
}


async def test_corners_scenario_replays_and_covers_its_corners(tmp_path: Path) -> None:
    (ref,) = await replay_case(CORNERS, tmp_path)
    text = ref.all_request_text()

    # Outcome: won with exactly the steered action stream, the RESET being
    # the model's own decision on the lost board.
    actions = ref.action_ids()
    assert actions == ["ACTION1", "ACTION1", "ACTION2", "ACTION3", "RESET", "ACTION2", "ACTION2"]
    assert ref.expected["state"] == "won"
    reset_tokens = ref.expected["per_action_tokens"][actions.index("RESET")]
    assert sum(reset_tokens) > 0  # the RESET cost an LLM call

    # Corner: nudge fired.
    assert "You have not acted yet. Investigate further if you need to, then act:" in text
    # Memory written from the nudged second response carries host metadata.
    assert "[#1 | response=2 | step=0 | level=1 | last_action=NONE]" in text
    # Corner: parallel tool call skipped with an explicit result.
    assert "not executed: an earlier tool call in this response executed actions" in text
    # Corner: in-snippet batch rejection (whole call, nothing executed).
    assert "unknown action 'TELEPORT'" in text
    # Corner: real Python -- open() writes and reads in the working
    # directory, and the listing is what the model wrote there.
    assert "['notes.txt'] kept" in text
    # Corner: level transition surfaced to the model (split absorbed).
    assert "You have progressed to a new level!" in text
    # Corner: the loss stops the batch early — the trailing DOWN never
    # executed — and the model is told to RESET itself.
    assert "RESET on your next turn to restart the level." in text
    # Corner: the lost board reaches the model in the present tense.
    assert "The game is over." in text
    assert "Executed actions:\n- DOWN (4096 pix changed, colors 4->5)\n- LEFT (4096 pix changed, colors 5->9)" in text
    assert "The code executed 2 actions in the previous sequence." in text
    # Corner: a non-RESET action on the lost board is refused.
    assert "RESET is the only valid action." in text


async def test_eviction_scenario_replays_and_eviction_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(EVICTION, tmp_path)
    assert ref.expected["state"] == "won"

    # Eviction fired: the retained history grew for a while (so the run
    # was NOT trivially over budget from the start) and then shrank (a
    # user-led block was dropped).
    records = ref.records()
    counts = [len(r["request"]["elements"]) for r in records]
    assert max(counts) >= 5, counts  # history genuinely accumulated first
    assert any(b < a for a, b in zip(counts, counts[1:])), counts
    # Explicit host-backed memory survived eviction even though early
    # conversation blocks are gone.
    final_text = ref.request_text(records[-1])
    assert "EVICTION-SURVIVOR" in final_text
    assert "[#1 | response=1 | step=0 | level=1 | last_action=NONE]" in final_text
    assert "padding block 1 " not in final_text  # the early block itself was evicted


async def test_hostile_host_calls_are_refused_without_ending_the_game(tmp_path: Path) -> None:
    """No host-call frame the model can author may execute code in the agent
    process or end the run (ullm host-call trust boundary; each route below).
    """
    # A1's reduce bomb writes this file iff it executes; a stale one from an
    # earlier run would mask a regression, so clear it first and after.
    _A1_EXEC_MARKER.unlink(missing_ok=True)
    try:
        (ref,) = await replay_case(HOSTILE, tmp_path)
        text = ref.all_request_text()

        # A1 proof, positive and direct: the reduce bomb writes _A1_EXEC_MARKER
        # if `builtins.exec` is ever resolved and run. Its ABSENCE is what shows
        # the payload never executed — an assertion the win alone cannot make,
        # because a regressed decoder would run exec, and any raise it triggered
        # would be swallowed by the same refusal path and still let the game win.
        assert not _A1_EXEC_MARKER.exists(), "the reduce bomb executed in the agent process"
    finally:
        _A1_EXEC_MARKER.unlink(missing_ok=True)

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1", "ACTION1", "ACTION2", "ACTION2"]

    # A1 + A2: a payload that is a forbidden class (the reduce bomb) or not a
    # valid pickle at all is REFUSED with the stable notice, not escalated.
    assert text.count(HOST_CALL_REFUSED_NOTICE) >= 2
    assert "HostPayloadRefused" in text  # A1: the reduce bomb refused by the restricted decoder
    # A3: an unknown host-function name is refused by name.
    assert "unknown host function" in text
    # A4: a bad argument shape reaching a real host function stays a
    # recoverable {'error': ...} (the seal validates and reports rather than
    # raising), not a crash — already proven by the win, but pin the message
    # so a regression is loud.
    assert "submit_action() received an invalid normalized action batch." in text
    # A cell that ends itself with SystemExit keeps its kernel: its output
    # comes back, nothing after the exit runs, and the next cell reads what the
    # exiting one bound.
    assert "before the exit" in text and "after the exit" not in text.replace("print('after the exit')", "")
    assert "session kept 42" in text


async def test_summarize_scenario_replays_and_covers_its_branches(tmp_path: Path) -> None:
    (ref,) = await replay_case(SUMMARIZE, tmp_path)
    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1"] * 16
    text = ref.all_request_text()

    # The request: thorough, and from the second summary on asking for what
    # still holds of the previous one; on level 2, the earlier levels too.
    assert "Your context window is nearly full. Write a thorough summary" in text
    assert "Make sure to also take over anything that is still relevant from the previous summary." in text
    assert "Include mechanics that were in play on earlier levels" in text
    # The rebuilt history: the summary, the end of thinking the request opened
    # its reply with stripped from its head.
    assert "[context compacted — the following is your own summary of your work so far]\n\nSUMMARY 1:" in text
    assert "[context compacted — the following is your own summary of your work so far]\n\nSUMMARY 2:" in text
    # The rebuilt history in the order things happened: the most recent tool
    # call kept with its result, then the summary and the closing message.
    preamble, wrapper = "[context compacted — this game is already under way", "your own summary of your work so far]"
    closing = "[context compacted — your `python` session"
    rebuilt = [
        request.split(preamble, 1)[1].split(closing, 1)[0]
        for request in map(ref.request_text, ref.records())
        if preamble in request and closing in request
    ]
    assert rebuilt and any("tool_result" in history.split(wrapper, 1)[0] for history in rebuilt)
    # Every resumed context retains the visual colour reference after the
    # older user prompts that carried it have been summarized away.
    assert all("Palette legend image: one swatch per color index" in history for history in rebuilt)
    # The closing message, over the listing of files and session objects:
    # the model's own, none of the harness's names.
    assert "[context compacted — your `python` session and files are intact; continue from here]" in text
    assert "Your files:\n- `notes.txt`" in text
    assert "Your `python` session, objects you defined:\n- `probe_" in text
    listings = [
        request.split("objects you defined:\n", 1)[1].split("\n\n", 1)[0]
        for request in map(ref.request_text, ref.records())
        if "objects you defined:\n" in request
    ]
    assert listings
    for listing in listings:
        for harness_name in ("`history`", "`available_actions`", "`submit_action`", "`np`", "`segment`", "`Counter`"):
            assert harness_name not in listing, (harness_name, listing)
    # Tool calls of the current turn a summary replaced are counted on its status.
    assert "tool call omitted>" in text or "tool calls omitted>" in text

    # Exercise the replay adapter against the agent's actual diagnostics:
    # successful summaries have no legacy memory-dump result field.
    from platypus.viewer_data import load_game_full

    payload = load_game_full(tmp_path / "replay_job", None, 0)
    boundaries = [
        section
        for event in payload["events"]
        for section in event.get("transcript_sections", [])
        if section["kind"] == "boundary" and "compaction" in section
    ]
    assert len(boundaries) >= 2
    assert all(section["compaction"]["mode"] == "summarize" for section in boundaries)
    assert all(
        "summary installed" in section["content"] and "FAILED" not in section["content"] for section in boundaries
    )
    assert all(
        context["compaction"]["mode"] == "summarize"
        for context in payload["contexts"]
        if context.get("closed_by") == "compaction"
    )


async def test_summarize_shows_the_models_world_model(tmp_path: Path) -> None:
    """With a world model in use, a summary's closing message carries
    ``world_model.py`` whole, as the model left it, and the request says so."""
    (ref,) = await replay_case(SUMMARIZE_WORLD_MODEL, tmp_path)
    assert ref.expected["state"] == "won"
    text = ref.all_request_text()
    assert "`world_model.py` will be shown in full after compaction" in text
    assert f"Your `world_model.py`:\n```python\n{SUMMARY_WORLD_MODEL.rstrip()}\n```" in text


async def test_regenerate_seal_scripted_references(tmp_path: Path) -> None:
    """Regenerates every scripted reference (stub-driven, no API spend);
    gated because it rewrites committed files. Each fresh reference is
    validated by immediate replay (I13)."""
    require_regeneration()
    for case in (CORNERS, EVICTION, HOSTILE, SUMMARIZE, SUMMARIZE_WORLD_MODEL):
        stub = _SummaryStub if case in (SUMMARIZE, SUMMARIZE_WORLD_MODEL) else ullm.backends.stub.StubBackend
        backend = stub(responses=SCRIPTS[case.name], on_exhausted="raise")
        await regenerate_case(case, backend=backend, tmp_path=tmp_path / case.name)


async def test_seal_diagnostics_spans(tmp_path: Path) -> None:
    """The seal's diagnostics instrumentation, exercised by the corners scenario
    live (scripted stub, real kernels): seal/analyze spans under moves,
    llm/turn spans under them with kernel/exec events, and — the seal's
    distinctive shape — conversation artifacts carrying PRE-HISTORY
    (earlier [UserPrompt, ResponseTurn] blocks of the persistent
    conversation, growing move over move)."""
    from platypus.diagnostics import BlockStore, blocks_dir, build_span_tree, game_stream_path, read_events
    from platypus.reference import game_run_dir

    backend = ullm.backends.stub.StubBackend(responses=SCRIPTS[CORNERS.name], on_exhausted="raise")
    benchmark = CORNERS.build(
        ReferenceRun(
            llm_config=LLMConfig(
                backend_factory=ullm.factory.PrebuiltBackend(backend=backend), sampling_profiles=CORNERS.profiles
            ),
            job_dir=tmp_path / "job",
            seeds={},
        )
    )
    await benchmark.run()
    (run,) = benchmark.game_runs
    assert run.state == "won"

    (root,) = build_span_tree(read_events(game_stream_path(game_run_dir(tmp_path / "job", run.run_id))))
    analyzes = [c for move in root.children for c in move.children if c.kind == "seal/analyze"]
    assert analyzes, "LLM-driven moves open seal/analyze spans"
    # Wins and losses fold into the snippet that caused them, so every move
    # is a decision move, and every analyze executed at least one action
    # in-snippet before its move ended.
    assert len(analyzes) == len(root.children)
    assert all(a.kv["executed"] >= 1 and a.outcome == f"executed {a.kv['executed']}" for a in analyzes)
    assert any(a.kv["nudges"] > 0 for a in analyzes)  # the scripted nudge corner fired

    turns = [c for a in analyzes for c in a.children if c.kind == "llm/turn"]
    assert turns and all(t.name == "analyzer" for t in turns)
    assert any(  # real kernel executions attributed to their turn
        e["kind"] == "kernel/exec" and e["values"]["duration_s"] > 0 for t in turns for e in t.events
    )

    # Pre-history: the conversation artifact of each successive turn
    # carries the earlier turns too (the persistent seal conversation).
    blocks = BlockStore(blocks_dir(tmp_path / "job"))

    def n_blocks(turn: Any) -> int:
        artifact = next(a for a in turn.artifacts if a["name"] == "conversation")
        structure = blocks.load(artifact["block"], "json")
        return sum(1 for element in structure["elements"] if element["kind"] == "turn")

    counts = [n_blocks(t) for t in turns]
    assert counts[0] == 1 and counts[-1] > 1
    assert all(b >= a for a, b in zip(counts[:-1], counts[1:], strict=True))
