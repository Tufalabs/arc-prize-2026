"""ARC-3 rollout augmentation: every frame of the run recolored, actions untouched.

Runs in the ARC-3 harness environment (its TAAF, arcengine and Platypus);
skipped elsewhere. A scripted engine checks each frame path exactly, including
TAAF's level-win safety frame; the real offline games check that scoring is
the engine's own level progress, unchanged by the colors.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

arcengine = pytest.importorskip("arcengine")
game_api = pytest.importorskip("taaf.game_api")

from taaf.game import GameState, RunSession  # noqa: E402

from harness_support.arc3.augmented_game import ColorPermutedGameAPI  # noqa: E402
from harness_support.augmentation import (  # noqa: E402
    applied_record,
    augmentation_seed,
    permute_colors,
    sample_arc3,
)

GameAPI = game_api.GameAPI
COLOR_MAP = sample_arc3(augmentation_seed(3, 1, "scripted")).color_map
ALL_COLORS = np.arange(16, dtype=np.int8).reshape(4, 4)


class ScriptedEngine:
    """Two levels of two moves each; the first win arrives as one frame.

    It keeps the frame data and arrays it hands out, as arcengine does, so the
    test can see that recoloring never writes into them.
    """

    def __init__(self):
        self.actions = []
        self.handed_out = []
        self.level = 0
        self.moves = 0

    def _raw(self, frames, state=arcengine.GameState.NOT_FINISHED):
        raw = arcengine.FrameDataRaw(
            game_id="scripted",
            state=state,
            levels_completed=self.level,
            win_levels=2,
            available_actions=[1, 2, 6],
        )
        raw.frame = frames
        self.handed_out.append((raw, list(frames), [frame.copy() for frame in frames]))
        return raw

    def start(self, game):
        game.game_id = game.external_game_id
        game.number_of_levels = 2
        game.base_actions_per_level = [2, 2]
        return GameState(raw=self._raw([ALL_COLORS.copy()]))

    def step(self, action):
        self.actions.append(action)
        self.moves += 1
        board = (ALL_COLORS + self.moves) % 16
        if self.moves == 2:
            # A level win reported with a single frame: TAAF adds a green one.
            self.level = 1
            return GameState(raw=self._raw([board]))
        if self.moves == 4:
            self.level = 2
            return GameState(
                raw=self._raw([board, (board + 1) % 16], state=arcengine.GameState.WIN)
            )
        return GameState(raw=self._raw([board, (board + 3) % 16]))


def _scripted(monkeypatch, game_class, **kwargs):
    engine = ScriptedEngine()
    monkeypatch.setattr(GameAPI, "_start_game", lambda self, session: engine.start(self))
    monkeypatch.setattr(GameAPI, "_execute_action", lambda self, action: engine.step(action))
    monkeypatch.setattr(GameAPI, "_finish_game", lambda self: None)
    game = game_class(env_name="scripted", external_game_id="scripted#s0", **kwargs)
    game.grid_size = (4, 4)
    return game, engine


def _actions():
    return [
        arcengine.ActionInput(id=arcengine.GameAction.ACTION1, data={}),
        arcengine.ActionInput(id=arcengine.GameAction.ACTION6, data={"x": 3, "y": 1}),
        arcengine.ActionInput(id=arcengine.GameAction.ACTION2, data={}),
        arcengine.ActionInput(id=arcengine.GameAction.ACTION1, data={}),
    ]


def _frames(state):
    return [np.asarray(frame) for frame in state.raw.frame]


def _play(monkeypatch, game_class, **kwargs):
    game, engine = _scripted(monkeypatch, game_class, **kwargs)
    session = RunSession()
    session.record_intermediate_states = True
    states = [game.start_game(session)]
    initial_current = game.current_state
    for action in _actions():
        states.append(game.execute_action(action))
    game.finish_game()
    return game, engine, states, initial_current


def test_every_frame_of_every_level_is_recolored_and_actions_pass_through(monkeypatch):
    plain, plain_engine, plain_states, _ = _play(monkeypatch, GameAPI)
    game, engine, states, initial = _play(
        monkeypatch, ColorPermutedGameAPI, color_map=COLOR_MAP
    )
    # The agent's initial state is the game's current state, recolored.
    assert initial is states[0]
    for before, after in zip(plain_states, states, strict=True):
        frames = _frames(after)
        expected = [permute_colors(frame, COLOR_MAP) for frame in _frames(before)]
        assert len(frames) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(frames, expected, strict=True))
        assert all(frame.dtype == np.int8 for frame in frames)
        assert after.levels_completed == before.levels_completed
        assert after.raw.state == before.raw.state
        assert after.available_actions == before.available_actions
        assert after.just_won_level == before.just_won_level
    # The single-frame level win carries TAAF's level-win frame, recolored too.
    assert len(plain_states[2].raw.frame) == 2 and (plain_states[2].raw.frame[0] == 14).all()
    assert (np.asarray(states[2].raw.frame[0]) == COLOR_MAP[14]).all()
    # The run's own record holds recolored states.
    recorded = game.game_run.intermediate_states
    assert [state.raw.frame[-1].tobytes() for state in recorded] == [
        state.raw.frame[-1].tobytes() for state in states
    ]
    assert np.array_equal(game.current_state.frame.data, states[-1].frame.data)
    # Actions reach the engine unchanged, in order.
    assert [a.model_dump() for a in engine.actions] == [a.model_dump() for a in _actions()]
    assert [a.model_dump() for a in engine.actions] == [
        a.model_dump() for a in plain_engine.actions
    ]
    # Scoring is the engine's level progress, which the colors do not change.
    assert game.game_run.state == plain.game_run.state == "won"
    assert game.game_run.levels_completed == plain.game_run.levels_completed == 2
    assert game.game_run.actions_per_level == plain.game_run.actions_per_level
    assert game.game_run.final_score == plain.game_run.final_score
    # The engine's frame data and buffers are never written.
    for raw, frames, copies in engine.handed_out:
        assert raw.frame == frames
        assert all(np.array_equal(f, c) for f, c in zip(frames, copies, strict=True))


def test_the_rendered_image_is_of_the_recolored_frame(monkeypatch):
    from platypus.contract import partition
    from platypus.llm import frame_image
    import taaf.game

    _, _, states, _ = _play(monkeypatch, ColorPermutedGameAPI, color_map=COLOR_MAP)
    # The Platypus observations the Duck reads (a level win splits in two).
    from platypus import Action

    observations = partition(states[2], Action.from_action_input(_actions()[1]), 2)
    assert [o.outcome for o in observations] == ["level_won", "playing"]
    assert (observations[0].frame.data == COLOR_MAP[14]).all()
    small = taaf.game.Frame(data=np.array([[0, 5, 15]], dtype=np.int8))
    recolored = taaf.game.Frame(data=permute_colors(small.data, COLOR_MAP))
    # The pixels are the palette colors of the recolored cells.
    rgb = np.array(
        [[round(channel * 255) for channel in taaf.game.ARC_COLORS[COLOR_MAP[c]]] for c in range(16)],
        dtype=np.uint8,
    )
    assert np.array_equal(frame_image(recolored).data, rgb[small.data])
    assert not np.array_equal(frame_image(recolored).data, frame_image(small).data)


def test_a_color_map_must_permute_all_sixteen_colors():
    with pytest.raises(ValueError, match="all 16"):
        ColorPermutedGameAPI(env_name="x", color_map=tuple(range(10)))
    with pytest.raises(ValueError, match="all 16"):
        ColorPermutedGameAPI(env_name="x", color_map=(0,) * 16)


def test_worker_builds_one_recolored_game_per_slot(tmp_path):
    """The worker's games: one color map per game (shared by its slots), recorded."""
    pytest.importorskip("duck.agent")
    pytest.importorskip("re_arc")
    from test_full_game_slots import _spec, _worker

    spec = _spec(tmp_path)
    spec["augmentation"] = {
        "version": 1, "environment": "arc3", "run_seed": 3, "window": 9,
        "dihedral": False, "colors": True,
    }
    games = _worker()._build_benchmark(spec).games
    assert [type(game) for game in games] == [ColorPermutedGameAPI] * 3
    assert [game.external_game_id for game in games] == ["ar25#s0", "ar25#s1", "ls20#s2"]
    records = [applied_record(spec, game.external_game_id) for game in games]
    assert records[0] == records[1] != records[2]
    for game, record in zip(games, records, strict=True):
        assert record["puzzle"] == game.env_name
        assert game.color_map == sample_arc3(augmentation_seed(3, 9, game.env_name)).color_map
        assert record["color_map"] == list(game.color_map)
    session = RunSession()
    try:
        for game in games:
            plain = GameAPI(env_name=game.env_name, external_game_id="plain")
            frames = [_frames(g.start_game(session)) for g in (game, plain)]
            assert all(
                np.array_equal(a, permute_colors(b, game.color_map))
                for a, b in zip(*frames, strict=True)
            )
            for started in (game, plain):
                started.finish_game()
    finally:
        session.close()


# re-arc-3 ac1cf3e's recorded best play of anchor_ice-0001 (its
# best_gameplay.txt, which the installed package does not ship): 0 is RESET,
# n is ACTIONn. It completes all seven levels in 249 actions.
ANCHOR_ICE_SOLVE = (
    "4231442324424132423131423143241423113144132313231411313243142413220313242414"
    "1332314132313232424113132431424132213242414314413244223231013242413144241324"
    "1324314231314241141321323142414232212313144241324132431423131424114133214231"
    "413323142432424114133"
)


def test_a_recorded_solve_scores_the_same_with_every_frame_recolored():
    """The real engine through all seven levels of a recorded solve."""
    re_arc = pytest.importorskip("re_arc")
    from pathlib import Path

    if not (Path(re_arc.__file__).parent / "environment_files/anchor_ice/0001").is_dir():
        pytest.skip("this re-arc-3 revision has no anchor_ice")
    actions = ["RESET" if a == "0" else f"ACTION{a}" for a in ANCHOR_ICE_SOLVE]
    env = "anchor_ice-0001"
    color_map = sample_arc3(augmentation_seed(3, 9, env)).color_map
    session = RunSession()
    pair = (
        ColorPermutedGameAPI(env_name=env, external_game_id=f"{env}#s0", color_map=color_map),
        GameAPI(env_name=env, external_game_id="plain"),
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            states = [game.start_game(session) for game in pair]
            transitions = 0
            for name in actions:
                if states[1].won:
                    break
                action = arcengine.ActionInput(id=arcengine.GameAction.from_name(name), data={})
                states = [game.execute_action(action) for game in pair]
                frames = [_frames(state) for state in states]
                assert len(frames[0]) == len(frames[1])
                assert all(
                    np.array_equal(a, permute_colors(b, color_map))
                    for a, b in zip(*frames, strict=True)
                )
                assert states[0].levels_completed == states[1].levels_completed
                assert states[0].just_won_level == states[1].just_won_level
                transitions += states[0].just_won_level
            for game in pair:
                game.finish_game()
        # TAAF's reconciliation with the engine's own scorecard found no mismatch.
        assert not [w for w in caught if "R11.12" in str(w.message)]
        assert transitions >= 2
        runs = [game.game_run for game in pair]
        assert runs[0].levels_completed == runs[1].levels_completed >= 2
        assert runs[0].actions_per_level == runs[1].actions_per_level
        assert runs[0].final_score == runs[1].final_score > 0
    finally:
        session.close()
