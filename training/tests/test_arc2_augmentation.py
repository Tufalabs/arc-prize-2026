"""ARC-2 rollout augmentation on the real TAAF task model, worker and Duck.

The first test needs only TAAF (numpy); the worker test needs the ARC-2
worker environment (vllm, the Duck), as tests/test_arc2_worker_construction.py.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_ARC2_DATA = (ROOT / "arc2-platypus/packages/taaf/src/taaf/data/arc2-eval.npz").exists()
HARNESS_PATHS = [
    ROOT / "arc2-platypus/packages" / name / "src" for name in ("duck", "platypus", "taaf")
] + [ROOT / "platypus/packages/ullm/src", ROOT / "scripts", ROOT]

# Shared by both programs: a puzzle, and a window whose augmentation both
# moves cells and changes colors, so that nothing passes by coincidence.
SETUP = """
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from taaf.datasets import load
from taaf.game import ArcGame, GameSpec, Submission
from taaf.grid import Grid
from harness_support.augmentation import (
    applied_record, augment_arc2_task, augmentation_seed, sample_arc2,
)

job_dir = Path(sys.argv[1])
task = next(t for t in load('arc2-eval') if len(t.to_answer) >= 2)
puzzle = task.game_id
grids = [g for p in task.pairs for g in (p.question, p.answer) if g is not None]
present = sorted({int(c) for g in grids for c in np.unique(g.data)})
def drawn(window):
    return sample_arc2(augmentation_seed(17, window, puzzle), present)
window = next(
    w for w in range(100)
    if drawn(w).transform != 'identity'
    and all(drawn(w).color_map[c] != c for c in present)
)
augmentation = drawn(window)
spec = {
    'job_dir': str(job_dir),
    'augmentation': {
        'version': 1, 'environment': 'arc2', 'run_seed': 17, 'window': window,
        'dihedral': True, 'colors': True,
    },
}
def same(grid, array):
    return isinstance(grid, Grid) and np.array_equal(grid.data, array)
"""


def _run(program, tmp_path, paths):
    result = subprocess.run(
        [sys.executable, "-c", SETUP + program, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.skipif(not HAS_ARC2_DATA, reason="Requires the private ARC-2 submodule")
def test_taaf_plays_and_scores_the_augmented_task(tmp_path):
    """Every grid is augmented, and TAAF scores against the augmented outputs."""
    program = """
from harness_support.arc2.results import enrich_arc2_result

slot = puzzle + '#s0'
augmented = augment_arc2_task(replace(task, game_id=slot), puzzle=puzzle, spec=spec)
assert type(augmented) is GameSpec and augmented.game_id == slot
for before, after in zip(task.pairs, augmented.pairs, strict=True):
    assert after.given == before.given
    assert same(after.question, augmentation.apply(before.question.data))
    assert same(after.answer, augmentation.apply(before.answer.data))
assert applied_record(spec, slot)['color_map'] == list(augmentation.color_map)
assert all(augmentation.color_map[c] == c for c in range(10) if c not in present)

def play(answers):
    game = ArcGame(spec_data=augmented)
    view = game.start_game()
    # The solver sees only augmented grids.
    for (question, answer), pair in zip(view.given, augmented.given, strict=True):
        assert question == pair.question and answer == pair.answer
    assert view.questions == tuple(pair.question for pair in augmented.to_answer)
    game.execute_action(Submission(attempts=tuple((a, a) for a in answers)))
    game.finish_game()
    return game.game_run

expected = [pair.answer for pair in augmented.to_answer]
original = [pair.answer for pair in task.to_answer]
right = play(expected)
assert right.final_score == 100.0 and all(p.correct for p in right.result.pairs)
# The answers of the unaugmented puzzle are wrong in augmented space.
wrong = play(original)
assert wrong.final_score == 0.0 and not any(p.correct for p in wrong.result.pairs)
# The worker's reward reads TAAF's per-output scoring of the standing answer.
for run, correctness in ((right, 1.0), (wrong, 0.0)):
    report = {'state': run.state, 'steps': []}
    enrich_arc2_result(
        report, benchmark_run=run.to_json_dict(),
        test_output_count=len(expected), malformed_call_penalty=0.05,
        max_episode_penalty=None,
    )
    assert report['arc2_outcome']['correctness'] == correctness
print('augmented task scored in augmented space')
"""
    _run(program, tmp_path, [ROOT / "arc2-platypus/packages/taaf/src", ROOT])


@pytest.mark.skipif(not HAS_ARC2_DATA, reason="Requires the private ARC-2 submodule")
def test_worker_builds_augmented_games_and_the_duck_renders_them(tmp_path):
    """The worker's ArcGames carry one augmentation per puzzle; the Duck's
    prompt images are renders of the augmented grids."""
    pytest.importorskip("vllm")
    program = """
from harness_support.arc2.worker import build_benchmark
from training.config import ExperimentConfig, resolve_context_contract
from duck import images
import ullm.conv

config = ExperimentConfig.load(Path(sys.argv[2]) / 'configs/grpo_6gpu_arc2_overfit.json')
context = resolve_context_contract(config.inference)
other = next(t.game_id for t in load('arc2-eval') if t.game_id != puzzle)
spec.update(
    dataset='arc2-eval', games=[puzzle] * 3 + [other], slot_seeds=list(range(4)),
    max_concurrent=4, max_runtime_s_per_game=900, rng_seed=7,
    base_url='http://127.0.0.1:1', served_model='policy',
    tokenizer_id='Qwen/Qwen3.8-27B', model_revision='pinned-revision',
    max_completion_tokens=context.max_completion_tokens,
    context_window_tokens=context.context_window_tokens,
    max_summary_tokens=context.max_summary_completion_tokens,
    max_served_sequence_tokens=context.max_served_sequence_tokens,
    summary_forbidden_token_ids=[248058],
    compact_to_tokens=context.reset_prompt_ceiling_tokens,
    safety_margin_tokens=context.safety_margin_tokens,
    max_episode_tokens=None, reasoning_effort='medium',
    compaction_trigger_tokens=context.compaction_trigger_tokens,
    sampling={'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
)
benchmark = build_benchmark(spec)
games = benchmark.games
assert [type(g) for g in games] == [ArcGame] * 4
slots = [puzzle + '#s0', puzzle + '#s1', puzzle + '#s2', other + '#s3']
assert [g.spec_data.game_id for g in games] == slots
# One augmentation for the puzzle's group, another for the other puzzle.
for game in games[:3]:
    for before, after in zip(task.pairs, game.spec_data.pairs, strict=True):
        assert same(after.question, augmentation.apply(before.question.data))
        assert same(after.answer, augmentation.apply(before.answer.data))
records = [applied_record(spec, slot) for slot in slots]
assert records[0] == records[1] == records[2] and records[0]['puzzle'] == puzzle
assert records[3]['puzzle'] == other and records[3]['seed'] != records[0]['seed']

# The images the Duck sends are renders of the augmented grids, not of the
# original ones.
agent = benchmark.solver.agent
assert agent.images == 'task+show'
def prompt_images(spec_data):
    prompt = agent._task_prompt(spec_data.view())
    return [
        element.data for element in prompt.content.elements
        if isinstance(element, ullm.conv.ImageContent)
        and not element.alt_text.startswith('ARC colour legend')
    ]
shown = prompt_images(games[0].spec_data)
by_hand = GameSpec(game_id='by-hand', pairs=tuple(
    replace(
        pair,
        question=Grid(augmentation.apply(pair.question.data)),
        answer=Grid(augmentation.apply(pair.answer.data)),
    )
    for pair in task.pairs
))
assert shown
assert all(np.array_equal(a, b) for a, b in zip(shown, prompt_images(by_hand), strict=True))
assert not any(
    a.shape == b.shape and np.array_equal(a, b)
    for a, b in zip(shown, prompt_images(task), strict=True)
)
# A small grid's pixels are the palette colors of its augmented cells.
small = np.array([[0, 1, 2]], dtype=np.int8)
cells = augmentation.apply(small)
assert np.array_equal(images.render(Grid(cells)), images.PALETTE[cells])
assert not np.array_equal(images.render(Grid(cells)), images.render(Grid(small)))
print('worker games, records and Duck images are augmented')
"""
    result = subprocess.run(
        [sys.executable, "-c", SETUP + program, str(tmp_path), str(ROOT)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, HARNESS_PATHS))},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
