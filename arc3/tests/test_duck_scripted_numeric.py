"""The numeric frame API, end to end, with an image attached.

`frame_api="numeric"` is one switch over three model-visible things: `.array`
carries color indices rather than one-character stand-ins, `.ascii` renders
those indices as hex, and `.segment()` reports an integer color. This records
a model using all three, and — because multimodal is on — the palette legend
image that explains the indices, which rides the first prompt of each level
and no other.

The corner that motivates the whole arm is the first scripted call: on a
character board `array == 5` matches nothing and raises nothing, so a model
reasoning in color indices draws a confident wrong conclusion. Here it
selects cells.

Multimodal is deliberately ON, unlike every other duck reference: the palette
image is attached only where an image already travels, so nothing else could
record it.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_duck_scripted_numeric'
"""

from __future__ import annotations

from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from duck import DuckAgent
from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from test_duck_scripted_reference import CornerGame

RNG_SEED = 20260825


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call, in order."""
    return [
        # The point of the arm: an integer comparison selects real cells.
        _python(
            "a = current_frame.array\n"
            "print('dtype', a.dtype)\n"
            "print('fill', int(a[0, 0]))\n"
            "print('matching cells', int((a == a[0, 0]).sum()))\n"
        ),
        # `.ascii` renders the same board as index characters.
        _python("print(current_frame.ascii.splitlines()[0][:16])\n"),
        # Segment colors are indices, so they index the legend directly.
        _python(
            "nodes = current_frame.segment()['nodes']\n"
            "print('colors', sorted({n['color'] for n in nodes}))\n"
            "print('types', sorted({type(n['color']).__name__ for n in nodes}))\n"
        ),
        # Arithmetic the character board cannot do at all.
        _python(
            "import numpy as _np\n"
            "print('unique', _np.unique(current_frame.array).tolist())\n"
            "print('counts', _np.bincount(current_frame.array.ravel().astype('int64'), minlength=16).tolist())\n"
        ),
        # Win level 1, which is what makes the next prompt a new level -- and
        # so the only other prompt the palette legend rides.
        _python("submit_action(['UP', 'UP'])\n"),
        _python("submit_action(['DOWN', 'DOWN'])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The duck on the numeric arm, with the board image switched on."""
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        frame_api="numeric",
        # Stated rather than inherited: this recording pins the pairing the
        # numeric arm was measured under. The class default is the verbose
        # form.
        numpy_scalar_repr="legacy",
        multimodal_context="current_grid",
        color_name_legend=False,  # the index legend is this arm's, not the colour names
        completion_cutoff_notice=True,
        expose_undo=True,
        max_llm_calls_per_move=8,
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


NUMERIC = ReferenceCase(
    name="duck_scripted_numeric",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_numeric_replays_and_covers_its_corners(tmp_path: Path) -> None:
    """The committed reference replays bit-exact, and every steered corner is
    still in it — so a future regeneration cannot quietly lose one."""
    (ref,) = await replay_case(NUMERIC, tmp_path)
    text = ref.all_request_text()

    assert ref.action_ids() == ["ACTION1", "ACTION1", "ACTION2", "ACTION2"]
    assert ref.expected["state"] == "won"

    # The array is integers, and an integer comparison finds cells.
    assert '"dtype int8' in text
    assert "matching cells 4096" in text
    # `.ascii` is index characters, not colour letters.
    assert '"0000000000000000' in text
    # Segment colours are ints.
    assert "'types', sorted" in text  # the snippet asked
    assert "types ['int']" in text
    # Arithmetic that the character board cannot do at all.
    assert "unique [0]" in text
    # The prompt teaches the index/glyph mapping where it stops agreeing.
    assert '10="a" sky blue' in text
    assert "`array == 5` selects every cell of color 5" in text
    # The palette legend rides a prompt, with its caption.
    assert "Palette legend image: one swatch per color index" in text


async def test_the_palette_rides_one_prompt_per_level(tmp_path: Path) -> None:
    """Reference material, not state: attached when a level starts and not
    again until the next one, so it costs pixels twice in a two-level game
    rather than once a turn."""
    (ref,) = await replay_case(NUMERIC, tmp_path)
    captions = ref.all_request_text().count("Palette legend image: one swatch per color index")
    # Every request re-sends the conversation, so the caption count grows with
    # the turns that followed it; what matters is that a later request carries
    # more than one, and that the count per request never grows unboundedly.
    per_request = [ref.request_text(record).count("Palette legend image:") for record in ref.records()]
    assert captions > 0
    assert max(per_request) == 2, per_request  # two levels, two attachments


async def test_regenerate_duck_scripted_numeric(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. The fresh reference is validated by
    immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(NUMERIC, backend=backend, tmp_path=tmp_path)
