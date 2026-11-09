"""What the model can see of the runtime, and what it reads when things break.

The other seal references pin what the agent *does*. This one pins what the
agent's surface *looks like from inside the REPL* — the module names the
model can import, what `help()` renders, the text of a Python error it
caused, and what the kernel answers when it reaches past what it is given.
All of that is prompt material the model acts on, and none of it was
covered by a committed reference before: a rename, a leaked internal type or
a changed traceback shape could all land green. The whale's scripted-surface
reference, for the seal's kernel.

One cell per way the seal's kernel differs from the duck's:

- introspection — `import runtime.game`, `dir()`, `help()`, and reprs;
- real Python — the standard library imports, `open()` in the working
  directory, a `class` in a cell, a trailing expression echoed;
- what is withheld — our source through `inspect` and `open`, a write
  outside the working directory;
- the whale's `Frame` (`.grid`, `.diff_sparse`, `.color_counts`) and
  the seal's `Node` with array pixels, including the two guesses the audit
  run found the model making (dict access on a node, `.array` on a frame);
- a traceback through the model's own function into ours, the output cap,
  cwd drift, and the `import submit_action` mistake;
- functions defined in one cell and read back from later ones: their source
  through `inspect`, a traceback through them from a longer cell, one the
  model prints itself, a library's own chained error, and a redefinition;
- acting, and `history` starting again at the next level;
- two modules the model authored: one reading the cell's preloaded names at
  import and in a function, one acting from inside a function and reading
  the moved board on its next line.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_surface'
"""

from __future__ import annotations

import re
from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from test_seal_scripted_reference import CornerGame

RNG_SEED = 20260827

# ---------------------------------------------------------------------------
# The cells the scripted model runs. Their OUTPUT is what this reference pins,
# so each one is chosen for a surface, not for realism.
# ---------------------------------------------------------------------------

#: The model-facing module surface: what it can import, what lives there, and
#: the module every class claims. The witness for any rename or eviction.
INTROSPECT = """\
import runtime
import runtime.game
print(sorted(dir(runtime.game)))
print(Frame.__module__, Turn.__module__, Node.__module__)
print(repr(Action('LEFT')), repr(Action('MOUSE', row=3, col=4)))
"""

#: `help()` with no argument, and on the two classes the audit found the
#: model guessing at.
HELP = """\
help()
help(Frame)
help(Node)
"""

#: Real Python: imports the duck's allowlist refused, a file written and read
#: back with `open()`, a class defined in a cell, and a trailing expression
#: echoed notebook-style.
REAL_PYTHON = """\
import time, importlib, sys, subprocess
open('notes.txt', 'w').write('kept\\n')
print(sorted(os.listdir('.')), open('notes.txt').read().strip())
class Probe:
    pass
Probe.__name__
"""

#: What is withheld, through every entry point: `inspect` on a class, `open`
#: on our file, and a write outside the working directory.
WITHHELD = """\
import inspect
print(inspect.getsource(Turn))
print(open(Frame.__init__.__code__.co_filename).read()[:20])
open(os.path.join('..', 'seal_surface_probe.txt'), 'w').write('x')
print(os.path.exists(os.path.join('..', 'seal_surface_probe.txt')))
"""

#: The whale's frame, on the level's first turn (no diff yet), and the
#: seal's segmentation with array-valued pixels.
FRAME_AND_NODES = """\
f = history[-1].frame
print(f.grid.dtype, f.grid.shape, f.step, f.level, f.diff_sparse)
print(f.color_counts)
seg = segment(f.grid)
print(seg)
n = seg[0]
print(n.pixels.shape, n.pixels.dtype, n.boundary.shape, n.children)
n
"""

#: The two guesses the audit run found in every game: a node read as a dict,
#: and a frame's board read as `.array`. Each is answered with the surface.
GUESS_NODE = "seg = segment(history[-1].frame.grid)\nseg[0]['pixels']\n"
GUESS_FRAME = "history[-1].frame.array\n"

#: The model's own function calling into ours with a bad argument: its frames
#: keep their lines, ours collapse to the boundary naming what it called.
TRACEBACK = """\
def f():
    return ascii_grid(np.zeros((2, 2, 2)))
f()
"""

#: Definitions for the cells after it. Each cell compiles under a name of its
#: own, so a function keeps its source once later cells have run, and every
#: one below reads its own lines rather than the newest cell's.
DEFINE = """\
def area(w, h):
    \"\"\"Width times height.\"\"\"
    return w * h
class Shape:
    def perimeter(self, w, h):
        return 2 * (w + h)
double = lambda x: 2 * x
def boom(depth):
    if depth:
        return boom(depth - 1)
    raise ValueError('raised by a function from an earlier cell')
"""

#: A later, shorter cell: the source of a function, a method and a lambda
#: from the cell above, and the line and file the recursive one starts at.
SOURCE = """\
import inspect
print(inspect.getsource(area))
print(inspect.getsource(Shape.perimeter))
print(inspect.getsource(double).strip())
print(inspect.getsourcelines(boom)[1], inspect.getsourcefile(boom))
"""

#: A later, longer cell raising through the earlier function: each of its
#: frames quotes the earlier cell's line, not whatever sits at that line
#: number here.
RAISE_EARLIER = "".join(f"step_{i} = {i}\n" for i in range(12)) + "boom(2)\n"

#: A traceback the model prints itself rather than one the kernel formats.
PRINTED = """\
import traceback
try:
    boom(0)
except ValueError:
    traceback.print_exc()
"""

#: A library's own chained error: `inspect` re-raises an IndexError as
#: OSError, so the first half of the chain holds only `inspect`'s frames.
CHAINED = """\
late = lambda: 0
late.__code__ = late.__code__.replace(co_firstlineno=999)
inspect.getsource(late)
"""

#: Redefined in a later cell: the source follows the newest definition.
REDEFINE = "def area(w, h):\n    return w * h * 1\nprint(inspect.getsource(area))\n"

#: More than the cap: the head and the tail survive, the middle is elided.
CAP = "print('row ' * 3000)\n"

#: cwd drift, and the note the next cell opens with; the file is visible
#: again only because the cwd came back.
DRIFT = "os.chdir('..')\nprint('moved')\n"
AFTER_DRIFT = "print(os.path.exists('notes.txt'))\n"

#: The import mistake the duck's kernel papered over.
IMPORT_MISTAKE = "import submit_action\n"

#: Acting, then reading what it changed while still in the cell.
ACT = """\
r = submit_action([Action('UP')])
print(r['executed_actions'], len(history))
print(history[-1].frame.diff_sparse)
history[-1]
"""

#: Two authored modules. `probe` reads the preloaded names at import time and
#: in a function -- the shape the audit run lost a call to, as a NameError.
#: `mover` acts from inside a function and reads the board and history on
#: the lines after, which only works if `history` moves inside the module.
PROBE_SOURCE = (
    "AT_IMPORT = (len(history), history[-1].frame.level, len(segment(history[-1].frame.grid)))\n"
    "def where():\n"
    "    return len(history), int(history[-1].frame.grid[0, 0]), available_actions\n"
)
MOVER_SOURCE = (
    "def act(name):\n"
    "    before = (len(history), int(history[-1].frame.grid[0, 0]))\n"
    "    result = submit_action([Action(name)])\n"
    "    return before, (len(history), int(history[-1].frame.grid[0, 0]), result['executed_actions'])\n"
)
AUTHOR_MODULES = (
    f"open('probe.py', 'w').write({PROBE_SOURCE!r})\n"
    f"open('mover.py', 'w').write({MOVER_SOURCE!r})\n"
    "import probe\n"
    "print(probe.AT_IMPORT, probe.where())\n"
)
MODULE_ACTS = "import mover, probe\nprint(mover.act('UP'))\nprint(probe.where(), len(history))\n"

#: The next level: `history` has started again, so its one turn has no
#: action and its frame has no diff. Read in its own cell: an acting cell's
#: output ends the game here and is never sent back.
NEW_LEVEL = "print(len(history), history[0].action, history[0].frame.diff_sparse, history[-1].frame.level)\n"


def _python(code: str) -> ullm.conv.LLMResponse:
    return tool_call("python", {"code": code})


def _surface_script() -> list[ullm.conv.LLMResponse]:
    return [
        # decision 1 (level 1): every surface cell, then one UP.
        _python(INTROSPECT),
        _python(HELP),
        _python(REAL_PYTHON),
        _python(WITHHELD),
        _python(FRAME_AND_NODES),
        _python(GUESS_NODE),
        _python(GUESS_FRAME),
        _python(TRACEBACK),
        _python(DEFINE),
        _python(SOURCE),
        _python(RAISE_EARLIER),
        _python(PRINTED),
        _python(CHAINED),
        _python(REDEFINE),
        _python(CAP),
        _python(DRIFT),
        _python(AFTER_DRIFT),
        _python(IMPORT_MISTAKE),
        _python(ACT),
        # decision 2 (level 1): the authored modules; the second UP, from
        # inside a module, wins the level.
        _python(AUTHOR_MODULES),
        _python(MODULE_ACTS),
        # decision 3 (level 2): history restarted; DOWN, DOWN wins the game.
        _python(NEW_LEVEL),
        _python("submit_action([Action('DOWN')] * 2)\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
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
        # The first decision spends eighteen non-acting calls before it acts.
        max_llm_calls_per_move=24,
        # Below what the scenario's long output prints, so the cut shows.
        tool_output_tokens=2425,
        # Acts without a world model, which the default gate refuses.
        use_world_model="none",
    )
    return taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


SURFACE = ReferenceCase(
    name="seal_scripted_surface",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_surface_reference_replays_and_every_cell_showed_its_surface(tmp_path: Path) -> None:
    (ref,) = await replay_case(SURFACE, tmp_path)
    text = ref.all_request_text()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1", "ACTION1", "ACTION2", "ACTION2"]

    # Introspection: the presented module, and every class claims it.
    assert (
        "'Frame', 'History', 'Node', 'Segmentation', 'Turn', 'Unknown', "
        "'WorldModelRequired', 'WorldModelStop', 'ascii_grid', 'color_names', 'from_ascii_grid', 'help', 'segment'"
        in text
    )
    assert "runtime.game runtime.game runtime.game" in text
    assert "Action('LEFT') Action('MOUSE', row=3, col=4)" in text
    assert "Available types: Frame, Turn, History, Action, DiffSparse, ColorCounts, Segmentation, Node" in text
    # Real Python: the imports, the file, the class, the echo.
    assert "['notes.txt'] kept" in text
    assert "\n'Probe'" in text
    # Withheld: our source through both entry points, and the write outside.
    assert "[Source code for the runtime's own internals is not available." in text
    assert "[Writes outside your working directory are not allowed." in text
    assert "Source code access denied" in text
    # The whale's frame and the seal's nodes.
    assert "int8 (64, 64) 0 1 None" in text
    assert "ColorCounts(" in text
    assert "Segmentation(" in text and "(4096, 2) int64 (4, 2) []" in text
    assert "Node(id=0, color=" in text
    # The two guesses, each answered with the surface.
    assert "Node is not a dict: read node.pixels rather than node['pixels']. See help(Node)." in text
    assert "'Frame' object has no attribute 'array'. Available: color_counts, diff, diff_sparse, grid." in text
    assert "Hint: Frame has no 'array'. Its documented surface:" in text
    # The traceback: the model's frames with lines, ours as the boundary.
    assert 'File "<repl>", line 3, in f' in text
    assert 'File "<harness>", in ascii_grid' in text
    assert "ValueError: ascii_grid expects a 2D array, got 3D." in text
    # Earlier cells' functions, read from later ones: their own source...
    assert 'def area(w, h):\n    """Width times height."""\n    return w * h\n' in text
    assert "    def perimeter(self, w, h):\n        return 2 * (w + h)\n" in text
    assert "double = lambda x: 2 * x" in text
    # ...the file it was compiled as, under the cell's own name, since a
    # cell's output is passed through as it printed it...
    assert re.search(r"^9 <repl-\d+>$", text, re.MULTILINE)
    # ...a traceback through them quoting their own lines from a longer cell...
    assert re.search(r'File "<repl>", line 11, in boom\n    return boom\(depth - 1\)', text)
    assert re.search(r"in boom\n    raise ValueError\('raised by a function from an earlier cell'\)", text)
    assert not re.search(r"in boom\n    step_", text)
    # ...the same when the model prints it; a cell's own name shows only
    # where the model asked for it (``getsourcefile``, above).
    assert 'File "<repl>", line 4, in <module>\n    boom(0)' in text
    assert {line for line in text.splitlines() if "<repl-" in line} == {
        line for line in text.splitlines() if re.fullmatch(r"9 <repl-\d+>", line)
    }
    # ...a library's chained error left as the library's...
    assert "in findsource\n" in text and "OSError: lineno is out of bounds" in text
    assert 'File "<harness>", in harness\nIndexError' not in text
    # ...and the newest definition's source after a redefinition.
    assert "    return w * h * 1\n" in text
    # The cap keeps both ends.
    cut = re.search(r"\n\.\.\. \[truncated \d+ chars of tool output\] \.\.\.\n", text)
    assert cut is not None
    assert "row" in text[cut.start() - 8 : cut.start()] and "row" in text[cut.end() : cut.end() + 8]
    # cwd drift is restored, with the note.
    assert "!!! Your cwd had drifted away from the working directory" in text
    # The import mistake is a plain module error.
    assert "ModuleNotFoundError: No module named 'submit_action'" in text
    # Acting mid-cell: the new turn is on history, with its diff.
    assert "[Action('UP')] 2" in text
    assert "DiffSparse(" in text
    assert "Turn(action=Action('UP'), outcome='playing')" in text
    # The next level starts history again. Read from the file rather than
    # the request text: the acting call that follows it ends the game, and a
    # turn's own tool results render only once a later request carries the
    # turn as history.
    raw = ref.path.read_text(encoding="utf-8")
    assert "1 None None 2" in raw
    # Authored modules, same reason (the module's UP won the level): the
    # preloaded names at import and in a function...
    assert "(2, 1, 1) (2, 1, ['RESET', 'UP', 'DOWN', 'LEFT'])" in raw
    # ...and an action from inside a module moves them on the next line —
    # here the winning one: the UP won level 1, so the module reads the
    # finished level's history, ending on the winning turn and its win
    # screen (TAAF's green, 14), before its function returns.
    assert "((2, 1), (3, 14, [Action('UP')]))" in raw
    assert "(3, 14, ['RESET', 'UP', 'DOWN', 'LEFT']) 3" in raw

    # Nothing machine-specific reached the model.
    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text and "packages/" not in text


async def test_regenerate_seal_scripted_surface(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend);
    gated because it rewrites a committed file. Validated by immediate
    replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_surface_script(), on_exhausted="raise")
    await regenerate_case(SURFACE, backend=backend, tmp_path=tmp_path / SURFACE.name)
