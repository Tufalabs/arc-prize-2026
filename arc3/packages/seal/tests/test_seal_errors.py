"""The rules that decide what a Python error looks like to the model.

Covered here rather than through a run because each rule is a decision about
one shape of traceback, and the shapes are cheaper to state than to provoke:
a chained exception, a harness-only stack, an address in a repr. The shape
the model actually received is pinned by the scripted surface reference.

Ported from the whale's ``test_errors.py``; the frames are the seal's.
"""

from __future__ import annotations

import os

from seal import errors

MODEL_FRAME = """  File "<repl>", line 6, in <module>
    return _apply(state, move)
"""
HARNESS_FRAME = """  File "/checkout/packages/seal/src/seal/harness/_repl_state.py", line 120, in install_state
    history._append_turn(entry)
"""
LIBRARY_FRAME = """  File "/venv/lib/numpy/core/_methods.py", line 47, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
"""


#: This package's own files, by path: the search's model wrapper carries no
#: name the harness list knows.
SEAL_DIR = os.path.dirname(os.path.abspath(errors.__file__))
SEARCH_FRAMES = (
    f'  File "{SEAL_DIR}/model.py", line 229, in step\n'
    "    prediction = self._step(state, action)\n"
    f'  File "{SEAL_DIR}/py_model.py", line 135, in _step\n'
    '    self._ns["do_move"](state, harness_game._move(model_move(action)))  # pyright: ignore[reportPrivateUsage]\n'
)
#: The model's helper module, in its working directory, holding do_move.
HELPER_FRAMES = (
    '  File "/work/seal-model-x/wm2.py", line 220, in do_move\n'
    "    pads = pad_p_set(state)\n"
    '  File "/work/seal-model-x/wm2.py", line 148, in pad_p_set\n'
    "    if is_pad(st, _px_cell(r, c)):\n"
)


def _traceback(*frames: str, exception: str = "KeyError: 'pos'") -> str:
    """A rendered traceback built from frame blocks, outermost first."""
    return "Traceback (most recent call last):\n" + "".join(frames) + exception


def test_the_frames_that_called_the_model_are_replaced_by_one_boundary() -> None:
    """Our function name, line number and source text are all prompt material
    otherwise — which is what froze the harness's decomposition, and what made
    the model read a chain starting inside a file it may not open."""
    out = errors.for_model(
        _traceback(HARNESS_FRAME, MODEL_FRAME),
        where="prepend",
        doing="pulling this level's state",
    )

    assert "install_state" not in out
    assert "history._append_turn" not in out
    assert 'File "<harness>", in prepend' in out
    assert "pulling this level's state" in out
    assert 'File "<repl>", line 6, in <module>' in out


def test_the_frames_the_model_called_are_kept() -> None:
    """Direction, not filename: a library the model called is as informative
    as any other, and dropping it would hide where its own call went wrong."""
    out = errors.for_model(_traceback(HARNESS_FRAME, MODEL_FRAME, LIBRARY_FRAME))

    assert "_amin" in out, "a frame below the model was dropped"
    assert 'File "_methods.py", line 47' in out


def test_a_traceback_with_no_model_frame_still_collapses_to_the_boundary() -> None:
    """The model's own code does not always reach the stack.

    The housekeeping prepended to a cell can raise before the model's code
    runs, so there is no model frame to normalise against — and passing the
    traceback through whole would hand the model our file, our line, our
    function name and the source text of our call. The exception line is
    the entire diagnosis; when the fault really is ours, the raw traceback
    is still in the run's spans.
    """
    out = errors.for_model(
        _traceback(HARNESS_FRAME, exception="RuntimeError: Cannot modify history."),
        where="prepend",
        doing="pulling this level's state",
    )

    assert "_repl_state.py" not in out, "our filename reached the model"
    assert "install_state" not in out, "our function name reached the model"
    assert "history._append_turn" not in out, "the source text of our call reached the model"
    assert 'File "<harness>", in prepend' in out
    assert "pulling this level's state" in out
    assert "RuntimeError: Cannot modify history." in out, "the diagnosis was lost"


def test_a_chained_exception_keeps_its_join() -> None:
    """Both segments are normalised, and the sentence explaining how they
    relate survives — without it the second traceback reads as unexplained."""
    joined = (
        _traceback(HARNESS_FRAME, MODEL_FRAME, exception="KeyError: 'pos'")
        + "\n\nDuring handling of the above exception, another exception occurred:\n\n"
        + _traceback(HARNESS_FRAME, MODEL_FRAME, exception="ValueError: bad state")
    )

    out = errors.for_model(joined)

    assert "During handling of the above exception" in out
    assert "install_state" not in out
    assert out.count('File "<repl>"') == 2


def test_a_path_inside_a_message_is_shortened_too() -> None:
    """The frame-level pass never sees this one: the path is in the exception
    text. Its ``mkdtemp`` segment differs between two runs on the same
    machine, so a reference recording it reproduces nowhere."""
    out = errors.for_model(
        _traceback(
            MODEL_FRAME,
            exception="ImportError: cannot import name 'MAZE' from 'solver' (/tmp/ullm_python_ab12/solver.py)",
        )
    )

    assert "/tmp/ullm_python_ab12" not in out
    assert "from 'solver' (solver.py)" in out


def test_an_object_address_is_masked() -> None:
    """Per-process, so a recording quoting one reproduces nowhere — including
    on the machine that recorded it."""
    out = errors.for_model(
        _traceback(MODEL_FRAME, exception="TypeError: <function f at 0x7f9c1d2e3f40> is not subscriptable")
    )

    assert "0x7f9c1d2e3f40" not in out
    assert "<function f> is not subscriptable" in out


def test_init_files_keep_their_parent() -> None:
    """``numpy/__init__.py`` and ``torch/__init__.py`` would otherwise be the
    same line in a multi-frame stack."""
    assert errors.shorten_path("/venv/lib/numpy/__init__.py") == "numpy/__init__.py"
    assert errors.shorten_path("/venv/lib/torch/__init__.py") == "torch/__init__.py"
    assert errors.shorten_path("/tmp/x/solver.py") == "solver.py"
    assert errors.shorten_path("<repl>") == "<repl>"


def test_no_boundary_is_announced_when_none_was_removed() -> None:
    """A REPL cell that raises in the model's own code has nothing above it.

    Announcing a boundary there would describe a call that never happened —
    and it would land on every plain REPL error the model makes, which is
    most of them.
    """
    out = errors.for_model(_traceback(MODEL_FRAME, LIBRARY_FRAME), where="prepend")

    assert "<harness>" not in out
    assert out.startswith("Traceback (most recent call last):")


HARNESS_BELOW = """  File "game.py", line 218, in ascii_grid
"""
PREPEND_FRAME = """  File "<harness-setup>", line 10, in <module>
"""
LOADER_FRAME = """  File "_repl_state.py", line 160, in refresh_state
"""


def test_a_harness_frame_below_the_model_keeps_only_its_public_name() -> None:
    """The model called it, so which call went wrong is worth saying — but
    our file, our line and the source CPython echoes from it are not."""
    out = errors.for_model(_traceback(MODEL_FRAME, HARNESS_BELOW, exception="ValueError: expects a 2D array"))

    assert 'File "game.py"' not in out
    assert "line 218" not in out
    assert 'File "<harness>", in ascii_grid' in out


def test_our_own_prepend_is_not_mistaken_for_the_model() -> None:
    """The housekeeping runs inside the cell, so before it was compiled under
    its own name every frame it produced read ``<repl>`` — indistinguishable
    from the model's code, which kept the whole harness stack below it and
    echoed the private store our state pull goes through.
    """
    out = errors.for_model(_traceback(PREPEND_FRAME, LOADER_FRAME, MODEL_FRAME, exception="SyntaxError: bad"))

    assert "<harness-setup>" not in out
    assert 'File "_repl_state.py"' not in out
    assert "refresh_state" not in out, "a harness name reached the model"
    assert 'File "<repl>", line 6, in <module>' in out, "the model lost its own frame"


def test_a_guess_at_our_own_api_gets_the_documentation() -> None:
    """The model cannot read our source, so a wrong attribute is a dead end
    unless the error says what the right ones are.

    Rendered from the harness's own ``help``, which is what the model would
    read had it asked — so the hint cannot drift from the docs, and it
    carries what ``dir()`` cannot: the access pattern in the docstring, and
    none of the dict mutators ``DiffSparse`` inherits.
    """
    out = errors.for_model(
        _traceback(MODEL_FRAME, exception="AttributeError: 'DiffSparse' object has no attribute 'changes'")
    )

    assert "Hint: DiffSparse has no 'changes'" in out
    assert "diff_sparse[(0, 3)]" in out, "the access pattern the model was guessing at is missing"
    assert ".total_pixels" in out
    for mutator in ("update", "popitem", "setdefault", "fromkeys"):
        assert mutator not in out, f"the hint advertises {mutator}, which would corrupt the diff"


def test_no_hint_for_what_is_not_ours() -> None:
    """A library's complaint is the model's business; answering everything
    would be noise."""
    out = errors.for_model(
        _traceback(MODEL_FRAME, exception="AttributeError: 'numpy.ndarray' object has no attribute 'grid'")
    )

    assert "Hint:" not in out


def test_a_hinted_text_is_not_hinted_again() -> None:
    """The tool normalises one cell's output twice — before the cap cuts,
    and once more on the way out — so the second pass must recognise the
    hint the first one added rather than append it again."""
    once = errors.for_model(
        _traceback(MODEL_FRAME, exception="AttributeError: 'Frame' object has no attribute 'array'")
    )
    assert once.count("Hint: Frame has no 'array'") == 1

    twice = errors.for_model(once)

    assert twice == once


def test_the_seal_surface_classes_get_the_hint() -> None:
    """``Node`` and ``Segmentation`` are the seal's additions to the whale's
    surface; a guess at either is answered the same way as at ``Turn``."""
    out = errors.for_model(_traceback(MODEL_FRAME, exception="AttributeError: 'Node' object has no attribute 'cells'"))

    assert "Hint: Node has no 'cells'" in out
    assert ".pixels" in out and "(N, 2)" in out


def test_a_cell_that_fails_to_compile_is_still_the_models_file() -> None:
    """CPython names a compile-time error's file ``<unknown>``; every other
    frame of the model's says ``<repl>``, and nine such tracebacks in the
    first post-rebuild audit run read as coming from somewhere else."""
    out = errors.for_model('  File "<unknown>", line 6\n    x = (1,\n         ^\nSyntaxError: unterminated string')

    assert 'File "<repl>", line 6' in out
    assert "<unknown>" not in out


def test_the_kernel_workers_frames_below_a_host_call_are_ours() -> None:
    """A timeout inside ``submit_action`` unwinds through ullm's worker and
    the stdlib below it; the worker's file and private function are ours."""
    out = errors.for_model(
        _traceback(
            MODEL_FRAME,
            '  File "game.py", line 780, in submit_action\n',
            '  File "/venv/ullm/tools/python_worker.py", line 56, in host_call\n',
            '  File "/usr/lib/python3.12/base64.py", line 88, in b64decode\n',
            exception="KeyboardInterrupt",
        )
    )

    assert "python_worker.py" not in out and "line 56" not in out
    assert 'File "<harness>", in submit_action' in out


def test_a_quoted_path_in_a_message_is_shortened_too() -> None:
    """A module's default repr quotes its file; the first full run printed
    one, job id and checkout name included."""
    out = errors.clean_text("<module 'solver_l4' from '/scratch/jobs/2314137/x/tmp/ullm_python_8bzq/solver_l4.py'>")

    assert out == "<module 'solver_l4' from 'solver_l4.py'>"


def test_a_harness_file_named_in_a_message_is_ours_too() -> None:
    """An import of a name the cell has not defined yet names the defining
    file of ``__main__``, which is the kernel worker's — a file of ours in a
    message rather than a frame, where shortening has nothing to shorten."""
    out = errors.clean_text("ImportError: cannot import name 'segment' from '__main__' (python_worker.py)")

    assert out == "ImportError: cannot import name 'segment' from '__main__' (<harness>)"


def test_the_models_own_file_in_a_message_keeps_its_name() -> None:
    """The same shape over a file the model wrote: it can open that one, so
    the name is the most useful part of the message."""
    out = errors.clean_text("ImportError: cannot import name 'do_move' from 'world_model' (world_model.py)")

    assert "world_model.py" in out and "<harness>" not in out


def test_a_model_file_other_than_world_model_is_the_models_own() -> None:
    """The search calls ``do_move`` straight from its own wrapper, and a
    ``world_model.py`` that imports ``do_move`` from a helper puts no frame of
    its own in between. Our frames go however our files are named, and the
    helper's stay, as a file in the model's working directory."""
    tb = _traceback(SEARCH_FRAMES, HELPER_FRAMES)
    out = errors.for_model(tb, where="world model", model_dir="/work/seal-model-x")

    assert "_step" not in out and "pyright" not in out and "model.py" not in out
    assert 'File "<harness>", in world model' in out
    assert 'File "wm2.py", line 220, in do_move' in out and "pad_p_set" in out
    # The same, with the directory set around the call rather than passed.
    with errors.model_directory("/work/seal-model-x"):
        assert errors.for_model(tb, where="world model") == out
