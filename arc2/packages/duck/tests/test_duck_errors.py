"""The rules that decide what a Python error looks like to the model.

Covered here rather than through a kernel because each rule is a decision
about one shape of traceback, and the shapes are cheaper to state than to
provoke. The frames are the duck's: ``check`` and its kin compiled under
``<harness-setup>``, the sandbox's patches and the import machinery under
this package, and the model's own files in its working directory.
"""

from __future__ import annotations

import os

from duck import errors

MODEL_FRAME = """  File "<repl>", line 6, in <module>
    out = solve(test_in[0])
"""
DUCK_DIR = os.path.dirname(os.path.abspath(errors.__file__))
#: A frame of ours above the model's: the sandbox's patched inspect, which a
#: chained library error can carry.
SANDBOX_FRAME = f"""  File "{DUCK_DIR}/sandbox.py", line 181, in _getsource
    return orig_getsource(obj, *args, **kwargs)
"""
CHECK_FRAME = """  File "<harness-setup>", line 240, in check
    raise CheckError(report)
"""
LIBRARY_FRAME = """  File "/venv/lib/numpy/core/_methods.py", line 47, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
"""
WORK = "/tmp/ullm_python_ab12"
CANDIDATE_FRAME = f"""  File "{WORK}/candidate.py", line 3, in fn
    return grid[5]
"""


def _traceback(*frames: str, exception: str = "IndexError: index 5 is out of bounds") -> str:
    """A rendered traceback built from frame blocks, outermost first."""
    return "Traceback (most recent call last):\n" + "".join(frames) + exception


def test_a_harness_frame_below_the_model_keeps_only_its_public_name() -> None:
    """The model called ``check``, so naming it says which call went wrong;
    our pseudo-file, our line and our source say nothing it can use."""
    out = errors.for_model(_traceback(MODEL_FRAME, CHECK_FRAME, exception="CheckError: check: FAILED"))

    assert 'File "<harness>", in check' in out
    assert "<harness-setup>" not in out and "line 240" not in out and "raise CheckError" not in out
    assert 'File "<repl>", line 6, in <module>' in out


def test_a_private_frame_of_ours_below_the_model_is_dropped() -> None:
    """The model can neither call nor read a private name, so the frame
    goes whole; the library frames around it stay."""
    out = errors.for_model(_traceback(MODEL_FRAME, SANDBOX_FRAME, LIBRARY_FRAME, exception="OSError: bad"))

    assert "sandbox.py" not in out and "_getsource" not in out and "orig_getsource" not in out
    assert 'File "_methods.py", line 47' in out


def test_the_frames_that_called_the_model_are_replaced_by_one_boundary() -> None:
    out = errors.for_model(_traceback(SANDBOX_FRAME, MODEL_FRAME))

    assert "_getsource" not in out
    assert 'File "<harness>", in harness' in out
    assert out.index("<harness>") < out.index("<repl>")


def test_a_traceback_with_no_model_frame_still_collapses_to_the_boundary() -> None:
    out = errors.for_model(_traceback(CHECK_FRAME, exception="RuntimeError: no model call active"))

    assert "<harness-setup>" not in out and "raise CheckError" not in out
    assert 'File "<harness>", in harness' in out
    assert "RuntimeError: no model call active" in out


def test_a_library_chain_with_no_frame_of_ours_is_kept() -> None:
    """``inspect`` re-raising an ``IndexError`` as ``OSError``: the first
    half is the library's, and stays."""
    chained = (
        _traceback(LIBRARY_FRAME, exception="IndexError: list index out of range")
        + "\n\nDuring handling of the above exception, another exception occurred:\n\n"
        + _traceback(MODEL_FRAME, SANDBOX_FRAME, LIBRARY_FRAME, exception="OSError: lineno is out of bounds")
    )
    out = errors.for_model(chained)

    assert "During handling of the above exception" in out
    assert out.count('File "_methods.py", line 47') == 2
    assert "_getsource" not in out


def test_a_file_in_the_working_directory_is_the_models_own() -> None:
    out = errors.for_model(_traceback(MODEL_FRAME, CANDIDATE_FRAME), model_dir=WORK)

    assert 'File "candidate.py", line 3, in fn' in out and "return grid[5]" in out
    assert "<harness>" not in out and WORK not in out


def test_no_boundary_is_announced_when_none_was_removed() -> None:
    out = errors.for_model(_traceback(MODEL_FRAME, LIBRARY_FRAME))

    assert "<harness>" not in out
    assert out.startswith("Traceback (most recent call last):")


def test_the_kernel_workers_frames_below_a_host_call_are_ours() -> None:
    """A timeout inside ``check``'s host call unwinds through ullm's worker."""
    out = errors.for_model(
        _traceback(
            MODEL_FRAME,
            CHECK_FRAME,
            '  File "/venv/ullm/tools/python_worker.py", line 56, in host_call\n',
            exception="KeyboardInterrupt",
        )
    )

    assert "python_worker.py" not in out and "line 56" not in out
    assert 'File "<harness>", in check' in out and 'File "<harness>", in host_call' in out


def test_a_path_inside_a_message_is_shortened_too() -> None:
    out = errors.for_model(
        _traceback(
            MODEL_FRAME, exception=f"ImportError: cannot import name 'fn' from 'candidate' ({WORK}/candidate.py)"
        )
    )

    assert WORK not in out and "from 'candidate' (candidate.py)" in out


def test_a_quoted_path_in_a_message_is_shortened_too() -> None:
    assert errors.clean_text(f"<module 'candidate' from '{WORK}/candidate.py'>") == (
        "<module 'candidate' from 'candidate.py'>"
    )


def test_a_harness_file_named_in_a_message_is_ours_too() -> None:
    out = errors.clean_text("ImportError: cannot import name 'solve' from '__main__' (python_worker.py)")

    assert out == "ImportError: cannot import name 'solve' from '__main__' (<harness>)"


def test_an_object_address_is_masked() -> None:
    out = errors.for_model(
        _traceback(MODEL_FRAME, exception="TypeError: <function f at 0x7f9c1d2e3f40> is not subscriptable")
    )

    assert "0x7f9c1d2e3f40" not in out and "<function f> is not subscriptable" in out


def test_init_files_keep_their_parent() -> None:
    assert errors.shorten_path("/venv/lib/numpy/__init__.py") == "numpy/__init__.py"
    assert errors.shorten_path("/tmp/x/candidate.py") == "candidate.py"
    assert errors.shorten_path("<repl>") == "<repl>"
