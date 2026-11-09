"""The address-space cap a search worker runs under.

It is the backstop for model-written code: the planner's own memory check
runs between expansions, so an allocation inside one reaches the machine
before anything looks. A worker that ran without the cap would have no second
line, and would fail somewhere else — an OOM kill, or a node taken out from
under the other games on it. So the cap is verified rather than attempted.
"""

from __future__ import annotations

import resource
from typing import Any

import pytest

from seal.search_process import _cap_address_space


def test_the_cap_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary path: asked for, taken, and read back as taken."""
    applied: list[tuple[int, int]] = []

    def record(which: int, pair: tuple[int, int]) -> None:
        del which
        applied.append(pair)

    monkeypatch.setattr(resource, "setrlimit", record)

    def report_4096(which: int) -> tuple[int, int]:
        del which
        return (4096, 4096)

    monkeypatch.setattr(resource, "getrlimit", report_4096)

    _cap_address_space(4096)

    assert applied == [(4096, 4096)]


def test_a_refused_cap_stops_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine that will not take the cap must not run model code anyway."""

    def refuse(*_args: Any) -> None:
        raise OSError("not permitted")

    monkeypatch.setattr(resource, "setrlimit", refuse)

    with pytest.raises(RuntimeError, match="could not cap its address space"):
        _cap_address_space(4096)


def test_a_silently_clamped_cap_stops_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """``setrlimit`` can succeed on a value the kernel then clamps.

    Reporting success while running under a ceiling nobody chose is the one
    outcome worse than refusing: everything downstream claims a bound that is
    not the one in force.
    """

    def accept(which: int, pair: tuple[int, int]) -> None:
        del which, pair

    monkeypatch.setattr(resource, "setrlimit", accept)

    def report_1024(which: int) -> tuple[int, int]:
        del which
        return (1024, 1024)

    monkeypatch.setattr(resource, "getrlimit", report_1024)

    with pytest.raises(RuntimeError, match="the process reports 1024"):
        _cap_address_space(4096)
