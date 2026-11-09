"""The matching settings on the host: ``strict_energy``, ``pixel_tolerance`` and
``expose_pixel_tolerance`` as starting values, what the model's setters may
change (a tolerance only where offered), and that the game keeps them.

The setters, the test and the per-move check reading them in a real kernel are
the ``seal_scripted_matching`` reference (``tests/test_seal_scripted_counter.py``).
"""

from __future__ import annotations

import pytest

from seal import SealAgent


def test_a_tolerance_starts_above_zero_only_where_offered() -> None:
    with pytest.raises(ValueError, match="requires expose_pixel_tolerance=True"):
        SealAgent(pixel_tolerance=2, expose_pixel_tolerance=False).prepare_for_backend()
    SealAgent(pixel_tolerance=2, expose_pixel_tolerance=True).prepare_for_backend()
    with pytest.raises(ValueError, match="pixel_tolerance must be a whole number"):
        SealAgent(pixel_tolerance=-1, expose_pixel_tolerance=True).prepare_for_backend()


def test_the_setters_change_the_game_settings_and_the_payload_carries_them() -> None:
    agent = SealAgent(expose_pixel_tolerance=True, pixel_tolerance=1)
    assert agent._state_payload()["matching"] == {  # pyright: ignore[reportPrivateUsage]
        "strict_energy": False,
        "pixel_tolerance": 1,
        "tolerance_offered": True,
    }

    reply = agent._set_matching({"strict_energy": True, "pixel_tolerance": 3})  # pyright: ignore[reportPrivateUsage]
    assert reply == {"matching": {"strict_energy": True, "pixel_tolerance": 3, "tolerance_offered": True}}
    # Kept by the agent, not the kernel: a level's new kernel reads them back.
    agent.budget_level = 2
    assert agent._state_payload()["matching"]["pixel_tolerance"] == 3  # pyright: ignore[reportPrivateUsage]


def test_the_host_refuses_what_the_model_may_not_set() -> None:
    offered, not_offered = SealAgent(expose_pixel_tolerance=True), SealAgent(expose_pixel_tolerance=False)

    assert "no pixel tolerance" in not_offered._set_matching({"pixel_tolerance": 2})["error"]  # pyright: ignore[reportPrivateUsage]
    assert "whole number" in offered._set_matching({"pixel_tolerance": -2})["error"]  # pyright: ignore[reportPrivateUsage]
    assert "whole number" in offered._set_matching({"pixel_tolerance": True})["error"]  # pyright: ignore[reportPrivateUsage]
    assert "True or False" in offered._set_matching({"strict_energy": 1})["error"]  # pyright: ignore[reportPrivateUsage]
    assert "unknown setting" in offered._set_matching({"wildcards": 1})["error"]  # pyright: ignore[reportPrivateUsage]
    mixed = offered._set_matching({"strict_energy": True, "pixel_tolerance": -1})  # pyright: ignore[reportPrivateUsage]
    assert "error" in mixed
    assert offered._matching() == {  # pyright: ignore[reportPrivateUsage]
        "strict_energy": False,
        "pixel_tolerance": 0,
        "tolerance_offered": True,
    }, "a refused call changes nothing"
