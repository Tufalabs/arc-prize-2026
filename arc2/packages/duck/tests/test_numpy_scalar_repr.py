"""``numpy_scalar_repr``: how the kernel prints a NumPy scalar inside a
container — what a printed row or coordinate list is made of — decided by
the agent and baked into the kernel prelude."""

from __future__ import annotations

import pytest
from duck import DuckAgent, kernel_code

LEGACY_LINE = "np.set_printoptions(legacy='1.25')"


def test_the_shipped_kernel_prints_plain_scalars() -> None:
    """The default is the quiet form: ``[5, 5, 5]``, not three constructor
    calls. The bare prelude says the same, so nothing that builds one
    without stating the setting changes what the model reads."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).numpy_scalar_repr
        == "legacy"
    )
    assert LEGACY_LINE in kernel_code.prelude(numpy_scalar_repr="legacy")
    assert LEGACY_LINE in kernel_code.prelude()


def test_the_verbose_repr_is_a_reachable_arm_and_nothing_else_is() -> None:
    """``np2`` leaves NumPy 2's own repr in place — a treatment a sweep can
    price — and anything else is refused at both seams."""
    assert LEGACY_LINE not in kernel_code.prelude(numpy_scalar_repr="np2")
    with pytest.raises(ValueError, match="numpy_scalar_repr"):
        DuckAgent(reply_cap_tokens=None, numpy_scalar_repr="plain")  # type: ignore[arg-type] — the refusal under test
    with pytest.raises(ValueError, match="numpy_scalar_repr"):
        kernel_code.prelude(numpy_scalar_repr="plain")


if __name__ == "__main__":
    pytest.main([__file__])
