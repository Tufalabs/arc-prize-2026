"""Configuration values shared across the duck's runtime layers."""

from __future__ import annotations

from enum import StrEnum


class CheckMode(StrEnum):
    """How ``check(fn)`` participates in solving and submission."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    MANDATORY = "mandatory"

    @property
    def enabled(self) -> bool:
        """Whether the kernel exposes ``check(fn)``."""
        return self is not CheckMode.DISABLED

    @classmethod
    def validated(cls, value: object) -> CheckMode:
        """The value as a ``CheckMode``, rejecting anything that is not one."""
        if not isinstance(value, cls):
            raise ValueError(f"check_mode must be a CheckMode; got {value!r}")
        return value


__all__ = ["CheckMode"]
