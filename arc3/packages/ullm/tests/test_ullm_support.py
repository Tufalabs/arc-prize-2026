"""BaseClass runtime type checking: what an unresolvable annotation costs."""

from dataclasses import dataclass
from typing import Optional

import pytest
import typeguard
import ullm.support


@dataclass
class _PartlyResolvable(ullm.support.BaseClass):
    """One field naming a module nobody imported, one ordinary field."""

    # No import of it anywhere, at runtime or under TYPE_CHECKING: this is the
    # shape a TYPE_CHECKING-only annotation takes when its module never loads.
    exotic: Optional["nonexistent_module.Thing"] = None  # type: ignore[name-defined] # noqa: F821
    count: int = 0


def test_one_unresolvable_annotation_does_not_disarm_the_whole_class():
    """The field that cannot resolve goes unchecked; its siblings do not.
    Bailing out of the class wholesale is the failure mode here — it turns
    the one routine whose job is to be loud into a silent pass."""
    ok = _PartlyResolvable(count=3)
    ok.check_constraints()  # resolvable fields are fine, unresolvable one ignored

    wrong = _PartlyResolvable(count="not an int")  # type: ignore[arg-type]
    with pytest.raises(typeguard.TypeCheckError):
        wrong.check_constraints()
