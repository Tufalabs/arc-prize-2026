from __future__ import annotations

GridPos = tuple[int, int]


def toggle_flags(active: frozenset[str], *flags: str) -> frozenset[str]:
    """Toggle one or more named flags in an immutable flag set."""
    out = set(active)
    for flag in flags:
        if flag in out:
            out.remove(flag)
        else:
            out.add(flag)
    return frozenset(out)


def apply_teleport(pos: GridPos, teleports: dict[GridPos, GridPos]) -> GridPos:
    """Return teleported destination for a tile or the original tile if none."""
    return teleports.get(pos, pos)


def is_gate_open(gate_id: str, active_flags: frozenset[str], required_flags: set[str]) -> bool:
    """Gate is open when its id is active or all required flags are active."""
    if gate_id in active_flags:
        return True
    return required_flags.issubset(active_flags)


def passable_with_dynamic_blocks(pos: GridPos, *, static_blocked: set[GridPos], closed_gates: set[GridPos]) -> bool:
    """Simple passability check that combines static and dynamic blocked cells."""
    return pos not in static_blocked and pos not in closed_gates
