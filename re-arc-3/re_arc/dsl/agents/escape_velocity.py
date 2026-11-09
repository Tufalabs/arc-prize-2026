from __future__ import annotations

from ..core import CachedProgramDslAgent

TILE = 4

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6


def _click(cell_x: int, cell_y: int) -> tuple[int, dict[str, int]]:
    return CLICK, {"x": cell_x * TILE + 2, "y": cell_y * TILE + 2}


def _click_belly(cell_x: int, cell_y: int) -> tuple[int, dict[str, int]]:
    # Edge-pixel aim at the cell's bottom edge: names the SOUTH (belly) face,
    # so a bolt arriving from the south-west lands the portal on that face.
    return CLICK, {"x": cell_x * TILE + 2, "y": cell_y * TILE + TILE - 1}


def _simple(action_id: int) -> tuple[int, dict[str, int]]:
    return action_id, {}


def _walk(action_id: int, count: int) -> list[tuple[int, dict[str, int]]]:
    return [_simple(action_id) for _ in range(count)]


LEVEL_PROGRAMS = [
    # Level 1: First Steps (no gun). Lift the box, walk through the
    # pre-placed portals carrying it, weight the button, walk to the goal.
    [
        _simple(RIGHT),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(DOWN),
        _simple(DOWN),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(SPACE),
        *_walk(RIGHT, 4),
    ],
    # Level 1: Two Gaps. Blue on the block's west face, orange on the west
    # wall, walk through; around the block, relocate orange to the block's
    # east face and blue to the goal brick, walk through onto the exit glow.
    [
        *_walk(RIGHT, 2),
        _click(8, 7),
        _simple(SPACE),
        _click(0, 7),
        *_walk(LEFT, 4),
        *_walk(UP, 2),
        *_walk(RIGHT, 3),
        *_walk(DOWN, 2),
        _click(9, 7),
        _simple(SPACE),
        _click(15, 7),
        _simple(LEFT),
    ],
    # Level 3: Special Delivery. Antechamber booster ride teaches pads over
    # water; then the classic: block shot from the water's edge, anchor on
    # the west wall, pad ride to the island, lift the box, post home, weight
    # the button, through the gate to the goal.
    [
        *_walk(DOWN, 2),
        *_walk(DOWN, 3),
        _simple(RIGHT),
        _simple(DOWN),
        _click(7, 7),
        _simple(SPACE),
        *_walk(DOWN, 4),
        _click(0, 12),
        *_walk(LEFT, 2),
        _simple(DOWN),
        _simple(UP),
        _simple(LEFT),
        _simple(LEFT),
        _simple(SPACE),
        _simple(RIGHT),
        *_walk(UP, 5),
        _simple(LEFT),
        _click(11, 1),
        _simple(LEFT),
        _simple(UP),
        *_walk(LEFT, 4),
        _simple(SPACE),
        *_walk(LEFT, 3),
        _simple(LEFT),
    ],
    # Level 4: Loading Dock. Beat 1 — escape the jail. The player is penned in
    # by dark (non-portalable) walls and a water moat; the ONLY portalable
    # surface in reach is the block (7,5) across the moat. Build an elbow on it
    # by striking two ADJACENT faces (blue on its south belly via a bottom-edge
    # shot from the SW, orange on its west face head-on), then ride the pen's
    # pad through the elbow — the flight turns the corner over the water and
    # drops you on the work platform. Beat 2 repeats the trick with the box
    # (caged behind dark walls so it can't be shot from the jail) as artillery:
    # rebuild the elbow on the east wall (14,5), feed the box to the pad, and it
    # sails over the water and drops onto the button. Gate opens, goal above.
    [
        # Beat 1 — adjacent-face elbow on the one block, ride the pad through it.
        _click_belly(7, 5),  # blue on the block's south belly (from the SW)
        _simple(SPACE),  # arm orange
        *_walk(UP, 3),  # (2,8) -> (2,5)
        _click(7, 5),  # orange on the block's west face (straight east)
        _simple(LEFT),  # step onto the pad -> flight elbow -> escape to (7,9)
        # Beat 2 — rebuild the elbow on the east wall, fire the box as artillery.
        *_walk(RIGHT, 2),
        *_walk(UP, 4),  # (7,9) -> (9,5)
        _click(14, 5),  # orange on the east wall's west face
        _simple(SPACE),  # arm blue
        *_walk(DOWN, 4),
        _simple(RIGHT),  # (9,5) -> (10,9)
        _click_belly(14, 5),  # blue on the east wall's south belly
        _simple(LEFT),
        *_walk(UP, 2),  # (10,9) -> stand on the box (9,7)
        _simple(SPACE),  # lift the box
        *_walk(UP, 2),  # carry to (9,5), pad due east
        _simple(SPACE),  # feed the box: flight -> elbow -> button -> gate
        *_walk(UP, 2),  # through the open gate onto the goal
    ],
    # Level 3: Round Trip. Phase 0: pass-through both faces of (2,13) to
    # extract the box from the sealed pocket (shot through the water hole).
    # Then twice over: drop the box as a flight elbow (W+N faces), ride the
    # pad through it, anchor the far post, walk back, evacuate the box's
    # portal onto a wall, lift, carry it through. Finally box on the button.
    [
        # Phase 0 — extract the box from the pocket. The pocket exit lands
        # west of the box whose west face holds the entry portal, so first
        # move that portal onto the pass-through wall, then mount the bare box.
        _simple(DOWN),
        _click(2, 13),
        _simple(SPACE),
        _simple(UP),
        _click(5, 14),
        _simple(DOWN),
        _simple(DOWN),
        *_walk(LEFT, 2),
        _click(2, 13),
        *_walk(RIGHT, 3),
        _simple(SPACE),
        *_walk(LEFT, 3),
        _simple(UP),
        # Phase 1 — elbow ride over water A.
        *_walk(RIGHT, 5),
        _simple(SPACE),
        _simple(UP),
        _click(7, 12),
        _simple(SPACE),
        _simple(LEFT),
        _simple(DOWN),
        _click(7, 12),
        *_walk(LEFT, 4),
        _simple(LEFT),
        # Phase 2 — ferry the box to region 2.
        *_walk(DOWN, 2),
        *_walk(LEFT, 2),
        _click(5, 9),
        _simple(DOWN),
        _simple(SPACE),
        *_walk(LEFT, 6),
        _click(0, 11),
        *_walk(RIGHT, 6),
        _simple(DOWN),
        _simple(SPACE),
        _simple(UP),
        *_walk(LEFT, 6),
        _simple(LEFT),
        # Phase 3 — elbow ride over water B.
        _simple(UP),
        *_walk(RIGHT, 7),
        _simple(SPACE),
        _simple(UP),
        _click(12, 7),
        _simple(SPACE),
        _simple(LEFT),
        _simple(DOWN),
        _click(12, 7),
        _simple(UP),
        *_walk(LEFT, 9),
        _simple(DOWN),
        _simple(LEFT),
        # Phase 4 — ferry the box to region 3.
        *_walk(LEFT, 2),
        _simple(DOWN),
        _click(10, 4),
        _simple(DOWN),
        _simple(SPACE),
        *_walk(LEFT, 7),
        *_walk(DOWN, 2),
        _click(5, 9),
        *_walk(RIGHT, 7),
        _simple(UP),
        _simple(SPACE),
        _simple(UP),
        *_walk(LEFT, 7),
        *_walk(DOWN, 2),
        _simple(DOWN),
        # Phase 5 — box on the button, through the door.
        *_walk(UP, 2),
        _simple(SPACE),
        *_walk(RIGHT, 2),
        _simple(UP),
    ],
    # Level 4: Air Mail. Keyhole shot through the open gate from its button,
    # portal into the east chamber, aim the artillery pair (east wall row 2 ->
    # sea wall row 12), then feed the box onto the pad: it flies through the
    # pair and lands on the islet button, opening the exit door for good.
    [
        *_walk(UP, 3),
        _simple(RIGHT),
        _click(11, 5),
        _simple(SPACE),
        _click(0, 5),
        *_walk(LEFT, 3),
        *_walk(DOWN, 5),
        *_walk(RIGHT, 3),
        _click(15, 12),
        _simple(SPACE),
        *_walk(UP, 8),
        _click(15, 2),
        _simple(DOWN),
        *_walk(LEFT, 3),
        *_walk(DOWN, 5),
        _simple(SPACE),
        *_walk(UP, 6),
        _simple(RIGHT),
        _simple(SPACE),
        *_walk(UP, 2),
    ],
    # Level 5: Return to Sender. Escape the sealed cell through the box
    # (45-degree anchor shot through the water slit + a portal on the box),
    # plant X outside, walk back in through it, lift the box (color-locked),
    # fire Y from inside while carrying, feed the launcher -> the box
    # pinballs Y->X->S2->S3 onto the door button -> then mail yourself after
    # it and land on your own parcel.
    [
        _click(6, 7),
        _simple(SPACE),
        _simple(DOWN),
        _simple(RIGHT),
        _click(3, 13),
        _simple(RIGHT),
        _simple(LEFT),
        *_walk(UP, 6),
        _simple(SPACE),
        _click(1, 2),
        *_walk(LEFT, 3),
        _simple(LEFT),
        _simple(UP),
        _simple(RIGHT),
        _simple(SPACE),
        _simple(DOWN),
        _simple(SPACE),
        _simple(UP),
        _simple(LEFT),
        _click(2, 8),
        _simple(DOWN),
        _simple(SPACE),
        _simple(DOWN),
        _simple(UP),
        _simple(RIGHT),
        _simple(RIGHT),
    ],
    # Level 8: Grand Tour. Keyhole out of the start cell; nook heist (45-degree
    # slit shot onto the nook's inner ceiling face, pair with A1, walk through,
    # lift the second box, walk back out the same portal); park it where the
    # old bounce pad used to be; shielded-anchor crossing to zone C; pocket
    # extraction; elbow ride back; re-entry; post-trick ferry home; then build
    # the elbow (both portals on the parked box, adjacent faces) and feed the
    # first box through the gate, through the elbow, down onto the islet.
    [
        _simple(LEFT),
        _click(9, 2),
        _simple(SPACE),
        _click(0, 2),
        _simple(LEFT),
        *_walk(DOWN, 2),
        *_walk(LEFT, 2),
        *_walk(DOWN, 4),
        _simple(LEFT),
        _click(2, 5),
        _simple(SPACE),
        _simple(RIGHT),
        *_walk(UP, 2),
        _click(5, 6),
        _simple(LEFT),
        _simple(DOWN),
        _simple(SPACE),
        *_walk(UP, 2),
        *_walk(UP, 2),
        *_walk(RIGHT, 5),
        *_walk(UP, 3),
        _simple(RIGHT),
        _simple(SPACE),
        # Detour down column 11 instead of 12: the gate G2 (12,5) sits closed
        # here (nobody is on button (5,2) yet), so skirt it and rejoin at (13,6).
        *_walk(DOWN, 3),
        _simple(LEFT),
        *_walk(DOWN, 2),
        *_walk(RIGHT, 2),
        _simple(SPACE),
        _click(13, 4),
        _simple(RIGHT),
        *_walk(LEFT, 8),
        _click(4, 14),
        _simple(SPACE),
        *_walk(UP, 2),
        _click(3, 14),
        *_walk(DOWN, 2),
        _simple(LEFT),
        _simple(LEFT),
        _simple(DOWN),
        _simple(RIGHT),
        _simple(SPACE),
        _click(4, 14),
        _simple(RIGHT),
        _simple(UP),
        _simple(RIGHT),
        _simple(SPACE),
        _simple(UP),
        _click(6, 13),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(DOWN),
        _click(6, 13),
        _simple(RIGHT),
        _simple(RIGHT),
        *_walk(DOWN, 5),
        _click(5, 6),
        _simple(LEFT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(SPACE),
        _click(9, 11),
        *_walk(LEFT, 3),
        _simple(DOWN),
        _simple(SPACE),
        _simple(UP),
        *_walk(RIGHT, 3),
        _simple(UP),
        *_walk(UP, 2),
        *_walk(RIGHT, 5),
        *_walk(UP, 3),
        _click(12, 1),
        _simple(DOWN),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(SPACE),
        _click(12, 1),
        _simple(LEFT),
        _simple(SPACE),
        *_walk(DOWN, 2),
        *_walk(LEFT, 3),
        _simple(UP),
        *_walk(LEFT, 3),
        _simple(UP),
        _simple(SPACE),
        _simple(DOWN),
        *_walk(RIGHT, 3),
        _simple(DOWN),
        *_walk(RIGHT, 3),
        *_walk(DOWN, 2),
        *_walk(RIGHT, 2),
        _simple(DOWN),
        _simple(RIGHT),
        _simple(RIGHT),
    ],
]


class EscapeVelocityDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        level_idx = int(self._current_level_idx or 0)
        return list(LEVEL_PROGRAMS[level_idx])


AGENT_CLASS = EscapeVelocityDslAgent
