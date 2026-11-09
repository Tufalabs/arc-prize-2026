# Mirrored

## Core Game

Mirrored is a discrete grid puzzle about controlling two main blocks at the same
time. Each main block occupies one grid cell. Up and down commands move both
main blocks in the same vertical direction. Left and right commands are mirrored:
the left-side block moves by the commanded horizontal delta, while the
right-side block moves by the opposite horizontal delta.

Each block resolves collision independently. If a command would push one block
into a solid wall, movable blocker, closed gate, or the edge of the grid, that
block stays where it is, but the other block can still move. This independent
pinning is the main puzzle device. The player must use maze walls, gates, and
movable blockers to change the relative positions of the two mirrored blocks.

The level is solved when all active main blocks have met. In these levels there
are always two active main blocks, so the win condition is for them to occupy
the same cell. A horizontal crossing also counts: if two horizontally adjacent
blocks swap places in one mirrored move, they are collapsed onto a shared cell
and treated as having met. Once a pair has met, both blocks become intangible
and the game advances.

## Objects And State

Logical grid coordinates use `(x, y)`, with `(0, 0)` at the top-left.

- `L`: the left-controlled main block. It uses the command's horizontal delta.
- `R`: the right-controlled main block. It uses the opposite horizontal delta.
- `#`: solid colored wall. Main blocks and movable blockers cannot enter it.
- `.`: open black floor.
- `X`: checkered hazard. A main block can move onto it, but doing so triggers a
  reset instead of a win check.
- `B`: clickable movable blocker. It is a one-cell obstacle for the main blocks.
- `G/g`: green gate family. `G` cells are gate bars, `g` cells are switch pads.
- `O/o`: orange gate family. `O` cells are gate bars, `o` cells are switch pads.
- `V/v`: violet gate family. `V` cells are gate bars, `v` cells are switch pads.

The player normally controls the mirrored main blocks. Clicking a blocker
selects that blocker instead; while selected, directional commands move only
that blocker by one cell. Clicking a main block or any non-blocker grid location
returns control to the mirrored pair. Only one blocker can be selected at a
time. Blockers are reset to their starting positions by hazard failure.

Switch pads are single colored cells that the main blocks can stand on. If any
active main block occupies a switch pad of a family, every gate bar of that same
family is removed and becomes passable. If no active main block occupies a
matching switch pad, those gates are solid again. Blockers do not open gates.

Hazards reset the arrangement rather than immediately advancing or solving the
level. After a main block lands on `X`, the hit block flashes for a short
sequence, then every clickable object in the level, including main blocks and
blockers, is returned to its level-start position. The action timer is not
restored. If the action counter exceeds 150, the level is lost. The interface
draws this limit as top and bottom progress bars.

## Level 1

Layout:

```text
cols 01234567890
00   ###########
01   ###......##
02   ##...#....#
03   #...###...#
04   #..#####..#
05   ##.######.#
06   ##.######.#
07   #...####..#
08   #....#....#
09   #..L.#.R..#
10   #....#...##
```

This is the basic mirrored-movement tutorial. The two cyan blocks start near the
bottom, separated by a central wall column. There are no hazards, blockers, or
gates. Up and down preserve their horizontal separation, while left and right
push them away from or toward each other because their horizontal movement is
mirrored.

To win, the player must route the blocks through the asymmetric corridors until
they meet. The important behavior is independent collision: one block can be
pinned against a wall or corridor end while the other continues moving. That
lets the player change which corridor each block occupies. A reconstruction
should allow one block to remain stuck while the other moves, otherwise this
level cannot be solved as intended.

The final contact can be either a shared cell or a horizontal crossing between
adjacent cells. No move is fatal in this level except exceeding the 150-action
limit.

## Level 2

Layout:

```text
cols 0123456789012
00   ......#......
01   ....L.#.R....
02   ......#......
03   ..#########..
04   ..#.#######..
05   .....X#......
06   .....X#......
07   .....X#......
08   XXX.XX#.XXXXX
09   ......#......
10   .............
11   .............
12   XXXXXXXXXXXXX
```

Level 2 adds checkered hazards. The two main blocks start high on opposite sides
of a central vertical wall. The lower half contains hazard stripes: a vertical
hazard segment at `x=5`, a long broken hazard row at `y=8`, and a full hazard
row along the bottom.

The goal is still to make the two main blocks meet. The hazards force the player
to exploit mirroring more carefully. One block must be held in a small safe
notch or against a wall while the other changes alignment, so that both can pass
the hazard bands without landing on them. After crossing into the lower area,
the blocks must be realigned on the same `y` coordinate and brought together.

If either main block lands on a hazard, both main blocks reset to `(4, 1)` and
`(8, 1)` after the flash sequence. There are no blockers or gates, so the reset
only affects the two main blocks.

## Level 3

Layout:

```text
cols 0123456789012
00   #############
01   ##.####.....#
02   #...#.B.###.#
03   #B#.#.###.#.#
04   #.#.#.#.....#
05   #.#...#.#####
06   #.#####.B...#
07   ......####..#
08   #####.#..#.##
09   ##....#.....#
10   ##..L.#.R...#
11   ##....#.....#
12   #############
```

Level 3 introduces clickable blockers. The two main blocks begin near the bottom
at `(4, 10)` and `(8, 10)`. Three blue blockers begin at `(1, 3)`, `(6, 2)`,
and `(8, 6)`. These blockers occupy one cell each and are solid to the main
blocks.

To win, the player must alternate between moving blockers and moving the
mirrored pair. Click a blocker, slide it out of the relevant corridor, then
return to main-block control and continue navigating. The blockers are not
targets; they are obstacles that must be cleared or positioned so that the main
blocks can use the maze's pinning points.

There are no hazards in this level. Bad blocker placement can trap a route, but
it does not kill the player; the level can still be recovered by selecting and
moving blockers again, as long as the 150-action limit is not exceeded.

## Level 4

Layout:

```text
cols 01234567890
00   ###########
01   #XXX###XXX#
02   #...###...#
03   #...###...#
04   #...XXX.R.#
05   #....B....#
06   #.L.XXX...#
07   #...###...#
08   #...###...#
09   #XXX###XXX#
10   ###########
```

Level 4 combines one movable blocker with hazards. The map is split into left
and right chambers by solid central wall sections and two central hazard strips.
The left main block starts at `(2, 6)`, the right main block starts at `(8, 4)`,
and the single blocker starts in the center at `(5, 5)`.

The blocker has two roles. First, it must be moved out of the central passage so
the mirrored pair can route through the level. Second, it can be placed as a
temporary stop so that one main block is pinned before it would be pushed into a
hazard. This is the first level where a blocker is not just clutter to clear; it
is a tool for creating a safe artificial wall.

To win, use the center blocker and the fixed walls to desynchronize the two main
blocks, bring them into compatible chambers, and then move them into the same
cell or through a horizontal crossing. Any main-block contact with the corner
hazards or central hazard strips resets the main blocks and the blocker to their
start positions.

## Level 5

Layout:

```text
cols 012345678901234
00   ###############
01   ...v...........
02   ...............
03   ...............
04   ...............
05   ###GGG####VVV##
06   .......#g.....o
07   .......##.....#
08   .......#.......
09   ##OOO#####VVV##
10   .......#.......
11   ...#...#.......
12   .R.v...#.....L.
13   .......#.......
14   ###############
```

Level 5 introduces colored gates and switch pads. The main blocks start near the
bottom corners, with `R` at `(1, 12)` and `L` at `(13, 12)`. There are no
hazards or blockers.

Gate families:

- Green: the `GGG` gate at row `5`, columns `3..5`, opens while a main block
  stands on the green switch `g` at `(8, 6)`.
- Orange: the `OOO` gate at row `9`, columns `2..4`, opens while a main block
  stands on the orange switch `o` at `(14, 6)`.
- Violet: both `VVV` gates at row `5`, columns `10..12`, and row `9`, columns
  `10..12`, open while a main block stands on either violet switch `v` at
  `(3, 1)` or `(3, 12)`.

To win, the player must use one mirrored block as a gate opener for the other.
For example, standing on a violet switch removes both violet gate bars, allowing
the opposite-side block to pass through the right-side lanes. The same principle
is then used for the green and orange gates. Because gate state is recomputed
after every movement action, a gate can close again as soon as the switch block
steps off its pad, so the switch holder often has to be pinned in place while
the other block moves.

The final task is still to make the two main blocks meet. The gates do not
change the win condition; they only control which passages are open while the
blocks are being aligned.

## Level 6

Layout:

```text
cols 0123456789012
00   XXXXXXXXXXXXX
01   X.....X.....X
02   X..o..X..g..X
03   X.....X.....X
04   X..L..X..R..X
05   X.....X.....X
06   XXGGGXXXOOOXX
07   X.....X.....X
08   X.....X.....X
09   X..g..B.....X
10   X.....X.....X
11   X.....X.....X
12   XXXXXXXXXXXXX
```

Level 6 is the full composition level. There are no solid wall sprites; the
structure is made from hazards, two gate families, and a movable blocker. The
main blocks start in the upper left and upper right chambers at `(3, 4)` and
`(9, 4)`. The blocker starts at `(6, 9)`, in the only gap of the central
vertical hazard divider.

Gate families:

- Green: the `GGG` gate at row `6`, columns `2..4`, opens while a main block
  stands on either green switch at `(9, 2)` or `(3, 9)`.
- Orange: the `OOO` gate at row `6`, columns `8..10`, opens while a main block
  stands on the orange switch at `(3, 2)`.

The hazards form the outer border, the central vertical divider, and dangerous
cells around the two gates. A main block that drifts into the border, the center
divider, or the hazard cells adjacent to the gates resets the whole clickable
state.

To win, the player must plan a sequence where one main block holds the correct
switch while the other passes through the matching gate, then repeat this with
the other gate family as needed. The movable blocker is essential because this
map has few solid non-hazard stopping points. It can be moved into positions
that stop one mirrored block before a hazard, keep a block on a switch long
enough for the other to cross, or prevent the mirrored movement from undoing a
gate setup.

The final meeting must happen on a safe open cell. Meeting on or after moving
through a hazard does not count, because the hazard reset is checked before the
meeting/win check.

## Reconstruction Checklist

- Use a discrete grid. Every moving object occupies one cell.
- Available player actions are up, down, left, right, click/select, and a no-op
  action. The puzzle mechanics are driven by the four directions and click.
- In main-control mode, vertical movement is shared by both main blocks and
  horizontal movement is mirrored.
- Resolve each main block's collision separately. One block being blocked must
  not cancel the other block's move.
- Main blocks are blocked by walls, selected/unselected blockers, closed gate
  bars, and grid boundaries.
- Main blocks are not blocked by switch pads; standing on a switch opens all
  gates of that color family.
- A gate family is open if at least one active main block occupies any switch of
  that family. Otherwise all gates in that family are closed.
- Blockers can be selected by clicking and then moved independently by direction
  actions. They do not open gates.
- Hazards reset all clickable sprites to their level-start positions after a
  short flash sequence. They do not restore the action timer.
- The meeting check happens after movement and hazard handling. Gate state is
  also refreshed after movement, before level completion is finalized.
- Two blocks sharing a cell solve the level. A horizontal adjacent swap also
  counts as a meeting.
- The level advances immediately after both main blocks have met.
- The action limit is 150 actions per level, shown as top and bottom progress
  bars.
