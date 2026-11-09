# CN04 - Connector

## Core Game

Connector is a free-placement construction puzzle played on a 20x20 grid. There is no avatar and no maze collision. The player selects a visible connector piece, moves it one grid cell at a time with the four direction actions, and uses the space action to either rotate it or change its size.

Every connector piece is an internally connected body with red endpoint cells. A level is solved when every visible endpoint cell is paired with exactly one endpoint of the same hidden endpoint polarity at the same grid coordinate. Endpoint adjacency does nothing; the two endpoint cells must overlap exactly. Body cells may overlap freely and never block movement.

There are two hidden endpoint polarities:

```text
A = red endpoint, internal polarity A
B = red endpoint, internal polarity B
```

Both polarities are drawn as red endpoints in the normal view. The hidden polarity matters only to the win check. An A endpoint must overlap another A endpoint, and a B endpoint must overlap another B endpoint. If three or more same-polarity endpoints overlap at one coordinate, none of those endpoints count as correctly paired. If an A and B visually overlap, the contact may look connected, but it does not satisfy either endpoint unless another same-polarity endpoint also completes the pair.

The masks below use this notation:

```text
. = transparent cell
# = connector body cell
A = hidden-polarity-A red endpoint
B = hidden-polarity-B red endpoint
o = black growth handle on an expandable piece
? = negative transparent cell used only to preserve one variant's bounding box
```

Coordinates are zero-indexed. A piece's `top-left` coordinate is the top-left of its rendered mask after its listed rotation. All masks in the level setup are shown in their starting rendered orientation.

## Controls And State

Clicking a visible, non-transparent cell selects that piece. Clicking the selected non-expandable piece again deselects it. Clicking a selected expandable piece on a non-handle cell also deselects it. The selected piece is drawn above the others.

Direction actions move the selected piece by one grid cell if its full rendered rectangle remains within the 20x20 board. There is no collision with other pieces.

Space rotates a non-expandable piece 90 degrees clockwise. After rotation, the piece is clamped so the whole rendered rectangle remains in-bounds.

Some later pieces are expandable stacks: multiple mutually exclusive variants at the same top-left coordinate. Only one variant is visible at a time. Space does not rotate these pieces; instead it cycles to the next variant in the stack. The cycle goes upward through the variants until the largest variant, then reverses back downward. Clicking the black `o` growth handle on the selected expandable piece also cycles the variant. Hidden variants do not count for the win check.

Levels 1 and 2 show all pieces in color, and the selected piece's body is drawn white. From level 3 onward, grey masking is enabled: inactive pieces are grey silhouettes, and selecting a piece reveals its actual connector endpoints. This is part of the hidden-information challenge.

The masks in this document collapse body colors to `#` because body color has no mechanical effect. For visual reconstruction, each nonselected piece can keep its own decorative body color, while the selected and grey-masked render rules above provide the gameplay-relevant distinction.

Every action counts against the level's action budget, shown as the bar along the top of the play area. If the budget is exhausted, the level fails and must be retried from its initial state.

## Level 1

Setup:

```text
Background: light cyan
Action budget: 75
Grey masking: off
Initial selection: left piece
```

Pieces:

```text
L1-A  top-left (3,3), rotation 90, size 5x6
#####
#...#
##.##
.#.#.
.#.#.
.B.A.

L1-B  top-left (12,9), rotation 0, size 4x7
.###
.#.#
A#.#
.#.#
B#.#
.#.#
.###
```

This is the tutorial for exact endpoint overlap. There are only two pieces and four endpoints total. To win, the A endpoint on one piece must overlap the A endpoint on the other, and the B endpoint on one piece must overlap the B endpoint on the other.

One solved arrangement is to leave L1-B at `(12,9)` with rotation `0`, rotate L1-A to rotation `0`, and place L1-A at `(7,10)`. This pairs the two A endpoints at `(12,11)` and the two B endpoints at `(12,13)`.

Important reconstruction details:

- Rotation is required unless the right piece is rotated instead.
- The two red endpoints on a piece are not interchangeable because their hidden polarities differ.
- No body contact, adjacency, or partial overlap matters; only exact endpoint-on-endpoint overlap counts.

## Level 2

Setup:

```text
Background: orange
Action budget: 100
Grey masking: off
Initial selection: upper-left bracket
```

Pieces:

```text
L2-A  top-left (4,11), rotation 0, size 7x7
...###.
.A.#.#A
.###.#.
B#...#.
.###.#.
...###.
....B..

L2-B  top-left (12,4), rotation 0, size 5x7
..A#.
...#.
...#B
...#.
...#A
B..#.
####.

L2-C  top-left (3,3), rotation 0, size 3x6
###
#.#
#.#
#.A
#..
#B.

L2-D  top-left (16,16), rotation 90, size 3x3
A.B
#.#
###
```

This level introduces selecting between multiple objects and building a compound connector. L2-A and L2-B each have four endpoints, so simply joining two pieces together still leaves unmatched endpoints. All twelve endpoints must be paired.

One solved arrangement is:

```text
L2-A: top-left (4,11), rotation 0
L2-B: top-left (8,12), rotation 0
L2-C: top-left (3,9), rotation 0
L2-D: top-left (12,14), rotation 0
```

The resulting endpoint pairs are:

```text
(5,12)  A: L2-A with L2-C
(10,12) A: L2-A with L2-B
(4,14)  B: L2-A with L2-C
(8,17)  B: L2-A with L2-B
(12,14) B: L2-B with L2-D
(12,16) A: L2-B with L2-D
```

Important reconstruction details:

- All endpoints are visible in this level.
- Every piece is movable and selectable; the puzzle is not a fixed-target lock.
- Multiple connector bodies may overlap. The only required overlaps are the six same-polarity endpoint pairs listed above or an equivalent all-paired arrangement.

## Level 3

Setup:

```text
Background: orange
Action budget: 125
Grey masking: on
Initial selection: left green path
```

Pieces:

```text
L3-A  top-left (10,3), rotation 90, size 6x2
.....B
B#####

L3-B  top-left (2,4), rotation 90, size 5x7
..##A
..#..
###..
#....
###..
..#..
..##B

L3-C  top-left (9,9), rotation 270, size 7x7
.B.....
###....
#.#....
#.#....
#.####.
#....#A
######.
```

This level adds the main hidden-state challenge. Inactive pieces are silhouettes, so the player has to select pieces to inspect their endpoint positions. The two visible red endpoint polarities still look the same, but the win check distinguishes A from B.

L3-A has two B endpoints. L3-B and L3-C each have one A and one B endpoint. A valid composition therefore uses L3-A as a double-B connector: each of its B endpoints must be paired with the B endpoint from one of the other pieces, and then the remaining A endpoints on L3-B and L3-C must be paired with each other.

One solved arrangement is:

```text
L3-A: top-left (10,3), rotation 90
L3-B: top-left (15,3), rotation 270
L3-C: top-left (9,4), rotation 270
```

The resulting endpoint pairs are:

```text
(15,3) B: L3-A with L3-B
(10,4) B: L3-A with L3-C
(15,9) A: L3-B with L3-C
```

Important reconstruction details:

- The level cannot be reconstructed from visible silhouettes alone; endpoint masks must exist underneath the grey render mask.
- The same-polarity rule matters here. A visually plausible red-on-red overlap can still be wrong if it pairs A with B.
- The level has only three endpoint pairs, but the pieces must be rotated and moved so all three pairs are present at once.

## Level 4

Setup:

```text
Background: blue
Action budget: 125
Grey masking: on
Initial selection: upper-left path
```

Pieces:

```text
L4-A  top-left (2,3), rotation 90, size 6x8
######
#....#
#.##.#
#.#..#
#.####
#.#..A
#.....
A.....

L4-B  top-left (11,6), rotation 0, size 5x4
A#.#A
.#.#.
.#.#.
.###.

L4-C  top-left (5,15), rotation 90, size 3x1
A#A

L4-D  top-left (14,13), rotation 0, size 4x6
A###
.#.#
.#.#
.#.#
.#.#
A###
```

All endpoints in this level have the same hidden polarity, so the challenge is geometric rather than polarity-based. The four pieces have eight A endpoints total, and all eight must be paired into four exact overlaps.

One solved arrangement is:

```text
L4-A: top-left (2,3), rotation 90
L4-B: top-left (7,8), rotation 270
L4-C: top-left (7,10), rotation 0
L4-D: top-left (2,10), rotation 90
```

The resulting endpoint pairs are:

```text
(7,8)  A: L4-A with L4-B
(2,10) A: L4-A with L4-D
(7,12) A: L4-B with L4-C
(7,10) A: L4-C with L4-D
```

Important reconstruction details:

- Grey masking still hides inactive endpoint layouts even though all endpoints share one polarity.
- The short two-ended bar is a normal rotatable piece; it is not an expandable stack.
- Since every endpoint is polarity A, any exact two-endpoint A overlap can satisfy a pair, but all four pairs must be present simultaneously.

## Level 5

Setup:

```text
Background: purple
Action budget: 150
Grey masking: on
Initial selection: central growing tree
```

This level introduces an expandable stack. The central tree starts in its smallest visible form. It cannot rotate; space changes which tree variant is visible. The largest tree is needed because earlier variants leave an odd number of B endpoints or do not expose enough sockets to attach the three outer pieces.

Central expandable tree variants, all with top-left `(7,4)` and rotation `0` at level start:

```text
L5-A1  size 4x4, initially visible
.#A.
A#..
.##B
.o..

L5-A2  size 4x5
.#A.
A#..
.##B
A#..
.o..

L5-A3  size 6x6
.#A...
A#....
.##B..
A#....
.####A
.o....

L5-A4  size 6x7
.#A...
A#....
.##B..
A#....
.####A
A#....
.o....

L5-A5  size 6x7
.#A...
A#....
.##B..
A#....
.####A
A#....
.###B.
```

Other pieces:

```text
L5-B  top-left (1,12), rotation 90, size 3x5
##A
#..
##A
#..
##A

L5-C  top-left (15,1), rotation 90, size 3x5
..A
B.#
#.#
#.#
###

L5-D  top-left (15,15), rotation 90, size 3x3
B..
#.A
###
```

The intended structure is a grown central tree with enough A sockets on its left and right branches, plus two B sockets for the two mixed-polarity outer pieces. L5-B has three A endpoints and attaches to three A endpoints on the tree. L5-C attaches to one A and one B endpoint. L5-D attaches to one A and one B endpoint.

One solved arrangement is:

```text
L5-A5: top-left (3,11), rotation 0
L5-B:  top-left (1,12), rotation 90
L5-C:  top-left (5,11), rotation 0
L5-D:  top-left (7,15), rotation 0
```

The resulting endpoint pairs are:

```text
(3,12) A: L5-A5 with L5-B
(3,14) A: L5-A5 with L5-B
(3,16) A: L5-A5 with L5-B
(5,11) A: L5-A5 with L5-C
(6,13) B: L5-A5 with L5-C
(8,15) A: L5-A5 with L5-D
(7,17) B: L5-A5 with L5-D
```

Important reconstruction details:

- Only the visible tree variant counts. Hidden smaller or larger variants should not contribute endpoints.
- The black `o` handle is part of the currently selected expandable piece and is used only for cycling variants; it is not an endpoint.
- Once the tree reaches L5-A5, pressing space again reverses the cycle toward smaller variants.

## Level 6

Setup:

```text
Background: blue
Action budget: 200
Grey masking: on
Initial selection: upper-left connector
```

This level has two expandable stacks. The central branching piece grows from a compact trunk into a larger multi-socket tree. The lower-right connector also expands, keeping the same number of endpoints but moving those endpoints farther apart. The intended solution expands both stacks before composing the pieces.

Central expandable stack, all with top-left `(6,6)` and rotation `90` at level start:

```text
L6-A1  size 3x6, initially visible
..A
.##
B#.
.#.
.o.
.?.

L6-A2  size 4x6
..A.
.##.
B#..
.#..
.##o
.A..

L6-A3  size 6x6
..A...
.##...
B#....
.#....
.####o
.A....

L6-A4  size 7x6
..A....
.##....
B#.....
.#....o
.######
.A....B

L6-A5  size 7x6
..A....
.##...o
B#....#
.#....#
.######
.A....B

L6-A6  size 8x6
..A..A#.
.##...#B
B#....#.
.#....#.
.######.
.A....B.
```

Upper-left connector:

```text
L6-B  top-left (1,1), rotation 90, size 5x6
#####
#...#
#...A
#....
##B..
A....
```

Lower-right expandable stack, all with top-left `(9,14)` and rotation `90` at level start:

```text
L6-C1  size 6x5, initially visible
..A.B.
..#.#.
.##.##
B#...#
.##o##

L6-C2  size 7x5
..A..B.
..#..#.
.##..##
B#....#
.###o##

L6-C3  size 8x5
..A...B.
..#...#.
.##...##
B#.....#
.####o##

L6-C4  size 9x5
..A....B.
..#....#.
.##....##
B#......#
.########
```

Other pieces:

```text
L6-D  top-left (15,1), rotation 90, size 4x4
####
#..#
A..#
..B#

L6-E  top-left (1,12), rotation 90, size 2x6
A.
#.
#.
#.
#.
#B
```

One solved arrangement using the largest versions of both expandable stacks is:

```text
L6-B:  top-left (1,1), rotation 90
L6-A6: top-left (3,3), rotation 90
L6-C4: top-left (2,8), rotation 90
L6-D:  top-left (8,1), rotation 90
L6-E:  top-left (1,6), rotation 90
```

The resulting endpoint pairs are:

```text
(5,3)  A: L6-B with L6-A6
(3,5)  B: L6-B with L6-A6
(1,6)  A: L6-B with L6-E
(8,3)  A: L6-A6 with L6-D
(10,4) B: L6-A6 with L6-D
(4,8)  A: L6-A6 with L6-C4
(9,8)  B: L6-A6 with L6-C4
(2,11) B: L6-C4 with L6-E
```

Important reconstruction details:

- The central stack and lower-right stack are independently expandable.
- Expandable stacks do not rotate in this implementation; their listed rotation stays fixed while cycling variants.
- The central stack's `?` cell is transparent and non-clickable, but it contributes to the rendered rectangle used for bounds.
- The level is solved by the same global rule as every other level: every visible endpoint must have exactly one same-polarity partner at the same coordinate.

## Implementation Checklist

- Use a 20x20 grid for every level.
- Selectable pieces are sprites with transparent, body, endpoint, and optional growth-handle cells.
- Direction actions translate the selected sprite by one grid cell within bounds.
- Space rotates normal pieces 90 degrees clockwise.
- Space cycles variants for expandable stacks instead of rotating them.
- Clicking a visible piece selects it; clicking the selected normal piece deselects it.
- Clicking the selected expandable piece's black handle cycles its variant.
- Inactive pieces are full-color in levels 1-2 and grey silhouettes in levels 3-6.
- Exact two-endpoint same-polarity overlap is the only win condition.
- Body overlap is allowed and has no mechanical effect.
- Hidden variants do not count toward endpoint pairing.
- The action budgets are 75, 100, 125, 125, 150, and 200 actions for levels 1-6 respectively.
