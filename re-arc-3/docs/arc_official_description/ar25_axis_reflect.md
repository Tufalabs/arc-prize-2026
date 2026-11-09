# AR25 - Axis Reflect

## Core Game

Axis Reflect is a discrete positioning puzzle played on a 21x21 grid. The board contains yellow target cells, blue movable polyominoes, and one or two cyan mirror axes. The player wins a level when every yellow target cell is covered by either a real blue cell or a reflected copy of a blue cell.

There is no player avatar and no collision maze. The active object is one of the movable blue shapes or one of the movable mirror axes. Direction actions translate the active object by one grid cell. Clicking a blue object or cyan axis selects it directly; the cycle/select action also advances through the selectable objects. Undo restores the previous positions of all blue shapes and axes. The step meter counts down on translations and on cycle/select actions; levels 1 and 2 start with 64 steps, levels 3 through 5 with 128, and levels 6 through 8 with 320. Running out of steps loses the level.

Mirror axes are full-board cyan lines:

- A vertical mirror line is represented by an `x` coordinate. Blue cells reflect to `(2 * axis_x - x, y)`. Vertical mirror lines can only be moved left and right.
- A horizontal mirror line is represented by a `y` coordinate. Blue cells reflect to `(x, 2 * axis_y - y)`. Horizontal mirror lines can only be moved up and down.

With multiple axes, reflection is recursive. Each real blue cell can produce reflected cells, those reflected cells can be reflected by the other axis, and so on, with duplicate positions suppressed. In normal play this behaves like a repeated mirror system: a shape can appear as the original, its vertical reflection, its horizontal reflection, and its reflection across both axes. Generated reflections are rendered as ghost-blue cells behind the real shapes.

Only the yellow targets matter for success. Extra real or reflected blue cells may exist away from the target pattern, including off the visible grid. A level advances after a move produces coverage on every yellow target. The next action then loads the following level.

## Reconstruction Model

Use a 21x21 logical grid with coordinates `(x, y)`, origin at the top left. Yellow target cells are single cells. Blue shapes are stored as masks where `X` is filled and `.` is transparent. A shape position is the top-left coordinate of its mask. Mirror lines are drawn as long lines, but their gameplay state is only the axis coordinate.

For each state update:

1. Select a shape or axis.
2. Translate it by one cell if the chosen direction is legal for that object type.
3. Recompute the reflection closure of all real blue cells through all active axes.
4. Overlay real blue cells on top of reflected ghost cells.
5. Check every yellow target coordinate; the level is solved if all targets have any real or reflected blue cell.
6. Push the pre-move positions onto the undo stack for real translations.

The blue masks used in the levels are:

```text
A, 3x3        B, 5x4        C, 4x4        D, 4x2
XXX           XXX..         X...          XXXX
..X           ..X..         X...          .XX.
..X           ..X..         X...
              ..XXX         XXXX

E, 5x2        F, 1x6        G, 5x5
X...X         X             XXXX.
XXXXX         X             ...X.
              X             ...XX
              X             ....X
              X             ..XXX
              X

H, 4x2        I, 5x5        J, 3x7        K, 5x4
.X..          .XXX.         ..X           .X.X.
XXXX          X...X         ..X           XXXXX
              X....         .XX           XXXXX
              X....         .X.           .X.X.
              .X...         XX.
                            X..
                            X..

L, 5x3        M, 3x5
XX...         XXX
X....         ..X
XXXXX         ..X
              ..X
              ..X
```

## Level 1

Level 1 introduces a fixed vertical mirror. The only selectable object is shape A, initially at `(6, 5)`. The vertical mirror line is fixed at `x = 10` and is not selectable.

Targets:

```text
(17,15), (18,15), (19,15), (17,16), (17,17)
```

The target cells form the horizontal reflection of shape A. To solve the level, move shape A so its reflected copy covers the target. One exact solution is to place the shape at top-left `(1, 15)`: its cells at `x = 1, 2, 3` reflect across `x = 10` to `x = 19, 18, 17`, producing the yellow pattern.

Important reconstruction details:

- The mirror is visible as a cyan vertical line but cannot be selected.
- The level teaches that the blue object itself does not need to touch the yellow cells; its reflected shadow may cover them.
- The win condition is target coverage only. The real shape remains on the opposite side of the mirror.

## Level 2

Level 2 makes the vertical mirror movable. Shape B starts at `(15, 6)`, and the vertical mirror line starts at `x = 12`. Both are selectable.

Targets:

```text
(3,14), (4,14), (5,14),
(3,15),
(3,16),
(1,17), (2,17), (3,17)
```

The targets again describe a reflected version of the blue object, but now the player must adjust both the object and the mirror line. The shape can move freely, while the vertical line can only slide left and right. A winning arrangement places the object and mirror so that shape B's left-right reflection lands on the lower-left target group.

Important reconstruction details:

- Objects marked with clickable dots are selectable; in this level both shape B and the vertical line can be selected.
- Moving the vertical line changes the reflection of all blue cells immediately.
- If the player spends too many steps in this tutorial level before using click selection, a brief flashing hint appears on the active object. It is feedback only and does not affect puzzle state.

## Level 3

Level 3 switches to a horizontal mirror and introduces multiple movable shapes. The horizontal mirror line starts at `y = 16`. Shape C starts at `(4, 7)`, and shape D starts at `(15, 9)`.

Targets:

```text
Top-left group:
(4,3), (5,3), (3,4), (4,4), (5,4), (6,4)

Top-right group:
(11,1), (12,1), (13,1), (14,1), (11,2), (11,3), (11,4)

Bottom-left group:
(3,14), (4,14), (5,14), (6,14), (4,15), (5,15)

Bottom-right group:
(11,14), (11,15), (11,16), (11,17), (12,17), (13,17), (14,17)
```

The two shapes must be positioned so that real cells and horizontal reflections cover all four yellow groups. The lower groups can be covered directly by the real objects, while corresponding upper groups can be produced by reflecting across the horizontal line, or vice versa after repositioning the line. Shape C is a 4x4 L-like block; shape D is a short 4x2 bar with a two-cell center on its lower row.

Important reconstruction details:

- The horizontal line can only move vertically.
- Both blue shapes are independent selectable objects.
- The level teaches that a single axis can coordinate more than one shape at the same time.

## Level 4

Level 4 keeps a single horizontal mirror but requires several objects to overlap into one combined target pattern. The horizontal mirror line starts at `y = 3`. Shape E starts at `(4, 6)`, and shape F starts at `(6, 10)`.

Targets:

```text
(13,3),
(13,4),
(13,5),
(11,6), (13,6), (15,6),
(11,7), (12,7), (13,7), (14,7), (15,7),
(13,8),
(13,10),
(11,11), (12,11), (13,11), (14,11), (15,11),
(11,12), (13,12), (15,12),
(13,13),
(13,14),
(13,15)
```

The yellow cells form a tall symmetric figure centered around `x = 13`, with dense five-cell crossbars near `y = 7` and `y = 11`. Shape F supplies the long vertical spine. Shape E supplies horizontal width and the side cells. The mirror line must be moved so the reflected portions complete the top and bottom halves of the target figure.

Important reconstruction details:

- The target is not intended to be one object. It is a composite made by overlapping two independently placed shapes and their reflections.
- Real and reflected blue cells may overlap freely. Overlap has no penalty and is often necessary to make the combined pattern read as one figure.
- Only one mirror axis is present, so all reflected copies are vertical flips in screen space.

## Level 5

Level 5 introduces two mirrors at once. Shape G starts at `(14, 12)`. The horizontal line starts at `y = 5`, and the vertical line starts at `x = 3`.

Targets:

```text
Upper-left cluster:
(4,5), (5,5), (6,5), (7,5), (7,6), (7,7)

Center cluster:
(8,7), (8,8), (6,9), (7,9), (8,9), (9,9), (10,9), (8,10), (8,11), (9,11), (9,12)

Lower-left cluster:
(4,13), (9,13), (10,13), (11,13), (12,13)

Upper-right/lower-right anchors:
(12,5), (12,13)
```

The player must use both mirror axes to generate a family of copies from a single shape. Shape G is a five-row bent path. With the vertical and horizontal lines placed correctly, the original, the two single-axis reflections, and the double reflection cover separated target clusters around the board.

Important reconstruction details:

- The vertical line is selectable and moves left/right; the horizontal line is selectable and moves up/down.
- A cell may reflect across one axis and then the other, so two perpendicular mirrors create diagonal counterpart copies even though no diagonal reflection exists directly.
- The level is solved by coverage, not by making the entire reflected pattern equal the yellow pattern. Extra blue cells away from listed targets are acceptable.

## Level 6

Level 6 is a large two-axis puzzle with two blue shapes. The horizontal line starts at `y = 0`, the vertical line starts at `x = 7`, shape H starts at `(17, 8)`, and shape I starts at `(14, 3)`.

Targets:

```text
(4,3), (8,3),
(5,4), (7,4),
(5,5), (7,5),
(1,6), (5,6), (7,6), (11,6),
(2,7), (3,7), (4,7), (8,7), (9,7), (10,7),
(2,9), (3,9), (4,9), (5,9), (7,9), (8,9), (9,9), (10,9),
(3,10), (9,10),
(3,12), (9,12),
(2,13), (3,13), (4,13), (5,13), (7,13), (8,13), (9,13), (10,13),
(2,15), (3,15), (4,15), (8,15), (9,15), (10,15),
(1,16), (5,16), (7,16), (11,16),
(5,17), (7,17),
(5,18), (7,18),
(4,19), (8,19)
```

The yellow pattern is a broad, repeated, bilaterally structured arrangement in the left and middle of the board. Shape I is a 5x5 broken outline; shape H is a small 4x2 bar. The player must move both shapes and both axes so the recursive reflection system fills the repeated columns and crossbars.

Important reconstruction details:

- The starting horizontal axis at `y = 0` can still be moved; it initially lies along the top edge.
- The puzzle depends on combining a large sparse shape with a small filler shape. Shape I supplies most of the diagonal and outline fragments; shape H fills short horizontal runs.
- Because the axes can be near or outside the visible target clusters, a useful reconstruction should treat mirror coordinates mathematically rather than as physical blockers.

## Level 7

Level 7 uses two axes and two shapes, but the success condition is intentionally looser than visual full-pattern matching. The horizontal line starts at `y = 5`, the vertical line starts at `x = 3`, shape J starts at `(17, 13)`, and shape K starts at `(5, 16)`.

Targets:

```text
Outer ring and diagonals:
(7,1), (9,1), (11,1), (13,1), (15,1), (17,1),
(13,2), (14,2), (15,2),
(7,3), (13,3), (14,3), (15,3), (17,3),
(9,4), (11,4), (13,4), (15,4),
(8,5), (16,5),
(9,7), (15,7),
(8,9), (16,9),
(7,11), (17,11),
(7,13), (9,13), (11,13), (13,13), (15,13), (17,13)

Inner lower block:
(9,10), (11,10), (13,10), (15,10),
(9,11), (10,11), (11,11),
(9,12), (10,12), (11,12)
```

The level asks the player to cover only the yellow cells, not to recreate a clean final silhouette. Shape J is a tall zigzag column, and shape K is a dense 5x4 block with notches. The two axes should be placed so reflected copies of the tall and dense shapes reach the ring, diagonals, and inner block at the same time.

Important reconstruction details:

- Extra reflected cells are expected and harmless. This is the clearest level demonstrating that yellow coverage is the only win test.
- The two shapes have very different densities. Shape J is useful for narrow diagonal/vertical target traces; shape K is useful for dense rectangular groups.
- Click selection matters for usability because four objects are selectable: two shapes and two axes.

## Level 8

Level 8 is the final two-axis placement puzzle. The horizontal line starts at `y = 5`, the vertical line starts at `x = 3`, shape L starts at `(13, 13)`, and shape M starts at `(7, 7)`.

Targets:

```text
Upper-left cluster:
(6,3), (7,3), (8,3),
(6,4), (6,5), (4,6), (5,6), (6,6), (4,7), (6,7),
(4,8), (5,8), (6,8), (7,8), (8,8)

Upper-right cluster:
(16,3), (17,3), (18,3),
(18,4), (18,5), (18,6), (19,6), (20,6), (18,7),
(16,8), (17,8), (18,8), (19,8), (20,8)

Lower-left cluster:
(4,14), (5,14), (6,14), (7,14), (8,14),
(4,15), (6,15), (4,16), (5,16), (6,16),
(6,17), (6,18), (6,19), (7,19), (8,19)

Lower-right cluster:
(16,14), (17,14), (18,14), (19,14), (20,14),
(18,15), (20,15), (18,16), (19,16), (20,16),
(18,17), (18,18), (16,19), (17,19), (18,19)
```

The target layout is four corner-like groups. Shape M is a narrow 3x5 corner/leg shape, and shape L is a wider 5x3 shape with a full bottom row. The two shapes and two axes must be arranged so their original and reflected copies populate all four quadrants.

Important reconstruction details:

- The four target clusters are related by horizontal and vertical reflection, but they are not all identical because the two different blue shapes contribute different pieces.
- The initial axes are both near the upper-left side; solving requires repositioning at least one shape and usually both axes.
- As in level 7, the target coverage test ignores surplus blue cells, so the reconstruction should not enforce exact equality between the produced blue pattern and the yellow target set.
