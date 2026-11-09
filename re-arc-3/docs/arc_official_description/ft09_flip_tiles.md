# Flip Tiles

## Core Game

Flip Tiles is a click-only pattern reconstruction game. Each level is a 32x32 logical grid rendered as a 64x64 playfield. Editable tiles are 3x3 sprites placed on a 4-pixel lattice, so neighboring tile top-left coordinates differ by 4 in `x` or `y`. The player clicks editable tiles; there is no movement action and no inventory.

Every editable tile has a center color state. A regular tile changes only itself. A special tile changes itself and additional neighboring tiles marked by magenta cells inside the special tile graphic. A level is solved when all target-marker constraints are satisfied at the same time.

The bottom row is a click budget. Effective tile clicks decrement it after the win check. Level 1 and 2 have 32 clicks, level 3 and 4 have 96 clicks, and level 5 and 6 have 128 clicks. Clicking empty space or a non-clickable target marker has no gameplay effect and does not spend the budget. If the budget reaches zero after a non-winning tile click, the level fails. Resetting the level restores all tile colors and the full budget.

## Global Reconstruction Model

Use these logical object types:

- `Hkx`: a regular editable tile. It is clickable and cycles its whole 3x3 body through the level's color cycle.
- `NTi`/special tile: a clickable tile with a magenta effect mask. It still has a normal center color state, but a click affects a 3x3 neighborhood instead of only the clicked tile.
- target marker: a non-clickable 3x3 pattern marker. Its center color is the target color for that local pattern. It checks the eight neighboring tile positions around itself, not its own center.

Color symbols used below:

```text
B = blue    (color id 9)
R = red     (color id 8)
O = orange  (color id 12)
G = green   (color id 14)
P = purple  (color id 15)
Y = yellow  (color id 11)
```

The magenta pixels on special tiles are effect indicators, not part of the level color cycle.

## Target Marker Rule

Each target marker is a 3x3 mask centered on a target color. For each of the eight neighboring lattice positions, if an editable tile exists there, the marker imposes one constraint:

```text
=  the neighboring tile must equal the marker's center color
!  the neighboring tile must be any other color in the level's cycle
C  the marker center; this is not an editable tile and is not checked
```

Missing neighboring tile positions are ignored. A tile can be constrained by more than one marker, so the final legal state is the intersection of all constraints. In two-color levels, `!` is equivalent to the other color. In three-color levels, `!` may leave two legal colors.

## Click Effects

Regular tile:

```text
. . .
. X .
. . .
```

Level 5 special tile:

```text
. X .
X X X
. X .
```

Level 6 special tile:

```text
. X .
. X .
. . .
```

`X` means "cycle the tile at this relative position if an editable tile exists there." The center `X` is always included for special tiles; the magenta pixels add the extra neighbor positions.

## Level 1

Level 1 is a single editable 3x3 block with a target marker in its center. Three completed example blocks are displayed elsewhere on the board, but only the lower-right editable block is checked. The editable block starts entirely blue. The color cycle is:

```text
B -> R -> B
```

There is one target marker at `(22, 22)`, with a red center and this mask:

```text
= ! !
= C =
= ! !
```

Final required state:

```text
x:    18  22  26
y18   R   B   B
y22   R  (R)  R
y26   R   B   B
```

To win, make the four `R` tiles red and leave the other four editable tiles blue. From the initial all-blue state, one direct solution is to click `(18,18)`, `(18,22)`, `(26,22)`, and `(18,26)` once each.

Important reconstruction details:

- The center `(22,22)` is the target marker, not an editable tile.
- Clicking outside a tile in this level flashes the frame around the editable block as a hint.
- The target check runs after every tile click; if the block matches, the game advances immediately.

## Level 2

Level 2 stacks two target patterns vertically so that their 3x3 neighborhoods overlap. All editable tiles start blue. The color cycle is:

```text
B -> O -> B
```

The upper target marker is at `(14, 11)`, with an orange center:

```text
= ! !
= C =
= ! =
```

The lower target marker is at `(14, 19)`, also with an orange center:

```text
= ! =
! C !
= = !
```

Final required state:

```text
x:    10  14  18
y07   O   B   B
y11   O  (O)  O
y15   O   B   O
y19   B  (O)  B
y23   O   O   B
```

To win, the shared middle row must satisfy both markers. From the initial all-blue state, click the seven orange final-state tiles once: `(10,7)`, `(10,11)`, `(18,11)`, `(10,15)`, `(18,15)`, `(10,23)`, and `(14,23)`.

Important reconstruction details:

- Both markers use the same target color, but they are checked independently.
- The overlapping row at `y=15` is constrained by both markers. The final state shown above is the intersection of the two pattern requirements.

## Level 3

Level 3 introduces several overlapping target patterns. All editable tiles start red. The color cycle is:

```text
R -> O -> R
```

There are four target markers:

```text
Marker at (14, 6), orange:
= = =
= C !
! = !

Marker at (10, 14), red:
! = !
! C =
= = !

Marker at (18, 14), red:
! = =
= C !
! = !

Marker at (14, 22), orange:
! = !
= C !
= = =
```

Final required state:

```text
x:     6  10  14  18  22
y02    .   O   O   O   .
y06    .   O  (O)  R   .
y10    O   R   O   R   R
y14    O  (R)  R  (R)  O
y18    R   R   O   R   O
y22    .   O  (O)  R   .
y26    .   O   O   O   .
```

To win, all four local patterns must be correct simultaneously. Because every editable tile is regular and the cycle has two colors, the solution is just a parity pattern: click each final `O` tile once and leave each final `R` tile untouched.

Important reconstruction details:

- The target markers at `(10,14)` and `(18,14)` overlap the orange markers above and below them.
- The target center cells are visual markers only; the editable area is the surrounding lattice.
- A tile constrained as `! orange` is red because this level has only red and orange.

## Level 4

Level 4 keeps regular self-flipping tiles but adds a three-color cycle. All editable tiles start blue. The color cycle is:

```text
B -> R -> O -> B
```

There are three target markers:

```text
Marker at (10, 11), orange:
! = !
! C !
! = !

Marker at (18, 11), blue:
! = !
! C !
! ! =

Marker at (14, 19), orange:
= ! !
! C !
= = =
```

Final accepted states:

```text
x:      6    10   14    18    22
y07   {BR}   O    R     B    {RO}
y11   {BR}  (O)   R    (B)   {RO}
y15   {BR}   O    R     R     B
y19    .    {BR} (O)   {BR}   .
y23    .     O    O     O     .
```

`{BR}` means either blue or red is accepted. `{RO}` means either red or orange is accepted. These are not wildcards; they come from exact `! orange` or `! blue` constraints in a three-color cycle.

To win, pick any accepted color for the ambiguous cells and cycle each tile to that color. One low-click target is to leave `{BR}` cells blue and set `{RO}` cells red.

Important reconstruction details:

- Repeated clicks matter: blue tiles need one click to become red and two clicks to become orange.
- The target marker at `(14,19)` forces the entire bottom row of editable tiles to orange.
- The three-color `!` rule is the new mechanic. Some cells are deliberately under-specified by a marker and remain legal in more than one color.

## Level 5

Level 5 introduces cross-effect special tiles. The board uses green and purple states. All editable tile centers start green. The color cycle is:

```text
G -> P -> G
```

Regular green tiles still toggle only themselves. The three special tiles are at `(11,6)`, `(11,14)`, and `(19,22)`. Clicking one toggles itself and the four orthogonal neighbor positions if those positions contain editable tiles:

```text
. X .
X X X
. X .
```

There are eight target markers:

```text
Marker at (7, 2), green:
! ! !
! C =
! = !

Marker at (19, 6), purple:
= ! !
! C !
= ! =

Marker at (3, 14), purple:
! ! =
! C !
! ! =

Marker at (11, 18), green:
= ! =
! C !
= ! =

Marker at (19, 14), green:
! = !
= C =
! = !

Marker at (27, 14), green:
! = !
= C !
! = !

Marker at (11, 22), green:
! ! !
= C =
! = !

Marker at (23, 26), green:
! = !
= C !
! ! !
```

Final accepted states:

```text
x:      3    7    11    15    19    23    27
y02     .   (G)   G     P     .     .     .
y06     .    G    sP    G    (P)    .     .
y10     G    P   {GP}   P     G     P     G
y14    (P)   G    sP    G    (G)    G    (G)
y18     G    P   (G)    P     G     P     G
y22     .    G   (G)    G     sP    G     .
y26     .    P    G     P     G    (G)    .
```

`sP` is a special editable tile whose final center color must be purple. `{GP}` at `(11,10)` means either green or purple is legal.

To win, solve the two-color parity puzzle with the cross toggles. One valid click set from the initial state is:

```text
(11,2), (15,2), (7,6), (11,6), (15,6),
(7,10), (15,10), (23,10),
(7,14), (11,14), (15,14),
(7,18), (15,18), (19,18), (23,18),
(15,22), (19,22), (23,22),
(7,26), (15,26), (19,26)
```

Important reconstruction details:

- Special tiles are still colored tiles. Their center color participates in the final pattern check.
- The magenta cross on a special tile adds neighbor effects, and the tile's own center also toggles.
- If an effect points to a missing lattice position or a target marker, that part of the effect does nothing.
- The level is won by the final color parity, not by clicking the target markers.

## Level 6

Level 6 uses only special tiles. All editable tile centers start yellow. The color cycle is:

```text
Y -> G -> Y
```

Every editable tile is a special tile with a magenta mark at the top. Clicking a tile toggles the clicked tile and the editable tile directly above it, if one exists:

```text
. X .
. X .
. . .
```

There are four target markers:

```text
Marker at (6, 3), green:
! ! !
! C !
= = !

Marker at (18, 11), green:
! = !
= C =
= = !

Marker at (10, 15), green:
! = =
= C =
! = !

Marker at (22, 23), green:
! = =
! C !
! ! !
```

Final required state:

```text
x:      2    6    10   14   18   22   26
y03     Y   (G)   .    .    .    .    .
y07     G    G    Y    Y    G    Y    .
y11     .    Y    G    G   (G)   G    .
y15     .    G   (G)   G    G    Y    .
y19     .    Y    G    Y    Y    G    G
y23     .    .    .    .    .   (G)   Y
```

To win, account for the upward side effect. A tile may already be correct, but clicking the tile below it can change it again. One valid click set from the initial state is:

```text
(2,3), (2,7), (10,7), (18,7),
(6,11), (10,11),
(6,15), (14,15), (18,15), (22,15),
(10,19), (22,19), (26,19)
```

Important reconstruction details:

- Top-row special tiles with no editable tile above only toggle themselves.
- Because every click affects the tile above, a natural way to solve or verify the level is bottom-up: decide lower rows first, then compensate for the changes they cause above.
- The final win check is still the same marker-constraint system used by every previous level.

## Implementation Checklist

- The game exposes only the click action.
- All editable tiles begin at the first color in the level's cycle.
- A click cycles affected tiles to the next color in that level's cycle, wrapping at the end.
- Target markers check only the eight surrounding lattice positions, and only if an editable tile exists there.
- `=` constraints require equality with the marker center color; `!` constraints require inequality with that center color.
- Multiple target markers combine by intersection of constraints.
- Special tile effects include the clicked tile plus the magenta effect-mask positions.
- The click budget is restored on level reset and decremented only after a non-winning effective tile click.
- Level completion happens as soon as all target markers are satisfied.
