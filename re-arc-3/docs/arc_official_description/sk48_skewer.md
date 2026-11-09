# Skewer

## Core Game

Skewer is a 6-pixel-grid stacking puzzle. The player controls one or more square skewer handles. A handle has a row or column of gray skewer segments extending from it. Colored 6x6 blocks act like marshmallows: they can be pushed, pierced onto a skewer segment, carried while pierced, transferred through intersections, and arranged into ordered sequences.

The bottom display shows the required sequence for each scoring skewer. A level is solved when every scoring skewer in the play area carries the same color sequence as its matching bottom display, read from the handle outward. The first gray segment at the handle is structural; target colors begin on the next visible target segment.

## Color Legend

The block colors used by the puzzles are:

```text
R = red    (8)
B = blue   (9)
O = orange (12)
G = green  (14)
```

Pink handles are the standard horizontal player skewers. Purple handles are the second controllable skewer used in the later two-pole levels. Cyan-marked and yellow-marked handles are fixed target sticks in level 4.

## Reconstruction Model

Use a 64x64 logical grid. All handles, colored blocks, black obstacles, and skewer cells occupy 6x6 top-left-aligned cells. Most levels use a 42x42 board at `(11, 6)` or `(11, 8)`, with usable cell positions spaced by 6 pixels. Level 1 uses a smaller 30x30 board at `(17, 12)`.

State needed for play:

- active skewer handle
- each handle's top-left position, orientation, and ordered segment list
- each colored block's top-left position
- per-level fixed board bounds and black obstacle positions
- undo history for handle positions, segment lengths, and block positions
- the 196-step move meter

The skewer orientation determines how directional actions behave:

```text
horizontal handle, rotation 0: forward = right, backward = left
vertical handle, rotation 90: forward = down, backward = up
```

A forward action extends the active skewer by one 6-pixel cell. Existing segments move one cell forward, and a new segment is inserted at the handle. A backward action retracts the skewer by one cell. The segment nearest the handle is removed, and the remaining segments move one cell backward. A skewer cannot retract below one segment.

Perpendicular movement slides the entire active handle and its complete segment list one cell, but only when a gray rail exists next to the handle in that direction. Horizontal skewers therefore move up and down on vertical rails. Vertical skewers move left and right on horizontal rails. If there is no rail segment for the requested slide, the action is rejected.

Colored blocks are moved by the same collision recursion as the skewer segments:

- A segment moving into a block pushes that block one cell in the same direction.
- A pushed block can push a chain of other blocks.
- If a block moves onto a skewer segment of the movement's axis, it becomes pierced by that skewer.
- A pierced block shares the exact coordinate of one skewer segment and moves with that segment.
- A block can sit at the intersection of a horizontal and a vertical skewer, making it count for both skewers.
- Movement is blocked by the board boundary, a black obstacle, an unmovable block chain, or an incompatible perpendicular skewer collision.

After each completed movement, the game scans each scoring skewer from its handle outward and records the colored blocks found on its segments. Empty segments are skipped. The resulting color list must begin with the full target list shown on the matching bottom display. In normal solutions there are no irrelevant extra blocks on the scoring skewer, but the important visible rule is that the required colors must appear in the displayed order.

Directional movement actions consume one point from the 196-step meter, including rejected direction attempts. Clicking a different controllable skewer and using undo do not consume the meter. Reaching zero causes game over. Reset restores the current level, while undo restores the previous completed movement state, including handle positions, skewer lengths, and block positions. Click selection itself is not part of the undo snapshot.

When all scoring skewers match their targets, the matched blocks blink for a short completion animation and the game advances to the next level.

## Level 1

Level 1 introduces a single horizontal pink skewer and a simple reorder target.

Layout:

```text
Board: 30x30 at (17, 12)
Active pink handle: (11, 36), horizontal
Initial pink segments: (11, 36), (17, 36)
Vertical rail: x = 13, y = 14, 20, 26, 32

Loose blocks:
  R at (41, 18)
  B at (41, 24)
  G at (41, 30)

Bottom target:
  pink display handle at (20, 56)
  required sequence: R, G, B
```

The pink handle sits just left of the board and can slide between the five board rows using the left vertical rail. Moving right extends the skewer into the board. Moving left retracts it. Moving up or down slides the whole skewer and any pierced blocks.

To win, the player must pierce and arrange the three loose blocks so the active pink skewer reads red, green, blue from the handle outward. The initial column is red, blue, green from top to bottom, so simply sweeping through it in order does not match the target. The useful lesson is that once a block is on the skewer, vertical sliding moves the carried load without changing the handle-outward order.

Important reconstruction details:

- There are no obstacles or secondary sticks.
- The bottom pink display is not controllable; it only defines the target order.
- The active skewer starts with one segment outside the board at the handle and one segment inside the board.

## Level 2

Level 2 keeps one pink skewer but expands the board and starts the loose blocks in a row that must be displaced and reordered.

Layout:

```text
Board: 42x42 at (11, 6)
Active pink handle: (5, 42), horizontal
Initial pink segments: (5, 42), (11, 42)
Vertical rail: x = 7, y = 8, 14, 20, 26, 32, 38

Loose blocks on board row y = 24:
  G at (29, 24)
  B at (35, 24)
  O at (41, 24)
  R at (47, 24)

Bottom target:
  pink display handle at (17, 56)
  required sequence: R, O, B, G
```

The active handle can slide through the full height of the board. The four loose blocks begin in the opposite useful order from the target. Because the skewer pushes whatever it hits, the player must first move blocks away from the packed row, then poke through them in a target-compatible order.

To win, arrange the active skewer as red, orange, blue, green from the handle outward. The level teaches that extending into a packed line is not just a pickup action: it also pushes blocks and can shift the row, so solving requires using empty board space before committing to the final impalement order.

Important reconstruction details:

- The level has no fixed obstacle.
- All four required blocks begin loose in the board, not already on the skewer.
- The skewer can reach rows from the board top to the board bottom through the left rail.

## Level 3

Level 3 introduces a fixed vertical stick that already holds blocks. The player still controls only the pink horizontal skewer.

Layout:

```text
Board: 42x42 at (11, 6)
Active pink handle: (5, 42), horizontal
Initial pink segments: (5, 42), (11, 42)
Vertical rail: x = 7, y = 8, 14, 20, 26, 32, 38

Fixed yellow-marked vertical stick:
  handle at (29, 0), vertical
  segments at (29, 0), (29, 6), (29, 12), (29, 18), (29, 24)
  carried blocks from top downward: G at (29, 6), B at (29, 12), R at (29, 18), O at (29, 24)

Bottom target:
  pink display handle at (17, 56)
  required sequence: R, O, B, G
```

Blocks on the vertical stick are already pierced vertically. They cannot be treated like independent loose blocks without first using the stick geometry. The pink skewer must interact with the stack row by row, moving blocks down along or away from the vertical stick until it can poke through them in the order needed for the pink target.

To win, the active pink skewer must read red, orange, blue, green from the handle outward. This is not the top-to-bottom order of the fixed stick, so the player must manipulate the mounted stack before collecting the final sequence.

Important reconstruction details:

- The fixed vertical stick is physical: its segments participate in collisions and can hold blocks, but its handle is not controllable.
- A block can be moved from the fixed stick onto the pink skewer only through legal skewer/block movement; it is not a pickup toggle.
- The target still checks only the active pink skewer against the bottom pink display.

## Level 4

Level 4 changes the objective from one carried stack to distributing one load across two fixed target sticks.

Layout:

```text
Board: 42x42 at (11, 6)
Active pink helper handle: (5, 42), horizontal
Initial pink helper segments: (5, 42), (11, 42), (17, 42), (23, 42), (29, 42)
Initial pink helper load from handle outward: B, G, R, O
Vertical rail: x = 7, y = 8, 14, 20, 26, 32, 38

Fixed cyan-marked vertical stick:
  handle at (23, 0), vertical
  segments at (23, 0), (23, 6), (23, 12), (23, 18)
  required sequence: B, G

Fixed yellow-marked vertical stick:
  handle at (35, 0), vertical
  segments at (35, 0), (35, 6), (35, 12), (35, 18)
  required sequence: R, O

Bottom displays:
  yellow-marked display at (11, 56), target R, O
  cyan-marked display at (35, 56), target B, G
```

The pink skewer is a carrier and separator, not a scoring target in this level. It starts with all four blocks already pierced in the order blue, green, red, orange. The two fixed vertical sticks hanging from the top are the scoring skewers. The player must separate the load and place the first two colors on the cyan-marked stick and the last two colors on the yellow-marked stick.

To win, the cyan-marked stick must read blue, green from its handle downward, and the yellow-marked stick must read red, orange from its handle downward. The blocks do not have to occupy the exact same vertical slots as the bottom UI, but their order along each fixed stick must match.

Important reconstruction details:

- Only the pink helper handle is controllable. The cyan-marked and yellow-marked top handles are fixed collision/target objects.
- The bottom displays are paired by handle marker color: cyan top to cyan bottom, yellow top to yellow bottom.
- Since the pink load begins as one continuous sequence, solving requires retracting, pushing, and poking blocks onto the two target sticks one by one rather than depositing the entire load at once.

## Level 5

Level 5 adds an immovable black obstacle that interrupts a horizontal line of blocks.

Layout:

```text
Board: 42x42 at (11, 6)
Active pink handle: (5, 24), horizontal
Initial pink segments: (5, 24), (11, 24), (17, 24), (23, 24)
Initial pink load from handle outward: B, B, B
Vertical rail: x = 7, y = 8, 14, 20, 26, 32, 38

Black obstacle:
  (29, 24)

Loose blocks to the right of the obstacle:
  R at (35, 24)
  R at (41, 24)
  R at (47, 24)

Bottom target:
  pink display handle at (20, 56)
  required sequence: R, B, R
```

The active skewer starts on the same row as the obstacle. Three blue blocks are already pierced on the pink skewer, and three red blocks sit beyond the obstacle. The obstacle blocks movement for skewers and blocks; it cannot be pushed, crossed, or pierced through.

To win, the pink skewer must read red, blue, red from the handle outward. The player must use the rail to move around the obstacle, push blocks into workable positions, collect red blocks from the far side, and interleave them with a retained blue block. Extending straight right from the start is blocked by the black square, so the puzzle requires alternating between sliding, pushing, retracting, and re-extending.

Important reconstruction details:

- The black obstacle is a single 6x6 solid cell at `(29, 24)`.
- It blocks both skewer segments and colored blocks.
- The target is shorter than the starting load. The player must discard or move surplus blue blocks out of the scoring order.

## Level 6

Level 6 introduces two controllable skewers. Clicking switches the active skewer between the pink horizontal pole and the purple vertical pole.

Layout:

```text
Board: 42x42 at (11, 8)

Pink skewer:
  handle at (5, 26), horizontal
  initial segments at (5, 26), (11, 26)
  vertical rail: x = 7, y = 10, 16, 22, 28, 34, 40
  required sequence: B, B, B

Purple skewer:
  handle at (29, 2), vertical
  initial segments at (29, 2), (29, 8)
  horizontal rail: y = 4, x = 13, 19, 25, 31, 37, 43
  required sequence: R, R, R

Black obstacle:
  (29, 26)

Loose blocks:
  R at (29, 32)
  R at (29, 38)
  R at (29, 44)
  B at (35, 26)
  B at (41, 26)
  B at (47, 26)

Bottom displays:
  pink display at (5, 56), target B, B, B
  purple display at (35, 56), target R, R, R
```

The pink skewer slides vertically on the left rail and extends horizontally into the board. The purple skewer slides horizontally on the top rail and extends downward into the board. The black obstacle sits where the central horizontal and vertical lanes would otherwise meet.

To win, the pink skewer must hold the three blue blocks, and the purple skewer must hold the three red blocks. The blocks start in regions naturally suited to different axes: blues are to the right on the pink row, while reds are below the purple column. The obstacle forces the player to coordinate both poles and move blocks around the central blockage instead of solving with one straight extension.

Important reconstruction details:

- Both top handles are clickable controllable objects. The active handle is visually highlighted.
- The bottom pink and purple displays are not controllable.
- A click changes the active skewer but does not itself change block positions or consume the move meter.

## Level 7

Level 7 keeps the two controllable skewers but requires one block to be pierced by both skewers so it can serve as the middle item in two different target patterns.

Layout:

```text
Board: 42x42 at (11, 8)

Pink skewer:
  handle at (5, 26), horizontal
  initial segments at (5, 26), (11, 26)
  vertical rail: x = 7, y = 10, 16, 22, 28, 34, 40
  required sequence: B, G, B

Purple skewer:
  handle at (29, 2), vertical
  initial segments at (29, 2), (29, 8)
  horizontal rail: y = 4, x = 13, 19, 25, 31, 37, 43
  required sequence: R, G, R

Loose blocks:
  R at (23, 20)
  B at (29, 20)
  B at (23, 26)
  R at (29, 26)
  G at (35, 32)

Bottom displays:
  pink display at (5, 56), target B, G, B
  purple display at (35, 56), target R, G, R
```

There is only one green block, but both target sequences require green in the middle. The intended solution is to make that green block a shared intersection block: it must lie on a pink horizontal segment and on a purple vertical segment at the same time. Then the same green block counts in both scans.

To win, the pink skewer must read blue, green, blue from left handle outward, while the purple skewer reads red, green, red from top handle downward. The player cannot solve this by collecting the green on only one skewer; the other target would be missing its middle block. The two poles must poke through the shared block from different axes and then carry or align their outer color blocks around it.

Important reconstruction details:

- The level has no black obstacle.
- One block may be on both skewers simultaneously when it is at a horizontal/vertical segment intersection.
- The win scan for each skewer independently includes the shared block at its position in that skewer's order.

## Level 8

Level 8 uses the same two-pole transfer idea with much shorter rails and smaller target sequences.

Layout:

```text
Board: 42x42 at (11, 8)

Pink skewer:
  handle at (5, 26), horizontal
  initial segments at (5, 26), (11, 26)
  restricted vertical rail: x = 7, y = 22, 28
  reachable handle rows: 20, 26, 32
  required sequence: B, G

Purple skewer:
  handle at (29, 2), vertical
  initial segments at (29, 2), (29, 8)
  restricted horizontal rail: y = 4, x = 25, 31
  reachable handle columns: 23, 29, 35
  required sequence: R, O

Loose blocks:
  B at (29, 14)
  G at (29, 38)
  R at (17, 26)
  O at (41, 26)

Bottom displays:
  pink display at (11, 56), target B, G
  purple display at (35, 56), target R, O
```

The blue and green blocks start on the central vertical line, where the purple skewer can naturally interact with them. The red and orange blocks start on the central horizontal row, where the pink skewer can naturally interact with them. The targets ask for the opposite distribution: pink needs blue and green, and purple needs red and orange.

To win, the player must use the crossing mechanics to transfer the vertical-line colors onto the pink skewer and the horizontal-line colors onto the purple skewer. The restricted rails limit the pink handle to three rows and the purple handle to three columns, so there is much less space to reorder mistakes. The same push, pierce, retract, and two-axis transfer rules from level 7 apply, but the geometry leaves fewer recovery paths.

Important reconstruction details:

- There is no black obstacle, but the short rails are the main constraint.
- Pink can only slide to rows 20, 26, and 32.
- Purple can only slide to columns 23, 29, and 35.
- The target sequences are shorter than in level 7, but the required transfer direction is stricter because each color begins on the other pole's natural axis.
