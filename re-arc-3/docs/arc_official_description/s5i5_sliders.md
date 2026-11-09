# Sliders

## Core Game

Sliders is a click-only puzzle played on a 64x64 render grid. There is no avatar. The player manipulates colored slider pieces by clicking matching controls. A level is solved when every fixed magenta target square is covered by a movable magenta marker square.

Most movable pieces are 3 pixels thick and have a gray cap on their anchored side. A slider's length is always an integer number of 3-pixel cells. Extending or retracting a slider changes its length by one cell, never below one cell. The gray cap remains fixed, and the free end moves:

- A cap on the left means the slider extends right.
- A cap on the right means the slider extends left.
- A cap on the top means the slider extends down.
- A cap on the bottom means the slider extends up.

Some pieces carry child pieces. When a parent translates, extends, retracts, or rotates, all of its children move with it recursively. A child can also have its own control, so stacked pieces can change their relative offsets after the parent has moved them.

The descriptions below use top-left pixel coordinates on the 64x64 board. For wall maps, cell `(cx, cy)` means the 3x3 pixel block whose top-left corner is `(3 * cx, 3 * cy)`.

## Controls and Rules

Slider controls are gray U-shaped panels containing a colored glyph. Clicking the short/left side of a horizontal control retracts every slider of that color; clicking the long/right side extends them. Vertical controls work the same way with top for retract and bottom for extend. A single control affects every solid piece whose interior color matches that control, so later puzzles use same-color coupling as an obstacle.

Plus-shaped controls appear from level 6 onward. Clicking a plus rotates every same-colored piece or assembly 90 degrees counterclockwise around its gray-capped base. Descendants rotate around the same pivot with the parent.

All slider pieces, carried blockers, and purple walls are solid. After each click, if any two solid pieces overlap, the attempted transformation is rejected and the affected pieces are restored. The click still costs one step. Each level has a step counter shown as the bottom row of the render; reaching zero fails the level. There are no lives or movement actions.

Fixed magenta targets are checked by exact top-left coordinate. A level advances only when each fixed target has a movable magenta marker at the same coordinate.

## Color Legend

The main colored controls and pieces are green, yellow, orange, cyan, red, blue, and pink. Purple pieces are static walls or blockers unless otherwise described. White pieces are carried wall attachments. Magenta marks are targets and movable endpoint markers.

## Level 1

Step limit: 50.

This level has two independent sliders and no walls. It teaches that the U-shaped controls extend and retract matching pieces.

Pieces:

```text
Yellow slider: (9,27), size 3x9, capped at top, extends down.
Green slider:  (27,9), size 6x3, capped at left, extends right.
Yellow marker: starts at (9,33), target at (9,51).
Green marker:  starts at (30,9), target at (51,9).
```

Controls:

```text
Green horizontal control at (36,18).
Yellow vertical control at (21,35).
```

To win, extend the yellow slider until its marker has moved 18 pixels down to `(9,51)`, and extend the green slider until its marker has moved 21 pixels right to `(51,9)`. Retraction is legal but unnecessary unless the player overshoots.

## Level 2

Step limit: 150.

This level introduces stacked sliders. Four one-cell blocks are linked in a chain. Moving an earlier block carries all later blocks, while moving a later block changes only that later part of the stack. The stack must be threaded through a purple wall layout so its endpoint marker reaches the right-side target.

Purple wall cells:

```text
(11,3) size 8x1
(11,4) size 1x5
(14,5) size 1x10
```

Stack, from parent to child:

```text
Orange block: (9,39), extends right, carries cyan.
Cyan block:   (12,39), extends up, carries yellow.
Yellow block: (12,36), extends right, carries green.
Green block:  (15,36), extends down, carries the marker.
Marker: starts at (15,36), target at (51,30).
```

Controls:

```text
Orange at (3,54), cyan at (18,54), yellow at (33,54), green at (48,54).
```

The target marker must end 36 pixels to the right and 6 pixels above its start. The wall layout prevents simply extending everything in place. The intended puzzle is to change the relative positions of the four blocks so the stack can squeeze around the vertical wall segments, then finish with the green marker exactly on `(51,30)`. Any block-wall or block-block overlap rolls the attempted click back.

## Level 3

Step limit: 200.

This level uses sliders as movable walls. Two final markers must be placed on two fixed targets along the same row, but two larger assemblies initially block the routes.

Static wall:

```text
Purple stopper cell at (15,9), pixel position (45,27).
```

Final markers:

```text
Pink marker: starts at (48,3), target at (48,27).
Cyan marker: starts at (6,27), target at (42,27).
```

Target sliders:

```text
Pink block at (48,3), extends down, carries the pink marker.
Cyan block at (6,27), extends right, carries the cyan marker.
```

Blocking assemblies:

```text
Blue block at (30,36), extends left, carries a red vertical slider.
Red vertical slider at (27,18), size 3x21, extends up, carries a white horizontal wall.
White horizontal wall at (27,15), size 24x3.

Green block at (54,24), extends up, carries an orange horizontal slider.
Orange horizontal slider at (39,21), size 18x3, extends left, carries a white vertical wall.
White vertical wall at (36,21), size 3x18.
```

Controls:

```text
Cyan at (26,45), orange at (45,45), green at (45,54),
red at (7,45), blue at (7,54), pink at (26,54).
```

To win, first move the red/white and orange/white wall assemblies out of the paths. The pink slider needs a clear vertical line from `(48,3)` to `(48,27)`, and the cyan slider needs a clear horizontal line from `(6,27)` to `(42,27)`. The purple stopper near `(45,27)` is a collision constraint on the target row, so the blockers must be arranged before extending the final pink and cyan target sliders.

## Level 4

Step limit: 100.

This level is a vertical "stream" puzzle. A yellow marker starts low in the board and must be extended upward to the top-center target. Several yellow child blocks share the same yellow control, so they move when the stream moves; their colored parent blocks must be used to shift them out of the stream's path.

Final marker:

```text
Yellow slider: (30,42), one cell, capped at bottom, extends up.
Marker: starts at (30,42), target at (30,9).
```

Coupled yellow blockers:

```text
Red block at (45,21) carries yellow block at (45,18).
Blue block at (51,15) carries yellow block at (51,12).
Green block at (9,18) carries yellow block at (9,15).
Orange vertical slider at (18,27), size 3x27, carries yellow block at (18,24).
```

Controls:

```text
Yellow at (25,51), orange at (3,54), green at (3,45),
blue at (48,45), red at (48,54).
```

The yellow stream marker must move 33 pixels upward to `(30,9)`. Because the yellow control affects every yellow piece, extending the stream also extends the yellow child blockers. The other colors are used to move those attached yellow pieces aside before the final yellow extension. The level is solved when the top-center target is covered, not when the visual stream merely reaches near it.

## Level 5

Step limit: 150.

This level combines target placement with a movable wall attachment. One target is already covered at the start; the other marker must be routed through gaps while preserving or restoring the already-covered target.

Purple wall cells:

```text
(5,3) size 1x3
(5,7) size 1x2
(12,12) size 1x1
```

Targets:

```text
Green-chain marker: starts at (9,9), target at (21,36).
Cyan marker: starts at (36,24), target at (36,24), already covered.
```

Movable target assemblies:

```text
Orange block at (9,6), extends down, carries green block at (9,9).
Green block at (9,9), extends right, carries the green-chain marker.

Cyan horizontal slider at (21,24), size 18x3, extends right,
carries the already-aligned marker at (36,24).
```

Movable wall assembly:

```text
Green block at (54,33), extends up, carries a blue horizontal slider.
Blue horizontal slider at (33,30), size 24x3, extends left, carries a white vertical wall.
White vertical wall at (30,30), size 3x18.
```

Controls:

```text
Green at (34,55), blue at (4,55), orange at (19,55), cyan at (49,55).
```

The movable white wall blocks the route to `(21,36)`. Use the green and blue controls to shift that wall assembly out of the way, then use the orange/green target chain to move the magenta marker around the purple blockers and onto `(21,36)`. The cyan marker begins solved, so moving the cyan slider is risky unless it is returned to `(36,24)` before the final state.

## Level 6

Step limit: 150.

This level introduces rotation. A three-segment jointed chain starts above a purple wall enclosure. The marker on the terminal segment must be folded and extended into the lower-right target position.

Purple wall cells:

```text
(1,9) size 2x1
(6,9) size 14x1
(1,10) size 1x4
(19,10) size 1x4
(15,13) size 4x1
```

Jointed chain:

```text
Green vertical slider at (39,12), size 3x6, extends up, carries yellow.
Yellow horizontal slider at (39,9), size 9x3, extends right, carries blue.
Blue vertical slider at (48,9), size 3x6, extends down, carries the marker.
Marker: starts at (48,12), target at (51,33).
```

Controls:

```text
Slider controls: blue at (6,54), yellow at (25,54), green at (44,54).
Rotation controls: blue plus at (9,45), yellow plus at (28,45), green plus at (47,45).
```

The target lies below the wall's top band and near its right side. Extension alone cannot place the marker there; the chain must be rotated at its colored bases and resized so it passes around the wall geometry. The plus controls rotate the relevant colored segment and its descendants, so changing the green base rotates the entire chain, changing yellow folds the yellow/blue tail, and changing blue rotates only the terminal blue segment and marker.

## Level 7

Step limit: 200.

This level is an orientation-and-hole puzzle. One marker is already aligned on the left; the active marker starts on a four-color chain at the right and must pass through the wall openings to reach the other fixed target.

Purple wall cells:

```text
(13,0) size 1x10
(0,9) size 2x1
(3,9) size 7x1
(11,9) size 2x1
(5,10) size 1x3
(5,14) size 1x2
(0,15) size 5x1
(6,15) size 15x1
```

Targets:

```text
Cyan marker: starts at (21,6), target at (21,6), already covered.
Orange-chain marker: starts at (54,15), target at (24,15).
```

Assemblies:

```text
Cyan horizontal slider at (21,6), size 15x3, extends left, carries the solved marker.
Red vertical slider at (12,6), size 3x15, extends down, used as a rotatable blocker.

Yellow block at (48,12), extends up, carries green.
Green horizontal slider at (48,9), size 6x3, extends right, carries blue.
Blue vertical slider at (54,9), size 3x6, extends down, carries orange.
Orange block at (54,15), extends left, carries the active marker.
```

Controls:

```text
Slider controls: green at (2,56), yellow at (2,49), blue at (24,49),
orange at (24,56), cyan at (52,49).
Rotation controls: green plus at (15,56), yellow plus at (15,49),
blue plus at (37,49), red plus at (58,56).
```

The active chain must be folded so it can pass through the wall holes instead of colliding with the long purple barriers. The three relevant openings constrain the chain's orientation: a segment that is too long or rotated the wrong way hits the wall and rolls back. The win state keeps the cyan marker on `(21,6)` and places the orange-chain marker on `(24,15)`.

## Level 8

Step limit: 200.

The final level combines same-color coupling, rotation inside a box, and several attached blockers. The only required marker starts near the top-left and must be extended down to the lower-left target. The path is blocked by two carried white-wall assemblies, and the green control is coupled to a green arm inside a purple box.

Purple wall cells:

```text
(4,7) size 1x1
(13,8) size 7x1
(13,9) size 1x6
(19,9) size 1x6
(8,14) size 1x1
(14,14) size 2x1
(17,14) size 2x1
```

Required target:

```text
Green target slider at (18,3), one cell, capped at top, extends down.
Marker: starts at (18,3), target at (18,42).
```

Left-side blockers:

```text
Blue block at (27,18), extends up, carries a pink horizontal slider.
Pink horizontal slider at (9,15), size 21x3, extends left,
carries a white vertical wall at (6,15), size 3x12.

Red block at (9,39), extends up, carries a cyan horizontal slider.
Cyan horizontal slider at (9,36), size 21x3, extends right,
carries a white vertical wall at (30,36), size 3x15.
```

Boxed cross assembly:

```text
Yellow center block at (48,32), inside the purple box.
It carries four one-cell arms:
red at (48,29), green at (45,32), orange at (48,35), blue at (51,32).
```

Controls:

```text
Top-right sliders: blue at (37,2), orange at (51,2),
red at (37,9), green at (51,9).
Bottom sliders: cyan at (6,54), pink at (20,54).
Yellow rotation plus at (46,15).
```

The green slider at `(18,3)` needs a long clear vertical lane to `(18,42)`. The pink/white and cyan/white blockers cross that lane and must be moved aside with their parent controls. Extending green also tries to move the green arm of the boxed cross, so the yellow plus must rotate the cross until the green arm is aligned with the box opening; otherwise the green extension collides with the purple box and is rejected. After the blockers are cleared and the boxed green arm can legally protrude, extend the top-left green target slider until its marker exactly covers `(18,42)`.
