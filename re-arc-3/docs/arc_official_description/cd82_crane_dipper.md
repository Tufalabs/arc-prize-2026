# Crane Dipper

## Core Game

Crane Dipper is a 10x10 pattern-copying puzzle. A reference pattern is shown in the upper-left frame. The editable 10x10 canvas starts as all white near the center of the board. The player moves a paint basket around the eight outer positions of a 3x3 ring around the canvas, selects a paint color from the top toolbar, and applies paint with Spacebar. Later levels add a smaller dipper that can be clicked to paint a narrow inset strip instead of a large half-plane or triangle.

Paint does not blend. Every application overwrites the affected canvas cells with the current selected color. The challenge is choosing the side, color, tool, and order so later operations overwrite only the regions that should change.

The level is solved after a paint operation when the editable canvas matches the reference on every cell except the two main diagonals. The visible reference still includes colors on those diagonal cells, but the win check ignores cells where `row == column` or `row + column == 9`. This matters because the triangular tools include their boundary diagonal cells.

## Color Legend

The paint colors used by this game are:

```text
. = white  (0)
P = purple (15)
O = orange (12)
Y = yellow (11)
G = green  (14)
R = red    (8)
B = blue   (9)
```

The black background and gray frame are not paint colors for the puzzle pattern.

## Board Layout

Use a 64x64 logical grid.

- The reference pattern is a 10x10 sprite at `(3, 3)`, framed by the gray upper-left display.
- The editable canvas is a 10x10 sprite at `(27, 34)` and starts all white on every level.
- The active large basket starts at the top slot around the canvas and starts purple.
- The top toolbar contains 5x5 color swatches at `y = 2`. Clicking any part of a swatch selects its center color and moves a 5-pixel white selection marker to `y = 7` under that swatch.
- Level 1 has swatches at `x = 35` for white and `x = 41` for purple.
- Level 2 has swatches at `x = 32` for white, `x = 38` for purple, and `x = 44` for orange.
- Levels 3-6 have swatches at `x = 21, 27, 33, 39, 45, 51, 57` for white, purple, orange, yellow, green, red, and blue.

The state needed to reconstruct play is:

- current basket slot, initially slot 0
- current paint color, initially purple
- current 10x10 canvas pixels, initially all white
- whether the level has the small dipper
- action count, capped at 100 actions

There is no player avatar and no collision maze. Mistakes persist as painted pixels until they are overwritten by later paint operations or the level is reset.

## Controls

Directional actions move the active basket one position around the eight-slot ring. The ring is:

```text
7 0 1
6 . 2
5 4 3
```

The center is not a valid position. Moving into the center or outside the ring leaves the basket where it is.

Spacebar applies the large basket at the current slot. The basket animates inward for seven frames, writes its paint mask at the turn-around point, then returns to its original slot.

Clicking a color swatch changes the current color. The active basket, and the small dipper when present, immediately recolor to the selected color.

In levels 3-6, a small dipper is visible only at slots 0, 2, 4, and 6. Clicking it applies its small strip mask. It also animates inward, writes the strip, and returns.

The bottom row is a 100-action meter. Each arrow, click, or Spacebar action increments the action count once. Reaching 100 actions causes game over. Reset restores the current level to its starting basket slot, selected color, blank canvas, swatches, and reference pattern.

## Paint Masks

Rows and columns below are 0-indexed within the 10x10 canvas.

Large basket masks:

```text
slot 0, top:         rows 0-4, all columns
slot 1, upper-right: cells where column >= row
slot 2, right:       columns 5-9, all rows
slot 3, lower-right: cells where column >= 9 - row
slot 4, bottom:      rows 5-9, all columns
slot 5, lower-left:  cells where column <= row
slot 6, left:        columns 0-4, all rows
slot 7, upper-left:  cells where column <= 9 - row
```

Small dipper masks:

```text
slot 0, top:    rows 0-2, columns 3-6
slot 2, right:  rows 3-6, columns 7-9
slot 4, bottom: rows 7-9, columns 3-6
slot 6, left:   rows 3-6, columns 0-2
```

The small dipper is removed while the basket is on a diagonal slot and reappears when the basket returns to a cardinal slot.

## Level 1

Level 1 teaches that the basket position determines which side of the canvas is painted. The level has only the white and purple swatches, starts with purple selected, and has no small dipper.

Target:

```text
..........
..........
..........
..........
..........
PPPPPPPPPP
PPPPPPPPPP
PPPPPPPPPP
PPPPPPPPPP
PPPPPPPPPP
```

To win, move the basket from the starting top slot to slot 4 at the bottom of the canvas and press Spacebar. This paints the bottom half purple while leaving the top half white. Pressing Spacebar at the starting top slot paints the wrong half and must be corrected with later overwriting or reset.

Important reconstruction details:

- Only the large horizontal basket is needed.
- The initial selected color is purple.
- The correct operation is one purple large-basket bottom-half paint.

## Level 2

Level 2 adds color selection. It has white, purple, and orange swatches, starts with purple selected, and still has no small dipper.

Target:

```text
PPPPPPPPPO
PPPPPPPPOO
PPPPPPPOOO
PPPPPPOOOO
PPPPPOOOOO
....OOOOOO
...OOOOOOO
..OOOOOOOO
.OOOOOOOOO
OOOOOOOOOO
```

This target is made by layering a purple top half with an orange lower-right anti-diagonal triangle. One constructive order is:

```text
1. Purple, slot 0 large basket: paint rows 0-4 purple.
2. Orange, slot 3 large basket: paint column >= 9 - row orange.
```

The order is important. If the orange triangle is painted first and the purple top half is painted afterward, the purple operation overwrites the orange cells in the top half and the reference cannot match.

Important reconstruction details:

- The orange swatch is introduced at the top toolbar.
- The diagonal boundary cells are visible in the reference but ignored by validation.
- No small strip operation exists yet.

## Level 3

Level 3 introduces the small dipper. All seven paint colors are available, the selected color still starts as purple, and the small dipper appears above, right, below, or left of the canvas when the basket is on one of those four slots.

Target:

```text
PPPOOOOPPP
PPPOOOOPPG
PPPOOOOPGG
PPPPPPPGGG
PPPPPPGGGG
PPPPPGGGGG
PPPPRGGGGG
PPPRRGGGGG
PPRRRGGGGG
PRRRRGGGGG
```

One constructive order is:

```text
1. Red, slot 4 large basket: paint the bottom half red.
2. Green, slot 2 large basket: paint the right half green.
3. Purple, slot 7 large basket: paint column <= 9 - row purple.
4. Orange, slot 0 small dipper: paint rows 0-2, columns 3-6 orange.
```

The small top strip is what makes the orange inset possible. A large orange pour from the top would recolor the entire top half and destroy the purple regions that must remain around the inset.

Important reconstruction details:

- The small dipper's color is always the current selected color.
- The small dipper operation overwrites only its 4x3 strip.
- The target uses four colors but can be produced with four overwrite operations.

## Level 4

Level 4 keeps the small dipper and changes the layering pattern. All seven swatches are available.

Target:

```text
BBBBBOOOOP
BBBBBOOOPP
BBBBBOOPPP
YYYBBOPPPP
YYYBBPPPPP
YYYBBPPPPP
YYYBBPPPPP
BBBBBPPPPP
BBBBBPPPPP
BBBBBPPPPP
```

One constructive order is:

```text
1. Orange, slot 0 large basket: paint the top half orange.
2. Purple, slot 3 large basket: paint column >= 9 - row purple.
3. Blue, slot 6 large basket: paint the left half blue.
4. Yellow, slot 6 small dipper: paint rows 3-6, columns 0-2 yellow.
```

This order leaves the right side purple, the left side blue, a narrow orange diagonal-adjacent band near the top, and a yellow inset on the left-middle edge.

Important reconstruction details:

- The yellow cells are not a separate large region; they are the left-side small dipper strip.
- The blue left-half paint must happen after the orange and purple operations so it overwrites the left side back to blue.
- The final yellow strip must be applied after the blue left half or it will be covered.

## Level 5

Level 5 uses the full palette and small dipper with a different arrangement of diagonals and the top inset.

Target:

```text
GBBRRRRBBO
GGBRRRRBOO
GGGRRRROOO
GGGGBBOOOO
GGGGGOOOOO
GGGGOOOOOO
GGGOOOOOOO
GGOOOOOOOO
GOOOOOOOOO
OOOOOOOOOO
```

One constructive order is:

```text
1. Blue, slot 0 large basket: paint the top half blue.
2. Red, slot 0 small dipper: paint rows 0-2, columns 3-6 red.
3. Green, slot 5 large basket: paint column <= row green.
4. Orange, slot 3 large basket: paint column >= 9 - row orange.
```

The final orange lower-right anti-diagonal must come after the green lower-left main-diagonal pour, because those two large triangles overlap in the lower half. The red top inset must be made with the small dipper so the surrounding blue top cells remain blue.

Important reconstruction details:

- The top strip is red, not produced by a diagonal tool.
- The green region is the lower-left main-diagonal large basket mask.
- The orange region is the lower-right anti-diagonal large basket mask and should be applied after green.

## Level 6

Level 6 is the final layering challenge. It uses the full palette and the small dipper.

Target:

```text
RRRPPPPRRR
RRRPPPPRRG
RRRPPPPRGG
YYYRRRRGGG
YYYRRRGGGG
YYYRRGGGGG
YYYR.GGGGG
RRR..GGGGG
RR...GGGGG
R....GGGGG
```

One constructive order is:

```text
1. Green, slot 2 large basket: paint the right half green.
2. Red, slot 7 large basket: paint column <= 9 - row red.
3. Yellow, slot 6 small dipper: paint rows 3-6, columns 0-2 yellow.
4. Purple, slot 0 small dipper: paint rows 0-2, columns 3-6 purple.
```

The untouched white wedge in the lower-middle left remains white because neither the red upper-left anti-diagonal nor the green right-half mask covers it. The two small strips then overwrite the red areas that need to become yellow and purple.

Important reconstruction details:

- The purple top-center area is a small dipper strip, not a large top-half pour.
- The yellow left-middle area is also a small dipper strip.
- The white cells in the lower-middle are intentional unpainted cells and must not be overwritten near the end.

## Implementation Checklist

- Render a 64x64 board with a top-left reference pattern, top color swatches, and a central 10x10 editable canvas.
- Keep the canvas as the only mutable pattern surface.
- Start each level with the basket at slot 0, current color purple, and an all-white canvas.
- Move the basket on the eight-slot ring and prevent movement into the center.
- Recreate the large basket masks and small dipper masks exactly as listed above.
- Apply paint by overwriting cells; do not mix or blend colors.
- In levels 1-2, omit the small dipper. In levels 3-6, show it only at cardinal slots.
- After every paint operation, compare the canvas to the level target while ignoring both diagonals.
- Advance to the next level immediately on a match.
- Maintain a 100-action limit and reset the current level to its initial state when reset is requested.
