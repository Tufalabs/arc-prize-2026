# Reticle

## Core Game

Reticle is a 64x64 pixel alignment puzzle. The playfield is black with a faint grid and a purple step meter on the bottom row. The player controls one movable reticle at a time. The four direction actions move the selected reticle by exactly 3 pixels. The space action switches selection to the next reticle in that level. Direction actions and space both spend one step; when the step meter reaches zero the level is lost and must be restarted.

The selected reticle is marked by a white center pixel. That white selector pixel is not a target color. When selection changes away from a reticle, its center is restored according to its geometry: crosses regain their colored center, hollow rectangles and diamonds keep the center empty, and the fixed-center diagonal crosses regain their colored center.

Targets are shown as small dark 3x3 frames with one colored center pixel. Only the colored center pixel is required for solving; the dark frame is visual guidance. A level is solved when every target center is covered by any non-transparent pixel of the same color from the current reticle arrangement. Extra colored pixels away from target centers are allowed, and reticles may overlap. If two reticles overlap, the later-drawn reticle can cover pixels from the earlier one, so final overlap must not hide any required target center.

Coordinates below are pixel coordinates in the 64x64 playfield, with `(0,0)` at the upper left. Sprite positions are top-left coordinates; some valid final placements extend partly outside the playfield as long as the reticle's center remains inside the 64x64 camera.

## Palette

The important visible colors are:

```text
6 magenta     7 pink        8 red        9 blue
10 cyan       11 yellow     12 orange    13 maroon
14 green
```

Gray is used for recolor-pad borders, dark gray is used for target frames, white marks the selected reticle center and shape-transform blocks, black is the background, and purple is the step meter.

## Reticle Geometry

The game uses a few reusable reticle shapes:

- Cross: one full horizontal line and one full vertical line crossing inside the sprite. Sizes include 19x19, 23x23, 25x25, 27x27, 29x29, and a wide 37x19 cross.
- Diagonal X: two one-pixel diagonals in a square. Some X reticles have a fixed colored center when inactive.
- Diamond: a one-pixel hollow diamond outline.
- Horizontal line: a one-pixel-high line.
- Hollow rectangle: a colored rectangular border with transparent inside.

All reticles move by 3-pixel increments from their starting positions. Target coordinates are also arranged on this 3-pixel rhythm.

## Recolor Pads

From level 4 onward, gray-framed colored pads can recolor the selected reticle. If a reticle touches a pad whose interior color differs from the reticle's current color, the overlapping pixels change first and the new color spreads through the whole connected reticle over a short animation. Movement is suspended until recoloring completes. Pads are not consumed. Touching an unwanted pad after a reticle has the correct color can ruin the final state, especially in level 8.

## Shape-Transform Blocks

From level 6 onward, white hollow blocks deform reticles on contact.

For hollow rectangles, horizontal contact with a transform block makes the rectangle narrower and taller by 3 pixels: `width -= 3`, `height += 3`. Vertical contact makes it wider and shorter by 3 pixels: `width += 3`, `height -= 3`. A transform that would shrink the contacted dimension below the minimum is blocked.

For crosses, contact can slide one stroke through the other in 3-pixel increments. Horizontal contact can shift the vertical stroke left or right within the cross's bounding box. Vertical contact can shift the horizontal stroke up or down. If the crossing point itself would collide with the transform block, or the stroke has no room to slide farther, the move is blocked. This is how later levels create off-center crosses whose horizontal and vertical strokes no longer cross at the geometric center.

## Level 1

Step limit: 100.

Starting reticles:

```text
Yellow 23x23 cross: top-left (10,16), center (21,27)
Blue 27x27 cross:   top-left (23,32), center (36,45), initially selected
```

Targets:

```text
Yellow: (15,3), (6,9), (24,9), (15,17)
Blue:   (48,16), (40,24), (53,24), (48,35)
```

This is the selection and movement tutorial. Space switches between the two crosses; arrows move the selected cross by 3 pixels. The yellow targets form a cross centered at `(15,9)`, and the blue targets form a cross centered at `(48,24)`.

One valid solved arrangement is:

```text
Yellow cross top-left (4,-2), center (15,9)
Blue cross top-left (35,11), center (48,24)
```

The yellow cross intentionally extends above the top edge in the solved state. This is legal because movement bounds are based on the reticle center, not the whole sprite box.

## Level 2

Step limit: 100.

Starting reticles:

```text
Orange 23x23 diagonal X: top-left (16,7), center (27,18), initially selected
Maroon 19x19 diamond:    top-left (30,21), center (39,30)
Blue 27x27 cross:        top-left (35,29), center (48,42)
```

Targets:

```text
Maroon: (21,3), (27,9), (12,12)
Orange: (9,39), (24,42), (9,57)
Blue:   (27,36), (15,48), (33,48), (27,60)
```

This level adds a third reticle and makes the matching less visually obvious. The player must infer which colored target cluster belongs to each geometry: the maroon cluster matches a diamond outline, the orange cluster matches a diagonal X, and the blue cluster matches a cross.

One valid solved arrangement is:

```text
Maroon diamond top-left (9,0)
Orange X top-left (7,37)
Blue cross top-left (14,35), center (27,48)
```

The blue cross covers the four blue targets with its vertical stroke at `x=27` and horizontal stroke at `y=48`; no target lies on the selected center pixel.

## Level 3

Step limit: 200.

Starting reticles:

```text
Red 23x23 diagonal X: top-left (7,37), center (18,48)
Red 25x25 diamond:    top-left (33,36), center (45,48)
Red 43x1 line:        top-left (9,45), center (30,45), initially selected
```

Targets:

```text
Red: (6,6), (45,6), (51,15), (21,21),
     (9,27), (48,30), (33,33), (21,39)
```

All reticles and targets share the same color, so color no longer identifies which object belongs where. The player must match by geometry. The horizontal line is the only shape that can naturally cover the two upper targets at the same `y`; the diamond covers the lower-left triangular cluster; the X covers the right-side diagonal cluster.

One valid solved arrangement is:

```text
43x1 line top-left (3,6), covering (6,6) and (45,6)
25x25 diamond top-left (6,18), covering (21,21), (9,27), and (21,39)
23x23 X top-left (31,13), covering (51,15), (48,30), and (33,33)
```

## Level 4

Step limit: 200.

Starting reticles:

```text
Magenta 27x27 cross:          top-left (41,23), center (54,36), initially selected
Cyan 21x21 fixed-center X:    top-left (14,11), center (24,21)
```

Recolor pads are six 6x6 squares:

```text
Cyan (4,4)       Orange (28,4)    Maroon (52,4)
Yellow (4,54)    Magenta (28,54)  Green (52,54)
```

Targets:

```text
Orange: (15,18), (27,30), (15,43)
Green:  (48,21), (33,24), (30,39)
```

This level introduces recoloring. The starting magenta and cyan colors are not the solution colors. The cross must become orange, and the diagonal X must become green. The pads stay active, so after a reticle has the correct color it must not touch another pad before final placement.

One valid solved arrangement is:

```text
Recolor the 27x27 cross orange, then place it top-left (2,17).
Recolor the 21x21 X green, then place it top-left (29,20).
```

The orange cross covers its three targets with vertical stroke `x=15` and horizontal stroke `y=30`. The green X covers its three targets along its two diagonals.

## Level 5

Step limit: 250.

Starting reticles:

```text
Orange 29x29 cross:             top-left (40,19), center (54,33)
Yellow 23x23 fixed-center X:    top-left (13,31), center (24,42), initially selected
Green 19x19 diamond:            top-left (21,9), center (30,18)
```

Recolor pads are five 6x6 squares:

```text
Yellow (3,3)   Cyan (54,3)
Green (3,27)
Blue (3,52)    Red (54,52)
```

Targets:

```text
Blue: (21,6), (39,6), (33,45), (24,51), (45,51), (33,60)
Red:  (51,27), (57,33), (42,36), (54,42)
```

This is a larger recoloring puzzle. The top blue pair fits the X, the lower blue diamond-like cluster fits the large cross, and the red cluster fits the diamond outline. The starting colors are distractors; the final colors must be blue and red.

One valid solved arrangement is:

```text
Recolor the 29x29 cross blue, then place it top-left (19,37).
Recolor the 23x23 X blue, then place it top-left (19,4).
Recolor the 19x19 diamond red, then place it top-left (42,27).
```

The red pad is in the lower-right and the blue pad is in the lower-left, so routing matters: a reticle already recolored blue or red should avoid crossing the other pad unless it is meant to change again.

## Level 6

Step limit: 200.

Starting objects:

```text
White 8x8 transform block: top-left (28,28)
Blue 25x25 cross:          top-left (36,3), center (48,15)
Yellow 19x19 hollow square: top-left (6,39), center (15,48), initially selected
```

Targets:

```text
Blue:   (12,6), (9,9), (30,9), (12,27)
Yellow: (45,30), (54,30), (45,57), (54,57)
```

This level introduces shape transformation. The blue targets form an off-center cross: the horizontal stroke must run at `y=9` from `x=9` through `x=30`, while the vertical stroke must run at `x=12` from `y=6` through `y=27`. The yellow targets are the four corners of a tall rectangle: they cannot be covered by the starting 19x19 square.

To solve the yellow shape, push the hollow square into the transform block horizontally three times. Each horizontal transformation reduces the width by 3 and increases the height by 3:

```text
19x19 -> 16x22 -> 13x25 -> 10x28
```

One valid solved arrangement is:

```text
Blue transformed 25x25 cross top-left (6,3), vertical stroke at relative x=6, horizontal stroke at relative y=6.
Yellow transformed 10x28 rectangle top-left (45,30).
```

The blue cross starts with both strokes centered at relative coordinate 12. Use the transform block to shift both strokes 6 pixels toward the upper-left before final placement. The yellow square must be pushed into the transform block horizontally three times to become the 10x28 rectangle.

## Level 7

Step limit: 300.

Starting objects:

```text
White 8x8 transform block: top-left (28,28)
Orange 13x13 hollow square: top-left (15,45), center (21,51)
Pink 37x19 wide cross:      top-left (6,45), center (24,54)
Cyan 19x19 cross:           top-left (3,39), center (12,48), initially selected
```

Recolor pads are five 5x5 squares across the top:

```text
Blue (15,2), Yellow (25,2), Red (35,2), Green (45,2), Magenta (55,2)
```

Targets:

```text
Red:    (9,9), (3,15), (36,15), (9,27)
Blue:   (57,18), (39,24)
Yellow: (45,30), (39,48), (51,48)
```

This level combines recoloring with shape transformation. The green and magenta pads are distractors; the target colors are red, blue, and yellow.

The red target set needs a wide cross whose horizontal stroke spans `x=3` through `x=36` at `y=15`, and whose vertical stroke runs at `x=9` from `y=9` through `y=27`. Recolor the wide cross red, place its top-left at `(0,9)`, and use the transform block to shift its vertical stroke from relative `x=18` to relative `x=9` and its horizontal stroke from relative `y=9` to relative `y=6`.

The blue target pair is opposite corners of a short wide rectangle. Recolor the hollow square blue and transform it vertically twice:

```text
13x13 -> 16x10 -> 19x7
```

Place the resulting 19x7 rectangle top-left `(39,18)`, covering the blue targets at its top-right and bottom-left corners.

The yellow target set needs a 19x19 cross with its horizontal stroke shifted down to the bottom. Recolor the small cross yellow, place it top-left `(36,30)`, and use the transform block to shift its horizontal stroke from relative `y=9` to relative `y=18`. Its vertical stroke covers `(45,30)` and its shifted horizontal stroke covers `(39,48)` and `(51,48)`.

## Level 8

Step limit: 400.

Starting objects:

```text
White 5x5 transform block: top-left (58,1)
White 5x5 transform block: top-left (34,55)
Cyan 13x13 hollow square:   top-left (48,39), center (54,45)
Orange 13x13 hollow square: top-left (45,42), center (51,48), initially selected
```

Recolor pads are dense and deliberately hazardous:

```text
Yellow (5,5)       Red (28,1)       Blue (28,8)
Magenta (5,20)     Orange (28,21)

Row at y=28:
Cyan (-3,28), Red (4,28), Green (11,28), Orange (18,28),
Blue (25,28), Green (32,28), Red (39,28), Cyan (50,28), Green (57,28)
```

Targets:

```text
Magenta: (6,45), (21,45), (6,54), (21,54)
Yellow:  (9,39), (15,39), (9,57), (15,57)
```

The final puzzle uses only two hollow rectangles, but both must be reshaped, recolored, and routed through a field of unwanted pads. The target groups overlap spatially in the lower-left area, so the rectangles can overlap in the final arrangement as long as neither hides the required target centers.

The magenta target group is a 16x10 rectangle: recolor one 13x13 square magenta, transform it vertically once to `16x10`, and place it top-left `(6,45)`.

The yellow target group is a 7x19 rectangle: recolor the other 13x13 square yellow, transform it horizontally twice:

```text
13x13 -> 10x16 -> 7x19
```

Place the resulting 7x19 rectangle top-left `(9,39)`.

The main hazard is the long row of recolor pads at `y=28`. The player must move through the gaps and shape-transform at the correct times so the rectangles do not pick up red, green, blue, orange, or cyan after receiving their final magenta or yellow colors. The upper-right transform block and lower transform block provide enough routes to change aspect ratios, but an incorrect approach direction changes the rectangle the wrong way.

## Reconstruction Checklist

- Use a 64x64 playfield with black background and a one-row step meter at the bottom.
- Direction actions move the selected reticle by 3 pixels; space cycles selection; all five actions spend one step.
- Movement bounds check only the reticle center against the 64x64 camera.
- Target frames are passive and non-blocking; only their colored center pixels are checked.
- The level advances immediately after every required target center is covered by a matching color.
- Recolor pads mutate the entire selected reticle to the pad's interior color and remain in place.
- Shape-transform blocks mutate crosses and hollow rectangles as described above; blocked transform attempts still spend the action step.
- There are no walls, pickups, enemies, or lives in this game. Failure comes from exhausting the step meter, and manual reset restarts the current level.
