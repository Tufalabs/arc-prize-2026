# Spill

## Core Game

Spill is a setup-and-execute flow puzzle. The player first arranges movable blue bars and, in later levels, blue corner deflectors. Pressing the spill action then runs a deterministic water simulation. The level is won only if every yellow basin is filled and no stream touches a failure wall.

Use a small logical pixel grid, not large tile cells. Levels 1-3 use a 16x16 grid; levels 4-6 use a 20x20 grid. Coordinates below use the unrotated logical grid with `(0,0)` at the top-left. Some levels render the whole board rotated 180 degrees; the stored geometry and rules are still the unrotated coordinates in the tables.

The main object colors are:

- Orange background.
- Blue outer frame and failure walls.
- Purple water.
- Yellow source markers and basins.
- Light-blue movable bars.
- Pink movable corner deflectors.
- Green filled basins.
- Yellow flashed walls on a failed spill.

## Setup Mode

The player has four direction actions, a spill action, and a click/select action. One movable object is selected at a time and is highlighted. At level start, the selected object is the movable bar or deflector closest to the top-left corner. Clicking a movable bar or deflector selects it. Direction actions move the selected object one logical pixel.

Every non-reset setup action costs one step: moving, clicking, and starting a spill all decrement the step meter by 1. The meter is shown as a horizontal bar at the top of the screen. If the meter reaches zero during setup, the level is lost. The step budgets are level-specific.

Movable pieces cannot be moved above row `y = 3`. They are also blocked by the outer frame, fixed failure walls, and a one-pixel safety halo around every basin. Other movable bars and deflectors do not permanently block movement in the same way; the controlled piece can be moved through them by the game's movement logic.

If a spill fails, the generated water is removed, walls and basins return to their unfilled colors, and the player returns to setup mode with the movable pieces left where they were. The step meter is not restored. After four failed spills, another spill action loses the level.

## Flow Rules

All streams are simulated as active water cells with a direction vector. Every source starts a downward stream `(0, +1)`.

At spill start:

1. The existing water cells below the top source markers become active.
2. Every yellow source marker embedded in a movable bar also creates a new downward water cell one cell below the marker, but only if that cell is empty and does not already contain water.

Each animation tick processes every active water cell:

1. If the next cell in the stream direction is empty, create water there and keep moving in the same direction.
2. If the next cell already contains water, continue through that water in the same direction.
3. If the next cell is a movable bar, try to create water in the two cells perpendicular to the stream direction, beside the current cell. New branches keep the original direction. This makes a horizontal bar split a falling stream outward until the branches pass the bar's ends, then fall again.
4. If the next cell is a basin, check the two perpendicular cells beside the current water cell. If both belong to that same basin, the basin is filled and that branch stops. Otherwise the basin acts like a splitter, creating perpendicular side branches where possible.
5. If the next cell is a corner deflector, check the same two perpendicular side cells. When exactly one side cell belongs to that same deflector and the other side is empty, the stream turns 90 degrees away from the deflector's occupied side. If the corner condition is not met, the deflector behaves like a splitter.
6. If the next cell is a failure wall, mark the wall wet and stop that branch.
7. If the next cell is only the decorative outer frame, the branch stops without filling a basin.

When no active streams remain, the spill succeeds if every basin is filled and no failure wall is wet. On success, advance to the next level.

## Shape Masks

A downward basin at `(x,y)` is a 3x2 U shape open at top center:

```text
X.X
XXX
```

It fills when a downward stream reaches `(x+1,y)` and then tries to enter `(x+1,y+1)`.

A left-opening basin, used on the right side of levels 5 and 6, has this 2x3 mask:

```text
XX
.X
XX
```

It fills when a right-moving stream reaches its open middle-left cell and then tries to enter the closed middle-right cell.

A right-opening basin, used on the left side of level 6, has this 2x3 mask:

```text
XX
X.
XX
```

It fills when a left-moving stream reaches its open middle-right cell and then tries to enter the closed middle-left cell.

Two corner deflectors are used:

```text
Right-turn corner       Left-turn corner
X.                      .X
XX                      XX
```

With falling water, the right-turn corner can send a stream to the right, and the left-turn corner can send a stream to the left when the water approaches the concave side correctly.

## Level 1

```text
Grid: 16x16
Step budget: 30
Display rotation: none
Sources: top marker at (9,0), active water at (9,1)
Failure wall: bottom row y=15
Basins: downward basins at (4,13) and (10,13)
Movable objects: length-5 horizontal bar at (3,4)
```

This is the basic splitter tutorial. The single top stream starts at column 9. The two basins are near the bottom, with open centers at columns 5 and 11. If the stream is allowed to fall straight, it misses both basins and wets the bottom wall.

To win, move the length-5 bar so it catches the falling stream and lets water spill around both ends. One valid final arrangement is:

```text
length-5 bar: (6,3)
```

With the bar spanning `x=6..10` at `y=3`, the stream spreads left and right along the bar's top edge, then drops into columns 5 and 11. Those two falling streams fill both downward basins.

## Level 2

```text
Grid: 16x16
Step budget: 45
Display rotation: 180 degrees
Sources: top marker at (5,0), active water at (5,1)
Failure wall: bottom row y=15
Basins: downward basins at (2,13), (6,13), and (10,13)
Movable objects:
  length-3 bar at (6,9)
  length-3 bar at (11,11)
  length-5 bar at (6,6)
```

The rendered board is upside down, but the underlying flow still runs downward in the unrotated coordinate system. The three basin centers are columns 3, 7, and 11. The player must make one source feed all three basins.

To win, arrange the three bars as a cascade of splitters. One valid final arrangement is:

```text
length-3 bar from (6,9):  (0,3)
length-3 bar from (11,11): (8,5)
length-5 bar from (6,6):  (4,3)
```

The length-5 bar near the top splits the source. The small bars catch and re-split the side branches so final falling streams reach columns 3, 7, and 11. Any branch that reaches the bottom wall makes the spill fail, so all three basin fills must happen before the stray streams reach `y=15`.

## Level 3

```text
Grid: 16x16
Step budget: 100
Display rotation: 180 degrees
Sources:
  top marker at (1,0), active water at (1,1)
  top marker at (6,0), active water at (6,1)
  top marker at (14,0), active water at (14,1)
Failure wall: bottom row y=15
Basins: downward basins at (1,13), (7,13), and (12,13)
Movable objects:
  length-4 bar at (10,10)
  length-5 bar at (1,8)
  length-6 bar at (8,7)
  length-6 bar at (1,5)
```

This level introduces multi-object setup. There are three independent top streams and four movable bars, so clicking to switch the selected object is part of the puzzle. The basin centers are columns 2, 8, and 13. In the starting layout, some streams can already reach basins, but at least one branch wets the bottom wall, so the starting layout is not a solution.

One valid final arrangement is:

```text
length-4 bar from (10,10): (9,11)
length-5 bar from (1,8):   (11,7)
length-6 bar from (8,7):   (3,9)
length-6 bar from (1,5):   (0,3)
```

The left source is controlled by the upper-left long bar and ends in the left basin. The middle source is split and guided to the center basin. The right source is caught by the right-side bars and guided to the right basin. The important reconstruction detail is that the player can reposition all four bars before pressing spill; the spill simulation itself is then fully automatic.

## Level 4

```text
Grid: 20x20
Step budget: 120
Display rotation: none
Sources:
  top marker at (7,0), active water at (7,1)
  embedded source marker in the length-7 bar
Failure wall: bottom row y=19
Basins: downward basins at (2,17), (8,17), (12,17), and (16,17)
Movable objects:
  length-4 bar at (12,13)
  length-4 bar at (14,10)
  length-5 bar at (12,5)
  length-5 bar at (5,5)
  length-7 source bar at (2,9), with its marker at local x=3
```

This level adds a movable source. The length-7 bar contains a yellow source pixel. Wherever the bar is placed, the embedded marker creates a new downward stream one cell below its absolute marker position at spill start.

The four basin centers are columns 3, 9, 13, and 17. The top stream alone cannot fill all of them; the movable source bar must be used as both an obstacle and an extra source.

One valid final arrangement is:

```text
length-4 bar from (12,13): (4,5)
length-4 bar from (14,10): (11,12)
length-5 bar from (12,5):  (5,4)
length-5 bar from (5,5):   (10,4)
length-7 source bar:       (10,15)
```

In that arrangement, the source bar spans `x=10..16` at `y=15`; its embedded marker is at `(13,15)` and creates water at `(13,16)`, directly above the basin centered at column 13. The other bars near the top split the fixed source into the left, center-left, and right basin columns. All four basins must be green at the end of the spill.

## Level 5

```text
Grid: 20x20
Step budget: 100
Display rotation: 180 degrees
Sources:
  top marker at (5,0), active water at (5,1)
  top marker at (13,0), active water at (13,1)
Failure walls:
  bottom row y=19
  right column x=19 for visible rows
Basins:
  left-opening basin at (17,6)
  downward basins at (2,17), (6,17), and (12,17)
Movable objects:
  length-3 bar at (11,9)
  length-4 bar at (7,13)
  length-5 bar at (2,9)
  right-turn corner at (8,5)
```

This level combines the normal falling-stream puzzle with a sideways basin. The display is rotated 180 degrees, so the right-side raw basin appears on the left side of the screen. That basin cannot be filled by a falling stream; it needs a right-moving stream entering its open middle-left cell.

The corner deflector is therefore essential. A falling stream must hit the concave side of the right-turn corner so the corner emits a horizontal stream to the right. The remaining three basins are filled by falling streams produced by the movable bars.

One valid final arrangement is:

```text
length-3 bar:      (4,10)
length-4 bar:      (5,7)
length-5 bar:      (8,11)
right-turn corner: (8,7)
```

With the corner at `(8,7)`, a falling branch reaches the corner and turns into a right-moving stream along row 7. That stream enters the side basin at `(17,7)` and fills it. The bars route the other branches into basin centers at columns 3, 7, and 13. Any stream that continues into `x=19` or `y=19` wets a wall and causes a reset to setup mode.

## Level 6

```text
Grid: 20x20
Step budget: 120
Display rotation: none
Sources:
  top marker at (9,0), active water at (9,1)
  embedded source marker in the length-5 bar
Failure walls:
  left column x=0 for visible rows
  right column x=19 for visible rows
  bottom row y=19
Basins:
  right-opening basin at (1,6)
  right-opening basin at (1,11)
  left-opening basin at (17,9)
  downward basin at (8,17)
Movable objects:
  length-4 vertical bar at (14,4)
  length-5 source bar at (7,10), with its marker at local x=2
  left-turn corner at (9,14)
  right-turn corner at (9,5)
```

The final level combines every mechanic. It has one fixed top source, one movable embedded source, a vertical splitter bar, two corner deflectors, two left-side sideways basins, one right-side sideways basin, and one normal bottom basin. The side walls make stray horizontal streams dangerous.

The key idea is to use the source bar as the first splitter and as the bottom-basin source at the same time. One valid final arrangement is:

```text
vertical length-4 bar: (4,8)
length-5 source bar:  (7,4)
left-turn corner:     (6,10)
right-turn corner:    (11,10)
```

In this arrangement, the length-5 source bar spans `x=7..11` at `y=4`, and its embedded marker is at `(9,4)`. The embedded marker creates a downward stream at `(9,5)`, which falls into the bottom basin centered at column 9.

The fixed top stream also hits the same source bar. The bar splits it into falling branches around columns 6 and 12. The left branch is turned left by the left-turn corner at `(6,10)`. That horizontal stream then hits the vertical bar at `(4,8)`. The vertical bar splits the left-moving stream into two left-moving rows, one at row 7 and one at row 12, which fill the two right-opening basins on the left side.

The right branch is turned right by the right-turn corner at `(11,10)`. It travels along row 10 and fills the left-opening basin on the right side. The level succeeds only when all four basins are filled and none of the three failure walls has been touched.

## Reconstruction Checklist

- Keep setup mode and spill mode separate.
- Direction, click, and spill actions consume setup steps; spill animation ticks do not.
- Failed spills reset generated water and basin/wall colors but leave movable pieces in place.
- Four failed spills are allowed; the next spill action loses.
- All sources start downward, including source pixels embedded in movable bars.
- Bars split streams perpendicular to the current direction; new branches keep the old direction.
- Basins fill only when water enters through their open middle cell and then hits the closed back wall with both side arms present.
- Corner deflectors turn streams only when the stream approaches the concave corner with one perpendicular neighbor occupied by that same corner.
- A level is won only after every basin has been filled and no failure wall is wet.
- Levels 2, 3, and 5 render rotated 180 degrees and remap controls/clicks to match the rotated display.
