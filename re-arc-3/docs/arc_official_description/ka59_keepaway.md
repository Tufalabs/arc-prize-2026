# Keepaway

## Core Game

Keepaway is a discrete push-and-position puzzle. The game runs on an integer
pixel grid, but all deliberate movement uses a 3-pixel quantum. The four
direction actions move the currently selected green agent by 3 pixels. Clicking
selects a different green agent; the selected agent has a black center marker.

The objective is always positional. Dark framed homes must contain matching
objects exactly one pixel inside the frame:

- A 5x5 square home is satisfied by any 3x3 square green agent at `(home.x + 1, home.y + 1)`.
- A 5x8 vertical home is satisfied by any 3x6 vertical green agent at `(home.x + 1, home.y + 1)`.
- An 8x5 horizontal home is satisfied by any 6x3 horizontal green agent at `(home.x + 1, home.y + 1)`.
- An 8x8 square home is satisfied by any 6x6 green agent at `(home.x + 1, home.y + 1)`.
- An 11x11 cross home is satisfied by any 9x9 yellow cross block at `(home.x + 1, home.y + 1)`.

When every home is satisfied at the same time, the level advances.

The terrain has two wall types. Gray hard walls block the active agent and also
stop pushed objects. Purple barriers block walking, but pushed objects pass
through them. This difference is central to the game: the player often has to
throw an agent, yellow block, or pusher through a purple barrier and then work
with the object on the other side.

## Push Rules

If the selected agent attempts to move into another movable object, the selected
agent stays in place and the contacted object is shoved in the attempted
direction. Movable objects include green agents, yellow blocks, and timed
pusher blocks. A shove moves the contacted object 3 pixels at a time for up to
five movement quanta, unless the object is overlapping a purple barrier; in that
case the shove continues until the object clears the barrier. The shove stops
early if it hits a gray hard wall. If the shoved object hits another movable
object, the second object is recursively shoved too, so chains of objects can be
moved together.

Directional inputs and clicks spend one step. Blocked directional inputs also
spend a step. Push and blast animation frames do not spend extra steps. If the
step meter reaches zero, the level is lost/reset to its starting state.

## Timed Pushers

Timed pushers are red/yellow blocks. On each completed movement action, every
timed pusher charges one row from red to yellow. When the final row charges, the
pusher fires, then resets to all red after its blast finishes. Click-only
selection changes spend a step but do not charge pushers.

Pusher direction is determined by rotation:

- Rotation 0 pushes downward.
- Rotation 90 pushes left.
- Rotation 180 pushes upward.
- Rotation 270 pushes right.

A firing pusher creates a 3-frame expanding blast. Any green agent, yellow
block, or timed pusher touched by the blast is moved 3 pixels per frame in the
pusher's direction. As with manual shoves, blast movement can carry objects
through purple barriers and can push chains of objects.

The 3x3 pusher starts with its first two rows already charged, so it fires after
one movement action at the beginning of a level. After it fires, it needs three
movement actions to charge again. A plain 6x6 pusher needs six movement actions
to charge. The special 6x6 pusher used in level 5 starts with its first two
rows charged and therefore needs four movement actions before its first shot.

## Sprite Masks

These masks are enough to rebuild the logical objects. `#` is a frame or solid
body cell, `.` is transparent or open interior, `C` is charged pusher color, and
`R` is uncharged pusher color.

```text
3x3 agent        3x6 vertical agent      6x3 horizontal agent
###              ###                     ######
###              ###                     ######
###              ###                     ######
                 ###
                 ###
                 ###

6x6 large agent
######
######
######
######
######
######

5x5 square home       5x8 vertical home      8x5 horizontal home
#####                 #####                  ########
#...#                 #...#                  #......#
#...#                 #...#                  #......#
#...#                 #...#                  #......#
#####                 #...#                  ########
                      #...#
                      #...#
                      #####

8x8 large home
########
#......#
#......#
#......#
#......#
#......#
#......#
########

9x9 yellow cross block       11x11 yellow cross home
...###...                    ...#####...
...###...                    ...#...#...
...###...                    ...#...#...
#########                    ####...####
#########                    #.........#
#########                    #.........#
...###...                    #.........#
...###...                    ####...####
...###...                    ...#...#...
                             ...#...#...
                             ...#####...

3x3 pusher, initial          6x6 pusher, uncharged       6x6 pusher, level 5 initial
CCC                          RRRRRR                      CCCCCC
CCC                          RRRRRR                      CCCCCC
RRR                          RRRRRR                      RRRRRR
                             RRRRRR                      RRRRRR
                             RRRRRR                      RRRRRR
                             RRRRRR                      RRRRRR
```

## Coordinate Notes

All coordinates below are top-left grid coordinates. Terrain slabs are inclusive
ranges. Coordinates outside the nominal grid are intentional; they create
off-screen walls, launchers, or boundary stops.

## Level 1

Step budget: 100. Nominal grid: 45x45.

Terrain:

```text
Purple barrier:
  y=12..32: x=24..29

Gray hard wall:
  y=-3..11: x=-3..47
  y=12..20: x=-3..-1, x=15..23, x=45..47
  y=21..23: x=-3..-1, x=45..47
  y=24..32: x=-3..-1, x=15..23, x=45..47
  y=33..47: x=-3..47
```

Objects:

```text
3x3 square agents: (9,21), (18,21)
5x5 square homes: (2,23) -> requires agent at (3,24)
                  (35,17) -> requires agent at (36,18)
```

This is the click-selection and purple-barrier tutorial. The gray walls form a
left room and a right room with a 3-pixel-high middle passage. The purple
vertical barrier blocks walking into the right room. To win, place one square
agent inside each square home. The right-side placement requires pushing an
agent through the purple barrier; the player cannot simply walk through it.
Since both agents are the same shape, either agent can occupy either home.

## Level 2

Step budget: 127. Nominal grid: 63x63.

Terrain:

```text
Purple barrier:
  y=0..23: x=24..62
  y=24..29: x=0..62
  y=30..62: x=21..29

Gray hard wall:
  y=-3..-1: x=-3..23
  y=0..23: x=-3..-1, x=21..23
  y=24..26: x=-3..-1
  y=27..29: x=-3..-1, x=30..65
  y=30..62: x=-3..-1, x=63..65
  y=63..65: x=-3..65
```

Objects:

```text
3x3 square agent:      (36,54)
3x6 vertical agent:    (33,42)
6x3 horizontal agent:  (39,33)
6x6 large agent:       (42,45)

5x5 square home:       (50,50) -> requires square agent at (51,51)
5x8 vertical home:     (8,8)   -> requires vertical agent at (9,9)
8x5 horizontal home:   (53,38) -> requires horizontal agent at (54,39)
8x8 large square home: (5,41)  -> requires large agent at (6,42)
```

Level 2 adds shape matching. Four green agents begin in the lower-right area,
but their four homes are split between the lower-right, lower-left, and
upper-left areas. The large purple cross of barriers prevents walking between
these regions. To win, each agent shape must be shoved into the corresponding
frame shape. The vertical and large agents must be sent through purple barriers
to reach the homes outside the starting region, and multiple adjacent agents can
be pushed as a chain when the route requires it.

## Level 3

Step budget: 100. Nominal grid: 54x54.

Terrain:

```text
Gray hard wall:
  y=-3..-1: x=-3..56
  y=0..53: x=-3..-1, x=54..56
  y=54..56: x=-3..56
```

Objects:

```text
3x3 square agent:       (39,9)
5x5 square home:        (29,23) -> requires agent at (30,24)

9x9 yellow cross blocks: (45,15), (15,30)
11x11 yellow homes:      (29,29) -> requires cross block at (30,30)
                         (14,14) -> requires cross block at (15,15)
```

This level is an open room with only a hard outer border. Yellow cross blocks
are introduced as non-clickable movable objects. They are shoved by the active
agent using the same push physics as green agents. To win, the square agent must
be parked in the square home and both yellow cross homes must contain a yellow
cross block. The yellow crosses are interchangeable, but the final top-left
positions must be exactly `(30,30)` and `(15,15)`.

## Level 4

Step budget: 127. Nominal grid: 54x54.

Terrain:

```text
Purple barrier:
  y=33..38: x=0..38

Gray hard wall:
  y=-3..0: x=-3..56
  y=1..14: x=-3..27, x=54..56
  y=15..32: x=-3..-1, x=54..56
  y=33..53: x=-3..-1, x=39..56
  y=54..56: x=-3..56
```

Objects:

```text
3x3 square agents: (15,24), (24,27)
5x5 square homes:  (44,23) -> requires agent at (45,24)
                   (32,47) -> requires agent at (33,48)

9x9 yellow cross block: (15,45)
11x11 yellow home:      (29,2) -> requires cross block at (30,3)
```

Level 4 combines agent placement, yellow-block placement, and a long purple
barrier. The top-right alcove contains the yellow cross home; the yellow cross
starts in the lower-left area below the purple wall. The right side also has a
square agent home in the middle corridor and another in the lower corridor. To
win, move one square agent into each square home and throw the yellow cross
block through the layout until it is centered in the top-right cross frame. The
horizontal purple barrier prevents direct walking access from below, so the
cross must be positioned by pushes that carry it across the barrier.

## Level 5

Step budget: 100. Nominal grid: 63x63.

Terrain:

```text
Purple barrier:
  y=-15..-13: x=-15..2
  y=-12..-1: x=-18..-16, x=3..5
  y=0..8: x=-18..-16
  y=9..11: x=-18..-16, x=18..35
  y=12..14: x=-15..-10, x=18..35
  y=15..20: x=18..35
  y=27..53: x=21..23, x=30..32
  y=54..56: x=21..32

Gray hard wall:
  y=-3..-1: x=-3..2, x=6..65
  y=0..62: x=-3..-1, x=63..65
  y=63..65: x=-3..65
```

Objects:

```text
3x3 square agent: (27,48)
5x5 square home:  (47,26) -> requires agent at (48,27)

6x6 timed pusher, partially charged, rotation 0: (24,30)
3x3 timed pusher, rotation 90:                  (51,27)
3x3 timed pusher, rotation 180:                 (-18,15)
3x3 timed pusher, rotation 270:                 (-21,-15)
3x3 timed pusher, rotation 0:                   (3,-18)
3x3 yellow X block:                             (-9,12)
```

Level 5 introduces timed pushers. The selected square agent starts inside a
purple U-shaped pen. A partially charged 6x6 pusher sits at the top of the pen
and pushes downward when it fires, so the player must time moves around its
charge cycle and can use the blast to be carried through the purple pen. A
small 3x3 pusher sits immediately to the right of the square home and fires
left; it is part of the final parking problem because the agent must end exactly
at `(48,27)` inside the home, not merely near it. The other small pushers and
the small yellow X block are off-screen timing objects and should still follow
the same charge, blast, and push rules.

To win, get the square agent out of the U-shaped pen, reach the right-side home,
and time the final approach so the left-firing pusher does not knock the agent
out of place or can be used to push the agent into the exact home alignment.

## Level 6

Step budget: 150. Nominal grid: 63x63.

Terrain:

```text
Purple barrier:
  y=30..35: x=0..29, x=33..62

Gray hard wall:
  y=-3..0: x=-3..65
  y=1..11: x=-3..0, x=30..32, x=63..65
  y=12..14: x=-3..0, x=21..41, x=63..65
  y=15..20: x=-3..0, x=21..23, x=30..32, x=39..41, x=63..65
  y=21..29: x=-3..0, x=30..32, x=63..65
  y=30..35: x=-3..-1, x=30..32, x=63..65
  y=36..44: x=-3..-1, x=63..65
  y=45..47: x=-3..-1, x=3..5, x=57..59, x=63..65
  y=48..50: x=-3..-1, x=63..65
  y=51..53: x=-3..-1, x=3..59, x=63..65
  y=54..62: x=-3..-1, x=63..65
  y=63..65: x=-3..65
```

Objects:

```text
3x3 square agent: (51,48)
5x5 square home:  (47,20) -> requires agent at (48,21)

9x9 yellow cross block: (6,15)
11x11 yellow home:      (2,2) -> requires cross block at (3,3)

6x6 timed pushers, rotation 0:   (24,15), (33,15)
6x6 timed pusher, rotation 180:  (42,45)
```

Level 6 is the first level where the timed pusher is required as a boundary
crossing tool. The square agent begins in the lower-right area, below a
horizontal purple barrier. Its square home is above that barrier at the
right-center of the board. Walking cannot cross the purple line, so the lower
right 6x6 pusher, which fires upward, must be used to shove the agent through
the purple barrier into the upper side.

The yellow cross block starts in the upper-left region and must be moved into
the cross home near the top-left corner. The two upper 6x6 pushers fire
downward and shape the timing and available space in the upper corridors. To
win, finish both goals simultaneously: the yellow cross block at `(3,3)` and
the square agent at `(48,21)`.

## Level 7

Step budget: 200. Nominal grid: 63x63.

Terrain:

```text
Purple barrier:
  y=6..17: x=21..23, x=39..41
  y=18..20: x=0..18
  y=36..38: x=22..48

Gray hard wall:
  y=3..5: x=21..63
  y=6..17: x=63
  y=18..20: x=19..32, x=36..50, x=63
  y=21..26: x=19..21, x=48..50, x=63
  y=27..29: x=19..21, x=63
  y=30..32: x=48..50, x=63
  y=33..35: x=19..21, x=48..50, x=63
  y=36..38: x=19..21, x=48..63
  y=39..45: x=19..21, x=63
  y=46..47: x=-1..21, x=63
  y=48: x=-1..21, x=48..50, x=63
  y=49..53: x=-1, x=48..50, x=63
  y=54..56: x=-1, x=42..50, x=63
  y=57..62: x=-1, x=63
  y=63: x=-1..63
```

Objects:

```text
3x6 vertical agent:   (54,42)
6x3 horizontal agent: (33,51)

5x8 vertical home:    (26,44) -> requires vertical agent at (27,45)
8x5 horizontal home:  (23,23) -> requires horizontal agent at (24,24)

9x9 yellow cross block: (9,9)
11x11 yellow home:      (8,35) -> requires cross block at (9,36)

6x6 timed pusher, rotation 0:   (45,6)
6x6 timed pusher, rotation 180: (42,48)
```

The final level combines shape-specific agent parking, yellow-cross parking,
purple-barrier throws, and movable timed pushers. The horizontal agent starts in
the lower-right corridor but belongs in the central horizontal home. The
vertical agent starts on the right and belongs in the lower-middle vertical
home. The yellow cross starts in the upper-left region and belongs in the
lower-left cross home.

The gray walls create narrow passages around the middle and right side, while
purple barriers block direct walking through the top-left and middle channels.
The upper pusher fires downward and the lower pusher fires upward. Both pushers
are themselves movable objects, so the solution depends on repositioning and
timing them as part of the puzzle, not only avoiding their blasts. To win, use
the pusher blasts and manual shoves to move objects through the purple barriers
and into their exact final alignments: vertical agent at `(27,45)`, horizontal
agent at `(24,24)`, and yellow cross block at `(9,36)`.
