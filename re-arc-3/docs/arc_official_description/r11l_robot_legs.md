# Robot Legs

## Core Game

Robot Legs is a click-only placement puzzle on a 64x64 playfield. The active objects are small jointed robots: a 5x5 central body and one or more 5x5 endpoint handles connected to the body by rendered line segments. The player does not drag the body directly. Instead, the player clicks an endpoint handle to select it, then clicks a destination cell to move that endpoint so its center lands on the clicked coordinate.

When a robot has two or more endpoints, its body is always recentered at the average of its endpoint centers. Moving one endpoint therefore translates and rotates the apparent robot indirectly. A robot with three or four endpoints is a multi-leg form: every endpoint can be moved independently, and the body stays at the centroid of all endpoints.

Each click is one action, and every level has a 60-action limit shown by a vertical meter on the left side of the playfield. Running out of actions fails the level. Reset restarts the level from its original positions, colors, pickups, and action count.

## Controls And State

Only click actions are used.

1. If the click lands inside any endpoint handle, that endpoint becomes selected. The selected endpoint is drawn differently from the other endpoints.
2. If the click does not select an endpoint and an endpoint is already selected, the selected endpoint is moved so its 5x5 handle is centered on the clicked playfield coordinate.
3. If the selected endpoint's destination would overlap a gray wall shape, the attempted placement is rejected immediately.
4. After a legal endpoint placement, any body attached to that endpoint is recentered from its endpoints.
5. If any robot body overlaps a cyan body-blocking shape at the end of the placement, the body flashes and the moved endpoint rewinds to its previous position. The fifth such body-block collision fails the level.
6. If target conditions are satisfied, the relevant target rings flash. When all required target conditions are satisfied at the same time, the game advances to the next level.

The drawn connecting lines are not colliders. Endpoint handles are blocked by gray wall masks, and robot bodies are checked against cyan body-blocking masks. Target rings and pickups are ordinary sprites for overlap tests, but they do not block movement.

## Target And Color Rules

Levels 1 through 4 use fixed colored robots. Each fixed robot has its own same-color or same-pattern hollow target ring. That robot's body must overlap its matching target. The win check is by robot-target pairing, not by exact pixel-perfect center alignment; any collision between the body and its matching target ring counts.

Levels 5 and 6 use neutral white robots and color pickups. The target rings are not assigned to specific robots. Any neutral robot body can satisfy any target if:

- the body overlaps that target ring, and
- the set of non-white colors currently painted onto the body exactly equals the target ring's color set.

Color arrangement on the body is less important than the set of colors present. Extra colors make a body fail that target. Pickups are consumed on contact with a neutral body and paint their colored pixels onto that body's 5x5 mask. There is no drop action, so collecting an unwanted color usually creates a dead state unless later pickups overwrite every cell of that color.

Some levels show single-color hollow rings and short half-ring markers that look like targets. These are visual landmarks or decoys. They are not checked for level completion.

## Reconstruction Coordinates

Coordinates below are zero-indexed playfield pixel coordinates. Object positions are top-left coordinates after rotation or mirroring. Bodies, endpoints, and pickups are 5x5 unless otherwise stated. Target rings are 7x7 hollow circles. The terrain maps downsample the 64x64 playfield into 4x4-pixel cells:

```text
#  gray wall area: endpoint placements here are rejected
~  cyan body blocker: body overlap flashes and rewinds
X  coarse cell containing both gray wall and cyan blocker
.  open playfield
```

Color names in the level descriptions are the visible colors: purple, orange, green, yellow, red, blue, pink, cyan, and maroon. Neutral robot bodies begin white.

## Level 1

Setup:

```text
Action limit: 60
Robots: 1
Targets: 1 required
Cyan blockers: none
Pickups: none
Initial selected endpoint: upper-left purple endpoint
```

Terrain:

```text
x 0123456789ABCDEF
00 ################
01 ####.##...######
02 #............###
03 ...............#
04 ...............#
05 ................
06 ................
07 ................
08 ................
09 ................
0A ................
0B ................
0C ..............##
0D #..........#####
0E #..........#####
0F ##........######
```

Objects:

```text
Purple two-endpoint robot:
  body:    (15,45)
  handles: (5,34), (25,57)
  target:  (36,18)
```

This is the endpoint-manipulation tutorial. The purple body starts in the lower-left half of the playfield, connected diagonally between two purple endpoint handles. The purple target ring is in the upper-middle area. The gray walls frame the top and lower-right edges but leave a large open black area.

To win, move one or both endpoints until the purple body overlaps the purple hollow target ring. The endpoints themselves do not need to touch the ring; only the central body matters. Since the body is positioned at the average of the two endpoint centers, placing one endpoint far across the board moves the body only halfway toward that endpoint.

Important reconstruction details:

- There are no cyan blockers in this level, so failed placements mostly come from clicking an endpoint into the gray wall.
- The body should recenter continuously after every accepted endpoint placement.
- The target flashes briefly when the body overlaps it, then the level advances.

## Level 2

Setup:

```text
Action limit: 60
Robots: 2
Targets: 2 required
Pickups: none
Initial selected endpoint: upper orange endpoint
```

Terrain:

```text
x 0123456789ABCDEF
00 ######..###.####
01 ##............##
02 #...............
03 #...............
04 ................
05 ......~~~~......
06 ....~~~~~~......
07 ...~~~~~.....~~~
08 ~~.~~~.......~~~
09 ~~..~~~......~~~
0A ~~~~~~~~......~~
0B X~~~~~~~......~~
0C XXXXX..........X
0D #XXXX.........##
0E #XXX##.......###
0F ######.....#####
```

Objects:

```text
Orange three-endpoint robot:
  body:    (22,10)
  handles: (15,4), (6,19), (47,7)
  target:  (37,48)

Purple two-endpoint robot:
  body:    (47,39)
  handles: (43,33), (52,46)
  target:  (54,15)
```

This level adds multiple robots, multiple targets, and cyan body blockers. The orange robot begins high on the board with three endpoints spread across the top-left and top-right. Its orange target is near the lower-middle. The purple robot begins in the lower-right, and its purple target is near the upper-right.

The cyan blocker forms a wide irregular island through the lower and middle playfield. Endpoint handles can be placed around it, but if either robot's body ends a placement overlapping cyan, that placement flashes and rewinds. The gray walls still reject endpoint placements directly.

To win, place the orange body on the orange target and the purple body on the purple target at the same time. The robots may be solved in either order. A body that is already on its target can be moved away again, so the final state must satisfy both overlaps simultaneously.

Important reconstruction details:

- The orange body is the centroid of three endpoints; moving one endpoint shifts the body by roughly one third of that endpoint's displacement.
- The purple body is the midpoint of its two endpoints.
- A cyan collision rewinds only the currently moved endpoint and the body attached to it.
- Target flashes do not make a target permanently solved; overlap must still be true for the final all-target check.

## Level 3

Setup:

```text
Action limit: 60
Robots: 2
Targets: 2 required
Pickups: none
Initial selected endpoint: left green endpoint
```

Terrain:

```text
x 0123456789ABCDEF
00 #XXX############
01 XXXX##..###.####
02 XX~~..........##
03 X~~.............
04 X~~............~
05 ~~..........~~~~
06 ~~...~~~~~~~~~~~
07 ~~...~~~..~~~~~~
08 ~~..........~~~~
09 ~~~......~....~~
0A ~~~~~~~~~~~~...~
0B ~~~~~~~~~~~~~..~
0C XX~~~~...~~~....
0D #####...........
0E #####...........
0F ######..........
```

Objects:

```text
Green four-endpoint robot:
  body:    (25,13)
  handles: (12,14), (21,19), (32,7), (37,14)
  target:  (52,50)

Purple two-endpoint robot:
  body:    (42,35)
  handles: (35,32), (50,38)
  target:  (31,54)
```

This level keeps the same rules as level 2 but makes the geometry more restrictive. The green robot starts near the upper-left with four endpoints and must be brought to the green target in the lower-right corner. The purple robot starts near the center-right and must be placed on the purple target near the bottom-center.

The cyan blocker is a large river-like shape running from the left and middle of the board into the lower-right. The green robot's many endpoints make it easy to create an invalid centroid that lies inside the cyan region even when all endpoint handles look clear.

To win, route both robot bodies around the cyan blocker and overlap both matching targets simultaneously.

Important reconstruction details:

- The four-endpoint green body should be recentered from all four endpoint centers after every green endpoint move.
- The cyan mask is the main challenge. The player must keep the body out of it, not merely keep the endpoint handles out of it.
- The lower gray wall area narrows the usable bottom route and makes the purple target harder to approach from below.

## Level 4

Setup:

```text
Action limit: 60
Robots: 3
Targets: 3 required
Pickups: none
Initial selected endpoint: lower-left endpoint of the purple/orange/green robot
```

Terrain:

```text
x 0123456789ABCDEF
00 ##...........###
01 ##............##
02 #...............
03 ................
04 ...............#
05 ........~~~....#
06 ##.....~~~~....#
07 ##....~~~~~....#
08 ##....~~~~.....#
09 ##....~~~......#
0A #...............
0B ................
0C ................
0D ................
0E #..............#
0F ###.....########
```

Objects:

```text
Purple/orange/green two-endpoint robot:
  body:    (29,11)
  handles: (21,18), (37,4)
  target:  (33,46)

Yellow/green three-endpoint robot:
  body:    (16,43)
  handles: (15,34), (25,50), (8,45)
  target:  (47,9)

Red/blue two-endpoint robot:
  body:    (44,42)
  handles: (44,50), (44,34)
  target:  (14,51)

Visual decoy rings:
  single-color rings: (9,14), (52,51), (17,5), (11,26), (50,25)
  short half-ring markers: (13,14), (56,51), (17,9), (11,30)
```

Level 4 introduces multi-color bodies. The body color pattern identifies which hollow multi-color target belongs to which robot:

- the upper purple/orange/green robot belongs on the lower-middle purple/orange/green target,
- the lower-left yellow/green robot belongs on the upper-right yellow/green target,
- the lower-right red/blue robot belongs on the lower-left red/blue target.

The cyan blocker is a compact island in the center. It is smaller than the blockers in levels 2 and 3, but all three robots have to pass around or pivot around it. The single-color rings and half-ring markers are visual clutter and should not be part of the win condition.

To win, place all three colored bodies on their matching target rings simultaneously.

Important reconstruction details:

- Multi-color bodies in this level are pre-painted. There are no pickups and no color mutation.
- Matching is by the robot's assigned target, not by a free color-set search.
- Decoy rings are rendered like targets but have no assigned robot body, so they do not participate in completion.
- The small cyan island should cause the same flash-and-rewind behavior as the larger cyan blockers in previous levels.

## Level 5

Setup:

```text
Action limit: 60
Neutral robots: 2
Required targets: 2
Pickups: 4
Cyan blockers: none
Initial selected endpoint: left endpoint of the upper neutral robot
```

Terrain:

```text
x 0123456789ABCDEF
00 ###..######...##
01 #......####...##
02 #......###......
03 ................
04 ................
05 ................
06 ................
07 ................
08 #...............
09 #...............
0A #...............
0B #...............
0C #...............
0D #...............
0E #..............#
0F ##.....#####..##
```

Neutral robots:

```text
Upper two-endpoint robot:
  body:    (32,32)
  handles: (23,33), (41,32)

Lower three-endpoint robot:
  body:    (40,50)
  handles: (32,53), (39,45), (50,53)
```

Required targets:

```text
Red/blue target:     (10,25)
Yellow/green target: (45,6)
```

Pickups:

```text
Blue pickup:   (12,38)
Green pickup:  (18,51)
Red pickup:    (31,17)
Yellow pickup: (54,41)
```

Visual decoys:

```text
Single-color rings: (54,15), (12,12), (50,25)
Short half-ring markers: (54,19), (12,16)
```

Level 5 changes the objective from moving pre-colored robots to building the right colors first. Both robot bodies begin white. A white body that touches a pickup absorbs that pickup's colored wedge pattern and removes the pickup from the board.

To win, one body must contain exactly red and blue and overlap the red/blue target, while the other body must contain exactly yellow and green and overlap the yellow/green target. Either neutral robot can be used for either target; the win check does not bind a specific body to a specific target. The single-color rings are decoys.

Important reconstruction details:

- Target matching uses the set of body colors, not the exact wedge locations.
- White body pixels do not count as target colors.
- A body with red, blue, and any extra color does not satisfy the red/blue target.
- Pickups are one-use and are restored only by resetting the level.
- There are no cyan body blockers, so the challenge is color assignment and endpoint positioning within the gray boundary.

## Level 6

Setup:

```text
Action limit: 60
Neutral robots: 2
Required targets: 2
Pickups: 9
Cyan blockers: none
Initial selected endpoint: upper endpoint of the left neutral robot
```

Terrain:

```text
x 0123456789ABCDEF
00 ########.....###
01 ##.#####.......#
02 ...............#
03 ................
04 ................
05 ................
06 ...............#
07 #.............##
08 #.............##
09 #.............##
0A #.............##
0B #...............
0C ................
0D ................
0E #..............#
0F ##............##
```

Neutral robots:

```text
Right two-endpoint robot:
  body:    (47,48)
  handles: (47,41), (48,55)

Left three-endpoint robot:
  body:    (10,14)
  handles: (2,17), (9,9), (20,17)
```

Required targets:

```text
Blue/green/purple target: (48,9)
Yellow/pink/cyan target:  (8,52)
```

Pickups:

```text
Red pickup:     (37,4)
Green pickup:   (32,15)
Blue pickup:    (9,31)
Pink pickup:    (43,28)
Cyan pickup:    (23,53)
Maroon pickup:  (54,21)
Orange pickup:  (36,55)
Purple pickup:  (14,43)
Yellow pickup:  (30,42)
```

Visual decoy:

```text
Single-color ring: (26,28)
Short half-ring marker: (26,32)
```

This is the final color-collection level. The board has two white robots and many more pickups than are needed. The upper-right target requires exactly the blue, green, and purple color set. The lower-left target requires exactly the yellow, pink, and cyan color set. The red, maroon, and orange pickups are wrong colors for both required targets and should be avoided unless a reconstruction intentionally allows later repainting to remove them completely.

To win, create two robot bodies whose color sets exactly match the two targets, then place those bodies on the matching target rings simultaneously. As in level 5, either body can satisfy either target if it has the required color set.

Important reconstruction details:

- The larger pickup set is the main difficulty. Extra colors prevent target satisfaction.
- Pickups paint fixed 5x5 wedge patterns onto the body and are removed immediately on body contact.
- The decoy ring near the center is ignored by the win check.
- There are no cyan body blockers; endpoint placement is constrained only by the gray outer walls and the action budget.

## Implementation Checklist

- Use a 64x64 playfield and click-only input.
- Generate click actions on playfield/display coordinates and convert clicks back to playfield coordinates before testing handles.
- Keep per-level action budget at 60.
- Track the selected endpoint globally; clicking another endpoint changes selection without moving it.
- Center the selected endpoint's 5x5 handle on the clicked destination.
- Reject endpoint destinations that collide with gray wall masks.
- Recenter each body at the integer average of its endpoint centers whenever any of its endpoints moves.
- Draw lines from each body center to every endpoint center for that robot.
- Check body collision against cyan masks only after the endpoint placement and body recentering.
- On cyan body collision, flash the body, rewind the moved endpoint to its previous position, and fail the level after five such collisions.
- For fixed-color levels, require each body to overlap its assigned target ring.
- For collection levels, require each target to be overlapped by some neutral body whose non-white color set exactly equals the target's color set.
- Consume pickups on neutral-body contact by painting the pickup's colored pixels into the body's 5x5 mask.
- Ignore visual decoy rings and half-ring markers for completion.
- Advance to the next level only after all required targets are satisfied simultaneously.
