# True Naming

## Core Game

True Naming is a click-only programming puzzle. The player does not move the
actor directly. Instead, each level has a right-side execution board with a
yellow 4x4 notched actor, a target outline, a black editable program grid, and a
large blue run button. Clicking bars in the program grid toggles instruction
bits. Clicking the blue button runs the current program from left to right.

The left side is a reference machine. In level 1 it is mostly off screen. From
level 2 onward it has selector buttons along the bottom. Clicking a selector
loads and immediately runs a fixed sample program on the left board, showing
which pattern corresponds to a movement or transformation. The selector does
not edit the right program; the player must copy the useful code into the right
program manually.

The actor is a 4x4 block with this mask, where `X` is colored and `.` is empty:

```text
XXXX
XXXX
XXXX
X..X
```

The target is a 6x6 outline of the same notched shape. For collision and win
checking, the target's effective state is the outline top-left plus the target
scale in both x and y. A level is solved only after a program finishes and the
right actor exactly matches the target x/y position, rotation, scale, and color.
Passing through the target before the last instruction is not enough.

## Execution Rules

Use a 64x64 render grid. The important movement quantum is 4 pixels. Most board
positions in levels 2-7 lie on x/y values spaced by 4 pixels. The actor stores:

- x/y top-left position
- rotation in degrees
- scale, clamped to at least 1
- color
- alive/visible state
- saved reset state, initially the level start and later updated by checkpoints

On a run button click, the current right program bitmasks are copied into an
execution list. Each game step executes one program slot:

1. Hide the previous instruction-slot highlight and show the current one.
2. Decode the slot bitmask as a command.
3. Apply movement, rotation, scale, or color.
4. If a movement or scale change overlaps a solid purple wall, undo that change
   and continue to the next instruction.
5. If the actor overlaps an active hazard, replace it with the multicolor
   failure sprite and ignore further movement until the run ends.
6. After the full program finishes, check checkpoints and the target.
7. If the target does not match, reset the actor to the saved reset state. The
   edited program stays as the player left it.

The top blue timer bar advances only on click actions while no program is
executing. In levels 1-5 it shifts left by 1 pixel per click. In levels 6-7 it
shifts left by 1 pixel every second click. If its right edge no longer overlaps
the track, the level is lost.

## Instruction Encoding

Each editable slot is a vertical stack of clickable bars. A lit/active bar
contributes one bit to the slot's integer code. In the full six-bit slots used
from level 2 onward, bits are ordered top to bottom:

```text
bit 0: top horizontal bar
bit 1: upper vertical bar
bit 2: middle horizontal bar
bit 3: lower vertical bar
bit 4: bottom horizontal bar
bit 5: bottom vertical bar
```

Level 1 uses compact two-bit slots, with bit 0 as the horizontal bar and bit 1
as the vertical bar.

Known command codes:

```text
0   no-op
1   move left 4 pixels
2   move right 4 pixels
3   move down 4 pixels
5   rotate clockwise 90 degrees
6   rotate counterclockwise 90 degrees
7   rotate 180 degrees
8   increase scale by 1
9   decrease scale by 1
10  move right 8 pixels
11  move right 8 pixels
12  move left 8 pixels
13  move left 8 pixels
14  set color to palette color 9
15  set color to palette color 8
16  rotate 270 degrees
33  move up 4 pixels
34  move left 4 pixels
63  set color to palette color 15
```

Codes not listed have no mapped action, but they still consume an instruction
slot and can affect timed hazards.

## Level 1

Level data:

```text
Right board: x=14, y=9, size 36x32
Actor start: x=30, y=13, rotation=0, scale=1, color=11
Target state: x=30, y=33, rotation=0, scale=1, color=11
Right program slots: 5 compact two-bit slots
Initial right program: [3, 0, 3, 0, 0]
Solid walls: none
Checkpoints: none
Hazards: none
```

This is the basic programming tutorial. The right board contains the actor near
the top and the target directly below it. Each compact slot can be empty or can
be changed into command `3`, the down command, by lighting both of its bars.
The initial program already has two down commands, so running it moves the actor
two cells down and then resets because it does not finish on the target.

To win, all five slots must be command `3`. The finished program is:

```text
[3, 3, 3, 3, 3]
```

That moves the actor from y=13 to y=33 in five 4-pixel steps. The x position,
rotation, scale, and color already match the target. If any slot remains a
no-op, the final y position is too high and the actor resets to the start after
the run.

## Level 2

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=45, y=24, rotation=180, scale=1, color=11
Target state: x=45, y=8, rotation=180, scale=1, color=11
Right program slots: 4 six-bit slots
Initial right program: [0, 0, 0, 0]
Solid walls: none
Checkpoints: none
Hazards: none
```

The left reference panel introduces selector buttons. There are two samples:

```text
Selector 1: [1, 1, 1, 1], starts offset to the right, demonstrates left motion
Selector 2: [33, 33, 33, 33], starts offset below, demonstrates upward motion
```

The right actor starts directly below the target, so the useful reference is the
vertical/up sample. The right program must set every slot to command `33`:

```text
[33, 33, 33, 33]
```

This moves the actor up by 16 pixels, from y=24 to y=8. Rotation, scale, and
color already match. A sideways command or no-op leaves the final position
wrong, and the actor returns to its starting state at the end of the run.

## Level 3

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=37, y=20, rotation=180, scale=1, color=11
Target state: x=53, y=12, rotation=180, scale=1, color=11
Right program slots: 6 six-bit slots
Initial right program: [0, 0, 0, 0, 0, 0]
Solid walls:
  vertical wall at x=45, y=4, size 4x12
  vertical wall at x=45, y=20, size 4x12
Checkpoints: none
Hazards: none
```

The left side becomes a four-direction code reference:

```text
Selector 1: [3, 3, 3, 3], down
Selector 2: [33, 33, 33, 33], up
Selector 3: [34, 34, 34, 34], left
Selector 4: [2, 2, 2, 2], right
```

The right board has two vertical purple wall segments at the same x position,
with a 4-pixel gap between them. The actor starts left of the lower segment.
The target is right of the upper segment. The intended route is to move up into
the gap, move right through the gap, then move up into the target row.

One winning program is:

```text
[33, 2, 2, 2, 2, 33]
```

Step by step, this places the actor at:

```text
(37,20) start
(37,16) up into the gap row
(41,16)
(45,16)
(49,16)
(53,16) right through the gap
(53,12) up to the target
```

Moving right first collides with the lower wall and is undone, so the actor must
enter the gap before crossing.

## Level 4

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=45, y=8, rotation=0, scale=2, color=11
Target state: x=41, y=24, rotation=0, scale=1, color=11
Right program slots: 6 six-bit slots
Initial right program: [0, 0, 0, 0, 0, 0]
Solid walls:
  horizontal wall at x=33, y=20, size 8x4
  horizontal wall at x=45, y=20, size 16x4
Checkpoints: none
Hazards: none
```

The reference selectors introduce resizing:

```text
Selector 1: [34, 34, 34], left
Selector 2: [8, 8, 8], grow
Selector 3: [3, 3, 3], down
Selector 4: [9, 9, 9], shrink
```

The two horizontal wall segments leave a one-cell vertical gap between x=41 and
x=45. The actor begins at scale 2 and cannot fit through that gap. The target
requires scale 1, so shrinking is both necessary for the final state and for the
route.

One winning program is:

```text
[9, 34, 3, 3, 3, 3]
```

This first shrinks the actor from scale 2 to scale 1, moves it left from x=45
to x=41, then moves it down four times from y=8 to y=24. It finishes in the gap
below the wall with the correct size. If the actor is not shrunk before trying
to pass the wall gap, the scale change or movement collides and is undone.

## Level 5

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=49, y=8, rotation=270, scale=1, color=11
Target state: x=49, y=20, rotation=0, scale=2, color=15
Right program slots: 6 six-bit slots
Initial right program: [0, 0, 0, 0, 0, 0]
Solid walls:
  horizontal wall at x=33, y=16, size 16x4
  horizontal wall at x=53, y=16, size 8x4
Checkpoints: none
Hazards: none
```

This level adds rotation and color transformation. The reference selectors are:

```text
Selector 1: [9, 9, 9], shrink
Selector 2: [8, 8, 8], grow
Selector 3: [3, 3, 3], down
Selector 4: [5, 5, 5], rotate clockwise 90 degrees
Selector 5: [0, 63, 0], set color to palette color 15
```

The wall is a horizontal barrier with a central gap at x=49. The actor starts
above that gap. The target is below it and also requires the actor to be larger,
rotated from 270 degrees to 0 degrees, and recolored to palette color 15.

One winning program is:

```text
[3, 3, 3, 8, 5, 63]
```

The three down commands move from y=8 to y=20 through the gap. The grow command
changes scale 1 to scale 2. The rotate command changes 270 degrees to 0
degrees, and command `63` changes the color to match the purple target. The
order of the final grow, rotate, and color commands can vary as long as the
actor does not collide and the final state matches exactly.

## Level 6

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=33, y=28, rotation=90, scale=1, color=11
Target state: x=37, y=8, rotation=90, scale=1, color=11
Right program slots: 6 six-bit slots
Initial right program: [0, 0, 0, 0, 0, 0]
Checkpoints:
  x=53, y=28
  x=49, y=20
  x=53, y=4
Solid walls:
  vertical wall at x=37, y=0, size 4x8
  vertical wall at x=45, y=8, size 4x12
  vertical wall at x=37, y=12, size 4x12
  horizontal wall at x=33, y=24, size 16x4
  vertical wall at x=53, y=8, size 4x20
Hazards: none
```

The left reference selectors return to the four basic directions, now in
three-step demonstrations:

```text
Selector 1: [3, 3, 3], down
Selector 2: [33, 33, 33], up
Selector 3: [2, 2, 2], right
Selector 4: [1, 1, 1], left
```

This level introduces checkered checkpoints. A checkpoint is not saved merely
by crossing it. The program must finish with the actor on the checkpoint, and
the actor's scale must match the checkpoint. At the end of that run, the
checkpoint becomes the new reset state, so the player can edit the program and
run another segment from that position.

A reconstruction can use this compact route:

```text
Run 1: [2, 2, 2, 2, 2, 0]       start -> checkpoint at (53,28)
Run 2: [1, 33, 33, 0, 0, 0]     (53,28) -> checkpoint at (49,20)
Run 3: [33, 33, 33, 33, 2, 0]   (49,20) -> checkpoint at (53,4)
Run 4: [1, 1, 1, 1, 3, 0]       (53,4) -> target at (37,8)
```

The wall layout forces this staged solution. The lower horizontal wall blocks a
direct route upward from the start. The tall right wall separates the lower
right checkpoint from the upper right checkpoint. The vertical walls near the
target require approaching from the top-right checkpoint, moving left above the
walls, then moving down into the target.

Failed runs reset to the last saved checkpoint rather than the original start.
The program grid remains editable after every reset, so the player can rewrite
the six slots between segments.

## Level 7

Level data:

```text
Right board: x=33, y=4, size 28x28
Actor start: x=41, y=28, rotation=180, scale=1, color=11
Target state: x=41, y=8, rotation=180, scale=1, color=11
Right program slots: 6 six-bit slots
Initial right program: [0, 0, 0, 0, 0, 0]
Visible checkpoints:
  x=53, y=24
  x=53, y=8
  x=33, y=12
Hidden start checkpoint:
  x=41, y=28
Solid walls:
  horizontal wall at x=37, y=4, size 12x4
  block at x=37, y=8, size 4x4
  block at x=45, y=8, size 4x4
  block at x=33, y=16, size 4x4
  vertical wall at x=49, y=24, size 4x8
Hazards:
  laser endpoints at x=37, y=16 and x=57, y=16
  full laser beam covers y=16 from x=37 through x=57
```

The reference selectors are the same as level 6:

```text
Selector 1: [3, 3, 3], down
Selector 2: [33, 33, 33], up
Selector 3: [2, 2, 2], right
Selector 4: [1, 1, 1], left
```

The new mechanic is the laser. The visible checkered endpoints are dangerous
when the beam is off. During a run, after the third instruction, the full
horizontal beam toggles on. If the actor is on any full-beam cell at that
moment, or moves through one while the beam is on, it fails and later resets.
At the end of a non-winning six-instruction run, the beam toggles back off
before the checkpoint/reset logic completes. This means each fresh segment
starts with only the endpoints dangerous.

The useful passable states on the laser row are:

```text
Beam off:  x=37 and x=57 are dangerous; x=41,45,49,53 are passable.
Beam on:   x=37,41,45,49,53,57 are all dangerous.
```

One winning route uses the two right checkpoints and avoids the full beam:

```text
Run 1: [2, 33, 33, 2, 2, 3]     start -> checkpoint at (53,24)
Run 2: [33, 33, 33, 0, 0, 33]   (53,24) -> checkpoint at (53,8)
Run 3: [0, 1, 3, 1, 1, 33]      (53,8) -> target at (41,8)
```

Run 1 moves right, then up twice to y=20 before the full beam turns on. It then
moves right while the beam is on but stays below y=16, and finishes by moving
down onto the lower-right checkpoint. Run 2 moves upward along x=53. It reaches
y=12 before the beam turns on, waits two no-op slots while the beam is on, and
then moves up to the upper-right checkpoint. Run 3 waits one slot, moves left,
moves down to y=12 as the beam turns on, moves left twice while staying above
the beam row, and then moves up into the target.

The left checkpoint at (33,12) is reachable, but it is not required for this
route. The hidden checkpoint at the start simply gives the same reset behavior
as earlier levels if the first run fails.

## Implementation Checklist

- Only click input is available.
- Clicks on program bars toggle bits; clicks on run buttons execute; clicks on
  reference selectors load and run the left demonstration.
- Right-side win checking happens only after the whole current program finishes.
- The right actor must match target position, rotation, scale, and color.
- Wall collisions undo the attempted movement or scale change but do not stop
  the program.
- A failed right run resets the actor to the latest saved reset state and
  preserves the edited program.
- Checkpoints save only when a run ends on them with matching scale.
- Level 7 lasers toggle after the third instruction of a run and toggle back at
  the end of a non-winning run.
- The top timer advances on clicks, not on automatic instruction ticks.
