# Dual Control

## Core Game

Dual Control is a discrete goal-reaching puzzle in which the player controls two different things:

- The green 2x2 player avatar, moved with the four direction actions.
- Clickable machinery on the right side of the screen, used to change bridges, teleporters, gates, and remote manipulators.

The objective in every level is to move the green avatar onto the yellow 2x2 goal. Reaching the goal immediately advances to the next level. The player moves in 2-pixel increments on a 64-pixel render grid. Most usable floor pieces are 4-pixel-aligned platforms, so the avatar's top-left coordinate is the useful reconstruction coordinate.

The play area is the left side of the board. A black machinery panel occupies the right side and is separated from the play field by a vertical dashed white barrier. White-bordered colored shapes on that panel are clickable controls. A click never moves the avatar directly unless the avatar is standing on a teleporter that belongs to the clicked control group.

## Movement, Support, And Failure

The avatar is a solid 2x2 sprite. A direction action attempts to move it 2 pixels. The move is rejected if the avatar would collide with a solid sprite. If the move has no collision, it is accepted only when the avatar's new top-left pixel is over a passable support sprite. Empty background, the black control panel, black visual gaps, and most solid obstacle states are not support. This means that a narrow bridge can be passable even if it is only a 2-pixel-high strip, as long as the avatar's top-left pixel remains on that strip.

Each level has a step counter shown as the bottom bar. Levels 1, 2, 3, and 4 start with 128, 192, 192, and 192 steps. Levels 5 and 6 start with 512 and 1024 steps because they involve much more remote manipulation. Direction moves consume one step. Successful remote-manipulator clicks also consume one step. Ordinary environment button clicks consume two steps: one for the machinery click and one for the no-movement click action that follows it.

If the avatar is left without support after a machinery click, the level plays a short failure animation, subtracts 20 steps, and restores the complete state snapshot from before that click. If the step counter reaches zero, the game ends. Moving into unsupported space by direction input is simply rejected rather than allowing the avatar to fall.

## Switch Groups

Most colored buttons and objects are organized into matching control groups. Clicking a button affects every active object in that group.

Numbered platform objects cycle to their next numbered state. For example, a two-state bridge changes from state 1 to state 2, and the next click changes it back to state 1. Larger moving bridges can have six or eight states. A cycled state can be:

- Passable support, drawn as a complete colored bridge or platform.
- A solid/blocked state, often shown as a checker or sparse version of the same color.
- An empty/invisible state, used for a missing bridge.
- A clickable button state, used when a trigger reveals a new control.

The important reconstruction rule is that buttons change topology without carrying the avatar. If a platform changes from under the avatar, the avatar remains at its coordinate and must still have support after the change.

## Triggered Buttons

Some levels contain small 4x2 trigger pads in the play field. Stepping on one removes that pad and permanently reveals the matching hidden button on the right panel. These are used to force the avatar to visit a region before a later control can be used.

Level 6 also has 2x2 sensor markers near the bottom-right. These do not disappear. A sensor reveals one remote-control button only while the avatar is standing on that sensor. Stepping away hides the button again. The player must therefore position the avatar on the correct sensor before clicking the corresponding remote-control command.

## Teleporters

Teleporters are 2x2 alternating-color tiles. They belong to a switch group. When the avatar is standing exactly on an active teleporter and the matching button is clicked, the avatar is moved to the paired teleporter in the same group. The machinery state then cycles as part of the same click.

The pair is determined by teleporter color family and numeric phase. In the simple levels there are two same-color teleporters, so the effect is simply "stand on one, click the matching button, appear at the other." Level 6 adds a color-cycle button that changes one teleporter's color family, thereby changing which other teleporter it pairs with.

## Remote Manipulators

Levels 5 and 6 add a remote manipulator controlled by separate arrow buttons and a grab button on the right panel.

The manipulator has an anchor point. When it is not carrying anything, the grab button tries to attach the object centered on that anchor. Once an object is grabbed, there is no release action in the level; the object remains attached and follows the manipulator. A failed grab only flashes/jiggles the manipulator.

In level 5, the manipulator follows a fixed U-shaped path:

- It starts at offset `(0, 0)` from its base position.
- It can move up and down only on the left column of the path.
- It can move left and right only on the bottom row or top row of the path.
- Each manipulator move changes the manipulator by 4 pixels.

In level 6, the manipulator follows a visible black guide track instead. Each 4-pixel move is legal only when the manipulator's next anchor point lands inside a black guide tile. If the manipulator is carrying a large bridge and the move would make that bridge overlap another active large bridge, the move is refused and the manipulator jiggles.

## Coordinate Legend

All positions below are top-left pixel coordinates in the game's render grid. Sizes are written as `width x height`.

Common objects:

- `P`: green avatar start, 2x2.
- `G`: yellow goal, 2x2.
- `floor`: static passable support.
- `button`: clickable right-panel control.
- `trigger`: play-field pad that reveals a hidden button.
- `sensor`: level 6 marker that temporarily reveals a remote-control button.
- `bridge/platform`: switchable support or obstacle.
- `teleporter`: 2x2 teleport tile.
- `manipulator`: remote claw/clamp.

## Level 1

Level 1 introduces the basic idea: the avatar and the bridge controls are separate. The avatar starts at `(10, 30)` on a lower-left floor block. The goal is at `(24, 10)` on a small upper platform. The player must use the right-panel buttons to configure a passable route upward and rightward.

Initial layout:

```text
Step limit: 128
P: (10,30)
G: (24,10)
Static floor: (8,28) size 6x6; (22,8) size 6x6; (8,20) size 4x4
Right-panel buttons:
  red group A button: (42,7) size 13x5
  blue group B button: (42,24) size 13x5
```

Switchable objects:

```text
Group A red bridge at (12,14), size 10x10:
  State 1: horizontal lower bar, covering x 12..21 and y 20..23.
  State 2: vertical right bar, covering x 18..21 and y 14..23.

Group B blue small bridges:
  Lower bridge at (8,24), size 4x4:
    State 1: sparse/solid checker, not useful as passable support.
    State 2: full passable support.
  Upper bridge at (18,10), size 4x4:
    State 2 starts active as full passable support.
    State 1 is the sparse/solid checker state.
```

To win, the player must alternately make the missing parts of the route passable. The lower blue bridge is needed to leave the starting block and reach the red bridge area. The red bridge must be rotated between horizontal and vertical states to connect the lower route to the upper route. The upper blue bridge must be in its passable state when the avatar crosses to the goal platform.

Important reconstruction details:

- The red bridge does not move the avatar. It only changes which red pixels are support.
- The blue control affects both blue bridges at once, so making the lower blue bridge usable can make the upper blue bridge unusable until the button is clicked again.
- The level has no triggers, teleporters, or remote manipulator.

## Level 2

Level 2 adds a hidden button that must be unlocked from inside the maze. The avatar starts at `(6, 22)`. The goal is at `(22, 4)`. The final route depends on a red bridge button that is initially absent from the control panel.

Initial layout:

```text
Step limit: 192
P: (6,22)
G: (22,4)
Static floor:
  (4,20), (20,4), (4,32), (8,16), (32,16), (16,44), each 4x4
Trigger for hidden red button: (16,45) size 4x2
Right-panel buttons:
  blue group B button: (46,30) size 13x5
  purple group C button: (46,12) size 13x5
  hidden red group A button: (46,21) size 13x5, revealed by the trigger
```

Switchable objects:

```text
Group A red bridge at (12,8), size 20x20:
  State 1 starts active as a horizontal bridge through the middle of the 20x20 box.
  State 2 is a vertical bridge through the middle of the same box.

Group B blue small bridges:
  Two lower-left blue 4x4 blocks at (4,24) and (4,28) start as sparse/solid checkers.
  One upper-left blue 4x4 block at (8,20) starts as full passable support.
  Clicking the blue button swaps each block to its opposite state.

Group C purple bridge at (8,24), size 20x20:
  State 1 starts active as a horizontal bridge.
  State 2 is a vertical bridge.
```

The intended structure is to use the initially available blue and purple controls to get from the start to the bottom trigger. The purple bridge must be rotated into a vertical route to reach the trigger at `(16,45)`. Stepping on that trigger removes it and reveals the red button on the right panel. After the red button is available, the red bridge can be rotated so the avatar can reach the upper route and the goal.

Important reconstruction details:

- The hidden red button should become visible and clickable permanently after the trigger is touched.
- The trigger itself is passable support and disappears after activation.
- The red bridge is impossible to control before the trigger is activated, even though the red bridge is visible in the play field.

## Level 3

Level 3 introduces teleporters. The avatar starts at `(30, 10)`, near the upper-right of the play field. The goal is at `(32, 30)`. The goal area cannot be reached by bridge switching alone; the avatar must stand on a teleporter and click the matching teleporter button.

Initial layout:

```text
Step limit: 192
P: (30,10)
G: (32,30)
Static floor:
  (28,8), (4,16), (12,36), (16,32), (30,28), (4,8), each 4x4
Trigger for hidden group D button: (16,34) size 4x2
Right-panel buttons:
  red group A button: (45,8) size 13x5
  blue group B button: (45,17) size 13x5
  purple teleporter group C button: (45,26) size 13x5
  hidden magenta group D button: (45,35) size 13x5, revealed by the trigger
```

Switchable objects:

```text
Group A red bridge at (12,8), size 20x20:
  State 1 starts horizontal.
  State 2 is vertical.

Group B blue small bridges:
  Sparse/solid checkers start at (4,12), (12,32), and (24,8).
  A full passable blue bridge starts at (8,16).
  Clicking the blue button swaps all of these.

Group C teleporters:
  Teleporter 1 starts at (6,8).
  Teleporter 2 starts at (12,38).
  Standing on either teleporter and clicking the purple button moves the avatar to the other.

Group D magenta gate pieces:
  Horizontal gate at (24,24), size 10x4, starts sparse/solid.
  Vertical gate at (16,24), size 4x8, starts sparse/solid.
  Clicking the revealed group D button changes both to full passable support.
```

To win, the player must use the red and blue controls to reach the teleporter area, stand on a purple checker teleporter, and click the purple button to transfer to the other teleporter. From there the avatar can reach the trigger that reveals the group D button. The group D button opens the magenta gate pieces needed for the last crossing to the goal.

Important reconstruction details:

- A teleporter click only teleports if the avatar is exactly on an active teleporter in that group.
- Clicking the teleporter button while away from a teleporter still cycles the teleporter visuals but does not move the avatar.
- The magenta gate button is hidden until the trigger is touched.

## Level 4

Level 4 combines teleporting with two kinds of moving bridge. The avatar starts at `(4, 10)`. The goal is at `(30, 36)`. The level has a long upper gap, lower sliding platforms, and a pair of teleporters.

Initial layout:

```text
Step limit: 192
P: (4,10)
G: (30,36)
Static floor:
  (4,10), (24,10), (24,18), (16,22), (6,22), (4,26),
  (28,34), (24,14), (4,22), (6,26), each 4x4
Additional static block: (4,16) size 6x4
Right-panel buttons:
  purple teleporter group C button: (46,9) size 13x5
  yellow bridge group E button: (54,18) size 6x6
  gray/L-shaped sliding group F button: (43,17) size 7x7
```

Switchable objects:

```text
Group C teleporters:
  Teleporter 1 starts at (18,24).
  Teleporter 2 starts at (4,16).
  Clicking the purple button teleports between them when the avatar is standing on one.

Group E horizontal bridge strips, size 16x2:
  Upper strip box at (8,12) starts in state 1, which is empty/no useful support.
  Lower strip box at (16,32) starts in state 5, a complete passable strip.
  The yellow button cycles each strip through eight frames: empty, small end pieces,
  larger end pieces, near-complete, complete, then shrinking back toward empty.

Group F sliding bridge strips, size 16x2:
  Strip box at (8,20) starts with support on the left side.
  Strip box at (4,30) starts with support on the left side.
  The gray/L-shaped button cycles each strip through six frames, shifting the
  6-pixel-long support segment from left to right.
```

To win, the player must time the bridge states rather than expecting platforms to carry the avatar. The yellow control makes the group E bridge appear, fill, and disappear across its fixed boxes. The gray/L-shaped control is the passability control for the black-looking gaps: it shifts the group F support segment through fixed positions, making different parts of those gaps walkable. The avatar must move only after a bridge has been placed under the next part of the route. The teleporter pair provides the shortcut between the upper-left area and the lower/middle area.

Important reconstruction details:

- The group E bridge does not translate the avatar; it only changes which pixels are currently support.
- The group F bridge is a sliding support window, not a moving object that pushes or carries.
- If a clicked bridge state removes the support under the avatar, the click is rolled back after the fall animation and 20-step penalty.

## Level 5

Level 5 introduces the remote manipulator and a carried block. The avatar starts at `(32, 34)`. The goal is at `(10, 50)`. The right side contains both normal environment buttons and a separate cluster of manipulator controls.

Initial layout:

```text
Step limit: 512
P: (32,34)
G: (10,50)
Remote manipulator: (2,24), size 8x8
Carryable block: (20,10), size 6x12
Trigger for hidden group D button: (10,18) size 4x2
Static floor:
  (30,34), (22,6), (26,30), (26,20), (10,34), (10,42),
  (10,50), (10,18), each 4x4
Right-panel environment buttons:
  blue group B button: (46,39) size 13x5
  purple teleporter group C button: (46,46) size 13x5
  hidden magenta group D button: (46,12) size 13x5, revealed by the trigger
  yellow bridge group E button: (54,20) size 6x6
  gray/L-shaped sliding group F button: (44,19) size 7x7
Remote-control buttons:
  up: (48,28) size 4x4
  down: (58,28) size 4x4
  left: (43,28) size 4x4
  right: (53,28) size 4x4
  grab: (48,34) size 9x3
```

Switchable objects:

```text
Group B blue small bridges:
  Sparse/solid checker at (26,34), size 4x4.
  Full passable bridge at (10,38), size 4x4.
  The blue button swaps them.

Group C teleporters:
  Teleporter 1 starts at (24,6).
  Teleporter 2 starts at (10,34).
  The purple button teleports between them when the avatar is standing on one.

Group D magenta route pieces:
  Horizontal piece at (14,50), size 10x4, starts as full passable support.
  Vertical piece at (26,46), size 4x8, starts sparse/solid.
  The hidden group D button swaps both pieces after it is revealed.

Group E horizontal strip at (14,44), size 16x2:
  Cycles through the same eight bridge-fill frames as level 4.

Group F sliding strip at (14,54), size 16x2:
  Cycles through the same six left-to-right support-window frames as level 4.
```

The carried block is the new central mechanic. The remote manipulator must be moved along its U-shaped track until its anchor is centered over the carryable block. Clicking grab attaches the block. After that, the block follows the manipulator whenever the manipulator moves, and there is no way to drop it. The block is passable support, so it can be used as a movable bridge for the avatar.

To win, the player must combine all earlier mechanics. The avatar needs to activate the trigger that reveals the group D button, use teleports and bridge switches to move between separated regions, and use the manipulator to carry the block into positions where it bridges otherwise unsupported gaps. The remote block is not optional: the lower goal route cannot be reconstructed as a continuous path without moving this support.

Important reconstruction details:

- The manipulator's four arrow buttons move the manipulator, not the avatar.
- The manipulator moves 4 pixels per successful remote-control click.
- The carryable block is grabbed only when its center coincides with the manipulator anchor.
- Once grabbed, it remains attached for the rest of the attempt.
- The player can stand on the carried block after it has been positioned, but it does not automatically transport the player while moving.

## Level 6

Level 6 is the final combined machinery level. It replaces the level 5 carried block with a large bridge that can be clamped, moved on a black guide track, rotated, and used together with color-changing teleporters. The avatar starts at `(28, 52)`. The goal is at `(46, 6)`.

Initial layout:

```text
Step limit: 1024
P: (28,52)
G: (46,6)
Remote clamp: (22,30), size 8x8
Large clampable bridge: (0,24), size 20x20
Trigger for hidden teleporter color-cycle button: (6,18) size 4x2
Trigger for hidden grab button: (34,48) size 4x2
Right-panel always-available buttons:
  purple/teleporter group C button: (45,23) size 13x5
  gray/L-shaped group F button: (53,5) size 7x7
Hidden or sensor-controlled buttons:
  color-cycle group D button: (49,46) size 4x4, revealed by the trigger at (6,18)
  grab button: (47,17) size 9x3, revealed by the trigger at (34,48)
  up button: (49,31) size 4x4, visible only while standing on sensor (34,56)
  right button: (53,35) size 4x4, visible only while standing on sensor (36,58)
  left button: (45,35) size 4x4, visible only while standing on sensor (32,58)
  down button: (49,39) size 4x4, visible only while standing on sensor (34,60)
```

Static support and guide geometry:

```text
Ordinary static floor:
  (8,20), (4,20), (4,16), (8,16),
  (4,4), (8,4), (8,8), (4,8),
  (32,4), (32,8), (36,8), (36,4), (38,4), (38,8),
  (46,8), (46,4), (42,8), (42,4), each 4x4
  plus larger floor pads at (32,56), (18,48), (6,48), (24,48), each 6x6
  and small floor tiles at (8,44), (6,44), each 4x4.

Black guide track for the clamp:
  (20,36), (28,28), (28,24), (28,20), (24,20), (20,20),
  (20,16), (16,32), (12,32), (8,32), (20,12), (20,8),
  (20,28), (24,32), (28,32), (24,36), (28,36), (24,28), (20,32),
  each 4x4.
```

Switchable and special objects:

```text
Group C large bridge and teleporters:
  Large bridge at (0,24), size 20x20:
    State 1 starts as a horizontal/centered bridge body.
    State 2 is the rotated vertical bridge body.
  Teleporter at (34,58), starts as one color family with phase 2.
  Teleporter at (4,4), starts as another color family with phase 2.
  Teleporter at (18,48), starts as purple family phase 1.
  Teleporter at (32,52), starts as purple family phase 2.
  Clicking the group C button both handles teleporting, if the avatar is on a
  paired teleporter, and cycles all group C bridge/teleporter phases.

Group D color-cycle button:
  Affects the teleporter at (18,48).
  Each click changes that teleporter's color family in this order:
  purple -> red/blue -> yellow/cyan -> orange/blue -> purple.
  Its numeric phase is preserved, so changing color changes which other
  phase-2 teleporter it can pair with.

Group F lower sliding strips:
  Multiple overlapping 16x2 strip boxes start at (6,54), (6,56), and (6,58).
  The group F button cycles their six-frame support windows together, producing
  the shifting lower platform near the bottom of the level.
```

The remote clamp starts on the black guide track. Unlike level 5, it does not use the fixed U-shaped path. A clamp move succeeds only if the next anchor position lies on the black guide. The clamp can attach the large 20x20 bridge when its anchor is centered on that bridge. After attachment, moving the clamp moves the bridge, and clicking the group C button can rotate the bridge between its two orientations.

To win, the player must first gain access to the needed controls. The lower-left trigger reveals the teleporter color-cycle button. The bottom-right trigger reveals the grab button. The four remote movement buttons are not constantly available; the avatar must stand on the matching 2x2 sensor to expose the desired direction button, then click it while still on that sensor.

The teleporter color is part of the puzzle. Initially, the teleporter at `(18,48)` pairs naturally with the teleporter at `(32,52)`. By using the color-cycle button, the player can make it pair with the teleporter at `(4,4)` or with the teleporter at `(34,58)`, depending on its current color family. The group C button is then used to perform the actual teleport when the avatar is standing on the correctly colored teleporter.

The clamped bridge is the other required tool. The player must move the clamp left to the large bridge, grab it, move it along the black guide track, and rotate it as needed so it becomes usable support in new places. The bridge cannot be freely dragged anywhere; every clamp move must follow the black guide, and bridge-overlap checks can reject a move. The final route to the goal requires both the correct teleport pairing and the large bridge positioned/oriented as support.

Important reconstruction details:

- The goal is on the right side of the dashed divider, so the avatar must ultimately cross into that side through constructed support rather than by walking through the divider.
- The color-cycle button changes teleporter family but does not itself teleport the avatar.
- The group C button both rotates/cycles the clamped bridge and activates teleporting for the current matching teleporter.
- Sensor-controlled remote buttons should disappear as soon as the avatar leaves their corresponding sensors.
- The bottom sliding platform controlled by group F is intentionally awkward: all overlapping strips cycle together, so the passable window must be timed with the avatar's position.

## Reconstruction Checklist

- Use a 64-pixel grid and move the avatar exactly 2 pixels per direction input.
- Treat support as a separate rule from collision: a move needs no solid collision and a passable support pixel under the avatar's top-left coordinate.
- Use `next level` progression only when the avatar's top-left coordinate equals the goal coordinate.
- Preserve the level step limits: 128, 192, 192, 192, 512, and 1024.
- Right-panel controls are clicked by position; they are not reachable by avatar movement.
- Button groups cycle all active matching objects together.
- Trigger pads reveal hidden buttons and then disappear.
- Teleporter clicks require the avatar to stand exactly on an active teleporter in the clicked group.
- Platform changes do not carry the avatar.
- A machinery click that removes support under the avatar rolls back to the pre-click state after a 20-step penalty.
- Level 5's carried block cannot be released after grabbing.
- Level 6's clamp moves only along the black guide track, and sensor markers temporarily reveal the four directional clamp buttons.
