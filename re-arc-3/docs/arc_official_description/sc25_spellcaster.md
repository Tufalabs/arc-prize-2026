# Spellcaster

## Core Game

Spellcaster is a 64x64-grid maze game with a movable player and a clickable spell board. The player is a 2x2 sprite drawn in two colors. It normally has scale 2, so it occupies a 4x4 pixel body and moves 4 pixels per direction action. The resize spell can shrink it to scale 1, where it occupies a 2x2 body and moves 2 pixels per direction action. Many passages are exactly 2 pixels high or wide, so the large body cannot enter them.

The win condition in every level is to touch the orange finish portal. Contact with the portal starts a short two-frame slide toward the portal, then advances to the next level. Wall pixels, colored gates, colored blast targets, and purple/blue corner blocks are collidable. The right edge contains a 32-segment action meter rendered as a 2-pixel-wide vertical bar.

Each level has a fixed action limit. Direction actions increment the spent-action counter, whether or not the move is productive. Clicking a spell-board cell also increments the counter. When a spell effect finishes, the counter increments once more. If the counter becomes greater than the level's limit, the level is lost. Yellow 2x2 recharge pickups reduce the spent counter by 10, down to a minimum of 0, then disappear until the level is restarted.

## Spell Board

The spell board is the 3x3 panel in the lower-right corner. Its cell top-left coordinates are:

```text
(24,49)  (29,49)  (34,49)
(24,54)  (29,54)  (34,54)
(24,59)  (29,59)  (34,59)
```

Clicking a cell toggles that cell on or off. A spell fires only when the 3x3 board exactly equals one of the level's available patterns; extra filled cells make the pattern invalid. After a spell resolves, the board is cleared.

Spell plaques on the left show which spells are available. In the first three levels, the active spell is preselected and the required cells flash as a tutorial hint. Later levels show plaques but do not pre-fill the memory work; the same three patterns must be remembered.

The three spell patterns are:

```text
Resize             Teleport           Blast
. X .              X X .              . X .
X . X              . X .              . X .
. X .              . . .              . X .
```

## Spell Effects

Resize toggles the player between scale 2 and scale 1. Shrinking always succeeds. Growing succeeds only if a 4x4 body can fit at or slightly up/left of the current position. The growth check tries offsets from -2 to 0 on each axis and rejects the spell if every possible 4x4 placement would collide with a wall or colored gate. A blocked growth attempt flashes the player and leaves it small.

Teleport moves the player to the next size-compatible teleport target. Large targets are 4x4 corner outlines. Small targets are 2x2 markers. The large-target index and small-target index are separate, and each advances after a successful teleport. Only the size-compatible target markers are emphasized: large target frames while large, small target frames while small.

Blast fires a projectile in the player's current facing direction, which is set by the last direction action. The projectile travels until it reaches the first wall, gate, finish portal, or blast target. It destroys only blast targets. A normal red/gold target removes itself and every normal gold gate in the level. A cyan target removes itself and every cyan gate in the level. Purple/blue corner blocks are movement obstacles but do not stop the blast scan.

## Coordinate Legend

Coordinates below are inclusive pixel rectangles in the 64x64 grid. A floor rectangle means the wall mask is open there; other collidable objects may still block part of that floor. Object coordinates are bounding boxes; corner-outline blockers and portals use their own pixel masks inside those boxes. The spell board, spell plaques, action meter, and hidden click colliders are user-interface objects and are not listed as playable maze floor.

Object labels:

```text
P  player start
F  finish portal
R  yellow recharge
T  teleport target
t  small teleport target
G  normal gold gate
g  cyan gate
B  normal blast target
b  cyan blast target
O  fixed purple/blue movement blocker
```

## Level 1

Action limit: 50. Available spell: resize.

Maze floor:

```text
x=16..42, y=19..20
x=23..42, y=21..22
```

Objects:

```text
P: x=39..42, y=19..22, scale 2
F: x=12..16, y=17..22
```

The player starts in the wider right side of a top corridor. The right part of the corridor is 4 pixels high, so the large 4x4 player can stand and move there. The left neck is only 2 pixels high at y=19..20. The finish portal touches the left end of that neck.

To win, fill the resize diamond on the spell board. The player shrinks to a 2x2 body, then can move left through the 2-pixel-high neck and touch the finish portal. The level cannot be solved at full size because the large body collides with the lower wall of the neck.

Important reconstruction details:

- The resize plaque is shown next to the board and is preselected.
- The first non-reset action starts a tutorial flash over the four resize-pattern cells.
- There are no gates, teleport targets, blast targets, recharges, or extra blockers.

## Level 2

Action limit: 25. Available spell: teleport.

Maze floor:

```text
x=31..34, y=11..38
x=27..30, y=15..22
x=35..38, y=15..22
x=27..30, y=31..38
x=35..38, y=31..38
```

Objects:

```text
P: x=31..34, y=35..38, scale 2
F: x=30..35, y=10..14
T: x=31..34, y=19..22
O: x=30..35, y=27..30
```

The maze is a 4-pixel-wide vertical shaft with side pockets. The player starts below a purple blocker that spans the shaft at y=27..30. The finish portal is above the blocker. Normal movement cannot pass through the blocker.

To win, fill the teleport angle pattern. The large player teleports to the 4x4 target at x=31..34, y=19..22, which is above the blocker. From there it can move upward into the finish portal.

Important reconstruction details:

- The teleport target is drawn as a 4x4 corner outline with a larger 6x6 active frame around it.
- The teleport plaque is preselected and can flash its required cells.
- The blocker remains in place; teleporting is what bypasses it.

## Level 3

Action limit: 50. Available spell: blast.

Maze floor:

```text
x=31..42, y=10..33
x=27..30, y=14..41
x=43..46, y=14..25
x=47..58, y=22..25
x=23..26, y=38..41
```

Objects:

```text
P: x=35..38, y=22..25, scale 2
F: x=22..26, y=37..42
G: x=27..30, y=34..36
B: x=55..58, y=22..25
O: x=47..50, y=21..26
```

The lower-left finish path is blocked by the normal gold gate at x=27..30, y=34..36. The matching blast target is far to the right in the horizontal branch, at x=55..58, y=22..25. The player begins on the same horizontal band as that target.

To win, make the player face right, fill the vertical-line blast pattern, and cast a blast across the branch into the normal target. Destroying that target removes the normal gold gate. The player can then travel down the vertical shaft and left into the finish portal.

Important reconstruction details:

- The purple blocker near x=47..50 is a movement obstacle, but the blast scan does not treat it as a blast-stopping object.
- If the blast is aimed into a wall or gate instead of the target, no gate is removed.
- There are no recharges, resize targets, or teleport targets.

## Level 4

Action limit: 35. Available spells: resize and blast.

Maze floor:

```text
x=27..46, y=11..38
x=07..10, y=28..31
x=11..26, y=29..30
x=47..54, y=35..38
```

Objects:

```text
P: x=35..38, y=19..22, scale 2
F: x=51..55, y=34..39
R: x=44..45, y=28..29
G: x=47..49, y=35..38
B: x=07..10, y=28..31
```

The central chamber connects to the left blast target through a 2-pixel-high tunnel at y=29..30. The large player cannot enter that tunnel. The right finish pocket is reachable only after the normal gold gate at x=47..49, y=35..38 is removed.

To win, cast resize to shrink, move through the narrow tunnel to the left side, face left, and cast blast into the normal target at x=07..10, y=28..31. This removes the right-side gate. Return through the tunnel and move into the finish portal. The recharge in the central chamber can be collected to buy 10 actions back.

Important reconstruction details:

- The resize and blast plaques are both present, but no spell is preselected at level start.
- The left chamber is exactly tall enough for the target and narrow enough to force the resize lesson.
- The right gate is a vertical rotated normal gate; removing it opens the x=47..54 finish pocket.

## Level 5

Action limit: 65. Available spells: resize, teleport, and blast.

Maze floor:

```text
x=15..18, y=11..46
x=35..38, y=11..22
x=51..54, y=11..38
x=31..34, y=15..18
x=39..42, y=15..18
x=11..14, y=39..46
x=19..30, y=39..40
x=03..10, y=43..46
```

Objects:

```text
P: x=35..38, y=15..18, scale 2
F: x=50..55, y=10..14
R: x=12..13, y=40..41
T: x=51..54, y=35..38
t: x=29..30, y=39..40
G: x=51..54, y=16..18
g: x=51..54, y=20..22
B: x=15..18, y=11..14
b: x=03..06, y=43..46
O: x=14..19, y=31..34
O: x=07..10, y=42..47
O: x=28..31, y=38..41
```

This level asks the player to combine all previous mechanics without a fresh pattern tutorial. The start is in the upper middle chamber. The finish is at the top of the right shaft, but two gates block that shaft: the normal gold gate at y=16..18 and the cyan gate at y=20..22. Each gate has its own blast target in the left-side maze.

The large teleport target is in the lower part of the right shaft. Teleporting there before opening the gates does not solve the level because both gates still block the climb to the finish. The small teleport target is in the 2-pixel-high lower connector, which is the route into the left maze.

One reliable solution structure is:

1. Cast resize in the starting chamber to shrink.
2. Cast teleport while small to move to x=29..30, y=39..40.
3. Move through the lower connector and collect the recharge if needed.
4. Blast the cyan target in the lower-left room from the right; this removes the cyan gate in the right shaft.
5. Climb the left shaft and blast the normal target at the top; this removes the normal gold gate in the right shaft.
6. Grow back to scale 2 in a 4-pixel-wide clear part of the left shaft.
7. Cast teleport while large to move to x=51..54, y=35..38, then climb the now-open right shaft to the finish.

Important reconstruction details:

- The large and small teleport targets are different physical objects. The small target should become emphasized only after the player is small.
- The recharge sits near the lower-left connector and is optional only with efficient routing.
- The two right-shaft gates are independent. Destroying only one target still leaves the other gate blocking the finish route.
- The corner blockers shape the movement route but should not absorb blast projectiles.

## Level 6

Action limit: 60. Available spells: resize, teleport, and blast.

Maze floor:

```text
x=33..36, y=09..32
x=41..52, y=17..20
x=41..44, y=21..24
x=49..52, y=21..32
x=29..32, y=29..36
x=53..56, y=29..40
x=17..20, y=33..38
x=37..52, y=37..40
x=17..18, y=39..42
x=13..16, y=41..42
```

Objects:

```text
P: x=41..44, y=21..24, scale 2
F: x=32..37, y=08..12
R: x=17..18, y=33..34
R: x=37..38, y=37..38
T1: x=53..56, y=37..40
T2: x=29..32, y=33..36
t:  x=13..14, y=41..42
G: x=42..44, y=37..40
g: x=33..36, y=14..16
B: x=17..20, y=33..36
b: x=37..40, y=37..40
O: x=12..15, y=40..43
```

The final level chains size-dependent teleporting with two separate blast-gate pairs. The finish is above the central shaft, but the cyan gate at x=33..36, y=14..16 blocks the climb. The lower horizontal passage is initially split by the normal gold gate at x=42..44, y=37..40. The normal target and cyan target each have a recharge underneath their top-left area, so destroying a target can expose a useful recharge pickup.

Teleport order is essential. While large, the first teleport goes to T1 at x=53..56, y=37..40. The next large teleport goes to T2 at x=29..32, y=33..36. While small, teleporting goes to the small target at x=13..14, y=41..42.

One reliable solution structure is:

1. Cast resize near the start to shrink.
2. Cast teleport while small to reach the lower-left small target.
3. Move into the left room and blast the normal target at x=17..20, y=33..36, usually from below. This removes the normal gate in the lower passage.
4. Grow back to scale 2 in the cleared 4-pixel-wide left room.
5. Cast teleport while large to reach T1 in the lower-right shaft.
6. Move left through the opened lower passage and blast the cyan target at x=37..40, y=37..40. This removes the cyan gate in the central shaft.
7. Cast teleport while large again to reach T2 in the lower central shaft.
8. Move from T2 into the central vertical shaft, climb through the removed cyan gate, and touch the finish portal.

Important reconstruction details:

- The first large teleport marker starts around T1 and moves to T2 after the first large teleport.
- The small target is framed by the blue corner blocker; the blocker constrains movement but leaves the target's center usable.
- The cyan gate must be removed before the final climb. Opening only the lower normal gate is not enough.
- The key state for this level is the teleport-target indices plus player scale; do not reset teleport order when a gate is opened.

## Implementation Checklist

- Use a 64x64 logical grid with pixel-precise collision.
- Direction movement is 4 pixels at scale 2 and 2 pixels at scale 1.
- The player's facing direction is the most recent direction action and controls blast direction.
- The spell board is a persistent 3x3 toggle grid and only exact patterns cast spells.
- Spell effects run as short animations; input is blocked while an effect is active.
- Resize clears the spell grid and toggles scale only if the destination body fits.
- Teleport chooses targets by current scale and cycles target indices after use.
- Blast destroys only normal or cyan blast targets, then removes the matching gate set.
- Recharges subtract 10 spent actions and are removed.
- Exceeding the per-level action limit loses the level.
- Touching the finish portal advances to the next level after the short slide animation.
