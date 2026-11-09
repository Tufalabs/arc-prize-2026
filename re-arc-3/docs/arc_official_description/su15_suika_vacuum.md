# Suika Vacuum

## Core Game

Suika Vacuum is a click-driven continuous-position merging puzzle. The board is a 64x64 pixel grid. The top 10 rows are a target display, rows 10 through 62 are the play area, and row 63 is a step meter. The player does not control an avatar. Instead, each click creates a shrinking circular vacuum marker, and nearby objects move toward that clicked point.

The objective is to put the required objects into the circular goal bubbles. A goal is satisfied by an object's center point being inside the goal bubble's 9x9 bounding box. Goals are interchangeable: if a level requires one size-3 friendly blob and one type-1 enemy, either required object may be placed in either goal, as long as the final counts inside all goals match the target display.

Most levels are about the same three operations:

- Pull friendly blobs together so equal sizes merge into the next size.
- Keep different sizes from touching during a merge, because incompatible contacts flash and undo the click.
- Use or avoid enemy ships. Enemy contact downgrades friendly blobs by one size, and later levels require enemy ships themselves to be merged and parked in goals.

Decorative panels and gray blocks do not block movement. The movement rules only use board bounds, dynamic object overlap, merge compatibility, enemy hits, and goal containment.

## Controls And Turn Model

The main action is a click on the play area. Valid click points are spaced on a 4-pixel grid: `x = 0, 4, 8, ... 60` and `y = 10, 14, 18, ... 62`. A click outside rows 10 through 62 does nothing.

Before every play-area click, the game stores an undo snapshot containing every friendly blob and enemy ship. The undo action restores the most recent snapshot, rebuilding all friendly blobs and ships at their stored sizes, types, and positions. Undo does not refund step meter energy.

On a click:

1. Every friendly blob within radius 8 of the clicked point is selected.
2. Every enemy ship within radius 8 of the clicked point is also selected.
3. A circular outline appears at the clicked point with radius 8, then shrinks over four animation ticks.
4. Selected friendly blobs move toward the clicked point for four ticks. On each tick, each axis moves by up to 4 pixels toward the target, so a blob can move at most 16 pixels per click on each axis.
5. Selected type-2 and type-3 enemy ships move toward the clicked point using the same 4-pixel-per-axis movement. Selected type-1 ships move along the vector from the ship to the clicked point at about 3.4 pixels per tick.
6. Unselected enemy ships chase the nearest non-hit friendly blob during the click animation. Type-1 and type-2 ships move 1 pixel per tick on each needed axis. Type-3 ships move 2 pixels per tick.
7. Enemy hits, enemy merges, and friendly merges resolve during or after the animation.
8. If the goal condition is met, the goal bubbles and target display flash black for a short win animation, then the next level starts.
9. Otherwise the click costs one step from the step meter.

The step meter is the black bar along the bottom row. If it reaches zero, the level fails. If all friendly blobs are removed, the level also fails. A reset restores the current level to its initial state, including the step meter, dynamic objects, enemy ships, undo history, and feedback colors.

## Friendly Blobs

Friendly blobs have discrete sizes even though they move in continuous pixel coordinates. The sprite is a solid square or rectangle filled with one ARC color index.

```text
size  dimensions  color index  value used by feasibility check
0     1x1         10           1
1     2x2         6            2
2     3x3         15           4
3     4x4         11           8
4     5x5         12           16
5     7x7         8            32
6     8x8         9            64
7     9x9         7            128
8     10x10       14           256
```

A merge is attempted after the four-tick pull finishes. The game looks only at selected friendly blobs. If two or more selected blobs of the same size have overlapping bounding boxes, that connected overlap group becomes one blob of the next size. The new blob is centered at the integer average of the old blob centers, then clamped inside the play area. If size-8 blobs merge, there is no size 9; the merged group is simply removed.

If selected friendly blobs of different sizes overlap when merge resolution runs, they flash black for 16 frames, the click is undone, and the step meter takes a penalty. The first illegal-contact penalty costs 2 steps, the second costs 4, the third costs 6, and so on.

Although the puzzles are designed around pairwise merges, the actual merge component can contain more than two blobs. Pulling three same-size blobs into one overlap group still creates only one next-size blob, so intended solutions should avoid accidental three-way merges.

After each click, the game also checks whether the remaining friendly mass is still enough to satisfy the required friendly targets. The mass of a size `n` blob is `2^n`. If the remaining mass is lower than the target mass, the goal bubbles and top target icons turn red as feedback. This is not an immediate loss, but the player must undo or reset to make the target possible again.

## Enemy Ships

Enemy ships are 5 pixels wide and 4 pixels high. They are patterned, not filled. `X` is a colored pixel and `.` is transparent.

```text
Type 1, color 7       Type 2, color 14      Type 3, color 13
..X..                 ..X..                 X.X.X
.X.X.                 .X.X.                 .X.X.
X...X                 X.X.X                 XXXXX
.XXX.                 .X.X.                 .XXX.
```

Enemy ships have two roles. First, they chase friendly blobs and downgrade them on contact. Second, equal-type ships can merge into stronger ships, which is required in the later levels.

When an enemy ship overlaps a friendly blob, the friendly blob is marked as hit:

- A size-0 blob shakes briefly and is then removed.
- A size-1 or larger blob shakes briefly, downgrades by one size, and is pushed away from the enemy.
- The enemy that caused the hit pauses briefly before chasing again.

Enemy ships merge globally, not just when selected. If two or more ships of the same type overlap, the connected group becomes one ship of the next type at the average center. Type 1 becomes type 2, and type 2 becomes type 3. Type-3 overlap has no higher type and removes the group. If ships of different types overlap, they flash black, the click is undone, and the illegal-contact step penalty is applied.

## Goals And Targets

Goal bubbles are 9x9 circular masks. Their visual shape is:

```text
..XXXXX..
.XXXXXXX.
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
.XXXXXXX.
..XXXXX..
```

For goal checking, the mask shape is not used. A friendly blob or enemy ship counts as inside a goal if its center point lies inside any goal bubble's 9x9 bounding box.

Each level's target display specifies exact counts for required friendly sizes and/or enemy ship types. Required counts must match exactly. Extra unrequested objects are ignored unless they are of a required size or enemy type and their centers are also inside goals.

Coordinates below are top-left pixel positions.

## Level 1

```text
Step budget: 32
Targets: one friendly size-2 blob
Goals: (44,11)
Friendly starts: size 2 at (3,58)
Enemies: none
```

This is the movement tutorial. The level starts with a single size-2 blob near the bottom-left corner and one goal bubble near the upper-right side. The decorative cross/guide art teaches that clicking pulls the blob toward the clicked point, but the only actual requirement is to get the blob's center into the goal at `(44,11)`.

To win, repeatedly click along a route from the blob toward the goal bubble. The blob can move at most 16 pixels per click on each axis, so several clicks are required. When its 3x3 sprite has center point inside the goal bubble's 9x9 box, the level is solved.

Important reconstruction details:

- There is no merge in this level.
- The click guide art disappears after the blob has been moved from its start.
- The lower play area contains decorative diagonal guide art, but it is not a wall.

## Level 2

```text
Step budget: 32
Targets: one friendly size-3 blob
Goals: (29,23)
Friendly starts:
  size 0 at (41,37), (18,37), (37,40), (16,41),
            (14,55), (16,57), (49,54), (47,56)
Enemies: none
```

This level introduces merging. It starts with eight size-0 dots arranged as four nearby pairs: two around the upper-left/middle-left area, two around the upper-right/middle-right area, two near the lower-left area, and two near the lower-right area. The target is one size-3 blob, which has value 8, exactly the value of all eight dots.

To win, merge the dots up the size chain:

1. Pull pairs of size-0 dots together to make four size-1 blobs.
2. Pull pairs of size-1 blobs together to make two size-2 blobs.
3. Pull the two size-2 blobs together to make one size-3 blob.
4. Move the size-3 blob into the goal at `(29,23)`.

Different sizes must not be pulled into overlap. If a size-0 dot overlaps a size-1 blob during merge resolution, the click flashes and is undone with a step penalty.

Important reconstruction details:

- The top target display shows the size-3 target as a 4x4 color-11 block.
- The friendly merge chain is the only active mechanic.
- A three-dot or three-blob pile still creates only one next-size blob, so the intended solution keeps merges pairwise.

## Level 3

```text
Step budget: 48
Targets: one friendly size-3 blob and one friendly size-2 blob
Goals: (5,46), (19,46)
Friendly starts:
  size 0 at (55,23), (61,23), (31,22), (31,15), (12,23), (8,28)
  size 1 at (46,22), (30,32), (18,16)
Enemies: none
```

This level asks for two finished blobs instead of one. The six size-0 dots and three size-1 blobs have total value 12. The target also has value 12: one size-3 blob has value 8, and one size-2 blob has value 4.

To win, partition the material so that one branch of merges produces a size-3 blob and another branch produces a size-2 blob. The two goal bubbles at the lower-left are interchangeable; the size-3 and size-2 blobs can be parked in either order as long as one of each required size is centered in a goal.

The main trap is mixing branches too early. Once a size-2 blob is meant to stay as the size-2 target, keep it away from the size-3 construction path. A size-2 touching a size-3 is an illegal merge contact, and a size-2 accidentally merging with another size-2 would become size 3, leaving no size-2 target unless enough smaller mass remains to rebuild it.

Important reconstruction details:

- There are no enemies, so the puzzle is purely about merge planning and click precision.
- The two goal bubbles share the same target pool; neither goal has a fixed size assignment.
- The 48-step budget allows more repositioning than level 2, but every play-area click still costs a step.

## Level 4

```text
Step budget: 48
Targets: one friendly size-3 blob
Goals: (1,53)
Friendly starts:
  size 0 at (5,26), (11,26), (31,27), (36,29),
            (33,47), (30,51), (12,47), (8,41)
Enemies:
  type 1 at (52,19)
```

This level adds the first active adversary. The eight size-0 dots are exactly enough mass to make one size-3 blob. A type-1 enemy ship starts near the upper-right. During every click animation, if the ship is not selected or cooling down, it moves toward the nearest friendly blob and will remove a size-0 dot or downgrade a larger blob on contact.

To win, merge all eight dots into a size-3 blob and move it into the lower-left goal. The enemy is only a hazard here. Any successful enemy hit reduces the remaining friendly mass below the size-3 requirement, causing the target display and goal to turn red. The player should undo immediately after such a hit or steer the enemy away before it reaches a blob.

Useful strategy:

- Build small pairs away from the enemy's path.
- When the enemy approaches, click near the enemy to select and pull it away, or move the active blob cluster away from it.
- Finish the size-3 merge before parking it in the goal at `(1,53)`.

Important reconstruction details:

- The enemy is not required in a goal in this level.
- Type-1 enemy contact with any size-0 dot removes that dot after the hit animation.
- Because the starting mass is exactly the target mass, no downgrade or removal can be tolerated unless it is undone.

## Level 5

```text
Step budget: 32
Targets: one friendly size-3 blob
Goals: (28,11)
Friendly starts:
  size 0 at (58,59), (44,53), (3,60), (14,54)
  size 1 at (14,28), (53,26), (6,25), (42,26)
Enemies:
  type 1 at (4,37), (46,37)
```

This level keeps the size-3 target but adds extra friendly mass and two type-1 enemies. The four size-1 blobs alone contain enough mass to produce the required size-3 blob. The four size-0 dots are additional material and can be used as decoys or backup, but they are not necessary for the final size-3 target.

To win, create one size-3 blob and park it in the upper goal at `(28,11)`. The most direct construction is to merge the four size-1 blobs into two size-2 blobs, then merge those into one size-3 blob. The enemies start between the upper size-1 blobs and lower size-0 dots, so they tend to cut across the playfield while chasing the nearest friendly. The puzzle is about using the surplus material and enemy behavior without allowing the final size-3 path to be downgraded.

Important reconstruction details:

- Extra friendly blobs outside the goal do not prevent the win.
- A single enemy hit on a nonessential blob can be acceptable if the remaining mass still includes a size-3 route.
- The 32-step budget is tight enough that repeated illegal-contact penalties or repeated failed attempts followed by undo can exhaust the level.

## Level 6

```text
Step budget: 32
Targets: one friendly size-3 blob and one type-1 enemy ship
Goals: (2,12), (52,53)
Friendly starts:
  size 5 at (33,32)
Enemies:
  type 1 at (16,34)
```

This level makes the enemy part of the objective. The player starts with one large size-5 blob and one type-1 ship. The target requires a size-3 friendly blob and a type-1 ship in the goal bubbles. Since the only friendly blob is size 5, it must be downgraded twice by enemy hits: size 5 becomes size 4, then size 4 becomes size 3.

To win:

1. Let or steer the type-1 enemy into the size-5 blob once, producing a size-4 blob.
2. Use the same mechanic a second time to produce a size-3 blob.
3. Move the size-3 blob into one goal bubble.
4. Move the type-1 enemy ship into the other goal bubble.

The two goals are interchangeable. A selected enemy ship does not perform its hit behavior during that click, which is useful when steering it into a goal after the downgrades have been completed.

Important reconstruction details:

- The target display contains both a friendly size-3 icon and a type-1 enemy icon.
- If the enemy hits the blob a third time, the blob becomes size 2 and no longer satisfies the target.
- If the enemy removes or over-downgrades the only friendly blob enough to make the target impossible, the goal feedback turns red.

## Level 7

```text
Step budget: 32
Targets: two friendly size-3 blobs
Goals: (19,13), (40,18)
Friendly starts:
  size 1 at (9,25), (20,35), (6,35), (30,37)
  size 5 at (51,46)
Enemies:
  type 1 at (12,51), (52,56)
```

This level combines deliberate downgrading with ordinary merging. The four size-1 blobs can merge into one size-3 blob: two pairs of size 1 make two size-2 blobs, and those merge into size 3. The size-5 blob must be downgraded twice by enemy hits to become the second required size-3 blob.

To win, build one size-3 blob from the four smaller blobs while also using one or both enemies to knock the size-5 blob down to size 3. Then park both size-3 blobs in the two upper goal bubbles. The enemies themselves are not target objects, so they should be kept out of the goals and away from the completed size-3 blobs after the downgrades are done.

Important reconstruction details:

- A hit on the small merge chain can destroy the required mass for that branch.
- A third hit on the downgraded size-5 branch over-downgrades it to size 2.
- The target count is exact for size 3. If a third size-3 blob somehow enters a goal, the count would be too high; unrelated sizes in goals are ignored.

## Level 8

```text
Step budget: 48
Targets: two friendly size-4 blobs and one type-2 enemy ship
Goals: (52,15), (3,15), (52,51), (3,51)
Friendly starts:
  size 3 at (13,42), (3,40)
  size 5 at (20,24)
Enemies:
  type 1 at (43,31), (29,53), (47,48)
```

This level introduces enemy merging as a target requirement. The player needs two size-4 friendly blobs and one type-2 enemy ship. The two starting size-3 blobs can merge into one size-4 blob. The size-5 blob must be downgraded once by an enemy hit to become the second size-4 blob. Separately, two type-1 enemy ships must overlap to merge into one type-2 ship, which then needs to be parked in a goal.

To win:

1. Merge the two size-3 blobs into one size-4 blob.
2. Use a type-1 enemy hit on the size-5 blob exactly once, making the second size-4 blob.
3. Merge two type-1 enemies into a type-2 enemy.
4. Move both size-4 blobs and the type-2 enemy into any three of the four goal bubbles.

The third type-1 enemy is surplus. It may be used to create the downgrade or kept away as a hazard. It should not be allowed to merge with the type-2 enemy, because different enemy types overlapping causes an illegal-contact undo, and creating a type-3 enemy would no longer satisfy the type-2 target.

Important reconstruction details:

- Enemy merge checking is global. Two unselected type-1 ships that overlap during a click will merge.
- Newly merged enemies pause briefly before chasing, giving a short window to park or steer them.
- The extra fourth goal bubble is unused; it exists to give room for the three required parked objects.

## Level 9

```text
Step budget: 48
Targets: one friendly size-4 blob, one friendly size-2 blob, and one type-3 enemy ship
Goals: (7,37), (49,51), (7,51)
Friendly starts:
  size 1 at (18,46), (23,52)
  size 5 at (35,48)
Enemies:
  type 1 at (51,13), (14,12), (15,22), (54,33)
```

The final level requires both friendly and enemy composition. The two size-1 blobs must merge into the required size-2 blob. The size-5 blob must be downgraded once to become the required size-4 blob. The four type-1 enemies must be combined through two enemy-merge stages: first make two type-2 enemies, then merge those two type-2 enemies into one type-3 enemy.

To win:

1. Merge the two size-1 blobs into a size-2 blob and protect it.
2. Use exactly one enemy hit on the size-5 blob to produce a size-4 blob.
3. Merge the four type-1 enemies into a single type-3 enemy:
   first pair type 1 with type 1 twice, then pair type 2 with type 2 once.
4. Park the size-2 blob, the size-4 blob, and the type-3 enemy in the three goal bubbles.

The danger is timing the enemy upgrades. Type-3 enemies chase twice as fast as type-1 and type-2 enemies. If the type-3 ship is created before the size-2 and size-4 friendly targets are ready or protected, it can quickly over-downgrade or remove them. Conversely, waiting too long can leave the enemy ships spread out and hard to merge within the 48-step budget.

Important reconstruction details:

- The enemy target count is exact: one type-3 enemy is required in a goal. A type-2 enemy in a goal does not satisfy it.
- The size-5 blob has enough mass to overshoot the target if left alone; it needs one downgrade, not a friendly merge.
- If a second enemy hit lands on the size-4 target, it becomes size 3 and no longer satisfies the level.
- The small size-2 target is fragile. Any enemy hit downgrades it to size 1, and another can remove its components from the intended solution.
