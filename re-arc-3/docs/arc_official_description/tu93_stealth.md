# Stealth

## Core Game

Stealth is a discrete graph-maze game. The visible maze is drawn as 3x3 square stopping nodes joined by 3x3 connector bars. Adjacent stopping nodes are six render pixels apart, so one direction input looks like a two-cell jump: the player slides one pixel at a time across the connector until centered on the next node.

The player is the blue 3x3 character. The goal is the green 3x3 square. Reaching the goal advances to the next level. If the player is destroyed, or the step meter reaches zero, the run fails. There are no lives or partial checkpoints inside a level.

Each directional input costs one step, including attempts into a wall or missing connector. A blocked input leaves every sprite in place and does not advance enemies. A valid input resolves in this order:

1. Decrement the step meter.
2. Rotate the player to face the input direction and move the player one graph edge.
3. If the player lands on any enemy or collectible enemy, that enemy explodes and is removed.
4. Trigger any ambushers that now have the player directly in front of them.
5. Move every red patroller one graph edge.
6. Move every active chaser one graph edge.
7. If any moving enemy lands on the player, the player explodes and is removed.
8. Reverse red patrollers whose next edge would hit a wall.
9. Activate or update chasers.
10. If the surviving player is on the goal, advance.

Explosions are stateful but purely visual: a 3x3 actor becomes a 5x5 burst, then a 7x7 burst, then disappears. The actor's original color is preserved in the burst. When the player causes the collision, the enemy explodes. When a moving enemy causes the collision, the player explodes.

## Actors

Player: blue 3x3 character. Its facing changes to the most recent valid move direction.

Goal: green 3x3 square. The level is solved when the player finishes an action on it.

Ambusher: usually orange, with a bright eye; one optional clone is blue but uses the same collision group. An ambusher is stationary until the player ends a move exactly one graph node in the ambusher's facing direction. It then changes its eye color and lunges one node forward. If that node contains the player, the player is destroyed. To remove an ambusher, step onto its own node from the side or rear before standing in its front node.

Red patroller: red 3x3 enemy. A red patroller moves one graph edge after every valid player move. It keeps its current direction until, after arriving at a node, the next edge ahead is not walkable; then it reverses direction. Red patrollers can be removed by stepping onto their current node, but if they move onto the player first, the player is destroyed.

Chaser: purple/brown 3x3 enemy. It starts inactive. If the player ends a move two graph nodes directly in front of the chaser, the chaser activates. Once active, it moves one graph edge after each valid player move and follows a delayed queue of the player's later directions. The delay lets the player lead it around loops, but standing still in its path or doubling back too soon is fatal.

## Map Legend

The maps below are logical graph maps. Coordinates are zero-based graph-node coordinates `(x, y)` with `(0, 0)` at the top-left node of that level's map. A blank means there is no node or connector.

```text
. empty stopping node
- horizontal connector
| vertical connector
P player start
G goal
A ambusher or collectible clone
R red patroller
C chaser
```

Directions in object lists use `up`, `down`, `left`, and `right`.

## Level 1

```text
Steps: 50
Player: (0, 0), facing right
Goal: (5, 5)
Enemies: none
```

Layout:

```text
P-. .-.-.
  | |   |
  . .-.-.-.
  | | |   |
  .-. .   .
      |   |
.-.-.-. .-.
| |
.-.-.-. .-.
  |   | | |
.-.   .-. G
```

This is the movement tutorial. The only task is to navigate the graph from the upper-left start to the lower-right goal. The route is intentionally indirect, forcing the player to recognize that one input moves to the next separated stopping node rather than to the immediately adjacent rendered square.

Failure can only come from spending all 50 steps. Blocked moves consume steps, so a reconstruction should reject movement where there is no connector but still decrement the meter.

Example safe route:

```text
RDDRURDDLLDRRDRURD
```

## Level 2

```text
Steps: 50
Player: (0, 2), facing up
Goal: (6, 0)
Ambusher: (4, 1), facing left
```

Layout:

```text
            G
            |
.-.-.-.-A-.-.
|   | | |
P   .-.-.
```

This level introduces facing-based ambushes. The direct-looking top corridor leads to `(3, 1)`, which is one node directly in front of the ambusher at `(4, 1)`. Ending a move there triggers the ambusher to lunge left into the player.

To win, take the lower loop to `(4, 2)` and move up onto the ambusher's node. Because the player lands on the ambusher itself, the ambusher explodes before it can act. Then continue right to `(6, 1)` and up to the goal.

Example safe route:

```text
URRDRRURRU
```

## Level 3

```text
Steps: 35
Player: (5, 4), facing up
Goal: (2, 4)
Ambushers:
- (2, 1), facing right
- (3, 1), facing down
- (0, 3), facing right
```

Layout:

```text
    .-.-.
    | | |
    A-A-.-.-.
      |     |
.-.-.-.-.-.-.
| |   | | |
A-.   .-.-.
  |       |
  .-G     P
```

The lower route toward the goal is guarded by the left ambusher at `(0, 3)`, while the upper cluster has two adjacent ambushers with different facings. Entering `(4, 1)` is safe, but entering `(3, 2)` below the down-facing ambusher or `(1, 3)` in front of the left ambusher at the wrong time is fatal.

To win, approach the two top ambushers from safe directions: go around the upper-right branch, enter `(2, 1)` from above, then enter `(3, 1)` from the left. After those are removed, route around to `(0, 2)`, move down onto the third ambusher at `(0, 3)`, then use the lower-left bend to reach the goal at `(2, 4)`.

Example safe route:

```text
UURULLULLDRDLLLDRDR
```

## Level 4

```text
Steps: 20
Player: (0, 4), facing right
Goal: (0, 1)
Ambusher: (1, 0), facing down
Red patroller: (3, 1), facing down
```

Layout:

```text
  A-.-.
  | | |
G-.-.-R
      |
      .-.
      | |
      . .
      | |
P-.-.-.-.
```

This level introduces the red patroller. The red enemy moves on the right vertical spine and reverses at the top and bottom. A direct trip across the bottom row collides with it as it descends into the lower-right nodes.

To win, use the lower-left edge as a timing area. Move back and forth until the red patroller has moved away from the lower crossing, then traverse the bottom row, climb the right side, cross the top, and remove the top ambusher from the side. Do not stand at `(1, 1)` while the ambusher at `(1, 0)` is still present, because that is directly in front of it. Once the ambusher is removed, step down and left into the goal.

Example safe route:

```text
RLRLRRRRUULUULLDL
```

## Level 5

```text
Steps: 50
Player: (7, 0), facing left
Goal: (3, 3)
Collectible clone: (7, 5), facing down
Red patrollers:
- (3, 2), facing down
- (2, 3), facing down
- (0, 3), facing right
- (3, 4), facing up
```

Layout:

```text
.-.-.-.-.-.-.-P
|     |
.     .
|     |
.   . R
|   | |
R-.-R-G-.-.-.-.
  | | | |     |
  . . R .     .
  |   | |     |
  .-.-.-.     A
```

The goal is in the center of several crossing red patrol paths. The red enemies can overlap or pass through the goal area on successive turns, so simply following the top path left and then down reaches the center at the wrong time and gets the player hit.

The blue clone at the lower-right dead end is removable like an ambusher, but it is not required to solve the level. It faces down into no useful node, so it mostly demonstrates that stepping onto another self-shaped actor makes that actor explode rather than granting a new player.

To win, synchronize the red patrols. A safe plan is to move left across the top, use the far-left vertical branch to wait, then descend into the lower-left loop and enter the goal from the right after the central red enemies have moved away.

Example safe route:

```text
LLLRLLLLLDUDUDUDUDDDRDDRRRUUL
```

## Level 6

```text
Steps: 60
Player: (6, 1), facing left
Goal: (0, 1)
Red patrollers:
- (3, 2), facing up
- (6, 5), facing left
Ambushers:
- (2, 3), facing down
- (5, 3), facing down
- (2, 5), facing up
- (3, 5), facing left
- (3, 6), facing up
- (2, 6), facing up
```

Layout:

```text
      .
      |
G-.-.-.-.-.-P
  |   | |
  .   R .
  |     |
  . A-.-.-A
  | |   | |
  .-.-.-.-.
    |     |
.-.-A-A-.-.-R
    | | |
    A-A-.
```

The start and goal share the top corridor, but the top red patroller makes a straight sprint unsafe. The lower half is a timed combat puzzle: several ambushers are arranged so that removing one from the wrong side leaves the player in the front cell of another, and the bottom red patroller sweeps through the same row.

To win, first time the two red patrollers from the upper-right area. Then enter the central lower cluster, remove the ambusher at `(2, 3)` from the right, remove `(5, 3)` from the left, and clear the bottom cluster in an order that never ends the player in front of a remaining ambusher. A safe order after entering the bottom is `(3, 6)`, `(3, 5)`, `(2, 6)`, `(2, 5)`. After the cluster is clear and the patrols are out of phase with the route, climb the left side and reach the goal.

Example safe route:

```text
LLDUDUDDLLRRRDDLDLUDLUULUUUL
```

## Level 7

```text
Steps: 30
Player: (0, 2), facing right
Goal: (6, 2)
Chaser: (2, 0), facing down
Ambusher: (5, 1), facing left
```

Layout:

```text
    C   .-.-.
    |   | | |
    .   .-A .
    |   | | |
P-.-.-. . . G
| |   | | |
.-.   . .-.
      | |
      .-.
```

The chaser is inactive at first. Moving to `(2, 2)` puts the player exactly two nodes below it, so the chaser wakes up. From then on it follows the player's route with a delay.

To win, deliberately wake the chaser, lead it down the left-middle spine and around the lower loop, then cross to the right while it is lagging behind. The ambusher at `(5, 1)` faces left, so avoid `(4, 1)` while it is alive. Remove it by moving up onto it from `(5, 2)`, then continue through the upper-right nodes and down to the goal.

Example safe route:

```text
RRRDDRURUUURDD
```

## Level 8

```text
Steps: 50
Player: (0, 5), facing right
Goal: (2, 0)
Ambusher: (1, 3), facing down
Chaser: (4, 1), facing down
```

Layout:

```text
    G-.-.
        |
        C
        |
        .
        |
  A-.-.-.-.
  | |
  . .
  | |
P-.-.
```

The lower-left route is blocked by a down-facing ambusher: moving from `(1, 4)` up to `(1, 3)` is safe because it lands on the ambusher, but standing at `(1, 4)` while it is alive causes it to lunge down. The chaser on the right vertical activates once the player reaches the lower right side below it.

To win, move through the lower path into the right branch to wake the chaser, then use the rectangular lower loop as cover. The chaser follows the trail around the loop, which gives enough delay to remove the ambusher from the right side and then climb the right vertical to the top path. Finish by crossing left to the goal while the chaser remains behind.

Example safe route:

```text
RRUURRLLLDDRUURRUUULL
```

## Level 9

```text
Steps: 50
Player: (4, 4), facing left
Goal: (4, 5)
Chaser: (3, 2), facing down
Red patrollers:
- (3, 1), facing left
- (6, 1), facing left
- (6, 2), facing down
Ambushers:
- (3, 5), facing right
- (3, 6), facing up
```

Layout:

```text
      .-.-.-.-.
      |     |
.-.-.-R-. .-R-.
      |   | |
    .-C   .-R-.
    | |     | |
    . .     .-.
    | |     |
    .-.-P .-.
          |
      A-G-.
      | | |
      A-.-.
```

This is the combined final level. The goal is near the start, but the player cannot go directly to it. The left ambusher at `(3, 5)` faces right into the goal node `(4, 5)`, so entering the goal while that ambusher remains alive causes it to lunge into the player. The second ambusher at `(3, 6)` guards the lower approach.

The first move toward the left activates the chaser, and the upper routes contain three red patrollers. To win, lead the chaser through the upper-left route, cross the upper-right patrol area only when the red enemies are out of the way, then descend the right side and circle around the bottom. Before finishing, remove `(3, 6)` from the side and then `(3, 5)` from below. Once both ambushers are gone, step right into the goal.

Example safe route:

```text
LLUURDDLUURUURRRDDRDLDLDDLLUR
```
