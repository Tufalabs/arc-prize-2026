# WA30 - Warehouse

## Core Game

Warehouse is a 4-direction grid puzzle about moving small crates onto blue target pads. The board is a 64x64 render grid divided into 16x16 logical cells; every normal actor, crate, and wall tile occupies one 4x4 cell and all movement is one cell at a time. The player is the green 4x4 worker with a white top edge. Crates are 4x4 black-framed squares with a blue center. Target pads and blue steal zones are traversable floor unless occupied by a crate or actor. A level is solved when every crate's top-left cell is inside any target-pad footprint and no crate is currently being carried.

The five actions are up, down, left, right, and the pickup/drop/neutralize action. Every action costs one step from the level's step bar, including blocked movement, pickup, drop, and neutralizing an enemy. If the step counter reaches zero before the win check succeeds, the level is lost and a reset restores the initial arrangement, step counter, removed enemies, and all carried-crate relationships.

The player has a facing direction. Directional movement changes the facing direction when the player is not carrying a crate, then attempts to move. The pickup/drop action uses the current facing direction:

- If the player is not carrying a crate and a crate is directly in front of the player, the crate becomes attached to the player.
- If the player is carrying a crate, the action drops it in its current adjacent cell.
- If the player is not carrying a crate and an adversary is directly in front of the player, the adversary is removed from the level.

When the player carries a crate, the crate keeps the same relative offset from the player. Directional movement moves both the player and the crate together if the player's destination and the crate's destination are clear. The player does not rotate while carrying, so dropping preserves the carried side. A carried crate can be pulled away from a wall or delivered to a pad, but a move is rejected if either the player or carried crate would collide with a solid actor or wall. Dashed wall cells block actors, but crates can be staged on those blocked cells as transfer shelves in the wall-transfer levels.

Helpers are orange 4x4 workers. Helpers are processed after each player action in level order. A helper does not need a facing direction: if it is adjacent to an unplaced crate, it attaches that crate. If that attachment happens, the helper phase ends for that tick. Otherwise, non-carrying helpers use shortest-path movement toward an empty cell adjacent to the nearest available crate. If a helper is carrying a crate, it uses shortest-path movement toward any carrier position that would put its attached crate onto a target pad. On the next helper update after the attached crate is already on a target pad, the helper releases it. The player must often move out of the helper's path or place crates where the helper can reach them.

Adversaries are purple 4x4 workers. They are processed after the helper phase in level order, using the same attachment and one-step movement model, but route crates to blue steal zones instead of target pads. If an adversary is adjacent to a crate, it can take it, including crates being carried by the player or a helper. If a non-carrying adversary attaches a crate, the adversary phase ends for that tick. If an adversary's carried crate reaches a steal zone, the adversary releases it there. The pickup/drop action neutralizes an adversary only when the player is facing the adversary. If the adversary was carrying a crate, neutralizing it leaves the crate at its current cell.

Crates and adversaries change border color as feedback. A loose crate is dark-framed; a crate directly in front of the player is highlighted; a crate carried by the player has a black border; a helper-carried or adversary-carried crate has a different border. An adversary is also highlighted when it is directly in front of the player.

## Reconstruction Model

Use these state fields for a faithful rebuild:

- `player_cell` and `player_facing`
- `remaining_steps`
- set of crate cells
- `carrier_by_crate` and `carried_crate_by_actor`
- helper cells
- adversary cells and removed flags
- static wall cells and dashed blocker cells
- target-pad cell footprints
- steal-zone cell footprints

On each action:

1. Spend one step.
2. Apply the player action, moving the player, moving a carried crate, attaching a faced crate, dropping the carried crate, or removing a faced adversary.
3. Process helpers in level order: release on target, otherwise attach an adjacent crate and end the helper phase, otherwise move one shortest-path step toward a useful crate or target delivery position.
4. Process adversaries in level order: release in a steal zone, otherwise attach or steal an adjacent crate and end the adversary phase, otherwise move one shortest-path step toward a useful crate or steal-zone delivery position.
5. Refresh crate/adversary highlight colors.
6. If all crates are on target pads and none are carried, advance to the next level.
7. If the step counter is zero and the level was not solved, fail the level.

Pathfinding for helpers and adversaries is ordinary four-neighbor shortest-path search over logical cells. The moving actor cannot enter solid wall cells, dashed blocker cells, occupied cells, or the board boundary. When carrying, the path search also checks that the attached crate's offset cell remains non-colliding.

## Map Legend

The compact maps below use 16x16 logical cells. Coordinates are shown as `column,row`, with `column = x / 4` and `row = y / 4`.

```text
. floor
P player start
H helper start
E adversary start
B crate start
T target pad footprint
Z blue steal-zone footprint
# solid wall or dashed blocker cell
~ visual dark band outside the playable corridor
```

If a crate starts on top of a target or steal-zone cell, the map shows `B`; the level text lists the underlying zone.

## Level 1

```text
Step limit: 200
Crates: 3
Target capacity: 3 cells

cols 0123456789012345
00   ................
01   ................
02   ................
03   ................
04   ................
05   ................
06   ...........B....
07   ....B..TTT......
08   ................
09   ........B.......
10   ................
11   ................
12   ........P.......
13   ................
14   ................
15   ................
```

Level 1 is the basic pickup and placement tutorial. The player starts near the lower center at `(8,12)`. The three crates start at `(11,6)`, `(4,7)`, and `(8,9)`. The only target pad is a horizontal 3-cell strip at row 7, columns 7 through 9.

To win, put all three crates onto the three target cells and drop them. There are no helpers, adversaries, walls, or steal zones. The player can approach a crate, face it, use pickup to attach it, carry it to the target strip, and use pickup/drop again to release it. Movement onto an occupied crate cell is blocked unless the crate is being carried away as part of the player's attached-crate movement.

## Level 2

```text
Step limit: 70
Crates: 5
Target capacity: 6 cells

cols 0123456789012345
00   ................
01   ................
02   ...P............
03   ................
04   ................
05   ..........B.....
06   ............B...
07   ...TT....B......
08   ...TT.......B...
09   ...TT.H.........
10   ...........B....
11   ................
12   ................
13   ................
14   ................
15   ................
```

Level 2 introduces a helper. The player starts at `(3,2)` and the helper starts at `(6,9)`, just to the right of the target area. The five crates start at `(10,5)`, `(12,6)`, `(9,7)`, `(12,8)`, and `(11,10)`. The target pad is a 2x3 rectangle covering columns 3-4 and rows 7-9.

The helper will automatically seek crates and carry them to the target pad. The player can also carry crates directly, but the intended lesson is cooperation: every player action also gives the helper one action, so the player should avoid standing in the helper's shortest path or occupying cells the helper needs for delivery. The target has one spare cell; any five of the six target cells can hold the five crates.

The level is won once all five crates are resting on the target rectangle. There are no walls or adversaries, so failure is only from spending all 70 actions before placement is complete.

## Level 3

```text
Step limit: 100
Crates: 5
Target capacity: 8 cells

cols 0123456789012345
00   ........#.......
01   ........#.......
02   ........#.......
03   ........B...H...
04   ..B.....#.......
05   .....B..#.......
06   ........#....TT.
07   ........#....TT.
08   ........B....TT.
09   ....P...#....TT.
10   ........#.......
11   ...B....#.......
12   ........#.......
13   ........#.......
14   ........#.......
15   ........#.......
```

Level 3 adds a full-height dashed wall at column 8. The player starts on the left side at `(4,9)`, while the helper starts on the right side at `(12,3)`. The target is on the right side, a 2x4 rectangle covering columns 13-14 and rows 6-9. Crates start at `(2,4)`, `(5,5)`, `(3,11)`, `(8,3)`, and `(8,8)`.

The wall blocks workers from crossing, so the player cannot personally enter the right-side target area. The important mechanic is that crates can be staged on the wall as transfer objects. The player can pick up a crate from the left, carry it next to the wall, and drop or leave it on column 8. The helper can then pick up that same crate from the right side and deliver it to the target. Two crates already begin on the wall column, making this transfer rule visible immediately.

To win, all five crates must end on the right-side target rectangle. The player should feed crates to the wall and keep enough clearance for the helper to take them. Because helper and player turns are synchronous, wasting player actions while blocking the helper can consume the 100-step budget without progress.

## Level 4

```text
Step limit: 100
Crates: 7
Target capacity: 7 cells

cols 0123456789012345
00   .......#........
01   ......B#......H.
02   .......#........
03   ..H....#........
04   .......#........
05   .....######...T.
06   .T...#B.BB#.....
07   .T...#....#..T..
08   .....#.P..#.....
09   ..T..#..B.#.....
10   .....#B..B#.....
11   .....######.....
12   ....#......#....
13   ...#........#...
14   ..#...H..TT..#..
15   .#............#.
```

Level 4 has three helpers and a central crate pen. The player starts inside the pen at `(7,8)`. Helpers start at `(14,1)`, `(2,3)`, and `(6,14)`. Seven crates start at `(6,1)`, `(6,6)`, `(8,6)`, `(9,6)`, `(8,9)`, `(6,10)`, and `(9,10)`.

The target pads together have exactly seven cells:

- Left vertical target at `(1,6)` and `(1,7)`.
- Left single target at `(2,9)`.
- Right single targets at `(13,7)` and `(14,5)`.
- Bottom horizontal target at `(9,14)` and `(10,14)`.

The dashed rectangle around the player is a transfer pen. The player can move inside it and carry crates to the pen boundary. Helpers outside the pen can collect crates placed where they can reach them and deliver them to their local targets. A solid vertical wall at column 7, rows 0-4, separates the upper helpers, and black diagonal wall cells at the bottom form side barriers around the lower helper's route.

To win, the player must distribute crates so the helpers can fill all seven target cells. Since target capacity exactly equals crate count, every crate matters. Dropping too many crates near one helper leaves another target uncovered, and standing on a transfer or path cell can stop a helper from completing its route before the 100-step limit.

## Level 5

```text
Step limit: 125
Crates: 6
Target capacity: 8 cells

cols 0123456789012345
00   .........#......
01   .........#..B...
02   .........#....B.
03   .........#......
04   .........#......
05   .........#......
06   ..TT..####......
07   ..TT.H.....B....
08   ..TT............
09   ..TT..####.P....
10   .........#......
11   .........#......
12   .........#...B..
13   .........#.....B
14   .........#.B....
15   .........#......
```

Level 5 is a cooperation and efficiency test. The player starts on the right side at `(11,9)` and the helper starts near the target at `(5,7)`. The target is a 2x4 rectangle at columns 2-3 and rows 6-9. Six crates start on the right half at `(12,1)`, `(14,2)`, `(11,7)`, `(13,12)`, `(15,13)`, and `(11,14)`.

The wall structure creates a divided warehouse. Column 9 is mostly blocked, with a narrow two-cell passage at rows 7-8. Additional wall bars at rows 6 and 9, columns 6-9, make the passage easy to clog. The player must ferry crates from the right side toward the helper and target side, while the helper routes them into the target rectangle.

The target has two extra cells, but the action budget is tighter than the open tutorial levels. The intended solution is not just to carry every crate personally; it is to keep the helper supplied and avoid blocking the doorway. The level ends when all six crates are resting on the target rectangle.

## Level 6

```text
Step limit: 75
Crates: 2
Target capacity: 4 cells
Steal-zone capacity: 4 cells

cols 0123456789012345
00   ...........##...
01   ...........##...
02   ...........##...
03   .......TT..##...
04   ....E..BT..##...
05   ...........##...
06   .............ZZ.
07   ...........##ZB.
08   ...........##...
09   ...........##...
10   ...........##...
11   ...........##...
12   ...........##...
13   .....P.....##...
14   ...........##...
15   ...........##...
```

Level 6 introduces an adversary. The player starts at `(5,13)`. The adversary starts at `(4,4)`. There are two crates: one starts at `(7,4)`, already inside the 2x2 target pad at columns 7-8 and rows 3-4, and the other starts at `(14,7)`, inside the 2x2 steal zone at columns 13-14 and rows 6-7.

The right side is blocked by a two-column vertical wall at columns 11-12, with a gap around row 6 connecting the central area to the steal zone. The adversary tries to take crates to the steal zone. Since one crate begins on the target and one begins in the steal zone, the player must both protect/recover crates and remove or outrun the adversary.

To neutralize the adversary, stand next to it, face it, and use the pickup/drop action. If the adversary carries a crate, the crate remains at its current position after the adversary is removed. To win, both crates must be on the target pad and uncarried before the 75-step counter expires.

## Level 7

```text
Step limit: 125
Crates: 2
Target capacity: 2 cells
Steal-zone capacity: 2 cells

cols 0123456789012345
00   ~~~~~~~~~~~~~~~~
01   ~~~~~~~~~~~~~~~~
02   ~~~~~~~~~~~~~~~~
03   ~~~~~~~~~~~~~~~~
04   ################
05   ................
06   ........B.......
07   ...T........Z...
08   ...T.P.B..E.Z...
09   ................
10   ################
11   ~~~~~~~~~~~~~~~~
12   ~~~~~~~~~~~~~~~~
13   ~~~~~~~~~~~~~~~~
14   ~~~~~~~~~~~~~~~~
15   ~~~~~~~~~~~~~~~~
```

Level 7 is a narrow horizontal corridor. Rows 4 and 10 are solid walls, and the dark bands above and below are visual padding outside the useful corridor. The player starts at `(5,8)`. The adversary starts to the right at `(10,8)`. Crates start at `(8,6)` and `(7,8)`. The target is a 1x2 vertical pad at column 3, rows 7-8. The steal zone is a matching 1x2 vertical pad at column 12, rows 7-8.

This level demonstrates that the adversary is active rather than static. After each player action, it moves toward crates and tries to carry them to the right-side steal zone. The closest crate is near the middle, so hesitation lets the adversary take control quickly. The player can either neutralize the adversary by facing it and using the action, or move and place crates efficiently enough that the adversary never completes a steal.

The win condition is exact: both crates must occupy the two target cells on the left and must be dropped. Because the target has no spare capacity, the final layout must fill both target cells.

## Level 8

```text
Step limit: 150
Crates: 13
Target capacity: 17 cells
Steal-zone capacity: 18 cells

cols 0123456789012345
00   ................
01   .......H........
02   .BBZ...B.B.TTTT.
03   .BBZ..B.B..TTTT.
04   .ZZZ....E.......
05   ................
06   ####..##########
07   ................
08   .P..............
09   #########..#####
10   ................
11   ......B..H......
12   ...ZZZ...B..TTT.
13   ...ZZZ.BB.B.TTT.
14   ...ZZZ..E...TTT.
15   ................
```

Level 8 has two work areas, two helpers, and two adversaries. The player starts in the middle-left lane at `(1,8)`. Helpers start at `(7,1)` in the upper section and `(9,11)` in the lower section. Adversaries start at `(8,4)` and `(8,14)`.

The target pads are on the right:

- Upper target: 4x2 rectangle at columns 11-14, rows 2-3.
- Lower target: 3x3 rectangle at columns 12-14, rows 12-14.

The steal zones are on the left:

- Upper steal zone: 3x3 rectangle at columns 1-3, rows 2-4. Four crates start inside or on this zone.
- Lower steal zone: 3x3 rectangle at columns 3-5, rows 12-14.

There are 13 crates total. They start at `(1,2)`, `(2,2)`, `(7,2)`, `(9,2)`, `(1,3)`, `(2,3)`, `(6,3)`, `(8,3)`, `(6,11)`, `(9,12)`, `(7,13)`, `(8,13)`, and `(10,13)`.

Two horizontal wall bands divide the map into upper, middle, and lower lanes. Row 6 is blocked except for the gap at columns 4-5. Row 9 is blocked except for the gap at columns 9-10. These gaps are the only efficient vertical transitions. The upper and lower helpers can solve their local target areas if crates are not stolen and if the player keeps lanes open.

The intended task is to neutralize or outmanage both adversaries in their separate sections, then allow the helpers and player to move crates to the right-side target pads. The upper adversary threatens crates near the upper steal zone; the lower adversary threatens the lower group. Because the step limit is 150 and many crates are already near danger zones, spending too long in one section lets the other adversary undo progress.

## Level 9

```text
Step limit: 70
Crates: 9
Target capacity: 13 cells
Steal-zone capacity: 4 cells

cols 0123456789012345
00   .........#......
01   .........#.H....
02   .........#...TT.
03   ..B..TTT.#######
04   .....TTT.#......
05   .B.B.TTT.#.B....
06   .........#...TT.
07   .BB.H....#..B...
08   .BZ.....P.....B.
09   ................
10   ..#.############
11   ..#.#...#...#...
12   ..#.#.#.#.#.#.#.
13   ..#.#.#.#.#.#.#.
14   ..#.#.#.#.#.#.#E
15   ..#...#...#...#.
```

Level 9 is the final timing puzzle. The player starts at `(8,8)`. Helpers start at `(4,7)` on the left and `(11,1)` in the upper-right area. The adversary starts deep in the lower-right maze at `(15,14)`. The nine crates start at `(2,3)`, `(1,5)`, `(3,5)`, `(1,7)`, `(2,7)`, `(12,7)`, `(1,8)`, `(14,8)`, and `(11,5)`.

The target pads together have 13 cells for the nine crates:

- Main target: 3x3 rectangle at columns 5-7, rows 3-5.
- Upper-right target: 2x1 strip at columns 13-14, row 2.
- Mid-right target: 2x1 strip at columns 13-14, row 6.

The only steal zone is a 2x2 square at columns 1-2, rows 7-8. Several left-side crates begin on or next to this zone, so they are vulnerable if the adversary reaches them later.

The upper board is split by a solid vertical wall at column 9. A dashed barrier across row 3, columns 10-15, shapes the upper-right helper's route. The lower half is a black maze of narrow vertical channels connected at the bottom and top, with the adversary starting near the lower-right end. The player must work quickly enough that the left helper can fill the main target area, the upper-right helper can fill the small right-side target strips, and the adversary is neutralized or prevented from stealing needed crates.

The target pads have spare cells, but the 70-step limit leaves almost no timing slack. A faithful reconstruction should preserve the exact walls, the left steal zone, the lower maze, and the helper/adversary synchronous update order. The level is won only when all nine crates are dropped on target cells.

## Implementation Checklist

- Board size is 64x64 pixels with 16x16 logical cells of size 4.
- Directional movement is one logical cell per action.
- Every action decrements the step counter before the win/loss check.
- The step bar renders along the bottom row of the board.
- Pickup/drop is the fifth action.
- Player-facing matters for pickup, drop, and neutralizing adversaries.
- Helpers and adversaries act after the player on every action.
- Helpers route crates to target pads and release them there.
- Adversaries route crates to steal zones and release them there.
- Adversaries can steal crates from the player or helpers by becoming adjacent to the crate.
- The player neutralizes an adversary only by facing it and using the action.
- A level advances only when every crate is on a target pad and no crate is carried.
- Timeout calls the loss/reset path; it does not award partial progress.
