# LS20 - Locksmith

Official task URL: https://arcprize.org/tasks/ls20

Local implementation: `re_arc/environment_files/ls20/9607627b/ls20.py`

Zip source: `Games/ls20 ...md` plus `Games/ls20/image*.png`

## Core Game

Locksmith is a discrete maze game about carrying a mutable "key" state through a 5x5-cell maze. The player moves the orange/blue body one cell at a time with the four direction actions. The bottom-left HUD shows the current key. Locks are black/framed squares that contain a small target key icon. A lock opens only when the player's current key exactly matches that lock's target shape, color, and rotation, and the player steps onto the lock cell.

The key has three independent state variables:

- Shape: one of six 3x3 masks.
- Color: orange, blue, green, or red, cycling in that order.
- Rotation: 0, 90, 180, or 270 degrees, cycling in that order.

The player changes the key by stepping on modifier tiles:

- White cross-shaped tile: rotate the current key 90 degrees clockwise.
- Four-color square tile: advance the key color in the cycle orange -> blue -> green -> red -> orange.
- White shape-marker tile: advance the key shape in the six-shape cycle.

The player has a 42-step energy bar shown along the bottom of the board. Most movement actions decrement the bar by 1. Yellow square pickups are recharges; stepping on one removes it and restores the energy bar to 42. If the meter expires, the player loses one of three lives, resets to the level start, resets the key to the level's starting state, restores picked-up recharges and opened locks, and resets moving/pushing objects. Losing all three lives fails the level.

There are also white pusher bars. When the player walks into one, it carries the player in the bar's direction until the next wall/lock stop, then the bar retracts to its start position while the player remains at the pushed destination. Some modifier tiles move automatically along invisible tracks once per player step. If the player's attempted move is blocked, those moving modifiers undo that step.

## Key Shape Masks

The code defines six key masks. `X` means a colored cell and `.` means empty.

```text
Shape 0    Shape 1    Shape 2
XX.        .X.        X.X
.XX        .X.        X.X
X.X        XXX        XXX

Shape 3    Shape 4    Shape 5
.XX        .X.        XXX
X.X        XX.        ..X
.X.        .XX        X.X
```

The shape modifier cycles `0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0`. The color modifier cycles `orange -> blue -> green -> red -> orange`. The rotation modifier cycles `0 -> 90 -> 180 -> 270 -> 0`.

## Reconstruction Model

Use a 64x64 render grid with 5x5 gameplay cells. The camera shows a 16x16 view. Most maze cells have top-left positions `x = 4 + 5 * column`, `y = 5 * row`. The player also occupies one 5x5 cell. Gray 5x5 blocks are solid walls.

Level state should include:

- `player_cell`
- `key_shape`
- `key_color`
- `key_rotation`
- `remaining_steps`
- `lives`
- per-lock completion flags
- removed recharge tiles
- moving modifier positions and directions
- pusher bar positions

On each direction action:

1. Move any moving modifier tiles one step along their invisible track.
2. Compute the player's target cell.
3. If the target is a wall, blocked lock, or blocked pusher, reject the move and undo modifier movement.
4. If the target contains a modifier, update key state.
5. If the target contains a recharge, restore the step meter and remove that recharge.
6. If the target contains a lock, compare key state to that lock's target shape/color/rotation. If it matches, remove that lock and its target icon. If it does not match, block the move and flash/fail feedback.
7. If the target contains a pusher, start the pusher animation and carry the player to the pusher's stop cell.
8. If all locks are removed, advance to the next level.
9. Otherwise decrement the step meter unless the move was a recharge move.

## Map Legend

The following compact maps are source-derived from the local implementation. They are approximate logical-cell maps, not pixel-perfect screenshots.

```text
# wall
. open floor
P player start
L lock destination cell
E recharge
R rotation modifier
C color modifier
S shape modifier
> pusher that carries right
< pusher that carries left
^ pusher that carries down
v pusher that carries up
```

## Level 1

Source data:

```text
Start key: shape 5, blue, rotation 270
Target lock: shape 5, blue, rotation 0
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   ############
02   ######L#####
03   ######.#####
04   ######.#####
05   ##........##
06   ##.R.#....##
07   ##...#....##
08   ###.##....##
09   ###...P...##
10   ############
11   ############
```

The level is a tutorial for rotation. The player starts in the lower middle with a blue shape-5 key rotated 270 degrees. The only lock is near the top center. The lock shows the same blue shape, but at rotation 0. The only modifier in the maze is the white rotation cross in the lower-left side of the accessible area.

To win, the player must route to the rotation cross enough times to rotate the carried key from 270 to 0, then walk up the central corridor into the lock. If the player enters the lock before the key orientation matches, movement into the lock is rejected and the lock remains.

Important reconstruction details:

- In level 1 only, the code shows a hint frame when the current key state matches an unopened lock. This teaches that the framed target is now enterable.
- There are no recharges, shape changes, color changes, pushers, or moving modifiers.
- The wall layout intentionally forces the player to detour to the rotation cross before entering the lock corridor.

## Level 2

Source data:

```text
Start key: shape 5, blue, rotation 0
Target lock: shape 5, blue, rotation 270
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   ###.......##
02   #.........##
03   #.E.#..#..##
04   #...#..##..#
05   ##.###..#..#
06   ##.###..#.##
07   ##.##..##.##
08   ##L##P.#...#
09   ########.R.#
10   #######E...#
11   ############
```

This level keeps the same target-shape and target-color idea but introduces energy recharge. The player starts near the center-right of the lower maze. The lock is on the lower-left side. The rotation cross is near the lower-right side. Two yellow recharge pickups sit on side paths: one near the upper-left portion of the maze and one near the lower-right.

To win, the player must rotate the blue shape-5 key from 0 to 270, then return to the lock. The maze is longer than level 1, so the energy meter matters. The recharges are optional only if the route is efficient; for a robust reconstruction they should restore the full step bar and disappear until reset.

Important reconstruction details:

- Recharges do not change the key. They only restore the step counter.
- The lower-right recharge is close to the rotation cross and teaches that recharges can be collected while solving the key state.
- The level has no color or shape modifier; only rotation is needed.

## Level 3

Source data:

```text
Start key: shape 5, orange, rotation 0
Target lock: shape 5, blue, rotation 180
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   #>.....#..^#
02   #.###.##.R.#
03   #.#...E#...#
04   #.#....###.#
05   #.#........#
06   #.#E.......#
07   #.###.####.#
08   #..#...###.#
09   #P.#.C.###.#
10   ####...###L#
11   ############
```

This level introduces color changing and pushers. The player starts in the lower-left. The target lock is in the lower-right. The target key is still shape 5, but it must be blue and rotated 180 degrees. The player starts with the same shape at rotation 0, but in orange.

The color modifier is in the lower middle. Stepping on it advances the key from orange to blue. The rotation cross is in the upper-right. The player must reach it twice to rotate 0 -> 90 -> 180. Yellow recharges are placed mid-route so the player can afford the long trip.

Two pusher bars are visible: one left-side horizontal/rightward pusher and one upper-right downward pusher. These are not static walls. When the player hits a pusher, it carries the player until the next stop. The level uses them as route shortcuts or gates through otherwise inconvenient corridors.

Important reconstruction details:

- Color and rotation are independent: the player can change color first or rotate first, as long as final state is shape 5, blue, rotation 180.
- The color tile is non-blocking as a modifier; entering its cell updates the key and lets movement continue.
- Pushers should animate and then retract, leaving the player at the pushed destination.
- Yellow recharges should be removable and restored only on death/reset.

## Level 4

Source data:

```text
Start key: shape 4, green, rotation 0
Target lock: shape 5, blue, rotation 0
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   #L...##...P#
02   ####.#..####
03   ###E....#..#
04   ##...#>.^..#
05   #..###.v<#.#
06   #..#S#C#...#
07   #>.^.###...#
08   #...<#..<#.#
09   ##.....#...#
10   ####..E##..#
11   ############
```

This level introduces shape changing and combines all three key-state dimensions. The player starts at the upper-right. The lock is at the upper-left. The target key is shape 5, blue, rotation 0. The player starts with shape 4, green, rotation 0.

The shape modifier is in the central-left area and the color modifier is near the center. To reach the target state, the player must step on the shape modifier once to advance shape 4 -> shape 5, and step on the color modifier enough times to cycle green -> red -> orange -> blue. No rotation change is required if the player avoids the rotation modifier.

The maze contains many pushers. They form one-way transit features and also local traps if used in the wrong order. The player must use them to pass narrow partitions and reach the modifier area, then escape back toward the lock.

Important reconstruction details:

- Shape, color, and rotation modifiers can all coexist. The key HUD should update immediately after every modifier contact.
- The target lock should remain blocked until all three dimensions match exactly.
- Pusher direction is part of each bar, not inferred from player approach. A left pusher always carries left, a right pusher always carries right, etc.
- The color modifier's cycle order matters: starting green means three color contacts are needed to reach blue.

## Level 5

Source data:

```text
Start key: shape 4, orange, rotation 0
Target lock: shape 0, red, rotation 180
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   ##...#^.E#L#
02   #E.S.#..##.#
03   ##.......#.#
04   ###.##>..#.#
05   #..<#C.v.#.#
06   #.######.^<#
07   #.R..#.....#
08   #......#>P.#
09   ##E..#####.#
10   ###.......v#
11   ############
```

This level makes the rotation modifier mobile. The player starts in the lower-right region. The lock is in the upper-right. The target key is shape 0, red, rotation 180. The player starts with shape 4, orange, rotation 0.

The shape modifier is near the upper-left-middle route. Because the shape cycle is 4 -> 5 -> 0, the player must step on the shape modifier twice. The color modifier is near the center; because the color cycle is orange -> blue -> green -> red, it must be used three times. The rotation cross starts on a hidden horizontal track in the lower-left area; it moves along that track as the player takes steps. The player must intercept it twice total to rotate 0 -> 90 -> 180.

The map has three recharges and several pushers. These are needed because the route from start to shape/color/rotation modifiers and back to the upper-right lock is long. The moving rotation tile means the solver cannot rely on a static modifier location; it must account for step timing and the tile's patrol.

Important reconstruction details:

- A moving modifier updates position once per player action before the player's movement is resolved.
- If the player's move is blocked, the modifier movement is undone for that action.
- The hidden track is not rendered, but it constrains the modifier's patrol.
- The level is solved only after collecting enough shape, color, and rotation changes; reaching the lock early should block.

## Level 6

Source data:

```text
Start key: shape 0, green, rotation 0
Lock A: shape 5, blue, rotation 90
Lock B: shape 0, red, rotation 180
Energy: 42 steps, 3 lives
Fog: off
```

Layout:

```text
cols 012345678901
00   ############
01   #E.....E#^.#
02   #.S.....#..#
03   #.#####...##
04   #.#...#..<##
05   #.......#..#
06   #.#.C.#.##.#
07   #.#####.##L#
08   #.....R.##.#
09   #E......##.#
10   ##..P..###L#
11   ############
```

This level introduces multiple sequential locks. There are two target locks on the right side, one above the other. Both must be opened before the level advances. The upper lock requires shape 5, blue, rotation 90. The lower lock requires shape 0, red, rotation 180. The player starts near the lower middle with shape 0, green, rotation 0.

The shape modifier, color modifier, and rotation modifier are all present, and some are on hidden tracks. The shape modifier starts in the upper-left and moves along a horizontal track. The color modifier starts in the central-left/middle area and moves around an invisible box-like track. The rotation modifier starts near the lower middle and moves on a horizontal track.

To reconstruct the intended challenge, make the player solve one lock, remove it, then continue with the current key state to solve the remaining lock. The code tracks each lock independently and removes the lock and its target icon once matched. The two lock states are different enough that the player must keep transforming the key after the first success.

Important reconstruction details:

- The locks are independent completion flags. Opening one lock removes only that lock.
- One of the lock frames has an extra tag and hides after completion; this prevents visual clutter around the completed upper/lower lock.
- Because the target states differ, the key state should persist between locks rather than resetting after opening the first lock.
- Moving modifiers and the 42-step meter make route order important.

## Level 7

Source data:

```text
Start key: shape 1, orange, rotation 0
Target lock: shape 0, red, rotation 180
Energy: 42 steps, 3 lives
Fog: on
```

Full source-derived layout:

```text
cols 012345678901
00   ############
01   #E....#E#E.#
02   #...#...#.R#
03   #..P##.##..#
04   #.###E.^#..#
05   #.......#..#
06   #.###.v<#..#
07   #...#.#....#
08   #C.S#.#....#
09   ##E##.###.##
10   #####L##..E#
11   ############
```

This level is the hidden-information version of the previous mechanics. The screenshot only shows a circular area around the player; the rest of the board is blacked out by fog of war. The underlying map still contains a normal 12x12-ish maze with a lock near the lower middle, a color modifier at the lower-left, a shape modifier near it, a rotation modifier on the right side, six recharges, and several pushers.

The target lock is shape 0, red, rotation 180. The player starts as shape 1, orange, rotation 0. To reach the target state, the player must change shape from 1 to 0 by cycling through the remaining shapes, change color from orange to red, and rotate twice. That means this level requires repeated contact with modifiers, not one simple visit to each. The fog makes route memory important because the player cannot see the whole maze and target at once.

The rotation modifier is mobile on a vertical hidden track on the right side. The player must locate it through exploration and time contacts while managing energy. Recharges are numerous but can be consumed only once per life. Pushers in the middle/right area can move the player through narrow passages or strand the player if used without knowing the hidden layout.

Important reconstruction details:

- Fog should be a render-only mask around the player; it should not alter collisions or game logic.
- The current key HUD is still shown even when fog is active.
- The full map exists from the start, but only tiles within about a 20-pixel radius of the player are visible.
- Reconstructing the level requires preserving hidden tracks and pusher behavior, because the visible screenshot alone does not reveal the full route.

## Implementation Checklist

- Player movement is grid-based and uses 5x5-cell steps.
- Only directional actions are available; there is no click or space interaction in LS20.
- Locks require exact `(shape, color, rotation)` equality.
- A failed lock attempt blocks movement and shows feedback; it does not consume or remove the lock.
- Shape/color/rotation modifiers mutate the current key on contact.
- Some modifiers move automatically on invisible tracks.
- Yellow recharge tiles restore the full energy bar and are removed until reset.
- The energy bar starts at 42 on every level.
- The player has three lives per level; step exhaustion consumes a life and resets transient level state.
- Pusher bars carry the player in a fixed direction until the next blocker, then retract without carrying the player back.
- Level 6 requires multiple locks to be removed before calling `next_level()`.
- Level 7 uses fog of war as a rendering mask over the same logical mechanics.
