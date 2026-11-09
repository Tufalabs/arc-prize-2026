# Ghost

## Core Game

Ghost is a discrete route-recording puzzle. The player controls a blue 7x7 body on a 10x10 logical board. Direction actions move one cell at a time on a six-pixel pitch. The goal is the blue outlined target square; the level is solved when the live player reaches the goal cell.

The fifth action, shown as Spacebar, records a ghost route. When Spacebar is pressed after the live player has moved, the current route rewinds all the way back to the start. The body that just moved becomes a pale ghost and keeps the route as a timed list of direction steps. A fresh live player appears at the start. On later live-player moves, each saved ghost attempts the route step with the same index as the live route: after the live player's first step, every ghost takes its first recorded step; after the second step, every ghost takes its second recorded step, and so on. When a ghost has exhausted its route, it remains at its final cell and can keep holding a pressure plate.

The Spacebar rewind is an animated reverse walk, not a silent teleport. During the rewind, the body releases and re-enters plates along the reverse path, gates update, and teleporter controllers can resolve. This matters most for yellow latches, because crossing a yellow plate during the rewind can send another toggle pulse.

Ghost movement is still collision-checked against the current gate state. If a recorded step is blocked by a closed gate, the ghost does not pass through it; this can desynchronize a solution. Timing gates, plates, and teleport triggers against the replay is the main puzzle.

The top-left HUD contains the ghost slots. Level 1 has two slots, so it supports one saved ghost plus the live player. Levels 2-7 have three slots, so they support two saved ghosts plus the live player. Pressing Spacebar after the last slot has been used clears the saved ghosts and returns to the first slot instead of adding another ghost.

The bottom horizontal bar is the time limit. Each non-reset command advances an internal counter; every second command shifts the 64-pixel bar one pixel left. The level fails when the bar has fully moved offscreen, after about 130 commands. A manual reset restores the current level to its initial state, including timer, gates, ghosts, helper agents, and teleport occupancy.

## Objects And Rules

Use a 64x64 render grid. Logical cell `(c, r)` has body top-left position `(1 + 6c, 1 + 6r)`. A movement step changes the body by exactly 6 pixels. The target goal art is a 9x9 square drawn one pixel up and left from the goal cell; reaching the goal means the player body is centered in that square.

Black maze regions are walkable. White/empty regions are outside the floor mask and cannot be entered. Visual wall strips and borders are decorative; the floor mask plus visible gates define collision.

Normal red pressure plates are momentary. When at least one body is on the plate, the connected red gate moves one cell open. When the last body leaves, the gate closes. A closing gate can destroy a body caught in it; active-player destruction fails the level.

Yellow pressure plates are latching toggles. Each entry onto the plate sends an activation pulse. If the connected yellow gate is closed, the pulse opens it; if it is already open, the next pulse closes it. Leaving the plate does not change the yellow gate. Passing over the same yellow plate twice therefore undoes the first toggle.

Purple pressure plates trigger teleporters. A teleporter consists of two purple outlined portal cells and a hidden controller. When its purple plate sends an activation, the controller waits briefly, plays a swap animation, then exchanges all bodies standing on the two portal cells. If only one endpoint is occupied, that body is moved to the other endpoint. Players, ghosts, and helper agents can all be teleported.

Green helper agents appear in levels 6 and 7. A helper moves automatically one cell after each successful live-player move. It follows a hidden track, initially trying to continue in its current direction, then turning if the next track cell is blocked or missing. It obeys gates, can press plates, and can be teleported. The live player should avoid overlapping the helper.

Gate opening directions are part of the gate. A gate opened by a plate moves one cell in its assigned direction and leaves its original cell passable. In the maps below, the text after each gate gives that opening direction.

## Map Legend

The maps are logical cell maps.

```text
#  outside the walkable floor
.  walkable floor
P  live player start
G  goal cell
A  green helper start
r  red momentary pressure plate
y  yellow latching pressure plate
p  purple teleporter trigger plate
D  red momentary gate
Y  yellow latching gate
T  purple portal endpoint
```

## Level 1

Layout:

```text
cols 0123456789
00   ##########
01   ##P...r###
02   ##.#.#####
03   ##...#####
04   ##.#######
05   ##.#######
06   ##D#######
07   ##.#######
08   ##.....G##
09   ##########
```

Objects and wiring:

- Start: `(2,1)`.
- Goal: `(7,8)`.
- Red plate `(6,1)` controls red gate `(2,6)`.
- Red gate `(2,6)` opens right.
- One saved ghost is available.

This is the route-recording tutorial. The start, the red plate, the gate corridor, and the goal are all on the same connected black floor, but the gate blocks the vertical route to the lower passage. The live player can reach the red plate from the start, but cannot hold it while also crossing the gate.

To win, record a short route from the start to the red plate, then press Spacebar. The route rewinds, leaving a ghost that will walk to `(6,1)` and remain there. On the next run, move the live player down the left vertical corridor. By the time the live player reaches the gate, the ghost is on the plate and the gate has moved right, opening the passage to the lower horizontal corridor and the goal.

Important reconstruction details:

- The ghost does not appear instantly at the plate. It replays the recorded route one step at a time, so the live player route is timed against the replay.
- The red gate recloses if the ghost leaves the plate. In the intended solution the ghost stops on the plate, keeping it open.
- The goal check happens only for the live player, not for the ghost.

## Level 2

Layout:

```text
cols 0123456789
00   ##########
01   #........#
02   #.####.#.#
03   #.D.G#.#.#
04   ######r.P#
05   ########.#
06   #.r..###.#
07   ####.###.#
08   #.....D..#
09   ##########
```

Objects and wiring:

- Start: `(8,4)`.
- Goal: `(4,3)`.
- Red plate `(6,4)` controls red gate `(6,8)`, which opens up.
- Red plate `(2,6)` controls red gate `(2,3)`, which opens down.
- Two saved ghosts are available.

This level requires a chain of recorded helpers. The right-side start area can reach the first red plate at `(6,4)`. That plate opens the lower gate at `(6,8)`, allowing the route through the lower corridor toward the left side of the maze. The left-side red plate at `(2,6)` opens the upper-left gate at `(2,3)`, which gives access to the goal chamber.

To win, first record a ghost that reaches and holds the right red plate. On the next run, use that open lower gate to reach the left-side red plate and record a second ghost that holds it. On the final run, both ghost paths replay: the first opens the lower route, the second opens the goal-side gate, and the live player can move to the goal.

Important reconstruction details:

- Both red gates are momentary. Each must be held by a body at the time it is needed.
- The two ghost paths replay from the same global step counter. Their route lengths must leave them on their plates before the live player arrives at the corresponding gate.
- If the second ghost cannot reach `(2,6)` because the first replay was mistimed, the upper-left gate never opens.

## Level 3

Layout:

```text
cols 0123456789
00   ##########
01   #........#
02   #.###.##.#
03   #P#G#y##Y#
04   ###.#.##.#
05   #..D#.r#.#
06   #.######.#
07   #.######.#
08   #..r..D..#
09   ##########
```

Objects and wiring:

- Start: `(1,3)`.
- Goal: `(3,3)`.
- Red plate `(3,8)` controls red gate `(3,5)`, which opens down.
- Red plate `(6,5)` controls red gate `(6,8)`, which opens up.
- Yellow plate `(5,3)` controls yellow gate `(8,3)`, which opens left.
- Two saved ghosts are available.

This level adds the yellow latch. The red gates still behave like momentary pressure gates, but the yellow gate is toggled by pulses from the yellow plate. A single entry onto `(5,3)` opens the yellow gate; a second entry closes it again.

The goal chamber is near the start but separated by walls and the red gate below it. The lower and right corridors require coordinated access through the red gates. A replay route can hold the lower-left red plate to open the gate below the goal, and another route can hold the mid-right red plate to open the lower-right gate. The yellow plate controls access across the upper-right side and must be crossed the correct number of times.

To win, arrange the saved route timing so the needed red gates are held open while the live player traverses the maze, and toggle the yellow gate exactly once when it must be open. Do not route a ghost or the player over the yellow plate a second time unless closing the yellow gate is intended.

Important reconstruction details:

- Yellow release events do nothing. Only entering the yellow plate toggles the gate.
- The yellow gate's current state persists until the next yellow-plate entry, even if the body that toggled it walks away.
- Red gates still close when their plates are empty, so their plates generally need parked ghosts.

## Level 4

Layout:

```text
cols 0123456789
00   ##########
01   #...P#...#
02   #.##.#.#.#
03   #.##...#.#
04   #D###r##p#
05   #.########
06   #...T#####
07   ##########
08   #G..T#####
09   ##########
```

Objects and wiring:

- Start: `(4,1)`.
- Goal: `(1,8)`.
- Red plate `(5,4)` controls red gate `(1,4)`, which opens right.
- Purple plate `(8,4)` triggers the portal pair `(4,6) <-> (4,8)`.
- Two saved ghosts are available.

This level introduces purple teleport timing. The two portals are vertically aligned in the lower middle of the maze. Pressing the purple plate does not simply open a door; after a short animation it swaps the bodies on the two portal endpoints. A ghost can therefore trigger the purple plate while the live player waits on one of the portals.

The red gate on the left controls the long vertical approach toward the lower-left goal area. The red plate is in the middle of the map and can be held by a ghost. The purple plate is on the far-right side and is used to move a body between the two portal cells, bypassing the separated lower regions.

To win, record a route that causes a ghost to reach the purple plate at the same step that the live player is standing on the portal endpoint that should be moved. A second route or the live player can handle the red plate, depending on timing. The final live player must end up in the lower-left region with the red gate open long enough to reach the goal.

Important reconstruction details:

- Teleporting swaps endpoint occupants; it is not a one-way warp.
- A purple trigger affects only bodies currently on its paired portal endpoints when the swap resolves.
- The trigger delay matters. The live player must be on the portal when the swap animation completes, not merely when the ghost begins walking toward the purple plate.

## Level 5

Layout:

```text
cols 0123456789
00   ##########
01   y##Y......
02   P##.##.##.
03   ....##.##.
04   .##y##Y..p
05   ....##.###
06   ###r##.T##
07   #G########
08   #.####.T##
09   #..D...###
```

Objects and wiring:

- Start: `(0,2)`.
- Goal: `(1,7)`.
- Yellow plate `(0,1)` controls yellow gate `(3,1)`, which opens left.
- Yellow plate `(3,4)` controls yellow gate `(6,4)`, which opens left.
- Red plate `(3,6)` controls red gate `(3,9)`, which opens up.
- Purple plate `(9,4)` triggers the portal pair `(7,6) <-> (7,8)`.
- Two saved ghosts are available.

This is a denser timing version of level 4. There are two separate yellow latch circuits, a red momentary gate in the lower passage, and a right-side purple teleporter. The start is on the left side, the purple trigger is far to the right, and the goal is in a lower-left pocket.

The two yellow gates must be toggled open without accidentally being toggled shut again by a repeated pass. The red gate near the bottom requires a body on the red plate at the time the lower corridor is used. The purple trigger must be timed so the live player or a recorded body is on the portal endpoint that should be swapped.

To win, use recorded routes to set the yellow gates, hold the red plate, and trigger the purple teleporter at the needed step. The live player then uses the opened/toggled route and the portal pair to reach the goal-side pocket.

Important reconstruction details:

- Both yellow gates are independent latches. Their state is not reset by leaving the yellow plates.
- The red gate is independent of the latches and closes as soon as `(3,6)` is empty.
- The portal pair is vertical: a body on `(7,6)` moves to `(7,8)` and a body on `(7,8)` moves to `(7,6)` when the purple trigger resolves.
- Because only two saved ghosts are available, a faithful puzzle requires one route to do multiple jobs by walking over several plates in the right order.

## Level 6

Layout:

```text
cols 0123456789
00   ##########
01   y..y.D.D..
02   #########.
03   #########A
04   Y..Y#r.r##
05   .##.###..P
06   .#......#.
07   .####.###.
08   .#y##Y.G#.
09   ...#######
```

Objects and wiring:

- Start: `(9,5)`.
- Helper start: `(9,3)`.
- Goal: `(7,8)`.
- Helper hidden track: row 1 from `(0,1)` through `(9,1)`, plus column 9 at `(9,1)`, `(9,2)`, `(9,3)`. The helper starts at the bottom of this track and first moves upward.
- Red plate `(7,4)` controls red gate `(7,1)`, which opens down.
- Red plate `(5,4)` controls red gate `(5,1)`, which opens down.
- Yellow plate `(3,1)` controls yellow gate `(3,4)`, which opens up.
- Yellow plate `(0,1)` controls yellow gate `(0,4)`, which opens up.
- Yellow plate `(2,8)` controls yellow gate `(5,8)`, which opens left.
- Two saved ghosts are available.

This level adds the green helper. The helper moves automatically after each successful live-player move, following its hidden track. It starts at `(9,3)`, climbs to the top row, and then tries to travel left. The red gates at `(7,1)` and `(5,1)` block the helper's top-row track unless the player or ghosts stand on the corresponding red plates.

The helper is needed because it can reach the top-row yellow plates. When it steps onto `(3,1)` or `(0,1)`, it toggles the yellow gates in the lower-left area. The player starts on the right and must open gates for the helper, while the helper's movement opens or closes gates that affect the player's later route. The lower yellow plate `(2,8)` controls the final yellow gate near the goal.

To win, coordinate the live player, ghosts, and helper. The player must time red-plate holds so the helper can pass the top-row red gates. The helper then toggles the lower-left yellow gates. The live player must also handle the bottom yellow gate `(5,8)` so the goal cell becomes reachable.

Important reconstruction details:

- Helper motion occurs only after a successful live-player move. Failed player moves consume time but do not advance the helper route.
- The helper is blocked by visible gates just like the player and ghosts. If the red gates are not open at the right steps, the helper turns around instead of passing through.
- The helper's yellow-plate contacts are pulses, so if it crosses the same yellow plate again on the return trip it toggles that gate back.
- Spacebar rewind also rewinds the live route and recorded moving bodies according to their stored movement history, so helper timing can be part of a recorded setup.

## Level 7

Layout:

```text
cols 0123456789
00   ##r##D..p#
01   ##.##.####
02   ##T#.....#
03   ######.#.#
04   p#T.P#.#.#
05   .###.#.#.#
06   Y##y...#T#
07   .##.######
08   .####G..T#
09   ...A######
```

Objects and wiring:

- Start: `(4,4)`.
- Helper start: `(3,9)`.
- Goal: `(5,8)`.
- Helper hidden track: row 9 from `(3,9)` left to `(0,9)`, then column 0 upward through `(0,4)`. The helper first moves left.
- Red plate `(2,0)` controls red gate `(5,0)`, which opens left.
- Yellow plate `(3,6)` controls yellow gate `(0,6)`, which opens right.
- Purple plate `(0,4)` triggers the left portal pair `(2,2) <-> (2,4)`.
- Purple plate `(8,0)` triggers the right portal pair `(8,6) <-> (8,8)`.
- Two saved ghosts are available.

This final level combines helper timing with teleporter timing. The helper begins at the lower-left end of the map and follows a track that would eventually take it up the left column to the purple plate at `(0,4)`. The yellow gate at `(0,6)` blocks that climb until the yellow plate at `(3,6)` has toggled it open. The red plate on the top row opens the red gate guarding the top-right route and the right purple trigger.

There are two independent purple teleport systems. The left purple trigger at `(0,4)` is on the helper's track; when the helper reaches it, it can teleport bodies between `(2,2)` and `(2,4)`. The right purple trigger at `(8,0)` swaps bodies between `(8,6)` and `(8,8)`. The player and ghosts must use these triggers as timed events rather than static doors.

To win, set up a route where the helper can pass the yellow gate and reach the left purple trigger at a useful time, while another route opens the top red gate and/or triggers the right teleporter. The live player must be standing on the correct portal endpoint when the corresponding purple trigger resolves, then use the resulting position to reach the goal at `(5,8)`.

Important reconstruction details:

- The yellow gate at `(0,6)` is a latch. Letting the helper or a ghost pulse `(3,6)` twice can close the helper's route again.
- The helper's purple trigger is not under direct player control once the helper starts moving; its arrival time is determined by the count of successful player moves and by whether gates block it.
- The two teleporter pairs are independent. Triggering the left pair does nothing to bodies on the right pair, and vice versa.
- As in earlier levels, the final goal must be reached by the live player after all replayed timing and teleport effects have settled.

## Reconstruction Checklist

- Use five actions: up, down, left, right, and Spacebar.
- Move bodies on a six-pixel grid; animate one pixel at a time until the target cell is reached.
- Store the live player's route as direction deltas. Spacebar rewinds the route through normal plate and gate logic, converts the old body into a ghost with that stored route, and creates a new live body at the start.
- Replay every saved ghost by route index alongside the live player's route index.
- Preserve collision during replay; recorded moves can fail if current gates block them.
- Trigger pressure plates after movement animations settle.
- Normal red gates open while occupied and close on release.
- Yellow gates toggle only on activation pulses.
- Purple triggers swap all occupants of their paired portal endpoints after a short delay.
- Helper agents move once per successful live-player move on their hidden track and obey gates.
- The timer bar shifts left one pixel every two non-reset commands and failure occurs when it has fully left the screen.
- The level is won only by the live player reaching the goal cell.
