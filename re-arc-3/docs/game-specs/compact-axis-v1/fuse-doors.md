Fuse Doors

This is a game about spending single-use keys becoming difficult because every opening permanently consumes a movable resource that may be needed to create the route home.

Authored level count: exactly 7.

Fuse Doors combines two normally separate decisions into one physical action. A fuse is both a Sokoban-style movable block and a permanently expendable access resource. Opening a door is always progress locally, but it may be the wrong global commitment. The settled board always exposes what has been spent, what remains movable, and which routes are permanently open.

1. Player-facing rules

The entire game uses UP, DOWN, LEFT, and RIGHT.

A movement action moves the player one logical cell orthogonally.

Moving into a fuse attempts to push it one cell in the same direction.

Fuses cannot be pulled or chain-pushed.

A fuse pushed into an unused socket of the same color is consumed. Its linked door opens permanently.

A mismatched socket rejects the fuse. Nothing moves.

Closed doors, walls, sockets, the core, the exit, and other fuses block fuse movement.

Open doors behave as ordinary floor for both the player and fuses.

Touching the core collects it. The player then visibly carries its emblem.

The exit becomes visibly active while the core is carried.

Entering the exit while carrying the core completes the level.

SPACE and clicks on any cell are accepted by step() but are intentional no-ops. No object is rendered as clickable, and clicking is not part of the game’s visual grammar.

There is no energy economy or gameplay timer. The evaluator caps are deliberately generous:

Levels	Maximum actions
1–3	256
4–5	1024
6–7	2048

A host-level restart is available outside the action space. Irreversible deadlocks otherwise remain visible rather than causing a hidden failure.

2. Implementation contract
2.1 Logical coordinates

Coordinates use zero-based (row, column) indexing.

(0,0) is the top-left logical cell.

Rows increase downward.

Columns increase rightward.

Every movement action changes exactly one coordinate by exactly one.

Boards vary in native dimensions but are always rendered on a 64×64 canvas.

2.2 State
LevelState:
    player_pos: (row, col)
    carrying_core: bool
    core_present: bool
    fuses:
        id -> {
            color: BLUE | YELLOW | RED,
            pos: (row, col)
        }
    sockets:
        id -> {
            color: BLUE | YELLOW | RED,
            pos: (row, col),
            used: bool,
            linked_door_id: id
        }
    doors:
        id -> {
            pos: (row, col),
            open: bool
        }
    step_count: int
    won: bool

Walls, the core coordinate, and the exit coordinate are static level data.

All authored doors begin closed. All authored sockets begin unused. The core begins uncollected.

There is no cursor, selected object, inventory cursor, facing state, or hidden object state.

2.3 Cell occupancy

A player may occupy:

floor;

an open door;

the core cell;

the exit cell.

A player may not occupy:

a wall;

a closed door;

any socket, used or unused;

a fuse.

A fuse may occupy:

floor;

an open door.

A fuse may not occupy:

a wall;

a closed door;

another fuse;

the core;

the exit;

a used socket.

An unused socket may receive a fuse only through the matching-consumption rule.

2.4 Exact action resolution

For a movement action with displacement d:

step_count += 1

q = player_pos + d

if q contains a fuse:
    z = q + d

    if z is an unused socket:
        if fuse.color != socket.color:
            reject as mismatch
        else:
            remove fuse
            socket.used = true
            socket.linked_door.open = true
            player_pos = q

    else if z is free floor or an open door:
        move fuse from q to z
        player_pos = q

    else:
        reject as blocked push

else if q is walkable for the player:
    player_pos = q

else:
    reject as blocked movement

if core_present and player_pos == core_pos:
    core_present = false
    carrying_core = true

if carrying_core and player_pos == exit_pos:
    won = true

Important ordering consequences:

In a successful socket push, the player advances into the fuse’s previous cell.

The door becomes logically open in that same step().

The player cannot traverse the newly opened door until a later action.

Core collection and exit validation happen after movement resolution.

No secondary fuse moves, chain pushes, automatic gravity, or delayed door effects occur.

A rejected action changes no gameplay state except step_count.

SPACE and click actions increment step_count and then make no state change.

2.5 Success validator

A level succeeds exactly when:

won == true

This is set only when the player’s settled position is the exit and carrying_core == true.

The validator ignores:

how many doors are open;

how many fuses remain;

which optional sockets remain unused;

the order in which the core and doors were handled;

rejected actions;

total actions, provided the cap is not exceeded.

Opening every door is neither required nor usually possible.

3. Rendering
3.1 Canvas and scaling

The render canvas is exactly 64×64 pixels.

For a native board of height H and width W:

cell_size = floor(60 / max(H, W))
origin_x = floor((64 - W * cell_size) / 2)
origin_y = floor((64 - H * cell_size) / 2)

Logical cell (r,c) occupies:

x = origin_x + c * cell_size
y = origin_y + r * cell_size

The unused canvas is color 5.

This produces:

6×6 pixel cells on the 7×9 boards;

5×5 pixel cells on the 11×11 board;

4×4 pixel cells on the 13×13 and 15×15 boards.

No text, numbers, arrows, labels, or HUD counters are rendered.

3.2 Color use
Element	Color IDs
Outside canvas	5
Floor	1
Wall face and bevel	3, with limited 2 and 4 edge pixels
Player	14, with 0 highlight
Core	6, with 0 center
Inactive exit emblem	6 and 4
Active exit	7, 6, and 0
Blue fuse family	9, with 10 highlight
Yellow fuse family	11, with 12 highlight
Red fuse family	8, with 13 shadow
Invalid-action flash	0
Open-door retracted posts	The door’s fuse-family color
3.3 Entity silhouettes

All important entities are connected multi-pixel shapes.

Player: a compact green cross with a two-pixel base. It occupies the center of one logical cell and never resembles a fuse.

Carried core: adds a magenta cap and white center to the player.

Blue fuse: a connected plus-shaped body.

Yellow fuse: a connected 2×3 capsule.

Red fuse: a connected five-pixel diamond.

Core: a magenta diamond with a white center.

Exit: a large cell-sized frame containing the same diamond silhouette as the core.

Socket: a wall cell containing a colored U-shaped rim and a dark cavity whose notch matches the corresponding fuse silhouette.

Used socket: the cavity is brightly filled with the fuse-family color but remains non-traversable.

Closed door: a full wall-width gate with two colored sliding bars.

Open door: floor-colored center with the two colored bars retracted against opposite edges.

Color and silhouette redundantly encode compatibility. Same-colored fuses are intentionally interchangeable.

Every socket is orthogonally adjacent to its linked door. A one-pixel colored conduit crosses their shared edge, so door ownership never has to be inferred from distance.

When the player is orthogonally adjacent to a fuse, the fuse edge facing the
player turns white. Its far edge previews the push result: active color for a
free floor cell, white for a matching socket, red for a mismatched socket, and
neutral gray for a blocked destination. A candidate socket repeats that cue on
its outer rim.

3.4 Animation and feedback

Logical state changes occur atomically during step(). Animation frames are visual interpolation only and do not consume actions.

Ordinary movement

Player movement settles immediately. A legal one-cell fuse push uses one intermediate interpolation frame.

Successful socket activation

Use four deterministic visual frames:

The fuse moves halfway toward the socket; the socket rim turns white.

The fuse contracts into the cavity; the player begins moving into the fuse’s former cell.

A colored pulse crosses the short conduit; the linked door bars retract halfway.

Render the settled used socket, open door, and new player position.

The environment does not accept another action until the settled frame is displayed.

Mismatch

A mismatched socket produces three frames:

White rim flash.

Fuse shifts one rendered pixel toward the socket.

Fuse returns to its original position.

No logical position changes.

A wall or ordinary blocked push instead produces a gray one-pixel shake without the socket-rim flash.

Core and completion

Core pickup uses two frames in which the core contracts into the player’s carried emblem.

The exit changes immediately from inactive magenta/dark gray to active pink/white.

Completion is set on the entry step, followed by a three-frame exit pulse.

4. Authored-map notation

The following ASCII maps are documentation only and are never rendered as text.

Symbol	Meaning
#	Wall
.	Floor
P	Initial player
F	Fuse; color given in the object table
s	Socket; color and link given in the module table
+	Initially closed door
C	Core
E	Exit

All coordinates in the tables are authoritative.

5. Progression summary
Level	Main discovery or composition	Verified shortest completion
1	Fuse consumption permanently opens a door; core activates exit	6
2	Sockets reject the wrong color	10 accepted movements; 11 calls with the intended probe
3	Same-colored sockets share a limited fuse budget	15
4	A fuse can become committed to a socket before being consumed	40
5	An open door can be valuable mainly as a standing cell	44
6	Open doors are maneuvering and temporary parking space; consumed fuses clear work lines	56
7	Two shared color budgets, staged direction changes, cross-region transport, and escape reservation	61

The action strings below are exact replays. Spaces and | characters only divide conceptual phases and are not actions.

6. Level 1 — First Burn

Native board: 7×9
Maximum actions: 256
Teaches: pushing, matching activation, permanent consumption, core collection, and exit activation.

#########
#...#...#
#...+C.E#
#.PFs...#
#...#...#
#...#...#
#########
Initial state

Player: (3,2)

Core: (2,5)

Exit: (2,7)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
B1	Blue	(3,3)
Module	Socket	Door	Role
M1	Blue (3,4)	D1 (2,4)	Only passage to the core
Player-facing discovery

The player, blue fuse, and blue socket are aligned. The linked door is immediately above the socket, with the conduit visible between them. D1 is the only route through the wall.

RIGHT pushes B1 into the socket. B1 disappears, the socket fills, and D1 retracts. The player is now standing in B1’s old cell. Nothing else changes simultaneously, so attribution is clean.

The core and exit use the same magenta emblem. Touching the core visibly places that emblem on the player and causes the exit to brighten. This provides the positive goal evidence in the first level without any symbolic legend.

Verified replay
R | URR | RR

Checkpoints:

Step	Result
1	B1 enters M1; B1 is consumed; D1 opens; player at (3,3)
4	Player enters (2,5) and collects the core
6	Player enters active exit (2,7)
Attribution and necessity check

There is one fuse, one socket, one door, one core, and one exit.

No unrelated object changes during the activation.

The player cannot reach the right chamber without opening D1.

The level cannot be completed by wandering randomly in the start room.

The opened door remains visibly open after the player leaves it.

7. Level 2 — Wrong Mouth

Native board: 7×9
Maximum actions: 256
Introduces exactly one causal unknown: sockets are color-specific.

#########
#....#..#
#....+CE#
#....s..#
#..PFs..#
#....+..#
#########
Initial state

Player: (4,3)

Core: (2,6)

Exit: (2,7)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
B1	Blue	(4,4)
Module	Socket	Door	Role
M1	Blue (3,5)	D1 (2,5)	Actual route to the goal chamber
M2	Yellow (4,5)	D2 (5,5)	Controlled mismatch experiment and apparent direct route
Player-facing discovery

B1 is immediately left of the yellow socket. Under the plausible older hypothesis “a fuse powers any socket,” RIGHT would open D2 and provide the most direct entrance into the right-hand goal chamber.

That experiment is safe:

R

The yellow rim flashes, B1 bumps and returns, and every logical object remains in place. The result distinguishes color matching from generic socket activation immediately.

The player can then move below B1, push it upward, approach from the left, and feed it into the visibly matching blue socket. D1 opens.

M2 is not decorative: it is the shortest-looking action under the specific misconception this level is designed to falsify.

Intended discovery replay
R [rejected probe] | DRULUR | URR | R

The verified shortest completion after the rule is known omits the first rejected probe:

DRULURURRR

Checkpoints:

Accepted step	Result
6	B1 enters blue M1; D1 opens; player at (3,4)
9	Core collected at (2,6)
10	Exit entered at (2,7)
Attribution and necessity check

The two sockets are adjacent to different direct-looking doors.

The mismatch action changes no other state.

The blue route cannot be explained as merely “use the closest socket”; the closest socket visibly rejects B1.

No yellow fuse or additional mechanism is present to complicate the observation.

The lower door remains closed throughout the successful trace, preserving the failed hypothesis as a visible counterfactual.

8. Level 3 — Keep the Way Back

Native board: 7×9
Maximum actions: 256
Introduces: a color is a shared finite budget, and reaching the core is not sufficient.

#########
#...#...#
#...+C..#
#.PFs...#
##......#
#E+s#...#
#########
Initial state

Player: (3,2)

Core: (2,5)

Exit: (5,1)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
B1	Blue	(3,3)
Module	Socket	Door	Role
M1	Blue (3,4)	D1 (2,4)	Immediate shortcut to the core
M2	Blue (5,3)	D2 (5,2)	Sole entrance to the exit alcove
Player-facing discovery

The nearest action is RIGHT, which spends B1 on M1 and opens a door immediately beside the core. This reaches the core quickly, but the exit is enclosed by walls and D2 remains closed.

There is an open route around the lower-right side to the core. The single fuse must therefore be reserved for M2. The board visibly contains one blue fuse and two blue sockets; there is no hidden inventory or replenishment.

The new insight is not “open both required doors.” Both cannot be opened. The player must identify which access can be replaced by walking and which cannot.

Verified replay
URDD | RRUU | DDLLLDL

Checkpoints:

Step	Result
4	B1 enters M2; D2 opens; player at (4,3)
8	Core collected at (2,5) through the open route
15	Player returns through D2 and enters exit (5,1)
Necessity check

Spending B1 on M1 leaves no fuse capable of opening D2.

The exit cannot be entered from above because (4,1) is a wall.

Exhaustive state traversal after the M1 spend contains no winning state.

M1 is a genuine route option rather than arbitrary clutter: it is the shortest route to the core, but the wrong global assignment.

The correct plan preserves escape access before collecting the core.

9. Level 4 — The Second Push Chooses

Native board: 11×11
Maximum actions: 1024
Introduces: a fuse can become irreversibly assigned before it reaches a socket.

###########
######....#
######.F..#
#####s+####
#.......#.#
#......#+.#
#...PF..sC#
#......##.#
##.#s.....#
###E+.....#
###########
Initial state

Player: (6,4)

Core: (6,9)

Exit: (9,3)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
B1	Blue	(6,5)
Y1	Yellow	(2,7)
Module	Socket	Door	Role
M1	Blue (3,5)	D1 (3,6)	Opens the yellow-fuse chamber
M2	Blue (6,8)	D2 (5,8)	Direct core shortcut
M3	Yellow (8,4)	D3 (9,4)	Opens the exit alcove
Player-facing discovery

B1 begins at a visible T-shaped commitment point.

Pushing right once moves B1 to (6,6), where it can still be redirected. Pushing right a second time moves it to (6,7). At that position:

north (5,7) is a wall;

south (7,7) is a wall;

east (6,8) is M2;

the player remains west of B1;

no route reaches B1’s east side.

B1 has not yet been consumed, but it can no longer be pulled, turned, or returned. Its only remaining push is east into M2. Thus the second eastward push is the actual commitment point.

M2 opens a short route to the nearby core but leaves the yellow chamber sealed. Correct play instead approaches B1 from below and pushes it north into M1.

D1 releases access to Y1. Y1 must be brought down into the lower working area, while the player separately loops through the right-side gap to collect the core and get to Y1’s pushing side. Y1 then opens D3.

Verified replay
DRUUU
|
RUUURRDLULDDDDDDLDDRRURRUU
|
DDLLLL
|
DLL

Raw replay:

DRUUURUUURRDLULDDDDDDLDDRRURRUUDDLLLLDLL

Checkpoints:

Step	Result
5	B1 enters M1; D1 opens
13	Y1 is shifted left to column 6 for the descent through D1
21	Y1 reaches (8,6)
31	Core collected at (6,9) after looping through the lower-right gap
37	Y1 enters M3; D3 opens
40	Exit entered at (9,3)
Anti-cleanliness and necessity check

Two eastward pushes commit B1 before consumption; this is a geometric consequence, not a special fuse state.

Spending B1 on M2 opens the attractive core shortcut but leaves no route to Y1 and therefore no way to open D3.

After the second eastward push, exhaustive traversal contains no winning continuation even though B1 is still present.

The core, Y1, and exit are spatially offset rather than encountered in use order.

The lower loop is required both to reach the core and to stand to the right of Y1. It is not a decorative corridor.

M2 plays a contrasting role from M1 despite identical blue-fuse rules: M1 preserves the solution network, while M2 collapses it into a dead end.

10. Level 5 — A Door to Stand In

Native board: 13×13
Maximum actions: 1024
Introduces: an opened doorway can be more valuable as one legal standing cell than as a shortcut.

#############
#.........C.#
#...........#
#...........#
#...........#
#.#+s#s+#####
#......F#...#
#..PF...#.F.#
#.......s...#
#.......+...#
#.......#.s##
#.......#.+E#
#############
Initial state

Player: (7,3)

Core: (1,10)

Exit: (11,11)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
B1	Blue	(7,4)
Y1	Yellow	(6,7)
B2	Blue	(7,10)
Module	Socket	Door	Role
M1	Blue (5,4)	D1 (5,3)	Upper-room shortcut
M2	Blue (5,6)	D2 (5,7)	Creates the standing cell above Y1
M3	Yellow (8,8)	D3 (9,8)	Opens the right chamber
M4	Blue (10,10)	D4 (11,10)	Opens the exit
Player-facing discovery

The upper room is already reachable through the open gap at (5,1). Consequently, D2 is not needed simply to enter that room.

Y1 sits immediately below D2 at (6,7). To push Y1 downward, the player must occupy (5,7), but that cell is a closed door. Opening D2 turns the door cell into the missing pushing position.

This makes the blue assignment nontrivial:

M1 appears to save distance to the distant core.

M2 does not initially look like a route to anything.

M2 is nevertheless the only blue spend that gives the player the geometry needed to move Y1.

B1 must first be shifted right from column 4 to column 6, then pushed upward into M2. The player can collect the core while traversing the already-open upper room, approach D2 from above, and push Y1 down.

Y1 is stopped at (8,7), then approached from the left and pushed right into M3. D3 opens access to B2. B2 is moved downward into M4, after which the player approaches D4 from its left side and enters the exit.

Verified replay
RRDRUU
|
DRUUUURUURR
|
LLLDDDDDDLDR
|
DRRUUURDDD
|
LDDRR

Raw replay:

RRDRUUDRUUUURUURRLLLDDDDDDLDRDRRUUURDDDLDDRR

Checkpoints:

Step	Result
6	B1 enters M2; D2 opens
17	Core collected at (1,10)
21–22	Y1 is pushed down through the newly walkable D2 cell
29	Y1 enters M3; D3 opens
39	B2 enters M4; D4 opens
44	Exit entered at (11,11)
Anti-cleanliness and necessity check

Opening D2 is not an access errand. The upper room is already accessible.

D2’s distinct necessary effect is making (5,7) a legal stance above Y1.

Spending B1 on M1 leaves Y1 immovable and B2 inaccessible; exhaustive traversal after that spend contains no win.

The core can be collected before or after moving Y1, so the level has an independent objective branch rather than one mandatory visible recipe.

B2 is visible long before it can be reached.

M4 is approached sideways after activation because the used socket remains non-traversable; opening the door does not erase the wall structure around it.

11. Level 6 — Clear, Park, Reverse

Native board: 15×15
Maximum actions: 2048
Combines: shared allocation, deliberate fuse removal, open-door maneuvering, direction reversal, and an independent core branch.

###############
#......#......#
#...C..#......#
#......#......#
#......#......#
#.##+s####s+#.#
#......#...F..#
#...PF.s......#
#......+..F...#
#......#......#
###########s+##
########......#
#########s.F..#
########E+....#
###############
Initial state

Player: (7,4)

Core: (2,4)

Exit: (13,8)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
Y1	Yellow	(7,5)
R1	Red	(6,11)
B1	Blue	(8,10)
B2	Blue	(12,11)
Module	Socket	Door	Role
M1	Yellow (5,5)	D1 (5,4)	Core shortcut; wrong yellow expenditure
M2	Yellow (7,7)	D2 (8,7)	Opens the central cross-route
M3	Blue (5,10)	D3 (5,11)	Adds maneuvering space for R1
M4	Red (10,11)	D4 (10,12)	Opens the lower chamber
M5	Blue (12,9)	D5 (13,9)	Opens the exit
Player-facing discovery

Y1 occupies the central push line. It can be redirected north into M1 or pushed east into M2.

M1 opens a door near the core, but the core already has a longer open route through the far-left gap. M2 instead opens the only useful cross-route into the right-hand system. Pushing Y1 into M2 both spends it and removes it from the central working line.

B1 must then be raised into M3. D3 opens directly above R1, but the shortest solution does something stronger than merely standing in the doorway:

R1 is first pushed upward into the newly opened D3 cell.

The player takes the outer upper-right loop.

The player gets above R1.

R1 is pushed back downward through its original location and onward into M4.

The same open door is therefore:

a legal fuse destination;

temporary parking space;

part of the route used to reverse the pushing direction.

M4 opens the lower chamber. B2 then opens D5 from inside that chamber.

Verified replay
ULLLUUUURRR
|
DDLLLDDDRRRRR
|
DRRDRRUUU
|
DRURRUULLDDDDD
|
RDDDLL
|
DLL

Raw replay:

ULLLUUUURRRDDLLLDDDRRRRRDRRDRRUUUDRURRUULLDDDDDRDDDLLDLL

Checkpoints:

Step	Result
11	Core collected at (2,4) through the open left route
24	Y1 enters M2; D2 opens
33	B1 enters M3; D3 opens
36	R1 is pushed upward into open D3
42	Player has looped above the parked R1
47	R1 enters M4; D4 opens
53	B2 enters M5; D5 opens
56	Exit entered at (13,8)
Anti-cleanliness and necessity check

M1 is a tempting core shortcut but consumes the only yellow fuse capable of opening M2. That spend has no winning continuation.

The core branch is independent and can be handled at several points before the exit.

R1 is visible before D3 opens, and B2 is visible before D4 opens; objects do not appear in their execution order.

D3 is not merely traversed once. It changes the legal set of fuse positions and enables a direction reversal.

The red column must eventually be cleared by consumption in M4 before the lower chamber is usable.

B1 and B2 share a color but have different spatial obligations. B1 cannot replace B2 at M5 because it must be consumed to create the R1 maneuver.

The shortest trace contains no repeated toggle or corridor padding; the long middle section performs a push reversal around fixed walls.

12. Level 7 — No Spare Shortcut

Native board: 15×15
Maximum actions: 2048
Final composition: two shared budgets, staged assignment, stance-door use, two cross-region openings, deliberate removal, core reservation, and long fuse transport.

###############
#......#......#
#...C..#......#
#.##+s##......#
#......+....F.#
#...PF.s#+s##.#
#......#.F....#
#......##s+####
#......#......#
########......#
#......#......#
#......s.F....#
##s....+..F...#
#E+....#......#
###############
Initial state

Player: (5,4)

Core: (2,4)

Exit: (13,1)

Carrying core: false

Selection/cursor: none

Fuse	Color	Position
Y1	Yellow	(5,5)
B1	Blue	(4,12)
R1	Red	(6,9)
Y2	Yellow	(11,9)
B2	Blue	(12,10)
Module	Socket	Door	Role
M1	Yellow (3,5)	D1 (3,4)	Direct core shortcut
M2	Yellow (5,7)	D2 (4,7)	Opens the right-side system
M3	Blue (5,10)	D3 (5,9)	Creates the stance above R1
M4	Red (7,9)	D4 (7,10)	Opens the lower-right region
M5	Yellow (11,7)	D5 (12,7)	Opens the lower cross-route
M6	Blue (12,2)	D6 (13,2)	Opens the exit alcove
Resource structure

The initial board visibly contains:

two yellow fuses and three yellow sockets;

two blue fuses and two blue sockets;

one red fuse and one red socket.

The two required yellow assignments are M2 and M5. M1 is the final large-scale shortcut temptation: spending either yellow fuse there leaves only one yellow fuse for two mandatory cross-routes.

The blue assignments are spatially constrained:

B1 must create the stance for R1.

B2 is the only blue fuse that can later reach the exit socket.

Player-facing discovery
Independent core branch

The core has an open route around the left-side wall. M1 offers a shorter door but consumes one-third of the visible yellow socket budget. A player who learned Level 3 can recognize that a walkable detour may be cheaper than a door.

First yellow assignment

Y1 is pushed east through the central line and consumed in M2. D2 opens access to the upper-right workspace.

Blue staging

B1 begins far to the right at (4,12). It cannot simply be pushed toward the nearest object. It must be:

approached from its right;

pushed left twice to (4,10);

approached from above;

pushed down into M3.

D3 opens at (5,9), immediately above R1.

Red stance

The player enters D3 from above and pushes R1 down into M4. D4 opens beside it, creating access to the lower-right region. D3 has again functioned as a required stance rather than a destination.

Second yellow assignment

Y2 is visible before the lower region opens. It is pushed left through (11,8) and consumed in M5. D5 opens one row below it. The socket remains solid, but the offset door creates an unobstructed horizontal transport row.

Final blue transport

B2 starts at (12,10). Once D5 is open, the player gets to B2’s right side and pushes it left through:

(12,9)
(12,8)
D5 at (12,7)
(12,6)
(12,5)
(12,4)
(12,3)
M6 at (12,2)

The eight consecutive left pushes are not walking padding. Every push transports the final remaining blue access resource through the cross-region door and into the exit socket. B2 is consumed, D6 opens, and the player enters the exit from the right.

Verified replay
ULLLUURRR
|
LLLDDDRRRRR
|
URRURRRRRDLLULD
|
LDD
|
RDDDDDLL
|
RRRDLLLLLLLL
|
DLL

Raw replay:

ULLLUURRRLLLDDDRRRRRURRURRRRRDLLULDLDDRDDDDDLLRRRDLLLLLLLLDLL

Checkpoints:

Step	Result
9	Core collected at (2,4) without spending yellow access
20	Y1 enters M2; D2 opens
31–32	B1 is shifted left into its staging column
35	B1 enters M3; D3 opens
38	R1 enters M4; D4 opens
46	Y2 enters M5; D5 opens
51–58	B2 is pushed across the lower region and consumed in M6
61	Exit entered at (13,1)
Anti-cleanliness and necessity check

Spending a yellow fuse on M1 leaves insufficient yellow budget for M2 and M5. Exhaustive traversal after that spend contains no win.

The core branch and the door network are independently approachable, so the board is not laid out as a simple visual checklist.

B1, R1, Y2, B2, the core, and the exit are all visible from the initial frame despite being used in a different dependency order.

D3 is required as a pushing stance, D4 as a regional entrance, and D5 as a transport crossing. Identical door behavior produces three distinct strategic roles through placement.

Y2 must be consumed before B2 can cross. Its disappearance and D5’s opening are both relevant to the lower workspace.

The final B2 movement is a commitment: once it begins moving left through D5, it cannot be retrieved from the exit-side wall geometry. The route must already be prepared.

No door can substitute for the open core route, and no core shortcut can substitute for either yellow cross-route.

Completion still depends on only one natural condition: carry the visible core into the visibly matching exit.

13. Contrasting uses of the same mechanics
Mechanic	Early role	Later contrasting roles
Fuse consumption	Open the only door in Level 1	Sacrifice a shortcut, clear a working line, remove a blocker, or permanently lose the only escape allocation
Same-colored sockets	Simple compatibility	Shared budget with more sockets than fuses; spatially forced assignment of otherwise interchangeable fuses
Open door	Passage	Standing cell, fuse parking cell, direction-reversal space, regional crossing, or transport aperture
Fuse position	One-step push	Pre-consumption commitment, staging column, temporary obstruction, cross-region cargo
Indirect route	Unnecessary in Level 1	Replaces an expensive core shortcut and preserves a fuse for an irreplaceable door
Core	Immediate destination	Independent obligation whose cheapest route competes with tempting fuse expenditure
Exit	Nearby finish	A route that must be funded or prepared before the core is collected

The operational rules never change between these uses. All strategic variation comes from wall geometry, approach direction, permanent consumption, and the order in which regions become maneuverable.
