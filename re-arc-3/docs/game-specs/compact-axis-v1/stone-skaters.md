Stone Skaters

This is a game about pushing stones becoming difficult because every pushed stone slides until something stops it, and every delivered stone changes what can stop the rest.

Authored level count: exactly 7.

Stone Skaters is a deterministic, top-down spatial puzzle. The agent walks normally, but a pushed stone continues in a straight line until it is stopped by geometry or captured by a compatible socket. A captured stone becomes a permanent obstacle, so solving one part of the board physically rewrites the remaining puzzle.

There is no hidden state, randomness, inventory, timer, velocity between actions, or symbolic checker. Every relevant fact is visible in a settled frame:

where the agent can stand;

where each loose stone is;

each stone’s compatibility type;

which sockets remain unfilled;

which stones are permanently captured;

which cells can stop the next slide.

The same operational rule gives stones contrasting roles: cargo, temporary bumpers, permanent brakes, route blockers, and obstacles that must be delayed.

1. Player-facing rules
Objective

Fill every socket with a stone of the same visible type.

An empty socket is flat and traversable. When a compatible loose stone enters it during a slide, the socket clamps around the stone immediately. The captured stone can never move again and becomes solid to both the agent and all other stones.

The level succeeds immediately after every socket is occupied by a captured compatible stone.

The following do not satisfy a socket:

the agent standing on it;

an incompatible loose stone resting on it;

a compatible stone merely aligned with it;

a loose stone that passed across it earlier.

Agent position, remaining step allowance, and delivery order are irrelevant once all sockets are correctly filled.

Compatibility types

There are two compatibility types:

Type A: blue, with a horizontal mark.

Type B: orange, with a vertical mark.

Any Type A stone can fill any Type A socket; identities such as A0 and A1 are only implementation and replay labels. The same applies to Type B.

Levels 1–3 use only Type A. Type B first appears in Level 4.

Actions
Input	Exact behavior
UP, DOWN, LEFT, RIGHT	Attempt one agent move in that cardinal direction. This may initiate a stone push.
SPACE	Wait/no-op. The settled puzzle state does not change.
Click on any render cell	No-op. No board object is clickable and no click affordance is rendered.

Only the four direction actions are meaningful gameplay actions. SPACE and clicks remain in the required action space but are deliberately inert.

A standard environment-level reset() restores the authored initial state; reset is not a player action and is not represented in the grid.

2. Logical geometry

Coordinates are one-indexed:

(row, column);

row increases downward;

column increases rightward.

Levels 1–5 use a 10 × 10 logical board. Levels 6–7 use 12 × 12.

Every board has a one-cell-thick outer wall:

row = 1 or row = H or column = 1 or column = W

All non-wall cells are smooth floor unless they contain a socket or entity. Empty sockets remain floor for movement and sliding.

The agent and every stone occupy exactly one logical cell. A directional movement action moves the agent by exactly one logical cell when unobstructed.

3. Formal state

A settled state contains:

agent_position: (row, column)

stones[i]:
    type: A | B
    position: (row, column)
    status: LOOSE | CAPTURED

goals[j]:
    type: A | B
    position: (row, column)

walls: set[(row, column)]

steps_taken: integer
done: boolean
truncated: boolean

selected_object: null
cursor_cell: null

Additional invariants:

No two stones occupy the same cell.

A captured stone is always on a socket of its own type.

A captured stone never becomes loose again.

No authored level begins with a captured stone.

A loose incompatible stone may overlap an unfilled socket.

No authored level begins with a loose stone on a compatible socket.

selected_object and cursor_cell remain null for the entire episode.

Rendering and animation never mutate the settled logical state.

Each level has max_steps = 1024. This is a non-strategic engine safety cap, more than twelve times the longest canonical solution and hundreds of times the early solutions. It is not displayed as energy or used as puzzle pressure.

Every submitted action increments steps_taken, including no-op actions. If the level is unsolved after action 1024, truncated = true; the board remains in its last settled state.

4. Exact transition rules
4.1 Normal movement

For a cardinal action with direction vector d, let:

adjacent = agent_position + d

If adjacent is:

outside the board;

a wall;

a captured stone;

then the agent bumps and remains in place.

If adjacent is ordinary floor or an unoccupied empty socket, the agent moves there.

If adjacent contains a loose stone, resolve a push.

4.2 Push resolution

Let the pushed loose stone be s, initially at p, and let:

next = p + d

A stone blocker is any other loose or captured stone. Stones are never chain-pushed.

If next is outside, a wall, or occupied by any other stone, the push fails:

the agent remains in place;

the loose stone remains at p;

no other state changes.

Otherwise the push succeeds. The stone repeatedly enters the next cell in direction d until one of two events occurs.

Event A: compatible capture

After the stone enters each new cell, check whether that cell contains an unfilled socket of the same type.

If so:

the slide ends immediately on that socket;

the stone becomes CAPTURED;

no cells beyond the socket are inspected;

the captured stone is solid from the end of this action onward.

If several same-type sockets lie on a ray, the first one entered captures the stone.

Event B: blocked next cell

If the cell after the stone’s current cell is outside, a wall, or occupied by another stone, the moving stone settles in its current cell while remaining LOOSE.

An incompatible empty socket never causes a stop. It behaves exactly like smooth floor.

After any successful push, the agent enters the pushed stone’s original cell p, regardless of the stone’s final distance or whether it was captured.

4.3 Resolution pseudocode
step(action):
    if done or truncated:
        return unchanged state

    steps_taken += 1

    if action is SPACE or CLICK:
        check_truncation()
        return

    d = direction(action)
    q = agent_position + d

    if q is outside or wall or captured stone:
        play_agent_bump()
        check_truncation()
        return

    if q is empty floor or unoccupied socket:
        agent_position = q
        validate()
        check_truncation()
        return

    s = loose stone at q
    first = q + d

    if first is outside or wall or occupied by another stone:
        play_failed_push_shudder(s)
        check_truncation()
        return

    path = []
    current = q
    captured = false

    loop:
        candidate = current + d

        if candidate is outside or wall or occupied by another stone:
            break

        current = candidate
        append current to path

        if current has unfilled socket with socket.type == s.type:
            captured = true
            break

    play_slide_animation(s, path, captured)

    s.position = current
    s.status = CAPTURED if captured else LOOSE
    agent_position = q

    validate()
    check_truncation()

validate() runs after the complete push and agent relocation, not at intermediate animation frames.

4.4 Win validation
done =
    for every goal g:
        there exists a stone s such that
            s.status == CAPTURED
            s.position == g.position
            s.type == g.type

On the transition from done = false to done = true:

reward is 1;

terminated = true;

completion feedback plays.

All other transitions have reward 0. Invalid movement and failed pushes carry no additional penalty.

5. Rendering contract
5.1 Canvas and board placement

The render canvas is always 64 × 64 pixels.

Each logical cell occupies a connected 4 × 4 pixel block.

For a board of width W and height H:

board_pixel_width  = 4W
board_pixel_height = 4H

origin_x = floor((64 - board_pixel_width) / 2)
origin_y = 8 + floor((48 - board_pixel_height) / 2)

Therefore:

a 10 × 10 board begins at pixel (12,12);

a 12 × 12 board begins at pixel (8,8).

The remaining canvas is color 5 black.

There are no grid lines or borders between ordinary floor cells. Adjacent floor cells form one continuous region.

5.2 Used palette
Role	Color IDs
Continuous floor	1
Wall face and shadow	3, 4
Outside-board background	5
Agent	6, 7
Invalid/bump flash	8
Type A body and mark	9, 10
Type B mark and body	11, 12
Captured clamp	14
Completion highlight	0, 14

Colors 2, 13, and 15 are unused.

5.3 Walls

A wall cell is filled with color 3. Its local bottom row and rightmost column are color 4, producing a simple raised block without decorating the floor or drawing a grid.

5.4 Loose stones

Within a 4 × 4 cell, a loose stone uses this connected mask:

. X X .
X X X X
X X X X
. X X .

The four corners remain uncovered. This lets a socket’s colored corners remain visible beneath an incompatible loose stone.

For Type A:

body: 9;

local pixels (1,1) and (1,2) in zero-indexed local coordinates: 10;

the light mark is horizontal.

For Type B:

body: 12;

local pixels (1,1) and (2,1): 11;

the light mark is vertical.

5.5 Empty sockets

An empty socket is flat and traversable.

Its 4 × 4 perimeter is colored with the type accent:

Type A perimeter: 10;

Type B perimeter: 11.

The center retains floor color 1 except for its compatibility mark:

Type A: local (1,1) and (1,2) are 9;

Type B: local (1,1) and (2,1) are 12.

The socket therefore visibly echoes both the color and mark orientation of its compatible stone.

5.6 Captured stones

A captured cell is rendered as a solid raised unit:

full 4 × 4 base: 4;

four corner clamp pixels: 14;

central 2 × 2: the captured stone’s body and type mark.

The original type remains readable, but the full-cell dark base and green clamp make its new solid status unmistakable.

5.7 Agent

The agent uses magenta rather than either compatibility color.

Connected mask:

. X X .
X X X X
. X X .
. X X .

body: 6;

local center pixels (1,1) and (1,2): 7.

The agent has no facing state.

5.8 Draw order

For each cell:

floor or wall;

empty socket, when present;

loose stone;

agent.

Captured stones replace the socket rendering with the captured rendering.

When a loose incompatible stone or the agent stands on a socket, the underlying socket’s corner pixels remain visible.

6. Animation and feedback

Animations are deterministic display sequences generated inside the same step() call. They do not consume actions, advance timers, or accept intervening input.

Successful multi-cell slide

For each logical cell traversed:

one frame with the stone displaced two rendered pixels toward the next cell;

one frame centered in the next logical cell.

Thus a slide of n logical cells uses 2n movement frames.

During these frames:

the agent remains at its pre-action position;

the moving stone is drawn above floor and sockets;

incompatible socket corners remain visible when crossed.

On the final centered frame, the agent snaps into the stone’s original logical cell.

Capture

After arrival on a compatible socket:

one frame shows the loose stone centered on the socket;

one frame replaces the perimeter with the dark raised base and green clamp;

one frame briefly changes the stone’s light mark to color 0.

The final captured state is then displayed.

Ordinary agent movement

A normal one-cell move changes immediately to the next settled frame. It has no mandatory interpolation.

Failed push

A stone with no available first sliding cell:

shifts one rendered pixel in the attempted direction with a color 8 edge flash;

returns to its original center.

The agent remains in place.

Agent collision

When the agent walks into a wall or captured stone, the agent shifts one pixel toward it and returns over two frames. The contacted edge flashes 8.

Completion

After the final capture and validation, all captured clamp corners alternate between 14 and 0 for six frames. No text, score, checkmark, or additional symbol appears.

There is no danger or death animation because the game has no hazards.

7. Progression overview
Level	Name	Principal discovery or composition	Verified canonical actions
1	First Glide	Unit movement, full slide, visible capture objective	3
2	Bank Turn	A fixed wall stops a stone immediately before itself	5
3	Borrowed Brake	A loose stone stops another and later becomes cargo	9
4	Build the Brake	A captured stone is a permanent stopping surface	25
5	Through the Wrong Ring	Incompatible sockets are harmless floor; nested bumper dependencies	51
6	Clear the Launch Side	A useful bumper can occupy the exact side needed for the next push	54
7	Interlocked Rink	Two staging branches, delayed fillers, permanent brakes, and a cross-branch final stop	85

The listed traces were replayed against the transition rules above. Other valid solutions are permitted.

8. Authored-map legend

The following maps are implementation schematics only. These characters are never rendered in the game.

#  wall
.  ordinary floor
@  agent
A  loose Type A stone
B  loose Type B stone
a  unfilled Type A socket
b  unfilled Type B socket
Q  loose Type A stone overlapping an unfilled Type B socket

Every map row is listed from row 1 at the top. Every character position is its one-indexed column.

For every level initially:

steps_taken = 0
done = false
truncated = false
selected_object = null
cursor_cell = null
max_steps = 1024
Level 1 — First Glide
Player-facing discovery

The first two useful actions move the agent one cell each. The third action pushes the only stone. The stone visibly traverses several cells rather than moving only one, enters the matching recessed socket, clamps, and completes the level.

There is no competing object, simultaneous rule change, or ambiguous stopper. The one successful push is positive goal evidence.

Initial state

Board: 10 × 10.

Internal walls: none.

Agent: (7,2).

A0: loose Type A stone at (5,3).

Type A socket: (5,7).

All sockets unfilled.

Selection/cursor: null / null.

##########
#........#
#........#
#........#
#.A...a..#
#........#
#@.......#
#........#
#........#
##########
Verified solution

Canonical trace:

UUR
Chunk	Exact effect
UUR	Agent moves (7,2) → (6,2) → (5,2), then pushes A0 right. A0 traverses (5,4),(5,5),(5,6),(5,7) and captures. Agent ends at (5,3).

Solution depth: 3 actions.

Attribution and necessity

The two U actions demonstrate one-cell agent movement before any stone changes.

The R action is the only action that changes the stone.

The multi-cell animation makes “slides through the intervening cells” distinguishable from teleportation.

The clamp and immediate completion make the socket’s purpose visible.

Removing any visible component removes either the only interaction or the objective.

Level 2 — Bank Turn
Player-facing discovery

The socket is not on the stone’s initial line. A wall protrudes above that line. Pushing upward makes the stone travel until it is immediately below the wall, rather than going to the outer edge. The resulting stop creates a horizontal launch toward the socket.

The first push is a controlled comparison with Level 1: the stone travels a different distance, and the only new causal feature in its path is the wall.

Initial state

Board: 10 × 10.

Internal wall: (4,4).

Agent: (8,4).

A0: loose Type A stone at (7,4).

Type A socket: (5,7).

Selection/cursor: null / null.

##########
#........#
#........#
#..#.....#
#.....a..#
#........#
#..A.....#
#..@.....#
#........#
##########
Verified solution

Canonical trace:

UULUR
Phase	Chunk	Exact effect
1	U	Push A0 upward. It traverses (6,4),(5,4) and stops because (4,4) is a wall. Agent ends (7,4).
2	ULUR	Agent reaches (5,3) and pushes right. A0 traverses (5,5),(5,6),(5,7) and captures. Agent ends (5,4).

Solution depth: 5 actions.

Attribution and necessity

The wall is directly adjacent to the final resting cell, so the stopping cause is spatially legible.

The distance differs from Level 1, ruling out a fixed slide length.

Without (4,4), the first upward push sends the stone to row 2, where the top boundary prevents a downward relaunch.

No loose bumper or captured obstacle appears yet.

Level 3 — Borrowed Brake
Player-facing discovery

The left stone begins in a short launch channel: walls above and below it make the rightward direction the only productive experiment. The stone travels right and stops immediately before the second loose stone.

The second stone is not a special “stopper” object. After serving as a bumper, it is delivered normally. This establishes the central dual role: every loose stone is simultaneously cargo and movable geometry.

Initial state

Board: 10 × 10.

Internal walls: (5,3), (7,3).

Agent: (6,2).

A0: loose Type A stone at (6,3).

A1: loose Type A stone at (6,7).

Type A sockets: (3,6), (3,7).

Selection/cursor: null / null.

##########
#........#
#....aa..#
#........#
#.#......#
#@A...A..#
#.#......#
#........#
#........#
##########
Verified solution

Canonical trace:

RRDRRUDRU
Phase	Chunk	Exact effect
1	R	Push A0 right. It traverses (6,4),(6,5),(6,6) and stops because A1 occupies (6,7). Agent ends (6,3).
2	RDRRU	Reach (7,6) and push A0 upward. It captures at (3,6). Agent ends (6,6).
3	DRU	Reach (7,7) and push A1 upward. It captures at (3,7). Agent ends (6,7).

Solution depth: 9 actions.

Attribution and necessity

The stopping object is loose and visibly distinct from the walls above and below the launch cell.

The first push ends at (6,6), immediately adjacent to A1; no wall lies farther along that direction before A1.

Without A1, A0 reaches column 9. Because column 10 is the outer wall on every row, the agent can never stand to its right and move it left again.

A1 is not a disposable tutorial prop: it must later fill the second socket.

Level 4 — Build the Brake
Player-facing discovery

The first known problem is another loose-bumper setup: orange B0 must use blue A1 to stop one column before it. B0 is then pushed upward into the orange socket at (5,7).

The captured orange stone rises into a solid brake. The blue channel stone A2 can then be launched right and stops at (5,6), immediately before the captured stone. From there it can turn upward toward its own socket.

The important contrast is state-based and physical:

the unfilled orange socket is flat;

the filled orange socket contains a solid stone;

the blue stone is stopped by the captured stone, not by the socket symbol.

Trying A2 before constructing the brake produces an immediate visible overshoot through the unfilled orange socket to (5,9). That mistake is irreversible but cheap to test under the generous reset and step allowance.

Initial state

Board: 10 × 10.

Internal walls: (4,3), (6,3).

Agent: (8,2).

B0: loose Type B stone at (8,4).

A1: loose Type A stone at (8,8).

A2: loose Type A stone at (5,3).

Type B socket: (5,7).

Type A sockets: (3,8), (2,6).

Selection/cursor: null / null.

##########
#....a...#
#......a.#
#.#......#
#.A...b..#
#.#......#
#........#
#@.B...A.#
#........#
##########
Verified solution

Canonical trace:

RRDRRRUDRUULLLLLLUURRDRRU
Phase	Chunk	Exact effect
1	RR	Move to (8,3) and push B0 right. It stops at (8,7) because A1 is at (8,8).
2	DRRRU	Reach (9,7) and push B0 up. It captures on the Type B socket at (5,7).
3	DRU	Reach (9,8) and push A1 up. It captures at (3,8).
4	ULLLLLLUUR	Reach (5,2) and push A2 right. It stops at (5,6) because captured B0 occupies (5,7).
5	RDRRU	Reach (6,6) and push A2 up. It captures at (2,6).

Final agent: (5,6).

Solution depth: 25 actions.

Dependency shape
A1 remains loose
        ↓
B0 can be parked at (8,7)
        ↓
B0 captures at (5,7)
        ↓
A2 gains its only usable brake

After B0 is parked:
A1 may be delivered independently.
Anti-degenerate check

Delivering A1 before using it as a bumper sends B0 to column 9, permanently losing the required leftward approach.

Launching A2 before B0 is captured sends A2 through the unfilled orange socket to (5,9). It can never move left afterward because column 10 is entirely wall.

The walls at (4,3) and (6,3) prevent a vertical escape from A2’s initial cell.

Consequently, the level cannot be completed while treating captured stones as non-solid or while ignoring the brake-building order.

Level 5 — Through the Wrong Ring
Player-facing discovery

The opening is a forced, safe causal demonstration of compatibility.

Orange B0 is confined above and below. The nearest socket on its rightward line is blue at (5,5); its matching orange socket is farther away at (5,8). Pushing right makes B0 visibly pass through the blue socket without clamping, then capture at the orange socket.

This observation cannot be explained by the older rule “every socket catches” or “sockets are obstacles.” The same blue socket is later filled normally by a blue stone, so it changes from harmless transit for one type to a required destination for another without changing its rule.

The rest of the board composes three kinds of brakes:

a loose blue stone that must remain available as a bumper;

a newly captured blue stone that becomes a permanent brake;

the captured opening stone, which later stops a different blue stone from another direction.

Initial state

Board: 10 × 10.

Internal walls: (4,3), (6,3).

Agent: (5,2).

B0: loose Type B stone at (5,3).

A1: loose Type A stone at (8,8).

B2: loose Type B stone at (6,4).

A3: loose Type A stone at (8,3).

B4: loose Type B stone at (3,3).

Type A sockets: (5,5), (3,7).

Type B sockets: (5,8), (4,4), (7,6).

Selection/cursor: null / null.

##########
#........#
#.B...a..#
#.#b.....#
#@B.a..b.#
#.#B.....#
#....b...#
#.A....A.#
#........#
##########
Verified solution

Canonical trace:

RLDDDRDRRRRUUUULLLLLUURURRRDDDDRDDDRUURULDDLLLUUDLU
Phase	Chunk	Exact effect
1	R	B0 passes through Type A socket (5,5) and captures at Type B socket (5,8).
2	LDDDR	Push A3 right from (8,3). It stops at (8,7) against loose A1 at (8,8).
3	DRRRRU	Push A3 upward. It captures at (3,7).
4	UUULLLLLUUR	Push B4 right. It stops at (3,6) against captured A3 at (3,7).
5	URRRD	Push B4 down. It captures at (7,6).
6	DDDRDDDRU	Push A1 upward from (8,8). It stops at (6,8) because captured B0 is at (5,8).
7	URUL	Push A1 left. It stops at (6,5) because loose B2 is at (6,4).
8	DDLLLUU	Push A1 upward one cell. It captures at (5,5), the blue socket crossed in phase 1.
9	DLU	Push B2 upward. It captures at (4,4).

Final agent: (6,4).

Solution depth: 51 actions.

Dependency network
A1 loose ──stops──> A3
A3 captured ──stops──> B4

B0 captured ──stops vertically──> A1
B2 loose ──stops horizontally──> A1
A1 captured ──frees──> B2 delivery
Anti-degenerate check

B0 cannot move vertically because of (4,3) and (6,3).

Pushing B0 left strands it at column 2, where the agent can never stand to its left.

Therefore every solution must push B0 right through the incompatible blue socket.

Moving A1 before it stops A3 strands A3 against the right boundary.

Moving B2 before it stops A1 strands A1 against the left boundary.

Before A3 is captured, B4 passes its blue socket and reaches column 9; the captured A3 is therefore a necessary permanent brake, not merely another completed goal.

Level 6 — Clear the Launch Side
Player-facing discovery

Orange B0 slides right across an incompatible blue socket and stops at (7,9) against blue A1.

The obvious next objective for B0 lies to the left. To push it left, the agent must stand at `(7,10)—the exact cell occupied by the bumper that just made the useful stop.

This is the local impasse:

the stone is in the right row;

the target is visibly reachable along that row;

ordinary movement cannot reach the required pushing side;

moving the bumper itself is the only useful nearby experiment.

Pushing A1 upward captures it at (4,10). The agent finishes that push at (7,10), precisely the newly cleared launch cell, and can immediately push B0 left.

The same captured A1 then becomes a brake for a different blue stone. A socket that was harmless transit early becomes dangerous if filled too soon.

Initial state

Board: 12 × 12.

Internal walls: none.

Agent: (7,3).

B0: loose Type B stone at (7,4).

A1: loose Type A stone at (7,10).

A2: loose Type A stone at (10,6).

B3: loose Type B stone at (3,6).

A4: loose Type A stone at (4,4).

Type A sockets: (4,10), (7,6), (9,9).

Type B sockets: (7,2), (6,9).

Selection/cursor: null / null.

############
#..........#
#....B.....#
#..A.....a.#
#..........#
#.......b..#
#b@B.a...A.#
#..........#
#.......a..#
#....A.....#
#..........#
############
Verified solution

Canonical trace:

RDRRRRRRULUULLLLLLURRRRURRDDDDDLDDDLLUUULUUUUUURDDDLDR
Phase	Chunk	Exact effect
1	R	B0 passes Type A socket (7,6) and stops at (7,9) against A1 at (7,10).
2	DRRRRRRU	Reach (8,10) and push A1 upward. It captures at (4,10), freeing (7,10).
3	L	From the newly freed cell (7,10), immediately push B0 left. It passes (7,6) again and captures at (7,2).
4	UULLLLLLUR	Push A4 right from (4,4). It stops at (4,9) against captured A1 at (4,10).
5	RRRURRD	Push A4 down. It passes incompatible Type B socket (6,9) and captures at (9,9).
6	DDDDLDDDLLU	Push A2 upward. It captures at (7,6), the socket that had to remain open for B0.
7	UULUUUUUURD	Push B3 down from (3,6). It stops at (6,6) against captured A2 at (7,6).
8	DDLDR	Push B3 right. It captures at (6,9).

Final agent: (6,6).

Solution depth: 54 actions.

Dependency network
A1 loose ──stops──> B0
A1 moved ──frees B0's reverse launch side
A1 captured ──stops──> A4

B0 completed ──allows──> A2 to fill shared crossing
A2 captured ──stops──> B3

A4 must cross (6,9)
before B3 permanently occupies (6,9)
Anti-degenerate check

Moving A1 before parking B0 makes B0 slide to column 11, where no leftward relaunch is possible.

Once B0 is parked, it cannot be pushed left until A1 vacates (7,10).

Filling the blue socket (7,6) with A2 before completing B0 makes the captured stone block B0’s required horizontal route.

Filling (6,9) with B3 before sending A4 downward makes the captured orange stone block A4.

Thus the same socket is useful as open transit in one phase and a permanent obstruction in another, entirely because of visible state and order.

Level 7 — Interlocked Rink
Player-facing discovery

The final level introduces no new operational rule. It asks the player to infer a dependency network from visible launch sides and stopping lines.

Two cargos begin loaning geometry from two different bumpers:

blue A0 begins directly over an incompatible orange socket at (10,5). It must leave that socket and slide upward against orange B1;

orange B3 must slide right against blue A4.

After staging those cargos, the original bumpers are moved and captured.

A second set of stones then fills the spaces the staged cargos left behind:

B2 captures on A0’s original socket and becomes the brake for A0’s downward return;

captured A4 becomes the brake for A5.

The two branches cross-link near the end:

B3 must vacate column 9 before A5 can descend;

captured A5 at (9,9) becomes the final horizontal brake for A0;

A0 stops at (9,8) and turns upward into the last socket.

Initial state

Board: 12 × 12.

Internal walls: none.

Agent: (11,5).

A0: loose Type A stone at (10,5), overlapping the unfilled Type B socket at the same coordinate.

B1: loose Type B stone at (3,5).

B2: loose Type B stone at (10,3).

B3: loose Type B stone at (8,3).

A4: loose Type A stone at (8,10).

A5: loose Type A stone at (5,4).

Type B sockets: (10,5), (3,9), (8,2).

Type A sockets: (5,10), (9,9), (6,8).

Socket (10,5) is unfilled but initially spatially occupied by incompatible loose A0.

Selection/cursor: null / null.

############
#..........#
#...B...b..#
#..........#
#..A.....a.#
#......a...#
#..........#
#bB......A.#
#.......a..#
#.B.Q......#
#...@......#
############

In the renderer, Q is not a special sprite. It is the normal blue loose-stone sprite drawn over the orange socket, whose orange corner pixels remain visible.

Verified solution

Canonical trace:

UULLLURDRRRRRRRUUULLLLLLLURDDRRRRRRDLUULUURDLLLLLUURLDDDDDDLLDRUUUUUUURRDDDDDLDRRDRRU
Phase	Chunk	Exact effect
1	U	Push A0 upward off the incompatible socket. It travels to (4,5) and stops against loose B1 at (3,5).
2	ULLLUR	Reach (8,2) and push B3 right. It stops at (8,9) against loose A4 at (8,10).
3	DRRRRRRRU	Reach (9,10) and push A4 up. It captures at (5,10).
4	UULLLLLLLUR	Push A5 right. It stops at (5,9) against captured A4 at (5,10).
5	DDRRRRRRDL	Reach (8,10) and push B3 left. It captures at (8,2), clearing column 9.
6	UULUURD	Reach (4,9) and push A5 down. It captures at (9,9).
7	LLLLLUUR	Reach (3,4) and push B1 right. It captures at (3,9), freeing (3,5), the pushing side above A0.
8	LDDDDDDLLDR	Reach (10,2) and push B2 right. It captures at (10,5), filling the socket vacated by A0.
9	UUUUUUURRD	Reach (3,5) and push A0 down. It stops at (9,5) against captured B2 at (10,5).
10	DDDDLDR	Reach (9,4) and push A0 right. It stops at (9,8) against captured A5 at (9,9).
11	RDRRU	Reach (10,8) and push A0 up. It captures at (6,8).

Final agent: (9,8).

Solution depth: 85 actions.

Dependency network
LEFT / VERTICAL BRANCH

B1 remains loose
      ↓
A0 stages at (4,5)
      ↓
B1 may capture at (3,9)
      ↓
cell (3,5) becomes A0's downward launch side

A0 leaves (10,5)
      ↓
B2 may capture at (10,5)
      ↓
B2 becomes A0's downward brake


RIGHT / HORIZONTAL BRANCH

A4 remains loose
      ↓
B3 stages at (8,9)
      ↓
A4 captures at (5,10)
      ↓
A4 becomes A5's horizontal brake

B3 moves left
      ↓
column 9 clears
      ↓
A5 captures at (9,9)


CROSS-LINK

A5 captured at (9,9)
      ↓
A0 can stop at (9,8)
      ↓
A0 turns upward into the final socket
Anti-degenerate check

If B1 moves before staging A0, A0 slides to row 2. The outer wall at row 1 prevents the agent from ever standing above it to push it downward.

If A4 moves before staging B3, B3 slides to column 11 and can never be pushed left.

A5 cannot obtain column 9 until A4 has been captured at (5,10).

A5 cannot descend to (9,9) while B3 remains at (8,9).

A0 cannot return downward safely until B2 fills (10,5); otherwise it reaches row 11 and loses its upward pushing side.

A0 cannot stop at (9,8) until A5 is captured at (9,9); without that brake it reaches column 11 and loses its leftward pushing side.

Every captured stone used as a brake is also a required goal occupant. There are no optional stopper objects or decoys.

9. Blind-player evidence audit
Level	Low-cost distinguishing experiment	Immediate evidence
1	Move above the start and push the only stone	It visibly traverses multiple cells, clamps, and completes the board.
2	First UP	The stone stops immediately below a wall at a distance different from Level 1.
3	First RIGHT	The stone stops immediately before another loose stone; the channel walls are not in its travel direction.
4	Capture B0, then launch A2	The raised captured stone stops A2 one cell before itself. An early launch instead visibly crosses the flat orange socket and overshoots.
5	First RIGHT	Orange B0 crosses a blue socket without changing state, then clamps at orange.
6	First RIGHT	B0 reaches a useful position, but its required reverse-side cell is visibly occupied by the bumper. Moving that bumper immediately frees the side.
7	No new rule	Every dependency is a composed version of evidence already seen in Levels 3–6.

The first three boards change only one relevant causal feature at a time. Later boards create difficulty through state-dependent geometry and order, not through hidden exceptions or additional device types.

10. Global necessity and consistency checks

Every authored level has exactly as many stones as sockets.

Every stone is required for validation; there are no spare stones.

Every internal wall constrains a launch direction or creates a required stopping geometry.

Empty sockets are always traversable, regardless of type.

Captured stones are always solid, regardless of type or level.

Loose stones always block other stones, regardless of whether they are ultimately used as cargo or bumpers.

No stone is ever chain-pushed.

No rule changes between levels.

No goal requires the agent to finish in a particular position.

No level uses a timer, energy total, click selection, hidden phase, or invisible dependency.

Later difficulty comes from which stopping surfaces must remain loose, which must become permanent, which routes must stay open, and whether the agent can still reach the required launch side.
