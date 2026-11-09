Worked for 38m 41s
Anchor Ice

Authored level count: exactly 7

This is a game about sliding becoming difficult because movable objects determine where movement can stop.

Anchor Ice is a top-down ice puzzle in which the player does not choose how far to move. Each directional action launches the agent until something physically stops it. Loose crates can be repositioned to create temporary stopping surfaces; crates that enter anchor sockets become permanent stopping surfaces. The challenge is therefore not merely reaching a location, but constructing the geometry that makes stopping there possible.

The complete game uses only walls, ice, one agent, loose crates, anchor sockets, anchored crates, and one exit. No new object categories appear after Level 3. Later depth comes entirely from changing the order and context in which crates serve as brakes, backstops, obstructions, and permanent supports.

1. Player-facing rules

The green exit pad is the objective. The agent must finish an action resting on the exit. Passing across it during a slide does not count.

UP, DOWN, LEFT, and RIGHT launch the agent in that direction.

The agent slides until one of the following happens:

It reaches a wall or anchored crate, in which case it stops in the preceding cell.

It reaches a loose crate that cannot move, in which case it stops in the preceding cell.

It reaches a loose crate with clear ice immediately beyond it, in which case it pushes the crate.

When a loose crate is pushed:

The crate slides in the push direction until stopped.

The agent always comes to rest in the crate’s original cell.

A moving crate stops before a wall, another loose crate, or an anchored crate.

A moving crate that enters an open anchor socket stops immediately and becomes permanently anchored, even when open ice continues beyond the socket.

A moving crate never pushes another crate.

Loose crates and anchored crates both stop the agent. Only loose crates can be pushed.

SPACE resets the current level to its exact initial state. There is no energy system, timer, or action budget, so experimentation is unrestricted.

Clicking any cell has no effect. There are no clickable objects, no cursor state, and no selected object.

2. State-machine contract
2.1 Logical space

Each level uses a rectangular orthogonal lattice.

Coordinates are written as (x, y):

(0, 0) is the upper-left cell.

x increases to the right.

y increases downward.

UP = (0, -1)

DOWN = (0, +1)

LEFT = (-1, 0)

RIGHT = (+1, 0)

Every authored level has a complete wall perimeter. The perimeter is physical wall geometry, not a decorative grid border.

All in-bounds cells not listed as walls are traversable ice. Exit cells and open socket cells are also traversable ice.

2.2 Static state
width: integer
height: integer
walls: set[(x, y)]
sockets: set[(x, y)]
exit: (x, y)

Static constraints:

walls, sockets, and exit are pairwise disjoint.

Every socket is on traversable floor.

The exit is on traversable floor.

No authored level begins with a crate on the exit or on a socket.

2.3 Dynamic state
agent: (x, y)
loose_crates: set[(x, y)]
anchored_crates: set[(x, y)]
status: ACTIVE | COMPLETE

Invariants:

anchored_crates ⊆ sockets

A socket is open exactly when it is not in anchored_crates.

A settled state never has a loose crate on an open socket; entering one immediately anchors it.

The agent, loose crates, anchored crates, and walls never share cells.

Crates are mechanically indistinguishable. Labels such as C1 in this document are only implementation references and are not rendered or stored as gameplay identity.

There is no hidden move history, momentum, facing direction, charge, or latent mode.

The settled frame completely specifies the gameplay state.

2.4 Directional transition

For a direction vector d:

q = agent

while q + d is empty traversable floor:
    q = q + d

blocker = q + d

“Empty traversable floor” includes ordinary ice, the exit, and open sockets, but excludes walls and all crates.

Case A: wall or anchored crate

If blocker is a wall, outside the board, or an anchored crate:

agent = q

If q equals the original agent position, this is a zero-displacement bump.

Case B: loose crate with blocked space behind it

If blocker contains a loose crate and blocker + d is a wall, outside the board, a loose crate, or an anchored crate:

agent = q
all crates remain unchanged

The agent may still have crossed several empty cells before stopping next to the unpushable crate.

There is no chain pushing.

Case C: pushable loose crate

If blocker contains a loose crate and blocker + d is empty traversable floor:

Remove the crate from blocker.

Set the agent’s destination to blocker.

Advance the crate one cell at a time in direction d.

After every entered cell, check for socket capture before considering the following cell.

If the crate enters an open socket:

anchored_crates += entered_socket
agent = blocker

The crate does not continue past the socket.

Otherwise, the crate stops in the last empty cell before a wall, loose crate, or anchored crate:

loose_crates += final_crate_cell
agent = blocker

The agent does not continue sliding after pushing. It always settles in the pushed crate’s original cell.

2.5 Exact collision table
Moving object	Encountered cell	Result
Agent	Empty ice, exit, or open socket	Continues
Agent	Wall or anchored crate	Stops before it
Agent	Loose crate with clear cell behind	Pushes it
Agent	Loose crate with blocked cell behind	Stops before it
Moving crate	Ordinary ice or exit	Continues
Moving crate	Open socket	Enters, anchors, and stops
Moving crate	Wall	Stops before it
Moving crate	Loose crate	Stops before it
Moving crate	Anchored crate	Stops before it
2.6 Win validation

Win validation occurs only after the full action has resolved into a settled state.

success = (agent == exit)

Everything else is ignored by the win validator:

Crates do not need to occupy sockets.

Sockets may remain open.

Any number of crates may remain loose.

Crate locations do not need to match the reference solution.

Passing across the exit during an animation does not count.

A crate passing across or resting on the exit does not count.

If a push leaves the agent in a crate’s former cell and that cell is the exit, the level succeeds.

After success, the exit performs its completion pulse and the level is frozen until the environment loads the next authored level.

3. Rendering contract
3.1 Canvas and board placement

Render target: exactly 64 × 64 pixels.

Every pixel is an integer color ID from 0–15.

Logical cell size: 5 × 5 rendered pixels.

Board pixel origin:

origin_x = floor((64 - 5 × width) / 2)
origin_y = floor((64 - 5 × height) / 2)

Pixels outside the logical board use color 0, white.

There are no lines between traversable cells.

Adjacent ice cells form one continuous light-blue surface.

Adjacent wall cells visually merge into connected dark structures.

The game renders no text, letters, numbers, arrows, or coordinate labels.

3.2 Colors used
Role	Color ID
Outside background	0
Ice	10
Walls and crate shadows	4
Agent	6
Agent bump highlight	7
Invalid/failure flash	8
Loose-crate body	12
Loose-crate highlight	11
Socket and clamp	13
Exit and completion	14
Exit center	0
Ice-scrape flash	9

Colors 1, 2, 3, 5, and 15 are unused in normal settled frames.

3.3 Entity silhouettes

The following are documentation sketches of the 5 × 5 pixel patterns. They are not text rendered in the game.

Legend:

i = underlying ice, color 10
P = agent, color 6
O = orange crate, color 12
Y = yellow crate highlight, color 11
M = maroon clamp, color 13
D = dark shadow, color 4
G = green exit, color 14
W = white, color 0
Agent
iiiii
iiPii
iPPPi
iiPii
iiiii
Loose crate

The dark lower-right pixels make the crate appear raised above the ice.

iiiii
iOOOi
iOYOD
iOOOD
iiDDi
Open socket

The four separated jaws leave an unmistakably open central recess.

iMiMi
MMiMM
iiiii
MMiMM
iMiMi
Anchored crate

The crate loses its raised shadow and is visibly enclosed by the clamp.

iMiMi
MOOOM
iOYOi
MOOOM
iMiMi
Exit
iGGGi
GiiiG
GiWiG
GiiiG
iGGGi

The exit is rendered before the agent and crates. When the agent occupies it, enough of the large green ring remains visible to communicate success.

3.4 Layer order

White outside background.

Ice and walls.

Exit.

Open sockets.

Anchored crates.

Loose crates.

Agent.

Temporary animation feedback.

3.5 Animation and commit timing

All gameplay changes occur as the deterministic result of one step() call. Animation frames are derived representations of that result and are not additional gameplay states.

No animation frame consumes an action. Inputs are ignored until the action’s animation has finished.

Agent-only slide

For every logical cell entered, show one transient frame with the agent centered in that cell. Use 35 ms per cell, with a maximum total slide animation of 280 ms.

The gameplay state is committed on the final frame.

Push

Animate the agent through all empty cells before the crate.

On the contact frame, move the agent into the crate’s original cell and move the crate into its first destination cell.

Keep the agent stationary there while the crate advances one logical cell per frame.

Commit the final agent and crate state together on the last frame.

This makes the rule “the agent settles in the vacated square” visually explicit.

Socket capture

When a moving crate first enters an open socket:

It stops immediately.

The loose-crate shadow disappears.

The maroon jaws close over two 45 ms frames.

The anchored-crate silhouette becomes the settled sprite.

The crate is logically anchored as soon as it enters the socket; the closing animation cannot be interrupted.

Blocked bump

When a direction produces zero displacement:

Offset the agent one rendered pixel toward the blocker for one 45 ms frame.

Render two blue scrape pixels behind it.

Return it to the original centered sprite.

The settled gameplay state does not change.

Completion

When the settled agent cell equals the exit:

Alternate the exit’s green and white pixels for four 60 ms frames.

Leave the final exit green.

Freeze gameplay after the pulse.

4. Curriculum overview
Level	Main discovery or reinterpretation	Verified minimum launches
1	Sliding continues to a blocker; crossing the exit is insufficient	4
2	Pushing moves a crate while the agent settles in its old cell	5
3	A socket captures a crate and turns it into a permanent stopper	5
4	One loose crate can position another by acting as its backstop	25
5	A precision backstop can immediately become a liability that must be removed	44
6	A crate should be used while loose before being spent as an anchored support	52
7	A fixed stopper can support a chain of successively placed stopping surfaces	62

All listed reference paths contain only state-changing launches. No reference path contains a bump, reset, click, or redundant reversal.

5. Authored levels

The ASCII layouts below are documentation only and are never rendered in-game.

# = wall
. = ice
A = agent
C = loose crate
S = open socket
X = exit

All perimeter cells are walls. The coordinate list is authoritative if a diagram and list ever disagree.

Level 1 — Cross, Then Stop
Purpose

Teach uninterrupted sliding and the exact-stop objective with one clean causal experiment.

Initial state
Board size: 7 × 7
Walls: full perimeter plus {(2,5)}
Agent: (1,1)
Loose crates: {}
Sockets: {}
Anchored crates: {}
Exit: (3,1)
Selection/cursor: none
Status: ACTIVE
#######
#A.X..#
#.....#
#.....#
#.....#
#.#...#
#######
Player-facing discovery

The exit is visibly on the agent’s row. Pressing RIGHT is the most natural first experiment.

That action causes the agent to pass directly across the green exit and stop at (5,1). There is no completion pulse. This immediately disproves the plausible hypothesis that merely touching or crossing the exit is sufficient.

The remaining geometry demonstrates how to create the correct approach:

DOWN reaches (5,5).

LEFT stops at (3,5) because of the wall at (2,5).

UP stops at (3,1), directly on the exit.

The final completion pulse provides positive evidence for the correct rule: the settled cell, not the traversed path, is validated.

Reference solution
RIGHT, DOWN, LEFT, UP

Shorthand:

R D L U
Reference replay
After action	Agent
1	(5,1); crossed exit without winning
2	(5,5)
3	(3,5); wall (2,5) supplies horizontal stop
4	(3,1); success
Validation

The level completes only after action 4, when the agent settles at (3,1).

Attribution check

There are no crates, sockets, decoys, or simultaneous state changes. The first action reveals both the sliding extent and the exit validator through one visibly continuous event. The single internal wall is necessary; removing it makes the authored level unsolvable.

Level 2 — The Vacated Square
Purpose

Teach crate pushing and, specifically, that the agent stops in the crate’s original cell rather than following behind it.

Initial state
Board size: 7 × 7
Walls: full perimeter plus
       {(1,2), (4,3), (3,4), (5,5)}
Agent: (1,1)
Loose crates: {(3,1)}
Sockets: {}
Anchored crates: {}
Exit: (4,5)
Selection/cursor: none
Status: ACTIVE
#######
#A.C..#
##....#
#...#.#
#..#..#
#...X##
#######
Player-facing discovery

At the start:

UP and LEFT meet the perimeter.

DOWN meets the wall at (1,2).

The only visibly productive direction is RIGHT, toward the loose crate.

Open ice is visible beyond the crate, inviting a push experiment.

Pressing RIGHT pushes the crate from (3,1) to (5,1). The agent visibly settles at (3,1), the crate’s former cell.

That exact x-coordinate is the important evidence. From (3,1), DOWN stops at (3,3) because of the wall at (3,4), opening the lower route. Had the agent followed the crate farther, the wall at (4,3) would prevent the same route.

Reference solution
R D L D R
Reference replay
After action	Agent	Loose crates
1	(3,1)	{(5,1)}
2	(3,3)	{(5,1)}
3	(1,3)	{(5,1)}
4	(1,5)	{(5,1)}
5	(4,5)	{(5,1)}

The wall at (5,5) stops the final rightward slide on the exit.

Validation

Success is checked after action 5 because the agent rests at (4,5). The crate’s final position is irrelevant.

Attribution check

The crate is the only new causal object. The first useful action directly shows both moving bodies and their final cells. No solution exists in the actual level without pushing the crate. Every internal wall changes solvability or creates a shorter unintended route if removed.

Level 3 — Clamp and Return
Purpose

Teach socket capture, visible permanent anchoring, and the fact that an anchored crate no longer moves when approached from another direction.

Initial state
Board size: 8 × 6
Walls: full perimeter
Agent: (1,1)
Loose crates: {(3,1)}
Sockets: {(4,1)}
Anchored crates: {}
Exit: (5,1)
Selection/cursor: none
Status: ACTIVE
########
#A.CSX.#
#......#
#......#
#......#
########
Player-facing discovery

The loose crate, open socket, and exit lie on one uncluttered row. Crucially, two open ice cells remain visibly beyond the socket. A player can therefore distinguish socket capture from ordinary wall stopping.

Pressing RIGHT:

Pushes the crate out of (3,1).

Moves it into the socket at (4,1).

Stops it immediately, even though (5,1) and (6,1) are open.

Changes it into the visibly lowered, clamped anchored sprite.

Leaves the agent at (3,1).

The player then travels around the open rectangle to the right side. The final LEFT action moves toward the anchored crate. Because the crate is permanent and unpushable, the agent stops at (5,1), the exit.

If the crate were merely a loose crate that happened to stop on the socket, the final LEFT action would push it and would not leave the agent on the exit. The solution therefore forces observation of the property that makes anchoring distinct.

Reference solution
R D R U L
Reference replay
After action	Agent	Loose crates	Anchored crates
1	(3,1)	{}	{(4,1)}
2	(3,4)	{}	{(4,1)}
3	(6,4)	{}	{(4,1)}
4	(6,1)	{}	{(4,1)}
5	(5,1)	{}	{(4,1)}
Validation

The exit is crossed by the crate during the initial action only if the socket rule were implemented incorrectly. Correct behavior captures the crate at (4,1). Success occurs only when the agent stops at (5,1) on action 5.

Attribution check

The socket is the only new causal unknown. There are no internal walls or unrelated objects. The open floor beyond the socket rules out the mistaken explanation that the crate stopped because it reached a boundary. No solution exists without socket capture.

Level 4 — Two Brakes
Purpose

Introduce crate-on-crate stopping and make both loose crates alternate between being movable tools and stopping surfaces.

Initial state

For documentation:

C1 begins at (2,4).

C2 begins at (3,2).

They are visually and mechanically identical.

Board size: 9 × 9
Walls: full perimeter plus
       {(1,3), (1,4), (2,6), (4,2),
        (5,2), (6,2), (4,5), (7,7)}
Agent: (2,1)
Loose crates: {(2,4), (3,2)}
Sockets: {}
Anchored crates: {}
Exit: (5,1)
Selection/cursor: none
Status: ACTIVE
#########
#.A..X..#
#..C###.#
##......#
##C.....#
#...#...#
#.#.....#
#......##
#########
Player-facing discovery

The first DOWN action encounters C1. The wall at (2,6) permits the crate to move only from (2,4) to (2,5), while the agent settles at (2,4). This is an immediate reuse of the “vacated square” rule in a less direct setting: the crate is useful because it creates a precise agent stop, not because it is moving toward a socket.

The player later moves C2 to (7,1) and returns C1 to (3,1). This creates the level’s decisive local impasse:

top row positions: C1 at x=3, exit at x=5, C2 at x=7

Pushing C1 right would normally send it to the far wall. Instead, C2 stops it at (6,1). The agent remains at (3,1). A second RIGHT launch cannot push the crate at (6,1) because C2 occupies (7,1), so the agent stops immediately before the pair at (5,1), on the exit.

Thus one crate first positions another, and the resulting adjacent pair then becomes the agent’s brake.

Reference solution
D R D L U L U R D L U R L D R U R D L U U L U R R
Key checkpoints
After action	Agent	Loose crates	Meaning
1	(2,4)	{(2,5),(3,2)}	C1 gives the first intermediate stop
5	(3,2)	{(2,5),(3,1)}	C2 has been pushed upward
8	(3,1)	{(2,5),(7,1)}	C2 is staged as a distant backstop
12	(2,5)	{(3,5),(7,1)}	C1 enters the central staging lane
20	(3,5)	{(3,1),(7,1)}	Both crates are aligned on the top row
24	(3,1)	{(6,1),(7,1)}	C1 stops against C2
25	(5,1)	{(6,1),(7,1)}	The pair stops the agent on the exit
Validation

Success depends only on the final agent cell (5,1). The adjacent crate pair is not an explicit validator requirement; it is the physical construction that makes the final stop possible.

Necessity and anti-cleanliness check

Both crates are indispensable; removing either makes the authored instance unsolvable. Every internal wall affects solvability or minimum solution length.

The objects are not encountered in use order:

The lower crate is touched first, staged in the lower chamber, and only much later returned to the exit row.

The upper crate leaves its starting area early but is not used as a backstop until the final phase.

The top row is visited repeatedly in materially different states.

The level cannot be reduced to a corridor or a sequence of independent errands. Its central insight is the physical collision between the two loose crates.

Level 5 — Borrowed Backstop
Purpose

Reuse crate-on-crate stopping in the opposite strategic role: a crate that enables an exact placement immediately becomes an obstruction that must be moved away.

Initial state

Documentation labels:

C1 = (2,2)

C2 = (5,5)

C3 = (5,7)

Board size: 10 × 10
Walls: full perimeter plus
       {(5,1), (7,2), (3,3), (4,4), (6,5),
        (1,6), (1,7), (7,7), (3,8), (6,8)}
Agent: (8,3)
Loose crates: {(2,2), (5,5), (5,7)}
Sockets: {}
Anchored crates: {}
Exit: (6,3)
Selection/cursor: none
Status: ACTIVE
##########
#....#...#
#.C....#.#
#..#..X.A#
#...#....#
#....C#..#
##.......#
##...C.#.#
#..#..#..#
##########
Player-facing discovery

The lower crate C3 is first raised one cell, then pushed left into column x=2. From below, the player pushes it upward.

C1 is already at (2,2), so C3 cannot continue to the top wall. It stops at (2,3), directly below C1.

This is the first contrasting reuse of crate-on-crate stopping:

In Level 4, the backstop remained useful and became part of the final brake.

Here, C1 has done its job as soon as C3 reaches (2,3).

C1 now blocks the very column through which C3 must later be removed.

The player must approach C1 from the right, push it left to (1,2), and then push C3 down from (2,3) to (2,8). The former precision tool has become a liability.

Only after clearing that region can C2 be moved from (5,5) to (5,2), nudged right to (6,2), and pushed down to (6,4). The final DOWN launch stops the agent at (6,3), immediately above that crate.

Reference solution
L U L D R D R U R U L L D L U R U L D L U L D L
D R D R U U L U L D R L U R D R U L D D

The line break is only for readability; the sequence is continuous and contains 44 launches.

Key checkpoints
After action	Agent	Loose crates	Meaning
8	(5,7)	{(2,2),(5,5),(5,6)}	C3 raised one cell
11	(5,6)	{(2,2),(2,6),(5,5)}	C3 moved into column 2
15	(2,6)	{(2,2),(2,3),(5,5)}	C1 stops C3 at the exact intermediate cell
22	(2,2)	{(1,2),(2,3),(5,5)}	The former backstop C1 is removed
23	(2,3)	{(1,2),(2,8),(5,5)}	C3 is sent away, reopening the region
29	(5,5)	{(1,2),(2,8),(5,2)}	C2 reaches the upper staging row
34	(1,2)	{(1,5),(2,8),(5,2)}	C1 is cleared from the horizontal route
35	(5,2)	{(1,5),(2,8),(6,2)}	C2 is aligned above the exit
43	(6,2)	{(1,5),(2,8),(6,4)}	C2 becomes the final brake
44	(6,3)	{(1,5),(2,8),(6,4)}	Success
Validation

No crate configuration is checked. The final crate at (6,4) matters only because it causes the agent to settle at exit (6,3).

Necessity and anti-cleanliness check

All three crates are necessary; removing any one makes the level unsolvable under the authored geometry. Every internal wall changes solvability or creates a shorter bypass if removed.

The level forms a dependency network:

C1 positions C3.

Moving C1 then permits moving C3.

Clearing both creates access to the approach angles needed for C2.

C2 becomes the final stopper.

The final stopper is not the crate that receives the most early attention, and the first useful pair must be dismantled rather than preserved.

Level 6 — Spend It Last
Purpose

Make anchoring a genuine commitment. One crate must serve as a loose backstop before it is spent as a permanent support. The anchored result then enables a different crate and a different pushing axis.

Initial state

Documentation labels:

C1 = (2,8)

C2 = (4,3)

C3 = (4,8)

Board size: 11 × 11
Walls: full perimeter plus
       {(1,4), (3,4), (5,4), (6,4),
        (8,5), (9,5), (2,6), (4,6),
        (6,8), (1,9), (4,9), (8,9)}
Agent: (3,9)
Loose crates: {(2,8), (4,3), (4,8)}
Sockets: {(3,1)}
Anchored crates: {}
Exit: (7,7)
Selection/cursor: none
Status: ACTIVE
###########
#..S......#
#.........#
#...C.....#
##.#.##...#
#.......###
#.#.#.....#
#......X..#
#.C.C.#...#
##.A#...#.#
###########
Player-facing discovery

C1 begins awkwardly left of the agent. It must first be nudged right to (3,8), raised to (3,5), moved across to (7,5), and finally pushed upward to (7,1).

At this point, the visible socket is far to the left, but anchoring C1 immediately is not yet useful. C1 still has an important loose-crate role.

The player stages C3 through the center and pushes it upward in column 7. Because C1 remains loose at (7,1), it stops C3 at (7,2). This creates a vertically adjacent pair with two different future roles.

Only now should C1 be pushed left into the socket at (3,1):

The top crate becomes permanently anchored.

Column 7 is freed.

C3 can be pushed downward from (7,2) to (7,9).

The new anchored crate at (3,1) creates a reliable horizontal stopping surface.

After the capture, moving left from (7,1) stops the agent at (4,1), directly beside the anchored crate. This exact stop supplies the downward pushing angle for C2 at (4,3).

Finally:

C2 is moved through (4,5) to (7,5).

It is pushed downward.

C3, waiting at (7,9), stops it at (7,8).

A second DOWN launch stops the agent on the exit at (7,7).

The same C1 has therefore been:

A loose movable object.

A backstop for C3.

A committed anchored object.

A horizontal agent stopper needed to access C2.

Reference solution
U L D R D R U L U R R D R U L D R U L D R L U R D L
U L U R D R U U R U L D U L D L U R D R U R D L D D

The continuous sequence has 52 launches.

Key checkpoints
After action	Agent	Loose crates	Anchored crates
7	(3,8)	{(3,5),(4,3),(4,8)}	{}
18	(7,5)	{(4,3),(4,8),(7,1)}	{}
30	(5,5)	{(4,3),(7,1),(7,5)}	{}
33	(7,5)	{(4,3),(7,1),(7,2)}	{}
37	(7,1)	{(4,3),(7,2)}	{(3,1)}
38	(7,2)	{(4,3),(7,9)}	{(3,1)}
40	(4,1)	{(4,3),(7,9)}	{(3,1)}
41	(4,3)	{(4,5),(7,9)}	{(3,1)}
46	(4,5)	{(7,5),(7,9)}	{(3,1)}
51	(7,5)	{(7,8),(7,9)}	{(3,1)}
52	(7,7)	{(7,8),(7,9)}	{(3,1)}
Validation

The socket must be used to solve the authored instance, but socket occupancy is not itself part of the validator. Success is still only agent == (7,7).

Necessity and anti-cleanliness check

No solution exists without anchoring a crate. All three loose crates move during the reference solution, and removing any one makes the level unsolvable.

Every listed internal wall changes solvability or changes the verified minimum route.

The dependencies are not a linear checklist:

C1 must remain loose long enough to stop C3.

Anchoring C1 then frees C3.

The anchored form of C1 enables the approach to C2.

C3, moved away after losing its original support, becomes C2’s final backstop.

Premature anchoring exchanges a still-needed mobile relationship for a fixed obstacle and destroys the intended pushing angles.

Level 7 — Build the Rail
Purpose

Use the established rules to construct a chain of stopping points. An anchored crate positions one loose crate; that loose crate positions another; the second loose crate finally positions the agent.

Initial state

Documentation labels:

C1 = (3,6)

C2 = (4,3)

C3 = (4,8)

Board size: 11 × 11
Walls: full perimeter plus
       {(1,4), (3,3), (5,3), (3,4), (5,4), (6,4),
        (8,5), (9,5), (2,6), (4,6),
        (6,8), (1,9), (4,9), (8,9)}
Agent: (3,9)
Loose crates: {(3,6), (4,3), (4,8)}
Sockets: {(3,1)}
Anchored crates: {}
Exit: (6,1)
Selection/cursor: none
Status: ACTIVE
###########
#..S..X...#
#.....#...#
#..#C#....#
##.#.##...#
#.......###
#.#C#.....#
#.........#
#...C.#...#
##.A#...#.#
###########
Player-facing discovery

The exit is on the upper rail at (6,1). Merely reaching the top row is easy; stopping at exactly x=6 is not. No wall or crate initially occupies the required neighboring cell.

The player must manufacture a stopping chain.

Phase 1: preserve the first crate as a loose backstop

C1 is moved from (3,6) through (3,5) and (7,5) to (7,1).

C3 is then routed through (5,8), (5,5), and (7,5), and pushed upward. C1 stops it at (7,2).

As in Level 6, anchoring C1 before this point would remove the loose backstop needed to place C3.

Phase 2: create the first permanent support

Approaching from the right, the player pushes C1 left. It slides into the socket at (3,1) and anchors.

The permanent stopper at (3,1) now allows the agent to stop at (4,1). From there, the player can push C2 downward from (4,3) to (4,5).

Phase 3: place the first loose link

C3, currently at (7,2), is pushed upward to (7,1) and then pushed left.

The anchored crate at (3,1) stops C3 at (4,1):

anchored C1 at x=3
loose C3 at x=4

At this moment the agent can move left from (7,1) only to (5,1). It crosses the exit at (6,1) but does not stop there, visibly reiterating the exact-stop rule from Level 1.

Phase 4: place the second loose link

C2 is moved from (4,5) across to (7,5) and then pushed up to (7,1).

From the right, the player pushes C2 left. C3 at (4,1) stops it at (5,1):

anchored C1 at x=3
loose C3 at x=4
loose C2 at x=5
exit at x=6
agent at x=7

The last LEFT action cannot push C2, because C3 is immediately behind it. The agent therefore stops at (6,1), completing the level.

The final construction is a physical dependency chain:

anchored crate → stops C3 → C3 stops C2 → C2 stops agent on exit
Reference solution
U D L U L U R R D R U L D R U L D R L U R D L U L U R D R U U R
U L L D U R D L U R U L L D L U R D R L D R D R U U R U L L

The sequence is continuous and contains 62 launches. There is one other minimum-length route, but both minimum solutions construct the same final stopping chain.

Key checkpoints
After action	Agent	Loose crates	Anchored crates	Meaning
15	(7,5)	{(4,3),(4,8),(7,1)}	{}	C1 preserved as top backstop
30	(7,5)	{(4,3),(7,1),(7,2)}	{}	C3 placed immediately below C1
34	(7,1)	{(4,3),(7,2)}	{(3,1)}	First permanent support built
35	(4,1)	{(4,3),(7,2)}	{(3,1)}	Anchor supplies the exact approach to C2
36	(4,3)	{(4,5),(7,2)}	{(3,1)}	C2 enters its staging lane
44	(7,1)	{(4,1),(4,5)}	{(3,1)}	Anchor stops C3 at (4,1)
45	(5,1)	{(4,1),(4,5)}	{(3,1)}	Agent crosses exit but cannot yet stop there
51	(4,5)	{(4,1),(7,5)}	{(3,1)}	C2 reaches the vertical launch column
57	(7,5)	{(4,1),(7,1)}	{(3,1)}	C2 reaches the top rail
61	(7,1)	{(4,1),(5,1)}	{(3,1)}	C3 stops C2 beside the exit
62	(6,1)	{(4,1),(5,1)}	{(3,1)}	Final crate pair stops the agent on the exit
Validation

Only the final agent position is checked. The visible chain is not an arbitrary configuration requested by a checker; it is the physical mechanism required to make the local exit stop possible.

Necessity and anti-cleanliness check

No solution exists without socket capture. Removing any crate makes the level unsolvable. Every listed internal wall alters solvability or minimum solution length.

The level is not a sequence of independently completed errands:

C1 cannot be anchored until it has positioned C3.

Anchoring C1 simultaneously removes one support and creates another.

C3 must be staged below C1, later moved onto the upper rail, and then preserved as a backstop.

C2 must be staged before C3 permanently occupies (4,1), but it cannot be placed in its final cell until after that occupation.

The exit is crossed unsuccessfully in the penultimate construction phase, giving immediate evidence that the chain is still one link short.

The wall at (3,3) prevents the central crate from being routed through the same clean approach used in Level 6. This forces the two loose crates through offset routes and prevents the solution from collapsing into a repeated recipe.

6. Replay-test expectations

These final states are implementation test vectors for the listed reference paths. They are not additional win requirements.

Level	Final agent	Final loose crates	Final anchored crates
1	(3,1)	{}	{}
2	(4,5)	{(5,1)}	{}
3	(5,1)	{}	{(4,1)}
4	(5,1)	{(6,1),(7,1)}	{}
5	(6,3)	{(1,5),(2,8),(6,4)}	{}
6	(7,7)	{(7,8),(7,9)}	{(3,1)}
7	(6,1)	{(4,1),(5,1)}	{(3,1)}
7. Required implementation tests

An implementation should pass at least the following deterministic tests.

Exit traversal

An agent that crosses the exit but settles elsewhere does not complete the level. Level 1 action 1 and Level 7 action 45 are authored test cases.

Vacated-square settlement

When an agent pushes a crate, its final cell is the crate’s original cell regardless of how far the crate travels. Level 2 action 1 is the minimal authored test.

Socket priority

A moving crate entering a socket anchors immediately before evaluating open cells beyond the socket. Level 3 action 1 must end with the crate at (4,1), not (6,1).

No chain pushing

An agent cannot push a loose crate when another crate is immediately behind it. The agent instead stops before the first crate.

Crate-on-crate stopping

A moving crate stops in the cell immediately before another crate. It does not push, overlap, replace, or pass through that crate. Level 4 action 24 must place the moving crate at (6,1).

Anchored collision

An anchored crate behaves as a blocker from every direction for both the agent and moving crates. It never returns to loose state before a level reset.

Open-socket traversal

The agent may traverse and rest on an unoccupied socket without changing it. Only a moving crate triggers capture.

Restart

SPACE restores all static and dynamic values to the authored initial state:

agent = initial_agent
loose_crates = initial_loose_crates
anchored_crates = {}
status = ACTIVE
Click behavior

Every click coordinate returns the unchanged settled frame. There are no hit regions, hover states, selected objects, or click animations.

Settled-frame completeness

Given a settled rendered frame, an observer can determine:

Agent position.

Every loose-crate position.

Every anchored-crate position.

Which sockets remain open.

Exit position.

Wall and ice geometry.

Whether the level is complete.

No strategically relevant value exists outside that visible state.
