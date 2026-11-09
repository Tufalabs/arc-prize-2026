Bumper Circuit

Exact authored level count: 7

This is a game about striking switches becoming difficult because each successful strike permanently changes the stopping geometry for later shots.

A puck is always stationary between actions. An arrow key launches it, and it keeps sliding until a solid object stops it. Diagonal bumpers bend trajectories without stopping the puck. Switches are solid targets: striking one parks the puck immediately beside it, lights its lamp permanently, and may raise a gate or rotate a bumper.

The puzzle is therefore not merely “reach every switch.” A completed switch may create the only useful stop for another shot, close a route that had to be used first, or change where a familiar banked launch goes.

1. Player-facing rules

The board visibly contains:

Entity	Physical role
Puck	The object controlled by arrow launches.
Wall	Permanent solid blocker.
Switch	Permanent solid target. Its central lamp shows unlit or lit.
Static bumper	Passable diagonal reflector with a fixed orientation.
Rotating bumper	Reflector whose bright diagonal rotates once when its linked switch is first struck. A faint alternate diagonal previews its future orientation.
Flat gate	Visible floor plate that the puck can cross.
Raised gate	Solid blocker created when its linked switch is first struck.
Cable	Non-colliding visual connection between a switch and the obstacle it changes.

The objective is visible both on the board and in the upper HUD: light every switch lamp.

There are no hidden variables, timed changes, enemies, or autonomous updates. Every meaningful board state can be read from a settled frame:

puck position;

lit and unlit switch lamps;

flat and raised gates;

current bumper diagonals;

remaining action budget.

Only the arrow actions are meaningful. SPACE and clicking any cell are deliberately unsupported idle actions; they never interact with an object.

2. Logical board and rendering contract
Coordinate system

Each level specifies a logical board of width W and height H.

Coordinates are zero-indexed (x,y).

(0,0) is the upper-left cell.

x increases rightward.

y increases downward.

Every perimeter cell—x=0, x=W-1, y=0, or y=H-1—is an implicit wall.

Level wall lists contain only additional internal wall cells.

Every unspecified interior cell is floor unless occupied by another listed object.

No internal cell grid or decorative border is rendered. The only visible divisions are physical walls and objects.

Mapping to the 64×64 canvas

Every logical cell occupies a 3×3 pixel block.

cell_size = 3
board_origin_x = floor((64 - 3W) / 2)
board_origin_y = 62 - 3H

Thus:

a 15×13 board occupies pixels x=9..53, y=23..61;

a 19×17 board occupies pixels x=3..59, y=11..61.

The logical cell center used for the puck, lamps, and cable routing is:

pixel_center(x,y) =
    (board_origin_x + 3x + 1,
     board_origin_y + 3y + 1)

Pixels outside the board and HUD use color 5.

Render palette use

Only the supplied integer color IDs are used.

ID	Use
0	Small white highlight and cable bridge highlight
1	Bright neutral metal, unlit lamp
2	Quiet cable, wall bevel, flat-gate rail, future bumper marks
3	Wall and switch body
4	Traversable floor
5	Outside-board background
6	Magenta link group, consistently used for switch A links
8	Invalid-action or failure pulse
9	Puck body
10	Puck highlight and D-link group
11	Lit switch lamp and completed HUD pip
12	Current bumper diagonal and E-link group
13	Raised gate body
14	Completion pulse and B-link group
15	F-link group
Core 3×3 silhouettes

These matrices show color IDs. They are schematic; exposed wall bevels tile continuously across adjacent wall cells.

Puck

4 10 4
10 9 10
4 10 4

Unlit switch

3 2 3
2 1 2
3 2 3

A lit switch changes only its center from 1 to 11. The switch remains solid.

Rotating \ bumper

12 4 2
4 12 4
2 4 12

The complete orange diagonal is current. The two faint opposite corners preview /. After rotation, the bright and faint diagonals exchange roles. A static bumper omits the two faint future marks.

Flat gate

2 4 2
4 1 4
2 4 2

It is visibly a floor plate but is entirely passable.

Raised gate

13 13 13
13  1 13
13 13 13

It occupies the whole logical cell as a solid blocker.

Cables

A cable is a one-pixel-wide polyline joining logical cell centers. Level descriptions provide exact logical waypoints.

Cable segments are drawn above floor but below walls and gameplay entities.

Both endpoints have a one-pixel collar in the cable’s link color.

Cables have no collision or state-machine effect.

Authored cable routes do not cross one another.

When a linked switch activates, the cable brightens for one presentation frame.

Documentation IDs such as A, B, and G1 are never rendered as text.

HUD

The upper edge contains:

A one-pixel action-budget bar: color 2 background and color 10 remaining portion.

One 3×3 lamp pip per switch, centered below the bar. Unlit pips use center color 1; lit pips use center color 11.

Pips are only a count/progress representation. Their individual identity is not needed because the natural objective is to light every visible board switch.

3. Exact state machine
Static level data
W, H
walls: set[(x,y)]
switches: switch_id -> (x,y)
bumpers: bumper_id -> ((x,y), initial_orientation)
gates: gate_id -> (x,y)
links: switch_id -> obstacle_id
start_position
initial_budget

Every linked obstacle has exactly one source switch in these seven levels.

Mutable state
puck_position: (x,y)
lit_switches: set[switch_id]
remaining_actions: integer
terminal: PLAYING | WON | FAILED
cursor: null
selected_object: null

Gate and bumper states may be stored explicitly or derived:

gate_raised[g] = source_switch(g) in lit_switches

bumper_orientation[b] =
    alternate(initial_orientation[b])
        if source_switch(b) in lit_switches
    else initial_orientation[b]

alternate("/") = "\" and alternate("\") = "/".

All switches begin unlit, all gates begin flat, and all linked rotating bumpers begin in their listed initial orientation.

Action semantics
Arrow actions

UP, DOWN, LEFT, and RIGHT launch the resting puck in the corresponding direction.

The puck does not move one cell and await another action. One arrow resolves the entire trajectory.

SPACE and click

SPACE and clicking any rendered cell:

do not move the puck;

do not select anything;

do not activate or pulse a particular object;

decrement the action budget like any other step() call.

There are no clickable cells, hover states, or hidden click affordances.

Collision and trajectory resolution

For an arrow action, use the obstacle state that existed at the beginning of the launch.

Repeatedly inspect the next logical cell:

Wall or raised gate: do not enter it. Stop in the current cell.

Switch: do not enter it. Stop in the current cell and register a strike on that switch.

Floor or flat gate: enter normally.

Bumper: enter its cell, immediately reflect the direction, and continue without stopping.

A bumper does not itself stop the puck. If the cell immediately after reflection is solid, the puck settles on the bumper cell.

Reflection mapping
Incoming	/ outgoing	\ outgoing
UP	RIGHT	LEFT
RIGHT	UP	DOWN
DOWN	LEFT	RIGHT
LEFT	DOWN	UP

An immediately adjacent switch is a valid strike even though the puck does not change cells. An immediately adjacent wall or raised gate is merely a blocked launch.

Already lit switches remain solid. Striking one again parks the puck normally but performs no additional activation.

Per-step update order

Every call to step(action) resolves in this exact order:

If the level is terminal, ignore the action and return the terminal frame.

Decrement remaining_actions by one.

For an arrow, calculate the full trajectory using the pre-action gate and bumper states.

Settle the puck.

If the collision was with an unlit switch:

add it to lit_switches;

light its board lamp and HUD pip;

permanently activate its linked obstacle, if any.

Check whether all level switches are lit.

If all are lit, set terminal=WON.

Otherwise, if the action budget is now zero, set terminal=FAILED.

Winning has priority over budget failure on the final action.

A linked obstacle changes only after the puck has settled at the switch. Therefore, a shot that strikes a bumper-rotation switch uses the bumper’s old orientation for its entire trajectory.

Trajectory-loop invariant

The authored boards have been checked over every reachable settled state and arrow action. No reachable launch repeats a directed state (cell,direction), so no infinite trajectory can occur.

An implementation should assert this property when loading the authored data.

Other load-time assertions

The implementation should also assert:

no static objects overlap;

the puck starts on ordinary floor;

no reachable switch activation raises a gate underneath the settled puck;

every listed reference sequence wins exactly on its final arrow;

no strict prefix of a reference sequence wins.

4. Animation and feedback

Animation is presentation-only. It does not create autonomous state ticks and never consumes additional actions.

Sliding

Partition the calculated path into maximal straight segments.

For each segment:

render one transit frame with a color-10 centerline trail across the traversed cells;

render the puck at that segment’s endpoint;

if the endpoint is a bumper, brighten its current diagonal to color 0 for one additional reflection frame.

After the final segment, render one settled frame without the trail.

A one-cell move can render immediately without a trail frame.

Switch and linked-object activation

After the settled frame:

Switch center changes to color 11.

Cable brightens for one frame.

A gate uses two deterministic transition frames:

lower half color 13;

complete raised-gate shape.

A rotating bumper uses two deterministic transition frames:

both diagonals visible, old one color 12, new one color 1;

new diagonal color 12, former diagonal reduced to faint color-2 corner marks.

The final transformed state is then shown. Player input is neither accepted nor queued during these frames.

Blocked and terminal feedback

A wall- or gate-blocked launch leaves the puck still and pulses the blocking object with color 8 for two frames.

Striking an already lit switch pulses its lamp with color 11.

Winning produces a three-frame color-14 outer-canvas pulse while preserving the final board state.

Budget failure produces a three-frame color-8 pulse.

5. Success condition

A level succeeds exactly when:

lit_switches == set(all authored switch IDs in that level)

Relevant:

every authored switch lamp.

Ignored:

final puck position;

final gate count;

final bumper orientation;

whether the last switch was linked;

whether lit switches have subsequently been struck again.

Allowed:

blocked launches;

revisiting earlier areas;

crossing a flat gate multiple times;

colliding with already lit switches.

There is no requirement to occupy an exit cell or restore any obstacle to a particular state.

6. Progression summary

The reference lengths below are verified shortest arrow-only solutions over the complete settled-state graph.

Level	Board	Main discovery or composition	Shortest launches	Budget
1	15×13	Sliding, wall stopping, visible all-lit objective	2	64
2	15×13	A bumper bends one continuous shot	5	96
3	15×13	A crossed flat gate later becomes a useful stopper	8	128
4	15×13	Two independently reachable switch branches must converge	30	256
5	19×17	The same bumper launch reaches opposite targets before and after rotation	41	384
6	19×17	Three independent branches construct the stops needed for the rotation switch	50	512
7	19×17	Nested impact staging and four permanent geometry changes	63	640
7. Authored levels
Level 1 — Catch the Wall
Player-facing discovery

The puck and one unlit switch are the only non-wall objects. They are not aligned. A rightward launch visibly slides until the boundary catches the puck; the resulting vertical alignment makes the second action obvious.

The switch and the lone unlit HUD pip illuminate together, immediately demonstrating the objective.

There are no decoys, cables, bumpers, gates, or unexplained simultaneous changes.

Exact instance
Board: 15×13
Budget: 64
Start puck: (3,3)
Internal walls: {}
Switches:
  A: (13,8), unlinked
Bumpers: {}
Gates: {}
Cursor/selection: null
Verified solution
RD
Action	Result
R	Puck stops at (13,3) before perimeter wall (14,3).
D	Puck strikes A at (13,8), settles at (13,7), lights the sole lamp, and wins.
Attribution and necessity

The first action changes only puck position and visibly demonstrates wall stopping. The second action changes only the target lamp and terminal feedback.

The switch cannot be reached in one launch because it shares neither row nor column with the start. Wrong first launches remain recoverable within the generous budget.

Level 2 — The Corner Shot
Player-facing discovery

The switch is visibly one cell above a \ bumper, but no ordinary cardinal lane reaches it. Two small wall blocks let the player construct a launch position on the bumper’s row.

At the local impasse (8,6), launching left appears to pass underneath the target. Instead, the puck enters the bright diagonal bumper, visibly turns upward, and strikes the switch in the same action.

This isolates reflection as the only new causal unknown.

Exact instance
Board: 15×13
Budget: 96
Start puck: (8,1)

Internal walls:
  {(7,11), (8,5)}

Switches:
  A: (3,5), unlinked

Static bumpers:
  M1: (3,6), orientation "\"

Gates: {}
Cursor/selection: null
Verified solution
RDLUL
Action	Settled puck
R	(13,1)
D	(13,11)
L	(8,11), stopped by wall (7,11)
U	(8,6), stopped by wall (8,5)
L	Enters M1 at (3,6), reflects upward, strikes A, and remains at (3,6)

The final micro-path is:

(8,6) → (7,6) → (6,6) → (5,6) → (4,6)
→ (3,6)[M1 "\"]
→ attempted (3,5)[A]
Attribution and necessity

Without M1, no reachable resting position offers a cardinal collision with A. Removing either internal wall destroys the launch position needed for the banked shot. The bumper is therefore not an optional shortcut or decorative object.

Level 3 — Cross It, Then Build It
Player-facing discovery

A flat gate is visibly embedded in the top lane. The intended route crosses it while it is flat, proving that it is passable rather than a wall.

The first switch has a direct magenta cable to that plate. Striking the switch visibly raises the gate after the puck settles. The second switch then presents a local alignment problem: ordinary boundary stops cannot place the puck on its column, but the newly raised gate can.

The same cell is therefore experienced first as traversable floor and later as a solid stopping surface.

Exact instance
Board: 15×13
Budget: 128
Start puck: (11,11)

Internal walls:
  {(3,5)}

Switches:
  A: (2,9), linked to G1
  B: (7,5), unlinked

Static bumpers:
  M1: (1,5), orientation "\"

Gates:
  G1: (6,1), initially flat

Link:
  A -> G1
  color: 6
  cable waypoints: [(2,9), (6,9), (6,1)]

Cursor/selection: null
Verified solution
ULDDRULD
Phase 1: cross the flat gate and raise it
ULDD

U: (11,11) → (11,1)

L: slides through flat G1=(6,1) and stops at (1,1)

D: enters M1=(1,5), reflects right, and stops at (2,5) before wall (3,5)

D: strikes A, settles at (2,8), and raises G1

Phase 2: use the raised gate
RULD

R: (2,8) → (13,8)

U: (13,8) → (13,1)

L: raised G1 stops the puck at (7,1)

D: strikes B and settles at (7,4)

Attribution and necessity

The flat gate is crossed before any activation, so it cannot be mistaken for a permanently solid wall. Its rise occurs only after A is struck and is shown by lamp, cable, and gate animation.

Without the bumper, A is unreachable. Without the wall, the banked staging position disappears. Without the gate activation, B is unreachable.

Level 4 — Two Branches, One Approach
Player-facing discovery

No new object rule is introduced. Instead, the known rules form the first dependency network.

Switches A and B are each reachable from the initial state. Each raises a different gate. The final switch C is not merely “after” them in a corridor: its approach requires a banked shot followed by two orthogonal landings, one supplied by each gate.

The player may take A or B first. The shorter route takes B first, but the alternate order remains valid, demonstrating that the level is not a single hard-coded recipe.

Exact instance
Board: 15×13
Budget: 256
Start puck: (13,2)

Internal walls:
  {
    (1,5), (3,1), (4,3), (6,5), (6,8), (6,11),
    (9,6), (9,11), (10,7), (11,1), (12,2)
  }

Switches:
  A: (10,3), linked to G1
  B: (5,3),  linked to G2
  C: (3,9),  unlinked

Static bumpers:
  M1: (9,2), orientation "\"

Gates:
  G1: (7,7), initially flat
  G2: (1,10), initially flat

Links:
  A -> G1
    color: 6
    waypoints: [(10,3), (7,3), (7,7)]

  B -> G2
    color: 14
    waypoints: [(5,3), (1,3), (1,10)]

Cursor/selection: null
Verified shortest solution
DLULULDLURDRURDLULURDULDRLDLDR
Phase 1 — branch B

Actions 1–13:

DLULULDLURDRU

The puck strikes B, settles at (5,4), and raises G2.

Phase 2 — branch A

Actions 14–21:

RDLULURD

The puck strikes A, settles at (10,2), and raises G1.

Phase 3 — convergence on C

Actions 22–30:

ULDRLDLDR

The important final construction is:

R from (4,2) enters M1=(9,2).

\ turns the puck downward.

Wall (9,6) catches it at (9,5).

L places it at (7,5).

Raised G1=(7,7) catches a downward launch at (7,6).

L reaches (1,6).

Raised G2=(1,10) catches a downward launch at (1,9).

R strikes C and leaves the puck at (2,9).

Alternate branch order

An A-first win exists in 32 launches:

DLULURDULDLURDRURDLULDRULDRLDLDR

Thus both branches are genuinely independently accessible. The final switch is newly strikeable only when both gate-producing switches are lit.

Anti-degenerate check

The static bumper and both gates are required. Removing any internal wall either makes the puzzle unsolvable or opens a shorter bypass that avoids the intended two-branch convergence. The irregular wall placement encodes stopping geometry rather than decorative density.

Level 5 — Before and After
Player-facing discovery

This level introduces the rotating bumper.

The central bumper R1 begins as \ and visibly carries faint / corner marks. Its magenta cable leads to switch A.

Two gate-building branches, B and D, can be approached independently. Both gates are required to reach A. Before reaching A, the player must launch through R1 in its initial orientation to hit B. Striking A then rotates R1, immediately and visibly.

Later, the exact same launch from (7,8) through the same bumper goes in the opposite vertical direction and strikes C. The level cannot be explained as a teleporter, a one-use arrow, or a simple unlock: the same physical reflection rule is observed in both orientations.

Exact instance
Board: 19×17
Budget: 384
Start puck: (9,15)

Internal walls:
  {
    (6,15), (10,2), (10,8), (11,6), (11,12),
    (12,1), (12,11), (13,7), (17,6), (17,14)
  }

Switches:
  A: (8,12), linked to R1
  B: (9,14), linked to G1
  C: (9,2),  unlinked
  D: (4,10), linked to G2

Rotating bumpers:
  R1: (9,8), initial orientation "\"

Gates:
  G1: (12,9), initially flat
  G2: (7,7), initially flat

Links:
  A -> R1
    color: 6
    waypoints: [(8,12), (9,12), (9,8)]

  B -> G1
    color: 14
    waypoints: [(9,14), (12,14), (12,9)]

  D -> G2
    color: 10
    waypoints: [(4,10), (7,10), (7,7)]

Cursor/selection: null
Verified shortest solution
LURDRULDLDLRULDLURRULULDLDLULDRURDRULDLUR
Actions	Sequence	New switch	Settled puck
1–11	LURDRULDLDL	D	(5,10)
12–18	RULDLUR	B	(9,13)
19–30	RULULDLDLULD	A	(8,11)
31–41	RURDRULDLUR	C	(9,3)
The distinguishing bumper observations

Before A:

puck (7,8)
RIGHT
→ (8,8)
→ R1 (9,8), orientation "\"
→ turns DOWN
→ stops at (9,13) against B=(9,14)

After A:

puck (7,8)
RIGHT
→ (8,8)
→ R1 (9,8), orientation "/"
→ turns UP
→ stops at (9,3) against C=(9,2)
Dependency and necessity

B and D may each be the first activated switch. A is newly strikeable only after both are lit. C is newly strikeable in a winning state only after A, B, and D.

Replacing R1 with a permanently fixed \ bumper makes the level unsolvable. Removing either gate makes the level unsolvable. The rotating bumper is therefore both a pre-activation tool and a post-activation tool, with no rule exception.

Level 6 — The Gate You Already Crossed
Player-facing discovery

The same arena is extended with switch E and gate G3.

On the first action, the puck crosses G3=(6,15) while it is flat. Much later, striking E raises it. The player must then approach the same cell from the opposite side and use it as a permanent stop.

This is not a new gate rule. It is a stronger contextual use of the rule established in Level 3:

first, preserve and cross the flat geometry;

then activate it;

later, exploit the obstruction it creates.

E, D, and B form three independently approachable branches. Their three raised gates jointly construct the route to rotation switch A.

Exact instance
Board: 19×17
Budget: 512
Start puck: (9,15)

Internal walls:
  {
    (10,2), (10,8), (11,6), (11,12),
    (12,1), (12,11), (13,7), (17,6), (17,14)
  }

Switches:
  A: (8,12), linked to R1
  B: (9,14), linked to G1
  C: (9,2),  unlinked
  D: (4,10), linked to G2
  E: (17,9), linked to G3

Rotating bumpers:
  R1: (9,8), initial orientation "\"

Gates:
  G1: (12,9), initially flat
  G2: (7,7), initially flat
  G3: (6,15), initially flat

Links:
  A -> R1
    color: 6
    waypoints: [(8,12), (9,12), (9,8)]

  B -> G1
    color: 14
    waypoints: [(9,14), (12,14), (12,9)]

  D -> G2
    color: 10
    waypoints: [(4,10), (7,10), (7,7)]

  E -> G3
    color: 12
    waypoints: [(17,9), (16,9), (16,15), (6,15)]

Cursor/selection: null
Verified shortest solution
LURDRULDLDRULURDRULDRDLURUURDRULDLDLULDRURDRULDLUR
Actions	Sequence	New switch	Settled puck
1–12	LURDRULDLDRU	E	(17,10)
13	L	D	(5,10)
14–25	URDRULDRDLUR	B	(9,13)
26–39	UURDRULDLDLULD	A	(8,11)
40–50	RURDRULDLUR	C	(9,3)
Important state-dependent moments

Action 1 crosses flat G3 while moving left across row 15.

Action 12 strikes E, raising G3.

Later, a leftward launch from the eastern side stops at (7,15) because raised G3 occupies (6,15).

From (7,15), G2 creates the vertical stop (7,8).

The initial \ bumper shot from there reaches B.

After all three branches are complete, their gates construct the staging sequence for A.

A rotates R1, and the familiar launch reaches C.

Dependency and necessity

E, D, and B can each be activated first in some valid partial route. A is newly strikeable only after all three are lit. C then requires the rotated geometry.

All three gates and the bumper rotation are individually necessary for completion. The reference path revisits the eastern loop because the same approach has a different meaning after each permanent activation; the repeated space is not a corridor-lengthening device.

Level 7 — Full Circuit
Player-facing discovery

The final level adds switch F and gate G4 without adding any new object category.

E must first raise G3, which creates the landing needed to strike F. F raises G4 immediately to the right of the central bumper. B and D raise the other two interior staging gates.

The route to A now requires a constructed three-stop staircase:

G1 fixes the correct row.

G4 fixes the adjacent bumper-side column.

G2 fixes the final approach to A.

Striking A finally rotates R1, after which the player must deliberately rebuild the familiar launch position and send the puck upward to C.

The objects are not encountered in the simple order in which they are listed. The level repeatedly returns to the same center and bottom regions under different permanent geometry.

Exact instance
Board: 19×17
Budget: 640
Start puck: (9,15)

Internal walls:
  {
    (10,2), (11,6), (12,1), (12,11),
    (13,7), (17,6), (17,14)
  }

Switches:
  A: (8,12), linked to R1
  B: (9,14), linked to G1
  C: (9,2),  unlinked
  D: (4,10), linked to G2
  E: (17,9), linked to G3
  F: (7,4),  linked to G4

Rotating bumpers:
  R1: (9,8), initial orientation "\"

Gates:
  G1: (12,9), initially flat
  G2: (7,7), initially flat
  G3: (6,15), initially flat
  G4: (10,8), initially flat

Links:
  A -> R1
    color: 6
    waypoints: [(8,12), (9,12), (9,8)]

  B -> G1
    color: 14
    waypoints: [(9,14), (12,14), (12,9)]

  D -> G2
    color: 10
    waypoints: [(4,10), (7,10), (7,7)]

  E -> G3
    color: 12
    waypoints: [(17,9), (16,9), (16,15), (6,15)]

  F -> G4
    color: 15
    waypoints: [(7,4), (10,4), (10,8)]

Cursor/selection: null
Verified shortest solution
LURDRULDLDRUDLURDRULDRDLURULDLDLURDRULDRDLURUURDRULDLDLULDRDLUR
Actions	Sequence	New switch	Settled puck
1–12	LURDRULDLDRU	E	(17,10)
13–25	DLURDRULDRDLU	F	(7,5)
26–32	RULDLDL	D	(5,10)
33–44	URDRULDRDLUR	B	(9,13)
45–58	UURDRULDLDLULD	A	(8,11)
59–63	RDLUR	C	(9,3)
Critical construction sequence

The gate-building events are:

E raises G3=(6,15).

Raised G3 catches the puck at (7,15), permitting an upward strike on F=(7,4).

F raises G4=(10,8).

D raises G2=(7,7).

B raises G1=(12,9).

During actions 54–58:

G1 stops the puck at (12,8).

A left launch stops at (11,8) against G4=(10,8).

The puck moves to (11,7).

G2=(7,7) stops the left launch at (8,7).

DOWN strikes A=(8,12) from the exact constructed column.

A rotates R1 from \ to /.

Actions 59–63 then return through the changed circuit:

RDLUR

The final RIGHT launch from (7,8) enters rotated R1, turns upward, and strikes C=(9,2) from (9,3).

Dependency and anti-degenerate check

The mature dependency structure is:

E -> F
B ─┐
D ─┼─> A -> rotated final approach to C
E ─┤
F ─┘

B and D remain independently approachable branches. F is unreachable before E has created the relevant bottom stop. A is newly strikeable only after B, D, E, and F are all lit.

Every gate and the bumper rotation is individually necessary. Replacing R1 with a permanently fixed bumper makes the level unsolvable. Removing any remaining internal wall either destroys a required landing or opens a shortcut that collapses the dependency structure.

8. Replay acceptance tests

An implementation should pass the following exact tests.

Level	Arrow sequence	New-switch order	Final puck
1	RD	A	(13,7)
2	RDLUL	A	(3,6)
3	ULDDRULD	A, B	(7,4)
4	DLULULDLURDRURDLULURDULDRLDLDR	B, A, C	(2,9)
5	LURDRULDLDLRULDLURRULULDLDLULDRURDRULDLUR	D, B, A, C	(9,3)
6	LURDRULDLDRULURDRULDRDLURUURDRULDLDLULDRURDRULDLUR	E, D, B, A, C	(9,3)
7	LURDRULDLDRUDLURDRULDRDLURULDLDLURDRULDRDLURUURDRULDLDLULDRDLUR	E, F, D, B, A, C	(9,3)

For every row:

the sequence must first satisfy the all-lit validator on its final arrow;

every strict prefix must remain nonterminal;

all returned gates and bumper orientations must reflect the final lit-switch state;

the final switch’s linked transformations, when present, must still be applied before the terminal frame is returned.
