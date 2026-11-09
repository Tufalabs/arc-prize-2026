Beam Paths

This is a game about aligning a beam becoming difficult because every reroute can revoke the access created by the previous alignment.

Beam Paths contains exactly 7 authored levels.

The player walks through compact rooms containing a single completely visible beam network. Bumping a mirror or pipe rotates it clockwise. A receiver opens its matching door only while the live beam reaches it. The resulting puzzle is not “set every component correctly”; the useful alignment changes as the player moves between regions. A route may first open a door, then need to be deliberately destroyed from the far side so that the same beam can open the next route.

There are no hidden connectivity checks. Every illuminated segment is drawn continuously from the emitter to its endpoint, including reflections and pipe interiors. When a route fails, the first stopping point is marked by a bright impact spark.

1. Action space

The environment exposes the required action space:

UP, DOWN, LEFT, RIGHT

SPACE

CLICK(x, y) on any rendered cell

Only the four directional actions have gameplay effects.

Directional action

Let q be the logical cell adjacent to the player in the chosen direction.

If q contains a rotatable mirror or pipe and is not separated by a closed door, the component rotates clockwise by one quarter-turn. The player remains in place.

Otherwise, if q is ordinary floor, an open door, or the exit, the player moves into q.

Otherwise, the action is rejected and the player remains in place.

Emitters and receivers are solid and cannot be rotated.

Unused actions

SPACE and every CLICK(x, y) are state-preserving no-ops. There is no cursor, selected object, or clickable region. Nothing in the render suggests clickability.

Exploration allowance

There is no move limit, timer, energy meter, death state, or irreversible failure. Every reachable nonterminal state in all seven authored levels retains a route to the exit.

2. Logical and rendered geometry
Logical coordinates

Every level uses a rectangular logical lattice.

Coordinates are one-based.

(1,1) is the upper-left cell.

x increases rightward.

y increases downward.

Each directional movement changes the player position by exactly one logical cell.

A wall cell is neither traversable nor optically transparent.

64×64 rendering

Each logical cell occupies a 4×4 pixel block.

For a level of logical width W and height H:

origin_x = floor((64 - 4W) / 2)
origin_y = floor((64 - 4H) / 2)

Logical cell (x,y) begins at:

pixel_x = origin_x + 4(x - 1)
pixel_y = origin_y + 4(y - 1)

The unused canvas is color 5.

Adjacent floor cells form one uninterrupted region. No grid lines or cell borders are drawn. Only authored wall cells and door structures divide the space.

Palette use
Purpose	Color ID
Outside canvas	5
Floor	1
Wall and dark casing	4, with 3 for raised component casing
Inactive pipe channel	2
Player	9
Mirror casing	7
Mirror reflective diagonal	0
Emitter core	12
Live beam	11
Beam endpoint and powered receiver center	0
Channel A receiver/door	6
Channel B receiver/door	15
Channel C receiver/door	8
Channel D receiver/door	10
Exit	14
Invalid-action pulse	13

Channel letters are documentation labels only. No letters or text appear in the rendered grid.

3. Visual entities
Player

The player occupies one logical cell and is rendered as a connected blue 4×4 ring:

Outer-ring pixels: color 9.

Central 2×2 pixels remain transparent to the underlying floor or beam.

The central opening keeps a beam visibly continuous when the player stands on it. The player never blocks, bends, absorbs, or otherwise affects a beam.

Emitter

An emitter is a solid 4×4 housing:

Casing: color 3.

Central 2×2 core: color 12.

The two center pixels on its facing edge: color 11.

Its direction is fixed and visually explicit. Every authored level has one emitter.

Two-sided diagonal mirror

A mirror is a solid pink cell with a bright diagonal:

Casing: color 7.

Reflective diagonal: color 0.

There are two effective orientations:

/

\

A clockwise quarter-turn toggles between them.

Mirrors are explicitly two-sided and always reflect an arriving orthogonal beam:

Mirror	Incoming travel	Outgoing travel
/	E	N
/	N	E
/	W	S
/	S	W
\	E	S
\	S	E
\	W	N
\	N	W

The illuminated incoming and outgoing half-segments meet continuously at the mirror center.

Straight pipe

A straight pipe is a solid casing with two visibly open opposite ends.

Effective orientations:

H: west–east

V: north–south

A clockwise quarter-turn toggles H ↔ V.

The beam passes only when its travel axis matches the open axis. Otherwise it stops against the closed casing.

Elbow pipe

An elbow pipe contains one connected L-shaped channel. Its four orientations are named by their open sides:

NE → SE → SW → NW → NE

This is also the clockwise rotation order.

A beam entering through either open side exits through the other. A beam arriving at either closed side stops at the casing.

For example, an SW elbow:

sends a beam entering from the west toward the south;

sends a beam entering from the south toward the west;

rejects beams arriving from the north or east.

Receiver

A receiver is solid and accepts a beam from any orthogonal direction.

The frame uses its channel color.

Side-center openings make it read as a socket rather than a wall.

Unpowered center: color 4.

Powered center: color 0, surrounded by a color 11 glow.

A receiver terminates the beam and powers its channel.

Door

A door occupies one logical cell in an opening through a wall.

Each door has a passage axis:

EW: player passage is left-to-right.

NS: player passage is up-to-down.

The door frame and receiver share the same channel color and the same paired-notch silhouette. This is the complete receiver–door association; there is no hidden numbering.

A door is:

open exactly when its channel is powered;

closed otherwise.

An open door is traversable. A closed door rejects entry.

Optical slit

A door contains a permanent narrow slit along its passage axis. A beam traveling along that axis passes through the door whether the player opening is retracted or closed. A perpendicular beam hits the frame and stops.

This rule is identical for every door. Level 6 is the first level whose solution materially uses the slit.

If a door loses power while the player occupies its cell, the player is not displaced. Collision is evaluated only when entering a destination cell, so the player may step out but cannot re-enter while it remains closed. No authored solution relies on this case.

Exit

The exit is a passable green frame using color 14. It is visually distinct from receivers and always lies in a terminal alcove or beyond a final door.

4. Beam state machine
Beam propagation

Each emitter creates one ray beginning at the center of its cell.

For every ray:

Advance toward the adjacent logical cell.

If the next cell is outside the board or is a wall, stop at the intervening boundary.

If the next cell is floor, the player, or the exit, continue in the same direction.

If it is a door:

continue if ray direction matches the door axis;

otherwise stop at its frame.

If it is a validly approached straight pipe, continue through it.

If it is a validly approached elbow, leave through its other open side.

If it is a pipe approached through a closed side, stop at the casing.

If it is a mirror, reflect according to the mirror table.

If it is a receiver, power that receiver’s channel and terminate.

If it is an emitter or other solid non-beam object, stop.

The beam endpoint is always shown as a persistent 2×2 white impact point edged by color 11.

To make propagation total even for non-authored arrangements, implementations must track directed states (cell, travel_direction). If a directed state repeats, propagation terminates before processing the repeated state and places the endpoint pulse at that cell. No authored level creates a loop.

Derived state

Receiver and door states are derived rather than independently mutable:

powered_channels =
    channels of all receivers reached by at least one beam

door_open(door) =
    door.channel in powered_channels

Every authored channel has one receiver, but the rule supports several receivers or doors sharing a channel.

step() resolution order

All gameplay changes occur only inside step(action).

1. Read the current player position, component orientations, and door states.

2. Resolve the action:
   a. directional bump and rotation;
   b. directional movement;
   c. invalid collision;
   d. SPACE/CLICK no-op.

3. Recompute every beam using the resulting component orientations.

4. Derive powered receivers and open doors from the new beam traces.

5. If the player now occupies the exit, latch level_complete = true.

6. Return the committed state and begin presentation-only animation.

Movement never changes beam geometry because the player is optically transparent.

At level initialization, the specified initial state and its derived beam/door state are constructed atomically before the first frame. After initialization, animations never mutate gameplay state.

5. Feedback and animation
Rotation

A bump performs one logical clockwise quarter-turn.

The logical orientation and resulting beam state are committed during step(). Inputs are then locked during a short deterministic presentation:

One frame highlights the clockwise edge of the component.

One frame replaces the old silhouette with the new orientation.

The old beam suffix disappears at the first changed component.

The new suffix is revealed outward in batches of up to four logical cells per frame.

Any receiver glow and door movement begins when the visual trace reaches the receiver.

Animation frames consume no actions.

Door motion

A changed door uses three frames:

closing: side panels extend toward the passage;

opening: panels retract into the surrounding wall;

the permanent axial slit remains visible throughout.

Logical passability already has its final value while this animation plays, but input remains locked, so visual and collision timing cannot conflict.

Invalid movement

A wall, closed door, emitter, or receiver gives:

no state change;

a two-frame color 13 pulse at the collision point;

a one-pixel player squash toward the obstacle.

Completion

Entering the exit immediately latches completion. The exit expands its green frame for four presentation frames. Receiver states, component orientations, and doors are ignored after completion.

There is no loss condition.

6. Authored-map notation

The following maps are implementation notation, not literal rendered glyphs.

#  wall
.  floor
P  initial player
E  emitter
M  mirror
I  straight pipe
L  elbow pipe
a/b/c/d  receivers for channels A/B/C/D
A/B/C/D  matching doors
X  exit

All coordinates and orientations below are authoritative if a map character is visually ambiguous.

7. Progression summary
Level	Main discovery or composition	Verified shortest solution
1	A visibly powered receiver retracts its matching door	5
2	Bumping a misaligned straight pipe rotates it and extends the beam	7
3	A diagonal mirror redirects rather than carries the beam straight	6
4	Prepare a future branch, open one door, cross, then restore the prepared branch	37
5	Elbows have closed sides and four orientations; the same elbow becomes a route switch	43
6	Stage a component behind one door, preserve it, return through another alignment, then reactivate the staged branch	60
7	Stage the final mirror from one side, build its shared prefix later, and reverse it from the opposite side	77

Counts include movement and successful bumps, and exclude optional invalid/no-op actions.

8. Authored levels
Level 1 — The already-open door

Size: 9×6

#########
#E.a#...#
#...#...#
#P..A.X.#
#...#...#
#########
Initial state
Entity	Coordinate	State
Player	(2,4)	no selection
Emitter	(2,2)	facing E
Receiver A	(4,2)	powered
Door A	(5,4)	EW, open
Exit	(7,4)	active

Initial beam:

Emitter (2,2)
→ floor (3,2)
→ receiver A (4,2)
Player-facing discovery

The first frame already contains the complete positive example:

a continuous yellow beam reaches the magenta receiver;

that receiver glows;

the identically colored door is visibly retracted;

the green exit is directly beyond it.

There are no rotatable objects, decoys, or alternate routes. The only useful hypothesis is that the open colored structure is a passage toward the green endpoint.

Exact solution
RRRRR

Action 3 enters open door A.

Action 5 enters the exit.

Attribution and necessity

The wall column at x=5 can be crossed only through door A. The level cannot be completed without traversing that door. Every visible object either demonstrates the objective or lies on the solution.

Level 2 — The beam frontier

Size: 10×6

##########
#E.I.a#..#
#..P..#..#
#.....A.X#
#.....#..#
##########
Initial state
Entity	Coordinate	State
Player	(4,3)	no selection
Emitter	(2,2)	facing E
Straight pipe	(4,2)	V
Receiver A	(6,2)	unpowered
Door A	(7,4)	EW, closed
Exit	(9,4)	active

Initial beam:

Emitter (2,2)
→ floor (3,2)
→ west casing of vertical pipe (4,2), stop
Player-facing discovery

The player begins immediately below the unfamiliar pipe. The exit route ends at a closed magenta door, and the beam has a bright endpoint on the pipe’s closed west face.

Walking upward cannot move into the solid component. Instead, the attempted movement rotates it clockwise. Its open ends visibly change from north–south to west–east, and the beam immediately extends through the interior to receiver A. The door retracts at the same moment.

The first bump is low-cost and has one unambiguous consequence. No other component can account for the change.

Exact solution
UDRRRRR

Key updates:

Action 1: bump pipe (4,2), V → H; A becomes powered.

Action 5: enter door A.

Action 7: enter exit.

Attribution and necessity

The partition at x=7 has no other opening. Without rotating the straight pipe, receiver A is never powered and the level is unsolvable.

Level 3 — Turning a corner

Size: 11×7

###########
#.....#...#
#E..MP#...#
#.....#...#
#...a.A.X.#
#.....#...#
###########
Initial state
Entity	Coordinate	State
Player	(6,3)	no selection
Emitter	(2,3)	facing E
Mirror	(5,3)	/
Receiver A	(5,5)	unpowered
Door A	(7,5)	EW, closed
Exit	(9,5)	active

Initial beam:

Emitter (2,3)
→ (3,3)
→ (4,3)
→ mirror / at (5,3)
→ north through (5,2)
→ wall boundary at (5,1), stop
Player-facing discovery

The player is directly right of the mirror but is not standing on either the old or intended beam path. This prevents player transparency from becoming a simultaneous tutorial unknown.

The visible diagonal already suggests a directional surface. Bumping left rotates / → \. The horizontal beam then visibly turns south, reaches receiver A, and opens the door.

A straight-pipe explanation cannot account for the observed right-angle path.

Exact solution
LDDRRR

Key updates:

Action 1: bump mirror (5,3), / → \; A becomes powered.

Action 4: enter door A.

Action 6: enter exit.

Attribution and necessity

The mirror is the only object capable of redirecting the beam toward (5,5). Door A is the only crossing through the x=7 partition.

Level 4 — Prepare, cross, restore

Size: 13×11

#############
#b...M#...#.#
#E...M....#.#
#.....#...#.#
#.....#...#.#
#....a#...#.#
#.....#...BX#
#.....A...#.#
#P....#...#.#
#.....#...#.#
#############
Initial state
ID	Coordinate	State
Player	(2,9)	no selection
Emitter	(2,3)	facing E
Top mirror M_top	(6,2)	/
Switch mirror M_switch	(6,3)	/
Receiver A	(6,6)	unpowered
Receiver B	(2,2)	unpowered
Door A	(7,8)	EW, closed
Door B	(11,7)	EW, closed
Exit	(12,7)	active

Initial trace:

Emitter (2,3)
→ east to M_switch (6,3)
→ north to M_top (6,2)
→ east into wall (7,2), stop

Useful configurations:

M_switch = /, M_top = \
    beam reaches receiver B

M_switch = \
    beam turns south and reaches receiver A
    M_top is not illuminated
Player-facing discovery

The path toward M_top crosses the live horizontal beam. Walking onto and off the beam leaves every segment unchanged, visibly establishing that the player is optically transparent.

The current endpoint is immediately east of M_top. Rotating M_top sends the beam west into receiver B. Door B opens on the far side of the map, but the player still cannot reach it because door A remains closed. This is positive evidence that the top mirror is a future alignment worth preserving.

Rotating M_switch then diverts the beam south to receiver A. The nearby door opens. After crossing A, the player can enter the one-cell niche at (7,3) and bump M_switch from its opposite side. The beam returns to the already-prepared B branch, closing A behind the player and opening the final door ahead.

Exact solution
UUUUURUURRRDRDDDDDRRRUUUUULLRDDDDRRRR

Key events:

Action	Event
11	M_top rotates / → \; B opens
13	M_switch rotates / → \; B closes and A opens
20	Player enters door A
28	M_switch rotates \ → /; A closes and B reopens
36	Player enters door B
37	Player enters exit
Recovery behavior

If the player crosses A without preparing M_top, rotating M_switch back to / produces a visible endpoint against the wall at (7,2). From the hub side, one further bump restores M_switch = \, reopening A. The player can return, fix M_top, and try again.

Anti-cleanliness and necessity

The top mirror is used before the door that grants access to the switch’s far side. Components are therefore not encountered in their final use order.

Both mirrors and both doors are individually necessary:

without M_top, B cannot be powered;

without M_switch, the beam cannot alternate between A and B;

A is the only start-to-hub passage;

B is the only route to the exit.

Level 5 — The four-sided elbow

Size: 15×13

###############
#......#....#.#
#......#....CX#
#b.I..M....c#.#
#......#....#.#
#......###B####
#......#....###
#E....L.....###
#......#....###
#......A....###
#P....a#....###
#......#....###
###############
Initial state
ID	Coordinate	State
Player	(2,11)	no selection
Emitter	(2,8)	facing E
B-branch straight pipe I_B	(4,4)	V
End mirror M_end	(7,4)	\
Elbow switch L_switch	(7,8)	NE
Receiver A	(7,11)	unpowered
Receiver B	(2,4)	unpowered
Receiver C	(12,4)	unpowered
Door A	(8,10)	EW, closed
Door B	(11,6)	NS, closed
Door C	(13,3)	EW, closed
Exit	(14,3)	active

Initial trace:

Emitter (2,8)
→ east through (3,8)…(6,8)
→ closed west casing of NE elbow (7,8), stop

Relevant elbow states for the west-arriving beam:

NE: west side closed; stop
SE: west side closed; stop
SW: west → south, reaching receiver A
NW: west → north, entering the B/C branch

With L_switch = NW:

beam travels north to M_end (7,4)

M_end = \:
    beam travels west through I_B to receiver B

M_end = /:
    beam travels east to receiver C
Player-facing discovery

The elbow is introduced at the exact live frontier.

The first bump changes NE → SE, but the beam still stops on the closed west side. This outcome rules out the mistaken model that every pipe orientation accepts every approach.

The second bump changes SE → SW. The west opening now lines up with the beam, the illuminated interior visibly bends downward, and receiver A powers. The new pipe is therefore demonstrated as a port-gated elbow rather than another mirror.

Later, the same elbow changes role. One further bump from SW → NW does not merely “open another door”: it preserves the incoming west connection while moving the beam from the southern A route into the northern shared branch. It is first an obstacle, then an access tool, then a branch selector.

Exact solution
UURUUUURUDDDDRRRUUDRRUULLRURRUUULLLLRURRRRR

Key events:

Action	Event
9	I_B rotates V → H
17	L_switch rotates NE → SE; beam remains blocked
18	L_switch rotates SE → SW; A opens
20	Player enters door A
25	L_switch rotates SW → NW; A closes and B opens
30	Player enters door B
36	M_end rotates \ → /; B closes and C opens
42	Player enters door C
43	Player enters exit
Blind-safe recovery

A player need not foresee the B-branch straight pipe.

If the player enters the central region with I_B = V, changing the elbow to NW makes the beam stop visibly against I_B. From the east-side elbow niche, three clockwise bumps cycle:

NW → NE → SE → SW

This restores receiver A and reopens the return door. The player can go back, align I_B, and repeat the crossing.

Anti-cleanliness and necessity

The straight pipe lies on a future route and may be staged before that route is live. The elbow is approached from different regions and has three visibly distinct strategic roles. The end mirror is changed only after the player has crossed two earlier doors.

All three rotatable components and all three doors are required for every solution.

Level 6 — Carry the alignment home

Size: 15×14

###############
#......#.#....#
#b....L..B.I.c#
#......#.#....#
#......#.#....#
#......#.#....#
#E....M..#....#
#......#.#....#
#.....a#.#....#
#......A.#....#
#P.....###....#
#......CX######
#......########
###############
Initial state
ID	Coordinate	State
Player	(2,11)	no selection
Emitter	(2,7)	facing E
Entry mirror M_entry	(7,7)	/
Branch elbow L_branch	(7,3)	SW
C-branch straight pipe I_C	(12,3)	V
Receiver A	(7,9)	unpowered
Receiver B	(2,3)	powered
Receiver C	(14,3)	unpowered
Door A	(8,10)	EW, closed
Door B	(10,3)	EW, open
Door C	(8,12)	EW, closed
Exit	(9,12)	active

Initial beam:

Emitter (2,7)
→ M_entry / at (7,7)
→ north to L_branch SW at (7,3)
→ west to receiver B (2,3)

Relevant configurations:

M_entry = \
    beam travels south to receiver A

M_entry = /, L_branch = SW
    beam travels to receiver B

M_entry = /, L_branch = SE, I_C = H
    beam travels east through door B's optical slit
    and reaches receiver C
Player-facing discovery

Receiver B and door B are visibly active on the first frame, but the player is on the wrong side of the A partition. Rotating M_entry to \ sacrifices the currently lit B route to open the local A route.

After crossing A, the player restores M_entry = /, enters B’s chamber, and rotates I_C. The pipe has no immediate effect because the elbow still points toward receiver B. This is deliberate staging of an established component, not the introduction of a hidden behavior.

Back in the hub, the player rotates the elbow through:

SW → NW → NE → SE

The two middle states reject the south-arriving beam, so the beam visibly collapses to the elbow. In SE, the beam travels east through the same door cell the player just used.

Receiver B loses power and its door closes, but the line remains uninterrupted through the permanent axial slit and reaches receiver C. The old access door has become part of the new beam route without changing its operational rule.

The player is now in the hub, whereas door C is accessible only from the starting region. They must therefore:

preserve L_branch = SE and I_C = H;

temporarily rotate M_entry back to A;

cross home;

restore M_entry = /.

The preserved downstream route immediately repowers C.

Exact solution
UUURURRRRDDDRRRUUULLRUUUURRRLLLLLLRDDDDLLRDDDLLLUUURDDDDDRRR

Key events:

Action	Event
9	M_entry rotates / → \; B closes and A opens
14	Player enters door A
20	M_entry rotates \ → /; A closes and B reopens
26	Player enters door B
28	I_C rotates V → H
29	Player returns through door B
32–34	L_branch: SW → NW → NE → SE; C opens
41	M_entry rotates / → \; C closes and A opens
46	Player returns through door A
52	M_entry rotates \ → /; C reopens
59	Player enters door C
60	Player enters exit
Recovery behavior

If I_C is still vertical when the elbow reaches SE, the beam stops visibly against the pipe. One additional elbow bump changes SE → SW, immediately restoring B and allowing another chamber visit.

Anti-cleanliness and necessity

This is not a sequence of unrelated stations:

door B is both a player passage in one phase and an optical conduit in another;

the side-chamber pipe is changed for a route that is not currently active;

receiver C can be powered while the player is on the wrong side of the map;

the solution must revisit the starting region while preserving two downstream orientations.

Every rotatable and every door is individually indispensable.

Level 7 — Preserve the longest prefix

Size: 16×15

################
#......#..#....#
#......#..B....#
#......#..#d...#
#......#..#....#
#......#..#....#
#b....L...IM####
#......#..#..#.#
#......#..#..#.#
#E....M...#c.#.#
#......#..#..#.#
#.....a#..C..#.#
#P.....A..#..DX#
#......#..#..#.#
################
Initial state
ID	Coordinate	State
Player	(2,13)	no selection
Emitter	(2,10)	facing E
Entry mirror M_entry	(7,10)	/
Branch elbow L_branch	(7,7)	SW
Shared-prefix straight pipe I_prefix	(11,7)	V
End mirror M_end	(12,7)	/
Receiver A	(7,12)	unpowered
Receiver B	(2,7)	powered
Receiver C	(12,10)	unpowered
Receiver D	(12,4)	unpowered
Door A	(8,13)	EW, closed
Door B	(11,3)	EW, open
Door C	(11,12)	EW, closed
Door D	(14,13)	EW, closed
Exit	(15,13)	active

Initial beam:

Emitter (2,10)
→ M_entry / at (7,10)
→ north to L_branch SW at (7,7)
→ west to receiver B (2,7)

Other routes:

M_entry = \
    beam reaches receiver A

M_entry = /, L_branch = SE, I_prefix = H:
    beam reaches M_end from the west

M_end = \:
    beam turns south to receiver C

M_end = /:
    beam turns north to receiver D
Spatial dependency

The three important regions are separated by actual geometry:

The start region reaches the hub only through A.

The hub reaches the north staging room only through B.

The hub reaches the lower final room only through C.

M_end blocks the only direct north–south opening on the right side.

I_prefix immediately west of M_end prevents the hub from bumping the end mirror.

Therefore M_end can be rotated only:

from its north side while inside the B room;

from its south side while inside the C room.

Player-facing discovery

The first phase repeats the now-familiar access sacrifice: rotate M_entry from the active B route to A, cross, then restore B from the hub.

Door B grants access to the north side of M_end. The current beam still points toward receiver B, so bumping M_end changes only its visible diagonal. The player must stage it as \, the orientation that will eventually turn an east-arriving beam downward toward receiver C.

This is not blind symbol manipulation. The receiver and door colors make the spatial consequence visible:

receiver C is below the mirror;

door C is the only hub entrance to the lower room;

receiver D is above the mirror, but its door is inside the still-unreachable final room.

After leaving the north room, the player redirects L_branch east. The beam does not immediately reach the staged mirror: it stops at the vertical I_prefix, with the player standing next to that exact frontier. Rotating the pipe horizontal completes the longest shared route built so far:

Emitter
→ M_entry
→ L_branch
→ I_prefix
→ M_end
→ receiver C

Door C opens. Inside the lower room, the player approaches the same end mirror from the opposite side and bumps it once more. Its rule has not changed: \ → /. The contextual consequence has changed completely. The beam leaves the entire shared prefix intact but swaps the final receiver from C to D. C closes behind the player, D opens ahead, and the exit becomes reachable.

Exact solution
UURURRRRDDDRRRUUULLRUUUUUUURRRRDDDLDURUULLLDDDDLLLLLRRRDDDDDRRURUUULUDRDDDDRR

Key events:

Action	Event
8	M_entry rotates / → \; B closes and A opens
13	Player enters door A
19	M_entry rotates \ → /; A closes and B reopens
29	Player enters door B
36	From the north, M_end rotates / → \; beam remains on B
42	Player returns through door B
50–52	L_branch: SW → NW → NE → SE
55	I_prefix rotates V → H; beam reaches staged M_end, C opens
61	Player enters door C
69	From the south, M_end rotates \ → /; C closes and D opens
76	Player enters door D
77	Player enters exit
Recovery behavior

Two plausible wrong plans produce informative, recoverable results.

End mirror not staged:
After aligning the east branch and I_prefix, the beam reaches M_end = / and powers D instead of C. The player remains in the hub and cannot use D. Rotating the elbow once more from SE → SW restores B, allowing another visit to the north side of the mirror.

Shared pipe left vertical:
After the elbow reaches SE, the beam stops against I_prefix immediately beside the player. One bump extends the route to the already-staged end mirror.

After crossing C, an incorrect D alignment is also recoverable because the end mirror remains adjacent from the lower side; it can be toggled back to C.

Anti-cleanliness and necessity

The mature dependency graph is not A then B then C then D in object order:

M_end is configured during the B phase before its incoming route exists.

I_prefix is configured later, only after the upstream elbow is redirected.

M_end is then reused from the opposite side to destroy the C access that brought the player there.

The final change preserves the emitter, entry mirror, elbow, and straight-pipe prefix while altering only the terminal reflection.

All four rotatable components and all four doors are individually required for completion.

9. Validation contract

The authored instances were checked over the complete finite state space of:

player coordinate × all effective component orientations

No timers, animation phases, or no-op actions contribute gameplay states.

Level	Shortest actions	Reachable states	Reachable dead states
1	5	23	0
2	7	43	0
3	6	60	0
4	37	284	0
5	43	1,402	0
6	60	1,336	0
7	77	3,363	0

A reachable dead state is a nonterminal state from which no sequence reaches the exit. None exist.

For every level, additionally:

forbidding traversal of any listed door makes the level unsolvable;

forbidding bumps of any listed rotatable component makes the level unsolvable;

no exit can be reached by bypassing its authored partitions;

invalid movements, SPACE, and clicks cannot improve a state;

no level depends on animation timing;

no component has a level-specific operational exception;

completion depends only on entering the visible green exit.
