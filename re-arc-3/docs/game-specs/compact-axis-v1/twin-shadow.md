Twin Shadow

This is a game about moving a linked pair becoming difficult because each command is reflected through two different obstacle fields.

Authored campaign size: exactly 7 levels.

Twin Shadow is a deterministic movement puzzle about turning obstruction into control. The player commands two agents at once, one in each of two fully visible rooms. Vertical commands move both agents in the same direction; horizontal commands are mirrored. Crucially, collision is resolved independently. A command can therefore hold one twin against a wall while moving the other, temporarily converting four shared commands into selective control.

Pressure plates and gates extend that same idea rather than creating a separate device puzzle. A twin can occupy a plate while surrounding walls absorb commands, allowing the other twin to pass through the corresponding gate. Goals eventually become useful blocking positions as well: reaching a goal from the correct side can let one twin remain docked while the other finishes.

There is no randomness, hidden information, timer, action limit, death state, or irreversible failure.

1. Global state-machine contract
1.1 Logical geometry

Each level contains two independent 7×7 logical rooms:

Room A: the left room, containing the magenta twin.

Room B: the right room, containing the blue twin.

Coordinates are (x,y).

x=0 is the leftmost column.

x=6 is the rightmost column.

y=0 is the top row.

y=6 is the bottom row.

Each logical cell renders as a 4×4 pixel region.

A wall occupies an entire logical cell.

A gate occupies the edge between two orthogonally adjacent floor cells.

Goals and plates are traversable floor cells.

The two agents never interact by collision because they inhabit separate rooms.

1.2 Persistent logical state
pos_A: (x,y)
pos_B: (x,y)
won: boolean

Level-authored immutable data:

walls_A: set of cells
walls_B: set of cells
goal_A: cell
goal_B: cell

links: list of:
    plate_room: A or B
    plate_cell
    gate_room: A or B
    gate_edge: unordered pair of adjacent cells
    motif: Diamond, Ring, or Bars

Gate openness is derived from agent positions and is not a persistent latch:

gate_open(link) =
    pos_A == plate_cell, if plate_room == A
    pos_B == plate_cell, if plate_room == B
1.3 Actions

Only the four directional actions affect the game.

Action	Twin A attempt	Twin B attempt
UP	(0,-1)	(0,-1)
DOWN	(0,+1)	(0,+1)
LEFT	(-1,0)	(+1,0)
RIGHT	(+1,0)	(-1,0)
SPACE	no movement	no movement
click any canvas cell	no movement	no movement

SPACE and click actions are intentionally inert. There are:

no clickable objects;

no clickable regions;

no cursor;

no selected object;

no hidden click affordance.

A no-op action does not animate or change any state.

1.4 Exact step() resolution order

For a directional action:

Read both current agent positions.

Compute which gates are open from pre-action plate occupancy.

Assign the corresponding movement delta to each agent.

Resolve each movement independently against:

the room boundary;

a wall destination cell;

a closed gate on the crossed edge.

A blocked agent remains in its current cell. The other agent’s movement is not cancelled.

Commit both resulting positions simultaneously.

Recompute plate occupancy and visible gate state from the new positions.

Recompute goal occupancy.

Set won=true exactly when:

pos_A == goal_A AND pos_B == goal_B

This gives two precise timing consequences:

Moving onto a plate does not open its gate early enough for the other twin to cross during that same action.

If a plate was occupied at the beginning of an action, the other twin may cross its gate while the plate-holder leaves during that same atomic action. The gate closes after both movements commit.

The authored solutions do not require the second edge case, but it is defined to remove validator ambiguity.

1.5 Collision details

An attempted movement is blocked when any of these is true:

candidate is outside x=0..6 or y=0..6
candidate is in the room's wall set
the unordered edge {current,candidate} is a gate edge
    and that gate was closed at action start

Gates block movement in both directions.

Agents never push walls, gates, plates, or goals. There are no agent-agent collisions.

1.6 Success and ignored state

A level succeeds only after a step on which both twins simultaneously occupy their own goals.

Relevant:

Twin A’s final cell.

Twin B’s final cell.

Ignored by validation:

current plate occupancy;

current gate states;

previous goal visits;

number of bumps;

number of actions;

whether every plate or gate was used.

Allowed:

One twin may reach and leave its goal any number of times.

One twin may remain on its goal for several actions while the other moves.

Both twins may bump simultaneously, although none of the verified intended traces needs such a step.

Unlimited experimentation is permitted.

Forbidden only by collision:

entering a wall;

crossing a closed gate;

leaving the room.

There is no loss condition.

1.7 Useful implementation invariant

For testing, define the horizontally mirrored coordinate of Twin B:

mirror_B = (6 - pos_B.x, pos_B.y)

When neither twin is blocked, Twin A and mirror_B receive the same displacement. Therefore, only one-sided obstruction can change their positional offset. This invariant is what makes Level 3 a clean diagnostic of independent collision resolution.

2. Rendering contract
2.1 Canvas layout

The render canvas is exactly 64×64 pixels.

Canvas background: color 4

Room A logical-cell origin: pixel (2,24)
Room B logical-cell origin: pixel (35,24)
Logical cell size: 4×4 pixels

Room A interior: x=2..29,  y=24..51
Room B interior: x=35..62, y=24..51

Room A one-pixel outer frame:
    x=1..30, y=23..52
Room B one-pixel outer frame:
    x=34..63, y=23..52

Only the two outer room frames are drawn. There are no internal grid lines or cell borders.

A three-pixel central gutter at x=31..33 contains a fixed mirror emblem. The emblem consists of a quiet vertical axis and two opposing colored wedges. It is not interactive.

2.2 Color assignment

Only integer color IDs 0–15 are used.

ID	Use
0	completion flash, bright active highlights
1	floor
2	outer frames, open-gate hinge remnants
3	walls
4	canvas background and room gutter
5	dark motif cutouts and deep shadow
6	Twin A
7	Twin A’s goal outline and A-half status outline
8	collision flash
9	Twin B
10	Twin B’s goal outline and B-half status outline
11	Diamond plate/gate pair
12	Ring plate/gate pair
14	Bars plate/gate pair

Colors 13 and 15 are unused.

2.3 Floor, walls, agents, and goals

Every floor cell is a solid 4×4 region of color 1.

Every wall cell is a solid 4×4 region of color 3.

Orthogonally adjacent wall cells merge visually; no lines divide them.

Each agent is a connected 2×2 shape centered in its logical cell.

Twin A uses color 6.

Twin B uses color 9.

An unoccupied goal is a one-pixel hollow outline around its 4×4 cell:

A goal: color 7.

B goal: color 10.

When occupied, the goal outline brightens and the agent remains visible in its center.

A goal has no collision behavior.

2.4 Plate and gate motifs

The motif names below are implementation labels only. No letters or text appear in the rendered grid.

Diamond: color 11, broad diamond silhouette.

Ring: color 12, hollow square-ring silhouette.

Bars: color 14, two separated parallel bars.

An unoccupied plate shows its motif over the floor color. An occupied plate:

visibly depresses by replacing its surrounding floor pixels with its link color;

draws a small color-0 center highlight;

pulses the matching connection rail once.

The agent sprite is drawn over the plate.

A closed gate is a two-pixel-thick link-colored bar on the shared edge:

For a left-right edge, use the last pixel column of the left cell and the first pixel column of the right cell.

For an up-down edge, use the last pixel row of the upper cell and the first pixel row of the lower cell.

An open gate retracts to two small color-2 hinge remnants, leaving a visibly clear central passage.

2.5 Visible link rail

The upper HUD contains up to three non-interactive connection rows:

Diamond row at y=12..13.

Ring row at y=16..17.

Bars row at y=20..21.

Each authored link has:

a miniature plate motif;

a continuous line in the link color;

a miniature gate motif;

a one-pixel source-room marker in color 6 or 9;

a one-pixel target-room marker in color 6 or 9.

The same motif and color appear on the actual plate and gate. Thus the player never needs to infer an arbitrary symbol mapping. When a plate is occupied, its entire rail brightens and the gate miniature retracts.

Unused rows are blank.

2.6 Simultaneous-goal emblem

A two-part completion emblem is centered at the top of the canvas.

Left half corresponds to Twin A and is outlined in color 7.

Right half corresponds to Twin B and is outlined in color 10.

A half fills with its agent color while that agent occupies its goal.

One filled half does not end the level.

Both filled halves trigger completion.

This is explicit positive goal evidence in Level 1.

2.7 Animation and feedback

Ordinary successful one-cell movement renders immediately and has no decorative interpolation.

Blocked movement

Logical positions are committed before animation. Then:

For one subframe, the blocked twin is drawn one pixel toward the obstacle.

The obstructing wall face, boundary segment, or gate bar flashes color 8.

On the next subframe, the twin returns to its stable cell center.

The other twin is already shown at its committed destination.

This animation consumes no player action.

Plate and gate transition

When a plate changes occupancy:

The new logical gate state is committed.

Over three render-only subframes, the plate depresses or rises.

The gate bar retracts or extends.

The connection rail pulses once.

Input is accepted after the stable frame.

Animation frames do not consume actions or alter collision timing.

Completion

On success:

Both goals brighten.

Both completion-emblem halves fill.

The central mirror seam flashes color 0 for six render-only subframes.

Further actions are ignored until the next level is loaded.

There is no danger or damage animation. Color 8 always means obstruction, not loss.

3. Level-documentation notation

The following authored maps are documentation only; their characters are not rendered.

.  floor
#  wall
S  initial agent cell
G  that room's goal
D  Diamond plate
O  Ring plate
T  Bars plate

Gates are listed separately because they lie between cells.

All level starts have:

won = false
all goals unoccupied
all plates unoccupied
all gates closed
selection = none
cursor = absent

The action traces use:

U = UP
D = DOWN
L = LEFT
R = RIGHT
4. The seven authored levels
Level 1 — Inward Reflection
Player-facing discovery

The two agents sit in short horizontal channels. Both goals lie inward, and the mirror emblem separates the rooms. Vertical experimentation from the relevant channel cells bumps both twins symmetrically, preventing vertical behavior from muddying the first causal observation.

Pressing RIGHT moves Twin A right and Twin B left. Both visibly approach their goals. Repeating the same command completes the level.

This establishes:

horizontal commands are mirrored;

both goals matter;

the objective is simultaneous occupancy.

Exact instance
Room A       Room B
.......      .......
.......      .......
.####..      ..####.
.S..G..      ..G..S.
.####..      ..####.
.......      .......
.......      .......

Initial state:

A start: (1,3)
B start: (5,3)
A goal:  (4,3)
B goal:  (2,3)

A walls:
    (1,2),(2,2),(3,2),(4,2),
    (1,4),(2,4),(3,4),(4,4)

B walls:
    (2,2),(3,2),(4,2),(5,2),
    (2,4),(3,4),(4,4),(5,4)

Links: none
Selection/cursor: none
Verified solution
RRR
After action	Twin A	Twin B	Visible result
1	(2,3)	(4,3)	Both move inward in opposite horizontal directions.
2	(3,3)	(3,3)	Both remain short of their differently placed goals.
3	(4,3)	(2,3)	Both goals and both emblem halves illuminate.

Verified shortest depth: 3 actions.

Attribution check

The useful experiment has an immediate result after one action.

No gate, plate, asymmetric obstacle, timer, or decoy is present.

The upper and lower wall strips suppress vertical displacement along the intended route.

A mistaken “both move right” hypothesis is disproved by Twin B’s first movement.

Three consecutive correct actions are required; the level is not a one-button accidental completion.

Level 2 — The Shared Rise
Player-facing discovery

Horizontal mirroring is already known. The new question is what vertical commands do.

The first UP moves both twins upward. A second immediate UP would bump both against matching wall bands. The visible gaps are at opposite horizontal ends of the rooms, so the known mirrored horizontal command moves both twins toward their respective opening. Once aligned with the gaps, UP moves both through simultaneously.

The level cleanly demonstrates that vertical movement is shared rather than mirrored.

Exact instance
Room A       Room B
.......      .......
.....G.      .G.....
.......      .......
######.      .######
.......      .......
.S.....      .....S.
.......      .......

Initial state:

A start: (1,5)
B start: (5,5)
A goal:  (5,1)
B goal:  (1,1)

A walls:
    (0,3),(1,3),(2,3),(3,3),(4,3),(5,3)

B walls:
    (1,3),(2,3),(3,3),(4,3),(5,3),(6,3)

Links: none
Selection/cursor: none
Verified solution
U | RRRRR | UUU | L
Steps	Actions	Twin A after phase	Twin B after phase	Purpose
1	U	(1,4)	(5,4)	Direct evidence that both move upward.
2–6	RRRRR	(6,4)	(0,4)	Move both toward the opposite wall gaps.
7–9	UUU	(6,1)	(0,1)	Pass the wall band together.
10	L	(5,1)	(1,1)	Mirrored horizontal docking.

Verified shortest depth: 10 actions.

Attribution check

The rooms are exact horizontal mirrors.

No intended action blocks only one twin.

Thus independent obstruction remains untested until Level 3.

The first UP alone distinguishes shared vertical movement from vertically mirrored movement.

The wall bands are necessary spatial structure: they force use of both the newly observed vertical rule and the established horizontal mirror.

Level 3 — The Clutch
Player-facing discovery

The goals require unequal horizontal corrections:

Twin A needs two cells to the right.

Twin B needs only one cell to the left.

If every horizontal action always moved both, those corrections could not coexist. A vertical wall stands immediately left of Twin A while the corresponding side of Room B is open.

The locally suggested experiment is LEFT. Twin A visibly bumps and stays still; Twin B moves right. The pair now has a one-cell offset. Two RIGHT commands then move Twin A two cells right and Twin B only back to the required column.

This establishes the central clutch rule: one twin being blocked does not cancel the other twin’s movement.

Exact instance
Room A       Room B
.......      .......
.......      .......
...G...      ..G....
#......      .......
#S.....      ...S...
#......      .......
.......      .......

Initial state:

A start: (1,4)
B start: (3,4)
A goal:  (3,2)
B goal:  (2,2)

A walls:
    (0,3),(0,4),(0,5)

B walls: none
Links: none
Selection/cursor: none
Verified solution
L | RR | UU
Steps	Actions	Twin A after phase	Twin B after phase	Purpose
1	L	(1,4)	(4,4)	A bumps against (0,4) while B moves right.
2–3	RR	(3,4)	(2,4)	The deliberately created offset becomes the required alignment.
4–5	UU	(3,2)	(2,2)	Shared vertical finish.

Verified shortest depth: 5 actions.
The listed trace is not the only length-5 solution, but every shortest solution requires a one-sided obstruction.

Attribution check

The first action changes only one agent’s position.

The wall face flashes at the same moment Twin B moves.

No second mechanic changes state simultaneously.

Without at least one one-sided block, the unequal required horizontal displacements are impossible.

The first blocked command initially moves Twin B away from its goal, demonstrating that a useful clutch action need not look locally goal-directed.

Level 4 — Hold the Diamond
Player-facing discovery

A Diamond plate appears in a three-sided alcove in Room A. A Diamond gate is the only opening through a vertical wall in Room B. Their matching motif, color, and connection rail make the relationship explicit.

After six ordinary commands, both twins arrive at a local impasse:

Twin A stands on the Diamond plate.

Twin B stands immediately beside the now-open Diamond gate.

A wall directly to Twin A’s right suggests repeating RIGHT.

The next RIGHT visibly produces three linked consequences:

Twin A bumps and remains on the depressed plate.

The gate remains open.

Twin B crosses the gate.

The same plate alcove then absorbs additional commands while Twin B navigates. Later, walls and boundaries swap which twin is anchored.

Exact instance
Room A       Room B
.......      .G#....
.......      ..#....
...##..      ..#....
...D#.G      .......
...##..      ..#....
...#...      ..#....
S......      ..#...S

Initial state:

A start: (0,6)
B start: (6,6)
A goal:  (6,3)
B goal:  (1,0)

A walls:
    (3,2),(4,2),
    (4,3),
    (3,4),(4,4),
    (3,5)

B walls:
    (2,0),(2,1),(2,2),
    (2,4),(2,5),(2,6)

Diamond plate:
    Room A, cell (3,3)

Diamond gate:
    Room B, edge (2,3) ↔ (3,3)

Selection/cursor: none

The Diamond gate is initially closed.

Verified solution
UUURRR | RRUL | UURRR | DDD | LLDDR | UUU
Steps	Actions	Twin A after phase	Twin B after phase	State consequence
1–6	UUURRR	(3,3)	(3,3)	A depresses Diamond; B is beside its gate.
7–10	RRUL	(2,3)	(1,2)	A holds Diamond for the crossing, then exits while B is blocked by the wall column.
11–15	UURRR	(5,1)	(0,0)	B’s upper-left boundaries become the clutch.
16–18	DDD	(5,4)	(0,3)	Both move down through different open regions.
19–23	LLDDR	(6,6)	(1,3)	A’s lower wall faces and B’s column exchange anchoring roles.
24–26	UUU	(6,3)	(1,0)	Simultaneous docking.

At step 7 specifically:

Before: A=(3,3), B=(3,3), Diamond open
Action: RIGHT
After:  A=(3,3), B=(2,3)

Twin A is blocked by wall (4,3) while Twin B crosses the gate.

Verified shortest depth: 26 actions.

Necessity and anti-cleanliness check

The Room B wall column makes the Diamond gate the only left-right crossing. Keeping that gate permanently closed makes the level unsolvable.

Treating the gate as permanently open lowers the shortest path to 13 actions, so plate timing is not decorative.

The C-shaped wall around Diamond supplies three useful blocking faces.

Wall (3,5) prevents a lower-flank shortcut; removing it lowers the shortest solution from 26 to 24.

The goals are not placed directly after the gate. More than half the solution occurs after the gate passage and requires alternating wall and boundary anchors.

Every repeated blocked command moves the other twin; none is padding.

Level 5 — Reciprocal Custody
Player-facing discovery

The Diamond relationship is now familiar, but a second linked pair appears:

Diamond in Room A opens a gate in Room B.

Ring in Room B opens a gate in Room A.

The Ring pair is visibly connected exactly like Diamond, so the player is not asked to infer a new symbolic language. The discovery is that either twin can become the holder.

Both rooms have barrier walls whose only openings are linked gates. The approach to the first plate simultaneously stages the other twin for its gate. Crossing that gate changes which region contains the useful anchor, eventually letting Twin B reach the Ring plate and hold it while Twin A passes.

The goals also begin changing role. Twin B touches its goal twice before the final action, but those early visits are strategically temporary.

Exact instance
Room A       Room B
....#..      S..#...
.##.#..      ...#...
.#D.#..      .......
.####.G      ..###.G
......#      ...###.
....##.      ..###O.
...S#..      ...###.

Initial state:

A start: (3,6)
B start: (0,0)
A goal:  (6,3)
B goal:  (6,3)

A walls:
    (4,0),
    (1,1),(2,1),(4,1),
    (1,2),(4,2),
    (1,3),(2,3),(3,3),(4,3),
    (6,4),
    (4,5),(5,5),
    (4,6)

B walls:
    (3,0),(3,1),
    (2,3),(3,3),(4,3),
    (3,4),(4,4),(5,4),
    (2,5),(3,5),(4,5),
    (3,6),(4,6),(5,6)

Diamond plate:
    Room A, cell (2,2)

Diamond gate:
    Room B, edge (2,2) ↔ (3,2)

Ring plate:
    Room B, cell (5,5)

Ring gate:
    Room A, edge (3,4) ↔ (4,4)

Selection/cursor: none

Both gates are initially closed.

Verified solution
UULLLUUUURRRDDL
LLLR
UULULLDDDDR
RRRRUL
URUUUU
DDD

Concatenated:

UULLLUUUURRRDDLLLLRUULULLDDDDRRRRRULURUUUUDDD
Steps	Actions	Twin A after phase	Twin B after phase	State consequence
1–15	UULLLUUUURRRDDL	(2,2)	(1,2)	A occupies Diamond; B is staged beside the Diamond barrier opening.
16–19	LLLR	(3,2)	(3,2)	B crosses Diamond while A is held in its alcove.
20–30	UULULLDDDDR	(1,4)	(5,5)	B reaches and depresses Ring; A is staged left of its gate.
31–36	RRRRUL	(5,3)	(6,5)	B holds Ring while A crosses the Ring gate and advances.
37–42	URUUUU	(6,0)	(6,0)	Post-gate offset shaping using walls and boundaries.
43–45	DDD	(6,3)	(6,3)	Both descend onto their goals.

Notable temporary goal visits:

Step 27: B is on its goal, A is at (0,2); no completion.
Step 39: B is again on its goal, A is at (6,1); no completion.
Step 45: both are on their goals; completion.

Verified shortest depth: 45 actions.

Necessity and anti-cleanliness check

The Room A wall at x=4, except for the Ring-gated opening at y=4, separates A’s start and goal.

The Room B wall at x=3, except for the Diamond-gated opening at y=2, separates B’s start and goal.

Keeping either gate permanently closed makes the level unsolvable.

The two plate holds face different geometric pressures and occur in opposite rooms.

The second plate is not simply placed immediately after the first gate. Shared movement must first route both twins into a useful joint state.

Twin B’s early goal visits establish that a goal is a floor location, not an absorbing terminal.

Twenty-three actions in the verified trace contain a deliberate one-sided block; each moves the unblocked twin.

Level 6 — Dock First
Player-facing discovery

The two reciprocal gate relationships return in a vertically inverted and structurally altered layout. The new strategic interpretation is visible at the end:

Twin B reaches its goal at the bottom of Room B.

Twin A remains four cells above its own goal.

The only useful remaining commands are DOWN.

Twin B’s bottom boundary absorbs every DOWN, preserving its glowing goal occupancy while Twin A descends.

This is the clean tutorial for dock without undocking. It contrasts directly with Level 5, where an early goal visit had to be abandoned.

Exact instance
Room A       Room B
..#S##.      ...###.
....#..      ...##O.
.......      ...###.
.####..      ...#...
.#D.#..      .......
.##.#..      ...##..
....#G.      S..#G..

Initial state:

A start: (3,0)
B start: (0,6)
A goal:  (5,6)
B goal:  (4,6)

A walls:
    (2,0),(4,0),(5,0),
    (4,1),
    (1,3),(2,3),(3,3),(4,3),
    (1,4),(4,4),
    (1,5),(2,5),(4,5),
    (4,6)

B walls:
    (3,0),(4,0),(5,0),
    (3,1),(4,1),
    (3,2),(4,2),(5,2),
    (3,3),
    (3,5),(4,5),
    (3,6)

Diamond plate:
    Room A, cell (2,4)

Diamond gate:
    Room B, edge (2,4) ↔ (3,4)

Ring plate:
    Room B, cell (5,1)

Ring gate:
    Room A, edge (3,2) ↔ (4,2)

Selection/cursor: none

Both gates are initially closed.

Verified solution
DDLLLDDDDRRRUUL
LLLR
DDLLDLUUUUR
RRRRL
DDDRRRLLDDR
DDDD

Concatenated:

DDLLLDDDDRRRUULLLLRDDLLDLUUUURRRRRLDDDRRRLLDDRDDDD
Steps	Actions	Twin A after phase	Twin B after phase	State consequence
1–15	DDLLLDDDDRRRUUL	(2,4)	(1,4)	A occupies Diamond and stages B.
16–19	LLLR	(3,4)	(3,4)	B crosses the Diamond gate.
20–30	DDLLDLUUUUR	(1,2)	(5,1)	B occupies Ring and stages A.
31–35	RRRRL	(4,2)	(6,1)	A crosses the Ring gate.
36–46	DDDRRRLLDDR	(5,2)	(4,6)	B reaches its goal from above.
47–50	DDDD	(5,6)	(4,6)	B bumps against the bottom boundary while A descends four cells.

During steps 47–50:

Twin B’s goal and the blue completion half remain continuously illuminated.

Every DOWN gives B a harmless boundary bump.

Twin A moves on every action.

Verified shortest depth: 50 actions.

Necessity and anti-cleanliness check

Each gate is again the sole opening through a barrier separating its twin from the goal region.

Keeping either gate permanently closed makes the level unsolvable.

The final four commands are not repeated walking padding: each is a selective movement enabled by B’s goal-side boundary.

Approaching B’s goal from above is strategically different from merely touching it.

The goal serves as a tool without acquiring any new rule; its operational behavior remains ordinary floor occupancy plus win feedback.

Twenty-eight actions in the verified trace contain a one-sided block.

Level 7 — Relay Chamber
Player-facing discovery

The final level adds no new operational mechanic. It contains three instances of the established plate-gate relationship:

Diamond in Room A opens an entry gate in Room B.

Ring inside Room B opens the central barrier in Room A.

Bars inside Room A opens a downward exit in Room B.

The Bars motif is connected through the same visible rail used by the two familiar links. It represents another plate-gate identity, not another rule.

The geometry forms an alternating relay:

A holds Diamond while B enters the upper chamber.

B holds Ring while A crosses its central barrier.

A reaches Bars from the opposite side and holds it while B exits downward.

After all three passages, the twins are still badly offset.

The remaining route alternates ordinary walls and boundaries.

Twin A finally docks first and uses the bottom boundary to hold position during six final DOWN commands.

Exact instance
Room A       Room B
.#####.      #O..##.
.D###T.      ##.....
.#####.      ....#..
...#...      ....#.S
.......      ##.##..
...##.#      .......
S..#..G      ......G

Initial state:

A start: (0,6)
B start: (6,3)
A goal:  (6,6)
B goal:  (6,6)

A walls:
    (1,0),(2,0),(3,0),(4,0),(5,0),
    (2,1),(3,1),(4,1),
    (1,2),(2,2),(3,2),(4,2),(5,2),
    (3,3),
    (3,5),(4,5),(6,5),
    (3,6)

B walls:
    (0,0),(4,0),(5,0),
    (0,1),(1,1),
    (4,2),
    (4,3),
    (0,4),(1,4),(3,4),(4,4)

Diamond plate:
    Room A, cell (1,1)

Diamond gate:
    Room B, edge (4,1) ↔ (5,1)

Ring plate:
    Room B, cell (1,0)

Ring gate:
    Room A, edge (2,4) ↔ (3,4)

Bars plate:
    Room A, cell (5,1)

Bars gate:
    Room B, edge (2,3) ↔ (2,4)

Selection/cursor: none

All three gates are initially closed.

Verified solution
UUUUUUDR
RRL
DDRURR
DRRURRUUL
DDDDR
DDDLDDLUL
UUUULURR
DDDDDD

Concatenated:

UUUUUUDRRRLDDRURRDRRURRUULDDDDRDDDLDDLULUUUULURRDDDDDD
Steps	Actions	Twin A after phase	Twin B after phase	State consequence
1–8	UUUUUUDR	(1,1)	(5,1)	B’s top boundary first anchors it; A then occupies Diamond.
9–11	RRL	(0,1)	(4,1)	B crosses Diamond while A is held, then A exits the plate alcove.
12–17	DDRURR	(2,3)	(1,0)	B reaches Ring; A is staged beside the Ring barrier.
18–26	DRRURRUUL	(5,1)	(2,0)	B holds Ring while A crosses and approaches Bars from its open side.
27–31	DDDDR	(6,1)	(2,4)	A holds Bars while B descends through the Bars gate.
32–40	DDDLDDLUL	(4,6)	(5,5)	Post-relay offset correction using lower wall faces.
41–48	UUUULURR	(6,6)	(6,0)	A reaches its goal; B is aligned above its own.
49–54	DDDDDD	(6,6)	(6,6)	A remains docked against the bottom boundary while B descends six cells.

Important individual transitions:

Step 9:
    Diamond is open.
    A attempts RIGHT into wall (2,1) and stays at (1,1).
    B crosses from (5,1) to (4,1).

Step 19:
    Ring is open.
    B remains held in its upper pocket.
    A crosses the Ring gate from (2,4) to (3,4).

Step 30:
    Bars is open.
    A remains held at (5,1).
    B crosses downward from (2,3) to (2,4).

Step 48:
    A reaches goal (6,6).
    B is at (6,0).

Steps 49–54:
    A is blocked by the bottom boundary on every DOWN.
    B moves from y=0 to y=6.

Verified shortest depth: 54 actions.

Necessity and anti-cleanliness check

Keeping Diamond, Ring, or Bars permanently closed makes the level unsolvable in each case.

Each plate is in a differently oriented alcove:

Diamond is entered from the left.

Ring is entered from the right side of its room geometry.

Bars is entered from the right and holds downward movement.

The Room B structure has a distinct entry and downward exit rather than a straight sequence of adjacent devices.

The Bars gate is crossed at step 30, leaving 24 further actions. The puzzle therefore does not end when the device relay is complete.

The final approach reuses goal docking from Level 6 under the opposite twin.

Forty-one actions in the verified trace deliberately block exactly one twin while moving the other.

The three links form alternating custody across the two rooms rather than three independent conditions checked at the end.

5. Campaign progression and verification summary
Level	Main discovery or reinterpretation	Verified shortest depth
1	Horizontal commands are mirrored; both goals form one objective.	3
2	Vertical commands move both twins in the same direction.	10
3	Obstruction resolves independently and can create offset.	5
4	A plate can be held as the other twin passes its gate.	26
5	Either twin can hold a cross-room plate; early goal visits may be temporary.	45
6	A correctly approached goal can safely anchor the completed twin.	50
7	Three familiar links form an alternating relay, followed by post-gate alignment and docking.	54

The apparently shorter Level 3 relative to Level 2 is intentional. Level 2 needs enough movement to establish a vertical route around a mirrored barrier, while Level 3 is a tightly isolated causal experiment. Difficulty thereafter rises through joint-state planning rather than corridor length.

Because gate states are derived entirely from the two positions, each level has at most 49 × 49 = 2401 logical joint-position states. Exhaustive breadth-first search under the specified update order verifies the listed shortest depths and final traces.

The resulting game remains one focused formal system:

the commands never change;

collision never changes;

plates never latch;

gates never acquire exceptions;

goals never become special blockers;

later depth comes entirely from where the same walls, temporary gates, plates, boundaries, and goals are placed.
