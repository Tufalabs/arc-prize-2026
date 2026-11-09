Worked for 55m 1s
Turnback Gates

Exact authored level count: 7

This is a game about crossing one-way gates becoming difficult because every crossing reverses the gate and rewrites the route available after the relic is collected.

The player explores compact structures containing a relic, an escape arch, and reversible one-way gates. Every gate visibly identifies the side from which it may currently be entered. Crossing from that side moves the agent through and flips the gate, so the destination side immediately becomes the permitted side.

This makes every successful crossing locally reversible, but not strategically neutral. A route used on the way inward may point the right way for retreat, the wrong way for the separate escape, or overwrite a gate that had already been prepared. Later levels are therefore about rehearsing escape routes backward, preserving useful orientations, and entering the relic region through one route so that a different route remains available afterward.

There are no timers, energy limits, random events, hidden modes, or loss states.

1. Core player-facing system

The board contains only:

connected floor corridors and walls;

one persistent blue agent;

one orange relic on a pedestal;

one boundary escape arch bearing the same orange relic emblem;

reversible one-way gates spanning selected floor edges.

The arch is closed until the relic is collected. Stepping onto the relic automatically picks it up. The relic then appears prominently on the agent, the pedestal remains visibly empty, and the arch changes from a closed maroon frame to an open green exit.

A gate has a yellow entry arrow. The arrow points toward the cell from which the gate may currently be crossed. It is deliberately an entry-side indicator rather than an arrow showing travel direction.

The permitted approach edge also carries a persistent green marker, while the
opposite approach edge is marked red. Crossing reverses the arrow and swaps
these two edge markers.

When the agent crosses:

the agent moves to the adjacent cell;

the gate reverses;

its arrow now points toward the agent’s destination cell.

Consequently, the agent may immediately cross back. A failed approach from the other side does not move or rotate the gate.

Central reversibility invariant

For any sequence of successful movements, performing the exact reverse sequence is legal and restores every traversed gate to its earlier orientation. Relic pickup is not reversed, but it never restricts movement.

This guarantees safe experimentation. A wrong plan may cost travel, but it never silently traps the player.

2. Implementation contract
2.1 Logical coordinates

Every level uses an 11×11 logical lattice:

x = 0…10, increasing rightward;

y = 0…10, increasing downward;

origin (0,0) at the upper-left.

A logical position is a pair (x,y).

Floor cells are explicitly listed by each level map. Two orthogonally adjacent floor cells are connected unless a gate spans their shared edge. Diagonal movement is never allowed.

The escape arch occupies a special non-floor cell with exactly one adjacent floor cell.

2.2 Action space

The environment accepts the provided action space exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any logical cell

Only the four directional actions are meaningful.

A directional action attempts to move the agent by exactly one logical cell.

SPACE is a legal no-op.

Clicking any cell is a legal no-op.

There are no selectable or clickable objects.

selection = null in every state.

There is no gameplay cursor state or hover state.

No visual object resembles a button or click target.

2.3 Static level definition
LevelDef:
    floor: set<Coord>
    start: Coord
    relic: Coord
    arch: Coord
    gates: ordered list<GateDef>

GateDef:
    id: G1, G2, ...
    a: floor Coord
    b: orthogonally adjacent floor Coord
    initial_entry_from: either a or b

No two gates occupy the same edge. No gate occupies the edge leading into the arch.

2.4 Runtime state
GameState:
    agent_pos: Coord
    relic_present: bool
    carrying_relic: bool
    gate_entry_from[g]: Coord for every gate g
    won: bool
    selection: null

Initial values in every level:

agent_pos = level.start
relic_present = true
carrying_relic = false
gate_entry_from[g] = g.initial_entry_from
won = false
selection = null

Derived visual state:

arch_open = carrying_relic
pedestal_empty = not relic_present

No action history is required. The settled frame exposes the complete gameplay state.

2.5 Directional transition order

For a directional action with offset d:

if won:
    no state change

target = agent_pos + d

if target == arch:
    if carrying_relic:
        agent_pos = arch
        won = true
    else:
        remain in place
        show closed-arch feedback
    stop

if target not in floor:
    remain in place
    show wall-bump feedback
    stop

if a gate g occupies edge {agent_pos, target}:
    if gate_entry_from[g] != agent_pos:
        remain in place
        gate does not rotate
        show barred-gate feedback
        stop

    agent_pos = target
    gate_entry_from[g] = target
else:
    agent_pos = target

if relic_present and agent_pos == relic:
    relic_present = false
    carrying_relic = true

Thus movement and any gate reversal are resolved before relic pickup. The resulting settled frame shows all consequences of that one action.

2.6 Exact success condition

A level succeeds only when a directional action attempts to enter the arch cell while carrying_relic == true.

Relevant:

the relic must previously have been collected;

the agent must enter the unique arch cell from its adjacent floor cell.

Ignored by validation:

final gate orientations;

the start marker;

the empty pedestal;

number of actions;

invalid actions previously attempted;

whether the agent revisited any location.

Allowed:

arbitrary exploration;

walking past the closed arch;

revisiting the pedestal;

restoring or disturbing gates after collecting the relic;

inserting any number of SPACE, click, wall-bump, or blocked-gate actions.

Forbidden only in the sense that it cannot win:

entering the closed arch without the relic.

After won = true, all further actions are no-ops until the host advances or resets the level.

3. Visual specification
3.1 Canvas and board mapping

Render on a 64×64 canvas using only integer color IDs 0–15.

Each logical cell is a 5×5 pixel region.

cell_left   = 4 + 5*x
cell_top    = 4 + 5*y
cell_right  = cell_left + 4
cell_bottom = cell_top + 4

The 11×11 board therefore occupies pixels 4…58 on both axes.

There are no internal cell borders or grid lines. Adjacent floor cells visually merge into connected corridors.

3.2 Palette usage
ID	Use
0	white completion highlight
1	floor
2	empty pedestal inset / neutral bump
3	gate body
4	walls and non-traversable board cells
5	canvas background and open arch interior
8	transient blocked-gate feedback
9	agent body
10	agent highlight and start marker
11	gate entry arrow
12	relic and carried-relic emblem
13	closed arch
14	open arch and completion pulse
15	relic pedestal

Colors 6 and 7 are unused.

3.3 Floor and walls

Every floor cell is filled with color 1.

Every non-floor logical board cell is filled with color 4.

Pixels outside the logical board use color 5.

No decorative wall texture is used.

3.4 Agent

The agent is a connected 3×3 blue shape centered in its logical cell:

body color 9;

center highlight color 10.

When carrying the relic, a five-pixel orange diamond in color 12 is drawn over the blue body. The blue silhouette remains visible around it.

3.5 Start marker

The start cell contains a persistent hollow 3×3 light-blue ring in color 10, drawn beneath the agent. It has no behavior after initialization.

3.6 Relic and pedestal

Before collection:

a connected purple pedestal in color 15 occupies the lower portion of the relic cell;

a five-pixel orange diamond in color 12 sits above it.

After collection:

the orange diamond disappears from the cell;

the purple pedestal remains with a color 2 empty inset;

the same orange diamond appears on the agent.

3.7 Escape arch

The arch cell is a connected 5×5 frame rotated to face its unique adjacent floor cell.

Closed state:

maroon frame and shutter, color 13;

orange relic diamond, color 12;

dark interior, color 5.

Open state:

shutter and orange emblem disappear;

frame becomes green, color 14;

interior remains color 5.

The matching orange shape on the relic and closed arch provides positive objective evidence in Level 1.

3.8 Gates

A gate spans the complete width of a floor passage. It consists of:

a two-pixel-thick dark bar in color 3;

a yellow triangular entry arrow in color 11.

For a horizontal pair of cells represented locally as a 10×5 rectangle:

gate body occupies columns 4 and 5;

when the left cell is the entry side, yellow pixels are placed at
(4,1), (4,2), (4,3), (3,2), (2,2);

when the right cell is the entry side, mirror those pixels horizontally.

Vertical gates use the same sprite rotated 90 degrees.

The arrowhead visibly protrudes into the permitted entry cell and points toward that cell. It is rendered after the agent so its orientation remains readable while the agent is adjacent.

Gate IDs such as G1 are documentation labels only and are never rendered.

4. Animation and feedback

All authoritative state changes happen during the single step() caused by the action. Animation subframes are presentation-only, reject further input, and consume no actions.

Successful gate crossing

Use four deterministic subframes:

old agent position and old arrow;

agent touching the gate boundary;

agent on the destination while the yellow arrow rotates;

settled state with the arrow pointing toward the destination cell.

The logical result is the settled fourth frame.

Failed gate approach

Use three subframes:

agent remains in place;

gate body flashes color 8 with a red barred line;

original gate appearance returns.

The gate orientation never changes.

Relic collection

Use three subframes:

agent arrives on the pedestal;

orange relic rises toward the agent;

settled frame shows the empty pedestal, carried relic, and open green arch.

Pickup, carry state, and arch opening belong to the same action.

Closed arch attempt

The agent remains in place. The orange emblem pulses once and the maroon frame briefly brightens. No persistent state changes.

Completion

On the successful arch-entry action:

the agent moves into the arch;

the arch frame pulses green;

a sparse white outline flashes around the agent;

the frame settles with won = true.

Ordinary floor movement is immediate and requires no tweening.

5. Progression overview

No new object category appears after the first gate in Level 2. Later depth comes entirely from geometry, gate orientation, route reuse, and ordering.

Level	Minimum directional actions	Main evidence or strategic development
1	2	Relic pickup opens the matching arch
2	5	A successful crossing reverses the gate and permits retreat
3	8	The entrance route and escape route can be different
4	25	Rehearse an escape chain backward; approach a blocked shortcut from its far side
5	43	Preserve a shared gate while changing from an outer route to an inner return
6	45	A gate is an entrance before the relic and the final exit afterward; the last crossing matters
7	61	Prepare and preserve two route families while a third route transfers the agent to the relic

The listed sequences are shortest successful directional sequences under the specified transition system, excluding insertable no-op or blocked actions. They were checked by exhaustive search over (agent position, gate orientation bits, carrying state).

6. Authored level notation

The maps below are implementation documentation only. Their letters are not rendered.

# = wall or non-floor cell
. = floor
S = start floor cell
R = relic floor cell
E = escape arch cell, not floor

Rows are labeled by y; character positions within each row are x = 0…10.

Solution notation:

U = UP
D = DOWN
L = LEFT
R = RIGHT
R4 = RIGHT repeated four times

No level overrides the global action or validation rules.

Level 1 — Relic and Arch

Intended depth: 2 directional actions
Purpose: Teach the objective without any gate changing simultaneously.

Player-facing discovery

The first frame contains only the agent, an adjacent orange relic, and a closed arch immediately beyond it. The arch carries the same orange diamond as the relic.

The only productive first action steps onto the relic. The relic appears on the agent and the arch immediately turns green. The next action enters it.

This establishes pickup, carrying, and success before gate behavior is introduced.

Map
y=0  ###########
y=1  ###########
y=2  ###########
y=3  ###########
y=4  ###########
y=5  ####SRE####
y=6  ###########
y=7  ###########
y=8  ###########
y=9  ###########
y=10 ###########
Initial state
start = (4,5)
relic = (5,5)
arch  = (6,5)

gates = []
agent_pos = (4,5)
carrying_relic = false
relic_present = true
won = false
selection = null
Shortest solution
R2

Resolution:

RIGHT: move to (5,5), collect the relic, open the arch.

RIGHT: enter (6,5) carrying the relic; win.

Attribution and necessity check

The first action changes only the objective-related state: relic pickup, carry display, empty pedestal, and matching arch opening. No gate or unrelated object changes at the same time.

Every visible component directly communicates or completes the objective.

Level 2 — The Arrow Turns

Intended depth: 5 directional actions
Purpose: Isolate permitted entry, reversal, and immediate retreat.

Player-facing discovery

The closed arch is behind the agent. A single gate points toward the agent, with an empty landing cell immediately beyond it and the relic one cell farther away.

The first RIGHT crosses the gate but does not collect the relic. The agent therefore sees the gate reverse as the sole new causal consequence. After collecting the relic, returning through the same gate is required to reach the arch. This proves that the reversed arrow is operational rather than decorative.

Map
y=0  ###########
y=1  ###########
y=2  ###########
y=3  ###########
y=4  ###########
y=5  ###ES.R####
y=6  ###########
y=7  ###########
y=8  ###########
y=9  ###########
y=10 ###########
Initial state
start = (4,5)
relic = (6,5)
arch  = (3,5)

agent_pos = (4,5)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(4,5) ↔ (5,5)	(4,5)	(4,5) → (5,5)
Shortest solution
R2 L3

Resolution:

Cross G1 from (4,5) to (5,5); G1 now permits (5,5) → (4,5).

Move to (6,5) and collect the relic.

Return to (5,5).

Cross G1 back to (4,5); G1 restores its initial orientation.

Enter the arch at (3,5).

Attribution and necessity check

The first gate crossing occurs onto an empty cell, so its reversal cannot be mistaken for a relic effect. The solution then requires the newly permitted reverse crossing. A mistaken hypothesis that the arrow is static cannot explain the successful retreat.

Level 3 — Leave by the Other Gate

Intended depth: 8 directional actions
Purpose: Establish that retracing the entrance is not necessarily the route to the visible escape.

Player-facing discovery

The agent enters a small central room through its west gate. The relic lies in a one-cell northern alcove, forcing the agent back to the central junction after collection.

At that point:

the west gate has reversed and permits a return toward the start;

the untouched east gate also permits departure from the room;

the now-open arch is visibly beyond the east gate.

The gate rule has not changed. The new insight is that two currently legal reversals have different strategic meanings.

Map
y=0  ###########
y=1  ###########
y=2  ###########
y=3  ###########
y=4  #####R#####
y=5  ##S.....E##
y=6  ###########
y=7  ###########
y=8  ###########
y=9  ###########
y=10 ###########
Initial state
start = (2,5)
relic = (5,4)
arch  = (8,5)

agent_pos = (2,5)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(3,5) ↔ (4,5)	(3,5)	(3,5) → (4,5)
G2	(6,5) ↔ (7,5)	(6,5)	(6,5) → (7,5)
Shortest solution
R3 U D R3

Phases:

R3: enter through G1 and reach the central cell (5,5).

U: collect the relic at (5,4).

D: return to the junction.

R3: leave through G2 and enter the arch.

Attribution and necessity check

Both gates are crossed by the shortest solution. The relic alcove makes the post-pickup route choice occur at a visible junction rather than allowing a straight line through the room.

Returning west is a safe but visibly unproductive hypothesis: it reaches the inert start marker rather than the green arch, and the player can cross G1 again to recover.

Level 4 — Rehearse from the Arch

Intended depth: 25 directional actions
Purpose: Cleanly introduce backward rehearsal and far-side use of a blocked shortcut.

Player-facing discovery

The agent begins beside the closed arch. The only useful route goes left through G1. Crossing it makes G1 point back toward the arch, visibly suggesting that the crossing has prepared the final return.

G2 and G3 form a downward chain. Each reverses as the agent descends, leaving a continuous upward route behind.

At (5,5), moving downward encounters G6 from its blocked side. Its arrowhead is visibly below the agent. The nearby leftward route is the only unexplored continuation. Going around that lower loop reaches the far side of G6 and the relic.

After collection, G6 provides a short upward handoff into the already-rehearsed G3–G2–G1 chain.

Map
y=0  ###########
y=1  #####..SE##
y=2  #####.#####
y=3  #####.#####
y=4  #####.#####
y=5  ###...#####
y=6  ###.#.#####
y=7  ###....R###
y=8  ###########
y=9  ###########
y=10 ###########
Initial state
start = (7,1)
relic = (7,7)
arch  = (8,1)

agent_pos = (7,1)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(7,1) ↔ (6,1)	(7,1)	(7,1) → (6,1)
G2	(5,1) ↔ (5,2)	(5,1)	(5,1) → (5,2)
G3	(5,3) ↔ (5,4)	(5,3)	(5,3) → (5,4)
G4	(3,5) ↔ (3,6)	(3,5)	(3,5) → (3,6)
G5	(3,7) ↔ (4,7)	(3,7)	(3,7) → (4,7)
G6	(5,5) ↔ (5,6)	(5,6)	(5,6) → (5,5)
Shortest solution
L2 D4 L2 D2 R4 L2 U6 R3

Detailed phases:

L2: cross G1 and reach (5,1). G1 now permits return toward the arch.

D4: cross G2 and G3, reaching (5,5). Both now point upward.

L2 D2 R4: go around the long side of the lower loop, cross G4 and G5, and collect the relic at (7,7).

L2: return to (5,7), the far side of G6.

U6: cross G6 upward, then reverse through G3 and G2 to (5,1).

R3: reverse through G1 and enter the arch.

A blind player may insert a diagnostic DOWN at (5,5). G6 flashes red without rotating, directly confirming that the arrowhead below marks the permitted side.

Attribution and anti-degenerate check

Every floor segment and all six gates participate in the shortest route.

G6 has contrasting roles under one unchanged rule:

before the relic, it blocks the direct downward route;

after the loop reaches its far side, it becomes the fastest return handoff.

A complete reversal of the inbound route remains a safe 29-action recovery solution. The 25-action solution depends on understanding and using G6 from its indicated side.

Level 5 — The Lower Return

Intended depth: 43 directional actions
Purpose: Nest a reversible relic excursion inside a larger route switch while preserving a shared gate.

Player-facing discovery

G1 leads from the upper area into a broad ring. At the first central junction, the visually short route toward the lower-left corner is oriented in the wrong direction: G5 and G4 point from the lower-left region back toward the center.

The player must therefore take the irregular outer side of the ring to reach the lower-left hub. A separate bottom spur leads to the relic through G2 and G3. Those two gates are immediately reversible, so the player can return to the lower-left hub without disturbing G1.

Now the formerly blocked G4–G5 chain points in exactly the useful direction: upward and right toward G1. G1 has been preserved since the initial descent and still permits the carrying agent to reverse upward.

Map
y=0  ###########
y=1  #S......E##
y=2  #####.#####
y=3  #.......###
y=4  #.#####.###
y=5  #.......###
y=6  #.#########
y=7  #......R###
y=8  ###########
y=9  ###########
y=10 ###########
Initial state
start = (1,1)
relic = (7,7)
arch  = (8,1)

agent_pos = (1,1)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(5,1) ↔ (5,2)	(5,1)	(5,1) → (5,2)
G2	(1,7) ↔ (2,7)	(1,7)	(1,7) → (2,7)
G3	(5,7) ↔ (6,7)	(5,7)	(5,7) → (6,7)
G4	(1,5) ↔ (1,4)	(1,5)	(1,5) → (1,4)
G5	(1,3) ↔ (2,3)	(1,3)	(1,3) → (2,3)
G6	(5,1) ↔ (6,1)	(5,1)	(5,1) → (6,1)
Shortest solution
R4 D2 R2 D2 L6 D2 R6 L6 U4 R4 U2 R3

Detailed phases:

R4 D2: reach (5,3) through G1.

R2 D2 L6 D2: travel around the outer side of the ring to (1,7).

R6: cross G2 and G3 and collect the relic at (7,7).

L6: reverse the relic spur through G3 and G2.

U4 R4: take the short return through G4 and G5 to (5,3).

U2: reverse G1 to (5,1).

R3: cross G6 and enter the arch.

Local experiment

From (5,3), the player can explore left toward the apparently short route. Approaching G5 from (2,3) produces a barred flash because its entry arrow lies at (1,3). The result is immediate and leaves the outer route as the useful hypothesis.

Anti-cleanliness and necessity check

The level contains:

a large ring with two directionally different sides;

a nested relic spur;

one shared gate, G1, that must remain untouched while the spur is resolved;

a separate final exit gate.

Every floor segment and every gate is used by the 43-action shortest solution.

A full retrace to the start and then to the arch takes 55 actions. The inner G4–G5 return is therefore a genuine route insight rather than decorative recovery.

Level 6 — The Last Open Throat

Intended depth: 45 directional actions
Purpose: Turn the same gate into an entrance before collection and the final exit afterward.

Player-facing discovery

An irregular upper approach leads to the closed arch-side cell (9,1). G4 points downward and carries the agent into the lower structure. As soon as it is crossed, G4 points upward toward the arch and becomes visibly useful for the eventual escape.

At (9,3), the direct vertical path toward the relic appears attractive, but G12 blocks it from above. Its entry arrow is lower in the passage. The only productive route is the long outer loop: left, down through an offset western bend, then across the bottom.

The relic is at the far end of that outer route. From the relic, G11 and G12 permit a direct vertical return. That return reaches the lower side of G4, which still points toward the arch because it has not been crossed since the initial entry.

Thus G4 is:

an entrance before the relic;

a state that must be preserved while traversing the outer loop;

the final exit gate after the relic.

Map
y=0  ###########
y=1  #S..#.....E
y=2  ###.#.###.#
y=3  ###...#...#
y=4  #######.#.#
y=5  #.......#.#
y=6  #.#######.#
y=7  #...#####.#
y=8  ###.#####.#
y=9  ###......R#
y=10 ###########
Initial state
start = (1,1)
relic = (9,9)
arch  = (10,1)

agent_pos = (1,1)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(3,1) ↔ (3,2)	(3,1)	(3,1) → (3,2)
G2	(5,3) ↔ (5,2)	(5,3)	(5,3) → (5,2)
G3	(7,1) ↔ (8,1)	(7,1)	(7,1) → (8,1)
G4	(9,1) ↔ (9,2)	(9,1)	(9,1) → (9,2)
G5	(9,3) ↔ (8,3)	(9,3)	(9,3) → (8,3)
G6	(7,5) ↔ (6,5)	(7,5)	(7,5) → (6,5)
G7	(3,5) ↔ (2,5)	(3,5)	(3,5) → (2,5)
G8	(1,7) ↔ (2,7)	(1,7)	(1,7) → (2,7)
G9	(3,9) ↔ (4,9)	(3,9)	(3,9) → (4,9)
G10	(7,9) ↔ (8,9)	(7,9)	(7,9) → (8,9)
G11	(9,9) ↔ (9,8)	(9,9)	(9,9) → (9,8)
G12	(9,5) ↔ (9,4)	(9,5)	(9,5) → (9,4)
Shortest solution
R2 D2 R2 U2 R4 D2 L2 D2 L6 D2 R2 D2 R6 U8 R

Detailed phases:

R2 D2 R2 U2 R4: follow the irregular upper route to (9,1), crossing G1–G3.

D2: cross G4 and reach (9,3). G4 now permits upward escape.

L2 D2 L6 D2 R2 D2 R6: traverse the long outer route through G5–G10 and collect the relic at (9,9).

U8: take the return-only vertical chain through G11, G12, and the preserved G4 to (9,1).

R: enter the arch.

Local experiment

After entering the lower structure, the player may move from (9,3) to (9,4) and try DOWN. G12 flashes red without rotating. Its arrowhead is visibly below, while G5 visibly permits the leftward route.

Last-crossing pressure

Crossing G4 upward before collecting the relic would reverse it downward again. The player could recover by crossing it once more, but the visible arrow would reveal that the useful final orientation had been overwritten.

The operational rule never changes; only the timing of G4’s last crossing matters.

Anti-cleanliness and necessity check

The outward route and return route overlap only at their endpoints and G4. The western detour prevents the outer route from collapsing into a simple rectangular lap.

All floor segments and all twelve gates are used by the shortest solution. Reversing the long outer route after collection yields a valid but longer 53-action solution; the 45-action solution requires recognizing the right column as a return-only chain.

Level 7 — Three Routes, One Relay

Intended depth: 61 directional actions
Purpose: Final composition of two preserved routes and two transfer routes, without adding any new rule or object.

Player-facing discovery

The final structure contains four route roles:

Top arch chain, G1–G2.
The agent begins beside the closed arch and crosses this chain leftward. It then points rightward for the final escape.

Prepared carrying route, G3–G7.
The agent winds through the irregular left half from the upper hub (5,1) to the lower transfer hub (5,9). Traversal makes the entire route point back toward the upper hub.

One-way transfer route, G8–G9.
At the lower hub, immediately reversing G3–G7 would erase the useful preparation. G8 instead points upward into an unexplored central route. Traversing that route consumes it: its gates then point back toward the upper-right staging region, not away from the lower hub.

Relic route, G10–G13.
The central transfer reaches a separate upper-right region. A final route descends to the relic and returns along the bottom to the lower hub. G13 initially prevents this route from being entered backward from the lower hub.

After collecting the relic and reaching the lower hub again:

G8 is now blocked from below;

G7 visibly permits entry into the preserved carrying route;

G3–G7 carry the relic back to the upper hub;

G2–G1 carry it through the arch.

The same yellow arrows that suggested each exploratory phase now expose the final route without requiring memory of hidden history.

Map
y=0  ###########
y=1  #####....SE
y=2  #####.#####
y=3  ###...#...#
y=4  ###.###.#.#
y=5  #...###.#.#
y=6  #.#####.#.#
y=7  #...#...#.#
y=8  ###.#.###.#
y=9  ###......R#
y=10 ###########
Initial state
start = (9,1)
relic = (9,9)
arch  = (10,1)

agent_pos = (9,1)
carrying_relic = false
relic_present = true
won = false
selection = null
Gate	Edge	entry_from	Initial allowed crossing
G1	(9,1) ↔ (8,1)	(9,1)	(9,1) → (8,1)
G2	(7,1) ↔ (6,1)	(7,1)	(7,1) → (6,1)
G3	(5,1) ↔ (5,2)	(5,1)	(5,1) → (5,2)
G4	(3,3) ↔ (3,4)	(3,3)	(3,3) → (3,4)
G5	(1,5) ↔ (1,6)	(1,5)	(1,5) → (1,6)
G6	(3,7) ↔ (3,8)	(3,7)	(3,7) → (3,8)
G7	(3,9) ↔ (4,9)	(3,9)	(3,9) → (4,9)
G8	(5,9) ↔ (5,8)	(5,9)	(5,9) → (5,8)
G9	(7,7) ↔ (7,6)	(7,7)	(7,7) → (7,6)
G10	(7,5) ↔ (7,4)	(7,5)	(7,5) → (7,4)
G11	(9,3) ↔ (9,4)	(9,3)	(9,3) → (9,4)
G12	(9,7) ↔ (9,8)	(9,7)	(9,7) → (9,8)
G13	(7,9) ↔ (6,9)	(7,9)	(7,9) → (6,9)
Shortest solution
L4 D2 L2 D2 L2 D2 R2 D2 R2
U2 R2 U4 R2 D6
L6 U2 L2 U2 R2 U2 R2 U2 R5

The line breaks mark logical phases only.

Phase 1: prepare the two eventual return routes
L4

Cross G1 and G2 from the arch side, reaching the upper hub (5,1). Both gates now point back toward the arch.

D2 L2 D2 L2 D2 R2 D2 R2

Follow the irregular G3–G7 route to the lower hub (5,9). Every crossed gate now points back toward (5,1).

At this point, directly reversing G7 would return to the upper hub but consume the preparation without finding the relic.

Phase 2: transfer to the separate relic approach
U2 R2 U4

Cross G8, traverse the central route, and cross G9 to reach (7,3). G8 and G9 now point in the opposite direction.

R2 D6

Enter the right route through G10, descend through G11 and G12, and collect the relic at (9,9).

Phase 3: hand the relic onto the prepared route

The first four actions of the next run move from the relic to the lower hub:

L4

This crosses G13 and reaches (5,9).

At this settled state:

G8 blocks upward travel because it was consumed during the transfer;

G7 accepts travel left into the prepared route;

reversing the right relic route remains possible but is visibly longer.

The compressed reference sequence continues left immediately:

L2 U2 L2 U2 R2 U2 R2 U2

This reverses G7 through G3 and returns to (5,1).

Finally:

R5

Cross G2 and G1 in reverse and enter the arch.

Epistemic checks

At the first visit to (5,9):

G8 visibly accepts upward entry;

G13 visibly rejects entry toward the relic route;

G7 accepts a reversible retreat but leads only to already explored territory.

This supplies a productive experiment without a hidden dependency.

At the second visit to (5,9), after collecting the relic:

G8 now visibly rejects the same upward attempt;

G7 still visibly accepts the carrying agent;

the changed contrast follows directly from the earlier crossings.

The player does not need to remember an invisible phase. Every strategically relevant difference is present in the arrows.

Anti-cleanliness and necessity check

Every floor segment and all thirteen gates are crossed by the shortest solution.

The level is not a sequence of devices encountered in final-use order:

G1–G2 are prepared first and used last.

G3–G7 are prepared, deliberately left unused through two other route families, and then reversed while carrying.

G8–G9 are useful only as a non-carrying transfer and become blockers later.

G10–G13 deliver the agent to the relic and back to the previously prepared hub.

An exact retrace from the relic to the original start followed by arch entry takes 73 actions. The verified 61-action route depends on preserving and reusing the left carrying route rather than merely backtracking.

7. Required implementation tests

An implementation should pass at least the following deterministic tests.

Gate unit tests

With entry_from = a, movement a → b succeeds.

After that movement, entry_from == b.

Immediate movement b → a succeeds and restores entry_from == a.

Attempting b → a before the first crossing fails and does not rotate the gate.

A failed wall, arch, or gate action changes no persistent state.

Relic and arch tests

Entering the relic cell sets:

relic_present = false;

carrying_relic = true;

arch visual to open.

Entering the closed arch without the relic fails.

Entering the open arch with the relic sets won = true.

Gate states do not affect validation once the agent enters the arch.

Input tests

SPACE changes no state.

Clicking every cell changes no state.

Directional actions always move at most one logical cell.

Inputs received during animation are ignored rather than queued.

Replay tests

The following exact sequences must win from the authored initial states:

Level	Winning sequence	Length
1	R2	2
2	R2 L3	5
3	R3 U D R3	8
4	L2 D4 L2 D2 R4 L2 U6 R3	25
5	R4 D2 R2 D2 L6 D2 R6 L6 U4 R4 U2 R3	43
6	R2 D2 R2 U2 R4 D2 L2 D2 L6 D2 R2 D2 R6 U8 R	45
7	L4 D2 L2 D2 L2 D2 R2 D2 R2 U2 R2 U4 R2 D6 L6 U2 L2 U2 R2 U2 R2 U2 R5	61

The game contains exactly seven levels, with no additional tutorial, test, or procedural levels included in the playable progression.
