Charge Pour

Exact authored level count: 7

This is a game about pouring one charge unit along an arrow becoming difficult because every pour sends usable empty capacity in the opposite direction.

Charge Pour is a click-only conservation puzzle. Every successful click performs the same tiny operation: one visible pip leaves one reservoir and enters another. Nothing is hidden, no pipe changes behavior, and pips have no identity. The difficulty comes entirely from the authored network: full reservoirs block incoming charge, one-way routes make some allocations irreversible, and the vacancy created at the source is often more strategically important than the pip that moved.

The campaign introduces no gameplay nouns beyond reservoirs, goal reservoirs, pipes, and directional transfer buttons. Later levels deepen the original rule through bottlenecks, shared headroom, directed cycles, and goals that must temporarily be drained and refilled.

1. Player-facing rules
Controls

Only clicking is used.

CLICK(x, y) attempts the transfer represented by the directional arrow button under that cell.

UP, DOWN, LEFT, RIGHT, and SPACE are accepted by the action API but are deliberate no-ops.

Clicking a reservoir, socket, bare pipe, background cell, or any other non-arrow region is a no-op.

There is no selection mode and no two-click interaction. One click directly attempts one transfer.

There is no energy limit, timer, or in-game move limit. Exploration is unlimited.

The host environment may expose its standard restart command, but restart is not a game action and is not rendered inside the 64×64 grid.

Every arrow is a prominent 5×5 button, rather than a tiny symbol painted somewhere on a pipe. These buttons are the only clickable objects.

Basic operation

Clicking an arrow from reservoir S to reservoir D:

Succeeds only when S contains at least one pip and D has at least one empty socket.

Removes exactly one pip from S.

Adds exactly one pip to D.

Leaves every other reservoir unchanged.

A click never pours an entire reservoir.

Failed transfers

A transfer fails when:

the source is empty;

the destination is full; or

both conditions hold.

The clicked arrow flashes red. An empty source rim pulses red; a full destination rim pulses red. If both block the transfer, both rims pulse. The settled state remains unchanged.

This makes failures directly attributable to visible conditions. There are no locks, cooldowns, ownership rules, or hidden prerequisites.

Objective

A reservoir with gold-rimmed sockets is a goal reservoir.

A level succeeds exactly when:

for every goal reservoir G:
    charge[G] == capacity[G]

Plain reservoirs have no independent final requirement. However, every authored level begins with total charge equal to combined goal capacity. Conservation therefore implies that filling all goals also leaves every plain reservoir empty.

Goals obey ordinary transfer rules. A goal may have an outgoing arrow and may need to be temporarily emptied. Gold denotes the required final state, not a one-way sink rule.

2. Core strategic consequences
Vacancy propagation

A successful S→D transfer creates one empty socket in S. In that sense:

the pip travels with the arrow;

the vacancy travels against the arrow.

A chain of full reservoirs must therefore be cleared from its downstream end toward its upstream end.

Repeated bottleneck clearing

A one-socket intermediate reservoir can carry arbitrarily many pips, but only one at a time. It must be emptied after each arrival before the next pip can enter.

Shared headroom

When several routes feed the same reservoir or goal, using a socket for one route may strand charge on another route. The sockets themselves are the visible resource; there is no hidden branch quota.

Directed-cycle circulation

A directed cycle with every socket full cannot move. One empty socket permits a neighboring pip to enter it, which relocates the empty socket to the pip’s former reservoir. Repeated transfers circulate the vacancy opposite the arrows.

Temporary goal drainage

Because goal reservoirs are mechanically ordinary reservoirs, a full gold reservoir may need to send charge onward and be refilled later. This is introduced only after the player has repeatedly observed conservation and capacity.

3. Rendering contract
Canvas and coordinates

Render canvas: exactly 64×64 cells.

Coordinates are zero-indexed.

(0,0) is the upper-left cell.

x increases rightward; y increases downward.

No text, letters, numbers, labels, or counters are rendered.

Reservoir IDs used below exist only in this document.

Palette

Only the supplied integer color IDs are used.

ID	Use
5	Background and empty socket interiors
4	Reservoir body and arrow-button backing
3	Pipes and ordinary socket rims
2	Plain reservoir outer rims
10	Charge pips and successful-transfer highlight
11	Goal outer rims, goal socket rims, and hover highlight
12	Transfer arrows
8	Rejected-transfer feedback
14	Completion pulse

No other colors are required.

Reservoir geometry

Each reservoir is defined by:

top-left = (x, y)
socket layout = columns × rows
capacity = number of actual sockets

Its rendered dimensions are:

width  = 5 × columns + 2
height = 5 × rows    + 2

Rendering:

Fill the full bounding box with color 4.

Draw a one-cell outer rim:

color 2 for a plain reservoir;

color 11 for a goal.

Candidate socket positions are enumerated row-major.

Render only the first capacity positions. Unused positions in a partially filled final row remain ordinary body color 4.

Each socket occupies a 4×4 region:

plain socket rim: 3;

goal socket rim: 11;

inner 2×2 area: 5 when empty, 10 when occupied.

All sockets in a goal are individually gold-rimmed, so the required final occupancy is visible even in a large goal.

For deterministic rendering, occupied sockets are always the first q sockets in row-major order. A transfer removes the highest-index occupied socket from the source and fills the lowest-index empty socket in the destination.

Conceptual socket module:

rrrr    r = socket rim
rpp r   p = 2×2 pip or empty interior
rpp r
rrrr

The actual module is a compact 4×4 implementation of this silhouette.

Pipes and arrow buttons

Pipes are one-cell-wide orthogonal lines in color 3.

A pipe does not itself receive clicks.

Each directed edge has its own 5×5 arrow-button plate centered at a specified coordinate.

Its clickable region is:

abs(click_x - arrow_center_x) <= 2
and
abs(click_y - arrow_center_y) <= 2

Arrow hitboxes never overlap one another or any reservoir bounding box.

The backing is color 4; the connected arrow silhouette is color 12.

Arrow silhouettes are rotated versions of one 5×5 pattern. For a right-facing arrow:

...A.
...AA
AAAAA
...AA
...A.

A denotes color 12, not rendered text.

If a front end supports pointer hover, the button’s outer cells change to color 11. Hover is presentation-only and is not stored in game state.

4. State-machine contract
Persistent state
level_index: 0..6
charge[r]: integer for every reservoir r
capacity[r]: immutable integer
goal[r]: immutable boolean
completed: boolean
selected_arrow: always None
feedback: None | Rejected(arrow_id, empty_source, full_destination)
animation: transient presentation state

There are no hidden pip identities, histories, phases, counters, or randomized values.

Every authored level begins with:

completed = false
selected_arrow = None
feedback = None
animation = None

The pointer begins outside the canvas, so no button is initially hover-highlighted.

step() resolution order

For one input action:

Clear feedback from the preceding action.

If completed == true, ignore the action.

If the action is not a click, perform a no-op.

For a click, find the unique arrow whose 5×5 hitbox contains the clicked cell.

If no arrow contains it, perform a no-op.

Read source and destination charge from the pre-action state.

If the source is empty or destination full:

do not alter charge;

set rejection feedback;

enqueue the rejection animation.

Otherwise, atomically perform:

charge[source]      -= 1
charge[destination] += 1

Assert the charge and capacity invariants.

Evaluate the goal condition on the post-transfer state.

Enqueue the successful-transfer animation.

If the goal condition is true, set completed = true and append completion feedback.

Only one transfer can occur in one step().

Required invariants

After initialization and after every step:

0 <= charge[r] <= capacity[r]             for every reservoir r
sum(charge[r]) == initial_total_charge

No arrow activation creates, deletes, recolors, or duplicates charge.

Animation

A successful transfer moves a pip over more than one render cell and therefore requires a short deterministic animation.

The state mutation occurs atomically during step(). The animation is then rendered from pre- and post-state snapshots:

Subframe 0: clicked arrow highlights with color 10.

Subframe 1: source’s removed socket is empty; a 2×2 moving pip appears at 25% of the Manhattan path.

Subframe 2: moving pip at 50%.

Subframe 3: moving pip at 75%.

Subframe 4: moving pip disappears and the destination socket is occupied.

The animation path is:

removed source socket center
→ specified source pipe port
→ listed pipe waypoints
→ specified destination pipe port
→ newly filled destination socket center

Progress is measured by total Manhattan path length. At a bend, the moving pip follows the bend rather than interpolating diagonally.

Animation subframes:

do not consume actions;

do not change logical state;

do not permit another action before the settled frame.

A rejected transfer has three presentation subframes: red arrow, relevant red rim pulse, settled unchanged frame.

On completion, all goal outer rims alternate 11, 14, 11, 14 over four additional subframes. No extra step is consumed. The final settled frame always contains enough information to infer the resulting state even when intermediate animation frames are unavailable.

5. Campaign progression
Level	Name	Main discovery or composition	Verified shortest successful clicks
1	First Pour	Clickable arrows, one-pip transfer, visible gold objective	4
2	Full Cup	Full destinations reject; clear a bottleneck repeatedly	5
3	Backward Air	A vacancy must be pulled backward through a full chain	12
4	Split Headroom	One flexible branch must divide across constrained goals	26
5	Turning Gap	Circulate a vacancy through a directed ring	40
6	Golden Conduit	Temporarily drain and refill a goal while splitting flow	48
7	Whole Circuit	Cycle circulation, goal drainage, staging, and reserved goal capacity	54

The counts above include successful transfer clicks only. Failed experiments and unrelated no-ops have no in-game penalty.

6. Authored levels
Level 1 — First Pour
Player-facing discovery

A partially filled gold reservoir provides positive evidence for the objective: one gold-rimmed socket already contains the same light-blue pip seen in the source.

The source contains two pips, but the one-slot middle reservoir can display only one. Clicking S→B therefore visibly moves one pip rather than dumping the whole source. Clicking B→G puts that pip into one marked goal socket. Repeating the pair finishes the visible pattern.

There are no unrelated objects, alternate routes, misleading buttons, or capacity failures.

Documentation-only schematic:

S(2/2) → B(0/1) → G★(1/3)
Initial reservoirs
ID	Role	Top-left	Socket layout	Capacity	Initial charge
S	Plain source	(4,27)	2×1	2	2
B	Plain bottleneck	(29,27)	1×1	1	0
G	Goal	(43,27)	3×1	3	1

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow center and orientation	Pipe centerline
S→B	(22,30), right	(16,30) → (28,30)
B→G	(39,30), right	(36,30) → (42,30)
Reference solution
S→B, B→G, S→B, B→G

Exactly four successful clicks.

Win validation
charge[G] == 3

By conservation, S=0 and B=0.

Attribution and necessity check

The first successful click changes exactly two visible counts: S:2→1, B:0→1.

The first B→G click changes one of the two initially empty gold sockets.

Removing either arrow makes completion impossible.

The partially filled goal is the only positive goal example needed; no legend or symbolic mapping is involved.

Random clicking outside the two visible orange buttons cannot alter state.

Level 2 — Full Cup
Player-facing discovery

The geometry is deliberately almost identical to Level 1. The only causal change is that the middle reservoir begins full.

A natural first attempt is S→B. It rejects immediately: the arrow flashes red and B’s visibly filled rim pulses. Clicking B→G creates an empty socket in B. The exact same S→B button now succeeds, distinguishing “destination capacity” from alternatives such as a disabled arrow or a required fixed sequence.

The one-slot reservoir must then be cleared after every arrival.

S(2/2) → B(1/1) → G★(0/3)
Initial reservoirs

The coordinates are intentionally identical to Level 1.

ID	Role	Top-left	Socket layout	Capacity	Initial charge
S	Plain source	(4,27)	2×1	2	2
B	Plain bottleneck	(29,27)	1×1	1	1
G	Goal	(43,27)	3×1	3	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow center and orientation	Pipe centerline
S→B	(22,30), right	(16,30) → (28,30)
B→G	(39,30), right	(36,30) → (42,30)
Reference solution
B→G, S→B, B→G, S→B, B→G

Exactly five successful clicks.

A failed exploratory S→B before this sequence changes no logical state and is not included in the solution depth.

Win validation
charge[G] == 3
Attribution and necessity check

The successful and failed attempts use the same arrow in two immediately consecutive visible capacity states.

No simultaneous rule changes obscure the evidence.

B cannot hold both source pips, so repeated clearing is structurally necessary.

Every successful move advances charge toward the only goal; early exploration cannot create an irreversible valid mistake.

Level 3 — Backward Air
Player-facing discovery

Three upstream reservoirs begin full. The two upstream transfer buttons reject because their destinations have no empty socket. Only the bottom transfer can initially succeed.

After B→G, the empty socket is visibly in B, allowing A→B. That moves the empty socket to A, allowing S→A. In three clicks, the player observes the vacancy moving upward while the pips move downward.

This cannot be explained by the older misconception “always click the arrow nearest the goal”: after the first click, the productive action moves one level upstream.

S(3/3)
  ↓
A(1/1)
  ↓
B(1/1)
  ↓
G★(0/5)
Initial reservoirs
ID	Role	Top-left	Socket layout	Capacity	Initial charge
S	Plain source	(24,2)	3×1	3	3
A	Plain reservoir	(29,15)	1×1	1	1
B	Plain reservoir	(29,28)	1×1	1	1
G	Goal	(24,43)	3×2	5	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow center and orientation	Pipe centerline
S→A	(32,11), down	(32,9) → (32,14)
A→B	(32,24), down	(32,22) → (32,27)
B→G	(32,39), down	(32,35) → (32,42)
Reference solution
B→G, A→B, S→A,
B→G, A→B, S→A,
B→G, A→B, S→A,
B→G, A→B, B→G

Exactly twelve successful clicks.

Win validation
charge[G] == 5
Attribution and necessity check

Initially, B→G is the only successful arrow.

Its visible consequence enables exactly one newly productive arrow, A→B.

That consequence in turn enables S→A.

All three arrows and all four reservoirs are required.

The level tests vacancy propagation rather than route choice, counting, or memorization.

Level 4 — Split Headroom
Player-facing discovery

This is the first branching network.

The upper-left branch contains four pips and has only one possible goal, the five-socket north goal. The lower-left source contains four more pips. Its east goal has only three sockets, so exactly one of its pips must take the shared north route through J→H; the other three must use the longer east route.

The same J→H arrow therefore has contrasting roles:

it is necessary once;

using it twice permanently consumes too much north-goal capacity and strands a pip in the constrained upper branch.

This consequence is visible from sockets and directed reachability. There is no hidden branch identity after pips merge.

The east route is a three-stage one-slot pipeline. Productive play overlaps its stages rather than moving one pip through the entire route before touching the next.

Documentation-only schematic:

C(3) → R(1) → H(0) → G_N★(0/5)
                   ↑
E(4) → J(0) ───────┘
          \
           → P(0) → Q(0) → G_E★(0/3)
Initial reservoirs
ID	Role	Top-left	Layout	Capacity	Initial charge
C	Constrained source	(1,5)	3×1	3	3
R	Upper bottleneck	(23,5)	1×1	1	1
H	Shared north reservoir	(35,5)	1×1	1	0
G_N	North goal	(44,19)	3×2	5	0
E	Flexible source	(1,40)	2×2	4	4
J	Fork reservoir	(19,42)	1×1	1	0
P	East pipeline stage	(32,42)	1×1	1	0
Q	East pipeline stage	(45,42)	1×1	1	0
G_E	East goal	(44,55)	3×1	3	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow	Pipe centerline
C→R	(20,8), right	(18,8) → (22,8)
R→H	(32,8), right	(30,8) → (34,8)
H→G_N	(46,8), right	(42,8) → (52,8) → (52,18)
E→J	(16,45), right	(13,45) → (18,45)
J→H	(38,29), up	(22,41) → (22,33) → (38,33) → (38,12)
J→P	(29,45), right	(26,45) → (31,45)
P→Q	(42,45), right	(39,45) → (44,45)
Q→G_E	(48,52), down	(48,49) → (48,52) → (52,52) → (52,54)
Reference solution
R→H, C→R, H→G_N,
R→H, C→R, H→G_N,
R→H, C→R, H→G_N,
R→H, H→G_N,

E→J, J→H, H→G_N,

E→J, J→P,
E→J, P→Q,
J→P, E→J,
Q→G_E, P→Q, J→P,
Q→G_E, P→Q, Q→G_E

Exactly twenty-six successful clicks.

Win validation
charge[G_N] == 5
and
charge[G_E] == 3
Anti-degenerate and necessity check

The constrained branch contributes four pips and cannot reach G_E.

G_E has room for only three of E’s four pips.

Consequently J→H is required exactly once in every winning allocation.

Using J→H twice leaves at most three north sockets for four constrained pips; because G_N has no outgoing arrow, that mistake is irreversible and visually predictable.

Every edge appears in the reference solution.

Removing either fork exit makes one goal impossible to fill.

The east stages are useful concurrently: the reference sequence stages pips in J, P, and Q rather than acting as a simple one-pip corridor.

The spatial fork and disconnected reachability constraint cannot be replaced by a counter without changing the puzzle.

Level 5 — Turning Gap
Player-facing discovery

Six one-socket reservoirs form a clockwise directed ring. D is the sole empty ring socket. Both external feeders point into currently full ring reservoirs and therefore reject.

Clicking C→D succeeds. The charge moves clockwise into D, while the visible dark socket moves backward to C. Clicking B→C, then A→B, continues that backward vacancy motion. This is an immediate, local experiment: every click moves the dark socket to the arrow’s source.

The ring has two exits:

a large nine-socket goal at D;

a small two-socket goal at A.

Feeders enter at different ring positions, so the useful location of the vacancy changes over the level. A transfer that advances one pip toward G_D may simultaneously move the vacancy away from the feeder that needs it next.

Documentation-only topology:

        A → B
        ↑   ↓
        F   C ← P_C
        ↑   ↓
P_E → E ← D
        A exits to G_A★
        D exits to G_D★

The arrow D→E closes the directed ring even though the shortest reference route banks charge at D instead. It remains a live continuation: after G_D fills, a pip arriving at D can only reach G_A by continuing through E→F→A.

Initial reservoirs
ID	Role	Top-left	Layout	Capacity	Initial charge
A	Ring reservoir	(21,20)	1×1	1	1
B	Ring reservoir	(34,20)	1×1	1	1
C	Ring reservoir	(34,33)	1×1	1	1
D	Ring reservoir	(34,46)	1×1	1	0
E	Ring reservoir	(21,46)	1×1	1	1
F	Ring reservoir	(21,33)	1×1	1	1
P_E	Feeder into E	(2,46)	2×1	2	2
P_C	Feeder into C	(50,20)	2×2	4	4
G_D	Large goal	(47,43)	3×3	9	0
G_A	Small goal	(18,2)	2×1	2	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow	Pipe centerline
A→B	(31,23), right	(28,23) → (33,23)
B→C	(37,29), down	(37,27) → (37,32)
C→D	(37,42), down	(37,40) → (37,45)
D→E	(31,49), left	(33,49) → (28,49)
E→F	(24,42), up	(24,45) → (24,40)
F→A	(24,29), up	(24,32) → (24,27)
P_E→E	(17,49), right	(14,49) → (20,49)
P_C→C	(46,25), left	(49,25) → (44,25) → (44,36) → (41,36)
D→G_D	(44,49), right	(41,49) → (46,49)
A→G_A	(24,14), up	(24,19) → (24,9)
Reference solution
C→D, B→C, A→B, F→A, E→F, P_E→E,
D→G_D,

C→D, B→C, A→B, F→A, E→F, P_E→E,
D→G_D,

C→D, B→C, A→B, F→A, E→F,
D→G_D,

C→D, B→C, D→G_D,

C→D, P_C→C, D→G_D,
C→D, P_C→C, D→G_D,
C→D, P_C→C, D→G_D,
C→D, P_C→C, D→G_D,
C→D, D→G_D,

A→G_A, F→A, A→G_A

Exactly forty successful clicks.

Win validation
charge[G_D] == 9
and
charge[G_A] == 2
Mechanic attribution

The first five ring transfers have an unambiguous visual interpretation:

initial vacancy: D
C→D  gives vacancy C
B→C  gives vacancy B
A→B  gives vacancy A
F→A  gives vacancy F
E→F  gives vacancy E

At that point P_E→E, which initially rejected, succeeds. The feeder activation is therefore directly explained by the moved vacancy rather than by elapsed time or a hidden sequence.

Anti-degenerate and necessity check

Both feeders are required to supply the combined goal capacity.

Both exits are required because G_D holds nine of eleven pips and G_A holds the remaining two.

The feeder ports occur at different ring positions, so the vacancy must be moved to different places in different phases.

Five of the six ring arrows occur in the shortest reference route. The sixth, D→E, is the principled continuation past the large exit and supports winning allocations that finish G_D before the two G_A pips are staged at A and F.

Removing the ring’s return side strands P_E; removing its upper side strands P_C.

The level cannot be reduced to a single chain without changing which feeder can use the vacancy and which exit a circulating pip can reach.

Level 6 — Golden Conduit
Player-facing discovery

G_0 begins completely full and gold. Earlier levels may have encouraged the harmless provisional belief that a filled goal should never be touched.

Immediately below it, A is full, so G_0→A rejects. Clicking A→H clears A. The now-enabled G_0→A is the only way to provide enough charge for the two empty downstream goals. Clicking it visibly removes a pip from a completed gold socket and sends it into the route.

The consequence is immediate and physical:

G_0 becomes incomplete;

A fills;

the downstream network gains charge;

G_0 gains the headroom needed to accept a refill pip.

No special goal exception is introduced. The familiar transfer rule is simply applied to a gold reservoir.

G_0 must emit seven pips over the level and receive seven replacement pips:

two from W;

five from E through the one-slot staging reservoir J.

The downstream hub H must allocate three pips left and five right. A’s initial pip is part of the downstream total, creating an offset between how much leaves G_0 and how much the lower goals receive.

W(2) ───────────→ G_0★(4/4) → A(1) → H
E(5) → J(0) ────→                 /   \
                                  L     R → T
                                  ↓         ↓
                               G_1★(3)   G_2★(5)
Initial reservoirs
ID	Role	Top-left	Layout	Capacity	Initial charge
W	Direct refill source	(2,5)	2×1	2	2
E	Staged refill source	(2,18)	2×3	5	5
J	Refill staging reservoir	(19,23)	1×1	1	0
G_0	Transit goal	(26,4)	2×2	4	4
A	Full outlet bottleneck	(29,22)	1×1	1	1
H	Shared downstream hub	(29,35)	1×1	1	0
L	Left branch stage	(16,35)	1×1	1	0
R	Right branch stage	(42,35)	1×1	1	0
T	Right branch stage	(55,35)	1×1	1	0
G_1	Left goal	(2,50)	3×1	3	0
G_2	Right goal	(45,49)	3×2	5	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow	Pipe centerline
W→G_0	(20,8), right	(14,8) → (25,8)
E→J	(16,26), right	(14,26) → (18,26)
J→G_0	(22,19), up	(22,22) → (22,18) → (31,18) → (31,16)
G_0→A	(32,19), down	(32,16) → (32,21)
A→H	(32,32), down	(32,29) → (32,34)
H→L	(25,38), left	(28,38) → (23,38)
L→G_1	(19,45), down	(19,42) → (19,46) → (10,46) → (10,49)
H→R	(39,38), right	(36,38) → (41,38)
R→T	(52,38), right	(49,38) → (54,38)
T→G_2	(58,45), down	(58,42) → (58,46) → (53,46) → (53,48)
Reference solution

Each comma-separated item is one successful click:

E→J,
A→H,
G_0→A,
W→G_0,
H→L,
A→H,
G_0→A,
W→G_0,
L→G_1,
H→L,
A→H,
G_0→A,
J→G_0,
E→J,
L→G_1,
H→L,
A→H,
G_0→A,
J→G_0,
E→J,
L→G_1,

H→R,
A→H,
G_0→A,
J→G_0,
E→J,
R→T,
H→R,
A→H,
G_0→A,
J→G_0,
E→J,
T→G_2,
R→T,
H→R,
A→H,
G_0→A,
J→G_0,
T→G_2,
R→T,
H→R,
A→H,
T→G_2,
R→T,
H→R,
T→G_2,
R→T,
T→G_2

Exactly forty-eight successful clicks.

Win validation
charge[G_0] == 4
and
charge[G_1] == 3
and
charge[G_2] == 5
Anti-degenerate and necessity check

The two lower goals require eight pips. Only one starts downstream of G_0, so seven transfers out of G_0 are mandatory.

Conservation then requires all seven refill pips from W and E to enter G_0.

Leaving G_0 untouched makes the lower goals impossible.

Draining it without coordinating refills leaves the gold goal incomplete.

H is not an errand station: its single socket is simultaneously the branch choice and the headroom needed to keep A and G_0 moving.

The left and right routes have unequal depth and unequal goal capacity, so their pipeline schedules differ.

Every reservoir and every arrow appears in the reference solution.

The new interpretation of a gold reservoir is reused in Level 7 rather than appearing only once.

Level 7 — Whole Circuit
Player-facing discovery

The final level combines the established consequences without introducing another object type.

A six-node ring begins with its sole vacancy at D. A full two-socket goal G_0 feeds the ring at A, but A is initially full. G_0 must emit six pips and be refilled by:

two pips from W;

four pips from X through J.

The cycle has two exits:

C→G_1;

E→G_2.

A separate one-pip source Q can reach only G_2. Therefore one of G_2’s six sockets must be used by, or at least reserved for, Q. Filling all six sockets from the ring first permanently strands Q. This directly reuses Level 4’s shared-headroom lesson in a cycle context.

The opening ring sequence:

C→D, B→C, A→B, F→A

moves the vacancy from D backward through C, B, and A to F, exposing different ports without changing any rule. Meanwhile X→J stages a refill before G_0 has space.

Documentation-only topology:

                    G_1★
                      ↑
              A → B → C
              ↑       ↓
G_0★ → A-port F       D
  ↑           ↑       ↓
W │           E ←─────
  │           ↓
X → J       G_2★ ← Q

The duplicate A in the schematic denotes the same ring reservoir receiving from G_0; it is not a separate object.

Initial reservoirs
ID	Role	Top-left	Layout	Capacity	Initial charge
A	Ring reservoir and G_0 inlet	(23,15)	1×1	1	1
B	Ring reservoir	(36,15)	1×1	1	1
C	Ring reservoir and G_1 exit	(36,28)	1×1	1	1
D	Initial ring vacancy	(36,41)	1×1	1	0
E	Ring reservoir and G_2 exit	(23,41)	1×1	1	1
F	Ring reservoir	(23,28)	1×1	1	1
G_0	Transit goal	(3,15)	2×1	2	2
W	Direct G_0 refill	(3,2)	2×1	2	2
X	Staged G_0 refill	(35,1)	2×2	4	4
J	One-slot refill stage	(22,3)	1×1	1	0
Q	Constrained G_2 source	(23,50)	1×1	1	1
G_1	Right goal	(50,23)	2×3	6	0
G_2	Left goal	(5,39)	2×3	6	0

Initial transient state: no selected object, no hover, no feedback, no animation.

Directed arrows
Transfer	Arrow	Pipe centerline
A→B	(33,18), right	(30,18) → (35,18)
B→C	(39,24), down	(39,22) → (39,27)
C→D	(39,37), down	(39,35) → (39,40)
D→E	(33,44), left	(35,44) → (30,44)
E→F	(26,37), up	(26,40) → (26,35)
F→A	(26,24), up	(26,27) → (26,22)
G_0→A	(18,18), right	(15,18) → (22,18)
W→G_0	(9,11), down	(9,9) → (9,14)
X→J	(31,6), left	(34,6) → (29,6)
J→G_0	(18,12), left	(25,10) → (25,12) → (12,12) → (12,14)
Q→G_2	(20,53), left	(22,53) → (17,53)
C→G_1	(46,31), right	(43,31) → (49,31)
E→G_2	(20,44), left	(22,44) → (17,44)
Reference solution
C→D,
B→C,
A→B,
F→A,
X→J,
Q→G_2,

C→G_1,
B→C,
A→B,
G_0→A,
W→G_0,

C→G_1,
B→C,
A→B,
G_0→A,
W→G_0,

C→G_1,
B→C,
A→B,
G_0→A,
J→G_0,
X→J,

C→G_1,
B→C,
A→B,
G_0→A,
J→G_0,
X→J,

C→G_1,
B→C,
A→B,
G_0→A,
J→G_0,
X→J,

C→G_1,
B→C,
A→B,
G_0→A,
J→G_0,

E→G_2,
D→E,
C→D,
B→C,
A→B,

E→G_2,
D→E,
C→D,
B→C,

E→G_2,
D→E,
C→D,

E→G_2,
D→E,
E→G_2

Exactly fifty-four successful clicks.

Win validation
charge[G_0] == 2
and
charge[G_1] == 6
and
charge[G_2] == 6

The reference final state is:

A=B=C=D=E=F=W=X=J=Q=0
G_0=2
G_1=6
G_2=6
Dependency and anti-degenerate check

The ring starts with only one vacancy, at D. Both G_0→A and the relevant upstream cycle moves are initially constrained by visible occupancy.

G_0 must output exactly six pips because W and X together supply six replacement pips and G_0 must finish full.

J has one socket, so X cannot simply dump its four pips; staging must be synchronized with vacancy creation in G_0.

Q has exactly one outgoing arrow, to G_2. If six other pips fill G_2 first, Q is visibly and permanently stranded.

C→G_1 and E→G_2 compete with continuing around the cycle. Banking a pip creates a vacancy at an exit node; circulating it preserves goal headroom but relocates the vacancy elsewhere.

The reference route uses five ring arrows. The remaining continuation, E→F, is the principled bypass when G_2 is full or its last socket is being reserved for Q; it carries surplus charge onward toward G_1.

The cycle, constrained source, staging reservoir, and transit goal form a dependency network rather than a sequence of independent stations.

Removing the cycle’s return side prevents downstream surplus from reaching the opposite exit. Removing either refill route leaves G_0 incomplete. Removing Q→G_2 makes total goal capacity unreachable.

No new mechanic is introduced in the final level. Every difficulty comes from the placement and interaction of previously demonstrated rules.

7. Epistemic-solvability audit
Foundational attribution
Level 1

The first useful click changes one visible pip at the source and one visible socket at the destination. No other state changes. The partially completed goal supplies positive evidence for the objective.

Level 2

The same transfer button is observed once while blocked by a full destination and once immediately after that destination is cleared. This isolates capacity as the causal variable.

Level 3

Each downstream move enables exactly one upstream move. The vacancy’s location is always visible, and no branch choice or new object competes with the inference.

Later mechanic evidence
Shared headroom

Level 4 makes the constrained contribution and both goal capacities simultaneously visible. The flexible fork must be used in both directions, and one fork edge changes from required to harmful after one use.

Level 7 reuses the same reasoning asymmetrically: Q has only one destination, so the ring must not consume all of that goal’s headroom.

Cycles

Level 5 begins with blocked feeder arrows and one conspicuous dark ring socket. One adjacent ring transfer moves that darkness to its source, giving immediate evidence for vacancy circulation.

Level 7 reuses the ring as a shared router rather than as a standalone throughput device.

Goal drainage

Level 6 presents a filled goal adjacent to a blocked outlet and two visibly empty downstream goals. Clearing the outlet makes the goal’s outgoing arrow immediately useful. The resulting loss of one gold pip and gain of downstream charge cannot be explained by an older “goals are sinks” rule.

Level 7 reuses the transit goal while adding refill staging and cycle headroom.

8. Implementation verification checklist

An implementation should include automated replay tests for all seven authored levels.

For each level:

Initialize exactly the listed capacities and charges.

Assert that total initial charge equals combined goal capacity.

Replay the listed reference solution.

Assert that every listed click succeeds.

Assert that no proper prefix satisfies the win condition.

Assert that the final state satisfies every goal exactly.

Assert that total charge remains invariant after every step.

Assert that each arrow hitbox is unique and does not overlap a reservoir or another arrow hitbox.

Test one empty-source rejection and one full-destination rejection wherever those conditions are reachable.

Confirm that directional inputs and SPACE leave all persistent state unchanged.

Confirm that animation frames consume no additional steps.

Confirm that clicking an outgoing arrow from a goal uses the same validation and transfer code as clicking an outgoing arrow from a plain reservoir.

The verified shortest successful-transfer depths for the specified count-state graphs are:

Level 1:  4
Level 2:  5
Level 3: 12
Level 4: 26
Level 5: 40
Level 6: 48
Level 7: 54

This gives Charge Pour one compact operational rule, a clean three-level causal curriculum, and four increasingly composed puzzles about the dual movement of charge and empty capacity.
