Pulse Gates

This is a game about rotating elbow gates becoming difficult because every rotation recomputes both the pulse network and the set of gates the player is still allowed to touch.

Authored campaign size: exactly 7 levels.

Pulse Gates is a click-only topology puzzle. The player can rotate only gates currently reached by the source pulse. Rotating one gate therefore changes two things at once:

the physical channel network;

the player’s future action set.

Progress is not equivalent to illuminating more territory. A useful click may deliberately extinguish most of the board, provided another illuminated approach is ready to take over.

1. Player-facing rules

The board contains one blue source reservoir, fixed channels, rotatable elbow gates, and one hollow green receiver.

Every gate is an elbow with exactly two adjacent open arms: NE, ES, SW, or WN.

A glowing gate may be clicked. A click rotates it one quarter-turn clockwise.

After every valid click, the old pulse disappears and a new pulse travels through the entire currently connected network.

Only gates reached by the settled new pulse glow and remain clickable.

A gate may extinguish itself by rotating away from its only source-connected approach.

The level is solved as soon as the settled pulse reaches and fills the receiver.

There are no hidden switches, remembered activations, one-way channels, delayed logical effects, energy costs, or timers.

The central invariant

Successive orientations share one arm:

Rotation	Shared arm
NE → ES	E
ES → SW	S
SW → WN	W
WN → NE	N

If the source is feeding a gate through that shared arm, the gate can remain reached after the click. If the source is feeding it only through the arm that disappears, the gate goes dark unless another route immediately takes over.

2. Action space

The provided action space is respected exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any rendered cell

Only clicks are used by the game.

UP, DOWN, LEFT, RIGHT, and SPACE are accepted no-op actions. There is no cursor, avatar, or persistent selection state.

Click behavior

A click is mapped to the logical cell under the clicked rendered pixel.

Clicking any part of a glowing gate’s complete 5×5 module performs a valid rotation.

Clicking a dark gate, fixed channel, source, receiver, empty cell, or outside the native board produces a two-frame blocked ripple and no logical state change.

The player never needs to hit a small hinge pixel or individual arm.

Gate halos make the full set of currently valid click targets visible in every settled frame.

The host environment’s normal episode reset may restore the authored initial state. Reset is not an in-game action and adds nothing to the action vocabulary.

There is no move budget. Invalid experiments cannot exhaust a resource.

3. Native geometry and rendering contract
Logical board

The native gameplay lattice is 11 columns × 9 rows.

Logical coordinates use (x,y).

(0,0) is the upper-left logical cell.

x ∈ [0,10]

y ∈ [0,8]

Orthogonal adjacency only.

The complete render canvas is 64×64 pixels.

Logical-to-render mapping

Logical cell (x,y) occupies the inclusive 5×5 pixel block:

pixel_x = 4 + 5x through 8 + 5x
pixel_y = 9 + 5y through 13 + 5y

Thus the native board occupies pixels x=4..58, y=9..53.

No logical grid lines, cell borders, board border, or decorative background pattern are drawn. Unoccupied cells merge into the background.

Port stencil

Within a 5×5 logical-cell block, use local coordinates (u,v), each in 0..4.

The central hub is:

HUB = {(u,v) | 1 ≤ u ≤ 3 and 1 ≤ v ≤ 3}

Port arms are:

N = {(1,0), (2,0), (3,0)}
E = {(4,1), (4,2), (4,3)}
S = {(1,4), (2,4), (3,4)}
W = {(0,1), (0,2), (0,3)}

A channel or gate with port set P renders HUB plus the arms in P. Reciprocal arms therefore meet as a continuous three-pixel-wide connection across adjacent cell boundaries.

Palette usage

Only the following integer color IDs are used:

ID	Use
0	white reached-gate halo and completion spark
1	unreachable gate arms
2	unreachable fixed channels and receiver inlet
3	dark gate hub
4	background and hollow receiver center
8	invalid-click ripple
9	source reservoir core
10	source reservoir body
11	active pulse fill
12	orange clockwise hinge cue
14	receiver rim and completed receiver fill

All remaining palette IDs are unused.

Entity appearance
Source

The source is a 5×5 reservoir module:

body: color 10;

central 3×3 core: color 9;

fixed outlet arms shown literally;

source-connected adjacent channels use pulse color 11.

The source cannot be clicked.

Fixed channel

A fixed channel uses the port stencil corresponding to its listed ports.

unreachable: color 2;

reached: color 11.

Straight, elbow, and T-shaped fixed channels all obey the same literal port rule. There is no hidden difference between shapes.

Rotatable gate

A gate is a 5×5 elbow module.

Unreached state:

arms: color 1;

hub: color 3;

clockwise crescent/chevron over the hub: color 12.

Reached state:

arms: color 11;

hub: color 0;

clockwise cue remains color 12;

the four local corner pixels flash or remain color 0 as the clickable halo.

The orientation is always visible from the elbow silhouette. The clockwise cue has no variable state; it only communicates the action’s fixed direction.

Receiver

The receiver is a hollow green socket:

outer rim: color 14;

center: color 4;

inlet arm: color 2 while unfilled.

When reached:

center fills with color 14;

inlet shows the arriving pulse;

four brief white completion sparks use color 0.

The receiver is visually distinct from both gates and the source and cannot be clicked.

4. Exact state machine
Authoritative state
level_id
orientation[g] ∈ {NE, ES, SW, WN} for every gate g
reached_cells                  derived
reached_gates                  derived
solved ∈ {false, true}
presentation_phase             render-only
last_clicked_gate              render-only or null

There is no persistent cursor or selected gate.

Port graph

For every occupied logical cell c, define its current port set P(c):

fixed channel: authored fixed ports;

source: authored outlet ports;

receiver: authored inlet ports;

gate: ports determined by its current orientation.

Two orthogonally adjacent occupied cells a and b are connected iff:

a has a port toward b; and

b has the reciprocal port toward a.

The pulse-reached region is the connected component containing the source cell.

A gate is reached only when its central cell belongs to this component. Merely pointing an arm at a bright adjacent channel is insufficient if the gate lacks the reciprocal open arm.

Valid-click update order

All logical changes occur synchronously inside step() after receiving the action.

1. Read the reached-gate mask from the pre-action settled state.

2. Resolve the clicked pixel to a logical cell and then, if present, a gate.

3. If the click does not target a pre-action reached gate:
      orientations unchanged
      reached region unchanged
      solved unchanged
      queue blocked-ripple feedback
      return

4. Otherwise:
      rotate clicked gate clockwise by exactly one orientation
      rebuild every cell's current port set
      recompute the source-connected component from scratch
      derive the new reached-gate mask
      set solved = true iff receiver is in the new component
      queue rotation and pulse animation

Important implementation consequences:

Click validity is checked before rotation.

Reachability is recomputed after rotation.

A clicked gate may be valid before the action and dark afterward.

No gate retains historical access.

The previous illuminated region has no gameplay authority.

A receiver reached by the recomputed graph solves immediately; no confirmation click is required.

Once solved=true, subsequent actions leave the completed frame unchanged.

Exact success condition

A level succeeds iff, after a valid gate rotation and full reachability recomputation:

receiver_cell ∈ reached_cells

Relevant:

current gate orientations;

current reciprocal channel connectivity;

source-component reachability.

Ignored:

number of reached gates;

total illuminated area;

orientations of gates outside the winning path;

click count;

previous pulse regions;

order in which equivalent commuting clicks were made.

Allowed:

extinguishing previously reached regions;

leaving gates dark;

reaching the receiver through any valid source-connected route.

Forbidden:

rotating a dark gate;

propagating through nonreciprocal ports;

treating visually adjacent but disconnected arms as connected;

preserving reachability from an earlier frame.

5. Animation and causal feedback

Animation is required because every valid action rotates an object and triggers a global physical recomputation.

Animation frames are presentation-only and consume no player actions, steps, energy, or timer units. step() is synchronous; another action is accepted only after the settled frame is shown.

Valid gate click

Rotation frame 0: the clicked hub flashes orange; old pulse fill begins dimming.

Rotation frame 1: render the union of the old and new elbow stencils in color 12, making the clockwise sweep visible.

Rotation frame 2: render only the new orientation.

Remove all previous pulse fill.

In the newly computed graph, calculate shortest logical-cell distance d(c) from the source.

Pulse frame k illuminates all final-reachable cells satisfying d(c) ≤ 2k.

When all final-reachable cells are shown, render the settled frame and the final clickable halos.

Disconnected old branches turn dark before the new wave begins. This makes contractions as legible as expansions.

Initial frame

On level load or reset:

orientations are set to their authored values;

reachability is computed once;

the pulse wave animates from the source without rotating anything;

the first actionable state is the fully settled frame.

No autonomous state updates occur afterward.

Invalid click

A dark gate or non-gate click produces:

one frame with a color-8 ring around the clicked logical cell;

one smaller color-8 ring;

the unchanged settled frame.

No pulse recomputation occurs.

Completion

When the final pulse reaches the receiver:

the receiver center fills color 14;

a three-frame white ring expands from the receiver;

the final completed state remains visible.

6. Orientation and notation

Orientations rotate as follows:

NE → ES → SW → WN → NE

Documentation gate IDs such as A, B, and C are not rendered. They exist only in this specification and test vectors.

In solution sequences:

A×3

means three separate clicks on A, with a complete rotation, pulse animation, and settled frame after each click. It is not one compound action.

7. Shared three-feed mesh construction

Levels 4–7 use a recurring source spine and gate-mesh grammar.

Source spine

Place exactly:

SRC  (0,0): S

FIX (0,1): NSE
FIX (1,1): EW

FIX (0,2): NS

FIX (0,3): NSE
FIX (1,3): EW

FIX (0,4): NS

FIX (0,5): NE
FIX (1,5): EW

This creates three permanently source-connected approaches from the west, ending beside logical gate positions (2,1), (2,3), and (2,5).

A gate beside one of these approaches is reached only if its current orientation actually includes W.

Mesh gate coordinates

A mesh node (column,row) maps to:

logical_x = 2 + 2·column
logical_y = 1 + 2·row

For every pair of authored gate nodes that are horizontally adjacent, place one EW fixed channel at their midpoint.

For every pair that are vertically adjacent, place one NS fixed channel at their midpoint.

No other mesh channels are present.

8. Campaign progression
Level	Name	Gates	New discovery or planning pressure	Verified shortest depth
1	One Bend	1	click rotates; pulse reaching receiver wins	1
2	The Lit Frontier	3	only pulse-reached gates become manipulable	3
3	Keep the Feed	3	a gate can strand itself and be re-fed from another side	4
4	Three-Feed Weave	6	deliberate contractions and repeated source handoffs	23
5	The Missing Corner	8	a familiar frontier becomes a temporary bridge and later a liability	37
6	Climb the Pulse	9	carry access from the lower mesh into a receiver above the network	51
7	Full Handoff	9	orchestrate several expansions, contractions, and right-side re-feeds	64

The first three levels are deliberately short because each isolates one causal unknown. The later click counts come from reversible access dependencies and forced revisits, not corridors, timers, or movement padding.

9. Authored levels
Level 1 — One Bend
Player-facing discovery

The pulse already reaches one glowing gate. The receiver sits immediately above it, while the gate’s current elbow points downward. The gate halo is the only click affordance.

Clicking the gate rotates its downward arm to the north, and the pulse visibly travels into the receiver.

The objective requires no legend or abstract validation: liquid-like energy visibly reaches a socket.

State-machine instance
SRC  (1,4): E

FIX (2,4): EW
FIX (3,4): EW
FIX (4,3): NS

GATE A (4,4): SW

RCV  (4,2): S

All unlisted cells are empty.

Initial interaction state:

selected_gate = null
presentation_phase = settled
reached_gates = {A}
solved = false
Exact solution
A

Transition:

A: SW → WN

The W source approach survives, while the new N arm connects to the receiver.

Final test state:

A = WN
reached_gates = {A}
receiver reached = true
Attribution and necessity check

There is exactly one visible gate, one valid click, and one changing rule.

The rotation and receiver lighting are immediate and spatially adjacent.

Removing the gate, either fixed segment, or receiver approach destroys the solution.

No unrelated object changes simultaneously.

The level is intentionally one click; it establishes the causal alphabet rather than testing planning.

Level 2 — The Lit Frontier
Player-facing discovery

Only gate A initially glows. Gates B and C are visibly present and oriented, but dark and without halos.

Clicking A sends the pulse upward and causes B to glow. Clicking B extends the reached frontier leftward and causes C to glow. Clicking C turns the pulse downward into the nearby receiver.

The player receives three immediate demonstrations that clickability is a consequence of the current pulse rather than a permanent property of gates.

State-machine instance
SRC  (0,6): E

FIX (1,6): EW
FIX (2,6): EW

GATE A (3,6): SW

FIX (3,5): NS
FIX (3,4): NS

GATE B (3,3): ES

FIX (2,3): EW

GATE C (1,3): NE

FIX (1,4): NS

RCV  (1,5): N

Initial interaction state:

selected_gate = null
reached_gates = {A}
solved = false

Clicking B or C initially produces the blocked ripple and changes nothing.

Exact solution
A → B → C

Settled transitions:

After action	Orientations changed	Reached gates
initial	—	{A}
A	A: SW→WN	{A,B}
B	B: ES→SW	{A,B,C}
C	C: NE→ES	{A,B,C} plus receiver

Final test state:

A = WN
B = SW
C = ES
receiver reached = true
Attribution and necessity check

Each action causes exactly one new gate to gain a halo.

The newly reached gate is spatially at the pulse frontier.

No dark gate is useful before the preceding expansion.

The three-gate U-shaped route is compact; it is not lengthened by an empty corridor.

All three gates and all listed fixed channels are on the only possible source-to-receiver route.

Level 3 — Keep the Feed
Player-facing discovery

Two gates, A and C, are initially illuminated from different source branches. The receiver lies west of A, but A cannot yet turn toward it safely.

Clicking A opens a downward route and illuminates B. Clicking B turns its elbow toward C. A second click on A then demonstrates the important danger: A rotates away from its original eastern feed, and the pulse collapses until only independently supplied C remains glowing.

That contraction creates a local impasse with an obvious experiment. C is now the only halo, and its next clockwise orientation visibly points west toward dark B. Clicking it feeds the chain from the opposite side, relighting B, A, and the receiver.

This is the first deliberate pulse handoff.

State-machine instance
SRC  (9,3): W

FIX (8,3): EW
FIX (7,3): EWS
FIX (6,3): EW
FIX (5,3): EW

GATE A (4,3): NE

FIX (3,3): EW
RCV  (2,3): E

FIX (4,4): NS
GATE B (4,5): WN

FIX (5,5): EW
GATE C (6,5): ES

FIX (7,4): NS
FIX (7,5): NS
FIX (7,6): NW
FIX (6,6): NE

The T-channel at (7,3) splits the source:

westward toward A;

downward and around to the south side of C.

Initial interaction state:

selected_gate = null
reached_gates = {A,C}
solved = false

B is dark. Its northern approach becomes reachable only after A opens south.

Exact illustrative solution
A → B → A → C

Settled transitions:

After action	Key change	Reached gates
initial	A fed from east; C fed from south	{A,C}
A	A: NE→ES; south route opens	{A,B,C}
B	B: WN→NE; points east toward C	{A,B,C}
A	A: ES→SW; original east feed disappears	{C}
C	C: ES→SW; westward alternate feed takes over	{A,B,C} plus receiver

Final test state:

A = SW
B = NE
C = SW
receiver reached = true

Other four-click orders exist when C is prepared before the contraction. They obey the identical rules; no special ordering exception is encoded.

Attribution and necessity check

The contraction after the second A click is required by the showcased solution and cannot be explained by the earlier “frontier only expands” hypothesis.

The only remaining halo is the exact gate that can restore access.

C is not a recovery decoy: every solution must manipulate A, B, and C.

The fixed source branch reaching C is visibly illuminated from the beginning.

The level remains recoverable after the demonstrated contraction rather than forcing an early reset.

Level 4 — Three-Feed Weave
Player-facing discovery

The board now exposes three source-fed western approaches at once. The receiver is immediately east of lower-right gate F, but neither lower gate can simply be turned toward it while surviving its source approach.

The crucial realization is that the middle-left gate C and lower-left gate E act as a reusable access pump:

one configuration broadens the lit region;

another deliberately contracts it;

the remote gates are adjusted during the broad phase;

the contraction changes which side supplies the next rotation.

Illuminating more gates is useful only when the newly reached gates are configured before the source handoff.

Mesh and coordinates

Use the shared source spine.

Gate-node set:

{(0,0), (1,0),
 (0,1), (1,1),
 (0,2), (1,2)}

Exact internal fixed channels:

FIX (3,1): EW
FIX (3,3): EW
FIX (3,5): EW

FIX (2,2): NS
FIX (2,4): NS
FIX (4,2): NS
FIX (4,4): NS

Receiver:

RCV (5,5): W
Gate state
ID	Coordinate	Initial orientation
A	(2,1)	ES
B	(4,1)	ES
C	(2,3)	SW
D	(4,3)	WN
E	(2,5)	SW
F	(4,5)	SW

Initial interaction state:

selected_gate = null
reached_gates = {C,E}
solved = false
Exact shortest solution example

Every entry below is a separate valid click followed by a full pulse settlement.

Phase	Click sequence	Reached gates after phase	Structural purpose
1	C → A → C×2 → E	{A,B,C,D,E}	use C to prepare A, contract, then let E energize the upper block
2	B → A×3 → C×2 → D×2	{A,B,C,E}	configure both upper gates while the broad feed exists
3	A → C×3	{A,C,E}	return the middle-left pump to the approach needed by the lower handoff
4	E → F → D → C×2 → F	{A,C,D,F} plus receiver	reach F, establish its alternate approach, then give it the final turn

Concatenated test vector:

C → A → C×2 → E
→ B → A×3 → C×2 → D×2
→ A → C×3
→ E → F → D → C×2 → F

Verified solution depth: 23 valid clicks.

Final test state, in A..F order:

[SW, SW, NE, SW, NE, NE]

Final reached gates:

{A,C,D,F}

Win validation:

receiver (5,5) reached = true
Why the authored instance is necessary

The apparent direct plan is to rotate E toward F. That fails physically: E loses its western source approach before it can provide the required eastward route. The upper loop must first provide another approach.

Exhaustive orientation-state verification establishes:

no solution exists while forbidding clicks on any one of A..F;

removing any one internal inter-gate segment makes the level unsolvable;

removing any one of the three source-feed branches makes the level unsolvable;

23 is the shortest valid solution depth.

The level is therefore not a checklist of independently adjusted elbows. Every gate participates in a cyclic access dependency.

Level 5 — The Missing Corner
Player-facing discovery

This level extends the weave to the right but leaves the upper-right mesh node absent. The empty notch is not decorative: there is no top route into the right column.

Gate H, beside the receiver, must first be reached from below and later supplied through the middle-right gate E. A gate that looks like the final frontier becomes a temporary bridge, goes dark during a handoff, and is revisited only after the left-side access pump has been rebuilt.

The solution repeatedly alternates between:

configuring the left two-column engine;

sending a pulse into the notched right wing;

preserving a new approach before collapsing the old one;

returning left to reconfigure the source side.

Mesh and coordinates

Use the shared source spine.

Gate-node set:

{(0,0), (1,0),
 (0,1), (1,1), (2,1),
 (0,2), (1,2), (2,2)}

The upper-right node (2,0) is absent.

Exact internal fixed channels:

FIX (3,1): EW

FIX (2,2): NS
FIX (4,2): NS

FIX (3,3): EW
FIX (5,3): EW

FIX (2,4): NS
FIX (4,4): NS
FIX (6,4): NS

FIX (3,5): EW
FIX (5,5): EW

There is no gate at (6,1), no channel at (5,1), and no channel at (6,2).

Receiver:

RCV (7,5): W
Gate state
ID	Coordinate	Initial orientation
A	(2,1)	ES
B	(4,1)	ES
C	(2,3)	SW
D	(4,3)	WN
E	(6,3)	ES
F	(2,5)	SW
G	(4,5)	SW
H	(6,5)	SW

Initial interaction state:

selected_gate = null
reached_gates = {C,F}
solved = false
Exact shortest solution example
Phase	Click sequence	Reached gates after phase	Structural purpose
1	C → A → C×2 → F	{A,B,C,D,F}	build the first broad left-side feed
2	B → A×3 → C×2 → D×2	{A,B,C,F}	configure the upper return while the middle is available
3	A → C×3 → F → G → D → C×2	{A,C,D,F,G}	open the lower route into the right wing and preserve its return
4	F×3 → C → G → H → E	{A,C,D,E,F,G,H}	reach H, then use E to establish its alternate vertical supply
5	D → A×3 → C×2 → D → H	{A,B,C,D,E,F,H} plus receiver	rebuild the left pump, re-energize the right side, and give H its final turn

Concatenated test vector:

C → A → C×2 → F
→ B → A×3 → C×2 → D×2
→ A → C×3 → F → G → D → C×2
→ F×3 → C → G → H → E
→ D → A×3 → C×2 → D → H

Verified solution depth: 37 valid clicks.

Final test state, in A..H order:

[ES, SW, WN, NE, SW, WN, NE, NE]

Final reached gates:

{A,B,C,D,E,F,H}

Win validation:

receiver (7,5) reached = true
Why the authored instance is necessary

A simple extension of the Level 4 plan fails because the missing upper-right node prevents the right wing from being supplied from above. H cannot be rotated to the receiver while relying only on the route that first reached it. E must acquire a different role as a vertical feeder, after which the left engine must be rebuilt.

Exhaustive verification establishes:

every one of the eight gates must be clicked in every solution;

every listed inter-gate segment is necessary;

all three source-feed branches are necessary;

the absent upper-right node materially changes the dependency graph;

37 is the shortest valid solution depth.

Level 6 — Climb the Pulse
Player-facing discovery

The complete 3×3 mesh is now present, but the receiver sits above upper-right gate C. The source enters from the left at three heights, so the receiver appears to be nearest the top feed. That apparent direct route is misleading for a principled reason: the top-row gates cannot all be oriented toward the receiver while remaining supplied solely from the west.

The pulse must first travel through the lower mesh, establish the right-hand loop, hand access back into the middle, and only then climb into C from a surviving approach.

The right column is not simply the final destination:

I and F first help create a return feed;

that feed is abandoned;

the lower pump is rebuilt;

C is reached during a later broad expansion;

the final contraction leaves exactly the upper route required by the receiver.

Mesh and coordinates

Use the shared source spine.

Gate-node set:

{(0,0), (1,0), (2,0),
 (0,1), (1,1), (2,1),
 (0,2), (1,2), (2,2)}

Exact internal fixed channels:

FIX (3,1): EW
FIX (5,1): EW

FIX (3,3): EW
FIX (5,3): EW

FIX (3,5): EW
FIX (5,5): EW

FIX (2,2): NS
FIX (4,2): NS
FIX (6,2): NS

FIX (2,4): NS
FIX (4,4): NS
FIX (6,4): NS

Receiver:

RCV (6,0): S
Gate state
ID	Coordinate	Initial orientation
A	(2,1)	WN
B	(4,1)	ES
C	(6,1)	ES
D	(2,3)	ES
E	(4,3)	SW
F	(6,3)	ES
G	(2,5)	SW
H	(4,5)	SW
I	(6,5)	SW

Initial interaction state:

selected_gate = null
reached_gates = {A,G}
solved = false
Exact shortest solution example
Phase	Click sequence	Reached gates after phase	Structural purpose
1	G → E → B → A×2 → D×2 → E×2	{A,B,D,G}	connect the initially separate top and lower fronts, then contract to a controllable core
2	A → D×3 → G → H → E → D×2 → G×3 → D	{A,D,E,G,H}	build and retime the lower-middle loop
3	H → I → F → E	{A,B,D,E,G}	visit the right column, use it to establish a return, then hand control back left
4	A×3 → D×2 → E → H×2 → E	{A,B,D,G}	preserve the right-side configuration while rebuilding the central supply
5	B×2 → A → D×3 → G → H → F → C	{A,B,C,D,E,F,G,H}	carry the pulse upward and make C manipulable
6	B → C → E×2 → D×2	{A,B,C,D,E} plus receiver	orient C north, then contract away the lower dependency without losing the upper feed

Concatenated test vector:

G → E → B → A×2 → D×2 → E×2
→ A → D×3 → G → H → E → D×2 → G×3 → D
→ H → I → F → E
→ A×3 → D×2 → E → H×2 → E
→ B×2 → A → D×3 → G → H → F → C
→ B → C → E×2 → D×2

Verified solution depth: 51 valid clicks.

Final test state, in A..I order:

[SW, ES, WN,
 NE, WN, WN,
 NE, WN, WN]

Final reached gates:

{A,B,C,D,E}

Win validation:

receiver (6,0) reached = true
Why the authored instance is necessary

The lower-right gates are not a detour inserted for length. Without configuring H, I, and F, the pulse cannot supply the side of C that survives its final rotations. Conversely, leaving the right wing maximally illuminated prevents the later central orientations from being reached.

Exhaustive verification establishes:

every gate A..I must be clicked at least once;

all twelve mesh connections are individually necessary;

all three source-feed branches are individually necessary;

forbidding any gate click or removing any single mesh edge makes the authored state unsolvable;

51 is the shortest valid solution depth.

Level 7 — Full Handoff
Player-facing discovery

The same complete mesh now places the receiver east of lower-right gate I. The receiver looks close to the bottom source feed, but I begins in SW: it cannot be turned to an east-facing orientation while relying only on the path that reaches it from the west.

The final level demands several complete pulse handoffs:

activate the upper and middle mesh from the lower source front;

deliberately collapse the network to preserve a new central approach;

use the upper-right gates as a temporary feed rather than the destination;

rebuild the left source pump;

establish a north-side approach into the lower-right region;

rotate I once while newly reached;

abandon that approach and reconstruct it from the opposite side;

rotate I into the receiver while its surviving arm remains supplied.

The lit area repeatedly shrinks from nearly the whole mesh to only three or four gates. Each contraction preserves a different remote orientation, so none is a reset to an earlier equivalent state.

Mesh and coordinates

Use the same complete 3×3 gate-node set and all twelve fixed mesh channels from Level 6.

Receiver:

RCV (7,5): W
Gate state
ID	Coordinate	Initial orientation
A	(2,1)	ES
B	(4,1)	SW
C	(6,1)	ES
D	(2,3)	SW
E	(4,3)	SW
F	(6,3)	WN
G	(2,5)	SW
H	(4,5)	SW
I	(6,5)	SW

Initial interaction state:

selected_gate = null
reached_gates = {D,G}
solved = false
Exact shortest solution example
Phase	Click sequence	Reached gates after phase	Structural purpose
1	D → A → D×2 → G → E	{A,B,D,E,G}	merge the middle and lower source fronts
2	A×3 → D×2 → E×2 → B×2 → A → D×3 → G	{A,D,G,H}	configure the upper return, then contract into the lower doorway
3	H → C → B → E	{A,D,E,G,H}	use the newly opened lower route to prepare the top-right side
4	D×2 → G×3 → D → E	{A,B,C,D,E,F,G}	expand across the upper and right mesh without yet touching I
5	F×2 → B → A×3 → D×2 → E×2	{A,B,D,G}	store the right-column orientations, then deliberately collapse back left
6	A → D×3 → G → E → D×2 → G×3 → D	{A,D,E,G,H}	rebuild a differently oriented lower handoff
7	H → I → F → E	{A,B,D,E,G}	reach and rotate I once, then transfer its supply away from the original route
8	A×3 → D×2 → E → I	{A,B,D,E,F,G,I} plus receiver	re-establish the alternate approach and rotate I east while preserving reach

Concatenated test vector:

D → A → D×2 → G → E
→ A×3 → D×2 → E×2 → B×2 → A → D×3 → G
→ H → C → B → E
→ D×2 → G×3 → D → E
→ F×2 → B → A×3 → D×2 → E×2
→ A → D×3 → G → E → D×2 → G×3 → D
→ H → I → F → E
→ A×3 → D×2 → E → I

Verified solution depth: 64 valid clicks.

Final test state, in A..I order:

[ES, SW, SW,
 WN, NE, SW,
 WN, NE, NE]

Final reached gates:

{A,B,D,E,F,G,I}

Win validation:

receiver (7,5) reached = true
Why the authored instance is necessary

The obvious bottom-row plan fails because I cannot progress from SW to an east-facing orientation while remaining fed only from the west. The upper and middle mesh must create a north-side feed, and that feed itself cannot be retained while preparing the final orientation unless the left pump is revisited.

The spatial layout therefore cannot be replaced by:

a counter;

a linear list of gate settings;

a corridor;

a sequence of independent rotations;

a “make all gates bright” objective.

Exhaustive verification establishes:

every gate A..I must be clicked in every solution;

every one of the twelve mesh connections is necessary;

every one of the three source-feed branches is necessary;

no solution exists when any one gate is made nonmanipulable;

no solution exists when any one mesh connection is removed;

64 is the shortest valid solution depth.

10. Implementation and validation checklist

An implementation is conformant only if all of the following hold:

Exactly 7 levels are present.

The render canvas is exactly 64×64.

Only color IDs 0–15 are emitted.

No text, gate letters, numbers, arrows made from text, grid borders, or HUD labels appear.

Gate documentation IDs are never rendered.

The complete 5×5 module of a reached gate is clickable.

Reachability is visibly persistent in settled frames.

A dark gate never rotates.

Click validity uses pre-action reachability.

Rotation occurs before pulse recomputation.

Pulse reachability is recomputed globally from scratch.

Connections require reciprocal ports.

A clicked gate may become dark in the same action.

Animation frames consume no steps.

There is no timer, energy bar, click budget, or hidden loss counter.

The receiver alone defines success.

Reaching more gates is neither required nor always beneficial.

Invalid clicks never alter gate orientation or reachability.

Keyboard actions remain no-ops.

Initial, intermediate, and final gate orientations for every listed test vector reproduce the stated reached-gate sets.

The verified minimum valid-click depths are:

Level 1:  1
Level 2:  3
Level 3:  4
Level 4: 23
Level 5: 37
Level 6: 51
Level 7: 64

The resulting game has one compact vocabulary—source, literal channels, clockwise elbows, and receiver—but supports expansion, self-stranding, alternate feeds, controlled contractions, frontier staging, and multi-phase pulse handoffs entirely through authored spatial context.
