Take It Back

This is a game about taking an artifact back becoming difficult because carrying it reverses which closed gates are legal and prevents any further gate changes.

Campaign size: exactly 7 authored levels.

The player enters every level empty-handed. The sole objective is visible on the board: retrieve the yellow artifact from its pedestal and reach the green receiver while still carrying it.

The route inward is rarely the route outward. Closed plain gates admit only an empty agent; closed artifact-marked gates admit only the carrier. Opening either kind removes that restriction, but the swung leaf becomes a wall on another floor cell. All gate preparation must eventually be completed before taking the artifact, because the carrier’s hands are visibly occupied.

There are no hidden switches, inventories, counters, random outcomes, enemies, or timed phases.

1. Action space

The game uses only:

UP, DOWN, LEFT, RIGHT

SPACE

clicking any cell

Clicks are deliberately unbound in this game family. A click consumes one action-budget unit but causes no logical or visual change. There is no cursor, selected object, or click target.

Directional actions move the agent by exactly one logical cell.

SPACE is positional:

If the empty agent is orthogonally adjacent to the loaded pedestal, retrieve the artifact.

Otherwise, if the agent occupies exactly one gate’s handle cell, attempt to toggle that gate.

Otherwise, show a brief neutral interaction ring and change nothing.

Authored states never make a pedestal interaction and gate-handle interaction simultaneously eligible.

2. Objective and validation

A level succeeds immediately after a step() if:

agent_position == exit_position
AND carrying_artifact == true

Nothing else is part of the win predicate.

Gate states, unused routes, action count, and pedestal position after collection are ignored. Entering the exit empty-handed is allowed but does not complete the level. The artifact cannot be dropped, returned, deposited early, or exchanged.

When the win predicate becomes true, the receiver flashes green and yellow for four render frames and the level becomes terminal.

3. Native geometry and rendering
Logical board

All levels use a 15 × 15 logical lattice with coordinates:

x = 0…14, increasing rightward
y = 0…14, increasing downward

A logical cell maps to a 4 × 4 pixel region on the 64 × 64 canvas:

pixel_origin_x = 2 + 4x
pixel_origin_y = 2 + 4y

Thus the playfield occupies pixels 2–61 on each axis. There are no decorative grid lines and no outline around every logical cell.

A one-pixel action-budget bar occupies canvas row y=0. Canvas row y=1 is black separation. The remaining outside margin is black.

Base colors
Element	Color ID
Outside canvas background	5
Non-floor space	4
Traversable floor	1
Authored wall-edge strip	3
Empty agent	9
Artifact	11
Pedestal	2 and 3
Exit receiver	14 with 11 socket
Empty-only gate	10 with 0 aperture
Carrier-only gate	15 with 11 artifact mark
Hinge and handle	12
Invalid-action feedback	8
Secondary feedback pulse	7

Only these IDs are used; all are within the required 0–15 range.

Agent

The agent is a connected blue 3 × 3 form centered inside its logical cell. When carrying, a yellow 2 × 2 diamond is rendered prominently against its upper/front half, with two blue arm pixels around it. The occupied logical footprint remains one cell.

The carrying state never needs to be remembered from a prior frame.

Pedestal and artifact

The pedestal occupies one solid logical cell. The agent cannot enter it.

A loaded pedestal has a dark-gray base and a centered yellow diamond. After collection, the yellow diamond disappears, but the base remains solid.

Exit

The exit is a traversable floor cell containing a green U-shaped receiver and the same yellow diamond silhouette used by the artifact. This visible match is the only goal cue needed.

In Level 1, collecting the artifact also causes the receiver to pulse once, providing positive evidence that the two belong together.

4. Gate model

Every gate is static level data with the following fields:

id
class                 EMPTY_ONLY or CARRIER_ONLY
initial_open           boolean
door_cell              one logical cell
axis_endpoint_a        floor cell
axis_endpoint_b        floor cell
handle_cell            cell from which SPACE operates the gate
swing_cell             cell occupied by the leaf while open

axis_endpoint_a, door_cell, and axis_endpoint_b are collinear.

Only these two movement edges connect through the gate:

axis_endpoint_a ↔ door_cell
door_cell ↔ axis_endpoint_b

Even when perpendicular floor cells touch the door cell geometrically, the frame blocks movement between them.

The handle cell always shares an edge with the door cell. Some authored gates have a handle cell that is non-floor. Such a handle is visibly wall-facing and cannot be reached; this is ordinary geometry, not a different gate type.

The swing cell shares either an edge or a corner with the door cell. A corner-adjacent swing cell represents a leaf pivoting around the shared hinge corner.

Closed gate passage

A closed gate is a selective aperture:

EMPTY_ONLY gate:
    passable iff carrying_artifact == false

CARRIER_ONLY gate:
    passable iff carrying_artifact == true

An authorized agent moves into the gate’s door cell normally. It then requires another directional action to move out through the opposite axis endpoint.

An unauthorized move leaves the agent in place:

An empty agent rejected by a carrier-only gate makes the yellow artifact mark pulse.

A carrier rejected by an empty-only gate shows the carried yellow diamond with a red slash.

Open gate passage

An open gate’s doorway admits either carrying state. Its class remains visible on the frame and relocated leaf.

The swing cell is occupied by the open leaf and behaves exactly like a wall for movement. A leaf may cover another gate’s handle cell; this is intentional in Levels 6 and 7.

Gate operation

A gate can be operated only when:

agent_position == gate.handle_cell
AND carrying_artifact == false
Closed → open

The operation succeeds if:

the swing cell is floor;

the swing cell does not contain the agent, pedestal, exit, gate doorway, or another open leaf.

The doorway becomes universally passable and the swing cell becomes occupied.

Open → closed

The operation succeeds if the door cell is unoccupied.

The swing cell is cleared and the doorway resumes its class-selective behavior.

Carrier attempt

If the carrier presses SPACE on a handle:

the handle rattles horizontally by one rendered pixel;

a red crossed-hands shape appears for three frames;

no gate state changes.

Blocked toggle

If a leaf cannot reach its destination, the hinge and conflicting cell flash red. The gate remains unchanged.

5. Fixed wall edges

Most walls are represented simply by absent floor cells. A few levels additionally specify blocked movement edges between two adjacent floor cells.

These are rendered as a single dark-gray strip on the shared boundary. They are actual authored walls, not internal grid decoration. No other cell boundaries are drawn.

6. Exact update order

Each action produces exactly one step().

1. If terminal, return the terminal state unchanged.

2. Record the submitted action and spend one budget unit.

3. Resolve the action using the old settled state.

   CLICK:
       no logical or visual effect.

   DIRECTION:
       a. Compute the adjacent target cell.
       b. Reject if the edge is an authored blocked edge.
       c. Reject if target is non-floor or the loaded/empty pedestal.
       d. Reject if target is occupied by an open gate leaf.
       e. Apply gate-frame axis restrictions.
       f. If entering a closed gate door, apply its class test.
       g. If not rejected, update agent_position.

   SPACE:
       a. Retrieve an adjacent loaded artifact if eligible.
       b. Else attempt the gate whose handle_cell equals agent_position.
       c. Else produce the neutral interaction ring.

4. Test the win predicate on the resulting state.

5. If won, success takes precedence even when this was the final budget unit.
   Otherwise, reaching zero budget ends the attempt in failure.

6. Return the new settled state.

There are no automatic state updates between actions.

7. Animation contract

Animations clarify causal motion but never consume actions or budget.

Gate swing

A successful toggle is logically committed by the current step(). Before the next action can be accepted, the renderer shows:

old leaf position;

leaf halfway around the orange hinge;

leaf mostly in the destination cell;

final settled gate state.

The hinge is a connected two-pixel orange shape. The closed-state swing destination has a small dark-gray L-shaped floor recess pointing toward the hinge, making the future occupied cell predictable.

Artifact retrieval

Over three render frames, the yellow diamond moves from the pedestal to the agent. The final frame is the settled carrying_artifact=true state.

Invalid movement

The agent shifts one rendered pixel toward the obstruction and returns. The logical position never changes.

8. State representation

A minimal runtime state is:

agent_position: (x, y)
carrying_artifact: boolean
pedestal_loaded: boolean
gate_open: dictionary gate_id → boolean
steps_remaining: integer
won: boolean
failed: boolean
transient_feedback: optional render-only event

There is always exactly one agent, one artifact, one pedestal, and one exit.

There is no cursor state and no selected-object state.

9. Level encoding

In each floor mask below:

# = non-floor
. = floor

Rows are listed in order y=0…14. Character position is x=0…14.

Gate door cells, swing cells, pedestal cells, and exit cells are included as floor in the mask and then overridden by their objects. A pedestal remains collision-solid.

Reference traces use:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

A finite-state breadth-first search over:

(agent position, carrying state, pedestal state, gate-open bitmask)

was used to verify the stated shortest trace lengths.

Level 1 — Take It
Player-facing discovery

The agent begins directly beside the yellow artifact. The green receiver above uses the same yellow diamond silhouette.

The productive experiment is immediate: SPACE. The artifact visibly transfers into the agent’s arms and the receiver pulses. Two upward moves then complete the level.

This isolates one causal unknown: how to take the artifact and what counts as success. There are no gates, handles, decoys, branches, or competing interactions.

Authored state
###############
###############
###.###########
###.###########
###..##########
###############
###############
###############
###############
###############
###############
###############
###############
###############
###############
Agent:              (3,4)
Carrying:           false
Pedestal:           (4,4), loaded
Exit:               (3,2)
Gates:              none
Blocked edges:      none
Selection/cursor:   none
Action budget:      96
One shortest solution
S U U

Length: 3 actions.

Necessity and attribution check

The exit cannot be completed empty-handed.

The pedestal blocks movement, so walking into the artifact is not a competing explanation.

SPACE causes only one visible state change: artifact leaves pedestal and appears on agent.

The matching yellow exit socket and collection pulse provide positive goal evidence.

Every floor cell participates in the solution.

Level 2 — The Handle
Player-facing discovery

A broad artifact-marked gate blocks the empty agent. The agent starts on the only orange handle cell. The artifact and receiver are visibly beyond the gate.

Pressing SPACE swings the leaf into its upper alcove, clearing the doorway. The moved leaf is visible but does not yet obstruct the solution.

This level introduces one new causal unknown: standing on a handle and pressing SPACE swings a gate.

The gate’s class-specific meaning is not required yet; “this closed gate blocks me” is sufficient for this lesson.

Authored state
###############
###############
###..##########
##....#########
###############
###############
###############
###############
###############
###############
###############
###############
###############
###############
###############
Agent:              (2,3)
Carrying:           false
Pedestal:           (5,3), loaded
Exit:               (4,2)
Initial open gates: none
Blocked edges:      none
Selection/cursor:   none
Action budget:      128
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
G	Carrier-only	Closed	(3,3)	(2,3) and (4,3)	(2,3)	(3,2)
One shortest solution
S R R S U

Length: 5 actions.

The two uses of SPACE are locally unambiguous: first on the handle, later adjacent to the pedestal.

Necessity and attribution check

The closed gate rejects the initial empty agent.

No route goes around it.

The only productive initial experiment is the visibly adjacent handle.

Opening immediately relocates the leaf and permits movement through the door.

The swing cell is a visible side alcove, so leaf occupation is observed without already becoming a planning burden.

Level 3 — Different Way Back
Player-facing discovery

The inward gate is narrow and plain. It admits the empty agent while remaining closed. The marked gate above the artifact staging cell rejects the empty agent.

After collection:

the marked gate admits the carrier;

the narrow gate rejects the carrier.

Both handles face non-floor wall pockets, preventing the already-known “just open it” solution. The player must trust the changed carrying state.

This introduces one new causal rule: the two closed frame shapes test opposite carrying states.

Authored state
###############
####.##########
####.##########
####.##########
##....#########
###############
###############
###############
###############
###############
###############
###############
###############
###############
###############
Agent:              (2,4)
Carrying:           false
Pedestal:           (5,4), loaded
Exit:               (4,1)
Initial open gates: none
Blocked edges:      none
Selection/cursor:   none
Action budget:      160
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
E	Empty-only	Closed	(3,4)	(2,4) and (4,4)	(3,5), non-floor	(3,3), non-floor
C	Carrier-only	Closed	(4,3)	(4,4) and (4,2)	(5,3), non-floor	(3,3), non-floor
Recommended discovery trace
R R U S L U U U

The first U is rejected by the carrier-only gate while empty. After collection, L is rejected by the empty-only gate. These two inexpensive probes establish both halves of the rule before the successful upward route.

One shortest solution
R R S U U U

Length: 6 actions.

Necessity and attribution check

“All closed gates block” is disproved when the empty agent crosses E.

“The marked gate is simply unlocked forever by collection” is distinguished by the nearby post-collection probe against E.

The handles cannot be reached, so gate operation cannot bypass the class test.

The artifact must remain visibly carried to traverse C and complete the level.

The route after collection is geometrically different rather than merely a reversed corridor.

Level 4 — Doorway or Floor
Player-facing discovery

The upper route reaches closed carrier gate H, but its handle is in a lower service lane. Gate G at the start grants access to that service lane.

Opening G immediately places its leaf on the upper route. The player can now reach H’s handle below, but opening H swings its own leaf onto the service path just used to arrive. A lower return loop remains available.

The player must:

use G’s doorway while accepting that its leaf blocks the upper route;

open H;

return around H’s new leaf;

close G to clear its swing cell;

use the now-clear upper route through open H.

This introduces one new causal property and uses it twice: an open gate creates a wall at its swing cell.

Authored state
###############
###############
###############
###############
###############
##########.####
##########.####
###.........###
###......######
######...######
###############
###############
###############
###############
###############
Agent:              (3,8)
Carrying:           false
Pedestal:           (11,7), loaded
Exit:               (10,5)
Initial open gates: none
Selection/cursor:   none
Action budget:      384
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
G	Carrier-only	Closed	(4,8)	(3,8) and (5,8)	(3,8)	(4,7)
H	Carrier-only	Closed	(8,7)	(7,7) and (9,7)	(8,8)	(7,8), corner-adjacent

Additional blocked wall edges:

(5,8) ↔ (5,7)
(6,8) ↔ (6,7)
(7,8) ↔ (7,7)

These three walls separate the upper route from the service lane. They are visible solid strips, not grid borders.

One shortest solution
S R R R R R
S D L L U L L L
S U R R R R R R R
S U U

Length: 26 actions.

Phase interpretation:

Open G and reach H’s side handle.

Open H and use the lower loop because H’s leaf blocks the arrival path.

Close G, clearing the upper route, and traverse open H.

Collect the artifact and reach the receiver without any further gate operation.

Necessity and attribution check

The direct upper route visibly ends at closed H.

H’s handle is reachable only through G’s service route.

G’s open leaf occupies (4,7), so the upper route cannot be used while G remains open.

H’s open leaf occupies (7,8), preventing immediate reversal through the service lane.

The fixed wall strips prevent leakage from the handle lane into the artifact side.

G must end closed while H must end open, despite both sharing the same operational rules.

Every floor branch appears in the verified solution.

Level 5 — Before Your Hands Are Full
Player-facing discovery

The lower branch leads quickly to the artifact. A tempting blind attempt is to collect it immediately, pass through the marked carrier gate C, and arrive at gate B’s handle.

Pressing SPACE there causes the crossed-hands feedback. Gate B remains closed and the carrier is trapped between empty-only gates.

The upper branch is therefore not an optional detour: it reaches B’s handle while the agent is still empty. B must be opened before the artifact is taken.

This introduces one new causal rule: carrying the artifact disables all gate handles.

The failure is physically local and quick: only four moves separate collection from the rattling handle.

Authored state
###############
###############
#########.#####
########....###
###......######
###.####.######
##.........####
###############
###############
###############
###############
###############
###############
###############
###############
Agent:              (2,6)
Carrying:           false
Pedestal:           (10,6), loaded
Exit:               (11,3)
Initial open gates: none
Blocked edges:      none
Selection/cursor:   none
Action budget:      512
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
A	Empty-only	Closed	(4,4)	(3,4) and (5,4)	(4,5), non-floor	(4,3), non-floor
D	Empty-only	Closed	(5,6)	(4,6) and (6,6)	(5,7), non-floor	(5,5), non-floor
C	Carrier-only	Closed	(8,5)	(8,6) and (8,4)	(9,5), non-floor	(7,5), non-floor
B	Empty-only	Closed	(9,3)	(8,3) and (10,3)	(8,3)	(9,2)
Blind-player impasse

Taking the direct lower route first produces:

... collect artifact
L U U U S

The carrier crosses C, reaches B’s handle, and the final S visibly fails because both hands are occupied.

The level must then be restarted; no reset action is added to the action space.

One shortest solution
R U U R R R R R U S
D L L L L L D D
R R R R R R S
L U U U R R R

Length: 32 actions.

Phase interpretation:

Use empty-only gate A to reach B’s handle and open B.

Return through A to the initial junction.

Use empty-only gate D to reach and collect the artifact.

Use closed carrier gate C, then the already-open B, to reach the exit.

Necessity and anti-cleanliness check

A reaches preparation space but not the artifact.

D reaches the artifact but not the exit after collection.

C is useless inward but becomes the only carrier return link.

B is useful only if operated during the earlier empty phase.

Opening B’s leaf does not create a universal linear corridor; the player must return to the original junction and choose a different branch.

The preparation order is derived from future carrying state, not from a checklist validator.

The authored route forms a four-connection dependency network: A → B preparation, D → artifact, C → return, B → exit.

Level 6 — Close What You Need
Player-facing discovery

Gate T begins closed at the starting handle. It must be opened to enter the preparation region, but its leaf settles on the future artifact route.

Inside, carrier gate F begins open. Its leaf visibly occupies B’s orange handle cell. F’s open doorway is useful because it lets the empty agent reach the inner handle of return gate R.

The correct preparation is:

open T and enter;

cross open F;

open R;

return across F;

close F, clearing B’s handle;

open B;

return and close T, clearing the artifact approach.

After collection, the carrier uses closed marked gates K and F, but open empty-only gates R and B.

The same closed carrier-gate rule now has a positive strategic role: closing F both clears a floor cell and prepares a carrier-only passage.

Authored state
###############
###############
###############
###############
###############
#########.#####
####.#.....####
##.....###...##
#####...####.##
#######.###...#
#######.....###
#######..######
###############
###############
###############
Agent:              (7,11)
Carrying:           false
Pedestal:           (13,9), loaded
Exit:               (2,7)
Initial open gates: F
Blocked edges:      none
Selection/cursor:   none
Action budget:      768
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
T	Carrier-only	Closed	(7,10)	(7,11) and (7,9)	(7,11)	(8,10)
F	Carrier-only	Open	(6,7)	(6,8) and (6,6)	(6,8)	(5,7)
R	Empty-only	Closed	(9,6)	(8,6) and (10,6)	(8,6)	(9,5)
B	Empty-only	Closed	(4,7)	(5,7) and (3,7)	(5,7)	(4,6)
D	Empty-only	Closed	(10,10)	(9,10) and (11,10)	(10,11), non-floor	(10,9), non-floor
K	Carrier-only	Closed	(11,7)	(10,7) and (12,7)	(11,6), non-floor	(11,8), non-floor
Local order evidence

Initially, F’s leaf occupies (5,7), which is exactly B’s handle. The obstruction is not abstract: the player cannot stand on the handle.

Closing F too early is recoverable. The empty agent remains at F’s outer handle and may reopen it to reach R.

Gate K prevents an empty agent who opened R from using the return passage as a shortcut to the artifact. Later, the carrier passes K without changing it.

One shortest solution
S U U U L U U R R S
L L D D S L U S
D R R D D D S
R U R R R U R S
U U L L U L L L L D D L U L L L

Length: 49 actions.

Phase interpretation:

Open T, cross F, and open R.

Return, close F, and open B.

Return to the starting side and close T.

Reach the artifact through fixed empty-only gate D.

Carry the artifact through closed K, open R, closed F, and open B.

Required pre-collection gate state:

T closed
F closed
R open
B open
D closed
K closed

Only the exit condition is validated; this state is required by the authored geometry, not by an attribute checker.

Necessity and anti-cleanliness check

T open is needed for preparation, but its leaf physically blocks the artifact route.

F open is needed to reach R, but its leaf physically blocks B’s handle and final path.

R must be opened before collection because it is empty-only.

B must be opened before collection for the same reason.

D prevents a carrier from reversing the inward route.

K prevents an empty shortcut and later becomes a useful closed carrier gate.

F changes from access tool to obstruction to carrier passage without changing its operational rule.

Wrong orders are recoverable before collection, supporting experimentation rather than memorization.

The verified solution uses every traversable branch.

Level 7 — The Prepared House
Player-facing discovery

The final level nests the prior dependencies.

T again grants preparation access while placing its leaf on the artifact approach. F begins open and its leaf covers B’s handle.

Beyond F, the preparation region splits into two genuinely independent tasks:

Q must be opened on the left.

R must be opened on the right.

These two may be prepared in either order. Only after both are ready should the player return and close F.

Closing F reveals B’s handle. Opening B then swings its leaf onto Q’s handle, visibly sealing the preparation phase. This is safe only because Q has already been opened.

The final carrier route is:

artifact
→ closed K
→ open R
→ closed F
→ open B
→ open Q
→ exit

No gate can be operated after commitment.

Authored state
###############
###############
###############
###############
####.##########
####.#####.####
###.........###
####....###...#
######..####..#
#######..###..#
########.###.##
########.....##
########..#####
###############
###############
Agent:              (8,12)
Carrying:           false
Pedestal:           (13,9), loaded
Exit:               (4,4)
Initial open gates: F
Selection/cursor:   none
Action budget:      1024
Gate	Class	Initial	Door	Axis endpoints	Handle	Swing
T	Carrier-only	Closed	(8,11)	(8,12) and (8,10)	(8,12)	(9,11)
F	Carrier-only	Open	(7,7)	(7,8) and (7,6)	(7,8)	(6,7)
Q	Empty-only	Closed	(4,6)	(4,7) and (4,5)	(5,6)	(3,6)
B	Empty-only	Closed	(5,7)	(6,7) and (4,7)	(6,7)	(5,6)
R	Empty-only	Closed	(10,6)	(9,6) and (11,6)	(9,6)	(10,5)
K	Carrier-only	Closed	(12,7)	(13,7) and (11,7)	(12,6), non-floor	(12,8)
D	Empty-only	Closed	(11,11)	(10,11) and (12,11)	(11,12), non-floor	(11,10), non-floor

Additional blocked wall edge:

(6,7) ↔ (6,6)

This wall prevents the outer B-handle cell from bypassing F into the inner preparation region. It is the only additional blocked edge in the level.

Nested handle dependencies

Initially:

F is open
F leaf occupies B handle at (6,7)

After F is closed:

B handle becomes reachable

After B is opened:

B leaf occupies Q handle at (5,6)

Therefore:

Q must be opened before B
R must be opened before F is closed
F must be closed before B can be opened
T must be closed before the artifact route is usable

The order between Q and R is deliberately unconstrained. The mature puzzle is a partial-order dependency network, not a single prescribed chain.

If the player closes F and opens B before preparing Q, the mistake remains recoverable while empty:

close B from its handle;

reopen F;

enter and open Q;

return, close F, and reopen B.

One shortest solution
S U U U L U U U
L L S R R R R S
L L D D S L U S
D R D R D D D S
R U R R R U U U R S
U L L U L L L L D D L U L L U U U

Length: 59 actions.

Phase interpretation:

Open T and cross open F into the inner preparation area.

Open Q and R. Their order may be exchanged.

Return, close F, and open B.

Return to T and close it, clearing the artifact route.

Travel through fixed empty-only gate D and collect the artifact.

Traverse closed K, open R, closed F, open B, and open Q to the exit.

Required pre-collection gate state:

T closed
F closed
Q open
B open
R open
K closed
D closed
Necessity and final anti-cleanliness check

Leaving T open blocks the artifact approach with its leaf at (9,11).

Leaving F open blocks B’s handle and the carrier’s outward route.

Closing F before opening R makes R’s handle unreachable to an empty agent.

Opening B before Q covers Q’s handle with B’s leaf.

Leaving Q closed blocks the carrier after B.

Leaving R closed blocks the carrier before F.

Opening is not universally beneficial: F and both carrier-only gates are most useful closed.

K blocks the empty preparation shortcut yet becomes a legal carrier passage.

D makes the artifact pickup a genuine commitment by denying the reverse route.

The wall between (6,7) and (6,6) prevents an unintended geometric bypass rather than enforcing an abstract rule.

The exit route, artifact route, two preparation branches, and all three strategically occupied swing cells participate in the solution.

The final route is completely visible in the settled pre-collection frame: no future trigger, hidden state, or checker requirement remains.

Campaign progression summary
Level	New discovery	Reinterpretation or composition	Shortest length
1	Retrieve artifact; carry it to matching exit	Positive goal evidence	3
2	Operate a visible handle with SPACE	Open doorway ignores gate class	5
3	Closed gate classes test opposite carrying states	Inward route cannot simply be reversed	6
4	An open leaf occupies a floor cell	One gate must stay open; another must be closed	26
5	Carrying disables gate operation	Prepare a return gate before commitment	32
6	A leaf can cover another handle	Close a carrier gate to clear a handle and prepare passage	49
7	Nested handle and route dependencies	Independent inner tasks converge into an irreversible prepared route	59

The seven levels remain one focused game family. Difficulty grows from understanding a visible state transition to preparing a spatially valid return network in which open and closed gates have context-dependent advantages, without introducing any unrelated devices or hidden predicates.
