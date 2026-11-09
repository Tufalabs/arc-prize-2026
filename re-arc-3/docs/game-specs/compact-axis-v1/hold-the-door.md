Hold the Door

This is a game about pulling statues into place becoming difficult because every open door needs a body left behind, while every pull needs empty space on the far side.

Authored campaign size: exactly 7 levels.

The player is not collecting keys or configuring devices. They are deciding when a statue is still movable equipment and when it must become part of the architecture. Every puzzle uses the same physical vocabulary: walk, grip, back away, occupy a plate, and pass through the gate that remains open.

There is no randomness, hidden information, move limit, energy system, timer, death state, or in-level scoring. Exploration is unlimited.

1. Native space and coordinates

The rendered canvas is 64×64 pixels.

The native gameplay lattice is 14×12 logical cells, with:

x = 1…14, increasing rightward.

y = 1…12, increasing downward.

Coordinates are 1-indexed.

One logical cell renders as a 4×4 pixel region.

Logical cell (x,y) begins at render pixel:

px = 4 + 4(x - 1)
py = 8 + 4(y - 1)

Thus the gameplay area occupies pixels x=4…59, y=8…55, leaving quiet margins around it.

Notation used below:

H[y, x1:x2]        all cells (x,y) with x1 ≤ x ≤ x2
V[x, y1:y2]        all cells (x,y) with y1 ≤ y ≤ y2
R[x1:x2, y1:y2]    inclusive rectangle
∪                   set union

Every coordinate not included in a level’s floor set is solid non-floor terrain.

2. Action space

The accepted action space is exactly:

UP
DOWN
LEFT
RIGHT
SPACE
CLICK(cell)

The game uses UP, DOWN, LEFT, RIGHT, and SPACE.

Clicks are deliberately not part of the mechanic:

CLICK on any logical or rendered cell is a no-op.

No object changes appearance on hover.

No cursor or selected-cell state is stored.

Nothing on the board is presented as clickable.

This avoids mixing arbitrary clicking into a physical movement game.

3. Settled game state

The complete persistent state is:

level_id
agent_position: (x,y)
agent_facing: UP | DOWN | LEFT | RIGHT
statue_positions: ordered tuple of (x,y)
gripped_statue_id: integer | null
won: boolean

Gate states are derived rather than stored.

Statue IDs exist only so that grip state and visual cap variants can be tracked. All statues have identical collision and movement rules.

There is no persistent animation phase, cursor, move counter, energy value, plate latch, gate timer, or hidden flag.

Every level initializes with:

gripped_statue_id = null
won = false
cursor state = none
4. Core physical rules
4.1 Walking and facing

An arrow action always changes the agent’s facing to that direction, even when movement is blocked.

When the agent is not gripping a statue, an arrow attempts to move the agent by one logical cell.

The move succeeds only if the destination:

belongs to the level’s floor set;

contains no statue;

is not occupied by a closed gate.

Walking into a statue never pushes it. The statue shudders but remains in place.

This facing-before-collision rule is important: several solution traces deliberately use a blocked arrow to face a statue before pressing SPACE.

4.2 Gripping

When no statue is gripped, SPACE examines the cell immediately in front of the agent.

If that cell contains a statue, the statue becomes gripped.

Otherwise SPACE is a no-op.

A gripped statue is shown by a persistent taut handle or chain between the agent and that statue.

When a statue is already gripped, SPACE releases it regardless of the agent’s current facing.

The player cannot switch directly from one statue to another. They must release, then grip again.

4.3 Pulling

Let:

s be the position of the gripped statue;

a be the position of the agent;

d = a - s.

Because the gripped statue is adjacent, d is one cardinal unit vector.

The statue is pulled only when the player chooses the arrow corresponding to d: the direction directly away from the statue.

If the agent’s backing destination a+d is traversable, the update is simultaneous:

statue_position' = a
agent_position'  = a + d

The statue therefore enters the cell the agent just vacated.

Example:

statue at (5,6)
agent at  (6,6)

RIGHT produces:

statue at (6,6)
agent at  (7,6)

While gripping:

Moving directly away attempts a pull.

Moving toward the statue does not move either object.

Moving perpendicular to the chain does not move either object.

These non-pull arrows still update facing.

The grip remains active until SPACE is pressed.

A pull fails if the agent’s backing destination is a wall, statue, or closed gate. The chain and obstructing object visibly shudder; all positions remain unchanged.

There is no pushing under any circumstance.

5. Plates and gates

Each plate belongs to exactly one gate. Documentation labels pairs A, B, C, and D; these letters are never rendered.

For pair j, let:

Pj = plate cell
Gj = set of gate cells

The gate is open exactly when the plate is currently occupied by the agent or any statue:

open(j) =
    agent_position == Pj
    OR
    any statue_position == Pj

This is recomputed after every positional update.

Consequences:

A plate is not a toggle.

Leaving it closes its gate immediately.

An agent can demonstrate a gate but normally cannot leave the plate and reach a distant gate.

A statue can become a persistent door-holder.

Pulling a statue onto a plate while the agent leaves that plate transfers occupancy continuously. The gate does not close for one frame between them.

All cells in a gate footprint open and close together. The seven authored levels use one-cell gate footprints, but the transition rule supports connected footprints.

Plates and gates have no click behavior.

5.1 Gate-closing collision rule

A gate may never close on an agent or statue.

After a proposed movement:

Proposed positions are calculated.

Plate occupancy is recomputed.

Proposed gate states are derived.

If any newly closed gate cell would contain the agent or a statue, the entire positional update is rejected.

On rejection:

Agent and statue positions remain as before the action.

Grip remains unchanged.

The new facing still applies.

The affected gate flashes and shudders.

Because positions were rejected, the old plate occupant remains and the gate remains open.

This prevents gate-jamming or crushing exploits without adding another game state.

6. Exact step() resolution order

All logical state changes occur only inside step().

1. If won is true:
      return the unchanged completed state.

2. Clear transient feedback from the previous step.

3. If the action is CLICK:
      make no logical change and return.

4. If the action is SPACE:
      a. If gripped_statue_id is not null, release it.
      b. Otherwise, grip the adjacent faced statue if one exists.
      c. Check success, although SPACE cannot normally alter position.
      d. Return.

5. The action is an arrow:
      a. Update agent_facing.
      b. Derive gate states from the state at action start.
      c. If no statue is gripped, propose a one-cell walk.
      d. If a statue is gripped, propose a pull only when the arrow
         points directly away from that statue.
      e. Reject if the agent destination is not floor, contains a
         statue, or is a gate closed at action start.
      f. Apply proposed agent/statue positions simultaneously.
      g. Recompute plate occupancy and gate states.
      h. Reject the positional update if a closed gate would overlap
         an occupant.
      i. Otherwise commit the proposed positions.

6. Set won = true exactly when agent_position == exit_position.

7. Produce the settled observation and any presentation-only frames.
7. Success and failure validation

A level succeeds exactly when:

agent_position == exit_position

Everything else is ignored by the success validator:

Facing is irrelevant.

Grip may be active or released.

Statues may occupy any legal cells.

The validator does not explicitly require every plate to be occupied.

The validator does not explicitly require every gate to be open.

Action count is irrelevant.

Click history is irrelevant.

The authored geometry naturally forces the required gates to be open when the agent reaches the exit.

A statue on the exit does not win. It merely blocks the agent until moved.

After success, further actions are no-ops and the completed board remains visible.

There is no terminal loss condition. The environment’s ordinary episode-reset API may restore the level, but reset is not a player action and is not part of step(action).

8. Rendering contract
8.1 Used colors

Only these integer color IDs are used:

Purpose	Color ID
Highlights, grip chain, pair motif	0
Wall top highlight	1
Wall body, statue base, depressed recess	2
Traversable floor	3
Background and distant non-floor	4
Deep wall shadow, inactive plate recess	5
Pair B	6
Pair D	8
Pair A	9
Agent	11
Statues	12
Exit	14
Pair C	15

Colors 7, 10, and 13 are unused.

The board is predominantly neutral. Saturated colors occupy only the small interactive modules and moving pieces.

8.2 Floor and walls

Contiguous floor cells render as one uninterrupted color-3 surface. No internal cell borders or grid lines are drawn.

A non-floor cell adjacent to floor is rendered as a connected wall mass:

wall body: color 2;

exposed upper edge: color 1;

lower or recessed shadow: color 5.

Borders are drawn only around the exterior contour of connected terrain, never between adjacent cells of the same type.

Non-floor cells farther from any floor are background color 4.

Logical collision always uses the floor set, not sprite pixels.

8.3 Agent sprite

Within a 4×4 logical cell, using local coordinates u,v ∈ 0…3:

Color 11 body pixels:

(1,1), (2,1), (1,2), (2,2)

Color 0 facing tab:

UP:    (1,0), (2,0)
DOWN:  (1,3), (2,3)
LEFT:  (0,1), (0,2)
RIGHT: (3,1), (3,2)

The facing tab changes even when an attempted move is blocked.

8.4 Statue sprite

All statues use color 12 and the same one-cell collision footprint.

Common body pixels:

row v=1: (0,1), (1,1), (2,1), (3,1)
row v=2: (0,2), (1,2), (2,2), (3,2)

Color-2 base pixels:

(1,3), (2,3)

The cap shape provides identity only:

S0 cap: (1,0), (2,0)
S1 cap: (0,0), (1,0)
S2 cap: (2,0), (3,0)
S3 cap: (0,0), (1,0), (2,0), (3,0)

These variants have no different behavior. The first level containing multiple variants requires both to be pulled in the same way, providing visible evidence that the caps are tracking aids rather than statue types.

When gripped, color-0 handle pixels appear on the touching edges of the agent and statue, joined by a taut one-pixel connector.

Before gripping, the adjacent statue currently faced by the agent has four
color-0 corner handles. This is derived targeting feedback for SPACE, not a
selection state.

8.5 Plate

A plate is a recessed 4×4 floor module.

Its perimeter uses the pair color.

Its center 2×2 pixels use color 5 when unoccupied.

Its center uses color 2 when occupied.

An occupying sprite is displayed one render pixel lower during the settling feedback, making the depression physically visible.

Four pair-colored corner pixels are redrawn above an occupant so the plate remains identifiable beneath a statue.

8.6 Gate

A closed gate fills its logical cell with the pair color and is visibly solid.

An open gate:

exposes ordinary color-3 floor;

leaves four small pair-colored corner blocks representing the retracted slab;

remains identifiable even when traversed.

A gate has no separate lock, timer, or memory.

8.7 Exit

The exit is a color-14 diamond inlay:

(1,0), (2,0),
(0,1), (3,1),
(0,2), (3,2),
(1,3), (2,3)

Its center is color 0.

The green perimeter remains partially visible beneath a statue. This is used in Level 1 as positive goal evidence.

8.8 Pair motifs and traces

Color is not the only pairing cue. Each pair also has a two-pixel color-0 motif on both its plate and gate:

Pair	Color	Motif pixels
A	9	north-middle (1,0),(2,0)
B	6	west-middle (0,1),(0,2)
C	15	south-middle (1,3),(2,3)
D	8	east-middle (3,1),(3,2)

A one-pixel orthogonal trace in the pair color connects plate and gate through the authored waypoints specified per level.

Trace rules:

Waypoints use logical coordinates.

Consecutive waypoints are joined horizontally or vertically.

The trace follows the reserved edge channel of the crossed logical cells rather than their center.

It is rendered above static terrain.

It has no collision and no interaction.

It never changes state.

Pair endpoint motifs remain visible when an occupant covers part of the line.

Pair labels such as “A” are never rendered.

9. Animation and feedback

Animations are presentation-only. They do not call step(), consume actions, or create intermediate collision states.

Ordinary walk

A successful walk snaps to the next logical cell. No ornamental walking cycle is required.

Pull

A successful pull uses four deterministic presentation frames:

old settled positions with taut chain;

agent begins backing while the statue tilts toward the vacated cell;

both sprites interpolate halfway;

new settled logical positions.

The logical transition is the simultaneous one-cell update defined earlier.

Plate and gate

If plate occupancy changes:

the occupant settles one pixel;

the plate recess changes;

the gate retracts or extends over two additional frames.

If occupancy transfers directly from the agent to a statue, the gate remains open and does not play a close–open flicker.

Invalid action

The blocking wall, statue, or gate shifts one render pixel toward the attempted motion and returns. Final positions are unchanged.

Completion

The exit alternates once between colors 14 and 0, then the completed state freezes.

10. Campaign progression
Level	Discovery or composition	Verified minimum actions
1	Statues are pull-only; grip and back away	7
2	Release, circle, and regrip from another side	15
3	Plates are momentary occupancy mechanisms	15
4	Stage a traveler before committing a permanent holder	36
5	Temporarily remove and then restore a holder	43
6	Pre-stage two travelers and perform an adjacent grip handoff	77
7	Allocate four statues while one plate changes from transit square to final assignment	102

The first three levels introduce one causal unknown at a time. Levels 4–7 add no unrelated device families; they derive difficulty from statue order, approach geometry, preserved backing lanes, revisits, and momentary occupancy.

In the reference traces below:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

Every character is one step() call. This includes arrows that only change facing because movement is blocked.

11. Authored Level 1 — The Covered Exit
Player-facing discovery

The green exit is visibly underneath the statue. The agent begins directly beside and facing that statue.

The most likely movement probe, LEFT, makes the statue shudder but does not push it. Pressing SPACE instead produces an unmistakable chain. Backing right then moves both pieces and reveals the complete green exit.

There are no gates, plates, alternate objects, or decoys. The only new causal package is:

Face a statue, grip it, and move away to pull it.

Exact initial state
Floor:
R[6:8, 5:6]

Agent:
position = (7,6)
facing   = LEFT
grip     = null

Statues:
S0 = (6,6)

Exit:
(6,6)

Pairs:
none

Initial gate states:
none

Cursor:
none

The statue and exit share a coordinate because the exit is terrain and the statue is an occupant.

Floor-cell count: 6.

Verified minimum solution
SRSULLD

Settled phases:

Actions	Result
S	Grip S0 while facing left
R	Agent (8,6), S0 (7,6); the exit is fully exposed
S	Release
ULLD	Circle above S0 and enter exit (6,6)

Minimum depth: 7 actions.

Attribution and necessity check

Walking into S0 cannot solve the level because pushing never occurs.

The exit is blocked until S0 moves.

The only useful first state change is the visible grip.

Every traversable cell is occupied by the agent or statue during the minimum trace.

The initial failed push probe is safe and reversible.

12. Authored Level 2 — Change Sides
Player-facing discovery

The statue again lies between the agent and the exit, but pulling it directly moves it away from the exit and leaves it in the corridor.

Remaining attached is now a local impasse: the agent cannot walk around while gripping. Pressing SPACE removes the chain. The lower two-cell-deep pocket visibly permits the agent to circle to another side, face the statue, and regrip it.

The level cannot be explained by the Level 1 strategy of one pull followed by walking around. The statue must be pulled in a different direction after regripping.

Exact initial state
Floor:
H[6, 4:7]
∪ R[6:7, 7:8]

Agent:
position = (6,6)
facing   = LEFT
grip     = null

Statues:
S0 = (5,6)

Exit:
(4,6)

Pairs:
none

Initial gate states:
none

Cursor:
none

Floor-cell count: 8.

Verified minimum solution
SRSDLUSDSRUULLL

Phase breakdown:

Actions	Settled result
SRS	Pull S0 right to (6,6); agent (7,6); release
DLUS	Circle to (6,7), use blocked U to face S0, then grip
D	Pull S0 down to (6,7) while agent backs to (6,8)
S	Release
RUULLL	Return through the cleared upper corridor and enter (4,6)

Minimum depth: 15 actions.

Attribution and necessity check

There is no gate or other cause that could explain the statue’s change of direction.

The lower pocket is used both to regrip and as the required backing square.

Pulling the statue only horizontally never clears the route to the exit.

Every floor cell is occupied during the minimum trace.

The player receives immediate feedback when SPACE removes and later restores the chain.

13. Authored Level 3 — Occupied Means Open
Player-facing discovery

This is the first plate and gate.

The agent begins one step left of the plate. The shortest solution deliberately begins:

RIGHT, RIGHT

The first RIGHT puts the agent on the plate and visibly retracts the connected gate. The second leaves the plate and visibly closes it again.

Those two observations distinguish the intended rule from all common alternatives:

It is not a permanent switch.

It is not a one-time trigger.

It is not based on proximity.

The traced gate follows current occupancy.

The statue is then pulled onto the plate. The gate stays open after the agent walks away, demonstrating that either kind of occupant works.

Exact initial state
Floor:
H[6, 4:11]
∪ H[5, 4:6]

Agent:
position = (4,6)
facing   = RIGHT
grip     = null

Statues:
S0 = (7,6)

Exit:
(11,6)

Pair A:
plate = (5,6)
gate  = {(9,6)}
trace = (5,6) → (5,7) → (9,7) → (9,6)

Exit:
(11,6)

Initial gate states:
A = closed

Cursor:
none

Floor-cell count: 11.

Verified minimum solution
RRSLLSURRDRRRRR

Phase breakdown:

Actions	Settled result
R	Agent occupies (5,6); A opens
R	Agent leaves for (6,6); A closes
SLL	Grip S0 and pull it left twice onto plate (5,6); A opens
S	Release
URRD	Use the upper detour to reach (6,6) on the far side of the statue
RRRRR	Cross gate (9,6) and enter exit (11,6)

Minimum depth: 15 actions.

Attribution and necessity check

Stepping onto and off the plate is both the distinguishing experiment and part of the minimum route.

A gate controlled as a toggle would produce a visibly different second observation.

The agent alone cannot remain on the plate and also reach the gate.

S0 is therefore necessary as a persistent holder.

Every traversable cell is touched during the minimum trace.

14. Authored Level 4 — Before You Leave It There
Player-facing discovery

Two statue cap variants and two traced pairs now appear.

S1 begins in a one-sided niche at (5,7). Its only possible movement is east onto plate A. This makes it visually tempting to commit S1 immediately.

Doing so creates a physical, local failure: S0 is directly below plate A at (6,8), and the square needed to stand above S0 is now occupied by the holder. S0 can no longer be pulled north into the gate lane.

The useful experiment is therefore to grip S0 while standing on plate A and pull it north through the plate before assigning S1 permanently.

The two cap variants are both pulled normally in the solution, visibly establishing that they are not different statue classes.

Exact initial state
Floor:
R[6:7, 5:8]
∪ {(5,7), (8,6), (9,6)}
∪ R[10:11, 6:9]
∪ {(12,6), (13,6), (14,6)}

Agent:
position = (7,8)
facing   = LEFT
grip     = null

Statues:
S0 = (6,8)
S1 = (5,7)

Pair A:
plate = (6,7)
gate  = {(9,6)}
trace = (6,7) → (6,4) → (9,4) → (9,6)

Pair B:
plate = (10,8)
gate  = {(13,6)}
trace = (10,8) → (10,10) → (13,10) → (13,6)

Exit:
(14,6)

Initial gate states:
A = closed
B = closed

Cursor:
none

Floor-cell count: 22.

Verified minimum solution
ULDSUUSRDDLSRSULSRRRRSDLUSDDSRUUURRR
Actions	Settled state after the phase	Purpose
ULDSUUS	Agent (6,5); S0 (6,6); S1 (5,7); both gates closed	Pull S0 through plate A’s column before committing a holder
RDDLSRS	Agent (7,7); S1 on A at (6,7); A open	Pull the niche statue onto A
ULSRRRRS	Agent (11,6); S0 (10,6); A open	Pull S0 through gate A into the right room
DLUSDDS	Agent (10,9); S0 on B at (10,8); A and B open	Circle and turn S0 downward onto B
RUUURRR	Agent reaches exit (14,6)	Return to the upper lane and cross B

Minimum depth: 36 actions.

Anti-degenerate and anti-cleanliness check

S1 first is a plausible plan, but it physically removes S0’s required destination-side access.

S0 first is not a checklist choice; it must be oriented through the plate cell while that cell remains available.

S1 is necessary as A’s eventual holder.

S0 is necessary as B’s eventual holder.

Both gates block the only route to the exit.

Every floor cell is occupied during the minimum trace.

The right region is a two-cell-wide circulation loop, not an empty corridor: it is needed to turn S0 downward and later regain the upper exit lane.

15. Authored Level 5 — Borrow the Holder
Player-facing discovery

S0 can immediately be pulled onto plate A, transferring occupancy from the agent to the statue without closing the gate.

The agent crosses A and pulls S1 left. This creates the central impasse:

S1 is now at (8,6).

To pull S1 right, the agent must reach its east side.

The only circulation route to that side passes through (8,7).

S0 is occupying (8,7) as A’s holder.

The solution is not to find another device. It is to take the known holder back.

Pulling S0 off A immediately closes the gate but opens the lower circulation route. The agent circles, briefly occupies the empty plate personally, and then pulls S0 back onto it from the opposite side. This leaves the agent on the correct side of both the holder and traveler.

The same plate occupant therefore has three contextual roles:

persistent gate support;

removable physical obstruction;

re-established support after the agent has changed sides.

Exact initial state
Floor:
H[6, 7:14]
∪ R[6:8, 7:8]
∪ {(9,7)}
∪ R[11:12, 7:9]

Agent:
position = (9,7)
facing   = LEFT
grip     = null

Statues:
S0 = (7,7)
S1 = (10,6)

Pair A:
plate = (8,7)
gate  = {(9,6)}
trace = (8,7) → (8,4) → (9,4) → (9,6)

Pair B:
plate = (11,8)
gate  = {(13,6)}
trace = (11,8) → (11,10) → (13,10) → (13,6)

Exit:
(14,6)

Initial gate states:
A = closed
B = closed

Cursor:
none

Floor-cell count: 21.

Verified minimum solution
LSRSURSLLSDRSLSDRRULSRSULSRRRSDLUSDDSRUUURR
Actions	Settled state after the phase	Purpose
LSRS	S0 on A (8,7); agent (9,7); A open	Initial agent-to-statue plate handoff
URSLLS	S1 (8,6); agent (7,6); A open	Pull S1 left through gate A
DRSLS	S0 removed to (7,7); agent (6,7); A closed	Unassign the holder to open the lower route
DRRULSRS	S0 restored to A; agent (9,7); A open	Circle, occupy A personally, then transfer it back to S0
ULSRRRS	S1 (11,6); agent (12,6)	Pull S1 right after reaching its correct side
DLUSDDS	S1 on B (11,8); agent (11,9); A and B open	Turn S1 downward onto B
RUUURR	Agent reaches (14,6)	Cross gate B

Minimum depth: 43 actions.

Anti-degenerate and anti-cleanliness check

Leaving S0 on A prevents access to the side of S1 required for the next pull.

Permanently removing S0 closes A and prevents later traversal.

The plate must be empty, agent-held, and statue-held at different moments.

The lower loop is not an optional recovery path; it is the only way to change sides while S1 blocks the upper lane.

Both statues and both pairs are necessary.

Every floor cell is touched during the minimum solution.

16. Authored Level 6 — The Handoff
Player-facing discovery

Three statues and three traced pairs are present, but there is no new object rule.

The opening shape reuses Level 4’s lesson twice:

S0 must be staged north to (5,8).

S2 must be staged north to (7,8).

Only then may the niche statue S1 be committed to plate A.

After A opens, S2 is moved through first and parked at (9,6), away from the shared horizontal lane. S0 then crosses and becomes B’s holder.

A new strategic use of the already-known grip rule appears at the central bottleneck. After S0 is placed on B:

the agent is still gripping S0;

S2 is immediately to the agent’s left;

walking while gripping cannot make progress.

The agent rotates left without moving, releases S0, presses SPACE again, and visibly transfers the chain to S2. This is not a special handoff command; it is a tight composition of the Level 2 release-and-regrip rule.

S2 can then be pulled through gate B and finally assigned to plate C.

Exact initial state
Floor:
R[5:7, 7:9]
∪ H[10, 5:6]
∪ {(4,9), (8,9)}
∪ H[8, 5:11]
∪ R[9:11, 5:7]
∪ {(10,4)}
∪ H[3, 7:10]
∪ H[2, 6:7]
∪ H[2, 9:10]

Agent:
position = (6,10)
facing   = UP
grip     = null

Statues:
S0 = (5,10)
S1 = (4,9)
S2 = (7,9)

Pair A:
plate = (5,9)
gate  = {(8,8)}
trace = (5,9) → (3,9) → (3,11) → (8,11) → (8,8)

Pair B:
plate = (10,7)
gate  = {(10,4)}
trace = (10,7) → (13,7) → (13,4) → (10,4)

Pair C:
plate = (8,3)
gate  = {(7,2)}
trace = (8,3) → (8,2) → (7,2)

Exit:
(6,2)

Initial gate states:
A = closed
B = closed
C = closed

Cursor:
none

Floor-cell count: 35.

Verified minimum solution
ULDSUUSRRDSUSLDDLSRSRRULSRRSULDSUUSRDDDLLLLSRRRRRSULDSULSSRSULDSUUUSLDRSLLSUL
Actions	Settled state after the phase	Purpose
ULDSUUS	Agent (5,7); S0 (5,8)	Pre-stage S0 through plate A’s column
RRDSUS	Agent (7,7); S2 (7,8)	Pre-stage S2
LDDLSRS	Agent (6,9); S1 on A (5,9)	Commit the niche statue only after both travelers are staged
RRULSRRS	Agent (10,8); S2 (9,8)	Pull S2 through A first
ULDSUUS	Agent (9,5); S2 (9,6)	Turn S2 north and park it clear of the shared row
RDDDLLLLSRRRRRS	Agent (11,8); S0 (10,8)	Revisit the left side and pull S0 through A
ULDSULSSRS	Agent (11,6); S0 on B (10,7); S2 (10,6)	Put S0 on B, release it, immediately regrip S2, and align S2 with B
ULDSUUU	Agent (10,2); S2 (10,3); still gripping S2	Pull S2 north through gate B
SLDRSLLS	Agent (7,3); S2 on C (8,3)	Circle and pull S2 west onto C
UL	Agent passes gate C and reaches exit (6,2)	Complete

Minimum depth: 77 actions.

Final reference allocation:

S1 holds A at (5,9)
S0 holds B at (10,7)
S2 holds C at (8,3)
Anti-degenerate and anti-cleanliness check

Committing S1 before staging S0 and S2 removes required backing access.

Leaving S2 in the horizontal gate lane blocks S0’s later traversal.

S0 cannot pass gate A before A has a persistent holder.

S2 cannot reach C until S0 holds B.

The wall at (8,2) prevents bypassing C from the eastern upper area.

The central adjacent-statue state requires an intentional release/regrip decision, not a long walk to a station.

The layout contains revisits and cross-dependencies: A supports two different travelers; B depends on one of them and enables the other.

Every traversable cell is occupied during the minimum trace.

17. Authored Level 7 — Everybody Has a Job
Player-facing discovery

Level 7 adds a fourth statue and a fourth plate–gate pair, but no new operational rule.

S3 begins at (4,8), visibly adjacent to the shared A-gate row. It cannot be moved usefully at first because S0 occupies (5,8), the square needed to stand beside and pull S3 east. S3 is therefore a necessary future resource whose route depends on an earlier statue vacating the lane.

The most important reinterpretation concerns plate D at (11,7).

S2 reaches D first. Pulling S2 north causes it to occupy D momentarily, visibly opening the final red gate. But S2 must immediately leave D and continue toward C; the briefly opened final gate is still unreachable.

Much later, S3 follows the now-cleared A route and becomes D’s permanent holder.

Thus the same plate has two sharply contrasting roles with an unchanged rule:

a transit square that opens its gate uselessly for a moment;

a final assignment that must remain occupied while another statue opens C.

The final corridor contains gate C, an ordinary separating cell, gate D, and the exit. The validator still checks only the exit, but the spatial route forces all four allocations to coexist.

Exact initial state
Floor:
R[5:7, 7:9]
∪ {(5,10), (6,10), (4,9), (4,8), (8,9)}
∪ H[8, 4:12]
∪ H[7, 9:12]
∪ H[6, 9:11]
∪ H[5, 10:11]
∪ {(10,4)}
∪ H[3, 7:10]
∪ {(4,2), (5,2), (6,2), (7,2), (9,2), (10,2)}

Agent:
position = (6,10)
facing   = UP
grip     = null

Statues:
S0 = (5,10)
S1 = (4,9)
S2 = (7,9)
S3 = (4,8)

Pair A:
plate = (5,9)
gate  = {(8,8)}
trace = (5,9) → (3,9) → (3,11) → (8,11) → (8,8)

Pair B:
plate = (10,7)
gate  = {(10,4)}
trace = (10,7) → (10,6) → (13,6) → (13,4) → (10,4)

Pair C:
plate = (8,3)
gate  = {(7,2)}
trace = (8,3) → (8,2) → (7,2)

Pair D:
plate = (11,7)
gate  = {(5,2)}
trace = (11,7) → (11,11) → (14,11) → (14,12)
        → (2,12) → (2,2) → (5,2)

Exit:
(4,2)

Initial gate states:
A = closed
B = closed
C = closed
D = closed

Cursor:
none

Floor-cell count: 39.

Verified minimum solution
ULDSUUSRRDSUSLDDLSRSRRULSRRRRSULDSUUSLDDDLLLLSRRRRRSULDSURSSLSDDLLLLSRRRRRRRSULDSUSULDSUUUSLDRSLLSULLL

Phase-level replay:

Actions	Settled state after the phase	Purpose
ULDSUUSRRDSUSLDDLSRS	Agent (6,9); S0 (5,8); S1 on A (5,9); S2 (7,8); S3 (4,8)	Stage S0 and S2, then commit S1 to A
RRULSRRRRSULDSUUS	Agent (11,5); S2 (11,6); D closed again	Pull S2 through A, turn it across D, briefly open D, then continue off the plate
LDDDLLLLSRRRRRSULDSU	Agent (10,6); S0 on B (10,7); S2 (11,6); grip still on S0	Revisit A, move S0 through, and establish B
RSSLS	Agent (9,6); S2 (10,6); no grip	Rotate while gripping S0, release, grip adjacent S2, and pull it into B’s column
DDLLLLSRRRRRRRSULDSUS	Agent (11,6); S3 on D (11,7); A, B, and D open	Revisit the far-left waiting statue and haul it through A to become D’s permanent holder
ULDSUUU	Agent (10,2); S2 (10,3)	Pull S2 through gate B
SLDRSLL	Agent (7,3); S2 on C (8,3); all four gates open	Turn S2 west onto C
SULLL	Release, pass C at (7,2), cross D at (5,2), enter exit (4,2)	Complete

Minimum depth: 102 actions.

Final reference allocation:

S1 holds A at (5,9)
S0 holds B at (10,7)
S2 holds C at (8,3)
S3 holds D at (11,7)

These identities are not validator requirements; the statues remain operationally interchangeable. Geometry and route availability make this the minimum reference allocation.

Anti-degenerate and anti-cleanliness check

S3 is visible from the beginning but cannot move until S0 vacates its approach square.

S2 cannot remain on D because it is also needed beyond gate B and on C.

D therefore cannot be solved by assigning the first statue that touches it.

S0 must hold B while S2 moves through the upper vertical route.

S1 must continue holding A because the solution revisits and traverses that bottleneck multiple times.

The final passage requires C and D simultaneously, but this is a route consequence rather than an explicit multi-clause validator.

D is used first as a temporary transit plate and later as a permanent support.

The long rightward statue hauls are not neutral corridor walks: each moves a limited shared resource through gate A and changes which backing and turning squares remain available.

Every one of the 39 traversable cells is occupied by an agent or statue during the minimum trace.

No object, pair, chamber, or waiting statue can be ignored.

18. Epistemic-solvability audit
Foundational attribution

Level 1: A blocked push and a visible chain isolate the pull rule. There are no simultaneous gate or state changes.

Level 2: The persistent chain creates the impasse, and SPACE visibly removes it. The only new useful region is the nearby circulation pocket needed for a different grip side.

Level 3: Stepping onto and then off the plate produces opposite gate states in two consecutive actions. This rules out toggle, permanence, and one-shot explanations before the player is required to use a statue as holder.

Reuse in contrasting contexts

A plate is an agent-operated demonstration in Level 3.

A plate is a temporary transit square in Levels 4 and 7.

A statue is permanent infrastructure in Level 4.

A holder becomes a removable obstruction in Level 5.

A gate-supporting statue is deliberately released beside another statue in Level 6.

D’s first occupant in Level 7 is unsuitable as its final occupant.

The same A gate supports one traveler in Level 4, a reassignment in Level 5, two pre-staged travelers in Level 6, and three separate traversals in Level 7.

No hidden rule burden

At every settled state, the player can see:

agent position and facing;

every statue position;

which statue is gripped;

all floor and wall geometry;

plate depression;

gate state;

the exact plate–gate pairing;

the exit;

every square required for a future pull.

No solution depends on an invisible timer, previous visit count, secret statue type, click target, or latent gate state.

19. Reference implementation audit

The authored instances were replayed under the exact transition system above.

Breadth-first search over settled states gives the following minimum step counts:

Level 1:   7
Level 2:  15
Level 3:  15
Level 4:  36
Level 5:  43
Level 6:  77
Level 7: 102

The search state includes agent position, facing, all statue positions, and grip identity. Clicks are no-ops and cannot shorten a solution.

For every level:

the supplied trace reaches the stated exit;

the supplied trace has the verified minimum length;

every static interactive object is used;

every traversable floor cell is occupied at least once by the agent or a statue in the supplied minimum trace;

no level is solvable by ignoring its newly introduced mechanic or planning consequence;

all difficulty follows from the same pull, occupancy, collision, and access rules.
