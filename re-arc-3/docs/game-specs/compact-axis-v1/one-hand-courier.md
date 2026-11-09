Worked for 27m 1s
One-Hand Courier

Identity: “This is a game about carrying keys becoming difficult because the parcel that must reach the exit occupies the same single hand.”

Exact authored level count: 7.

The player is a courier with one hand slot. Keys are useful either while carried, as temporary permission to cross matching gates, or while left in sockets, as infrastructure that holds a particular gate open. The parcel uses the same hand slot but matches nothing. Every late puzzle therefore asks the player to turn a temporarily navigable map into a route that remains navigable while carrying the parcel.

There are no counters, hidden modes, timers, enemies, consumable objects, or global configuration predicates. Difficulty comes entirely from where objects are staged before a key is reassigned and a previously open route closes.

1. Objective and action space

The game uses:

UP, DOWN, LEFT, RIGHT

SPACE

Clicking is not used. A click on any cell leaves the state unchanged, produces no animation, and creates no cursor or selection state. No object is rendered as clickable.

The exact success condition is:

After an action resolves, the courier occupies the exit cell and the hand contains the parcel.

Keys, sockets, loose objects, and gate states are ignored by the success validator. Entering the exit empty-handed or carrying a key is allowed but does not complete the level. The exit briefly pulses with its parcel silhouette to indicate what is missing.

There is no action budget, energy system, timer, or loss condition. Exploration is unlimited. The host environment may provide an external restart, but restart is not an in-world action.

2. Native geometry and rendering
Logical board

Every level uses an 11 × 9 orthogonal lattice.

Coordinates are (x,y).

x = 1…11, increasing rightward.

y = 1…9, increasing downward.

A movement action changes the courier position by exactly one logical cell.

Gates occupy edges between adjacent floor cells rather than cells themselves.

Walls are impassable.

Loose objects do not block movement. The courier may occupy the same cell as one loose object.

At most one loose object may occupy a floor cell.

Mapping to the 64×64 canvas

Canvas background: color 5.

Logical tile size: 5×5 rendered pixels.

Board origin: rendered pixel (4,14).

Logical cell (x,y) occupies:

pixel_x = 4 + 5(x−1) … 8 + 5(x−1)

pixel_y = 14 + 5(y−1) … 18 + 5(y−1)

This gives a 55×45 board footprint. The remaining upper area contains the hand slot.

Contiguous wall cells are rendered as merged solid regions, with no individual cell borders. Ordinary floor cells also have no grid outlines.

Palette
Role	Color IDs
Canvas background	5 black
Walls	4 very dark gray
Floor	1 light gray
Neutral outlines and gate posts	3 dark gray
Empty socket interior	2 gray
Courier	0 white, with 3 center detail
A-key family	9 blue and 10 light blue
B-key family	11 yellow and 12 orange
Parcel and exit	6 magenta and 7 pink
Invalid-action feedback	8 red
Completion feedback	14 green

Purple and maroon are unused.

Object silhouettes

These are exact conceptual 5×5 masks. A period is transparent.

A = color 9, a = color 10:

.AAA.
AA.AA
.AaA.
..A..
..A..

B = color 11, b = color 12:

..B..
.BBB.
BBbBB
..B..
..B..

P = color 6, p = color 7:

.PPP.
PPpPP
PpppP
PPpPP
.PPP.

The A and B silhouettes are deliberately different in shape as well as color. The same silhouette appears on its loose key, sockets, and matching gates.

Courier

The courier is a connected white 3×4 figure centered in the logical cell, with one dark-gray center pixel. When sharing a cell with a loose object, the object is rendered first and remains visible around the courier.

When carrying something:

A 3×2 miniature of the carried object appears immediately above the courier’s body within the same logical cell.

The full object also appears in the hand slot.

Hand slot

A 9×9 framed module is rendered at pixels x=3…11, y=3…11.

Empty: dark interior with a white open-bracket silhouette.

Occupied: the carried object’s full 5×5 mask.

There is exactly one compartment; no numeral or text is shown.

Sockets

A socket occupies a whole logical floor cell.

Empty: a hollow outline of its matching key mask on a dark-gray inset.

Occupied: the inserted key’s solid mask fills the socket.

A key in a socket is not simultaneously shown in the hand or on the floor.

A one-pixel cable connects every socket to its linked gate.

A-key cables use color 10; B-key cables use color 12. Cables are visual only and do not affect movement.

For deterministic rendering, draw each cable from the socket-cell center to the linked gate-edge midpoint, moving horizontally first and then vertically. Cables are drawn over floor and wall regions but beneath objects, the courier, sockets, and gates. The authored layouts do not create ambiguous cable crossings.

Gates

A gate is drawn across its specified edge.

A closed gate has:

two neutral posts;

a two-pixel-thick colored bar spanning the edge;

a small version of the matching key emblem.

An open gate retains its posts but retracts the center bar, leaving an unmistakable floor-colored gap.

Exit

The exit is a full-cell magenta frame containing a hollow parcel silhouette. It is traversable in every state.

3. Complete state-machine contract
Persistent state
courier_position: floor coordinate
hand: NONE or object_id
object_location[object_id]:
    HAND
    FLOOR(x,y)
    SOCKET(socket_id)
socket_occupant[socket_id]: NONE or matching key_id
won: boolean

There is no selected object, cursor, orientation state, action history, latent gate mode, or hidden inventory.

A gate’s open state is derived rather than independently stored:

gate_open(gate) =
    gate has a linked socket
    AND that socket currently contains its matching key

The portable-only gate in Level 2 has no linked socket and is therefore closed in every settled frame.

Movement resolution

For UP, DOWN, LEFT, or RIGHT:

Compute the orthogonally adjacent destination cell.

If it is not floor, do not move.

If no gate occupies the crossed edge, move normally.

If an open gate occupies the edge, move normally regardless of hand contents.

If a closed gate occupies the edge:

If the hand contains a key of the gate’s type, complete the move. The key remains in hand.

Otherwise, do not move.

After movement, run the success validator.

Gates are bidirectional. A matching carried key works from either side.

A portable crossing does not leave the gate open. In the settled frame after the movement, the gate is closed unless its socket is occupied.

SPACE resolution

Socket interaction has priority over ordinary pickup and dropping.

Courier standing on a socket
Hand	Socket	Result
Empty	Occupied	Retrieve the key into the hand; linked gate closes
Matching key	Empty	Insert the key; hand becomes empty; linked gate opens
Empty	Empty	No state change; hollow socket pulses
Wrong key or parcel	Empty	No state change; mismatch feedback
Any object	Occupied	No state change; swapping is forbidden

Wrong objects remain in the hand.

Courier standing on a non-socket floor cell

If the cell is the exit, SPACE never drops an object there.

Otherwise:

Hand	Loose object under courier	Result
Empty	Present	Pick up the object
Occupied	Absent	Drop the held object
Empty	Absent	No state change
Occupied	Present	No state change; objects cannot be swapped or stacked

The parcel uses these exact pickup and drop rules.

Parcel restrictions

The parcel:

can be carried and dropped like a key;

matches no gate;

matches no socket;

cannot be inserted;

cannot be dropped on the exit or a socket;

wins only while still in the hand.

Update and animation order

Logical state changes occur only when step(action) is called.

After an action is validated, the final logical state is committed immediately. The following presentation frames visualize that committed transition. Inputs are ignored until the animation settles, and animation frames do not invoke additional steps.

Ordinary movement

Ordinary one-cell movement renders immediately, without ornamental animation.

Portable gate crossing

A successful carried-key crossing uses three presentation frames:

Gate halves retract while the courier is drawn on the source side.

Courier is drawn centered over the gate edge.

Courier appears in the destination cell and the gate closes again.

The settled state is the destination position with the key still in hand.

Socket insertion or retrieval

Insertion or retrieval uses three frames:

Key moves between the hand indicator and socket.

Cable briefly brightens.

Linked gate finishes retracting or closing.

The socket occupancy, hand state, and gate state are already logically committed before frame 1.

Invalid interaction

Blocked or mismatched gate: gate rattles one pixel in each direction and flashes color 8.

Wrong socket object: socket and hand slot flash color 8.

Attempted stacking or invalid drop: held object rises one pixel and settles back.

Full-hand pickup attempt: both the held object and floor object flash color 8.

No gameplay state changes.

Completion

On success:

The parcel silhouette fills the exit.

Exit changes to color 14.

Courier and parcel settle into one centered completion silhouette.

won=true remains visible and subsequent inputs do nothing.

4. Authored-level notation

The maps below are implementation blueprints, not in-game text.

#  wall
.  ordinary floor
@  courier start
a  loose A key
b  loose B key
p  loose parcel
x  exit
1–5 socket IDs defined below each map

Every non-# character is a floor cell. Gates are edges and are listed separately.

Unless explicitly stated:

the hand starts empty;

all sockets start empty;

all gates start closed;

won=false;

there is no cursor or selected-object state.

In solution traces:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

The reference traces were replayed in the specified state machine. Their stated lengths are minimal under these rules.

5. Level 1 — Empty the Hand

Teaching goal: The hand holds exactly one object; SPACE transfers an object between hand and floor; the parcel must remain carried at the exit.

Player-facing discovery

The courier begins visibly carrying an A key. A parcel lies one step to the right, and the parcel-shaped exit is farther along the same open area.

The simplest experiment, SPACE, immediately drops the visible held key and empties the hand slot. If the player first walks onto the parcel and presses SPACE, the pickup fails visibly because the hand is full. Moving to an empty cell and pressing SPACE then drops the key.

This creates direct evidence for one-hand capacity without introducing gates, sockets, decoys, or unrelated terrain.

Blueprint
  12345678901
1 ###########
2 ###########
3 ###########
4 ###########
5 ##@p..x####
6 ###########
7 ###########
8 ###########
9 ###########

Initial state:

Component	State
Courier	(3,5)
Hand	A key
A key	HAND
Parcel	FLOOR(4,5)
Exit	(7,5)
Gates/sockets	None
Verified solution depth: 6 actions
S | R S | R R R

Resolution:

Drop A at (3,5).

Enter (4,5) and pick up the parcel.

Carry the parcel through (5,5) and (6,5) into the exit at (7,5).

Final relevant state:

courier = (7,5)
hand = parcel
A key = FLOOR(3,5)
won = true
Attribution and necessity check

The initial key is not a decoy: it must leave the hand before the parcel can be collected. The exit silhouette supplies positive goal evidence. No other object or simultaneous rule can explain the visible full-hand failure or the result of dropping.

6. Level 2 — Permission in Hand

Teaching goal: A carried matching key permits passage through a closed gate, but the gate closes behind the courier.

Player-facing discovery

The only route to the parcel crosses a blue A gate. Approaching empty-handed causes a visible rattle. The A key lies directly beside the start and has the same color and forked silhouette as the gate.

Picking it up and repeating the move causes the gate to retract and the courier to cross. The gate then visibly closes behind the courier while the key remains in the hand.

Dropping is already established by Level 1 and is required before collecting the parcel.

Blueprint
  12345678901
1 ###########
2 ###########
3 ###########
4 #######p.##
5 #@a......##
6 ########x##
7 ###########
8 ###########
9 ###########

Initial state:

Component	State
Courier	(2,5)
A key	FLOOR(3,5)
Parcel	FLOOR(8,4)
Exit	(9,6)
GA	A gate on edge (5,5) ↔ (6,5), no socket
Verified solution depth: 13 actions
R S | R R R R S | R U S | R D D

Resolution:

Pick up A at (3,5).

Cross GA, continue to (7,5), and drop A.

Move to (8,4) and collect the parcel.

Carry it through (9,4) and (9,5) into (9,6).

Attribution and necessity check

The gate edge is the sole connection between the two regions. There is no socket and no alternate route, so successful passage cannot be attributed to proximity, direction, stepping twice, or a permanent gate toggle. The player directly observes that the matching object must be in the hand.

7. Level 3 — Leave Permission Behind

Teaching goal: Inserting a key into its socket holds the linked gate open for any carried object.

Player-facing discovery

The player can initially solve the first half using the Level 2 hypothesis: carry A through the gate. But on reaching the parcel, the hand is occupied. Dropping A allows the parcel to be collected, but the closed gate then blocks the return journey.

The only unfamiliar object is the hollow A-shaped socket immediately beside the gate, with a visible cable joining them. Inserting A causes the key to leave the hand and the gate to retract immediately. The open gate then admits the parcel.

This forces observation of persistent latching; the puzzle cannot be solved by the older “carry the matching key” rule alone.

Blueprint
  12345678901
1 ###########
2 ###########
3 ###########
4 ####1######
5 ##@a..p####
6 ##x########
7 ###########
8 ###########
9 ###########

Socket and gate:

Map ID	Socket	Type	Linked gate
1	SA at (5,4)	A	GA
—	GA	A	Edge (5,5) ↔ (6,5)

Initial objects:

courier = (3,5)
A = FLOOR(4,5)
parcel = FLOOR(7,5)
exit = (3,6)
Verified solution depth: 14 actions
R S | R U S | D R R S | L L L L D

Resolution:

Pick up A.

Stand on SA and insert A; GA opens.

Cross empty-handed and collect the parcel.

Return through the still-open gate and enter the exit.

Attribution and necessity check

Carrying A through the gate cannot solve the level because the parcel must return through the same edge. The occupied socket, lit cable, and persistently retracted gate provide three mutually reinforcing visual consequences of the insertion.

8. Level 4 — Move the Latch

Teaching goal: An inserted key can be retrieved, retrieval closes its gate, and the same key can be reassigned to another socket. Cargo must be staged on the useful side before retrieval.

Player-facing discovery

The left A gate must first be latched open so that the B key can be carried out of the starting region. But the exit lies behind a different A gate, and there is only one A key.

Once B has been staged in the central region, the courier encounters a local impasse: leaving A in socket 1 preserves a route that is no longer useful, while socket 2 visibly controls the exit gate. Returning empty-handed to occupied socket 1 and pressing SPACE pulls A back into the hand and immediately closes the old gate.

The player learns that a socket is not a permanent commitment, but moving its key changes which side of the map remains connected.

Blueprint
  12345678901
1 ###########
2 ###########
3 ###########
4 #a#1..2####
5 #@.....x###
6 #b#..3.####
7 ####p######
8 ###########
9 ###########

Sockets and gates:

Map ID	Socket	Type	Linked gate
1	SA1 at (4,4)	A	GA1
2	SA2 at (7,4)	A	GA2
3	SB at (6,6)	B	GB
—	GA1	A	Edge (3,5) ↔ (4,5)
—	GA2	A	Edge (7,5) ↔ (8,5)
—	GB	B	Edge (5,6) ↔ (5,7)

Initial objects:

courier = (2,5)
A = FLOOR(2,4)
B = FLOOR(2,6)
parcel = FLOOR(5,7)
exit = (8,5)
Verified solution depth: 39 actions
U S
| D R R U S
| D L L D S
| U R R U R R S
| L L S
| R R R S
| L S
| D D S
| L D S
| U U R R R

State-changing phases:

Pick up A.

Cross GA1 carrying A and insert it in SA1.

Return through the open gate and pick up B.

Carry B into the central region and drop it at (6,4).

Retrieve A from SA1; GA1 closes with B and the courier safely on the central side.

Insert A in SA2; GA2 opens.

Pick up staged B.

Insert B in SB; GB opens.

Collect the parcel.

Carry it through the open B route and open exit gate.

Final gate state:

GA1 closed
GA2 open via A in SA2
GB open via B in SB
Anti-degenerate check

Without SA1, B cannot cross GA1 while A occupies the hand.

Without staging B before retrieving A, B remains sealed in the left region.

A cannot remain in SA1, because GA2 must be open while the parcel occupies the hand.

B and SB are necessary because the parcel is behind GB.

Both gate edges are topological chokepoints; there are no wall bypasses.

9. Level 5 — Empty the Vault

Teaching goal: One latch can support several cargo trips, and a key may cross its gate portably before becoming a stationary latch on the far side.

Player-facing discovery

A and the courier begin in the central hub. Both B and the parcel are inside the western vault behind GA1. Socket 1 is on the vault side.

The courier must:

carry A into the vault;

leave A in socket 1;

ferry B and the parcel out separately;

return for A only after both are staged.

The B socket is beyond its own gate. This creates a contrasting use of a familiar rule: B must first be carried through the closed B gate, then inserted from the far side so the courier can return empty-handed for the parcel.

Blueprint
  12345678901
1 ###########
2 ###########
3 ####a######
4 #b1#.2##3.#
5 #.........#
6 #p.#@.##.x#
7 ####.######
8 ###########
9 ###########

Sockets and gates:

Map ID	Socket	Type	Linked gate
1	SA1 at (3,4)	A	GA1
2	SA2 at (6,4)	A	GA2
3	SB at (9,4)	B	GB
—	GA1	A	Edge (3,5) ↔ (4,5)
—	GA2	A	Edge (6,5) ↔ (7,5)
—	GB	B	Edge (8,5) ↔ (9,5)

Initial objects:

courier = (5,6)
A = FLOOR(5,3)
B = FLOOR(2,4)
parcel = FLOOR(2,6)
exit = (10,6)
Verified solution depth: 58 actions
U U U S
| D D L L U S
| L S
| D R R R R S
| L L L D L S
| U R R R S
| L L U S
| D R R U R S
| D S
| R R R U S
| D L L L L S
| R R R R D R

State-changing phases:

Collect A.

Enter the vault through GA1 and insert A in SA1.

Collect B.

Carry B out and stage it at (6,5).

Return to the vault and collect the parcel.

Carry the parcel out and stage it at (5,5).

Retrieve A; the vault gate closes.

Carry A back through the closed matching gate and insert it in SA2, opening the east route.

Pick up staged B.

Carry B through open GA2, cross closed GB portably, and insert it in far-side SB.

Return empty-handed through both open gates and pick up the parcel.

Carry the parcel through the prepared route to the exit.

Anti-cleanliness check

This is not a list of independent switches:

A cannot be moved to the eastern socket until two different cargo objects have been extracted through its first gate.

B’s gate permission and latching roles occur on opposite sides of the same gate.

The parcel’s staging location matters because A and B must both be manipulated after the parcel leaves the vault.

Neither the vault phase nor the final-route phase can be removed without breaking the dependency structure.

10. Level 6 — Close Behind the Cargo

Teaching goal: Closing an old route can be part of the solution, but only after every required object has been transferred to the correct side.

Player-facing discovery

The A key initially opens the route from the western supply area into the central chamber. It must later move to socket 2, opening a second bridge into the right-hand staging region.

The exit gate’s B socket is not directly accessible. It lies in a small lower alcove behind a third A gate. Reaching that socket requires moving A from socket 2 to socket 3, which closes the bridge back to the central chamber.

The player must therefore stage both B and the parcel in the right-hand region before moving A. The old bridge closing is no longer merely a danger; it becomes a safe commitment because nothing useful remains behind it.

Blueprint
  12345678901
1 ###########
2 ###########
3 #a#1.#2####
4 #@........#
5 #b##p#3.#x#
6 #######.4##
7 ###########
8 ###########
9 ###########

Sockets and gates:

Map ID	Socket	Type	Linked gate
1	SA1 at (4,3)	A	GA1
2	SA2 at (7,3)	A	GA2
3	SA3 at (7,5)	A	GA3
4	SB at (9,6)	B	GB
—	GA1	A	Edge (3,4) ↔ (4,4)
—	GA2	A	Edge (6,4) ↔ (7,4)
—	GA3	A	Edge (8,5) ↔ (8,6)
—	GB	B	Edge (8,4) ↔ (9,4)

The cable from socket 4 runs upward through wall cell (9,5) to GB, making the side-alcove relationship visible from the initial frame.

Initial objects:

courier = (2,4)
A = FLOOR(2,3)
B = FLOOR(2,5)
parcel = FLOOR(5,5)
exit = (10,5)
Verified solution depth: 62 actions
U S
| D R R U S
| D L L D S
| U R R R R S
| L U L S
| D R R R U S
| D L L D S
| U R R R S
| L L S
| R S
| U S
| D D S
| U S
| D R D R S
| L U U S
| R R D

State-changing phases:

Pick up A and latch GA1 with SA1.

Fetch B and stage it at (6,4).

Retrieve A from SA1.

Carry A through GA2 and insert it in SA2.

Return through open GA2, collect the parcel, and stage it at (8,4).

Return for B and stage it at (7,4).

Retrieve A from SA2; GA2 closes behind all required cargo.

Insert A in SA3, opening the lower alcove.

Pick up B and carry it into socket 4; GB opens.

Return to (8,4), collect the staged parcel, and cross the newly open B gate into the exit region.

Final state:

A = SOCKET(SA3)
B = SOCKET(SB)
parcel = HAND
GA1 closed
GA2 closed
GA3 open
GB open
Anti-cleanliness check

Moving A to socket 3 before transferring the parcel leaves the parcel behind GA2.

Moving A before transferring B leaves no object capable of activating the exit gate.

B cannot reach its socket unless GA3 is open.

The parcel cannot simply use the lower A route because that route terminates in a socket alcove, not the exit.

The mature puzzle has a network of supply, staging, service, and exit routes rather than one corridor of devices.

All premature commitments remain experimentally recoverable: A can be moved back from SA3 to SA2 before the final departure.

11. Level 7 — Temporary Roads

Teaching goal: The same key can successively serve as portable permission, extraction latch, service-route latch, and final-route latch. Both keys must end in different roles from those they served initially.

Player-facing discovery

The parcel begins inside a lower vault behind a B gate. A first opens the supply gate so B can enter the central region.

At the vault, the player reaches the now-familiar full-hand impasse: carrying B permits entry, but prevents collection of the parcel. Socket 2 inside the vault lets B hold that gate open. The parcel can then be removed and staged near the final route.

However, B is still not ready for its final socket. It must first occupy service socket 3, opening an upper B gate so A can be carried to remote final socket 5. Only after A opens the final A gate can B be retrieved from the service socket and moved through the final B gate.

The same operational rules never change. The strategic meaning of each socket changes because of where it sits in the route network.

Blueprint
  12345678901
1 ###########
2 ###########
3 #a#####..##
4 #@#1..3.5##
5 #......##x#
6 #b##.#.4..#
7 ####.2#####
8 ####.p#####
9 ###########

Sockets and gates:

Map ID	Socket	Type	Linked gate
1	SA_supply at (4,4)	A	GA_supply
2	SB_vault at (6,7)	B	GB_vault
3	SB_service at (7,4)	B	GB_service
4	SB_final at (8,6)	B	GB_final
5	SA_final at (9,4)	A	GA_final
—	GA_supply	A	Edge (3,5) ↔ (4,5)
—	GB_vault	B	Edge (5,6) ↔ (5,7)
—	GB_service	B	Edge (7,4) ↔ (8,4)
—	GB_final	B	Edge (7,6) ↔ (8,6)
—	GA_final	A	Edge (9,6) ↔ (10,6)

The cable from socket 5 runs downward through wall cell (9,5) to GA_final. Thus the final A relationship is visible even though the socket is reached through the upper service route.

Initial objects:

courier = (2,4)
A = FLOOR(2,3)
B = FLOOR(2,6)
parcel = FLOOR(6,8)
exit = (10,5)
Verified solution depth: 68 actions
U S
| D D R R U S
| D L L D S
| U R R R D D R S
| D S
| U L U U R R D S
| U L L D D R S
| L U U U R R S
| L L L S
| R R R R R S
| L L S
| D D R S
| L S
| R R R U

State-changing phases:

Pick up A.

Cross GA_supply and insert A in socket 1.

Return for B.

Carry B through the open supply gate and then through closed GB_vault; insert B in socket 2.

Pick up the parcel.

Carry the parcel out through the latched vault gate and stage it at (7,6), beside the final route.

Return to socket 2, retrieve B, and cross the now-closed vault gate while carrying B.

Insert B in service socket 3, opening GB_service.

Return to socket 1 and retrieve A. The supply gate closes; all required cargo is already beyond it.

Carry A through the open service gate to socket 5. Insert it, opening GA_final.

Return through the service gate and retrieve B from socket 3; the temporary service route closes.

Carry B through closed GB_final and insert it in far-side socket 4, holding the final B gate open.

Step back to (7,6) and collect the staged parcel.

Carry the parcel through GB_final, then through GA_final, and enter the exit at (10,5).

Final infrastructure:

A = SOCKET(SA_final)
B = SOCKET(SB_final)
GA_supply closed
GB_vault closed
GB_service closed
GB_final open
GA_final open
parcel = HAND
Anti-degenerate check

Every authored component is necessary:

GA_supply and socket 1 are required to carry B out of the starting region without occupying the hand with A.

GB_vault and socket 2 are required because B must be left behind before the parcel can be picked up.

The parcel must be staged before B leaves the vault role.

GB_service and socket 3 are required to transport A to the final A socket while B is not in the hand.

A cannot remain in the supply socket because the parcel’s final route includes GA_final.

B cannot be placed in its final socket early, because it must first open the service route for A.

Socket 4 is on the far side of GB_final, forcing B to act first as portable permission and then as a latch.

Both final gates must be open simultaneously because the parcel matches neither one.

Walls prevent bypasses between the service room, final corridor, and vault.

The level is not a sequence of unrelated errands. It is a route-replacement problem: two temporary roads are deliberately closed after cargo transfer, while the same two keys are reassigned to create a different parcel-only road.

12. Curriculum and epistemic-solvability audit
Level	New discovery or reinterpretation	Productive local experiment	Immediate evidence
1	One visible hand slot; SPACE transfers objects	Press SPACE while holding A, or try to collect the parcel with a full hand	A drops; full-hand pickup visibly fails
2	A carried matching key permits crossing	Try the gate empty, then while carrying the visually matching key	Rattle versus animated crossing
3	A socket converts permission into persistent infrastructure	Insert A into the only unfamiliar hollow A shape	Key leaves hand, cable brightens, gate retracts
4	Keys can be retrieved and reassigned; retrieval closes the old gate	Press SPACE on occupied socket 1 with an empty hand	A enters hand and GA1 immediately closes
5	A portable key can cross first and become a far-side latch afterward	Carry B through its closed gate and place it in the visible socket beyond	Courier can return empty through the same gate
6	Closing a route is safe after staging all required cargo	Move A from socket 2 to socket 3 only after transferring B and the parcel	The old bridge closes, but both cargo objects remain visibly usable
7	One key can cycle through several unchanged roles in a dependency network	Move B among vault, service, and final sockets	Each move opens one concrete route while closing another

The first three levels introduce one central causal addition at a time. Later levels add no arbitrary object vocabulary. They deepen the same rules through side choice, far-side sockets, repeated cargo trips, temporary infrastructure, route closure, and key reassignment.

The complete settled frame always reveals:

courier position;

hand occupancy;

every loose object;

every occupied socket;

every open or closed gate;

every socket-to-gate relationship;

the parcel and exit;

whether the level is complete.

No successful plan depends on hidden history or remembering a transient event.
