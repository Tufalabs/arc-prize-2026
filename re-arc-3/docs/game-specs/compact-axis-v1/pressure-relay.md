Pressure Relay

“This is a game about holding gates open with carried objects becoming difficult because the same one-slot cargo must also be transported, spent, and ultimately taken to the exit.”

Pressure Relay contains exactly 7 authored levels.

It is a deterministic top-down transport puzzle. Every important state is visible: loose objects, the carried object, depressed plates, open gates, consumed keys, recessed locks, and the final artifact. The game has no timer, energy limit, random events, hidden switches, enemies, or autonomous motion.

The depth comes from one rule family: a loose object can support a pressure plate, but an object in the player’s single inventory slot cannot. A weight is therefore not merely cargo; it is part of the current route. Keys and the artifact obey the same physical rule, but have additional future obligations.

No new object category is introduced after Level 3. Levels 4–7 deepen the consequences of the existing rules.

1. Player-facing rules
Objective

Reach the green exit pedestal while carrying the unique orange artifact.

The exit contains a large orange diamond-shaped recess matching the artifact. Reaching it empty-handed, carrying a weight, or carrying a key does not complete the level. No other gate, lock, plate, or loose-object state matters to validation.

Controls

UP, DOWN, LEFT, RIGHT: move exactly one logical cell.

SPACE:

while empty-handed, pick up the loose object under the agent;

while carrying, drop the carried object on the current cell if that cell can support it.

Clicking any rendered cell is accepted by the action interface but is an inert no-op. There are no clickable objects, cursor states, or click affordances.

Objects are never picked up automatically.

Carrying

The inventory holds exactly one object.

Weights, keys, and the artifact are all carryable. Loose objects do not block movement: the agent may walk onto an object and then press SPACE.

A direct swap is impossible. If the agent carries one object while standing on another, SPACE fails rather than exchanging them. This makes nearby parking cells physically important.

Pressure plates and gates

Each plate is visibly connected to exactly one gate by a matching conduit.

A plate is active exactly when a loose carryable object rests on its cell.

The following activate a plate:

a weight;

a key;

the artifact.

The following do not:

the agent;

an object currently being carried;

an object on an adjacent cell.

A linked gate is traversable exactly while its plate is active. Picking an object off a plate closes the linked gate during that same action transition. Dropping an object onto a plate opens it during that same transition.

Keys and locks

A closed lock blocks movement.

Attempting to enter a lock while carrying its matching key:

consumes the key;

permanently opens the lock;

moves the agent into the lock cell;

all in the same directional action.

The open lock remains visibly recessed and traversable for the rest of the level. A wrong object or wrong key does nothing except produce collision feedback.

Only one red key shape is needed in this campaign, so the player learns the relationship from direct shape and color matching rather than an arbitrary symbolic code.

2. Exact state-machine contract
2.1 Native space

The gameplay lattice is 13 × 13.

Coordinates are:

x = 1…13, left to right;

y = 1…13, top to bottom.

Each level explicitly defines which cells are floor. Every unspecified cell is solid wall.

Each logical movement action changes one coordinate by exactly one:

UP: (x, y - 1)

DOWN: (x, y + 1)

LEFT: (x - 1, y)

RIGHT: (x + 1, y)

There is no diagonal movement.

2.2 Mutable state
player_pos: (x, y)
inventory: NONE | item_id
loose_position[item_id]: (x, y) for every non-carried, non-consumed item
open_locks: set[lock_id]
completed: boolean

Item metadata is immutable:

item_kind[item_id]: WEIGHT | KEY | ARTIFACT
key_shape[item_id]: RED_KEY when item_kind == KEY

Plate and gate states are derived rather than independently stored:

plate_active[channel] =
    any loose_position[item_id] == plate_position[channel]

gate_open[channel] = plate_active[channel]

This prevents a plate and gate from becoming inconsistent.

2.3 Cell layers

A logical cell has one terrain layer and optional entity layers.

Terrain may be:

wall;

ordinary floor;

pressure plate;

gate;

lock;

exit pedestal.

A cell may additionally contain:

at most one loose carryable;

the agent.

The agent and one loose object may occupy the same cell. Two loose objects may not.

2.4 Droppable cells

An object may be dropped only on:

ordinary floor;

a pressure plate.

It may not be dropped on:

walls;

a gate cell, even while the gate is open;

a lock cell, even after the lock opens;

the exit pedestal;

a cell already containing a loose object.

The starting cell is ordinary floor unless explicitly marked otherwise.

2.5 step() resolution order

The sole state-mutating entry point is:

step(action)

Rendering and animation never mutate game state.

For every action:

A. Directional action

Compute the target cell.

If the target is wall or outside the authored floor set, reject movement.

If the target is a closed gate, reject movement.

If the target is a closed lock:

if inventory contains its matching key, consume the key, mark the lock open, and move into the target;

otherwise reject movement.

Otherwise move into the target. Loose objects do not block movement.

Recompute all plate and gate states.

Run the success validator.

B. SPACE

If inventory is empty:

if the current cell contains one loose object, remove it from loose_position and place its ID in inventory;

otherwise make no state change.

If inventory is occupied:

if the current terrain is droppable and there is no loose object there, place the carried object on the current cell and clear inventory;

otherwise make no state change.

Then:

recompute all plate and gate states;

run the success validator.

SPACE never performs a swap.

C. Click

For every click coordinate:

step(CLICK(px, py))

returns the same gameplay state. There is no cursor or selected module.

D. Completion

After every resolved action:

completed =
    player_pos == exit_position
    AND inventory != NONE
    AND item_kind[inventory] == ARTIFACT

Completion is latched once true.

The exit does not consume the artifact. Loose weights, open gates, open locks, and unused keys are ignored by validation.

2.6 Impossible gate-overlap case

In the authored levels, a gate cannot lose support while the agent occupies that gate cell:

loose-object state changes only at the agent’s current cell;

no plate shares a cell with a gate;

an object cannot be dropped on a gate.

An implementation may assert this invariant rather than inventing a gate-crushing rule.

2.7 Exploration and failure

There is:

no energy;

no action budget;

no timer;

no automatic loss condition.

Invalid actions produce feedback but do not otherwise punish exploration. The host environment may restart an episode, but restart is not part of the game’s action space.

3. Rendering contract
3.1 Canvas

The render is exactly 64 × 64 pixels using only integer color IDs 0–15.

The logical board is rendered at four pixels per logical cell.

For logical cell (x, y), its top-left render pixel is:

X = 6 + 4 × (x - 1)
Y = 9 + 4 × (y - 1)

Thus the gameplay board occupies:

X = 6…57
Y = 9…60

The region above the board is reserved for the inventory indicator.

There are no internal grid lines or cell borders. Adjacent floor cells merge into continuous regions. Wall outlines appear only at actual floor–wall boundaries.

No text, digits, letters, or action labels are rendered. The ASCII diagrams later in this specification are developer notation only.

3.2 Palette assignment
Role	Color IDs
Outer background	5
Wall body and wall edge	4, 3
Ordinary floor	1
Neutral inset/shadow	2
Agent	10 with 9 outline
Weight	2 with 3 shadow
Red key and lock	8, 13
Artifact	12 with 0 center
Exit	14 ring with 12 artifact recess
Channel α	9, 10
Channel β	6, 7
Channel γ	11, 12
Channel δ	15, 0
Failure flash	8

No antialiasing or intermediate colors are used.

3.3 Entity silhouettes

All major entities are connected multi-pixel shapes.

Agent

A connected 3 × 3 rounded figure inside its logical cell:

outline 9;

center 10.

When standing over a loose object, the object’s outer corners remain visible.

Weight

A connected beveled 3 × 3 block:

gray body 2;

dark lower/right edge 3.

Its neutral appearance communicates that it has no matching lock or destination of its own.

Key

A connected 3 × 4 silhouette within the tile:

a 2 × 2 red head;

a maroon shaft and tooth.

The matching closed lock contains the same silhouette at larger scale.

Artifact

A conspicuous connected 3 × 3 orange diamond using 12, with a white center pixel 0.

Exit pedestal

A full-tile green frame using 14, containing the same orange diamond silhouette as the artifact. The matching shape is visible from Level 1 onward.

Plate

A full 4 × 4 module:

channel-colored outer rim;

neutral raised center while empty;

center shifted down one pixel and filled with the channel accent while occupied.

An object never completely hides the rim.

Gate

A full-cell barrier with two dark posts and a channel-colored crossbar.

Closed: the crossbar spans the traversable opening.

Open: the bar retracts into the posts, leaving visible channel-colored end caps.

Lock

A full-tile red/maroon slab with the key silhouette cut through its center.

After opening, the slab recesses into the surrounding wall, leaving a dark floor-like opening with a thin maroon rim.

3.4 Inventory and carried-object display

A 6 × 6 inventory frame is rendered at pixels:

x = 29…34
y = 1…6

Empty: dark center with a gray outline.

Occupied: a 4 × 4 version of the carried object appears inside.

The carried object also appears in the upper portion of the agent’s logical cell. This makes the inventory state legible without text and prevents confusion between a loose object under the agent and a carried object.

3.5 Pressure-channel identity

Each plate and its gate share:

primary color;

accent color;

one large center-cut pattern;

a visible conduit.

The center patterns merely reinforce direct matching; the player never has to memorize a symbolic legend.

The conduit is a one-pixel orthogonal line drawn underneath entities. Inactive conduits show the primary color as separated segments. Active conduits become continuous and gain accent pixels.

Each level specifies the conduit polyline as logical tile centers. Conduits are visual overlays and never affect movement. They may be etched into wall surfaces where necessary.

3.6 Animation and feedback

All gameplay state changes are committed by step() before the first feedback frame. Animation frames do not consume actions and do not call step().

The environment presents the feedback frames before accepting another action.

Plate and gate transition

Four deterministic frames:

object settles or lifts;

plate center depresses or rises;

gate bar retracts or extends halfway;

gate reaches its committed state and the conduit stabilizes.

Lock opening

Five deterministic frames:

carried key contracts toward the lock;

lock halves separate;

lock halves recess;

agent interpolates into the lock cell;

open-lock state stabilizes.

The key is already consumed and the agent already occupies the target in the committed state.

Invalid movement

Wall: agent nudges toward the wall for two frames.

Closed gate: gate crossbar shakes for three frames.

Closed or wrong lock: lock silhouette flashes 8 for two frames.

Invalid SPACE

Carrying onto an occupied cell: loose object and inventory border flash 8.

Carrying on nondroppable terrain: current terrain and inventory border flash 8.

Empty-handed with no object: the current floor cell gives one brief 8 outline pulse.

Completion

Six frames:

exit ring pulses 14;

artifact and exit recess alternate 12 and 0;

all active conduits remain in their actual state.

There is no danger animation because the game has no hazards.

4. Authored-map notation

The diagrams below are exact 13 × 13 implementation maps, not player-facing graphics.

#  wall
.  ordinary floor
s  initial agent position
w  loose weight
k  loose red key
a  artifact
e  exit pedestal
1  channel α plate
A  channel α gate
2  channel β plate
B  channel β gate
3  channel γ plate
C  channel γ gate
4  channel δ plate
D  channel δ gate
L  closed red lock

An item can cover a plate in the initial state. Level 2’s coordinate table explicitly records that overlap even though the map can show only one character.

Reference replays use:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

A slash separates conceptual phases and does not represent an action.

The listed depths are shortest state-changing solutions under the state machine above. No invalid actions or clicks are included.

5. Curriculum overview
Level	New evidence	Distinguishing observation
1	Pickup, inventory, artifact objective	Walking over the artifact does not collect it; SPACE visibly carries it, and only the matching exit accepts it
2	Loose occupancy continuously controls gates	One occupied plate/open gate and one empty plate/closed gate are visible initially; picking up the weight closes the first gate immediately
3	Keys are consumed but locks stay open	The solution must return through the same lock after the key has disappeared
4	A key is physically valid pressure	The only object capable of opening the first gate is the key; the weight is visibly behind that gate
5	A weight can be relayed through an alternate pressure route	Removing the weight closes the original route, but a key-held second route remains open
6	The artifact is also valid pressure	The mandatory lock has already consumed the key, leaving the artifact as the only possible support
7	No new rule; complete composition	The artifact must replace the weight on one plate so the weight can open a remote final route
6. Authored levels
Level 1 — Carry It Home
Player-facing discovery

The player sees only three meaningful things:

the agent;

the orange artifact;

a green pedestal containing the artifact’s orange silhouette.

Walking across the artifact does not remove it. Pressing SPACE while standing on it makes the artifact appear both above the agent and in the inventory frame. Entering the pedestal with it immediately completes the level.

The player may cheaply test the simpler hypothesis “touching the artifact is enough” by walking over it to the exit. The pedestal flashes its empty artifact recess but does not complete, making the missing carried object visually obvious.

Exact initial state
#############
#############
#############
#############
#############
###....######
###sa.e######
#############
#############
#############
#############
#############
#############
Player:       (4, 7)
Inventory:    EMPTY
Selection:    NONE
Artifact:     (5, 7)
Exit:         (7, 7)
Locks:        none
Plates/gates: none
Completed:    false

All marked floor cells are ordinary droppable floor except the exit.

Reference solution

Depth: 4 actions

R S R R

Checkpoints:

Move onto the artifact.

Pick it up.

Move through (6,7).

Enter the exit at (7,7) while carrying it.

Attribution check

There are no plates, gates, locks, weights, keys, decoys, or simultaneous state changes. The visual change after SPACE can only be attributed to pickup, and the completion can only be attributed to carrying the matching artifact into the pedestal.

Level 2 — Held, Not Toggled
Player-facing discovery

The opening frame itself provides positive and negative evidence:

the weight already rests on channel α’s plate;

α’s plate is depressed and α’s gate is open;

channel β’s plate is empty;

β’s gate is closed.

The player begins on the far side of α’s open gate. To solve the forward obstruction, the player must cross α, pick up the weight, and move it to β.

Picking up the weight immediately raises α’s plate and closes α’s gate behind the player. This rules out the plausible mistaken rule that a plate permanently toggles its gate.

Dropping the same weight on β opens the second gate.

Exact initial state
#############
#############
#############
#############
#############
#..#...#...##
#s.Aw.2Ba.e##
#..#...#...##
#############
#############
#############
#############
#############
Player:       (2, 7)
Inventory:    EMPTY
Selection:    NONE

Weight:       (5, 7), loose and initially on plate α
Artifact:     (9, 7)
Exit:         (11, 7)

Plate α:      (5, 7), initially ACTIVE
Gate α:       (4, 7), initially OPEN
Conduit α:    [(5,7), (4,7)]

Plate β:      (7, 7), initially INACTIVE
Gate β:       (8, 7), initially CLOSED
Conduit β:    [(7,7), (8,7)]

Locks:        none
Completed:    false
Reference solution

Depth: 12 actions

R R R S / R R S / R R S / R R

Checkpoints:

Cross open gate α and stand on the weight at (5,7).

Pick up the weight; α closes behind the player.

Carry it to plate β at (7,7) and drop it.

Cross gate β and pick up the artifact at (9,7).

Enter the exit at (11,7).

Attribution check

The initial paired comparison makes occupancy causality visible before the player acts. The required pickup then demonstrates closure. No key, lock, alternate object, or route can explain either transition.

The first gate is not decorative: it must be crossed while supported, and it visibly closes when the support is removed.

Level 3 — A Spent Key Leaves a Passage
Player-facing discovery

The exit is immediately to the player’s left, but the artifact is beyond a red lock to the right.

The red key and lock have the same large silhouette. Carrying the key into the lock consumes it and retracts the lock. The artifact must then be carried back through that same location to the exit.

This return trip proves that the lock remains open after the key disappears; a mistaken “the key only lets me pass once” hypothesis cannot explain the solution.

Exact initial state
#############
#############
#############
#############
#############
##....#...###
##esk.L.a.###
##....#...###
#############
#############
#############
#############
#############
Player:       (4, 7)
Inventory:    EMPTY
Selection:    NONE

Red key:      (5, 7)
Red lock:     (7, 7), CLOSED
Artifact:     (9, 7)
Exit:         (3, 7)

Plates/gates: none
Open locks:   empty set
Completed:    false

The wall column at x=7 has no other opening, so the lock is the sole connection between the chambers.

Reference solution

Depth: 13 actions

R S / R R R R / S / L L L L L L

Checkpoints:

Pick up the key at (5,7).

Move through (6,7) and enter the lock at (7,7), consuming the key.

Continue to the artifact at (9,7) and pick it up.

Return through the now-recessed lock and enter the exit at (3,7).

Attribution check

There are no pressure devices. The only state change accompanying the key’s disappearance is the lock’s permanent retraction. Returning through the same cell makes permanence necessary rather than merely decorative feedback.

Level 4 — Borrowed Pressure
Player-facing discovery

The gate blocks the only route to the weight. The only loose carryable available on the starting side is the key.

Known rules no longer suffice under the mistaken hypothesis “only weights activate plates.” Placing the key on the nearby plate immediately depresses it and opens the gate, revealing that the key has ordinary physical pressure.

The key is also needed at the lock beyond the gate. The player therefore has to:

use the key as temporary pressure;

retrieve the weight;

park the weight;

remove the key, closing the gate;

park the key;

install the weight on the plate;

reclaim the key and spend it at the lock.

Because direct swapping is forbidden, the nearby floor cells are a genuine staging area rather than decorative room space.

Exact initial state
#############
#############
#############
#############
#############
##....#....##
##sk1.A.w..##
##....#....##
#########L###
########ea.##
########...##
#############
#############
Player:       (3, 7)
Inventory:    EMPTY
Selection:    NONE

Red key:      (4, 7)
Weight:       (9, 7)
Artifact:     (10, 10)
Exit:         (9, 10)

Plate α:      (5, 7), INACTIVE
Gate α:       (7, 7), CLOSED
Conduit α:    [(5,7), (7,7)]

Red lock:     (10, 9), CLOSED
Completed:    false

The wall at x=7 makes gate α the only connection between the left and upper-right rooms. The lower artifact chamber is reachable only through the lock.

Reference solution

Depth: 35 actions

R S
/ R S
/ R R R R S
/ L L L U L S
/ D S
/ R S
/ U L S
/ D S
/ R S
/ R R D R R D D S L

State checkpoints:

Pick up the key and drop it on plate α.

Cross gate α and pick up the weight.

Return and park the weight at (5,6).

Pick the key off plate α; the gate closes.

Park the key at (6,7).

Pick up the weight and drop it on plate α.

Pick up the key from (6,7).

Cross the gate, unlock (10,9), pick up the artifact, and enter the exit.

Necessity and anti-degeneracy

The weight is initially inaccessible, so the key must activate the plate.

The key cannot remain on the plate because the lock requires it.

The gate cannot remain open after the key is removed unless the weight replaces it.

Direct object swapping is impossible, so at least two staging placements are required.

The artifact cannot substitute for either object because it is behind both the gate and lock.

The key’s operational rule does not change: it is always a carryable loose object and therefore always valid pressure. Its role changes only because of placement and future need.

Level 5 — The Relay Loop
Player-facing discovery

This level establishes the central relay insight cleanly.

The weight can open upper gate α, allowing entry to the middle region. But the weight cannot simply be picked up and carried through α: picking it up closes α before the player can cross.

A key and plate β are visible in the middle region. Putting the key on β opens the lower route back to the starting room. The player can then:

return through still-open α;

pick up the weight, closing α;

carry the weight through key-held gate β;

advance the weight to plate γ.

Only after the weight reaches γ can the player retrieve the key from β, cross gate γ, and open the lock. The lock is physically beyond gate γ, so the game never invites an irreversible premature key use.

Exact initial state
#############
#.....#...#e.
#.....#...#.a
#.sw1.A...#L#
#.....#.k.#..
#.....#..3C..
#.....#...#..
#.....#...#..
#.....B.2.###
#.....#...###
#############
#############
#############
Player:       (3, 4)
Inventory:    EMPTY
Selection:    NONE

Weight:       (4, 4)
Red key:      (9, 5)
Artifact:     (13, 3)
Exit:         (12, 2)

Plate α:      (5, 4), INACTIVE
Gate α:       (7, 4), CLOSED
Conduit α:    [(5,4), (7,4)]

Plate β:      (9, 9), INACTIVE
Gate β:       (7, 9), CLOSED
Conduit β:    [(9,9), (7,9)]

Plate γ:      (10, 6), INACTIVE
Gate γ:       (11, 6), CLOSED
Conduit γ:    [(10,6), (11,6)]

Red lock:     (12, 4), CLOSED
Completed:    false

The upper α crossing and lower β crossing connect the same two large regions at different vertical positions. Gate γ leads to the eastern staging area; the lock then leads to the artifact pocket.

Reference solution

Depth: 57 actions

R S R S
/ R R R D R S
/ D D D D S
/ U U U U U L L L L S
/ D D D D D R R R U U U R R S
/ D D D L S
/ U U U R R R U U U R S U L

State checkpoints:

Put the weight on plate α.

Cross α and pick up the key at (9,5).

Put the key on plate β, opening the lower route.

Return through α and pick up the weight; α closes.

Carry the weight down through gate β and place it on plate γ at (10,6).

Return to plate β and retrieve the key; β closes.

Carry the key through gate γ, unlock (12,4), collect the artifact, and reach the exit.

Necessity and anti-cleanliness

Removing the weight from α closes the upper crossing before the weight can traverse it.

Gate β is therefore the only possible route for the carried weight.

Only the key is available to support β during the transfer.

The key must later be reclaimed because the lock is mandatory.

Gate γ prevents the key from being spent before the weight relay is complete.

The two connections are vertically offset, so the level is a loop with changing accessibility, not a left-to-right device list.

The weight changes from static anchor at α to transported support at γ. The key changes from route support at β to consumable lock opener.

Level 6 — Cargo as Support
Player-facing discovery

The weight first holds lower gate α. The key must then be carried through α and immediately consumed by the mandatory lock leading into the central room.

This ordering is important: by the time the player reaches the artifact and plate β, the key no longer exists. The weight is still stranded on plate α.

The final gate γ visibly needs pressure, but no loose weight or key is present in the central room. The artifact is the only available carryable. Dropping it on β immediately opens the upper alternate route.

The artifact must temporarily cease being the objective and become infrastructure:

artifact holds β;

player returns to the weight;

weight is carried through β;

weight is placed on γ;

artifact is reclaimed;

artifact is carried through γ to the exit.

Exact initial state
#############
#.....#######
#.....###..##
#.....B..2.##
#.k...###..##
#.....###3.Ce
#.....###..##
#.....###..##
#.sw1.A.L.a##
#.....###..##
#.....#######
#############
#############
Player:       (3, 9)
Inventory:    EMPTY
Selection:    NONE

Weight:       (4, 9)
Red key:      (3, 5)
Artifact:     (11, 9)
Exit:         (13, 6)

Plate α:      (5, 9), INACTIVE
Gate α:       (7, 9), CLOSED
Conduit α:    [(5,9), (7,9)]

Plate β:      (10, 4), INACTIVE
Gate β:       (7, 4), CLOSED
Conduit β:    [(10,4), (7,4)]

Plate γ:      (10, 6), INACTIVE
Gate γ:       (12, 6), CLOSED
Conduit γ:    [(10,6), (12,6)]

Red lock:     (9, 9), CLOSED
Completed:    false

Gate α leads to a one-cell vestibule followed immediately by the lock. The central room containing the artifact and plate β is unreachable without consuming the key. Gate β is a separate upper connection back to the western room.

Reference solution

Depth: 63 actions

R S R S
/ U U U U L L S
/ D D D D R R R R R R R R S
/ U U U U U L S
/ D D D D D L L L L L S
/ U U U U U R R R R R D D S
/ U U S
/ D D R R R

State checkpoints:

Put the weight on plate α.

Retrieve the key from (3,5).

Carry it through gate α and the mandatory lock, then pick up the artifact.

Put the artifact on plate β.

Return through the still-open lower route and pick up the weight; α closes.

Carry the weight through gate β and put it on plate γ.

Retrieve the artifact from β; β closes.

Carry the artifact through gate γ into the exit.

Necessity and anti-cleanliness

The mandatory lock consumes the key before plate β becomes accessible, so key-as-β-support cannot replace the new artifact interaction.

Carrying the artifact directly back through α cannot win because the exit lies behind gate γ.

Leaving the artifact on γ cannot win because picking it up would close γ before traversal.

The weight must migrate from the lower western plate to the central final plate.

The artifact is useful first as cargo, then as temporary route support, then as cargo again.

The upper and lower crossings create a forced revisit with different active routes rather than a linear sequence.

This is the first level in which the visually privileged objective object obeys the same pressure rule as ordinary cargo.

Level 7 — Closed Circuit
Player-facing discovery

No new operational rule appears. The final level composes the established rules into a cross-linked route.

The first half extends the Level 5 relay:

weight holds plate α;

key holds plate β;

weight travels through gate β;

weight advances to plate γ;

key is reclaimed and used at the lock.

The second half creates a stronger exchange:

weight on plate γ currently holds the only route to the far room;

the artifact has been retrieved through that route;

the weight must leave γ and travel to remote plate δ;

the artifact must therefore replace the weight on γ;

weight on δ opens a different gate near the exit;

the player returns through artifact-held γ, reclaims the artifact, and uses the remotely opened exit gate.

Plate δ and its gate are not adjacent or sequential. A long purple conduit visibly links the far lower plate to the upper exit gate. This cross-link turns the final arrangement into a dependency network:

artifact holds route to weight’s destination
weight at destination opens artifact’s exit route
Exact initial state
#############
###########..
#.....#...De.
#.sw1.A...###
#.....#.k.#.a
#.....#...#L#
#.....#...#..
#.....#..3C..
#.....B.2.#.4
#.....#...#..
#############
#############
#############
Player:       (3, 4)
Inventory:    EMPTY
Selection:    NONE

Weight:       (4, 4)
Red key:      (9, 5)
Artifact:     (13, 5)
Exit:         (12, 3)

Plate α:      (5, 4), INACTIVE
Gate α:       (7, 4), CLOSED
Conduit α:    [(5,4), (7,4)]

Plate β:      (9, 9), INACTIVE
Gate β:       (7, 9), CLOSED
Conduit β:    [(9,9), (7,9)]

Plate γ:      (10, 8), INACTIVE
Gate γ:       (11, 8), CLOSED
Conduit γ:    [(10,8), (11,8)]

Plate δ:      (13, 9), INACTIVE
Gate δ:       (11, 3), CLOSED
Conduit δ:    [(13,9), (13,4), (11,4), (11,3)]

Red lock:     (12, 6), CLOSED
Completed:    false

The artifact pocket at y=5 and the exit pocket at y=2…3 are separated by solid wall. The exit cannot be approached from the artifact side. Gate δ is the only entrance to the exit pocket.

Reference solution

Depth: 86 actions

R S R S
/ R R R D R S
/ D D D D S
/ U U U U U L L L L S
/ D D D D D R R R U R R S
/ D L S U R R R U U U R S
/ L D D D L L U S D S D S U U S D S D S
/ U R R D R S
/ U L L L S
/ U U U U U R R

Detailed checkpoints:

Phase 1: Establish the first route

Pick up the weight at (4,4).

Drop it on plate α at (5,4).

Cross gate α and collect the key at (9,5).

Phase 2: Build the alternate return

Drop the key on plate β at (9,9).

Return through gate α.

Pick up the weight from plate α; gate α closes.

Carry the weight through key-held gate β.

Phase 3: Advance the weight and spend the key

Put the weight on plate γ at (10,8), opening gate γ.

Retrieve the key from plate β; gate β closes.

Carry the key through gate γ.

Enter lock (12,6), consuming the key.

Pick up the artifact at (13,5).

Phase 4: Exchange artifact and weight at plate γ

Return through gate γ while the weight still holds it.

The reference staging arrangement is:

drop the artifact at (10,7);

pick the weight off plate γ at (10,8), closing gate γ;

park the weight at (10,9);

retrieve the artifact from (10,7);

drop the artifact onto plate γ at (10,8), reopening gate γ;

retrieve the weight from (10,9).

No direct swap is possible, so all six object-state changes are structurally required.

Phase 5: Open the remote exit

Carry the weight through artifact-held gate γ.

Put it on plate δ at (13,9).

Gate δ at (11,3) opens remotely.

Phase 6: Reclaim the objective

Return through gate γ.

Pick the artifact off plate γ; gate γ closes behind the player.

Move north through the hub.

Cross open gate δ and enter the exit at (12,3) while carrying the artifact.

Necessity and anti-degeneracy

The key must hold β before it is consumed; otherwise the weight cannot leave plate α.

The lock is behind gate γ, so the key cannot be spent before the weight reaches γ.

The artifact must replace the weight on γ because the weight is the only remaining object that can be transported to δ.

Putting the artifact on δ is insufficient: picking it up would close the exit gate, and the weight would remain stranded on the opposite side of γ.

Leaving the weight on γ is insufficient because nothing else can hold δ while the artifact is carried.

The far plate opens a gate in a different region rather than the next gate along the route. Spatially removing this cross-link would destroy the puzzle.

The staging cells (10,7) and (10,9) are necessary because occupied plates cannot be directly swapped.

All four pressure channels participate in the shortest solution, and omitting any one disconnects a required transfer.

7. Reuse in contrasting roles

The campaign uses a small rule set repeatedly rather than adding a parade of devices.

Entity	Early role	Contrasting later role
Weight	Static doorstop in Level 2	Relayed cargo in Level 5; final remote support in Level 7
Key	Permanent lock opener in Level 3	Temporary gate support in Levels 4, 5, and 7; something that must be delayed before spending
Artifact	Pure objective cargo in Levels 1–5	Temporary route support in Level 6; replacement support and final cargo in Level 7
Pressure gate	Simple bottleneck in Level 2	Alternate transport route in Level 5; remotely cross-linked exit route in Level 7
Parking floor	Forgiving movement space early	Required exchange resource in Levels 4 and 7

The operational rules never change. Every role reversal comes from:

which side of a gate an object occupies;

whether another object must later be carried;

whether a key still has to be spent;

whether removing support destroys the return route;

whether the exit is connected to the current region;

whether a replacement object has been staged before release.

The resulting puzzle family is about maintaining a physically self-consistent route while the very objects holding that route open are also the objects that must travel through it.
