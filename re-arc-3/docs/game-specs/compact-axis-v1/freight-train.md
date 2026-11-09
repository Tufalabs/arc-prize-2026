Freight Train

Exact authored level count: 7

This is a game about pushing lines of carts becoming difficult because compatible carts permanently clasp into rigid trains.

Freight Train is a compact warehouse-pushing game with one irreversible event: when a visible hook touches a visible socket, the two carts lock together forever. A longer train can transmit force from places the worker cannot reach, but every future push must accommodate the train’s entire footprint.

Nothing is hidden. There are no switches, counters, inventories, timers, or conditional floor types. Difficulty comes entirely from walls, approach direction, coupler orientation, permanent joins, and the order in which rigid footprints are created.

1. Player-facing system

The player learns four increasingly strong interpretations of the same physical rule:

Touching carts transmit a push.

Only hook-to-socket contact creates a permanent clasp.

A clasped train can be pushed sideways and used as a long or offset ram.

Every member of a train needs destination space, so extra reach also creates extra obstruction.

Cargo colors provide the objective directly: each cart must finish on the bay with the same cargo color. A correctly delivered cart receives an immediate white halo.

There is no action budget. Invalid experiments do not damage the state, and SPACE restarts the current level, allowing generous exploration after an irreversible bad coupling.

2. Native board and rendering contract
Logical coordinates

Every level uses a 12×12 logical board.

Coordinates are (row, column).

Both are zero-indexed from 0 through 11.

Row increases downward.

Column increases rightward.

A cart, worker, or bay occupies one logical cell.

Every movement action translates the worker by exactly one logical cell.

A cart push translates every affected cart by exactly one logical cell.

The authored floor masks use:

. — floor

# — wall

These characters are documentation only. No characters or text are rendered in the game.

64×64 rasterization

The 12×12 board occupies a 60×60 region.

Logical cell size: 5×5 rendered pixels

Board origin: rendered pixel (2,2)

Logical cell (r,c) maps to:

x = 2 + 5c ... 6 + 5c

y = 2 + 5r ... 6 + 5r

The two-pixel outer margin is color 5.

There are no internal grid lines or tile borders.

Connected wall cells therefore appear as continuous shapes rather than a bordered logic grid.

Color use
Role	Color ID
Correct-delivery halo	0
Floor	1
Cart frame	2
Undercarriage	3
Wall	4
Outside board / socket notch	5
Worker	6
Worker wall-bump flash	7
Blocked-push impact	8
Blue cargo B	9
Completion pulse	10
Hooks and permanent clasps	11
Orange cargo O	12
Maroon cargo M	13
Green cargo G	14
Purple cargo P	15

The cart frame and warehouse remain quiet gray. Saturated color appears mainly in small cargo emblems, the worker, and transient feedback.

Bays

A bay is floor with four cargo-colored corner brackets. Relative to its 5×5 cell, draw the cargo color at:

(0,0) (0,1)       (0,3) (0,4)
(1,0)                   (1,4)

(3,0)                   (3,4)
(4,0) (4,1)       (4,3) (4,4)

A cart on a matching bay changes those brackets to color 0. On the first matching frame, they briefly pulse color 10, then remain color 0.

A wrong cart does not activate the bay. The bay’s colored corners remain visible around the cart.

Worker

The worker is a connected 3×3 shape centered in its logical cell:

All central 3×3 pixels: color 6

Center pixel: color 0

The worker has no facing state.

Carts and couplers

Documentation notation:

B H→ — blue cargo, horizontal cart, hook facing east, socket facing west

O H← — orange cargo, hook facing west, socket facing east

G V↑ — green cargo, hook facing north, socket facing south

Orientation affects only the coupler faces. It does not constrain movement direction. A horizontal cart can be pushed north or south without rotating.

For the canonical H→ cart:

Draw a central 3×3 cart frame in color 2.

Draw a five-pixel plus-shaped cargo emblem in its cargo color:

(1,2), (2,1), (2,2), (2,3), (3,2)

Draw undercarriage pixels (4,1), (4,2), (4,3) in color 3.

Draw the east hook as a yellow T:

(1,4), (2,4), (3,4), (2,3) in color 11

Draw the west socket as:

(1,0), (3,0) in color 11

(2,0) in color 5, creating a visible central notch

H← is the horizontal mirror. Vertical variants are rotations of this template.

A hook is therefore a solid T-shaped face; a socket has a conspicuous open notch. Side faces contain neither.

Visible permanent coupling

When compatible ends lock:

Replace the two touching face patterns with a 3×2 clasp in color 11.

Connect the undercarriage through the seam with color 3.

Keep the clasp visible for the remainder of the level.

For a horizontal seam, the clasp occupies rows 1..3 on the touching edge pixels of both cells. The vertical version is the rotated equivalent.

Initial trains are rendered with these clasps from the first frame.

3. Actions
Cardinal movement

UP, DOWN, LEFT, and RIGHT are the puzzle actions.

Let d be the chosen direction and q = worker_position + d.

If q is a wall or outside the 12×12 board, nothing moves.

If q is unoccupied floor or a bay, the worker moves to q.

If q contains a cart, the game attempts a push.

SPACE

SPACE restores the complete authored initial state of the current level:

Worker position

Cart positions

Permanent locks

Goal halos

won = false

The reset is performed inside step(SPACE). It is accompanied by a single render-only perimeter pulse in color 10.

SPACE is never required by an intended solution and is excluded from solution-depth counts.

Clicking

A click on any rendered cell is accepted by the action interface but is a no-op.

No object is clickable.

No selection outline is shown.

selected_id remains null.

cursor_state remains null.

This game deliberately uses physical contact rather than arbitrary click interactions.

4. State machine
Static level state
floor[r][c]              boolean
bays[cargo]              set of cells
cart_specs[id]           {
    cargo,
    axis,                H or V
    hook_direction       U, D, L, or R
}
initial_worker           cell
initial_cart_positions   array of cells
initial_locks            set of unordered cart-ID pairs
Dynamic state
worker_position
cart_positions[id]
locks                    permanent cart-to-cart seams
won
selected_id = null
cursor_state = null

No timer, energy, turn phase, queue, or hidden state exists.

Rigid components

The connected components of locks are rigid trains. A one-cart component is an ordinary cart.

Every member of a rigid component always translates by the same vector. Components never rotate, bend, or split.

Push propagation

When the worker attempts to push a cart:

Initialize affected with every cart in the contacted cart’s rigid component.

For every cart i in affected, inspect cart_positions[i] + d.

If that destination contains a cart from another rigid component, add that entire component to affected.

Repeat until no component is added.

If any affected cart’s destination is a wall or outside the board, the entire push fails.

Otherwise, translate every affected cart by d simultaneously.

Move the worker into the originally contacted cart cell.

Resolve new couplings.

Recompute bay matches and the win condition.

This closure rule permits:

A loose line of carts to transmit a push

A broadside train to push multiple separate carts simultaneously

One train to push another train

A distant member of a train to cause a push to fail

Coupling test

For two orthogonally adjacent carts A and B, let d point from A to B.

They permanently couple exactly when:

A has a hook on face d and B has a socket on face -d, or

A has a socket on face d and B has a hook on face -d

Cargo color is irrelevant.

The following contacts never couple:

Hook to hook

Socket to socket

Side to side

Side to hook

Side to socket

After every successful push, all newly compatible adjacent pairs are locked simultaneously. If several seams appear in one action, all are added before validation.

At level initialization, compatible adjacent authored carts are normalized into initial trains before the first render.

Important movement consequences

The worker may push any member of a train.

A train may translate along or perpendicular to its long axis.

No cart ever rotates.

There is no pulling.

A train’s entire translated footprint must be clear.

An uncoupled cart can transmit force and then later move away.

A compatible contact made during a push is permanent from the end of that action onward.

5. Update and animation order

All logical changes occur only in response to step().

Ordinary walking

Walking renders immediately at the destination cell. No multi-frame animation is required.

Successful pushes

A push uses a short deterministic translation animation because several objects can move from one cause.

Render affected carts and the worker at offsets:

2/5 tile
4/5 tile
5/5 tile

All affected carts use the same interpolation fraction.

The final logical positions become active on the 5/5 frame.

New clasps appear immediately after that final translation frame.

Bay halos and won are evaluated after clasps appear.

Intermediate frames consume no player actions.

New actions are not accepted until the stable final frame.

Blocked cart push

No logical state changes.

For two render frames, draw a color-8 bar on every affected cart face whose destination edge is obstructed. Flash the actual distant obstruction, not merely the cart touched by the worker.

This feedback is central to Level 5.

Worker against wall

The worker remains still and flashes color 7 for one frame. No red cart-impact bar is shown.

Completion

When all carts are correctly delivered:

Set won = true.

Pulse every active bay halo 0 → 10 → 0.

Freeze cardinal and click actions.

Permit SPACE to replay the level, or permit the host to load the next level.

6. Objective and validation

A cart is correct exactly when its position belongs to bays[cart.cargo].

A level is won when:

for every cart id:
    cart_positions[id] is in bays[cart_specs[id].cargo]

The validator ignores:

Worker position

Number of actions

Number or shape of trains

Which route was used

Whether a cart was previously correct

Temporary occupation of bays by wrong carts

The validator allows:

The worker standing on a bay

A wrong cart temporarily occupying a bay

A correct cart leaving a bay and returning

Several delivered carts remaining coupled

Every authored level uses unique cargo colors per cart, so its final required cart coordinates are unambiguous.

7. Curriculum overview
Level	Main evidence	Intended depth
1. First Delivery	Walking, one-cell pushing, matching bay	2
2. Loose Line	Push propagation without permanent coupling	6
3. The Clasp	Hook-to-socket contact creates a rigid train	7
4. Far-End Lift	A train member reaches a cart the worker cannot push directly	28
5. Three-Wide Gate	A distant train member can block the whole push	40, plus one diagnostic failed push
6. Extension Route	Move an extension solo, then form and route a three-cart train	47
7. Two Departures	One train positions the carts that must later form another train	50

For route strings below, spaces and | are explanatory separators. Concatenate the U, D, L, and R symbols to replay the route.

Every level starts with:

won = false
selected_id = null
cursor_state = null
matched_bays = empty
8. Authored levels
Level 1 — First Delivery
Player-facing discovery

The matching blue cart and blue bay make the objective visible without a legend. The worker must first cross one empty cell, then walk into the cart. The cart moves onto the bay, the bay gains a white halo, and the level ends.

There is only one possible causal interaction. Nothing else changes during the push.

Floor mask
############
############
############
############
###......###
###......###
###......###
############
############
############
############
############
Initial state
Worker: (5,3)

C0: B H→ at (5,5)

Bays:
B: (5,6)

Initial locks: none
selected_id: null
cursor_state: null
Exact validator result
C0 must end at (5,6)
Worker position is ignored
Verified route
RR

State progression:

R: worker (5,3) → (5,4)

R: C0 (5,5) → (5,6) and worker enters (5,5)

The blue bay halo appears on the second action.

Attribution check

The first action demonstrates ordinary walking.

The second demonstrates pushing and matching.

No coupler contact, second cart, decoy bay, or unrelated state change is present.

Random wall bumps cannot accidentally satisfy the validator; the cart must be pushed in the visibly correct direction.

Level 2 — Loose Line
Player-facing discovery

Two carts begin touching, but their touching faces are both hooks. Pushing the rear blue cart moves both carts east. No clasp appears.

After the blue cart reaches its bay, the worker approaches the orange cart from below and pushes it north alone. This final action rules out the tempting mistaken hypothesis that all touching carts have become one object.

The level establishes one compound fact:

A contiguous cart line transmits a push, but incompatible faces do not create a permanent train.

Floor mask
############
############
############
############
##.......###
##.......###
##.......###
##.......###
##.......###
############
############
############
Initial state
Worker: (6,2)

C0: B H→ at (6,3)
C1: O H← at (6,4)

Touching faces:
C0 east face = hook
C1 west face = hook
Result = incompatible, no lock

Bays:
B: (6,5)
O: (5,6)

Initial locks: none
selected_id: null
cursor_state: null
Exact validator result
C0 must end at (6,5)
C1 must end at (5,6)
Verified route
RR | D RR U

Raw sequence:

RRDRRU

State progression:

First R: both carts move from columns 3,4 to 4,5.

Second R: both move to 5,6; C0 activates the blue bay.

D RR: worker reaches (7,6).

U: only C1 moves from (6,6) to (5,6).

Necessity and misconception test

C0 must move two cells east.

Because C1 begins immediately ahead, both east pushes require propagation.

C1 must then leave the line in a perpendicular direction.

A rule such as “touching carts are always glued” predicts the final U incorrectly.

A rule such as “only the touched cart moves” predicts the first R incorrectly.

The absent clasp visually agrees with the later separation.

Level 3 — The Clasp
Player-facing discovery

The orange cart sits in a pocket where the worker cannot stand below it. The blue cart can be pushed east into the gap.

This time the meeting faces are a hook and a socket. A thick yellow clasp appears immediately. The worker then pushes upward on only the blue member, but both carts move north together onto their bays.

The layout deliberately resembles Level 2 while producing a contrasting result.

Floor mask
############
############
############
############
############
#####..#####
##.....#####
##....######
##....######
############
############
############
Initial state
Worker: (8,2)

C0: B H→ at (6,4)
C1: O H→ at (6,6)

Potential joining faces after C0 moves east:
C0 east face = hook
C1 west face = socket

Bays:
B: (5,5)
O: (5,6)

Initial locks: none
selected_id: null
cursor_state: null
Exact validator result
C0 must end at (5,5)
C1 must end at (5,6)
Verified route
UU | RR | D R U

Raw sequence:

UURRDRU

State progression:

UU R: worker reaches (6,3).

Second R: C0 moves to (6,5), touches C1, and locks to it.

D R: worker reaches (7,5).

U: pushing C0 north translates the whole train to row 5.

Necessity and attribution check

C1 cannot move alone:

East destination (6,7) is a wall.

The eastern worker stance (6,7) is a wall.

South destination and stance route are blocked by (7,6).

The worker cannot stand directly below C1.

C0 must therefore become the accessible handle of a two-cart train.

The new property cannot be explained by Level 2’s loose propagation rule: the decisive push is perpendicular to the seam and contacts only C0. Without permanent coupling, C1 would remain at (6,6).

Level 4 — Far-End Lift
Player-facing discovery

The green cart must move north, but the cell directly beneath it is occupied by the orange cart. Moving the orange cart east merely wedges it in a dead pocket.

The blue cart can instead be routed to the orange cart’s exposed socket. Once joined, pushing upward under the blue member shifts the whole train. The orange member, one cell away from the worker, pushes the green cart north.

The green contact is against the side of a horizontal cart, so it does not lock. The train later moves away while the green cart remains delivered.

This is the first level where making a train is useful primarily because of where its far member can exert force.

Floor mask
############
############
#####....###
#####.....##
##........##
##......####
##......####
##...#..####
##......####
############
############
############
Initial state
Worker: (8,2)

C0: B H→ at (7,3)
C1: O H→ at (4,8)
C2: G V↑ at (3,8)

Bays:
B: (8,6)
O: (8,7)
G: (2,8)

Initial locks: none
selected_id: null
cursor_state: null
Exact validator result
C0 must end at (8,6)
C1 must end at (8,7)
C2 must end at (2,8)
Verified route
R UUU L U RRRR
| D R U
| L UU R D
| RR D L
| U L DDDD

Raw sequence:

RUUULURRRRDRULUURDRRDLULDDDD
Phase detail

Route and couple the handle

R UUU L U RRRR

C0 moves from (7,3) to (4,3).

It is then pushed east to (4,7).

Its hook meets C1’s west socket at (4,8).

C0 and C1 become a horizontal train.

Use the far end

D R U

The worker reaches (5,7).

Pushing north on C0 moves the train from row 4 to row 3.

C1’s destination contains C2, so C2 is propagated north to (2,8).

The C1–C2 contact is side-to-end and does not clasp.

Move away from the delivered cart

L UU R D

The worker loops above the train.

The train moves down one row to (4,7),(4,8), visibly separating from C2.

Align with the descent

RR D L

The worker enters the pocket that C1 previously blocked.

One west push moves the train to columns 6,7, clearing the wall beneath column 8.

Deliver the train

U L DDDD

The worker reaches the northern side of the right member.

Four south pushes place C0 and C1 on row 8.

Necessity and anti-cleanliness check

C2’s required southern stance is initially occupied by C1.

C1 cannot be pushed west because its eastern stance is sealed behind C1 and C2.

Pushing C1 east into (4,9) makes it unrecoverable because (4,10) is a wall.

C0 is therefore necessary both as a coupler and as the accessible pushing handle.

After the lift, the train cannot descend in column 8; it must first move down, use the newly opened side pocket, shift west, and then descend.

The object-use order is not simply the visual left-to-right order: the green cart is delivered before the train, and its movement opens the worker route used for later staging.

Level 5 — Three-Wide Gate
Player-facing discovery

A three-cart train lies between the worker and the rest of the warehouse. The most obvious upward push appears clear above the near member, but fails. The two farther members flash red against walls.

The nearby lower corridor allows the player to shift the train east until all three members align with a three-cell-wide shaft.

As the train rises, its right member pushes the purple cart ahead of it. The purple cart is inaccessible before the train clears the barrier, so this displacement is unavoidable rather than decorative. After the train parks, the purple cart must be brought back down, moved across the old staging corridor, and dropped into the left delivery pocket.

Floor mask
############
############
###........#
###........#
###........#
###........#
######...###
#..###....##
#.........##
#...#.....##
############
############
Initial state
Worker: (9,2)

C0: B H→ at (8,2)
C1: O H→ at (8,3)
C2: G H→ at (8,4)
C3: P V↑ at (5,8)

Initial locks:
C0--C1
C1--C2

Bays:
B: (4,3)
O: (4,4)
G: (4,5)
P: (9,1)

selected_id: null
cursor_state: null
Exact validator result
C0 must end at (4,3)
C1 must end at (4,4)
C2 must end at (4,5)
C3 must end at (9,1)
Distinguishing experiment

From the initial state, try:

U

The push fails.

C0’s destination (7,2) is floor.

C1’s destination (7,3) is a wall.

C2’s destination (7,4) is a wall.

The state is unchanged. Red impact bars appear above C1 and C2, not above C0.

This immediately distinguishes full-footprint blocking from a plausible “only the touched end matters” rule.

Verified route after the experiment
L U RRRR
| D R UUUU
| RRR U LLL
| UU RR DDDDD
| R D LLLLLLL
| U L D

Raw successful sequence:

LURRRRDRUUUURRRULLLUURRDDDDDRDLLLLLLLULD

Successful route depth: 40

Discovery route depth including the failed diagnostic U: 41

Phase detail

Align the train

L U RRRR

The worker gets west of the train.

Four east pushes move the train from columns 2..4 to 6..8.

Enter the isolated pushing pocket

D R UUUU

The worker enters the previously unreachable lower-right pocket.

Four north pushes move the train to row 4.

During the last two pushes, C2 propagates force into C3:

C3 (5,8) → (4,8) → (3,8)

No clasp appears because C2 contacts C3 with its side.

Park the three-cart train

RRR U LLL

The worker reaches the train’s eastern side.

Three west pushes place the train at (4,3..5), activating all three adjacent bays.

Return the purple cart downward

UU RR DDDDD

The worker reaches (2,8).

Five south pushes move C3 from (3,8) to (8,8).

Cross the old staging corridor

R D LLLLLLL

The worker gets east of C3.

Seven west pushes move it along row 8 to (8,1).

Final drop

U L D

The worker reaches (7,1).

C3 moves south onto (9,1).

Necessity and anti-cleanliness check

The worker begins in the left lower pocket.

The initial train blocks the only horizontal route into the right staging pocket.

The wall at (9,4) prevents walking around beneath the train.

C3 cannot be approached until the train has crossed the barrier and opened the upper room.

The train cannot ascend before shifting because two distant footprint cells are blocked.

C3 cannot travel directly across row 9 because (9,4) is a wall; it must cross on row 8 and then drop at column 1.

The train is first a liability, then a ram, then a delivered object.

C3 is first an obstruction, then displaced cargo, then a forced revisit.

Level 6 — Extension Route
Player-facing discovery

The green cart occupies a narrow left shaft. Its lateral neighbors are walls, so it can only move north. The blue-orange pair blocks the separate two-cell passage.

The green cart must therefore travel alone through the narrow route and be staged at the upper right. The worker then returns through a different lower route, pushes the pair upward, and joins the green cart only after all three are above their restrictive passages.

The completed three-cart train pushes two separate carts north in one action, using its broad front. It must then shift into a three-wide delivery chute. The rightmost chute column has no usable worker stance above it, so the green member cannot be delivered separately; the full three-cart train is required.

Floor mask
############
#####......#
#####......#
#####.......
#...........
##..##..#...
###.##..#...
###.#...#...
###.#...#...
#.......#...
#########...
############
Initial state
Worker: (9,4)

C0: B H→ at (5,6)
C1: O H→ at (5,7)
C2: G H→ at (8,3)
C3: P V↑ at (3,6)
C4: M V↑ at (3,8)

Initial locks:
C0--C1

Bays:
B: (10,9)
O: (10,10)
G: (10,11)
P: (2,6)
M: (2,8)

selected_id: null
cursor_state: null
Exact validator result
C0 must end at (10,9)
C1 must end at (10,10)
C2 must end at (10,11)
C3 must end at (2,6)
C4 must end at (2,8)
Verified route
L UUUU L U RRRRR
| LLLL DDDDD RRR UUU
| UU
| L U RRR
| L UU RR D
| DDDDDDD

Route depth: 47

Phase detail

Send the extension through the solo route

L UUUU L U RRRRR

The worker enters the shaft below C2.

Four north pushes move C2 from (8,3) to (4,3).

The worker reaches its west side.

Five east pushes stage C2 at (4,8).

The lower shaft’s neighboring cells are walls, so C2 cannot be shifted sideways to bypass this route.

Return to the pair

LLLL DDDDD RRR UUU

The worker retreats west to column 3.

It returns to row 9.

It crosses the lower chamber to column 6.

It climbs to (6,6), directly below C0.

This forced revisit means the extension and pair are prepared through different spatial branches rather than encountered in solution order.

Form the three-cart train and use its broad front

UU

First U:

C0 and C1 move from row 5 to row 4.

C1’s east hook meets C2’s west socket.

The train becomes C0--C1--C2 at columns 6..8.

Second U:

The three-cart train moves to row 3.

C0 propagates into C3 at (3,6), moving it to (2,6).

C2 propagates into C4 at (3,8), moving it to (2,8).

Both bays activate simultaneously.

Neither target couples because the contacts are against the train’s side.

Align with the delivery chute

L U RRR

The worker reaches (3,5).

Three east pushes move the train to (3,9..11).

Reach an accessible top stance

L UU RR D

The worker loops through the top shelf to (2,9).

This uses the gap between the two delivered carts.

Descend

DDDDDDD

Seven south pushes place the train on row 10.

Necessity and anti-cleanliness check

C2 blocks the one-cell shaft and cannot move laterally in its lower position.

C0–C1 block the alternative two-cell route.

The upper area is therefore inaccessible until one of the authored cart routes is used.

The pair cannot occupy the green bay at (10,11).

A solo C2 positioned at (3,11) could not be pushed south because (2,11) is a wall.

Joining C2 to the pair is therefore required to carry it down by pushing an accessible member at column 9.

The two target carts create a broad-front interaction: a single push has two physically separate consequences.

The resulting long train is useful during the broadside push but cumbersome during alignment and requires the top loop before descent.

Level 7 — Two Departures
Player-facing discovery

No new rule is introduced.

The worker begins trapped beneath the green extension cart. Pushing it north eventually joins the blue-orange pair. Because the worker remains trapped beneath the new train, the only productive next push moves the whole train north.

That push simultaneously displaces the purple and maroon carts. Those two carts will later become a second train.

The first three-cart train must be moved into the right delivery chute before the future two-cart train can descend: its initial row occupies the second train’s required destination cells. A wall prevents the purple-maroon pair from shifting left while still on row 2, so it must be lowered into the row just vacated by the first train.

The level therefore contains two coupled structures with a real dependency network:

The first train moves the future members of the second.

The second cannot be assembled and routed until the first vacates the center.

Each chute has an outer column that a solo cart cannot enter from a valid worker stance.

The trains must be built in the correct phases, but no operational rule changes.

Floor mask
############
#.........##
#...#......#
#...........
............
..###.###...
..###.###...
..###.###...
..###.###...
..###.###...
..#######...
############
Initial state
Worker: (9,5)

C0: G H→ at (8,5)
C1: B H→ at (4,6)
C2: O H→ at (4,7)
C3: P H→ at (3,5)
C4: M H→ at (3,7)

Initial locks:
C1--C2

Bays:
G: (10,9)
B: (10,10)
O: (10,11)
P: (10,0)
M: (10,1)

selected_id: null
cursor_state: null
Exact validator result
C0 must end at (10,9)
C1 must end at (10,10)
C2 must end at (10,11)
C3 must end at (10,0)
C4 must end at (10,1)
Verified route
UUUU
| U
| L U RRRR
| U R DDDDDDD
| UUUUUUU L
| L
| U LL DD
| RR D
| LLLLL
| U L DDDDDD

Route depth: 50

Phase detail

Construct the first train

UUUU

C0 moves from (8,5) to (4,5).

Its east hook meets C1’s west socket.

The first train becomes C0--C1--C2 at (4,5..7).

The worker remains at (5,5). The shaft walls leave no route around the train.

Move the future second train

U

The first train moves north to (3,5..7).

Its left member pushes C3 from (3,5) to (2,5).

Its right member pushes C4 from (3,7) to (2,7).

C3 and C4 remain separate because there is a gap at (2,6).

Shift the first train into the right chute

L U RRRR

The worker reaches (3,4).

Four east pushes move the train to (3,9..11).

Deliver the first train

U R DDDDDDD

The worker reaches (2,9).

Seven south pushes place the train at (10,9..11).

The outer member at column 11 is carried by the train; there is no floor at (2,11) from which it could have been sent down alone.

Return to the future second train

UUUUUUU L

The worker climbs the right chute to (2,9).

It moves left to (2,8), immediately east of C4.

Create the second train

L

C4 moves from (2,7) to (2,6).

C3’s east hook meets C4’s west socket.

The second train becomes C3--C4 at (2,5..6).

Lower it into the vacated center

U LL DD

The worker reaches (1,5).

Two south pushes move the pair:

row 2 → 3

row 3 → 4

This lowering was impossible while the first train occupied row 3.

It is also necessary because (2,4) is a wall; the pair cannot shift left while remaining on row 2.

Get to the eastern side

RR D

The worker reaches (4,7).

Align with the left chute

LLLLL

Five west pushes move the pair from columns 5,6 to columns 0,1.

Deliver the second train

U L DDDDDD

The worker reaches (3,1), above the pair’s right member.

Six south pushes place the pair at (10,0..1).

Necessity and anti-cleanliness check

The worker cannot leave the initial shaft without moving C0 into the first train.

Once C0 joins, the shaft still has no side exit; the broadside push into C3 and C4 is forced.

C3 and C4 cannot descend while the first train occupies (3,5..7).

C3 and C4 cannot shift left on row 2 because (2,4) is a wall.

The leftmost delivery column cannot be solved by a solo cart:

(3,0) is a wall, so a cart at (4,0) has no direct worker stance above it.

Column 2 is sealed below row 4, so a cart cannot be sent down column 1 and later shifted into column 0.

The rightmost delivery column similarly requires the first rigid train because (2,11) is a wall.

The purple and maroon carts appear before the first train is finished, but their eventual meaning is not “deliver these now.” They are first moved as loose obstacles, then become a train after the central space changes.

The two train routes use opposite sides of the warehouse and different widths, preventing the level from reducing to a single corridor or repeated recipe.

9. Implementation conformance checks

An implementation should satisfy these deterministic checkpoints.

Coupler contrast

After Level 2’s first R:

C0 = (6,4)
C1 = (6,5)
locks = empty

After the joining push in Level 3, the prefix UURR produces:

C0 = (6,5)
C1 = (6,6)
locks = {C0--C1}
Far-end propagation

After the first 13 actions of Level 4:

C0 = (3,7)
C1 = (3,8)
C2 = (2,8)
locks = {C0--C1}

C2 must not be locked to C1.

Full-footprint failure

Level 5’s initial U must:

leave worker and every cart unchanged
flash north edge of C1 against (7,3)
flash north edge of C2 against (7,4)

The near cart C0 has floor at (7,2) and therefore does not receive an obstruction flash.

Multi-target broadside propagation

After the first 29 actions of Level 6:

C0, C1, C2 = (3,6), (3,7), (3,8)
C3 = (2,6)
C4 = (2,8)
locks = {C0--C1, C1--C2}

Neither C3 nor C4 may be included in that train.

Two final components

At the end of Level 7:

positions:
C0 = (10,9)
C1 = (10,10)
C2 = (10,11)
C3 = (10,0)
C4 = (10,1)

locks:
C0--C1
C1--C2
C3--C4

won = true

These locks are consequences of the intended route, but the generic validator still checks only cargo-to-bay occupancy.
