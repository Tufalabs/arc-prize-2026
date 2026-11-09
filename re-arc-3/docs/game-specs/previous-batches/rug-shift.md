Rug Shift

Identity:

This is a game about moving a rug becoming difficult because blocked cargo stops while the rug keeps changing what it supports.

Authored scope: exactly 7 levels.

The player directly moves one large patterned rug. Parcels move only when the rug fully supports them at the beginning of an action. If a parcel would hit a wall or another stationary parcel, the parcel remains where it is while the thin rug continues sliding beneath or away from it. The puzzle is therefore not conventional pushing: the player controls the relative position of rug and cargo by deliberately arranging moments when they move together and moments when they separate.

The only gameplay objects are:

the rug;

colored parcels;

matching colored docks;

raised stop-walls that block parcels but not the rug.

There are no switches, keys, counters, doors, hidden states, timers, or clickable devices.

1. Player actions

The action space is exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any rendered cell

Only the four directional actions affect gameplay.

A directional action attempts to translate the entire rug by exactly one logical cell. It never directly targets a parcel.

SPACE and all clicks are inert no-ops. They call step(), increment the internal step counter, and leave the game state unchanged. No object or region is clickable, no selection highlight is rendered, and:

selected_object = null
cursor_state = null

in every level.

There is no energy, timer, hard step limit, or failure state. Exploration is unrestricted.

2. Objective and validator

Every parcel has exactly one dock of the same color and shape.

A level succeeds after a step() when, for every parcel i:

parcel_i.world_footprint == dock_i.world_footprint

The following are deliberately ignored by the validator:

whether the rug is still beneath a delivered parcel;

the rug’s final position;

parcel support status;

the order in which docks were reached;

the step count;

parcels passing over nonmatching docks earlier in the solution.

A partial overlap with a dock does not count. The parcel footprint must match the dock footprint exactly.

There are no extra docks and no two parcels share a color. All seven levels start unsolved.

3. Logical geometry
Board

Every level uses the same logical board:

width  = 12 cells
height = 12 cells
x ∈ {0, …, 11}
y ∈ {0, …, 11}

(0,0) is the upper-left cell. Positive x is rightward and positive y is downward.

All 144 cells are floor. There are no internal grid borders or grid lines.

Rectangle notation
Rect(x, y, w, h)

means the set:

{(x+i, y+j) | 0 ≤ i < w and 0 ≤ j < h}

Parcel coordinates always identify the upper-left cell of their rectangular footprint.

Rug shapes use local coordinates. A rug with local footprint S and origin (rx,ry) occupies:

{(rx+sx, ry+sy) | (sx,sy) ∈ S}

The rug never rotates.

4. Rug templates
STEP rug

Used in Levels 4 and 5.

..##
####
##..

Exact local footprint:

STEP = {
    (2,0), (3,0),
    (0,1), (1,1), (2,1), (3,1),
    (0,2), (1,2)
}

A 2×2 parcel is fully supported only at local parcel origins:

(0,1)  lower-left bay
(2,0)  upper-right bay

There are no other 2×2 support placements in this shape.

POCKET rug

Used in Level 6.

...##
#####
##...
##...

Exact local footprint:

POCKET = {
    (3,0), (4,0),
    (0,1), (1,1), (2,1), (3,1), (4,1),
    (0,2), (1,2),
    (0,3), (1,3)
}

A 2×3 parcel is fully supported only at local origin:

(0,1)

A 2×2 parcel is fully supported at:

(0,1), (0,2), or (3,0)
CROWN rug

Used in Level 7.

..##..
.####.
##..##
##..##

Exact local footprint:

CROWN = {
    (2,0), (3,0),
    (1,1), (2,1), (3,1), (4,1),
    (0,2), (1,2), (4,2), (5,2),
    (0,3), (1,3), (4,3), (5,3)
}

A 2×2 parcel is fully supported only at:

A = (0,2)  lower-left bay
B = (2,0)  upper-middle bay
C = (4,2)  lower-right bay
5. Exact state machine
State
GameState:
    level_index: int in [0,6]
    rug_origin: (int x, int y)
    fixed_rug_shape: set of local cells
    parcels:
        id
        color_id
        width
        height
        origin
    walls: set of absolute cells
    docks:
        parcel_id
        color_id
        footprint
    step_count: nonnegative int
    complete: bool
    selected_object: null
    cursor_state: null

There is no other logical state. In particular, parcels have no hidden carried/unloaded flag. Carrying is recomputed from geometry at each action.

Support

For parcel i with current footprint P_i and current rug footprint R:

supported(i) ⇔ P_i ⊆ R

Support is all-or-nothing.

A parcel with three of four cells on the rug is not carried. A 2×3 parcel with five of six cells on the rug is not carried.

Support is tested at the beginning of the directional action. A rug movement may finish underneath a stationary parcel and make it fully supported, but that parcel does not move until the next action.

Wall layers

Walls are raised parcel stoppers:

a parcel may not occupy a wall cell;

the rug may occupy and pass through wall cells;

docks and the rug may overlap;

the rug may move beneath stationary parcels;

walls are rendered above the rug.

This layer distinction is first demonstrated in Level 2.

Directional update order

For a direction vector d, execute the following exact order.

1. Validate the rug translation

Compute:

R' = R translated by d

If any cell of R' lies outside the 12×12 board:

the entire action is invalid;

the rug does not move;

no parcel movement is considered;

step_count still increments;

completion is rechecked but cannot newly change.

Walls, docks, and parcels do not invalidate a rug translation.

2. Find initially eligible parcels

A parcel is initially eligible if it is fully supported in the pre-action state:

E = {i | P_i ⊆ R}
3. Remove wall-blocked parcels

For every i ∈ E, calculate:

destination_i = P_i translated by d

Remove i if its destination:

intersects any wall cell; or

lies outside the board.

The latter cannot normally occur when a fully supported parcel follows a valid rug move, but it is specified for completeness.

Call the remaining set M₀.

4. Resolve parcel-on-parcel blocking

Parcels are solid. A moving group may travel as a convoy, but a parcel cannot move into a parcel that will remain stationary.

Repeatedly apply:

M_(k+1) = {
    i ∈ M_k |
    destination_i does not intersect the current footprint
    of any parcel j ∉ M_k
}

Stop when the set no longer changes.

Because all remaining parcels translate by the same vector, parcels remaining in the movable set preserve their relative positions and cannot collide with one another.

This rule produces two important cases:

Two adjacent, supported parcels can move together.

If a front parcel is removed because it hits a wall, a rear parcel is also removed if its destination would overlap the now-stationary front parcel.

The second case is visibly demonstrated in Level 6 and reused in Level 7.

5. Commit once

Atomically commit:

rug_origin += d

for each parcel i in final movable set M:
    parcel_i.origin += d

all other parcels:
    unchanged

Support is then recomputed only for the next action.

6. Validate success

After committing movement, check exact parcel-to-dock equality. If all parcels match:

complete = true

Further inputs are ignored until the environment wrapper advances or resets the level.

Pseudocode
step(action):
    step_count += 1

    if complete:
        return unchanged_state

    if action is SPACE or click:
        validate_win()
        return unchanged_state

    d = direction_vector(action)
    proposed_rug = translate(rug, d)

    if proposed_rug is not wholly inside board:
        play_boundary_feedback()
        validate_win()
        return unchanged_state

    movable = {
        i for each parcel
        if footprint(parcel_i) is a subset of current_rug
        and translate(footprint(parcel_i), d) avoids walls
        and translate(footprint(parcel_i), d) is inside board
    }

    repeat:
        old_movable = movable
        for i in old_movable:
            if destination(i) overlaps any parcel j not in movable:
                remove i from movable
    until movable == old_movable

    old_state = copy(state)

    rug_origin += d
    for i in movable:
        parcel_i.origin += d

    validate_win()
    render_transition(old_state, state, movable)
6. Rendering contract
Canvas mapping

The render canvas is 64×64 pixels.

The 12×12 board occupies a 48×48 region with pixel origin (8,8).

Logical cell (x,y) maps to:

pixel x: 8 + 4x through 11 + 4x
pixel y: 8 + 4y through 11 + 4y

The surrounding eight-pixel margin is white.

No logical cell borders are rendered. The board is one contiguous light-gray region ending directly against the white margin.

Palette use
0   white       outer background
1   light gray  floor and unsupported parcel windows
2   gray        shadows
3   dark gray   parcel tape and wall side faces
4   very dark   wall top faces
5   black       parcel outlines
6   magenta     Level 7 magenta parcel and dock
8   red         red parcels and docks
9   blue        dark rug pattern
10  light blue  light rug pattern
11  yellow      brief collision and completion feedback
12  orange      Level 6 long parcel and dock
14  green       green parcels and docks
15  purple      Level 6 small parcel and dock

Colors 7 and 13 are unused.

Rug

Each occupied rug cell is filled with color 9 or 10 according to the parity of its fixed local coordinate:

color = 9 if (local_x + local_y) is even
color = 10 otherwise

The pattern is anchored to the rug, not the board, so it visibly travels with the rug. No outlines separate rug cells.

Parcels and visible support

A parcel is a continuous rectangular crate:

one-pixel color-5 exterior outline;

interior filled with its parcel color;

a color-3 central tape band.

Every logical cell in the parcel footprint contains a centered 2×2 support window. The support window shows the layer beneath that logical cell:

rug color 9 or 10 if the rug occupies the cell;

floor color 1 otherwise.

Thus a 2×2 parcel visibly presents four support windows. All four must show rug colors for the parcel to move. This marking communicates actual gameplay state rather than decorative information.

A 2×3 parcel has six corresponding windows.

Docks

A dock has the same logical footprint and color as its parcel. It is rendered as four thick corner brackets around its footprint rather than as a solid block.

Dock brackets are rendered above floor and rug and remain visible when partially overlapped by a parcel. They do not collide with anything.

This gives Level 1 a visible target silhouette and makes partial versus exact alignment immediately legible.

Walls

Walls are contiguous dark shapes:

color 4 top face;

color 3 one-pixel lower and right side face;

no symbol or text.

Walls render above the rug. During a slip, the rug pattern visibly enters beneath the wall while the wall remains fixed.

7. Animation and feedback

All logical state changes happen exactly once inside step(). Animation is an observation-side interpolation between the saved old state and the already committed new state.

Valid directional movement

A valid one-cell rug translation is displayed over four noninteractive subframes, one rendered pixel per subframe.

The rug interpolates four pixels.

Parcels in the final movable set interpolate with it.

Blocked and unsupported parcels remain fixed.

Walls remain fixed and render above the moving rug.

Inputs received during subframes are ignored rather than queued.

Subframes consume no steps.

This animation is especially required when a parcel remains fixed: the player must see the patterned rug physically slide beneath or away from it.

Collision feedback

When a supported parcel is removed from the movable set:

a wall collision flashes the parcel edge touching the wall in color 11 for the first subframe;

a parcel collision flashes the touching edges of both parcels in color 11 for the first subframe.

The flash does not alter logical state.

Invalid board-edge movement

The rug visually offsets one pixel toward the board edge and returns over two subframes. No logical coordinate changes.

Completion

On completion, all correctly matched dock brackets alternate between their parcel color and color 11 for four subframes. There is no failure or danger animation because the game has neither failure nor danger states.

8. Curriculum overview
Level	Main discovery or composition	Verified example depth
1	Rug and fully supported parcel travel together; docks require exact alignment	2
2	A wall stops cargo but not the rug, changing parcel offset	4
3	Partial support does not carry; rug can slide underneath to pick up	7
4	An irregular rug has discrete support bays; a wall corner enables a bay handoff	15
5	One wall can split multiple cargo, park one parcel, and force a revisit	23
6	Different parcel shapes compete for support; parcels can block one another	31
7	Three occupied bays, parcel-mediated stops, route constraints, and an outer-bay exchange	40
9. Authored levels
Level 1 — Ride the Rug
Player-facing discovery

A red 2×2 parcel sits on a blue patterned rug. A hollow red 2×2 dock silhouette is directly to its right.

Pressing RIGHT moves both the rug pattern and the parcel. After one action the parcel overlaps only one dock column, visibly showing that contact is insufficient. A second RIGHT produces an exact match and immediate completion feedback.

This isolates two facts:

Arrow keys control the rug, not an avatar or cursor.

A fully supported parcel moves with the rug.

Initial state
board: 12×12

rug:
    shape = Rect(0,0,3,2)
    origin = (2,5)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (3,5)
        dock = Rect(5,5,2,2)

walls: none

selected_object: null
cursor_state: null

Initial footprints:

rug    = Rect(2,5,3,2)   # x=2..4, y=5..6
parcel = Rect(3,5,2,2)   # x=3..4, y=5..6
dock   = Rect(5,5,2,2)   # x=5..6, y=5..6
Verified solution
RR

Depth: 2 meaningful actions.

State progression:

start:   rug (2,5), parcel (3,5)
R:       rug (3,5), parcel (4,5)   # partial dock overlap
R:       rug (4,5), parcel (5,5)   # exact dock match
Attribution check

There are no walls, extra parcels, misleading shapes, or simultaneous state changes. The rug pattern and parcel translate by the same amount, making the causal relationship visible. The dock is a hollow same-color target silhouette, so the objective does not depend on written instructions.

Level 2 — Slip Against the Wall
Player-facing discovery

The rug begins flush against the board’s right edge. The dock lies two cells farther right than the parcel, but RIGHT cannot move the rug.

A dark wall touches the parcel’s left side. Moving LEFT is the nearby productive experiment:

the rug enters the wall cells;

the parcel flashes at the wall and remains fixed;

the support windows reveal that its offset on the rug has changed.

Repeating the slip moves the parcel from the rug’s left end to its right end. The rug can then move right twice and reach the dock.

The wall is not a conventional obstacle for the controlled object. It is a brake for cargo.

Initial state
rug:
    shape = Rect(0,0,4,2)
    origin = (8,5)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (8,5)
        dock = Rect(10,5,2,2)

walls:
    Rect(7,5,1,2)

selected_object: null
cursor_state: null

Initial geometry:

wall   = x=7,    y=5..6
parcel = x=8..9, y=5..6
rug    = x=8..11,y=5..6
dock   = x=10..11,y=5..6
Verified solution
LLRR

Depth: 4 meaningful actions.

start: rug (8,5), parcel (8,5), local parcel offset (0,0)

L:     rug (7,5), parcel (8,5)
       parcel was wall-blocked
       local offset becomes (1,0)

L:     rug (6,5), parcel (8,5)
       parcel was wall-blocked again
       local offset becomes (2,0)

R:     rug (7,5), parcel (9,5)
R:     rug (8,5), parcel (10,5), complete

A blind player may first try RIGHT. That action gives the board-edge bump but changes no state, creating the intended local impasse before the wall experiment.

Attribution check

Only one causal unknown has been added: walls block parcels but not the rug. The wall is directly adjacent to the parcel, and the first useful interaction produces immediate motion of the rug alongside visible nonmotion of the parcel.

Two slips are necessary. With zero slips the parcel cannot move farther right at all; with one slip its maximum origin is (9,5), still one cell short of the dock.

Level 3 — Get All Four Corners Under
Player-facing discovery

The parcel begins with only its left column supported: two support windows show blue rug and two show gray floor.

The dock is above and to the right. Moving the rug toward the right is a distinguishing experiment. The parcel does not move, disproving the plausible rule “any overlap carries.” Instead, the rug slides under the parcel and all four support windows become blue.

Only on the following action does the parcel begin travelling with the rug.

Initial state
rug:
    shape = Rect(0,0,3,3)
    origin = (2,5)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (4,5)
        dock = Rect(8,3,2,2)

walls: none

selected_object: null
cursor_state: null

Initial support:

rug    = x=2..4, y=5..7
parcel = x=4..5, y=5..6

supported cells = {(4,5), (4,6)}
unsupported cells = {(5,5), (5,6)}
Verified solution
RUURRRR

Depth: 7 meaningful actions.

start: rug (2,5), parcel (4,5), partial support

R:     rug (3,5), parcel (4,5), now fully supported
UU:    rug (3,3), parcel (4,3)
RRRR:  rug (7,3), parcel (8,3), complete
Attribution check

There are no walls or additional parcels. The first RIGHT action changes only one relevant relationship: the amount of support beneath the stationary parcel. The result cannot be explained by the Level 1 transport rule or Level 2 wall rule.

Level 4 — Change Bays at the Corner
Player-facing discovery

The first irregular rug has two visibly complete 2×2 support bays connected by a narrow middle band.

The parcel begins in the lower-left bay. Moving downward initially works. On the third DOWN, the horizontal wall arm catches the parcel while the rug slips one cell downward. The parcel is now over the rug’s missing notch and becomes unsupported.

The nearby irregular rug silhouette suggests sliding it sideways. Two LEFT actions place the upper-right bay under the stationary parcel. The same parcel is now carried from a different local position, allowing it to reach the far-right dock.

Initial state
rug:
    shape = STEP
    origin = (4,2)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (4,3)
        dock = Rect(10,9,2,2)

walls:
    Rect(3,3,1,4)   # vertical arm:   x=3, y=3..6
    Rect(3,7,4,1)   # horizontal arm: x=3..6, y=7

selected_object: null
cursor_state: null

The parcel starts at local rug origin (0,1), the lower-left support bay.

Verified solution
DDDLLRRRDDDDRRR

Depth: 15 meaningful actions.

Phases and key states:

start:
    rug (4,2)
    parcel (4,3) in lower-left bay

DDD:
    rug (4,5)
    parcel (4,5)
    third D is wall-blocked; parcel is now unsupported

LL:
    rug (2,5)
    parcel (4,5)
    parcel is now in the upper-right bay at local offset (2,0)

RRR:
    rug (5,5)
    parcel (7,5)

DDDD:
    rug (5,9)
    parcel (7,9)

RRR:
    rug (8,9)
    parcel (10,9), complete
Anti-cleanliness and necessity

The dock at x=10 cannot be reached while the parcel remains in the lower-left bay: the STEP rug’s maximum origin is x=8, so that bay can carry a parcel only as far as x=8. The upper-right bay is therefore mandatory.

The horizontal wall arm creates the visible handoff moment. The vertical arm prevents a direct early left-carry shortcut and makes the wall an actual corner rather than a single-purpose brake.

The later route is not a corridor: the player must first change the parcel’s local support bay, then choose an order that clears the wall arm before moving toward the lower-right dock.

Level 5 — Split the Load
Player-facing discovery

Red and green parcels occupy the two support bays of the STEP rug. They initially move together.

The first route toward the green dock leads across one horizontal wall:

the red parcel reaches the wall’s vertical range first and remains behind;

the green parcel continues moving;

the two cargo paths visibly separate despite the same rug action.

The same wall later catches the green parcel on its dock while the rug leaves. The rug must then return beneath the abandoned red parcel and carry it in the opposite direction.

The wall’s operational rule never changes. Its role changes from splitter to parking brake because of parcel position and movement direction.

Initial state
rug:
    shape = STEP
    origin = (4,4)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (4,5)
        dock = Rect(8,10,2,2)

    green:
        color = 14
        shape = 2×2
        origin = (6,4)
        dock = Rect(2,1,2,2)

walls:
    Rect(2,3,2,1)   # x=2..3, y=3

selected_object: null
cursor_state: null

Red starts in local bay (0,1). Green starts in local bay (2,0).

Verified solution
ULUULLLDDRRRDDDDDDRRRRR

Depth: 23 meaningful actions.

Phase 1: split the parcels
ULUU

Result after action 4:

rug   = (3,1)
red   = (3,4)
green = (5,1)

On the third action, UP, red’s destination intersects wall cell (3,3), so red remains at (3,4) while green moves upward.

Phase 2: deliver and park green
LLL

Green reaches its dock:

rug   = (0,1)
red   = (3,4)
green = (2,1)

Then:

DD

The first DOWN would move green into the same horizontal wall, so green stays on its dock while the rug moves away. The second DOWN continues with green unsupported.

Phase 3: return beneath red
RRR

Result:

rug = (3,3)
red = (3,4)

The final RIGHT places red in the rug’s lower-left support bay. Red does not move until the following action.

Phase 4: deliver red
DDDDDDRRRRR

Final state:

rug   = (8,9)
red   = (8,10)
green = (2,1)
Anti-cleanliness and necessity

The required displacements are incompatible:

red:   (4,5) → (8,10) = (+4,+5)
green: (6,4) → (2,1)  = (-4,-3)

They cannot remain a fixed convoy. The wall is the only authored structure that selectively separates them.

The solution contains a forced revisit: after green is parked, the rug is far from red and must be routed back beneath it. The objects are not encountered in final-use order, and the level cannot be reduced to visiting two docks sequentially.

Level 6 — Long Crate, Small Pocket
Player-facing discovery

The POCKET rug contains:

a lower-left area large enough for the orange 2×3 parcel;

a small upper-right 2×2 bay.

The first obvious attempt is to move right toward the purple dock.

On the first RIGHT:

the small purple parcel hits the nearby wall and stays;

the long orange parcel moves one cell.

On the second RIGHT:

purple remains stationary and unsupported;

orange’s destination overlaps purple’s current footprint;

orange also stays while the rug moves.

This is the first explicit parcel-on-parcel blocking demonstration. The immediate contact flash distinguishes it from another wall collision.

A short detour places only the small parcel in the upper-right bay. It can then be delivered and parked before the rug returns for the long parcel.

Initial state
rug:
    shape = POCKET
    origin = (1,4)

parcels:
    orange:
        color = 12
        shape = 2×3
        origin = (1,5)
        dock = Rect(0,9,2,3)

    purple:
        color = 15
        shape = 2×2
        origin = (4,4)
        dock = Rect(10,0,2,2)

walls:
    Rect(6,4,1,2)    # first purple brake
    Rect(10,2,2,1)   # brake below purple dock

selected_object: null
cursor_state: null

Initially:

orange occupies the only 2×3 bay at local origin (0,1)
purple occupies the small bay at local origin (3,0)
Verified solution
RRULLDUUUURRRRRRDLLLLLDDDLLDDDD

Depth: 31 meaningful actions.

Phase 1: expose parcel collision
RR

After the two actions:

rug    = (3,4)
orange = (2,5)
purple = (4,4)

Action details:

first R:
    purple is wall-blocked
    orange moves from (1,5) to (2,5)

second R:
    orange proposes (3,5)
    orange's 2×3 destination overlaps purple at row y=5
    orange is parcel-blocked
    rug alone moves to (3,4)
Phase 2: selectively acquire purple
ULLD

Result:

rug    = (1,4)
orange = (2,5), not fully supported
purple = (4,4), fully supported in local bay (3,0)

The detour is necessary because moving directly left through rug origin (2,4) would momentarily put orange in its only valid 2×3 bay and carry it back with the rug.

Phase 3: deliver purple
UUUURRRRRR

Result:

rug    = (7,0)
orange = (2,5)
purple = (10,0), on dock

Then:

D

Purple’s proposed destination intersects the wall at y=2, so it remains on its dock while the rug leaves.

Phase 4: retrieve orange
LLLLLDDD

Result:

rug    = (2,4)
orange = (2,5), fully supported at local origin (0,1)
purple = (10,0)
Phase 5: deliver orange
LLDDDD

Final state:

rug    = (0,8)
orange = (0,9)
purple = (10,0)
Anti-cleanliness and necessity

The long parcel cannot occupy the small upper-right support bay. The small parcel can use that bay, but the orange parcel’s shifted position after the opening collision prevents the rug from carrying both during the pickup detour.

The two target displacements are in opposite directions, and the extreme purple dock is reachable only while purple uses local bay (3,0). The dock wall is necessary to preserve purple while the rug crosses the board to retrieve orange.

Parcel collision is not a special-case interaction: it follows the same stationary-obstacle resolution used globally and is reused as a planning tool in Level 7.

Level 7 — Exchange the Outer Bays
Player-facing discovery

The CROWN rug has three discrete 2×2 support bays, all initially occupied:

red     in left bay A
green   in middle bay B
magenta in right bay C

Their docks imply different reach requirements:

green’s top-right dock is reachable only from middle bay B;

magenta’s bottom-left dock is reachable only from left bay A;

red’s bottom-right dock is reachable only from right bay C.

Green begins in the correct kind of bay. Red and magenta must effectively exchange outer-bay roles without ever rotating the rug or parcels.

The opening experiment is to move right:

Magenta reaches the upper-right wall and stops.

Red continues until magenta itself blocks red.

Green reaches its dock.

A downward action parks green against the same wall while the rug leaves.

The rest of the puzzle uses parcels as temporary walls, an offset lower obstruction, a coupled descent, and a final selective unload.

Initial state
rug:
    shape = CROWN
    origin = (2,0)

parcels:
    red:
        color = 8
        shape = 2×2
        origin = (2,2)
        dock = Rect(10,10,2,2)

    green:
        color = 14
        shape = 2×2
        origin = (4,0)
        dock = Rect(8,0,2,2)

    magenta:
        color = 6
        shape = 2×2
        origin = (6,2)
        dock = Rect(0,10,2,2)

walls:
    Rect(9,2,1,2)   # upper-right brake
    Rect(2,6,2,2)   # offset lower block
    Rect(0,9,2,1)   # lip above magenta dock

selected_object: null
cursor_state: null

Initial local placements:

red     = A = (0,2)
green   = B = (2,0)
magenta = C = (4,2)
Verified solution
RRRRDLDDDLLUURRULLUDDRDDDDLDDLLLURDRRRRR

Depth: 40 meaningful actions.

The path can be read as:

RRRRD | LDDDLL | UURRULLU | DDRDDDDLDD | LLL | URD | RRRRR
Phase 1 — Deliver green and separate the outer parcels
RRRRD

State after action 5:

rug     = (6,1)
red     = (5,2)
green   = (8,0), on dock
magenta = (7,2)

Causal sequence:

On the second RIGHT, magenta’s destination intersects wall column x=9; magenta stays.

Red keeps moving.

On the fourth RIGHT, red’s proposed destination overlaps stationary magenta, so red stays.

Green reaches (8,0).

DOWN would move green into the same upper-right wall, so green remains on its dock while the rug leaves.

The wall has already served as both a splitter and a dock brake.

Phase 2 — Acquire magenta in the middle bay
LDDDLL

State after action 11:

rug     = (3,4)
red     = (5,2)
green   = (8,0)
magenta = (5,4)

The rug first moves below magenta, acquires it in middle bay B, then carries it downward and left. Red remains stationary.

Phase 3 — Use parcels as temporary blockers
UURRULLU

State after action 19:

rug     = (3,0)
red     = (7,2), in right bay C
green   = (8,0), on dock
magenta = (3,2), in left bay A

Important interactions:

The first UP would move magenta into red, so red acts as a temporary wall and magenta stays while the rug slips.

After two RIGHT actions, red is carried toward the right bay.

The next UP would move red into delivered green, so green now acts as a temporary wall. Red stays while magenta moves.

LLU finishes placing magenta in A while red occupies C.

No parcel has changed shape or acquired a special state. Their roles changed solely through position.

Phase 4 — Coupled descent around the lower block
DDRDDDDLDD

State after action 29:

rug     = (3,8)
red     = (7,10)
green   = (8,0)
magenta = (3,10)

Red and magenta now move as a two-parcel convoy in the outer bays.

The rug first descends, shifts right, and then continues downward while remaining to the right of the 2×2 lower wall block. It returns left only after both parcels have cleared the block’s vertical range.

The lower wall is structurally necessary even though the verified route avoids direct contact with it: without this block, the paired cargo can descend two columns earlier and a 38-action shortcut bypasses the intended route dependency.

Phase 5 — Align magenta
LLL

State after action 32:

rug     = (0,8)
red     = (4,10)
green   = (8,0)
magenta = (0,10), on dock

Both outer parcels travel left together until magenta reaches its dock. Red remains supported in the opposite outer bay.

Phase 6 — Preserve magenta and stage red
URD

State after action 35:

rug     = (1,8)
red     = (5,10)
green   = (8,0)
magenta = (0,10), on dock

On UP, magenta’s destination intersects the wall lip at y=9, so it remains delivered. Red moves upward with the rug. RIGHT and DOWN place red back on the bottom row without reacquiring magenta.

Phase 7 — Deliver red
RRRRR

Final state:

rug     = (6,8)
red     = (10,10)
green   = (8,0)
magenta = (0,10)
complete = true
Anti-cleanliness and necessity

This level forms a dependency network rather than a chain:

green must be delivered early so the upper wall can release the rug;

stationary magenta blocks red during the opening;

stationary red later blocks magenta;

delivered green later blocks red;

red and magenta must then be jointly carried;

the lower wall constrains where that joint descent can occur;

magenta must be selectively stopped at its dock while red remains carried.

The three docks are not an attribute checklist. Their extreme positions make different rug bays geometrically necessary. Removing spatial layout or replacing the board with three independent counters would destroy the puzzle.

10. Implementation invariants

An implementation is conformant only if all of the following hold:

The game contains exactly seven authored levels.

Every directional action translates the rug by one logical cell or does nothing at the board boundary.

Walls never block the rug.

Support is tested before movement and requires the entire parcel footprint.

Unsupported parcels never move automatically.

A newly supported parcel waits until the next action.

Wall-blocked parcels remain fixed while the rug still moves.

Parcel collisions use the stable movable-set procedure, allowing convoys but preventing overlap with stationary parcels.

No parcel, wall, or dock changes shape or orientation.

Docks never block movement.

Completion requires exact same-color footprint equality and ignores rug support.

SPACE and every click are no-ops.

There is no cursor selection state, timer, energy system, failure state, hidden information, randomness, or procedural level variation.

Animation never advances logical state and never consumes actions.

The internal logical grid has no rendered cell borders.
