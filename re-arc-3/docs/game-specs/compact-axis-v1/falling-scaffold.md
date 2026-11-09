Falling Scaffold

One-sentence identity

This is a game about placing and removing supports becoming difficult because gravity settles the entire structure after every action.

Authored content: exactly 7 levels.

Falling Scaffold is a side-view construction puzzle in which every brace is simultaneously architecture, a climbing surface, and a one-slot inventory resource. Loose slabs never move sideways and can never be lifted again, so removing one support is a visible commitment: the resulting slab may become a bridge, a stair, a ceiling, or a permanent obstruction. Nothing important is hidden. At every action boundary, the settled frame shows every slab, brace, socket, route, carried object, and remaining approach.

No new object category is introduced after Level 3. Later depth comes entirely from support height, approach direction, staged falls, one-edge support, slab-on-slab support, route sacrifice, and reuse of the same brace in a different location.

Progression
Level	Name	New understanding or composition	Verified solution depth
1	First Rung	Exposed braces can be entered and climbed	2
2	Missing Rung	A carried brace can be placed in an outlined socket	3
3	Drop Bridge	Removing a loaded brace releases its slab; gravity resolves immediately	8
4	Chosen Height	One brace under one edge determines the resting height of an entire rigid slab	13
5	Second Landing First	A slab can be lowered in stages; its second catch must be prepared before the first bridge is sacrificed	25
6	Seal, Then Climb	A useful underpass must be permanently sealed; one slab can support a larger overhanging slab	26
7	One Pull, Three Steps	Three staged slabs, two relocated braces, forced revisits, hand-capacity pressure, and a final staircase cascade	51

L, R, U, D, and S in replay strings mean LEFT, RIGHT, UP, DOWN, and SPACE.

1. Native geometry and rendering
Logical coordinates

Coordinates are integer cells:

x increases from left to right.

y increases from bottom to top.

(0,0) is the lower-left active cell.

A player coordinate denotes the cell occupied by the climber.

A slab (x,y,w) occupies (x,y) through (x+w-1,y).

All coordinate ranges below are inclusive.

Notation:

row(y, a..b) = {(x,y) | a ≤ x ≤ b}

col(x, a..b) = {(x,y) | a ≤ y ≤ b}

rect(a..b, c..d) = {(x,y) | a ≤ x ≤ b and c ≤ y ≤ d}

Each level has an explicit active width W and height H. Cells outside 0 ≤ x < W, 0 ≤ y < H are out of bounds, rendered as background, and cannot be entered.

64×64 mapping

Each logical cell is rendered as a 4×4 pixel module.

For a level with active size W×H:

origin_x = floor((64 - 4W) / 2)
origin_y = floor((64 - 4H) / 2)

Logical cell (x,y) occupies:

pixel_x = origin_x + 4x ... origin_x + 4x + 3
pixel_y = origin_y + 4(H - 1 - y) ... origin_y + 4(H - 1 - y) + 3

The active region has no enclosing border and no internal grid lines. Negative space is color 0.

Levels	Active size	Pixel origin
1–2	8×5	(16,22)
3	14×6	(4,20)
4	14×8	(4,16)
5	14×10	(4,12)
6	14×9	(4,14)
7	15×14	(2,4)

No text, digits, arrows, or labels are rendered in the grid.

2. Visual vocabulary

Only integer color IDs 0–15 are used.

Entity	Rendering
Air and exterior	0
Fixed terrain	Main fill 4; exposed upper pixel edge 3. Orthogonally adjacent terrain cells merge without seams.
Loose slab	A connected horizontal shape, normally 2 cells wide and wider where explicitly authored. Main fill 12; bottom pixel row and short end caps 13. No seam between the logical cells of one slab.
Exposed brace	Open lattice using 9 for its outer rails and 10 for its diagonal interior. Air holes remain 0.
Loaded brace	Same lattice, but its top rail is 11. This makes “the brace currently bearing weight” visible without a legend.
Empty brace socket	A thin local 2 outline inside its one-cell module. Only empty sockets show the outline; occupied sockets are visually replaced by the brace.
Climber	Connected body in 6, with a conspicuous facing nub in 7 on its left or right side.
Carried brace	A small 9/10 lattice pack attached to the climber’s back, opposite the facing nub. It has no separate collision footprint.
Exit hatch	Outer frame 14, inner opening 5. The green frame remains visible when the player occupies the hatch.
Invalid action	The blocked object or target cell flashes 8.
Crushing/falling failure	The player and colliding slab flash broadly in 8, then the attempt terminates unsuccessfully.
Completion	Hatch and outer exposed terrain edges pulse 14.
SPACE target cue	A brace that can be picked up, or an empty socket that can receive the carried brace, has a color-11 outline while faced.
Color 15	Unused.

A brace’s local 4×4 lattice can use this exact pattern:

9  9  9  9
9  0 10  9
9 10  0  9
9  9  9  9

For a loaded brace, replace the first row with 11.

A hatch uses:

14 14 14 14
14  5  5 14
14  5  5 14
14 14 14 14

When the climber overlaps an exposed brace, draw the brace first and the climber’s body over its center, leaving peripheral lattice visible.

3. Complete state contract
State:
    level_id
    player = (px, py)
    facing ∈ {LEFT, RIGHT}
    carrying_brace ∈ {false, true}
    occupied_brace_sockets ⊆ level.sockets
    slabs = {
        slab_id: (x_left, y, width)
    }
    terminal ∈ {false, success, failure}

There is no hidden phase, timer, energy value, selected object, pointer mode, or persistent cursor.

The following are level data, not mutable state:

terrain
sockets
slab widths and horizontal x positions
exit coordinate
active width and height

Slabs retain their IDs and horizontal coordinates throughout an attempt. Their y coordinate is the only mutable slab coordinate.

All authored initial states are already fully settled. Loading a level must not cause an unsolicited gravity animation.

Conservation invariant

Except when an attempt is reset:

number_of_board_braces + carrying_brace = initial_total_braces

Braces are never created, consumed, or destroyed.

4. Occupancy and support
Solid and passable cells

For player movement:

Fixed terrain is solid.

Slab cells are solid.

A loaded brace is solid.

An exposed brace is passable and climbable.

Empty sockets and the hatch are passable.

The climber never pushes any solid object.

A brace at (bx,by) is loaded exactly when a slab currently occupies (bx,by+1).

Slab support

A slab is rigid. It is supported when at least one cell in its footprint has one of the following immediately below it:

fixed terrain;

any brace;

a supported slab.

One supporting cell arrests the entire slab. Unsupported portions may cantilever arbitrarily; there is no torque or balance calculation.

The player does not support slabs.

A brace placed under an already settled slab does not raise it. Slabs only move downward.

Player support

The player does not fall when either condition holds:

The player currently overlaps an exposed brace and is therefore gripping it.

The cell immediately below contains terrain, a brace, or a slab.

Otherwise, gravity moves the player downward.

5. Actions
LEFT and RIGHT

Let dx=-1 for LEFT and dx=+1 for RIGHT.

Set facing to the requested direction even if movement will fail.

Let the adjacent cell be (px+dx, py).

If the adjacent cell is passable, move into it.

Otherwise, if the adjacent cell is solid and (px+dx, py+1) is passable, step up exactly one cell to (px+dx, py+1).

Otherwise, remain in place and produce the visible bump feedback.

Resolve gravity.

This means walking into open air is allowed; gravity then determines where the player lands.

Blocked horizontal input is also the game’s turn-in-place operation. The facing nub flips before the bump flashes, so this is visible rather than hidden. Later supports deliberately require approaching from a side and turning toward a loaded brace.

UP

UP attempts to move to (px,py+1).

It succeeds only when:

the destination is passable and in bounds; and

either the current cell or destination cell contains an exposed brace.

Then gravity resolves.

This permits both:

moving upward while inside a lattice;

leaving a brace and standing one cell above it.

It also permits sparse ladders with a one-cell air gap between braces: leave one brace into the supported air cell, then climb into the next brace.

DOWN

DOWN is symmetric:

destination (px,py-1) must be passable and in bounds;

the current cell or destination must contain an exposed brace.

After movement, gravity resolves. If the player leaves the bottom of a brace into unsupported air, the player may continue falling during the same action.

SPACE

SPACE targets the horizontally adjacent cell:

target = (px-1, py) when facing LEFT
target = (px+1, py) when facing RIGHT

Exactly one of these transitions may occur:

Pick up

If:

carrying_brace = false; and

the target contains a brace, whether exposed or loaded;

then:

remove that brace from the board;

set carrying_brace = true.

If the brace was loaded, its slab becomes unsupported immediately before gravity resolution.

Place

If:

carrying_brace = true;

the target is a socket;

the target contains no terrain, slab, brace, or player;

then:

put a brace in the target socket;

set carrying_brace = false.

A brace may be placed immediately below a slab. It supports the slab at its current height but never lifts it.

Invalid SPACE

All other cases flash the target cell in 8. No object or inventory state changes, but gravity is still resolved.

There is no vertical pickup, remote manipulation, pushing, throwing, or selecting.

Clicking

There are deliberately no clickable gameplay objects. Clicking is not part of the mechanic.

A click on any cell:

is a silent no-op;

does not establish a cursor or selection;

does not alter facing, inventory, terrain, braces, slabs, player position, or the action budget.

A click outside the active rectangle is a completely silent no-op.

6. Gravity and update order

Every action is resolved by one call to step(). There is no passive update between actions.

step(state, action):
    1. Apply the direct action mutation.
    2. Resolve all slab and player gravity to a fixed point.
    3. If alive and the settled player coordinate equals the hatch:
           return terminal success
       otherwise:
           return the settled nonterminal state
Slab gravity microtick

To model falling stacks without dependence on iteration order:

Mark every slab directly supported by terrain or a brace as anchored.

Repeatedly mark a slab anchored if any footprint cell is immediately above an already anchored slab.

Every unanchored slab moves downward by one cell simultaneously.

Repeat until all slabs are anchored.

Thus a stack disconnected from terrain or braces falls together. When its lowest slab reaches a catch, upper slabs can continue settling until they rest on it.

If a slab would cross below y=0, the attempt terminates in failure. No authored slab fall line permits this.

Player during a gravity microtick

After determining which slabs move:

If the player was standing directly above a moving slab, the player rides that slab downward by one cell.

Otherwise, move the slabs and check for collision with the player.

If a slab occupies the player’s cell, the player is crushed.

If a falling slab loads a brace currently occupied by the player, the player is pinched inside the newly loaded structure and is also crushed.

If alive, apply one row of player gravity unless the player is gripping or supported.

Repeat with the next microtick.

Passing through the hatch during a fall does not count. Only the final settled coordinate is validated.

Deterministic animation

step() computes the final logical state and a sequence of render events. Animation playback does not mutate gameplay state.

Ordinary walking, climbing, pickup, and placement render immediately.

Pickup or placement receives one local highlight frame.

Each one-row gravity microtick is shown for exactly two render frames.

Input is rejected while those frames play.

Animation frames consume no actions and no budget.

On landing, the actual supporting brace rail or slab contact edge flashes 11 for one frame.

Invalid movement uses a two-frame one-pixel bump-and-return with an 8 outline.

Crushing produces four red frames and terminal failure.

Completion produces six green pulse frames and terminal success.

7. Objective, failure, and exploration

The green hatch is the sole goal.

Success is exactly:

terminal != failure
and gravity is fully settled
and player == exit_coordinate

The following are explicitly irrelevant to validation:

whether a brace is still carried;

where unused braces remain;

player facing;

positions of slabs other than how they physically enable the hatch;

whether lower corridors remain open;

number of actions taken.

It is legal to finish while carrying a brace.

There is no gameplay timer, energy meter, move budget, or score threshold. Attempts may use unlimited actions. A harness may impose a nonsemantic safety cap of at least 2,048 calls, but this cap must not be rendered or used by the puzzle validator.

Crushing or falling out of the active board ends the current attempt. The normal environment reset restores the exact authored initial state and is not a player action.

8. Authored levels
Level 1 — First Rung
Player-facing discovery

The green hatch is immediately visible one cell above and to the right of the climber. A single blue lattice occupies the cell below it. Walking right visibly places the climber inside the lattice rather than colliding with it. UP then enters the hatch and immediately produces the green completion pulse.

The terrain wall to the right prevents walking around the hatch or mistaking the brace for an ordinary decorative object. There are no slabs, sockets requiring interpretation, decoys, timing constraints, or failure hazards.

This gives positive goal evidence after only two attributable actions.

Exact initial state
W = 8
H = 5

terrain =
    row(0, 0..7)
    ∪ {(4,1), (4,2)}

sockets = {(3,1)}
initial braces = {(3,1)}
slabs = {}

player = (2,1)
facing = RIGHT
carrying_brace = false
exit = (3,2)

selected_object = null
cursor = null
Verified path
R U

RIGHT: player enters exposed brace (3,1).

UP: player moves to hatch (3,2) and wins.

Attribution and necessity

The only unfamiliar causal property is that an exposed brace is enterable and climbable. The hatch cannot be reached by ordinary walking or one-cell ledge stepping because the right wall blocks a step-around approach.

Level 2 — Missing Rung
Player-facing discovery

The layout is visually the same as Level 1, but the blue lattice is missing. Its outlined socket remains, and a blue lattice pack is visibly attached to the climber.

Ordinary movement reaches a local impasse: the hatch is above an empty cell, and UP does nothing. Because the climber begins facing the outlined socket, SPACE is the cheapest discriminating experiment. It immediately inserts the carried lattice into the outline, recreating the exact successful structure from Level 1.

Only brace placement is new.

Exact initial state
W = 8
H = 5

terrain =
    row(0, 0..7)
    ∪ {(4,1), (4,2)}

sockets = {(3,1)}
initial braces = {}
slabs = {}

player = (2,1)
facing = RIGHT
carrying_brace = true
exit = (3,2)

selected_object = null
cursor = null
Verified path
S R U

SPACE: place the carried brace in (3,1).

RIGHT: enter the brace.

UP: enter hatch (3,2).

Attribution and necessity

The visual comparison with Level 1 rules out “the outline itself is climbable.” The player must observe the carried brace leave the climber and appear in the socket before the previously learned climbing action works.

Level 3 — Drop Bridge
Player-facing discovery

A two-cell pit separates two raised platforms. Above the pit is an orange slab resting on a brace whose top rail is yellow. The player can reach the left side of that brace, but ordinary walking cannot cross the pit.

Removing the visibly loaded brace causes an immediate four-row cascade. The slab lands on the broad pit floor, filling the two-cell gap. The resulting settled state is unmistakable: the former overhead weight is now the bridge.

This is the first gravity lesson, but the experiment is safe. The slab has a full-width catch floor and cannot disappear or land ambiguously.

Exact initial state
W = 14
H = 6

terrain =
    rect(0..4, 0..2)
    ∪ rect(7..13, 0..2)
    ∪ {(5,0), (6,0), (4,3)}

sockets = {(5,4)}
initial braces = {(5,4)}

slabs = {
    A: (x=5, y=5, width=2)
}

player = (2,3)
facing = RIGHT
carrying_brace = false
exit = (9,3)

selected_object = null
cursor = null
Verified path
R R S R R R R R

Compact form:

R² S R⁵

Key states:

After R², player is at (4,4), directly left of support (5,4).

After action 3, the brace is carried and slab A has settled from y=5 to y=1.

The player walks across A’s top and onto the right platform.

Action 8 reaches hatch (9,3).

Attribution and necessity

The pit is two cells wide, so one-cell ledge stepping cannot cross it. The slab is the only object capable of making a continuous surface. The old brace-climbing rule cannot explain the slab’s multi-row movement after pickup.

Level 4 — Chosen Height
Player-facing discovery

The player begins with a carried brace beside an empty lower socket. A slab is visibly held much higher by another brace. The hatch sits on the left platform at a height that the slab can reach only if it is caught above the floor.

The first useful experiment is to put the carried brace in the lower socket. The player then climbs the fixed right ladder and removes the upper support. The slab visibly falls until only its right edge contacts the newly placed brace. The entire rigid slab stops there and becomes a platform to the hatch.

This forces observation of the distinct property: one supported edge fixes the height of the whole slab. Merely knowing that braces can be placed or removed is insufficient.

Exact initial state
W = 14
H = 8

terrain =
    row(0, 0..13)
    ∪ rect(0..6, 0..3)
    ∪ rect(9..13, 0..1)
    ∪ {(9,5), (10,5)}

sockets =
    {(8,2), (8,6)}
    ∪ col(11, 2..5)

initial braces =
    sockets - {(8,2)}

slabs = {
    A: (x=7, y=7, width=2)
}

player = (9,2)
facing = LEFT
carrying_brace = true
exit = (5,4)

selected_object = null
cursor = null
Verified path
S R R U U U L L S L L L L

Compact form:

S R² U³ L² S L⁴

Key states:

Action 1 places a brace at (8,2).

Actions 2–8 reach (9,6), to the right of upper support (8,6).

Action 9 removes the upper brace.

A falls from y=7 to y=3, held by the single brace below its right cell.

Four LEFT actions cross the slab and reach hatch (5,4).

Anti-degenerate check

Removing the upper support before placing the lower brace makes A fall to y=1. Its top is then at y=2, permanently too low to connect to the hatch platform at y=4. Because slabs never rise, that mistake cannot be repaired within the attempt.

Level 5 — Second Landing First
Player-facing discovery

Slab A begins on an upper support and has a second visible brace lower in its fall line. Removing the upper brace does not send A to the floor: it stops at the intermediate brace and becomes a high bridge.

The carried upper brace must now be taken down to the empty low socket before the intermediate support is touched. The player then climbs back up from the right, crosses the same slab that was just lowered, falls onto the small side ledge, and removes the intermediate support from its opposite side.

A falls a second time and is caught at the hatch height.

The slab therefore has two contrasting roles under exactly the same rule:

after the first release, it is a temporary high bridge needed to reach its own second support;

after the second release, it becomes the permanent low platform to the hatch.

Exact initial state
W = 14
H = 10

terrain =
    row(0, 0..13)
    ∪ row(1, 8..10)
    ∪ {(8,3), (9,3)}
    ∪ {(5,3), (5,4)}
    ∪ {(8,7), (9,7)}

sockets =
    {(7,8), (6,5), (7,2)}
    ∪ col(10, 4..7)

initial braces =
    sockets - {(7,2)}

slabs = {
    A: (x=6, y=9, width=2)
}

player = (8,8)
facing = LEFT
carrying_brace = false
exit = (7,4)

selected_object = null
cursor = null
Verified path
S R R R L L L S L U R R R U U U L L L L L R S R R

Compact form:

S R³ L³ S L U R³ U³ L⁵ R S R²

Key states:

Action 1 removes upper support (7,8). A falls from y=9 to y=6 on intermediate support (6,5). The upper brace is now carried.

Moving right from the upper shelf deliberately falls to the lower-right floor.

Action 8 places the carried brace at lower catch (7,2).

Actions 9–16 return to the upper-right route through ladder x=10.

Actions 17–21 cross A’s top and fall onto side ledge (5,4), leaving the player at (5,5).

RIGHT at action 22 is blocked by the loaded brace at (6,5) but visibly turns the climber toward it.

Action 23 removes the intermediate brace. A settles at y=3 on catch (7,2).

Two RIGHT actions reach hatch (7,4).

Anti-degenerate and necessity notes

Releasing (6,5) before placing (7,2) sends A to the floor and leaves the hatch unreachable.

The right ladder is needed to return to the high side after placing the catch.

The side ledge (5,4) is needed to access the intermediate support without standing in A’s fall line.

The solution is not an object checklist: the placed brace is useful only later, while the temporarily supported slab is simultaneously the route used to reach its next release.

Level 6 — Seal, Then Climb
Player-facing discovery

A wide slab A hangs two cells above the floor, creating a visible underpass from the player’s starting side to a small support-access step on the left. The green hatch is high on the far right.

The player must first walk under A. From the left step, the support can be removed. A then falls into the exact corridor just traversed, permanently sealing it. The player is now committed to the left side, carrying A’s brace.

Nearby is a sparse vertical scaffold whose bottom socket is empty. Placing the carried brace there opens the only climb to the upper landing. From that landing, the player drops beside slab B’s support and releases it.

B falls onto only the rightmost cell of A. Despite two cells overhanging empty space, the whole rigid slab stays. Its top becomes the final traverse to the hatch.

This level gives the familiar objects three different contextual meanings:

A is first a ceiling over a useful route.

A becomes a permanent barrier and the base of a stair.

A’s removed brace becomes a ladder rung.

B is supported by another slab rather than a brace, with only one cell of overlap.

Exact initial state
W = 14
H = 9

terrain =
    row(0, 0..13)
    ∪ {(2,1)}
    ∪ row(7, 1..4)
    ∪ {(5,4)}
    ∪ {(9,3)}

sockets =
    {(3,2), (6,5)}
    ∪ {(0,1), (0,3), (0,5), (0,7)}

initial braces =
    sockets - {(0,1)}

slabs = {
    A: (x=3, y=3, width=4),
    B: (x=6, y=6, width=3)
}

player = (7,1)
facing = LEFT
carrying_brace = false
exit = (9,4)

selected_object = null
cursor = null

Here row(7,1..4) means terrain cells (1,7) through (4,7).

Verified path
L L L L L R S L S L U U U U U U R R R R R S R R R R

Compact form:

L⁵ R S L S L U⁶ R⁵ S R⁴

Key states:

Actions 1–4 move left beneath slab A.

Action 5 steps onto terrain (2,1), leaving the player at (2,2).

RIGHT is blocked by loaded support (3,2) but turns toward it.

Action 7 removes that support. A falls from y=3 to y=1, occupying corridor cells x=3..6. The route back is gone.

Action 9 places A’s brace into missing ladder socket (0,1).

Actions 10–16 climb the sparse sequence at y=1,3,5,7.

Five RIGHT actions cross the upper landing and fall to side-access cell (5,5).

Action 22 removes B’s support (6,5).

B falls from y=6 to y=2. Its left cell at (6,2) is supported by A’s rightmost cell (6,1); cells (7,2) and (8,2) cantilever.

Four RIGHT actions traverse B and step onto terrain (9,3), entering hatch (9,4).

Anti-cleanliness and necessity notes

The underpass cannot be removed from the design without removing the commitment: the player must use it while it exists and knowingly destroy it. The bottom scaffold socket, every sparse rung, the upper landing, fall-side ledge, and hatch pedestal are each required for reachability. No other brace is accessible as a replacement for A’s support brace.

The wider slabs introduce no new physics. They make already learned rigidity and one-edge support produce a stronger spatial consequence.

Level 7 — One Pull, Three Steps
Player-facing discovery

No new rule is introduced. The level asks the player to coordinate every established consequence.

Three two-cell slabs are vertically staggered:

A is low and left.

B is higher and shifted one cell right.

C is highest and shifted another cell right.

Their final positions can form an exact three-step staircase, but their supports are approached from alternating sides.

The dependency network is:

Release A to make the first step and obtain the brace that opens the middle-right ascent.

Use that ascent to reach B from its left.

Release B onto A and move B’s brace to the missing bottom of the far-left sparse ladder.

Climb to C’s upper support.

Release C only to its intermediate support.

Put the now-obsolete upper brace back into its socket to empty the one-brace hand.

Cross C while it is still high.

Reach the intermediate support from the right and release it.

C lands on B and completes the staircase directly beneath the hatch.

The same brace rule has three strategic roles in one level: supporting weight, filling a route gap, and serving as inventory parking after it can no longer affect the slab.

Exact initial state
W = 15
H = 14

terrain =
    row(0, 0..14)
    ∪ row(4, 0..5)
    ∪ row(4, 10..12)
    ∪ row(11, 1..6)

sockets =
    {(6,1), (6,5)}
    ∪ {(7,12), (8,7)}
    ∪ {(9,5), (9,7)}
    ∪ col(13, 2..4)
    ∪ {(0,5), (0,7), (0,9), (0,11)}

initial braces =
    sockets - {(9,5), (0,5)}

slabs = {
    A: (x=5, y=2, width=2),
    B: (x=6, y=6, width=2),
    C: (x=7, y=13, width=2)
}

player = (7,1)
facing = LEFT
carrying_brace = false
exit = (7,4)

selected_object = null
cursor = null

Named supports and routes:

A support                 = (6,1)
B support                 = (6,5)
C upper support           = (7,12)
C intermediate support    = (8,7)

middle-route socket       = (9,5), initially empty
middle upper rung         = (9,7)

right access tower        = (13,2), (13,3), (13,4)

left ladder bottom        = (0,5), initially empty
left sparse rungs         = (0,7), (0,9), (0,11)
Verified path
S R R R R R R U U U L L L S L U U L L L L
R S L L L L S L U U U U U U
R R R R R R
S S
R R R D L S L L

Exact unbroken replay string:

SRRRRRRUUULLLSLUULLLLRSLLLLSLUUUUUURRRRRRSSRRRDLSLL

Compact form:

S
R⁶ U³ L³
S L U² L⁴
R S
L⁴ S L U⁶
R⁶
S S
R³ D L S L²
Replay checkpoints
First landing

Action 1: remove support (6,1).

A falls from y=2 to y=1.

Final A footprint: (5,1),(6,1).

Player carries A’s brace.

Open the middle route

Actions 2–10 move across the floor and climb the right tower.

Actions 11–13 reach (10,5).

Action 14: place A’s brace at (9,5).

LEFT, UP, UP reaches fixed rung (9,7).

Four LEFT actions cross intermediate brace (8,7), then B’s top, and fall to (5,5).

Second landing

Action 22, RIGHT, turns toward loaded B support (6,5) without entering it.

Action 23: remove B’s support.

B falls from y=6 to y=2, supported by A at their shared x=6.

Final B footprint: (6,2),(7,2).

Player carries B’s brace.

Reuse B’s brace as access

Actions 24–27 move to (1,5).

Action 28: place B’s brace at sparse-ladder bottom (0,5).

Actions 29–35 climb through (0,5),(0,7),(0,9),(0,11).

Actions 36–41 cross the high platform to (6,12).

First C release

Action 42: remove upper support (7,12).

C falls from y=13 to y=8, caught by intermediate support (8,7).

C footprint is now (7,8),(8,8).

Player carries the upper brace.

Free the hand without changing C

Action 43: put that brace back into socket (7,12).

C does not rise; it remains at y=8.

This empties the player’s one-brace inventory so the intermediate support can later be removed.

The reinserted brace also provides a stable first cell at the edge of the high platform.

Cross and release from the opposite side

Actions 44–46 cross the reinserted brace and C, ending above right-side rung (9,7).

Action 47 descends into (9,7).

Action 48, LEFT, is blocked by loaded intermediate support (8,7) but visibly turns toward it.

Action 49: remove (8,7).

C falls from y=8 to y=3, supported by B at their shared x=7.

Final settled staircase:

A: x=5..6 at y=1
B: x=6..7 at y=2
C: x=7..8 at y=3

Action 50 moves left and falls onto C’s top at (8,4).

Action 51 moves left into hatch (7,4).

Anti-degenerate and dependency notes

Without socket (9,5), A’s brace cannot open the middle ascent.

Without B’s released brace in (0,5), the gaps between the fixed sparse rungs cannot be crossed.

Reusing B’s brace at its old support instead leaves the high platform unreachable.

C’s upper brace must be parked before the intermediate brace can be picked up; carrying two braces is impossible.

The intermediate support is accessible only from the right after C has become a high bridge.

Releasing the intermediate support before B has formed the second step gives C the wrong final height.

The right tower, middle upper rung, left sparse rungs, both empty route sockets, and all three slabs participate in every successful dependency structure.

The level is not encountered in usage order: the player repeatedly crosses between the right tower, central service slab, far-left ladder, high platform, and right side of C. The final release is physically adjacent to the beginning of the middle route but cannot be performed during the first visit.

9. Curriculum and anti-degeneracy audit
Foundational attribution

Level 1: only brace climbing changes the outcome.
Level 2: only placement is new, in an otherwise repeated scene.
Level 3: only loaded-support removal and gravity are new; the broad catch makes the result safe and immediate.

No foundational level contains an unused brace, irrelevant slab, misleading socket, tight budget, hidden dependency, optional decoy, or simultaneous unexplained rule change.

Contrasting mechanic roles
Mechanic	Early role	Later contrasting roles
Brace	Climbable rung	Loaded support, chosen catch, missing ladder segment, route bridge, inventory parking
Slab	Bridge across a pit	Height-selected platform, temporary bridge, permanent corridor seal, base step, overhanging step
Gravity	Safe one-slab demonstration	Controlled two-stage descent, irreversible route sacrifice, multi-slab staircase
One-edge support	Existing support under a small slab	Authored catch height, large slab cantilevering from one cell of another slab
Carry capacity	Obvious carried tutorial object	Forces relocation before a second support may be removed
Facing	Visible movement direction	Blocked movement becomes a deliberate turn toward a loaded support
State-space checks

Under the specified action semantics, the replay strings above are shortest successful traces when click no-ops are excluded.

Additional reachability checks confirm:

Level 4 is unsolvable without its lower catch socket.

Level 5 is unsolvable without its lower catch or right return ladder.

Level 6 is unsolvable without the missing bottom socket, any one of the sparse fixed rungs, the underpass access step, the upper landing, the B-support side ledge, or the exit pedestal.

Level 7 is unsolvable without the middle-route socket, the left-ladder socket, any left sparse rung, the right access tower, or the right-side rung beside C’s intermediate support.

These checks are not validator predicates. They verify that the authored geometry, rather than an arbitrary checklist, makes every central feature matter.
