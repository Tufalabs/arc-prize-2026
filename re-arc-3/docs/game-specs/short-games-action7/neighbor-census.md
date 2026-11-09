Neighbor Census

This is a game about flipping one switch becoming difficult because every switch can change several overlapping exact neighbor reports.

Authored level count: exactly 7.

Neighbor Census is a compact, fully visible constraint game. Every cell is a reversible switch. Some cells additionally carry a census halo that reports how many of their orthogonal neighbors must be on. The puzzle is complete when every census halo is simultaneously exact.

The game introduces no secondary devices, inventories, timers, locked tiles, or hidden modes. Its depth comes entirely from consequences of the same local rule:

a useful switch may repair several censuses at once;

the same switch may overfill another;

a census cell can itself be used as a switch;

one contributing neighbor can be exchanged for another while preserving one report but changing several others;

a shared switch that was invaluable earlier can later become impossible to keep on.

All exploratory actions are reversible, and there is no move limit.

1. Player-facing rules
Controls

UP, DOWN, LEFT, RIGHT: move the persistent selection cursor by one logical cell.

SPACE: toggle the selected cell between off and on.

Click actions are accepted by the environment but are deliberate no-ops. There are no clickable objects and no click affordances.

Movement does not wrap.

An attempted move beyond the rectangular board leaves the cursor in place and gives a brief visual nudge.

Cell behavior

Every board position contains one switch cell.

Off cell: gray face.

On cell: blue face.

Ordinary cell: only has the switch face.

Census cell: has the same switch face, with a target tally, four directional neighbor lamps, and a status halo layered around it.

For a census cell at (r,c):

count(r,c)=
(r
′
,c
′
)∈N
4
	​

(r,c)
∑
	​

1[on(r
′
,c
′
)]

where N4 contains only in-bounds orthogonal neighbors.

Important consequences:

Diagonal cells never count.

A census cell’s own state does not count toward its own target.

Its state does count when it is an orthogonal neighbor of another census.

Missing positions beyond the rectangular board do not count.

A census is satisfied only when its count is exactly its target.

Both shortage and overfill are failures.

Visible objective

The level succeeds when every census cell has a continuous green status halo.

No particular cursor location, action count, ordinary-cell pattern, or census-cell face state is checked beyond what follows from the neighbor equations. Short and overfull halos are both unacceptable.

2. Visual and rendering contract
Canvas and logical board

Render canvas: exactly 64×64.

Logical boards are full rectangles with no internal holes.

Maximum authored board size: 6×6.

Coordinates are zero-indexed:

row increases downward;

column increases rightward.

There are no grid lines or decorative internal borders. Cells are separated by the dark negative space built into their sprites.

Scaled cell sprite

Each logical cell uses a base 10×10 integer-pixel sprite.

Choose:

scale = min(3, floor(60 / (10 * max(H, W))))

scale is always at least 1 for the authored boards.

Each base sprite pixel becomes a scale×scale block. Center the complete rendered board in the 64×64 canvas:

board_width  = 10 * scale * W
board_height = 10 * scale * H
origin_x = floor((64 - board_width)  / 2)
origin_y = floor((64 - board_height) / 2)

The top-left base-pixel coordinate for logical cell (r,c) is:

px = origin_x + 10 * scale * c
py = origin_y + 10 * scale * r

All sprite geometry below is stated in unscaled 10×10 local coordinates and then nearest-neighbor scaled.

Base cell face

The switch face occupies:

x in [2,8)
y in [2,8)

Colors:

off: color 2

on: color 9

Census markings are drawn over this same face; they never replace its state.

Census target strip

A census target is a horizontal, nondirectional four-slot strip inside the face.

Slots are:

slot i: x in [3+i, 4+i), y in [4,6), for i = 0,1,2,3

For targets 1–4:

the leftmost target slots use color 11;

remaining slots use color 4.

For target 0:

all four slots use color 13.

Thus zero has an explicit maroon marker rather than being represented by missing yellow marks. Level 2 teaches this marker directly.

The strip only communicates a number. Its left-to-right arrangement has no spatial or directional meaning.

Live orthogonal-neighbor lamps

A census has four possible two-by-two lamps:

north: x in [4,6), y in [0,2)
east:  x in [8,10), y in [4,6)
south: x in [4,6), y in [8,10)
west:  x in [0,2), y in [4,6)

For each in-bounds neighbor:

neighbor off: color 3

neighbor on: color 10

For a missing out-of-bounds direction, leave the lamp position as background color 5.

The lamps change solely from the corresponding neighbor states. The target does not affect lamp colors.

Census status halo

The semantic halo lies on the perimeter of the local rectangle:

x in [1,9)
y in [1,9)

Render status as follows:

Short: four open L-shaped corner brackets in color 11.

Exact: complete continuous perimeter in color 14.

Overfull: complete continuous perimeter in color 8.

The exact halo is visually different from merely having enough filled spokes: only equality closes the green perimeter.

Selection cursor

Render the selected cell with four outer L-corners in color 6. Draw selection last, so it remains visible on ordinary and census cells.

Selection corners occupy the three local pixels nearest each sprite corner, for example at the top-left:

(0,0), (1,0), (0,1)

with rotationally symmetric corners elsewhere.

For an invalid edge move:

cursor coordinates do not change;

on the returned observation only, selection corners are displaced one scaled pixel in the attempted direction and use color 7;

the nudge flag clears at the beginning of the next step().

Palette usage
Role	Color ID
Background	5
Cell off	2
Cell on	9
Existing neighbor off	3
Existing neighbor on	10
Empty target slot	4
Positive target slots	11
Explicit target zero	13
Short halo	11
Exact halo	14
Overfull halo	8
Selection	6
Invalid-move nudge	7

Colors 0, 1, 12, and 15 are unused in ordinary settled frames.

Animation and feedback

Ordinary movement and toggling are immediate.

A SPACE step returns a settled frame in which:

the selected face has changed;

every affected neighbor lamp has changed;

every affected census halo has been recomputed.

There are no intermediate animation frames and no animation consumes actions.

On completion, the returned terminal frame additionally draws a small color-14 block in each canvas corner. All census halos are already green in that same frame.

3. State-machine contract
State
H: integer
W: integer
on[H][W]: boolean
target[H][W]: optional integer in 0..4
cursor_r: integer
cursor_c: integer
nudge_direction: optional {UP, DOWN, LEFT, RIGHT}
terminal: boolean

There is no randomness and no hidden state.

Update order

Every state transition occurs only inside step(action).

step(action):
    if terminal:
        return unchanged terminal observation

    clear nudge_direction

    if action is UP/DOWN/LEFT/RIGHT:
        candidate = cursor + corresponding unit offset
        if candidate is in bounds:
            cursor = candidate
        else:
            nudge_direction = action

    else if action is SPACE:
        on[cursor_r][cursor_c] = not on[cursor_r][cursor_c]

    else if action is a click:
        do nothing

    simultaneously recompute every census live_count

    for every census:
        if live_count < target: status = SHORT
        if live_count = target: status = EXACT
        if live_count > target: status = OVER

    if every census is EXACT:
        terminal = true

    return settled rendered observation

No census is processed before another for causal purposes. All counts are derived from the same post-action on array.

Validation details

Relevant:

in-bounds orthogonal neighbor states;

every census target;

simultaneous exactness.

Ignored:

diagonal states relative to a census;

a census’s own state for its own count;

cursor position;

prior action history;

temporary earlier overfills or shortages;

total number of steps;

click history.

Allowed:

temporary breakage of satisfied halos;

any action ordering that reaches a satisfying assignment;

toggling a cell repeatedly;

continued exploration without penalty before completion.

4. Authored-data notation

The matrices below are implementation documentation and are not rendered as text.

For target matrices:

. means an ordinary cell;

0–4 means a census cell with that target.

For state matrices:

0 means off;

1 means on.

Action traces use:

U, D, L, R for cursor movement;

X for SPACE.

Each comma-separated group ends by toggling the reached cell. For example:

UL X

means UP, LEFT, SPACE.

The coordinate toggle lists are authoritative. The traces are verified examples, not action-order validators. Later puzzles have unique final assignments, but many toggle orderings can reach them.

5. Progression overview
Level	Board	Main discovery	Toggles	Verified total actions
1	2×1	A lit orthogonal neighbor fills a spoke and can close a halo	1	1
2	1×3	Exact zero requires active subtraction	2	4
3	1×3	Census cells are switches; self-state is excluded	3	6
4	4×4	One selected census switch can alter four reports at once	10	25
5	5×5	Exchange which neighbors supply a preserved count	19	50
6	6×6	A formerly helpful shared switch can be prohibited	22	55
7	6×6	Two adjacent census hubs require interlocked substitutions	26	64

Levels 1–3 form the clean causal curriculum. From Level 4 onward, the boards form dependency networks rather than linear sequences.

6. Authored levels
Level 1 — One Lit Spoke
Player-facing discovery

A single census sits above its only possible neighbor. Its target strip contains one yellow unit, its sole existing spoke is dark, and its halo is open. The only neighboring switch is already selected.

Pressing SPACE:

turns that cell blue;

fills the census’s south lamp;

closes the halo in green;

completes the level.

This is direct positive goal evidence. The player sees the target, the contributing direction, and successful equality in one action.

State-machine instance

Dimensions:

H = 2
W = 1

Targets:

1
.

Initial states:

0
0

Initial cursor:

(1,0)

Canonical final states:

0
1

The census’s own state is not constrained by another census in this introductory level. The canonical trace leaves it off, but either self-state is accepted if its neighbor is on.

Verified path

Toggle order:

(1,0)

Actions:

X

Toggles: 1

Cursor moves: 0

Total actions: 1

Attribution and necessity check

The selected cell is the census’s only valid neighbor.

No other object changes simultaneously.

Completion occurs with exactly one lit spoke, matching the one-unit target.

A hypothesis such as “the census wants its own cell on” is contradicted by the unchanged census face and successful neighbor toggle.

Level 2 — Zero Means None
Player-facing discovery

A zero-target census lies between two on cells. Its two horizontal lamps are filled and its halo is red.

The player begins on the left contributing cell. Turning it off reduces the visible count from two to one, but the census remains red. Turning off the second neighbor reduces the count to zero and closes the halo.

This teaches three things without another device:

the maroon target strip means zero;

turning off is productive, not merely undo;

a report must be exact rather than “at least good enough.”

State-machine instance

Dimensions:

H = 1
W = 3

Targets:

. 0 .

Initial states:

1 0 1

Initial cursor:

(0,0)

Canonical final states:

0 0 0

As in Level 1, the isolated observer’s own face is not constrained by another census. The verified trace leaves it off.

Verified path

Toggle order:

(0,0), (0,2)

Actions:

X, RR X

Toggles: 2

Cursor moves: 2

Total actions: 4

Attribution and necessity check

After the first toggle, one filled spoke visibly remains and the halo remains overfull.

Both endpoint states must be changed; one random toggle cannot complete the level.

The solution cannot be explained by the Level 1 heuristic “turn things on until green.”

There are no diagonal cells, secondary censuses, or unrelated components.

Level 3 — The Observer Switches
Player-facing discovery

All three cells are census cells. There is no ordinary switch available, creating a clean local impasse for any hypothesis that marked cells are immutable clues.

The selected center census has target two. The edge censuses each have target one.

Toggling the center census on:

changes its face from gray to blue;

leaves its own zero-of-two report unchanged;

simultaneously closes both edge halos, because the center is their shared neighbor.

Toggling the left and right census cells then fills the center census’s two lamps. Each edge census’s own halo remains based only on the center.

This forces observation of the exact property that distinguishes census cells from static clues.

State-machine instance

Dimensions:

H = 1
W = 3

Targets:

1 2 1

Initial states:

0 0 0

Initial cursor:

(0,1)

Unique final states:

1 1 1
Verified path

Toggle order:

(0,1), (0,0), (0,2)

Actions:

X, L X, RR X

Toggles: 3

Cursor moves: 3

Total actions: 6

Attribution and necessity check

The board contains no ordinary cells, so marked-cell participation is mandatory.

The first toggle changes two neighboring halos and not the selected cell’s own halo.

After toggling only one edge, the center visibly reports one of two and remains short.

The satisfying assignment is unique.

Level 4 — Four-Way Hinge
What it teaches

This is the first two-dimensional composition.

The selected cell at (1,1) is itself a census cell with target two. Its own halo starts green. It is orthogonally adjacent to four other census cells, all of which begin exactly one short.

Toggling the selected census on:

leaves its own green halo unchanged;

fills one spoke in each of its four neighboring censuses;

closes all four neighboring halos simultaneously.

This is the unmistakable introduction to a high-leverage shared switch. The rest of the board prevents that single dramatic action from being the whole solution: peripheral reports remain wrong, and several of the newly satisfied census cells must later have their own switch faces changed.

State-machine instance

Dimensions:

H = 4
W = 4

Targets:

0 2 . .
1 2 3 3
1 3 . .
. . 2 1

Initial states:

0 1 1 0
1 0 0 0
0 0 1 0
1 1 1 1

Initial cursor:

(1,1)

Unique final states:

0 0 1 1
0 1 1 0
0 1 1 1
0 1 0 0
Initial local evidence

Before the first toggle:

census (1,1), target 2: exact
census (0,1), target 2: short by 1
census (1,0), target 1: short by 1
census (1,2), target 3: short by 1
census (2,1), target 3: short by 1

After toggling (1,1), all four neighboring reports become exact while (1,1) remains exact.

No diagonal census changes.

Verified path

Toggle order:

(1,1),
(0,1),
(1,2),
(2,1),
(1,0),
(3,0),
(3,2),
(3,3),
(2,3),
(0,3)

Actions:

X,
U X,
DR X,
DL X,
UL X,
DD X,
RR X,
R X,
U X,
UU X

Toggles: 10

Cursor moves: 15

Total actions: 25

Player reasoning outline

Use the selected center switch to repair four short reports at once.

Notice that the surrounding census faces are themselves switches and still feed the outer reports.

Resolve the red zero census in the northwest without abandoning the central shared contribution.

Use the lower census states to supply the two bottom targets.

Finish the right-side pair with different contributors rather than treating the four central halos as independent tasks.

Anti-degenerate and anti-cleanliness check

The central selected state differs between initial and final configurations; it cannot be ignored.

Every one of the 16 cell states participates in at least one census equation.

The final satisfying assignment is unique.

Removing any one census equation introduces at least one additional satisfying assignment.

The solution is not a corridor or ordered checklist: the first action repairs four reports, while later changes to those reporters affect spatially separate constraints.

Level 5 — Preserve and Reroute
What it teaches

The center census (2,2) has target two and begins green.

Its initial contributing neighbors are:

north (1,2) = on
south (3,2) = on
west  (2,1) = off
east  (2,3) = off

Simply preserving the green center is not enough. Two outer reports expose that the wrong pair supplies its count:

census (2,4) is short and needs the east cell’s influence;

zero census (4,2) is overfull because of the south cell.

Two count-preserving substitutions solve these local conflicts.

First substitution: north → east

Initially:

center (2,2) is exact at two;

census (1,3) is exact at three;

census (2,4) is short at two of three.

Turn east (2,3) on:

center becomes overfull at three;

(1,3) becomes overfull;

(2,4) becomes exact.

Then turn north (1,2) off:

center returns to two;

(1,3) returns to three;

(2,4) stays exact.

The center and upper report have preserved counts, but influence has moved eastward.

Second substitution: south → west

Initially:

center is exact;

census (3,1) is exact at two;

zero census (4,2) is overfull at one.

Turn west (2,1) on, then south (3,2) off:

the center returns to two;

(3,1) returns to two because both west and south neighbor it;

(4,2) loses its only on neighbor and becomes exact zero.

This demonstrates that two neighbors can be interchangeable for one census while being very different globally.

State-machine instance

Dimensions:

H = 5
W = 5

Targets:

2 2 . 3 .
. . . 3 .
. 4 2 . 3
3 2 2 . 2
. . 0 . .

Initial states:

1 0 0 1 0
0 0 1 0 1
0 0 0 0 0
1 0 1 1 1
0 0 1 0 0

Initial cursor:

(2,3)

Unique final states:

0 1 1 1 1
1 1 0 1 1
1 1 1 1 1
1 1 0 0 1
1 0 0 0 1
Verified path

The first four toggles perform the two visible substitutions:

east on,
north off,
west on,
south off

Full toggle order:

(2,3),
(1,2),
(2,1),
(3,2),
(3,1),
(4,0),
(2,0),
(0,0),
(1,0),
(1,1),
(0,1),
(0,2),
(0,4),
(2,4),
(4,4),
(4,2),
(3,3),
(2,2),
(1,3)

Actions:

X,
UL X,
DL X,
DR X,
L X,
DL X,
UU X,
UU X,
D X,
R X,
U X,
R X,
RR X,
DD X,
DD X,
LL X,
UR X,
UL X,
UR X

Toggles: 19

Cursor moves: 31

Total actions: 50

Player reasoning outline

Treat the green center not as “finished,” but as an invariant whose suppliers may change.

Replace north with east to move one unit of influence toward the upper-right cluster.

Replace south with west to remove influence from the zero target without changing the lower-left anchor.

Use the now-established swap principle in the top-left cluster, where candidate neighbors overlap different pairs of census cells.

Toggle the center census’s own face later; this does not alter its target-two halo but is required by neighboring reports.

Anti-degenerate and anti-cleanliness check

All four central neighbor states differ between initial and final configurations.

The center begins and ends at count two, so satisfying it alone cannot explain the solution.

Both substitutions are forced by the unique final state.

Every cell participates in at least one census equation.

The final satisfying assignment is unique.

Every census equation is necessary: deleting any one creates another satisfying assignment.

The two swaps affect overlapping but nonidentical report sets, so the board cannot be reduced to independent local chores.

Level 6 — The Helpful Cell Must Go
What it teaches

The cell (2,2) is a shared on switch surrounded by four census cells:

north (1,2): target 0
west  (2,1): target 2
east  (2,3): target 1
south (3,2): target 3

Initially:

the north zero census is overfull by one;

west is exact;

east is exact;

south is exact.

Thus the shared switch performs three useful jobs but is categorically forbidden by the fourth. This reverses the role of the Level 4 hinge.

The nearby zero target makes the productive experiment clear: turn the central cell off.

Immediate result:

north becomes exact;

west becomes short by one;

east becomes short by one;

south becomes short by one.

The player now understands the cost of obeying the zero census and can search for substitute suppliers.

Two nearby cells provide the reroute:

turning (2,0) on restores west;

turning (3,3) on simultaneously restores east and south.

The formerly shared hub is replaced by two geometrically distinct contributions.

State-machine instance

Dimensions:

H = 6
W = 6

Targets:

. . . 0 1 .
1 0 0 . . .
1 2 . 1 4 1
. . 3 . . .
. 3 4 . 4 0
. 2 2 . 0 .

Initial states:

0 0 0 1 1 0
1 0 1 0 0 1
0 1 1 0 0 0
1 1 0 0 1 0
1 0 1 0 1 0
1 0 0 0 0 1

Initial cursor:

(2,2)

Unique final states:

0 0 0 0 0 0
0 0 0 0 1 1
1 0 0 1 0 1
1 1 1 1 1 0
0 1 1 1 0 1
0 1 1 0 1 0
Verified local opening
toggle (2,2) off:
    north fixed
    west, east, south become short

toggle (2,0) on:
    west fixed

toggle (3,3) on:
    east and south fixed together
Verified full path

Toggle order:

(2,2),
(2,0),
(3,3),
(3,2),
(2,3),
(0,3),
(0,4),
(1,4),
(2,5),
(4,5),
(5,5),
(5,4),
(4,4),
(4,3),
(5,2),
(5,1),
(5,0),
(4,0),
(4,1),
(2,1),
(1,2),
(1,0)

Actions:

X,
LL X,
DRRR X,
L X,
UR X,
UU X,
R X,
D X,
DR X,
DD X,
D X,
L X,
U X,
L X,
DL X,
L X,
L X,
U X,
R X,
UU X,
UR X,
LL X

Toggles: 22

Cursor moves: 33

Total actions: 55

Player reasoning outline

Remove the shared cell because target zero makes every solution with it on impossible.

Read the three new shortages as a map of what that switch used to supply.

Replace its west contribution with (2,0).

Replace its east and south contributions together with (3,3).

Continue through the outer board using the same principle: preserve already exact reports while relocating contribution through adjacent marked and ordinary cells.

Several census cells must themselves be toggled, so treating halos as immutable clues fails.

Anti-degenerate and anti-cleanliness check

The initially helpful central switch is forced off in the unique solution.

Both replacement cells are forced on.

The replacement branches are asymmetric: one repairs one census, the other repairs two.

Every cell participates in at least one census equation.

The final satisfying assignment is unique.

Every census equation is necessary.

The zero census is not a decorative prohibition: it removes the most locally efficient shared switch and forces the entire rerouting structure.

Level 7 — Double Census Exchange
What it combines

The final level centers on two adjacent census hubs:

A = (2,2), target 2
B = (2,3), target 2

Both halos begin green, but their contributing neighbors are globally wrong.

Because A and B are adjacent:

A’s switch state counts toward B;

B’s switch state counts toward A;

neither state counts toward its own halo.

This turns the Level 3 self-exclusion rule into part of an interlocked exchange.

Initial suppliers

For hub A:

north A  (1,2) = off
south A  (3,2) = off
west A   (2,1) = on
hub B    (2,3) = on

So A has two contributions: west plus B.

For hub B:

north B  (1,3) = on
south B  (3,3) = off
hub A    (2,2) = off
east B   (2,4) = on

So B has two contributions: north plus east.

Required final suppliers

Hub A must ultimately be supplied by:

north A + south A

Hub B must ultimately be supplied by:

hub A + south B

Thus both green hubs must preserve their counts while replacing all of their original suppliers.

Playable local impasse and discovery sequence

The cursor begins on hub A.

Turn A on.

A’s own halo does not change.

B becomes overfull at three.

Turn B’s north neighbor (1,3) off.

B returns to two.

One of B’s suppliers has been exchanged for A.

Turn B’s south neighbor (3,3) on.

B becomes overfull again.

Turn B’s east neighbor (2,4) off.

B returns to two.

B is now supplied by A and south.

Turn A’s north neighbor (1,2) on.

A becomes overfull.

Turn A’s west neighbor (2,1) off.

A returns to two.

Turn A’s south neighbor (3,2) on.

A becomes overfull again.

Turn hub B itself off.

B’s own halo remains unchanged.

A returns to two, now supplied by north and south.

This eight-toggle sequence repeatedly produces a visible red-overfull state followed by exact restoration. No hidden convention is needed to infer the exchange.

State-machine instance

Dimensions:

H = 6
W = 6

Targets:

. 0 3 . . .
. 2 . 2 1 0
1 3 2 2 . 1
. . . . 3 0
. . . 3 . .
2 1 . 3 3 .

Initial states:

0 0 1 1 1 1
1 1 0 1 1 1
0 1 0 1 1 1
0 1 0 0 1 0
0 0 1 0 1 0
0 0 1 0 0 1

Initial cursor:

(2,2)

Unique final states:

0 1 0 1 1 0
0 0 1 0 0 0
1 0 1 0 0 0
1 1 1 1 0 1
1 0 0 1 1 0
0 1 1 1 1 1
Verified full path

The first eight toggles are the hub exchanges described above.

Full toggle order:

(2,2),
(1,3),
(3,3),
(2,4),
(1,2),
(2,1),
(3,2),
(2,3),
(1,4),
(1,5),
(0,5),
(2,5),
(3,5),
(3,4),
(5,4),
(5,3),
(4,3),
(4,2),
(5,1),
(4,0),
(3,0),
(2,0),
(1,0),
(1,1),
(0,1),
(0,2)

Actions:

X,
UR X,
DD X,
UR X,
ULL X,
DL X,
DR X,
UR X,
UR X,
R X,
U X,
DD X,
D X,
L X,
DD X,
L X,
U X,
L X,
DL X,
UL X,
U X,
U X,
U X,
R X,
U X,
R X

Toggles: 26

Cursor moves: 38

Total actions: 64

Player reasoning outline

Recognize the two green hubs as invariants rather than completed subproblems.

Use A’s own state as a replacement supplier for B.

Replace B’s north and east suppliers with A and south.

Replace A’s west and B suppliers with north and south.

Follow the consequences into four outer regions:

upper-right zero and one targets;

right-edge one and zero targets;

lower census pair;

left-side zero and three targets.

Use marked census states throughout, sometimes preserving their own green halos while changing distant neighboring reports.

Finish with the upper-left exchange, where the target-zero cell eliminates the superficially symmetric alternative.

Anti-degenerate and anti-cleanliness check

Both central halos begin satisfied, so a “repair only non-green cells” strategy fails.

All eight local supplier and hub states differ between initial and final configurations.

A and B each act both as observer and as counted neighbor.

The two hub equations alone allow many arrangements; the surrounding dependency network forces one specific exchange.

Every cell participates in at least one census equation.

The final satisfying assignment is unique.

Every census equation is necessary.

The outer reports do not appear in usage order. Changes in the hub cluster affect several distant branches that must be reasoned about together.

7. Epistemic-solvability audit
Level 1

Before the only meaningful action, the player sees:

one target unit;

zero filled spokes;

one selected possible neighbor.

The result of SPACE immediately distinguishes “neighbor contributes” from alternatives.

Level 2

The player sees:

an explicit zero marker;

two filled spokes;

an overfull red halo.

Turning one selected neighbor off immediately reduces the count but does not complete the report. This distinguishes exact zero from “reduce it somewhat.”

Level 3

There are no ordinary switches. Toggling the selected census:

visibly changes its face;

leaves its own halo unchanged;

changes both neighboring halos.

No older rule or simpler mistaken explanation accounts for all three observations.

Level 4

The selected central census is surrounded by four short census halos. One toggle closes all four and leaves its own report unchanged. The shared-influence property cannot be mistaken for an effect on only one nearest marker.

Level 5

The center is already green. Nearby short and overfull reports show that merely preserving its total is insufficient. An add-then-remove pair visibly overfills and restores two reports while repairing a third, directly teaching influence substitution.

Level 6

A red zero census and three green censuses all point to the same on switch. Turning it off produces one immediate repair and three immediate deficits. The tradeoff is visible before any long plan is needed.

Level 7

Both hubs are green, but surrounding reports are not. Toggling hub A leaves A unchanged while overfilling B. The nearby alternative supplier allows an immediate paired correction, establishing the final interlocked exchange through local evidence.

8. Authored-instance verification requirements

An implementation test suite should assert all of the following:

There are exactly 7 levels.

Every target is between zero and the cell’s in-bounds orthogonal degree.

No initial level is already complete.

The listed action trace for each level reaches a terminal state.

Every intermediate count is computed from the post-action switch array.

Diagonal toggles never affect a census.

Toggling a census never affects its own live count.

Toggling a census does affect every adjacent census.

Overfull and short states both fail validation.

Cursor position does not affect validation.

Clicks do not change state.

Out-of-bounds movement does not change cursor coordinates.

Levels 3–7 have exactly one satisfying switch assignment.

Every cell in Levels 4–7 is a neighbor of at least one census.

In Levels 4–7, removing any single census constraint permits at least one additional satisfying assignment.

The verified solution depths are:

Level 1:  1 toggle,  0 moves,  1 total action
Level 2:  2 toggles, 2 moves,  4 total actions
Level 3:  3 toggles, 3 moves,  6 total actions
Level 4: 10 toggles, 15 moves, 25 total actions
Level 5: 19 toggles, 31 moves, 50 total actions
Level 6: 22 toggles, 33 moves, 55 total actions
Level 7: 26 toggles, 38 moves, 64 total actions

The result is one focused game family with only two cell categories and one local counting law. Its progression comes from increasingly consequential spatial overlap rather than accumulating devices or exceptions.
