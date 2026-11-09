Majority Cells

Exact authored level count: 7

This is a game about copying a cell’s local majority becoming difficult because every correction rewrites the votes needed by nearby corrections.

1. Player-facing concept

The board is made only of binary cells. Every cell visibly shows:

its current state as a large symbol;

its target state on a smaller attached tab;

whether the two currently match.

Clicking a cell does not let the player choose its new state. Instead, the clicked cell observes its existing orthogonal neighbors and copies whichever state has a strict majority. The clicked cell itself does not vote. A tied vote leaves it unchanged.

The first levels teach this as a local editing rule. Later levels exploit a less obvious consequence: after a cell changes, it immediately becomes a different voter for all adjacent future clicks. A correction can therefore enable one neighbor while permanently blocking another.

There are no switches, doors, counters, moving avatars, hidden weights, random events, or automatic cellular-automaton updates. The entire seven-level progression comes from one operation applied to carefully authored spatial patterns.

2. Controls and action contract

The environment exposes exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any render-canvas cell

Only direct clicks on game-cell modules are semantically used.

UP, DOWN, LEFT, RIGHT, and SPACE are accepted actions but are no-ops. They do not move a cursor, select a cell, or change game state.

A click anywhere inside a visible cell module—including its attached target tab—clicks that logical cell. The target tab is not separately editable. Clicks in gutters or outside the board are no-ops.

There is no persistent selected cell and no strategic cursor state.

All actions increment the technical episode step counter. Every level has max_steps = 1024, which is intentionally far above the verified solution length. There is no visible energy meter, timer, or action budget. The host’s normal reset() operation restores the exact authored initial state and is an environment lifecycle operation, not a player action.

3. Logical coordinate system

Each level is an R × C rectangular logical lattice.

Coordinates are written as:

(row, column)

Rows are numbered top to bottom from 1. Columns are numbered left to right from 1.

Two logical cells are neighbors exactly when their coordinates differ by one in one axis and zero in the other. Diagonal cells never vote.

Matrices in this document use:

0 for the hollow blue state;

1 for the solid orange state.

These digits are documentation notation only. No digits or text are rendered in the game.

4. Visual specification
4.1 Palette

Canvas background: color 4, very dark gray

Cell backing: color 3, dark gray

Inactive target-check bracket: color 2, gray

State 0: color 9, blue

State 0 highlight: color 10, light blue

State 1: color 12, orange

State 1 highlight: color 11, yellow

Matched target-check bracket and completion pulse: color 14, green

Technical max-step failure pulse: color 8, red

Hollow centers and empty sprite pixels: color 4

Only these colors are required.

4.2 Cell sprite

The minimum cell sprite is a connected 5 × 5 shape. Local sprite rows are numbered 0–4 and columns 0–4.

The backing occupies:

row 0: columns 2–4
row 1: columns 0–4
row 2: columns 0–4
row 3: columns 0–3
row 4: columns 0–3

This gives the cell a large lower-left current-state plate and a visibly attached upper-right target tab.

Current-state symbol

The current-state symbol occupies sprite rows 2–4, columns 0–2.

State 0:

its eight perimeter pixels are color 9;

the center pixel at local (row 3, column 1) is color 4.

State 1:

all nine pixels are color 12.

Thus current states differ through both color and solid-versus-hollow shape.

Target tab

The target symbol occupies:

(row 0, columns 3–4)
(row 1, columns 3–4)

Target 0 is a miniature hollow symbol:

(0,3), (0,4), and (1,3) are color 9;

(1,4) is color 4.

Target 1 is a solid 2 × 2 block of color 12.

Match indicator

A four-pixel L-shaped border surrounds the left and lower sides of the target tab:

(0,2), (1,2), (2,3), (2,4)

It is color 14 when current == target, and color 2 otherwise.

Consequently, the initial set of mismatched cells is visible directly as the set of cells lacking green target brackets. No additional mismatch glyphs or grid markings are rendered.

4.3 Scaling and placement

The 5 × 5 sprite is scaled by the largest integer factor that fits within its level’s module size and then centered in that module. Nearest-neighbor integer scaling is used; no antialiasing is required.

Modules are separated by canvas-colored gaps. There are no continuous grid borders and no borders drawn through the internal board.

Level	Board	Module size	Gap	Board origin (x,y)
1	2 × 2	20	4	(10,10)
2	2 × 7	7	1	(4,24)
3	3 × 3	13	2	(10,10)
4	8 × 8	6	1	(4,4)
5	9 × 9	5	1	(5,5)
6	9 × 9	5	1	(5,5)
7	10 × 10	5	1	(2,2)

For hit testing, the entire square module is clickable, not only its painted sprite pixels.

5. Formal state machine
5.1 Persistent state
current[R][C] : binary cell states
target[R][C]  : immutable binary target states
step_count    : integer, initially 0
won           : Boolean, initially false

Derived render state:

matched[r][c] := current[r][c] == target[r][c]

Transient feedback may store:

last_clicked
pre_click_current
neighbor_coordinates
neighbor_states
vote_kind
post_click_current

Transient feedback does not participate in future votes or victory validation.

There is no selected-cell variable.

5.2 Click update

For a click mapped to logical cell q = (r,c):

Snapshot the complete current matrix.

Collect only existing orthogonal neighbors of q from that snapshot.

Count state-zero and state-one neighbors:

n0 = number of neighbors in state 0
n1 = number of neighbors in state 1

The clicked cell does not contribute to either count.

Compute:

if n1 > n0: next_q = 1
if n0 > n1: next_q = 0
if n0 == n1: next_q = snapshot[q]

Commit current[q] = next_q.

Every other cell retains its snapshot state.

Recompute all match indicators.

Validate victory.

Increment step_count.

No neighbor is updated automatically. A changed cell influences future clicks only after this step() has completed.

For clicks outside a module, and for directional or SPACE actions, only step_count changes.

5.3 Boundary behavior

Corner cells have exactly two voters. They change only when both neighbors agree. Disagreeing corner neighbors are a tie.

Non-corner edge cells have exactly three voters and therefore cannot tie.

Interior cells have four voters and tie on a 2–2 split.

There are no absent-neighbor default values.

5.4 Clicking an already-correct cell

Matched cells remain fully clickable. A player may damage a correct cell if its neighbors hold an opposing majority.

The target tab never votes and never protects the cell from changing.

5.5 Win condition

After each action, the level succeeds exactly when:

for every logical coordinate (r,c):
    current[r][c] == target[r][c]

The validator ignores:

animation phase;

vote highlights;

last-click metadata;

click history;

step count, except for the technical cap;

which cells were ever clicked;

whether the final action changed a cell.

After won becomes true, further actions are ignored and the final settled board remains visible.

There is no additional requirement concerning order, number of clicks, remaining occurrences of a state, or intermediate configurations.

6. Feedback and animation

State is committed only by step(). Animation is a deterministic visualization of the old and new snapshots and does not invoke additional state transitions.

Input is locked while the short animation plays. Animation subframes consume no player steps.

Strict majority that changes the cell

All voting neighbors brighten for one render subframe:

state 0 voters use color 10;

state 1 voters use color 11.

The gutter segments between winning voters and the clicked module flash in the winning highlight color. The clicked current symbol fills inward from those sides.

The renderer shows the settled post-click state and updated target bracket.

Strict majority already matching the cell

The same voters brighten, but the current symbol does not refill. A one-subframe green ring around the current symbol confirms that a real strict majority was found.

This distinguishes a valid majority no-change from a tied vote.

Tie

The current symbol temporarily shows a blue-orange split:

left half state 0 blue;

right half state 1 orange.

It then returns to its unchanged settled state. No flow is shown.

Completion and failure

Completion produces one green pulse around the board bounding box after all target brackets turn green.

The game has no hazards or ordinary loss state. If the technical 1024-step cap is reached, the board receives one red perimeter pulse and the episode terminates unsuccessfully.

The settled board always contains all strategically relevant information; remembering animation is never required.

7. Useful implementation invariant

Let:

A(current) = number of orthogonally adjacent cell pairs with equal current states

Whenever a click actually flips its cell, A must strictly increase. A tie or same-state majority leaves A unchanged.

This follows because a flipped cell had strictly more disagreeing than agreeing neighbors before the click and exchanges those two counts afterward.

This is not a separate player rule, but it is a useful automated implementation test. It catches accidental self-voting, diagonal voting, simultaneous updates, and incorrect tie handling.

8. Late-level checkerboard-tree construction

Levels 4–7 use a common consequence of the original rule rather than introducing a new object.

In their initial checkerboards, every cell begins opposite to every orthogonal neighbor. Therefore every cell is initially capable of flipping.

Suppose a target-mismatched cell has not yet been clicked:

an adjacent mismatch that has already been corrected now matches this unclicked cell’s current state;

an adjacent mismatch not yet corrected still opposes it.

For an edge or interior cell, one corrected neighbor still leaves an opposing strict majority. Two corrected neighbors remove that majority: an interior cell ties, while a three-neighbor edge cell receives a majority for its existing state. A corner loses its usable majority after only one adjacent correction.

The authored mismatch masks in Levels 4–7 are connected induced trees. If the unique mismatched board corner is treated as the root and cells are clicked parent-before-child, every non-root cell has exactly one corrected tree neighbor when clicked and therefore flips successfully.

The masks are visible in play because precisely those cells lack green target brackets.

In the diagrams below:

X means current != target;

. means current == target.

The letters themselves are never rendered.

9. Authored levels
Level 1 — Lone Dissenter

Board: 2 × 2
Verified changing depth: 1 click
Initial transient state: no selection, last_clicked = none, won = false

Player-facing discovery

Three cells already have green target brackets. The lone hollow blue cell has an orange target tab and a gray bracket. Its two neighbors are both orange.

Clicking it immediately turns it orange, makes its bracket green, and completes the board. This establishes:

cells are the click targets;

the attached tab is the objective;

a clicked cell copies its neighbors’ majority;

completion means all target brackets are green.

No explanatory legend or symbolic inference is required.

Current
01
11
Target
11
11
Example solution
(1,1)

Cell (1,1) has two existing neighbors, both in state 1, so it changes from 0 to 1.

Necessity and attribution

The board has exactly one mismatch, and one click changes at most one cell. The one-click solution is optimal.

Every visible component participates directly in either the vote or the visible objective. There are no decoys, failure states, ties, or simultaneous changes.

Level 2 — The Cell Does Not Vote

Board: 2 × 7
Verified changing depth: 5 clicks
Initial transient state: no selection, last_clicked = none, won = false

Player-facing discovery

The first useful click is an edge cell in state 0 with three neighbors:

two neighbors in state 1;

one neighbor in state 0.

It changes to 1.

Had the clicked cell counted itself, the vote would have been two votes for each state and it would not have changed. The highlighted voters visibly exclude the clicked cell, making the evidence diagnostic rather than merely compatible with a self-voting rule.

The two upper changes then become support for the lower row, establishing that a newly changed cell immediately votes in later actions.

Current
1101011
1100011
Target
1111111
1111111
Initial mismatch mask
..X.X..
..XXX..
Example solution
(1,3)
(1,5)
(2,3)
(2,5)
(2,4)

Vote details:

(1,3) sees states 1,1,0 and becomes 1.

(1,5) sees states 1,1,0 and becomes 1.

(2,3) now sees states 1,1,0 and becomes 1.

(2,5) now sees states 1,1,0 and becomes 1.

(2,4) sees three state-1 neighbors and becomes 1.

Necessity and attribution

There are five mismatches, so at least five state-changing clicks are required. The listed sequence changes every mismatched cell exactly once and is optimal.

All intended clicks are non-corner edge clicks with three voters, so the teaching path introduces no tied vote. The first result specifically distinguishes neighbor-only voting from plausible self-voting.

The lower middle cell cannot be completed before support has been established on both sides, so the level is not solved by five independent attribute adjustments.

Level 3 — Split Center

Board: 3 × 3
Verified solution: 3 changing clicks
Blind discovery path: 4 actions including one diagnostic tie
Initial transient state: no selection, last_clicked = none, won = false

Player-facing discovery

The central cell is a visible mismatch. Its four neighbors are evenly divided: the horizontal pair is state 1, while the vertical pair is state 0.

Clicking the center produces the split-vote feedback and no state change.

The top edge mismatch is nearby and has three voters, two of which are state 1. Clicking it cannot tie and changes it to 1. The center now has a 3–1 majority and can change.

This cleanly teaches both tie behavior and boundary leverage.

Current
101
101
101
Target
111
111
111
Initial mismatch mask
.X.
.X.
.X.
Blind discovery path
(2,2)   tied 2–2; no change
(1,2)   changes to 1
(2,2)   now changes to 1
(3,2)   changes to 1

A player who already understands ties can omit the first probe and solve in three clicks.

Necessity and attribution

The three state-changing clicks are optimal because exactly three cells differ from their targets.

The initial center click cannot be explained by an older “majority always changes the cell” hypothesis: its split feedback explicitly reveals the tie. The edge click then changes under the same unchanged voting rule, showing that boundary geometry—not a special edge exception—removes the possibility of a tie.

All six initially correct cells are stable or tied under the starting configuration, making exploratory clicks relatively forgiving.

Level 4 — One-Way Front

Board: 8 × 8
Verified optimal depth: 27 changing clicks
Initial transient state: no selection, last_clicked = none, won = false

What this level adds

This is the first checkerboard-ordering puzzle.

Initially, every orthogonal pair disagrees, so every cell appears locally ready to change. The target-mismatched cells form one winding path. Correcting a path cell changes it from a supporter of its next neighbor into a vote for that neighbor’s current state.

The top-left corner is the local impasse that teaches the correct direction. If (1,2) is corrected before (1,1), the corner’s two neighbors disagree and the corner can no longer move. The player receives an immediate tie rather than an abstract rejection.

The useful front must therefore begin at the corner and advance continuously along the path.

Current
01010101
10101010
01010101
10101010
01010101
10101010
01010101
10101010
Target
10101011
10101000
00101011
11101010
00101011
10101000
01101011
10101010
Initial mismatch mask
XXXXXXX.
......X.
.XXXXXX.
.X......
.XXXXXX.
......X.
..XXXXX.
........
Example solution
(1,1) (1,2) (1,3) (1,4) (1,5) (1,6) (1,7)
(2,7)
(3,7) (3,6) (3,5) (3,4) (3,3) (3,2)
(4,2)
(5,2) (5,3) (5,4) (5,5) (5,6) (5,7)
(6,7)
(7,7) (7,6) (7,5) (7,4) (7,3)

Every listed click changes its cell directly to its target.

Necessity and anti-degeneracy

The target differs in 27 cells, and a click changes at most one cell. The solution therefore meets a 27-click lower bound and is optimal.

The mismatch graph has 27 vertices, 26 orthogonal edges, and one connected component: it is a path. Its only included board corner is (1,1).

In any optimal sequence, (1,1) must be clicked before its path neighbor. Every later path cell must have exactly one earlier corrected path neighbor. Starting a second front elsewhere eventually puts two corrected neighbors around an unclicked cell, removing its usable majority.

The layout is intentionally a single clean path because this level isolates the new “corrections consume support” consequence. Levels 5–7 immediately replace the path with branching dependency networks.

Level 5 — Junctions First

Board: 9 × 9
Verified optimal depth: 46 changing clicks
Initial transient state: no selection, last_clicked = none, won = false

What this level adds

The mismatch path now branches repeatedly.

The root begins in state 1 and must become 0, demonstrating that the ordering principle is symmetric between the two states.

At the first major junction, (1,4) has two target-mismatched descendants, (1,5) and (2,4). Both descendants are initially flippable. If the player corrects both before the junction, they become two votes for the junction’s current state, and (1,4) no longer changes.

The player must therefore correct junctions before spreading into their branches. Once a junction is secured, its different branches may be advanced in either order or interleaved.

Current
101010101
010101010
101010101
010101010
101010101
010101010
101010101
010101010
101010101
Target
010101011
010001010
010101010
010101110
010101010
011101010
010101010
010100010
110101011
Initial mismatch mask
XXXXXXXX.
...X.....
XXXXXXXXX
......X..
XXXXXXXXX
..X......
XXXXXXXXX
.....X...
.XXXXXXX.

The mask is one connected induced tree rooted at (1,1).

Example solution by depth layers

Click every depth layer before proceeding to the next. Cells within one layer may be clicked in any order.

D0:  (1,1)
D1:  (1,2)
D2:  (1,3)
D3:  (1,4)
D4:  (1,5), (2,4)
D5:  (1,6), (3,4)
D6:  (1,7), (3,3), (3,5)
D7:  (1,8), (3,2), (3,6)
D8:  (3,1), (3,7)
D9:  (3,8), (4,7)
D10: (3,9), (5,7)
D11: (5,6), (5,8)
D12: (5,5), (5,9)
D13: (5,4)
D14: (5,3)
D15: (5,2), (6,3)
D16: (5,1), (7,3)
D17: (7,2), (7,4)
D18: (7,1), (7,5)
D19: (7,6)
D20: (7,7), (8,6)
D21: (7,8), (9,6)
D22: (7,9), (9,5), (9,7)
D23: (9,4), (9,8)
D24: (9,3)
D25: (9,2)

Every non-root cell has exactly one already-corrected mismatch neighbor when clicked. Its other existing neighbors still hold the opposing checkerboard state, so it receives a strict majority for its target.

Necessity and anti-cleanliness

There are 46 mismatches, giving a 46-click lower bound. The mismatch graph has 46 vertices and 45 edges and is connected, so it is a tree. The layer sequence reaches the target in exactly 46 clicks.

The authored instance is not a row-by-row recipe:

connectors occur at columns 4, 7, 3, and 6;

each connector creates two spatially separated arms;

some lower branches become available while upper branches remain unfinished;

several same-depth cells can be interleaved;

correcting both sides of an uncorrected junction creates a physically visible local block.

The solution is a partial order over branches rather than one fixed coordinate list.

Level 6 — Keep a Spark

Board: 9 × 9
Verified optimal depth: 57 changing clicks
Initial transient state: no selection, last_clicked = none, won = false

What this level adds

This level makes the “a state cannot arise from nowhere” consequence necessary.

The starting checkerboard contains 40 state-1 cells:

39 of them currently mismatch and ultimately need to become 0;

only (5,4) is an already-correct state-1 cell;

18 current state-0 cells must eventually become 1.

A tempting plan is to click all 39 orange mismatches first. That leaves only the single orange cell at (5,4). No other cell then has a strict state-1 majority, so none of the 18 required new orange cells can be created. The target is unreachable from that configuration.

The immediate local version appears at the root. (1,1) is a blue corner targeting orange. Its two orange neighbors are both themselves orange-to-blue mismatches. Consuming either support first makes the corner tie. The player must grow a new orange cell before spending its old orange voters.

The same state can therefore be:

a target value to create;

temporary support that must be preserved;

a target mismatch that should be consumed later.

No operational exception is involved.

Current
010101010
101010101
010101010
101010101
010101010
101010101
010101010
101010101
010101010
Target
101010100
000000000
100010001
000000000
101110101
000000000
100010001
000000000
001010100
Initial mismatch mask
XXXXXXXX.
X.X.X.X.X
XX.XXX.XX
X.X.X.X.X
XXX.XXXXX
X.X.X.X.X
XX.XXX.XX
X.X.X.X.X
.XXXXXXX.

The lone initial state-1 cell outside the mismatch tree is (5,4).

Example solution by depth layers

Click layers in order; coordinates inside a layer may be clicked in any order.

D0:  (1,1)

D1:  (1,2), (2,1)

D2:  (1,3), (3,1)

D3:  (1,4), (2,3), (3,2), (4,1)

D4:  (1,5), (5,1)

D5:  (1,6), (2,5), (5,2), (6,1)

D6:  (1,7), (3,5), (5,3), (7,1)

D7:  (1,8), (2,7), (3,4), (3,6), (4,3),
     (4,5), (6,3), (7,2), (8,1)

D8:  (5,5)

D9:  (5,6), (6,5)

D10: (5,7), (7,5)

D11: (4,7), (5,8), (6,7), (7,4), (7,6), (8,5)

D12: (5,9), (9,5)

D13: (4,9), (6,9), (9,4), (9,6)

D14: (3,9), (7,9), (9,3), (9,7)

D15: (2,9), (3,8), (7,8), (8,3),
     (8,7), (8,9), (9,2), (9,8)
Necessity and anti-cleanliness

The board contains 57 mismatches. The mismatch graph is a connected tree with 57 vertices and 56 edges. The listed route changes each mismatch once, so 57 clicks are both sufficient and necessary.

The matched cell (5,4) is not decorative. It remains a state-1 voter beside the central branches and is the only old orange cell not scheduled for consumption. Clicking it as part of an “erase all orange” plan destroys the final source state entirely.

The dependency geometry is not a linear chain:

the first half grows several fronts from the upper-left;

those fronts converge near the central preserved voter without forming a mismatch cycle;

lower and right branches become available at uneven depths;

consuming old state-1 cells too early prevents distant target-1 cells from ever acquiring a majority.

Level 7 — The Living Tree

Board: 10 × 10
Verified optimal depth: 69 changing clicks
Initial transient state: no selection, last_clicked = none, won = false

Final composition

This level uses only the original cell rule, but combines all earlier consequences:

the clicked cell does not vote;

corners are tie-prone;

edge cells provide reliable 2–1 majorities;

a correction immediately changes later votes;

junctions must be resolved before multiple descendants;

both states must be grown before their existing supporters are consumed;

a target-correct cell may be essential as a voter even though it should never be clicked.

The mismatch tree is much denser than in Level 5. Nearby branches visually appear capable of being processed in a row scan, but offset matched gaps prevent those branches from actually connecting. The player must reason from orthogonal adjacency rather than visual density.

Of the 50 initial state-1 cells:

48 need to become 0;

only (1,10) and (10,1) remain target-correct;

21 initial state-0 cells need to become 1.

If all 48 orange mismatches are consumed before any new orange cell is grown, the two surviving orange corner cells are isolated from one another. Neither supplies a majority anywhere, while the target still needs 21 new orange cells.

The top-left root supplies the immediate first experiment: changing either adjacent orange support first prevents the blue corner from acquiring its orange target.

Current
0101010101
1010101010
0101010101
1010101010
0101010101
1010101010
0101010101
1010101010
0101010101
1010101010
Target
1000100011
0000000000
1010101010
0000000000
1000001000
0001000001
1000000000
0001010101
1000000000
1101000100
Initial mismatch mask
XX.XXX.XX.
X.X.X.X.X.
XXXXXXXXXX
X.X.X.X.X.
XX.X.XXX.X
X.XXX.X.XX
XX.X.X.X.X
X.XXXXXXXX
XX.X.X.X.X
.XXXX.XXX.

The two initial state-1 cells outside the mismatch tree are:

(1,10)
(10,1)
Example solution by depth layers

Click every cell in a depth before proceeding. Within one depth, order is arbitrary.

D0:  (1,1)

D1:  (1,2), (2,1)

D2:  (3,1)

D3:  (3,2), (4,1)

D4:  (3,3), (5,1)

D5:  (2,3), (3,4), (4,3), (5,2), (6,1)

D6:  (3,5), (7,1)

D7:  (2,5), (3,6), (4,5), (7,2), (8,1)

D8:  (1,5), (3,7), (9,1)

D9:  (1,4), (1,6), (2,7), (3,8), (4,7), (9,2)

D10: (3,9), (5,7), (10,2)

D11: (2,9), (3,10), (4,9), (5,6),
     (5,8), (6,7), (10,3)

D12: (1,9), (10,4)

D13: (1,8), (9,4), (10,5)

D14: (8,4)

D15: (7,4), (8,3), (8,5)

D16: (6,4), (8,6)

D17: (5,4), (6,3), (6,5), (7,6),
     (8,7), (9,6)

D18: (8,8)

D19: (7,8), (8,9), (9,8)

D20: (8,10), (10,8)

D21: (7,10), (9,10), (10,7), (10,9)

D22: (6,10)

D23: (5,10), (6,9)

Every listed click changes its cell exactly once to its target state.

Necessity and final anti-cleanliness check

There are 69 target mismatches, so every solution requires at least 69 state-changing clicks. The mismatch graph has:

69 vertices
68 orthogonal edges
1 connected component

It is therefore a tree. The listed route solves it in exactly 69 clicks and is optimal.

Because (1,1) is the tree’s only included board corner, an optimal solution must treat it as the root. For every other tree cell, an optimal sequence must provide exactly one earlier corrected tree neighbor:

zero earlier neighbors would create an additional source;

two earlier neighbors would remove the opposing majority;

a tree with 68 edges cannot support multiple sources without eventually requiring a forbidden two-parent cell.

This does not force one recipe. At many depths, several branches can be interleaved. What is forced is the causal partial order between each junction and its descendants.

The mature puzzle cannot be replaced by a corridor, counter, or checklist:

every cell is directly available from the first frame;

the spatial tree is not presented in use order;

matched gaps determine which apparent branches are actually adjacent;

fixing a correct target value at the wrong time can consume a required voter;

local progress can destroy remote reachability;

both binary states repeatedly alternate between resource, result, and liability.

10. Progression summary
Level	New discovery or pressure	Verified depth
1	Visible target matching and direct majority copying	1
2	The clicked cell does not vote; changed cells vote immediately afterward	5
3	Interior ties and dependable three-neighbor edges	3 changing / 4 discovery
4	Correcting one cell can close the opportunity for another	27
5	Junctions must precede multiple branches; either binary state may propagate	46
6	Existing instances of a state are a consumable resource that must seed replacements	57
7	Dense branch ordering, state preservation, asymmetric gaps, and interleaved fronts	69

No new interactable object category is introduced after Level 1. All later depth comes from the same cells, the same target tabs, and the same orthogonal majority update.

11. Implementation verification checklist

An implementation should pass all of the following deterministic checks.

Clicking an edge cell with neighbor states 1,1,0 changes it to 1 even when its own current state is 0.

Clicking an interior cell with neighbor states 0,0,1,1 leaves it unchanged and reports a tie.

Clicking a corner whose two neighbors disagree leaves it unchanged.

Clicking a cell whose strict majority already equals its current state leaves it unchanged but reports a strict-majority confirmation, not a tie.

Target tabs never enter vote counts.

Only the clicked cell may change during one action.

A state-changing click strictly increases the number of equal orthogonal pairs.

The four late mismatch masks have graph sizes:

Level 4: (vertices, edges) = (27,26)

Level 5: (46,45)

Level 6: (57,56)

Level 7: (69,68)

Simulating each listed solution reaches its exact target matrix.

Every listed click in Levels 1, 2, and 4–7 changes the clicked cell. Level 3’s first discovery click is the sole deliberate tied no-change action.

The initial and final equal-adjacency counts are:

Level 1:  2  -> 4
Level 2:  12 -> 19
Level 3:  6  -> 12
Level 4:  0  -> 48
Level 5:  0  -> 72
Level 6:  0  -> 86
Level 7:  0  -> 111

Victory depends only on the settled current == target comparison and not on following the authored example route.
