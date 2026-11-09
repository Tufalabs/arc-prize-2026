Cross Press

Exact authored level count: 7

This is a game about turning cells off becoming difficult because each press flips an overlapping cross rather than one cell.

Cross Press is a deterministic click-only puzzle. Every visible module is a pressable cell, every cell is either lit or dark, and the entire objective is to leave the board completely dark.

There are no hidden properties, obstacles, counters, timers, move limits, or special cells. The same operation applies everywhere. Difficulty comes entirely from overlapping cross footprints, deliberate presses on dark cells, and the reduced footprints available along boundaries.

1. Player-facing rules

A click flips:

the clicked cell;

its existing orthogonal neighbor above;

its existing orthogonal neighbor below;

its existing orthogonal neighbor to the left;

its existing orthogonal neighbor to the right.

Diagonal cells never change. Effects do not wrap across the board.

An interior click therefore flips five cells. An edge click flips four. A corner click flips three.

The state of the clicked cell does not alter the operation. Lit and dark cells are equally pressable.

The player wins immediately when every cell is dark.

Meaningful action subset

Only direct cell clicks are used.

UP, DOWN, LEFT, RIGHT, and SPACE are accepted by the harness but are inert no-ops.

A click inside a cell module presses that cell.

A click in a gutter, outside the board, or on the outer frame is a no-op.

There is no persistent cursor or selected object.

Harness-level reset is available, but there is no reset object inside the grid.

Clicking belongs to the central mechanic because the chosen click location is the center of the cross footprint. Movement would add navigation without adding a relevant decision.

2. Exact logical model

Rows are numbered from top to bottom starting at 1. Columns are numbered from left to right starting at 1.

For a board of height H and width W, the persistent gameplay state is:

lit[H][W]            boolean
solved               boolean
last_pressed         optional (row, column), rendering feedback only
last_affected        set of coordinates, rendering feedback only

Every level begins with:

solved = false
last_pressed = null
last_affected = empty

There is no persistent selection state:

selected_cell = null

For a valid click at (r,c), define:

affected(r,c) =
    {(r,c), (r-1,c), (r+1,c), (r,c-1), (r,c+1)}
    intersected with the board

All members of affected(r,c) toggle simultaneously:

new_lit[i][j] =
    old_lit[i][j] XOR ((i,j) is in affected(r,c))
step() resolution order

For every action:

Discard the previous last_pressed and last_affected feedback.

If solved is already true, leave all gameplay state unchanged.

If the action is a valid cell click, compute its affected set from board geometry.

Toggle all affected cells simultaneously from the pre-action state.

Set last_pressed and last_affected.

Set solved = true exactly when no entries of lit remain true.

Render the resulting state.

Keyboard actions and invalid clicks perform steps 1, 2, 6, and 7 but do not alter lit.

Consequences of the rule

Press footprints do not depend on state, so:

pressing the same cell twice restores every cell it affected;

press order does not affect the final board;

a cell pressed an even number of times has no net effect;

a cell pressed an odd number of times is equivalent to pressing it once;

overlapping footprint cells cancel when toggled twice.

These are consequences of the single toggle rule, not additional exceptions.

3. Objective and validator

The only success predicate is:

solved == true
iff
for every cell (r,c): lit[r][c] == false

The validator ignores:

click order;

click history;

last_pressed;

last_affected;

step count;

how many times a cell was pressed before its final parity;

whether presses were made while their centers were lit or dark.

Allowed:

experimentation;

no-op inputs;

pressing a cell twice to undo it;

solving with any ordering of the required odd-parity presses.

There is no failure state and no move budget. Once solved, the board is locked and subsequent actions are no-ops.

4. Rendering contract
Canvas and logical board

The render canvas is exactly 64×64 integer-color pixels.

Each logical cell is a connected 5×5 module. Adjacent modules are separated by one pixel of negative space.

For a board of width W and height H:

content_width  = 6W - 1
content_height = 6H - 1

x0 = floor((64 - content_width) / 2)
y0 = floor((64 - content_height) / 2)

Cell (r,c) occupies:

x = x0 + 6(c-1) through x0 + 6(c-1) + 4
y = y0 + 6(r-1) through y0 + 6(r-1) + 4

Only those 5×5 rectangles are cell hit regions.

A one-pixel outer board frame is drawn immediately around the full content rectangle. There are no additional region dividers or decorative borders inside the board.

Cell silhouette

Every cell uses this 5×5 structure:

BBBBB
BSSSB
BSSSB
BSSSB
BBBBB

B is the module perimeter. S is the visible state center.

This makes dark cells just as obviously clickable as lit cells.

Colors

Only the following integer IDs are used:

ID	Use
2	ordinary cell perimeter
3	outer board frame
4	canvas background and one-pixel gutters
5	dark cell center
10	affected-neighbor pulse
11	lit cell center
12	pressed-center pulse
14	completion frame

A dark cell has perimeter 2 and center 5.
A lit cell has perimeter 2 and center 11.

No text, digits, arrows, or legends are rendered in the game.

Press feedback

A valid click commits the logical toggles immediately. The returned frame then shows:

the pressed cell perimeter in color 12;

every other affected cell perimeter in color 10;

all 3×3 centers already displaying their new persistent state.

The pulse lasts for that returned frame only. It is cleared on the next step() and consumes no additional action.

There are no independent animation ticks. The settled state remains fully readable under the pulse.

Completion feedback

When the board becomes all dark:

the outer board frame becomes color 14;

every cell perimeter becomes color 14;

all centers remain visibly dark in color 5.

This completion frame remains stable. It provides positive goal evidence without adding a separate target panel.

5. Progression overview

“Solution depth” below means the number of cells that must be pressed an odd number of times in the unique duplicate-free solution.

Level	Size	Main discovery or composition	Solution depth
1	3×3	One click leaves the whole board dark	1
2	7×7	Overlapping crosses cancel collateral changes	5
3	7×7	A correct press may have to begin on darkness	4
4	7×7	Dark edge cells provide clipped four-cell footprints	27
5	8×8	Corners provide three-cell elbows with contrasting roles	34
6	8×8	A boundary tail becomes a liability coupling two regions	43
7	8×10	Dense global composition of overlap, darkness, edges, and corners	53

Level 3 is intentionally one click shorter than level 2. Forcing every duplicate-free solution to begin on a dark cell produces cleaner evidence than padding the tutorial with unrelated presses.

Solution-mask notation

In the authored specifications:

initial-board rows use 1 for lit and 0 for dark;

solution masks use X for “press once” and . for “do not press.”

For levels 4–7, the exact example replay is:

scan solution-mask rows from top to bottom;

within each row, scan left to right;

click every X once.

Because presses commute, this row-major order is only a reproducible implementation test, not a hidden intended ordering.

6. Authored levels
Level 1 — Single Impression

Board: 3×3
Initial lit count: 5
Intended solution depth: 1
Teaches: click affordance, full interior cross, all-dark objective.

Player-facing discovery

The first board is itself shaped like one press footprint. The only cell lying at the center of all five lights is (2,2).

Clicking it immediately darkens every cell and turns the board frame green. The visible before-and-after establishes both the operation and the objective without a textual legend.

There are no decoys or simultaneous state changes. The central press is an interior press, so its footprint is the complete five-cell cross.

Initial state
010
111
010

Lit cells:

(1,2)
(2,1) (2,2) (2,3)
(3,2)

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
...
.X.
...
Example replay
click (2,2)

The post-click state is:

000
000
000
Attribution and necessity check

The one useful action causes one immediately attributable five-cell change.

No edge or corner press is used in the solution.

A center-only mistaken rule predicts that four lights should remain, so the result directly distinguishes the actual cross rule.

The level cannot be solved while ignoring the cross effect.

The completion frame is the only additional visual response.

Level 2 — Diagonal Echo

Board: 7×7
Initial lit count: 9
Intended solution depth: 5
Teaches: overlapping footprints and cancellation.
Reuses: full interior crosses and the all-dark objective.

Player-facing discovery

A sparse diagonal runs through the board, but the two ends carry small side branches. The five conspicuous diagonal cells (2,2) through (6,6) are all lit and all lie away from the boundary.

Clicking (2,2) does not simply remove one light. It removes three existing lights while creating two adjacent lights. The remaining shape resembles the same disturbance shifted down and right, suggesting the next diagonal center.

Successive diagonal presses share two footprint cells. Those shared cells toggle twice and cancel, allowing collateral changes from one press to be repaired by the next.

This is a clean overlap tutorial:

all required centers begin lit;

all required centers use full five-cell crosses;

no boundary clipping or deliberate dark-center press is required.

Initial state
0100000
1100000
0010000
0001000
0000100
0000011
0000010

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
.......
.X.....
..X....
...X...
....X..
.....X.
.......
Example replay
(2,2) → (3,3) → (4,4) → (5,5) → (6,6)

After the first click at (2,2), the state is:

0000000
0010000
0110000
0001000
0000100
0000011
0000010

The newly lit cells at (2,3) and (3,2) are precisely the two cells shared with the next diagonal cross. Pressing (3,3) toggles both again.

Attribution and necessity check

Every required center is initially lit.

Every required press is interior and affects exactly five cells.

Consecutive required crosses overlap in exactly two cells.

Treating each press as an isolated local clear cannot explain the shifted residual pattern.

The board contains no unrelated components or optional lights.

The five-click solution is unique up to ordering and duplicate-pair insertions.

Level 3 — Dark Hub

Board: 7×7
Initial lit count: 6
Intended solution depth: 4
Teaches: pressing a dark cell deliberately.
Reinterprets: overlap cancellation as a cause of dark press centers.

Player-facing discovery

Six lights surround a compact dark T-shaped region. Clicking any visible light produces spill away from the cluster. The dark cell (4,4) lies at the geometric hub of the lower three lights and between the two lateral lights.

Clicking (4,4) appears locally unfavorable:

the lit cell (5,4) turns dark;

four dark cells turn lit.

The result is nevertheless much more structured: it exposes three complete cross footprints centered at (3,4), (4,3), and (4,5).

The crucial evidence is stronger than merely including a dark cell in the solution: all four odd-parity solution centers are dark in the initial frame. Consequently, every duplicate-free correct first move must click darkness.

No boundary press participates, so the observation cannot be explained by clipped edge behavior.

Initial state
0000000
0001000
0000000
0100010
0011100
0000000
0000000

Lit coordinates:

(2,4)
(4,2) (4,6)
(5,3) (5,4) (5,5)

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
.......
.......
...X...
..XXX..
.......
.......
.......

The required centers are:

(3,4), (4,3), (4,4), (4,5)

All four are initially dark.

Example replay

A pedagogically useful order is:

(4,4) → (3,4) → (4,3) → (4,5)

After clicking the initially dark hub (4,4):

0000000
0001000
0001000
0111110
0010100
0000000
0000000

The other three required crosses are now locally visible.

Attribution and necessity check

Any duplicate-free solution must begin on a dark cell.

All four required presses are interior five-cell crosses.

The level cannot be solved by pressing only cells that are initially lit.

The apparent increase from six to nine lights after the hub press visibly disproves a greedy “always reduce the number of lights” hypothesis.

The subsequent structured residual makes the experiment productive rather than punitive.

No boundary or corner rule is required.

Level 4 — Broken Frame

Board: 7×7
Initial lit count: 15
Intended solution depth: 27
Introduces: edge leverage and four-cell clipped crosses.
Reuses: deliberate dark presses and overlap cancellation.

Player-facing discovery

The board contains a nearly closed inner frame whose upper-left cell is missing. A full-cross strategy aimed only at the lit frame repeatedly spills into both sides of the contour.

The dark top-edge cell (1,4) is directly aligned above the upper lit run. Clicking it gives immediate boundary evidence:

(1,4): dark → lit
(1,3): dark → lit
(1,5): dark → lit
(2,4): lit  → dark

There is no cell above (1,4), so only four cells change. Nothing wraps to the bottom.

This press is part of the unique solution. The level requires sixteen boundary presses, but deliberately requires no corner presses. It therefore isolates the four-cell edge footprint before the three-cell corner footprint appears.

The missing cell at (2,2) breaks the otherwise symmetric frame. The player cannot mirror a solution from one side to the other; boundary work on the left and right has different consequences.

Initial state
0000000
0011110
0100010
0100010
0100010
0111110
0000000

The lit contour consists of:

upper arm (2,3) through (2,6);

side arms at columns 2 and 6, rows 3 through 5;

lower arm (6,2) through (6,6).

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
.XXXXX.
X.XXX.X
XX.X..X
XXX....
XX...XX
X...XX.
.XX.X..
Canonical row-major replay

Click each X in the mask once, scanning top to bottom and left to right. This produces exactly 27 valid clicks.

The first five clicks are therefore:

(1,2), (1,3), (1,4), (1,5), (1,6)

The final five are:

(6,6), (7,2), (7,3), (7,5)

with (6,5) immediately before (6,6) in the full row-major scan.

Intended reasoning

A useful decomposition is not “clear the top, then the sides, then the bottom.” Instead, the player reasons about overlapping footprint families:

top-edge presses alter the upper arm and prepare cells used by second-row presses;

left-edge presses compensate for the missing (2,2) frame cell;

interior presses cancel spill shared by the two vertical sides;

bottom-edge presses close residuals created by both side systems.

These families overlap. None is independently solvable and permanently finished before the others.

Necessity and anti-cleanliness check

Sixteen of the 27 required presses lie on an edge.

None lies at a corner, cleanly separating edge clipping from corner clipping.

Eighteen required cells are dark in the initial state.

Ignoring boundary cells cannot win because the press-parity solution is unique.

Restoring the missing frame cell (2,2) changes 21 entries of the unique solution mask; the notch is not decorative.

The frame is spatially asymmetric and cannot be replaced by a sequence of independent devices.

The level has no move limit, so the player can click an edge cell twice to inspect and undo its footprint.

Level 5 — Opposed C-Frames

Board: 8×8
Initial lit count: 33
Intended solution depth: 34
Introduces: three-cell corner leverage.
Combines: dark presses, edge clipping, cancellation, and coupled contours.

Player-facing discovery

Two opposing C-shaped contours occupy the same board:

an outer C uses the top, left, and bottom sides;

an inner reverse C uses two interior horizontal arms and the right side.

A notch at (1,4) breaks the outer top arm.

The dark top-right corner (1,8) sits beside the lit endpoint (1,7). Clicking it visibly toggles only:

(1,8), (1,7), (2,8)

This is the first required corner press. The footprint is a three-cell elbow, not a four-cell edge cross and not a wrapped cross.

The same corner rule then takes contrasting roles:

(1,1) begins lit and is pressed as a direct boundary clear;

(1,8) begins dark and is pressed to create a temporary vertical endpoint;

(8,8) also begins dark and is needed to resolve the lower reverse-C arm;

(8,1) is not required.

The operational rule never changes; only position and surrounding state change its strategic meaning.

Initial state
11101110
10000000
10111111
10000001
10000001
10111111
10000000
11111110

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
XXXX..XX
X.....X.
X.XX.X..
X.XXX.XX
X.X..X.X
X....XXX
XX.X....
...XXXXX
Canonical row-major replay

Click every X once in row-major order. The replay begins:

(1,1), (1,2), (1,3), (1,4), (1,7), (1,8),
(2,1), (2,7), ...

and ends:

(8,4), (8,5), (8,6), (8,7), (8,8)

Total: 34 meaningful clicks.

Intended reasoning

The two C-frames are not independent shapes:

presses near column 1 alter both the outer vertical arm and cells that receive spill from the inner arms;

presses near column 8 jointly control the inner vertical arm and the endpoints of both horizontal contours;

the top notch prevents a symmetric treatment of top and bottom;

corner elbows control endpoint parity but can create debt on the adjacent edge.

A plan that tries to erase the outer C completely before touching the inner C causes the shared seam cells to be relit.

Necessity and anti-cleanliness check

Twenty required presses are boundary presses.

Three required presses are corners.

Twelve required centers begin dark.

The corner rule is necessary, not merely visible.

The three required corners begin in contrasting states.

Filling the notch at (1,4) changes 22 of the 64 solution-mask entries.

Neither C can be removed or solved as an isolated checklist item.

The solution is a coupled parity network rather than a left-to-right device route.

Level 6 — Hooked Chamber

Board: 8×8
Initial lit count: 30
Intended solution depth: 43
New interpretation: a familiar boundary endpoint becomes a liability.
Combines: all established footprints and global net-effect reasoning.

Player-facing discovery

A large top-left hook is connected to an inner rectangular chamber. The chamber extends into a one-cell tail at (8,7) on the bottom edge.

The tail resembles a final isolated endpoint, but treating it as an ordinary light to erase is misleading. Clicking the required tail cell (8,7) produces:

(8,7): lit  → dark
(7,7): lit  → dark
(8,6): dark → lit
(8,8): dark → lit

It clears two lights while creating two new bottom-edge lights, including one at the corner. The edge footprint that served as precise leverage in level 4 now transfers the unresolved state sideways.

The player must treat the tail as part of the chamber’s boundary parity, not as a detachable final light.

Initial state
11111100
11100000
10111110
10100010
10100010
10100010
00111110
00000010

The principal authored structures are:

top arm: row 1, columns 1–6;

left hook and connector: column 1 through row 6, plus row 2, columns 1–3;

chamber: rows 3–7, columns 3–7;

bottom tail: (8,7).

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
XXX.XXXX
X.XXX.X.
X..XXX..
XXX.XXX.
.XXXX.XX
XXXXXX..
X.X...XX
.XXX.XX.
Canonical row-major replay

Click every X once in row-major order.

The replay begins with:

(1,1), (1,2), (1,3), (1,5), (1,6), (1,7), (1,8)

and ends with:

(8,2), (8,3), (8,4), (8,6), (8,7)

Total: 43 meaningful clicks.

Intended reasoning

The level supports several overlapping hypotheses, but visible experiments distinguish them:

Clearing the outer hook alone relights its connector through chamber presses.

Clearing the chamber alone leaves boundary residuals inherited from the hook.

Treating the tail as independent moves its light sideways rather than removing it.

Presses along rows 3–6 influence both the chamber contour and the left-hook seam.

The useful abstraction is the net effect of a set of crosses. The player can still reason locally, but local regions must be selected according to which neighboring footprints will later cancel their spill.

Necessity and anti-cleanliness check

Forty-three distinct odd-parity presses are required.

Twenty-two required centers begin dark.

Nineteen required presses are on the boundary.

Both top corners are required, while neither bottom corner is required.

Removing only the tail light (8,7) changes 27 entries of the solution mask and reduces the solution depth from 43 to 32.

The tail therefore creates a real global dependency rather than visual decoration.

The layout cannot be replaced by a corridor, ordered station sequence, or independent collection of contours.

Level 7 — Crossed Hooks

Board: 8×10
Initial lit count: 39
Intended solution depth: 53
Final composition: overlap cancellation, deliberate darkness, edge and corner leverage, asymmetric coupling, and reasoning by total footprint.

Player-facing discovery

The final board contains two unequal hooks that cross through a narrow interior seam:

a long upper-left hook follows the top and left boundaries;

a right-facing hook begins on row 3, reaches the right boundary, descends, and returns along the bottom;

a lower shelf on row 6 reaches from the left boundary into the center;

the lit bridge cell (4,3) links the apparent structures through overlapping footprints.

The bridge is not merely a connector in the drawn contour. Clicking the required (4,3) changes:

(4,3): lit  → dark
(3,3): lit  → dark
(4,2): dark → lit
(4,4): dark → lit
(5,3): dark → lit

Two lights disappear and three appear. The action transfers unresolved parity from the upper arm into the seam and lower chamber. This immediately disproves a greedy local-clearing approach while producing a predictable physical consequence.

All four corners are required:

(1,1) and (8,10) begin lit;

(1,10) and (8,1) begin dark.

Thus the same three-cell corner footprint must be used both directly and indirectly in four different surrounding contexts.

Initial state
1111111100
1000000000
1011111111
1010000001
1000000001
1111110001
0000000001
0011111111

Auxiliary state:

solved = false
selected_cell = null
last_pressed = null
last_affected = empty
Unique press mask
XXX.XXXXXX
X.XXX...X.
XXXXX.X...
.XXX.X...X
XXXXXX..X.
X...XXXXXX
XX.X.XXX.X
X..XXX.XXX
Canonical row-major replay

Click every X once in row-major order.

The first row contributes:

(1,1), (1,2), (1,3),
(1,5), (1,6), (1,7), (1,8), (1,9), (1,10)

The final row contributes:

(8,1), (8,4), (8,5), (8,6), (8,8), (8,9), (8,10)

Total: 53 meaningful clicks.

Intended reasoning

The mature puzzle forms a dependency network rather than a chain:

top-boundary presses affect the upper hook and establish cells later shared with the right hook;

left-boundary presses influence both the upper hook and lower shelf;

right-boundary presses join the upper and lower arms of the second hook;

bottom presses interact with the shelf through the right-hand return path;

the bridge at (4,3) changes which cancellations are possible between the upper and lower structures;

corner presses control endpoints but also create neighboring edge obligations.

No contour appears in the exact order it should be handled. The useful moves are distributed around all four boundaries and throughout the interior.

A player may reason by local cross sums, row effects, reversibility, or experimentally maintained hypotheses. All strategically relevant state remains visible after every action.

Necessity and anti-cleanliness check

Fifty-three distinct odd-parity presses are required.

Twenty-six required centers begin dark.

Twenty-four required presses lie on the boundary.

All four corners are necessary.

Removing the bridge light (4,3) changes 31 of the 80 solution-mask entries and lowers the solution depth from 53 to 44.

The bridge is therefore a global coupling feature, not a decorative imperfection.

The unequal hook lengths prevent rotational or reflection copying.

Solving one visible contour in isolation necessarily disturbs another because their cross footprints overlap through the seam.

There is no timer, counter, hidden condition, or special-case behavior producing the difficulty.

7. Progression and epistemic-solvability audit
Foundational attribution
Level 1

One central action produces one complete cross change and success. The objective and action effect are simultaneously visible.

Level 2

Only one new causal property is required: different crosses can overlap and their shared changes cancel. Every solution center is lit and interior.

Level 3

Only one new causal interpretation is required: a dark center can be the correct press. Every solution center is initially dark, making the evidence unavoidable in any duplicate-free solution.

Later mechanic development
Boundary rule reuse

Level 4 uses noncorner edge clipping as a precision tool.

Level 5 sharpens that observation into three-cell corner elbows.

Level 6 turns a bottom-edge endpoint into a liability that transfers light.

Level 7 requires all four corners under different initial states.

Dark-center reuse

Level 3 requires darkness as the first correct action.

Level 4 uses dark cells in the outer band to affect an inner contour.

Level 5 uses dark corners to manage contour endpoints.

Level 7 places 26 required presses on initially dark cells throughout a dense dependency network.

Overlap reuse

Level 2 uses diagonal overlaps to cancel harmless collateral.

Level 3 uses adjacency overlap to make all required centers initially dark.

Level 5 couples opposing contours.

Levels 6 and 7 make local clearing plans fail because later footprints re-enter earlier regions.

No operational rule changes between these uses.

8. Implementation verification

Each level can be verified over GF(2).

Let A be the board’s cross-incidence matrix:

A[target_cell, pressed_cell] = 1
iff target_cell is the pressed cell or an existing orthogonal neighbor

Let:

b = flattened initial lit matrix
p = flattened authored press mask

Every authored level satisfies:

A p = b  (mod 2)

Therefore:

b XOR A p = 0

The selected board dimensions all have full-rank cross matrices:

Dimensions	Matrix rank
3×3	9 / 9
7×7	49 / 49
8×8	64 / 64
8×10	80 / 80

Consequently, each initial state has one unique press-parity solution. Any other winning action trace can differ only by:

reordering presses;

inserting an even number of additional presses on one or more cells.

This prevents unintended shorter parity solutions while preserving unlimited reversible experimentation.

Required regression tests

An interior click changes exactly five cells.

A noncorner edge click changes exactly four cells.

A corner click changes exactly three cells.

No effect wraps to an opposite edge.

Diagonal cells never change.

All affected cells use the same pre-action state and toggle simultaneously.

Clicking the same cell twice restores the previous logical board.

Clicking a dark cell uses exactly the same footprint as clicking a lit cell.

Keyboard actions and off-module clicks leave lit unchanged.

The feedback pulse never obscures the 3×3 persistent state center.

Success occurs immediately after the step that leaves all cells dark.

The validator ignores pulse state, click order, and action count.

The game contains exactly seven authored levels in the order specified above.
