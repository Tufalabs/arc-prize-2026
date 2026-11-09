Growth Frontier

Exact authored level count: 7

This is a game about adding one edge-adjacent cell becoming difficult because every addition changes both the reachable frontier and the colors that future cells will receive.

The player grows one connected region across an irregular field of target cells. Every inactive frontier cell visibly previews the color it would receive now. Clicking it permanently commits that color, changes the region’s silhouette, and updates nearby previews.

Design note: preserving depth after the tutorial

With only fixed target cells and exact previews, the system has a useful but limiting monotonic property: whenever a target currently matches its preview, it is always safe to activate immediately, and nonadjacent matching targets commute. That makes the original rules exceptionally learnable, but mature boards would eventually reduce to “click every visible match.”

The specification therefore introduces exactly one related cell variant after the three-level foundational curriculum:

Open-ring cells activate, count as support, and update the frontier exactly like target cells, but have no required result color. They still must be activated for completion. Choosing when to use an open-ring cell can rescue a high target, close a low target’s window, or do both in sequence.

This is not a station, switch, or separate subsystem. It is the same growth operation with its own target requirement omitted. All other mechanics remain those of the supplied idea.

1. Player-facing rules
The board

The board is an irregular set of logical square cells on a four-neighbor lattice.

One yellow source cell starts active. Every other playable cell begins inactive and is one of:

A target cell, with a colored target ring and one to four target pips.

Starting in level 4, an open-ring cell, with a broken light-gray ring and no target pips.

Void coordinates are black background and are not cells.

The frontier

An inactive playable cell is on the frontier when at least one of its four edge-neighbors is active. Diagonal contact never counts.

Every frontier cell receives conspicuous white corner brackets. Its center previews what would happen if it were clicked now:

Active edge-neighbors	Result color	Color ID	Center pips
1	Blue	9	1
2	Green	14	2
3	Orange	12	3
4	Magenta	6	4

A non-frontier cell has no preview and no white brackets.

Clicking

Clicking an inactive frontier cell:

Counts its active edge-neighbors using the pre-click state.

Permanently activates the clicked cell.

Gives it the corresponding result color and pip count.

Recomputes the frontier and all previews.

A target cell may be activated while its preview does not match its ring. This is allowed, but the mismatch is permanent and prevents completion.

An open-ring cell accepts any result count. Its center still shows the count it received, and it supports later growth normally.

Objective

Completion occurs exactly when:

Every playable cell is active.

Every target cell’s committed result count equals its target count.

The following are ignored by the validator:

The source cell’s color requirement; it has none.

The committed result count of open-ring cells.

The number of invalid clicks or unused directional actions.

The exact activation order, provided the final target results are correct.

A target mismatch is forbidden for completion. There is no in-level undo or recoloring. The host environment’s ordinary level reset is the recovery mechanism.

There is no energy, timer, move budget, or penalty for experimentation.

2. Formal state-machine contract
2.1 Logical geometry

Coordinates use:

x = 0, 1, ... from left to right.

y = 0, 1, ... from top to bottom.

Four-neighbor offsets: (-1,0), (1,0), (0,-1), (0,1).

Per level:

P             set of playable coordinates
source        one coordinate in P
kind[c]       TARGET or OPEN for every c != source
target[c]     integer 1–4 when kind[c] == TARGET

Void coordinates inside a level’s rectangular bounding box are absent from P.

The active region is always connected because only frontier cells can activate.

2.2 Dynamic state
active                 set of active coordinates
result_count[c]        1–4 for each activated non-source cell
won                    Boolean
selected_cell          always null
feedback               transient render-only feedback record or null

Initial state for every level:

active = {source}
result_count = {}
won = false
selected_cell = null
feedback = null

There is no persistent cursor, selected module, phase counter, queue, budget, or hidden state.

For an inactive cell c:

active_neighbor_count(c) =
    number of coordinates in N4(c) ∩ active

frontier(c) =
    c ∈ P
    and c ∉ active
    and active_neighbor_count(c) > 0

The preview count of a frontier cell is exactly active_neighbor_count(c).

2.3 Action semantics

The provided action space is respected exactly:

UP
DOWN
LEFT
RIGHT
SPACE
CLICK(px, py)

Only clicking has gameplay effect.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic state-preserving no-ops. They do not dismiss feedback early, consume resources, or advance any timer.

Click hit testing

Each logical coordinate owns a module-sized 7×7 pixel hit region. Clicking anywhere in that region addresses that logical coordinate; the player never needs to locate a tiny pixel.

If the click addresses:

An inactive frontier cell: activate it.

An inactive non-frontier playable cell: reject without state change.

An already-active cell: reject without state change.

A void coordinate or position outside the board: reject without state change.

Exact transition order

For step(CLICK(px, py)):

If won == true, preserve state and ignore the click.

Hit-test the click to obtain logical coordinate c, or null.

If c == null or c ∉ P, leave gameplay state unchanged and set outside-board feedback.

If c ∈ active, leave gameplay state unchanged and set active-cell feedback.

Otherwise compute:

n = active_neighbor_count(c)

If n == 0, leave gameplay state unchanged and set missing-contact feedback.

If n > 0, atomically perform:

active ← active ∪ {c}
result_count[c] ← n

Recompute every inactive cell’s frontier status and preview from the new active set.

Set:

won =
    active == P
    and for every TARGET cell c:
            result_count[c] == target[c]

Select the appropriate render animation.

The clicked cell uses the pre-click count. It does not count itself, and no other cell automatically activates.

All active cells count as support, including:

The source.

Correctly matched targets.

Mismatched targets.

Open-ring cells of any committed result color.

2.4 Determinism and reset

There is no randomness.

A reset restores the exact authored initial state. Reset is supplied by the surrounding environment and is not another gameplay action.

3. Rendering contract
3.1 Canvas and placement

Render on a 64×64 canvas using only integer color IDs 0–15.

For a level with bounding width W and height H:

board_pixel_width  = 7W
board_pixel_height = 7H

origin_x = floor((64 - 7W) / 2)
origin_y = floor((64 - 7H) / 2)

Logical cell (x,y) owns the slot:

[origin_x + 7x, origin_x + 7x + 6]
×
[origin_y + 7y, origin_y + 7y + 6]

There is no enclosing board rectangle and no shared internal grid. Modules float against the black background with negative space between their rings.

The diagrams later in this specification are implementation notation only. No letters, digits, coordinates, or text are rendered in the game.

3.2 Palette use
Role	Color ID
Canvas and void	5
Inactive center	4
Active-region connectors	2
Target count 1	9
Target count 2	14
Target count 3	12
Target count 4	6
Open ring	1
Source	11
Frontier brackets	0
Invalid or mismatch pulse	8

Colors 7, 10, 13, and 15 are not needed.

3.3 Cell module

Within a cell’s local 7×7 slot, local coordinates run from (0,0) to (6,6).

Target ring

The target ring is the perimeter of the local 5×5 square from (1,1) through (5,5), filled with the target color.

Target-pip pixels replace ring pixels at:

1 pip:  (3,1)

2 pips: (1,3), (5,3)

3 pips: (3,1), (1,5), (5,5)

4 pips: (3,1), (5,3), (3,5), (1,3)

Pip ink:

count 1 or 4: color 0
count 2 or 3: color 5
Cell center

The interior is the local 3×3 area from (2,2) through (4,4).

Inactive non-frontier: fill with color 4, no center pips.

Inactive frontier: fill with the prospective result color.

Active: fill with the committed result color.

Preview/result pip locations are:

1 pip:  (3,3)

2 pips: (2,3), (4,3)

3 pips: (3,2), (2,4), (4,4)

4 pips: (2,2), (4,2), (2,4), (4,4)

Use the same pip-ink mapping as the target ring.

A correct target therefore has matching ring and center colors. A mismatch retains two visibly different colors.

Open-ring cell

An open-ring cell uses color 1 on the same 5×5 perimeter, but omits the four ring corners:

(1,1), (5,1), (1,5), (5,5)

It has no ring pips. Its center previews and records its actual neighbor count exactly like any other cell.

The four gaps and absence of pips are the visible statement that no result color is required.

Source

The source uses a solid yellow 5×5 square from (1,1) through (5,5), with the center pixel (3,3) in color 0. It has no target pips and is active from the first frame.

3.4 Frontier brackets

Every inactive frontier cell receives color-0 bracket pixels at:

(0,1), (1,0),
(5,0), (6,1),
(0,5), (1,6),
(5,6), (6,5)

No other inactive cell receives these brackets.

This makes the major click targets full-module, conspicuous objects rather than hidden pixel affordances.

3.5 Connected active silhouette

Before drawing cells, draw a color-2 connector between every pair of orthogonally adjacent active cells.

For a horizontal active pair, fill the two-pixel gap between their modules with a 2×3 bridge aligned to the cell centers. For a vertical pair, use a 3×2 bridge.

Connectors are visual only. They make the active region read as one connected shape without drawing a permanent grid.

3.6 Feedback and animation

Animations are deterministic render-only subframes. They consume no actions, budget, or state-machine steps. A new player action is accepted only after the settled frame.

Valid activation

The gameplay state is committed during the step() transition before animation.

Render three subframes:

The clicked center displays only its central result-colored pixel.

The full 3×3 result center appears with result pips.

The active connector appears, white frontier brackets are recomputed, and affected neighboring previews show their new color and pip count.

Only frontier previews whose counts changed receive a one-subframe white pip flash.

Mismatching target activation

Use the valid activation sequence, then pulse the four corners of the clicked module in color 8 for two subframes.

The red pulse is temporary. The permanent evidence is the differing target-ring and center colors.

Non-frontier playable-cell click

State remains unchanged. For two subframes, draw four short color-8 inward-pointing bars at the side midpoints of the clicked module. This communicates that it has no active edge contact.

Active-cell click

State remains unchanged. Pulse its outer ring in color 2 for one subframe.

Void or outside-board click

State remains unchanged. Draw a small color-8 3×3 hollow square centered as closely as possible on the click point for two subframes.

Completion

Set won = true during the final activation step. Then pulse the outer boundary of the entire active silhouette color 0, original, color 0, original over four subframes.

No text, score, or completion glyph appears.

4. Authored-level notation

The maps below are exact logical data:

S  source
1  target count 1
2  target count 2
3  target count 3
4  target count 4
O  open-ring cell
.  void

Digits and O are not rendered as text. They compile to the rings, colors, and pips defined above.

Every listed solution is one mechanically verified successful activation order. Other orders are accepted when they produce the same valid final state.

Meaningful solution depth counts successful cell activations, not rejected clicks, no-op controls, animations, or resets.

5. Level progression overview
Level	Playable cells	Open rings	Meaningful clicks	New evidence
1	4	0	3	Edge-touching frontier and preview-to-result match
2	8	0	7	Waiting can raise a preview from 1 to 2
3	12	0	11	Growing around a dormant cell can produce count 3
4	30	3	29	Open rings accept different result counts; their timing matters
5	42	2	41	A cell can require all four edge-neighbors
6	49	3	48	Offset high targets force interleaved branch preparation
7	56	4	55	Four coupled growth fronts form one dependency network
6. Exact authored levels
Level 1 — First Contact
Purpose

Teach:

Only edge-adjacent inactive cells are clickable.

The center is a prospective result preview.

Clicking commits the preview.

Matching the ring is desirable.

Completion means filling the entire visible playable silhouette correctly.

No other causal rule changes.

Board
      x=012
y=0   ..1
y=1   S11
P = {(0,1), (1,1), (2,1), (2,0)}
source = (0,1)
open cells = {}

Initial interaction state:

active = {(0,1)}
selected_cell = null
feedback = null

Initial frontier:

(1,1): preview count 1, target count 1

Cells (2,1) and (2,0) are visible but initially non-frontier.

Player-facing discovery

The sole white-bracketed module is (1,1). Its blue one-pip center already agrees with its blue one-pip ring.

Clicking (1,1) immediately:

Makes it solid blue.

Connects it visibly to the yellow source.

Removes its frontier brackets.

Causes (2,1) to receive the same blue one-pip preview.

This first click acts as the positive solved example required for a matching objective. The player does not need a textual legend.

A curious click on (2,0) before growth is also safe: it produces the red missing-contact feedback and no state change.

Verified solution
01 (1,1)
02 (2,1)
03 (2,0)

All three activations occur with one active edge-neighbor.

Win validation

After step 3:

active == P
result_count[(1,1)] == 1
result_count[(2,1)] == 1
result_count[(2,0)] == 1

The completion silhouette pulse begins.

Attribution and necessity check

Every visible cell either demonstrates the source, the sole legal frontier action, or a later continuation of the same causal chain. There are no decoys, competing mechanics, timing choices, or misleading affordances.

Random clicking is not a reliable solution: only one cell is initially actionable, and distant clicks visibly fail.

Level 2 — Waiting Makes Two
Purpose

Teach that a cell’s preview is not a fixed property of the ring. It changes when another edge-neighbor joins the active region.

The level demonstrates the same count-2 event twice.

Board
      x=0123
y=0   S1..
y=1   1211
y=2   ..21
P = {
    (0,0), (1,0),
    (0,1), (1,1), (2,1), (3,1),
    (2,2), (3,2)
}
source = (0,0)
open cells = {}

Initial interaction state:

active = {(0,0)}
selected_cell = null
feedback = null

Initial frontier:

(1,0): preview 1, target 1
(0,1): preview 1, target 1

The green two-pip target at (1,1) is initially diagonal to the source and therefore has no preview.

Player-facing discovery

Click either initial blue target.

Suppose (1,0) is clicked first. Cell (1,1) becomes frontier and displays:

target ring: green, 2 pips
preview:     blue, 1 pip

The other source neighbor (0,1) is still a matching blue target. Clicking it raises (1,1) from one active neighbor to two. The center visibly changes from blue/one-pip to green/two-pip.

This rules out plausible mistaken hypotheses such as:

A cell’s result is copied from its ring.

A frontier cell always receives blue.

The first neighboring activation permanently fixes its future color.

The bent tail repeats the same observation at (2,2).

Verified solution

The first two clicks may be exchanged.

01 (1,0)
02 (0,1)
03 (1,1)
04 (2,1)
05 (3,1)
06 (3,2)
07 (2,2)

Key checkpoints:

After step 1:
    preview_count(1,1) = 1

After step 2:
    preview_count(1,1) = 2

Before step 7:
    active neighbors of (2,2) are (2,1) and (3,2)
    preview_count(2,2) = 2
Necessity check

Both source-adjacent target-1 cells are necessary supports for (1,1). Clicking (1,1) after only one support visibly commits the wrong color and prevents completion.

The second count-2 target is not decorative repetition: it confirms that the count rule depends on local geometry, not proximity to the source.

Attribution check

Only one new causal unknown is present: an untouched frontier preview can rise when another edge-neighbor activates. No open rings, count-3 targets, count-4 targets, holes, or timing traps appear.

Level 3 — Three-Sided Bay
Purpose

Teach count 3 and the spatial principle of growing around a dormant intended center.

The player sees one target’s preview progress through counts 1, 2, and 3 before activation.

Board
      x=0123
y=0   1121
y=1   S312
y=2   1122
P = {(x,y) | 0 <= x <= 3 and 0 <= y <= 2}
source = (0,1)
open cells = {}

Initial interaction state:

active = {(0,1)}
selected_cell = null
feedback = null

Initial frontier:

(0,0): preview 1, target 1
(0,2): preview 1, target 1
(1,1): preview 1, target 3
Player-facing discovery

The orange three-pip cell (1,1) touches the source immediately, but its center is initially blue with one pip.

The upper branch (0,0) → (1,0) provides a second active neighbor. The lower branch (0,2) → (1,2) provides a third.

The center therefore changes visibly:

1 active neighbor: blue
2 active neighbors: green
3 active neighbors: orange

The right half is not reachable before (1,1) activates, so the count-3 mechanic cannot be ignored.

Verified solution

The two two-cell support branches may be exchanged.

01 (0,0)
02 (1,0)
03 (0,2)
04 (1,2)
05 (1,1)
06 (2,1)
07 (2,0)
08 (2,2)
09 (3,0)
10 (3,1)
11 (3,2)

Key checkpoints:

Initial:
    preview_count(1,1) = 1

After step 2:
    preview_count(1,1) = 2

After step 4:
    preview_count(1,1) = 3

After (1,1) activates, (2,1) becomes the physical gateway to the right portion.

Necessity check

The source, upper support, and lower support are all required to produce count 3. Neither branch is optional.

Activating (1,1) at count 1 or 2 creates a permanent, visually explicit mismatch. Ignoring the central mechanic leaves the right half unreachable.

Attribution check

There is one new causal consequence: three independently placed edge supports produce the orange result. The board is rectangular only to keep that evidence clean; every cell participates in the solution.

Level 4 — Open Commitments
Purpose

Introduce the one new cell variant: the open-ring cell.

The level provides three contrasting observations:

An open ring can be the only available bridge.

The same open-ring appearance accepts a count-3 result, ruling out “gray means count 1.”

An open ring may need to wait until a neighboring count-1 window is secured, then become support for a count-3 target.

Board
      x=0123456
y=0   ..1111.
y=1   .11O121
y=2   SO33222
y=3   .111.O1
y=4   .122231
y=5   ...11..
source = (0,2)

open cells = {
    (1,2),
    (3,1),
    (5,3)
}

All displayed non-void coordinates are playable.

Initial interaction state:

active = {(0,2)}
selected_cell = null
feedback = null

Initial frontier:

(1,2): open ring, preview count 1

No target cell is initially frontier.

Player-facing discovery
First observation: visibly clickable but ringless

The only white-bracketed cell is the open ring at (1,2). Ordinary known actions cannot progress elsewhere. Clicking it:

Activates it blue with one result pip.

Leaves its broken gray outer ring.

Produces no mismatch pulse.

Opens the upper and lower left branches.

This establishes that the unfamiliar module belongs to the growth system.

Distinguishing observation: it is not secretly a count-1 target

Before solution step 12:

open cell (3,1) has preview count 3
target (3,2) has preview count 2 and target count 3

The direct right-side continuation remains unavailable at its required counts. Clicking (3,1) commits an orange three-pip center inside the same broken gray ring and produces no mismatch.

It immediately raises (3,2) from count 2 to count 3.

This is the necessary distinguishing experiment: the open-ring rule cannot be explained as an ordinary count-1 target with unusual decoration.

Contrasting role: preserve a low window first

Late in the level, (5,3) is an open ring adjacent to:

Blue target (6,3), which must activate with one neighbor.

Orange target (5,4), which ultimately needs three neighbors.

Activating (5,3) before securing (6,3) raises the blue target to count 2 and permanently spoils it.

After (6,3) is fixed, the same open-ring activation raises (5,4) from count 2 to count 3. The open cell is first a liability, then a required support.

Verified solution
01 (1,2)
02 (1,1)
03 (2,1)
04 (2,0)
05 (3,0)
06 (4,0)
07 (4,1)
08 (1,3)
09 (2,3)
10 (3,3)
11 (2,2)
12 (3,1)
13 (3,2)
14 (5,0)
15 (5,1)
16 (4,2)
17 (6,1)
18 (5,2)
19 (6,2)
20 (1,4)
21 (2,4)
22 (3,4)
23 (3,5)
24 (4,5)
25 (4,4)
26 (6,3)
27 (6,4)
28 (5,3)
29 (5,4)

Open-ring committed counts on this path:

(1,2) receives 1
(3,1) receives 3
(5,3) receives 2

Key checkpoints:

Before step 12:
    preview_count(3,2) = 2

After step 12:
    preview_count(3,2) = 3

Before step 28:
    result_count(6,3) is already fixed at 1
    preview_count(5,4) = 2

After step 28:
    preview_count(5,4) = 3
Necessity and anti-cleanliness check

(1,2) is the source’s sole outgoing bridge.

(3,1) supplies the third support needed to cross the central count-3 bottleneck.

(5,3) supplies the third support for (5,4).

The void at (4,3) prevents a direct central sweep and separates the lower convergence from the upper route.

The right-side open ring is visible well before it is safe to use.

The solution is not a corridor or station errand. Progress requires returning between spatially separate frontier branches and interpreting the same open-ring rule first as permission, then as danger, then as support.

Level 5 — Four Around
Purpose

Introduce the magenta four-pip result and require genuine growth around a dormant center.

Count 4 is demonstrated twice:

Once as a central cell surrounded through an asymmetric detour.

Once as a closing cross whose final support is an open-ring cell that had to be delayed.

Board
      x=01234567
y=0   ..1111..
y=1   .112O21.
y=2   11342121
y=3   S211.O41
y=4   .1223111
y=5   ..123221
y=6   ...1122.
source = (0,3)

open cells = {
    (4,1),
    (5,3)
}

Initial interaction state:

active = {(0,3)}
selected_cell = null
feedback = null

Initial frontier:

(0,2): preview 1, target 1
(1,3): preview 1, target 2
Player-facing discovery
First four-neighbor center

The target at (3,2) has four pips. Its four supports are:

left:  (2,2)
right: (4,2)
up:    (3,1)
down:  (3,3)

The cell itself cannot provide support for any of them. The player must route around it.

The open ring (4,1) is essential because (4,2) targets count 2 while (3,2) is still inactive. Before the center can activate, (4,2) receives its two supports from (4,1) and (5,2).

Before solution step 19:

preview_count(3,2) = 4

The center preview changes to magenta with four pips and can be committed.

This directly distinguishes count 4 from count 3: the cell is visibly orange at three supports and magenta only after the fourth.

Second four-neighbor cross

The target at (6,3) has supports:

(6,2)
(7,3)
(6,4)
open cell (5,3)

Before the open cell activates, it previews count 3.

However, (5,3) is also adjacent to blue target (5,4). The player must approach (5,4) from the right and commit it at count 1 before using the open cell.

Then:

click (5,3)
→ preview_count(6,3) changes 3 → 4
→ click (6,3)

The same open cell is therefore both a threat to a low target and the fourth support for a high target.

Verified solution
01 (0,2)
02 (1,2)
03 (1,1)
04 (2,1)
05 (2,0)
06 (3,0)
07 (3,1)
08 (1,3)
09 (1,4)
10 (2,3)
11 (3,3)
12 (2,2)
13 (4,0)
14 (4,1)
15 (5,0)
16 (5,1)
17 (5,2)
18 (4,2)
19 (3,2)
20 (6,1)
21 (6,2)
22 (7,2)
23 (7,3)
24 (7,4)
25 (6,4)
26 (5,4)
27 (5,3)
28 (6,3)
29 (7,5)
30 (6,5)
31 (5,5)
32 (2,4)
33 (2,5)
34 (3,4)
35 (3,5)
36 (3,6)
37 (4,6)
38 (4,5)
39 (4,4)
40 (5,6)
41 (6,6)

Open-ring committed counts:

(4,1) receives 2
(5,3) receives 2

Key checkpoints:

Before step 19:
    preview_count(3,2) = 4

Before step 27:
    result_count(5,4) = 1
    preview_count(6,3) = 3

After step 27:
    preview_count(6,3) = 4
Dependency structure

The board has three interacting regions rather than one ordered chain:

The left and upper routes jointly prepare the first count-4 center.

The right boundary secures (5,4) and prepares the second count-4 cross.

The lower-left route later converges with the right route at target (4,4), which needs three supports from distinct approaches.

The void at (4,3) prevents the first center from becoming a direct route into the lower composition. The player must revisit the lower-left frontier after completing right-side work.

Necessity and anti-cleanliness check

Omitting or postponing (4,1) leaves (4,2) unable to reach target count 2 before the first count-4 center.

Prematurely activating (5,3) spoils (5,4).

Delaying (5,3) forever leaves (6,3) at count 3.

The target at (4,4) requires convergence from the lower-left and lower-right structures, so neither is an independent checklist branch.

Both count-4 targets are necessary and appear in distinct geometric roles.

Level 6 — Offset Crosses
Purpose

Combine established mechanics into three offset count-4 structures.

The important supports are deliberately prepared out of spatial order:

A top open ring helps a later right-hand center.

A nearby count-1 bridge must be secured before the region closes around it.

A left center is completed after part of the right center is already prepared.

A lower open ring must be delayed until a separate branch reaches its vulnerable neighbor.

No new operational rule appears.

Board
      x=01234567
y=0   .112111.
y=1   11212221
y=2   122.O412
y=3   S34O1131
y=4   1111.241
y=5   .12311O1
y=6   ..112222
source = (0,3)

open cells = {
    (4,2),
    (3,3),
    (6,5)
}

Count-4 targets:

(2,3)
(5,2)
(6,4)

Initial interaction state:

active = {(0,3)}
selected_cell = null
feedback = null

Initial frontier:

(0,2): preview 1, target 1
(0,4): preview 1, target 1
(1,3): preview 1, target 3
Player-facing structure
Top preparation creates a closing bridge

The upper route reaches open ring (4,2). Activating it makes target (4,3) frontier with exactly one active neighbor.

That blue target lies between several future supports. It must be secured before growth arrives from (3,3) or (5,3).

The open ring is therefore not simply “use whenever seen.” Its activation creates an urgent low-count window elsewhere.

Left count-4 center

Target (2,3) uses:

(1,3) target 3
(2,2)
(2,4)
open cell (3,3)

The top and bottom source branches first prepare (1,3) and the two vertical supports. Target (3,4) is secured at count 1 before open cell (3,3) joins.

Before step 26:

preview_count(2,3) = 4
Right count-4 center

Target (5,2) uses:

open cell (4,2)
(5,1)
(6,2)
(5,3)

Three of those supports are prepared before the left center is finished. The fourth, (5,3), is reached through the central bridge. This creates a dependency network rather than “finish left, then start right.”

Before step 28:

preview_count(5,2) = 4
Lower count-4 center

Target (6,4) uses:

(5,4)
(7,4)
(6,3)
open cell (6,5)

The left-bottom branch must first commit (5,5) at count 1. A plausible wrong plan is to descend the far-right edge to (7,5), activate open cell (6,5), and only later approach (5,5). That gives (5,5) a second active neighbor and permanently spoils it.

Once (5,5) is fixed, the open ring becomes the fourth support for (6,4).

Verified solution
01 (0,2)
02 (0,1)
03 (1,1)
04 (1,0)
05 (2,0)
06 (2,1)
07 (3,1)
08 (3,0)
09 (4,0)
10 (4,1)
11 (4,2)
12 (4,3)
13 (5,0)
14 (5,1)
15 (6,0)
16 (6,1)
17 (6,2)
18 (1,2)
19 (2,2)
20 (0,4)
21 (1,4)
22 (2,4)
23 (1,3)
24 (3,4)
25 (3,3)
26 (2,3)
27 (5,3)
28 (5,2)
29 (7,1)
30 (7,2)
31 (7,3)
32 (7,4)
33 (6,3)
34 (1,5)
35 (2,5)
36 (2,6)
37 (3,6)
38 (3,5)
39 (4,5)
40 (5,5)
41 (5,4)
42 (7,5)
43 (6,5)
44 (6,4)
45 (4,6)
46 (5,6)
47 (6,6)
48 (7,6)

Open-ring committed counts:

(4,2) receives 1
(3,3) receives 2
(6,5) receives 2

Key checkpoints:

Before step 26:
    preview_count(2,3) = 4

Before step 28:
    preview_count(5,2) = 4

Before step 43:
    preview_count(6,4) = 3

After step 43:
    preview_count(6,4) = 4
Dependency and anti-cleanliness check

The voids at (3,2) and (4,4) prevent a rectangular sweep:

(3,2) separates the top support network from the left center.

(4,4) forces the bottom-left and right-edge routes to meet around, rather than through, the middle.

The three high targets do not form a single dependency chain. Work for (5,2) begins before (2,3) is completed, while work for (6,4) uses both the far-right route and the lower-left route.

Every open ring is a necessary support for a different count-4 target. Each also creates or threatens a nearby lower-count window.

Level 7 — Whole Frontier
Purpose

Present the complete system as one dense but readable dependency network.

There are:

Four open rings.

Four count-4 targets.

Three important count-3 convergence cells.

Two offset interior voids.

Three initial growth branches.

No new visual or operational rule is introduced. Difficulty comes entirely from deciding which frontier branch should advance before each open-ring commitment.

Board
      x=01234567
y=0   .112111.
y=1   .1212221
y=2   11314O22
y=3   122.1141
y=4   S24O2.O1
y=5   12112112
y=6   .1234O32
y=7   ..111112
source = (0,4)

open cells = {
    (5,2),
    (3,4),
    (6,4),
    (5,6)
}

Count-4 targets:

(4,2)
(2,4)
(6,3)
(4,6)

Initial interaction state:

active = {(0,4)}
selected_cell = null
feedback = null

Initial frontier:

(0,3): preview 1, target 1
(0,5): preview 1, target 1
(1,4): preview 1, target 2

The player may begin upward or downward. The middle target must wait.

Player-facing dependency network
Upper open ring and upper count-4 center

Open ring (5,2) has two coupled effects:

It supplies the right support for count-4 target (4,2).

It makes (5,3) available at count 1.

The player should commit (5,3) while that low window is open. Later central growth would raise it.

Target (4,2) ultimately uses:

(3,2)
open cell (5,2)
(4,1)
(4,3)

The target-3 cell (2,2) is deliberately activated earlier with supports from the top-left, top, and top-right. It then helps prepare the upper support of the left-middle structure, coupling what would otherwise be independent branches.

Left-middle open ring

Target (2,4) uses:

(1,4)
(2,3)
(2,5)
open cell (3,4)

Before activating (3,4), the player approaches and commits (3,5) at count 1. The open cell then becomes the fourth support for (2,4).

Because (3,4) can also be reached from the east through (4,4), advancing the east branch too aggressively creates a plausible premature-open plan. The visible blue ring at (3,5) exposes the danger.

East open ring

Target (6,3) uses:

(5,3)
(7,3)
(6,2)
open cell (6,4)

The top route prepares three supports early, but the open ring is deliberately approached from below and from the right.

Target (6,5) must be fixed at count 1 before (6,4) joins. Then the open ring raises (6,3) from three to four.

Bottom open ring

Target (4,6) uses:

(3,6)
open cell (5,6)
(4,5)
(4,7)

The bottom route first reaches blue target (5,7). Activating open cell (5,6) before (5,7) would raise that blue target to count 2.

Once the low target is secured, the open ring supplies the fourth support for (4,6).

Final closing window

Target (6,6) requires count 3. Immediately before its activation, its supports are:

open cell (5,6)
(6,5)
(6,7)

Target (7,6) is still inactive. Activating (7,6) first would raise (6,6) from three to four and spoil it.

Thus the final actions still require interpretation; they are not corridor cleanup.

Verified solution
01 (0,3)
02 (0,2)
03 (1,2)
04 (1,1)
05 (1,0)
06 (2,0)
07 (2,1)
08 (3,1)
09 (3,0)
10 (4,0)
11 (4,1)
12 (3,2)
13 (5,0)
14 (5,1)
15 (5,2)
16 (5,3)
17 (6,0)
18 (6,1)
19 (6,2)
20 (7,1)
21 (7,2)
22 (7,3)
23 (2,2)
24 (1,3)
25 (2,3)
26 (1,4)
27 (0,5)
28 (1,5)
29 (2,5)
30 (3,5)
31 (3,4)
32 (2,4)
33 (4,3)
34 (4,2)
35 (7,4)
36 (4,4)
37 (4,5)
38 (5,5)
39 (6,5)
40 (7,5)
41 (6,4)
42 (6,3)
43 (1,6)
44 (2,6)
45 (2,7)
46 (3,7)
47 (3,6)
48 (4,7)
49 (5,7)
50 (5,6)
51 (4,6)
52 (6,7)
53 (6,6)
54 (7,6)
55 (7,7)

Open-ring committed counts:

(5,2) receives 1
(3,4) receives 1
(6,4) receives 2
(5,6) receives 2

Key checkpoints:

Before step 32:
    preview_count(2,4) = 4

Before step 34:
    preview_count(4,2) = 4

Before step 41:
    preview_count(6,3) = 3

After step 41:
    preview_count(6,3) = 4

Before step 51:
    preview_count(4,6) = 4

Before step 53:
    preview_count(6,6) = 3
Structural irregularity

The authored irregularities all have consequences:

The void at (3,3) separates the upper-left and center routes, making (3,4) approachable from two temporally different directions.

The void at (5,4) prevents direct passage between the central and east open rings.

The missing upper-left and lower-left cells make the source’s upper and lower approaches unequal.

The top route prepares parts of two count-4 centers before either is completed.

The lower-left route supports both the left-middle and bottom centers.

The right boundary supports the east center and creates access to its open ring.

Open rings are not encountered in the order they are ultimately committed.

Anti-degenerate check

The level cannot be solved by:

Activating every open ring as soon as it appears.

Completing one spatial quadrant before touching another.

Following a perimeter sweep.

Clicking all count-1 targets first globally.

Treating each count-4 target as an independent four-item checklist.

Each of those plans oversaturates at least one visible low or medium target. The important failures arise from concrete adjacency and timing, not hidden exceptions.

7. Implementation acceptance checks

An implementation should satisfy all of the following:

The game exposes exactly 7 levels.

The only effective gameplay input is clicking a frontier cell.

A clicked cell’s result uses the active set from before that click.

Diagonal cells never contribute to a count.

Source, open-ring, matched, and mismatched active cells all count as support.

A target mismatch remains active and can support growth, but prevents victory.

Open-ring cells accept results 1, 2, 3, or 4 and retain their broken gray ring.

All inactive frontier cells show both white brackets and an exact prospective color/pip preview.

Non-frontier clicks never activate a cell.

Directional actions and SPACE never change gameplay state.

Animation subframes never consume actions.

No level has a budget, timer, energy reserve, or exploration penalty.

The listed solution orders activate every non-source playable cell exactly once.

Every target activation in the listed paths has a pre-click active-neighbor count equal to its map digit.

The meaningful solution depths are exactly 3, 7, 11, 29, 41, 48, 55.

Victory ignores source and open-ring result colors but requires every playable cell to be active and every colored target ring to match its committed center.
