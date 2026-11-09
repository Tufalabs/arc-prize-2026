Drop & Pair
Identity

This is a game about deleting equal adjacent pairs becoming difficult because every deletion rewrites the relative column heights that determine all future pairs.

Exact authored level count: 7.

Drop & Pair has one operational rule and no secondary devices. Clicking either member of an orthogonally adjacent, equal-colored pair removes both blocks. Blocks above the removed cells then fall straight down within their original columns. Matches created by the fall remain until explicitly clicked.

The player wins when the board contains no blocks.

The interesting choice is therefore rarely “can I remove this pair?” It is “what profile will the remaining columns have after I remove it?”

1. Player-facing rules
Controls

The action space is exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any canvas cell

Only clicking is functional. The four directional actions and SPACE call step() but leave the board unchanged.

There is no avatar, persistent cursor, selected block, timer, move limit, energy counter, or hidden state. The colored blocks are the only major objects on the board and are rendered as raised, module-sized click targets.

Clicking is intrinsic to the mechanic: the player directly chooses which currently available pair will be consumed as a height adjustment. Replacing clicks with movement would add navigation unrelated to the puzzle.

Basic interaction

Two blocks form a removable pair when:

they have the same integer color ID;

they share one orthogonal edge;

neither block has another equal-colored orthogonal neighbor.

The third condition is normally invisible because all seven authored boards satisfy the stronger invariant that no state reachable through valid pair removals contains an ambiguous block with two equal neighbors.

Clicking either member of a removable pair:

briefly outlines both blocks;

removes both simultaneously;

lets every block above either removed position fall vertically;

leaves newly formed pairs in place.

There is no horizontal compression and no automatic cascade.

Clicking an occupied block with no equal neighbor gives an invalid-action shake and crossed overlay. Clicking an empty cell or outside the board has no logical effect.

Objective

After each valid removal and settlement:

win = every column contains zero blocks

Nothing else is checked. In particular:

the number of clicks is ignored;

invalid clicks are allowed;

removal order is not compared with an authored solution;

animation state and feedback state are ignored;

an empty board is sufficient and necessary.

If blocks remain but no removable pair exists, the board is visibly deadlocked. Deadlock is not a separate hidden failure state; the unchanged blocks show the failure directly. Restarting the level is handled by the host environment, not by any gameplay action.

2. Logical board and coordinates

Each level uses a rectangular logical board of width W and height H.

Coordinates are:

x = 0 at the leftmost column;

y = 0 at the floor;

increasing y moves upward.

A column is stored bottom-to-top:

columns[x] = [color_at_y0, color_at_y1, ...]

All initial columns are compact. At every settled state, an occupied cell exists exactly when:

0 <= y < len(columns[x])

The empty cells above each column remain visible as faint wells.

Pair notation used below

The authored solution traces use:

H(x, y, K)

for a horizontal pair of color K occupying (x,y) and (x+1,y), and:

V(x, y, K)

for a vertical pair of color K occupying (x,y) and (x,y+1).

The standardized witness click is the left cell of H or the lower cell of V. Clicking the other member has exactly the same result.

All coordinates in a trace refer to the current settled state after all preceding clicks, not to persistent block identities.

3. Colors and non-text visual coding

The actual game renders no letters or text. The following letters are documentation shorthand only:

Shorthand	Color ID	Palette color
M	6	magenta
R	8	red
B	9	blue
C	10	light blue
Y	11	yellow
O	12	orange
G	14	green
P	15	purple

Only those eight IDs are used for blocks.

To make equality readable without relying exclusively on hue, each block contains a fixed geometric inset. Local coordinates refer to the block’s 4×4 body:

Color ID	Inset pixels
6	(1,1)
8	(2,1)
9	(1,2)
10	(2,2)
11	(1,1),(2,1)
12	(1,2),(2,2)
14	(1,1),(2,2)
15	(2,1),(1,2)

Inset ink is color 0 on IDs 6, 8, 9, and 15, and color 5 on IDs 10, 11, 12, and 14. The surrounding colored pixels keep every block a connected multi-tile shape.

4. 64×64 rendering contract
Board placement

Every logical cell uses a 5×5 pixel module:

4×4 visible body;

one-pixel right and bottom gutter.

For level dimensions W,H:

board_pixel_width  = 5 * W
board_pixel_height = 5 * H

origin_x = floor((64 - board_pixel_width) / 2)
origin_y = floor((62 - board_pixel_height) / 2)

The two vertically reserved pixels allow a floor line below the board.

For logical (x,y), let:

render_row = H - 1 - y
module_x = origin_x + 5*x
module_y = origin_y + 5*render_row

The module is:

[module_x, module_x+4] × [module_y, module_y+4]

The 4×4 body is:

[module_x, module_x+3] × [module_y, module_y+3]
Static colors

Canvas and gutters: color 0.

Empty cell bodies: color 1.

Floor: a two-pixel horizontal strip of color 4 at rows
origin_y + 5H and origin_y + 5H + 1.

Occupied cell body: the block’s color ID.

Inset symbol: color 0 or 5 as defined above.

There are no internal borders around cells. Negative space comes from the white one-pixel gutters, not nested outlines.

Click regions

The entire 5×5 module of an occupied logical cell is its hit region, including its gutter. This prevents pixel hunting.

A canvas click maps to a logical cell only when it lies inside the board rectangle. Clicking the module of an empty cell is invalid.

Level-one click cue

On the first load of level 1 only, all occupied blocks depress visually by one pixel for one 80 ms frame and return. This is a non-state-changing affordance cue applied uniformly to all blocks; it does not identify which colors match.

5. Exact state transition

The state relevant to gameplay is:

columns: list[list[int]]
won: bool
feedback: NONE | VALID_PAIR | INVALID_BLOCK | EMPTY_CLICK | COMPLETE
selected: always null

There is no persistent cursor or selected object.

step(click(px,py))

If won is true, return the unchanged completed state.

Map the canvas click to logical (x,y).

If the click is outside the board or the logical cell is empty:

leave columns unchanged;

set feedback = EMPTY_CLICK;

return.

Enumerate equal-colored orthogonal neighbors of (x,y).

If the count is zero:

leave columns unchanged;

set feedback = INVALID_BLOCK;

return.

If the count exceeds one:

leave columns unchanged;

use the same invalid feedback;

raise a development assertion in debug builds because an authored-board invariant has been violated.

Let the unique neighbor be (x2,y2). Snapshot both pre-removal coordinates.

Remove both blocks simultaneously:

for a vertical pair, delete indices y and y+1 from one column;

for a horizontal pair, delete index y from both affected columns.

Preserve the bottom-to-top order of every surviving block. Compacting each list produces gravity.

Recompute adjacency only after all affected columns have settled.

Do not automatically remove any new pair.

Set:

won = all(len(column) == 0 for column in columns)

Set feedback to COMPLETE if won, otherwise VALID_PAIR.

UP, DOWN, LEFT, RIGHT, and SPACE perform a no-op transition with no gameplay feedback.

Resolution order examples

For V(x,y,K):

the two blocks at indices y and y+1 are removed together;

every old block at index j > y+1 moves to j-2.

For H(x,y,K):

index y is removed from columns x and x+1;

every old block above y in either affected column moves down one;

the two columns settle independently.

No pair is detected halfway through a fall.

6. Animation and feedback

Animations do not consume actions, do not alter move budgets, and do not expose an intermediate decision state. A call to step() is resolved atomically; the next action is accepted only after the settled frame.

Valid removal

A valid click produces four deterministic render phases:

Outline, 70 ms: both selected block bodies receive a color-11 perimeter.

One-third fall, 70 ms: pair blocks are absent; falling blocks are rendered one-third of the way toward their final positions.

Two-thirds fall, 70 ms: falling blocks are rendered two-thirds of the way down.

Settled: render the committed state.

For a survivor falling d logical rows, its total displacement is 5d pixels. Intermediate displacement at phase k ∈ {1,2} is:

floor((5*d*k + 1) / 3)

A vertical removal can therefore show a two-row drop in one column; a horizontal removal shows one-row drops in two columns.

Invalid occupied click

The clicked block receives a high-contrast X:

(0,0),(1,1),(2,2),(3,3)
(3,0),(2,1),(1,2),(0,3)

The X uses color 8, except on a red block where it uses color 0. The visual body shifts left one pixel, right one pixel, and returns. The logical board never changes.

Completion

After the final blocks settle:

all empty wells remain visible;

the floor flashes color 14 for two render frames;

won becomes true.

7. Progression overview
Level	Valid clicks in witness	Main discovery
1	2	Equal orthogonal pairs are clickable; pairs do not auto-clear
2	5	Blocks fall vertically and new matches wait for a click
3	7	Horizontal and vertical removals produce different height changes
4	25	Remove an upper pair to expose a lower sequence before spending a tempting pair
5	40	Preserve a currently valid bridge pair until another profile change makes it safe
6	45	Overlapping horizontal pairs transmit a height correction across columns
7	50	Coordinate several fronts while preserving a valid but premature pair

The traces below are verified winning witnesses, not recipes enforced by the validator. Several later removals can commute. The essential restrictions arise from the board geometry and column profiles.

8. Authored levels
Level 1 — Two Orientations

Purpose: Teach the clickable objects, equality, both orthogonal orientations, explicit removal, and the empty-board objective without gravity or misleading blocks.

Dimensions: W=4, H=2
Meaningful solution depth: 2 clicks
Initial transient state: selected=null, feedback=NONE, won=false

Documentation sketch, top row first:

. . . Y
B B . Y

Exact columns, bottom-to-top:

x0 = [9]
x1 = [9]
x2 = []
x3 = [11, 11]

Initial removable pairs:

H(0,0,B)
V(3,0,Y)
Player-facing discovery

The level contains only two obvious pairs, one horizontal and one vertical. Clicking either member of a pair removes exactly those two blocks. The other visible pair remains, demonstrating immediately that matching does not cause automatic clearing.

After the second click, the board is empty and gives completion feedback.

Verified witness
1 H(0,0,B)
2 V(3,0,Y)

The reverse order also wins.

Attribution and necessity check

Every visible block communicates the single causal rule or participates in the solution. There are no unmatched blocks, falling blocks, decoys, timing conditions, or simultaneous secondary changes. The two orientations prevent the mistaken hypothesis that only horizontal or only vertical equality matters.

Level 2 — The Drop

Purpose: Isolate gravity and establish that newly created matches remain until clicked.

Dimensions: W=2, H=6
Meaningful solution depth: 5 clicks
Initial transient state: selected=null, feedback=NONE, won=false

P .
G .
O P
Y G
B O
B Y

Exact columns:

x0 = [9, 9, 11, 12, 14, 15]
x1 = [11, 12, 14, 15]

Initial removable pair:

V(0,0,B)

No other initial block is clickable.

Player-facing discovery

The only useful experiment is the blue vertical pair at the bottom of the left column. Removing it causes the four blocks above it to fall exactly two rows. They then line up with the four blocks in the right column:

Y Y
O O
G G
P P

All four matches remain visible rather than cascading. The player must choose and click them.

This distinguishes the intended rule from several plausible mistakes:

removed space does not remain as a hole;

blocks do not move diagonally;

both affected cells of a vertical pair vanish simultaneously;

newly formed matches are not automatic.

Verified witness
1 V(0,0,B)
2 H(0,0,Y)
3 H(0,0,O)
4 H(0,0,G)
5 H(0,0,P)

After the first click, the four horizontal pairs can be removed in any order.

Attribution and necessity check

The initial blue pair is the sole available action, so its visible two-row fall cannot be attributed to another removal. Every remaining block is part of a newly created pair. There is no deadlock reachable in this level.

Level 3 — One Row or Two

Purpose: Teach that pair orientation is not cosmetic. A horizontal pair lowers two columns by one; a vertical pair lowers one column by two.

Dimensions: W=3, H=6
Meaningful solution depth: 7 clicks
Initial transient state: selected=null, feedback=NONE, won=false

. G C
. R R
. P C
. B M
B P R
G M R

Exact columns:

x0 = [14, 9]
x1 = [6, 15, 9, 15, 8, 14]
x2 = [8, 8, 6, 10, 8, 10]

Initial removable pairs:

V(2,0,R)   # losing opening
H(1,4,R)   # winning opening
Player-facing discovery

Both available pairs are red, but they modify the profile differently.

Clicking the upper horizontal pair lowers columns 1 and 2 by one. This preserves the bottom vertical red pair while positioning the two light-blue blocks so they can later become vertical.

Clicking the bottom vertical pair first lowers only column 2 by two. The player can continue through three forced matches, but the final six blocks are visibly unmatched. The experiment is cheap and its consequence is spatially inspectable.

Thus the level falsifies the tempting hypothesis that every currently valid pair is harmless progress.

Verified witness
1 H(1,4,R)
2 V(2,0,R)
3 V(2,1,C)
4 H(1,0,M)
5 H(0,1,B)
6 V(1,0,P)
7 H(0,0,G)
Necessity check

H(1,4,R) is the only winning opening. Choosing V(2,0,R) first leads through the forced sequence:

H(1,0,M)
H(0,1,B)
V(1,0,P)

and then to a deadlock with blocks remaining.

The lesson comes entirely from the different geometry of the two legal removals; there is no new object or hidden exception.

Level 4 — Expose Before Aligning

Purpose: Introduce a required exposure sequence. A tempting floor pair must wait while an upper pair creates a vertical match and then a horizontal match.

Dimensions: W=8, H=8
Meaningful solution depth: 25 clicks
Initial transient state: selected=null, feedback=NONE, won=false

Y . O . Y . B .
P . P . O R G .
M . G . M C P .
P R Y R C P R .
C M C C B C G .
B G Y M B P O .
B Y R P C O P .
C M M O R Y O B

Exact columns:

x0 = [10, 9, 9, 10, 15, 6, 15, 11]
x1 = [6, 11, 14, 6, 8]
x2 = [6, 8, 11, 10, 11, 14, 15, 12]
x3 = [12, 15, 6, 10, 8]
x4 = [8, 10, 9, 9, 10, 6, 12, 11]
x5 = [11, 12, 15, 10, 15, 10, 8]
x6 = [12, 15, 12, 14, 8, 15, 14, 9]
x7 = [9]

Initial removable pairs:

V(0,1,B)
V(4,2,B)
H(1,0,M)   # valid but no winning continuation if taken immediately
H(2,3,C)
Player-facing discovery

The central local structure supplies the new evidence.

Clicking the upper light-blue horizontal pair:

H(2,3,C)

removes one block from columns 2 and 3. In column 2, the yellow block above falls directly onto another yellow block, visibly creating:

V(2,2,Y)

Removing that vertical yellow pair drops column 2 by two and exposes green at (2,2), now aligned with green at (1,2):

H(1,2,G)

The immediate sequence is therefore:

upper horizontal pair
→ newly formed vertical pair
→ newly formed horizontal pair

The visible magenta pair on the floor is deliberately tempting. Removing it before the exposure work destroys the profile required to clear the board.

Other safe initial work can be interleaved, so the puzzle is not validating one script, but the central exposure sequence is unavoidable.

Verified witness
 1 H(2,3,C)   2 V(2,2,Y)   3 H(1,2,G)   4 H(1,0,M)   5 H(2,1,P)
 6 V(4,2,B)   7 V(0,1,B)   8 V(0,0,C)   9 V(4,1,C)  10 H(3,1,M)
11 H(0,1,M)  12 H(4,1,O)  13 H(5,1,P)  14 V(0,0,P)  15 H(0,0,Y)
16 V(6,0,O)  17 H(1,0,R)  18 H(5,2,P)  19 V(5,1,C)  20 H(2,0,O)
21 H(5,1,R)  22 H(3,0,R)  23 H(4,0,Y)  24 V(6,0,G)  25 H(6,0,B)
Anti-cleanliness and necessity check

The board has four opening pairs distributed across three regions. Three openings have winning continuations, while the floor magenta pair does not. This makes the board a dependency network rather than a single exposed chain.

The isolated blue block in column 7 is not a decoy: it is the final partner of H(6,0,B). The uneven heights of columns 3 and 7 prevent the board from reducing to a corridor or a list of independent pairs.

Level 5 — Preserve the Bridge

Purpose: Turn an available pair into a resource that must be preserved. The pair is harmful when first created but necessary after an upper profile has been corrected.

Dimensions: W=12, H=10
Meaningful solution depth: 40 clicks
Initial transient state: selected=null, feedback=NONE, won=false

M B . . . . . . . . Y .
B C . . . . . R . P G .
M P . . G . . C . B C .
Y R . . C P . R . O G .
B Y . . R Y . G . R O .
P M . . M B Y R . G B .
Y M C R Y G R Y . G Y .
P G O Y P Y P Y O R B .
B Y B O M P B R P C Y .
R G P B C Y C G R Y B O

Exact columns:

x0  = [8,9,15,11,15,9,11,6,9,6]
x1  = [14,11,14,6,6,11,8,15,10,9]
x2  = [15,9,12,10]
x3  = [9,12,11,8]
x4  = [10,6,15,11,6,8,10,14]
x5  = [11,15,11,14,9,11,15]
x6  = [10,9,15,8,11]
x7  = [14,8,11,11,8,14,8,10,8]
x8  = [8,15,12]
x9  = [11,10,8,14,14,8,12,9,15]
x10 = [9,11,9,11,9,12,14,10,14,11]
x11 = [12]

Initial removable pairs:

V(1,3,M)
V(7,2,Y)
V(9,3,G)

All three have winning continuations.

Player-facing discovery

The clearest local experiment is on the right.

The first three witness removals descend column 7:

V(7,2,Y)
V(7,1,R)
V(7,0,G)

This creates a red horizontal pair at the floor:

H(7,0,R)

It looks like immediate progress, but clicking it now has no winning continuation. The pair must remain visible.

Column 9 provides a nearby alternative:

V(9,3,G)
V(9,2,R)

Those removals expose an orange horizontal pair across columns 8 and 9:

H(8,2,O)

Removing the orange pair alters column 8 without spending the floor red pair. The red bridge is now safe and can lower columns 7 and 8 together:

H(7,0,R)

The same operational pair has therefore occupied two contrasting roles:

premature liability before the orange pair;

necessary height-changing bridge after the orange pair.

No state marker changes. Its meaning changes only because the surrounding column profiles change.

Verified witness
 1 V(7,2,Y)   2 V(7,1,R)   3 V(7,0,G)   4 V(9,3,G)   5 V(9,2,R)
 6 H(8,2,O)   7 H(7,0,R)   8 V(1,3,M)   9 H(0,3,Y)  10 V(0,2,P)
11 V(0,1,B)  12 H(0,1,Y)  13 V(1,0,G)  14 H(0,0,R)  15 H(6,0,C)
16 H(5,1,P)  17 V(5,0,Y)  18 H(5,2,Y)  19 H(4,2,P)  20 H(3,2,Y)
21 V(4,1,M)  22 H(9,2,B)  23 V(10,1,Y) 24 V(10,0,B) 25 H(10,0,O)
26 H(9,1,C)  27 V(10,0,G) 28 H(9,0,Y)  29 H(8,0,P)  30 H(1,0,P)
31 H(0,1,B)  32 V(0,0,M)  33 H(2,0,B)  34 H(3,1,R)  35 V(4,0,C)
36 H(2,0,O)  37 H(1,0,C)  38 H(4,0,G)  39 H(5,0,B)  40 H(6,0,R)
Anti-cleanliness and necessity check

The bridge is not merely delayed by an abstract prerequisite. Spending it early physically lowers column 8 and prevents the orange blocks at (8,2) and (9,2) from ever meeting at the required phase.

The board has active fronts on the far left, center, and right. Their operations can partly interleave, but the right bridge ordering cannot be ignored. The singleton orange block in column 11 is necessary for H(10,0,O) and closes a dependency running through columns 8–11.

Level 6 — Coupled Wave

Purpose: Reuse horizontal pairs as overlapping two-column adjustments. Each removal prepares a pair that shares one of its columns with the next removal.

Dimensions: W=12, H=10
Meaningful solution depth: 45 clicks
Initial transient state: selected=null, feedback=NONE, won=false

M B . . G . . R . . Y .
B C . . C P . C . P G .
M P . . R Y . R . B C .
Y R . O M P M G . O G .
B Y . R C B P M . R O .
P M . Y O C Y R . G B .
Y M C O Y G R Y . G Y .
P G O B P Y P Y O R B .
B Y B R M P B R P C Y .
R G P R C Y C G R Y B O

Exact columns:

x0  = [8,9,15,11,15,9,11,6,9,6]
x1  = [14,11,14,6,6,11,8,15,10,9]
x2  = [15,9,12,10]
x3  = [8,8,9,12,11,8,12]
x4  = [10,6,15,11,12,10,6,8,10,14]
x5  = [11,15,11,14,10,9,15,11,15]
x6  = [10,9,15,8,11,15,6]
x7  = [14,8,11,11,8,6,14,8,10,8]
x8  = [8,15,12]
x9  = [11,10,8,14,14,8,12,9,15]
x10 = [9,11,9,11,9,12,14,10,14,11]
x11 = [12]

Initial removable pairs:

V(1,3,M)
V(3,0,R)
V(7,2,Y)   # valid but initially losing
V(9,3,G)
Player-facing discovery

The yellow vertical pair in column 7 resembles an immediately useful pair from level 5, but its context has changed. Removing it before correcting columns 3–7 has no winning continuation.

The center entry point is:

V(3,0,R)

Removing it exposes:

H(3,4,O)

That horizontal removal lowers columns 3 and 4 and prepares:

H(4,4,C)

The overlapping sequence continues:

H(5,5,P)
H(6,5,M)

Each pair shares one column with the next. The one-row correction therefore propagates rightward:

columns 3–4
→ columns 4–5
→ columns 5–6
→ columns 6–7

Only after the final magenta removal has adjusted column 7 does V(7,2,Y) acquire a winning continuation.

The left and far-right fronts remain available for interleaving. The witness deliberately alternates among fronts so that the level is not experienced as a single corridor of matches.

Verified witness
 1 V(3,0,R)   2 V(9,3,G)   3 H(3,4,O)   4 V(1,3,M)   5 H(4,4,C)
 6 V(9,2,R)   7 H(5,5,P)   8 H(0,3,Y)   9 H(6,5,M)  10 V(7,2,Y)
11 V(0,2,P)  12 V(0,1,B)  13 V(7,1,R)  14 V(7,0,G)  15 H(0,1,Y)
16 V(1,0,G)  17 H(0,0,R)  18 H(8,2,O)  19 H(7,0,R)  20 H(6,0,C)
21 H(5,1,P)  22 V(5,0,Y)  23 H(5,2,Y)  24 H(4,2,P)  25 H(3,2,Y)
26 V(4,1,M)  27 H(9,2,B)  28 V(10,1,Y) 29 V(10,0,B) 30 H(10,0,O)
31 H(9,1,C)  32 V(10,0,G) 33 H(9,0,Y)  34 H(8,0,P)  35 H(1,0,P)
36 H(0,1,B)  37 V(0,0,M)  38 H(2,0,B)  39 H(3,1,R)  40 V(4,0,C)
41 H(2,0,O)  42 H(1,0,C)  43 H(4,0,G)  44 H(5,0,B)  45 H(6,0,R)
Anti-cleanliness and necessity check

The four central horizontal pairs are not independent checklist items. Removing one changes both the column it received from the previous pair and the column needed by the next pair.

Meanwhile, the left and right fronts create a genuine dependency network: they can be advanced between central-wave steps, but none substitutes for the central correction. The initially valid yellow pair changes from liability to tool without any exception to its removal rule.

Level 7 — Whole-Board Orchestration

Purpose: Compose exposure, pair preservation, overlapping height corrections, and multiple safe opening fronts. The player must distinguish currently safe work from the one valid pair that remains premature.

Dimensions: W=12, H=11
Meaningful solution depth: 50 clicks
Initial transient state: selected=null, feedback=NONE, won=false

M B . . G . . . . P Y .
B C . . C . . R . B G .
M P . . R P . C . O C .
Y R . O M Y . R . R G .
B Y . R C P M G . G O .
P M . Y O B P M O M B .
Y M . O Y C Y R P G M .
P G C B P G R Y B R Y .
B Y O R M Y P Y P C B .
C C B G G P B R P B Y .
R G P R C Y C G R Y B O

Exact columns:

x0  = [8,10,9,15,11,15,9,11,6,9,6]
x1  = [14,10,11,14,6,6,11,8,15,10,9]
x2  = [15,9,12,10]
x3  = [8,14,8,9,12,11,8,12]
x4  = [10,14,6,15,11,12,10,6,8,10,14]
x5  = [11,15,11,14,10,9,15,11,15]
x6  = [10,9,15,8,11,15,6]
x7  = [14,8,11,11,8,6,14,8,10,8]
x8  = [8,15,15,9,15,12]
x9  = [11,9,10,8,14,6,14,8,12,9,15]
x10 = [9,11,9,11,6,9,12,14,10,14,11]
x11 = [12]

Initial removable pairs:

V(1,4,M)
V(7,2,Y)   # valid but initially losing
V(8,1,P)
H(0,1,C)
H(3,1,G)
Player-facing discovery

The initial board exposes five legal choices across three fronts.

On the right, removing:

V(8,1,P)

immediately creates:

H(8,1,B)

Removing that blue pair alters columns 8 and 9 and exposes a magenta horizontal pair higher up across columns 9 and 10.

In the center, the initial green pair:

H(3,1,G)

prepares the vertical red trigger and the overlapping correction wave from level 6.

On the left, the initial light-blue pair:

H(0,1,C)

changes the coordinate of the magenta vertical pair in column 1 and opens a separate exposure sequence.

The yellow vertical pair in column 7 remains visibly legal throughout much of this work, but it is still premature. It becomes safe only after the center wave reaches the magenta pair across columns 6 and 7.

The mature puzzle therefore has several partial orders rather than one prescribed phase list:

the left cap can be advanced early or delayed;

the right exposure branch can be interleaved with center work;

the center wave must preserve the column-7 yellow pair;

later right-edge removals depend on profiles established much earlier;

final floor pairs join regions that began as separate fronts.

Verified witness
 1 V(8,1,P)   2 H(3,1,G)   3 H(0,1,C)   4 H(8,1,B)   5 V(1,3,M)
 6 H(9,4,M)   7 V(3,0,R)   8 V(9,3,G)   9 H(3,4,O)  10 V(9,2,R)
11 H(4,4,C)  12 H(0,3,Y)  13 H(5,5,P)  14 H(6,5,M)  15 V(7,2,Y)
16 V(0,2,P)  17 V(0,1,B)  18 V(7,1,R)  19 V(7,0,G)  20 H(0,1,Y)
21 V(1,0,G)  22 H(0,0,R)  23 H(8,2,O)  24 H(7,0,R)  25 H(6,0,C)
26 H(5,1,P)  27 V(5,0,Y)  28 H(5,2,Y)  29 H(4,2,P)  30 H(3,2,Y)
31 V(4,1,M)  32 H(9,2,B)  33 V(10,1,Y) 34 V(10,0,B) 35 H(10,0,O)
36 H(9,1,C)  37 V(10,0,G) 38 H(9,0,Y)  39 H(8,0,P)  40 H(1,0,P)
41 H(0,1,B)  42 V(0,0,M)  43 H(2,0,B)  44 H(3,1,R)  45 V(4,0,C)
46 H(2,0,O)  47 H(1,0,C)  48 H(4,0,G)  49 H(5,0,B)  50 H(6,0,R)
Anti-cleanliness and necessity check

Four of the five initial pairs have winning continuations; the column-7 yellow pair does not. Thus the opening is neither a single forced action nor a random click-fest.

The board’s asymmetry is consequential:

column 2 is short and later becomes a bridge between left and center;

column 8 begins with a vertical purple pair that exposes the right branch;

column 11 contains one orange block whose only eventual partner depends on several right-edge removals;

columns 3–7 carry the overlapping correction wave;

columns 0–1 and 9–11 can be advanced at different times but eventually feed pairs back toward the center.

Removing any region from the authored instance changes the solvability structure; the board cannot be reduced to a corridor, timer, or independent checklist.

9. Mechanic reuse in contrasting roles

The operational rule never changes, but pair meaning changes with context:

Pair role	Example
Immediate harmless progress	Either pair in level 1
Generator of many new matches	Bottom blue pair in level 2
Losing height adjustment	Bottom red pair in level 3
Exposure tool	Upper light-blue pair in level 4
Tempting pair that must wait	Floor magenta pair in level 4
Bridge that is first harmful, later necessary	Floor red pair in level 5
Overlapping transfer of a height correction	Central staircase in level 6
Persistent valid liability amid several safe fronts	Column-7 yellow pair in levels 6 and 7

This gives the game systemic depth without adding devices, keys, switches, counters, ownership states, or symbolic predicates.

10. Reference validation and implementation tests

The authored states were exhaustively enumerated under all sequences of valid pair clicks. States are deduplicated by the tuple of bottom-to-top columns.

Level	Initial blocks	Reachable settled states	Reachable nonempty deadlocks	Ambiguous reachable states
1	4	4	0	0
2	10	17	0	0
3	14	17	1	0
4	50	326	9	0
5	80	4,704	11	0
6	90	10,472	23	0
7	100	17,136	29	0

Recommended unit tests:

Every initial column is compact and contains only IDs {6,8,9,10,11,12,14,15}.

Every initial block count is even.

Every supplied witness trace ends with all columns empty.

At every reachable state, each block has at most one equal-colored orthogonal neighbor.

A valid click removes exactly two blocks.

A vertical removal preserves the order of survivors and lowers all blocks above it by two.

A horizontal removal preserves both affected column orders and lowers blocks above the removed row by one.

Newly adjacent pairs remain after settlement.

Invalid clicks leave columns byte-for-byte unchanged.

No directional action or SPACE changes columns, won, or any hidden puzzle variable.

The only transition to won=true occurs when the total block count reaches zero.

The game exposes exactly 7 levels.
