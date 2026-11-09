Shift Mosaic

This is a game about cyclically shifting one line becoming difficult because every crossing shift changes which tiles the other lines contain.

Authored level count: exactly 7.

Shift Mosaic is a reversible permutation puzzle. Every move is easy to understand: activate an arrow beside a row or column, and that entire line shifts one cell with wraparound. The depth comes from interference. A column shift extracts one member of every row it crosses; reversing a row after such an extraction restores the row’s positions but not its former membership. This lets the player perform precise transfers using only broad cyclic shifts.

There are no locks, counters, timers, hidden tile properties, random events, or failure states. Experimentation is unrestricted.

1. Player-facing rules

The board is a square mosaic completely filled with tiles. Each board cell has a fixed recessed target sample. Each movable tile carries a corresponding colored motif.

A tile whose motif exactly matches the target sample in its current socket receives a bright white fitted outline. An unmatched tile has a gray outline. The level completes when every tile is fitted.

Every row has:

A left-edge handle pointing left.

A right-edge handle pointing right.

Every column has:

A top-edge handle pointing up.

A bottom-edge handle pointing down.

Activating a handle shifts its whole row or column one logical cell in the arrow’s direction. The tile pushed beyond the board wraps to the opposite end.

Matched tiles are never locked. Shifting a completed line removes its fitted outlines immediately. Applying the opposite handle immediately afterward restores the previous state exactly—but only if no crossing line was shifted in between.

That qualification is the heart of the game:

Shift a row.

Shift a crossing column, extracting one tile and inserting another.

Reverse the row.

Reverse the column.

Most of the board returns to its earlier arrangement, while a small set of tiles around the intersection remains rerouted.

2. Action space

The game uses:

UP

DOWN

LEFT

RIGHT

SPACE

Click actions are accepted by the harness but have no effect. There are zero clickable regions, no hover feedback, and no click-based control mode.

Directional actions move the focus frame among perimeter handles. SPACE activates the focused handle. A board tile is never individually selected.

There is no move limit, energy meter, timer, score requirement, or optimality requirement. Any sequence that reaches the solved mosaic is accepted.

3. Visual contract
3.1 Canvas and logical lattice

The render canvas is exactly 64×64 pixels.

Logical board coordinates are:

Rows r = 0 … N−1, top to bottom.

Columns c = 0 … N−1, left to right.

One logical cell is a 10×10-pixel slot.

A line shift moves every affected tile by exactly one logical cell, or 10 rendered pixels.

For board size N, define:

board_x = board_y = floor((64 − 10N) / 2)

Thus:

Board	Pixel extent
3×3	(17,17) through (46,46)
4×4	(12,12) through (51,51)
5×5	(7,7) through (56,56)

The cell at logical (r,c) has pixel origin:

sx = board_x + 10c
sy = board_y + 10r

Do not draw internal grid lines or cell borders. Cell separation comes from negative space, target samples, and tile silhouettes.

3.2 Tile and socket module

Every 10×10 logical slot contains:

A fixed 4×4 target sample in its upper-left corner.

A movable connected 9×9 notched plaque occupying most of the slot.

A 4×4 motif on the plaque.

Documentation-only sketch:

TTTT......
TTTT.PPPPP
TTTT.PPPPP
TTTT.PPPPP
.....PPPPP
.PPPPPPPPP
.PPPPPPPPP
.PPPPPPPPP
.PPPPPPPPP
.PPPPPPPPP

T is the fixed target sample. P is the movable plaque. These letters are not rendered.

Exact regions:

Target sample: offsets (0…3, 0…3).

Plaque base square: offsets (1…9, 1…9).

Plaque notch removed from the base square: offsets (1…4, 1…4).

Tile motif: offsets (5…8, 5…8).

The plaque is therefore one connected multi-pixel shape. Its notch exposes the target sample beneath it.

Plaque rendering:

Interior: color 1.

Unmatched contour: color 2.

Matched contour: color 0.

Motif pixels: the tile’s motif hue.

The contour includes both the outer boundary and the inner boundary around the notch, so a correct tile visibly “fits” around its target sample.

Target rendering:

Inactive motif pixels: color 3.

Active motif pixels: the socket’s required motif hue.

When solved, every plaque has a white contour. Rows form consistent hue bands, columns repeat consistent motif shapes, and the white contours make the mosaic visibly regular.

3.3 Tile identity encoding

Documentation labels such as A3 are never rendered.

A tile’s target row determines its hue:

Documentation row	Logical row	Color ID
A	0	9 blue
B	1	14 green
C	2	11 yellow
D	3	12 orange
E	4	6 magenta

A tile’s target column determines its connected 4×4 motif:

G1              G2              G3              G4              G5
.##.            ....            ##..            .##.            ##..
.##.            ####            ##..            ####            ###.
.##.            ####            ##..            ####            .###
.##.            ....            ####            .##.            ..##

For example:

A3 is a blue G3 tile.

D1 is an orange G1 tile.

E5 is a magenta G5 tile.

The target sample at (r,c) displays hue H_r and motif G_{c+1}. Consequently, every target is visible directly; the player never needs an arbitrary symbolic legend.

For smaller boards, only the first N row hues and first N motifs are used.

3.4 Background, handles, focus, and feedback

Use only these color IDs:

Role	Color ID
Background	5
Board backing	4
Target-sample inactive pixels and wrap rail	3
Inactive plaque contour and handles	2
Plaque interior	1
Fitted outline	0
Focus frame	7
Invalid-direction bump	8
Shift/wrap pulse	10
Completion pulse	15
Motif hues	9, 14, 11, 12, 6

No other colors are required.

Handle geometry

Each handle is a connected 5×5 arrow shape. The right-facing bitmap is:

..#..
...#.
#####
...#.
..#..

Left, up, and down handles are rotations or reflections of this shape.

Handle centers are:

L_r = (board_x − 4,       board_y + 10r + 5)
R_r = (board_x + 10N + 3, board_y + 10r + 5)

U_c = (board_x + 10c + 5, board_y − 4)
D_c = (board_x + 10c + 5, board_y + 10N + 3)

The handle hit/render box is the 5×5 square centered there. Inactive handles use color 2. The focused handle uses color 10 and receives a one-pixel color-7 frame around its box.

Wrap rail

Draw a one-pixel color-3 rail immediately outside the board:

left   x = board_x − 1
right  x = board_x + 10N
top    y = board_y − 1
bottom y = board_y + 10N

The rail forms a complete rectangle. Every handle touches the relevant rail side. During a shift, only the two entry/exit points belonging to the activated line pulse color 10.

The rail is visual communication, not an object and not a clickable region.

4. Focus navigation

The focus state is always one handle ID:

L_r, R_r, U_c, or D_c

Let the current handle center be p. For a directional action with unit vector d, consider every other handle center q satisfying:

dot(q − p, d) > 0

For each candidate define:

forward = dot(q − p, d)
side    = abs(dot(q − p, perpendicular(d)))

Select the candidate minimizing:

(side, forward, canonical_rank)

Canonical rank is:

L_0 … L_(N−1),
U_0 … U_(N−1),
R_0 … R_(N−1),
D_0 … D_(N−1)

Consequences include:

RIGHT from L_r goes directly to R_r.

DOWN from U_c goes directly to D_c.

UP or DOWN normally moves along a left or right handle bank.

LEFT or RIGHT normally moves along a top or bottom bank.

At a corner, movement can enter the adjacent handle bank.

If no candidate exists, focus does not change. Its frame moves one pixel toward the requested edge, flashes color 8, and returns during the same step.

Focus never wraps around the perimeter.

For canonical test replays, navigate to each requested handle using a shortest path in this focus graph. Ties use action order:

UP, LEFT, DOWN, RIGHT

No shortest focus path in the authored boards exceeds five directional actions.

5. Formal state machine
5.1 Persistent state
N                 board dimension
P[N][N]           tile identity occupying each socket
focus             one perimeter handle ID
done              Boolean

The target arrangement is immutable:

Target[r][c] = tile whose identity is (r,c)

A diagnostic step counter may exist outside gameplay, but it must not affect rendering, validation, failure, or state transitions.

There is no selected tile, tile orientation, lock state, energy, timer, random seed, or hidden history.

5.2 Shift operations

Rows and columns are cyclic arrays.

For row r:

R_r: P'[r,c] = P[r,(c−1) mod N]
L_r: P'[r,c] = P[r,(c+1) mod N]

For column c:

D_c: P'[r,c] = P[(r−1) mod N,c]
U_c: P'[r,c] = P[(r+1) mod N,c]

Every unaffected cell retains its old value.

All tiles in the chosen line move simultaneously. There are no collisions or empty cells.

L_r and R_r are exact inverses on the current row. U_c and D_c are exact inverses on the current column.

Importantly:

R_r, L_r

restores the whole board, but:

R_r, D_c, L_r, U_c

generally does not, because the column shift changes which tile is present in row r before the row is restored.

5.3 step() resolution
Directional action

Read pre-step focus.

Find the next focus handle using the navigation rule.

Update focus or perform the invalid-direction bump.

Do not alter the tile permutation.

Return the settled frame.

SPACE

Read the currently focused handle.

Copy the pre-step permutation for animation.

Compute the entire cyclic line permutation atomically.

Assign the post-step permutation.

Recompute every fitted/unfitted contour from the new permutation.

Set done = true exactly when all N² cells match.

Produce the deterministic shift animation.

Return the settled post-step frame.

Focus remains on the activated handle.

Click

Any click coordinate from (0,0) through (63,63) is a no-op:

No tile moves.

Focus does not move.

No click marker appears.

No budget is consumed because no budget exists.

Terminal state

Once done is true, all subsequent actions are no-ops and the completed board remains visible.

6. Shift animation

A line shift must have a short deterministic animation because one action moves several objects and wraps one tile.

The logical state changes only through the step() call. Animation microframes are presentation derived from the pre-state and post-state; they are not additional game states and consume no actions.

For a shift of one 10-pixel logical cell:

Render moving plaques at pixel offset 3.

Render moving plaques at pixel offset 7.

Render the settled post-state at pixel offset 10.

For wrapping, render duplicate periodic copies displaced by the full board width or height, clip them to the board, and pulse both relevant rail endpoints. Thus the outgoing tile visibly exits one side while its periodic copy enters the other.

Target samples never move.

Fitted outlines on the moving line are suppressed during the two transient frames. They are recomputed and shown only in the settled frame, preventing an outline from appearing attached to an intermediate position.

On completion:

Show the fully fitted settled frame.

Pulse the outer rail color 15 for two presentation frames.

Leave the solved settled frame visible.

There is no failure or danger animation because the game has no failure or danger state.

7. Documentation notation

Rows and columns in move sequences are zero-indexed.

R3 means activate the right handle of logical row 3 once.

D1² means activate the bottom handle of logical column 1 twice.

Superscripts always mean repeated one-cell shifts, not a hidden multi-cell action.

For compact authored solutions, define the documentation-only crossing loop:

C(r,c;h,v) =
    R_r^h,
    D_c^v,
    L_r^h,
    U_c^v

For example:

C(1,1;2,2)
=
R1, R1, D1, D1, L1, L1, U1, U1

C is not a player command, object, or special rule. It is only shorthand for ordinary handle activations.

Matrices use documentation tile IDs. The target for an N×N level is always:

A1 A2 ... AN
B1 B2 ... BN
...
8. Authored levels
Level 1 — One Turn
Contract

Board: 3×3

Board origin: (17,17)

Initial focus: R1, center (50,32)

Initial matched cells: 6/9

Initial state:

A1 A2 A3
B2 B3 B1
C1 C2 C3
Player-facing discovery

Rows A and C already have matching target samples, tile motifs, and white fitted contours. Row B contains exactly the correct three green tiles, but their motifs are visibly rotated one position.

The right handle for row B is initially focused. Pressing SPACE shifts all three green plaques right. B1 leaves the right edge and visibly enters from the left through the wrap rail. The entire row becomes fitted, followed by completion feedback.

This one action distinguishes the intended rule from plausible alternatives:

It does not move only the nearest tile.

It does not swap two adjacent tiles.

It moves all three tiles simultaneously.

It wraps the edge tile.

Completing every target ends the level.

Exact solution
R1

Canonical raw action stream:

SPACE

Depth:

Shift activations: 1

Raw controller actions: 1

Attribution and necessity check

Only R1 solves in one activation. Every visible matched cell supplies positive goal evidence. The other handles are the fixed reversible control surface and provide clean alternative experiments, not hidden devices or decoys.

Level 2 — Two Axes
Contract

Board: 3×3

Board origin: (17,17)

Initial focus: U0, center (22,13)

Initial matched cells: 3/9

Initial state:

C1 A2 B3
A1 B2 C3
B1 C2 A3

The middle column is fully fitted. The left and right columns contain the correct column motifs but cyclically wrong row hues.

Player-facing discovery

Horizontal shifting cannot fix the outer columns because it leaves every tile in its current row. The wrong row hues are direct evidence that a vertical transfer is required.

Activating focused U0 produces:

A1 A2 B3
B1 B2 C3
C1 C2 A3

The entire left column becomes fitted while the right column remains visibly cyclic. The bottom handle D2 then shifts the right column down, placing A3, B3, and C3.

This teaches the orthogonal form of the same line-shift rule without introducing another device.

Exact solution
U0, D2

Canonical raw action stream:

SPACE,
DOWN, RIGHT, RIGHT, SPACE

The two column repairs commute, so D2, U0 is also valid.

Depth:

Shift activations: 2

Raw controller actions for the canonical order: 5

Attribution and necessity check

No sequence containing only row shifts can solve the board, because row shifts cannot change a tile’s logical row. Both wrong outer columns must be moved vertically. The middle fitted column is positive evidence and is not touched by the intended solution.

Level 3 — The Crossing
Contract

Board: 3×3

Board origin: (17,17)

Initial focus: R1, center (50,32)

Initial matched cells: 6/9

Initial state:

A1 B1 A3
B2 A2 B3
C1 C2 C3

Only three tiles are wrong:

B1 is above its socket.

B2 is left of its socket.

A2 occupies the central socket.

They form a visible three-cycle around the intersection of row 1 and column 1.

Player-facing discovery

No whole wrong line has a common cyclic offset. A row-only or column-only explanation therefore stops making progress.

The focused R1 is a low-cost experiment:

R1 moves B2 onto its socket but moves fitted B3 away.

D1 carries B1 into row 1. The fitted count visibly drops.

L1 restores the row’s broad arrangement while retaining the transferred member.

U1 restores the column and completes the three-cycle.

Exact settled states:

After R1:

A1 B1 A3
B3 B2 A2
C1 C2 C3

After D1:

A1 C2 A3
B3 B1 A2
C1 B2 C3

After L1:

A1 C2 A3
B1 A2 B3
C1 B2 C3

After U1:

A1 A2 A3
B1 B2 B3
C1 C2 C3

This is the first forced demonstration that a correct tile may need to move away and that a reversal after a crossing shift preserves a transfer rather than erasing it.

Exact solution
C(1,1;1,1)

Expanded:

R1, D1, L1, U1

Canonical raw action stream:

SPACE,
DOWN, DOWN, LEFT, SPACE,
LEFT, LEFT, UP, SPACE,
UP, UP, RIGHT, SPACE

Depth:

Shift activations: 4

Raw controller actions: 13

Attribution and necessity check

An exhaustive breadth-first search gives a minimum of four shift activations. Neither row-only nor column-only play can solve the three-cycle.

There is also no path to the goal whose fitted-cell count is nondecreasing after every activation. The temporary loss of fitted outlines is therefore not optional tutorial theater; it is a forced property of the instance.

Level 4 — Borrowed Strip
Contract

Board: 4×4

Board origin: (12,12)

Initial focus: R3, center (55,47)

Initial matched cells: 5/16

Initial state:

A1 A2 A3 A4
B4 C2 D3 C1
C3 D4 B2 C4
B3 B1 D2 D1

The entire top row is fitted. C4 is also fitted. The lower three rows contain interleaved foreign row hues rather than independent row rotations.

Player-facing discovery

The initial focused action has a clear local consequence: R3 wraps D1 from the lower-right corner into its correct lower-left socket.

The broader impasse is that the pristine A row cannot be treated as locked. Several foreign-colored tiles below it must move vertically, and every vertical shift temporarily removes one fitted A tile. The player must use paired row/column reversals to borrow the completed row as transit space and return it intact.

A second interpretation appears near the end: a row should be shifted only after it contains the correct membership. Row B does not initially contain only B tiles. After the crossing transfers, it becomes:

B4 B1 B2 B3

Only then is a direct left shift meaningful.

Exact solution
R3;

C(1,1;2,2);

C(3,1;2,1);

C(1,2;1,2);

C(2,2;2,1);

L1

Expanded activation count:

1 + 8 + 6 + 6 + 6 + 1 = 28

Important checkpoint after 15 activations:

A1 A2 A3 A4
B4 C1 D3 B3
C3 C2 B2 C4
D1 D2 B1 D4

Checkpoint immediately before the final L1, after 27 activations:

A1 A2 A3 A4
B4 B1 B2 B3
C1 C2 C3 C4
D1 D2 D3 D4

Matched-cell regression sequence at phase boundaries:

5 → 6 → 6 → 9 → 10 → 12 → 16

Depth:

Shift activations: 28

Canonical raw controller actions: 91

Necessity and anti-cleanliness check

The foreign row hues below row A prove that column shifts are necessary. Therefore at least part of the initially completed A strip must be moved; preserving every fitted tile is impossible.

The second crossing loop intentionally produces no net outline gain at its boundary. It reroutes C2, C1, and B3 into positions needed by the following branch. Its evidence is positional, not a score increase.

Treating the six displayed phases as indivisible coarse operations, only 4 of their 720 possible orders solve. Those four alternatives arise from a genuinely independent branch, so the level is neither an unordered checklist nor a single corridor-like chain.

Level 5 — Two Repair Fronts
Contract

Board: 5×5

Board origin: (7,7)

Initial focus: R0, center (60,12)

Initial matched cells: 7/25

Initial state:

A2 C2 B2 B3 E5
B1 B4 A5 D3 B5
C1 E4 E1 C4 C5
D1 A1 D4 A4 D2
E3 E2 C3 A3 D5
Player-facing discovery

The top row begins with A2 one position left of its socket. Pressing focused R0 immediately fits A2, but also wraps E5 into the far-left socket. This creates an obvious local question: can the new top-row arrangement be used as a temporary carrier rather than immediately reversed?

Completing the first crossing loop fits both A1 and A2. From there, two repair fronts become visible:

A left/middle front using columns 1 and 2.

A right/bottom front using columns 3 and 4.

They overlap through rows 0, 1, 3, and 4. Some phases can be interleaved, but phases sharing a carrier cannot be arbitrarily reordered.

The same paired reversal has contrasting roles:

The first loop opens the top row to construct a correct prefix.

The sixth loop uses the bottom row to assemble three tiles at once.

The final loop reopens the top row after most of the board is complete.

Exact solution
C(0,1;1,2);

C(3,1;2,1);

C(3,3;1,2);

C(1,3;2,1);

C(1,2;1,1);

C(4,2;2,2);

C(4,4;1,1);

C(0,4;2,1)

Activation count:

6 + 6 + 6 + 6 + 4 + 8 + 4 + 6 = 46

After the fifth loop, at activation 28:

A1 A2 A5 A4 E5
B1 B2 B3 B4 B5
C1 C2 E1 C4 C5
D1 D2 D3 D4 E4
E3 E2 C3 A3 D5

After the seventh loop, at activation 40:

A1 A2 A5 A4 E5
B1 B2 B3 B4 B5
C1 C2 C3 C4 C5
D1 D2 D3 D4 D5
E1 E2 E3 E4 A3

The final loop resolves the remaining A3, A5, and E5 cycle.

Matched-cell regression sequence:

7 → 9 → 11 → 13 → 15 → 17 → 20 → 22 → 25

Within the first loop, the count temporarily falls to 6 before rising to 9.

Depth:

Shift activations: 46

Canonical raw controller actions: 153

Necessity and anti-cleanliness check

Each loop changes membership at a crossing; replacing the layout with independent row rotations would remove the puzzle.

Among all 8! = 40,320 permutations of the eight listed coarse loops, exactly 160 solve. Thus:

The loops are not eight independent errands.

Several disjoint fronts genuinely can be interleaved.

Overlapping carrier lines impose substantial order constraints.

The supplied route is one valid topological order, not a hidden recipe required by the validator.

Level 6 — False Finish
Contract

Board: 5×5

Board origin: (7,7)

Initial focus: R2, center (60,32)

Initial matched cells: 8/25

Initial state:

D1 A4 A3 C3 A1
E3 A2 B5 B4 C2
B2 E5 C1 C4 A5
B3 D2 D3 D4 D5
E1 C5 E4 B1 E2

Row D appears almost complete:

B3 D2 D3 D4 D5

Its missing D1 is visibly located at the upper-left socket.

Player-facing discovery

A greedy player can attempt to move D1 directly down column 0, but that leaves the obstructing B3 and the surrounding carrier relationships unresolved.

The initially focused row is row C. Its C4 tile is already fitted. Activating R2 moves that correct tile away, immediately reducing the fitted count from 8 to 7. Because the opposite handle remains available, this is a safe but unmistakably non-greedy experiment.

Two right shifts place useful members of row C around column 0. The subsequent column shift, row reversal, and column reversal prepare the left-side transfer without yet producing a fitted tile. The next loop then places both A1 and D1, completing the visually tempting D-row edge in a way compatible with the remaining puzzle.

The familiar mechanics change roles over the solution:

Row D begins as an apparent finish line.

Row C begins as sacrificial workspace.

Row D later becomes a stable reference repeatedly crossed and restored.

Row C ends as the final uniformly offset strip.

Exact solution
C(2,0;2,1);

C(0,0;1,2);

C(0,3;2,1);

C(4,3;1,2);

C(4,1;2,2);

C(1,1;1,1);

C(1,4;2,1);

C(2,4;1,2);

L2²

Activation count:

6 + 6 + 6 + 6 + 8 + 4 + 6 + 6 + 2 = 50

After 12 activations:

A1 A4 A3 C3 B3
B2 A2 B5 B4 C2
C4 E5 C1 E3 A5
D1 D2 D3 D4 D5
E1 C5 E4 B1 E2

Immediately before the final two shifts, at activation 48:

A1 A2 A3 A4 A5
B1 B2 B3 B4 B5
C4 C5 C1 C2 C3
D1 D2 D3 D4 D5
E1 E2 E3 E4 E5

Row C is now a pure two-step cyclic offset. Two L2 activations complete it.

Matched-cell regression sequence:

8 → 7 → 9 → 10 → 12 → 14 → 17 → 19 → 20 → 25

Depth:

Shift activations: 50

Canonical raw controller actions: 151

Necessity and anti-cleanliness check

Moving the final L2² phase to the beginning and then executing the other eight phases leaves only 17 of 25 cells fitted. The final row correction does not commute with its earlier use as a carrier.

Among the 9! = 362,880 permutations of the nine coarse phases, 780 solve. This permits several independent repair-front interleavings while still enforcing a dense set of ordering constraints.

The first phase is the progression’s sharpest reinterpretation of matched outlines: it has a necessary staging purpose even though its boundary state has one fewer fitted tile.

Level 7 — Closed Mosaic
Contract

Board: 5×5

Board origin: (7,7)

Initial focus: R0, center (60,12)

Initial matched cells: 1/25

Initial state:

E1 A3 E3 C5 D1
B2 E2 D4 C2 A5
A1 A2 C3 B4 C1
D3 E5 B3 D5 B5
A4 D2 E4 C4 B1

Only C3 at the center begins fitted.

Player-facing discovery

There are no new rules or objects. The puzzle asks the player to build a stable structure out of a densely interdependent permutation.

The initial focused R0 has immediate evidence: one right shift fits A3. A second right shift removes that fit but places the top-row members around column 0 for a localized crossing loop. Completing the loop raises the total from one fitted tile to three and establishes a reusable first hypothesis.

The solution develops several fronts around a cycle of shared carriers:

Top-left through row 0 and column 0.

Left-center through row 2 and column 0.

Center-right through row 2 and column 3.

Bottom-right through row 4 and column 3.

Bottom-left through row 4 and column 1.

Upper-left through row 1 and column 1.

Upper-right through row 1 and column 4.

Lower-right through row 3 and column 4.

Center closure through row 3 and column 2.

Final top-center three-cycle through row 0 and column 2.

At activation 24, the central C row becomes fully fitted. It remains fitted at every later coarse checkpoint, but later column shifts temporarily open and restore it. The same completed line is simultaneously:

A visible progress landmark.

A temporary transit line.

A constraint on where subsequent interference should occur.

Exact solution
C(0,0;2,1);

C(2,0;1,2);

C(2,3;2,1);

C(4,3;1,2);

C(4,1;2,1);

C(1,1;1,2);

C(1,4;2,1);

C(3,4;1,2);

C(3,2;2,2);

C(0,2;1,1)

Activation count:

6 + 6 + 6 + 6 + 6 + 6 + 6 + 6 + 8 + 4 = 60

Central-row checkpoint at activation 24:

A1 A3 E3 A4 D1
B2 E2 D4 B4 A5
C1 C2 C3 C4 C5
D3 E5 B3 D5 B5
E1 D2 A2 E4 B1

Final three-cycle checkpoint at activation 56:

A1 A3 E3 A4 A5
B1 B2 B3 B4 B5
C1 C2 C3 C4 C5
D1 D2 D3 D4 D5
E1 E2 A2 E4 E5

The last loop reroutes A2, A3, and E3 without disturbing the four completed rows at its boundary.

Matched-cell regression sequence:

1 → 3 → 6 → 8 → 10 → 12 → 15 → 16 → 19 → 22 → 25

Depth:

Shift activations: 60

Canonical raw controller actions: 193

Necessity and anti-cleanliness check

Every row and every column is activated in both directions somewhere in the intended solution. No line can be removed from the authored dependency graph.

Adjacent coarse phases show mixed ordering behavior:

The first and second loops cannot be swapped.

The second and third can be interleaved.

The third and fourth cannot be swapped.

The fourth and fifth can be interleaved.

The fifth and sixth cannot be swapped.

The sixth and seventh can be interleaved.

The seventh and eighth cannot be swapped.

This creates a dependency network with multiple active fronts rather than a single ten-stage corridor.

The central row’s completed state cannot be treated as a lock. Later phases necessarily cross it, and their paired reversals restore it. The board’s spatial structure therefore remains essential through the final move.

9. Progression audit
Level	New discovery or reinterpretation	Shift depth
1	A handle cyclically shifts an entire row; wrap and all-matched completion	1
2	The identical operation applies vertically; arrow direction matters	2
3	Crossing shifts change line membership; paired reversals localize a transfer	4
4	A fully completed strip can be borrowed and restored; correct membership must precede line alignment	28
5	Several localized transfers form partially independent repair fronts	46
6	Fitted outlines are not a monotonic objective; a correct tile can be an intentional staging sacrifice	50
7	Completed lines become reusable structural backbones inside a full dependency network	60

No level introduces a new noun after the foundational curriculum. Difficulty comes entirely from placement, line membership, crossing order, temporary breakage, restoration, and phase dependency.

The same operational rules acquire contrasting roles:

A row shift directly solves level 1.

A row shift stages a column transfer in level 3.

A completed row becomes temporary workspace in level 4.

Paired reversals isolate one repair front in level 5.

A fitted tile becomes an obstruction that must move in level 6.

A completed central row becomes both backbone and transit line in level 7.

10. Epistemic-solvability audit
Foundational attribution

Level 1: Six fitted examples visibly establish what “correct” looks like. The focused action moves a whole line and visibly wraps a tile, ruling out local-swap interpretations.

Level 2: Wrong row hues cannot be corrected horizontally. The board itself demonstrates why vertical movement is required.

Level 3: Only a crossing transfer can solve the visible three-cycle. Exhaustive search confirms that temporary loss of fitted cells is unavoidable.

Later experimentation

Later boards retain four forms of evidence:

Direct target visibility. Every tile’s destination remains visible through hue and motif.

Line-membership evidence. A row containing foreign hues visibly requires vertical transfers before horizontal alignment can solve it.

Immediate causal feedback. Every shift moves exactly one line, pulses its wrap endpoints, and recomputes outlines in the settled frame.

Safe reversibility. The opposite handle can undo an exploratory shift immediately, with no budget penalty.

Each late level begins with the focus on a productive local experiment, but the focus remains where the player leaves it afterward; it does not trace the rest of the solution.

11. Validator and regression contract

The implementation should assert all of the following:

number_of_levels == 7

For every level:

flatten(initial_grid) is a permutation of flatten(target_grid)

Inverse identities hold for every arbitrary legal permutation:

L_r(R_r(P)) == P
R_r(L_r(P)) == P
U_c(D_c(P)) == P
D_c(U_c(P)) == P

Crossing operations are not incorrectly collapsed:

R_r, D_c, L_r, U_c

must be applied sequentially to the current permutation, not optimized into two cancelling pairs.

Completion is exactly:

done iff for all r,c: P[r][c] == Target[r][c]

The following are ignored by completion:

Focus position.

Number of steps.

Which handles were used.

Whether the supplied intended route was followed.

Click history.

Animation phase.

Replay regression counts:

Level 1:  6 → 9
Level 2:  3 → 6 → 9
Level 3:  6 → 6 → 4 → 6 → 9
Level 4:  5 → 6 → 6 → 9 → 10 → 12 → 16
Level 5:  7 → 9 → 11 → 13 → 15 → 17 → 20 → 22 → 25
Level 6:  8 → 7 → 9 → 10 → 12 → 14 → 17 → 19 → 20 → 25
Level 7:  1 → 3 → 6 → 8 → 10 → 12 → 15 → 16 → 19 → 22 → 25

Applying every authored activation sequence to its specified initial matrix must produce the row-major target exactly. The intended routes are regression witnesses and difficulty references, not hidden optimality conditions.
