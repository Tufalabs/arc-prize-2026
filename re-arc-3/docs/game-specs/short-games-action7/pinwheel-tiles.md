Pinwheel Tiles

Exact level count: 7

“This is a game about turning a small square clockwise becoming difficult because every overlapping turn can be used only once.”

Pinwheel Tiles is a fully observable permutation puzzle. The player clicks fixed pinwheel hubs; each hub rotates the tiles in its visibly marked 2×2 or 3×3 footprint exactly 90° clockwise, then permanently cracks. The same tile cells may belong to several footprints, so a tile can be transferred from one rotational orbit to another—but only if the receiving pivot has not already been spent.

The interesting tension is that a click changes two things at once:

It permutes several tiles.

It permanently removes that permutation from the remaining action set.

There are no keys, counters, invisible modes, tile orientations, random outcomes, or checker-like secondary requirements. Every fact needed for planning is visible in the settled frame.

1. Player-facing rules

The player sees a compact collection of socket pads. Every socket has a required symbol indicated by colored corner marks. A movable tile sits over each socket:

A correct tile receives a bright white rim.

An incorrect tile has a dark rim, while the socket’s corner marks remain visible around it.

When every tile matches its socket, the level completes automatically.

A visible pinwheel badge marks every clickable pivot:

A small footprint outline shows exactly which square of cells it affects.

Clicking an unused pivot rotates the tiles in that footprint one quarter-turn clockwise.

The hub itself and its footprint do not move.

After the rotation, the hub becomes gray and cracked.

A cracked hub cannot be used again.

Clicking a cracked hub or any ordinary board cell produces a brief red blocked pulse and changes nothing.

For a 3×3 footprint, the center tile remains in its cell under that rotation. An overlapping footprint may still move the tile occupying that center.

Tiles orbit without rotating their own graphics. Symbol orientation is therefore not a state variable.

2. Action space

Only cell clicking has gameplay effect.

Input	Effect
Click an unused pivot badge	Execute its clockwise rotation and spend it
Click a spent pivot badge	Red blocked pulse; no logical change
Click any other canvas cell	Red blocked pulse; no logical change
UP, DOWN, LEFT, RIGHT, SPACE	Hard no-op

There is no persistent selected object and no logical cursor. A click acts immediately.

Level restart is supplied by the host environment’s normal reset() operation, outside the 64×64 observation and outside the puzzle action space. Retries are unlimited. There is no in-game energy, timer, or move budget.

3. Visual contract
3.1 Canvas and native board

The render canvas is always 64×64 pixels.

The native board is an H × W array of logical socket cells, with H,W ≤ 6 in the authored campaign.

Each socket occupies 7×7 pixels. Adjacent sockets are separated by a 2-pixel gutter, giving a pitch of 9 pixels.

board_width  = 9W - 2
board_height = 9H - 2

origin_x = floor((64 - board_width) / 2)
origin_y = floor((64 - board_height) / 2)

For zero-index logical cell (r,c), the socket box is:

x = origin_x + 9c ... origin_x + 9c + 6
y = origin_y + 9r ... origin_y + 9r + 6

There are no row or column grid lines. Socket pads are separated islands on the dark background. Only pivot footprints create larger enclosing outlines.

3.2 Base colors
Use	Color ID
Canvas background and gaps	4
Socket pad	3
Tile slab	1
Ordinary incorrect tile rim	4
Correct tile rim	0
Unused hub body	10
Hub arrow	12
Spent hub body	2
Spent crack	5
Blocked-click pulse	8
Completion border pulse	7
Exhausted-unsolved border pulse	13
3.3 Tile and socket construction

A tile is a connected 5×5 slab occupying local socket pixels (1..5,1..5). It has:

a neutral color-1 body;

a 3×3 colored central motif;

colored identity tabs at selected tile corners;

a white perimeter when correct or dark perimeter when incorrect.

A socket’s requirement remains visible through colored L-shaped marks at selected outer corners:

TL: (0,0), (1,0), (0,1)
TR: (6,0), (5,0), (6,1)
BL: (0,6), (1,6), (0,5)
BR: (6,6), (5,6), (6,5)

The tile has corresponding one-pixel identity tabs at local slab corners:

TL: (1,1)
TR: (5,1)
BL: (1,5)
BR: (5,5)

When tile and socket agree, their inner tabs and outer brackets line up radially. Thus matching does not require memorizing a symbolic legend.

3.4 Symbol catalogue

The letters below are documentation identifiers only. No letters are rendered.

ID	Accent	3×3 motif (# = accent)	Active corners
A	9	.#. / ### / .#.	TL, BR
B	9	### / .#. / .#.	TR, BL
C	6	#.. / #.. / ###	TL, BL
D	6	#.# / ### / #.#	TR, BR
E	11	##. / .## / ##.	TL, TR
F	11	### / #.# / ###	BL, BR
G	14	#.. / ##. / ###	TL, TR, BR
H	14	#.# / ### / .#.	TR, BL, BR

Only four saturated symbol colors are present. Symbols sharing a color have strongly different silhouettes and corner signatures.

3.5 Pivot badges and hit regions

Pivots are fixed overlays, not movable board contents.

2×2 pivot

For a 2×2 footprint with zero-index top-left (r0,c0), the 4×4 visible badge and click hitbox is:

x = origin_x + 9c0 + 6 ... origin_x + 9c0 + 9
y = origin_y + 9r0 + 6 ... origin_y + 9r0 + 9

It straddles the central gutter intersection of the four affected sockets.

3×3 pivot

For a 3×3 footprint with zero-index top-left (r0,c0), its badge overlays the center socket (r0+1,c0+1).

If that socket begins at pixel (px,py), the click hitbox is:

x = px + 1 ... px + 5
y = py + 1 ... py + 5

The visible badge is a connected plus-like pinwheel occupying that region. The tile beneath remains identifiable through its outer corner tabs, socket marks, and rim.

All authored pivot hitboxes are disjoint. The level validator must reject a level containing intersecting pivot hitboxes.

3.6 Footprint outlines

Each pivot has a square outline one pixel outside the pads in its footprint. The outline remains present after use.

Four quiet styles are used:

Style	Rendering
0	Continuous color-1 line
1	Color-1 line with repeating draw, draw, gap pattern
2	Continuous color-2 line with white caps at top-left and bottom-right
3	Alternating color-2/color-3 line with white caps at top-right and bottom-left

The badge’s outer ring repeats the same style. Any two footprints sharing at least one logical cell receive different styles.

If multiple outline segments request the same pixel, sort their style IDs and alternate them along the shared segment by boundary-pixel index. The geometric center of each footprint and its matching badge ring remain the primary association cues.

3.7 Drawing order

Background.

Footprint outlines.

Socket pads and requirement marks.

Tile slabs.

Tile rims, motifs, and identity tabs.

Pivot badges.

Blocked pulse, completion pulse, or exhausted pulse.

No text, decorative pattern, legend panel, move counter, or HUD appears.

4. Formal rotation rule

For a footprint of side length s ∈ {2,3} and zero-index top-left (r0,c0), read from a copied pre-rotation tile array:

new_tiles[r0 + j][c0 + s - 1 - i]
    = old_tiles[r0 + i][c0 + j]

for every 0 ≤ i,j < s.

Equivalent examples:

2×2 clockwise

a b       c a
c d   ->  d b
3×3 clockwise

a b c       g d a
d e f   ->  h e b
g h i       i f c

The 3×3 center e maps to itself.

Pivots, sockets, footprint outlines, and tile graphics do not rotate. Only tile positions are permuted.

5. State machine
5.1 Logical state
GameState:
    level_id: integer 1..7
    height: integer
    width: integer
    sockets[height][width]: symbol ID A..H
    tiles[height][width]: symbol ID A..H
    pivots[]:
        id: implementation-only identifier
        top_row: integer
        left_col: integer
        size: 2 or 3
        used: boolean
        style: 0..3
    done: boolean
    transient_pulse:
        kind: none | blocked | exhausted | completed
        pixel_x, pixel_y: optional

The following are not state variables:

tile orientation;

score;

selected pivot;

move count;

pivot charge beyond the visible used bit;

random seed;

hidden solution flags.

matched[r][c] is derived as:

tiles[r][c] == sockets[r][c]
5.2 step() resolution order
step(action):
    clear prior transient pulse

    if done:
        return unchanged completed frame

    if action is not CLICK:
        return unchanged frame

    pivot = hit_test(action.x, action.y)

    if pivot is absent:
        set blocked pulse at clicked cell
        return unchanged logical frame

    if pivot.used:
        set blocked pulse on pivot
        return unchanged logical frame

    old_tiles = copy(tiles)

    compute complete rotated tile array from old_tiles
    assign tiles = rotated result
    set pivot.used = true

    recompute all matched rims
    done = every board cell matches

    if done:
        set completed pulse
    else if every pivot is used:
        set exhausted pulse

    play deterministic rotation animation
    return final settled frame

The copied source array is mandatory. An in-place traversal must not create a cascading rotation.

The pivot becomes spent in the same logical transaction as the tile permutation. Win validation happens after both updates.

5.3 Win condition

A level succeeds exactly when:

for every logical cell (r,c):
    tiles[r][c] == sockets[r][c]

Relevant:

tile symbol identity;

socket symbol identity.

Ignored:

which physical copy of a repeated symbol occupies a socket;

click order;

unused pivots;

number of invalid clicks;

pivot style;

animation phase.

Allowed:

completing with unused pivots, in principle;

repeated symbols.

Forbidden only in the sense of non-completion:

any single mismatched cell.

All seven authored shortest solutions happen to require every listed pivot, but this is a property of the instances, not an extra validator condition.

5.4 Exhausted state

If every pivot has been spent and the board is not solved, the outer canvas border pulses color 13 once. The board then remains visible and inert until the host resets it. There is no automatic asynchronous reset.

6. Rotation animation and feedback

An accepted pivot click must visibly communicate the multi-cell motion.

The animation is contained inside the triggering step() and consumes no additional actions.

Hub flash: the unused arrow flashes color 0.

Lift frame: affected tiles receive a light-blue color-10 ghost rim.

Orbit frames: tile centers follow two fixed intermediate points on a clockwise quarter-circle around the footprint center. Tile graphics remain upright.

Settle frame: tiles snap to their destination socket centers.

Commit feedback: matched rims are redrawn and the hub becomes gray and cracked.

For a 3×3 rotation, the center tile remains in place and briefly pulses during the orbit frames, explicitly distinguishing it from the eight moving tiles.

At the logical level, each source center follows the transform from (x,y) to (-y,x) relative to the footprint center. Intermediate pixel centers are precomputed rather than simulated physically.

The final logical state is committed during the same step() before its settled observation is returned. Human-facing clients may display all subframes; an agent receiving only the settled frame still sees:

the changed arrangement;

the newly cracked pivot;

the unchanged footprint;

all updated correctness rims.

Invalid clicks produce a two-frame color-8 pulse. They never consume or alter a pivot.

Completion produces a three-frame color-7 border pulse after the final hub crack becomes visible. No text overlay obscures the final board.

7. Strategic rules that emerge from the single rotation operation
Rotational orbits

A 2×2 pivot has one four-cell orbit.

A 3×3 pivot has:

one four-corner orbit;

one four-edge-center orbit;

one stationary center.

Overlap transfer

A shared cell is an exit from one pivot’s orbit and an entrance to another’s. A tile can cross a distance no single footprint spans by being staged in that shared cell.

Noncommuting order

Two overlapping clockwise rotations generally satisfy:

rotate(A), then rotate(B) ≠ rotate(B), then rotate(A)

Because each operation disappears after use, the player cannot repair a bad order by simply applying an inverse.

Centers are contextual anchors

The center of a 3×3 pivot is stationary only under that pivot:

It can protect a correct tile.

It can trap an incorrect tile.

A neighboring footprint can load the correct tile into it before use.

A neighboring footprint can repair it after its own pivot has already been spent.

Another large footprint can move a center tile that appeared protected.

The rule never changes; only geometry does.

8. Authored campaign

Coordinates below are 1-indexed (row,column), with (1,1) at the top-left.

For a 2×2 pivot whose footprint starts at (r,c), its hub lies between rows r/r+1 and columns c/c+1.

For a 3×3 pivot, the hub overlays the footprint’s center cell.

Every level begins with:

all pivot.used = false
done = false
selected_pivot = null
transient_pulse = none
animation = none

Letters in matrices are documentation IDs, not rendered characters.

Level 1 — One Turn Home

Purpose: Teach socket matching, the click affordance, the affected footprint, and fixed clockwise motion.

Intended depth: 1 accepted click.

Player-facing discovery

The two cells in the left column already have white rims. The four cells under the single pinwheel do not. This is positive goal evidence: matching symbols produce bright rims.

There is one saturated pinwheel hub and no competing object. Clicking it visibly cycles four distinct tiles clockwise. The left column remains untouched, all six rims become white, and the level completes.

The one action distinguishes clockwise from counterclockwise because the four affected symbols are asymmetrical.

Board

Dimensions: 2 × 3

Sockets:

A B C
D E F

Initial tiles:

A C F
D B E
Pivots
ID	Size	Footprint	Style
P	2	rows 1–2, columns 2–3	0
Replay
P

State after P:

A B C
D E F

Matched-cell checksum:

2 -> 6
Attribution check

The matched left column communicates the objective.

The sole pivot communicates the only useful experiment.

Every moving symbol has a visibly distinct destination.

No one-use planning is required yet.

No unrelated component changes simultaneously.

Necessity check

P is the only accepted operation and is necessary. Shortest depth is 1.

Level 2 — The Cracked Shortcut

Purpose: Establish that each individual hub is usable only once.

Intended depth: 2 accepted clicks. A natural teaching trace includes one additional blocked diagnostic click.

Player-facing discovery

Two familiar 2×2 pivots overlap in the middle column. Initially, the only two mismatches both lie inside the left footprint, making the left hub the productive first experiment.

After clicking L, the two mismatches move into the shared column. Under the plausible but wrong hypothesis that pivots are reusable, clicking L a second time would solve the board. Instead:

L is visibly cracked;

a second click on it produces a red blocked pulse;

R remains bright and unused.

Clicking R solves the same two displaced symbols.

The repeated A/B arrangement is deliberate. In a hypothetical reusable version, L,L would solve. In the actual game, the distinct one-use rule changes the required solution to L,R.

Board

Dimensions: 2 × 3

Sockets:

A A B
A B B

Initial tiles:

B A B
A A B
Pivots
ID	Size	Footprint	Style
L	2	rows 1–2, columns 1–2	0
R	2	rows 1–2, columns 2–3	1
Minimal replay
L, R

After L:

A B B
A A B

After R:

A A B
A B B

Natural diagnostic trace:

L, L again [blocked], R

Matched-cell checksum for accepted rotations:

4 -> 4 -> 6

The first click does not increase the number of matches; it physically relocates the mismatched pair into the receiving footprint.

Wrong first experiment

Clicking R first produces:

B A A
A B B

The incorrect A at r1c3 is now outside the only remaining footprint L, making the failure spatially visible.

Attribution check

The only new causal unknown is pivot availability:

Both operations are already-familiar clockwise 2×2 turns.

The cracked appearance and blocked re-click immediately test one-use behavior.

The right pivot is adjacent to the local impasse.

No tight budget punishes the experiment.

Necessity check

Shortest depth is 2. The only solving accepted-click order is L,R; both pivots are required.

Level 3 — Shared Column

Purpose: Teach genuine overlap transfer and noncommuting order.

Intended depth: 2 accepted clicks.

Player-facing discovery

The board uses the same two overlapping footprints as Level 2, so there is no new noun.

Tile D begins at r2c3 but belongs at r2c1. No single footprint contains both cells:

R can move D from exclusive right cell r2c3 into shared cell r2c2.

L can then move it from r2c2 into exclusive left cell r2c1.

This makes the shared column an observable transfer point.

Board

Dimensions: 2 × 3

Sockets:

A B C
D E F

Initial tiles:

B C F
A E D
Pivots
ID	Size	Footprint	Style
L	2	rows 1–2, columns 1–2	0
R	2	rows 1–2, columns 2–3	1
Replay
R, L

After R:

B E C
A D F

D is now visibly staged at shared cell r2c2.

After L:

A B C
D E F

Matched-cell checksum:

1 -> 2 -> 6
Wrong order

Clicking L first produces:

A B F
E C D

D is still in the right footprint. R can move it only as far as the shared column, but L has already been spent. The target at r2c1 is visibly unreachable.

Attribution check

This cannot be explained by the Level 2 rule alone:

A unique tile must cross from one exclusive side to the other.

The shared cell is necessarily occupied at the intermediate state.

Reversing the same two clicks fails.

The failure is a stranded tile outside all remaining relevant footprints, not an abstract rejection.

Necessity check

Shortest depth is 2. The only solving order is R,L.

Level 4 — Still Point

Purpose: Introduce 3×3 rotation and its stationary center, then immediately reuse existing 2×2 operations in a non-monotonic composition.

Intended depth: 6 accepted clicks.

Player-facing discovery

The new 3×3 hub M surrounds a visibly correct D at r2c2. Several symbols around that center are displaced along the corner and edge orbits.

Clicking M produces the distinguishing observation:

eight surrounding tiles orbit clockwise;

the central D does not move;

its white rim remains visible throughout;

the larger hub cracks exactly like a 2×2 hub.

Later, pivot Q deliberately reduces the number of correct cells. This prevents the player from treating white-rim count as a score that must rise every turn: Q stages tiles for both lower pivots.

Board

Dimensions: 4 × 4

Sockets:

G A A D
B D E C
H H F G
B E C F

Initial tiles:

D A E F
A D G C
G B H F
H E B C
Pivots
ID	Size	Footprint	Style
M	3	rows 1–3, columns 1–3	0
T	2	rows 1–2, columns 3–4	2
R	2	rows 2–3, columns 3–4	1
Q	2	rows 3–4, columns 2–3	2
B	2	rows 3–4, columns 3–4	3
L	2	rows 3–4, columns 1–2	1

Footprints with a shared cell always have different styles.

Example replay
M, T, R, Q, B, L

B and L may be swapped at the end.

Matched-cell checksum:

4 -> 7 -> 8 -> 11 -> 9 -> 13 -> 16

The drop from 11 to 9 occurs on Q.

Key states

After M:

G A D F
B D A C
H G E F
H E B C

The central D at r2c2 has remained stationary and correct.

After R:

G A A D
B D E C
H G F F
H E B C

After the apparently regressive Q:

G A A D
B D E C
H E G F
H B F C

Q has broken two visible matches but placed the lower symbols into the correct incoming positions for B and L.

Player reasoning

Before the first action, known 2×2 operations cannot account for the ring-like displacement around the matched center. The unfamiliar larger outline is located exactly around that local structure.

After the 3×3 observation, every subsequent action follows established rules. The challenge becomes deciding when a correct-looking local arrangement must be sacrificed for two downstream footprints.

Anti-cleanliness and necessity check

Shortest depth is 6.

Every pivot is necessary.

There are exactly two solving click orders: the listed path and the same path with final B,L swapped.

The level cannot be reduced to a corridor or checklist: Q simultaneously modifies both lower branches and must occur before either branch closes.

A greedy “never decrease matches” policy fails.

Level 5 — Load the Anchor

Purpose: Reinterpret a stationary 3×3 center from protected asset into a cell that must first be loaded through an overlapping 2×2 pivot.

Intended depth: 8 accepted clicks.

Player-facing discovery

The center of 3×3 pivot M is r2c2. Its socket requires H, but the cell initially contains G.

The correct H is visibly adjacent at r1c2. Because the player already knows M cannot move its own center, clicking M cannot solve that mismatch. The nearby 2×2 pivot C provides the local experiment:

C rotates H from r1c2 into r2c2;

the center immediately gains a white rim;

M can then rotate its ring while preserving the newly loaded H.

The same center rule has not changed. Its strategic meaning has.

Board

Dimensions: 5 × 5

Sockets:

D G C B H
A H A A F
E D B C F
D G E E C
F A H B G

Initial tiles:

C H C H F
B G F A A
D A D C G
D G A E B
E F B E H
Pivots
ID	Size	Footprint	Style
C	2	rows 1–2, columns 1–2	1
M	3	rows 1–3, columns 1–3	0
T	2	rows 1–2, columns 3–4	1
R	2	rows 1–2, columns 4–5	0
L	2	rows 3–4, columns 1–2	2
E	2	rows 3–4, columns 2–3	1
Q	3	rows 3–5, columns 3–5	2
Z	2	rows 4–5, columns 1–2	0
Example replay
C, M, R, E, Z, Q, T, L

Matched-cell checksum:

6 -> 7 -> 9 -> 11 -> 10 -> 11 -> 18 -> 21 -> 25

E temporarily reduces the count from 11 to 10.

Center-loading trace

Initial upper-left region:

C H C
B G F
D A D

After C:

B C C
G H F
D A D

H has entered the stationary center at r2c2.

After M:

D G B
A H C
D F C

The ring moved; the center H did not.

Dependency structure

The mandatory precedence relationships, after removing transitive consequences, are:

C -> M
M -> T
M -> E
R -> T
E -> Q
E -> Z
Z -> L

This is not a single chain:

R can be performed on a separate upper-right branch.

E feeds both the large lower-right pivot and the lower-left branch.

T, Q, Z, and L allow several valid interleavings once their inputs are staged.

Anti-cleanliness and necessity check

Shortest depth is 8.

All eight pivots are necessary.

Exhaustive search on the actual repeated-symbol board finds 75 valid full click orders.

The local center-loading dependency C -> M is unavoidable in every solution.

E is necessary despite briefly worsening the visible match count.

No pivot is a decoy or optional errand.

Level 6 — Crossed Centers

Purpose: Show that a center stationary under one 3×3 pivot can be moved by another 3×3 pivot, including after its own pivot has been spent.

Intended depth: 10 accepted clicks.

Player-facing discovery

Three 3×3 footprints lie diagonally:

A is centered at r2c2.

B is centered at r3c3.

C is centered at r4c4.

The centers of A and B lie in each other’s rotating rings.

Initially:

B’s center at r3c3 is correct.

A’s center is incorrect.

C’s center is incorrect.

Using A necessarily rotates the correct tile out of B’s center. Pivot B cannot repair its own center because that cell is stationary under B. Later, 2×2 pivot Y must restore it.

Conversely, B rotates the correct H into A’s center after A has already been spent. A cracked pivot therefore does not freeze the tile beneath its hub; it only removes that pivot’s own operation.

Board

Dimensions: 5 × 5

Sockets:

E C E C D
F H B B A
D A D G A
A C G H B
E H G F F

Initial tiles:

E F D C A
C A G B B
E F D B F
D H A C H
A E G H G
Pivots
ID	Size	Footprint	Style
A	3	rows 1–3, columns 1–3	1
B	3	rows 2–4, columns 2–4	0
C	3	rows 3–5, columns 3–5	2
N	2	rows 1–2, columns 4–5	3
W	2	rows 3–4, columns 1–2	2
S	2	rows 4–5, columns 1–2	1
E	2	rows 2–3, columns 4–5	1
D	2	rows 4–5, columns 4–5	1
X	2	rows 1–2, columns 3–4	2
Y	2	rows 3–4, columns 2–3	3
Example replay
X, A, N, B, C, W, E, D, Y, S

Matched-cell checksum:

5 -> 4 -> 6 -> 10 -> 13 -> 14 -> 12 -> 15 -> 18 -> 21 -> 25

The first click and W both reduce the immediate number of matches.

Center-state checksum

At the three 3×3 centers:

Moment	A center r2c2	B center r3c3	C center r4c4
Initial	A, wrong	D, correct	C, wrong
After A	A, wrong	G, wrong	C, wrong
After B	H, correct	G, wrong	B, wrong
After D	H, correct	G, wrong	H, correct
After Y	H, correct	D, correct	H, correct

A’s correct center is installed by B, not by A.

Dependency network

X straddles the seam between the A branch and the N branch. It stages an input for each, so it must occur first.

A transitive reduction of the mandatory ordering is:

X -> A
X -> N

A -> B
N -> B

B -> C
B -> W

C -> E
C -> D
C -> Y

W -> Y
Y -> S

Thus:

A and N are independent after X.

They merge at B.

C and W split again.

E and D may interleave.

C and W both feed Y.

Y repairs the crossed center before S closes the southwest branch.

Anti-cleanliness and necessity check

Shortest depth is 10.

Every pivot is necessary.

Exhaustive search finds 64 solving full orders.

The level contains genuine branching and merging dependencies rather than a single route.

A correct center acts first as an asset, then as a required sacrifice, then as a repair target.

Spatial overlap—not a hidden predicate—creates every ordering constraint.

Level 7 — Whole Pinwheel

Purpose: Combine one-use scheduling, transfer, large-ring orbits, loaded centers, centers repaired after use, protected centers, deliberate temporary damage, and a branching dependency network.

Intended depth: 12 accepted clicks.

Player-facing discovery

The 6×6 board contains five 3×3 footprints and seven 2×2 bridges. Despite the density:

every hub has a disjoint hitbox;

no overlapping footprints share a style;

every center and overlap is visible;

there are no new object types.

Several center roles coexist:

M’s center is wrong and must be loaded by C before M.

NW’s center remains wrong when NW is used; M repairs it afterward.

NE’s center is loaded by X before NE.

SW’s center remains wrong through its own rotation and is repaired later by Y.

SE’s center begins correct and is preserved.

Board

Dimensions: 6 × 6

Sockets:

A E A B B B
F G D E H A
C D C D F H
A C G C E B
H B H D G A
F D F G C E

Initial tiles:

A B B B A E
E H B D E H
A F D D B D
H G C C F A
G A D E G C
C H F C G F
Pivots
ID	Size	Footprint	Style
NW	3	rows 1–3, columns 1–3	2
NE	3	rows 1–3, columns 4–6	2
SW	3	rows 4–6, columns 1–3	2
SE	3	rows 4–6, columns 4–6	2
M	3	rows 2–4, columns 2–4	0
T	2	rows 1–2, columns 3–4	1
L	2	rows 3–4, columns 1–2	1
R	2	rows 3–4, columns 5–6	0
B	2	rows 5–6, columns 3–4	0
C	2	rows 3–4, columns 3–4	1
X	2	rows 2–3, columns 4–5	3
Y	2	rows 4–5, columns 2–3	3

Pivots sharing style 2 occupy non-overlapping quadrant footprints. Every overlapping pair has a different style.

Example replay
T, NW, C, M, SW, X, NE, R, SE, B, Y, L

Matched-cell checksum:

6 -> 7 -> 10 -> 9 -> 10 -> 12
  -> 13 -> 18 -> 20 -> 26 -> 29 -> 32 -> 36

C deliberately drops the count from 10 to 9 while loading M’s stationary center.

Center-state progression

Initially:

Pivot center	Initial	Required
NW at r2c2	H	G
NE at r2c5	E	H
M at r3c3	D	C
SW at r5c2	A	B
SE at r5c5	G	G

Key consequences:

NW is used while its own center is still wrong.

C moves the required C from r4c3 into M’s center r3c3.

M preserves that C and moves the required G into the already-spent NW center.

X loads the NE center with H.

NE preserves its newly correct center.

SW rotates while its center remains wrong.

Y later installs B into the spent SW center.

SE preserves its initially correct center throughout its own turn.

Exact dependency network

Every solution obeys:

T -> NW -> C -> M

After M, two branches open:

M -> X -> NE -> R -> SE -> B
M -> SW -----------------> B

They merge at B, followed by:

B -> Y -> L

SW is not optional. It may be placed at any of five positions:

before X;

between X and NE;

between NE and R;

between R and SE;

between SE and B.

This produces exactly five solving click orders. The example replay uses the first position.

Why the schedule is inspectable

T stages tiles into the overlap between NW and the northern half of the board.

NW must be consumed before M repairs its center; otherwise NW would rotate that repaired center away.

C visibly loads M’s center.

X, R, and the eastern large pivots form a spatially continuous transfer branch.

SW is independent of the eastern branch but must occur before B, which consumes the cells where the branches meet.

Y repairs the visibly wrong SW center.

L closes the last two southwest mismatches.

The player is scheduling a network, not following objects in the order they appear on the board.

Anti-cleanliness and necessity check

Shortest depth is 12.

All 12 pivots are necessary.

Exactly five full click orders solve.

The flexible SW operation is a real branch, not a decoy: omitting it or moving it outside the interval M < SW < B makes completion impossible.

The board contains deliberate non-monotonic progress, delayed center repair, two dependency branches, and a merge.

No new rule appears in the final level.

9. Authored verification summary

The state graph for verification is (tile matrix, used-pivot mask). The values below are for the actual repeated-symbol instances, not for artificially unique tile labels.

Level	Shortest accepted clicks	Required pivots	Solving shortest orders
1	1	1/1	1
2	2	2/2	1
3	2	2/2	1
4	6	6/6	2
5	8	8/8	75
6	10	10/10	64
7	12	12/12	5

The accepted-click depths are intentionally lower than movement-game targets. Each click is an irreversible macro-permutation affecting four or nine cells and removing one operation from the future action set. Inflating the final level to 50 clicks would require roughly 50 simultaneously legible one-shot hubs, undermining the core visual and epistemic constraints. Complexity instead comes from the number and structure of possible schedules, not action padding.

10. Implementation validation checklist

A level definition is valid only when all of the following hold:

num_levels == 7

1 <= height,width <= 6
len(sockets) == len(tiles) == height
every row length == width

every symbol is A..H
multiset(tiles) == multiset(sockets)

every pivot:
    size is 2 or 3
    footprint is in bounds
    used is initially false
    hitbox is in the 64×64 canvas

all pivot hitboxes are pairwise disjoint

for every pair of pivots whose footprints share a cell:
    style IDs differ

initial tiles do not already satisfy every socket

replaying the documented path:
    uses no pivot twice
    reaches exact socket equality
    marks every clicked pivot spent

direction keys and SPACE:
    never change tiles, pivots, done, or selection

ordinary and spent clicks:
    never change tiles or used bits

Recommended regression assertions:

shortest_depths = [1, 2, 2, 6, 8, 10, 12]
shortest_order_counts = [1, 1, 1, 2, 75, 64, 5]
11. Epistemic-solvability audit

The foundational curriculum gives one useful causal distinction at a time:

Level 1: a hub rotates its outlined cells clockwise, and matching produces white rims.

Level 2: the used hub is permanently unavailable; the natural repeated click is visibly blocked.

Level 3: a shared cell transfers a tile between footprints, making order consequential.

Every later development is a consequence of those observations:

A larger square has visible corner, edge, and center orbits.

A center is stationary only relative to its own pivot.

Correct cells may need to be displaced for downstream transfers.

Overlap creates partial-order dependencies.

One branch can be scheduled flexibly while still being necessary.

Wrong orders strand identifiable symbols outside the footprints that remain.

There are no optional stations, arbitrary checkers, hidden exceptions, symbolic codes, or unrelated late devices. All seven levels use the same tiles, sockets, footprint outlines, and one-shot clockwise pivots.
