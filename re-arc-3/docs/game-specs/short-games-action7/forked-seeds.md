Forked Seeds

Exact campaign size: 7 authored levels.

This is a game about splitting one seed into two becoming difficult because every daughter occupies a landing cell another branch may need later.

Forked Seeds is a spatial scheduling game disguised as simple growth. A click replaces one seed with two daughters in the two directions printed on its cell. The operation always gains one seed, but it also consumes one net empty cell: two vacancies are filled while the parent becomes the new vacancy. Later puzzles are therefore about moving and preserving holes as much as growing seeds.

There are no switches, inventories, counters, hidden mappings, movable avatars, or timing rules. Every consequence comes from the same visible fork-and-vacate operation.

1. Player-facing rules

The board is a connected or partially connected region of pale planting cells surrounded by dark unusable ground.

Each planting cell contains a faint fork marking with exactly two cardinal arrowheads. The marking is an immutable property of the cell:

A seed occupying that cell displays the same two directions strongly.

A daughter born in another cell immediately adopts that destination cell’s fork.

Direction is not inherited from the parent.

Forks may point around a corner, such as north-east, or in opposite directions, such as east-west.

Gold broken rings mark target planting cells. The objective is:

Occupy every gold-ringed cell with a seed at the same time.

A seed’s generation, size, ancestry, and fork direction do not affect whether it satisfies a ring.

Clicking an occupied seed cell attempts to split it:

Read the cell’s two fork directions.

Inspect the two adjacent destination cells.

If both destinations are usable planting cells and both are empty, remove the parent and place one daughter in each destination.

If either destination is unusable or occupied, nothing changes. There is no partial split.

Seeds cannot otherwise move, merge, rotate, or be removed. Consequently, an unfortunate legal split can permanently consume a landing cell needed by another branch.

Only direct cell clicks are used. The entire occupied cell is the hit region; the player never has to click a tiny seed pixel.

2. Logical and rendering geometry
2.1 Native board

The native logical lattice is 10 rows × 10 columns.

Coordinates are written (r,c):

(1,1) is the upper-left logical cell.

Rows increase downward.

Columns increase rightward.

Every level specifies a subset U of usable planting cells. Any logical cell not in U is dark ground.

The 10×10 lattice is not outlined. Adjacent planting cells visually touch, so no internal grid borders are drawn.

2.2 Pixel mapping

The render canvas is exactly 64×64 pixels.

Each logical cell occupies a 6×6 pixel module:

Cell (r,c) has upper-left pixel
x = 2 + 6(c−1),
y = 2 + 6(r−1).

It covers local pixel coordinates u,v ∈ {0,…,5}.

The logical board occupies pixels 2…61 in each axis.

The remaining two-pixel outer margin is dark ground.

A click anywhere in the 6×6 module maps to that logical cell. Clicking a module outside U, or outside the logical board, is a ground click and has no state effect.

2.3 Palette

Only these integer color IDs are needed:

ID	Use
0	White seed arrowheads and completion highlight
1	Pale planting-cell surface
2	Faint empty-cell fork preview
3	Seed drop shadow
4	Dark ground and unused canvas
5	Deep seed outline where needed
8	Invalid-split feedback
10	Empty-cell click feedback
11	Gold target ring
14	Green seed body

No text, numbers, labels, counters, or legends are rendered.

3. Cell and seed rendering
3.1 Planting cells

Every cell in U is filled with color 1. There is no generic border between neighboring usable cells.

An empty planting cell shows its fork in color 2. The marking consists of a central hub, stems, and two arrowheads. The following local masks are used:

Central hub: {2,3} × {2,3}.

North stem: (2,1),(3,1).

East stem: (4,2),(4,3).

South stem: (2,4),(3,4).

West stem: (1,2),(1,3).

Arrowhead masks:

N: tips (2,0),(3,0) and wings (1,1),(4,1).

E: tips (5,2),(5,3) and wings (4,1),(4,4).

S: tips (2,5),(3,5) and wings (1,4),(4,4).

W: tips (0,2),(0,3) and wings (1,1),(1,4).

The hub, applicable stems, and applicable arrowheads are all color 2 while the cell is empty.

3.2 Target rings

A target cell uses a broken gold ring rather than a full square border. This avoids turning adjacent targets into an internal grid.

The ring mask is:

Four corners: (0,0),(5,0),(0,5),(5,5).

Two centered pixels on each edge:

top (2,0),(3,0)

right (5,2),(5,3)

bottom (2,5),(3,5)

left (0,2),(0,3)

These pixels use color 11.

When an occupied seed’s arrowhead overlaps an edge-center ring pixel, the white arrowhead is drawn above that pixel. The four gold corner pixels are always redrawn last, so an occupied target remains visibly ringed.

3.3 Seed bodies and generations

Each seed stores an integer generation g.

Initial seeds have g=0.

Both daughters receive g_parent + 1.

Generation never changes legality or validation.

The green body mask shrinks for the first two generations:

g=0: central 4×4 block, u,v ∈ {1,2,3,4}.

g=1: central 2×2 block plus four two-pixel arms:
{2,3}×{2,3},
{2,3}×{1,4},
{1,4}×{2,3}.

g≥2: central 2×2 block {2,3}×{2,3}.

The relevant directional stems are added to the body in color 14, keeping even a minimum-size seed connected. Arrowhead tips and wings are then drawn in color 0.

Before drawing the body, its mask is shifted one pixel down and right, clipped to the cell, and drawn in color 3. This produces a simple raised appearance.

The rendered size clamps at g=2; deeper generations remain at the minimum readable size.

4. Input contract

The supplied action space is:

UP

DOWN

LEFT

RIGHT

SPACE

click on any canvas cell

Forked Seeds uses only clicks.

UP, DOWN, LEFT, RIGHT, and SPACE are accepted no-ops. They do not move a cursor, alter selection, or mutate puzzle state.

There is no persistent cursor or selected object. Every level starts with:

selected_cell = null
feedback = null

A direct click is interpreted as follows:

Occupied usable cell: attempt its split.

Empty usable cell: no mutation; briefly highlight that cell in color 10.

Dark ground: no mutation.

Target border, arrowhead, seed shadow, or seed body: all count as clicking that logical cell.

5. Formal state machine
5.1 Immutable level definition

Each level contains:

U                 set of usable cells
fork[c]           unordered pair of distinct cardinal directions for every c in U
T                 set of target cells, T ⊆ U
initial_seeds     map from occupied cell to generation

Direction vectors are:

N = (-1, 0)
E = ( 0,+1)
S = (+1, 0)
W = ( 0,-1)

The specification shorthand NE, EW, and so forth names the two directions; it is not rendered as text.

5.2 Runtime state
seeds             map cell -> generation
done              Boolean
feedback          transient visual-feedback descriptor or null
animation         transient render-only descriptor or null
step_count        technical harness count

There may never be more than one seed in a cell.

5.3 Split transition

For a clicked seed at cell p, let:

fork[p] = {d1, d2}
q1 = p + d1
q2 = p + d2

The split is legal exactly when:

q1 ∈ U
q2 ∈ U
q1 ∉ seeds
q2 ∉ seeds

Legality is evaluated against the complete pre-action occupancy state.

For a legal split with parent generation g:

remove p
insert q1 with generation g+1
insert q2 with generation g+1

Removal and both insertions are one atomic transition:

O' = (O \ {p}) ∪ {q1,q2}

No daughter is inserted before the other. No chain reaction occurs.

Every legal split therefore satisfies the population invariant:

|O'| = |O| + 1
5.4 Invalid split

If either destination is blocked:

The parent remains.

No daughter appears, including in a free destination.

Every blocked arrow turns color 8.

An occupied blocking destination receives a red rim.

An unusable destination shows a red patch immediately beyond the corresponding cell edge.

The parent shakes by one pixel toward and away from the blocked direction over three alternating subframes.

If both destinations are blocked, both are indicated.

The state hash before and after an invalid split must be identical.

5.5 Success validation

After every state update, evaluate:

done = all(t in seeds for t in T)

Relevant:

Any seed generation satisfies a target.

A target ring remains active while occupied.

Initially occupied targets count.

Seeds on unringed cells are allowed.

Remaining legal splits are irrelevant once all targets are occupied.

Ignored:

Generation values.

Number of unringed seeds.

Whether seeds are terminal.

Which ancestry produced a target seed.

Forbidden only by the transition rules:

Multiple seeds in one cell.

Seeds outside U.

Partial splits.

The level completes immediately when done becomes true.

6. Animation and timing

There is no gameplay timer, energy value, or action budget. A technical truncation guard of 4096 step() calls may be used by the harness; it is not rendered and is far above any authored solution length.

6.1 Successful split animation

A legal split uses four deterministic render subframes:

The parent kernel compresses toward its center.

The parent disappears; two daughter kernels appear two pixels along their respective directions.

The kernels move to four pixels from the parent center.

The daughters settle at the destination centers and reveal the destination cells’ fork arrows.

The logical state is committed during the initiating step() call. Animation frames merely interpolate between a cached old render and the already committed new state.

Animation subframes:

do not call step();

do not consume the technical action cap;

do not permit or queue additional input.

If the final split completes the level, the normal split animation finishes and all gold rings then pulse 11 → 0 → 14 → 11.

6.2 Other feedback

Empty usable-cell click: three-frame color-10 pulse, no mutation.

Invalid split: six short feedback frames, no mutation.

Ordinary settled states contain all information necessary to reason about the puzzle; animation is never the only evidence for a rule.

7. Authored-level notation

For large levels, usable and target cells are specified by row.

For example:

r2: 3–7

means (2,3),(2,4),(2,5),(2,6),(2,7).

All ranges are inclusive.

Every listed usable cell in Levels 4–7 either:

is clicked in the canonical verified solution, or

is occupied at completion.

There are no decorative pale planting cells.

Level 1 — First Fork

Teaches: one click creates both indicated daughters, while the parent disappears.

Intended depth: 1 successful click.

Player-facing discovery

A single large seed sits at (5,5). Its arrows point east and south, directly toward two empty gold-ringed cells. There are no other usable cells.

The obvious experiment distinguishes the intended rule from several plausible alternatives:

The seed does not move along one chosen arrow.

The player does not choose between the arrows.

The parent does not remain.

Both daughters appear from one click.

The completion pulse immediately provides positive goal evidence.

State-machine instance
Usable cells and forks
(5,5) = SE
(5,6) = NE
(6,5) = SW

Initial seeds
(5,5), generation 0

Targets
(5,6)
(6,5)

Selection/cursor
none

The target cells’ forks point into dark ground, so seeds placed there cannot split successfully.

Canonical solution
1. click (5,5)

Final occupancy:

(5,6)
(6,5)
Attribution and necessity check

The two targets are exactly the two visible destinations of the only seed. No unrelated object changes during the action. The population lower bound is one split, and the replay reaches the objective in one split.

Level 2 — Forks Have Addresses

Teaches: daughters use the fork printed on their destination cell, and daughters can themselves split.

Intended depth: 5 successful clicks.

Player-facing discovery

The initial seed points north and south. The two destination cells visibly preview east-west forks.

After the first split, the two smaller seeds therefore point sideways rather than continuing north-south. The target rings sit beyond those lateral forks, so the player has a clean reason to click the daughters.

The upper-left and upper-right grandchildren then display different corner forks. Clicking them produces daughters in the visibly previewed directions.

There are no occupied blockers, convergences, internal target rings, or irreversible order traps. Every successful split order remains recoverable.

State-machine instance
Usable cells and forks

(5,5) = NS

(4,5) = EW
(6,5) = EW

(4,4) = NW
(4,6) = NE

(6,4) = SW
(6,6) = SE
(3,4) = NW
(4,3) = SW
(3,6) = NE
(4,7) = SE

The last six cells point into dark ground and are unsplittable under the ordinary rule.

Initial seeds
(5,5), generation 0

Targets
(6,4), (6,6)
(3,4), (4,3)
(3,6), (4,7)

Selection/cursor
none
Canonical solution
1. click (5,5)
2. click (4,5)
3. click (6,5)
4. click (4,4)
5. click (4,6)

Clicks 2 and 3 may be exchanged. Clicks 4 and 5 may occur whenever their respective parents exist.

Attribution and necessity check

The first daughter action visibly differs from the parent action: an east-west cell creates lateral daughters. This cannot be explained by a mistaken “all generations retain the root direction” rule.

No target seed is clicked. All five internal transitions are required to produce the six leaf targets, and the five-click population lower bound is achieved.

Level 3 — Room for Both

Teaches: a split is all-or-nothing; both destinations must be usable and empty.

Intended depth: 5 successful clicks, plus an optional diagnostic invalid click.

Player-facing discovery

Two initial seeds touch.

The seed at (5,4) points:

east into the occupied seed at (5,5);

south into an empty gold-ringed cell at (6,4).

A natural first experiment is to click (5,4). It fails visibly:

the east arrow and occupied destination flash red;

the south target remains empty;

the parent remains;

no partial daughter appears.

The adjacent seed at (5,5) is legal. Splitting it immediately vacates the blocking cell. The formerly blocked seed can then split normally.

The remaining branch structure merely reuses the recursive behavior established in Level 2.

State-machine instance
Usable cells and forks

(5,4) = ES
(5,5) = NE

(4,5) = NE
(5,6) = SE
(6,6) = SE

(6,4) = SW
(3,5) = NW
(4,6) = NE
(5,7) = NE
(7,6) = SW
(6,7) = SE

The last six cells point into dark ground.

Initial seeds
(5,4), generation 0
(5,5), generation 0

Targets
(6,4)
(3,5), (4,6)
(5,7)
(7,6), (6,7)

Selection/cursor
none
Discovery action

This click is optional and does not count toward solution depth:

click (5,4)

Expected result: invalid feedback and a byte-for-byte unchanged logical state.

Canonical successful solution
1. click (5,5)
2. click (4,5)
3. click (5,6)
4. click (6,6)
5. click (5,4)

After the first action, click 5 may be performed earlier without harming the other branches.

Final occupancy is:

all six targets
plus the unringed parking seed at (5,5)
Attribution and necessity check

The only new causal observation is the blocked, all-or-nothing split. No target seed is used as transit, and no convergence scheduling is required.

Four successful splits cannot suffice: covering the south target requires splitting (5,4), and that split unavoidably creates an additional unringed daughter at (5,5). The verified minimum is therefore five.

Level 4 — Borrowed Rings

Teaches: an occupied target may be a transit cell that must be vacated and later refilled.

Intended depth: 25 successful clicks.

Player-facing discovery

The initial north-south split opens two fan-shaped growth regions:

an upper fan that is wider and shallower;

a lower fan that is narrower and deeper.

The target silhouettes are deliberately offset between the two regions.

After the first five canonical clicks, a seed occupies the gold target (3,4), while further gold targets remain immediately north and east of it. The seed’s NE fork is the only useful local continuation. Clicking it:

visibly empties a currently satisfied ring;

fills two farther cells;

creates a vacancy at (3,4).

Later, splitting (4,4) refills (3,4).

The lower fan repeats the same rule in a different geometry: target (7,4) is split and later refilled from (6,4).

This is the first level where “never disturb a completed target” is demonstrably the wrong hypothesis.

State-machine instance
Usable cells
r1:     3–6
r2–r4: 3–7
r5:     3
r6–r9: 3–6
r10:    3–5
Fork field
(5,3) = NS
Every usable cell with r < 5 = NE
Every usable cell with r > 5 = SE
Targets
r1:     3–6
r2–r3: 4–7
r4:     7
r6:     6
r7–r9: 4–6
r10:    3–5

There are 39 usable cells and 26 targets.

Initial seeds
(5,3), generation 0

Selection/cursor
none
Canonical verified solution
 1– 5: (5,3), (4,3), (6,3), (3,3), (7,3)
 6–10: (3,4), (7,4), (2,4), (8,4), (2,5)
11–15: (8,5), (3,5), (7,5), (2,6), (9,5)
16–20: (4,4), (9,4), (2,3), (6,4), (4,5)
21–25: (6,5), (3,6), (8,3), (4,6), (9,3)

At completion, the occupied set is exactly the target set.

Necessity and anti-cleanliness check

The two fans are not copies:

the upper fan has four rows and five columns of growth extent;

the lower fan has five rows and four columns;

their missing far corners and final target frontiers differ.

Within each fan, cells have two possible parents, creating genuine convergence dependencies. Filling an eastern landing too early can block the western parent that must pass through it.

The final 26 targets require at least 26−1=25 splits by the population invariant, and the replay achieves that bound. The non-negative firing-count equations have a unique solution: all 25 transition cells in the trace must fire once. Twelve of those cells are final targets, so target-transit behavior is structurally required rather than decorative.

Level 5 — Keep One, Spend One

Combines: convergence order, target preservation, target transit, and repeated use of released parent cells.

Intended depth: 41 successful clicks.

Player-facing discovery

Two initial seeds already occupy gold rings:

(5,5)
(4,6)

Both are legal at reset, and both point toward the shared cell (5,6):

(5,5) has an NE fork and wants (4,5) plus (5,6).

(4,6) has an SE fork and wants (5,6) plus (4,7).

Whichever seed splits first occupies the shared landing and blocks the other.

The surrounding target geometry resolves the ambiguity. In a minimum solution:

(4,6) must split and be refilled much later;

(5,5) must remain occupied throughout.

Thus two visually identical target seeds take opposite strategic roles solely because of their spatial context.

The level also introduces genuine repeated relays. Three unringed cells must each be occupied, split, refilled from another direction, and split again:

(5,8) at clicks 6 and 19
(8,6) at clicks 21 and 29
(7,3) at clicks 11 and 38

A parent vacancy is no longer just a one-time opening. It becomes a reusable junction.

State-machine instance
Usable cells
r1:  3–5
r2:  3–8
r3:  3–8
r4:  2–9
r5:  2–10
r6:  1–9
r7:  1–8
r8:  2–8
r9:  5–8
r10: 6
Clockwise fork field

For every usable cell:

r≤5, c≤5  = NE
r≤5, c≥6  = SE
r≥6, c≥6  = SW
r≥6, c≤5  = NW

The four regions visibly form a clockwise circulation. This is not a hidden regional rule: every individual empty cell displays its actual fork.

Initial seeds
(5,5), generation 0, initially on a target
(4,6), generation 0, initially on a target
Targets
r1:  3–5
r2:  4, 5, 8
r3:  4, 6–8
r4:  2, 5–7, 9
r5:  3–6, 9–10
r6:  1–9
r7:  1–2, 4–5, 7
r8:  2, 5, 7
r9:  5–8
r10: 6

There are 61 usable cells and 43 targets.

Selection/cursor
none
Canonical verified solution
 1–10: (4,6), (4,7), (5,7), (6,7), (7,7),
       (5,8), (7,6), (7,5), (6,8), (7,4)

11–20: (7,3), (6,3), (7,8), (5,3), (4,3),
       (5,9), (4,8), (3,3), (5,8), (6,2)

21–30: (8,6), (5,2), (9,6), (3,4), (8,5),
       (4,4), (8,7), (2,4), (8,6), (2,5)

31–40: (7,2), (2,6), (8,4), (8,8), (2,3),
       (8,3), (3,6), (7,3), (3,5), (3,7)

41:    (2,7)

Important replay events:

click 1:  (4,6) vacates its initial target and occupies shared cell (5,6)
click 17: (4,8) refills reusable hinge (5,8)
click 27: (8,7) refills reusable hinge (8,6)
click 36: (8,3) refills reusable hinge (7,3)
click 37: (3,6) refills the original target (4,6)

At completion, the occupied set is exactly the 43 targets.

Necessity and anti-cleanliness check

This is a dependency network across all four directional regions, not four independent quadrants. The three repeated hinge cells tie distant sections together and must be reopened at different phases.

The target population gives the lower bound:

43 targets − 2 initial seeds = 41 splits

The canonical trace reaches it.

The integer firing-count solution is unique:

(5,5) fires zero times.

(4,6) fires once.

(5,8), (8,6), and (7,3) each fire twice.

Every other fired cell appears once.

Cells absent from the trace have firing count zero.

Thus preserving one initial target, spending the other, and reusing the three hinges are forced in every minimum-length solution.

Level 6 — Sideways Shelf

Combines: target transit with opposite-direction forks that propagate a vacancy laterally.

Intended depth: 46 successful clicks.

Player-facing discovery

The upper and right-hand regions use corner forks, but the lower-left region consists entirely of familiar EW forks. The rule itself is not new—Level 2 already established opposite-direction daughters—but its strategic role changes.

In the lower-left shelf, clicking a seed requires an empty cell on both its left and right. The result fills those two cells and leaves the parent cell empty. A vacancy can therefore be relayed horizontally:

... occupied, empty, occupied ...

becomes a new arrangement one position farther along the row.

After the first fifteen canonical clicks, target seed (6,4) is the local route toward the left-side targets. Its two horizontal destinations (6,3) and (6,5) are open. Clicking it fills both and temporarily empties its own ring. Later:

(6,3) refills (6,4) while extending to (6,2);

(6,2) continues the relay toward (6,1).

The same target mechanic now behaves like a sliding hole rather than a branching fan.

A contrasting target appears immediately at (5,6). It is created by the first split and must be preserved in a minimum solution, while nearby target (4,4) is split and refilled.

State-machine instance
Usable cells
r1:     2–5
r2–r3: 1–5
r4–r5: 1–10
r6–r7: 1–9
r8–r9: 1–6
r10:    6
Fork field
r≤5, c≤5  = NW
r≤5, c≥6  = SE
r≥6, c≥6  = SW
r≥6, c≤5  = EW
Initial seeds
(5,5), generation 0
(4,6), generation 0

Neither initial seed is on a target.

Targets
r1:  2–5
r2:  1–4
r3:  1–4
r4:  1–4, 10
r5:  1, 6, 8–10
r6:  1, 3–8
r7:  1, 3–9
r8:  1, 3–6
r9:  1, 3–6
r10: 6

There are 65 usable cells and 48 targets.

Selection/cursor
none
Canonical verified solution
 1–10: (4,6), (5,5), (4,7), (5,4), (5,7),
       (6,7), (4,4), (4,3), (6,6), (6,5)

11–20: (5,8), (7,6), (3,3), (5,3), (8,6),
       (6,4), (9,6), (7,5), (8,5), (5,9)

21–30: (8,4), (4,8), (3,4), (6,8), (9,5),
       (2,3), (8,3), (8,2), (6,9), (6,3)

31–40: (2,4), (2,2), (6,2), (4,9), (7,4),
       (4,5), (3,2), (9,4), (4,2), (7,3)

41–46: (5,2), (7,2), (3,5), (9,3), (2,5), (9,2)

Important shelf relay:

click 10: (6,5) creates (6,4) and (6,6)
click 16: (6,4) creates (6,3) and (6,5)
click 30: (6,3) creates (6,2) and refills (6,4)
click 33: (6,2) creates (6,1) and refills (6,3)

Important target contrast:

click 1 creates target seed (5,6); it is never split
click 7 splits target (4,4)
click 36 refills target (4,4) from (4,5)

At completion, occupancy is exactly the 48-target set.

Necessity and anti-cleanliness check

The shelf is not a separate device. It is a region of ordinary cells whose visible forks happen to point east-west. Its behavior follows the global rule exactly.

The upper-left crown, right-hand turns, southeast basin, and horizontal shelf cannot be completed as independent errands. Shelf seeds consume landing cells that the turning regions later use to refill targets.

The target population proves a minimum of:

48 − 2 = 46 splits

The firing-count vector is unique:

all 46 cells in the trace fire exactly once;

usable transition cell (5,6) has count zero and must remain occupied in a minimum solution;

29 fired cells are also final targets, so temporary target loss is pervasive rather than optional.

Level 7 — Canopy Lock

Combines: all previous consequences in one dense but readable branching canopy.

Intended depth: 52 successful clicks.

Player-facing discovery

The only initial seed already occupies target (5,5). Almost every remaining ring lies beyond it, so the first action must deliberately break the apparent progress.

The opening relay is compact and legible:

(5,5) splits
(5,6) carries the branch east
(5,7) carries it farther
(6,6) turns south-west
(6,5) refills the original target (5,5)

The root ring is therefore lost on click 1 and restored on click 5.

The rest of the level develops three interacting fronts around the clockwise field. Interior targets repeatedly serve as transit cells before being refilled from another direction:

(5,6) is clicked at step 2 and refilled at step 15
(6,6) is clicked at step 4 and refilled at step 8
(6,5) is clicked at step 5 and refilled at step 18
(4,6) is clicked at step 15 and refilled at step 25

The final target silhouette is irregular: a narrow northern tip, a dense central sweep, a leftward lower canopy, and offset lower-right leaves. It is not a rectangle to be filled by indiscriminate expansion.

State-machine instance
Usable cells
r1:     5
r2–r4: 5–10
r5:     2–10
r6–r9: 1–9
r10:    6–9
Clockwise fork field
r≤5, c≤5  = NE
r≤5, c≥6  = SE
r≥6, c≥6  = SW
r≥6, c≤5  = NW
Initial seed
(5,5), generation 0, initially on a target
Targets
r1:     5
r2:     10
r3–r4: 6–10
r5:     2–10
r6–r8: 1–8
r9:     1, 6–8
r10:    6–9

There are 68 usable cells and 53 targets.

Selection/cursor
none
Canonical verified solution
 1–10: (5,5), (5,6), (5,7), (6,6), (6,5),
       (6,4), (5,8), (6,7), (6,8), (6,3)

11–20: (6,2), (4,5), (7,6), (5,9), (4,6),
       (7,7), (3,5), (7,5), (7,4), (6,9)

21–30: (7,8), (7,3), (2,5), (7,2), (3,6),
       (8,6), (7,9), (8,5), (8,7), (2,6)

31–40: (8,4), (8,8), (8,9), (8,3), (8,2),
       (9,6), (4,7), (4,8), (9,7), (4,9)

41–50: (9,5), (3,7), (9,4), (3,8), (3,9),
       (9,8), (2,7), (9,9), (9,3), (2,8)

51–52: (2,9), (9,2)

At completion, the occupied set is exactly the 53 targets.

Necessity and anti-cleanliness check

The minimum follows immediately from population:

53 targets − 1 initial seed = 52 splits

The canonical replay achieves this bound.

The firing-count equations have a unique non-negative solution: each of the 52 transition cells in the canonical trace fires exactly once. Thirty-seven of those transition cells are also final targets, so a strategy that treats gold rings as permanent endpoints cannot solve the level.

The level typically presents several simultaneously legal seeds, but their orders are not interchangeable. Advancing one frontier closes landing cells on the next frontier; a later branch can pass only if the relevant target or junction was vacated first. Difficulty therefore comes from scheduling the canopy, not from extra devices, a hidden counter, or a long corridor.

8. Campaign progression
Level	Successful clicks	New evidence or strategic interpretation
1	1	One seed becomes two; parent disappears; both targets matter simultaneously
2	5	Fork behavior belongs to the destination cell; daughters recursively split
3	5	Both landing cells must be open; blocked splits are completely atomic
4	25	Target seeds can be transit cells and later be refilled
5	41	Identical target seeds may need opposite treatment; parent cells can be reused repeatedly
6	46	Opposite-direction forks relay vacancies horizontally and require staging
7	52	Dense composition of convergence, preservation, transit, and branch scheduling

The role of a target seed changes repeatedly without changing its operational rule:

Levels 1–3: endpoint to preserve.

Level 4: temporary transit seed.

Level 5: one initial target must be preserved while another must be spent.

Level 6: one target anchors the layout while others relay a moving vacancy.

Level 7: the root and much of the final canopy are temporarily vacated and repaired.

Likewise, the parent vacancy evolves from immediate visual evidence into the game’s main spatial resource.

9. Implementation validation checklist

An implementation should pass the following checks.

Universal transition tests

A legal split increases population by exactly one.

An invalid split leaves occupancy and generations unchanged.

A free destination receives no daughter when the twin destination is blocked.

Daughters use fork[destination], not fork[parent].

Both daughters receive parent_generation + 1.

Target rings remain visible around occupied seeds.

Clicking any pixel in an occupied 6×6 cell attempts that seed’s split.

Empty-cell clicks do not create persistent selection.

Keyboard actions are state-preserving no-ops.

Animation playback never mutates logical state or consumes actions.

Authored replay tests

For every level:

Reset to the listed initial state.

Click the center pixel of each canonical coordinate in order.

Assert that every listed click is legal.

Assert that done is false before the last successful click.

Assert that done becomes true on the final listed click.

Assert the final target coverage exactly as documented.

Additional Level 3 test:

reset
click (5,4)
assert state unchanged
assert only the east destination is reported as blocked
assert no seed appears at the free south target
Structural audits

The authored minimum depths are:

L1 = 1
L2 = 5
L3 = 5
L4 = 25
L5 = 41
L6 = 46
L7 = 52

For Levels 4–7, the canonical final occupancy equals the target set, and the population lower bound proves optimality.

For Levels 4–7, solving the integer transition-incidence equations against the documented initial and final occupancy produces the firing counts described in each level. This catches accidental fork changes, omitted cells, wrong target coordinates, or a daughter placed in the wrong direction even when a superficially similar replay still appears possible.
