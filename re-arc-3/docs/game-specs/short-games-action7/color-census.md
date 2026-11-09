Color Census

“This is a game about recoloring an entire color population becoming difficult because every arrival irreversibly fuses the groups that future clicks can address.”

Authored level count: exactly 7.

Color Census is a click-only population-routing game. Every piece has a current color and matching emblem. Clicking one exposed piece advances every piece of that color along a visible transition graph. The pieces never move.

The operation is simple; its consequences are not. When a population enters an occupied color, the two populations become indistinguishable and can never again be controlled separately. Later levels add fixed census covers that hide particular pieces from direct clicking without shielding them from global recoloring. A covered population must acquire an exposed representative through a carefully planned merge.

1. Player-facing rules
The central operation

Clicking an exposed piece:

Reads that piece’s current color.

Finds every piece on the board currently showing that color and emblem.

Advances all of them by one arrow in the transition legend.

Leaves every piece in its original position.

Updates the current census.

Only one transition edge is traversed per click. A population that arrives in another occupied color merges there but does not continue moving until another click.

Irreversible consolidation

Suppose three blue pieces already exist and two red pieces transform into blue. There are now five blue pieces. The game retains no origin marker. Clicking any exposed blue piece later transforms all five.

This is not merely a visual grouping convention. There is no hidden state with “old blue” and “former red” subgroups.

Census covers

Introduced in level 4:

An ordinary unframed piece is exposed and clickable.

A piece surrounded by a thick light-gray frame is covered and cannot itself be clicked.

A covered piece still recolors whenever an exposed piece of the same current color is clicked.

Covers are fixed to board positions and never open, move, or change.

If a completely covered population merges with a population containing an exposed position, the combined color now has an exposed representative.

Covers do not create another command or device errand. They only change which global populations can currently be selected.

Objective

The side census board shows, for each revealed color:

the color and emblem;

its one-step destination;

current population pips;

fixed target pips;

a green seal when current count equals target count.

The level is won automatically when every color count matches its target simultaneously.

No submit action is required.

2. Transition system

Specification shorthand such as R, B, or G is never rendered as text. It is used only in this document and in solution traces.

The persistent transition graph is:

R ─┐
   ├→ B → G → O → P → M → C ─┐
Y ─┘          ↑               │
              └───────────────┘

Thus G → O → P → M → C → G is a five-color cycle, while red and yellow independently feed blue, and blue feeds the cycle.

Exact color and emblem table

In each 5×5 mask below:

. is the current gameplay color.

# is color 5, black.

The mask changes whenever the piece changes color.

Shorthand	Color ID	Emblem mask	Destination
R	8	...../..#../..#../..#../.....	B, ID 9
B	9	...../.##../.##../...../.....	G, ID 14
G	14	...../.#.../..#../...#./.....	O, ID 12
O	12	...../...#./..#../.#.../.....	P, ID 15
P	15	...../..#../.###./..#../.....	M, ID 6
M	6	...../.#.#./...../.#.#./.....	C, ID 10
C	10	...../..#../.#.#./..#../.....	G, ID 14
Y	11	...../...../.###./...../.....	B, ID 9

The different emblems make distant members of a population identifiable without relying exclusively on hue.

3. Native geometry and rendering
Canvas and logical board

Render canvas: exactly 64×64 integer-color cells.

Logical piece lattice: 5 columns × 8 rows.

Logical coordinates: (c,r), where:

c ∈ {0,1,2,3,4}, left to right;

r ∈ {0,1,2,3,4,5,6,7}, top to bottom.

No lattice lines or internal board border are rendered.

The slot for logical coordinate (c,r) has pixel origin:

slot_x = 2 + 7c
slot_y = 4 + 7r

The playfield therefore occupies pixels x=2…36, y=4…59.

Exposed piece

An exposed piece occupies:

x = slot_x + 1 … slot_x + 5
y = slot_y + 1 … slot_y + 5

Its exact clickable hit region is those 25 body pixels. The one-pixel slot margin remains background.

The center pixel used by canonical replays is:

click_x = 5 + 7c
click_y = 7 + 7r
Covered piece

A covered piece uses the same inner 5×5 colored body, surrounded by a 7×7 perimeter in color 1, light gray.

Conceptually:

1111111
1ppppp1
1ppppp1
1ppppp1
1ppppp1
1ppppp1
1111111

Here p is the normal color-and-emblem body. The rendered game uses integer colors, not these characters.

The entire 7×7 covered slot is blocked from clicking. Clicking either the frame or the visible inner body produces an invalid-action ripple.

Palette use
Role	Color ID
Background	4
Black emblems	5
Cover frames and target sockets	1
HUD separators	2
Inactive seal	3
Valid-action pulse	0
Completed census seal	14
Gameplay colors	6, 8, 9, 10, 11, 12, 14, 15

Color 13 is unused.

4. Census and transition HUD

The HUD occupies x=39…62, y=4…59.

There are eight fixed seven-pixel-high row positions in this display order:

R, B, G, O, P, M, C, Y

For row index i, its top pixel is:

row_y = 4 + 7i
Row layout

Within a revealed row:

x=39…41, y=row_y+2…row_y+4: 3×3 source color/emblem chip.

x=42…43: light-gray right arrow.

x=44…46: 3×3 destination color/emblem chip.

x=48…52, y=row_y+1…row_y+4: current-count pip field.

x=53: gray vertical separator.

x=55…59, y=row_y+1…row_y+4: target-count socket field.

x=61…62, y=row_y+1…row_y+5: match seal.

The current and target fields are both 5×4, with capacity for 20 pieces. Pips are filled left-to-right, then top-to-bottom.

Current pips use the row’s gameplay color.

Target sockets use color 1.

The seal is color 14 when current_count == target_count, otherwise color 3.

The largest authored level contains 17 pieces, so the pip capacity is sufficient even after a complete accidental consolidation.

Progressive but persistent legend revelation

At reset, a row is revealed when either:

that color currently has at least one piece, or

that color has a nonzero target count.

A revealed source row always shows its immediate destination chip, even when the destination’s own row is not yet revealed.

When a valid click creates a previously unseen destination color, that destination row becomes visible in its fixed row position. Once visible during an attempt, a row never disappears.

Thus the player is never asked to predict an unshown immediate transition. Longer paths are discovered one causal edge at a time.

5. Exact state-machine contract
Persistent state
pieces:
    fixed list of:
        logical coordinate (c,r)
        current_color_id
        covered: bool

target_count[color_id]:
    integer for each of the eight gameplay colors

revealed[color_id]:
    bool

step_count:
    integer

done:
    bool

failed:
    bool

selected:
    always None

cursor:
    always None

Current counts and row seals are derived from pieces.

There is no stored cohort identity, source history, merge ancestry, selected piece, or movable cursor.

Action space

The environment accepts exactly:

UP
DOWN
LEFT
RIGHT
SPACE
CLICK(x,y)

Only CLICK is used for gameplay.

UP, DOWN, LEFT, RIGHT, and SPACE are inert no-ops.

Every action, including a no-op or invalid click, increments step_count.

There is no keyboard cursor.

SPACE is not a submit action.

Valid-click resolution

For a click on an exposed piece body:

Increment step_count.

Read the clicked piece’s pre-action color source.

Read destination = next[source].

Snapshot:

affected = every piece whose pre-action color == source
preexisting_destination = every piece whose pre-action color == destination

Atomically set the current color of every piece in affected to destination.

Do not apply another transition during this step.

Set revealed[source] = true and revealed[destination] = true.

Recompute current counts and seals.

If every count equals its target, set done = true.

If not done and step_count >= 512, set failed = true.

A win on step 512 takes precedence over the safety cap.

Invalid clicks

A click is invalid when it lands on:

an empty/background cell;

any part of a covered 7×7 slot;

the HUD;

an exposed slot’s one-pixel margin;

an out-of-board decorative margin.

It changes no gameplay state other than step_count.

Win validation

For the eight gameplay colors:

done ⇔ ∀ color:
    number_of_pieces_currently_showing(color)
    == target_count[color]

The validator explicitly ignores:

which positions carry each color;

which initial populations contributed to a merged group;

whether winning pieces are exposed or covered;

transition-row revelation;

the route or click order used;

the number of clicks, provided the safety cap has not already terminated the attempt.

Any state with the target count vector wins, including an unanticipated origin assignment.

Exploration allowance

Every level has the same max_steps = 512.

There is no energy meter, shrinking budget, timer, or early automatic failure when a bad merge makes the target unreachable. An experiment can continue until reset or the generous safety cap.

6. Animation and feedback

Animations are deterministic render events, not additional state-machine steps.

The environment’s settled observation is always the final post-action frame.

Valid recoloring

A valid click produces three brief UI frames:

Every affected source piece receives a color-0 outer pulse, including affected covered pieces.

Their bodies and emblems switch to the destination appearance.

Census pips update and matching seals change.

When a merge occurs, pieces that already had the destination color receive a one-frame color-1 outer pulse in frame 2. This makes the collision visible without implying a second transition.

The logical recoloring is one atomic update. Animation frames consume no steps and cannot accept input.

Invalid click

A two-frame color-1 crossed ripple appears around the clicked cell. The settled frame is unchanged.

Completion

Once the target vector is reached:

all visible row seals are green;

every piece receives one four-frame white-to-green outer halo;

final colors and emblems remain visible underneath;

done is already true before the completion animation begins.

7. Replay notation

In the level solutions below, a token such as G means:

Click the center of the row-major first exposed piece currently showing the green diagonal emblem.

Row-major means increasing r, then increasing c.

Because positions and covers are fixed, this convention turns every color-token sequence into a fully determined pixel-coordinate replay. The implementer does not need to choose arbitrary representatives.

A multiplier such as:

(P O G C M) × 4

means repeat that exact five-click block four times.

8. Progression summary
Level	Main discovery	Verified minimum valid clicks
1. One Click, Two Sites	A click affects every distant member of the source color	1
2. Origins Disappear	Populations that meet become one future click target	2
3. Clear the Destination	A destination must sometimes be vacated before an arrival	4
4. Borrowed Handle	Covered populations can be moved through an exposed merged member	15
5. The Moving Vacancy	Full-cycle consolidation creates a vacancy that must be rotated	20
6. Four Required Sums	Covered addends, delayed feeders, and four final cohorts interact	40
7. Collection Order	Every irreversible merge must contribute to the final partition	51

The minimums were checked against the count-only win validator, not merely against the narrated origin assignment.

9. Authored levels
Level 1 — One Click, Two Sites
Player-facing discovery

Two red pieces are placed far apart, with one blue piece between them spatially but not adjacent to either. The census asks for zero red and three blue.

Clicking either red piece causes both distant red pieces to pulse and become blue. Both census rows seal, and the completion feedback appears immediately.

This one-action setup supplies positive evidence for both:

the global same-color relation;

the exact-count objective.

There are no unrelated pieces, covers, decoys, or alternative goals.

Initial state
Current color	Coordinates	Access
R, count 2	(0,1), (4,6)	Exposed
B, count 1	(2,3)	Exposed

Display-order census vector [R,B,G,O,P,M,C,Y]:

Initial: [2,1,0,0,0,0,0,0]
Target:  [0,3,0,0,0,0,0,0]

Initially revealed rows:

R, B

Selection/cursor:

None / None
Intended replay
R

Canonical click: logical (0,1), pixel center (5,14).

Post-click census:

B = 3
all other colors = 0
Attribution and necessity check

The two red pieces are distant, making the global effect impossible to mistake for adjacency.

The click changes no other rule or object.

Clicking blue first sends the singleton to green and makes the target unreachable.

Exhaustive state search gives minimum depth 1, with red as the only successful first click.

Level 2 — Origins Disappear
Player-facing discovery

The physical layout is deliberately unchanged from level 1. Only the target changes.

The player already knows that both red pieces move together. Now the target asks for all three pieces to become green:

Red enters the occupied blue population.

Clicking any resulting blue piece advances all three to green.

The mandatory second click proves that the original blue and former red pieces no longer retain separate operational identities.

Initial state
Current color	Coordinates	Access
R, count 2	(0,1), (4,6)	Exposed
B, count 1	(2,3)	Exposed

Census:

Initial: [2,1,0,0,0,0,0,0]
Target:  [0,0,3,0,0,0,0,0]

Initially revealed rows:

R, B, G

Selection/cursor:

None / None
Intended shortest replay
R B

Intermediate states:

Step	Census
0	R2, B1
1	B3
2	G3

The canonical replay clicks (0,1) twice: after step 1 that exposed position is blue.

Contrasting safe experiment

Clicking blue first is not immediately fatal:

B R B

also solves the level in three valid clicks. This allows exploratory play without obscuring the shortest causal demonstration.

Attribution and necessity check

A mistaken rule in which identical colors preserve hidden origins predicts that clicking the original blue might move only one piece.

The required second click visibly moves all three and directly falsifies that hypothesis.

No one-click state has the target count.

Verified minimum depth: 2.

Level 3 — Clear the Destination
Player-facing discovery

The same three-piece layout is reused once more. The target now requests:

B2 and P1

Sending red into blue immediately would create an inseparable blue population of size 3. Since the target requires blue 2 and a separate singleton, that plan cannot work.

The blue singleton must leave first. Its journey reveals the cycle one edge at a time:

B → G → O → P

Once blue is vacant, red can enter it without merging with the singleton.

Initial state
Current color	Coordinates	Access
R, count 2	(0,1), (4,6)	Exposed
B, count 1	(2,3)	Exposed

Census:

Initial: [2,1,0,0,0,0,0,0]
Target:  [0,2,0,0,1,0,0,0]

Initially revealed rows:

R, B, P

The green row appears after step 1 and the orange row after step 3.

Selection/cursor:

None / None
Intended replay
B R G O
Step	Action	Census	Meaning
0	—	R2, B1	Blue destination occupied
1	B	R2, G1	Blue is vacated
2	R	B2, G1	Red enters empty blue
3	G	B2, O1	Singleton continues
4	O	B2, P1	Target reached
Attribution and necessity check

The first state-changing click must be blue.

Clicking red first creates B3; no later operation can split that group into sizes 2 and 1.

The new strategic property cannot be explained by the older “merge everything toward the target” heuristic.

Verified minimum depth: 4.

Level 4 — Borrowed Handle
New visual rule: census covers

Three populations are completely covered:

the two light-blue pieces;

the green singleton;

the purple singleton.

The two covered light-blue pieces flank a central exposed magenta piece:

covered C — exposed M — covered C

Clicking either covered light-blue piece produces the crossed ripple and no census change. The nearby magenta row visibly points to light blue.

Clicking magenta changes its exposed position into light blue. There are now three light-blue pieces, and the newly recolored central position is exposed. Clicking that exposed light-blue position moves all three—including both covered pieces.

This is the local distinguishing experiment:

covers block direct selection;

covers do not block global recoloring;

exposure is inherited by the combined population through its exposed member.

Initial state
Color	Coordinates	Access
B, count 1	(0,6)	Exposed
G, count 1	(4,1)	Covered
O, count 2	(1,1), (3,6)	Exposed
P, count 1	(4,6)	Covered
M, count 1	(2,3)	Exposed
C, count 2	(1,3), (3,3)	Covered

Census:

Initial: [0,1,1,2,1,1,2,0]
Target:  [0,0,0,0,0,7,1,0]

That is:

Target M7, C1

Initially revealed rows:

B, G, O, P, M, C

Selection/cursor:

None / None
Optional diagnostic action

Click covered C at logical (1,3), pixel center (12,28):

no state change
crossed ripple
step_count += 1

Then click exposed M at (2,3), pixel center (19,28). The central position becomes an exposed C.

Intended valid-click replay
M C G B O G P O M P C G O M P

Verified valid-click depth: 15.

Census milestones
Step	Census
0	B1, G1, O2, P1, M1, C2
1	B1, G1, O2, P1, C3
2	B1, G4, O2, P1
3	B1, O6, P1
4	G1, O6, P1
5	G1, P7
8	P1, M7
10	M1, C7
15	M7, C1
Structural interpretation

The final size-7 population is:

initial G1 + O2 + P1 + M1 + C2 = 7

The initial blue singleton is deliberately kept separate and eventually becomes the final light-blue singleton.

The same cover has contrasting consequences:

Covered C is initially an obstacle.

Covered G becomes a useful addend once the exposed light-blue handle enters it.

Covered P becomes the final addend that completes the size-7 cohort.

Anti-degenerate check

The shortest solution with covers is 15 valid clicks.

With all covers removed, the same count target has a six-click solution.

Thus covers are causally necessary rather than decorative.

The shortest covered solution begins with M; beginning with O remains recoverable but requires 16 clicks.

The singleton cannot be merged into the size-7 group because no splitting operation exists.

Level 5 — The Moving Vacancy
Player-facing discovery

All five cycle colors are occupied:

G → O → P → M → C → G

Therefore every possible first click collides with the next population. A first consolidation is unavoidable.

The target asks for purple 10. The initial purple and magenta populations have sizes 4 and 6, and 4 + 6 = 10. No other adjacent pair produces 10 while leaving the required singleton populations intact.

Therefore purple must be clicked first:

P4 → M6

This creates a merged magenta population of 10 and leaves purple empty.

The empty color is now a movable vacancy:

With purple empty, clicking orange moves orange into purple without merging.

Orange becomes empty, so green can move into orange.

Green becomes empty, so light blue can move into green.

Light blue becomes empty, so magenta can move into light blue.

A player who instead clicks a color whose destination is occupied causes an additional irreversible collapse.

Initial state

All pieces are exposed.

Color	Coordinates
G, count 1	(2,0)
O, count 2	(0,2), (4,7)
P, count 4	(1,6), (3,1), (4,4), (0,5)
M, count 6	(0,0), (4,1), (2,2), (1,4), (3,6), (2,7)
C, count 3	(4,0), (0,7), (3,3)

Census:

Initial: [0,0,1,2,4,6,3,0]
Target:  [0,0,1,2,10,0,3,0]

Initially revealed rows:

G, O, P, M, C

Selection/cursor:

None / None
Intended replay
(P O G C M) × 4

Expanded:

P O G C M
P O G C M
P O G C M
P O G C M

Verified minimum depth: 20.

Census milestones
Step	Census
0	G1, O2, P4, M6, C3
1	G1, O2, M10, C3
5	G3, O1, P2, C10
10	G10, O3, P1, C2
15	G2, O10, P3, C1
20	G1, O2, P10, C3

After the initial merge, each five-click block advances the size-10 population by one position around the cycle while carrying the vacancy backward. It must travel:

M → C → G → O → P

The other three populations are necessarily displaced and restored along the way.

Anti-cleanliness and necessity check

The target size 10 uniquely identifies the intended adjacent merge P4 + M6.

Any different first click destroys the required partition; exhaustive search finds no recovery.

The player cannot leave the other target-matching populations untouched: the vacancy rotation necessarily routes all four surviving cohorts.

The 20 clicks are not corridor padding. Each one moves a cohort into the only vacant color and prevents an unwanted second merger.

Verified minimum depth: 20, with P as the only successful first action.

Level 6 — Four Required Sums
Player-facing discovery

Two covered populations act as inaccessible addends:

blue has three covered pieces;

purple has one covered piece.

Nearby exposed feeders visibly suggest the two access-producing mergers:

R2 → covered B3  gives 5
O4 → covered P1  gives 5

The target also requires populations of sizes 4 and 3:

M1 + C3 = 4
Y1 + G2 = 3

The challenge is not merely to form these sums. All four resulting cohorts must remain separate and end at four different cycle colors.

Yellow and red both feed blue. Yellow must be delayed until the size-5 blue population has left; otherwise it joins that population and creates size 6.

Initial state
Color	Coordinates	Access
R, count 2	(0,1), (4,6)	Exposed
Y, count 1	(2,7)	Exposed
B, count 3	(1,1), (3,4), (0,7)	Covered
G, count 2	(4,1), (2,5)	Exposed
O, count 4	(0,0), (2,2), (1,5), (4,7)	Exposed
P, count 1	(2,3)	Covered
M, count 1	(4,4)	Exposed
C, count 3	(1,0), (0,4), (3,7)	Exposed

The exposed red at (0,1) is adjacent to a covered blue at (1,1). The exposed orange at (2,2) is directly above the covered purple at (2,3). These local arrangements suggest the relevant handle-producing tests without making adjacency part of the transition rule.

Census:

Initial: [2,3,2,4,1,1,3,1]
Target:  [0,0,0,5,4,5,3,0]

That is:

Target O5, P4, M5, C3

Initially revealed rows:

all eight rows

Selection/cursor:

None / None
Intended origin partition

Origins are shown here only to explain the intended reasoning; they are not validator state.

R2 + B3 = 5
O4 + P1 = 5
M1 + C3 = 4
Y1 + G2 = 3

In the supplied replay, these end as:

O5 = O4 + P1
P4 = M1 + C3
M5 = R2 + B3
C3 = Y1 + G2

The two size-5 groups may exchange roles in another valid solution because the validator sees only counts.

Intended replay
R O G B Y
(M P O G C) × 2
M P O G B C
(M P O G C) × 3
M P O G

Expanded length:

5 + 10 + 6 + 15 + 4 = 40 clicks
Early causal phase
Step	Action	Consequence
1	R	Exposed red joins covered blue: B5 with an exposed handle
2	O	Exposed orange joins covered purple: P5 with an exposed handle
3	G	Green vacates the cycle entry destination
4	B	The first size-5 group enters green
5	Y	Yellow becomes a separate blue singleton
6	M	Magenta joins light blue, producing C4

Yellow then waits in blue while the other groups are routed. At step 20, the yellow singleton leaves blue. At step 21, the original green pair enters its color and merges with it to produce the required size 3.

Census milestones
Step	Census
0	R2, Y1, B3, G2, O4, P1, M1, C3
1	Y1, B5, G2, O4, P1, M1, C3
2	Y1, B5, G2, P5, M1, C3
4	Y1, G5, O2, P5, M1, C3
6	B1, G5, O2, P5, C4
10	B1, G4, O5, P2, M5
20	G1, O5, P4, M5, C2
21	G3, O5, P4, M5
30	O5, P3, M5, C4
40	O5, P4, M5, C3
Dependency-network check

This is not a fixed linear station sequence:

The red/blue, orange/purple, and magenta/light-blue consolidations can be staged in more than one order.

Exhaustive search finds optimal solutions beginning with R, O, or M.

Yellow must remain outside blue until the size-5 blue group has departed.

Green must be vacated before accepting that size-5 group.

The future size-3 group is assembled only after the other populations have rotated into safe positions.

The two size-5 groups are interchangeable under the count-only validator, creating alternative valid schedules rather than a single recipe.

Anti-degenerate check

Verified minimum: 40 valid clicks.

Making every piece exposed reduces the minimum to 26, so the two covered populations materially alter the puzzle.

Every initial population contributes to exactly one required final size.

Joining yellow to blue too early creates size 6, which cannot appear in any winning count vector.

Joining the magenta/light-blue group to another cohort destroys the only available size 4.

Level 7 — Collection Order
Player-facing discovery

Three separate covered collectors are visible:

Covered blue singleton beside exposed yellow pair.

Covered purple triplet beside exposed orange singleton.

Covered green singleton surrounded by an exposed light-blue triplet.

Their arrows and target counts suggest three immediate candidate additions:

Y2 + B1 = 3
O1 + P3 = 4
C3 + G1 = 4

These first three consolidations can occur in any order. This makes the opening an actual dependency set rather than a prescribed chain.

The remaining exposed populations are:

R1
M2

The magenta pair must remain unmerged because it is the only available size-2 population. The red singleton must eventually join one of the size-4 populations to form the required size 5—but only after red can enter a now-vacant blue without being absorbed into the size-3 group.

The required final populations are:

2, 4, 3, 5

Every irreversible merge must contribute to one of those four numbers.

Initial state
Color	Coordinates	Access
R, count 1	(4,0)	Exposed
Y, count 2	(0,1), (4,7)	Exposed
B, count 1	(1,1)	Covered
G, count 1	(1,5)	Covered
O, count 1	(2,3)	Exposed
P, count 3	(2,2), (3,2), (3,3)	Covered
M, count 2	(0,7), (4,4)	Exposed
C, count 3	(0,5), (2,5), (1,6)	Exposed

The three covered collectors are spatially paired with plausible exposed sources:

yellow at (0,1) beside blue at (1,1);

orange at (2,3) beside the purple cluster;

light blue around green at (1,5).

The physical layout suggests experiments but does not add adjacency-dependent exceptions. The legend remains the sole transformation rule.

Census:

Initial: [1,1,1,1,3,2,3,2]
Target:  [0,0,0,2,4,3,5,0]

That is:

Target O2, P4, M3, C5

Initially revealed rows:

all eight rows

Selection/cursor:

None / None
Required origin partition

Again, these origin names are explanatory only.

Final O2 = initial M2

Final P4 = initial G1 + C3

Final M3 = initial Y2 + B1

Final C5 = initial R1 + O1 + P3

The four irreversible merges are therefore:

Y2 joins B1
O1 joins P3
C3 joins G1
R1 later joins the O1+P3 group

Every merge has a visible numerical purpose.

Intended replay
Y O C
(G M P O C) × 2
G B R
(M P O G C) × 2
M P O G B C
(M P O G C) × 3
M P O G

Expanded length:

3 + 10 + 3 + 10 + 6 + 15 + 4 = 51 clicks
Opening phase

The supplied replay chooses:

Step	Action	Consequence
1	Y	Y2 + covered B1 → exposed B3
2	O	O1 + covered P3 → exposed P4
3	C	C3 + covered G1 → exposed G4

The order of these three actions may be permuted without changing the three resulting group sizes.

Delayed red injection

By step 15, the size-3 group has left blue. Only then is red advanced:

step 15: B3 leaves its blue position
step 16: R1 enters now-empty blue

The red singleton travels independently until step 31:

step 31: B1 → G1
step 32: a C4 population enters G and merges with R1

That produces the required size-5 group.

If red enters blue before the size-3 group leaves, the result is size 4. That collapse destroys both the required size 3 and the intended size 5.

Census milestones
Step	Census
0	R1, Y2, B1, G1, O1, P3, M2, C3
1	R1, B3, G1, O1, P3, M2, C3
2	R1, B3, G1, P4, M2, C3
3	R1, B3, G4, P4, M2
10	R1, B3, O2, P4, C4
15	R1, G3, O4, P2, M4
16	B1, G3, O4, P2, M4
30	B1, O2, P4, M3, C4
31	G1, O2, P4, M3, C4
32	G5, O2, P4, M3
40	G3, P5, M2, C4
45	G4, P3, M5, C2
47	G2, O4, P3, M5
48	G2, O4, P3, C5
49	G2, O4, M3, C5
50	G2, P4, M3, C5
51	O2, P4, M3, C5
Why the long routing is meaningful

After the four required consolidations, four surviving cohorts of sizes 2, 3, 4, and 5 must be interleaved around the five-color cycle.

Only colors containing exposed positions are selectable. Vacating one destination changes which cohort may advance without another consolidation. Several apparent shortcuts instead merge two target-sized groups and permanently reduce the number of nonzero populations below four.

The repeated five-color fragments are therefore not time padding. They are vacancy rotations that preserve four independently needed groups while changing their phase around the cycle.

Anti-degenerate and necessity check

Verified minimum: 51 valid clicks.

The target has one winning abstract state under the exact covered-state model.

Valid optimal openings may begin with Y, O, or C, reflecting the independent initial collector merges.

Making every piece exposed reduces the same count target to a nine-click solution. The covers create the mature puzzle rather than merely decorating it.

Merging the magenta pair loses the only attainable size 2.

Moving red into occupied blue too early destroys the required size 3.

Failing to combine red with the correct size-4 cohort leaves no attainable size 5.

Every visible population and every cover contributes to the final dependency network.

10. Implementation verification summary

The authored initial states were exhaustively searched at the population-state level, with:

fixed covered/exposed positions;

exact directed transitions;

atomic one-edge clicks;

irreversible same-color consolidation;

count-only goal aggregation across all possible origin assignments.

The resulting minimum state-changing click depths are:

Level 1:  1
Level 2:  2
Level 3:  4
Level 4: 15
Level 5: 20
Level 6: 40
Level 7: 51

Cover-removal comparisons:

Level 4: covered 15, all exposed 6
Level 6: covered 40, all exposed 26
Level 7: covered 51, all exposed 9

This confirms that the late layouts obtain their depth from global population control, irreversible addition, destination vacancy, and exposed-handle routing—not from hidden validation conditions, movement padding, arbitrary devices, or decorative blockers.
