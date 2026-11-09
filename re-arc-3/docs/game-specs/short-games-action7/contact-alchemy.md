Contact Alchemy

This is a game about flipping one tile becoming difficult because the click also stamps the tile’s old face across its entire contact boundary.

The shipped game contains exactly 7 authored levels. Set num_levels = 7.

Only direct tile clicks are meaningful. There is no movement, cursor navigation, move limit, timer, score, inventory, randomness, or hidden history.

1. Player-facing game

Every reagent tile visibly contains two layers:

Its center shows its current face.

Its broad outer frame shows the face that center must have at completion.

There are only two faces:

Amber, shown with a yellow/orange family and a large solid cross.

Violet, shown with a purple/blue family and a large hollow diamond.

Clicking any reagent tile immediately flips its center. Simultaneously, all edge-adjacent reagent tiles are rewritten to the clicked tile’s old face. A neighbor already showing that old face appears unchanged; an opposite neighbor visibly transmutes.

The objective is simply:

Make every center match its own frame.

A completed board is a naturally readable mosaic: every tile has a center and frame from the same face family.

Why the system has depth

A click is both an inversion and a local overwrite. It always leaves the source opposing a uniform neighborhood, so solving is not equivalent to independently toggling mismatches. Correct tiles may need to be used as stamps, temporarily spoiled, restored by a second click, or preserved until a boundary has been oriented.

The player can eventually derive four useful facts without any hidden information:

Boundary stamping: after one click, all neighbors show the source’s old face while the source shows the opposite face.

Double transmutation: clicking the same tile twice restores its center but rewrites all its neighbors to the opposite of its original face.

Order shapes seams: overlapping stamps do not commute. The later source controls shared boundary tiles.

Finishing signatures: a non-isolated final click must end on a target tile whose target face opposes every target neighbor.

2. Logical notation

The matrices below are implementation notation only. No letters are rendered in the game.

A = Amber face = binary state 0

V = Violet face = binary state 1

. = absent logical cell

Coordinates are zero-indexed (row, column) in the displayed matrix.

All cells marked A or V are present reagent tiles. Two present cells are neighbors exactly when their logical coordinates differ by one in one axis and zero in the other. There is no diagonal adjacency, wrapping, or adjacency across an absent coordinate.

Replay shorthand

The engine has only ordinary clicks. The following are specification abbreviations:

C(r,c) — click (r,c) once.

D(r,c) — click (r,c) twice consecutively.

P(a → b) — click adjacent tile a, then adjacent tile b.

P is not a separate action. In every reference replay where it appears, its two cells are opposite immediately before the first click. Under that condition, clicking a and then b restores both endpoint faces while stamping the external boundary with a’s original face. Reversing the pair stamps the opposite face.

3. Exact state machine
Persistent state
LevelState:
    rows: integer
    columns: integer
    present[row, column]: boolean
    face[row, column]: {0, 1}       # meaningful only when present
    target[row, column]: {0, 1}     # meaningful only when present
    terminal: boolean

There is no persistent selected tile or cursor:

selected_cell = null
cursor_state = null

Animation may retain a transient pre-action snapshot for rendering, but that snapshot must not influence later game logic.

Input space

The engine action space remains exactly:

UP
DOWN
LEFT
RIGHT
SPACE
CLICK(x, y)

This game uses only CLICK.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic no-ops. They do not change faces, targets, terminal state, or any hidden counter.

Valid click mapping

A pixel click is valid when it lies anywhere inside the 6×6 rendered body of a present reagent tile, including its target frame. The center and frame are not separate clickables.

A click in:

a one-pixel tile gap,

an absent logical slot,

or the background outside the board

is invalid and leaves the state unchanged.

Valid click transition

For a valid clicked tile p, take a snapshot of the current faces and set:

old = face[p]
next_face = face.copy()

next_face[p] = 1 - old

for each q in orthogonal_neighbors(p):
    next_face[q] = old

All assignments use the same pre-click source value old and commit simultaneously.

This direct assignment is equivalent to the player-facing formulation:

opposite neighbors copy the source’s old face;

matching neighbors remain unchanged.

No target frame participates in this calculation.

Resolution order inside step()

For every input action:

Resolve whether the action is a meaningful valid click.

If valid, snapshot the old source face and current neighbor faces.

Compute the complete next face array.

Commit that array atomically.

Evaluate the win validator.

Render the deterministic transmutation feedback.

If the validator passed, render completion feedback and set terminal = true.

There are no autonomous updates between calls to step().

Exact success condition

A level succeeds exactly when:

for every logical coordinate p:
    if present[p]:
        face[p] == target[p]

The validator ignores:

absent cells,

animation overlays,

click count,

click history,

which tile was clicked last,

invalid clicks,

and unused directional or SPACE inputs.

Any action sequence satisfying the equality is allowed. Reference replays are not required to be unique.

Once terminal becomes true, subsequent actions leave the solved board frozen.

There is no loss condition.

4. Rendering contract
Canvas and board placement

Render canvas: 64×64 pixels

Background: color 4 very dark gray

No text

No outer grid border

No internal lattice lines

Each logical slot has a pitch of 7 pixels. A present tile occupies a 6×6 body, leaving a one-pixel background gutter before the next slot.

For a level with matrix dimensions R × C:

render_width  = 7*C - 1
render_height = 7*R - 1

origin_x = floor((64 - render_width) / 2)
origin_y = floor((64 - render_height) / 2)

Logical tile (r,c) occupies:

x = origin_x + 7*c ... origin_x + 7*c + 5
y = origin_y + 7*r ... origin_y + 7*r + 5

A 9×9 matrix therefore occupies 62×62 pixels and begins at (1,1).

Absent slots render only as background. The game must not draw a rectangular board outline around them.

Tile construction

Each reagent is a connected 6×6 shape:

FFFFFF
FCCCCF
FCCCCF
FCCCCF
FCCCCF
FFFFFF

F is the immutable target frame.

C is the current-face panel.

Amber target frame

Base outer ring: color 12 orange.

Mid-edge frame accents at local pixels
(0,2),(0,3),(5,2),(5,3),(2,0),(3,0),(2,5),(3,5)
use color 11 yellow.

Violet target frame

Base outer ring: color 9 blue.

Four frame corners (0,0),(0,5),(5,0),(5,5) use color 10 light blue.

Amber current face

Inner 4×4 fill: color 11 yellow.

Large solid cross: color 4.

Cross pixels:

(1,2),(1,3),
(2,1),(2,2),(2,3),(2,4),
(3,1),(3,2),(3,3),(3,4),
(4,2),(4,3)
Violet current face

Inner 4×4 fill: color 15 purple.

Large hollow diamond: color 0 white.

Diamond pixels:

(1,2),(1,3),
(2,1),(2,4),
(3,1),(3,4),
(4,2),(4,3)

A mismatch therefore has visibly different center and frame colors as well as incompatible cross/diamond geometry. The player never has to distinguish two near-neighbor colors alone.

Click affordance

Every present tile appears as a raised, frame-enclosed module. Its entire 6×6 body is one large hit region.

No background region, gap, animation wave, or completion glow is clickable.

Animation and feedback

The face array is committed before animation. Animation never consumes an input or causes another state transition.

Valid click

Use four short deterministic visual subframes:

The source already shows its new persistent face. Its frame receives a one-frame white outline.

A narrow old-face-colored wave crosses each existing orthogonal contact.

Neighbors that changed receive a white outline; considered neighbors that remained unchanged receive a gray outline.

All overlays disappear, leaving the settled persistent state.

Input is ignored while these subframes are playing. A renderer that exposes only settled frames may omit the intermediate overlays; all causal evidence remains visible in the final state.

Invalid click

Render a three-frame crossed ripple in color 8 red at the clicked position. Do not alter any tile.

Completion

After the final transmutation wave, pulse the outermost pixels of all reagent frames once with color 14 green. Then freeze the solved mosaic. The green overlay is feedback only and is not part of the validator.

5. Authored progression
Level	Title	Main discovery or pressure	Reference depth
1	First Transmutation	Matching center to frame; source flips	1 click
2	Unequal Witnesses	Opposite neighbors copy; matching neighbors remain	1 click
3	Keep the Heart	Double transmutation and guarded arms	2 clicks
4	Diamond of Stamps	Overlapping double stamps impose order	8 clicks
5	The Keystone	Oriented seam pairs and a unique finishing site	25 clicks
6	Borrowed Seams	Temporary regression, revisits, and several plausible finishers	41 clicks
7	Bent Crucible	Contact loops and an interior double-click finish	55 clicks

The first three levels are intentionally microscopic. Each isolates one causal fact and contains no decorative cells. The jump after Level 4 comes from interacting contact neighborhoods rather than movement padding.

6. Level specifications
Level 1 — First Transmutation
Player-facing discovery

The board contains one pressable reagent. Its Amber center sits inside a Violet target frame. There is no competing object or alternative interpretation.

Clicking the tile changes its center to Violet and immediately produces completion feedback. This simultaneously teaches:

which object is clickable,

that clicking flips its face,

that the frame is a target rather than a second state,

and that matching all centers to frames is the objective.

This one-action solved example is the positive goal evidence for the entire game.

State-machine instance

Dimensions: 1×1

Initial current faces:

A

Target frames:

V

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
C(0,0)

The clicked source has old face A, no neighbors, and becomes V.

Intended depth: 1 click.

Validation and attribution check

The board succeeds after that click because its only current face equals its target.

No simultaneous neighbor effect exists, so the player can attribute the visible change solely to the source-flip rule. There is no irrelevant component and no alternate tile to test.

Level 2 — Unequal Witnesses
Player-facing discovery

The center reagent is Amber but wants Violet. It has three neighbors:

the north neighbor is already Amber and wants to remain Amber;

the west and east neighbors are Violet but both want Amber.

Clicking the center:

flips the source to Violet;

leaves the already-matching north neighbor Amber;

rewrites both opposite Violet neighbors to Amber.

A mistaken “every adjacent tile toggles” hypothesis predicts that the north tile would change. It visibly does not. A mistaken “all neighbors copy the new source” hypothesis predicts Violet neighbors. They visibly become Amber instead.

The single settled frame therefore identifies the old-face copying rule.

State-machine instance

Dimensions: 2×3

Initial current faces:

.A.
VAV

Target frames:

.A.
AVA

Present coordinates:

(0,1), (1,0), (1,1), (1,2)

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
C(1,1)

Exact transition:

source (1,1): A -> V
north  (0,1): A -> A
west   (1,0): V -> A
east   (1,2): V -> A

Intended depth: 1 click.

Validation and attribution check

The center click is the only one-click solution. Every visible tile participates in the distinguishing experiment. Nothing changes for an unrelated reason.

Level 3 — Keep the Heart
Player-facing discovery

The central Amber reagent already matches its Amber frame. Its four inner arms are Amber but want Violet. Behind every arm is a Violet tip that already matches.

A direct click on an arm appears locally useful—the arm becomes Violet—but it stamps its old Amber face onto the matching outer tip, merely moving the mismatch outward. This makes the known one-at-a-time repair strategy visibly self-defeating.

Clicking the correct center once also appears bad:

the center becomes Violet and mismatches;

the Amber arms remain Amber.

Clicking that same center again causes the important discovery:

the center returns to Amber;

all four arms become Violet;

the outer tips remain untouched.

The second click gives persistent evidence that a repeated source can rewrite an entire surrounding cross without sacrificing its own final face.

State-machine instance

Dimensions: 5×5

Initial current faces:

..V..
..A..
VAAAV
..A..
..V..

Target frames:

..V..
..V..
VVAVV
..V..
..V..

The center is (2,2). The inner arms are:

(1,2), (2,1), (2,3), (3,2)

The guarded outer tips are:

(0,2), (2,0), (2,4), (4,2)

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
D(2,2)

Expanded:

1. C(2,2)
2. C(2,2)

After click 1:

center: V
inner arms: A
outer tips: V

After click 2:

center: A
inner arms: V
outer tips: V

Intended depth: 2 clicks.

Validation and attribution check

An exhaustive search of this nine-tile state space gives shortest distance 2, with the center double-click as a shortest solution.

The outer tips are necessary teaching evidence: they make direct arm clicks visibly destructive rather than merely less efficient.

Level 4 — Diamond of Stamps
Player-facing discovery

This level contains four ordinary tiles that happen to occupy contextual “hub” positions:

northwest: (2,2), face A
northeast: (2,4), face V
southwest: (4,2), face V
southeast: (4,4), face A

Each hub has the guarded arm-and-tip structure from Level 3, so double-clicking each hub is a justified known action.

The new pressure comes from the four shared connector tiles between the hubs. A double-click on either adjacent hub can rewrite a connector. Because the two hubs have opposite faces, they stamp opposite results. The target connector therefore specifies which hub must act later.

The visible connector targets induce this partial order:

northwest before northeast
northwest before southwest
northeast before southeast
southwest before southeast

Northeast and southwest may be exchanged. The level is therefore a diamond dependency network, not a left-to-right chain.

Reversing one required pair leaves a physically visible connector with the wrong face. The failure is an overwritten seam, not an abstract ordering rejection.

State-machine instance

Dimensions: 7×7

Initial current faces:

..V.A..
..A.V..
..AVV..
..V.A..
..VAA..
..V.A..
..A.V..

Target frames:

..V.A..
..V.A..
..AAV..
..A.V..
..VVA..
..A.V..
..A.V..

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
D(2,2)
D(2,4)
D(4,2)
D(4,4)

Expanded action ranges:

1-2: D(2,2)   # northwest
3-4: D(2,4)   # northeast
5-6: D(4,2)   # southwest
7-8: D(4,4)   # southeast

Intended depth: 8 clicks.

Key causal checkpoints

After D(2,2), its guard arm is fixed, but its east and south shared connectors are Violet.

After D(2,4), the north shared connector becomes Amber. Reversing these two doubles would leave that connector Violet.

After D(4,2), the west shared connector becomes Amber.

After D(4,4), the east and south connectors become Violet, completing the diamond.

Validation and anti-degeneracy

An exhaustive search over the 16-tile state gives shortest distance 8. No solution with fewer clicks exists.

The shared connectors cannot be treated as independent toggles: each lies in two overlapping stars. The four guarded arms force all four hub regions to participate, while the connector geometry determines their useful order.

Level 5 — The Keystone
Player-facing discovery

The board expands the Level 4 structure into a 3×3 network of contextual hubs at:

(2,2), (2,4), (2,6)
(4,2), (4,4), (4,6)
(6,2), (6,4), (6,6)

The boundary hubs retain guarded arms and tips, so known double-stamp experiments remain locally justified.

The first new local impasse appears around (6,4):

the hub and its lower arm are opposite and both can be preserved;

the two horizontal connector targets beside the hub require the arm’s old Amber face;

either endpoint clicked once would spoil itself;

clicking the arm and then the hub restores both endpoints while stamping the two side connectors Amber.

This teaches the oriented pair use of two ordinary clicks. Reversing the order would stamp Violet instead. The same consequence is reused on the right edge and then the upper-left edge.

The target also contains a unique visible finishing signature:

(4,3) wants V
its only neighbors, (4,2) and (4,4), both want A

It is the only target tile in the level that opposes all its existing target neighbors. That makes it a natural candidate for the final click. Clicking it early is unproductive because later center and west stamps overwrite it.

State-machine instance

Dimensions: 9×9

Initial current faces:

..V.A.V..
..A.V.A..
..AAVAA..
..A.A.V..
AVVAAVVVA
..A.A.V..
..AAVAA..
..A.V.A..
..V.A.V..

Target frames:

..V.A.V..
..V.A.V..
..AVVAA..
..A.A.A..
AAAVAVVAA
..A.V.A..
..AAVAA..
..V.A.V..
..V.A.V..

Every A or V position is present. Every . is absent.

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
 1. D(6,4)              # actions 1-2
 2. D(6,6)              # actions 3-4
 3. D(6,2)              # actions 5-6
 4. D(2,2)              # actions 7-8
 5. D(4,6)              # actions 9-10
 6. P((7,4) -> (6,4))   # actions 11-12
 7. D(2,6)              # actions 13-14
 8. P((4,7) -> (4,6))   # actions 15-16
 9. D(4,4)              # actions 17-18
10. D(2,4)              # actions 19-20
11. P((1,2) -> (2,2))   # actions 21-22
12. D(4,2)              # actions 23-24
13. C(4,3)              # action 25

Intended depth: 25 clicks.

Every P begins with opposite endpoint faces in this replay.

Final checkpoint

After action 24, exactly these tiles mismatch:

(4,2), (4,3)

Their current values are:

(4,2) = V, target A
(4,3) = A, target V

Tile (4,4) is already Amber.

Clicking (4,3) with old face Amber:

flips (4,3) to Violet;

stamps both neighbors (4,2) and (4,4) Amber;

completes the board.

Necessity and anti-cleanliness

For the listed replay:

no strict prefix reaches the target;

deleting any one of the 25 clicks while retaining all later coordinates fails to reach the target.

This establishes that the replay contains no removable padding, though alternate global solutions are allowed and global shortestness is not asserted.

The layout is not a checklist of nine hub states. The lower, right, and upper-left seam pairs overwrite shared connectors, while the central and west regions converge on the delayed keystone click.

Level 6 — Borrowed Seams
Player-facing discovery

This level uses a 3×4 hub network:

hub rows:    2, 4, 6
hub columns: 1, 3, 5, 7

Top hubs have arms at row 1 and tips at row 0. Bottom hubs have arms at row 7 and tips at row 8. Midpoint cells connect neighboring hubs.

No tile has a new operational rule. The new difficulty comes from using established actions in deliberately inconvenient roles.

Several earlier lessons reverse strategically:

A guarded arm click, previously evidence of a bad plan, is now useful because a later hub double will repair the displaced guard.

A tile that looks like a possible finisher may be clicked early to prepare a seam.

The top-middle oriented pair must be used twice, with intervening stamps changing what its boundary means.

Correct regions are intentionally overwritten because their old faces are the needed reagent for another boundary.

The target has four nontrivial last-click signatures:

(4,3), (4,7), (7,7), (8,7)

The target alone narrows the candidates, but does not prescribe one. The current neighborhood near the end determines which candidate is productive.

State-machine instance

Dimensions: 9×9

Initial current faces:

.V.A.V.A.
.A.V.A.V.
.AAVAAVV.
.A.A.V.A.
.VAAAVVA.
.A.V.V.V.
.AVVAAVV.
.A.V.A.V.
.V.A.V.A.

Target frames:

.V.A.V.A.
.V.A.V.A.
.AAVAAAV.
.V.V.A.V.
.VVAVVVA.
.A.V.A.V.
.AVVVAVV.
.V.A.V.A.
.V.A.V.V.

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
 1. D(6,1)              # actions 1-2
 2. D(2,7)              # actions 3-4
 3. D(2,3)              # actions 5-6
 4. D(2,5)              # actions 7-8
 5. D(6,3)              # actions 9-10
 6. D(4,5)              # actions 11-12
 7. P((7,1) -> (6,1))   # actions 13-14
 8. D(4,1)              # actions 15-16
 9. C(7,7)              # action 17
10. D(4,3)              # actions 18-19
11. P((1,7) -> (2,7))   # actions 20-21
12. D(6,5)              # actions 22-23
13. P((1,3) -> (2,3))   # actions 24-25
14. D(4,7)              # actions 26-27
15. D(2,1)              # actions 28-29
16. D(6,7)              # actions 30-31
17. P((7,5) -> (6,5))   # actions 32-33
18. C(4,4)              # action 34
19. P((3,5) -> (4,5))   # actions 35-36
20. P((1,3) -> (2,3))   # actions 37-38
21. P((3,7) -> (4,7))   # actions 39-40
22. C(4,3)              # action 41

Intended depth: 41 clicks.

Every P begins with opposite endpoint faces.

Important reinterpretations

D(6,1) initially uses the lower-left hub as a ring stamp. The later pair (7,1) → (6,1) uses that same hub as the second endpoint of an oriented seam.

The top-middle pair (1,3) → (2,3) appears at actions 24–25 and again at actions 37–38. Its operational behavior is unchanged; intervening contact stamps make its later boundary consequence different.

C(4,4) at action 34 increases the number of mismatches from four to six. It is still productive because it establishes the old faces required by the three final oriented pairs. The player receives immediate visible evidence of the sacrifice and has unlimited room to test the hypothesis.

Final checkpoint

After action 40, exactly these three cells mismatch:

(4,3), (4,4), (3,3)

Current local state around (4,3):

source (4,3) = V, target A

neighbors:
(5,3) = V, target V
(3,3) = A, target V
(4,4) = A, target V
(4,2) = V, target V

Clicking (4,3) flips it to Amber and stamps all four neighbors Violet, solving the level in one physically legible final event.

Necessity and anti-cleanliness

For the listed replay:

no strict prefix is solved;

all 41 individual clicks are deletion-essential.

The board cannot be reduced to an ordered list of stations. Its top, bottom, and center dependencies interleave, one pair is revisited, and an apparent regression is required before the finishing neighborhood can exist.

Level 7 — Bent Crucible
Player-facing discovery

The final level preserves the Level 6 tile vocabulary and adds only four ordinary reagent cells:

left wing:  (3,0), (4,0)
right wing: (4,8), (5,8)

Together with existing cells, these form two bent 2×2 contact loops:

left loop:
(3,0) -- (3,1)
  |         |
(4,0) -- (4,1)

right loop:
(4,7) -- (4,8)
  |         |
(5,7) -- (5,8)

These are not new devices. They are ordinary tiles whose extra contact path changes which stamp reaches which seam.

The loops are used throughout the puzzle rather than appended at the end:

the left wing is oriented almost immediately;

the right wing is double-stamped early;

its lower bend is used much later as an oriented pair;

the adjacent east hub is then revisited;

the final solution closes at an interior four-neighbor site, not at either wing.

The mature insight is that a currently correct center may need to be clicked twice at the very end. The first click visibly makes the board worse; the second converts the entire prepared neighborhood.

State-machine instance

Dimensions: 9×9

Initial current faces:

.V.A.V.A.
.A.V.A.V.
.AAVAAVV.
AA.A.V.A.
AVAAAVVAA
.A.V.V.VA
.AVVAAVV.
.A.V.A.V.
.V.A.V.A.

Target frames:

.V.A.V.A.
.V.A.V.A.
.AAVAAAV.
VA.A.A.V.
AVVAAVAAV
.A.V.A.VV
.AVVVAVV.
.V.A.V.V.
.V.A.V.A.

Initial auxiliary state:

selected_cell = null
terminal = false
animation = none
Reference replay
 1. D(6,7)              # actions 1-2
 2. P((4,1) -> (4,0))   # actions 3-4
 3. C(7,7)              # action 5
 4. D(2,7)              # actions 6-7
 5. D(4,8)              # actions 8-9
 6. D(2,3)              # actions 10-11
 7. D(4,3)              # actions 12-13
 8. P((4,7) -> (5,7))   # actions 14-15
 9. C(4,4)              # action 16
10. D(6,3)              # actions 17-18
11. D(6,1)              # actions 19-20
12. D(2,1)              # actions 21-22
13. D(2,5)              # actions 23-24
14. D(6,3)              # actions 25-26, revisit
15. P((1,3) -> (2,3))   # actions 27-28
16. P((7,1) -> (6,1))   # actions 29-30
17. D(6,5)              # actions 31-32
18. D(4,1)              # actions 33-34
19. D(4,5)              # actions 35-36
20. D(4,7)              # actions 37-38
21. P((1,7) -> (2,7))   # actions 39-40
22. P((4,8) -> (5,8))   # actions 41-42
23. D(4,7)              # actions 43-44, revisit
24. C(4,3)              # action 45
25. P((3,5) -> (4,5))   # actions 46-47
26. P((7,5) -> (6,5))   # actions 48-49
27. P((1,3) -> (2,3))   # actions 50-51
28. P((3,7) -> (4,7))   # actions 52-53
29. D(4,5)              # actions 54-55

Intended depth: 55 clicks.

Every P begins with opposite endpoint faces.

Contact-loop roles

The left pair (4,1) → (4,0) acts before most of the central network. Its stamp reaches (3,0) through the bent contact geometry, so the wing is not an independent side task.

The right tile (4,8) is first used as a double-stamp source and later as the first endpoint of (4,8) → (5,8). The same tile therefore changes from a ring tool into a seam tool without changing its rule.

Hub (4,7) is doubled, affected by the right bend, doubled again, and finally used as an oriented pair endpoint. Its repeated roles arise entirely from timing and contact geometry.

Final double-transmutation checkpoint

After action 53, only two cells mismatch:

(4,6), (5,5)

The intended finishing source (4,5) is already Violet and already matches its Violet target.

Its four neighbors currently are:

(5,5) = V, target A
(3,5) = A, target A
(4,6) = V, target A
(4,4) = A, target A

The target therefore calls for an Amber ring around a Violet center.

Action 54 clicks the correct Violet center once:

center becomes A
all four neighbors become V

The visible mismatch count rises from two to five. This is deliberate and directly recalls Level 3.

Action 55 clicks the same center again:

center returns to V
all four neighbors become A

The entire five-tile target cross completes simultaneously.

The final target has five nontrivial last-click candidates:

(2,3), (4,5), (8,7), (4,0), (3,0)

Only the prepared current state makes the interior candidate (4,5) capable of resolving every remaining mismatch.

Necessity and anti-cleanliness

For the listed replay:

the final target is not reached at any strict prefix;

all 55 individual clicks are deletion-essential.

The bent wings are directly used as sources or pair endpoints and also change the contact neighborhoods of existing cells. Removing their spatial loops changes the replay transitions themselves.

The solution is not a sweep through objects in visible order. Both wings are used early, central regions are revisited, the top-middle pair appears twice, and the last two clicks intentionally move away from completion before producing the finishing ring.

7. Implementation verification suite

An implementation should ship with deterministic replay tests.

Generic transition tests
Source and selective neighbors

Using Level 2’s initial state, clicking (1,1) must produce exactly its target state.

Double transmutation

For any source p with pre-double face b:

after C(p), C(p):
    face[p] == b

    for every q in neighbors(p):
        face[q] == 1 - b

All non-neighbor cells must equal their pre-double values unless separately clicked.

Simultaneity

A neighbor changed during a click must not recursively transmute its own neighbors. Only the directly clicked source generates a wave.

Target independence

Changing target[p] in a test fixture while keeping the current faces identical must not change the result of a click.

Authored replay tests

For every level:

apply(reference_replay, initial_faces) == target_faces

Expected reference lengths:

Level 1:  1
Level 2:  1
Level 3:  2
Level 4:  8
Level 5: 25
Level 6: 41
Level 7: 55

No strict prefix of any reference replay may satisfy the target.

For Levels 5–7, deleting any single click from the reference replay while retaining all later click coordinates must fail to produce the target.

For Level 4, exhaustive shortest-path validation should return distance 8.

Finishing-signature tests

Define a nontrivial target finishing candidate as a present tile with at least one neighbor such that:

for every q in neighbors(p):
    target[q] == 1 - target[p]

Expected candidate sets:

Level 5:
{(4,3)}

Level 6:
{(4,3), (4,7), (7,7), (8,7)}

Level 7:
{(2,3), (4,5), (8,7), (4,0), (3,0)}

These signatures are planning evidence only. The validator must never require the player to finish at one of the reference sites.

Input and terminal tests

Directional inputs and SPACE leave all persistent state unchanged.

Invalid background clicks leave all persistent state unchanged.

Clicking any pixel inside a tile’s 6×6 body invokes that tile.

Clicking the one-pixel gap between tiles invokes neither tile.

After success, every subsequent action leaves the solved frame unchanged.

The level loader exposes exactly the seven instances above and reports num_levels = 7.
