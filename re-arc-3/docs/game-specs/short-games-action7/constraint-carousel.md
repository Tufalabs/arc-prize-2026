Constraint Carousel

Authored level count: exactly 7

This is a game about rotating four tokens becoming difficult because every irreversible quarter-turn changes the color inventories of several overlapping frames.

Constraint Carousel is a click-only puzzle about local permutations and overlapping set constraints. A pivot always performs the same simple operation: rotate its four surrounding tokens clockwise, then become permanently spent. The challenge comes from understanding which tokens cross each goal frame’s boundary, preserving frames that already contain one token of each color, and ordering overlapping pivots so later rotations receive the colors they need.

There is no fixed target picture. Any settled board on which every authored goal frame contains exactly one token of each gameplay color is valid.

1. Gameplay vocabulary
Gameplay colors

Only four colors are used for tokens:

Role	Color ID
Magenta token	6
Blue token	9
Yellow token	11
Green token	14

Other rendered colors:

Role	Color ID
Complete frame / white feedback	0
Gray structural overlap	2
Spent bracket	3
Pivot socket / hollow-pip center	4
Background	5
Live pivot arrow and bracket	10

No text, numbers, or documentation IDs appear in the rendered grid.

Logical coordinates

Token positions use zero-indexed logical coordinates (r,c), with rows increasing downward and columns increasing rightward.

Frc denotes the goal frame whose 2×2 footprint has top-left token (r,c).

Prc denotes the pivot whose 2×2 rotation footprint has top-left token (r,c).

These names are documentation identifiers only. They are not rendered.

Active cells

A logical token cell exists exactly when it belongs to at least one listed goal-frame footprint or pivot footprint.

Cells shown as -- in a level matrix are inactive:

they render as background color 5;

they contain no token;

they are ignored by the validator;

no listed frame or pivot ever includes them.

Thus every rendered token participates in at least one goal or transformation. There are no inert corner tokens added merely to make a rectangle.

2. Rendering contract
Canvas and board placement

The render canvas is 64×64 pixels.

For a logical board of height H and width W, define:

x0 = 3 + floor((64 - 10W) / 2)
y0 = 3 + floor((64 - 10H) / 2)

(x0,y0) is the top-left pixel of token (0,0).

An active token (r,c) is a connected 4×4 square:

x = x0 + 10c ... x0 + 10c + 3
y = y0 + 10r ... y0 + 10r + 3

There are no per-cell wells or cell borders. Tokens float on the black background, with goal frames and pivot brackets providing the relevant grouping.

Goal frames

A frame Frc surrounds the four tokens:

(r,c)       (r,c+1)
(r+1,c)     (r+1,c+1)

Its outer rectangle is:

L = x0 + 10c - 3
R = x0 + 10c + 16
T = y0 + 10r - 3
B = y0 + 10r + 16

The frame is a two-pixel-wide square outline.

Incomplete frame: color 2.

Complete frame: color 0.

If overlapping frame strokes share pixels, complete white has rendering priority over gray. The individual corner pips remain unambiguous.

No borders are drawn around ordinary logical cells.

Frame inventory pips

Each goal frame has four 3×3 pips occupying its four outer corners:

Frame corner	Color represented
Northwest	6
Northeast	9
Southeast	11
Southwest	14

For a pip whose represented color is C:

Missing, count = 0       Exact, count = 1       Duplicate, count >= 2

C C C                    C C C                  C 5 C
C 4 C                    C C C                  C 5 C
C C C                    C C C                  C 5 C

Thus a player can immediately see:

which colors are absent;

which colors occur exactly once;

which colors occur too many times.

The duplicate display need not distinguish two copies from three or four copies; the rendered tokens themselves expose the exact count.

A frame is complete precisely when all four pips are solid.

With the ten-pixel lattice spacing, authored pip modules never geometrically overlap, even when their frames share tokens.

Pivot sockets

A pivot Prc occupies the 6×6 central junction between its four tokens:

x = x0 + 10c + 4 ... x0 + 10c + 9
y = y0 + 10r + 4 ... y0 + 10r + 9

This entire 6×6 module is its clickable region.

A live socket has background color 4 and a clockwise arrow in color 10. One acceptable 6×6 arrow mask is shown below, where a means color 10 and . means color 4:

. a a a a .
a . . . . a
. . . . . a
. . . . . a
. . a a a a
. . . a . .

The socket’s operational bracket is a thin color-10 set of L-shaped corner marks around:

x = x0 + 10c - 1 ... x0 + 10c + 14
y = y0 + 10r - 1 ... y0 + 10r + 14

Each L-shaped corner mark has arms three pixels long. This bracket is one pixel inside a coincident goal frame, so the two remain visually distinct.

After use:

the arrow disappears;

the socket becomes a color-4 square with a color-2 X;

its bracket changes from color 10 to color 3.

A live bracket has rendering priority over a spent bracket where neighboring corner marks happen to share pixels.

The pivot’s position at the center of its bracket makes the rotated region locally obvious. No connector tracing or arbitrary clickable hunting is required.

3. State machine
State variables
tokens: map from active logical cell -> one of {6,9,11,14}
goals: set of 2×2 top-left coordinates
pivots: map from 2×2 top-left coordinate -> LIVE or SPENT
selection: always null
animation_snapshot: null or the most recent valid rotation
solved: Boolean

There is no hidden random state, timer, energy, score, cursor selection, or persistent hover state.

All listed pivots begin live in all seven authored levels.

Rotation operation

For pivot Prc, let its current token block be:

a b
c d

A clockwise rotation produces:

c a
d b

Equivalently:

new[r,c]       = old[r+1,c]
new[r,c+1]     = old[r,c]
new[r+1,c+1]   = old[r,c+1]
new[r+1,c]     = old[r+1,c+1]

The pivot then becomes spent.

There are no chain reactions. One click rotates exactly one displayed bracket using the current settled token arrangement.

Action handling

The declared action space remains:

UP, DOWN, LEFT, RIGHT, SPACE, click(x,y)

Only clicking a live pivot socket changes gameplay state.

UP, DOWN, LEFT, RIGHT, and SPACE: no state change.

Click on a token, goal frame, status pip, bracket, inactive area, or background: no state change.

Click on a spent socket: no state change; its X shifts one pixel right and then left over two render-only frames.

Click anywhere in a live pivot’s 6×6 hit region: perform its rotation and spend it.

Invalid actions do not spend pivots and have no budget cost. A subtle one-frame gray flash on the outer canvas edge may acknowledge a keyboard or background no-op.

There is no persistent selected pivot. A click invokes a pivot directly.

step() resolution order

For a click on a live pivot:

Save the pre-action token positions for animation.

Apply the clockwise token permutation atomically.

Mark that pivot spent.

Recompute all goal-frame color counts.

Set solved = true exactly when every authored goal frame is complete.

Produce the deterministic rotation animation.

Return the final settled observation.

No state changes occur between calls to step(). Intermediate animation frames are render-only and accept no input.

Rotation animation

The four tokens visibly circulate around the perimeter of their 2×2 block:

old top-left moves right;

old top-right moves down;

old bottom-right moves left;

old bottom-left moves up.

Render three post-action motion frames using displacement magnitudes:

3 pixels, 7 pixels, 10 pixels

At the ten-pixel displacement, all tokens occupy their final logical positions.

During the first two motion frames, goal pips and outlines retain their pre-action appearance. On the final settled frame they update together, making the inventory change attributable to that one rotation.

Animation frames consume no steps and no budget. The pivot is logically spent as soon as the valid click is processed.

Success and dead states

A goal frame is valid iff the multiset of its four tokens is exactly:

{6, 9, 11, 14}

The level succeeds iff every listed goal frame is valid simultaneously.

The validator ignores:

exact token positions within an already valid frame;

unframed 2×2 regions;

unused live pivots;

whether all pivots have been spent;

inactive logical cells;

any global repeating pattern.

Unused pivots are explicitly allowed.

On success, the final solved frame remains visible for one settled observation before the host advances.

There is no automatic terminal failure. If all pivots are spent while some goal remains incomplete, the board stays visible with its hollow and duplicate pips. The host may restart the level outside the game action space.

4. Systemic invariants

The operational rule never changes between levels.

Bracket conservation. A rotation cannot change the four-color inventory of a goal frame coincident with its bracket. It can only change positions within that frame.

Boundary exchange. A goal frame changes inventory only when tokens cross its boundary. Pivots touching a frame are not automatically useful; the direction and exact overlap matter.

Shared contribution. A token in two overlapping goal frames is counted independently by both.

Irreversibility. Every pivot provides one clockwise turn and is then gone.

Noncommutativity. Overlapping pivots can carry a shared token to different destinations depending on their order.

Multiple valid pictures. The validator checks inventories rather than a target arrangement. Distinct terminal boards may all be correct.

5. Authored levels
Level 1 — One Bright Neighbor
Player-facing discovery

The left frame is already complete: its outline is white and all four pips are solid. The overlapping right frame is gray, with a hollow blue pip and a split yellow pip. This gives positive evidence for the objective before any action.

There is exactly one live pivot, centered inside the complete left frame. Clicking the only obvious arrow rotates those four tokens. The left frame remains complete while the right frame immediately becomes complete, matching the visual example.

This one action establishes:

what can be clicked;

what clockwise rotation does;

what a spent pivot looks like;

that the objective is one token of each color rather than a fixed picture.

State-machine instance
Board: 2 rows × 3 columns
Render origin: (x0,y0) = (20,25)
Selection/cursor state: null

Initial tokens:

09 14 06
06 11 11

Goals:

F00, F01

Live pivots:

P00

Initial inventories in color order (6,9,11,14):

F00 = (1,1,1,1), complete
F01 = (1,0,2,1), missing blue and duplicating yellow

Intended and unique solution:

P00

Bright-frame trace:

Start:       {F00}
After P00:   {F00,F01}

Canonical terminal tokens:

06 09 06
11 14 11

Intended depth: 1 meaningful action.

Attribution check: There is only one pivot and no unrelated object. The already-complete frame is deliberate positive goal evidence, not a decoy; its tokens visibly move while its inventory remains valid.

Level 2 — Same Four, New Places
Player-facing discovery

The sole goal frame is missing green and contains two magenta tokens. Two adjacent live pivots are visible.

P00 coincides exactly with the goal. Clicking it rotates the four tokens, spends the pivot, and leaves every status pip unchanged. This is the clean distinguishing experiment for conservation: positions changed, but the frame still has the same duplicate and missing colors.

The remaining offset pivot can now exchange a boundary token and repair the frame. Clicking the offset pivot first produces a different but still invalid inventory, after which the coincident pivot cannot change it.

State-machine instance
Board: 2 × 3
Render origin: (20,25)
Selection/cursor state: null

Initial tokens:

06 09 11
06 11 14

Goals:

F00

Live pivots:

P00, P01

Initial inventory:

F00 = (2,1,1,0)

Unique solution:

P00 → P01

Replay:

Start:       F00 = (2,1,1,0), incomplete
After P00:   F00 = (2,1,1,0), still incomplete
After P01:   F00 = (1,1,1,1), complete

The wrong first click P01 changes the frame to (2,0,1,1). Spending P00 afterward cannot alter that inventory.

Canonical terminal tokens:

06 09 06
11 14 11

Intended depth: 2 meaningful actions.

Attribution check: Only one goal inventory is displayed. The first rotation’s unchanged pips cannot be attributed to another frame, simultaneous mechanism, or hidden condition.

Level 3 — Handoff
Player-facing discovery

Two diagonal goal frames share the central token. Both begin incomplete:

F01 lacks blue and duplicates yellow;

F10 lacks green and duplicates magenta.

The first useful turn lights F10. The second lights F01 but extinguishes F10, making the shared-cell consequence unmistakable. The final pivot repairs both simultaneously.

The level teaches that a temporarily completed frame is not necessarily finished work and that overlapping rotations must be ordered.

State-machine instance
Board: 3 × 3
Render origin: (20,20)
Selection/cursor state: null

Initial tokens:

06 14 11
06 11 06
06 09 09

Goals:

F01, F10

Live pivots:

P00, P01, P11

Initial inventories:

F01 = (1,0,2,1)
F10 = (2,1,1,0)

Unique solution:

P00 → P11 → P01

Bright-frame trace:

Start:       {}
After P00:   {F10}
After P11:   {F01}
After P01:   {F01,F10}

Canonical terminal tokens:

06 09 06
11 14 11
06 09 06

Intended depth: 3 meaningful actions.

Necessity: All three pivots are required. Of the six possible full pivot orders, exactly this one succeeds. Removing either goal admits a one- or two-click shortcut.

Level 4 — Choose a Turn
Player-facing discovery

The player now encounters two legitimate ways to route colors through the upper pair of overlapping frames. The two route pivots are not equivalent:

one family uses P12 and leaves P02 live;

the other uses P02 and leaves P12 live.

Both produce valid but visibly different terminal boards. Spending both route pivots is unnecessary and disrupts the useful token placement.

The lower frame is spatially separated from the upper pair but coupled to them through P11, P01, and the final P22 repair. The objects are not laid out in their usage order.

State-machine instance
Board: 4 × 4
Render origin: (15,15)
Inactive cells: (2,0), (3,0)
Selection/cursor state: null

Initial tokens:

06 09 14 09
06 11 11 09
-- 14 09 14
-- 14 06 11

Goals:

F01, F02, F21

Live pivots:

P00, P01, P02, P11, P12, P22

All three goals begin incomplete.

Route A: use the middle-right pivot
P11 → P00 → P12 → P01 → P22

P02 remains live.

Bright-frame trace:

Start:       {}
After P11:   {F21}
After P00:   {F01,F21}
After P12:   {F01}
After P01:   {F01,F02}
After P22:   {F01,F02,F21}

Terminal tokens:

06 09 06 09
14 11 14 11
-- 09 06 14
-- 14 11 09
Route B: use the upper-right pivot
P00 → P02 → P11 → P01 → P22

P12 remains live.

Bright-frame trace:

Start:       {}
After P00:   {F01}
After P02:   {}
After P11:   {F01}
After P01:   {F01,F02}
After P22:   {F01,F02,F21}

Terminal tokens:

06 14 06 14
11 09 11 09
-- 09 06 09
-- 14 11 14

Intended depth: 5 meaningful actions.

Anti-degenerate check:

The shortest depth is five.

There are 26 solving five-click sequences, but all belong to exactly two pivot-subset families: use P12 or use P02.

Every pivot belongs to at least one shortest solution.

Removing F01, F02, or F21 reduces the shortest depth to four, two, or two respectively.

The two terminal matrices demonstrate directly that no hidden fixed picture is being checked.

Level 5 — Break the Good
Player-facing discovery

Two frames begin bright, but preserving them makes the remaining colors unreachable. The first required click, P20, rotates an incomplete coincident frame whose own inventory cannot improve; instead it deliberately extinguishes both bright neighbors.

The next click, P11, is a pure staging turn. It does not light a frame, but its visible token movement positions the colors needed by the upper pivots. Conservation, previously a safe invariant, is now a planning constraint: a coincident pivot may need to be used for position even when it cannot repair its own frame.

Later, P12 temporarily damages F21 after four frames have become complete, and P22 performs the final two-frame repair.

State-machine instance
Board: 4 × 4
Render origin: (15,15)
Inactive cell: (0,3)
Selection/cursor state: null

Initial tokens:

06 09 14 --
06 11 06 14
14 09 06 09
11 11 14 11

Goals:

F00, F10, F11, F20, F21

Live pivots:

P00, P01, P10, P11, P12, P20, P21, P22

Unique solution:

P20 → P11 → P00 → P01 → P10 → P21 → P12 → P22

Bright-frame trace:

Start:       {F10,F21}
After P20:   {}
After P11:   {}
After P00:   {F10}
After P01:   {F00}
After P10:   {F00,F21}
After P21:   {F00,F10,F20,F21}
After P12:   {F00,F10,F20}
After P22:   {F00,F10,F11,F20,F21}

Canonical terminal tokens:

06 09 06 --
11 14 11 14
06 09 06 09
11 14 11 14

Intended depth: 8 meaningful actions.

Anti-degenerate check:

All eight pivots are required.

Exhaustive ordering search finds exactly one solution.

Removing any one goal lowers the shortest depth from eight to between four and seven.

The two initially complete frames are liabilities during the opening rather than finished checklist items.

The top-right inactive cell removes a token that would otherwise be visually inert.

Level 6 — Two Feeds, One Release
Player-facing discovery

Seven goal frames form an irregular connected dependency network. The right branch, lower branch, and upper transfer can be prepared with some freedom, but every branch must deliver the correct color placement to the central P11.

Several frames alternate between useful and damaged states. The final central pivot repairs three outstanding frames at once while preserving four already-complete frames.

This level is not a single dependency chain. Independent branch actions can interleave, but each branch contains internal ordering constraints.

State-machine instance
Board: 5 × 5
Render origin: (10,10)
Inactive cells: (0,0), (3,4), (4,0), (4,4)
Selection/cursor state: null

Initial tokens:

-- 09 06 06 11
09 14 06 14 09
06 09 11 14 06
11 14 11 11 --
-- 09 06 09 --

Goals:

F01, F02, F11, F13, F20, F31, F32

Live pivots:

P01, P02, P03,
P10, P11, P12,
P20, P21, P22

One intended solution:

P03 → P22 → P12 → P21 → P20 → P10 → P01 → P02 → P11

Bright-frame trace:

Start:
{F11,F20,F31}

After P03:
{F11,F13,F20,F31}

After P22:
{F11,F20,F31,F32}

After P12:
{F01,F20,F31,F32}

After P21:
{F01,F32}

After P20:
{F01,F11,F31,F32}

After P10:
{F31,F32}

After P01:
{F11,F31,F32}

After P02:
{F11,F13,F31,F32}

After P11:
{F01,F02,F11,F13,F20,F31,F32}

Canonical terminal tokens:

-- 09 06 09 06
11 14 11 14 11
06 09 06 09 06
11 14 11 14 --
-- 09 06 09 --

Intended depth: 9 meaningful actions.

Anti-degenerate check:

Every one of the nine pivots is required.

Exactly 28 of the 9! = 362,880 full pivot orders solve the level.

Every goal is individually critical. Removing one produces a shortcut of depth four through eight.

The 28 solutions arise from interleaving partially independent branch work, not from optional or irrelevant pivots.

All rendered tokens lie in a listed frame or pivot footprint.

Level 7 — Full Carousel
Player-facing discovery

The finale begins with four complete frames, but the only viable opening dismantles three of them. The upper, lower, and right-side token streams must be staged before the inner pivots are spent.

Three established roles of the same pivot rule appear together:

P21 changes several visible inventories without immediately lighting or extinguishing a frame.

P04 changes the inventory of F14 but leaves it incomplete.

P34 changes no goal inventory at all. It is a necessary positional staging turn that places tokens for the later P23 and P24 boundary exchanges.

Near the end, P22 repairs three frames but deliberately breaks F23. The last pivot, P24, repairs both the right frame and that temporarily damaged central frame.

State-machine instance
Board: 6 × 6
Render origin: (5,5)

Inactive cells:
(0,0), (0,1),
(1,0),
(4,0),
(5,0), (5,1), (5,2), (5,5)

Selection/cursor state: null

Initial tokens:

-- -- 06 09 09 14
-- 14 06 14 06 11
09 09 11 14 09 14
06 11 14 09 11 09
-- 09 06 11 11 06
-- -- -- 06 14 --

Goals:

F02, F11, F14, F22, F23, F31, F32

Live pivots:

P04,
P12, P13,
P20, P21, P22, P23, P24,
P32, P34,
P43

One intended solution:

P43
→ P20
→ P12
→ P21
→ P13
→ P32
→ P04
→ P34
→ P23
→ P22
→ P24

Bright-frame trace:

Start:
{F11,F14,F31,F32}

After P43:
{F11,F14,F31}

After P20:
{F11,F14}

After P12:
{F14}

After P21:
{F14}

After P13:
{F02}

After P32:
{F02,F23}

After P04:
{F02,F23}

After P34:
{F02,F23}

After P23:
{F02,F22,F23}

After P22:
{F02,F11,F22,F31,F32}

After P24:
{F02,F11,F14,F22,F23,F31,F32}

Although the binary bright-frame set is unchanged after P21 and P04, their missing and duplicate pip identities visibly change. P34 is the sole pure positional turn: its four tokens visibly rotate, and the turn is necessary for the two following boundary exchanges.

Canonical terminal tokens:

-- -- 06 09 06 09
-- 14 11 14 11 14
06 09 06 09 06 09
11 14 11 14 11 14
-- 09 06 09 06 09
-- -- -- 14 11 --

Intended depth: 11 meaningful actions.

Anti-degenerate check:

No state reachable in zero through ten pivot activations satisfies all seven goals.

Since exactly eleven pivots exist, every pivot is required.

Across all 11! = 39,916,800 full pivot orders, exactly 4,107 solve—approximately 0.0103%.

Those orders reach four distinct valid terminal token arrangements, preserving the game’s non-picture objective.

Every goal is individually critical. Removing one yields a shortcut of depth:

Remove F02: depth 5
Remove F11: depth 6
Remove F14: depth 6
Remove F22: depth 6
Remove F23: depth 8
Remove F31: depth 7
Remove F32: depth 7

The active-cell silhouette is exactly the union of the goal and pivot footprints, so no decorative tokens remain outside the dependency network.

6. Curriculum and epistemic-solvability audit
Level	New causal lesson	Distinguishing experiment and evidence
1	Objective and pivot click	One obvious arrow transforms a gray frame into the same bright state as the visible solved example.
2	A coincident rotation preserves inventory	P00 visibly moves tokens while every pip on F00 stays unchanged.
3	Overlapping pivots require order	One frame lights, then the other lights while the first goes dark, then both light together.
4	A solution may leave a pivot unused and end in different pictures	Two shortest pivot-subset families produce two visibly different valid boards.
5	Completed frames may need to be sacrificed; conservation can force staging	The only solution first destroys both bright frames, then uses a turn that produces positional progress without completion.
6	Branches can interleave but must feed a shared release	Twenty-eight solutions exist, yet all nine pivots and all seven frames remain necessary.
7	Some turns manipulate future reach rather than current inventories	P34 is a visible, necessary positional rotation that prepares later boundary exchanges without directly affecting a goal.

The meaningful solution depths are:

1, 2, 3, 5, 8, 9, 11

Each counted action is an irreversible pivot activation. There are no corridor steps, repeated no-ops, duplicated pivot charges, or mechanical padding. The final eleven-click instance still exposes almost forty million raw pivot orders.

7. Implementation regression oracle

An implementation should satisfy all of the following tests:

Level 1:
minimum depth 1
unique solution P00

Level 2:
minimum depth 2
unique solution P00,P01

Level 3:
minimum depth 3
unique solution P00,P11,P01

Level 4:
minimum depth 5
26 solving length-5 sequences
every shortest sequence uses:
    {P00,P01,P11,P22,P12}
or
    {P00,P01,P11,P22,P02}

Level 5:
minimum depth 8
unique solution:
P20,P11,P00,P01,P10,P21,P12,P22

Level 6:
minimum depth 9
28 solving full orders
all nine pivots required

Level 7:
minimum depth 11
4,107 solving full orders
four distinct solved terminal token states
all eleven pivots required

For every level:

frame_ok(Frc) :=
    sorted(tokens in its four cells) == [6,9,11,14]

level_ok :=
    every listed frame satisfies frame_ok

No additional target-pattern, all-pivots-spent, global-count, or position-specific validation may be added.
