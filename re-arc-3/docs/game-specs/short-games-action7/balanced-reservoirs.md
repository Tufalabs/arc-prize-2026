Balanced Reservoirs

Exact authored level count: 7

This is a game about moving one visible charge unit along an arrow becoming difficult because shared reservoirs must satisfy several overlapping equalities at once.

Balanced Reservoirs begins as an immediately readable redistribution toy. Every click has one local physical effect: one yellow charge token crosses one visible arrow. The complication is that reservoirs belong to outlined balance groups, and a shared reservoir contributes its current count to every group touching it. A transfer that repairs one badge may therefore split another.

The game never adds valves, capacities, currencies, switches, or hidden arithmetic. Later difficulty comes entirely from the arrangement of the same reservoirs, directed links, group overlaps, and conserved tokens.

1. Player-facing rules

Only arrow plates are clickable. Clicking one transfers exactly one charge unit from the arrow’s tail reservoir to its head reservoir.

Charge is conserved. Tokens are never created, destroyed, merged, or split.

An empty source cannot transfer. The selected arrow flashes red, jolts one pixel backward, and returns to its original position. The settled state is unchanged.

Each colored outline is a balance group. Its badge is:

split red when its member reservoirs have different counts;

level green when all its member reservoirs have the same count.

Zero is a valid equal count. A group containing only empty reservoirs is balanced.

A shared reservoir belongs fully to every outline touching it. Its count is not copied; the same physical tokens affect all attached badges.

Different disconnected balance components may finish at different heights. Only overlaps force two groups to share a common count.

The level succeeds when every authored group badge is simultaneously balanced.

Temporary imbalance is allowed.

Tokens may pass through reservoirs that were already balanced.

There is no predetermined target count stored in the validator.

There is no requirement to minimize clicks.

There are no capacities, timers, move limits, or transfer costs.

The only active player action is clicking. UP, DOWN, LEFT, RIGHT, and SPACE are accepted but leave the state unchanged.

2. Logical geometry and rendering contract
2.1 Canvas and native board

The render canvas is exactly 64×64 pixels, indexed from (0,0) at the upper-left to (63,63) at the lower-right.

The native gameplay space is a 4×4 logical lattice. Logical position (c,r), where c,r ∈ {0,1,2,3}, has pixel center:

center_x = 8 + 16*c
center_y = 8 + 16*r

Logical lattice lines are not rendered. The only visible borders are authored group outlines.

Specification diagrams below use the logical lattice for compactness. Their row and column separators do not appear in the game.

2.2 Reservoirs

A reservoir centered at (cx,cy) occupies the connected 11×11 footprint:

x = cx-5 ... cx+5
y = cy-5 ... cy+5

Rendering:

outer one-pixel rim: color 1;

transparent-looking interior: color 4;

background outside objects: color 5;

charge tokens: color 11.

Each reservoir has 25 possible display slots at local offsets:

x offsets: -4, -2, 0, 2, 4
y offsets, bottom to top: 4, 2, 0, -2, -4

Slots fill left-to-right on the bottom row, then upward. Each token is a separated 1×1 square. The separation is essential: the amount must remain countable from the current frame alone.

There is no gameplay capacity. All authored levels contain at most 19 total tokens, so even a legal state concentrating every token in one reservoir fits in the 25 display slots.

2.3 Directed arrow plates

Every transfer link connects orthogonally adjacent logical positions.

A one-direction link uses a 5×5 dark-gray plate centered halfway between its reservoirs. A white arrow on the plate shows the source and destination.

plate background: color 3;

arrow: color 0;

click hit region: the entire 5×5 plate, not merely the white arrow pixels.

Thus the intended interaction is “click this arrow module,” not “find the arrow’s exact tip.”

Opposing arrows

When both directions are permitted, they are two distinct, non-overlapping 5×5 plates.

For horizontally adjacent reservoirs:

eastbound arrow: upper lane, center (mid_x, cy-4);

westbound arrow: lower lane, center (mid_x, cy+4).

For vertically adjacent reservoirs:

southbound arrow: right lane, center (cx+4, mid_y);

northbound arrow: left lane, center (cx-4, mid_y).

This placement is invariant across all levels.

Only arrow plates are clickable. Reservoirs, tokens, badges, outlines, reference panels, and background cells are not clickable. Clicking them is a no-op.

2.4 Balance-group outlines

A balance group is authored as a connected set of logical cells.

For rendering, its region is the union of the corresponding 16×16 logical cell areas. Only the outer contour of that union is drawn. Internal boundaries between adjacent member cells are omitted, so the result is a single polyomino-like outline rather than a logic grid.

Group colors, in authored order, are:

Group order	Color ID
1	9 blue
2	6 magenta
3	15 purple
4	12 orange
5	13 maroon

No authored reservoir belongs to more than two groups.

When two group contours coincide along a segment:

the earlier group uses the contour pixel closest to the outside of its region;

the later group uses the immediately adjacent pixel toward the inside of its own region;

both strands follow corners with a one-pixel miter.

This produces two visibly parallel colored strands rather than overwriting one group with another.

Reservoir membership tabs

A reservoir has one 2×2 colored rim tab for each group containing it.

For a reservoir in one group, the tab is placed on its north rim. For a reservoir in two groups, groups are sorted by authored order and assigned:

first group  -> north tab
second group -> east tab

Tabs extend one pixel outside the gray reservoir rim. Shared reservoirs are therefore recognizable immediately without following the entire outline.

2.5 Balance badges

Every group has one authored badge_edge, written as reservoir:side.

The 5×5 badge is centered eight pixels from that reservoir’s center in the named direction. Every badge edge faces a logical cell not belonging to that group, and no badge overlaps an arrow plate.

The group contour runs directly into the badge frame.

Badge rendering:

outer 5×5 frame: the group’s color;

inner 3×3 field: color 4;

unequal state: two red ledges at visibly different heights, color 8;

equal state: one centered horizontal green beam with a short central support, color 14.

The shape is a balance pictogram, not rendered text.

2.6 Color usage
ID	Use
0	arrows, completion pulse
1	reservoir rims
2	inactive demonstration frame
3	arrow plates
4	reservoir and badge interiors
5	background
6	second group
7	brief click/transfer highlight
8	unequal badge and blocked-arrow flash
9	first group
10	transfer-animation trail
11	charge tokens
12	fourth group
13	fifth group
14	balanced badge core
15	third group

No text, numerals, or letters are rendered in the game grid.

3. Formal state machine
3.1 Persistent state
level_index
counts[reservoir_id]       nonnegative integer
terminal                    boolean
last_event                  none | transfer | blocked | irrelevant_click

Group balance is derived rather than independently stored:

balanced(group) =
    all(counts[m] == counts[group.members[0]]
        for m in group.members)

There is no selected reservoir, selected token, or persistent cursor.

Every level begins with:

terminal = false
last_event = none
animation_phase = settled
selected_link = none

The pointer itself is external UI state and is not rendered in benchmark observations.

3.2 step(action) resolution order

For every action:

If terminal is already true, return the unchanged terminal state.

If the action is UP, DOWN, LEFT, RIGHT, or SPACE, return the unchanged settled state.

If the action is a click:

hit-test all arrow plates;

authored plates are disjoint, so at most one link can match;

if no plate matches, set last_event = irrelevant_click and leave counts unchanged;

otherwise let the matching link be source → destination.

If counts[source] == 0:

do not change either count;

set last_event = blocked;

play the blocked animation;

recompute nothing else.

Otherwise:

atomically decrement counts[source];

atomically increment counts[destination];

set last_event = transfer;

play the transfer animation;

recompute every badge from the resulting counts.

After badge recomputation, set:

terminal = all(balanced(g) for g in level.groups)

If terminal became true, play the completion pulse and report level success.

Only one token can move per call to step().

3.3 Animation

Animations are deterministic visual subframes generated by the same action. They do not consume additional player steps, do not advance timers, and do not accept queued input.

Successful transfer: five subframes

Pre-transfer frame.

The highest-index occupied source slot becomes a moving token at the source rim.

The token appears at the center of the arrow plate with a color-10 one-pixel trail.

The token appears at the destination rim.

Settled post-transfer frame; the destination’s next slot is filled and badges update.

The authoritative state returned by the action is the settled fifth frame.

Blocked transfer: three subframes

The arrow and plate flash color 8.

The white arrow graphic shifts one pixel opposite its pointing direction.

The arrow returns; the original settled state is restored.

Badge transitions

When a badge changes between unequal and equal, its core briefly flashes color 0 during the final transfer subframe, then settles to red or green.

Completion

When the final required badge becomes equal, all group contours pulse color 0 for four visual frames. No additional step is consumed.

4. Level progression summary
Level	Main discovery or composition	Verified path depth
1	click an arrow; equal counts satisfy the badge	1
2	opposing arrows are separate directional controls	6
3	a shared reservoir couples two groups	6
4	balanced zero-groups must be uniformly raised	27
5	an already-balanced group must act as a conduit	41
6	disconnected components can require different heights	44
7	a fixed visible anchor pins one component while two flows merge and split	52

link×n below means clicking that link plate on n consecutive calls to step().

5. Authored levels
Level 1 — One Unit Level
Player-facing discovery

The player sees:

one reservoir with two tokens;

one empty reservoir;

a red split badge around the pair;

one obvious arrow from the full reservoir to the empty one;

a small, already-balanced demonstration in a separate inert panel.

The arrow is the only active-looking control. One click moves a token and immediately makes the live pair visually match the solved example.

This establishes the objective without requiring the player to infer an arbitrary symbolic badge meaning.

Static solved reference

A noninteractive reference panel occupies pixels:

x = 36 ... 63
y = 1 ... 15

It has:

color-2 outer frame;

color-4 interior;

two miniature five-pixel-wide reservoirs, each containing one color-11 token;

a blue group contour;

an equal green balance badge.

It contains no arrow plates and is ignored by hit-testing and validation.

Initial state
r0: .  .  .  .
r1: A  B  .  .
r2: .  .  .  .
r3: .  .  .  .
Reservoir	Logical position	Initial count
A	(0,1)	2
B	(1,1)	0

Initial transient state:

settled; selected_link = none; last_event = none
Group
ID	Color	Members	Badge edge
G1	9	{A,B}	B:E

Badge center: (32,24).

Link
ID	Direction
AB	A → B

This is a single centered plate.

Verified solution
AB

State change:

[A,B] = [2,0] → [1,1]

G1 changes from unequal to equal, and the level completes.

Intended solution depth: 1 successful click.

Attribution check

Nothing else changes on the click. There is only one live group, one arrow, and one movable unit of surplus. The token motion, badge transition, and success occur in the same local region, so the result cannot plausibly be attributed to another object.

Level 2 — Two Arrowheads
Player-facing discovery

A central reservoir holds all nine tokens. The left and right reservoirs are empty.

Each adjacent pair has two visibly separate opposing arrow plates. The player must use:

a west-pointing arrow to fill the left reservoir;

an east-pointing arrow to fill the right reservoir.

Thus the result cannot be explained by a fixed screen direction such as “links always move right.” Clicking either reverse arrow while its endpoint is empty produces the red blocked jolt without risking charge.

Initial state
r0: .  .  .  .
r1: L  C  R  .
r2: .  .  .  .
r3: .  .  .  .
Reservoir	Position	Initial count
L	(0,1)	0
C	(1,1)	9
R	(2,1)	0

Initial transient state:

settled; selected_link = none; last_event = none
Group
ID	Color	Members	Badge edge
G1	9	{L,C,R}	C:N

Badge center: (24,16).

Links
ID	Direction	Plate lane
LC	L → C	eastbound upper lane
CL	C → L	westbound lower lane
CR	C → R	eastbound upper lane
RC	R → C	westbound lower lane
Verified solution
CL×3, CR×3

Checkpoints:

[L,C,R]
start   = [0,9,0]
CL×3    = [3,6,0]
CR×3    = [3,3,3]

The two blocks may be interleaved without changing the required net transfers.

Intended solution depth: 6 successful clicks.

Attribution check

The level contains no overlaps, transit structure, decoys, or secondary objective. The left and right results mirror one another, while the required arrowheads point in opposite screen directions. A click on an empty-source reverse plate gives immediate evidence that the plate specifies a source rather than merely an undirected gap.

Level 3 — Shared Rim
Player-facing discovery

The center reservoir has two differently colored rim tabs and lies inside two overlapping outlines.

The right group starts balanced at [1,1]. Any useful transfer from A into the shared reservoir immediately splits that previously green badge. The player therefore sees, on the first productive action, that the same reservoir count participates in both groups.

Both physical links are reversible, making experiments safe during this foundational overlap lesson.

Initial state
r0: .  .  .  .
r1: A  S  B  .
r2: .  .  .  .
r3: .  .  .  .
Reservoir	Position	Initial count
A	(0,1)	7
S	(1,1)	1
B	(2,1)	1

Initial transient state:

settled; selected_link = none; last_event = none
Groups
ID	Color	Members	Badge edge
G1	9	{A,S}	A:N
G2	6	{S,B}	B:N

Badge centers:

G1: (8,16)
G2: (40,16)

S receives the north tab for G1 and east tab for G2.

Links
ID	Direction
AS	A → S
SA	S → A
SB	S → B
BS	B → S

Each adjacent pair uses opposing plates.

Verified solution
AS×3, AS, SB×2

Important checkpoints:

[A,S,B]
start      = [7,1,1]   G2 is balanced
AS×3       = [4,4,1]   G1 becomes balanced
AS         = [3,5,1]   G1 must be broken again
SB×2       = [3,3,3]   both groups balance

Intended solution depth: 6 successful clicks.

Necessity and misconception check

Because G1 requires A=S and G2 requires S=B, all three final counts must match. Conservation fixes that count at three.

Every winning sequence must change S and therefore affect both badges. The reverse links permit recovery but cannot shorten the six-click net requirement:

net A→S = 4
net S→B = 2

The level cannot be solved by treating the two outlines independently or by preserving every green badge.

Level 4 — Raise the Zeroes
Player-facing discovery

Eight tokens begin in H. Every other reservoir is empty.

Two whole groups are already green because all of their members contain zero. Earlier play has shown that green means only “equal now,” not “finished forever.” The sole active source leads into a winding transfer network, and the first token that reaches F or E visibly splits a zero-balanced group.

The three group overlaps form a dependency cycle:

G1 and G2 share E;

G1 and G3 share F;

G2 and G3 share C.

The group dependency structure therefore does not merely mirror the directed transfer path.

Initial state
r0: .  .  G  H
r1: .  .  F  E
r2: .  A  C  D
r3: .  .  B  .
Reservoir	Position	Count
H	(3,0)	8
G	(2,0)	0
F	(2,1)	0
E	(3,1)	0
D	(3,2)	0
C	(2,2)	0
A	(1,2)	0
B	(2,3)	0

Initial transient state:

settled; selected_link = none; last_event = none
Groups
ID	Color	Members	Badge edge
G1	9	{H,G,F,E}	G:W
G2	6	{E,D,C}	E:N
G3	15	{F,C,A,B}	F:W

Badge centers:

G1: (32,8)
G2: (56,16)
G3: (32,24)
Links
HG: H → G
GF: G → F
FE: F → E
ED: E → D
DC: D → C
CA: C → A
CB: C → B

All are single-direction links.

Target deduction

The overlap graph connects all eight reservoirs into one equality component. There are eight visible tokens, so the only final distribution is:

H=G=F=E=D=C=A=B=1

The validator does not store that target; it follows from the groups and conservation.

Verified solution
HG×7,
GF×6,
FE×5,
ED×4,
DC×3,
CA,
CB

Replay checkpoints, in order [H,G,F,E,D,C,A,B]:

start = [8,0,0,0,0,0,0,0]
HG×7  = [1,7,0,0,0,0,0,0]
GF×6  = [1,1,6,0,0,0,0,0]
FE×5  = [1,1,1,5,0,0,0,0]
ED×4  = [1,1,1,1,4,0,0,0]
DC×3  = [1,1,1,1,1,3,0,0]
CA    = [1,1,1,1,1,2,1,0]
CB    = [1,1,1,1,1,1,1,1]

Intended solution depth: 27 successful clicks.

Necessity and anti-cleanliness check

Every arrow is the sole entrance to at least one required downstream reservoir. The forced net transfers are:

HG=7, GF=6, FE=5, ED=4, DC=3, CA=1, CB=1

The winding route turns five times and branches at C. Meanwhile, the group overlaps form a triangle rather than following the route in a line. Replacing the layout with a corridor would remove the distinction between the transfer network and the equality network.

The central insight is not “visit the groups in order.” It is that several green zero-groups must be raised together to the globally forced height of one.

Level 5 — Balanced Conduit
Player-facing discovery

C1, C2, and C3 begin at [1,1,1], so their blue conduit badge is green. S contains eight tokens.

The only way out of S is into C1. Thus progress requires deliberately breaking the already-correct conduit group. The seven surplus units then travel through all three conduit members before reaching J.

At J, the flow must split asymmetrically:

two units enter the short left branch;

four units enter the long right branch;

J keeps one.

The right group occupies a compact 2×2 area, but its arrows run around three sides of that area. B4 is physically adjacent to B1, yet there is no direct B1→B4 plate.

Initial state
r0: .   C2  C1  S
r1: .   C3  .   .
r2: A1  J   B1  B4
r3: A2  .   B2  B3
Reservoir	Position	Count
S	(3,0)	8
C1	(2,0)	1
C2	(1,0)	1
C3	(1,1)	1
J	(1,2)	0
A1	(0,2)	0
A2	(0,3)	0
B1	(2,2)	0
B2	(2,3)	0
B3	(3,3)	0
B4	(3,2)	0

Initial transient state:

settled; selected_link = none; last_event = none
Groups
ID	Color	Members	Badge edge
G1	9	{C1,C2,C3}	C3:W
G2	6	{S,C1}	S:S
G3	15	{C3,J,A1,B1}	C3:E
G4	12	{A1,A2}	A1:N
G5	13	{B1,B2,B3,B4}	B1:N

Badge centers:

G1: (16,24)
G2: (56,16)
G3: (32,24)
G4: (8,32)
G5: (40,32)
Links
SC1:   S  → C1
C1C2:  C1 → C2
C2C3:  C2 → C3
C3J:   C3 → J

JA1:   J  → A1
A1A2:  A1 → A2

JB1:   J  → B1
B1B2:  B1 → B2
B2B3:  B2 → B3
B3B4:  B3 → B4
Target deduction

The overlap network connects all 11 reservoirs. Total charge is 11, so every reservoir must finish at one.

The important consequence is that G1’s initial [1,1,1] is indeed its final distribution, but its members must temporarily carry the seven-unit packet.

Verified solution
SC1×7,
C1C2×7,
C2C3×7,
C3J×7,

JA1×2,
A1A2,

JB1×4,
B1B2×3,
B2B3×2,
B3B4

Replay checkpoints in order:

[S,C1,C2,C3,J,A1,A2,B1,B2,B3,B4]

start    = [8,1,1,1,0,0,0,0,0,0,0]
SC1×7    = [1,8,1,1,0,0,0,0,0,0,0]
C1C2×7   = [1,1,8,1,0,0,0,0,0,0,0]
C2C3×7   = [1,1,1,8,0,0,0,0,0,0,0]
C3J×7    = [1,1,1,1,7,0,0,0,0,0,0]
JA1×2    = [1,1,1,1,5,2,0,0,0,0,0]
A1A2     = [1,1,1,1,5,1,1,0,0,0,0]
JB1×4    = [1,1,1,1,1,1,1,4,0,0,0]
B1B2×3   = [1,1,1,1,1,1,1,1,3,0,0]
B2B3×2   = [1,1,1,1,1,1,1,1,1,2,0]
B3B4     = [1,1,1,1,1,1,1,1,1,1,1]

Intended solution depth: 41 successful clicks.

Necessity and anti-cleanliness check

The directed cuts force the packet sizes:

SC1=C1C2=C2C3=C3J=7

left branch:
JA1=2, A1A2=1

right branch:
JB1=4, B1B2=3, B2B3=2, B3B4=1

The two branches are deliberately unequal. The compact right-hand placement creates a tempting spatial shortcut between B1 and B4, but the visible arrows require the U-shaped route. No extra link or reservoir is ornamental.

G1 has a contrasting role to earlier green badges: it begins and ends balanced at the same height, yet must be used as a temporary conduit.

Level 6 — Different Heights
Player-facing discovery

The level contains two disconnected balance components:

X consists of four reservoirs;

Y consists of five reservoirs.

The Y groups begin balanced at zero. The X groups are unequal and contain all 14 tokens.

C and G are joined by a reversible pair of arrows. This creates a safe local experiment at the key conceptual impasse:

click C→G once:
    both zero-balanced Y badges split

click G→C once:
    the exact initial distribution returns

The player can therefore test whether charge is allowed to enter an already-green independent component before making an irreversible routing commitment deeper inside Y.

The final state necessarily demonstrates that disconnected components need not share a common height.

Initial state
r0: A  B  .  .
r1: .  D  C  .
r2: .  E  G  J
r3: .  .  H  I
Reservoir	Position	Count
A	(0,0)	5
B	(1,0)	5
D	(1,1)	1
C	(2,1)	3
G	(2,2)	0
E	(1,2)	0
H	(2,3)	0
I	(3,3)	0
J	(3,2)	0

Initial transient state:

settled; selected_link = none; last_event = none
Groups
ID	Color	Members	Badge edge
GX1	9	{A,B,D}	A:S
GX2	6	{D,C}	D:S
GY1	15	{G,E}	E:S
GY2	12	{G,H,I,J}	J:N

Badge centers:

GX1: (8,16)
GX2: (24,32)
GY1: (24,48)
GY2: (56,32)
Links
AB: A → B
BD: B → D
DC: D → C

CG: C → G
GC: G → C

GE: G → E

GH: G → H
HI: H → I
IJ: I → J

CG and GC use the standard opposing vertical lanes. All other links are one-direction plates.

Target deduction

Let every X reservoir finish at x and every Y reservoir finish at y.

The visible total gives:

4x + 5y = 14

For nonnegative integer counts, the only solution is:

x = 1
y = 2

Thus X must lose ten units and Y must rise from zero to two per reservoir.

This target is not hard-coded. It is a consequence of group sizes and conserved visible charge.

Verified solution

The shortest authored path does not need the optional reversible probe:

AB×4,
BD×8,
DC×8,
CG×10,

GE×2,

GH×6,
HI×4,
IJ×2

Replay checkpoints in order [A,B,D,C,G,E,H,I,J]:

start = [5,5,1,3,0,0,0,0,0]
AB×4  = [1,9,1,3,0,0,0,0,0]
BD×8  = [1,1,9,3,0,0,0,0,0]
DC×8  = [1,1,1,11,0,0,0,0,0]
CG×10 = [1,1,1,1,10,0,0,0,0]
GE×2  = [1,1,1,1,8,2,0,0,0]
GH×6  = [1,1,1,1,2,2,6,0,0]
HI×4  = [1,1,1,1,2,2,2,4,0]
IJ×2  = [1,1,1,1,2,2,2,2,2]

Intended solution depth: 44 successful clicks.

A blind-discovery path may add the reversible probe CG, GC without changing the eventual solution.

Necessity and anti-cleanliness check

The required cut flows are:

AB=4
BD=8
DC=8
net C→G=10
GE=2
GH=6
HI=4
IJ=2

Y is deliberately asymmetric:

E is a one-edge short branch requiring two units;

H–I–J is a three-edge U-route carrying six, then four, then two units.

G and J are adjacent in the 2×2 Y group, but no direct G→J arrow exists. The authored route, rather than group membership alone, determines how charge reaches J.

The reverse G→C plate is not a fake alternative route. It is positioned at the precise new conceptual impasse and permits a reversible experiment about independent balance heights.

Level 7 — Pinned Exchange
Player-facing discovery

The final level combines:

two independent balance components;

overlap chains inside each component;

an isolated visible anchor;

two charge streams merging at b3;

a winding U-route through the B component;

a delayed final split at b5.

Reservoir a5 has one token and no incident arrows. Because it belongs to GA2, it visibly pins:

a3 = a4 = a5 = 1

The overlap at a3 then pins all of GA1 to one as well. The player does not have to gamble on the destination height before using the one-way bridge.

After the five A reservoirs are fixed at one, conservation leaves 14 tokens for the seven B reservoirs, forcing each B reservoir to finish at two.

Some B groups begin green at zero, but the earlier curriculum has established that a locally balanced group can still require a uniform shift.

Initial state
r0: a1  a2  b1  .
r1: a4  a3  b3  b2
r2: a5  b6  b5  b4
r3: .   .   b7  .
Reservoir	Position	Count
a1	(0,0)	5
a2	(1,0)	4
a3	(1,1)	3
a4	(0,1)	2
a5	(0,2)	1
b1	(2,0)	4
b3	(2,1)	0
b2	(3,1)	0
b4	(3,2)	0
b5	(2,2)	0
b6	(1,2)	0
b7	(2,3)	0

Initial transient state:

settled; selected_link = none; last_event = none
Groups
ID	Color	Members	Badge edge
GA1	9	{a1,a2,a3}	a3:S
GA2	6	{a3,a4,a5}	a5:E
GB1	15	{b1,b3,b2}	b1:E
GB2	12	{b2,b4,b5}	b4:S
GB3	13	{b5,b6,b7}	b7:E

Badge centers:

GA1: (24,32)
GA2: (16,40)
GB1: (48,8)
GB2: (56,48)
GB3: (48,56)
Links
a1a2: a1 → a2
a2a3: a2 → a3
a4a3: a4 → a3
a3b3: a3 → b3

b1b3: b1 → b3

b3b2: b3 → b2
b2b4: b2 → b4
b4b5: b4 → b5

b5b6: b5 → b6
b5b7: b5 → b7

a5 has no link plate. Its one visible token is immobile and serves as a physical target anchor, not as a new object type.

Target deduction

From isolated a5 and GA2:

a3=a4=a5=1

From the shared a3 and GA1:

a1=a2=a3=1

Therefore all five A reservoirs consume five of the 19 visible tokens.

The remaining 14 tokens belong to the seven-reservoir B equality component:

b1=b2=b3=b4=b5=b6=b7=2

Equivalently, the global equation is:

5·1 + 7·2 = 19
Verified solution
a1a2×4,
a2a3×7,
a4a3,

b1b3×2,
a3b3×10,

b3b2×10,
b2b4×8,
b4b5×6,

b5b6×2,
b5b7×2

Replay checkpoints in order:

[a1,a2,a3,a4,a5,b1,b3,b2,b4,b5,b6,b7]

start    = [5,4,3,2,1,4,0,0,0,0,0,0]
a1a2×4   = [1,8,3,2,1,4,0,0,0,0,0,0]
a2a3×7   = [1,1,10,2,1,4,0,0,0,0,0,0]
a4a3     = [1,1,11,1,1,4,0,0,0,0,0,0]
b1b3×2   = [1,1,11,1,1,2,2,0,0,0,0,0]
a3b3×10  = [1,1,1,1,1,2,12,0,0,0,0,0]
b3b2×10  = [1,1,1,1,1,2,2,10,0,0,0,0]
b2b4×8   = [1,1,1,1,1,2,2,2,8,0,0,0]
b4b5×6   = [1,1,1,1,1,2,2,2,2,6,0,0]
b5b6×2   = [1,1,1,1,1,2,2,2,2,4,2,0]
b5b7×2   = [1,1,1,1,1,2,2,2,2,2,2,2]

Intended solution depth: 52 successful clicks.

Necessity and anti-cleanliness check

Each required flow follows directly from a visible one-way cut:

a1→a2 = 4
a2→a3 = 7
a4→a3 = 1
a3→b3 = 10

b1→b3 = 2

b3→b2 = 10
b2→b4 = 8
b4→b5 = 6

b5→b6 = 2
b5→b7 = 2

Every arrow is necessary.

The spatial instance is not a linear checklist:

a1 and a4 feed a3 through different approaches;

b1 supplies a second stream directly into b3;

b3 is vertically adjacent to b5, but there is no direct arrow;

charge must travel right to b2, down to b4, then left to b5;

b5 belongs to both GB2 and GB3 and becomes the final split point;

a5 is never clicked, but its fixed visible count is essential evidence that determines the entire A target.

A wrong cross-component amount fails physically: surplus charge becomes stranded downstream of a one-way cut. The player has enough visible evidence to determine the correct amount before making that commitment.

6. Mechanic reuse in contrasting roles
Mechanic	Early role	Later contrasting roles
Balanced badge	immediate success evidence in level 1	a green badge must be broken in level 3; zero-green groups must be raised in level 4; a green conduit is disturbed and restored in level 5
Shared reservoir	simple two-group coupling in level 3	three overlaps form a dependency cycle in level 4; shared nodes become merge, branch, and target-propagation points later
Directed link	one obvious transfer in level 1	opposing directional controls in level 2; safe reversibility in level 3; winding one-way commitments in levels 4–5; a reversible allocation probe in level 6
Conservation	confirms one-for-one transfer	fixes a common height in levels 3–5; distinguishes independent component heights in level 6; combines with a fixed anchor in level 7
Temporary imbalance	incidental evidence in level 3	required uniform group shifting in level 4; transport through an already-correct group in level 5; moving ten units into a zero-balanced component in level 6
Spatial adjacency	reservoirs sit beside their links	later, adjacency without an arrow is deliberately not a shortcut, as in B1/B4, G/J, and b3/b5
7. Validator and implementation invariants

The implementation should assert the following for every authored level:

number of levels == 7

all reservoir coordinates are distinct
all links join orthogonally adjacent reservoirs
all link hitboxes are disjoint
all badge hitboxes are disjoint from link hitboxes
all group member sets are connected on the logical lattice
every reservoir belongs to at least one group
no reservoir belongs to more than two groups
all initial counts are nonnegative
sum(counts) remains constant after every step

Success must be implemented solely as:

all(
    all(counts[m] == counts[group.members[0]]
        for m in group.members)
    for group in level.groups
)

The validator must not:

compare against the solution vectors documented above;

require a particular click order;

require minimum depth;

require all reservoirs on the board to share one count;

treat a green badge as locked;

forbid temporary imbalance;

impose a reservoir capacity;

count irrelevant clicks against a budget;

infer transfers from reservoir clicks;

advance state during animation without a new step().

This preserves the central promise of the game: the local rule is always one visible token crossing one visible arrow, while the puzzle comes from understanding which globally compatible distribution those local moves must create.
