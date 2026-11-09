Fusion Links

“This is a game about merging equal tokens becoming difficult because every merge chooses one surviving endpoint and can permanently cut off unfinished mass.”

Fusion Links contains exactly 7 authored levels.

It is a click-only spatial reduction game. Every visible connector joins two token sockets and has an orange cap marking the endpoint that survives. Clicking an available connector between equal-tier tokens consumes both tokens and places one token of the next tier at the capped endpoint.

The arithmetic is simple. The difficulty is structural: an available connector may be valid but premature. Using it can empty a socket that still connects an unfinished branch to the goal.

1. Core player-facing system

Each token has tier t and mass 2
t
. Two equal-tier tokens fuse into one token of tier t+1.

The capped endpoint of the clicked link receives the result. The other endpoint becomes empty permanently: because every fusion requires both endpoints to be occupied, no later action can refill an empty socket.

Consequently:

Fusion is also directed transport.

A token may be visibly eligible for several links, but consuming it through one link destroys the other possibilities.

A connector can be active before it is strategically safe.

Large tokens often have to wait while another branch is reduced to the same tier.

The last capped connector entering the goal can be read backward as a required equal split of the total mass.

The objective is to leave exactly one token of the outlined tier in the outlined goal socket.

2. Action space

The environment accepts the required action space:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x, y) on any canvas cell

Only CLICK is used by the game.

UP, DOWN, LEFT, RIGHT, and SPACE are inert and return the unchanged settled frame. They do not select anything and do not reset the level.

There is no persistent cursor or selection state:

selected_connector = null
selected_socket = null
cursor_state = null

Every connector contains a visibly prominent fusion bead. Clicking any cell in that bead’s hit region attempts that connector immediately. Tokens, sockets, thin routing lines, goal outlines, and background cells are not clickable.

This makes clicking part of the central mechanic: the player is directly choosing both the pair to fuse and, through the visible cap, which endpoint survives.

3. Native geometry and rendering
3.1 Canvas

Render canvas: 64 × 64

Coordinates: (x, y), with (0, 0) at the top-left

No board border

No internal grid lines

No text

No HUD, energy meter, move counter, or timer

Background: color 5

The native game space is a graph of socket centers at authored pixel coordinates. It is not a 64×64 movement lattice.

3.2 Palette usage
Purpose	Color ID
Background	5
Empty socket / very dim link	4
Blocked structural link	3
Ordinary connector body	2
Bright connector body	1
Token outer bands	9
Token inner bands	10
Available fusion bead	11
Result-end cap	12
Goal outline	14
Invalid interaction	8
Irrecoverably severed component	13
Completion glint	0

Colors 6, 7, and 15 are unused.

3.3 Sockets and tokens

A socket is a quiet 3×3 dark-gray plate centered on its authored coordinate. An empty socket remains visible after its token is consumed.

Tier is redundantly encoded by disk diameter and concentric bands:

Tier	Diameter
0	2 cells
1	3 cells
2	4 cells
3	5 cells
4	6 cells
5	7 cells
6	8 cells

Tier-0 tokens are connected 2×2 disks, never single pixels.

For tiers 1–6, rasterize a rounded filled disk of the listed diameter. Starting at the outside, successive one-cell erosion bands alternate colors 9, 10, 9, and so on. Diameter alone is sufficient to distinguish every tier, while the bands make comparisons robust when tokens are close together.

Tokens are rendered above thin connector routes.

3.4 Goal socket

The goal is a normal socket surrounded by a one-cell color-14 outline whose interior aperture matches the required token diameter.

The outline has small port gaps wherever an incident connector enters, so that nearby beads, caps, and local tokens remain readable. The ring is not clickable.

A smaller token may occupy the goal during play. Success occurs only when its tier equals the outlined tier.

3.5 Connectors

Every connector has:

a straight horizontal, vertical, or diagonal centerline;

a 3×3 fusion bead near its midpoint;

an orange color-12 cap immediately before the endpoint where the result appears.

The cap remains visible even while the connector is blocked, so direction never has to be inferred from history.

Connector states are derived entirely from their two endpoints:

Endpoint state	Rendering
Both occupied, equal tiers	body 1–2, clear bead 11
Both occupied, unequal tiers	body 3, bead 4 with dark transverse bar
At least one endpoint empty	body and bead dim to 4
Invalid click transient	bead and endpoint rims pulse 8

Only the fusion bead is clickable. Its hit region is the full 3×3 square centered on the authored bead coordinate.

All authored bead hit regions are disjoint.

4. Logical state
level_id: integer
terminal: RUNNING | SUCCESS | FAILURE
failure_reason: null | SEVERED | STALLED

socket:
    id
    center_x
    center_y
    occupied: boolean
    tier: 0..6 or null

link:
    id
    endpoint_u
    endpoint_v
    target_endpoint       # u or v
    bead_center_x
    bead_center_y
    route_cells
    cap_cells

goal_socket_id
goal_tier

selected_connector = null
selected_socket = null

An authored level contains no hidden links, hidden tiers, randomized states, or delayed state changes.

5. Exact action and update semantics
5.1 Mergeability

A link is mergeable exactly when:

u.occupied
and v.occupied
and u.tier == v.tier
and u.tier < 6

Goal status, socket degree, and whether the merge is strategically wise do not affect mergeability.

Thus a connector entering the goal can be active much too early.

5.2 step(CLICK(x, y))

Resolve (x, y) against fusion-bead hit regions.

If no bead contains the click:

logical state is unchanged;

show a one-frame color-1 point at the clicked cell.

If the bead belongs to a blocked or dim link:

logical state is unchanged;

play the invalid animation.

If the link is mergeable:

read both tiers from the pre-action state;

identify target from the orange cap;

identify source as the other endpoint;

play the merge animation;

remove both original tokens;

create one token of tier old_tier + 1 at target;

leave source empty;

recompute every connector’s visual state;

run terminal validation in the order below.

No other connector resolves during the same action. There are no cascades.

5.3 Terminal validation order

After a successful merge commits:

A. Success
token_count == 1
and goal_socket.occupied
and goal_socket.tier == goal_tier

The validator does not require a separate click on the goal.

B. Immediate severance failure

Construct the undirected graph induced by currently occupied sockets, retaining a link only when both of its endpoints are occupied.

If the occupied sockets form more than one connected component, the state is irrecoverable: an emptied articulation socket has stranded live tokens behind it.

Set:

terminal = FAILURE
failure_reason = SEVERED

Every occupied component not containing the goal receives a one-cell maroon color-13 halo.

This is not an extra puzzle restriction. Empty sockets can never be refilled, so a separated component can never reach the goal. The early check merely reports that physical consequence immediately.

C. Stalled failure

If the occupied graph remains connected, but no link is mergeable:

terminal = FAILURE
failure_reason = STALLED

All remaining fusion beads pulse color 13.

5.4 Exploration and retries

There is no in-game action budget or timer.

Invalid clicks and inert actions cannot fail a level. The first three levels contain only one mergeable connector at each settled state, so no valid exploratory click is fatal.

A terminal failure ends the current harness episode. The next episode begins from the authored initial state; resetting is not itself a puzzle action.

6. Animation contract

Animation ticks do not call step(), consume actions, or alter logical state. New input is rejected until the animation settles.

Valid fusion

A valid fusion uses four deterministic display phases:

The target token brightens from 9 to 10; the source token contracts by one raster layer.

A ghost of the source token moves one-third of the way along the connector route toward the cap.

The ghost moves two-thirds of the way to the cap.

The source reaches the target; both old sprites disappear and the larger result sprite appears at the capped endpoint.

The logical merge commits only in phase 4.

For a very short local connector, repeated interpolated raster positions may be skipped, but the source must visibly travel toward the capped endpoint before the result appears.

Invalid connector

Three display phases:

bead turns color 8;

endpoint sprites shift one cell perpendicular to the connector;

sprites return and the settled barred state is restored.

No logical state changes.

Completion

After the final merge commits, the goal outline flashes:

14 → 0 → 14

The level then remains in terminal success.

7. Reusable reduction pod

Levels 4–7 use a repeated eight-socket spatial structure. It is not framed or labelled in the rendered grid; proximity and connector geometry provide the grouping.

Uppercase letters identify macro pods. Lowercase letters identify sockets inside one pod.

7.1 Base west-facing pod

For a pod root at (X, Y), the unrotated W orientation has these socket offsets:

Local socket	(dx, dy)
r	(0, 0)
a	(-5, -5)
b	(-5, 0)
c	(-10, -10)
d	(-5, 5)
e	(-15, -10)
f	(-5, -10)
g	(-10, -5)

The root is r. A completed base-tier-0 pod produces tier 3 at r.

7.2 Orientation transforms

Apply the transform to socket offsets and bead offsets before adding (X, Y):

Orientation	Transform
W	(dx, dy)
N	(-dy, dx)
E	(-dx, -dy)
S	(dy, -dx)

The fan occupies the named side of its root and leaves the other three cardinal rays clear for macro connectors.

7.3 Local links

All routes are straight segments.

Link	Endpoints	Result endpoint	Bead offset in W
rd	r–d	r	(-3, 3)
bg	b–g	b	(-8, -3)
rb	r–b	r	(-3, 0)
af	a–f	a	(-5, -8)
ce	c–e	c	(-13, -10)
ac	a–c	a	(-8, -8)
ra	r–a	r	(-3, -3)

The intended hierarchy is:

rd + bg → rb
af + ce → ac
rb + ac → ra

This hierarchy is not separately marked. It is exposed by topology:

d, g, f, and e are terminal branch sockets.

b and c must retain their terminal partners before being consumed.

a and r are shared endpoints.

Higher links are often active from the first frame, but their non-target endpoint still carries unfinished branches.

7.4 Pod frontier variants

A frontier variant is simply a visibly partially reduced instance of the same pod. Omitted sockets and already-used links are not drawn; they are not inert empty objects.

Every offset-0 variant below has total mass 8 and reduces to tier 3 at r.

Variant	Initial tokens	Visible links	Canonical local sequence
P8	r,a,b,c,d,e,f,g : tier 0	all seven	rd bg rb af ce ac ra
P7R	r:1; a,b,c,e,f,g:0	all except rd	bg rb af ce ac ra
P7A	a:1; r,b,c,d,e,g:0	all except af	rd bg rb ce ac ra
P6RA	r,a:1; b,c,e,g:0	bg rb ce ac ra	bg rb ce ac ra
P5R	r:2; a,c,e,f:0	af ce ac ra	af ce ac ra
P5A	a:2; r,b,d,g:0	rd bg rb ra	rd bg rb ra
P4	r,b,a,c:1	rb ac ra	rb ac ra
P2	r,a:2	ra	ra

P8+1 uses the P8 geometry and sequence but begins all eight tokens at tier 1. It has mass 16 and produces tier 4 at r.

In solution listings, F.rd means click the bead belonging to local link rd in pod F.

8. Universal macro anchor positions

Later levels use some or all of these root coordinates:

Pod	Root coordinate
F	(16, 16)
B	(32, 16)
E	(48, 16)
D	(16, 32)
R	(32, 32)
A	(48, 32)
G	(32, 48)
C	(48, 48)

Conceptual placement:

F       B       E

D       R       A

        G       C

Only links listed for the current level are drawn. There are no enclosing pod boxes or macro-grid borders.

All macro bead coordinates are the midpoint of their endpoint roots.

9. Progression overview
Level	Main discovery	Initial tokens	Intended successful clicks
1	Equal fusion, destination cap, goal	3	2
2	Fusion transports a growing token	6	5
3	A large front waits for an equal front	7	6
4	Active shared endpoints can be premature	32	31
5	Read the final split backward through uneven branches	48	47
6	Reversing the final cap reverses which whole half survives	56	55
7	Unscaffolded recursive reduction at two spatial scales	64	63

Invalid clicks are not included in these depths.

10. Authored levels
Level 1 — One Size Short
Player-facing discovery

The goal already contains a tier-1 token inside a visibly larger tier-2 outline. Its incoming connector is barred because its other endpoint holds a smaller token.

A second pair of equal small tokens is the only available fusion. Clicking it creates a tier-1 token at the capped endpoint and immediately clears the formerly barred goal connector.

The player can safely test the barred connector first; it only produces the red invalid pulse.

Initial state

Selection and cursor state: null.

Socket	Coordinate	Initial tier	Notes
a	(18, 32)	0
b	(32, 32)	0
g	(46, 32)	1	Goal socket

Goal:

goal_socket = g
goal_tier = 2

Links:

Link	Direction	Bead center
ab	a–b → b	(25, 32)
bg	b–g → g	(39, 32)

Initial connector state:

ab = active
bg = unequal/barred
Exact solving path
ab, bg

After ab, socket a empties and b becomes tier 1. After bg, only g remains, at tier 2.

Attribution and necessity check

The first frame contains no unrelated object. ab is the only valid action, and its three effects are simultaneous and local:

source socket a empties;

a larger token appears at capped endpoint b;

connector bg visibly changes from barred to available.

The three-node graph has two links, and both are required.

Level 2 — Carry the Fusion
Player-facing discovery

A staircase of increasingly large stationary tokens bends across the board. Only the first pair is initially equal.

Each fusion moves the growing mass onto the orange-capped endpoint. That endpoint is beside the next equal-sized token, so a new connector becomes available in a different direction.

A mistaken hypothesis such as “the result stays on the left,” “the result stays at the first token,” or “fusion does not move mass” is disproved by the first click.

Initial state

Selection and cursor state: null.

Socket	Coordinate	Initial tier
a	(10, 12)	0
b	(22, 12)	0
c	(22, 26)	1
d	(38, 26)	2
e	(38, 42)	3
g	(52, 42)	4

Goal:

goal_socket = g
goal_tier = 5

Links:

Link	Direction	Bead center
ab	a–b → b	(16, 12)
bc	b–c → c	(22, 19)
cd	c–d → d	(30, 26)
de	d–e → e	(38, 34)
eg	e–g → g	(45, 42)

Only ab is initially active. Every successful click activates exactly the next connector.

Exact solving path
ab, bc, cd, de, eg

Token progression:

b: 0+0 → 1
c: 1+1 → 2
d: 2+2 → 3
e: 3+3 → 4
g: 4+4 → 5
Attribution and necessity check

Each action changes one destination, one vacated socket, and one following connector. No simultaneous independent rule changes occur.

The bent geometry is necessary: the player sees the growing token turn downward, rightward, downward, and rightward according to the caps rather than following a fixed screen direction.

Level 3 — Meet the Large Front
Player-facing discovery

The goal already holds a tier-5 token inside a tier-6 outline. Its only incoming connector is barred against a tier-4 token.

A descending chain of visible tiers runs away from that tier-4 front:

4 ← 3 ← 2 ← 1 ← 0 + 0

The goal-side token cannot consume smaller pieces individually. The entire opposite front must be grown to tier 5 first.

Trying the final connector gives an immediate red pulse. The only active connector is the distant tier-0 pair, providing a productive experiment rather than a blind search.

Initial state

Selection and cursor state: null.

Socket	Coordinate	Initial tier
a	(50, 14)	0
b	(36, 14)	0
x	(22, 14)	1
y	(22, 30)	2
z	(22, 46)	3
h	(38, 46)	4
g	(38, 30)	5

Goal:

goal_socket = g
goal_tier = 6

Links:

Link	Direction	Bead center
ab	a–b → b	(43, 14)
bx	b–x → x	(29, 14)
xy	x–y → y	(22, 22)
yz	y–z → z	(22, 38)
zh	z–h → h	(30, 46)
hg	h–g → g	(38, 38)

Only ab is initially active.

Exact solving path
ab, bx, xy, yz, zh, hg
Attribution and necessity check

The final link remains visibly barred through tiers 1–4 and clears only when h reaches tier 5. This directly distinguishes equality from plausible mistaken rules such as:

a larger token may absorb smaller ones;

any connector entering the goal is valid;

total mass rather than equal endpoint tier determines availability.

No valid action can irreversibly fail this level.

Level 4 — Shared Roots
Player-facing discovery

Four repeated fan-shaped regions are joined into an asymmetric larger tree. Every starting token is tier 1, so every connector is initially available—including many that are unsafe.

Inside each pod, a shared root touches several available links. The terminal branch tokens have only one route forward. Fusing one of those terminal branches preserves the shared root and immediately bars its higher links by increasing its tier.

Fusing a higher link first empties an internal socket and visibly separates unfinished tokens. The disconnected branch receives the maroon severance halo immediately.

At macro scale the same question repeats: the final F–B link is already available, but F and B still anchor entire unfinished pods.

Pod instances

Selection and cursor state: null.

Pod	Root	Orientation	Variant
F	(16,16)	W	P8+1
D	(16,32)	W	P8+1
B	(32,16)	N	P8+1
E	(48,16)	N	P8+1

Each pod starts with eight tier-1 tokens and reduces to tier 4 at its local r.

Goal:

goal_socket = B.r
goal_tier = 6
Macro links
Link	Direction	Bead center
M.FD	F.r–D.r → F.r	(16, 24)
M.BE	B.r–E.r → B.r	(40, 16)
M.FB	F.r–B.r → B.r	(24, 16)

All local and macro connectors are initially active.

Exact solving path
F.rd, F.bg, F.rb, F.af, F.ce, F.ac, F.ra,
D.rd, D.bg, D.rb, D.af, D.ce, D.ac, D.ra,
M.FD,
B.rd, B.bg, B.rb, B.af, B.ce, B.ac, B.ra,
E.rd, E.bg, E.rb, E.af, E.ce, E.ac, E.ra,
M.BE,
M.FB

Successful clicks: 31.

Why every part is necessary

The initial graph has 32 occupied sockets and 31 connectors. It is a connected tree. Each successful fusion removes exactly one occupied socket, so reaching one token requires all 31 connectors to be used exactly once.

The high links are therefore not decoys. F.rb, for example, is required later, but using it before F.bg empties F.b and strands F.g.

Anti-cleanliness check

The level is not a line or checklist:

four local reductions can be interleaved;

each pod contains multiple initially available links;

the same active state means “use now” for terminal links and “reserve” for internal links;

the macro links repeat the same endpoint competition at a larger spatial scale;

wrong valid actions fail through a visible cut in the authored network, not through an abstract rejection.

Level 5 — Backward Split
Player-facing discovery

Eight mass-8 pods begin at visibly different stages of reduction. Some contain many small tokens; others contain only a few large fronts.

The final connector M.RA enters the goal and is active on the first frame because both endpoint roots happen to be tier 2. Clicking it would create only a tier-3 token in a tier-6 goal and would sever the entire A side.

The visible goal outline and the two large sides provide the intended backward inference:

tier 6 at R
requires tier 5 at R and tier 5 at A
each tier 5 requires two tier-4 pod pairs
each tier 4 requires two completed tier-3 pods

The two final halves contain equal mass but very different numbers and placements of starting tokens.

Pod instances

Selection and cursor state: null.

Pod	Root	Orientation	Variant	Starting tokens
R	(32,32)	S	P5R	5
D	(16,32)	W	P8	8
B	(32,16)	N	P7A	7
F	(16,16)	W	P8	8
A	(48,32)	E	P2	2
E	(48,16)	N	P8	8
C	(48,48)	E	P4	4
G	(32,48)	S	P6RA	6

Total starting tokens: 48.

Every pod has mass 8 and finishes as tier 3 at its r.

Goal:

goal_socket = R.r
goal_tier = 6
Macro links
Link	Direction	Bead center
M.RD	R.r–D.r → R.r	(24, 32)
M.AE	A.r–E.r → A.r	(48, 24)
M.BF	B.r–F.r → B.r	(24, 16)
M.CG	C.r–G.r → C.r	(40, 48)
M.RB	R.r–B.r → R.r	(32, 24)
M.AC	A.r–C.r → A.r	(48, 40)
M.RA	R.r–A.r → R.r	(40, 32)

Initial root tiers:

R=2, D=0, B=0, F=0, A=2, E=0, C=1, G=1

Initially active macro links:

M.BF
M.CG
M.RA

All three are premature.

Exact solving path
A.ra,
E.rd, E.bg, E.rb, E.af, E.ce, E.ac, E.ra,
M.AE,

C.rb, C.ac, C.ra,
G.bg, G.rb, G.ce, G.ac, G.ra,
M.CG,
M.AC,

R.af, R.ce, R.ac, R.ra,
D.rd, D.bg, D.rb, D.af, D.ce, D.ac, D.ra,
M.RD,

B.rd, B.bg, B.rb, B.ce, B.ac, B.ra,
F.rd, F.bg, F.rb, F.af, F.ce, F.ac, F.ra,
M.BF,
M.RB,

M.RA

Successful clicks: 47.

Critical local evidence

Pod A begins as P2: A.r and A.a are equal tier-2 tokens joined by local A.ra. At the same time, A.r is equal to R.r across the final macro link.

Thus A.r visibly touches two available links:

A.ra, which finishes its local pod;

M.RA, which would sacrifice the still-unfinished entire A half.

Clicking A.ra creates tier 3 at A.r and bars M.RA, immediately confirming the intended scale distinction.

Anti-cleanliness check

The fast A half takes 19 clicks to reach tier 5; the more decomposed R half takes 27. The already-large tokens are not merely shortcuts:

they make dangerous high-level links become available early;

they change which local endpoint must be preserved;

they create a long waiting front on one side;

their placements differ between the two final halves.

Every visible socket and every listed connector participates in the successful reduction.

Level 6 — Carry the Whole Half
Player-facing discovery

The same macro locations now form a different reduction tree. The goal has moved to A, and the cap on the final R–A connector points toward A.

In Level 5, A was eventually sacrificed into R. Here the identical geometric relationship has the opposite role: the complete R half must be transported into A.

Two compressed pods also place their prebuilt tier-2 token on opposite local sides:

F uses P5A, with its large token initially away from the pod root;

E uses P5R, with its large token already at the pod root.

The same local fusion rule therefore acts as transport in one pod and as waiting/preservation in the other.

Pod instances

Selection and cursor state: null.

Pod	Root	Orientation	Variant	Starting tokens
F	(16,16)	W	P5A	5
D	(16,32)	W	P7R	7
B	(32,16)	N	P8	8
R	(32,32)	S	P8	8
E	(48,16)	N	P5R	5
A	(48,32)	E	P8	8
G	(32,48)	S	P7A	7
C	(48,48)	E	P8	8

Total starting tokens: 56.

Goal:

goal_socket = A.r
goal_tier = 6
Macro links
Link	Direction	Bead center
M.FD	F.r–D.r → D.r	(16, 24)
M.BR	B.r–R.r → R.r	(32, 24)
M.DR	D.r–R.r → R.r	(24, 32)
M.EA	E.r–A.r → A.r	(48, 24)
M.GC	G.r–C.r → C.r	(40, 48)
M.AC	A.r–C.r → A.r	(48, 40)
M.RA	R.r–A.r → A.r	(40, 32)

Initial root tiers:

F=0, D=1, B=0, R=0, E=2, A=0, G=0, C=0

Initially active macro links:

M.BR
M.GC
M.AC
M.RA

Each is structurally premature.

Exact solving path
F.rd, F.bg, F.rb, F.ra,
D.bg, D.rb, D.af, D.ce, D.ac, D.ra,
M.FD,

B.rd, B.bg, B.rb, B.af, B.ce, B.ac, B.ra,
R.rd, R.bg, R.rb, R.af, R.ce, R.ac, R.ra,
M.BR,
M.DR,

E.af, E.ce, E.ac, E.ra,
A.rd, A.bg, A.rb, A.af, A.ce, A.ac, A.ra,
M.EA,

G.rd, G.bg, G.rb, G.ce, G.ac, G.ra,
C.rd, C.bg, C.rb, C.af, C.ce, C.ac, C.ra,
M.GC,
M.AC,

M.RA

Successful clicks: 55.

Structural roles

The left half follows:

F → D → R
B ─────→ R

The right half follows:

E → A
G → C → A

The final action transports the completed tier-5 token at R eastward into the tier-5 token at A.

Reversing only the final cap would change the required survivor and would produce a different puzzle; no endpoint has a special goal exception.

Anti-cleanliness check

This is not Level 5 with a relocated outline:

the macro connector tree is different;

the final source side is reversed;

pre-fused large tokens occur on opposite local branches;

one macro front travels through two result endpoints before reaching the final link;

several initially available macro links must be preserved for different lengths of time.

Level 7 — Full Fusion Network
Player-facing discovery

All scaffolding is removed. Every one of the 64 starting tokens is tier 0, every pod is complete, and every connector on the board is initially available.

There are no pre-fused large disks to reveal the hierarchy. The player must recognize the same reduction structure recursively:

terminal local branches must contract before shared local endpoints;

completed pods become the terminal branches of the macro network;

macro pair results must be preserved for their higher links;

the two tier-5 fronts finally meet at B–R.

The macro tree is not the one used in either preceding level.

Pod instances

Selection and cursor state: null.

Pod	Root	Orientation	Variant
F	(16,16)	W	P8
D	(16,32)	W	P8
B	(32,16)	N	P8
E	(48,16)	N	P8
R	(32,32)	W	P8
G	(32,48)	S	P8
A	(48,32)	E	P8
C	(48,48)	E	P8

Total starting tokens: 64, all tier 0.

Goal:

goal_socket = R.r
goal_tier = 6
Macro links
Link	Direction	Bead center
M.FD	F.r–D.r → F.r	(16, 24)
M.BE	B.r–E.r → B.r	(40, 16)
M.FB	F.r–B.r → B.r	(24, 16)
M.RG	R.r–G.r → R.r	(32, 40)
M.AC	A.r–C.r → A.r	(48, 40)
M.RA	R.r–A.r → R.r	(40, 32)
M.BR	B.r–R.r → R.r	(32, 24)

Every local and macro connector is initially active.

Exact solving path
F.rd, F.bg, F.rb, F.af, F.ce, F.ac, F.ra,
D.rd, D.bg, D.rb, D.af, D.ce, D.ac, D.ra,
M.FD,

B.rd, B.bg, B.rb, B.af, B.ce, B.ac, B.ra,
E.rd, E.bg, E.rb, E.af, E.ce, E.ac, E.ra,
M.BE,
M.FB,

R.rd, R.bg, R.rb, R.af, R.ce, R.ac, R.ra,
G.rd, G.bg, G.rb, G.af, G.ce, G.ac, G.ra,
M.RG,

A.rd, A.bg, A.rb, A.af, A.ce, A.ac, A.ra,
C.rd, C.bg, C.rb, C.af, C.ce, C.ac, C.ra,
M.AC,
M.RA,

M.BR

Successful clicks: 63.

Dependency structure

The upper macro half is:

D → F ┐
      ├→ B
E ─→ B┘

The lower macro half is:

G → R ┐
      ├→ R
C → A ┘

Then:

B → R(goal)

At the beginning, M.BR is visually available. Clicking it would move only two tier-0 roots into a tier-1 token and sever nearly the entire upper half. The connector is nevertheless necessary as the final action.

Anti-cleanliness check

The 63-click depth comes entirely from the required binary reduction of 64 visible mass units. There are no corridors, repeated no-ops, timers, counters, filler tokens, or unused links.

The late difficulty comes from:

eight rotated spatial subgraphs;

local and macro links sharing the same visual vocabulary;

every connector being available before its proper reduction scale;

two different macro dependency shapes meeting at one final link;

the need to preserve capped endpoints through several levels of contraction.

11. Global authored-level invariants

Every level satisfies the following:

Exact mass:
The sum of 2
tier
 over all starting tokens equals 2
goal_tier
.

Single natural objective:
A correctly sized token in the goal necessarily contains all starting mass.

No cleanup predicate:
Success is not “goal correct AND all other tokens removed” as an independent checklist. Exact mass and fusion conservation make those the same physical event, though the validator checks both explicitly.

No optional devices:
Every visible socket carries starting mass or is produced as an empty consequence of play. Every authored connector lies in the required reduction tree.

No hidden connectivity:
All endpoints, routes, caps, beads, token tiers, empty sockets, and blocked states are visible in every settled frame.

No special goal behavior:
The goal socket follows ordinary fusion rules. It can contain undersized tokens and participate in premature valid fusions.

Monotonic occupancy:
Each valid fusion reduces the occupied socket count by exactly one. Empty sockets never refill.

Determinism:
There is no randomness, simultaneous chain reaction, hidden timer, or action-order tie.

Learnable failure:
Premature endpoint consumption produces a visibly separated live branch and immediate severance feedback.

Mechanical reuse with contrasting roles:
The same capped fusion may transport a growing front, preserve a junction, sacrifice a completed branch, wait behind an unequal front, or determine which entire half survives—solely because of visible tier, direction, and topology.
