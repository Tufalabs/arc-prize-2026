Pipe Rotors

Authored level count: exactly 7 (num_levels = 7).

This is a game about rotating pipes becoming difficult because every rotation changes which pipes remain editable.

Pipe Rotors is not a conventional “rotate everything until the picture looks connected” puzzle. Fluid from the source acts as both the transported substance and the editing permission: only a pipe currently carrying fluid may be turned. A legal turn can therefore extinguish the clicked pipe, lock an entire prepared region, or transfer editing access to a different route.

The settled frame exposes all strategically relevant state. Live pipes have bright blue interiors; inaccessible pipes remain visibly oriented but hollow and gray. Empty and filled terminal vessels remain spatially visible throughout the level.

1. Core implementation contract
1.1 Action space

The environment accepts exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click at any rendered cell

Only clicks are used by the game.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic no-ops. They do not move a cursor, change selection, rotate a pipe, or consume a resource.

There is no retained cursor or selected object. Every click directly targets the module under the clicked rendered cell.

There is no energy meter, action limit, countdown, or timer. Blocked experiments are therefore free apart from the action itself, and the early distinguishing experiments never modify puzzle state.

1.2 Native geometry and 64×64 rendering

The logical play space is a fixed 7×7 orthogonal lattice, with coordinates:

row r = 0…6, increasing downward;

column c = 0…6, increasing rightward.

Each logical cell is rendered as an 8×8 pixel module:

pixel_x = 4 + 8*c … 11 + 8*c
pixel_y = 4 + 8*r … 11 + 8*r

This occupies pixels 4…59 in each dimension and leaves a quiet four-pixel margin around the board.

The renderer draws no lattice lines, cell borders, coordinate labels, or text. Empty logical cells are simply background. The coordinate maps in this document are implementation diagrams and are not shown in-game.

A pipe’s complete 8×8 module is its hit region. The player never has to click a narrow pipe stroke or exact pixel.

Clicks in the outer margin or an empty logical cell are no-ops.

1.3 Palette
Use	Color ID
Canvas and empty lattice	4
Deep hollow bore	5
Rotor collar	3
Inactive pipe wall	2
Live pipe outer wall	9
Live fluid interior	10
Source reservoir	9 and 10
Empty terminal shell	1
Empty terminal interior	5
Filled terminal interior	14
Legal rotation pulse	11
Blocked-click ring	8

Colors 6, 7, 12, 13, and 15 are unused.

1.4 Module silhouettes
Rotatable pipe

Every rotatable pipe has a rounded 6×6 rotor collar, color 3, occupying local pixels 1…6 with its four corner pixels omitted. The pipe channel is drawn above it.

The channel consists of a central 4×4 hub at local pixels x=2…5, y=2…5, plus one arm for each port:

Port	Outer arm pixels
N	x=2…5, y=0…3
E	x=4…7, y=2…5
S	x=2…5, y=4…7
W	x=0…3, y=2…5

The inner bore is the centered two-pixel strip inside those masks.

When hollow:

outer channel: color 2;

bore: color 5.

When source-connected:

outer channel: color 9;

bore: color 10.

Thus disconnected pipes preserve a large, unambiguous view of their current ports.

Source

The source is a square reservoir without a rotor collar:

outer body: local x=1…6, y=1…6, color 9;

inner body: local x=2…5, y=2…5, color 10;

fixed outlet uses the same arm geometry as a pipe port.

All authored sources have one fixed port. The source is always bright.

Terminal vessel

A terminal is a square vessel without a rotor collar:

shell: local x=1…6, y=1…6, color 1;

interior: local x=2…5, y=2…5;

fixed neck uses the corresponding port-arm geometry.

Its interior is color 5 while unpowered and color 14 while connected. Terminals are graph endpoints and never transmit fluid onward.

The different square mounts make source and terminals visibly non-rotating; only round-collared pipe modules invite rotation.

2. Pipe state and orientation

Directions are N, E, S, and W.

A connection exists across an orthogonal cell boundary only if both adjacent objects have reciprocal ports. A pipe with an east port does not connect to a neighbor that lacks a west port.

Unpaired ports and ports facing empty cells are harmless. There is no leakage rule.

2.1 Orientation states
Kind	State	Visible ports	Clockwise successor
Straight I	0	NS	state 1
Straight I	1	EW	state 0
Elbow L	0	NE	state 1
Elbow L	1	ES	state 2
Elbow L	2	SW	state 3
Elbow L	3	WN	state 0
Branch T	0	NEW	state 1
Branch T	1	NES	state 2
Branch T	2	ESW	state 3
Branch T	3	NSW	state 0

The T names list all present ports. For example, NSW is missing its east port.

A disconnected pipe retains its kind and orientation indefinitely.

3. Connectivity and click resolution

Let live(state) be the set of source and pipe nodes reachable from the source through reciprocal pipe connections. Terminals may be reached by this search but do not extend it.

Eligibility is checked against the settled pre-action state.

step(action):
    clear transient feedback

    if action is UP, DOWN, LEFT, RIGHT, or SPACE:
        return unchanged settled state

    map click pixel to logical cell

    if cell contains a rotatable pipe:
        pre_live = source_reachability(current orientations)

        if pipe not in pre_live:
            orientation does not change
            set feedback = BLOCKED
            return unchanged settled state

        rotate pipe one quarter-turn clockwise

        post_live = source_reachability(new orientations)
        set every pipe's live appearance from post_live
        set each terminal filled iff terminal is reachable
        set won iff every authored terminal is reachable
        return new settled state

    if cell contains source or terminal:
        set feedback = FIXED_MOUNT
        return unchanged settled state

    return unchanged settled state

Important consequences:

A live pipe may legally rotate into a state in which it becomes hollow.

Pipes beyond that rotation may also become hollow and locked.

A pipe supplied through a loop may remain live after losing its original feed.

A previously configured hollow region preserves its orientations and may become editable again through a different route.

There are no recursive device activations or delayed logical events. All consequences are graph connectivity.

3.1 Feedback and animation

Logical state changes occur only inside the action’s step() call. Animation is a presentation payload generated from the pre-state and committed post-state; animation frames never perform additional logical updates.

A legal rotation produces four deterministic presentation frames:

Pre-orientation with a clockwise quarter-ring in color 11.

Old port arms in color 2 and new arms overlaid in color 11.

New orientation with live bores temporarily color 9.

Authoritative settled frame: live bores color 10, hollow bores color 5, terminals updated.

A newly lit pipe has a one-frame color 11 edge pulse in frame 4. A pipe that just became hollow drains directly to colors 2/5.

A blocked hollow-pipe click produces one frame containing a color 8 ring around the pipe’s rotor collar. It does not rotate, recompute connectivity, or change terminal state.

Clicking a fixed source or terminal produces a one-frame color 1 square pulse around its mount.

Animation frames consume no actions and no budget. Input is ignored until the settled frame is shown. A harness that exposes only one observation per action should expose frame 4.

3.2 Success condition

A level succeeds exactly when, after resolving a legal rotation:

every fixed terminal vessel is reachable from the source simultaneously

Relevant:

reciprocal ports;

pipe orientations;

source reachability;

all authored terminal vessels.

Ignored:

whether every pipe is live;

whether every pipe is part of the final network;

unused or open-facing pipe ports;

how many terminals were filled earlier;

the number of clicks used.

Allowed:

the clicked pipe may be hollow in the winning state;

temporary terminal filling and emptying;

temporary disconnection of large regions;

extra live branches.

Once won = true, the state latches. All terminal interiors pulse 14–10–14, and subsequent actions are ignored until level transition.

There is no separate loss state. A player can create an unsalvageable orientation and use the environment’s standard level reset.

4. Progression overview
Level	Main discovery	Verified minimum
1. Quarter Turn	Clockwise rotation, reciprocal ports, visible terminal objective	1
2. Live Edge	Hollow pipes cannot rotate; advance the bright frontier	5
3. Prepare, Then Restore	All terminals must be filled at once; configure a branch before restoring its old feed	7
4. Borrowed Loop	Alternate feeds permit repeated turns and legal self-disconnection	28
5. Lower Handoff	A source-side branch is both working access and a terminal branch	40
6. Side-Spur Staging	Prepare an offset tail early, abandon it, and return after central handoffs	42
7. Five-Way Relay	Sacrifice an already filled vessel and choreograph several non-monotonic handoffs	51

Levels 1–3 form the clean causal curriculum. Levels 4–7 introduce no unrelated devices; their depth comes entirely from placement, orientation, loops, preserved dark state, and source-access handoffs.

The minimum depths were verified over the complete orientation-state graphs. The supplied replays contain no blocked clicks.

5. Authored levels

Lowercase letters in documentation maps denote terminals. S is the source. Uppercase letters are rotatable pipes. These letters are never rendered.

Level 1 — Quarter Turn

Teaches: reciprocal ports, direct clicking, clockwise rotation, and the visible objective.

Meaningful depth: 1 legal click.

Documentation map
      c0 c1 c2 c3
r0     .  .  S  .
r1     .  .  A  n
Player-facing discovery

The source visibly fills the elbow from above. The empty vessel is immediately to its right, and its neck faces left. There is only one rotor collar on the board.

Clicking the elbow rotates its west-facing arm clockwise to the east. The vessel fills immediately and the level completes. A counterclockwise rule would have produced a south-facing arm instead, so the visible result distinguishes rotation direction.

Initial state

Source S: (0,2), fixed port S.

Terminal n: (1,3), fixed port W, initially empty.

A: (1,2), elbow L/WN, initially live.

Initial live pipe set: {A}.

Selected object/cursor: None.

Active control: direct click on A.

Reference replay
A

Equivalent coordinate replay:

(1,2)

After the click:

A: WN → NE;

A remains connected through its north port;

A.E reciprocates with terminal n.W;

n fills;

success latches.

Reference final orientation vector, ordered [A]:

[NE]
Attribution and necessity check

There is no unrelated object and no simultaneous secondary change. Clicking the only pipe has one immediate consequence: its two visible arms rotate clockwise. The terminal cannot fill in the initial orientation or through any non-click action.

Level 2 — Live Edge

Teaches: a hollow pipe is locked; a live boundary pipe can advance editing access.

Meaningful depth: 5 legal clicks. A safe diagnostic click on any hollow pipe may precede them.

Documentation map
      c0 c1 c2 c3 c4 c5 c6
r0     .  .  .  .  .  n  G
r1     .  .  .  .  .  .  F
r2     .  .  .  .  .  .  E
r3     .  .  S  A  B  C  D
Player-facing discovery

The vessel-adjacent rotor G is visibly hollow. Clicking it gives a red blocked ring and leaves its ports unchanged. That outcome rules out “click any pipe to rotate it.”

Only A is bright. Its missing east port forms a sharp live/hollow boundary beside B. Turning A immediately lights B; subsequent useful turns advance the fluid frontier around the bend and up to the vessel.

Every legal click in the reference route causes an immediate new region to become bright. There are no decoys or simultaneous terminal requirements.

Initial state

Source S: (3,2), fixed port E.

Terminal n: (0,5), fixed port E, initially empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(3,3)	T/NSW	live
B	(3,4)	T/NSW	hollow
C	(3,5)	I/EW	hollow
D	(3,6)	L/SW	hollow
E	(2,6)	T/ESW	hollow
F	(1,6)	T/NES	hollow
G	(0,6)	T/NES	hollow

Initial live pipe set: {A}.

Selected object/cursor: None.

Reference replay
A, B, D, E, G

Coordinate replay:

(3,3), (3,4), (3,6), (2,6), (0,6)

Resolution:

A: NSW → NEW; B lights.

B: NSW → NEW; passive straight C and elbow D light.

D: SW → WN; E lights.

E: ESW → NSW; already aligned F and G light.

G: NES → ESW; its west port meets terminal n.E.

Reference final orientations [A,B,C,D,E,F,G]:

[NEW, NEW, EW, WN, NSW, NES, ESW]
Attribution and necessity check

The safe blocked click on G changes nothing except the red ring. The nearby bright/hollow boundary then singles out A as the productive experiment.

C and F are not optional decorations even though the reference route does not rotate them. They are the only reciprocal conduits across their respective gaps; removing either breaks the source-to-terminal route.

Level 3 — Prepare, Then Restore

Teaches: terminal filling is simultaneous, and a downstream rotor should be configured while a temporary branch still supplies it.

Meaningful depth: 7 legal clicks.

Documentation map
      c0 c1 c2
r0     .  n  S
r1     A  B  C
r2     D  E  F
r3     b  .  .
Player-facing discovery

The upper terminal n begins filled, giving positive evidence that filled green vessels are the objective. The lower terminal’s adjacent elbow D is hollow, so directly clicking it is blocked.

E is the bright lower frontier. Rotating it opens the lower-right route. Turning B then sacrifices the already filled upper vessel but illuminates the entire lower loop, including D.

The lower vessel first fills after the second click on D, while the upper vessel remains empty. The level does not complete. This is the distinguishing evidence that both vessels must be supplied in the same settled state.

The remaining clicks preserve the lower feed while restoring B’s north port.

Initial state

Source S: (0,2), fixed port S.

Terminal n: (0,1), fixed port S, initially filled.

Terminal b: (3,0), fixed port N, initially empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(1,0)	T/ESW	hollow
B	(1,1)	T/NES	live
C	(1,2)	T/NEW	live
D	(2,0)	L/WN	hollow
E	(2,1)	T/NSW	live
F	(2,2)	T/ESW	hollow

Initial live pipe set: {B,C,E}.

Initial terminal state: n=filled, b=empty.

Selected object/cursor: None.

Reference replay
E, B, D, D, F, C, B

Coordinate replay:

(2,1), (1,1), (2,0), (2,0), (2,2), (1,2), (1,1)

Key consequences:

E admits fluid to F.

B opens west and empties n; A and D become live.

D: WN → NE; the east feed is prepared.

D: NE → ES; it remains live from the east and fills b.

F changes how the right side will feed the loop.

C hands source access from the upper edge to the lower-right route.

B: ESW → NSW; its north port restores n without losing b.

Reference final orientations [A,B,C,D,E,F]:

[ESW, NSW, NES, ES, NEW, NSW]
Attribution and necessity check

The new objective property cannot be mistaken for an older rule: at step 4, one vessel is visibly full and the other visibly empty, and the level remains active.

A, although never clicked, is necessary. It carries the temporary west-side feed that keeps D editable through two clockwise turns. The second turn of D would not be legal under a simple one-way frontier strategy.

Level 4 — Borrowed Loop

Teaches: a loop is working access, not merely part of the final plumbing. A live pipe can be legally turned into darkness and later recovered from another side.

Verified minimum: 28 legal clicks.

Documentation map
      c0 c1 c2 c3 c4
r0     .  .  n  .  .
r1     .  A  B  C  .
r2     w  D  E  F  e
r3     .  G  H  I  .
r4     .  .  S  b  .
Player-facing discovery

The source initially illuminates only the center, center-right, and lower-center rotors. Four empty vessels surround the network.

The first few turns reach the upper-left loop. The upper elbow B then remains or becomes bright through different neighbors as its original feed is removed. This is the first forced observation that “where the pipe was originally powered from” is not a persistent property: only current reciprocal reachability matters.

At step 10, a legal turn leaves only H bright. This is not failure; H is the deliberately preserved source-side foothold. Turning it re-enters the network through the lower perimeter.

The center E is essential working access but is intentionally hollow in the winning configuration.

Initial state

Source S: (4,2), fixed port N.

Terminals:

n: (0,2), port S;

w: (2,0), port E;

e: (2,4), port W;

b: (4,3), port N.

All four terminals initially empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(1,1)	T/NEW	hollow
B	(1,2)	L/ES	hollow
C	(1,3)	T/NEW	hollow
D	(2,1)	L/WN	hollow
E	(2,2)	T/ESW	live
F	(2,3)	T/NEW	live
G	(3,1)	T/NSW	hollow
H	(3,2)	T/NES	live
I	(3,3)	T/NES	hollow

Initial live pipe set: {E,F,H}.

Selected object/cursor: None.

Reference replay

The following sequence is an exact 28-click solution:

E, C, B, A, D, B, B, D, G, E,
H, C, B, F, F, B, F, I,
D, D, I, D, B, B, D, F, F, D

Coordinate groups:

1–10:
(2,2), (1,3), (1,2), (1,1), (2,1),
(1,2), (1,2), (2,1), (3,1), (2,2)

11–18:
(3,2), (1,3), (1,2), (2,3),
(2,3), (1,2), (2,3), (3,3)

19–28:
(2,1), (2,1), (3,3), (2,1), (1,2),
(1,2), (2,1), (2,3), (2,3), (2,1)

Useful settled checkpoints:

After click	Filled terminals
4	w
6	n
10	none; only H remains live
11	e
14	e, b
21	w, b
23	n, b
26	n, e, b
28	n, w, e, b

Reference final orientations [A…I]:

[NES, NE, ESW, SW, NEW, NES, NEW, ESW, NSW]

In that reference final state, A and E are hollow. This is valid because all four vessels remain reachable through the perimeter routes.

Anti-cleanliness and necessity check

This is not four independent terminal errands. Individual vessels repeatedly fill and empty as the editable region moves around the square. The source-side bottleneck, left loop, right loop, and terminal feeds form a dependency network.

An exhaustive search over all 4^9 = 262,144 orientation states finds no solution under 28 legal rotations. Every rotor appears in the supplied minimum-length replay. The seemingly optional center rotor is required as temporary access even though it is not part of the final live network.

Level 5 — Lower Handoff

Teaches: a branch close to the source can be an access manifold early and a terminal branch late. Downstream preparation may require several turns separated by an upstream handoff.

Verified minimum: 40 legal clicks.

Documentation map
      c0 c1 c2 c3 c4
r0     .  .  n  .  .
r1     .  A  B  C  .
r2     w  D  E  F  e
r3     .  G  H  I  .
r4     .  .  J  K  .
r5     .  .  S  b  .
Player-facing discovery

The source enters through J, while K forms a side spur directly above the lower terminal. The tempting interpretation is that K is simply the last local adjustment. It is not.

The lower spur is part of the working source manifold throughout the puzzle. The central square must first be opened and repeatedly handed between its left and right feeds. Near the end, K must be clicked twice, then temporarily supplied through a changed H route, and only then clicked a third time.

At step 20, every pipe is visibly bright while every terminal is empty. This provides strong visible evidence against the mistaken objective “make all pipes connected.”

Initial state

Source S: (5,2), fixed port N.

Terminals:

n: (0,2), port S;

w: (2,0), port E;

e: (2,4), port W;

b: (5,3), port N.

All terminals initially empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(1,1)	T/NSW	hollow
B	(1,2)	L/ES	hollow
C	(1,3)	T/NEW	hollow
D	(2,1)	L/NE	hollow
E	(2,2)	L/ES	hollow
F	(2,3)	T/NEW	hollow
G	(3,1)	T/ESW	hollow
H	(3,2)	L/NE	hollow
I	(3,3)	T/NES	live
J	(4,2)	T/NES	live
K	(4,3)	T/NEW	live

Initial live pipe set: {I,J,K}.

Selected object/cursor: None.

Reference replay
I, F, H, H, G, H, I, F, F, H,
E, A, D, G, E, H, H, C, B, A,
C, E, D, D, D, B, E, H, H, F,
F, E, B, D, D, H, K, K, H, K

Coordinate groups:

1–10:
(3,3), (2,3), (3,2), (3,2), (3,1),
(3,2), (3,3), (2,3), (2,3), (3,2)

11–20:
(2,2), (1,1), (2,1), (3,1), (2,2),
(3,2), (3,2), (1,3), (1,2), (1,1)

21–30:
(1,3), (2,2), (2,1), (2,1), (2,1),
(1,2), (2,2), (3,2), (3,2), (2,3)

31–40:
(2,3), (2,2), (1,2), (2,1), (2,1),
(3,2), (4,3), (4,3), (3,2), (4,3)

Useful checkpoints:

After click	Visible consequence
1	e briefly fills
20	every pipe is live, but no terminal is filled
23	w fills
30	e fills
32	n and e are filled
37	n, e, and b are filled
39	w and b remain filled after a major handoff
40	all four terminals fill

Reference final orientations [A…K]:

[NES, NE, ESW, SW, SW, NES, NEW, SW, NSW, NES, NSW]
Anti-cleanliness and necessity check

The lower spur is not encountered only when it is needed. It is bright from the first frame, participates in source access, is left untouched through most of the solution, then requires the interleaved suffix:

K, K, H, K

The intervening H turn is structurally necessary to change which side keeps K editable.

An exhaustive reverse search over all 4^11 = 4,194,304 orientation states verifies a 40-click minimum.

Level 6 — Side-Spur Staging

Teaches: an offset tail can be prepared early, become irrelevant while dark, and later be recovered without undoing three already supplied central terminals.

Verified minimum: 42 legal clicks.

Documentation map
      c0 c1 c2 c3 c4 c5
r0     .  .  n  .  .  .
r1     .  A  B  C  .  .
r2     w  D  E  F  e  .
r3     .  G  H  I  .  .
r4     .  .  J  K  L  t
r5     .  .  S  .  .  .
Player-facing discovery

The lower-right tail is visibly offset from the central 3×3 region. K is clicked near the beginning, long before terminal t can be preserved. Most of the puzzle then happens back in the central loops.

The central terminal states are highly non-monotonic: the same left and top branches repeatedly alternate between being working access and being supplied outputs. After three central terminals are finally stable, the player returns to the already prepared tail and completes it with K, L.

The tail is therefore neither a final corridor nor a separate errand. Its early orientation affects whether it can be re-entered after the central handoffs.

Initial state

Source S: (5,2), fixed port N.

Terminals:

n: (0,2), port S;

w: (2,0), port E;

e: (2,4), port W;

t: (4,5), port W.

All terminals initially empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(1,1)	T/NSW	hollow
B	(1,2)	L/SW	hollow
C	(1,3)	T/NES	hollow
D	(2,1)	T/ESW	hollow
E	(2,2)	L/NE	hollow
F	(2,3)	L/ES	hollow
G	(3,1)	T/ESW	live
H	(3,2)	T/NSW	live
I	(3,3)	L/ES	hollow
J	(4,2)	T/NES	live
K	(4,3)	T/ESW	live
L	(4,4)	T/NSW	live

Initial live pipe set: {G,H,J,K,L}.

Selected object/cursor: None.

Reference replay
G, H, K, I, H, H, I, F, E, E,
H, G, D, A, D, D, E, A, E, E,
B, B, D, I, I, F, C, B, E, E,
F, F, B, B, D, H, I, H, H, I,
K, L

Coordinate groups:

1–10:
(3,1), (3,2), (4,3), (3,3), (3,2),
(3,2), (3,3), (2,3), (2,2), (2,2)

11–20:
(3,2), (3,1), (2,1), (1,1), (2,1),
(2,1), (2,2), (1,1), (2,2), (2,2)

21–30:
(1,2), (1,2), (2,1), (3,3), (3,3),
(2,3), (1,3), (1,2), (2,2), (2,2)

31–40:
(2,3), (2,3), (1,2), (1,2), (2,1),
(3,2), (3,3), (3,2), (3,2), (3,3)

41–42:
(4,3), (4,4)

Useful checkpoints:

After click	Filled terminals
7	e
11	w
21	n
27	n, w
31	w, e
35	n, w
40	n, w, e
42	n, w, e, t

Reference final orientations [A…L]:

[NES, WN, ESW, NSW, WN, ES, NEW, ESW, WN, NES, NEW, NEW]
Anti-cleanliness and necessity check

The right tail is visible and partly live from the start, but completing it immediately is not the puzzle. K is staged at step 3 and revisited at step 41 after the central network has undergone several complete feed reversals.

The central region is not traversed in a simple object order. E, H, and I each alternate between bridge, alternate supply, and deliberate cut.

An exhaustive search over 4^12 = 16,777,216 orientation states verifies that 42 legal clicks are necessary.

Level 7 — Five-Way Relay

Teaches and combines: deliberate sacrifice of an already solved-looking branch, dark-state preservation, alternate feeds, repeated source handoffs, and simultaneous restoration of five vessels.

Verified minimum: 51 legal clicks.

Documentation map
      c0 c1 c2 c3 c4
r0     .  .  n  .  .
r1     .  A  B  C  .
r2     w  D  E  F  e
r3     .  G  H  I  .
r4     .  J  K  L  .
r5     .  l  S  r  .
Player-facing discovery

The lower-left terminal l begins filled. This is not a free completed subgoal: the first reference turn deliberately empties it and transfers working access toward the middle.

The source sits between two bottom vessels, but neither bottom branch is an independent final adjustment. The left branch is repeatedly used as a feed into the central network. The right branch stays prepared while dark and is restored only after the three upper vessels are stable.

The first click gives an especially strong piece of causal evidence: a legal rotation of bright J immediately extinguishes the already filled lower-left terminal. Nothing hidden has happened—the visible port moved away from the vessel, and the network’s editing frontier changed with it.

Initial state

Source S: (5,2), fixed port N.

Terminals:

n: (0,2), port S;

w: (2,0), port E;

e: (2,4), port W;

l: (5,1), port N;

r: (5,3), port N.

Initial terminal state:

l=filled;

n,w,e,r=empty.

Pipes:

ID	Coordinate	Kind/orientation	Initial access
A	(1,1)	T/NSW	hollow
B	(1,2)	L/ES	hollow
C	(1,3)	T/NEW	hollow
D	(2,1)	T/ESW	hollow
E	(2,2)	L/ES	hollow
F	(2,3)	L/WN	hollow
G	(3,1)	L/ES	hollow
H	(3,2)	T/NES	hollow
I	(3,3)	T/NEW	hollow
J	(4,1)	T/ESW	live
K	(4,2)	T/ESW	live
L	(4,3)	T/NEW	live

Initial live pipe set: {J,K,L}.

Selected object/cursor: None.

Reference replay
J, K, E, H, H, J, G, G, G, D,
A, D, D, E, C, B, A, E, C, E,
B, B, B, D, E, E, F, F, E, G,
H, F, E, H, G, G, I, I, E, G,
E, H, I, F, F, F, B, B, D, J, L

Coordinate groups:

1–10:
(4,1), (4,2), (2,2), (3,2), (3,2),
(4,1), (3,1), (3,1), (3,1), (2,1)

11–20:
(1,1), (2,1), (2,1), (2,2), (1,3),
(1,2), (1,1), (2,2), (1,3), (2,2)

21–30:
(1,2), (1,2), (1,2), (2,1), (2,2),
(2,2), (2,3), (2,3), (2,2), (3,1)

31–40:
(3,2), (2,3), (2,2), (3,2), (3,1),
(3,1), (3,3), (3,3), (2,2), (3,1)

41–51:
(2,2), (3,2), (3,3), (2,3), (2,3),
(2,3), (1,2), (1,2), (2,1), (4,1), (4,3)

Key settled checkpoints:

After click	Filled terminals
Initial	l
1	none
3	w
5	w, l
6	w
21	n
27	w, e
39	w
45	w, e
49	n, w, e
50	n, w, e, l
51	n, w, e, l, r

Reference final orientations [A…L]:

[NES, WN, ESW, NSW, WN, ES, NE, ESW, NSW, NES, NSW, NES]
Anti-cleanliness and necessity check

The initially filled terminal prevents the level from reading as a clean expansion exercise. Productive play begins by making visible progress appear worse.

All twelve rotors are used. The two bottom vessels are not independent final toggles:

J is clicked at both the beginning and step 50;

K changes the initial source distribution;

L is preserved for the final reciprocal connection;

the upper three terminals must already survive before the final two clicks.

The solution repeatedly returns to B, D, E, F, G, H, and I; the layout cannot be reduced to a corridor or an ordered checklist.

An exhaustive search over all 4^12 = 16,777,216 orientation states verifies that no solution exists below 51 legal clicks.

6. Implementation invariants and test assertions

An implementation should satisfy all of the following:

Eligibility uses pre-rotation connectivity.
A pipe that is live before a click rotates even when the resulting orientation makes it hollow.

Blocked clicks are perfectly inert.
Orientation vector, live set, terminals, and win state remain byte-for-byte unchanged.

Disconnected orientation persists.
Becoming hollow never resets, normalizes, or randomizes a pipe.

Connectivity is reciprocal.
A one-sided facing port is not an edge.

Terminals are endpoints.
A terminal cannot bridge two pipe regions or relay source access.

All terminals are evaluated post-rotation and simultaneously.

Extra live pipes are allowed.
Conversely, a winning state may contain hollow pipes, including pipes used earlier in the solution.

There is no hidden phase, cursor, pressure value, flow direction, or pipe ownership.
The settled image is a complete observation of gameplay state, except for the already-latched won flag represented by the completion pulse.

No action other than clicking a live pipe can change orientation.

Exactly seven level initializers exist, in the order specified above.
