Mirror Run

Exact authored level count: 7

This is a game about rotating one mirror becoming difficult because the rotation can change the entire downstream route and the side from which every later mirror is struck.

Mirror Run is a click-only deterministic beam-routing game. A single source continuously emits one beam. Every valid click rotates exactly one mirror between / and \, clears the previous trail, and recomputes the complete route from the source. The settled beam is both the current state and a visible proof of what every mirror did.

The game introduces no devices beyond the source, mirrors, receiver, and absorbing outer frame. Depth comes from spatial arrangement: later boards fold wrong routes back through familiar mirrors from different directions, so a mirror that was useful on one approach can become the reason the beam escapes on another.

1. Player-facing rules
Objective

Make the beam enter the ring-shaped receiver.

The receiver accepts the beam from any cardinal direction. It turns green immediately when entered, the beam remains visible, and the level completes.

Controls

Click any visibly raised mirror module: rotate it by 90°, alternating between / and \.

Click any non-mirror cell: no effect.

UP, DOWN, LEFT, RIGHT, SPACE: no effect.

Every mirror is clickable whether or not the current beam reaches it.

There is no click limit, energy, timer, irreversible state, or penalty for experimentation.

Clicking the same mirror twice restores exactly the previous orientation and beam route.

Clicking belongs to the core mechanic rather than acting as an auxiliary interface: changing a mirror is the only way to change the puzzle state.

Reflection rule

Directions below are the beam’s direction of travel when it enters the mirror.

Incoming direction	/ exits	\ exits
RIGHT	UP	DOWN
LEFT	DOWN	UP
UP	RIGHT	LEFT
DOWN	LEFT	RIGHT

Mirrors never rotate when struck by the beam. Only clicks rotate them.

Beam termination

The beam continues until exactly one of these happens:

It enters the receiver: success.

It leaves the 13×13 logical board: failure, shown by a persistent impact glow on the outer frame.

It strikes a mirror: it turns and continues.

Empty-cell crossings do not interact. A horizontal and vertical segment may cross in empty space without splitting, blocking, or changing the beam.

2. Epistemic curriculum

The first three levels each isolate one causal question.

Level 1: Is the visible diagonal rotatable?

The initial settled beam already demonstrates that a mirror reflects. The only unresolved question is whether the conspicuously raised mirror can be changed. One click changes its diagonal and sends the beam directly into the receiver.

Level 2: Does a recomputed beam continue through more than one mirror?

Clicking the first mirror preserves the source-to-first-mirror prefix but replaces the entire suffix. The new trail reaches a second mirror and misses elsewhere. Clicking that second mirror completes the route.

Level 3: Does the same face always send the beam toward the same screen direction?

No. Only the upstream mirror is clicked. A downstream / mirror remains visibly unchanged, but the new route strikes it from below rather than from the left, so its exit changes from UP to RIGHT. This observation cannot be explained by mirror rotation because that mirror did not rotate.

After Level 3, the complete operational system is established. Levels 4–7 introduce no hidden rules and no new object categories. They develop consequences of the same reflection table:

preparing currently dark mirrors;

reasoning backward from the receiver;

distinguishing a wrong orientation from a wrong approach;

recognizing paths that traverse familiar mirrors in reverse order;

delaying an upstream commitment until a long suffix is prepared.

3. Logical geometry
Coordinate system

Every level uses a 13×13 logical board.

Coordinates are (x, y).

x = 0…12, increasing rightward.

y = 0…12, increasing downward.

Beam directions are UP, DOWN, LEFT, and RIGHT.

Source, receiver, and mirror coordinates are distinct.

Empty cells are traversable.

The outer frame lies immediately outside logical coordinates -1 and 13.

Documentation labels such as A, B, and C identify mirrors in this specification only. They are never rendered.

Beam stepping

Starting at the source coordinate:

Set direction = source_direction.

Advance one logical cell in direction.

If the new coordinate is outside 0…12, terminate at the frame.

If it is the receiver coordinate, terminate successfully.

If it contains a mirror, apply the reflection table and continue from that mirror’s center.

Otherwise continue in the same direction.

The source is used only as the initial origin. Exhaustive validation confirms that no authored orientation assignment routes the beam back through the source cell.

4. Rendering contract
Canvas

Render size: 64×64 pixels.

Canvas background: color 5.

Board interior origin: pixel (6,6).

Logical cell size: 4×4 pixels.

Logical cell (x,y) occupies:

pixel x = 6 + 4x … 9 + 4x
pixel y = 6 + 4y … 9 + 4y

Board interior therefore occupies pixels 6…57 on each axis.

The only board border is the one-pixel outer frame:

left x=5

right x=58

top y=5

bottom y=58

Do not draw internal grid lines or internal region borders.

Empty board cells use color 4.

Restricted palette
Role	Color ID
White mirror face and source arrow	0
Raised mirror housing	1
Outer frame	2
Old-trail fade	3
Board field	4
Canvas margin	5
Frame impact	8
Inactive receiver ring	10
Beam	11
Source housing / impact center	12
Active receiver	14

Other colors remain unused.

Mirrors

A mirror occupies its entire 4×4 logical cell, making the hit region module-sized and obvious. Fill is color 1; the white diagonal is color 0.

/:

1 1 1 0
1 1 0 1
1 0 1 1
0 1 1 1

\:

0 1 1 1
1 0 1 1
1 1 0 1
1 1 1 0

The complete 4×4 rectangle is clickable, including its diagonal and housing.

Source

The source cell is filled with color 12.

For a RIGHT-facing source, draw a color-0 shaft at local pixels (0,1), (1,1), and (2,1), with arrowhead pixels (2,0), (3,1), and (2,2). Rotate this pattern for the other three source directions.

The source is not clickable.

Receiver

Inactive:

10 10 10 10
10  4  4 10
10  4  4 10
10 10 10 10

Active:

14 14 14 14
14 11 11 14
14 11 11 14
14 14 14 14

The receiver is not clickable.

Beam

Use color 11.

The visual anchor of logical cell (x,y) is:

(7 + 4x, 7 + 4y)

Draw one-pixel-wide horizontal or vertical lines between successive anchors. Draw the beam before mirror, source, and receiver modules so the modules remain readable. The trail should visibly meet each module edge.

At an empty-cell crossing, draw an ordinary color-11 plus-shaped intersection. It has no gameplay effect.

Frame impact

A failed route leaves a persistent impact marker where it meets the outer frame.

Left or right impact: three vertical color-8 pixels centered on the beam coordinate, with the center pixel color 12.

Top or bottom impact: three horizontal color-8 pixels, with the center pixel color 12.

The impact remains visible in the settled frame until another valid mirror click recomputes the beam.

5. Animation and feedback

All logical changes occur atomically inside the step() that receives the action. Animation is render-only and never consumes additional actions.

Valid mirror click

When a mirror is clicked:

Cache the old trace for rendering.

Toggle the clicked mirror’s logical orientation.

Recompute the complete new trace.

Set completed = true if and only if the new trace enters the receiver.

Begin the following render sequence:

one frame with the old trail faded to color 3 and the clicked mirror showing both diagonals as a white X;

one frame showing the new diagonal with no trail;

reveal the new trail in source-to-terminal order.

For a trace with L straight legs, reveal ceil(L / 6) legs per render frame, so beam regrowth takes at most six frames. This chronological reveal is important when a wrong route revisits a mirror or crosses itself.

The orientation and full beam result are already committed before the first animation frame. Inputs received while the animation is playing are discarded rather than queued.

Clicking an unlit mirror

The mirror visibly rotates. The existing beam briefly fades and redraws identically if the rotated mirror remains outside the route. This is deliberate evidence that:

the click was accepted;

the new orientation persists;

the mirror currently has no causal effect on the beam.

Failure

The new frame impact pulses for two render frames and then remains steady.

Completion

The receiver changes to its active rendering for three frames of glow, then remains steady. The level-success signal is set immediately during recomputation, not after the glow animation.

Invalid actions

Keyboard actions and clicks on non-mirror cells produce no animation and no state change.

6. State-machine contract
Persistent state
level_index: integer in 0…6
mirror_orientation: map[mirror_id] -> SLASH | BACKSLASH
beam_legs: ordered list of axis-aligned logical segments
beam_hits: ordered list of (mirror_id, incoming_direction, outgoing_direction)
terminal_kind: FRAME | RECEIVER
terminal_location: receiver coordinate or outside-board impact coordinate
completed: boolean
Transient render state
last_clicked_mirror: mirror_id | null
old_beam_legs: ordered list, used only during animation
animation_phase: SETTLED | ROTATION_FLASH | BEAM_REVEAL | IMPACT | COMPLETE
animation_frame: nonnegative integer

There is no persistent selected object and no keyboard-controlled cursor.

Every level initializes with:

last_clicked_mirror = null
animation_phase = SETTLED
animation_frame = 0
completed = false

The initial beam is computed from the authored initial orientations before player input.

Click hit testing

For a canvas click (px,py):

if 6 <= px <= 57 and 6 <= py <= 57:
    x = floor((px - 6) / 4)
    y = floor((py - 6) / 4)
else:
    no logical cell

Toggle a mirror only if a mirror exists at (x,y).

Update order
step(action):
    if completed:
        return unchanged state

    if animation is active:
        discard action
        return unchanged logical state

    if action is UP, DOWN, LEFT, RIGHT, or SPACE:
        return unchanged state

    if action is a click not landing on a mirror:
        return unchanged state

    old_beam_legs = beam_legs
    toggle clicked mirror
    beam_legs, beam_hits, terminal_kind, terminal_location = recompute()
    completed = (terminal_kind == RECEIVER)
    begin render-only animation
Trace pseudocode
recompute():
    position = source_position
    direction = source_direction
    seen_entries = empty set
    trace = empty ordered list

    loop:
        next = position + delta(direction)

        if next is outside the 13×13 board:
            append segment ending at the outer frame
            return FRAME

        append next cell to current beam segment

        if next == receiver_position:
            return RECEIVER

        if next contains mirror M:
            key = (M, direction)
            assert key not in seen_entries
            add key to seen_entries

            direction = reflect(M.orientation, direction)
            position = next
            begin new beam leg
            continue

        position = next

A repeated (mirror, incoming_direction) would imply an infinite deterministic loop. It is an authoring error. The released seven levels have been exhaustively checked and contain no such state under any possible mirror assignment.

Win validation

The validator checks only:

terminal_kind == RECEIVER

It does not compare the board against a hidden target orientation vector. Orientations of mirrors outside the successful route are ignored. This matters in Level 3, where the upper demonstration mirror can have either orientation once the beam takes the lower route.

The receiver terminates the trace immediately. Anything geometrically beyond it is irrelevant.

7. Solution-depth policy

A binary, reversible, globally clickable mirror puzzle has an important structural property: once the winning configuration is known, each incorrectly oriented mirror needs at most one click. Forcing 25–50 optimal clicks would require dozens of relevant mirrors or pointless double rotations, both of which would undermine readability and create mechanical padding.

Mirror Run therefore uses two distinct depth measures:

Optimal toggle depth: number of mirror rotations required from the authored initial state.

Causal trace depth: number of mirror encounters that must be predicted in the successful beam.

Later levels are intended to produce many hypothesis-testing clicks during blind play, but do not force repetitive input after the configuration has been understood.

Level	Mirrors	Verified optimal clicks	Winning mirror encounters	First-solve experimental target
1	1	1	1	1–3 clicks
2	2	2	2	2–6 clicks
3	5	1	4	2–8 clicks
4	6	5	6	8–20 clicks
5	8	5	8	12–30 clicks
6	10	6	10	18–40 clicks
7	12	8	12	25–60 clicks

The experimental targets are authoring goals, not enforced budgets.

8. Authored levels
Level 1 — First Turn
Player-facing discovery

The beam leaves the source, strikes the only raised mirror, turns downward, and hits the bottom frame. The receiver lies directly above that mirror. There is one obvious local experiment: click the mirror.

The initial trail already proves that the diagonal reflects the beam. The click only needs to reveal that the mirror rotates and the route is recomputed.

Initial state

Source: (0,6), direction RIGHT

Receiver: (5,2)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(5,6)	\	/

Initial trace:

source
→ A at (5,6), RIGHT entering \ becomes DOWN
→ bottom frame at outside coordinate (5,13)
Intended solution

Click mirror A at (5,6).

A changes from \ to /.

The old downward suffix disappears.

The new trace is:

source
→ A, RIGHT entering / becomes UP
→ receiver at (5,2)

The receiver activates and the level completes.

Attribution and necessity check

There is only one clickable object.

The receiver’s position gives positive evidence for the desired consequence.

No unrelated state changes occur.

The level is not solvable without rotating A.

Random clicking on empty space cannot progress the puzzle.

Verified optimal depth: 1 click.

Level 2 — Second Bounce
Player-facing discovery

The initial beam again fails at the first mirror. Rotating it does not solve the level; instead, it reveals a second reflection and a new frame impact. The unchanged source-to-A prefix and completely new downstream suffix make recomputation visually explicit.

Initial state

Source: (0,10), direction RIGHT

Receiver: (0,4)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(4,10)	\	/
B	(4,4)	/	\

Initial trace:

source
→ A, RIGHT entering \ becomes DOWN
→ bottom frame at (4,13)
Intended solution

Click A at (4,10).

A becomes /.

New trace:

source
→ A, RIGHT entering / becomes UP
→ B, UP entering / becomes RIGHT
→ right frame at (13,4)

Click B at (4,4).

B becomes \.

New trace:

source
→ A, RIGHT entering / becomes UP
→ B, UP entering \ becomes LEFT
→ receiver at (0,4)

The two clicks may legally be performed in the opposite order because unlit mirrors are globally clickable. The lit-first sequence is the intended blind-play experiment.

Attribution and necessity check

The first click cannot accidentally solve the level.

Its distinctive outcome proves that the beam continues after a reflection.

B is visibly struck before the second click, so there is no arbitrary click hunt.

Both orientations are fixed in the only winning assignment.

Verified optimal depth: 2 clicks.

Level 3 — Same Face, New Side
Player-facing discovery

The initial beam reaches mirror C from the left. C is a /, so the beam exits upward and misses.

Clicking only upstream mirror A reroutes the beam around the bottom of the board. The new route reaches the same unchanged / mirror C from below. C now sends the beam rightward into the receiver.

Because C never rotates, the observation forces the conclusion that mirror behavior depends on entry side.

Initial state

Source: (0,8), direction RIGHT

Receiver: (11,3)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Required at success
A	(4,8)	/	\
B	(4,3)	/	Either orientation
C	(9,3)	/	/
D	(4,11)	\	\
E	(9,11)	/	/

Initial trace:

source
→ A, RIGHT entering / becomes UP
→ B, UP entering / becomes RIGHT
→ C, RIGHT entering / becomes UP
→ top frame at (9,-1)
Intended solution

Click A at (4,8), changing it from / to \.

The complete new trace is:

source
→ A, RIGHT entering \ becomes DOWN
→ D, DOWN entering \ becomes RIGHT
→ E, RIGHT entering / becomes UP
→ C, UP entering / becomes RIGHT
→ receiver at (11,3)

C visibly remains / throughout. Its old exit was UP; its new exit is RIGHT solely because its incoming direction changed from RIGHT to UP.

Attribution and necessity check

Only A changes orientation.

C is present in both the initial and successful traces.

The result cannot be explained by “a slash always points the beam upward” or “clicking A also rotates downstream mirrors.”

B is not a decoy: it is necessary to create the clean initial approach to C from the left. Once the lower route is chosen, B is unlit and its orientation is correctly ignored by the validator.

D and E are necessary to create the contrasting approach from below.

Verified optimal depth: 1 click.

Level 4 — Dark Suffix
Player-facing discovery

The initial beam takes a shortcut from A directly to E. E is already in its eventual correct orientation, but because it is struck from the left rather than from below, it sends the beam upward into the frame.

This is the first board where “the last lit mirror must be wrong” is an unreliable heuristic. The receiver instead suggests tracing the required final bends backward and orienting currently dark mirrors before changing A.

Preparation is not an order-dependent rule: the clicks commute. It is the low-noise solving insight.

Initial state

Source: (5,0), direction DOWN

Receiver: (11,11)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(5,2)	\	/
B	(3,2)	\	/
C	(3,8)	/	\
D	(9,8)	\	/
E	(9,2)	/	/
F	(11,2)	/	\

Initial trace:

source
→ A, DOWN entering \ becomes RIGHT
→ E, RIGHT entering / becomes UP
→ top frame at (9,-1)

A productive diagnostic click is E. Rotating E sends the beam downward into D and then to the right frame, showing that merely changing the currently lit mirror does not construct a route to the receiver. The click can be reversed without penalty.

Canonical solution

Prepare the suffix from the receiver backward while A continues to hold the initial route:

Click F at (11,2): / → \.

Click D at (9,8): \ → /.

Click C at (3,8): / → \.

Click B at (3,2): \ → /.

Each mirror visibly changes, but the beam remains on the initial A → E → top frame route.

Click A at (5,2): \ → /.

The new beam is:

source
→ A, DOWN entering / becomes LEFT
→ B, LEFT entering / becomes DOWN
→ C, DOWN entering \ becomes RIGHT
→ D, RIGHT entering / becomes UP
→ E, UP entering / becomes RIGHT
→ F, RIGHT entering \ becomes DOWN
→ receiver at (11,11)
Anti-degenerate and necessity check

E begins in its correct final orientation but produces a failure under the wrong approach.

Clicking A immediately exposes B’s wrong orientation, so a forward trial strategy remains informative but noisier than backward preparation.

Every mirror is encountered by the only winning trace.

Flipping any one winning orientation breaks success.

The irregular route cannot be replaced by a linear corridor without removing the approach trap at E.

Verified optimal depth: 5 clicks.
Winning trace depth: 6 mirror encounters.

Level 5 — Reverse Fold
Player-facing discovery

The initial route does not merely stop at the first wrong mirror. It enters the middle of the intended route from the wrong side, traverses several correct mirrors in reverse spatial order, returns to A from above, and then escapes through the right frame.

This gives familiar mirrors contrasting roles without changing their rules:

B, C, and D are correct throughout.

In the initial trace, they carry the beam backward through the fold.

In the successful trace, they carry it forward toward the receiver.

A is encountered twice in the initial trace, with different incoming directions.

Initial state

Source: (0,7), direction RIGHT

Receiver: (12,9)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(7,7)	/	\
B	(7,9)	/	/
C	(5,9)	\	\
D	(5,5)	/	/
E	(7,5)	\	/
F	(7,2)	\	/
G	(10,2)	/	\
H	(10,9)	/	\

Initial trace:

source
→ A, RIGHT entering / becomes UP
→ E, UP entering \ becomes LEFT
→ D, LEFT entering / becomes DOWN
→ C, DOWN entering \ becomes RIGHT
→ B, RIGHT entering / becomes UP
→ A, UP entering / becomes RIGHT
→ right frame at (13,7)

The chronological beam animation is important here: it makes clear that A is struck once from the left and later from below.

Canonical solution

Prepare the currently unlit final suffix:

Click H at (10,9): / → \.

Click G at (10,2): / → \.

Click F at (7,2): \ → /.

The initial six-hit route remains unchanged.

Click E at (7,5): \ → /.

Because A is still /, E is currently approached from below. The shortened temporary route becomes:

source → A → E → right frame at (13,5)

This is useful evidence: E’s correct final orientation is still not sufficient under the temporary approach.

Click A at (7,7): / → \.

Winning trace:

source
→ A, RIGHT entering \ becomes DOWN
→ B, DOWN entering / becomes LEFT
→ C, LEFT entering \ becomes UP
→ D, UP entering / becomes RIGHT
→ E, RIGHT entering / becomes UP
→ F, UP entering / becomes RIGHT
→ G, RIGHT entering \ becomes DOWN
→ H, DOWN entering \ becomes RIGHT
→ receiver at (12,9)
Anti-degenerate and necessity check

The initial objects are not encountered in eventual use order.

The wrong path physically folds through E–D–C–B and revisits A.

E has three visibly distinct contextual roles across experimentation:

initial backward turn;

temporary rightward miss;

final upward turn.

All eight mirrors are required in the only winning assignment.

The key insight is not a checklist of mirror states; it is recognizing that the current trail is using a valid-looking middle route backward.

Verified optimal depth: 5 clicks.
Winning trace depth: 8 mirror encounters.

Level 6 — Cross-Weave
Player-facing discovery

The initial beam follows a seven-hit route:

A → B → F → E → D → C → B → frame

B is encountered twice. Its first wrong turn jumps forward to F; after the beam travels backward through E, D, and C, it returns to B from below. B’s unchanged / orientation then sends the beam right rather than up.

The board therefore behaves like a small dependency network rather than a row of mirrors used in sequence.

Initial state

Source: (2,12), direction UP

Receiver: (1,6)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(2,8)	/	/
B	(6,8)	/	\
C	(6,10)	\	\
D	(11,10)	/	/
E	(11,5)	\	\
F	(6,5)	/	\
G	(6,1)	/	\
H	(3,1)	\	/
I	(3,10)	\	/
J	(1,10)	/	\

Initial trace:

source
→ A, UP entering / becomes RIGHT
→ B, RIGHT entering / becomes UP
→ F, UP entering / becomes RIGHT
→ E, RIGHT entering \ becomes DOWN
→ D, DOWN entering / becomes LEFT
→ C, LEFT entering \ becomes UP
→ B, UP entering / becomes RIGHT
→ right frame at (13,8)
Canonical solution

Prepare the receiver-side route while it is dark:

Click J at (1,10): / → \.

Click I at (3,10): \ → /.

Click H at (3,1): \ → /.

Click G at (6,1): / → \.

The initial reverse weave remains visible.

Click F at (6,5): / → \.

The current wrong approach to F now sends the beam left, collapsing the reverse weave:

source → A → B → F → left frame at (-1,5)

Click B at (6,8): / → \.

Winning trace:

source
→ A, UP entering / becomes RIGHT
→ B, RIGHT entering \ becomes DOWN
→ C, DOWN entering \ becomes RIGHT
→ D, RIGHT entering / becomes UP
→ E, UP entering \ becomes LEFT
→ F, LEFT entering \ becomes UP
→ G, UP entering \ becomes LEFT
→ H, LEFT entering / becomes DOWN
→ I, DOWN entering / becomes LEFT
→ J, LEFT entering \ becomes UP
→ receiver at (1,6)
Anti-degenerate and necessity check

B and F form one cross-linked pair: the wrong B route jumps directly to F.

C and I form another potential cross-linked pair under alternative orientations.

The initial beam revisits B from a different direction rather than merely stopping at the first error.

Clicking B first is a productive but noisy experiment: it enters the intended prefix and then fails farther downstream.

Every mirror is on the only winning trace.

The spatial weave, especially the shared x=6 and y=10 lines, is causally necessary; replacing it with a corridor would remove the alternate approaches and revisit.

Verified optimal depth: 6 clicks.
Winning trace depth: 10 mirror encounters.

Level 7 — Knotted Return
Player-facing discovery

The source and receiver both sit near the top of the board, but the successful beam must make a long asymmetric return through all twelve mirrors.

The initial beam jumps from C directly to I, then traverses H, G, and F in reverse:

A → B → C → I → H → G → F → frame

Several locally plausible mirror orientations are already correct. The central problem is to prepare a future route whose mirrors will later be struck from completely different sides.

Initial state

Source: (2,0), direction DOWN

Receiver: (8,0)

Initial interaction state: no selection, no cursor, settled animation

Mirror	Coordinate	Initial	Winning
A	(2,2)	\	\
B	(7,2)	\	\
C	(7,9)	\	/
D	(5,9)	/	\
E	(5,3)	/	\
F	(2,3)	\	/
G	(2,6)	\	\
H	(8,6)	\	\
I	(8,9)	/	\
J	(10,9)	\	/
K	(10,2)	/	\
L	(8,2)	/	\

Initial trace:

source
→ A, DOWN entering \ becomes RIGHT
→ B, RIGHT entering \ becomes DOWN
→ C, DOWN entering \ becomes RIGHT
→ I, RIGHT entering / becomes UP
→ H, UP entering \ becomes LEFT
→ G, LEFT entering \ becomes UP
→ F, UP entering \ becomes LEFT
→ left frame at (-1,3)
Canonical solution

First prepare mirrors that are not yet on the active trail:

Click L at (8,2): / → \.

Click K at (10,2): / → \.

Click J at (10,9): \ → /.

Click E at (5,3): / → \.

Click D at (5,9): / → \.

The visible beam does not change, but all five mirror rotations persist.

Click F at (2,3): \ → /.

The beam now reaches the prepared E and D before returning to still-wrong C:

A → B → C → I → H → G → F → E → D → C
→ bottom frame at (7,13)

C is encountered twice, once from above and later from the left. The second encounter is not a loop because the incoming direction differs.

Click I at (8,9): / → \.

The current shortcut from C now terminates quickly:

A → B → C → I
→ bottom frame at (8,13)

Click C at (7,9): \ → /.

This is the final upstream commitment. The complete prepared route appears:

source
→ A, DOWN entering \ becomes RIGHT
→ B, RIGHT entering \ becomes DOWN
→ C, DOWN entering / becomes LEFT
→ D, LEFT entering \ becomes UP
→ E, UP entering \ becomes LEFT
→ F, LEFT entering / becomes DOWN
→ G, DOWN entering \ becomes RIGHT
→ H, RIGHT entering \ becomes DOWN
→ I, DOWN entering \ becomes RIGHT
→ J, RIGHT entering / becomes UP
→ K, UP entering \ becomes LEFT
→ L, LEFT entering \ becomes UP
→ receiver at (8,0)
Anti-degenerate and necessity check

The initial trail reaches C, I, H, G, and F in an order different from the winning route.

Wrong C jumps directly to I.

After F changes, the trail reaches E and D before returning to C from another direction.

H and L form a second latent cross-pair: the wrong orientation at L would send the beam back down toward H.

F lies immediately below A, but its useful final role is reached from the right rather than from below.

The receiver’s proximity to the source is visually tempting, but no direct or short route can reach it.

Every one of the twelve mirrors is encountered by the only winning beam.

Flipping any one mirror from the winning state causes a physical reroute or frame impact rather than failing an abstract predicate.

Verified optimal depth: 8 clicks.
Winning trace depth: 12 mirror encounters.

9. Exhaustive authored validation

All possible orientation assignments were enumerated under the exact trace algorithm.

Level	Mirror assignments checked	Winning assignments	Infinite-loop assignments
1	2	1	0
2	4	1	0
3	32	2	0
4	64	1	0
5	256	1	0
6	1,024	1	0
7	4,096	1	0

Level 3 has two winning assignments only because mirror B is outside the successful lower route and is intentionally ignored by the receiver-contact validator. Its presence is still necessary for the clean initial comparison at mirror C.

For Levels 4–7:

every mirror lies on the unique successful trace;

every mirror’s orientation is fixed in the unique winning assignment;

no mirror is a decorative decoy;

no orientation assignment produces an infinite beam loop;

no level depends on a click budget, hidden state, text, arbitrary symbol mapping, or special-case reflection behavior.
