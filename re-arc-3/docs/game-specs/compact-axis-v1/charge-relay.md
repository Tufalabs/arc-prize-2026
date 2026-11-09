Charge Relay

This is a game about transferring charge becoming difficult because every landing changes both which contacts are open and which straight lines are live.

Authored level count: exactly 7.

The game uses UP, DOWN, LEFT, and RIGHT. SPACE and clicks are accepted by the environment but have no gameplay function. There are no clickable objects, no cursor, and no selected-object state.

The central tension is that movement and circuit configuration are the same action. Crossing a gate is evaluated using the charge type before the move, but landing immediately adopts the destination token’s type and retunes every gate on the board. Meanwhile, receptors care only about the currently visible horizontal and vertical conduction lines—not where the charge has previously traveled.

1. Player-facing rules

The board consists of orthogonally touching relay tokens. Every token has one intrinsic type:

Type	Color ID	Glyph
A	9	vertical bar
B	11	horizontal bar
C	6	X-shaped mark

Exactly one token holds the charge. It has a bright white corona.

Pressing a direction attempts to transfer the corona to the neighboring relay token in that direction. A successful transfer:

moves the corona one logical token;

changes the active charge type to the destination token’s intrinsic type;

opens every gate keyed to that new type;

closes every other keyed gate;

recalculates all visible conduction stripes;

recalculates every receptor.

There is no visited state and no stored charge in previously occupied tokens.

Live conduction

The holder produces two independent live stripes:

a horizontal stripe through the maximal uninterrupted row segment containing the holder;

a vertical stripe through the maximal uninterrupted column segment containing the holder.

Stripes do not turn corners. A token reached horizontally shows a horizontal inner stripe; a token reached vertically shows a vertical inner stripe. The holder shows both, forming a cross.

A gap or closed gate stops the corresponding stripe. Intermediate token types do not affect propagation. Only the current holder type determines which gates are open.

Gates

A gate occupies a contact between two adjacent tokens and has one key type, A, B, or C.

A gate is open exactly when:

gate.key == intrinsic_type[current_holder]

Transfer through a gate is evaluated before moving. Therefore, a gate may be crossed from an A token because it is an A gate, but immediately close behind the charge if the destination is a B or C token.

A closed gate blocks both transfer and conduction.

Sources and receptors

A source–receptor pair specifies one horizontal or vertical interval of relay tokens.

A horizontal pair is active exactly when every relay coordinate from its source endpoint to its receptor endpoint belongs to the current horizontal conduction stripe. A vertical pair uses the vertical stripe.

The holder does not need to lie inside the source–receptor interval. It may feed the interval from farther along the same uninterrupted row or column. Levels 6 and 7 require this.

Sources are always orange. Inactive receptors are hollow gray shapes. An active receptor becomes a solid green shape.

Objective

A level succeeds immediately after a settled action if every visible receptor is active in the same recomputed frame.

The validator ignores:

travel history;

previously energized tokens;

how many times a token was visited;

which sources the holder passed;

gates not relevant to the current stripes;

unused SPACE or click actions, except that they consume an action.

There is no confirmation action. Success latches as soon as the final receptor vector becomes all true.

2. Exact state machine
Persistent state
holder: relay coordinate
step_count: integer
done: boolean
truncated: boolean
last_transition: visual-feedback descriptor only

The following values are derived, not independently stored:

current_type = token_type[holder]
open_gate_set
horizontal_energized_set
vertical_energized_set
receptor_active[]

There is no cursor, selection, inventory, visited set, receptor latch, gate memory, or token charge memory.

step(action) order

For every action:

1. Save the old state for transition rendering.
2. Let departure = holder.
3. Let departure_type = token_type[departure].

4. If action is directional:
      destination = departure + direction

      a. If destination contains no token:
           movement is blocked by a gap.

      b. Else if the contact has a gate whose key != departure_type:
           movement is blocked by that gate.

      c. Otherwise:
           holder = destination.

5. If action is SPACE or a click:
      holder remains unchanged.
      Click coordinates are not stored.

6. Increment step_count exactly once.

7. Set current_type = token_type[holder].

8. Recompute every gate:
      open iff gate.key == current_type.

9. Recompute the maximal horizontal and vertical conduction
   segments containing holder.

10. Recompute every receptor from those two segments.

11. If all receptors are active:
       done = true.
    Else if step_count >= level.max_steps:
       truncated = true.

12. Construct deterministic transition frames from the old and
    committed states. Transition rendering never mutates state.

Winning takes precedence over truncation if the final allowed action completes the level.

Blocked actions

A blocked action leaves holder, current type, gate configuration, stripes, and receptor states unchanged. It still increments step_count.

For a gate block:

the blocked gate pulses with color 8;

the holder glyph pulses between its intrinsic color and color 0;

all currently legal neighboring contacts briefly show color 10.

For a gap block:

a three-pixel color-8 mark flashes on the attempted side of the holder;

legal neighboring contacts briefly show color 10.

Exploration budgets

These are episode caps, not puzzle resources. They are deliberately much larger than the reference solutions.

Level	Reference length	Maximum actions
1	3	64
2	6	96
3	5	128
4	25	320
5	32	448
6	44	640
7	56	896

All nonwinning states reachable before completion in the seven authored instances retain a route to a winning state. A mistaken legal move may cost actions, but it does not irreversibly ruin the level.

3. Rendering contract
Canvas and lattice

The observation is a 64 × 64 grid of integer color IDs.

The native relay lattice is 11 × 11, with zero-based coordinates:

row increases downward
column increases rightward

A relay token at logical (r,c) occupies:

x = 4 + 5c through 8 + 5c
y = 3 + 5r through 7 + 5r

Each relay is therefore a connected 5 × 5 shape. Adjacent logical tokens touch edge-to-edge. There are no rendered grid borders. Missing logical coordinates remain background.

The action-budget bar occupies:

x = 4..59
y = 61..62

Its track is color 3 and its remaining fraction is color 10.

Color use
Role	Color ID
Background	5
Normal relay body	4
Charged relay interior	1
Corona	0
Inactive structural detail	2 or 3
A type and A gates	9
B type and B gates	11
C type and C gates	6
Conduction stripe	10
Source badge	12
Active receptor	14
Invalid-action pulse	8

No other color IDs are required.

Type glyph masks

Coordinates below are relative (dx,dy) positions inside a token.

A glyph, color 9:
(2,1), (2,2), (2,3)

B glyph, color 11:
(1,2), (2,2), (3,2)

C glyph, color 6:
(1,1), (3,1), (2,2), (1,3), (3,3)

The glyph is always rendered, whether the token is energized or not.

Corona

The holder uses body color 1 instead of 4 and receives color-0 corner brackets:

top-left:     (0,0), (1,0), (0,1)
top-right:    (3,0), (4,0), (4,1)
bottom-left:  (0,3), (0,4), (1,4)
bottom-right: (4,3), (3,4), (4,4)

This remains unique even when terminal badges cover one corner.

Conduction stripes

A horizontally energized token receives color 10 on its center row:

(0,2), (1,2), (2,2), (3,2), (4,2)

A vertically energized token receives color 10 on its center column:

(2,0), (2,1), (2,2), (2,3), (2,4)

The type glyph is drawn over the stripe. The holder receives both masks.

Gate silhouettes

For horizontal neighbors (r,c) and (r,c+1), the gate occupies the central three pixels on both touching edge columns. For vertical neighbors, rotate this shape by 90 degrees.

A closed gate is a solid 2 × 3 keyed-color slab.

An open gate retains keyed-color rails on the two outer rows of that slab but clears its center row, producing an unmistakable split shutter. A conduction stripe passes through the cleared center when aligned with the gate.

Terminal badges

Terminal badges are overlays inside endpoint-token corners. They do not occupy logical coordinates and never block transfer.

For a horizontal pair:

the source token receives a solid orange 2 × 2 badge in its northwest corner;

the receptor token receives a northeast-corner badge.

For a vertical pair:

the source token receives a solid orange 2 × 2 badge in its northwest corner;

the receptor token receives a southwest-corner badge.

An inactive receptor is a three-pixel color-2 L shape within its assigned corner. When active, the fourth corner pixel fills and all four pixels become color 14. This is the receptor’s completed-line symbol.

The authored intervals in Level 7 are disjoint and ordered, so each source is visually paired with the next receptor along the same row or column.

Draw order
1. background
2. relay bodies
3. conduction stripes
4. intrinsic type glyphs
5. gates
6. holder corona
7. source and receptor badges
8. action-budget bar
Animation

Ordinary same-type transfers that do not change a receptor are immediate.

When a transfer changes type or changes any receptor:

the corona appears on the destination for one frame;

changed gates split open or close over two frames;

stripes grow outward from the holder by one logical token per frame, in all four directions simultaneously;

receptors update on the final stripe-growth frame.

The logical state was already committed by step(). These frames are pure visualization, accept no input, and consume no actions.

Completion adds six frames in which active stripes alternate between colors 10 and 14. Blocked attempts use three pulse frames.

4. Authored-instance notation

Maps below are exact logical layouts, not rendered text.

A  type-A token
B  type-B token
C  type-C token
.  no token

The start coordinate is listed separately, so the character at that coordinate still specifies its intrinsic type.

Gate notation:

H(r,c)=K

means a K-keyed gate between (r,c) and (r,c+1).

V(r,c)=K

means a K-keyed gate between (r,c) and (r+1,c).

Ranges are inclusive. Every adjacent token contact not listed as a gate is ungated.

Receptor notation:

H[(r,c1) -> (r,c2)]
V[(r1,c) -> (r2,c)]

The first coordinate is the source endpoint and the second is the receptor endpoint.

All levels begin with an A holder, so A gates are initially open and B/C gates are initially closed.

Level 1 — Crooked Spark
Player-facing discovery

The target B token visibly contains a horizontal source–receptor badge pair. The charge starts three transfers away on a short bent chain.

The penultimate position is directly below the target. From there the target receives a vertical stripe, but its horizontal receptor remains hollow. Moving onto the target produces the holder’s horizontal stripe, fills the receptor, and visibly changes the active type from A to B.

This teaches that the corona itself transfers between adjacent tokens and that landing adopts the destination glyph. The source–receptor badge supplies positive goal evidence without text.

Exact state
   01234567890
00 ...........
01 ...........
02 ...........
03 ...........
04 ....B......
05 ...AA......
06 ...A.......
07 ...........
08 ...........
09 ...........
10 ...........
Start:              (6,3), type A
Selection/cursor:   none
Gates:               none
Receptor:            H[(4,4) -> (4,4)]
Maximum actions:     64
Initial receptor:    inactive
Reference solution
U R U

Coordinate trace:

(6,3) A
(5,3) A
(5,4) A
(4,4) B  -> receptor active

Reference depth: 3 actions.

Attribution and necessity

There are no gates, alternate devices, or simultaneous changes. Every visible relay is on the only productive route.

The state at (5,4) is the distinguishing observation: the target is vertically striped but its horizontal receptor is still inactive. The final move therefore cannot be mistaken for “any nearby charge activates the target.”

Level 2 — The Live Rail
Player-facing discovery

A five-token horizontal rail carries the source and receptor. The charge approaches through a crooked all-A lead-in.

While the holder is below the left endpoint, the endpoint receives only a vertical stripe and the receptor remains inactive. Entering the row makes a horizontal stripe appear across all five rail tokens at once.

This isolates the new rule that conduction extends through an entire uninterrupted straight run and does not turn.

Exact state
   01234567890
00 ...........
01 ...........
02 ...........
03 ....AAAAA..
04 ....A......
05 ...AA......
06 ..AA.......
07 ..A........
08 ...........
09 ...........
10 ...........
Start:              (7,2), type A
Selection/cursor:   none
Gates:               none
Receptor:            H[(3,4) -> (3,8)]
Maximum actions:     96
Initial receptor:    inactive
Reference solution
U R U R U U

Coordinate trace:

(7,2)
(6,2)
(6,3)
(5,3)
(5,4)
(4,4)
(3,4) -> full horizontal rail and receptor active

Reference depth: 6 actions.

Any holder coordinate on (3,4..8) would satisfy the receptor, but (3,4) is the only rail coordinate reachable before the level completes.

Attribution and necessity

All tokens have the same type, and there are no gates. The only new causal event on the final action is the horizontal stripe expanding through the rail.

A holder-only interpretation fails because the holder occupies only (3,4), while the receptor at (3,8) activates through the visibly continuous stripe.

Level 3 — Retune on Landing
Player-facing discovery

The known horizontal receptor rail is visibly broken by a closed B gate. An A gate immediately before the B token is open.

The charge approaches as type A. It crosses the A gate, lands on B, and causes three simultaneous but causally ordered effects:

the A gate behind closes;

the remote B gate ahead opens;

the horizontal stripe reaches the receptor.

This establishes the global gate rule and the departure-type crossing rule.

Exact state
   01234567890
00 ...........
01 ...........
02 ...........
03 ...........
04 ...ABACA...
05 ...A.......
06 ..AA.......
07 ..A........
08 ...........
09 ...........
10 ...........
Start:              (7,2), type A
Selection/cursor:   none
Maximum actions:     128

Gates:
H(4,3)=A
H(4,5)=B

Receptor:
H[(4,4) -> (4,7)]

Initial state:

A gate H(4,3): open
B gate H(4,5): closed
Receptor: inactive
Reference solution
U R U U R

The final transfer is:

(4,3) A --A gate--> (4,4) B

Immediately after landing:

H(4,3)=A closes
H(4,5)=B opens
horizontal stripe covers (4,4..7)
receptor activates

Reference depth: 5 actions.

Attribution and necessity

The B gate is inside the required receptor interval, so the level cannot be solved by treating gates as movement-only barriers. Its opening must also extend the live stripe.

The A gate is necessary for the final transfer, while the B gate is necessary for the resulting receptor activation. The two gate roles are visibly distinct on the same action.

Level 4 — Enter at the Middle
Player-facing discovery

The source and receptor sit at opposite ends of the top row. Six B gates divide that row.

An older strategy—approaching the source endpoint—is available through the upper-left branch. At (0,1), however, the A holder cannot enter the B-gated row. This is a recoverable local impasse.

The useful route instead reaches (1,4) from below and enters the B token at (0,4). Landing in the middle opens every B gate simultaneously and powers the entire row in both directions.

This is the first required side-feed: the terminal endpoints are not useful entrances.

Exact state
   012345678
00 .ACABCAC.
01 .A..AA...
02 .AABACBA.
03 .......A.
04 .AACABAB.
05 .A.......
06 .CAABA...
07 .....C...
08 .AABAA...

All coordinates outside this 9 × 9 rectangle are gaps.

Start:              (8,1), type A
Selection/cursor:   none
Maximum actions:     320

Receptor:
H[(0,1) -> (0,7)]

Gates:

H(0,c)=B for c=1..6

V(7,5)=A
V(5,1)=C
V(3,7)=B
V(1,5)=C

Initial state:

V(7,5)=A is open.
All six top-row B gates are closed.
The other C and B route gates are closed.
The receptor is inactive.
Reference solution

Full action string:

RRRRUULLLLUURRRRRRUULLULU

Grouped:

R×4, U×2, L×4, U×2, R×6, U×2, L×2, U, L, U

Reference depth: 25 actions.

Key one-way crossings:

Action	Transfer	Gate	Landing type
5	(8,5) -> (7,5)	A	C
11	(6,1) -> (5,1)	C	A
19	(4,7) -> (3,7)	B	A
23	(2,5) -> (1,5)	C	A
25	(1,4) -> (0,4)	ungated	B

The final B landing opens all six H(0,1..6) gates. The horizontal stripe grows from (0,4) to both terminals.

Anti-degenerate and anti-cleanliness check

The upper-left branch ending at (0,1) is not an arbitrary dead end. It is the plausible application of Level 2’s endpoint-entry strategy, and it fails for the newly established gate reason.

The middle entry at (1,4) is indispensable. No non-B top-row token can traverse the B gates, and (0,4) is the only B token on the receptor row.

Every token not on the reference route belongs either to the receptor rail or to the recoverable endpoint experiment. The four one-way crossings create approach asymmetry without introducing another object type.

Level 5 — The Crossing
Player-facing discovery

A horizontal receptor and a vertical receptor visibly cross at (4,4). Both arms are segmented by C gates.

During the route, the player reaches multiple arm endpoints while holding A or B. Each endpoint remains locally isolated because its C gates are closed. Powering one geometric direction is not enough; both receptors must be live in the same settled frame.

The only holder coordinate that lies on both required lines is the C intersection token (4,4). It is entered from below through two deliberately ungated contacts.

Two apparent shortcut loops are directionally reversed by the types on their endpoints. They can take the player backward, but they cannot bypass the authored route in the forward direction.

Exact state
   012345678
00 AAAABABAA
01 B.A.A.A.A
02 AAB.B.AAA
03 A...A...A
04 AAAACAAAA
05 A...A...A
06 ABA.AAABA
07 ..A.A....
08 AAB.A....
Start:              (8,0), type A
Selection/cursor:   none
Maximum actions:     448

Receptors:
H[(4,0) -> (4,8)]
V[(0,4) -> (8,4)]

Cross gates:

H(4,c)=C for c=0..7

V(r,4)=C for r in {0,1,2,3,6,7}

The two contacts below the target are ungated:

(6,4) <-> (5,4)
(5,4) <-> (4,4)

Route gates:

H(6,1)=A
V(1,2)=B
V(0,6)=B
H(6,7)=A

Reverse-only shortcut gates:

V(1,0)=B
H(0,6)=A

The first shortcut appears close from (2,0), but (2,0) is A and cannot cross the B gate upward. From the later B token (1,0), the same gate permits a return downward.

The second appears close from (0,6), but that token is B and cannot cross the A gate rightward. From the later A token (0,7), it permits a return leftward.

Reference solution

Full action string:

RRUULLUUUURRUURRRRDDRRDDDDLLLLUU

Grouped:

R×2, U×2, L×2, U×4,
R×2, U×2, R×4, D×2,
R×2, D×4, L×4, U×2

Reference depth: 32 actions.

Key keyed crossings:

Action	Transfer	Gate	Landing type
5	(6,2) -> (6,1)	A	B
13	(2,2) -> (1,2)	B	A
19	(0,6) -> (1,6)	B	A
27	(6,8) -> (6,7)	A	B

Final approach:

(6,4) A
(5,4) A
(4,4) C

On the C landing, every cross gate opens. The holder’s horizontal and vertical stripes reach all four cross endpoints in the same frame.

Anti-degenerate and anti-cleanliness check

The only winning holder coordinate is (4,4). Any other coordinate can belong to at most one of the two required directional stripes.

The two extra loops are not decorative branches: each is a visually tempting shortcut whose direction is determined by the established departure-type rule. Entering either loop later is recoverable but loses route progress.

Every remaining token is part of the route, one of the two conductive arms, or one of those two asymmetric shortcut loops.

Level 6 — Beyond the Bracket
Player-facing discovery

The horizontal source–receptor interval ends at (3,5), while the vertical pair occupies column 7. The terminal intervals therefore do not visibly cross.

The relay row continues past the horizontal receptor through (3,6) to (3,7). Those two extension contacts are B-gated. The correct holder position is the B token (3,7), outside the horizontal terminal interval but still on its extended conductor run.

An endpoint probe is available at the top of the vertical pair. Reaching (0,7) as A leaves the first B gate below it closed. The endpoint itself is not a magic activation square.

Immediately before the final move, the holder is C at (3,8). It is horizontally adjacent to the correct junction, but the B extension gates remain closed. Moving left onto B opens the extension and powers both pairs.

Exact state
   012345678
00 AABAAACA.
01 A.....AA.
02 A.BAA.AAA
03 AACAAAABC
04 BAA.A..A.
05 ....A..A.
06 ACA.CABAA
07 A.C....AA
08 A.AABACAB
Start:              (8,0), type A
Selection/cursor:   none
Maximum actions:     640

Receptors:
H[(3,0) -> (3,5)]
V[(0,7) -> (8,7)]

Required B-gated runs:

H(3,c)=B for c=0..6
V(r,7)=B for r=0..7

The horizontal pair stops at column 5, but gates H(3,5) and H(3,6) are still needed to feed it from (3,7).

Route gates:

V(0,6)=C
V(2,2)=B
V(2,8)=A
V(5,4)=C
V(6,2)=A
V(7,8)=B

The contact (3,7) <-> (3,8) is ungated and is the final approach.

Reference solution

Full action string:

UURRDDRRRRRRUULLLLUUUULLDDLLUUUURRRRRRDDRRDL

Grouped:

U×2, R×2, D×2, R×6,
U×2, L×4, U×4, L×2,
D×2, L×2, U×4, R×6,
D×2, R×2, D, L

Reference depth: 44 actions.

Key keyed crossings:

Action	Transfer	Gate	Landing type
5	(6,2) -> (7,2)	A	C
13	(8,8) -> (7,8)	B	A
19	(6,4) -> (5,4)	C	A
25	(2,2) -> (3,2)	B	C
39	(0,6) -> (1,6)	C	A
43	(2,8) -> (3,8)	A	C

Final action:

(3,8) C --ungated LEFT--> (3,7) B

After landing:

all row-3 B gates open;
all column-7 B gates open;
H[(3,0) -> (3,5)] becomes active from beyond its receptor;
V[(0,7) -> (8,7)] becomes active;
the level completes.
Anti-degenerate and anti-cleanliness check

The holder cannot satisfy both pairs from inside the horizontal terminal interval because no coordinate there lies on column 7. The player must reason about the conductor’s continuation beyond the receptor badge.

The top endpoint branch at (0,7) provides an immediate, recoverable falsification of “stand on a terminal endpoint.” Its B gate remains visibly closed under the A token.

Every token outside the reference route belongs to one of the two required conductor runs. There are no inert maze branches.

Level 7 — Four Arms, One Frame
Player-facing discovery

Four source–receptor intervals occupy two common axes:

horizontal interval 1: row 2, columns 0..3
horizontal interval 2: row 2, columns 5..10

vertical interval 1: column 8, rows 0..2
vertical interval 2: column 8, rows 4..10

The intervals are separated by conductor-only bridge positions. The holder must therefore preserve one uninterrupted row and one uninterrupted column long enough to energize four separate receptor brackets.

The only coordinate belonging to the common row and common column is (2,8). It is a C token, matching every gate in the required row and column.

Several visually short routes are directionally unusable:

the top bridge from (0,4) toward (0,5) is C-gated, but (0,4) is B;

the bottom bridge from (10,6) toward (10,7) is C-gated, but (10,6) is A;

the lower-right branch reaches the bottom vertical receptor at (10,8), but its upward C gate remains closed while that token holds A.

These routes are usable in the reverse direction and therefore demonstrate asymmetry rather than acting as arbitrary walls.

Exact state
   01234567890
00 AABABCBAA.B
01 A...A.B.A.A
02 BAAABAAACAA
03 A...A.B.A.A
04 CAA.AAA.AAB
05 ..B.....A..
06 AAA.CAAABAA
07 A...A...A.A
08 BACAA.AABAB
09 ......B.A.C
10 AABAAAACAAB
Start:              (10,0), type A
Selection/cursor:   none
Maximum actions:     896

Receptors, in validator order:

R0 = H[(2,0) -> (2,3)]
R1 = H[(2,5) -> (2,10)]
R2 = V[(0,8) -> (2,8)]
R3 = V[(4,8) -> (10,8)]

Required C-gated row:

H(2,c)=C for c=0..9

Required C-gated column:

V(r,8)=C for r=2..9

The two approach contacts above the target are ungated:

(0,8) <-> (1,8)
(1,8) <-> (2,8)

Route gates:

V(9,6)=A
V(7,10)=B
V(6,4)=C
V(7,0)=B
V(5,2)=A
V(3,0)=C
V(0,4)=B
V(3,6)=A
H(0,6)=B

Reverse-only bridge gates:

H(0,4)=C
H(10,6)=C
Reference solution

Full action string:

RRRRRRUURRRRUULLLLLLDDLLLLUURRUULLUUUURRRRDDDDRRUUUURRDD

Grouped:

R×6, U×2, R×4, U×2,
L×6, D×2, L×4, U×2,
R×2, U×2, L×2, U×4,
R×4, D×4, R×2, U×4,
R×2, D×2

Reference depth: 56 actions.

Key one-way crossings:

Action	Transfer	Gate	Landing type
7	(10,6) -> (9,6)	A	B
13	(8,10) -> (7,10)	B	A
21	(6,4) -> (7,4)	C	A
27	(8,0) -> (7,0)	B	A
31	(6,2) -> (5,2)	A	B
35	(4,0) -> (3,0)	C	A
43	(0,4) -> (1,4)	B	A
49	(4,6) -> (3,6)	A	B
53	(0,6) -> (0,7)	B	A

Final approach:

(0,8) A
(1,8) A
(2,8) C

On the last landing, every C gate in row 2 and column 8 opens. Horizontal and vertical stripes grow outward in parallel. The resulting receptor vector is:

[R0, R1, R2, R3] = [true, true, true, true]
Contextual roles

The same C gates have several roles without changing their rule:

On row 2 they prevent ordinary travel along the apparent goal rail.

On column 8 they make the bottom receptor endpoint a trap for the naïve endpoint strategy.

At the final C holder they become the links that preserve all four receptor spans.

At the two reverse-only bridges, C gates reject an apparent forward shortcut but permit a later retreat from the opposite side.

The same live-stripe rule is also used in contrasting ways:

R1 and R2 include the holder.

R0 is powered across a separate left-hand interval.

R3 is powered entirely below the holder, across a conductor-only bridge at (3,8).

Anti-degenerate and anti-cleanliness check

Only (2,8) lies on both common axes. Its intrinsic C type is also the only type that opens every required contact. Position alone and type alone are each insufficient as explanations; the final state visibly satisfies both.

The lower-right endpoint branch, top reverse bridge, and bottom reverse bridge each instantiate an established misconception:

terminal endpoints are not goals by themselves;

geometric closeness does not imply directional accessibility;

a gate crossed from one side may reject the return or the apparent forward shortcut.

Every non-route token belongs to a required receptor run or one of those concrete asymmetric experiments. No object is present solely to make the map denser.

5. Validator reference

The implementation should reproduce the following winning holder sets and shortest arrow-only path lengths:

Level	Winning holder coordinate(s)	Shortest length
1	{(4,4)}	3
2	{(3,4), (3,5), (3,6), (3,7), (3,8)}	6 from the authored start
3	{(4,4)}	5
4	{(0,4)}	25
5	{(4,4)}	32
6	{(3,7)}	44
7	{(2,8)}	56

Required behavioral tests:

Departure-type test: In Level 3, the final A gate permits entry into B, then closes after landing.

Remote-gate test: That same landing opens the nonadjacent B gate inside the receptor interval.

No persistence test: Moving away from an active line immediately removes stripes and deactivates its receptor unless the new holder still conducts through the interval.

No turning test: A horizontal stripe reaching an intersection does not continue vertically unless the holder itself occupies that intersection.

Outside-interval test: In Level 6, holder (3,7) activates horizontal interval (3,0..5).

Simultaneous-frame test: Level 7 succeeds only when all four receptor booleans are true after one shared recomputation.

Blocked-action test: A blocked transfer changes only step_count and visual feedback.

Render-purity test: Animation frames never alter holder, gates, stripes, receptors, or budget.

Unused-action test: SPACE and all clicks leave holder unchanged, store no cursor state, and consume one action.

Exploration test: Before success, every state reachable from each authored start has a directed route to a winning state.
