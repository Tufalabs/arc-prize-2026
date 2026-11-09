Vanishing Levers

Exact authored level count: 7

This is a game about flipping an entire row or column becoming difficult because every lever works only from a dark host and disappears after one pull.

1. Player-facing concept

The board is a rectangular array of recessed dark cells and bright glowing cells. Every lever sits inside one cell and bears a large double-ended horizontal or vertical arrow.

Pulling a lever flips every cell in the indicated row or column, including the cell beneath the lever. The lever then folds beneath a spent cap and can never act again.

An unspent lever is usable only when its host cell is dark. A glowing host closes a conspicuous clamp around the handle. Consequently, a line flip can:

open a previously clamped crossing lever;

close an open crossing lever;

repair a dark cell beneath an old spent cap;

create an irreparable dark cell beneath a lever already spent;

commit the player to one coherent family of row and column flips while making a complementary family impossible.

The objective is always visible and physical: make every board cell glow at the same time. Unspent levers are not themselves objectives.

No text, numbers, counters, target diagrams, or symbolic legends appear in the rendered game.

2. Logical state

Coordinates are written as (row, column), one-indexed in this document, with (1,1) at the top-left.

For a level of height H and width W, the logical state is:

board[H][W]          boolean; 0 = dark, 1 = glowing
levers[]             fixed host coordinate and orientation
lever.spent          boolean
selected_lever_id    one lever or spent cap
terminal              boolean
feedback_kind         none, clamped, spent, no-directional-target, win
step_count            integer; diagnostic only

A clamp does not need independent state:

clamped(lever) =
    not lever.spent
    and board[lever.row][lever.column] == 1

There is at most one lever or spent cap in any cell.

Cells beneath levers remain ordinary objective cells. Lever graphics never replace or shield the illumination bit.

Lever orientations

H: flips all cells (r,1) through (r,W).

V: flips all cells (1,c) through (H,c).

The arrows are double-ended. There is no directional difference between left and right, or between up and down.

3. Action semantics

The supported action space is exactly:

UP, DOWN, LEFT, RIGHT, SPACE, CLICK(any cell)

Only the four directions and SPACE have gameplay functions.

3.1 Selector movement

The selector moves between lever objects rather than through empty board cells. Unspent, clamped, and spent levers all remain selector nodes.

This makes every movement action a spatial selection decision rather than empty cursor travel.

For the selected object at (r,c), calculate the next selection as follows.

RIGHT

Consider every lever with c' > c. Choose the minimum tuple:

(abs(r' - r), c' - c, r', c', object_id)
LEFT

Consider every lever with c' < c. Choose the minimum tuple:

(abs(r' - r), c - c', r', c', object_id)
DOWN

Consider every lever with r' > r. Choose the minimum tuple:

(abs(c' - c), r' - r, r', c', object_id)
UP

Consider every lever with r' < r. Choose the minimum tuple:

(abs(c' - c), r - r', r', c', object_id)

Thus the selector prefers a lever exactly aligned with the requested direction, then the nearest parallel line.

If no candidate exists in the requested half-plane:

the selected object does not change;

the selector ring briefly nudges in that direction;

no board or lever state changes.

Selection movement never changes illumination or lever availability.

3.2 SPACE

SPACE acts on the selected lever.

The checks occur in this order:

Spent check.
If the selected lever is spent, nothing changes. Its cap pulses.

Clamp check.
If the selected lever is unspent but its host cell glows, nothing changes. Its clamp flashes and rattles.

Valid pull.
If the selected lever is unspent and its host cell is dark:

capture the affected row or column from the pre-action state;

toggle every cell in that line simultaneously;

mark the lever spent;

retain the selector on its new spent cap;

derive all other clamp appearances from the new board;

validate completion.

The pulled lever’s host is dark before the action and therefore glows immediately afterward.

There are no automatic lever activations or chain pulls.

3.3 Clicking

Every click action, including a click directly on a lever, is an inert no-op:

it does not select an object;

it does not pull a lever;

it produces no special click feedback;

it only increments the diagnostic step count.

There is deliberately no click affordance or click mode.

3.4 Terminal state

Once the level is complete, later actions produce no state change.

There is no energy meter, timer, or pull quota. A recommended harness safety limit is 512 actions per episode, but this is not rendered and is not part of the puzzle.

4. Exact success condition

After every valid pull:

success = all(board[r][c] == 1 for every board cell)

Relevant:

every board cell, including cells beneath unspent levers and spent caps.

Ignored:

selector position;

which levers remain unspent;

whether remaining unspent levers are open or clamped;

action count;

failed pull attempts;

which of multiple valid lever families was used.

Allowed:

completing with many untouched levers;

any valid order or subset that produces the all-glowing board.

Forbidden:

even one dark board cell.

The green completion frame is feedback, not an additional requirement.

5. Update and animation order

Logical state changes only during step(action). Calling render() never advances the game.

Valid pull animation

A valid pull has four deterministic render-only subframes:

The selected handle begins folding and the selected-line outline brightens.

Every affected cell flashes light blue simultaneously.

All affected cells display their committed new illumination states; the handle is replaced by its spent cap.

Clamps settle into their newly derived positions.

The logical board and spent flag are committed before subframe 1. The animation does not consume additional actions. A headless implementation may return only subframe 4.

There is no left-to-right or top-to-bottom sweep. Such a sweep would falsely imply an ordering among cells.

Invalid pull feedback

Clamped: the maroon jaws move inward one pixel, flash red, and return.

Spent: the gray cap briefly becomes white and then returns to gray.

No selector destination: the blue selector ring nudges and returns.

Each invalid animation is render-only and leaves the logical state unchanged.

Completion

If the committed board is all glowing:

terminal becomes true immediately;

the outer frame changes to green;

the frame pulses once;

the final board and all remaining lever states stay visible.

6. Visual contract
6.1 Canvas and board geometry

Canvas: exactly 64×64.

Canvas and cell gutters: color 5.

Every logical cell is a connected 5×5 pixel tile.

Adjacent tiles have a 1 pixel negative-space gutter.

Board pixel size:

pixel_width  = 6 × W - 1
pixel_height = 6 × H - 1

Center the board using:

origin_x = floor((64 - pixel_width) / 2)
origin_y = floor((64 - pixel_height) / 2)

For the largest 10×10 level, the board is 59×59 pixels.

There are no permanent interior grid lines or borders. The black gutters are negative space rather than outlined cells.

A one-pixel outer frame sits two pixels outside the board bounding box, clipped only if necessary:

idle: color 3;

definite immobility with an unsolved board: color 13;

completion: color 14.

The immobility frame is shown only when there is no unspent lever on a dark host. It does not run a hidden future-solvability test.

6.2 Cell appearance

An empty dark cell:

outer 5×5 tile: color 4;

central 3×3 recess: color 5.

An empty glowing cell:

outer 5×5 tile: color 11;

central 3×3 core: color 12.

A lever sprite occupies only the central 3×3, leaving the one-pixel cell rim visible. Therefore, the host cell’s illumination remains readable under every lever state.

6.3 Lever appearance

All unspent handles use color 6. Orientation tips use color 0.

H: a three-pixel horizontal central bar with white end tips and a short upright handle.

V: the transposed silhouette.

Open clamp:

two separated jaws in color 1;

visible space remains between each jaw and the handle.

Closed clamp:

jaws move against the handle;

jaws use color 13.

Spent cap:

connected central 3×3 shape in colors 2 and 3;

no upright magenta handle;

a white horizontal or vertical groove preserves the former orientation.

No IDs or letters are rendered.

6.4 Selector and line preview

Selected lever: one-pixel ring in color 9.

Affected line: one transient rectangular outline around the entire row or column footprint in color 10.

The preview never changes the board and never obscures the host rim.

A spent cap still shows its former affected line when selected.

This is one outline around the full line, not a border around every internal cell.

6.5 Palette usage
ID	Use
0	orientation tips and cap groove
1	open clamp
2	spent cap
3	recessed detail and idle frame
4	dark cell
5	canvas, gutters, deep recess
6	unspent handle
8	invalid-action flash
9	selector
10	selected-line preview and flip flash
11	glowing cell rim
12	glowing cell core
13	closed clamp and immobility frame
14	completion frame

Colors 7 and 15 are unused.

7. Later-level parity construction

Levels 4–7 use a deliberately verifiable construction.

Let:

D[r,c] = 1 when the initial cell is dark
x[r]   = 1 when row r is pulled
y[c]   = 1 when column c is pulled

A cell finishes glowing exactly when:

D[r,c] = x[r] XOR y[c]

On a full rectangular board, choosing x[1] determines every y[c], which then determines every other x[r]. Therefore, a solvable board has exactly two complementary row/column parity solutions.

Levels 4–7 contain exactly one unspent lever in every row and every column. Their initial matrices are authored so that:

one named family is a winning subset;

its complete complement is the only other winning subset;

every unspent lever belongs to one of those two real solutions;

spending even one line from each family makes both winning subsets impossible.

Initial spent caps do not provide additional line operations.

This makes the unfamiliar open lever in the other family a real alternative rather than a decorative decoy. Removing any unspent lever destroys one entire valid solution family and makes the commitment problem easier.

8. Authored levels

In the matrices below:

1 = glowing
0 = dark

The digits, IDs, arrows, and coordinates are documentation only and are never drawn as text.

For action traces:

U D L R = selector actions
S       = SPACE

Failed probes are included when they provide the intended discovery evidence.

Level 1 — One Line
Teaches

SPACE pulls the selected lever.

A lever flips its entire indicated line.

An all-glowing board completes the level.

Player-facing discovery

There is one dark horizontal strip and one selected horizontal lever. No other object can compete as an explanation. Pulling it immediately lights the entire board and turns the frame green.

The spent cap is visible in the completed frame, but permanence is not yet required for the solution.

Exact initial state

Board size: 1×3

000

Objects:

ID	State	Host	Orientation
A	unspent, open	(1,2)	H

Initial selection: A

Intended trace
S

After A:

111

The board completes immediately.

Depth

Valid pulls: 1

Selector actions: 0

Total authored controls: 1

Attribution and necessity

A is the only action-bearing object.

Every cell is affected by A.

No simultaneous mechanic change can explain the result.

Removing any cell or the lever destroys the one-action causal demonstration.

Level 2 — It Stays Gone
Teaches

A pulled lever permanently becomes spent.

A spent lever cannot be reused.

Different rows still require different levers.

Player-facing discovery

The player lights the first row. The selector remains on the visibly flattened handle. Pressing SPACE again produces a cap pulse rather than another toggle. The second row remains dark, and the only productive action is selecting its separate lever.

Exact initial state

Board size: 2×3

000
000

Objects:

ID	State	Host	Orientation
A	unspent, open	(1,2)	H
B	unspent, open	(2,2)	H

Initial selection: A

Intended discovery trace
S S D S

Resolution:

First S: A flips row 1 and becomes spent.

111
000

Second S: spent-cap pulse; no board change.

D: select B.

Final S: B flips row 2 and the level completes.

Minimum success trace
S D S
Depth

Valid pulls: 2

Minimum controls: 3

Authored discovery controls: 4

Attribution and necessity

The failed second pull directly distinguishes permanent spending from an ordinary repeatable toggle.

A is necessary for row 1.

B is necessary for row 2.

No clamp behavior is encountered before Level 3.

Level 3 — Open the Clamp
Teaches

An unspent lever on a glowing host is clamped.

A crossing line can darken that host and open the clamp.

Clamp state is derived immediately from host illumination.

Player-facing discovery

The initially selected vertical lever is visibly upright but trapped by maroon jaws. Its host rim glows. SPACE makes the clamp rattle.

An adjacent horizontal lever sits on a dark host. Pulling it flips the top row: the selected vertical lever’s host visibly becomes dark and its jaws open. Pulling the newly opened lever completes the board.

Exact initial state

Board size: 2×2

10
01

Objects:

ID	State	Host	Orientation
A	unspent, clamped	(1,1)	V
B	unspent, open	(1,2)	H

Initial selection: A

Intended discovery trace
S R S L S

State landmarks:

First S: clamp flash; matrix remains:

10
01

Select and pull B:

01
01

A’s host (1,1) is now dark, so A opens.

Pull A:

11
11
Depth

Valid pulls: 2

Minimum controls without probe: 4

Authored discovery controls: 5

Attribution and necessity

A cannot act before B.

B cannot finish the bottom-left cell.

The fixed glowing cell (2,2) makes the final result unambiguous.

The only new causal unknown is the dark-host requirement.

Level 4 — Choose a Chain
Teaches and combines

A level may finish with many unspent levers.

Two coherent complementary lever families can both solve the same board.

Mixing those families spends irreplaceable line parities and makes completion impossible.

Correct row/column crossings toggle their intersection twice.

A wrong mixed crossing can darken the host of a spent lever, creating visible spent-cell debt.

Player-facing discovery

Only A and E begin open. Everything else is visibly clamped.

Pulling A immediately opens B, providing a strong local hypothesis: continue the causally opening chain. E remains open but has not participated in that causal change. Following A–B–C–D completes the board while E–H remain untouched.

A player who instead pulls A, then E, then F sees F darken the host beneath A’s spent cap. Both the dark debt and the unavailable cap are visible; no abstract checker is needed to explain the failure.

Exact initial state

Board size: 4×4

1010
1010
0101
0101

Objects, in stable implementation order:

ID	State	Host	Orientation	Causal role
A	open	(1,2)	H	first root
B	clamped	(1,1)	V	A opens B
C	clamped	(2,1)	H	B opens C
D	clamped	(2,3)	V	C opens D
E	open	(3,1)	H	second root
F	clamped	(3,2)	V	E opens F
G	clamped	(4,2)	H	F opens G
H	clamped	(4,4)	V	G opens H

Initial selection: A

Dependencies:

A → B → C → D
E → F → G → H

Winning subsets:

Family α: A B C D
Family β: E F G H

These are the only winning lever subsets.

Intended trace
S L S D S R S

Pulled lever order:

A, B, C, D

Immediately before D, the board is:

1101
1101
1101
1101

The selected preview around D therefore encloses the sole entirely dark column. Pulling D completes the board.

Alternative valid lever order:

E, F, G, H
Depth

Irreversible pulls: 4

Authored controls: 7

Anti-degenerate and necessity check

Every unspent lever belongs to one complete valid family.

Removing any A–D lever destroys family α.

Removing any E–H lever destroys family β.

Pulling from both families cannot be repaired because every row and column has only one unspent lever.

The two chains occupy crossing rows and columns; replacing the layout with a corridor would erase the double-toggle and spent-debt consequences.

The level cannot be solved by pulling every available lever.

Level 5 — Leave a Final Line
Teaches and combines

A winning family can contain multiple open roots and an internal fork.

A newly open lever is useful evidence, but the player must still remain within one parity family.

The line preview can certify a decisive final pull: every cell in that line is visibly dark beforehand.

A familiar open root can be either the beginning of a valid alternative or a fatal mixed commitment.

Player-facing discovery

The selector starts on clamped C. Pressing SPACE confirms that the old clamp rule still applies.

A is directly to its right and visibly open. Pulling A opens C. Meanwhile B is another open root in the same upper family, while G is the root of the lower complementary family.

The intended route develops two causally related parts of the upper family. Immediately before F, every cell in column 3 is dark and every other cell glows. F’s preview therefore presents positive, local evidence that it is a viable final action.

Exact initial state

Board size: 6×6

111000
111000
111000
000111
000111
000111

Objects:

ID	State	Host	Orientation	Causal role
A	open	(1,4)	H	α root
B	open	(2,6)	H	α root
C	clamped	(1,1)	V	A opens C
D	clamped	(2,2)	V	B opens D
E	clamped	(3,2)	H	D opens E
F	clamped	(3,3)	V	E opens F
G	open	(4,1)	H	β root
H	clamped	(4,4)	V	G opens H
I	clamped	(4,5)	V	G opens I
J	clamped	(5,5)	H	I opens J
K	clamped	(6,4)	H	H opens K
L	clamped	(5,6)	V	J opens L

Initial selection: C

Dependencies:

Family α:
A → C
B → D → E → F

Family β:
G → H → K
G → I → J → L

Winning subsets:

Family α: A B C D E F
Family β: G H I J K L

The order inside a family may interleave its independent branches.

Intended discovery trace
S R S L S R R S L S D S R S

Pulled lever order:

A, C, B, D, E, F

The first S is the intentional clamp probe on C.

Immediately before F:

110111
110111
110111
110111
110111
110111

Column 3 is the only dark line. F toggles it to glow and completes the board.

A valid lower-family order is:

G, H, K, I, J, L
Depth

Irreversible pulls: 6

Minimum controls for the documented family: 13

Authored discovery controls: 14

Anti-cleanliness and necessity check

The upper solution has two spatially separated roots rather than a single object sequence.

The lower solution forks after G.

Objects are not laid out in their complete use order.

All twelve unspent levers participate in a valid winning family.

Mixing A or B with G commits row parities from opposite solutions and cannot be repaired.

F is not a separate goal device; it is an ordinary column lever whose authored placement produces a naturally visible final-line state.

Level 6 — Old Debts
Teaches and combines

A level may begin with already-spent levers.

A dark cell beneath a spent cap is ordinary board debt: it still must glow, but that lever cannot act.

Debt may be repaid by another lever in the same line or by a perpendicular crossing.

The same operational rule has contrasting strategic roles in the two valid families.

One action can both repay a debt and unlock the next lever.

Player-facing discovery

The selector begins on S3, a horizontal spent cap whose host cell is dark. SPACE produces the familiar spent pulse.

Pressing UP selects F, whose vertical preview passes directly through S3, but F is clamped. Pressing RIGHT selects adjacent open E. Pulling E darkens F’s host and opens it. Pulling F then:

lights the dark cell beneath S3;

leaves S3 spent;

opens G at (3,3).

This is an immediate physical demonstration of debt repayment rather than an abstract explanation.

The alternative lower family repairs the same three debts with row pulls instead of the intended upper family’s column pulls.

Exact initial state

Board size: 7×7

1110000
1110000
1110000
1110000
0001111
0001111
0001111

Unspent objects:

ID	State	Host	Orientation	Causal role
A	open	(1,4)	H	α root
B	clamped	(1,1)	V	A opens B
C	clamped	(2,1)	H	B opens C
D	clamped	(2,2)	V	C opens D
E	open	(4,4)	H	α root
F	clamped	(4,3)	V	E opens F
G	clamped	(3,3)	H	F opens G
H	open	(5,2)	H	β root
I	clamped	(5,4)	V	H opens I
J	clamped	(6,4)	H	I opens J
K	clamped	(6,5)	V	J opens K
L	open	(7,1)	H	β root
M	clamped	(7,6)	V	L opens M
N	clamped	(7,7)	V	L opens N

Initial spent debts:

ID	Host	Former orientation	Initial host
S1	(5,1)	H	dark
S2	(6,2)	V	dark
S3	(7,3)	H	dark

Initial selection: S3

Dependencies:

Family α:
A → B → C → D
E → F → G

Family β:
H → I → J → K
L → M
L → N

Winning subsets:

Family α: A B C D E F G
Family β: H I J K L M N

Debt repairs:

Debt	Family α repair	Family β repair
S1 at (5,1)	B flips column 1, perpendicular to S1	H flips row 5
S2 at (6,2)	D flips column 2	J flips row 6, perpendicular to S2
S3 at (7,3)	F flips column 3, perpendicular to S3	L flips row 7

No repair-direction exception exists. Whichever surviving line crosses the cell toggles it normally.

Intended discovery trace
S U S R S L S U S R U S L S D S R S

Interpretation:

S on S3: spent pulse.

U, S on F: clamp rattle.

R, S on E: pull E and open F.

L, S on F: repair S3 and open G.

Continue with G, A, B, C, D.

Pulled lever order:

E, F, G, A, B, C, D

Immediately before D:

1011111
1011111
1011111
1011111
1011111
1011111
1011111

Column 2 is entirely dark. It contains S2’s outstanding debt. Pulling D both pays that final debt and completes the board.

A valid lower-family order is:

H, I, J, K, L, M, N
Depth

Irreversible pulls: 7

Failed causal probes: 2

Total authored controls: 18

Anti-cleanliness and necessity check

The three debts are restored at different phases rather than forming a consecutive checklist.

S3 is restored early while simultaneously unlocking G.

S1 is restored in the middle.

S2 remains as the visible final-line debt.

The alternative family gives the same caps different contextual repair roles without changing any rule.

Every unspent lever belongs to one complete valid family.

Every spent cap begins on a dark objective cell that must be restored in either solution.

Level 7 — Alternating Commitments
Teaches and combines

Parity families need not occupy clean contiguous quadrants.

One root can reveal a fork of two dependency arms.

The player must preserve commitment while working across spatially alternating rows and columns.

Several old debts are repaired in an offset order.

The final pull is simultaneously a line completion, a debt payment, and the last member of the selected family.

Player-facing discovery

The initial board is an alternating glow pattern rather than the large blocks used in Levels 4–6. The player can no longer identify a family merely as “top” or “bottom.”

The selector starts on dark spent cap X0. Two complementary roots exist:

A begins the odd-row/odd-column family.

K begins the even-row/even-column family.

Pulling A immediately opens both B and D, visibly revealing a fork. The player has evidence for continuing those two arms. K remains open, but mixing it into the chosen family would spend an even row and eliminate both complete winning subsets.

The five debts are placed on alternating dark cells. During the intended family:

some are repaired by columns;

X0 is repaired by a row;

X3 remains on the sole dark final column until J is pulled.

Exact initial state

Board size: 10×10

1010101010
0101010101
1010101010
0101010101
1010101010
0101010101
1010101010
0101010101
1010101010
0101010101
Family α: odd rows and odd columns
ID	State	Host	Orientation	Dependency
A	open	(1,10)	H	root
B	clamped	(1,1)	V	A → B
C	clamped	(9,1)	H	B → C
F	clamped	(9,3)	V	C → F
E	clamped	(3,3)	H	F → E
D	clamped	(1,9)	V	A → D
G	clamped	(7,9)	H	D → G
H	clamped	(7,7)	V	G → H
I	clamped	(5,7)	H	H → I
J	clamped	(5,5)	V	I → J

Dependency network:

              ┌→ B → C → F → E
A, the root ──┤
              └→ D → G → H → I → J
Family β: even rows and even columns
ID	State	Host	Orientation	Dependency
K	open	(2,1)	H	root
L	clamped	(2,2)	V	K → L
M	clamped	(10,2)	H	L → M
P	clamped	(10,4)	V	M → P
O	clamped	(4,4)	H	P → O
N	clamped	(2,10)	V	K → N
Q	clamped	(8,10)	H	N → Q
R	clamped	(8,8)	V	Q → R
S	clamped	(6,8)	H	R → S
T	clamped	(6,6)	V	S → T

Dependency network:

              ┌→ L → M → P → O
K, the root ──┤
              └→ N → Q → R → S → T
Initial spent debts
ID	Host	Former orientation	Initial host
X0	(5,6)	H	dark
X1	(2,3)	V	dark
X2	(4,7)	H	dark
X3	(6,5)	V	dark
X4	(8,9)	H	dark

Initial selection: X0

Winning subsets:

Family α:
A B C D E F G H I J

Family β:
K L M N O P Q R S T

They are the only winning subsets.

Debt roles
Debt	Family α repair	Family β repair
X0 (5,6)	I flips row 5	T flips column 6
X1 (2,3)	F flips column 3	K flips row 2
X2 (4,7)	H flips column 7	O flips row 4
X3 (6,5)	J flips column 5	S flips row 6
X4 (8,9)	D flips column 9	Q flips row 8

The former orientation of each cap alternates which family repairs it perpendicularly. The operational cell-toggle rule remains identical.

Intended discovery trace
S
U U R S
L L S
D D S
R S
U S
U U R S
D S
L S
U S
L L S

The first S is the spent-cap probe on X0.

Pulled lever order:

A, B, C, F, E, D, G, H, I, J

This completes one fork arm before returning through the selector graph to complete the other. Other interleavings of the two arms are valid.

Immediately before J:

1111011111
1111011111
1111011111
1111011111
1111011111
1111011111
1111011111
1111011111
1111011111
1111011111

Column 5 is the sole dark line. It contains X3’s final outstanding debt. Pulling J:

toggles the entire dark column to glow;

repairs X3;

spends the final family-α lever;

completes the level.

A valid family-β lever order is:

K, L, M, P, O, N, Q, R, S, T
Depth

Irreversible pulls: 10

Selector hops in the documented route: 17

Failed causal probes: 1

Total authored controls: 28

Anti-cleanliness and necessity check

The alternating matrix removes the earlier quadrant shortcut without changing the underlying rule.

Each winning family has a fork rather than one dependency chain.

The two arms are spatially interwoven with spent caps and opposite-family levers.

The other family remains a complete valid solution, not a collection of decoys.

Removing any unspent lever destroys one full family and simplifies the commitment problem.

Mixing roots A and K makes both exact winning subsets impossible.

Every spent cap is a dark objective cell and is restored at a distinct point in either solution.

The last pull has three concrete consequences rather than satisfying an additional hidden predicate.

9. Curriculum and epistemic audit
Level	New evidence available to a blind player	Low-cost distinguishing experiment
1	One selected lever, one dark line	Pull once; whole line changes and frame confirms objective
2	Pulled handle visibly becomes a cap	Press SPACE again; cap pulses without toggling
3	Selected upright lever has closed jaws and glowing host	Failed pull, then adjacent crossing visibly opens it
4	Two open roots; pulling one opens only its local chain	Continue the newly opened chain and finish with the other half untouched
5	Several roots, but only one family creates a coherent dark final line	Compare selected-line previews before the last pull
6	Selected dark spent cap cannot act; nearby crossing lever is clamped	Open the crossing, pull it, and watch the cap’s host glow
7	Alternating pattern and two roots; chosen root opens a fork	Follow both newly opened arms while preserving the untouched complementary family

The first three levels each introduce at most one causal unknown. Later levels introduce no additional device categories or cell states.

10. Implementation acceptance tests

An implementation is conformant when all of the following hold:

Level count is exactly 7.

The initial matrices and object coordinates match this specification.

render() without step() never changes logical state.

Pulling a valid lever toggles all cells in its line simultaneously.

The host cell is included in its own line.

A lever is valid exactly when it is unspent and its host is dark.

The pulled lever becomes spent in the same logical transition.

A spent lever takes precedence over host darkness during validation.

Other clamps update immediately from the committed board.

The selector remains on the spent cap after a pull.

Directional selection uses the specified tuple ordering and includes spent caps as candidates.

Clicking never selects or pulls anything.

Completion depends only on all board cells glowing.

Levels 4–7 accept both documented complementary lever subsets.

Levels 4–7 reject every mixed lever subset.

Level 3’s first SPACE leaves the matrix unchanged.

Level 3 becomes 01 / 01 after B and 11 / 11 after A.

Level 6’s F pull changes S3’s host from dark to glowing and opens G.

Level 7’s documented pre-final state has only column 5 dark.

Animation frames consume no player actions and introduce no sequential cell-toggle semantics.
