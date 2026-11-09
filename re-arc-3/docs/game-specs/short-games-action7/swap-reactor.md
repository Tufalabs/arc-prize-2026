Swap Reactor

Exact authored level count: 7

This is a game about swapping neighboring reactor tiles becoming difficult because every completed match immediately removes support and changes all later alignments.

Swap Reactor is a deterministic click-only puzzle played on a compact reactor grid. Each click exchanges two orthogonally adjacent cells. Whenever the exchange creates a horizontal or vertical run of at least three equal active tiles, every currently matched tile deactivates simultaneously. The remaining tiles then settle downward, possibly creating another match and another settling event.

The objective is visible directly on the board: deactivate every tile carrying a target ring. A target ring moves with its tile during swaps and settling and disappears only when that tile deactivates.

The puzzle is therefore not merely “make three.” A direct target match may be unavailable or actively harmful. The player must often stage several near-matches, remove the correct supports, preserve useful tiles through an earlier reaction, or arrange several columns so that one decisive click launches a multi-round cascade.

1. Player-facing rules

The player sees:

A framed reactor grid.

Bright active tiles in four states, distinguished by both color and a large internal symbol.

Dark inactive sockets.

Thick gray connectors between every pair of orthogonally adjacent cells.

White-and-orange target rings on tiles that must be deactivated.

A fixed light-blue downward arrow beneath the reactor.

The only useful action is clicking a connector.

Clicking a connector swaps the complete contents of the two cells on its sides. This includes active tiles, inactive sockets, tile state, and any target ring.

After the swap:

Find every horizontal and vertical contiguous run of at least three active tiles with the same state.

Deactivate the union of all such runs simultaneously.

Settle all surviving active tiles downward within their own columns.

Search for matches again.

Repeat until the reactor is stable.

The player wins when no target rings remain.

There is no loss condition, move budget, energy meter, or game timer. An implementation may impose a generous external episode horizon of 512 actions for every level, but this is not rendered and is not part of the puzzle logic.

2. Action space

The environment exposes exactly:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x, y) on any rendered cell of the 64×64 canvas

Only CLICK on a connector is mechanically useful.

UP, DOWN, LEFT, RIGHT, SPACE, clicks on tile bodies, clicks on inactive socket bodies, clicks on the frame, clicks on the arrow, and clicks on the background are invalid no-ops. They still call step() and increment the external step count, but do not modify the reactor.

There is:

no logical cursor;

no selected tile;

no first-click/second-click exchange mode;

no persistent selected connector.

Every valid interaction is a single direct connector click.

3. Native board and coordinate system

Each level uses an H × W logical grid.

Coordinates are one-indexed:

(1,1) is the top-left cell.

Row numbers increase downward.

Column numbers increase rightward.

Connector notation used in this specification:

H(r,c) is the horizontal connector between (r,c) and (r,c+1).

V(r,c) is the vertical connector between (r,c) and (r+1,c).

Thus, clicking V(1,2) exchanges the cells at (1,2) and (2,2).

All connectors that exist geometrically are displayed and valid, including connectors:

between two active tiles;

between an active tile and an inactive socket;

between two inactive sockets.

A valid connector click always gives visible feedback, even when it exchanges identical contents or produces no match.

4. Exact 64×64 rendering contract
4.1 Board placement

The canvas is exactly 64×64 pixels.

Each logical cell occupies a connected 6×6 pixel body. Adjacent cell bodies are separated by a 3-pixel connector region.

Constants:

CELL_SIZE = 6
CONNECTOR_GAP = 3
PITCH = 9

For a board of width W and height H:

board_pixel_width  = 9W - 3
board_pixel_height = 9H - 3

x0 = floor((64 - board_pixel_width) / 2)
y0 = floor((56 - board_pixel_height) / 2) + 1

The top-left pixel of logical cell (r,c) is:

x = x0 + 9(c - 1)
y = y0 + 9(r - 1)

Its body occupies inclusive pixel coordinates:

[x, x+5] × [y, y+5]

A two-pixel-thick outer frame surrounds the complete board footprint. There are no decorative borders around individual cells.

4.2 Connector hitboxes

For a cell beginning at pixel (x,y):

H(r,c) hitbox = [x+6, x+8] × [y,   y+5]
V(r,c) hitbox = [x,   x+5] × [y+6, y+8]

The 3×3 region where a horizontal gap and vertical gap cross is background and is not clickable. Horizontal and vertical hitboxes are therefore disjoint.

A horizontal connector is rendered as:

full 3×6 hitbox in color 2;

central two rows in color 1.

A vertical connector is rendered as:

full 6×3 hitbox in color 2;

central two columns in color 1.

On a valid click, the complete hitbox is rendered in color 7 for the returned feedback frame.

4.3 Used colors
Role	Color ID
Canvas background	5
Outer reactor frame	3
Connector body	2
Connector core	1
Valid-click pulse	7
Inactive socket	4 with a 2×2 center in 5
Target ring	0 with midpoint ticks in 12
Down arrow	10
Invalid-action frame flash	8
Completion frame pulse	alternating 14 and 0
4.4 Active tile states

The letters below are documentation notation only. No letters or text are rendered.

State	Body color	Large internal symbol
A	6	white X
B	9	white vertical bar
C	11	dark horizontal bar
D	14	white plus

Local symbol coordinates use (u,v) inside the 6×6 tile, with (0,0) at its top-left.

State A: white X

Color 0 at:

(1,1), (4,1),
(2,2), (3,2),
(2,3), (3,3),
(1,4), (4,4)
State B: white vertical bar

Color 0 at:

u ∈ {2,3}, v ∈ {1,2,3,4}
State C: dark horizontal bar

Color 4 at:

u ∈ {1,2,3,4}, v ∈ {2,3}
State D: white plus

Color 0 on the union of:

u ∈ {2,3}, v ∈ {1,2,3,4}
u ∈ {1,2,3,4}, v ∈ {2,3}
4.5 Target ring

A target-bearing tile retains its normal state body and symbol.

Its perimeter pixels are rendered in color 0, except for orange midpoint ticks in color 12 at:

(2,0), (3,0),
(2,5), (3,5),
(0,2), (0,3),
(5,2), (5,3)

The ring is part of the tile record. It moves during swaps and settlement.

4.6 Inactive socket

An inactive socket is a 6×6 block in color 4, with its central 2×2 pixels in color 5.

It is not an absent board cell. It remains connected to its neighbors by normal clickable connectors.

4.7 Downward-settling indicator

A fixed downward arrow is centered beneath the reactor:

shaft: x=31..32, y=57..59
head:  x=29..34, y=60
       x=30..33, y=61
       x=31..32, y=62

All arrow pixels use color 10.

The arrow never changes direction. It is not clickable.

5. State representation

A sufficient implementation state is:

cells[H][W]:
    Empty
    or Tile {
        state: A | B | C | D,
        target: Boolean
    }

terminal: Boolean
feedback_kind: none | valid_connector | invalid_action
feedback_connector: optional H(r,c) or V(r,c)
step_count: nonnegative integer

There is no hidden tile identity required for gameplay. Debug implementations may attach unique IDs, but equal-state untargeted tiles are operationally interchangeable.

At every level reset:

terminal = false
feedback_kind = none
feedback_connector = null
step_count = 0
selection = null
cursor = absent

All seven authored initial boards are stable and contain no initial match.

6. Exact step() semantics
step(action):

    step_count += 1
    clear previous feedback latch

    if terminal:
        return unchanged terminal state

    if action is not a click inside exactly one connector hitbox:
        feedback_kind = invalid_action
        return unchanged reactor state

    feedback_kind = valid_connector
    feedback_connector = clicked connector

    swap the complete two cell records

    repeat:
        M = union of all horizontal and vertical matching runs

        if M is empty:
            break

        remove every cell in M simultaneously
        settle all columns downward simultaneously

    terminal = no cell has target == true

    return the final stable state
6.1 Match detection

A horizontal or vertical run matches when:

it is contiguous;

every cell is active;

every tile has the same state;

its length is at least three.

Target status is ignored when comparing states.

A maximal run of length four, five, or six deactivates in full. It is not divided into separate trios.

Diagonal runs never match.

6.2 Simultaneous match union

At one cascade round, calculate all matching runs before removing anything.

If a tile belongs to both a horizontal and vertical run, it is included once in the removal set.

If several unrelated runs exist, they all deactivate before any tile settles.

This rule is essential in Levels 5 and 7.

6.3 Settlement

After a removal round, every column settles independently.

For each column:

survivors = active tiles read from top to bottom
new column = Empty repeated (H - len(survivors)) times,
             followed by survivors in the same order

Settlement:

never changes a tile’s state;

never changes its target flag;

never reorders surviving tiles within a column;

does not move tiles sideways.

Settlement occurs only after at least one tile deactivates.

A swap that produces no match does not invoke gravity. Consequently, active tiles can deliberately be swapped sideways through inactive sockets.

6.4 Win validation

Success is checked only after complete cascade closure.

success =
    terminal
    and every cell has target == false

The following are irrelevant to success:

remaining untargeted active tiles;

remaining inactive sockets;

where surviving tiles are located;

how many swaps were used;

how many non-target tiles were deactivated.

On success, the reactor becomes terminal and later actions are no-ops.

7. Animation and feedback

All logical consequences of one action belong to the same step() call. Animation subframes do not accept input and do not consume actions or the external horizon.

The official state after the call is always the fully settled stable state.

7.1 Adjacent swap

A valid click produces:

One frame with the clicked connector in color 7.

One frame showing the exchanged tile contents.

Match processing, if applicable.

The logical swap is committed before the first match scan.

For renderers without subframe playback, the returned stable frame retains the clicked connector in color 7 for that observation.

7.2 Match removal

For every cascade round:

Overlay every member of the simultaneous removal set in color 0 for one frame.

Remove all marked tiles together.

Begin settlement.

This makes intersections and unrelated simultaneous runs visually distinguishable from sequential clearing.

7.3 Settlement

If one or more tiles move downward:

calculate every tile’s final row first;

animate all falling tiles synchronously;

advance each moving tile by at most one logical row per subframe;

do not scan for matches at intermediate positions;

commit the settled state after the final fall subframe;

only then scan for the next cascade round.

7.4 Invalid action

An invalid action leaves the reactor unchanged and renders the outer frame in color 8 for one returned observation.

7.5 Completion

When the last target disappears, the frame alternates between colors 14 and 0 for two subframes and then remains color 14.

8. Authored progression

* in the documentation means that the tile carries a target ring. . means an inactive socket.

Level	Name	Size	Main discovery	Reference clicks	Clear rounds
1	The Exchange	2×3	Connector click, swap, match, visible objective	1	1
2	Remove the Supports	5×3	A target can survive and move during settlement	2	2
3	No Pause Between Reactions	5×3	Newly formed matches resolve in the same step	1	2
4	Replace the Support	5×5	Stage payload and sacrificial support before ignition	4	2
5	One Flash, Many Arms	6×6	Intersecting and unrelated runs clear simultaneously	4	6
6	Stage Before Ignition	5×6	Near-matches must be prepared before an immediate clear	4	3
7	Read the Reactor Twice	6×6	Full composition; a familiar cascade becomes only the opening	16	9

The click counts are not padded with reversible transit. Later depth comes from irreversible staging choices and multi-round consequences. In particular, the final level has sixteen deliberate clicks and nine separate clear rounds.

9. Level 1 — The Exchange
Player-facing discovery

The target ring surrounds a magenta X tile. Another magenta X is on the same row, while a third sits directly beneath the blocking blue tile.

The target cannot currently be part of a trio. The connector between the misplaced lower magenta tile and the blocking blue tile is the most local, visually supported experiment.

Clicking it immediately exchanges the tiles, creates a row of three magenta X tiles, removes the ringed tile, and completes the level.

This establishes the complete foundational causal chain without unrelated objects.

State-machine detail
Size: 2 rows × 3 columns
Targets: 1
Initial selection: none
Initial feedback: none

       c1  c2  c3
r1     A*  B   A
r2     .   A   .

The initial board contains no match.

Reference solution
1. V(1,2)

This swaps:

(1,2): B
(2,2): A

The temporary swapped board is:

A*  A  A
.   B  .

Cascade round 1 detects:

horizontal A run: (1,1), (1,2), (1,3)

All three A tiles deactivate simultaneously. The target disappears. The remaining B tile occupies (2,2) after settlement.

Win validation
remaining target count = 0
terminal = true
Attribution and anti-degeneracy

The only target is visibly one swap away from a trio.

Every active tile participates in either the swap or the resulting match.

V(1,2) is the unique one-click winning connector.

No direction action, tile-body click, or background click can imitate the result.

The connector pulse makes even an incorrect experiment attributable to the clicked edge.

10. Level 2 — Remove the Supports
Player-facing discovery

The ringed A tile is far above two untargeted A companions at the bottom corners. Directly beneath the target is a nearly complete vertical D run.

The ordinary A match is impossible from the initial arrangement. The nearby productive experiment is instead to complete the D column.

Doing so removes three non-target supports. The ringed A visibly falls but does not disappear, proving that:

rings move with tiles;

falling is not deactivation;

removing unrelated tiles can reposition the objective.

The target then sits immediately above the bottom-center C tile. Swapping those two cells forms the required A row.

State-machine detail
Size: 5 rows × 3 columns
Targets: 1
Initial selection: none
Initial feedback: none

       c1  c2  c3
r1     .   A*  .
r2     .   D   .
r3     D   B   .
r4     .   D   .
r5     A   C   A

The initial board contains no match.

Reference solution
1. H(3,1)
2. V(4,2)
Action 1: H(3,1)

Swap (3,1)=D with (3,2)=B.

Column 2 becomes:

r1 A*
r2 D
r3 D
r4 D
r5 C

Cascade round 1 removes:

vertical D run: (2,2), (3,2), (4,2)

After downward settlement:

       c1  c2  c3
r1     .   .   .
r2     .   .   .
r3     .   .   .
r4     B   A*  .
r5     A   C   A

The target ring is now visibly at (4,2).

Action 2: V(4,2)

Swap (4,2)=A* with (5,2)=C.

The bottom row becomes:

A  A*  A

Cascade round 1 of this step removes:

horizontal A run: (5,1), (5,2), (5,3)

The target disappears.

Attribution and anti-degeneracy

No one-click solution exists.

The first click clears no target, so the player cannot mistake support removal for direct objective satisfaction.

The ring’s fall from (1,2) to (4,2) is immediate and unambiguous.

The bottom C tile is necessary: it prevents the target from initially settling into the final row and provides the exchange partner for the second click.

The D at (3,1) and B at (3,2) create the sole local one-swap completion of the support column.

11. Level 3 — No Pause Between Reactions
Player-facing discovery

This level deliberately resembles Level 2. The same D support column can be completed with the same local exchange.

The difference is below it: the two A companions are already supported at row 4.

When the D run disappears, the ringed A falls directly between them. The new A trio flashes and deactivates before another action can be taken.

This distinguishes the intended rule from plausible alternatives such as:

only one match resolves per click;

the board settles but does not recheck;

the player must click again to continue a reaction.

State-machine detail
Size: 5 rows × 3 columns
Targets: 1
Initial selection: none
Initial feedback: none

       c1  c2  c3
r1     .   A*  .
r2     .   D   .
r3     D   B   .
r4     A   D   A
r5     B   C   D

The initial board contains no match.

Reference solution
1. H(3,1)
Cascade round 1

After swapping the D and B at row 3, remove:

vertical D run: (2,2), (3,2), (4,2)

After settlement, the board temporarily becomes:

       c1  c2  c3
r1     .   .   .
r2     .   .   .
r3     B   .   .
r4     A   A*  A
r5     B   C   D
Cascade round 2

Before returning control, detect and remove:

horizontal A run: (4,1), (4,2), (4,3)

The target disappears and the level completes.

Attribution and anti-degeneracy

H(3,1) is the unique one-click win.

The first reaction is visually identical in kind to Level 2’s support clear.

The new evidence is specifically the second flash and deactivation during the same step.

Bottom-row B, C, and D tiles support the two side A tiles so that they do not fall away before the target arrives.

No new object type is introduced.

12. Level 4 — Replace the Support
Player-facing discovery

The reactor contains exactly three A tiles:

the ringed A near the top;

two potential companions in the lower-right portion.

They cannot be brought together directly without disturbing the structure that must place them.

A vertical C support run is almost present in column 3, but completing it too early would not put the A tiles in the correct relative positions. The player must first stage both the falling target and the two surviving companions.

The final swap has a double meaning:

it moves an A companion out of the trigger column and into the future target row;

it moves a C tile into that column, completing the sacrificial support run.

The C run disappears, the target falls into the vacated location, and the prepared A row reacts automatically.

State-machine detail
Size: 5 rows × 5 columns
Targets: 1
Initial selection: none
Initial feedback: none

       c1  c2  c3  c4  c5
r1     B   A*  B   D   B
r2     C   C   D   B   C
r3     D   B   C   B   C
r4     D   A   D   C   A
r5     C   B   B   D   D

The initial board contains no match.

Reference solution
1. H(1,2)
2. H(2,2)
3. H(4,2)
4. H(4,3)
Staging actions

After actions 1–3, no match has occurred. The exact board is:

       c1  c2  c3  c4  c5
r1     B   B   A*  D   B
r2     C   D   C   B   C
r3     D   B   C   B   C
r4     D   D   A   C   A
r5     C   B   B   D   D

The three important structures are now visible:

A target at (1,3);

two C tiles already at (2,3) and (3,3);

A companions at (4,3) and (4,5) with C at (4,4).

Trigger action: H(4,3)

Swap:

(4,3): A
(4,4): C

Row 4 becomes:

D  D  C  A  A
Cascade round 1

Remove:

vertical C run: (2,3), (3,3), (4,3)

The ringed A falls within column 3 from (1,3) to (4,3).

The board now contains:

row 4, columns 3–5: A* A A
Cascade round 2

Remove:

horizontal A run: (4,3), (4,4), (4,5)

The target disappears.

Necessity and anti-cleanliness

There are exactly three A tiles, so every A is required for the target clear.

The target is not touched by the trigger click.

The C run is useful only after the target and companions have been staged.

The A at (4,3) is first a payload tile and then must vacate the support column at the trigger moment.

The C at (4,4) is initially a separator and later becomes the final sacrificial support.

Exhaustive search over all connector sequences finds no win in three or fewer clicks.

The solution depends on the irregular geometry of column 3 and row 4; it cannot be replaced by a corridor or independent checklist.

13. Level 5 — One Flash, Many Arms
Player-facing discovery

Several ringed B and D tiles are distributed around an asymmetrical 6×6 reactor.

An ordinary D clear near the top first changes the relative heights of several columns. Two subsequent swaps stage the bottom B tiles and preserve the right-side payload. The final exchange creates a long B run along the bottom.

That run starts a five-round cascade. During its third round, a horizontal B run and a vertical B run intersect while a separate D run also exists. All eight matched cells flash and vanish together before any settling occurs.

The intersecting B footprint is not ornamental: clearing only one arm first would produce a different settlement and leave a ringed B alive.

State-machine detail
Size: 6 rows × 6 columns
Targets: 5
Initial selection: none
Initial feedback: none

       c1  c2  c3  c4  c5  c6
r1     A   D   B   D   A   D*
r2     C   D   C   D   D   C
r3     C   A   B   B   D   D
r4     A   C   D   D   B   A
r5     A   D*  B*  D   B   A
r6     D   B*  C   B   D   B*

For implementation discussion only, name the targets by initial position:

T1 = D* at (1,6)
T2 = D* at (5,2)
T3 = B* at (5,3)
T4 = B* at (6,2)
T5 = B* at (6,6)
Reference solution
1. H(2,2)
2. H(2,5)
3. H(6,2)
4. V(5,5)
Action 1: open the unequal columns

H(2,2) exchanges D and C, producing:

row 2: C C D D D C

Remove the D run at (2,3)–(2,5) and settle.

Result:

       c1  c2  c3  c4  c5  c6
r1     A   D   .   .   .   D*
r2     C   C   B   D   A   C
r3     C   A   B   B   D   D
r4     A   C   D   D   B   A
r5     A   D*  B*  D   B   A
r6     D   B*  C   B   D   B*
Actions 2–3: stage the delayed reaction
2. H(2,5)

This swaps the A and C at (2,5) and (2,6).

3. H(6,2)

This swaps target T4 with the C at (6,3), placing T4 at (6,3).

Immediately before the final trigger:

       c1  c2  c3  c4  c5  c6
r1     A   D   .   .   .   D*
r2     C   C   B   D   C   A
r3     C   A   B   B   D   D
r4     A   C   D   D   B   A
r5     A   D*  B*  D   B   A
r6     D   C   B*  B   D   B*
Action 4: V(5,5)

Swap (5,5)=B with (6,5)=D.

The bottom row becomes:

D  C  B*  B  B  B*

This starts the following exact cascade.

Cascade round	Simultaneous removal set	Targets removed
1	B at (6,3)–(6,6)	T4, T5
2	D at (5,2)–(5,4)	T2
3	B union {(4,3),(5,3),(5,4),(5,5),(6,3)} and D at (4,4)–(4,6)	T3
4	A at (4,6),(5,6),(6,6)	none
5	D at (6,4)–(6,6)	T1
The round-3 intersection

At the start of round 3, the B matches are:

vertical:   (4,3), (5,3), (6,3)
horizontal: (5,3), (5,4), (5,5)

Their union contains five cells and forms a T-shaped intersection.

At the same moment, a separate D run exists at:

(4,4), (4,5), (4,6)

The implementation must therefore flash and remove all eight cells before settling anything.

Necessity and anti-cleanliness

The five targets are cleared by four different states of the same cascade, not by five independent errands.

Action 1 is not a target clear; it changes the offsets required for later rounds.

Actions 2 and 3 do not immediately match anything. They prepare surviving tiles for post-settlement roles.

The final clicked edge is not adjacent to four of the five initial targets.

The intersecting B arms are operationally necessary; they clear T3 and alter three columns simultaneously.

Exhaustive search finds no win in three or fewer clicks.

The level uses no new object type. Its new mechanic is a stricter interpretation of the already established simultaneous-removal rule.

14. Level 6 — Stage Before Ignition
Player-facing discovery

Two targets are present:

a C target embedded in an almost complete vertical support;

an A target on the upper-right side.

The visible C match is tempting, but completing it immediately commits the reactor to settling. Any trio that becomes complete reacts at once; it cannot be stored for later.

The player must first arrange two near-matches in the upper rows. Only then should the C support be ignited.

After the C target clears, an inactive socket appears between a B tile and two other B tiles. The nearby connector remains visibly active. Swapping the B sideways into that socket completes a B run, whose removal then aligns the A target for a second reaction.

This is the first level in which an empty socket becomes an active transport resource rather than merely the residue of a clear.

State-machine detail
Size: 5 rows × 6 columns
Targets: 2
Initial selection: none
Initial feedback: none

       c1  c2  c3  c4  c5  c6
r1     C   A   B   B   A   B
r2     C   B   C   B   B   A*
r3     B   D   C   C   B   C
r4     D   A   A   C*  D   D
r5     C   C   D   B   B   C

The initial board contains no match.

Reference solution
1. H(1,2)
2. H(1,3)
3. H(4,3)
4. H(2,2)
Actions 1–2: stage the upper payload

After H(1,2) and H(1,3), row 1 is:

C  B  B  A  A  B

The exact board is:

       c1  c2  c3  c4  c5  c6
r1     C   B   B   A   A   B
r2     C   B   C   B   B   A*
r3     B   D   C   C   B   C
r4     D   A   A   C*  D   D
r5     C   C   D   B   B   C

No match has yet occurred.

The upper B and A structures are both incomplete, so they can survive until the support clear.

Action 3: H(4,3)

Swap (4,3)=A with (4,4)=C*.

Column 3 becomes:

r1 B
r2 C
r3 C
r4 C*
r5 D

Cascade round 1 removes:

vertical C run: (2,3), (3,3), (4,3)

The C target disappears.

After settlement:

       c1  c2  c3  c4  c5  c6
r1     C   B   .   A   A   B
r2     C   B   .   B   B   A*
r3     B   D   .   C   B   C
r4     D   A   B   A   D   D
r5     C   C   D   B   B   C

The local impasse is now explicit:

row 2, columns 2–5: B . B B

There is a visible connector between the left B and the socket.

Action 4: H(2,2)

Swap (2,2)=B with the inactive socket at (2,3).

This produces:

row 2, columns 3–5: B B B
Cascade round 1

Remove:

B at (2,3), (2,4), (2,5)

After settlement, row 2 contains:

columns 4–6: A A A*
Cascade round 2

Remove:

A at (2,4), (2,5), (2,6)

The final target disappears.

Necessity and anti-cleanliness

The first two clicks arrange two different future reactions before any immediate reward.

The C target acts both as an objective and as the sacrificial support whose removal opens the B socket.

The empty socket is not decorative: the final B run cannot be completed without swapping through it.

Gravity does not move the B sideways; the player must use the connector.

The B clear is not an objective clear, but it is required to align the A target.

Exhaustive search finds no win in three or fewer clicks.

The level reuses immediate deactivation as a liability: completing the C run before staging the upper payload loses the compact cascade setup.

15. Level 7 — Read the Reactor Twice
Player-facing discovery

This level intentionally begins with the same tile arrangement as Level 5, but two additional rings are present:

an A target at the top-left;

a C target at the upper-right.

The familiar four-click intersection cascade is still useful, but it is no longer a complete solution.

The extra C ring changes the meaning of an already familiar staging swap: it must be moved before the broad cascade so that it survives and lands in a useful bottom socket.

After the opening cascade, the formerly dense reactor becomes a sparse transport space. The player must:

bring two C branches together to clear the C target;

use the remaining unringed C tiles as a sacrificial support clear;

move the final A target and its last companion along separate branches;

join those branches for the final reaction.

The familiar mechanism therefore changes roles several times without changing its operational rule.

State-machine detail
Size: 6 rows × 6 columns
Targets: 7
Initial selection: none
Initial feedback: none

       c1  c2  c3  c4  c5  c6
r1     A*  D   B   D   A   D*
r2     C   D   C   D   D   C*
r3     C   A   B   B   D   D
r4     A   C   D   D   B   A
r5     A   D*  B*  D   B   A
r6     D   B*  C   B   D   B*

This is exactly Level 5’s tile arrangement, with additional rings on:

A at (1,1)
C at (2,6)

The initial board contains no match.

Reference solution
 1. H(2,2)
 2. H(2,5)
 3. H(6,2)
 4. V(5,5)

 5. H(6,4)
 6. H(6,3)
 7. V(5,1)
 8. H(6,1)
 9. V(5,2)

10. H(3,1)
11. H(3,2)
12. V(3,3)

13. V(3,1)
14. V(4,1)
15. H(6,2)
16. V(5,3)
Phase 1: the familiar cascade becomes an opening

Actions 1–4 are structurally the same as Level 5:

1. H(2,2)
2. H(2,5)
3. H(6,2)
4. V(5,5)

Action 2 now has an additional role: it moves the C target from (2,6) to (2,5) before the broad cascade.

The five-round cascade clears the five targets inherited from Level 5. The A and C targets survive.

After complete settlement, the exact board is:

       c1  c2  c3  c4  c5  c6
r1     A*  .   .   .   .   .
r2     C   D   .   .   .   .
r3     C   C   .   .   .   .
r4     A   A   .   .   .   .
r5     A   C   .   .   .   .
r6     D   C   .   .   C*  .

Remaining targets:

A* at (1,1)
C* at (6,5)

The old solution is visibly insufficient, but it has opened the transport geometry needed for the rest of the level.

Phase 2: converge two C branches

Two independent preparations are visible:

move C* left through bottom-row sockets;

rearrange the lower-left cells so that two ordinary C tiles can meet it.

The two branches may be partially interleaved.

Move the target branch
5. H(6,4)
6. H(6,3)

These swaps move C*:

(6,5) → (6,4) → (6,3)

No match occurs and no gravity runs. The target remains exactly where each swap places it.

Prepare the donor branch
7. V(5,1)
8. H(6,1)

Action 7 exchanges the A and D at the bottom of column 1.

Action 8 exchanges the resulting bottom-left A with the C at (6,2), producing:

bottom row, columns 1–3: C A C*
Converge the branches
9. V(5,2)

Swap the C at (5,2) with the A at (6,2).

The bottom row becomes:

C C C*

Remove the horizontal C run. The C target disappears.

After settlement:

       c1  c2  c3  c4  c5  c6
r1     .   .   .   .   .   .
r2     A*  .   .   .   .   .
r3     C   D   .   .   .   .
r4     C   C   .   .   .   .
r5     A   A   .   .   .   .
r6     D   A   .   .   .   .

Only the A target remains.

Phase 3: turn the remaining C tiles into sacrificial support

The C target is gone, but three ordinary C tiles remain. Their role changes from possible companions to disposable support.

10. H(3,1)
11. H(3,2)
12. V(3,3)

After actions 10 and 11, the C from (3,1) has been moved to (3,3) through the socket at (3,2).

Action 12 moves it down to (4,3).

This completes:

horizontal C run: (4,1), (4,2), (4,3)

The three C tiles deactivate and all columns settle.

Exact resulting board:

       c1  c2  c3  c4  c5  c6
r1     .   .   .   .   .   .
r2     .   .   .   .   .   .
r3     A*  .   .   .   .   .
r4     D   .   .   .   .   .
r5     A   A   .   .   .   .
r6     D   A   .   .   .   .

The C clear has moved the target A from row 2 to row 3 and exposed the final staging space.

Phase 4: converge the final A branches

Two preparations can again be reasoned about separately:

lower the target A into row 5;

move the bottom A companion into column 3.

Target branch
13. V(3,1)
14. V(4,1)

The target moves:

(3,1) → (4,1) → (5,1)

After action 14, row 5 begins:

A* A .
Companion branch
15. H(6,2)

Move the bottom A from (6,2) to (6,3) through an inactive socket.

Final convergence
16. V(5,3)

Swap the inactive socket at (5,3) with the A at (6,3).

Row 5 becomes:

A* A A

Remove the horizontal A run. The final target disappears and the level completes.

Dependency structure

The reference path has phases, but it is not a single rigid chain:

Actions 5–6 and actions 7–8 are two branches that can be interleaved; action 9 requires both.

After the sacrificial C clear, target-placement actions 13–14 and companion-placement action 15 are separate preparations; action 16 requires both.

The C tiles change from target companions to sacrificial supports.

The empty sockets change from cascade residue to transport lanes.

The Level 5 intersection changes from a complete solution to a controlled demolition that creates the second half of the puzzle.

Necessity and anti-degeneracy

The inherited four-click cascade leaves exactly two rings, so replaying the old recipe cannot win.

Action 2 must account for the new C target before the large cascade changes the board.

The C target cannot be cleared without combining the right-side target branch and left-side donor branch.

The final A alignment depends on the later unringed C clear.

The A target and final A companion approach their meeting row through different routes.

All three post-cascade C tiles are consumed.

The remaining D and A tiles provide necessary supports and exchange partners during the final positioning.

Exhaustive search finds no win in four or fewer clicks.

The reference route uses sixteen meaningful clicks and nine deterministic clear rounds, with no repeated no-op movement.

16. Mechanic reuse in contrasting roles
Mechanic	Early role	Later contrasting roles
Three-tile match	Directly clears the target in Level 1	Removes unrelated support in Levels 2 and 4; opens sockets in Level 6; synchronizes many columns in Levels 5 and 7
Downward settlement	Brings a target toward companions in Level 2	Automatically finishes a cascade in Level 3; must be delayed until payloads are staged in Levels 4 and 6
Target ring	Marks the immediate payload in Level 1	Survives and moves in Level 2; must be protected through a broad cascade in Level 7
Untargeted run	Incidental in the foundation	Sacrificial trigger in Levels 4, 6, and 7
Empty socket	Visible aftermath of clearing	Exchange destination and lateral transport resource in Levels 6 and 7
Simultaneous removal	Background consistency rule in early levels	Necessary intersecting-clear mechanic in Level 5; board-clearing opening tool in Level 7
Long cascade	Immediate reward in Level 3	Must be engineered through offsets in Level 5; useful but insufficient opening in Level 7

No mechanic changes its operational behavior between levels. Strategic differences arise entirely from geometry, timing, target placement, and which tiles must survive earlier reactions.

17. Epistemic curriculum audit
Level 1

The same-state arrangement identifies one local connector experiment. The result combines swap, match, and success in one immediately attributable action.

Level 2

The target cannot initially match. The nearly complete support run is directly beneath it. Completing that run visibly moves the ring without removing it, separating settlement from deactivation.

Level 3

The opening arrangement reuses Level 2’s evidence. The only changed consequence is that the landing position already completes a trio. The second reaction occurs before control returns.

Level 4

Known rules permit staging without immediate reaction. The player sees one target column, two C supports, and two A companions. The final swap visibly replaces an A with the C that triggers the fall.

Level 5

The final action produces a large visible match. During the cascade, both arms of the B intersection and the separate D run flash together, directly demonstrating simultaneous union before settlement.

Level 6

The C clear creates a local B . B B impasse. The connector next to the socket remains visibly active. Clicking it produces an immediate B reaction and then an A reaction.

Level 7

The player recognizes Level 5’s board but sees two extra objective rings. After the known cascade, the sparse board exposes two local branch-convergence problems rather than introducing an unfamiliar object or hidden rule.

18. Reference implementation assertions

A conforming implementation should satisfy all of the following tests.

All levels:
    reset board contains no match
    reset terminal == false
    reset selection == null
    settle direction == DOWN
Level 1:
    V(1,2) produces exactly 1 clear round
    target count after step == 0
Level 2:
    no single connector click wins
    after H(3,1), target is at (4,2)
    after subsequent V(4,2), target count == 0
Level 3:
    H(3,1) produces exactly 2 clear rounds in one step
    target count after step == 0
Level 4:
    first three reference clicks produce no clear
    H(4,3) produces exactly 2 clear rounds
    first round is C at column 3, rows 2–4
    second round is A at row 4, columns 3–5
Level 5:
    H(2,2) produces 1 clear round
    H(2,5) and H(6,2) produce no clear
    V(5,5) produces exactly 5 clear rounds
    its round-3 removal union contains exactly 8 cells
    final target count == 0
Level 6:
    first two reference clicks produce no clear
    H(4,3) clears the C target and leaves (2,3) empty
    H(2,2) swaps B into that socket
    that step produces exactly 2 further clear rounds
Level 7:
    after reference action 4, remaining targets are:
        A* at (1,1)
        C* at (6,5)

    after action 9, remaining target is:
        A* at (2,1)

    after action 12, remaining target is:
        A* at (3,1)

    after action 16:
        target count == 0
        terminal == true

These assertions, together with the global update order and exact initial matrices, fully determine all seven authored instances without requiring implementation-specific interpretation.
