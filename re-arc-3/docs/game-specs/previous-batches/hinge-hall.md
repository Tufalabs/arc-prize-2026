Hinge Hall

This is a game about pushing hinged walls, becoming difficult because opening one passage makes the wall occupy another.

Authored level count: exactly 7.

Hinge Hall is a movement puzzle in which every movable wall is both an obstruction and a piece of route geometry. A push never removes a wall: it rotates the wall by ninety degrees around a fixed hinge. The player therefore solves levels not by opening everything, but by deciding where each wall should obstruct, whether its handle will remain reachable, and whether another wall needs its sweep space first.

The only meaningful controls are UP, DOWN, LEFT, and RIGHT. SPACE and clicks are accepted by the action interface but have no gameplay effect. There are no hidden clickable regions.

1. Player-facing rules

The player sees:

a blue avatar;

a recessed green exit;

dark static walls;

movable gray wall leaves;

large yellow-orange hinge pivots;

a yellow-orange handle at each leaf’s free end;

faint light-blue sweep guides around each hinge.

Walking into ordinary empty floor moves the player one logical cell.

Walking sideways into the handle at the free end of a leaf attempts to rotate that leaf ninety degrees in the direction of movement. If the complete destination and swept area are empty, the leaf rotates and the player enters the cell that the handle vacated.

A leaf cannot:

pass through a static wall;

pass through another leaf;

pass through another hinge;

push another leaf as part of its sweep;

rotate when struck on an inner segment;

rotate when pushed radially along its own length.

A failed rotation leaves everything unchanged and shows exactly what blocked the attempt.

The objective is simply:

Move the player onto the visible green exit cell.

No final leaf arrangement is checked independently. The necessary arrangement emerges from the geometry.

2. Logical geometry
2.1 Coordinate system

Every level uses a rectangular logical lattice.

Origin: (0,0) at the upper-left.

Positive x: right.

Positive y: down.

Directions:

Name	Vector
N	(0,-1)
E	(1,0)
S	(0,1)
W	(-1,0)

N, E, S, and W are documentation abbreviations only. No letters are rendered in the game.

Each logical cell renders as a 4×4 pixel area.

2.2 Leaf representation

A wall leaf has:

a fixed hinge coordinate h;

a length L, either 2 or 3;

a current orientation vector d ∈ {N,E,S,W}.

The hinge itself occupies h and is impassable.

The leaf occupies:

leafCells(h,L,d)={h+kd∣1≤k≤L}.

Its handle, or free end, is:

f=h+Ld.

All occupied leaf cells are collision cells. The handle is a visual marking on the final occupied cell, not a separate logical object.

Leaves are reversible. They have no locks, ownership, activation state, or one-way exceptions.

2.3 Tangential pushing

For a directional action with movement vector m, let:

q=p+m

where p is the player position before the action.

A rotation is attempted only when all of the following are true:

q is the free-end cell of one leaf.

m is perpendicular to the leaf’s current orientation:

d⋅m=0.

Equivalently, the player is standing at f-m and moves toward f.

The candidate new orientation is exactly m.

For example, for a north-pointing leaf:

standing west of its handle and pressing RIGHT attempts N→E;

standing east of its handle and pressing LEFT attempts N→W;

approaching from north or south does not rotate it.

On a successful rotation:

the leaf orientation becomes m;

the player position becomes the old free-end coordinate f.

Thus the player steps into the cell vacated by the handle.

Conceptual example, not rendered:

Before N→E, length 2          After

      handle                  player
player → leaf                   ·
         leaf                   ·
         hinge                  hinge—leaf—handle
3. Discrete swept-area rule

Checking only the final wall position would make rotations appear to pass through one another. Hinge Hall therefore uses an explicit discrete swept envelope.

For a rotation from orientation d to perpendicular orientation m, define:

Destination cells
D={h+km∣1≤k≤L}.
Interior sweep cells
I={h+ad+bm∣1≤a≤L−1,1≤b≤L−1}.
Complete clearance mask
M=D∪I.

The rotation succeeds only if every coordinate in M:

is a floor coordinate in the authored map;

contains no other hinge;

contains no cell belonging to another leaf.

The current cells of the leaf being pushed are ignored during its own clearance test. The player’s current position is not a blocker; for a valid tangential approach it lies outside M anyway.

This rule gives a length-2 leaf a three-cell mask and a length-3 leaf a seven-cell mask.

Important consequence

A rotation may fail even when every final destination cell looks empty. Another leaf can obstruct only an interior sweep cell. Level 3 forces the player to distinguish this from simple destination collision.

4. Exact action semantics

The action set is exactly:

UP

DOWN

LEFT

RIGHT

SPACE

click on any rendered cell

Only the four directional actions are meaningful.

4.1 Directional action resolution

For a directional action with vector m, resolution is:

Clear the previous transient failure feedback.

Compute q = player + m.

If q is floor and contains no hinge or leaf cell:

move the player to q;

check victory.

Otherwise, if q is the free end of a leaf and m is perpendicular to that leaf:

construct the candidate sweep mask;

collect all blockers;

if there are no blockers, atomically rotate the leaf and move the player to q;

otherwise, leave the logical state unchanged and display failure feedback.

Otherwise:

leave the logical state unchanged;

display ordinary collision feedback.

After an accepted move or push, set complete=true exactly when the player coordinate equals the goal coordinate.

There is no push priority ambiguity because valid authored states never contain overlapping leaves.

4.2 Inner-leaf and radial contact

If the target is:

an inner leaf segment;

a hinge;

a handle approached parallel to the leaf;

a static wall;

outside the authored floor;

the action is blocked. It never translates the leaf and never causes a partial rotation.

4.3 SPACE

SPACE leaves all logical state unchanged. It produces a brief neutral gray player outline so the action is visibly acknowledged.

4.4 Clicks

CLICK(x,y) leaves all logical state unchanged for every coordinate.

There are:

no clickable leaves;

no clickable hinges;

no clickable floor regions;

no selected-object state;

no cursor state stored by the game.

No object is styled as a button or click target.

5. State and update contract

The persistent logical state is:

level_index
player_position
leaf_orientation[leaf_id]
complete

The map, hinge coordinates, leaf lengths, and goal are immutable level data.

Transient render feedback may contain:

blocked_move_coordinate
attempted_leaf_id
attempted_orientation
blocking_coordinates
pre_rotation_snapshot
post_rotation_snapshot

Transient feedback is not used by the validator.

All logical changes occur inside step(action). Animation frames never perform additional logical updates.

There are no:

timers;

energy values;

move budgets;

limited retries;

hidden phases;

random events;

automatic leaf movement;

delayed actions;

queue state.

Exploration is unlimited in every level.

6. Victory validation

A level succeeds exactly when:

complete = (player_position == goal_position)

Relevant:

the player’s final logical coordinate.

Ignored:

leaf orientations;

number of actions;

number of failed actions;

whether earlier routes remain open;

transient feedback;

animation frame;

use of SPACE or clicks.

Allowed:

the goal may initially be covered by a leaf;

a push may expose the goal;

the final route behind the player may be closed;

a successful push may itself land the player on the goal.

Forbidden only through ordinary collision:

entering a wall;

entering a hinge;

entering an occupied leaf cell without a successful handle push;

rotating through a blocked sweep mask.

After completion, further actions leave the completed state unchanged. Level advancement is handled by the host environment.

7. Rendering contract
7.1 Canvas and board placement

The render canvas is always 64×64.

For a logical board of width W and height H, each logical cell is 4×4 pixels and the board origin is:

o
x
	​

=⌊
2
64−4W
	​

⌋,o
y
	​

=⌊
2
64−4H
	​

⌋.

No borders are drawn between logical cells. Adjacent floor and wall cells appear as contiguous regions.

7.2 Colors

Only these integer color IDs are used:

Element	Color ID
White player center / completion glint	0
Floor	1
Movable leaf body	2
Static wall	3
Outside-board background	4
Failed push ghost / collision	8
Player body	9
Sweep guides	10
Hinge and handle outer material	11
Hinge center / blocker emphasis	12
Goal and completion pulse	14

Colors 5, 6, 7, 13, and 15 are unused.

7.3 Drawing order

Outside background.

Static wall and floor regions.

Goal inlay.

Sweep guides.

Leaf bodies.

Hinge pivots and free-end handles.

Player.

Failure or completion feedback.

7.4 Entity silhouettes
Static wall

Every # cell is filled 4×4 with color 3. Adjacent wall cells have no internal outlines.

Floor

Every authored floor cell is filled 4×4 with color 1.

Leaf

Every logical leaf cell is a filled 4×4 block of color 2, producing a connected 2×1, 3×1, 1×2, or 1×3 structure.

No per-cell border is drawn inside a leaf.

Hinge

The hinge cell uses a large brass disk:

yellow 11 around the outside;

orange 12 in the central 2×2 pixels.

The hinge is visually larger and warmer than the handle, making the fixed end unambiguous.

Handle

The free-end cell receives a yellow 11 cap with a small orange 12 center on its outward-facing edge. The cap rotates with the leaf. It is the only part of a leaf that invites player contact.

Player

The player is a compact blue 9 shape with a white 0 center. It remains visually distinct from sweep guides and the gray wall family.

Goal

The goal is a green 14 recessed floor frame. Its four corner pixels remain visible even when a leaf occupies the cell, so an initially covered goal is still identifiable.

This overlay is visual only; a covered goal remains physically occupied.

8. Sweep guides and feedback
Sweep guides

For each leaf, consider both perpendicular candidate orientations.

A guide is drawn when the candidate sweep mask is compatible with the static floor geometry and other fixed hinges, ignoring the current orientations of other movable leaves. Therefore:

a sweep that is permanently embedded in static wall has no guide;

a sweep currently blocked by another leaf still has its guide;

overlapping guides expose likely dependencies.

The guide is a one-pixel light-blue 10 outline around the union of the discrete sweep-mask cells. It has no per-cell grid borders.

Successful rotation animation

A successful push uses five deterministic render-only poses:

0°

22.5°

45°

67.5°

90°

The leaf is rendered as a rigid rectangle pivoting around the hinge center.

The logical final state is committed by the triggering step() before another action can be accepted. The intermediate poses interpolate cached pre-step and post-step states and consume no player actions or budget. The player appears in the old handle cell on the final pose.

Blocked push feedback

For four render-only frames:

the attempted final leaf is outlined with a red 8 alternating-pixel ghost;

all blocking cells are outlined in orange 12;

a blocking leaf or hinge briefly receives an orange perimeter;

the actual leaf remains in its original orientation.

When only an interior sweep cell is blocked, the red final outline may appear unobstructed while the actual blocker is highlighted inside the sweep guide. This is intentional evidence for the swept-area rule.

Blocked movement

A non-push collision displays a two-frame red 8 contact outline on the struck cell.

Completion

Completion produces an eight-frame green 14 pulse around the goal and player. It consumes no action and does not alter leaf state.

9. Progression
Level	Main discovery	Contrasting reuse	Verified depth
1	Tangential handle push	Opening one side occupies another	10
2	Continued control requires the correct handle side	Same leaf is first an entrance gate, then an exit gate	12
3	Interior sweep space matters	Two independent leaves jointly block one long sweep	19
4	A route should sometimes be closed deliberately	Leaf is both bridge control and commitment barrier	28
5	A correct wall must be moved away and restored	A leaf is first a blocker, then a staging tool; another blocks a handle position	40
6	A leaf can open one corridor while sealing the route just used	A delayed rotation depends on another wall’s previous state	46
7	Nested sweep dependencies require a forced revisit	A final leaf clears one push while cutting off the starting side	60

Only one object family is introduced: hinged wall leaves. Length 3 is introduced in Level 3 and then repeatedly reused. Later levels add consequences and dependency structures rather than unrelated devices.

10. Authored levels
Documentation map legend

The following maps are implementation data, not rendered symbols.

#  non-floor static wall
.  floor
S  floor containing the initial player position
G  floor containing the goal

Rows are listed from y=0 downward. Characters within a row are listed from x=0 rightward.

Hinges and leaves are overlaid using the tables below. Every hinge and leaf cell lies on authored floor.

Solution traces use:

U = UP
D = DOWN
L = LEFT
R = RIGHT

Every action in each trace is accepted; failed experiments are not counted.

Level 1 — The wall does not disappear

Intended depth: 10 actions

Player-facing discovery

The player begins in the upper-right room and can see the green exit in the lower-left room. The central opening is occupied by a vertical movable leaf. The leaf’s handle is immediately reached from its right side, and its leftward sweep guide enters an empty alcove.

Ordinary movement reaches a local impasse at the handle. Pressing LEFT rotates the leaf west and immediately creates a crossing. At the same time, the settled wall visibly occupies the west side of the lower chamber: pushing has relocated the obstruction rather than removed it.

This is sufficient first-level goal evidence because the blue avatar, visible green floor exit, and physically blocked route make the objective directly legible.

State-machine detail

Board: 9×9

Pixel origin: (14,14)

Start: (7,2)

Goal: (1,6)

Selection/cursor state: none

#########
#...#...#
#...#..S#
#.......#
#.......#
#.......#
#G..#...#
#...#...#
#########
Leaf	Hinge	Length	Initial orientation	Initial occupied cells	Initial free end
A	(4,5)	2	N	(4,4), (4,3)	(4,3)
Verified solution
DLLLDLLLDD

Structured trace:

DLL
L  — push A: N→W
DLLLDD

The push occurs from (5,3). The player lands at the vacated handle coordinate (4,3).

Necessity and anti-degeneracy

The static divider at x=4 blocks both the upper and lower bypasses. Movement alone cannot reach the left chamber.

If pushes of A are disabled, exhaustive state search finds no solution. No other object or alternate route can explain the crossing, so the tutorial specifically demonstrates hinged rotation.

Level 2 — Keep control of the handle

Intended depth: 12 actions

Player-facing discovery

The first push opens the entrance into the central space, but the same leaf settles horizontally across the visible route toward the exit. The player has not “finished with” the wall.

After entering, the player reaches the handle from above. A downward push rotates the wall again, this time parking it in a lower recess and opening the east passage.

The wall therefore has two contrasting roles in one small level:

first it blocks entry;

after being moved, it blocks exit;

the player must preserve access to the useful side of its handle to continue.

The lower side of the horizontal handle is static wall, so the continuation cannot be mistaken for “touch the handle from any side.”

State-machine detail

Board: 13×10

Pixel origin: (6,12)

Start: (2,1)

Goal: (10,5)

Selection/cursor state: none

#############
##S##########
##.##########
##......#####
#####...#####
#####.....G##
#####..######
#####.#######
#############
#############
Leaf	Hinge	Length	Initial orientation	Initial occupied cells	Initial free end
A	(5,5)	2	N	(5,4), (5,3)	(5,3)
Verified solution
DDRRRRRDDRRR

Structured trace:

DDRR
R  — push A: N→E
RRD
D  — push A: E→S
RRR

Push details:

Action index	Player before	Transition	Player after
5	(4,3)	A N→E	(5,3)
9	(7,4)	A E→S	(7,5)
Necessity and anti-degeneracy

The narrow geometry prevents walking around either orientation. A must be pushed twice; merely learning “rotate the vertical wall once” leaves the exit physically blocked.

Disabling pushes of A makes the level unsolvable. The second push is forced from a different approach direction, establishing handle-side access rather than merely repeating Level 1.

Level 3 — The empty destination is not enough

Intended depth: 19 actions

Player-facing discovery

A long central leaf covers the green exit. Rotating it east appears to be the obvious objective, but two shorter leaves occupy different parts of its swept region.

The lower short leaf directly occupies cells near the long leaf’s final route. Moving it east makes the long leaf’s final destination visibly empty. A subsequent attempt to rotate the long leaf still fails: the upper short leaf is highlighted at (9,5), an interior sweep cell rather than a final destination cell.

This is the distinct tutorial evidence for swept-area collision. “Only the destination matters” cannot explain the failed attempt.

The two short leaves are independent:

either one may be cleared first;

clearing only one is insufficient;

both feed into the long leaf’s rotation.

This is the first dependency network rather than a single ordered chain.

State-machine detail

Board: 13×11

Pixel origin: (6,10)

Start: (6,4)

Goal: (7,6), initially occupied by leaf C

Selection/cursor state: none

#############
#############
#############
######......#
######S....##
#######...###
#######G..###
#######....##
#########...#
#############
#############
Leaf	Role	Hinge	Length	Initial orientation	Initial occupied cells
A	Upper sweep brace	(9,3)	2	S	(9,4), (9,5)
B	Lower sweep brace	(9,8)	2	N	(9,7), (9,6)
C	Exit leaf	(7,7)	3	N	(7,6), (7,5), (7,4)
Critical collision evidence

For C rotating N→E:

Destination:
(8,7), (9,7), (10,7)

Interior sweep:
(8,6), (9,6), (8,5), (9,5)

Initially:

B occupies (9,7) and (9,6);

A occupies interior cell (9,5).

After B rotates east, all destination cells are clear, but A still blocks the interior sweep at (9,5).

Verified solution
URRDDDRLURULULLDRDD

Structured trace:

URRDDD
R  — push B: N→E
LU
R  — push A: S→E
ULULLD
R  — push C: N→E
DD

Push details:

Action index	Player before	Transition	Player after
7	(8,6)	B N→E	(9,6)
10	(8,5)	A S→E	(9,5)
17	(6,4)	C N→E	(7,4)

The final two moves traverse C’s vacated source cells and reach the exposed goal.

Necessity and anti-degeneracy

Dependency structure:

A ─┐
   ├─ clear C's sweep ─→ C opens the goal
B ─┘

A and B are not interchangeable decorative keys: they occupy different classes of obstruction, one interior and one partly final.

Disabling pushes of A, B, or C individually makes the level unsolvable.

Level 4 — Cross before you close it

Intended depth: 28 actions

Player-facing discovery

The upper leaf A blocks the sweep of both later leaves. Moving A west opens access to the lower loop.

Leaf B begins horizontal. Its northward rotation clears the region needed by the long exit leaf C, but it also occupies (6,5) and (6,4), closing the only connection back to the upper-left region.

The player must therefore:

move A;

cross into the lower region while B is still horizontal;

approach B from below;

rotate B north, deliberately closing the connection behind the player;

use the east loop to approach C.

The new lesson is forced by physical position. Rotating B is useful only after the player is on the correct side of the route it will close.

State-machine detail

Board: 13×11

Pixel origin: (6,10)

Start: (3,2)

Goal: (9,6), initially occupied by C

Selection/cursor state: none

#############
#############
###S.....####
#####....####
######.....##
#####......##
#####....G.##
#####......##
########...##
#############
#############
Leaf	Role	Hinge	Length	Initial orientation	Initial occupied cells
A	Shared upper blocker	(7,3)	2	S	(7,4), (7,5)
B	Commitment gate	(6,6)	2	E	(7,6), (8,6)
C	Exit leaf	(9,7)	3	N	(9,6), (9,5), (9,4)
Dependency geometry

Before A moves:

A at (7,5) blocks B’s E→N sweep.

A at (7,5) also blocks C’s N→W sweep.

Before B moves:

B at (7,6) and (8,6) blocks C’s N→W sweep.

Thus the dependency graph contains both direct and indirect edges:

A → B
A → C
B → C

The player-side commitment is an additional spatial condition, not a validator predicate.

Verified solution
RRRRRDDDLLLDDRRRUDDRRUUUULDD

Structured trace:

RRRRRDDD
L  — push A: S→W

LLDDRRR
U  — push B: E→N, closing the upper connection

DDRRUUUU
L  — push C: N→W

DD

Push details:

Action index	Player before	Transition	Player after
9	(8,5)	A S→W	(7,5)
17	(8,7)	B E→N	(8,6)
26	(10,4)	C N→W	(9,4)
Necessity and anti-degeneracy

Pushing B before entering the lower/east side would strand the player away from C. The failure is not an abstract “wrong order” rejection: B physically occupies the only return cells.

A, B, and C are individually indispensable under exhaustive search. The asymmetrical east loop is required after commitment; replacing the board with a linear corridor would remove the side-selection problem.

Level 5 — Move the correct wall away, then restore it

Intended depth: 40 actions

Player-facing discovery

Leaf A initially points east and blocks the sweep of long leaf B. Rotating A north clears B, apparently solving the immediate obstruction.

However, A’s north orientation then occupies (8,6) and (8,7), which lie in the sweep of exit leaf C. A has become a new liability.

After B rotates west, its vacated vertical source opens an upper route to the opposite side of A’s handle. The player can now rotate A east again, restoring its original orientation for a different reason than before.

Near the goal, leaf D introduces a second contextual use of the same rule. D’s handle occupies (5,8), exactly where the player must stand to push C east. Pushing D east both removes the obstruction and deposits the player at the required stand cell, allowing an immediate second push into C’s handle.

Thus:

A must be moved away and restored;

B creates the route needed for restoration;

D is both an obstacle and a positioning tool;

C is blocked by two independent conditions.

State-machine detail

Board: 15×14

Pixel origin: (2,4)

Start: (13,12)

Goal: (6,7), initially occupied by C

Selection/cursor state: none

###############
###############
###############
###############
#####...#######
#####.......###
###......##...#
###.#.G.....#.#
###....#....#.#
#####..#....#.#
#####...##....#
##########.####
##########...S#
###############
Leaf	Role	Hinge	Length	Initial orientation	Initial occupied cells
A	Reversible central shuttle	(8,8)	2	E	(9,8), (10,8)
B	Long east gate	(11,9)	3	N	(11,8), (11,7), (11,6)
C	Exit leaf	(6,5)	3	S	(6,6), (6,7), (6,8)
D	Handle-position blocker	(5,10)	2	N	(5,9), (5,8)
Dependency geometry

A initially blocks B because B’s N→W sweep contains (10,8).

After A rotates north:

A occupies (8,7), (8,6)

Both cells lie in C’s S→E sweep.

B’s initial vertical source blocks access to the upper return route. After B rotates west, those source cells are vacated and the player can reach (7,6), the stand cell for restoring A N→E.

For C rotating S→E, the required player stand is (5,8). D initially occupies that coordinate with its handle.

Dependency network:

A-away → B
B → access to reverse side of A
A-restored ─┐
            ├→ C
D-moved ────┘
Verified solution
LLLUUUUDDRRRUUUULLULLLLDRULULLDDLLDDRRRU

Structured trace:

LLLUUU
U  — push A: E→N

DDRRRUUUUL
L  — push B: N→W

ULLLLD
R  — push A: N→E

ULULLDDLLDDR
R  — push D: N→E

R  — push C: S→E
U  — enter goal

Push details:

Action index	Player before	Transition	Player after
7	(10,9)	A E→N	(10,8)
18	(12,6)	B N→W	(11,6)
25	(7,6)	A N→E	(8,6)
38	(4,8)	D N→E	(5,8)
39	(5,8)	C S→E	(6,8)
Necessity and anti-degeneracy

Leaving A north allows movement but makes C’s rotation impossible. Restoring A is not cosmetic or part of a final checklist; it clears physical sweep cells.

D cannot simply be walked around because the relevant resource is not access to C’s vicinity but access to the exact tangential stand cell (5,8).

A, B, C, and D are all individually indispensable.

Level 6 — Open one corridor, seal another

Intended depth: 46 actions

Player-facing discovery

Level 6 reuses the Level 5 structure but adds one more ordinary hinged leaf, E, at the upper central bottleneck.

E initially points north. It has two simultaneous consequences:

it occupies (9,5), blocking C’s eventual sweep;

it blocks the direct route to the reverse side of A.

Rotating E east would solve both problems, but E cannot initially move: its eastward destination includes (11,6), occupied by B.

The player must first perform the familiar temporary movement of A and rotate B west. Only then is E’s eastward sweep available.

After the player traverses the upper loop and rotates E east:

E vacates (9,4) and (9,5), opening the western descent;

E occupies (10,6) and (11,6), sealing the eastern route just used.

The player is now committed to the west side, but this time the closure is created by a leaf whose immediate purpose was also to clear another sweep.

State-machine detail

Board: 15×14

Pixel origin: (2,4)

Start: (13,12)

Goal: (6,7), initially occupied by C

Selection/cursor state: none

###############
###############
###############
########...####
#####......####
#####.......###
###...........#
###.#.G.....#.#
###....#....#.#
#####..#....#.#
#####...##....#
##########.####
##########...S#
###############
Leaf	Role	Hinge	Length	Initial orientation	Initial occupied cells
A	Reversible central shuttle	(8,8)	2	E	(9,8), (10,8)
B	Long east gate	(11,9)	3	N	(11,8), (11,7), (11,6)
C	Exit leaf	(6,5)	3	S	(6,6), (6,7), (6,8)
D	Handle-position blocker	(5,10)	2	N	(5,9), (5,8)
E	Upper corridor plug	(9,6)	2	N	(9,5), (9,4)
Critical delayed rotation

For E rotating N→E:

Destination:
(10,6), (11,6)

Interior sweep:
(10,5)

B initially occupies (11,6), so an early attempt highlights B even though the player can see the desired east orientation.

After B rotates west, (11,6) is clear and E can rotate.

Dependency network:

A-away → B → E
                 ↘
E opens return → A-restored
E clears C ────────┐
A-restored ─────────┼→ C
D-moved ────────────┘
Verified solution
LLLUUUUDDRRRUUUULLULUULLDRDLLDRUULLLDDLLDDRRRU

Structured trace:

LLLUUU
U  — push A: E→N

DDRRRUUUUL
L  — push B: N→W

ULUULLD
R  — push E: N→E

DLLD
R  — push A: N→E

UULLLDDLLDDR
R  — push D: N→E

R  — push C: S→E
U  — enter goal

Push details:

Action index	Player before	Transition	Player after
7	(10,9)	A E→N	(10,8)
18	(12,6)	B N→W	(11,6)
26	(8,4)	E N→E	(9,4)
31	(7,6)	A N→E	(8,6)
44	(4,8)	D N→E	(5,8)
45	(5,8)	C S→E	(6,8)
Necessity and anti-degeneracy

E is not an optional added wall:

without moving E, C’s sweep remains blocked at (9,5);

E cannot move before B because of (11,6);

E’s final orientation changes which side of the board remains connected.

The upper loop is traversable only during the appropriate phase. It is not a redundant alternate corridor.

A, B, C, D, and E are individually indispensable.

Level 7 — The cellar hinge

Intended depth: 60 actions

Player-facing discovery

The final level retains all Level 6 dependencies and expands the lower-left region into an asymmetrical cellar loop.

Leaf D still needs to rotate east to place the player beside C. Its final east destination includes (7,10). A new leaf F initially points north and occupies exactly that coordinate.

An attempted D push therefore displays:

D’s apparently reasonable red eastward outline;

F highlighted specifically at (7,10).

F’s opposite approach cell (8,10) is static wall. It can only be rotated east by reaching (6,10), which lies behind D’s hinge and requires descending through the lower cellar route.

Once F rotates east:

D’s destination is clear;

F occupies (8,12) and (9,12);

the short connection toward the original start corridor is sealed;

the player is left on the useful western side.

The player must then revisit the upper part of the cellar, rotate D, immediately rotate C, and step onto the exit.

F has the same operational rule as every earlier leaf, but its contextual role combines three established ideas:

it blocks another leaf’s destination;

it requires approach from one available side;

moving it deliberately closes the route behind the player.

State-machine detail

Board: 15×14

Pixel origin: (2,4)

Start: (13,12)

Goal: (6,7), initially occupied by C

Selection/cursor state: none

###############
###############
###############
########...####
#####......####
#####.......###
###...........#
###.#.G.....#.#
###....#....#.#
####...#....#.#
####....##....#
####.....#.####
#######......S#
###############
Leaf	Role	Hinge	Length	Initial orientation	Initial occupied cells
A	Reversible central shuttle	(8,8)	2	E	(9,8), (10,8)
B	Long east gate	(11,9)	3	N	(11,8), (11,7), (11,6)
C	Exit leaf	(6,5)	3	S	(6,6), (6,7), (6,8)
D	Handle-position blocker	(5,10)	2	N	(5,9), (5,8)
E	Upper corridor plug	(9,6)	2	N	(9,5), (9,4)
F	Cellar plug	(7,12)	2	N	(7,11), (7,10)
Critical lower dependency

For D rotating N→E:

Destination:
(6,10), (7,10)

Interior sweep:
(6,9)

F initially occupies destination coordinate (7,10).

For F rotating N→E:

Required player stand:
(6,10)

Destination:
(8,12), (9,12)

Interior sweep:
(8,11)

The alternative stand coordinate for N→W would be (8,10), which is static wall. F therefore has only one reachable rotation direction without any hidden one-way rule.

Dependency network:

A-away → B → E → route to restore A
                         ↓
A-restored ──────────────┐
E-moved ─────────────────┼→ C
F-moved → D-moved ───────┘

The graph does not specify a validator checklist. Every edge corresponds to an explicit occupied destination, interior sweep cell, handle position, or traversable route.

Verified solution
LLLUUUUDDRRRUUUULLULUULLDRDLLDRUULLLDDLLDDRDDDRRURDLLLUUURRU

Structured trace:

LLLUUU
U  — push A: E→N

DDRRRUUUUL
L  — push B: N→W

ULUULLD
R  — push E: N→E

DLLD
R  — push A: N→E

UULLLDDLLDDRDDDRRU
R  — push F: N→E, sealing the start-side connection

DLLLUUU
R  — push D: N→E

R  — push C: S→E
U  — enter goal

Push details:

Action index	Player before	Transition	Player after
7	(10,9)	A E→N	(10,8)
18	(12,6)	B N→W	(11,6)
26	(8,4)	E N→E	(9,4)
31	(7,6)	A N→E	(8,6)
50	(6,10)	F N→E	(7,10)
58	(4,8)	D N→E	(5,8)
59	(5,8)	C S→E	(6,8)

The final UP reaches (6,7).

Necessity and anti-degeneracy

Trying D before F gives a physically informative failure at (7,10). F cannot be approached from the convenient east side, forcing the lower loop rather than a nearby station visit.

After F moves, its new position blocks the direct route back toward the start. This turns the cellar traversal into a commitment and a forced revisit, not a detachable side errand.

A, B, C, D, E, and F are individually indispensable. No added leaf can be ignored.

11. Replay and validator checks

The authored traces were replayed under the exact sweep-mask and movement rules above.

Level	Shortest verified solution length	Final orientation vector in shown trace	Every leaf indispensable
1	10	A=W	Yes
2	12	A=S	Yes
3	19	A=E, B=E, C=E	Yes
4	28	A=W, B=N, C=W	Yes
5	40	A=E, B=W, C=E, D=E	Yes
6	46	A=E, B=W, C=E, D=E, E=E	Yes
7	60	A=E, B=W, C=E, D=E, E=E, F=E	Yes

“Indispensable” here means that exhaustive reachability search finds no winning state when rotations of that particular leaf are forbidden while every other action remains available.

The final orientations are diagnostic replay results, not additional win requirements.

12. Required implementation test cases

An implementation should pass the following focused checks in addition to replaying all seven solution traces.

Basic contact

Walking into an inner leaf segment does not rotate the leaf.

Walking radially into a handle does not rotate the leaf.

Walking tangentially into a handle rotates the leaf only when its full mask is clear.

On success, the player enters the old free-end cell.

A leaf never translates away from its hinge.

Sweep collision

In Level 3, after moving B east but before moving A:

C’s east destination cells are clear;

C’s N→E push fails;

(9,5) is reported as blocked by A.

Moving A east then makes the same C push legal.

No leaf may sweep another leaf even when their final states do not overlap.

Commitment geometry

In Level 4, B cannot rotate north before A moves west.

After B rotates north, its cells (6,5) and (6,4) prevent return to the upper-left region.

This closure does not trigger failure; closing routes is legal.

Delayed access

In Level 5, A east blocks B.

A north clears B but blocks C.

After B rotates west, A can be restored east from (7,6).

D’s initial handle at (5,8) prevents the player from occupying C’s required stand coordinate.

Pushing D east places the player at (5,8) and enables an immediate C push.

Nested dependency

In Level 6, E N→E fails before B moves because B occupies (11,6).

After B moves west, the same E rotation succeeds.

E east vacates (9,4) and (9,5) while occupying (10,6) and (11,6).

Final cellar dependency

In Level 7, D N→E fails while F occupies (7,10).

F cannot be rotated west because the required stand cell (8,10) is static wall.

F N→E succeeds from (6,10).

F east clears D while occupying (8,12) and (9,12).

The player can still reach D through the authored western cellar return.

These checks capture the game’s essential identity: every useful opening is also a newly placed wall, and the important question is not merely whether a leaf can move, but what geometry that movement leaves behind.
