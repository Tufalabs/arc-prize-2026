Beacon Relay

This is a game about approaching landmarks, becoming difficult because the side from which the current landmark is entered determines which landmark becomes the next objective.

Authored level count: exactly 7 levels.

Beacon Relay overlays two structures on the same board:

a fixed physical maze through which the player walks;

a visible cardinal relay network connecting the beacons.

Entering a beacon is therefore both a movement and an instruction. The physical route determines the side of entry; the side of entry determines the next objective; completing beacons changes later relay rays because lit beacons are skipped.

1. Controls

The exposed action space remains:

UP, DOWN, LEFT, RIGHT

SPACE

click on any cell

Only the following are meaningful:

Action	Effect
UP, DOWN, LEFT, RIGHT	Attempt to move one logical maze cell.
SPACE	Immediately reset the current level to its authored initial state.
Click	No effect. There are no clickable objects, no cursor, and no selected state.

There is no energy, timer, life count, or move limit. Players can explore indefinitely and use SPACE after entering an unrecoverable relay order.

Reference solution lengths count arrow actions only. Animation frames never count as actions.

2. Logical board and rendering
Logical geometry

Every level uses an 11×11 logical lattice:

x = 0…10, increasing left to right;

y = 0…10, increasing top to bottom;

movement changes one coordinate by exactly one;

diagonal movement is impossible.

A logical cell is either:

wall;

traversable floor;

a traversable beacon cell;

the exit cell.

Beacon and exit cells are listed explicitly in each level.

64×64 rendering

The logical board occupies a 55×55-pixel area.

Canvas: 64×64

Board origin: pixel (4,6)

Logical cell size: 5×5

Cell (x,y) occupies:

pixel x = 4 + 5x ... 8 + 5x
pixel y = 6 + 5y ... 10 + 5y

The center pixel of cell (x,y) is:

(6 + 5x, 8 + 5y)

There are no grid lines or cell borders. Adjacent floor cells merge into continuous regions.

Palette
Role	Color ID
Outer canvas	5 black
Walls	4 very dark gray
Floor	1 light gray
Signal rails	3 dark gray
Player	9 blue, with center pixel 0 white
Hollow unlit beacon	11 yellow
Active beacon	12 orange with 11 yellow center
Completed beacon	14 green
Rejected beam	8 red
Closed exit	13 maroon
Open exit	10 light blue
Completion border	14 green
Beacon silhouette

A beacon fills most of its 5×5 logical cell.

Hollow unlit beacon:

..*..
.***.
**.**
.***.
..*..

Completed beacon and active beacon use the filled version:

..*..
.***.
*****
.***.
..*..

These sketches describe pixel occupancy only; no characters are rendered in the game.

The active beacon alternates its orange and yellow regions after each movement action. This pulse_phase is visual only and has no effect on rules. The initial active beacon is already shown in its bright filled state, making it immediately distinguishable from hollow inactive beacons.

The player is rendered over the cell contents as a 3×3 diamond:

.*.
***
.*.

Its four arms are color 9; its center is color 0. Enough of the beacon remains visible beneath the player to show whether the occupied beacon is lit.

Signal rails

For every row and column:

collect all beacon centers on that line;

sort them by coordinate;

connect each consecutive pair with a one-pixel-wide rail through their centers.

Rails:

are visible across floors and walls;

do not make a wall traversable;

do not interact at crossings;

are not switches or physical objects;

exist regardless of which beacons are lit.

A beam can travel through multiple consecutive rail segments when completed beacons are skipped.

Exit

The exit is a module-sized 5×5 gate.

Closed: maroon bars fill the passage and the cell is non-traversable.

Open: the bars slide outward and become light blue; the center becomes traversable floor.

The exit opens only when the final beacon is completed.

3. Exact state machine
State
player_pos: Coord
active_beacon: beacon index or null
lit_mask: one Boolean per beacon
exit_open: Boolean
completed: Boolean
pulse_phase: 0 or 1, visual only
transient_animation: render-only event or null
selection: always null
cursor: absent

Initial invariant for every authored level:

player_pos = S
active_beacon = A
lit_mask = all false
exit_open = false
completed = false
selection = null

During play:

if active_beacon != null, exactly one unlit beacon is active;

an active beacon is never also marked lit;

exit_open is true exactly when every beacon is lit.

Ordinary movement

For an arrow action with vector d, let:

destination = player_pos + d

If the destination is a wall or the closed exit:

the player stays in place;

no relay state changes;

the struck wall or gate flashes white for one frame.

If the destination is floor, a lit beacon, or an inactive unlit beacon:

the player moves there;

no relay state changes.

Inactive and completed beacons are traversable. Only entry into the currently active beacon can fire a relay.

Entering the active beacon

The entry direction is the arrow vector used to move into the beacon. There is no persistent player facing.

For example:

moving RIGHT into a beacon sends its beam right;

moving UP into a beacon sends its beam up.

Before changing state, determine how many unlit beacons exist.

Case A: the active beacon is the sole remaining unlit beacon

The entry is accepted from any traversable adjacent side, even when there is no rail in that direction.

In the same step():

move the player onto the beacon;

mark the beacon lit;

set active_beacon = null;

set exit_open = true.

This is the validator rule that permits a difficult beacon to be reserved as the final target.

Case B: other unlit beacons remain

Search strictly forward from the active beacon along the attempted movement vector.

Candidate beacons must:

lie on the same row or column;

lie strictly forward, not behind;

be unlit;

not be the current active beacon.

Completed beacons are ignored during this search. The closest remaining unlit beacon by cardinal distance is the candidate successor.

If a candidate exists:

move the player onto the current active beacon;

mark that beacon lit;

make the candidate the new active beacon.

If no candidate exists:

reject the entire movement;

keep the player outside the beacon;

do not light the current beacon;

do not change the active beacon;

show a red rejected beam.

The target search is hypothetical until validation succeeds. A rejected entry never partially lights a beacon.

Exit validation

The level is won exactly when:

exit_open == true
and
the player moves into the exit cell

At success:

all beacons must already be lit by construction;

the player’s previous route and entry directions are ignored;

move count is ignored;

beacon occupancy outside the exit is irrelevant;

no additional exact configuration is checked.

Reset and clicks

SPACE restores every variable to the authored initial state, including beacon states, player position, exit state, and pulse phase.

A click on any canvas cell returns the same logical state and same rendered observation. No cell has a click affordance.

4. Animation and update order

Logical changes occur only inside step().

Accepted non-final relay

After the state is validated and committed:

one frame shows the entered beacon turning green;

the bright yellow beam advances one logical cell per render frame;

each completed beacon crossed by the beam flashes white for one frame and returns to green;

on the final frame, the target changes from hollow yellow to filled orange/yellow.

No action is accepted during these frames, and they consume no moves.

Rejected relay

Logical state remains unchanged.

If a rail exists in the attempted direction, a red beam advances to the end of the connected rail and retracts.

If no rail exists in that direction, a two-pixel red stub flashes from the beacon’s corresponding arm.

The player remains visibly outside the beacon.

Final beacon

After committing the final beacon:

the beacon turns green;

no outgoing beam is drawn;

the exit bars slide outward over three frames;

the open exit remains light blue.

Ordinary movement

Normal one-cell movement renders immediately. It does not need interpolated animation.

5. Authored-map notation

The maps below are implementation notation, not the actual rendering.

# = wall

. = floor

S = player start

X = exit

A–H = beacon centers

Every row contains 11 characters corresponding to x=0…10; the 11th character is x=10. Letters are never drawn in the game.

Every listed reference action sequence was replayed against the state machine. A breadth-first state search found no shorter successful sequence.

Level 1 — First Hand-Off

Intended depth: 9 actions

Authored layout
    x=0........10
y=0 ###########
y=1 ###########
y=2 #######X###
y=3 #######.###
y=4 #######.###
y=5 #S.A...B###
y=6 ###########
y=7 ###########
y=8 ###########
y=9 ###########
y=10###########
S = (1,5)
X = (7,2)
A = (3,5), initially active
B = (7,5)
lit = {}
selection = null
Player-facing discovery

A is a filled, bright beacon while B is hollow. The first two rightward moves enter A from the west. A turns green and a beam visibly travels east to B, which becomes active.

Entering B completes it despite having no unlit successor because it is the final beacon. The exit then opens above it.

This is the positive goal evidence for the game: the player sees one complete relay hand-off and the gate response without needing a legend or symbolic reference panel.

State-machine solution

Relay order:

A → B → exit

Replay:

RR [A→B] | RRRR [B final] | UUU [exit]

Full action string:

RRRRRRUUU

Relay events occur on actions 2 and 6; success occurs on action 9.

Necessity and anti-degeneracy

A cannot be omitted because it demonstrates that entering the active beacon lights it and remotely selects another beacon. B cannot be omitted because the level must separately demonstrate the final-beacon exception and gate opening.

The simple L-shaped route is restricted to this first tutorial. No rule choice is hidden behind distance or maze complexity.

Level 2 — The Side Is the Instruction

Intended depth: 17 actions

Authored layout
    x=0........10
y=0 ###########
y=1 ###########
y=2 ##..B...C##
y=3 ##.#####.##
y=4 ##...###.##
y=5 ###SA###X##
y=6 ###..######
y=7 ###########
y=8 ###########
y=9 ###########
y=10###########
S = (3,5)
X = (8,5)
A = (4,5), initially active
B = (4,2)
C = (8,2)
lit = {}
selection = null
Player-facing discovery

The player begins immediately west of A. The apparently obvious RIGHT action tries to enter A while sending its beam east. There is no unlit beacon east of A, so the entry is rejected; A flashes a red eastward stub and the player remains at S.

A tiny loop directly below the player allows entry into A while moving upward. That beam follows the visible vertical rail to B.

When B becomes active, approaching it directly from below would again send the wrong ray. The bent left-side route reaches B from the west, allowing B to select C to its east.

The same directional rule is therefore observed twice in contrasting local geometries rather than taught by prose.

State-machine solution

Relay order:

A → B → C → exit

Replay:

DRU [A→B]
ULLUURR [B→C]
RRRR [C final]
DDD [exit]

Full action string:

DRUULLUURRRRRRDDD

Relay events occur on actions 3, 10, and 14; success occurs on action 17.

Necessity and anti-degeneracy

The cell immediately west of A creates a plausible rejected entry. The lower two-cell pocket is necessary to obtain the upward entry vector.

The left bend near B is not movement padding: without it, B could not be entered while moving right, and C could not become active. A route that merely reaches B from below does not solve the relay.

Level 3 — Branch Before Route

Intended depth: 23 actions

Authored layout
    x=0........10
y=0 ###########
y=1 ###X#######
y=2 ###.#######
y=3 ##.C...D.##
y=4 ##.#####.##
y=5 ##.#####.##
y=6 ##.#####.##
y=7 ##.AS##B.##
y=8 ##......###
y=9 ###########
y=10###########
S = (4,7)
X = (3,1)
A = (3,7), initially active
B = (7,7)
C = (3,3)
D = (7,3)
lit = {}
selection = null
Player-facing discovery

A has two relationally meaningful successors:

entering A while moving right selects B;

entering A while moving up selects C.

The shortest physical maneuver is to step down, left, and up into A, selecting C. That accepted beam looks promising, but it commits the relay to the wrong branch.

After A→C→D, B remains unlit. D would have to send downward to B, requiring entry into D from its north side (7,2). That cell is a wall. Every accessible side of D rejects, exposing the dead order physically rather than through an abstract validator message.

The correct plan loops around A’s lower side, enters from the west, and uses:

A → B → D → C

The player must solve the objective graph and the approach geometry together.

State-machine solution

Replay:

DLLUR [A→B]
DRRRRU [B→D]
RUUUUL [D→C]
LLLL [C final]
UU [exit]

Full action string:

DLLURDRRRRURUUUULLLLLUU

Relay events occur on actions 5, 11, 17, and 21; success occurs on action 23.

Dependency network

This is not a single dependency chain.

A can select B or C.

B can select D.

C can select D.

D can potentially point toward B or C depending on what remains unlit.

The physical inaccessibility of (7,2) makes the order A→C→D unrecoverable while leaving A→B→D→C valid.

Necessity and anti-degeneracy

C is physically visible and reachable long before it should be completed. D’s blocked north approach is the structural reason the wrong branch fails; removing that wall would make the early-C order solvable.

The lower route, right column, and top row each serve a different required entry direction. They cannot be replaced by one corridor without changing the puzzle.

Level 4 — Signals Pass Through History

Intended depth: 37 actions

Authored layout
    x=0........10
y=0 ###########
y=1 ######...##
y=2 ######C#.##
y=3 ######.#.##
y=4 ###X##.#.##
y=5 ###.##.#.##
y=6 #.S.##.#.##
y=7 #.A.##B..##
y=8 ##......###
y=9 ###E..D.###
y=10###########
S = (2,6)
X = (3,4)
A = (2,7), initially active
B = (6,7)
C = (6,2)
D = (6,9)
E = (3,9)
lit = {}
selection = null
Player-facing discovery

After A→B, B must be entered upward from its southern pocket to select C.

When C becomes active, the short vertical trunk approaches C from below. Entering from that side would send the beam upward, where there is no unlit successor, so the entry is rejected.

The right-hand loop reaches the north side of C. Entering C while moving down sends a beam toward D. B lies directly between them and is already green. The beam visibly crosses B, makes B flash white, and continues to D.

This forces observation of the distinct rule:

A completed beacon does not block or reactivate; the relay skips it and continues to the first unlit beacon.

State-machine solution

Relay order:

A → B → C → D → E

Replay:

LDR [A→B]
DRRRRU [B→C]
RRUUUUUULLD [C→D, skipping lit B]
DDDDDDRDL [D→E]
LLL [E final]
UUUUU [exit]

Full action string:

LDRDRRRRURRUUUUUULLDDDDDDDRDLLLLUUUUU

Relay events occur on actions 3, 9, 20, 29, and 32; success occurs on action 37.

Dependency network

B has two possible meaningful outgoing targets:

north to C;

south to D.

Taking B→D early is unrecoverable:

D→E leaves C unlit, but E has no ray to C;

D→C leaves E unlit, but C has no ray to E.

Thus C must be completed before D, specifically so C’s downward beam can skip B.

Necessity and anti-degeneracy

B’s position between C and D is mechanically necessary. If B were absent or not previously lit, the level would not test skipping.

The right loop is necessary because C must be entered from above. The direct trunk exists to create a concrete local misconception, not as a decorative alternative.

Level 5 — Save the Awkward Beacon

Intended depth: 42 actions

Authored layout
    x=0........10
y=0 ###....####
y=1 ###.##.####
y=2 ###E##..D..
y=3 #X..######.
y=4 ###F##..C..
y=5 ######.#...
y=6 ######.###.
y=7 ######.###.
y=8 ##.A##..B#.
y=9 ##.S.###...
y=10####.....##
S = (3,9)
X = (1,3)
A = (3,8), initially active
B = (8,8)
C = (8,4)
D = (8,2)
E = (3,2)
F = (3,4)
lit = {}
selection = null
Player-facing discovery

The player begins directly below A. Pressing UP enters A while moving north. The beam immediately selects F, which looks like progress.

F then creates a visible local impasse:

north neighbor (3,3) is traversable;

south (3,5), west (2,4), and east (4,4) are walls;

entering F from its only accessible side means moving down;

that downward ray encounters only the already-lit A and then terminates while other unlit beacons remain.

The entry is therefore rejected. F cannot pass the relay onward in this state.

Resetting and using the small western hook enters A while moving right, selecting B instead.

F appears as a tempting shortcut again when C is active: entering C while moving left would select F. It remains the same dead active beacon. The correct entry into C points north to D.

After completing B, C, D, and E, E sends downward to F. F is now the sole unlit beacon. The exact same north-side entry that previously rejected is accepted, because the last beacon requires no successor.

State-machine solution

Relay order:

A → B → C → D → E → F

Replay:

LUR [A→B]
DRDRRRRUU [B→C]
DRRUUUULLU [C→D]
RRUULL [D→E]
LLUULLLDD [E→F]
DD [F final]
ULL [exit]

Full action string:

LURDRDRRRRUUDRRUUUULLURRUULLLLUULLLDDDDULL

Relay events occur on actions 3, 12, 22, 28, 37, and 39; success occurs on action 42.

Dependency network

F is linked into the relay from two different earlier positions:

A can select F northward;

C can select F westward.

Both shortcuts are relationally valid but strategically fatal. The correct order preserves F until E can select it as the last objective.

This is not a rule exception: F’s outgoing rays and accessible sides never change. Only the set of remaining unlit beacons changes.

Necessity and anti-degeneracy

F is simultaneously:

a dangerous early successor;

a shortcut temptation in the middle;

the required final beacon.

Its one-sided physical access makes the final-beacon validator quirk necessary rather than convenient. Removing F, opening its south or west neighbor, or allowing a final beacon to require an outgoing ray would fundamentally change the solution.

Level 6 — Return Beam

Intended depth: 51 actions

Authored layout
    x=0........10
y=0 ###########
y=1 #..X#######
y=2 #.####....#
y=3 #G##..C..D.
y=4 #.##.#.##..
y=5 #.##.#.##..
y=6 #..#...##..
y=7 ##..S#.##..
y=8 #F.A..B..E.
y=9 #..#...####
y=10###########
S = (4,7)
X = (3,1)
A = (3,8), initially active
B = (6,8)
C = (6,3)
D = (9,3)
E = (9,8)
F = (1,8)
G = (1,3)
lit = {}
selection = null
Player-facing discovery

The lower row contains:

F — A — B — E

The upper row contains:

G — C — D

The initial short approach enters A while moving left and selects F. F can then only use its southern approach to select G. G becomes active too early and cannot point to the remaining right-side beacons because its west approach is a wall. The branch is dead.

The correct western approach selects B. B then presents another trap: walking straight along the lower row enters B from the west and selects E, again causing E→F→G too early. The player instead stages below B and enters upward to select C.

C, D, and E each require a different approach:

C from the west, to select D eastward;

D from the north, to select E downward;

E from the east, to send west.

E’s westward beam crosses completed B and A before reaching F. Both completed beacons visibly flash as they are skipped.

Afterward, the player walks physically back across B and A. Reaching F from the east is not enough: that entry points west and rejects. The player must pass F’s vicinity, drop into its lower pocket, and re-enter upward to select G.

G is now last and accepts entry.

State-machine solution

Relay order:

A → B → C → D → E → F → G

Replay:

LLDR [A→B]
RDRRU [B→C]
UULLUUURR [C→D]
URRRD [D→E]
DDDDRDL [E→F, skipping B and A]
LLLLLLLDLU [F→G]
RUULUUU [G final]
UURR [exit]

Full action string:

LLDRRDRRUUULLUUURRURRRDDDDDRDLLLLLLLLDLURUULUUUUURR

Relay events occur on actions 4, 9, 18, 23, 30, 40, and 47; success occurs on action 51.

Dependency network

Several tempting accepted transitions exist:

A→F;

B→E;

C→D;

C→G;

E→D or E→F, depending on approach and lit state.

Both A→F and B→E eventually activate G while C or D remains. G’s useful eastward beam would require entry from (0,3), which is a wall. Those orders are unrecoverable.

In the correct order, completing A and B first changes E’s westward ray from a sequence of nearby landmarks into a long skip to F.

Necessity and anti-degeneracy

The return across the lower row is not redundant travel:

on the first visit, A and B are active objectives whose entry sides must be controlled;

on the return, both are completed landmarks that the beam and player traverse harmlessly;

F, initially a liability, is now needed to reach the reserved final beacon G.

The same objects have different strategic roles solely because of relay history and player approach.

Level 7 — Crossed Relay

Intended depth: 61 actions

Authored layout
    x=0........10
y=0 ##.....X###
y=1 ##.###..###
y=2 #.E..D..H##
y=3 ..###.##.##
y=4 .#.####..##
y=5 ..F##C..G##
y=6 ....#....##
y=7 ###..##..##
y=8 #.A..B...##
y=9 #..S#.#####
y=10###...#####
S = (3,9)
X = (7,0)
A = (2,8), initially active
B = (5,8)
C = (5,5)
D = (5,2)
E = (2,2)
F = (2,5)
G = (8,5)
H = (8,2)
lit = {}
selection = null
Player-facing discovery

The beacons form an incomplete three-row lattice:

E — D — H
|   |   |
F — C — G
|   |
A — B

The visible network offers many apparently valid successors:

A can point east to B or north to F;

C can point north to D, west to F, or east to G;

D can point west to E or east to H;

F can point through C toward G;

G can point north to H.

The short initial A→F branch is accepted but unrecoverable. It consumes the left wing before the central and upper dependencies have been staged.

The correct first half is:

A → B → C → D

While D is active, the physical route to D passes directly over H at (8,2). H is hollow and inactive, so the player can occupy it without firing it. The player must resist treating physical contact as completion and continue around to D’s east side.

D has a tempting eastward ray to H. Activating H here is fatal:

H’s east neighbor (9,2) is a wall, so it cannot be entered while moving left toward D or E;

H’s north neighbor (8,1) is a wall, so it cannot be entered while moving down toward G;

its accessible west and south sides point toward empty east and north rays.

H is therefore usable only as the final beacon.

D must instead point west to E. E points down to F. When F is entered from the west, its eastward beam passes through completed C and activates G. G then points north to H, which finally accepts entry from below.

State-machine solution

Relay order:

A → B → C → D → E → F → G → H

Replay:

LLUR [A→B]
DRDRRUU [B→C]
RRUULLU [C→D]
RRURUULLL [D→E; route crosses inactive H]
RUULLLLDD [E→F]
LDLDDRR [F→G, skipping lit C]
DRDDRRRRUURU [G→H]
UUU [H final]
LUU [exit]

Full action string:

LLURDRDRRUURRUULLURRURUULLLRUULLLLDDLDLDDRRDRDDRRRRUURUUUULUU

Relay events occur on actions 4, 11, 18, 27, 36, 43, 55, and 58; success occurs on action 61.

Dependency network

The correct order is selected from a dense directional network, not presented as a sequence of isolated stations.

Critical dependencies include:

B must remain available when A fires east;

D must be selected before the left wing E–F is consumed;

F must remain unlit until C is completed, allowing F’s later ray to skip C and reach G;

H must remain unlit until every other beacon is complete;

physical passage over H must not be confused with activation.

Both A→F and D→H lead to states from which the exit is unreachable.

Necessity and anti-degeneracy

Every beacon has more than a checklist role:

C is first an objective, later a completed signal bridge;

H is first traversable terrain, then a dangerous premature target, then the final objective;

F is an early branch, a vertical receiver from E, and the source of a skip through C;

D divides the upper network into the fatal H shortcut and the required E branch.

The layout cannot be reduced to a corridor or station order. Its solution depends on both the two-dimensional maze and the changing unlit-beacon graph.

6. Progression summary
Level	Main discovery	Contrasting later use
1	Entering the active beacon selects a successor; final beacon opens exit.	Final-any-side rule later becomes a planning tool.
2	Entry direction is the beam direction; invalid rays reject movement.	Rejection later distinguishes wrong physical approaches from wrong relay orders.
3	Objective order and physical accessibility form a dependency network.	Later levels contain several accepted but fatal branches.
4	Completed beacons are skipped.	Levels 6 and 7 use multi-beacon and strategically staged skips.
5	A hard beacon can need to be reserved as final.	G in level 6 and H in level 7 reuse this with different geometry.
6	The same corridor and beacons change meaning on a forced return.	Level 7 combines traversal, inactive occupancy, skipping, and reservation.
7	Full relay planning across a dense cardinal network.	Culmination; no new noun or exception is introduced.
7. Required implementation tests

These are useful unit tests for the validator.

Rejected directional entry

Level 2 initial state:

player = (3,5)
active = A at (4,5)
action = RIGHT

Expected:

player remains (3,5)
A remains active and unlit
lit_mask unchanged
red eastward rejection shown
Skip one completed beacon

Level 4 after A and B are lit:

active = C at (6,2)
player = (6,1)
action = DOWN

Expected target:

D at (6,9)

B at (6,7) is crossed and skipped.

Skip multiple completed beacons

Level 6 after A, B, C, and D are lit:

active = E at (9,8)
player = (10,8)
action = LEFT

Expected target:

F at (1,8)

B at (6,8) and A at (3,8) are both skipped.

Same entry rejected early and accepted late

Level 5, F at (3,4):

player = (3,3)
action = DOWN

If A is lit but other unlit beacons remain:

reject

If F is the sole unlit beacon:

enter F
light F
open exit
Inactive-beacon traversal

Level 7 while D is active and H is unlit but inactive:

player moves onto H at (8,2)

Expected:

player moves normally
H stays hollow and inactive
D remains active
no beam is emitted
Rail and wall independence

For every level:

walls block the player;

walls never block a signal beam;

rail crossings never redirect or split a beam;

the first unlit beacon on the ray is always the sole successor.
