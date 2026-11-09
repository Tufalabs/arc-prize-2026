Worked for 33m 18s
Bridge Stock

Exact authored level count: 7

This is a game about crossing gaps becoming difficult because every permanent crossing spends a plank that may be the only way to reach later stock.

Game identity

Bridge Stock is a top-down spatial resource puzzle. The player moves through a visible arrangement of floor islands, walls, open void, socketed one-cell gaps, loose wooden planks, and a green exit.

A carried plank is not placed manually. Attempting to enter a socketed gap automatically consumes one plank, covers the gap with a permanent wooden bridge, and moves the player onto it. That bridge then behaves exactly like ordinary floor: it can be recrossed freely and can be left in any orthogonal direction.

The central tension is therefore visible and physical:

A plank is both inventory and a future piece of terrain.

Spending a plank may open access to several more planks.

An installed bridge may become a repeatedly used junction or return route.

A convenient shortcut may consume the one plank required by an unavoidable later crossing.

Mistakes remain visible: every spent plank is still present in the particular gap that consumed it.

There are no hidden rules, timers, enemies, switches, symbolic predicates, or secondary completion conditions.

1. Player actions

The environment exposes exactly:

UP

DOWN

LEFT

RIGHT

SPACE

clicking any canvas cell

Only the four directional actions have gameplay effects.

SPACE and every click are legal no-ops. They leave the complete state and settled frame unchanged. Nothing is rendered as clickable, there is no cursor state, and there is no selected object.

Each directional action attempts to move the player by exactly one logical cell.

There is no move budget, energy meter, timer, or action penalty. Exploration is unlimited. An ordinary environment-level reset restores the authored initial state but is not an in-game action.

2. Exact success condition

A level is won immediately when the player enters the exit cell.

The validator is:

won = player_position == exit_position

The following are irrelevant to validation:

remaining carried planks;

uncollected loose planks;

unbridged gaps;

the number or location of installed bridges;

the route used to reach the exit.

Entering the exit wins even with unused stock or loose planks remaining. No authored reference solution relies on such leftovers except where explicitly noted.

Once won becomes true, all subsequent actions return the unchanged completed frame.

There is no automatic failure state. A wasteful bridge placement may leave the level visibly unsolvable, in which case the player can use the external level reset.

3. Logical state

Every level uses a fixed logical board of 15 columns × 12 rows.

Coordinates are (x,y) with:

x = 0…14, increasing left to right;

y = 0…11, increasing top to bottom.

The complete persistent state is:

terrain[12][15]:
    VOID
    WALL
    FLOOR
    GAP
    BRIDGE

player_position: (x, y)

loose_planks: set[(x, y)]

inventory: nonnegative integer

exit_position: (x, y)

status:
    RUNNING
    WON

There is no strategically relevant history. In every settled frame:

the player position is visible;

every loose plank is visible;

every installed bridge is visible;

every uninstalled gap is visible;

the exact inventory count is visible;

the exit and all traversable terrain are visible.

The authored states never exceed five carried planks.

4. Terrain and interaction rules
4.1 Floor

A floor cell is freely traversable. Entering it moves the player one cell.

4.2 Loose plank

A loose plank is an overlay on a floor cell.

When the player enters that cell:

the player moves onto the cell;

the loose plank is removed;

inventory increases by exactly one.

Collection is compulsory and permanent until the level is reset.

There is no carrying capacity, dropping action, or way to leave a plank on a different floor cell.

4.3 Socketed gap

A gap is an impassable one-cell hole with four clearly visible orange socket tabs.

A gap may be approached from any orthogonally adjacent traversable cell.

When the player attempts to enter an unbridged gap:

Inventory greater than zero

Inventory decreases by exactly one.

The target terrain changes from GAP to BRIDGE.

The player moves onto the new bridge.

Placement is automatic. The player cannot decline to spend the plank while moving into the gap.

Inventory equal to zero

The player does not move.

The gap remains unbridged.

No other state changes.

The orange socket tabs flash red and the empty inventory tray gives a brief shake.

4.4 Installed bridge

An installed bridge is permanent for the rest of the attempt.

It behaves exactly like floor:

entering it costs nothing;

leaving it costs nothing;

it can be crossed any number of times;

the player may leave it through any orthogonally adjacent floor, bridge, loose-plank, or exit cell.

This all-direction behavior is important. A bridge can become a junction connecting areas above, below, left, and right of the original gap.

Installed bridges cannot be removed, picked up, pushed, or overwritten.

4.5 Wall

A wall blocks movement and never consumes inventory.

A blocked wall action produces a one-frame bump of the player sprite toward the wall, followed by the unchanged settled state.

4.6 Unmarked void

Unmarked void blocks movement and can never receive a plank.

It has no orange sockets, clearly separating it from a bridgeable gap. Attempting to enter void does not consume inventory.

5. Directional step() resolution order

For a directional action while status == RUNNING:

1. Compute target = player_position + direction.

2. If target is outside the 15×12 board:
       leave state unchanged.

3. If terrain[target] is VOID:
       leave state unchanged;
       emit void-block feedback.

4. If terrain[target] is WALL:
       leave state unchanged;
       emit wall-bump feedback.

5. If terrain[target] is GAP:
       if inventory == 0:
           leave state unchanged;
           emit empty-gap feedback;
       else:
           inventory -= 1;
           terrain[target] = BRIDGE;
           player_position = target;
           emit bridge-install animation.

6. Otherwise, if terrain[target] is FLOOR or BRIDGE:
       player_position = target.

7. If player_position is in loose_planks:
       remove that coordinate from loose_planks;
       inventory += 1;
       emit collection animation.

8. If player_position == exit_position:
       status = WON;
       emit completion animation.

9. Return the settled observation.

Authored level data never places a loose plank or exit overlay on a gap, wall, or void.

6. Animation and feedback contract

Animations are deterministic integer-pixel sequences belonging to the action that caused them. They do not call step(), accept input, or consume additional actions.

An implementation that exposes only one observation per action returns the final settled frame.

Ordinary movement

Ordinary floor and bridge movement is immediate. No interpolation is required.

Plank collection

A three-frame sequence:

The board plank brightens from color 12 to color 11.

It contracts toward the player while the next inventory slot brightens.

The board plank disappears and the inventory icon appears.

The persistent collection state is committed on frame 3.

Bridge installation

A four-frame sequence is required because one movement action visibly transforms terrain:

The player’s carried-plank marker extends toward the target gap.

A narrow orange board spans the target cell along the entry direction.

Cross-braces unfold, producing the standard square bridge platform.

Inventory decreases, the player appears centered on the bridge, and the bridge receives its final colors.

The persistent transition is committed on frame 4. The entry direction affects only the unfolding animation; the settled bridge is the same all-direction platform.

Empty-gap attempt

Two transient frames:

The gap sockets change from orange 12 to red 8.

The empty inventory tray shifts one pixel and returns.

No persistent state changes.

Completion

When the player enters the exit:

the exit ring changes from green 14 to white 0;

the player and exit alternate green and white for three additional frames;

the final frame retains a green outer ring and white center.

No text is displayed.

7. Rendering contract
7.1 Canvas and board placement

The render canvas is exactly 64×64 pixels, containing integer color IDs only.

Each logical cell is 4×4 rendered pixels.

The board occupies:

pixel_x = 2 + 4*x
pixel_y = 14 + 4*y

Thus the board spans pixels:

x = 2…61
y = 14…61

The upper strip, primarily y = 2…9, contains the inventory display.

No grid lines or cell borders are drawn. Adjacent floor cells form continuous regions, while void merges with the black background.

7.2 Palette usage
ID	Use
0	white completion highlight and exit center
1	floor
2	wall highlight and inactive HUD details
3	wall body and empty inventory slots
4	very-dark HUD backing
5	unmarked void and gap interior
8	invalid empty-gap flash
9	player body
10	player center highlight
11	brief plank/install highlight
12	plank, bridge, and gap-socket wood
13	plank ends and bridge cross-braces
14	exit
6, 7, 15	unused

The normal settled board therefore uses only quiet gray structure plus blue for the player, orange for the plank system, and green for the objective.

7.3 Central 4×4 silhouettes

The following matrices specify color IDs within one logical cell.

Socketed gap
5  12 12 5
12 5  5  12
12 5  5  12
5  12 12 5

The four-sided sockets communicate that the hole is bridgeable from any orthogonal direction.

Loose plank on floor
1  1  1  1
13 12 12 13
13 12 12 13
1  1  1  1
Installed bridge
13 12 12 13
12 13 13 12
12 13 13 12
13 12 12 13

It remains unmistakably wooden but fully covers the former hole.

Exit
1  14 14 1
14 0  0  14
14 0  0  14
1  14 14 1
Player

The player is a connected blue 3×3 body using colors 9 and 10, centered in its logical cell. When inventory is nonzero, a connected 1×2 orange pack using 12 and 13 is attached to its left side. Exact inventory is still communicated by the HUD rather than by the pack size.

When standing on the exit, the green corner pixels remain visible around the player.

7.4 Inventory HUD

The HUD uses five fixed plank slots because five is the maximum inventory reached by these seven authored levels.

HUD backing: pixels x=2…41, y=2…9, color 4.

Player badge: a 4×4 blue marker at x=4…7, y=4…7.

Slot i, for i=0…4, begins at (10+6i,4) and occupies 4×3 pixels.

Empty slot: color 3.

Filled slot: orange center 12 with maroon end pixels 13.

The first inventory slots are filled left to right.

No numerals or text are rendered.

7.5 Render priority

From back to front:

void/background;

floor or wall;

gap or bridge;

exit overlay;

loose plank;

player;

transient feedback;

HUD.

8. Authored map encoding

All seven maps below are complete 15×12 initial boards.

~  unmarked void
#  wall
.  floor
G  unbridged socketed gap
P  loose plank on floor
S  player start on floor
E  exit on floor

No level begins with an installed bridge.

Run notation compresses repeated actions: R4 means four separate RIGHT actions.

The listed reference lengths count successful directional actions only. Exploratory blocked attempts may be inserted without changing state.

Progression overview
Level	Main evidence or planning pressure	Verified reference length
1	Carrying a plank lets movement install a bridge	2
2	Walking onto a loose plank adds visible stock	6
3	A bridge persists, is free to reuse, and becomes a junction	12
4	A convenient bridge can be strategically wasteful	25
5	One bridge can support several branches; stock must reach the next supply	44
6	An early supply route must be preserved for a later exit journey	70
7	Two bridge junctions, branch ordering, exact stock, and one redundant shortcut	73
9. Level 1 — The First Crossing
Player-facing discovery

The player begins with one filled inventory slot. A single socketed hole lies between the player and the green exit.

Moving toward the exit immediately produces the central causal event: the carried plank unfolds over the gap and the player moves onto the resulting wooden surface. One more move enters the exit.

There are no loose planks, choices, decoys, walls, or secondary routes.

Initial state
Initial inventory: 1
Start:             (5,6)
Exit:              (7,6)
Loose planks:      none
Gaps:              (6,6)
Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~~~~~~~~~
02  ~~~~~~~~~~~~~~~
03  ~~~~~~~~~~~~~~~
04  ~~~~~~~~~~~~~~~
05  ~~~~~~~~~~~~~~~
06  ~~~~~SGE~~~~~~~
07  ~~~~~~~~~~~~~~~
08  ~~~~~~~~~~~~~~~
09  ~~~~~~~~~~~~~~~
10  ~~~~~~~~~~~~~~~
11  ~~~~~~~~~~~~~~~
Reference solution
R2

Exact resolution:

RIGHT: consume the only plank, replace gap (6,6) with a bridge, and move to (6,6).

RIGHT: move to exit (7,6) and win.

Inventory trace:

1 → install (6,6) → 0
Attribution and necessity check

The first useful action changes exactly three visible things belonging to one causal event:

a filled inventory slot empties;

the socketed hole becomes a wooden platform;

the player advances onto it.

The exit is unreachable without installing the bridge. No unrelated object changes simultaneously.

10. Level 2 — Fetch Before Crossing
Player-facing discovery

The direct route is visually identical to Level 1, but the inventory tray is empty. Trying RIGHT produces the empty-gap flash without movement.

A single loose plank is visible two cells above the player. Walking onto it visibly removes the board object and fills one inventory slot. Returning to the gap now reproduces the successful installation learned in Level 1.

The extra one-cell approach to the plank ensures collection is seen as the result of entering its cell, rather than as a remote or adjacency effect.

Initial state
Initial inventory: 0
Start:             (5,6)
Exit:              (7,6)
Loose planks:      (5,4)
Gaps:              (6,6)
Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~~~~~~~~~
02  ~~~~~~~~~~~~~~~
03  ~~~~~~~~~~~~~~~
04  ~~~~~P~~~~~~~~~
05  ~~~~~.~~~~~~~~~
06  ~~~~~SGE~~~~~~~
07  ~~~~~~~~~~~~~~~
08  ~~~~~~~~~~~~~~~
09  ~~~~~~~~~~~~~~~
10  ~~~~~~~~~~~~~~~
11  ~~~~~~~~~~~~~~~
Reference solution
U2, D2, R2

Inventory trace:

0 → collect (5,4) → 1 → install (6,6) → 0

A blind discovery trajectory will commonly begin with a blocked RIGHT, followed by the six-action solution.

Attribution and necessity check

The plank at (5,4) is the only source of stock and must be entered. The direct gap cannot be crossed initially. No other object or rule changes during collection, so the filled HUD slot is unambiguously caused by stepping onto the plank.

11. Level 3 — Paid Once
Player-facing discovery

The player has one plank and only one initially reachable gap. Crossing it spends the plank and enters a small supply alcove containing two loose planks.

After collecting both, the only way forward is to step back onto the already installed bridge. Inventory does not decrease. From that bridge, the player turns right into a route containing two new gaps.

This cleanly establishes three consequences of one rule:

a bridge remains after the player leaves it;

recrossing costs nothing;

the installed tile behaves like floor and can be exited sideways.

The two collected planks are then consumed by the two final gaps.

Initial state
Initial inventory: 1
Start:             (5,7)
Exit:              (10,6)
Loose planks:      (3,5), (4,5)
Gaps:              (5,6), (7,6), (9,6)
Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~~~~~~~~~
02  ~~~~~~~~~~~~~~~
03  ~~~~~~~~~~~~~~~
04  ~~~~~~~~~~~~~~~
05  ~~~PP.~~~~~~~~~
06  ~~~~~G.G.GE~~~~
07  ~~~~~S~~~~~~~~~
08  ~~~~~~~~~~~~~~~
09  ~~~~~~~~~~~~~~~
10  ~~~~~~~~~~~~~~~
11  ~~~~~~~~~~~~~~~
Reference solution
U2, L2, R2, D, R5

Resolution by phase:

U2: install (5,6) and enter the supply alcove.

L2: collect both planks.

R2: return to (5,5).

D: step onto bridge (5,6) without spending stock.

R5: leave the bridge sideways, install (7,6) and (9,6), and enter the exit.

Inventory trace:

1
→ install (5,6): 0
→ collect two:   2
→ recross (5,6): 2
→ install two:   0
Attribution and necessity check

There is no alternative gap choice at the beginning, so the foundational experiment cannot cause an accidental dead end. The final route is inaccessible until (5,6) has become a bridge. Both loose planks and both final gaps are necessary.

A mistaken hypothesis that bridges disappear, cost again, or only support straight-through motion is directly contradicted by the required return and right turn.

12. Level 4 — The Expensive Shortcut
Player-facing discovery

The start contains one plank. Two more planks are visible on the eastern side of a short wall barrier.

Gap (8,8) offers an immediate crossing through the barrier. A free floor route goes up, around the wall, and back down to the same supply area.

The exit lies to the left behind exactly three gaps. The player therefore needs all three available planks for the exit route. Spending one on the attractive shortcut leaves only two after collecting the supply.

This is the first deliberately recoverable-looking but strategically losing commitment. Its failure is concrete: the optional bridge remains at (8,8), while one of the three exit gaps remains visibly open.

Initial state
Initial inventory: 1
Start:             (7,8)
Exit:              (0,8)
Loose planks:      (10,6), (12,6)
Gaps:              (2,8), (4,8), (6,8), (8,8)
Walls:             (8,6), (8,7)
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING

Gap (8,8) is optional and wasteful. The other three gaps form the exit route.

Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~~~~~~~~~
02  ~~~~~~~~~~~~~~~
03  ~~~~~~~~~~~~~~~
04  ~~~~~~~~~~~~~~~
05  ~~~~~~~...~~~~~
06  ~~~~~~~.#.P.P~~
07  ~~~~~~~.#.~~~~~
08  E.G.G.GSG.~~~~~
09  ~~~~~~~~~~~~~~~
10  ~~~~~~~~~~~~~~~
11  ~~~~~~~~~~~~~~~
Reference solution
U3, R2, D, R3,
L3, U, L2, D3,
L7

Phases:

Follow the western side of the barrier upward.

Cross its free opening at row 5.

Descend to the two supply planks and collect both.

Return through the same free upper route.

Move left through the three required gaps to the exit.

Inventory trace:

1
→ collect (10,6), (12,6): 3
→ install (6,8), (4,8), (2,8): 0
Anti-degenerate and anti-cleanliness check

The wall creates a genuine topological cycle: a short paid edge and a longer free route connect the same two sides. Removing the geometry would remove the puzzle.

The optional gap is not arbitrary decoration. It offers useful access to the actual supply and saves substantial movement, but it competes directly with an unavoidable future crossing. Randomly crossing every visible gap cannot solve the level.

13. Level 5 — Junction Investment
Player-facing discovery

The player approaches gap (7,6) with two planks. Installing it creates a four-way wooden junction.

Three possibilities become available from that bridge:

south leads to one nearby plank;

north contains two consecutive gaps and then four planks;

east is the exit route, containing four gaps.

After paying for the central bridge, only one plank remains. Neither the northern chain nor the eastern route can be completed. The nearby southern plank provides the productive experiment: leave the bridge downward, collect it, and return.

With two planks, the player can cross the northern chain, collect four planks, return over all three existing bridges for free, and spend the four-plank stock on the exit route.

A player who enters the first northern gap too early can return and recover, so experimentation is supported without hiding the resource requirement.

Initial state
Initial inventory: 2
Start:             (3,6)
Exit:              (14,3)

Loose planks:
    (4,9)
    (5,1), (6,1), (8,1), (9,1)

Gaps:
    central junction: (7,6)
    northern chain:   (7,4), (7,2)
    exit route:       (9,6), (11,6), (12,4), (13,3)

Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~PP.PP~~~~~
02  ~~~~~~~G~~~~~~~
03  ~~~~~~~.~~~~.GE
04  ~~~~~~~G~~~~G~~
05  ~~~~~~~.~~~~.~~
06  ~~~S...G.G.G.~~
07  ~~~~~~~.~~~~~~~
08  ~~~~~...~~~~~~~
09  ~~~~P.~~~~~~~~~
10  ~~~~~~~~~~~~~~~
11  ~~~~~~~~~~~~~~~
Reference solution
R4,

D2, L2, D, L,
R, U, R2, U2,

U5,
L2, R4, L2,
D5,

R5, U3, R2

Phase interpretation:

R4: install the central junction (7,6).

Southern trip: collect (4,9) and return to the junction.

Northern trip: install (7,4) and (7,2), collect all four northern planks, and return over the same route.

Eastern trip: install the four exit-route gaps and enter (14,3).

Inventory trace:

2
→ central junction: 1
→ southern plank:   2
→ northern gaps:    0
→ four north planks:4
→ four exit gaps:   0
Necessity check

Total available stock is:

initial 2 + loose 5 = 7

Exactly seven bridges are required:

central 1 + north 2 + exit 4 = 7

Therefore:

the central bridge is mandatory;

the southern plank is mandatory;

both northern gaps are mandatory;

all four northern planks are mandatory;

every exit-route gap is mandatory.

The central bridge has contrasting roles: initial expense, access junction, return tile, and final departure point.

14. Level 6 — The Route You Need Twice
Player-facing discovery

The start is a central hub with two major branches.

The left bridge (6,6) leads into a region containing two planks. That same region also contains the beginning of the final five-gap exit route. Entering the final route early is tempting, but doing so would leave too little stock to reach the large eastern supply.

The upper bridge (7,5) leads to two more planks and then a winding four-gap chain. Beyond that chain is a chamber containing five planks.

The efficient structure is:

Visit the left region for its two planks.

Return to the hub without beginning the exit chain.

Enter the upper region and collect its two planks.

Spend exactly four planks reaching the eastern chamber.

Collect five planks and return across the permanent chain.

Reuse the left bridge and spend those five planks on the exit route.

Thus the left bridge changes role from supply entrance to final-route access, while the upper bridge and four-chain serve as both outbound investment and free return terrain.

Initial state
Initial inventory: 2
Start:             (7,6)
Exit:              (7,10)

Left-region planks:
    (3,7), (5,8)

Upper-region planks:
    (5,3), (9,3)

Eastern-chamber planks:
    (9,8), (10,8), (11,8), (9,10), (11,10)

Gaps:
    left bridge:        (6,6)
    upper bridge:       (7,5)

    eastern chain:
        (10,4), (12,4), (13,6), (12,7)

    final exit route:
        (2,8), (1,9), (2,10), (4,10), (6,10)

Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~~~~~~~~~
02  ~~~~~~~~~~~~~~~
03  ~~~~~P...P~~~~~
04  ~~~~~~~...G.G.~
05  ~~~~~~~G~~~~~.~
06  ~~~...GS~~~~~G~
07  ~~~P..~~~~~.G.~
08  ~.G..P~~~PPP~~~
09  ~G~~~~~~~...~~~
10  ~.G.G.GE~P.P~~~
11  ~~~~~~~~~~~~~~~
Reference solution
L3, D, L, D, R2, U2, R2,

U3, L2, R4, D,
R4, D3, L2,

D, L2, D2, R2, U3,

R2, U3, L4,
L2, D2,

L2, D2, L4, D2, R6

Phase interpretation:

Phase A: early left-region visit
L3, D, L, D, R2, U2, R2

Install (6,6), collect (3,7) and (5,8), and return to the hub.

Phase B: upper supply and eastern investment
U3, L2, R4, D

Install (7,5) and collect (5,3) and (9,3).

R4, D3, L2

Install all four eastern-chain gaps and enter the eastern chamber.

Phase C: chamber sweep and return
D, L2, D2, R2, U3

Collect all five chamber planks.

R2, U3, L4,
L2, D2

Return across the four eastern bridges and the upper bridge to the hub. No stock is spent during the return.

Phase D: reuse left access and exit
L2, D2, L4, D2, R6

Cross the already installed left bridge, enter the final route, install its five gaps, and reach the exit.

Inventory trace
Initial:                         2

Install left bridge:             1
Collect two left planks:         3

Install upper bridge:            2
Collect two upper planks:        4

Install four eastern bridges:    0
Collect five chamber planks:     5

Return across all bridges:       5
Install five exit bridges:       0
Necessity and anti-cleanliness check

Total stock is exactly:

2 initial + 2 left + 2 upper + 5 chamber = 11

The map contains exactly eleven gaps, and all eleven must be installed.

The level is not a sequence of independent collection errands:

the left region must be entered in an early phase and revisited in the final phase;

beginning its exit chain early prevents access to the eastern supply;

the upper branch both provides stock and carries the only route to the eastern chamber;

the four eastern bridges must later function as a free return path;

the exit sits inside a previously visited region rather than after the last supply chamber.

15. Level 7 — Shared Stock Network
Player-facing discovery

The final level contains two bridge junctions.

First junction: (5,6)

Once installed, it connects:

the starting side;

a southern one-plank branch;

the northern two-gap supply route;

the eastern trunk toward the second junction.

A separate gap at (3,4) appears to offer a faster northern approach. It reaches the same pre-chain area already opened by the first junction. Because the first junction is still required to reach the southern stock and eastern trunk, paying for (3,4) creates a redundant bridge rather than replacing a required one.

Second junction: (9,6)

Once installed, it connects:

the western trunk;

a free northeastern branch containing two planks;

a southeastern two-gap branch containing three planks;

the five-gap exit route.

The two eastern supply branches can be visited in either order once the second junction is active. The reference route visits the southeastern branch first because its return aligns efficiently with the northeastern branch and the final route.

The total available stock exactly covers every required bridge but not the redundant northern shortcut.

Initial state
Initial inventory: 2
Start:             (3,6)
Exit:              (12,10)

Southern first-junction plank:
    (3,9)

Northern planks:
    (7,1), (9,1), (9,3)

Northeastern planks:
    (11,4), (13,4)

Southeastern planks:
    (6,10), (8,10), (7,11)

Gaps:
    redundant shortcut:
        (3,4)

    first junction:
        (5,6)

    northern chain:
        (5,3), (6,2)

    second junction:
        (9,6)

    southeastern chain:
        (9,8), (8,9)

    final route:
        (11,6), (12,7), (13,8), (14,9), (13,10)

Walls:             none
Initial bridges:   none
Selection/cursor:  none / none
Status:            RUNNING
Map
    012345678901234
00  ~~~~~~~~~~~~~~~
01  ~~~~~~~P.P~~~~~
02  ~~~~~.G...~~~~~
03  ~~~~~G~..P~~~~~
04  ~~.G..~~~~.P.P~
05  ~~.~~.~~~..~~~~
06  ~~.S.G...G.G.~~
07  ~~~~~.~~~.~~G~~
08  ~~~...~~~G~~.G.
09  ~~~P~~~.G.~~~~G
10  ~~~~~~P.P~~~EG.
11  ~~~~~~~P.~~~~~~
Reference solution
R2,

D2, L2, D, U, R2, U2,

U4, R2,
U, R2, D2, U,
L4, D4,

R4,

D3, L,
D2, L, U, L, R, U, R2, U3,

U, R, U, R3,
L3, D2,

R2, D2, R2, D2, L2

Total: 73 directional actions.

Phase interpretation
Phase A: install first junction and collect southern stock
R2
D2, L2, D, U, R2, U2

Install first junction (5,6).

Leave it southward.

Collect (3,9).

Return to the junction.

Inventory:

2 → first junction: 1 → southern plank: 2
Phase B: northern chain and three-plank chamber
U4, R2
U, R2, D2, U
L4, D4

Travel north from the junction.

Install (5,3) and (6,2).

Collect (7,1), (9,1), and (9,3).

Return over both bridges to the first junction.

Inventory:

2 → two northern gaps: 0 → three planks: 3
Phase C: install second junction
R4

Move along the eastern trunk and install (9,6).

Inventory:

3 → second junction: 2
Phase D: southeastern branch
D3, L,
D2, L, U, L, R, U, R2, U3

Leave the second junction downward.

Install (9,8) and (8,9).

Collect all three southeastern planks.

Return to the second junction over the same bridges.

Inventory:

2 → two southeast gaps: 0 → three planks: 3
Phase E: northeastern branch
U, R, U, R3,
L3, D2

Collect (11,4) and (13,4), then return to (10,6), adjacent to the final route.

Inventory:

3 → two northeastern planks: 5
Phase F: final five-gap route
R2, D2, R2, D2, L2

Install:

(11,6)
(12,7)
(13,8)
(14,9)
(13,10)

Then enter exit (12,10).

Inventory:

5 → five final gaps → 0
Exact resource proof

Available stock:

initial:             2
southern branch:     1
northern chamber:    3
northeastern branch: 2
southeastern branch: 3
total:              11

Required nonredundant bridges:

first junction:      1
northern chain:      2
second junction:     1
southeastern chain:  2
final route:         5
total:              11

Gap (3,4) would be a twelfth installation.

It reaches the northern pre-chain area but does not eliminate:

the first junction, which remains necessary for the southern branch and eastern trunk;

either northern-chain gap;

the second junction;

either southeastern gap;

any final gap.

Consequently, installing (3,4) makes completion impossible by exactly one plank. The failure remains fully visible rather than being declared by an abstract checker.

Anti-cleanliness check

The mature level forms a dependency network rather than a single line:

the first junction unlocks three strategically different branches;

the northern and southern branches jointly fund later expansion;

the second junction exposes two supply branches whose order is flexible;

the southeastern route is first an expense and later a free return path;

the northeastern branch is free but spatially offset from the final route;

the redundant shortcut is encountered early but judged using future stock needs;

the exit route bends around the eastern edge and ends back toward the center rather than simply continuing past the final supply.

Every loose plank is necessary. Every gap except (3,4) is necessary. The one optional gap tests a specific established misconception: treating every profitable-looking bridge as worth installing.

16. Required implementation tests

An implementation should pass at least the following deterministic checks.

Bridge placement

Starting Level 1 and applying RIGHT once must produce:

player_position = (6,6)
inventory = 0
terrain[6][6] = BRIDGE
status = RUNNING
Empty-gap behavior

Starting Level 2 and applying RIGHT must leave all persistent state unchanged.

Collection

Starting Level 2 and applying UP, UP must produce:

player_position = (5,4)
inventory = 1
loose_planks = empty
Free recrossing and sideways exit

In Level 3, after:

U2, L2, R2

the player is at (5,5) with inventory 2. Applying DOWN, RIGHT must:

move onto bridge (5,6);

then move sideways to floor (6,6);

leave inventory equal to 2.

Void does not consume stock

In any level, moving from a traversable cell toward adjacent ~ must leave inventory unchanged.

Permanent bridge

After leaving an installed bridge and later returning, its terrain must still be BRIDGE.

No-op actions

For every running and won state:

step(SPACE)
step(click at any pixel)

must return the unchanged persistent state.

Level completion

Entering the authored exit coordinate must win regardless of remaining inventory or unrelated unbridged gaps.

Settled-frame sufficiency

Two persistent states that differ in any of the following must render differently:

player position;

inventory count;

loose-plank set;

gap-versus-bridge terrain;

running-versus-won status.

No strategically distinct settled states may share the same rendered frame.
