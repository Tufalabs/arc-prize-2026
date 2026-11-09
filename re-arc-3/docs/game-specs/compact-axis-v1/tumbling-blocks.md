Worked for 43m 9s
Tumbling Blocks

This is a game about pushing a cube becoming difficult because every step changes which face points where.

Authored level count: exactly 7.

The route taken by a cube is simultaneously a spatial route and a rotation program. Walls determine not only where the cube can travel, but where the pusher can stand for the next roll. Terminals record successful face contacts permanently, while docks demand the cube’s final position and complete orientation. Later levels become difficult through turn order, closed reorientation loops, constrained departure sides, temporary staging, and two cubes competing for the same turning spaces.

There is no action budget, timer, damage state, or other pressure against experimentation.

1. Logical space and controls

The gameplay lattice is 9×9, with coordinates:

x = 0…8, increasing left to right.

y = 0…8, increasing top to bottom.

UP changes y by −1.

DOWN changes y by +1.

LEFT changes x by −1.

RIGHT changes x by +1.

Only UP, DOWN, LEFT, and RIGHT are active controls.

SPACE and cell clicks are not used. They produce a brief neutral pulse on the pusher but leave every gameplay variable unchanged. There are no clickable-looking objects and no cursor or selected-object state in any level.

A movement action behaves as follows:

If the neighboring cell is open, the pusher moves into it.

If the neighboring cell contains a cube, the action attempts to push it one cell in the same direction.

The push succeeds only if the cube’s destination is an open floor or dock cell unoccupied by another cube.

On a successful push, the cube rolls one cell and changes orientation; the pusher enters the cube’s former cell.

There is no pulling, chain pushing, rotating in place, or swapping.

Walls, terminal wall cells, board edges, and other cubes block rolls.

A blocked cube visibly rocks toward the obstruction and returns. Neither its position nor orientation changes.

Walking into a wall produces a brief edge flash and no state change.

2. Rendering contract
Canvas and board

Canvas: 64×64 pixels.

Logical cell size: 7×7 pixels.

The 9×9 board occupies pixels 0…62 on both axes.

Pixel row 63 and column 63 use color 5 as a quiet outer separator.

Do not draw per-cell grid lines.

Adjacent wall cells render as one connected dark mass, not as individually bordered tiles.

Structural colors
Purpose	Color ID
Floor	0 white
Cube chassis	1 light gray
Neutral rims and empty-dock status	2 gray
Shadows and recessed wall details	3 dark gray
Walls and terminal bodies	4 very dark gray
Outer background and glyph marks	5 black
Blue cube identity pins	9 blue
Pink cube identity pins	7 pink
Pusher	15 purple
Incorrect face/contact feedback	13 maroon
Lit terminal / correct dock ticks	14 green

The identity pins matter only when multiple cubes appear. A blue-pinned cube can activate only blue-pinned terminals and satisfy only the blue-pinned dock; likewise for pink. Levels 1–5 repeatedly show the blue cube, dock, and terminals with matching pins before the second identity appears.

Cube faces

Every cube has the same six physical faces:

Face ID	Color	Symbol
YD	11 yellow	hollow diamond
BP	10 light blue	plus
GB	14 green	vertical bar
OR	12 orange	ring
MX	6 magenta	X
RD	8 red	dot

The 3×3 dark glyph masks are:

YD     BP     GB     OR     MX     RD
.#.    .#.    .#.    ###    #.#    ...
#.#    ###    .#.    #.#    .#.    .#.
.#.    .#.    .#.    ###    #.#    ...

These are geometric marks, not rendered text.

Cube silhouette

Within a 7×7 logical cell:

Draw a shadow first, shifted one pixel southeast, using color 3.

The main cube body is a connected 5×5 schematic.

The central 3×3 region shows the current top face’s color and glyph.

The north, east, south, and west faces appear as three-pixel edge bands on the corresponding edges.

Two chassis corner pixels show the cube identity color, 9 or 7.

A 2×2 southeast chip shows the bottom face color beside the shadow. A dark notch distinguishes the chip from the body.

Thus a settled cube visibly exposes all six face identities:

Top through the central symbol.

Four sides through directional edge bands.

Bottom through the inset chip.

No orientation history needs to be remembered.

Docks

A dock is a floor cell, not an obstacle.

It shows:

Required north, east, south, and west faces as colored rails along the cell’s outer edges.

Required top face as a colored central symbol while empty.

When occupied, a small top-face tab remains visible above the cube.

Matching identity pins.

Persistent status ticks:

Gray when empty.

Maroon when occupied by the wrong cube or wrong orientation.

Green when occupied correctly.

A cube on a dock can still be pushed away. Dock satisfaction is live, not permanent.

Terminals

A terminal occupies a wall cell and is therefore solid.

Its plate is drawn on the edge facing its contact cell. The plate visibly shows:

The accepted face’s color and glyph.

Its cube identity pins.

A gray rim while unlit.

A green rim while lit.

On a wrong matching-cube contact, the terminal rim and the cube’s contacting edge pulse maroon. A wrong-identity contact produces a neutral gray pin pulse instead. Neither failure has a persistent penalty.

3. Cube orientation algebra

Every authored cube starts in the same standard orientation:

O0 = (top, north, east, south, west, bottom)
   = (YD, BP, GB, OR, MX, RD)

For an orientation

O = (T, N, E, S, W, Btm)

the four legal rolls are:

UP:
(S, T, E, Btm, W, N)

DOWN:
(N, Btm, E, T, W, S)

RIGHT:
(W, N, T, S, Btm, E)

LEFT:
(E, N, Btm, S, T, W)

Consequences that follow without exceptions:

Four rolls in the same direction restore the full orientation.

A roll followed immediately by its inverse restores both position and orientation.

Perpendicular rolls generally do not commute.

A closed spatial route can return a cube to its starting cell with a different orientation.

4. Terminal contact semantics

Each terminal has:

wall_position
contact_position
contact_side ∈ {north, east, south, west}
accepted_face
identity
lit

A terminal is evaluated only when a matching cube finishes a legal roll in its contact cell.

For example, a terminal at (3,2) with contact cell (3,3) is north of the cube. It compares the cube’s north face after the roll settles.

Rules:

Position and orientation commit at the end of the roll.

Every unlit terminal adjacent to the moved cube is then checked.

A matching identity and accepted contacting face lights the terminal permanently.

A wrong face produces only the maroon pulse.

A wrong cube identity cannot light the terminal.

Pusher-only movement does not trigger terminals.

A lit terminal never turns off and ignores later contacts.

No authored level begins with a cube already in a terminal contact cell.

5. Update and animation order

A legal cube push requires a short deterministic animation.

Successful roll animation

The animation has four presentation subframes:

Cube translated two pixels toward the destination; old top begins compressing.

Cube translated four pixels; the rolling side becomes visually dominant.

Cube translated six pixels; the new top and edge bands appear.

Cube settles at the full seven-pixel displacement with its final orientation.

At subframe 4:

Cube position and orientation commit.

Pusher enters the cube’s former logical cell.

Terminal checks occur.

Dock states are recomputed.

The win validator runs.

Animation frames consume no actions, budget, or game time. Input is locked until the settled frame. A step() call is the only source of gameplay-state change; there are no autonomous updates between actions.

Failed push animation

A blocked cube shifts one pixel toward the obstruction, flashes the obstructed edge, and returns. The logical state never changes.

Completion

When the validator first becomes true:

All terminal rims and correct-dock ticks flash green for three subframes.

The level enters complete = true.

Further actions leave the settled state unchanged.

6. Success validator

A level succeeds exactly when all of the following hold after an action settles:

Every required terminal is lit.

Every cube occupies its identity-matching dock.

Every docked cube’s full six-face orientation equals the dock orientation.

No cube remains off-dock.

The validator ignores:

Pusher position.

Number of actions taken.

Incorrect terminal attempts.

Previous wrong dock occupations.

Transient animation and feedback state.

A lit terminal remains satisfied after its cube leaves. A dock does not remain satisfied after its cube leaves.

7. Authored-map notation

The following ASCII maps are implementation documentation only. No text is rendered in the game.

#  wall or non-floor cell
.  floor
P  pusher
C  single blue cube
A  blue cube
B  pink cube
T  terminal wall cell
d  single blue dock
1  blue dock
2  pink dock

A map specifies every logical cell exactly. T is blocking wall terrain; docks are floor.

Action replays use:

U = UP
D = DOWN
L = LEFT
R = RIGHT

Separators shown in some replay strings are explanatory only and are not actions.

Level 1 — First Tumble

Purpose: Establish walking, pushing, rolling, and orientation-matched docking through one short physical route.

Intended solution depth: 4 actions.

#########
#########
#....####
#PC..####
#.#d.####
#....####
#########
#########
#########
Initial state

Pusher: (1,3)

Blue cube: (2,3)

Cube orientation: O0

Blue dock: (3,4)

Dock orientation:

(BP, GB, YD, MX, RD, OR)

Terminals: none

Selection/cursor: none

Player-facing discovery

The pusher, cube, and nearby dock form a visible physical objective. Pressing RIGHT immediately rolls the cube and visibly changes its top, directional bands, and bottom chip.

The wall at (2,4) prevents the cube from simply being pushed downward from its original column. After the first roll, the open space above the cube gives the pusher an obvious way to get behind it for a downward push.

The second roll puts the cube on the dock with every visible face matching. The dock ticks turn green and completion follows immediately.

Verified solution

Cube push route:

R, D

Exact action replay:

RURD

Final settled state:

Pusher: (3,3)

Cube: (3,4)

Cube orientation: (BP, GB, YD, MX, RD, OR)

Complete: true

Attribution check

The first push has one immediate consequence: the cube moves one cell and visibly tumbles. No terminal, second cube, decoy, budget, or simultaneous state change is present.

The wall at (2,4) is necessary: it creates the turn and prevents the first level from being solved by repeatedly pressing a single direction.

Level 2 — Same Top, Wrong Sides

Purpose: Establish that the complete route order matters, not merely final position or top face.

Intended solution depth: 11 actions.

#########
#......##
#.PC.#.##
#....d.##
#......##
#......##
#########
#########
#########
Initial state

Pusher: (2,2)

Blue cube: (3,2)

Cube orientation: O0

Blocking wall: (5,2)

Blue dock: (5,3)

Dock orientation:

(BP, YD, MX, RD, GB, OR)

Terminals: none

Selection/cursor: none

Player-facing discovery

The obvious direct push to the right works once, then the cube reaches the wall at (5,2) and rocks. Open space above and below presents two short routes around the obstruction.

Two especially natural cube routes are:

Upper route: U, R, D, D, R
Lower route: D, D, R, R, U

They have:

The same five roll directions as a multiset: one U, two D, and two R.

The same endpoint, (5,3).

The same final top face, BP.

The same final bottom face, OR.

But they do not produce the same side orientation.

Upper route result:

(BP, RD, GB, YD, MX, OR)

Lower route result:

(BP, YD, MX, RD, GB, OR)

The dock requests the lower result. A player who watches only the top sees an apparent match after the upper route, but the directional rails visibly disagree. This makes the side bands causally necessary rather than decorative.

Verified solution

Cube push route:

D, D, R, R, U

Exact action replay:

URDDLDRRDRU

Final settled state:

Pusher: (5,4)

Cube: (5,3)

Cube orientation: (BP, YD, MX, RD, GB, OR)

Complete: true

Attribution check

The wall at (5,2) removes the three-push shortcut R,R,D, which would otherwise reach the required orientation. The two longer routes create the distinguishing experiment.

No new object accompanies the orientation lesson. A wrong route remains recoverable in the open lower room.

Level 3 — The Face Plate

Purpose: Introduce terminal contact, accepted side faces, and permanent terminal memory.

Intended solution depth: 5 actions.

#########
#########
###T#####
#....d.##
#..C...##
#..P...##
#......##
#########
#########
Initial state

Pusher: (3,5)

Blue cube: (3,4)

Cube orientation: O0

Blue terminal:

Wall: (3,2)

Contact cell: (3,3)

Contact side: north

Accepted face: YD

Initially unlit

Blue dock: (5,3)

Dock orientation:

(BP, YD, MX, RD, GB, OR)

Selection/cursor: none

Player-facing discovery

The cube is directly between the pusher and the unfamiliar wall plate. The only immediate push toward the plate is UP.

That roll places the cube at (3,3) with orientation:

(OR, YD, GB, RD, MX, BP)

Its north band is now YD, exactly matching the plate. The terminal lights immediately.

The terminal wall prevents another northward roll, creating a local impasse that forces a turn. The pusher walks around the cube and pushes it east twice to the familiar dock orientation from Level 2. The terminal remains visibly lit after the cube leaves, establishing permanence through play.

Verified solution

Cube push route:

U, R, R

Exact action replay:

ULURR

Final settled state:

Pusher: (4,3)

Cube: (5,3)

Cube orientation: (BP, YD, MX, RD, GB, OR)

Terminal lit: true

Complete: true

Attribution check

The terminal is the only new causal object. The first cube roll simultaneously exposes the accepted north face and lights the adjacent plate; no second rule changes at that moment.

Ignoring the terminal cannot complete the level even if the dock is correct.

Level 4 — Five Is One, but Order Still Matters

Purpose: Make the four-roll straight cycle useful, then combine it with noncommuting turn order and permanent terminal state.

Intended solution depth: 26 actions.

........#
PC.d....#
........#
...##...#
...##...#
........#
.......T#
........#
#########
Initial state

Pusher: (0,1)

Blue cube: (1,1)

Cube orientation: O0

Central wall block:

(3,3), (4,3), (3,4), (4,4)

Blue terminal:

Wall: (7,6)

Contact cell: (6,6)

Contact side: east

Accepted face: BP

Initially unlit

Blue dock: (3,1)

Dock orientation:

(OR, MX, YD, GB, RD, BP)

Selection/cursor: none

Player-facing discovery

The central wall offers two plausible routes from the northwest start to the southeast terminal contact:

Down first:  D×5, R×5
Right first: R×5, D×5

During either five-cell straight run, the cube visibly returns to its starting orientation after the fourth roll. The fifth roll therefore has the same orientation effect as a single roll in that direction.

The two routes arrive at the same contact cell after the same number of pushes, but their orientations differ.

Down-first contact orientation:

(MX, RD, BP, YD, OR, GB)

The east face is BP, so the terminal lights.

Right-first contact orientation:

(BP, GB, YD, MX, RD, OR)

The east face is YD, so the terminal gives a dull maroon pulse. The mismatch is visible on the exact edge touching the plate.

After the correct contact, the cube must continue north five cells and then west three cells to its dock. The terminal remains lit, but the cube carries the terminal-contact orientation into the remaining route.

Verified solution

Cube push route:

D×5, R×5, U×5, L×3

Exact action replay:

URDDDDDLDRRRRRDRUUUUURULLL

Final settled state:

Pusher: (4,1)

Cube: (3,1)

Cube orientation: (OR, MX, YD, GB, RD, BP)

Terminal lit: true

Complete: true

Necessity and anti-cleanliness check

The level is not a linear tour:

Two equally long branches reach the same terminal cell.

The branch distinction is orientation, not distance or object collection.

The central block is required to give the order choice spatial meaning.

Three-cell-wide outer regions are needed for the pusher to change pushing sides.

The terminal wall fixes which departure directions remain possible after contact.

The dock lies on a revisited upper region rather than immediately after the terminal.

Removing orientation checks, the central block, or the terminal would collapse the puzzle.

Level 5 — Return Changed

Purpose: Introduce a closed spatial loop as a deliberate orientation tool, then reuse carried orientation in an irregular terminal route.

Intended solution depth: 49 actions.

....T....
PC.......
.........
...##....
...##.#..
...##...T
.d.......
.........
.........
Initial state

Pusher: (0,1)

Blue cube: (1,1)

Cube orientation: O0

Walls:

(3,3), (4,3)

(3,4), (4,4)

(3,5), (4,5)

(6,4)

Blue terminal T1:

Wall: (4,0)

Contact cell: (4,1)

Contact side: north

Accepted face: YD

Initially unlit

Blue terminal T2:

Wall: (8,5)

Contact cell: (7,5)

Contact side: east

Accepted face: GB

Initially unlit

Blue dock: (1,6)

Dock orientation:

(OR, RD, MX, YD, GB, BP)

Selection/cursor: none

Player-facing discovery

The immediate-looking plan is to push east toward T1. Without reorientation, the cube reaches (4,1) with BP on its north side, while the plate requests YD. The pulse shows the mismatch.

The start area contains a compact square route:

(1,1) → (2,1) → (2,2) → (1,2) → (1,1)

Rolling through it with:

R, D, L, U

returns the cube to its original cell but changes its orientation from O0 to:

(MX, YD, OR, RD, BP, GB)

The north face is now YD. The cube has visibly returned to the same position while becoming usable in a new way. Pushing east again lights T1.

The middle walls then force an offset sequence of turns toward T2. T2’s wall occupies (8,5), so after the cube lights it at (7,5), the pusher cannot stand directly east of the cube to push west. The cube must first be carried down to (7,6). The pusher then takes the lower flank and begins the final westward run.

Thus T2 is both a goal and a physical liability: satisfying it commits the cube to a particular departure.

Verified solution

Cube push route:

R, D, L, U,
R×4,
D, R, D, R, D×3,
L×6

Exact replay, split by logical phase:

RURDRDLDLU
LURRRR
DRRUULDLDRURDLDRURDDD
LDDRRULLLLLL

Concatenated replay:

RURDRDLDLULURRRRDRRUULDLDRURDLDRURDDDLDDRRULLLLLL

Final settled state:

Pusher: (2,6)

Cube: (1,6)

Cube orientation: (OR, RD, MX, YD, GB, BP)

T1 lit: true

T2 lit: true

Complete: true

Necessity and anti-cleanliness check

The start loop is not optional: the direct terminal approach exposes the wrong north face.

The later route is not a sequence of adjacent devices:

T1 is followed by an asymmetric wall weave.

The isolated wall at (6,4) prevents a shorter diagonal correction.

T2 changes the available pusher stance after activation.

The dock is reached only after carrying the resulting orientation across a six-cell westward run.

The same terminal rule first acts as an orientation test and later as a route commitment.

Level 6 — Shared Turning Bay

Purpose: Introduce two cubes as mutually blocking spatial objects. One cube must be staged temporarily so the other can use the shared center.

Intended solution depth: 51 actions.

###...###
###...###
###.....#
........T
..PBA...1
.........
###...T##
###....##
###.2..##
Initial state

Pusher: (2,4)

Blue cube A: (4,4)

Pink cube B: (3,4)

Both orientations: O0

Blue terminal:

Wall: (8,3)

Contact cell: (7,3)

Contact side: east

Accepted face: RD

Initially unlit

Pink terminal:

Wall: (6,6)

Contact cell: (5,6)

Contact side: east

Accepted face: MX

Initially unlit

Blue dock 1:

Position: (8,4)

Required orientation:

(YD, BP, GB, OR, MX, RD)

Pink dock 2:

Position: (4,8)

Required orientation:

(RD, GB, BP, MX, OR, YD)

Selection/cursor: none.

Player-facing discovery

The first obvious RIGHT action attempts to push pink B from (3,4) into blue A at (4,4). The push fails visibly: B rocks against A. This is not a new collision exception—other cubes obey the same solid-destination rule as walls.

The open region below the pair lets the pusher reach the south side of A. Pushing A north twice clears the shared center, but A is not solved; it is only staged.

Pink B then travels through the center into the lower-right arm. Its terminal wall at (6,6) blocks the direct route to the pusher stance needed for the later left push. The pusher must flank around through row 8. Once B reaches its dock, it becomes a persistent obstacle while A is retrieved.

A returns through the center, travels east, briefly rolls north into its terminal contact, rolls south again, and only then enters its dock. The north/south excursion restores A’s pre-excursion orientation but records the terminal contact.

The operational rules never change:

A is first an obstacle.

Then A is a temporarily preserved resource.

B is first the active payload.

Docked B later becomes part of A’s route geometry.

Verified solution

Logical cube pushes:

Stage A: U, U

Solve B:
R, R, D, D, D, L, D

Solve A:
D, D, R, R, R, U, D, R

Exact replay by phase:

DRRUU

LLDRRURDDDLDDRRULULD

UUUULUURDDLDRRRDRULUURDLDR

Concatenated replay:

DRRUULLDRRURDDDLDDRRULULDUUUULUURDDLDRRRDRULUURDLDR

Final settled state:

Pusher: (7,4)

Blue A: (8,4)

Blue orientation: (YD, BP, GB, OR, MX, RD)

Pink B: (4,8)

Pink orientation: (RD, GB, BP, MX, OR, YD)

Both terminals lit

Complete: true

Necessity and anti-cleanliness check

B cannot enter the center until A moves.

Moving A south instead of north occupies B’s required lower route.

The pink terminal wall creates the lower flanking requirement; it is not merely a validator.

A’s direct four-roll east route would satisfy its dock but leave the blue terminal unlit.

B’s occupied dock changes the pusher’s retrieval route for A.

The level cannot be reduced to independently solving two cubes; the first cube’s temporary position controls access for the second.

Level 7 — Interleaved Circuit

Purpose: Final composition. Both cubes must be partially advanced, restaged, and completed in an interleaved order. Correctly docking either cube too early destroys a pusher stance needed for the other.

Intended solution depth: 85 actions.

####..###
###...###
###.....#
##......T
##PBA...1
##.......
###...T##
###....##
####2..##
Initial state

Pusher: (2,4)

Blue cube A: (4,4)

Pink cube B: (3,4)

Both orientations: O0

Pink terminal:

Wall: (8,3)

Contact cell: (7,3)

Contact side: east

Accepted face: GB

Initially unlit

Blue terminal:

Wall: (6,6)

Contact cell: (5,6)

Contact side: east

Accepted face: RD

Initially unlit

Blue dock 1:

Position: (8,4)

Required orientation:

(GB, YD, BP, RD, OR, MX)

Pink dock 2:

Position: (4,8)

Required orientation:

(MX, BP, YD, OR, RD, GB)

Selection/cursor: none.

Player-facing discovery

The opening repeats the visible collision from Level 6: B cannot move right because A occupies its destination. This time A must be pushed north three cells into the upper chamber.

B then crosses the center, travels to (7,4), and rolls north into (7,3), lighting the pink terminal. B is not docked; it is deliberately parked in an upper-east bay.

With the center clear, A travels from the north chamber down to (4,6). The lower court supports the established orientation loop:

R, D, L, U

The first rightward roll enters (5,6) and lights A’s terminal. Completing the loop returns A to (4,6) with a changed orientation. A is then pushed north four cells to (4,2), clearing the center again while preserving that new orientation.

Only now can B be retrieved. B rolls down to (7,4), west through the center, and south to the pink dock.

Finally A returns to the center and rolls east four cells into the blue dock.

Why the order is forced

This is a dependency network rather than a list of terminal errands.

B cannot begin before A is staged.
The initial chain push is blocked.

B cannot be docked before A performs the lower loop.
A’s loop requires the pusher to flank through the lower chamber, including the route around (4,8). A pink cube parked on dock 2 blocks that flank.

A cannot be docked before B leaves the east chamber.
After B rolls down from (7,3) to (7,4), pushing B west requires the pusher to stand at (8,4). That cell is blue dock 1. A cube already docked there would permanently trap B because there is no pulling.

A must be restaged north before B crosses the center.
Otherwise A itself occupies the cells needed for B’s westward passage.

Thus a “finish one cube, then finish the other” policy fails in either order.

Verified solution

Logical push phases:

1. Stage A north:
   A: U×3

2. Park B in the east bay:
   B: R×4, U

3. Send A south, reorient it, and restage it:
   A: D×5, R, D, L, U, U×4

4. Retrieve and dock B:
   B: D, L×3, D×4

5. Finish A:
   A: D×2, R×4

Exact replay by phase:

DRRUUU

DLLDRRRRDRU

LUULUULDDDDDLDRURDLDDRRULDLUUUUU

RURRDLDDRRULLLULDDDD

UUUULUURDDLDRRRR

Concatenated replay:

DRRUUUDLLDRRRRDRULUULUULDDDDDLDRURDLDDRRULDLUUUUURURRDLDDRRULLLULDDDDUUUULUURDDLDRRRR

Final settled state:

Pusher: (7,4)

Blue A: (8,4)

Blue orientation: (GB, YD, BP, RD, OR, MX)

Pink B: (4,8)

Pink orientation: (MX, BP, YD, OR, RD, GB)

Both terminals lit

Complete: true

Necessity and anti-cleanliness check

The final level has no new object category. Its depth comes entirely from established rules:

Cube collision.

Identity-matched terminals and docks.

Permanent terminal state.

Live dock occupancy.

Noncommuting turns.

Closed-route reorientation.

Pusher-side access.

Terminal walls as physical obstructions.

The blue dock is simultaneously a future goal and a stance that must remain empty. The pink dock is simultaneously a future goal and part of the blue cube’s lower flanking route. The center is used by A, then B, then A again. Removing the spatial layout would remove the puzzle.

8. Implementation conformance checks

An implementation should satisfy these deterministic checks before loading the authored levels.

Orientation checks

Starting from O0:

RIGHT × 4 = O0
DOWN × 4  = O0
RIGHT, LEFT = O0
UP, DOWN    = O0

The Level 5 loop must produce:

O0 + R,D,L,U
= (MX, YD, OR, RD, BP, GB)

The two Level 2 routes must produce:

U,R,D,D,R
= (BP, RD, GB, YD, MX, OR)

D,D,R,R,U
= (BP, YD, MX, RD, GB, OR)
Collision checks

For adjacent pusher, cube B, and cube A in one line:

P B A .

pushing toward A must change no logical position or cube orientation. Only B’s blocked-rock feedback plays.

Terminal checks

Compare the cube face after the roll, not before it.

A matching face on a wrong-identity cube does not light the terminal.

A correct contact permanently sets lit = true.

Moving the cube away does not change lit.

A cube merely standing in a contact cell without having just rolled there does not activate an unlit terminal.

Dock checks

Dock status is derived after every settled action.

Correct occupancy does not lock the cube.

Moving the cube away immediately removes dock satisfaction.

Exact identity and complete orientation are required.

Pusher position never affects validation.

Settled-frame sufficiency

From any settled frame, an observer can determine:

Pusher position.

Every cube’s position, identity, and six-face orientation.

Every terminal’s accepted face, identity, mount direction, and lit state.

Every dock’s identity, target orientation, and current correctness.

All walls and possible pushing stances.

No strategically relevant state depends on hidden history, elapsed time, an unseen counter, or an animation phase.
