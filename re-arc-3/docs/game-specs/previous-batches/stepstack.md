Stepstack

This is a game about pushing loose blocks into staircases, becoming difficult because the scaffold needed to climb can obstruct its own construction or be needed elsewhere afterward.

Authored level count: exactly 7.

Stepstack is a side-view construction puzzle with only four visible nouns: the avatar, terrain, identical loose blocks, and a green exit. Blocks can be pushed horizontally but never pulled. Unsupported blocks fall, stack, and become walkable structure. The same block may begin as a catcher, become temporary access to a work platform, and finish as the lowest step of the exit staircase.

The objective is always physical and visible: move the blue avatar into the green exit cell. Block arrangement, move count, facing, and unused terrain are irrelevant once the avatar enters the exit.

1. Logical space and coordinates

Every level uses a 15×11 logical lattice.

Columns: x = 0…14, left to right.

Rows: y = 0…10, top to bottom.

A coordinate is written (x,y).

Outside the lattice is void and cannot be entered.

Each movement action traverses one edge of the logical movement graph. A climb or descent changes both x and y, but is still one logical move.

The bottom authored floor is normally row y=10.

The level diagrams below contain exactly 15 characters per row and 11 rows:

#: fixed terrain

B: loose block

> or <: avatar and initial facing

E: exit

.: empty cell

Letters such as A, B, C, D, G, and H in the prose are documentation labels based on initial position. They are not rendered. All loose blocks look and behave identically.

2. Rendering contract
Canvas

Render canvas: 64×64 pixels.

Gameplay origin: pixel (2,10).

Logical cell size: 4×4 pixels.

Gameplay area: pixels x=2…61, y=10…53.

All remaining pixels use background color 4.

Do not draw logical grid lines.

Adjacent terrain cells visually merge into continuous shapes; do not draw internal borders between them.

Colors used
Role	Color ID
Empty/background	4
Terrain body	3
Exposed terrain top	1
Block outline	13
Block body	12
Block top highlight	11
Avatar body	9
Avatar face/direction	10
Exit	14
Exit interior/highlight	0
Invalid-action flash	8
Undo flash	6

No color outside 0–15 may be used.

Terrain

First merge all terrain cells. Fill the merged silhouette with color 3. For every exposed horizontal top edge—terrain cell present, cell immediately above absent—draw its uppermost pixel row in color 1.

Exposed side edges may use color 2, but internal cell boundaries must remain invisible.

Loose block

A block occupies one logical cell but is a connected multi-pixel object:

Four corner pixels: color 13

Central 2×2: color 12

The two upper central pixels: color 11

Remaining edge pixels: color 12

Individual block boundaries remain visible when stacked because counting vertical block height is mechanically important.

Avatar

The avatar occupies one logical cell. Render a connected blue body in color 9, with a light-blue protruding face in color 10 on the facing side. Facing must be readable even when a horizontal action only turns the avatar.

Exit

The exit is a passable 4×4 green arch:

Outer frame: color 14

Central opening: color 0

Render it behind the avatar when the avatar enters it. A block may theoretically occupy the exit cell and obscure it, although no authored solution requires this.

3. State

A stable game state contains:

player: (x,y)
facing: LEFT or RIGHT
blocks: unordered set of occupied cells
terrain: fixed set of occupied cells
exit: one passable cell
undo_stack: stack of prior stable states
won: boolean
transient_feedback: none, invalid, braced, undo, or completion

There is no selected block, cursor, inventory, energy, score requirement, or timer.

All seven levels use:

selected_object = null
cursor_state = absent
clickable_cells = empty set

The game itself has no action limit. SPACE provides unlimited single-step undo back to the initial state, allowing extensive experimentation.

4. Action semantics
LEFT and RIGHT

Let d=-1 for LEFT and d=+1 for RIGHT.

The action first sets facing=d.

Let:

adjacent = (player.x+d, player.y)
beyond   = (player.x+2d, player.y)

Resolution:

Horizontal walk

If adjacent is empty and the cell directly below it is terrain or a block, move the avatar to adjacent.

Horizontal push

If adjacent contains exactly one loose block and beyond is inside the board and empty:

Move that block from adjacent to beyond.

Tentatively move the avatar into adjacent.

Apply the braced-push safety rule below.

Settle all unsupported blocks.

Blocked action

Terrain, a second block in beyond, or the board edge prevents translation. The avatar may still visibly turn if the requested facing changed.

Blocks do not need support at their horizontal destination. A pushed block may immediately fall several rows.

UP: climb one step

Using the current facing direction d, define:

adjacent = (player.x+d, player.y)
landing  = (player.x+d, player.y-1)

The climb succeeds exactly when:

adjacent is terrain or a loose block;

landing is inside the board;

landing contains neither terrain nor a block.

The avatar moves to landing.

This permits climbing a one-cell step, climbing onto a loose block, or climbing over a loose block. It does not permit jumping, climbing a two-cell vertical face, or moving vertically without an adjacent support.

DOWN: descend one step

Using facing direction d, define:

side    = (player.x+d, player.y)
landing = (player.x+d, player.y+1)
support = (player.x+d, player.y+2)

The descent succeeds exactly when:

side is empty;

landing is empty;

support is terrain or a block.

The avatar moves to landing.

The avatar never free-falls multiple rows. A descent is always one diagonal step. This difference between block gravity and controlled avatar descent is central to several levels: a construction platform can be too high to descend directly onto the final stack.

SPACE: undo

SPACE restores the most recent stable state from undo_stack.

Undo is unlimited.

Intermediate animation frames are never stored.

Rejected actions and clicks are not stored.

If there is no prior state, flash the outer canvas edge in color 6 for two frames.

Undo does not itself invoke gravity.

CLICK

Clicking any cell is an exact no-op:

no selection;

no state change;

no gravity;

no history entry;

no clickable visual affordance.

Clicking is not part of Stepstack’s mechanic.

5. Block gravity

Terrain and other blocks support loose blocks. The avatar does not support blocks.

After every successful walk, climb, descent, or push:

Partition blocks in each column into vertically connected components.

A component is unsupported if the cell immediately below its bottom block is empty.

During one gravity tick, all unsupported components move down one logical row.

Repeat until every component is supported.

Vertical order within a column is preserved.

Blocks never slide diagonally.

If a block would leave the board or pass through the avatar, the action invokes the safety resolution below.

Initial states are not automatically settled during level loading. This matters in Level 1: its visibly suspended block falls only after the first successful player action.

6. Braced support pushes

Ordinary pushes advance the avatar into the block’s former cell. There is one deterministic safety consequence when pushing a load-bearing block:

Simulate the push with the avatar advancing normally.

Simulate gravity.

If a falling block would occupy the avatar’s advanced cell, repeat the simulation with the avatar remaining in its original cell.

If the original cell is safe, accept this braced push:

the pushed block moves;

the avatar remains in place;

the unsupported stack settles.

If both positions would be struck, reject the entire push.

This is not a separate action and is not a block subtype. It is the same collision rule everywhere. Level 4 creates the first situation where it is observable.

A braced push visibly communicates its cause: the avatar leans into the push but recoils to the original cell while the supported block above settles into the vacated column.

7. Update order

For every step(action):

If won=true, return the completed observation unchanged.

Resolve SPACE or click immediately if applicable.

Save the current stable state as a tentative undo snapshot.

Update facing for LEFT or RIGHT.

Resolve avatar translation or block push.

For pushes, apply normal-versus-braced safety simulation.

Settle blocks under gravity.

If the action was rejected, restore the entire pre-action position and block set. The requested horizontal facing may remain visible.

If the logical state changed, append the pre-action stable state to undo_stack.

Validate success.

Play the required deterministic animation.

Return the final stable observation.

Success is checked after all gravity has settled.

8. Animation and feedback

Animation frames do not consume actions and are not independently interactive.

Ordinary movement

Walks, climbs, and descents update immediately. No ornamental movement animation is required.

Horizontal block push

Animate the block across its four rendered pixels in exactly four frames. The logical destination is committed at the end of the fourth frame.

Falling

For every logical row fallen:

four interpolation frames;

one rendered pixel downward per frame.

A vertical component moves together. If an action causes both a push and a fall, show the four-frame horizontal slide first, followed by the falling frames.

Braced support push

Show:

Four-frame horizontal block slide.

Avatar face pixels briefly change to color 0 while the avatar remains in the original cell.

The released component falls using normal gravity animation.

Invalid push

Flash for two frames in color 8:

the blocked destination;

the pushed block;

every block vertically supported by that block.

Then return to the unchanged state.

Completion

When the avatar occupies the exit after settling, alternate the exit frame between colors 14 and 0 for four frames, then complete the level.

9. Progression
Level	Main discovery	Minimal reference depth
1. The Falling Step	Push, gravity, and one-step climbing	9
2. Outside First	A block pushed beneath an overhang becomes unusable	11
3. Borrowed Stair	Catch a falling block, then move the access block into the final stair	25
4. The Load-Bearing Block	Push a support out safely and let the structure settle	26
5. Finish the High Work	Visit the far upper block before dismantling access	45
6. The Delayed Riser	Build the base stack before releasing a middle-height block	58
7. Five-High Handoff	A braced push transfers two stacked blocks into successive final heights	66

The listed paths are minimal under the transition contract above.

10. Authored levels
Level 1 — The Falling Step
Player-facing discovery

The exit is visibly two cells above the floor. A block floats above the floor between the avatar and the wall. The first successful movement causes it to fall straight down. Pushing it to the wall creates the missing first step.

No additional goal diagram is needed: the green exit, high wall, and falling block provide direct physical evidence.

Initial state
terrain = {
  y=8:  x=10–14
  y=9:  x=10–14
  y=10: x=0–14
}

player = (2,9), facing RIGHT
blocks = {(6,7)}
exit = (11,7)

selection/cursor = none

The block at (6,7) is intentionally unsupported. Do not settle it on reset.

...............
...............
...............
...............
...............
...............
...............
......B....E...
..........#####
..>.......#####
###############
Reference solution — 9 actions
R×6, U×2, R

Replay checkpoints:

First R: player reaches (3,9); block A falls from (6,7) to (6,9).

Remaining five R: A is pushed to (9,9) and the player ends at (8,9).

First U: player climbs onto A at (9,8).

Second U: player climbs the wall to (10,7).

Final R: player enters the exit at (11,7).

Final block set:

{(9,9)}
Necessity and anti-degeneracy

The wall occupies both (10,9) and (10,8), so it is two cells higher than the floor and cannot be climbed directly. The only loose block is therefore necessary. Random horizontal movement cannot finish the level because two correctly timed climbs are required after exact block placement.

Level 2 — Outside First
Player-facing discovery

The floor block appears to belong directly beneath the upper ledge. Pushing it one cell too far, however, places it under a solid overhang. The avatar cannot stand on it because the landing cell is terrain, and the block cannot be pulled back.

The correct position is outside the overhang. From there it provides access to the upper block, which becomes the second step near the exit wall.

Initial state
terrain = {
  y=6:  x=12–14
  y=7:  x=12–14
  y=8:  x=8–14
  y=9:  x=11–14
  y=10: x=0–14
}

player = (2,9), facing RIGHT
blocks = {
  A=(5,9)
  B=(10,7)
}
exit = (13,5)

selection/cursor = none
...............
...............
...............
...............
...............
.............E.
............###
..........B.###
........#######
..>..B.....####
###############
Reference solution — 11 actions
R×4, U×2, R×2, U×2, R

Replay checkpoints:

R×4: A moves to (7,9); player ends at (6,9).

First U: climb A to (7,8).

Second U: climb the overhang to (8,7).

R: walk to (9,7).

R: push B from (10,7) to (11,7).

First U: climb B to (11,6).

Second U: climb the high wall to (12,5).

Final R: enter (13,5).

Final block set:

{(7,9), (11,7)}
Local impasse

If A is pushed from (7,9) to (8,9), UP targets (8,8), which is terrain. The blocked landing and A flash red together. A cannot be pulled left because the avatar cannot reach its right side through the solid structure.

Necessity and anti-degeneracy

A is required to reach the upper ledge. B is required because the upper ledge at y=7 remains two levels below the final wall top at y=5. Neither block is decorative, and the correct stopping coordinate of A is the central insight.

Level 3 — Borrowed Stair
Player-facing discovery

Three dependencies are visible:

A floor block near the right edge can catch the upper falling block.

A second floor block provides access to the construction bridge.

That access block is not allowed to remain there: after the upper work, it must be moved into the final staircase.

Pushing the upper block without first staging its catcher causes it to fall to floor height, one row too low to form the needed two-block stack.

Initial state
terrain = {
  y=6:  x=11–12
  y=7:  x=10, x=14
  y=8:  x=9,  x=14
  y=9:  x=14
  y=10: x=0–14
}

player = (11,9), facing RIGHT
blocks = {
  A=(7,9)    temporary access block
  B=(12,5)   upper falling block
  C=(12,9)   catcher
}
exit = (14,6)

selection/cursor = none
...............
...............
...............
...............
...............
............B..
...........##.E
..........#...#
.........#....#
.......B...>B.#
###############
Reference solution — 25 actions
R,
L×4,
U, D, R,
U×4,
R,
L, D×4,
R×4,
U×3

Replay checkpoints:

R: push C from (12,9) to (13,9).

L×4: return to (8,9).

U, D: climb over A and descend to its left at (6,9).

R: push A to its temporary position (8,9).

U×4: climb A and the irregular fixed steps to the bridge at (11,5).

R: push B from (12,5) into column 13. It falls to (13,8), resting on C.

L, D×4: descend the access staircase to (7,9).

R×4: push A from (8,9) to its final position (12,9).

U: climb A to (12,8).

U: climb the B–C stack to (13,7).

U: enter the exit at (14,6).

Final block set:

{(12,9), (13,8), (13,9)}
Dependency network

C and A are independent setup branches. The upper drop depends on both:

without C, B settles at (13,9);

without A at (8,9), the bridge is unreachable.

The final ascent then depends on moving A away from its temporary access position.

Necessity and anti-degeneracy

The bridge is at y=5, while the final stack’s walkable top is at y=7. The avatar cannot descend directly from the bridge onto the stack because DOWN moves only one row and the intermediate landing is unsupported. This explicitly prevents the upper construction route from doubling as the final route.

Level 4 — The Load-Bearing Block
Player-facing discovery

A two-block vertical stack sits on an elevated shelf. At the construction perch:

climbing is impossible because the top block occupies the landing cell;

the only productive action is pushing the bottom block.

The push visibly behaves differently from an unloaded push. The avatar braces instead of entering the vacated cell, the bottom block falls to the floor, and the top block settles one row. Pushing that top block next causes it to fall onto the first, producing the final two-block stack.

Initial state
terrain = {
  y=6:  x=8–9
  y=7:  x=7, x=11–14
  y=8:  x=6, x=11–14
  y=9:  x=11–14
  y=10: x=0–14
}

player = (7,9), facing RIGHT
blocks = {
  A=(3,9)   reusable access block
  H=(9,4)   top supported block
  G=(9,5)   load-bearing bottom block
}
exit = (12,6)

selection/cursor = none
...............
...............
...............
...............
.........B.....
.........B.....
........##..E..
.......#...####
......#....####
...B...>...####
###############
Reference solution — 26 actions
L×3,
U, D, R×2,
U×4,
R×2,
L, D×4,
R×4,
U×3, R

Replay checkpoints:

L×3, U, D: reach the left side of A.

R×2: move A to temporary position (5,9).

U×4: climb to the construction perch (8,5).

First R, a braced push:

G moves from (9,5) to column 10 and falls to (10,9);

H falls from (9,4) to (9,5);

player remains at (8,5).

Second R:

H moves to (10,5);

H falls to (10,8), resting on G;

player advances to (9,5).

L, D×4: return to floor coordinate (4,9).

R×4: move A to (9,9).

U: climb A to (9,8).

U: climb the H–G stack to (10,7).

U: climb the wall to (11,6).

R: enter (12,6).

Final block set:

{(9,9), (10,8), (10,9)}
Distinct tutorial evidence

The first R cannot be explained as an ordinary push:

under ordinary advance, H would fall into the avatar’s new cell (9,5);

the visible recoil and immediate settling demonstrate the braced-push rule;

no other action resolves the local impasse.

Necessity and anti-degeneracy

The construction perch at y=5 is two rows above the final stack’s walkable top at y=7, so the player must return and reuse A. G and H are both required for the two-block inner stair; A is required as the outer stair.

Level 5 — Finish the High Work
Player-facing discovery

Two blocks occupy the high bridge. The nearer one blocks access to the farther one. Pushing the nearer block toward the chute first merely jams both blocks together.

The player must:

climb over the nearer block;

push the far block into the chute;

climb back over the near block;

send the near block into the same chute;

only then descend and dismantle the access scaffold.

This is the first level where the highest work must be completed before rebuilding toward the exit.

Initial state
terrain = {
  y=5:  x=9–12
  y=6:  x=8,  x=14
  y=7:  x=7,  x=14
  y=8:  x=6, x=12, x=14
  y=9:  x=14
  y=10: x=0–14
}

player = (9,9), facing RIGHT
blocks = {
  A=(4,9)    temporary/final scaffold
  D=(10,9)   floor catcher
  E=(10,4)   nearer upper block
  F=(12,4)   farther upper block
}
exit = (14,5)

selection/cursor = none
...............
...............
...............
...............
..........B.B..
.........####.E
........#.....#
.......#......#
......#.....#.#
....B....>B...#
###############
Reference solution — 45 actions
R×3,
L×7,
U, D, R,
U×5,
U, D, R,
L, U, D,
R×3,
L×3, D×5,
R×6,
U×4

Replay checkpoints:

R×3: push D from (10,9) to catcher position (13,9).

L×7: return to (5,9).

U, D, R: cross A and move it to temporary position (5,9).

U×5: climb to bridge entry (9,4).

U: climb over E to (10,3).

D: descend to (11,4), between E and F.

R: push F into column 13; it settles at (13,8) on D.

L, U, D: return over E to (9,4).

R×3: push E into column 13; it settles at (13,7).

L×3, D×5: descend to (4,9).

R×6: push A to (11,9).

U: climb A to (11,8).

U: climb the fixed step to (12,7).

U: climb the three-block stack to (13,6).

U: enter the exit at (14,5).

Final block set:

{(11,9), (13,7), (13,8), (13,9)}
Wrong plan and consequence

Pushing E right before visiting F moves E into the only gap separating the two upper blocks. They become an unpushable pair. Moving A toward the exit before dropping both upper blocks removes the only route back to the bridge.

Necessity and anti-degeneracy

D, E, and F must form a three-block column. With only two blocks in the chute, its walkable top is one row too low. A is required both for bridge access and the final outer step. The upper bridge remains too high for a direct descent onto the completed stack.

Level 6 — The Delayed Riser
Player-facing discovery

The floor catcher begins on the wrong side of the avatar. It must first be climbed over so it can be pushed right rather than left.

A fifth block, G, waits beneath the construction bridge. It looks convenient, but releasing it before the three-block base exists makes it fall to floor height. G must remain untouched until:

the floor catcher is staged;

both upper blocks have fallen;

the access block has been moved into its final position;

the player has climbed to G’s pushing side.

Here a familiar block changes from apparent early shortcut to deliberately delayed final riser.

Initial state
terrain = {
  y=5:  x=9–12, x=14
  y=6:  x=8, x=14
  y=7:  x=7, x=11–12, x=14
  y=8:  x=6, x=10, x=14
  y=9:  x=14
  y=10: x=0–14
}

player = (6,9), facing RIGHT
blocks = {
  A=(2,9)
  D=(4,9)
  E=(10,4)
  F=(12,4)
  G=(12,6)
}
exit = (14,4)

selection/cursor = none
...............
...............
...............
...............
..........B.B.E
.........####.#
........#...B.#
.......#...##.#
......#...#...#
..B.B.>.......#
###############
Reference solution — 58 actions
L, U, D, R×9,
L×9,
U, D, R×3,
U×5,
U, D, R,
L, U, D,
R×3,
L×3, D×5,
R×4,
U×3, R, U×2

Replay checkpoints:

L, U, D: climb over D and land on its left at (3,9).

R×9: push D to (13,9).

L×9: return to (3,9).

U, D, R×3: cross A and move it to temporary position (5,9).

U×5: reach bridge entry (9,4).

U, D, R: climb over E and push F into the chute. F settles at (13,8).

L, U, D: return over E.

R×3: push E into the chute. E settles at (13,7).

L×3, D×5: descend to (4,9).

R×4: push A to final position (9,9).

U×3: climb A and the two fixed steps to (11,6).

R: push G from (12,6) to (13,6), where the existing three-block stack catches it.

U: climb G to (13,5).

U: enter (14,4).

Final block set:

{(9,9), (13,6), (13,7), (13,8), (13,9)}
Contrasting roles

D is initially an obstacle that must be crossed, then becomes the bottom catcher.

A is temporary access, then an outer final step.

G is harmful if used early and essential if delayed.

E and F form supporting structure rather than direct steps encountered by the avatar.

Necessity and anti-degeneracy

The upper stack contains exactly three blocks before G is moved. Without that base, G falls below (13,6) and cannot produce the required fourth level. The final wall begins at (14,5), so the four-block stack is necessary.

Level 7 — Five-High Handoff
Player-facing discovery

The capstone composes every prior consequence.

At floor level, the catcher again has to be crossed before it can be pushed to the chute. The construction bridge is higher, so the player must complete both upper drops and fully return to the floor.

Near the exit is a vertical pair:

G, the bottom block, is aligned with the existing three-block chute stack.

H rests directly above G.

Pushing G is a loaded push. The avatar braces; G falls onto the distant stack at exactly the fourth height, while H falls into G’s former cell. A second push sends H into the fifth and final height.

The same operational rule seen in Level 4 now performs a two-stage structural handoff rather than merely dismantling a stack.

Initial state
terrain = {
  y=3:  x=9–12
  y=4:  x=8, x=14
  y=5:  x=7, x=14
  y=6:  x=6, x=11–12, x=14
  y=7:  x=5, x=10, x=14
  y=8:  x=4, x=9, x=14
  y=9:  x=14
  y=10: x=0–14
}

player = (5,9), facing RIGHT
blocks = {
  A=(1,9)
  D=(3,9)
  E=(10,2)
  F=(12,2)
  H=(12,4)   top block of the middle pair
  G=(12,5)   load-bearing bottom block
}
exit = (14,3)

selection/cursor = none
...............
...............
..........B.B..
.........####.E
........#...B.#
.......#....B.#
......#....##.#
.....#....#...#
....#....#....#
.B.B.>........#
###############
Reference solution — 66 actions
L, U, D, R×10,
L×10,
U, D, R×2,
U×7,
U, D, R,
L, U, D,
R×3,
L×3, D×7,
R×5,
U×4,
R×2,
U×2

Replay checkpoints:

L, U, D: cross D and reach (2,9).

R×10: push D from (3,9) to (13,9).

L×10: return to (2,9).

U, D, R×2: cross A and move it to temporary position (3,9).

U×7: climb the tall access route to bridge entry (9,2).

U, D, R: cross E and push F into the chute. F settles at (13,8).

L, U, D: return over E.

R×3: push E into the chute. E settles at (13,7).

L×3, D×7: return to floor coordinate (2,9).

R×5: push A to final position (8,9).

U×4: climb A and the fixed steps to (11,5).

First R, a braced loaded push:

G moves from (12,5) into column 13;

G falls to (13,6), resting on the D–F–E stack;

H falls from (12,4) to (12,5);

player remains at (11,5).

Second R:

H moves from (12,5) to (13,5);

player advances to (12,5).

First U: climb H to (13,4).

Second U: enter the exit at (14,3).

Final block set:

{
  (8,9),
  (13,5),
  (13,6),
  (13,7),
  (13,8),
  (13,9)
}
Dependency network

The final loaded handoff works only if the earlier branches are complete:

D must be at (13,9).

F must be at (13,8).

E must be at (13,7).

Only then does G settle at `(13,6)).

Only then can H remain at (13,5).

A must be at (8,9) to reach the pushing side at (11,5).

If any one of D, E, or F is missing, G and H settle too low and the wall at x=14 remains unclimbable.

Push-side preservation reprise

A must stop at (8,9). Pushing it to (9,9) places it beneath fixed terrain at (9,8), recreating the Level 2 overhang trap at the beginning of the final ascent.

Anti-degeneracy

The upper bridge is at y=2, while the three-block chute stack initially offers a walkable top at y=6. There is no direct descent or accidental shortcut. All six blocks are present in the minimal final construction, and every revisit changes the structural role of at least one block.

11. Final design audit
Small rule set

The entire game uses:

one avatar;

fixed terrain;

identical loose blocks;

one exit;

horizontal pushing;

one-step climbing and descending;

vertical gravity;

the deterministic braced-push collision consequence.

No switches, keys, colors, counters, pattern legends, enemies, timers, or block subtypes are introduced.

Contextual reuse

The same block rule produces contrasting roles:

Level 1: a falling block becomes a permanent step.

Level 2: a block becomes useless if placed too far inward.

Level 3: a block catches another, while a separate step is borrowed and later reclaimed.

Level 4: a load-bearing block is pushed out so the structure can settle.

Level 5: blocks are upper work that must be completed before access is dismantled.

Level 6: a visible block must be deliberately delayed.

Level 7: a loaded push transfers two blocks into consecutive remote heights.

No checklist validator

The validator checks only:

player_position == exit_position

All structural requirements arise from physical reachability, not from hidden conjunctions over block locations.

No optional authored objects

Every loose block in every level is necessary for the minimal solution. Fixed terrain irregularities determine:

where a block can be pushed from;

which heights catch a falling block;

whether the avatar can descend;

which scaffolds must be preserved;

where a block becomes trapped under an overhang;

whether an upper task remains reachable.

Removing the spatial structure would change or destroy each puzzle rather than merely shorten its route.
