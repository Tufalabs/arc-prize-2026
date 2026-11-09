Compass Stretch

This is a game about making directional leaps, becoming difficult because using one direction lengthens its future reach while shortening the reach of the opposite direction.

Exact authored level count: 8.

Compass Stretch is a movement puzzle in which position is only half the state. Every successful leap changes the compass that determines the next leap. The same arrow can therefore mean “move two cells” now, “move four cells” later, and “overshoot the only useful junction” after that.

The game uses no hidden information, random events, autonomous movement, click interactions, keys, switches, or level-specific rule exceptions. All eight levels use the same floor, seal, exit, avatar, and compass rules. Progression comes from increasingly consequential geometry.

1. Native game space and notation
Logical space

Every level uses a 17×17 logical lattice.

Coordinates are (x, y).

(0, 0) is the upper-left logical cell.

UP decreases y.

DOWN increases y.

LEFT decreases x.

RIGHT increases x.

A cell is either traversable floor or black void.

There are no diagonal connections.

The map has no rendered tile borders.

The authored maps below use notation only:

# = void

. = ordinary floor

P = initial player cell

1–7 = seals, numbered only for this specification

X = exit

Every non-# character is traversable floor. The actual game renders no letters, numbers, or other text. All seals are visually identical; their positions distinguish them.

Render mapping to 64×64

The render canvas is exactly 64×64 pixels.

Background: color 5.

Logical board origin: rendered pixel (1, 7).

Every logical cell occupies a solid 3×3 pixel block.

Logical cell (x, y) maps to:

pixel x = 1 + 3x through 3 + 3x

pixel y = 7 + 3y through 9 + 3y

The logical board therefore occupies pixels x=1..51, y=7..57.

Adjacent floor cells visually merge. There are no internal outlines or grid borders.

The compass display occupies pixels x=53..63, y=7..17.

The action-budget bar occupies pixels x=2..61, y=61..62.

2. Shared state machine
State variables
level_id: integer 1..8
floor: set of logical coordinates
player: logical coordinate
reach_n: integer 1..4
reach_s: integer 1..4
reach_e: integer 1..4
reach_w: integer 1..4
activated_seals: bit mask
exit_open: boolean
selection: null
step_count: nonnegative integer
step_cap: authored per level
attempt_locked: boolean
done: boolean

The compass invariants are always:

reach_n + reach_s = 5
reach_e + reach_w = 5

All levels begin with:

(N, S, E, W) = (2, 3, 2, 3)
activated_seals = 0
exit_open = false
selection = null
step_count = 0
attempt_locked = false
done = false

There is no cursor state and no selected object.

3. Action semantics

The available action space remains exactly:

UP, DOWN, LEFT, RIGHT, SPACE, click(x, y)

Only the four directional actions participate in the puzzle.

Directional actions

For direction d, let:

L be the current reach of d;

opp(d) be the opposite direction;

(dx, dy) be the unit vector of d.

The attempted path is:

player + k·(dx,dy), for k = 1..L

A directional action is valid only if every cell in that path, including the landing cell, belongs to floor.

Objects do not block movement:

seals are floor;

the closed exit is floor;

the open exit is floor;

activated seals remain floor.

The player visibly passes over intermediate cells but interacts only with the exact landing cell.

Successful directional action

A successful move resolves in this exact order:

Read L from the pre-action compass.

Validate the complete path using the pre-action position and compass.

Move the player to the final path cell.

If the landing cell contains an inactive seal, activate it permanently.

Attempt to transfer one pip from the opposite arm to the used arm:

if reach_opp > 1, increment reach_d and decrement reach_opp;

if reach_opp = 1, the pair is saturated and both lengths remain unchanged.

Set exit_open = true if every authored seal is active.

Set done = true if and only if:

the player is on the exit cell; and

exit_open = true.

Check the action cap only if the level did not complete.

Thus movement distance is always determined before the pip transfer caused by that move.

Invalid directional action

If any intended path cell is void or outside the 17×17 lattice:

player position is unchanged;

compass lengths are unchanged;

seal state is unchanged;

exit state is unchanged;

a red path preview is shown;

the action still increments step_count.

There is no death or automatic reset.

SPACE

SPACE restarts the current level from its exact authored initial state:

player, compass, seals, exit, and budget are reset;

step_count becomes zero;

the attempt is unlocked.

SPACE is never part of a reference solution.

Clicking

A click on any cell is a deliberate no-op.

There are no clickable objects or regions.

Clicking does not select anything.

A click increments step_count.

No click can contribute to level completion.

This keeps clicking out of the movement formal system rather than attaching arbitrary click behavior to floor objects.

Exhausted attempt

When step_count reaches the level cap without completion:

attempt_locked becomes true;

the budget bar turns maroon;

directional actions and clicks produce no further state change;

SPACE remains available to restart.

If the winning directional action is the final allowed action, the win takes priority over locking.

4. Seals, exit, and exact validation
Seals

Every seal begins inactive.

A seal activates only when the player lands on it.

Passing over it has no effect.

Activation is permanent for the current attempt.

Revisiting or passing over an active seal is allowed.

Seal order is not directly validated.

Exit

The closed exit is traversable.

Landing on it before all seals are active does not win.

Passing over it never wins.

It changes visibly from orange to green when the final seal activates.

The player must subsequently land on the green exit unless already there. Seal and exit coordinates never overlap, so every authored level requires an actual final exit landing.

Validator

A level is complete exactly when:

player == exit_position
AND activated_seals == full_seal_mask
AND the last action was a successful directional action

Ignored by validation:

final compass configuration;

total action count, provided it is within the cap;

order of seal activation;

number of invalid attempts;

visits to the closed exit;

revisits to active seals.

Forbidden:

winning by passing over the exit;

winning while standing elsewhere;

activating a seal by passing over it;

movement through a void cell;

hidden alternative click completion.

5. Visual language

Only integer color IDs 0–15 are used.

Entity	Color IDs	Rendering
Void	5	Black background
Floor	1	Solid connected 3×3 regions; no borders
Player	9, 10	Connected five-pixel plus shape; blue arms with light-blue center
Inactive seal	6	Magenta 3×3 outer ring, floor-colored center
Active seal	7, 11	Pink outer ring with yellow center
Closed exit	12, 4	Orange 3×3 frame with very-dark center
Open exit	14, 0	Green frame with white center
Compass pips	10	Light-blue one-pixel chains
Moving transfer pip	11	Yellow
Invalid preview	8	Red path centers and red landing outline
Spent budget	3	Dark gray
Available budget	10	Light blue
Low-budget warning	12	Orange
Exhausted budget	13	Maroon
Player and underlying objects

The player is drawn as:

.9.
9A9
.9.

where A is color 10. The corners of the underlying seal or exit remain visible when the player stands on it.

Compass panel

The compass panel has a very-dark-gray background, color 4.

A 3×3 miniature player sits at its center.

Four one-pixel pip chains extend north, south, east, and west.

The number of pixels in each chain is the corresponding current reach.

Maximum length four fits exactly within the 11×11 panel.

The two opposite chains visibly total five pips.

No digits or direction letters are drawn.

The used arm flashes white during a move.

A transferred pip moves in yellow from the opposite tip to the used tip.

Budget bar

For budget B and current count t, the number of available pixels is:

ceil(60 × max(0, B - t) / B)

Available pixels are light blue and spent pixels are dark gray. The remaining segment becomes orange below 10% of the cap.

6. Animation and causal feedback

No animation frame consumes a player step. Inputs are ignored until the current animation finishes.

Valid leap

A valid leap uses this deterministic visual sequence:

One frame shows a light-blue centerline through every path cell.

For k=1..L, one frame places the avatar on path cell k.

Intermediate seals and the exit do not react.

On the landing frame, the logical move, seal activation, and compass update are committed.

If a pip transfers, three frames show a yellow pip leaving the opposite arm and joining the used arm.

If the pair was saturated, both affected arm tips pulse yellow for two frames instead.

A newly activated seal fills from magenta to pink/yellow over three frames.

If that was the final seal, the exit changes from orange to green over four frames.

A winning landing produces an eight-frame green outward pulse.

The logical state is committed only once, on the landing frame. The remaining frames only explain the committed transition.

Invalid leap

An invalid action shows six frames:

red center dots over every in-bounds intended path cell;

a red 3×3 outline at the intended landing coordinate if it is in bounds;

the first missing or out-of-bounds segment flashes most strongly;

the corresponding compass arm flashes red.

The board then returns to the unchanged state.

7. Progression and exploration budget
Level	Main discovery forced by the instance	Verified shortest solution	Step cap
1	A move uses its old reach, then changes the return reach; seals require landing	8	160
2	Repeating a direction grows it until it overshoots	14	240
3	Departure and return are not inverses; position and compass state separate	23	420
4	A spatial circuit can return to one junction with a different compass	32	640
5	A useful junction can be crossed repeatedly without being landable	43	900
6	Both conserved directional pairs must be coordinated	48	1050
7	A reach needed for the final exit must survive a remaining branch	53	1350
8	The player must reconstruct a previously seen exit approach after a full tour	61	1600

The listed paths were exhaustively checked with breadth-first search over:

(player position, N/S state, E/W state, activated-seal mask)

They are shortest under the stated rules. Longer valid solutions are accepted. In every level, every traversable logical cell is crossed by the reference path; the maps contain no unused decorative branches.

In action strings below:

U = UP
D = DOWN
L = LEFT
R = RIGHT

Vertical bars divide conceptual phases and are not actions.

8. Authored levels
Level 1 — The Shortened Return
Player-facing discovery

The player begins at the end of a short vertical stem. Only DOWN is legal.

The first DOWN leap travels three cells. It visibly passes over the only magenta seal without activating it, then transfers one pip from north to south. North has therefore shortened from two to one.

The seal is now exactly one cell above the player. Pressing UP immediately lands on it, turns it pink, and makes the distant orange exit green. This is positive first-level evidence for both the objective and the landing-only rule.

The remainder reuses the same transfer rule horizontally. The lower offset path prevents the level from being solved merely by moving directly toward the green exit.

Initial authored state
Budget: 160
P = (8,3)
S1 = (8,5)
X = (4,7)
Compass = (N,S,E,W) = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
########P########
########.########
#####...1########
######.#.########
####X#.##########
###....##########
#################
#################
#################
#################
#################
#################
#################
#################
Reference replay
DU | LRDLRU

Full action sequence:

DULRDLRU

Key states:

Action 1, D: (8,3) → (8,6). S1 is passed over. Compass becomes (1,4,2,3).

Action 2, U: (8,6) → (8,5). S1 activates. Compass returns to (2,3,2,3). Exit opens.

Action 8, U: player lands on X=(4,7) and wins with compass (2,3,2,3).

Necessity and anti-degeneracy

At the initial state, DOWN is the only valid directional action.

The forced first move crosses S1 but cannot activate it, so the player must observe the distinction between passing and landing.

A mistaken rule such as “touching a seal anywhere along the path activates it” is visibly falsified.

The exit cannot be reached from the seal by a direct reverse action; the lower asymmetry forces horizontal transfer use.

All 14 floor cells participate in the eight-action shortest path.

Level 2 — One Cell Behind
Player-facing discovery

Only RIGHT is initially legal. The horizontal rail appears to lead directly to S1, but successive rightward moves have lengths two, three, and four.

The third RIGHT leap crosses S1 at (10,8) and lands one cell beyond it at (11,8). The west arm is now exactly one, so LEFT lands back on the seal. The player sees that repeating the same action changed the action itself.

The second seal is one cell north of the central junction. Ordinary UP initially jumps over it. The player must use the lower vertical rail—including a visit to the still-closed exit—to tune north to exactly three, land on S2, and then reconstruct the exit landing.

Initial authored state
Budget: 240
P = (2,8)
S1 = (10,8)
S2 = (10,7)
X = (10,12)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
#################
#################
#################
#################
##########2######
##P.......1.#####
##########.######
##########.######
##########.######
##########X######
##########.######
#################
#################
#################
Reference replay
RRRL | DUDUUU | DDDU

Full sequence:

RRRLDUDUUUDDDU

Checkpoints:

Actions 1–3: RIGHT leaps have lengths 2, 3, 4, ending at (11,8) after passing S1.

Action 4, L: west length one lands on S1. Compass becomes (2,3,3,2).

Action 8, U: lands on the closed exit at (10,12). No completion occurs.

Action 10, U: lands on S2 at (10,7). Compass becomes (4,1,3,2) and the exit opens.

Action 14, U: lands again on (10,12) and wins.

Necessity and anti-degeneracy

S1 is not reachable by repeatedly choosing RIGHT and hoping to stop there; the third leap necessarily overshoots.

S2 is not activated by crossing it during the later three-cell UP.

The closed exit is deliberately on the useful calibration path. It is a location with two different strategic roles: tuning surface first, objective later.

Removing the short cell beyond S1 eliminates the visible overshoot-and-return experiment.

The reference path uses every one of the 16 traversable cells and is shortest at 14 actions.

Level 3 — The Non-Inverse Tour
Player-facing discovery

S2 is visible one cell to the right of the player, but RIGHT is invalid: east reach is two, and the second cell is void. The nearby left spur is the only useful local experiment.

Moving left and returning does not restore the original movement capacity. The sequence around the spur gradually changes west/east reach until a three-cell RIGHT finally lands on S2.

From S2, LEFT moves only one cell, despite the final approach having moved three cells right. This is the first forced demonstration that the return action is not the inverse of departure.

The closed exit is then used as part of the upper route. The player descends the right side, crosses the bottom to S1, and climbs the left side before approaching the exit with a newly prepared compass.

Initial authored state
Budget: 420
P = (8,8)
S1 = (5,13)
S2 = (9,8)
S3 = (11,4)
X = (8,3)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
#####...X...#####
#####.##.##3#####
#####.##.##.#####
#####.##.##.#####
#####.##.##.#####
###.....P2#.#####
#####.#####.#####
#####.#####.#####
#####.#####.#####
#####.#####.#####
#####1......#####
#################
#################
#################
Reference replay
LRLRRR | LUURD | DDDLLL | UUUURR

Full sequence:

LRLRRRLUURDDDDLLLUUUURR

Checkpoints:

Action 6, R: S2 activates at (9,8), compass (2,3,4,1).

Action 7, L: the one-cell return lands at (8,8), not at the earlier left position.

Action 9, U: lands on the closed exit at (8,3).

Action 11, D: S3 activates at (11,4), compass (3,2,4,1).

Action 17, L: S1 activates at (5,13), compass (1,4,1,4).

Action 23, R: lands on the now-open exit with compass (4,1,3,2).

Dependency network

This is not a hard S1 → S2 → S3 key chain.

S2’s calibration spur prepares horizontal reach for the top crossing.

The upper-right descent prepares south and west reach for the bottom-left seal.

The bottom-left ascent prepares north reach for the final top approach.

Every permutation of the three seal orders is reachable under the rules, but their shortest completion lengths differ. The authored shortest order is S2 → S3 → S1; alternative orders begin at 26 actions and rise to 41.

The ordering therefore emerges from conserved reach and geometry rather than an explicit lock.

Necessity and anti-degeneracy

Going UP first reaches the objective region but leaves every seal inactive, establishing that proximity to the exit is not progress by itself.

The gap immediately right of S2 prevents a direct eastward landing from the initial compass state.

The left spur, upper route, right descent, bottom route, and left ascent are all traversed by the shortest solution.

All 42 floor cells are used; there are no unused outer-loop decorations.

Verified shortest depth: 23 actions.

Level 4 — Pocket Circuit
Player-facing discovery

The first two seals are easy to see but leave the compass badly skewed. The upper loop supplies S3, after which the player descends to the long right rail.

S2 sits inside a small lower-right pocket. Entering the pocket from (13,8) uses:

D, L

After activating S2, the route:

R, U, U

returns to the exact same junction (13,8).

The player has returned spatially, but north has changed from two to three. The local compass transfer animation makes the result unmistakable. The pocket is therefore simultaneously a seal route and a calibration circuit.

Initial authored state
Budget: 640
P = (7,8)
S1 = (7,11)
S2 = (11,11)
S3 = (8,3)
S4 = (4,8)
X = (7,13)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
#####...3..######
#####.####.######
#####.####.######
#####.####.######
#####.####.######
##..4..P.......##
#######.##.##.###
#######.#####.###
#######1###2..###
#######.#########
#######X#########
#################
#################
#################
Reference replay
D | UUL | RLRRUDUR | LRDDDURLDL | RUULLLRRDD

Full sequence:

DUULRLRRUDURLRDDDURLDLRUULLLRRDD

Checkpoints:

Action 1, D: S1 activates at (7,11).

Action 4, L: S4 activates at (4,8).

Action 12, R: S3 activates at (8,3).

Actions 20–25 form the pocket circuit:

begin at (13,8) with compass (2,3,3,2);

D,L activates S2 at (11,11);

R,U,U returns to (13,8);

resulting compass is (3,2,3,2).

Action 32, D: lands on X=(7,13).

Dependency network

S1 and S4 can be approached early from different branches.

S3 controls the state used to enter the right rail.

S2’s pocket both completes a seal branch and changes the north/south pair needed to escape toward the exit.

The final left-side calibration reuses S4’s branch after S4 is already active.

S4 is consequently a target early and merely useful terrain later; its operational rule never changes.

Necessity and anti-degeneracy

A mistaken “closed circuits restore the compass” model is directly falsified by the return to (13,8) with a different north/south split.

The right pocket is not a dead-end errand: its return state is required for the final route.

The exit stem cannot be entered correctly directly after S1.

All 38 traversable cells are used by the 32-action shortest solution.

Level 5 — The Junction You Cross
Player-facing discovery

The upper-left route contains three closely placed seals, but reach values make them behave very differently.

On the first ascent:

the player passes over S2 while moving from (4,6) to (4,3);

the next RIGHT passes over S3 and lands on S1.

Returning west passes over S3 again. The nearby seal remains visibly magenta despite being crossed from both directions. It cannot be collected by “trying harder” on the same rail.

The player must leave the region, use the lower spur and central shaft to change the compass, then revisit the top with west reach exactly three. Only then can LEFT land on S3.

Initial authored state
Budget: 900
P = (8,8)
S1 = (7,3)
S2 = (4,4)
S3 = (5,3)
S4 = (11,14)
S5 = (8,7)
X = (9,5)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
####.3.1.########
####2###.########
####.###.X...####
####.###.###.####
####.###5###.####
#.......P.......#
####.###.########
########.########
########.########
########.########
########.########
########...4#####
########.########
#################
Reference replay
LLRRUUR | LLD | DDURRLDDUR | LLUUDU | UL | RRDDDUDUURRLLUL

Full sequence:

LLRRUURLLDDDURRLDDURLLUUDUULRRDDDUDUURRLLUL

Checkpoints:

Action 6, U: passes over S2.

Action 7, R: passes over S3 and activates S1 at (7,3).

Action 9, L: passes over S3 a second time.

Action 10, D: lands on S2 at (4,4).

Action 20, R: activates S4 at (11,14).

Action 26, U: activates S5 at (8,7).

Action 28, L: finally lands on S3 at (5,3) with west reach three.

Action 43, L: lands on the exit at (9,5).

Dependency network

The upper-left cluster exposes S1, S2, and S3 together, but does not permit them to be completed together.

S1 and S2 prepare entry to the lower route.

S4’s lower spur changes east/west reach.

S5’s central shaft changes north/south reach.

Those two preparations converge to make the delayed S3 landing possible.

After S3, the same upper and central structures must be crossed in a different state to reach the exit.

S3 is therefore an obstacle-like liability during the first visit and a goal during the second, despite obeying one unchanged landing rule.

Necessity and anti-degeneracy

S3 is visible and crossed near the beginning but activates at action 28, preventing an object-order recipe.

The bottom spur is necessary compass preparation rather than a remote switch.

S5 cannot simply be collected on the initial central pass because the current vertical reach crosses it.

Every one of the 45 floor cells is used by the 43-action shortest solution.

Level 6 — Four-Petal Coupling
Player-facing discovery

The board resembles four offset petals around a central cross. Each petal changes one compass pair while placing the player where the other pair matters.

The early upper move activates S5 and leaves north/south at (4,1). Descending to S3 then leaves the compass at approximately the opposite spatial commitment: south and east are long while north and west are short.

At that point, neither UP nor LEFT behaves like a useful return. The player must combine motion on both axes rather than treating vertical and horizontal calibration as separate chores.

The four petal routes are reused in contrasting ways:

a petal first contains a seal;

later it is a return loop;

still later its short edge is used solely to alter the other directional pair.

Initial authored state
Budget: 1050
P = (8,8)
S1 = (14,6)
S2 = (8,14)
S3 = (11,13)
S4 = (2,10)
S5 = (8,3)
S6 = (3,5)
X = (11,5)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
#####...5########
########.########
###6####...X##.##
###.####.#####1##
###.####.#####.##
#.......P.......#
##.#####.########
##4#####.########
##.#####.########
########.########
########...3#####
########2########
########.########
#################
Reference replay
UU | LRRDDDDR | LLUUUDLRLU | DDRLRDU | URRRRLUD | DLLLRRDDU | UUUR

Full sequence:

UULRRDDDDRLLUUUDLRLUDDRLRDUURRRRLUDDLLLRRDDUUUUR

Checkpoints:

Action 2: S5 at (8,3), compass (4,1,2,3).

Action 10: S3 at (11,13), compass (1,4,4,1).

Action 20: S6 at (3,5), compass (4,1,1,4).

Action 27: S4 at (2,10), compass returns to the balanced (2,3,2,3).

Action 35: S1 at (14,6), compass (3,2,3,2).

Action 44: S2 at (8,14), compass (2,3,3,2).

Action 48: exit (11,5), compass (4,1,4,1).

Dependency network

The level has two intertwined cycles rather than one sequence:

S5 ↔ S3 establishes the upper-to-lower vertical cycle.

S6 ↔ S4 establishes the left cycle.

S1 ↔ S2 establishes the right-to-bottom cycle.

The central cross is revisited under several compass configurations.

Preparing the state for one cycle generally damages the immediately opposite return, so the player must use another petal to recover.

No petal is encountered only when needed. The upper-left and central branches, for example, are crossed both before and after their seals activate.

Necessity and anti-degeneracy

After S3, both north and west are length one; a one-axis calibration plan cannot recover the route.

S1 is visually accessible from the central rail much earlier than action 35, but the wrong vertical state makes its return harmful.

S2 is visible on the central vertical and is deliberately delayed until action 44.

All 45 traversable cells are used.

Verified shortest depth: 48 actions.

Level 7 — Keep Three North
Player-facing discovery

The closed exit is immediately above the player’s starting region, but north reach is two. The first UP leaps past its row.

The player eventually reaches and lands on the closed exit at action 34. At that moment, every seal except S6 has been activated. S6 lies in the far-right lower pocket.

The obvious temptation is to treat the exit visit as nearly finished and rush through the remaining branch. But the final exit approach must come from (11,10) with north reach exactly three. Repeated UP actions would leave north at four; excessive DOWN would reduce it to one or two.

The final branch must therefore be solved while deliberately preserving—or reconstructing—the exact final northward reach.

Initial authored state
Budget: 1350
P = (11,8)
S1 = (10,7)
S2 = (2,13)
S3 = (5,8)
S4 = (2,8)
S5 = (14,8)
S6 = (14,10)
X = (11,7)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#################
#################
#################
#################
##..........#####
##.##.....1X.####
##4##3.##..P..5##
##........#.#####
##..#..##.....6##
##..##.##########
##..##.##########
##2.##.##########
###.##.##########
###.##.##########
#################
Reference replay
ULRDRR | LLULLRD | ULD | DUD | URDUUURDURLR |
LRLLLDDDUDUUURRDRLR | LLU

Full sequence:

ULRDRRLLULLRDULDDUDURDUUURDURLRLRLLLDDDUDUUURRDRLRLLU

Checkpoints:

Action 6: S5 activates at (14,8).

Action 13: S3 activates at (5,8).

Action 16: S4 activates at (2,8).

Action 19: S2 activates at (2,13).

Action 31: S1 activates at (10,7).

Action 33 crosses both S1 and the closed exit horizontally without interacting.

Action 34 lands on the closed exit at (11,7) with compass (4,1,3,2).

Action 50: S6 activates at (14,10), compass (3,2,4,1).

Actions 51–52 tune west/east while returning to (11,10).

Action 53 uses north reach three to land on the exit. Final compass becomes (4,1,2,3).

Dependency network

The left cluster contains S2, S3, and S4 but uses both upper and lower approaches.

S1 and the exit share the top connector but have different landing requirements.

S5 is reached early from the central-right route.

S6 is nearby in the rendered map but postponed because entering its pocket at the wrong compass state destroys the final exit approach.

The player must leave a closed exit, solve S6, and return through a different compass state.

The exit is thus a visible future obligation, a traversable calibration cell, and the final destination, all under one unchanged rule.

Necessity and anti-degeneracy

Ending at the exit is not a final checklist item placed after the seals; the exit is beside the start and is physically crossed and landed upon much earlier.

Greedily maximizing north during the last branch makes the final UP overshoot.

Returning from S6 by the spatially shortest route does not necessarily produce north reach three.

Every one of the 60 floor cells appears in the 53-action reference replay.

Level 8 — Reconstruct the Approach
Player-facing discovery

The exit lies at the top of a narrow shaft. Before activating any seal, the player reaches (7,2) with south reach one and lands on the closed exit at action 9.

That early visit is effectively a physical demonstration of the final required approach:

stand one cell above X with south reach one

The rest of the level then pulls the compass through both saturated extremes:

the paired central seals are crossed before being landed upon;

the lower-left shaft drives both south and west long;

the upper-left seal requires reversing vertical saturation;

the right pocket repeatedly crosses S4;

S7 is reached only after another overshoot-and-backstep;

the route home must restore the exact one-cell downward landing seen near the beginning.

The final insight is not merely “save a direction.” It is “reconstruct a previously observed joint spatial and compass state after a full dependency network.”

Initial authored state
Budget: 1600
P = (4,9)
S1 = (3,14)
S2 = (9,8)
S3 = (11,8)
S4 = (14,10)
S5 = (4,5)
S6 = (8,10)
S7 = (15,11)
X = (7,3)
Compass = (2,3,2,3)
Activated mask = 0
Exit open = false
Selection/cursor = null
#################
#################
#######.#########
#######X#########
####.##.#########
####5##.#########
####.##.#######.#
####............#
####.##..2.3....#
#...P..#.######.#
###..#..6.....4.#
###..##########7#
###.###########.#
###.#############
###1#############
#################
#################
Reference replay
LRRURUDUDDDR | RLLL | ULRRD | LULDUD |
UURUUD | DRRLDRRL | LLRRRUDU | UUDLLLRLUDUD

Full sequence:

LRRURUDUDDDRRLLLULRRDLULDUDUURUUDDRRLDRRLLLRRRUDUUUDLLLRLUDUD

Checkpoints:

Action 9, D: lands on the closed exit at (7,3) before any seal is active.

Action 12, R: crosses S2 and activates S3 at (11,8).

Action 16, L: crosses S3 and activates S2 at (9,8).

Action 21, D: activates S6 at (8,10).

Action 27, D: activates S1 at (3,14), compass (1,4,1,4).

Action 32, U: passes over S5.

Action 33, D: backsteps onto S5 at (4,5).

Action 40, R: crosses S4.

Action 41, L: activates S4 at (14,10).

Action 46 crosses S4 again while preparing S7.

Action 48 passes over S7.

Action 49, U: backsteps onto S7 at (15,11). The exit opens.

Action 60 reaches (7,2) with compass (4,1,1,4).

Action 61, D: south reach one reproduces the early demonstrated landing on X=(7,3).

Dependency network

The authored structure contains several interacting cycles:

The upper exit shaft is visited both before and after the complete seal tour.

S2 and S3 form an overshoot pair: each is crossed during the approach to the other.

S6 connects the central rail to the lower-left cycle.

S1 and S5 require opposite vertical extremes.

S4 and S7 share the right pocket but require different backsteps.

Returning from the right pocket sets the horizontal pair needed for the central crossing, while the left and upper routes rebuild south reach one.

No single seal directly unlocks the next. Several branches can be entered in alternate orders, but their resulting compass states change the cost and reachability of the remaining branches.

Necessity and anti-degeneracy

The exit is encountered at action 9, before any seal, so the level cannot read as “collect seven objects and then walk to the end.”

S2, S3, S4, S5, and S7 are each visibly passed over before their activation.

The final route is not a reversal of the opening route; it returns to the same shaft with a different horizontal state and must reproduce only the required south reach.

The final validator has no hidden compass predicate. The compass matters solely because it determines whether the last physical landing is possible.

All 58 floor cells are traversed by the verified 61-action shortest solution.

9. Why the system has depth without extra devices

Compass Stretch obtains its later difficulty from four facts that never change:

A move has an exact, visible length.

A successful move changes the future length of that direction and its opposite.

Intermediate cells are crossed but do not interact.

Completion requires exact seal and exit landings.

Those rules produce several contrasting uses for the same geometry:

a long arm is a shortcut on one rail and an overshoot liability on another;

a closed exit is useless as an objective but valuable as calibration terrain;

a seal branch may be worth entering before its seal is landable;

a loop may collect a seal, tune one pair, damage another pair, or preserve location while changing capability;

returning to a familiar junction can represent a completely new state;

an activated seal remains strategically relevant because its surrounding floor can still tune the compass;

a direction can be strengthened for an immediate target, weakened to catch an intermediate junction, or protected for the eventual exit.

No operational rule changes between levels or between objects. The progression comes entirely from authored placement, approach direction, conserved reach, delayed revisits, and the consequences of committing to a route.
