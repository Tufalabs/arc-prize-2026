Toggle Labyrinth

This is a game about toggling doors becoming difficult because every necessary crossing changes the return route.

Authored level count: exactly 7.

Toggle Labyrinth is a deterministic, fully visible retrieval game. The player enters a compact labyrinth, collects one artifact automatically, and carries it to a matching exit recess. Floor plates invert linked doors whenever the player enters them. Because plates can close doors as readily as they open them, movement through the labyrinth also edits the path that remains available afterward.

There are no hidden counters, one-shot switches, random events, timers, energy limits, or secret interactions. Every strategically relevant state is visible in the settled board.

1. Player actions

The environment exposes the required action space:

UP, DOWN, LEFT, RIGHT

SPACE

Click on any canvas cell

Only the four directional actions have gameplay effects.

SPACE and all click actions are intentional no-ops:

They do not move the player.

They do not trigger a plate.

They do not clear blocked-movement feedback.

They do not create or move a cursor.

No object is visually presented as clickable.

An external level-reset facility may be provided by the host environment, but reset is not an in-level action.

2. Objective and validator

The artifact and exit are visible from the first frame.

Entering the artifact cell automatically sets has_artifact = true. The artifact then disappears from the floor and becomes visibly attached to the player.

A level is won exactly when, after resolving a successful movement action:

player_position == exit_position
and has_artifact == true

Everything else is ignored by the validator:

Door states do not need to match a target configuration.

Plates do not need to be used a prescribed number of times.

The player may enter or cross the exit before collecting the artifact; this does nothing.

The artifact cannot be dropped.

Extra movement, blocked attempts, SPACE, and clicks are allowed.

There is no step limit or timer.

The blocked-facing indicator is irrelevant to success.

This gives the player unlimited room for experimentation, especially in the foundational levels.

3. Logical geometry

Each level has its own rectangular logical tile map. Coordinates use:

(0,0) at the upper-left.

x increasing rightward.

y increasing downward.

Each directional movement action attempts to move the player by exactly one logical tile.

All authored door cells have two opposite traversable neighbors, so every door has an unambiguous horizontal or vertical passage axis.

Documentation map notation

The authored maps below use:

#  wall
.  ordinary floor
s  initial player position
o  artifact
x  exit recess
p  plate P
q  plate Q
r  plate R
A–J  implementation door IDs

These characters are documentation only. No letters, coordinates, or other text are rendered in the game canvas.

4. Rendering contract
4.1 Canvas and tile projection

The render canvas is exactly 64×64 pixels.

Every logical tile is 4×4 rendered pixels. For a level of width W and height H:

origin_x = floor((64 - 4W) / 2)
origin_y = floor((64 - 4H) / 2)

Logical cell (x,y) occupies:

pixel x: origin_x + 4x through origin_x + 4x + 3
pixel y: origin_y + 4y through origin_y + 4y + 3

Pixels outside the logical map use color 5.

There are no visible internal grid lines. Contiguous floor and wall regions render as continuous shapes rather than bordered tiles.

4.2 Palette usage

Only integer color IDs 0–15 are used.

Role	Color ID
Background outside map	5
Walls	4
Wall-to-floor edge highlight	3
Floor	1
Open-door recessed jambs	2
Closed-door body	13
Closed-door latch stripe / blocked mark	8
Player	14
Artifact and exit outline	15
Artifact highlight	7
Plate P and its cables	11
Plate Q and its cables	6
Plate R and its cables	9
P activation flash	0
Q activation flash	7
R activation flash	10

Color is not the sole door-state cue. Closed and open doors have different silhouettes.

4.3 Main shapes

A . below means that the underlying floor remains visible.

Artifact:

.  15 15 .
15  7  7 15
15  7  7 15
.  15 15 .

Exit recess:

.  15 15 .
15  4  4 15
15  4  4 15
.  15 15 .

Player:

.  14 14 .
14 14 14 14
14 14 14 14
.  14 14 .

When carrying the artifact, two upper-right player pixels become color 15, creating a clearly visible attached purple load.

A plate of color c fills its cell as a recessed pad:

c c c c
c 3 3 c
c 3 3 c
c c c c

The permanent cables emerging from its edges make it unmistakably interactive.

4.4 Doors

A door occupies one logical passage cell.

For a left-right passage:

Closed: a solid vertical panel of colors 13 and 8 occupies the center of the cell.

Open: the panel is split and recessed into the upper and lower wall sides, leaving a continuous floor-colored opening.

For an up-down passage, the pattern is transposed.

A closed door blocks entry from every direction. An open door is ordinary traversable terrain.

Every door has one small colored cable socket per controlling plate. A door controlled by P and Q therefore has both yellow and magenta sockets.

4.5 Deterministic cable routing

The logical link sets in the level definitions are authoritative. Cable geometry is generated deterministically so an implementer does not need to hand-author additional routes.

Use a half-tile boundary lattice:

Lattice coordinates are (u,v) with 0 ≤ u ≤ 2W, 0 ≤ v ≤ 2H.

Horizontal edges are allowed only where v is even.

Vertical edges are allowed only where u is even.

This restricts cables to logical tile boundaries and edge midpoints.

A cell (x,y) has four cable ports:

up:    (2x+1, 2y)
right: (2x+2, 2y+1)
down:  (2x+1, 2y+2)
left:  (2x,   2y+1)

For every plate-door link, choose a minimum-cost route from any plate port to any socket port on the linked door. Route links in this order:

P links by ascending door ID,
then Q links,
then R links.

Base edge costs are:

Boundary between two wall cells: 1

Boundary between wall and non-wall: 2

Boundary between two non-wall cells: 6

Outer map boundary: 3

Add:

+2 for an edge already used by another cable from the same plate, allowing legible shared trunks.

+20 for an edge already used by a different plate color.

+6 for crossing a vertex occupied by a different plate color.

Resolve equal-cost alternatives by:

Source-port order UP, RIGHT, DOWN, LEFT

Target-port order UP, RIGHT, DOWN, LEFT

Path direction order UP, RIGHT, DOWN, LEFT

Map half-lattice position (u,v) to pixel position:

(origin_x + 2u, origin_y + 2v)

Draw a one-pixel cable along the resulting path. Same-colored shared segments merge into a visible bus. At an unavoidable crossing between different colors, the later-routed cable receives a three-pixel bridge, with the crossing cable remaining visible on either side.

No other tile boundaries are drawn.

5. State machine
5.1 Logical state
level_id
player_position: (x,y)
has_artifact: bool
artifact_present: bool
door_open: map[door_id -> bool]
blocked_facing: NONE | UP | DOWN | LEFT | RIGHT
won: bool

There is no selected object and no gameplay cursor.

All level state resets between levels.

5.2 Movement resolution

For UP, DOWN, LEFT, or RIGHT, resolve one step() in this exact order:

If won == true, ignore the action.

Compute the adjacent target cell.

Test the target using the pre-action door states.

If the target is outside the map, a wall, or a closed door:

Keep the player in place.

Set blocked_facing to the attempted direction.

Do not trigger the plate under the player.

Do not toggle doors.

Do not pick up the artifact.

Do not test victory.

Otherwise:

Move the player to the target.

Clear blocked_facing.

If the target contains a plate:

Invert every linked door atomically.

An open linked door closes.

A closed linked door opens.

If the target contains the artifact and artifact_present == true:

Set artifact_present = false.

Set has_artifact = true.

If the target is the exit and has_artifact == true:

Set won = true.

All doors controlled by a plate toggle simultaneously. There is no link priority.

5.3 Fresh entry

A plate triggers on every successful move into its cell.

It does not trigger when:

The player remains on it after a blocked movement.

The player uses SPACE.

The player clicks.

The player leaves it.

The player simply continues standing on it.

Leaving and later re-entering the plate triggers it again, regardless of direction or prior use count.

No authored level starts the player on a plate.

6. Animation and feedback

Logical state changes occur entirely inside the action’s step() call. Animation frames are deterministic visual output only and do not consume actions or modify state.

Blocked movement

A blocked directional action returns a two-frame shake:

Player displaced one pixel toward the obstacle.

Player restored to its logical cell.

A two-pixel color-8 notch remains on the attempted side of the player until the next successful movement. A later blocked attempt updates the notch direction.

Plate activation

Entering a plate returns four visual frames:

Player appears on the plate; the plate depresses and its cable trunks flash.

Every affected door begins sliding.

Doors continue sliding simultaneously.

Settled authoritative door forms are shown.

The logical door states are committed before frame 1. No input is accepted between these frames.

Cable flash colors are:

P: color 0

Q: color 7

R: color 10

Artifact pickup

Artifact pickup returns three visual frames:

Player enters the artifact cell.

The artifact lifts toward the player.

The carried-artifact attachment appears.

The artifact is logically carried from the original movement step onward.

Completion

The winning step returns six frames. The carried artifact moves into the matching recess, which fills with color 15 and receives a color-14 rim. won is already true from the movement step.

7. Authored levels
Level 1 — The Matching Recess

Purpose: Teach movement, automatic pickup, carrying, and the sole objective.

Board: 7×5

#######
#.....#
#x.so.#
#.....#
#######
Initial state
Player:   (3,2)
Artifact: (4,2)
Exit:     (1,2)
Doors:    none
Plates:   none
has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

The artifact is immediately beside the player. The exit has the same purple silhouette but is visibly hollow. Moving right causes the artifact to become attached to the avatar. Returning left causes the carried shape to settle into its matching recess.

A player who first moves left can enter the exit harmlessly and see that the still-visible artifact is missing from it. This is a cheap, unambiguous experiment rather than a punishment.

One shortest replay

R L3

Step 1: collect the artifact at (4,2).

Step 4: enter the exit at (1,2) and win.

Verified depth: 4 successful movement actions.

Attribution and necessity check

There are no doors, plates, decoys, or unrelated symbols. The only state change before victory is artifact pickup. The level cannot be won without collecting the artifact.

Level 2 — One Cable, One Door

Purpose: Introduce the causal effect of entering a visibly linked plate.

Board: 9×5

#########
#..p#...#
#x..A.o.#
#..s#...#
#########
Initial state
Player:   (3,3)
Artifact: (6,2)
Exit:     (1,2)

Door A: (4,2), closed

Plate P: (3,1)
P links: {A}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

The wall at x=4 divides the map. Door A is the only crossing and is visibly closed. From (3,2), moving right produces the blocked shake and persistent right-facing mark.

Plate P is directly above that local impasse, with one uninterrupted yellow cable terminating at A. Entering P immediately retracts A. No other object changes.

The solution never needs to enter P a second time, so this level does not prematurely require the player to infer repeat triggering.

Door trace
Initial: A closed
Enter P: A open
One shortest replay

U2 D R3 L5

Step 2: enter P; A opens.

Step 6: collect the artifact.

Step 11: reach the exit.

Verified depth: 11 successful movement actions.

Attribution and necessity check

The wall barrier has no second opening. With P disabled or A permanently closed, the artifact is unreachable. The immediate single-door movement isolates the plate-door relationship from all later complications.

Level 3 — The Door You Closed Behind You

Purpose: Establish the single new causal rule that plates are repeatable inverters, not one-shot openers.

Board: 11×7

###########
###########
###########
#xB.p.A.o.#
####.######
####s######
###########
Initial state
Player:   (4,5)
Artifact: (8,3)
Exit:     (1,3)

Door A: (6,3), closed
Door B: (2,3), open

Plate P: (4,3)
P links: {A, B}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

The player enters P from below. A opens, but the already-open B visibly closes at the same moment. This rules out the hypothesis that plates merely “open their doors.”

After collecting the artifact, B blocks the exit route. Returning left necessarily enters P again. The second entry closes A and reopens B, showing that the plate has no consumed or one-shot state.

The relevant local impasse is directly visible on the return trip: B is closed, P lies between the player and B, and entering it produces the needed physical reversal.

Door trace
Initial:      A closed, B open
First P:      A open,   B closed
Second P:     A closed, B open
One shortest replay

U2 R4 L7

Step 2: first P entry; A opens and B closes.

Step 6: collect the artifact.

Step 10: second P entry; A closes and B opens.

Step 13: win.

Verified depth: 13 successful movement actions.

Attribution and necessity check

A must open to reach the artifact. B must later open to reach the exit. Therefore both the initial inversion and repeat entry are necessary; a one-shot-opener interpretation cannot explain a successful playthrough.

Level 4 — Shared Cable

Purpose: Introduce overlapping link sets and selective change through cancellation.

Board: 13×11

#############
#############
#############
#######q#####
###...A.#####
###...##.o..#
###.#.B..##.#
###p#######C#
###.######..#
#s..######.x#
#############
Initial state
Player:   (1,9)
Artifact: (9,5)
Exit:     (11,9)

Door A: (6,4),  closed
Door B: (6,6),  closed
Door C: (11,7), open

Plate P: (3,7)
P links: {A, C}

Plate Q: (7,3)
Q links: {B, C}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

Entering P opens A but closes C. B remains closed, so the known single-plate rule does not yet produce a route to the artifact and exit.

A leads to a northern dead-end containing Q. Entering Q opens B and reopens C. The shared door C has now been toggled twice, while A and B have each been toggled once:

Initial: A closed, B closed, C open
After P: A open,   B closed, C closed
After Q: A open,   B open,   C open

The player must then backtrack through A to the earlier branch, use B to reach the artifact, and use C for the separate lower exit route. Q is deliberately not placed directly before B; the player has to understand what changed and return to the useful junction.

One shortest replay

R2 U5 R4 U D L2 D2 R3 U R3 D4

Plate and objective events:

Step 4: enter P.

Step 12: enter Q.

Step 22: collect the artifact.

Step 28: win.

Verified depth: 28 successful movement actions.

Anti-cleanliness and necessity check

This is not a linear P–Q–artifact corridor:

Q is a dead-end operation requiring backtracking.

B belongs to a different branch from Q.

C lies on the later exit branch and initially appears usable, but P closes it.

No single plate creates the required configuration.

Disabling either plate or fixing any door closed removes every winning path.

Level 5 — Do Not Retrace the Turnstile

Purpose: Turn a familiar plate into a return-path liability. Introduce a forced-crossing arrangement made only from established doors and plates.

Board: 13×13

#############
#############
#############
##########o.#
######.AqD..#
#####..####.#
#####.....B.#
###p..#######
###.#########
###.#########
###...##...##
#s..#..C.#.x#
#############
Initial state
Player:   (1,11)
Artifact: (10,3)
Exit:     (11,11)

Door A: (7,4),  closed
Door B: (10,6), open
Door C: (7,11), closed
Door D: (9,4),  closed

Plate P: (3,7)
P links: {A, B}

Plate Q: (8,4)
Q links: {A, B, C, D}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

Door A, plate Q, and door D form the horizontal sequence:

A – Q – D

Both flanking doors are linked to Q and begin in opposite states. This is not a new object type; it is a spatial consequence of the existing toggle rule.

The journey proceeds as follows:

Enter P:

A opens.

B closes.

Reach A–Q–D.

Enter Q from the A side:

A closes behind the player.

D opens ahead.

B reopens as the alternate return.

C opens as the eventual exit route.

Collect the artifact beyond D.

Return through B, avoiding Q.

Enter P again from the upper side:

A opens.

B closes.

C and D remain open because P does not control them.

Cross C to the exit.

Door trace
Initial: A closed, B open,   C closed, D closed
After P: A open,   B closed, C closed, D closed
After Q: A closed, B open,   C open,   D open
After P: A open,   B closed, C open,   D open
Why retracing Q is wrong

Returning from the artifact through Q toggles C closed. The flanking A/D geometry also prevents the player from simply touching Q an arbitrary extra time and remaining on the original side: odd Q parity places the player on the D side, while completing a return through Q produces even parity.

Any recovery after retracing Q must eventually enter Q again and still use B. B cannot be ignored.

One shortest replay

R2 U4 R2 U2 R U R4 U D R D2 L6 D L2 D3 R2 D R3 U R2 D R

Events:

Step 6: first P.

Step 14: Q.

Step 17: artifact.

Step 30: second P.

Step 44: win.

Verified depth: 44 successful movement actions.

Anti-cleanliness and necessity check

The outward path and return path are physically different loops. B begins open but is closed whenever the player first reaches the upper network, so it cannot be used as an early shortcut. Q later reopens it from the far side.

Every door is traversed in the verified route. With P or Q disabled, or with any door permanently closed, no successful route exists.

Level 6 — State Handoff

Purpose: Reuse the turnstile as a setup loop, then make a different plate close the prepared route behind the artifact and open the final route ahead.

Board: 15×15

###############
###############
###############
###############
###############
###############
######...AqE..#
#####..######.#
#####.......B.#
###p..#########
###.###########
###.#####..or.#
###...##..###D#
#s..#..C.####x#
###############
Initial state
Player:   (1,13)
Artifact: (11,11)
Exit:     (13,13)

Door A: (9,6),   closed
Door B: (12,8),  open
Door C: (7,13),  closed
Door D: (13,12), closed
Door E: (11,6),  closed

Plate P: (3,9)
P links: {A, B}

Plate Q: (10,6)
Q links: {A, B, C, E}

Plate R: (12,11)
R links: {C, D}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Player-facing discovery

The upper setup loop reuses the now-established turnstile construction:

A – Q – E

P opens A and closes B. Q closes A behind, opens E ahead, reopens B, and opens C. The player must then return through B and enter P again to emerge on the lower side while preserving C:

Initial: A closed, B open,   C closed, D closed, E closed
After P: A open,   B closed, C closed, D closed, E closed
After Q: A closed, B open,   C open,   D closed, E open
After P: A open,   B closed, C open,   D closed, E open

C now leads to the artifact. Immediately beyond the artifact is R, while D ahead remains visibly closed.

Entering R produces the final physical handoff:

C closes behind
D opens ahead

The player is already on the useful side of the newly closed C, so this closure is helpful rather than harmful.

One shortest replay

R2 U4 R2 U2 R U R7 D2 L8 D L2 D3 R2 D R3 U R U R4 D2

Events:

Step 6: first P.

Step 16: Q.

Step 32: second P.

Step 46: collect artifact.

Step 47: enter R; C closes and D opens.

Step 50: win.

Verified depth: 50 successful movement actions.

Anti-cleanliness and necessity check

The upper loop does not contain the artifact. Its purpose is to generate the C-open state while returning the player to C’s useful side. This is not reducible to visiting an isolated switch station: A, Q, E, and B determine which side of the loop the player occupies after the toggles.

R reverses the role of C:

Before R, C is the required entrance.

After R, closing C prevents useless retracing and D becomes the only completion route.

P, Q, and R are all mandatory. Fixing any of A–E closed removes every winning path.

Level 7 — The Entire Journey Is the Switch

Purpose: Compose three forced-crossing arrangements, two overlapping setup cancellations, asymmetric outbound and return routes, and a final route that exists only at odd R parity.

Board: 15×15

###############
###############
######.AqF....#
#####..#####..#
###...######..#
###G#......B..#
###p#.#######.#
###H#.#######.#
#s..I.#######.#
#.#.#########.#
#.#..C..oDrE..#
#J#############
#x#############
###############
###############
Initial state
Player:   (1,8)
Artifact: (8,10)
Exit:     (1,12)

Door A: (7,2),  open
Door B: (11,5), open
Door C: (5,10), closed
Door D: (9,10), open
Door E: (11,10), closed
Door F: (9,2),  closed
Door G: (3,5),  closed
Door H: (3,7),  open
Door I: (4,8),  closed
Door J: (1,11), closed

Plate P: (3,6)
P links: {B, C, G, H}

Plate Q: (8,2)
Q links: {A, B, C, F}

Plate R: (10,10)
R links: {C, D, E, I, J}

has_artifact: false
blocked_facing: NONE
selection/cursor: none / hidden
Spatial motifs

Three plates are placed between opposite-state linked doors:

Vertical P crossing:   G
                       P
                       H

Horizontal Q crossing: A – Q – F

Horizontal R crossing: D – R – E

The operational rule is unchanged. The layouts merely make entering a plate commit the player to the newly opened side.

Phase 1: open C on the wrong side

The starting lower hub has two visible impasses:

C blocks the artifact route.

J blocks the exit recess.

H is open below P, while G is closed above it. Entering P from below gives:

Initial open: A, B, D, H
After P open: A, C, D, G

Consequences:

H closes behind.

G opens ahead, forcing the player upward.

C opens, but the player is now on the wrong side of the labyrinth to use it.

B closes.

This immediately demonstrates that having the desired door open is insufficient if a crossing placed the player on the wrong side.

Phase 2: Q prepares the return and cancels C

The upper route leads through A–Q–F. Entering Q from the left gives:

After Q open: B, D, F, G

Consequences:

A closes behind.

F opens ahead.

B reopens as the alternate route back to the upper hub.

C closes again.

The player uses B, not Q, to return to the P side.

Phase 3: reopen C on the useful side

The player approaches P from above through G. Entering P again gives:

After second P open: C, D, F, H

Consequences:

G closes.

H opens.

The player is forced downward into the starting hub.

C reopens while the player is now on its useful side.

This is the same P operation as before, but its strategic meaning is reversed by approach direction and current overlapping parity.

Phase 4: artifact and R commitment

The player crosses C and collects the artifact at (8,10). D is open on the left of R, while E is closed on the right.

Entering R produces:

After R open: E, F, H, I, J

Consequences:

D closes behind.

E opens ahead, forcing the player rightward.

C closes, sealing the old artifact route.

I opens as a later upper-to-lower return route.

J opens as the final exit gate.

The player cannot remain on the artifact side with J open by repeatedly crossing R: returning through R would close J again. Odd R parity both opens J and places the player on the E side.

Phase 5: reverse Q, then use the route R prepared

E leads to the far-right upper network. The player approaches Q from the right through F.

Entering Q in reverse gives:

After second Q open: A, B, C, E, H, I, J

F closes behind and A opens ahead, placing the player back in the upper hub.

The direct P route is unavailable because G is closed. This makes I—the gate opened earlier by R—the necessary return to the starting hub. From there, J leads to the exit.

One shortest replay

R2 U4 R2 U R U R6 D3 L7 U L2 D6 R10 U8 L7 D L D5 L4 D4

Events:

Step 4: first P.

Step 13: first Q.

Step 32: second P.

Step 41: collect artifact.

Step 43: R.

Step 59: second Q.

Step 76: win.

Verified depth: 76 successful movement actions.

Anti-cleanliness and necessity check

The level is a dependency network rather than an ordered checklist:

P opens C before the player can use it.

Q deliberately closes C while opening the alternate return B.

The second P entry reopens C from the useful side.

R creates three geographically separate consequences: artifact exit E, return gate I, and final gate J.

Q must later be crossed in the opposite direction.

Re-entering R to seek another route closes J, so the R turnstile cannot replace I.

I is not encountered when R opens it; its meaning becomes relevant only after the reverse Q crossing.

J is visible beside the start from the first frame but cannot be opened by any early local action.

An exhaustive state audit finds no win if:

P, Q, or R is disabled.

Any one of doors A–J is held permanently closed.

8. Progression summary
Level	New discovery or composition	Plate-entry sequence	Shortest verified moves
1	Artifact pickup and matching exit	—	4
2	Entering a linked plate changes a door	P	11
3	Plates are repeatable inverters	P, P	13
4	Shared links cancel while unshared links remain changed	P, Q	28
5	Retracing a turnstile plate destroys the prepared return state	P, Q, P	44
6	One plate hands the route from an entrance door to an exit door	P, Q, P, R	50
7	Side, parity, overlap, and return geometry govern the whole journey	P, Q, P, R, Q	76

The same objects acquire contrasting roles without exceptions:

A plate first opens a route, later closes the route behind the player.

Re-entering a plate restores a useful state in Level 3 but destroys one in Level 5.

A shared door is a temporary casualty in Level 4 and a side-positioning resource in Level 7.

An initially open door may be useless until reached from the correct side.

Closing a door can trap the player, protect a prepared state, or commit the player to the only viable return.

The artifact never changes the rules; it makes the second half of those rules unavoidable.

9. Verification contract

The authored instances were checked as finite state spaces over:

(player_position, door_bitmask, has_artifact)

The listed replays are shortest successful directional-action sequences under the specified transition order.

For Levels 2–7:

Disabling any authored plate’s toggle effect eliminates all winning paths.

Holding any authored door permanently closed eliminates all winning paths.

Every door in the level is traversed by the documented shortest replay.

Blocked actions, SPACE, and clicks cannot improve a solution because they do not alter position, doors, or artifact state.

Thus the level mechanisms are not decorative, and the later puzzles cannot be solved by ignoring the mechanics introduced in their boards.
