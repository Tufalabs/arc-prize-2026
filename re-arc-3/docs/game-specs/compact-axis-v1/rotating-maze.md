Rotating Maze

Exact authored level count: 7

This is a game about rotating arrows becoming difficult because the same routes control both the agent and continuously moving cargo.

Rotating Maze is a compact routing game with six object categories: floor, walls, an agent, rotatable arrows, cargo balls, cargo bays, and an exit. Every arrow obeys one rule everywhere. The depth comes from when and from which side the player can reach each arrow, whether a ball is already committed to motion, and whether a route useful for cargo is also usable—or dangerous—for the agent.

The game uses UP, DOWN, LEFT, RIGHT, and SPACE. There are no clickable objects. Click actions are accepted because they belong to the supplied action space, but clicking any cell is a no-op and never selects anything.

1. Player-facing rules
Objective

Every free ball must be captured by an empty circular bay. When the last ball is captured:

Every filled bay remains visibly occupied and illuminated.

The closed exit loses its yellow ball-shaped lock and becomes a green open doorway.

The agent must then enter the open exit.

Levels 1 and 2 contain no balls or bays, so their exits begin open.

Ball identity is irrelevant. Any ball may enter any empty bay.

Movement

UP, DOWN, LEFT, and RIGHT:

set the agent’s visible facing direction;

attempt to move the agent one logical cell;

update facing even when movement is blocked.

Ordinary movement ends after one cell.

Rotating arrows

SPACE rotates the arrow directly in front of the agent clockwise:

UP → RIGHT → DOWN → LEFT → UP

The agent does not need to stand on a special control tile. Physical adjacency and facing are the complete manipulation rule.

Every arrow has:

a light-blue rotatable rim;

a dark-blue arrowhead;

one of four visible orientations.

If there is no arrow directly ahead, SPACE changes nothing. The agent’s facing tip and the faced cell edge briefly pulse red.

Agent deflection

When the agent enters an arrow, it is immediately ejected one cell in the arrow’s direction.

If the ejection enters another arrow, the chain continues. Reaching ordinary floor or an open exit ends the chain.

The agent never occupies an arrow in a settled frame.

If an arrow chain:

points into a wall, ball, bay, closed exit, or outside the board; or

repeats an arrow and would cycle,

the entire attempted entry is rejected. The agent stays on the ordinary cell from which it tried to enter. The obstructed arrow or cycle flashes red.

An arrow that sends the agent back to its source ordinary cell is valid. The animation visibly travels out and back, while the agent’s facing remains the direction of the original input.

Cargo launching

Moving into an adjacent ball attempts to push and launch it.

A launched ball:

begins moving in the push direction;

continues across ordinary floor without waiting for another action;

changes direction whenever it crosses an arrow;

continues across floor after the turn;

is captured immediately upon entering an empty bay;

stops on the final ordinary floor cell before an obstruction.

Obstructions for a moving ball are:

walls and board edges;

another ball;

a filled bay;

either state of the exit;

the agent’s new position after the push.

If the ball cannot vacate its starting cell, the push fails and neither agent nor ball moves.

A successful push first reserves the ball’s old cell for the agent. The ball then completes its entire route, and the agent settles in the ball’s former cell.

Arrows are transient route cells

Neither the agent nor a ball can remain on an arrow in a settled frame.

When a ball is about to enter an arrow, the directed contiguous arrow run is checked. If that run is blocked or cyclic, the ball stops on the ordinary cell immediately before the first arrow.

This makes failed routing legible: the ball remains visibly beside the arrow whose output is unusable, instead of resting on top of and obscuring the arrow.

Ball-route cycle handling

A launch tracks every (arrow coordinate, incoming direction) encountered during that launch.

If the same pair would be entered again, the ball stops before that repeated arrow entry. Every arrow participating in the detected cycle flashes red.

There is therefore no infinite movement, hidden momentum, or timing-dependent interception.

Bays

An empty bay:

is impassable to the agent;

captures the first ball that enters it.

A filled bay:

remains permanently filled;

is impassable to both agent and balls;

cannot be emptied or pushed.

Exit

The exit is closed exactly when either:

at least one authored ball remains free; or

at least one bay remains empty.

A closed exit is solid.

An open exit is passable to the agent and solid to balls.

Victory occurs only when the agent enters the open exit during a directional action or arrow-chain resolution. Final arrow orientations, agent facing, and action count are irrelevant.

2. Exact action contract
Direction action resolution

For a direction d:

Set agent_facing = d.

Let q = agent_position + d.

If q contains a ball, attempt a launch.

Otherwise, if q is solid for the agent, keep the same position and render a bump.

Otherwise, if q is an arrow, resolve the contiguous agent arrow chain.

Otherwise, move the agent to q.

If the settled agent position is the open exit, set won = true.

SPACE resolution

Let q = agent_position + agent_facing.

If q contains an arrow, replace its orientation with its clockwise successor.

Otherwise, leave all logical state unchanged and emit invalid-action feedback.

SPACE never changes facing.

Click resolution

CLICK(x, y) on any canvas cell:

never selects an object;

never rotates an arrow;

never moves the agent or a ball;

never changes facing;

emits a universal one-frame gray ripple centered on the clicked location.

There is no persistent cursor or selected-object variable.

State variables

An implementation needs only:

level_id
board_width
board_height
walls: set[cell]
arrows: map[cell -> orientation]
agent_position: cell
agent_facing: orientation
balls: set[cell]
bays: set[cell]
filled_bays: set[cell]
exit_position: cell
exit_open: bool
won: bool
last_feedback: render-only transition feedback

There is no velocity, momentum, phase clock, selected arrow, hidden switch state, or autonomous update.

Step timing

All logical changes happen inside step(action).

A completed step() computes the final settled state before another action is accepted. Animation frames are a deterministic visualization of that one transition; they do not call an autonomous update function and do not consume additional actions.

3. Ball trace algorithm

For a ball at start launched in direction d:

Temporarily remove the launched ball from the ball set.

Treat start as occupied by the pushing agent for the rest of the trace.

Set current = start.

Inspect next = current + d.

Then repeat:

If next is solid for balls, stop at current.

If next is an empty bay, fill that bay and end the trace.

If next is ordinary floor, set current = next and continue with the same direction.

If next is an arrow:

Record the arrow entry and incoming direction.

Resolve contiguous arrows until a non-arrow cell is reached.

If the run is blocked or repeats an arrow-entry state, do not enter the first arrow; stop at current.

If the run exits into an empty bay, capture the ball.

If it exits onto ordinary floor, set current to that floor cell, set d to the final arrow’s direction, and continue.

If the final result leaves the ball at start, the push is invalid and the agent remains in its original cell. Otherwise, the agent occupies start.

After capture:

exit_open =
    balls is empty
    AND filled_bays == bays
4. Rendering contract
Canvas and logical lattice

Render canvas: 64×64

Color values: integer IDs 0–15 only

Logical cell size: 5×5 pixels

Maximum authored board: 12×10 logical cells

Logical coordinates are zero-indexed:

x increases left to right;

y increases top to bottom.

For a board of size W×H:

origin_x = floor((64 - 5W) / 2)
origin_y = 8 + floor((50 - 5H) / 2)

The board is centered within a 50-pixel-high gameplay region from canvas rows 8–57.

There are no grid lines or internal cell borders. Adjacent wall cells merge into one continuous dark region, and adjacent floor cells merge into one light region.

Colors
ID	Use
0	canvas background and optional highlight
1	floor
2	arrow plate interior and click feedback
3	optional transition shadow only
4	walls and closed-exit interior
5	open doorway interior
6	agent body
7	agent facing tip
8	blocked or invalid feedback
9	arrowhead
10	rotatable arrow rim
11	cargo ball and exit’s ball lock
12	empty bay rim
13	closed exit frame
14	filled-bay glow, open exit, completion feedback
15	unused
Exact local silhouettes

Local pixel coordinates run from (0,0) to (4,4).

Agent

For an upward-facing agent:

color 6 at
{(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,3)}

color 7 at (2,0)

all remaining pixels retain floor color 1.

Rotate this mask for the other facings.

Ball

Color 11 wherever:

abs(px - 2) + abs(py - 2) <= 2

All other cell pixels retain floor color 1.

Empty bay

Color 12 on the outer ring where:

max(abs(px - 2), abs(py - 2)) == 2

except the four corner pixels, which remain floor color 1.

The center remains visibly empty.

Filled bay

Use the same ring in color 14. Fill the central 3×3 area with color 11.

Rotatable arrow

outer perimeter: color 10;

inner 3×3 plate: color 2;

upward arrowhead and shaft: color 9 at
{(2,0),(1,1),(2,1),(3,1),(2,2),(2,3)}.

Rotate the color-9 mask for other orientations. The color-10 rim remains fixed during rotation.

Closed exit

perimeter: color 13;

central 3×3: color 4;

center pixel (2,2): color 11.

Open exit

perimeter: color 14;

central 3×3: color 5;

no yellow lock.

Animation

Ordinary one-cell movement renders immediately.

Use animation only for the following:

Arrow rotation: two intermediate frames, then the new orientation. Logical orientation is committed by the same step() call.

Agent arrow travel: two frames per arrow crossed: centered on plate, then centered on the next cell.

Ball launch: two frames per logical cell crossed. At an arrow, show the ball centered briefly on the plate before continuing in the new direction.

Blocked move: two-frame compression toward the obstruction, then return.

Blocked arrow run: the entrant remains before the arrow; involved rims flash color 8 for one frame.

Capture: two frames of the ball shrinking into the bay, followed by the filled-bay settled frame.

Exit opening: after the final capture, a three-frame upward gate slide. The next action is accepted only after the open settled frame.

Completion: one-frame color-14 perimeter pulse.

Animation frames never consume action budget.

5. Authored-level notation

The following diagrams are documentation only. The rendered game contains no text.

#  wall
.  ordinary floor
A  agent
o  ball
^ > v <  initial arrow orientation
B  empty bay
X  exit location

X does not indicate whether the exit begins open or closed; that is stated for each level.

The host may use the generous technical truncation caps below. They are not an in-world timer and are not rendered. A host-level episode restart is outside the player action space.

Level	Main discovery	Canonical actions	Host cap
1	Agent follows arrows	3	64
2	SPACE rotates the front arrow	5	96
3	Ball launch, capture, and exit opening	6	160
4	Balls turn on arrows; prepare downstream	33	256
5	Wall staging and contiguous arrow chains	58	512
6	Shared junctions and riding cargo routes	59	640
7	Cargo blocks an arrow handle; final far-side reroute	59	800

R×3 means three separate RIGHT actions. SPACE×2 means two separate SPACE actions.

6. Exact authored levels
Level 1 — Follow the Arrow
Player-facing discovery

The green exit is visible to the right of a blue arrow. The agent cannot walk directly to the exit because walls surround the tiny route.

The only useful path leads upward into the arrow. Entering it immediately carries the agent right into the doorway.

This introduces exactly one causal unknown: an arrow redirects the agent according to its visible head.

State-machine instance

Board: 5×6

Agent: (1,4), facing RIGHT

Arrow: (2,2) = RIGHT

Exit: (3,2), initially open

Balls: none

Bays: none

Selection/cursor: none

Host cap: 64

#####
#####
##>X#
##.##
#A.##
#####
Canonical solution
RIGHT, UP, UP

Resolution:

RIGHT: agent moves to (2,4).

UP: agent moves to (2,3).

UP: agent enters (2,2), is ejected right, and enters the open exit at (3,2).

Canonical depth: 3 actions.

Attribution and necessity

The arrow already points at the visible exit, so no symbolic inference is required.

The final UP produces exactly one novel effect.

Every other neighboring cell is a wall.

There are no decoys, controls, balls, or simultaneous state changes.

The level cannot be solved by walking around the arrow.

Level 2 — Turn Before Entering
Player-facing discovery

The route shape is nearly identical to Level 1, but the arrow now points downward. Entering it sends the agent back to the approach floor instead of toward the exit.

Once the agent stands below the arrow, SPACE visibly rotates the rimmed plate clockwise. Three rotations point it right, restoring the route learned in Level 1.

This introduces exactly one causal unknown: SPACE rotates the adjacent arrow in front of the agent.

State-machine instance

Board: 5×6

Agent: (2,4), facing UP

Arrow: (2,2) = DOWN

Exit: (3,2), initially open

Balls: none

Bays: none

Selection/cursor: none

Host cap: 96

#####
#####
##vX#
##.##
##A##
#####
Canonical solution
UP, SPACE×3, UP

Resolution:

UP: agent reaches (2,3), directly below the arrow.

Three SPACE actions rotate DOWN → LEFT → UP → RIGHT.

UP: the agent enters the right-facing arrow and is ejected into the exit.

Canonical depth: 5 actions.

Distinguishing experiment

A blind player may instead try a second UP before rotating. The agent visibly enters the down arrow and returns to (2,3). This rules out the mistaken hypotheses that arrows are decoration, ordinary floor, or one-way doors.

Trying SPACE then causes an immediate visible quarter-turn without changing any other object.

Attribution and necessity

SPACE from the starting cell targets ordinary floor and has no effect; the player must first approach.

The exit cannot be reached while the arrow points down, left, or up.

No other component changes when the arrow rotates.

The rim distinguishes the arrow as the only plausible manipulable object.

Level 3 — One Push, One Delivery
Player-facing discovery

A closed maroon exit with a yellow ball lock is visible to the left. A yellow ball is directly in front of the agent, aligned with an orange circular bay.

Pushing right launches the ball across two intermediate floor cells in one action. The ball settles into the bay, the bay illuminates, and the exit opens. The agent must then turn the known arrow left to return through the doorway.

The single push reveals the complete cargo causal package:

a pushed ball continues without additional inputs;

a bay captures it;

filling the last bay opens the exit.

State-machine instance

Board: 9×5

Agent: (3,2), facing RIGHT

Ball: (4,2)

Bay: (7,2)

Arrow: (2,2) = UP

Exit: (1,2), initially closed

Selection/cursor: none

Host cap: 160

#########
#########
#X^Ao..B#
#########
#########
Canonical solution
RIGHT,
LEFT,
SPACE×3,
LEFT

Resolution:

RIGHT:

ball leaves (4,2);

crosses (5,2) and (6,2);

enters bay (7,2);

bay fills and exit opens;

agent settles at (4,2).

LEFT: agent moves to (3,2), now facing the arrow.

SPACE×3: arrow rotates UP → RIGHT → DOWN → LEFT.

LEFT: agent enters the arrow and is ejected into exit (1,2).

Canonical depth: 6 actions.

Attribution and necessity

The ball’s multi-cell animation directly falsifies the hypothesis that pushing moves it only one cell.

Bay filling occurs before the exit animation, making the causal order visible.

The exit is solid before capture.

The arrow is necessary for the return path and reuses the first two levels’ rules.

There are no unrelated objects or alternative routes.

Level 4 — Downstream First
Player-facing discovery

The bay lies around two corners. A straight launch cannot reach it.

The upper arrow initially points right into a wall. A premature launch moves the ball to (4,2), where it visibly stops immediately before that unusable arrow. The ball remains recoverable.

The route to the bay requires:

upper arrow (5,2) pointing DOWN;

lower arrow (5,6) pointing RIGHT.

The lower arrow is farther downstream. Because the ball cannot pause between the arrows, both must be prepared before the successful launch.

State-machine instance

Board: 11×9

Agent: (1,2), facing RIGHT

Ball: (2,2)

Upper arrow J₁: (5,2) = RIGHT

Lower arrow J₂: (5,6) = UP

Bay: (8,6)

Exit: (1,7), initially closed

Selection/cursor: none

Host cap: 256

###########
#.....#####
#Ao..>#####
#.###.#####
#.###.#####
#.###.#####
#....^..B##
#X#########
###########
Canonical solution
Prepare the downstream arrow
DOWN×4, RIGHT×3, SPACE

The agent reaches (4,6) and rotates J₂ from UP to RIGHT.

Prepare the upper turn
LEFT×3, UP×5, RIGHT×2, DOWN, RIGHT, SPACE

The agent reaches (4,2) and rotates J₁ from RIGHT to DOWN.

Return and launch
LEFT, UP, LEFT×2, DOWN, RIGHT

The final RIGHT launches the ball:

(2,2)
→ (3,2)
→ (4,2)
→ J₁ at (5,2), turn DOWN
→ (5,3)
→ (5,4)
→ (5,5)
→ J₂ at (5,6), turn RIGHT
→ (6,6)
→ (7,6)
→ bay (8,6)
Enter the exit
LEFT, DOWN×5

Canonical depth: 33 actions.

Necessity and anti-degeneracy

The bay is neither horizontally nor vertically aligned with the starting ball.

J₁ = DOWN is required to enter the lower route.

J₂ = RIGHT is required to reach the bay.

Leaving J₁ = RIGHT makes the ball stop before it, with the wall-facing arrow visibly explaining why.

Preparing only J₁ while J₂ = UP creates a visible route cycle; the cycle detector stops the ball before the repeated arrow entry and flashes both arrows.

The two arrows are independent preparation branches that converge at the launch; they may be prepared in either order.

The exit is placed on a forced revisit of the left spine, not at the end of the cargo line.

Level 5 — Stage at the Wall
Player-facing discovery

The ball’s first useful destination is not the bay. It must be deliberately launched right until it stops at (7,3) against the wall at (8,3).

That staged position changes the useful push direction: the agent can later stand above it and launch it downward.

However, staging the ball at (7,3) blocks the only ordinary-floor approach to arrow J at (7,5). Therefore J must be turned left before committing the ball to the wall.

A second arrow K at (6,5) is directly adjacent to J. The ball crosses both without pausing:

J points LEFT → immediately enter K → K points DOWN

This is the first authored demonstration of a contiguous arrow chain.

State-machine instance

Board: 12×10

Agent: (1,3), facing RIGHT

Ball: (2,3)

Arrow J: (7,5) = DOWN

Arrow K: (6,5) = RIGHT

Bay: (6,8)

Exit: (9,7), initially closed

Staging wall: (8,3)

Selection/cursor: none

Host cap: 512

############
#.........##
#.#####.#.##
#Ao.....#.##
#.#####.#.##
#.....>v#.##
#.####.##.##
#.####.##X##
######B#####
############
Canonical solution
Prepare the arrow that will become inaccessible
UP×2, RIGHT×6, DOWN×3, SPACE

At (7,4), rotate J from DOWN to LEFT.

Return and stage the ball
UP×3, LEFT×6, DOWN×2, RIGHT

The final RIGHT launches the ball along row 3. It stops at (7,3) immediately before wall (8,3).

The agent settles at (2,3).

Prepare the second arrow
LEFT, DOWN×2, RIGHT×4, SPACE

At (5,5), rotate K from RIGHT to DOWN.

Reach the opposite side of the staged ball and relaunch
LEFT×4, UP×4, RIGHT×6, DOWN×2

The first DOWN reaches (7,2). The second pushes the staged ball from (7,3).

Ball route:

(7,3)
→ (7,4)
→ J at (7,5), turn LEFT
→ K at (6,5), turn DOWN
→ (6,6)
→ (6,7)
→ bay (6,8)
Enter the exit
UP×2, RIGHT×2, DOWN×6

Canonical depth: 58 actions.

Necessity and anti-degeneracy

The wall at (8,3) is a tool: it creates the only stable position from which the ball can be pushed downward.

The bay cannot be reached from the initial horizontal launch.

Staging at (7,3) cuts the top access column to J; J must be prepared first.

If J = LEFT but K = RIGHT, the contiguous pair points into a two-arrow cycle. The ball stops before J, and both rims flash red.

K = DOWN is therefore necessary.

K may be prepared either before or after staging, while J has the stricter pre-staging dependency. This creates a partial-order network rather than one fixed checklist.

The exit is in a separate right-hand spine, forcing the agent to leave the cargo area after delivery.

Level 6 — Borrowed Route
Player-facing discovery

Two balls must use one central junction.

Central arrow J at (6,4) has two cargo jobs:

UP sends a ball to the northern bay;

RIGHT sends a ball toward the eastern bay.

The eastern downstream arrow K cannot be reached on foot from the west. The agent must first point J right and deliberately enter it, landing on the east side at (7,4). From there, the agent can rotate K.

Returning is not a separate mechanism. The agent faces left, rotates J from the far side until it points left, and rides it back west.

After both deliveries, the same prepared rightward route becomes the agent’s bridge to the exit.

State-machine instance

Board: 12×10

Agent: (1,4), facing RIGHT

West ball: (2,4)

South ball: (6,7)

Junction J: (6,4) = DOWN

East arrow K: (8,4) = DOWN

North bay: (6,1)

East bay: (10,4)

Exit: (10,1), initially closed

Selection/cursor: none

Host cap: 640

############
######B##.X#
######.##.##
######.##.##
#Ao...v.v.B#
#.###..#####
#.###..#####
#.###.o#####
#......#####
############
Canonical solution
Reach the central junction
DOWN×4, RIGHT×4, UP×4, LEFT, RIGHT

The agent stands at (5,4), facing J.

Point J east and ride it
SPACE×3, RIGHT

J rotates DOWN → LEFT → UP → RIGHT.

Entering it lands the agent at (7,4).

Prepare K, then return across J
SPACE×3,
LEFT,
SPACE×2,
LEFT

SPACE×3 rotates K from DOWN to RIGHT.

The first LEFT enters right-facing J and is ejected back to (7,4). The agent now faces left.

SPACE×2 rotates J from RIGHT to LEFT.

The final LEFT crosses J and lands at (5,4).

Send one ball north
LEFT, RIGHT, SPACE,
DOWN×4, LEFT×4, UP×4, RIGHT

LEFT, RIGHT restores a right-facing manipulation stance.

SPACE rotates J from LEFT to UP.

The route around the lower loop reaches (1,4).

RIGHT launches the west ball through J into bay (6,1).

Reorient and send the other ball east
RIGHT×3, SPACE,
DOWN×4, RIGHT, UP

Reach (5,4).

SPACE rotates J from UP to RIGHT.

Reach (6,8).

UP launches the south ball through J, through right-facing K, and into bay (10,4).

The exit opens.

Ride the completed cargo route to the exit
UP×3, RIGHT, UP×3, RIGHT

The third UP enters J from below and ejects the agent to (7,4).

RIGHT enters K and ejects the agent to (9,4).

UP×3, RIGHT enters exit (10,1).

Canonical depth: 59 actions.

Necessity and anti-degeneracy

K is inaccessible until the agent deliberately rides J.

K = RIGHT is necessary for the east bay.

One ball must pass through J = UP, and one must pass through J = RIGHT.

Sending both balls to the same output cannot work because the first filled bay becomes solid.

The balls are intentionally interchangeable. Swapping which physical ball fills which bay is valid, but it does not remove either required junction orientation.

J plays three contrasting roles without changing rules:

transport into the east region;

two-way cargo junction;

final bridge to the exit.

K is first a remote preparation problem, then cargo infrastructure, then agent transportation.

Level 7 — The Occupied Handle
Player-facing discovery

The final level introduces no new operational rule.

A ball already occupies (4,5), the only ordinary cell from which arrow K at (4,4) can be rotated:

left of K is a wall;

right of K is a bay;

above K is another arrow;

below K is the occupied cell.

K begins pointing right. The ball covering its handle therefore cannot be redirected. Pushing it upward necessarily sends it through K into the adjacent bay. Capturing that ball frees (4,5).

The agent can then stand on the newly cleared cell and rotate K upward.

The second ball must be launched right until it stages on that same cell. Once it occupies the handle, K becomes inaccessible again, so K must already point up. The second ball is then relaunched upward through the contiguous K → J chain, routed east by J, turned north by the remote arrow, and captured.

Finally, the exit lies north while J still points east. The agent rides the chain to the far side, rotates J from there, and uses the newly redirected route to enter the exit.

State-machine instance

Board: 12×8

Agent: (1,3), facing RIGHT

Movable staging ball: (2,5)

Handle-blocking ball: (4,5)

Central junction J: (4,3) = DOWN

Handle arrow K: (4,4) = RIGHT

Remote east arrow L: (10,3) = RIGHT

Adjacent bay: (5,4)

Northern east bay: (10,1)

Exit: (4,1), initially closed

Selection/cursor: none

Host cap: 800

############
####X#####B#
#....#####.#
#A..v.....>#
#.##>B######
#.o.o#######
#....#######
############
Canonical solution
Prepare the remote east arrow and return
UP,
RIGHT×2,
DOWN,
LEFT,
RIGHT,
SPACE×3,
RIGHT,
RIGHT×4,
SPACE×3,
LEFT×4,
SPACE×2,
LEFT

Detailed result:

The agent reaches (3,3) and faces right.

SPACE×3 rotates J from DOWN to RIGHT.

RIGHT rides J to (5,3).

RIGHT×4 reaches (9,3).

SPACE×3 rotates L from RIGHT to UP.

LEFT×4 returns to (5,3).

SPACE×2 rotates J from RIGHT to LEFT.

LEFT rides J back to (3,3).

Capture the ball occupying the handle
LEFT×2, DOWN×3, RIGHT×3, UP

The agent reaches (4,6) and pushes the handle-blocking ball upward.

Its route is forced:

(4,5)
→ K at (4,4), turn RIGHT
→ bay (5,4)

The agent settles at (4,5), exactly where the ball had been.

Reconfigure the freed handle and central junction
SPACE×3,
UP,
LEFT,
RIGHT,
SPACE×2

SPACE×3 rotates K from RIGHT to UP.

UP enters the contiguous chain:

K = UP;

then J = LEFT;

agent lands at (3,3).

LEFT, RIGHT gives a right-facing stance.

SPACE×2 rotates J from LEFT to RIGHT.

Stage and relaunch the second ball
LEFT×2, DOWN×2, RIGHT,
LEFT, DOWN, RIGHT×3, UP

The first RIGHT launches the ball from (2,5):

(2,5) → (3,5) → (4,5)

It stops at (4,5) before wall (5,5).

The agent loops to (4,6). The final UP relaunches the ball:

(4,5)
→ K at (4,4), turn UP
→ J at (4,3), turn RIGHT
→ (5,3)
→ (6,3)
→ (7,3)
→ (8,3)
→ (9,3)
→ L at (10,3), turn UP
→ (10,2)
→ bay (10,1)

The second bay fills and the exit opens.

Final far-side reroute
UP,
LEFT,
SPACE×3,
LEFT,
UP

UP from (4,5) enters K = UP, then J = RIGHT, landing at (5,3).

LEFT enters right-facing J and returns to (5,3), setting facing left.

SPACE×3 rotates J from RIGHT to UP.

LEFT enters J from the east and is ejected upward to (4,2).

UP enters the open exit at (4,1).

Canonical depth: 59 actions.

Necessity and anti-degeneracy

The first ball physically occupies the sole manipulation cell for K; no other side of K is usable.

K cannot be rotated before the first capture.

The first ball is therefore forced into the adjacent bay without requiring color or identity matching.

Before the first ball is removed, prematurely pushing the second ball right merely stops it at (3,5) behind the occupied handle. The obstruction is visible and the position remains recoverable.

After the first capture, K must be changed to UP.

Once the second ball stages at (4,5), K is inaccessible again. Incorrect preparation is therefore a visible commitment, not a hidden validation failure.

J = RIGHT is required for the second delivery.

J = UP is required for the final exit route.

L = UP is required for the remote northern bay.

The same ordinary cell (4,5) serves three contextual roles:

inaccessible manipulation handle;

freed control position;

cargo staging location.

The same arrow J serves:

east-region access;

cargo routing;

a far-side reversible gate;

final exit transportation.

No new noun, exception, or validator condition appears in the final level.

7. Progression and epistemic audit
Foundational attribution
Level 1

The only novel event is agent deflection. The arrow already points toward the open doorway, so its effect is visually predicted before it occurs.

Level 2

The established deflection rule creates a local impasse. SPACE changes only the arrow’s orientation, immediately and visibly. Three rotations also reveal the clockwise order rather than merely teaching a binary toggle.

Level 3

The first cargo push is low-cost and unambiguous:

one action causes uninterrupted travel;

the ball visibly enters the bay;

the bay visibly fills;

only then does the exit open.

No unrelated mechanism changes during the demonstration.

Contrasting mechanic roles
Mechanic	Early role	Later contrasting roles
Arrow	guides agent to exit	cargo turn, blocked gate, bridge, shared junction, manipulation lock
SPACE	correct one local route	prepare downstream, rotate from far side, preserve orientation before access is lost
Continuous ball travel	direct delivery	irreversible junction traversal, wall staging, shared-route pressure
Wall collision	visible failure	deliberate staging tool
Agent deflection	forced movement	region access, return bridge, final transportation
Ball occupancy	movable obstacle	route blocker, arrow-handle lock, staging commitment
Filled bay	completion evidence	permanent obstruction preventing duplicate delivery
Mature dependency structure

Level 4: two independent arrow-preparation branches converge at one launch.

Level 5: J must precede staging, while K may be prepared on either side of that commitment.

Level 6: remote preparation, two interchangeable cargo deliveries, junction reorientation, and final transport form a network rather than a fixed object order.

Level 7: remote-arrow preparation, forced first capture, handle reconfiguration, staging, shared-chain routing, and final far-side rotation create a strict but physically visible partial order.

8. Validator and implementation invariants

An implementation should enforce these assertions after every settled step():

agent_position is not an arrow
no free ball position is an arrow
agent_position is not in bays
balls and filled_bays are disjoint
filled_bays is a subset of bays
all free balls occupy distinct cells
exit_open == (balls is empty and filled_bays == bays)
won implies agent_position == exit_position
won implies exit_open

Additional behavioral tests:

A directional input updates facing even when blocked.

SPACE rotates only the arrow exactly one cell ahead.

Clicks never change logical state.

Agent motion stops on the first ordinary floor after an arrow chain.

Ball motion continues across ordinary floor after an arrow turn.

Neither entity settles on an arrow.

A blocked contiguous arrow run rejects entry.

A ball-route arrow-entry cycle terminates visibly and deterministically.

Balls treat both exit states as solid.

Filled bays never reopen.

Exit validation occurs only after the complete ball trace and capture have settled.

The seven canonical sequences above end in victory with action counts:

Level 1: 3
Level 2: 5
Level 3: 6
Level 4: 33
Level 5: 58
Level 6: 59
Level 7: 59
