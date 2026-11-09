Tumbling Colors

Exact authored level count: 7

This is a game about rolling a cube one cell at a time becoming difficult because every move also changes which colored face points toward every future obstacle.

The player controls one six-colored cube. Position and orientation are inseparable: a spatially convenient move may destroy the orientation needed later, while an apparent detour may be the only way to rotate the cube without abandoning a useful location.

The game has four persistent ideas and no late-game exception rules:

Rolling changes position and orientation.

Checkpoints inspect the top face.

Gates inspect the face rolling toward them.

The final dock inspects the complete visible orientation.

Difficulty comes from arranging those rules in loops, bottlenecks, ordered revisits, and routes that must be used for different purposes at different times.

1. Player-facing visual language
The cube

The cube has six permanently colored faces:

Face in the initial orientation	Color
Up U	11 yellow
Down D	15 purple
North N, toward UP	8 red
South S, toward DOWN and visually shown as the front	14 green
East E, toward RIGHT and visually shown as the right side	9 blue
West W, toward LEFT	6 magenta

The fixed opposite pairs are therefore:

yellow 11 opposite purple 15;

red 8 opposite green 14;

blue 9 opposite magenta 6.

The rendered cube always shows the current U, S, and E faces. Because opposite pairs never change, the hidden D, N, and W faces are inferable from the visible three once the player has observed the early rolls.

Floor and walls

Traversable floor is color 1. Internal walls are color 4. The outer background is color 0.

There are no cell outlines or internal grid borders. Adjacent floor cells visually merge into one connected region, and adjacent walls merge into solid shapes. The logical lattice becomes evident from the cube’s fixed roll distance and from aligned docks and checkpoints.

A move into a wall, outside the authored bounding box, or into a non-floor coordinate does not move or rotate the cube.

Dock

There is exactly one dock per level. It is a recessed 9×9-pixel pad occupying one logical cell.

The dock displays three colored registration rails:

the top rail is the required U face;

the lower/front rail is the required S face;

the right rail is the required E face.

Thus a dock target is always written as the tuple:

(required U, required S, required E)

The arrangement mirrors the three faces visibly rendered on the cube; no symbolic legend is needed.

Dock frame states:

color 3: checkpoints remain incomplete;

color 10: all checkpoints are complete and the dock is ready;

color 12 for two feedback frames: the cube entered but cannot finish;

color 14: successful lock.

The cube may freely roll onto and off an incorrect or inactive dock. A rejection never bounces or resets it.

Checkpoints

Each checkpoint is a large diamond/ring occupying one logical cell. Its center color is the required top-face color.

Checkpoint ring states:

color 10: currently active;

color 2: future checkpoint, not yet active;

color 14: completed.

Only the currently active checkpoint can complete. Completion is permanent.

On entering an active checkpoint:

if the post-roll top face matches its center, it completes;

otherwise its ring flashes color 12 and remains active.

On entering a future checkpoint:

it cannot complete, even when the top face matches;

its gray ring briefly lightens while the actual active checkpoint pulses;

no state changes.

This makes ordering visible rather than hidden.

Face-sensitive gates

A gate is a colored threshold on the shared edge between two floor cells. It is stateless and never stays open.

When crossing in direction dir, the gate checks the cube face pointing in that direction before the roll:

Movement	Face checked
UP	N
DOWN	S
LEFT	W
RIGHT	E

Equivalently, it checks the face that would become the bottom face as the cube rolls across the threshold.

If the color matches, the cube crosses and rolls normally. If it does not match, the cube stays in place and does not rotate. The gate flashes color 12.

The same gate rule applies from both directions. Crossing a gate from the reverse side therefore generally checks a different cube face.

2. Controls

Only the four arrow actions are gameplay actions.

Action	Result
UP	Attempt one-cell roll north
DOWN	Attempt one-cell roll south
LEFT	Attempt one-cell roll west
RIGHT	Attempt one-cell roll east
SPACE	Atomically reset the current level to its authored initial state
Click on any canvas cell	Recognized no-op; no object is selected and no cursor is rendered

Clicking is deliberately unused. Nothing on the board presents a click affordance.

All levels have:

selected_object = null
cursor_state = null
3. Exact orientation transition

Store orientation as:

(U, D, N, S, E, W)

For every successful roll, read all values from the old orientation and assign:

RIGHT: (U,D,N,S,E,W) <- (W,E,N,S,U,D)
LEFT:  (U,D,N,S,E,W) <- (E,W,N,S,D,U)
DOWN:  (U,D,N,S,E,W) <- (N,S,D,U,E,W)
UP:    (U,D,N,S,E,W) <- (S,N,U,D,E,W)

Blocked moves do not apply any part of this permutation.

All seven levels begin with:

position = authored start coordinate
orientation = (11,15,8,14,9,6)
checkpoint_index = 0
terminal = false
selected_object = null
cursor_state = null
step_count = 0
feedback = none
4. step() resolution order

All logical changes occur only in response to step(action).

For a nonterminal state:

Clear feedback from the preceding action.

If the action is SPACE, restore the complete initial state and return.

If the action is a click, increment the harness step counter but change no gameplay state.

For an arrow, compute the neighboring logical coordinate.

If that coordinate is not floor, produce wall-bump feedback and stop. Position and orientation remain unchanged.

If the crossed edge contains a gate, compare its color with the appropriate pre-roll directional face.

On a gate mismatch, produce gate-rejection feedback and stop. Position and orientation remain unchanged.

Otherwise, compute the new orientation, animate the roll, and atomically commit the new position and orientation at the final animation frame.

If the new coordinate is the active checkpoint, compare the post-roll top face and update checkpoint_index if matching.

If the new coordinate is the dock, validate it using the now-updated checkpoint index and post-roll orientation.

If the dock validation succeeds, set terminal = true.

Only after success validation may the external harness apply its episode truncation limit.

Gate checks therefore use the old orientation; checkpoint and dock checks use the new orientation.

5. Exact success condition

A level succeeds immediately after a successful roll when all of the following hold:

cube_position == dock_position
checkpoint_index == number_of_checkpoints
(U, S, E) == dock_requirement

The validator ignores:

route taken;

number of wall or gate bumps;

visits to future or completed checkpoints;

earlier visits to the dock;

clicks;

orientation of hidden faces except as already implied by (U,S,E);

action count, provided the harness has not truncated the episode.

Allowed:

crossing completed checkpoints;

entering the dock early;

leaving an incorrect dock;

crossing the same gate any number of times;

revisiting rooms and loops;

taking non-shortest but valid solutions.

Forbidden only by physics:

entering walls or non-floor coordinates;

crossing a gate with the wrong directional face;

completing a future checkpoint out of order.

The validator must never compare the player’s action string with the example solution.

6. Rendering contract
Canvas and logical coordinates

The render canvas is exactly 64×64 pixels.

Each authored level has a logical bounding box. Logical coordinates use:

x increasing left to right;

y increasing top to bottom.

Each logical cell is 9×9 pixels.

For a bounding box of width w and height h:

origin_x = floor((64 - 9*w) / 2)
origin_y = floor((64 - 9*h) / 2)

For a logical coordinate (x,y) in a bounding box beginning at (xmin,ymin):

pixel_x = origin_x + 9*(x-xmin)
pixel_y = origin_y + 9*(y-ymin)

The cell occupies the inclusive pixel range:

[pixel_x, pixel_x+8] × [pixel_y, pixel_y+8]

Every coordinate inside the bounding box that is not listed as floor is a wall cell. Everything outside the bounding box is background and nontraversable.

Adjacent cells receive no separating outlines.

Cube mask

The following is a specification mask, not runtime text:

.........
...kk....
..kTTk...
.kTTTTk..
kTTTTTTk.
kFFFkRRRk
kFFFkRRRk
.kFFkRRk.
..kk.kk..

Render symbols as:

T: current U color;

F: current S color;

R: current E color;

k: color 5;

.: transparent.

The dock’s perimeter rails are drawn after the cube so they remain visible when the cube begins or ends on the dock.

Dock geometry

Within its 9×9 cell:

Fill the recessed interior with color 3.

Draw a one-pixel perimeter in the current dock-frame color.

Draw these rails over the perimeter:

required U: local row 0, columns 2–6;

required S: local row 8, columns 0–4;

required E: local column 8, rows 4–8.

If occupied, draw the cube and then redraw the three rails above its black perimeter pixels.

Checkpoint geometry

Specification mask:

....A....
..AAAAA..
.AAAAAAA.
.AACCCAA.
AAACCCAAA
.AACCCAA.
.AAAAAAA.
..AAAAA..
....A....

A: checkpoint ring-state color;

C: required top-face color;

.: underlying floor.

Gate geometry

For a gate between horizontally adjacent cells, draw a two-pixel-wide vertical threshold straddling the shared edge. The colored portion is five pixels long and centered; its two end caps are color 5.

For a gate between vertically adjacent cells, transpose the same mask.

Draw order:

background
walls and floor
dock/checkpoint bases
gates
cube
feedback flashes and dock registration rails
7. Animation and feedback

Animation frames never consume actions and never accept input.

Successful roll

Every successful roll uses four deterministic render frames:

Cube in the old cell with the old orientation.

Cube translated three pixels toward the destination and visibly leaning over the shared edge.

Cube translated six pixels; use the new face assignment, showing the leading face folding under.

Cube in the destination cell with the standard mask and new orientation.

Logical position and orientation are committed only at frame 4. step() returns the final observation.

A successfully crossed gate flashes color 10 during frames 2–3.

Blocked wall move

Three feedback frames:

standard position;

cube shifted two pixels toward the wall, with the contacted wall edge flashing color 12;

standard position restored.

No logical state changes.

Rejected gate move

The same lean-and-return sequence is used, but only the gate threshold flashes color 12. This distinguishes a gate mismatch from ordinary collision.

Checkpoint feedback

Checkpoint state is committed at the end of the roll step.

Correct active checkpoint: ring changes to color 14 over three render frames; the next ring simultaneously changes from color 2 to color 10.

Incorrect active checkpoint: ring flashes color 12 twice.

Future checkpoint: future ring briefly changes 2 → 1 → 2 while the active ring pulses color 10.

Dock feedback

Inactive or mismatched dock: frame flashes color 12 twice; cube remains freely movable.

Successful dock: frame changes to color 14, pulses outward for four frames, and terminal success is committed in the same action.

8. Progression overview
Level	Discovery or composition	Verified shortest depth	Shortest action strings
1	Horizontal rolling and dock correspondence	2	1
2	The shortest spatial route can give the wrong orientation	4	1
3	Top-sensitive checkpoint and dock prerequisite	6	1
4	A directional face gate, crossed in both directions	15	1
5	Ordered checkpoints; matching a future checkpoint does nothing	34	8
6	Two gates; one room must prepare entry into another	46	6
7	Three-stage circuit and restoration of an earlier dock orientation	64	12

There is no gameplay timer or energy system. Recommended harness truncation caps are deliberately generous:

Level 1: 128
Level 2: 160
Level 3: 192
Level 4: 384
Level 5: 768
Level 6: 1024
Level 7: 1536

SPACE gives unlimited fresh attempts. The caps are not displayed and are not part of puzzle reasoning.

9. Authored levels

The maps below are documentation only. No characters or numerals are rendered in the game.

Legend:

·  ordinary floor
#  wall
S  cube start
D  dock
X  cube starts on dock
1  first checkpoint
2  second checkpoint
3  third checkpoint

Gate locations and colors are listed separately.

Level 1 — First Tumble
Player-facing discovery

The cube begins two cells from a brightly active dock. The dock’s lower/front rail is green, exactly matching the cube’s currently visible green front face. This is positive partial-goal evidence: the pad appears to be asking for the same three face positions shown on the cube.

The first RIGHT immediately swaps the top and side colors while leaving the green front unchanged. The second RIGHT places the cube on a perfectly matching dock and produces the green lock animation.

There is no competing rule, object, route, or simultaneous state change. A blind player can attribute the result entirely to rolling.

State-machine instance
bbox: x=1..3, y=1..1

x:     1 2 3
y=1:   S · D
floor_by_y:
  y=1: {1,2,3}

start: (1,1)
initial_orientation: (11,15,8,14,9,6)

checkpoints: []
gates: []

dock:
  position: (3,1)
  required (U,S,E): (15,14,6)
  colors: (purple, green, magenta)

selection/cursor: null
recommended action cap: 128
Intended solution

Grouped:

RR

Exact action string:

RR

Orientation evidence:

Initial visible:       (11,14,9)  = yellow, green, blue
After first RIGHT:     (6,14,11)  = magenta, green, yellow
After second RIGHT:    (15,14,6)  = purple, green, magenta
Validation and necessity

The only successful state is the dock with (U,S,E)=(15,14,6). One RIGHT is insufficient; blocked random directions do not rotate the cube. The two-step sequence is the unique shortest solution.

Attribution check: every useful action causes one visible roll; every visible object communicates either the mover, route, or objective.

Level 2 — The Longer Way
Player-facing discovery

The lower row deliberately resembles Level 1. The natural hypothesis is to repeat RIGHT, RIGHT.

That familiar route reaches the dock but produces:

(U,S,E) = (15,14,6)

The new dock instead asks for:

(U,S,E) = (11,8,6)

The dock flashes orange and leaves the cube in place. The upper row is the only new structural possibility. Going up, crossing above the dock, and coming down produces the required orientation.

This cleanly falsifies the simpler hypothesis “only the destination matters.” The exact same spatial destination accepts one route and rejects another.

State-machine instance
bbox: x=1..3, y=1..2

x:     1 2 3
y=1:   · · ·
y=2:   S · D
floor_by_y:
  y=1: {1,2,3}
  y=2: {1,2,3}

start: (1,2)
checkpoints: []
gates: []

dock:
  position: (3,2)
  required (U,S,E): (11,8,6)
  colors: (yellow, red, magenta)

selection/cursor: null
recommended action cap: 160
Intended solution

Grouped:

U RR D

Exact:

URRD

Key comparison:

Direct RR:
  position = dock
  visible orientation = (15,14,6)
  result = reject

Detour URRD:
  position = dock
  visible orientation = (11,8,6)
  result = success

The vertical rolls also bring the previously hidden red face into view, completing the visual introduction of all six face colors.

Validation and necessity

The middle lower cell is necessary as the familiar but incorrect route from Level 1. The upper row is necessary as the orientation-changing alternative. Removing either destroys the causal contrast.

The intended four-action path is the unique shortest solution.

Attribution check: only one new consequence is required—route order affects final orientation.

Level 3 — Come Back Green
Player-facing discovery

A large light-blue checkpoint ring now sits between the start and dock. Its center is green.

The obvious first move, RIGHT, lands on it with a magenta top. The ring flashes orange and stays active. This directly falsifies “touching a checkpoint is enough.”

The nearby upper square suggests a loop. Rolling UP, RIGHT, DOWN reaches the dock with every dock rail already matching, but the dock frame is still inactive because the checkpoint remains unfinished. Rolling LEFT returns to the checkpoint with green on top; it fills green, the dock frame becomes active, and one final RIGHT succeeds.

The player therefore sees both parts of the checkpoint rule physically:

top color controls checkpoint activation;

an uncompleted checkpoint prevents an otherwise perfect dock match.

State-machine instance
bbox: x=1..3, y=1..2

x:     1 2 3
y=1:   · · ·
y=2:   S 1 D
floor_by_y:
  y=1: {1,2,3}
  y=2: {1,2,3}

start: (1,2)

checkpoints, in order:
  1. position: (2,2)
     required top: 14 green

gates: []

dock:
  position: (3,2)
  required (U,S,E): (6,15,14)
  colors: (magenta, purple, green)

selection/cursor: null
recommended action cap: 192
Intended solution

Grouped:

R · URD · L · R

Exact:

RURDLR

Key states:

Step 1, first checkpoint entry:
  position = (2,2)
  top = 6 magenta
  checkpoint requires 14 green
  checkpoint remains active

Step 4, first dock entry:
  visible orientation = (6,15,14), exactly matching dock
  checkpoint_index = 0
  dock rejects solely because checkpoint is incomplete

Step 5, checkpoint revisit:
  top = 14 green
  checkpoint completes

Step 6:
  same matching dock orientation is restored
  level succeeds
Validation and necessity

The upper row is not padding: it is the smallest loop that demonstrates returning near the same location with a different top face. The dock is crossed before it is usable, establishing that pads are traversable terrain rather than terminal walls.

The path is the unique shortest solution.

Attribution check: the first checkpoint entry, early perfect dock entry, checkpoint completion, and final success occur on separate actions with distinct feedback.

Level 4 — Two-Sided Threshold
Player-facing discovery

A magenta gate is the only connection between two rooms. A dark wall directly above it prevents spatial bypass.

From the start, RIGHT reaches the gate approach. The cube’s visible east/right face is yellow, not magenta. Attempting another RIGHT makes the cube lean into the threshold, which flashes orange; the cube neither moves nor rotates.

The left 2×2 loop returns the cube to the same approach cell with magenta facing east. The gate now accepts it.

The far checkpoint is also magenta and activates after two more rightward rolls. The shortest retreat then approaches the same gate from the opposite side with the wrong west face. The upper-right loop changes the west face to magenta, allowing the reverse crossing.

Thus the gate is immediately reused with the same operational rule but a different checked face.

State-machine instance
bbox: x=2..6, y=1..2

x:     2 3 4 5 6
y=1:   D · # · ·
y=2:   S · · · 1
floor_by_y:
  y=1: {2,3,5,6}
  y=2: {2,3,4,5,6}

start: (2,2)

gate:
  edge: ((3,2),(4,2))
  required color: 6 magenta

checkpoints:
  1. position: (6,2)
     required top: 6 magenta

dock:
  position: (2,1)
  required (U,S,E): (8,15,6)
  colors: (red, purple, magenta)

selection/cursor: null
recommended action cap: 384
Intended solution

Grouped by structural purpose:

R · ULDR · R · RR · ULDL · L · LU

Exact:

RULDRRRRULDLLLU

Key states:

After initial R at (3,2):
  east face = 11 yellow
  gate requires 6 magenta
  direct crossing fails

After ULDR, again at (3,2):
  east face = 6 magenta
  crossing RIGHT succeeds

At (6,2):
  top = 6 magenta
  checkpoint completes

After ULDL, at (4,2):
  west face = 6 magenta
  crossing LEFT succeeds

Final dock:
  visible orientation = (8,15,6)
Validation and necessity

The wall (4,1) is essential: without it, the gate can be bypassed. The left loop is required to prepare the outbound face; the right loop is required to prepare the reverse face. The checkpoint forces the player to enter the far room, and the dock forces a return.

The 15-action path is the unique shortest solution.

Anti-cleanliness check: the dock is traversed during the initial left-room loop before it can finish the level, and the gate changes from outbound obstacle to inbound obstacle without changing its rule.

Level 5 — Not Yet
Player-facing discovery

Two checkpoints are visible:

checkpoint 1 at the upper left is bright and active;

checkpoint 2 at the lower right is gray and future.

The red gate blocks the route left. A direct leftward approach reaches it with the wrong hidden west face. The nearby lower-right square is the only useful experiment.

The required orientation loop deliberately crosses future checkpoint 2. At that moment the cube’s top is magenta, exactly matching checkpoint 2’s magenta center, but its ring stays gray and checkpoint 1 pulses. This is unambiguous evidence that checkpoint order matters; failure cannot be explained by top mismatch.

After checkpoint 1 completes, checkpoint 2 becomes active. On the first return, the cube reaches it with red on top, now demonstrating the opposite case: correct order but wrong orientation. A circuit around the same lower-right square returns with magenta on top and completes it.

The same checkpoint has three contrasting roles:

matching but future;

active but mismatched;

active and matched.

State-machine instance
bbox: x=1..6, y=1..3

x:     1 2 3 4 5 6
y=1:   1 · · # # #
y=2:   · · · · S ·
y=3:   # D · # · 2
floor_by_y:
  y=1: {1,2,3}
  y=2: {1,2,3,4,5,6}
  y=3: {2,3,5,6}

start: (5,2)

gate:
  edge: ((3,2),(4,2))
  required color: 8 red

checkpoints, in order:
  1. position: (1,1)
     required top: 15 purple

  2. position: (6,3)
     required top: 6 magenta

dock:
  position: (2,3)
  required (U,S,E): (14,9,11)
  colors: (green, blue, yellow)

selection/cursor: null
recommended action cap: 768
Intended solution

Grouped:

DRULLL · DLULU · RDDRUR · RDRULDR · ULLL · ULDRDL

Exact:

DRULLLDLULURDDRURRDRULDRULLLULDRDL

Important event trace:

Step 2:
  enter checkpoint 2 while it is future
  top = 6 magenta, exactly matching its target
  no completion; checkpoint 1 pulses

Step 6:
  first red-gate crossing, right region -> left region

Step 8:
  enter dock before any checkpoint is complete

Step 11:
  checkpoint 1 completes with top 15 purple
  checkpoint 2 becomes active

Step 14:
  revisit dock with only checkpoint 1 complete

Step 17:
  cross red gate left -> right

Step 20:
  enter active checkpoint 2 with top 8 red
  mismatch flash

Steps 21–24:
  loop around the lower-right square
  return to checkpoint 2 with top 6 magenta
  checkpoint 2 completes

Step 28:
  cross red gate right -> left

Step 34:
  enter dock with (U,S,E)=(14,9,11)
Validation and necessity

Every one of the 13 floor cells occurs in at least one shortest solution. The two disconnected-looking lower pockets are orientation loops, not decorative alcoves. The dock is encountered twice before success, and checkpoint 2 is encountered in all three meaningful order/orientation states.

Verified shortest depth is 34. There are eight shortest action strings; the validator accepts all of them.

Anti-cleanliness check: the active checkpoint is not the first checkpoint encountered, the dock is not encountered only at the end, and the lower-right loop means different things before and after checkpoint 1.

Level 6 — Borrowed Rooms
Player-facing discovery

The cube begins on the inactive dock at the center. Checkpoint 1 is in the eastern room.

The apparent route is UP toward the green eastern gate. After that move, however, the cube’s east face is blue, so the green gate rejects it. The magenta western gate is immediately passable because the initial west face is magenta.

The west room therefore has to be entered before the active eastern checkpoint. Its lower loop is borrowed as an orientation operator: it changes the cube so that returning through the magenta gate and approaching the green gate succeeds.

After checkpoint 1, the eastern room must be looped through to prepare the green gate from the reverse direction. Checkpoint 2 then requires the western room.

Completing checkpoint 2 still does not make the central dock usable: the cube returns to it with the wrong triplet. The player must cross the eastern gate again, use the upper-east loop solely as an orientation tool, and return.

Neither room has a permanent “checkpoint room” or “route room” identity. Each alternates between destination, orientation workspace, and obstacle.

State-machine instance
bbox: x=1..7, y=1..4

x:     1 2 3 4 5 6 7
y=1:   # # # # # · ·
y=2:   2 · # · · · ·
y=3:   · · · X # · 1
y=4:   · · · # # # #
floor_by_y:
  y=1: {6,7}
  y=2: {1,2,4,5,6,7}
  y=3: {1,2,3,4,6,7}
  y=4: {1,2,3}

start: (4,3), on the dock

gates:
  west gate:
    edge: ((3,3),(4,3))
    required color: 6 magenta

  east gate:
    edge: ((4,2),(5,2))
    required color: 14 green

checkpoints, in order:
  1. position: (7,3)
     required top: 14 green

  2. position: (1,2)
     required top: 15 purple

dock:
  position: (4,3)
  required (U,S,E): (11,8,6)
  colors: (yellow, red, magenta)

selection/cursor: null
recommended action cap: 1024
Intended solution

Grouped:

LLLDRURR
URRDR
ULDRULLL
DLDLULU
RDDLURRR
URRRULDLLD

Exact:

LLLDRURRURRDRULDRULLLDLDLULURDDLURRRURRRULDLLD

Structural event trace:

Step 1:
  cross magenta west gate

Steps 2–7:
  use west lower loop to prepare return

Step 8:
  cross west gate back onto the dock
  checkpoint 1 still incomplete

Step 10:
  cross green east gate

Step 13:
  checkpoint 1 completes with green top

Steps 14–21:
  eastern loop prepares west face for reverse green-gate crossing

Step 22:
  return to dock with only checkpoint 1 complete

Step 23:
  cross west gate toward checkpoint 2

Step 28:
  checkpoint 2 completes with purple top

Steps 29–36:
  west loop prepares return
  reach dock with all checkpoints complete but orientation (9,11,14)
  required orientation is (11,8,6), so dock rejects

Steps 37–45:
  cross the green gate again
  use upper-east loop purely for final orientation
  return through the green gate

Step 46:
  roll down onto dock with (11,8,6)
Validation and necessity

Both gates are crossed four times. All 17 floor cells occur in a shortest solution.

Removing the western lower loop prevents preparation for the first eastern excursion. Removing the upper-eastern cells prevents final dock alignment. Neither checkpoint can be treated as a simple errand because the opposite room is required before or after it.

Verified shortest depth is 46, with six shortest action strings.

Anti-cleanliness check: the route to checkpoint 1 begins by moving away from it; the dock is revisited in three different states; and the final eastern excursion occurs after every checkpoint is already complete.

Level 7 — Crooked Circuit
Player-facing discovery

The final level retains the two gate colors and central dock but expands each side into offset upper and lower loops.

Checkpoint order is:

lower east;

upper west;

upper east.

The active first checkpoint lies east, but the green east gate again cannot initially be crossed. The player must use the western lower loop to prepare it.

During the first eastern excursion, the route passes checkpoint 3 at the upper right. The cube arrives there with yellow on top, exactly matching checkpoint 3’s yellow target, but it is still future and remains gray. The player must continue downward to checkpoint 1.

After checkpoint 1, the eastern middle circuit prepares the return through the green gate. Checkpoint 2 then requires the western upper route. The western lower loop must subsequently be used again, now to prepare the route back toward checkpoint 3.

After checkpoint 3 completes, the cube can return to the central dock—but arrives with the wrong orientation. Importantly, the final required dock orientation was already seen at the same dock earlier, at steps 8 and 42, when checkpoints were incomplete. The final insight is therefore not discovering a new state but recreating a previously useful state at the correct phase.

The player must cross the western gate one final time, use a small upper-west circuit, and restore the earlier dock orientation.

State-machine instance
bbox: x=1..7, y=1..5

x:     1 2 3 4 5 6 7
y=1:   # # # # # · 3
y=2:   2 · # # # · ·
y=3:   · · # · · · ·
y=4:   · · · X # · 1
y=5:   · · · # # # #
floor_by_y:
  y=1: {6,7}
  y=2: {1,2,6,7}
  y=3: {1,2,4,5,6,7}
  y=4: {1,2,3,4,6,7}
  y=5: {1,2,3}

start: (4,4), on the dock

gates:
  west gate:
    edge: ((3,4),(4,4))
    required color: 6 magenta

  east gate:
    edge: ((4,3),(5,3))
    required color: 14 green

checkpoints, in order:
  1. position: (7,4)
     required top: 11 yellow

  2. position: (1,2)
     required top: 6 magenta

  3. position: (7,1)
     required top: 11 yellow

dock:
  position: (4,4)
  required (U,S,E): (9,11,14)
  colors: (blue, yellow, green)

selection/cursor: null
recommended action cap: 1536
Intended solution

Grouped by conceptual phase:

LLLDRURR
URRURULDDDR
LUURDLLL
DLLUUL
DDRRDLURR
URRURU
LDRDLLL
DLLLURDRR

Exact:

LLLDRURRURRURULDDDRLUURDLLLDLLUULDDRRDLURRURRURULDRDLLLDLLLURDRR
Key event trace
Steps 1–8:
  cross west gate, use lower-west loop, return to dock
  dock orientation is already the final target (9,11,14)
  checkpoints remain incomplete

Step 10:
  cross green east gate

Step 14:
  enter future checkpoint 3
  top = 11 yellow, exactly matching its target
  checkpoint stays future and gray

Step 19:
  checkpoint 1 completes with yellow top

Steps 20–27:
  eastern circuit prepares reverse green-gate crossing

Step 28:
  return to dock with only checkpoint 1 complete

Step 29:
  cross west gate

Step 33:
  checkpoint 2 completes with magenta top

Steps 34–42:
  lower-west loop prepares return
  arrive at dock again with the final target orientation (9,11,14)
  checkpoint 3 is still incomplete

Step 44:
  cross green gate

Step 48:
  checkpoint 3 completes with yellow top

Steps 49–55:
  return through the eastern circuit and green gate

Step 56:
  enter dock with all checkpoints complete
  current orientation = (15,8,9)
  target = (9,11,14)
  dock rejects

Steps 57–63:
  cross west gate once more
  use the upper-west mini-loop to restore the earlier dock orientation

Step 64:
  cross west gate onto the dock
  orientation = (9,11,14)
  all checkpoints complete
  success
Validation and necessity

The west gate is crossed six times and the east gate four times. All 21 floor cells occur in a shortest solution.

The upper-east cells are needed both for the early future-checkpoint observation and checkpoint 3 itself. The lower-east cells are needed for checkpoint 1. The lower-west loop prepares two different eastward phases, while the upper-west loop is required only after all checkpoints complete. The central dock is a staging surface, an early perfect-but-inactive state, a midgame perfect-but-inactive state, a late active-but-mismatched state, and finally the goal.

Verified shortest depth is 64, with twelve shortest action strings.

Anti-cleanliness check: the three checkpoints do not define three isolated errands. The future third checkpoint is encountered during the first phase, the final dock orientation appears twice before it is usable, and the last loop serves no checkpoint at all—it restores a previously observed orientation under a changed global state.
