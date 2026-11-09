Telescopic Causeway

This is a game about pushing a rigid beam through a wall, becoming difficult because every segment extended as a bridge on one side is retracted from the other.

Exact authored level count: 7.

Telescopic Causeway is a movement puzzle about conserving useful bridge length. A beam is never created or destroyed: pushing one handle transfers exactly one logical cell of beam from the pushed side to the opposite side. The same shift may extend a route, remove a return path, expose a handle, bury a handle in a sleeve, or place the player on the wrong side of the mechanism.

The player’s visible objective is always simply to reach the green landing pad. Beam states are tools, not checklist conditions.

1. Native geometry and rendering
Logical board

All levels use a 20×17 orthogonal lattice:

x = 0..19, left to right.

y = 0..16, top to bottom.

Movement is four-directional.

All coordinates and rectangles below are inclusive.

Every unlisted cell is blue void.

Notation:

Rect(x=a..b, y=c..d) is the inclusive Cartesian rectangle.

Cells{(x1,y1), ...} is an explicit cell set.

A ∪ B is set union.

A \ B removes cells in B from A.

64×64 render

Logical cell size: 3×3 rendered pixels.

Board origin: rendered pixel (2,10).

Board footprint: 60×51 pixels, covering rendered pixels x=2..61, y=10..60.

The top strip x=2..61, y=3..5 is the step-cap bar.

There are no grid lines and no internal borders between logical cells.

Logical cell (x,y) maps to:

render_x = 2 + 3x .. 4 + 3x
render_y = 10 + 3y .. 12 + 3y
Palette

Only the supplied integer color IDs are used.

Role	Color ID
Void	9 blue
Walkable stone floor	1 light gray
Wall and blocking terrain	4 very dark gray
Deep sleeve recess	5 black
Beam A body	11 yellow
Beam B body	6 magenta
Beam C body	15 purple
Beam D body	10 light blue
Endpoint-handle center	0 white
Avatar	0 white with center pixel 8 red
Goal landing pad	14 green with a white central inset
Invalid push preview	8 red
Used portion of step bar	3 dark gray
Remaining portion of step bar	1 light gray
Entity silhouettes

Beam body: Every logical beam cell is a filled 3×3 block in that beam’s color. Adjacent segments have no separating outlines, so the beam reads as one continuous strip through the wall.

Handle: The two endpoint cells have the same body color but contain a single white center pixel. Only these square endpoint cells can be pushed.

Avatar: A white diamond occupying the center and four edge-middle pixels of a 3×3 cell, with a red center pixel. It remains readable over both gray floor and saturated beams.

Goal: A connected 2×2 logical-cell landing pad, except where a larger floor island surrounds it. The exact green goal cells are specified per level.

Sleeve: Sleeve cells are openings in a dark wall band. The beam is always drawn continuously across those openings. The surrounding wall shape, rather than decorative markings, identifies the sleeve.

Draw order

Void.

Static floor.

Goal pad over its floor cells.

Walls.

Beam bodies and handles.

Avatar.

Temporary red invalid-push preview.

Step-cap bar.

A beam may lie over static floor. The beam is drawn above the floor, but the floor remains present and becomes visible again if the beam retracts.

2. Action space

The environment exposes exactly:

UP, DOWN, LEFT, RIGHT, SPACE, CLICK(x,y)

Only the four directions are meaningful puzzle actions.

UP, DOWN, LEFT, and RIGHT attempt one logical-cell movement or one beam push.

SPACE is a no-op.

Clicking any cell is a no-op.

There are no clickable objects, selections, cursors, or hidden click regions.

SPACE, clicks, blocked moves, and rejected pushes still increment step_count.

No level requires or rewards clicking.

External level reset is handled by the environment and is not mapped to a puzzle action.

All levels therefore begin with:

selected_object = null
cursor_state = none
3. State-machine contract
Persistent state
player: (x, y)

for each beam i:
    orientation_i: H or V
    fixed_axis_i: row y for H, column x for V
    length_i: positive integer
    start_i:
        leftmost x if H
        topmost y if V
    sleeve_interval_i:
        inclusive x interval if H
        inclusive y interval if V

step_count: integer
status: PLAYING | WON | FAILED

Static level state contains:

floor_cells
wall_cells
goal_cells
max_steps

There are no timers, enemies, random variables, hidden phases, or autonomous updates.

Derived beam cells

For horizontal beam i:

beam_cells_i = {(start_i + k, fixed_y_i) | k = 0..length_i-1}

For vertical beam i:

beam_cells_i = {(fixed_x_i, start_i + k) | k = 0..length_i-1}

The first and last cells are handles.

Support

A player cell is supported when it contains either:

static floor, or

at least one beam segment.

Void without a beam is unsupported and cannot be entered.

Walls are never traversable.

Ordinary movement

For a directional action, let q be the adjacent target cell.

The player moves to q when:

q is inside the board;

q is not a wall;

q is supported; and

the move is not an axial inward approach to an endpoint handle.

A handle can be entered normally:

from either perpendicular side;

from the next segment inside the same beam;

when moving outward from the handle onto a supported cell.

Only approaching a handle from outside, along the beam’s axis and toward the beam, attempts a push.

Horizontal pushes

For a horizontal beam occupying x=s..s+L-1 on row y:

Push right

player = (s-1, y)
action = RIGHT
target = left handle (s, y)
proposed start = s+1
player destination = old left handle (s, y)

Push left

player = (s+L, y)
action = LEFT
target = right handle (s+L-1, y)
proposed start = s-1
player destination = old right handle (s+L-1, y)
Vertical pushes

For a vertical beam occupying y=s..s+L-1 in column x:

Push down

player = (x, s-1)
action = DOWN
target = top handle (x, s)
proposed start = s+1
player destination = old top handle (x, s)

Push up

player = (x, s+L)
action = UP
target = bottom handle (x, s+L-1)
proposed start = s-1
player destination = old bottom handle (x, s+L-1)
Push legality

A proposed shift is valid only when all of the following hold:

Every proposed beam cell is inside the board.

No proposed beam cell is a wall.

No proposed beam cell overlaps another beam.

Every cell of that beam’s sleeve interval remains covered by the proposed beam.

The player destination—the old pushed handle cell—contains static floor after the active beam retracts from it.

Rule 5 is the support rule. The avatar steps into the handle’s old location while pushing. If the beam was the only support there, the push would strand the avatar and is rejected.

Floor does not block a beam. Floor cells under endpoints act as visible docking and pushing pads.

If an axial handle push is invalid, the action does not become an ordinary move onto the handle.

Conserved length

A beam’s length never changes. A valid shift translates every beam cell by one lattice unit. Therefore:

one cell appears at the extending end;

one cell disappears at the pushed end;

the sleeve remains covered;

the total number of beam cells remains constant.

There are no remote pushes or automatic chain reactions. One action can shift at most one beam.

Update order

Every call to step(action) performs exactly:

If terminal, return the unchanged terminal state.

Increment step_count.

Determine whether the action is a push attempt or an ordinary movement attempt.

Validate against the pre-action positions of all other beams.

Commit either:

the new beam start and new player cell;

an ordinary player move; or

no positional change.

Check victory.

If not victorious and step_count >= max_steps, set status = FAILED.

Reaching the goal on the final allowed step wins; victory is checked before cap failure.

Victory condition

A level is won immediately when:

player ∈ goal_cells

Everything else is ignored:

Beam positions do not need to match a target.

Beams may remain overextended.

Unused floor and handles do not matter.

Invalid actions taken earlier do not matter.

Only the final player position and the step cap are validated.

This keeps the objective physically natural rather than conjunctive.

4. Animation and feedback
Ordinary movement

Ordinary one-cell movement is rendered immediately. It has no multi-frame animation.

A blocked ordinary move leaves the avatar in place and briefly changes its white pixels to gray for two render frames. This is visual feedback only.

Valid beam shift

A beam shift returns a deterministic four-frame transition derived from the pre-step and post-step snapshots:

Original state.

Whole beam translated one-third of a logical cell.

Whole beam translated two-thirds of a logical cell.

Beam at its new cells; avatar displayed in the old pushed-handle cell.

The authoritative logical state is produced once by step(). Animation does not mutate state or call step() again. Inputs are ignored until the four visual frames finish. Animation frames consume no step budget.

Invalid beam shift

No positional state changes.

For four render frames:

the cell that would retract is rendered red;

the proposed extension cell is rendered red.

If the extension hits a wall, the red preview is drawn over that wall cell. This simultaneously reveals the attempted gain and the required loss.

Completion

The green goal pad alternates green and white for six render frames, then the board freezes in WON. These frames consume no steps.

5. Progression
Level	Main discovery	Verified shortest directional path	Step cap
1	One-for-one transfer	10	128
2	Far-side control and return loss	20	192
3	One beam as another beam’s control footing	32	320
4	Deliberate overextension to expose access	42	420
5	Vertical beams and nested control transfer	56	600
6	Temporary staging followed by restoration	68	760
7	Two independent preparation branches	78	1000

The caps are approximately ten or more times the reference lengths, allowing extensive experimentation. They are not puzzle resources.

In the replay notation below:

U = UP
D = DOWN
L = LEFT
R = RIGHT
R×5 = five consecutive RIGHT actions
Level 1 — Borrowed Cell
Player-facing discovery

The beam almost reaches the right island, but one blue void cell remains. The avatar begins immediately outside the left handle. Pushing once visibly transfers the leftmost segment to the right and closes the gap.

After the push, the avatar is still outside the new handle. Pressing right again gives the red loss/gain preview because the new handle is over void and cannot support the avatar after another shift. A small gray spur above the handle suggests approaching it from the side instead.

The first frame already shows a white avatar, a nearly complete bridge, and a green landing pad. No symbolic reference panel is needed.

State-machine instance
Floor =
    Rect(x=2..4, y=7..9)
  ∪ Cells{(5,7)}
  ∪ Rect(x=10..13, y=7..9)

Walls =
    Rect(x=6..7, y=3..13)
  \ Cells{(6,8), (7,8)}

Goal =
    Rect(x=11..12, y=7..8)

Player start = (3,8)
Selection/cursor = none
max_steps = 128

Beam A:

orientation = H
row = 8
length = 5
sleeve x = 6..7
initial start = 4
initial cells = (4,8)..(8,8)
initial handles = (4,8), (8,8)

The initial right endpoint (8,8) is separated from the right floor by void cell (9,8).

Verified reference replay
R, U, R, D, R×6

Key states:

R pushes A from start 4 to 5.

A becomes (5,8)..(9,8).

Player becomes (4,8).

U,R,D enters the new left handle (5,8) from above.

R×6 crosses the beam, enters the right island, and reaches (11,8).

Length: 10 directional actions.

Necessity and anti-cleanliness

Without shifting A, (9,8) remains unsupported.

A second right shift is rejected because old handle cell (5,8) has no floor beneath it.

The spur (5,7) is necessary for perpendicular boarding.

There is no alternate land route and no sequence of random axial moves that reaches the goal.

Level 2 — Don’t Take the Third
Player-facing discovery

The initial beam already connects the starting island to the large eastern platform. The target is instead on a disconnected western island.

The player crosses east and discovers that the right handle can now be pushed left. Two left shifts make the western endpoint reach the target island while leaving the eastern endpoint beside a side-entry ledge.

A third left shift looks greedily better: it extends farther into the target island. It also retracts the eastern handle into the wall sleeve. The avatar is then unable to enter the beam and cannot return. The player must stop at the merely sufficient length.

State-machine instance
Floor =
    Rect(x=5..6, y=9..11)
  ∪ Cells{(5,8)}
  ∪ Rect(x=1..2, y=7..9)
  ∪ Rect(x=9..13, y=7..10)

Walls =
    Rect(x=7..8, y=3..13)
  \ Cells{(7,8), (8,8)}

Goal =
    Rect(x=1..2, y=7..8)

Player start = (5,9)
Selection/cursor = none
max_steps = 192

Beam A:

orientation = H
row = 8
length = 7
sleeve x = 7..8
initial start = 5
initial cells = (5,8)..(11,8)
Verified reference replay
U, R×7, L×2, U, L, D, L×7

Key states:

U boards A from below.

R×7 traverses A and exits onto (12,8).

L×2 pushes A from start 5 → 4 → 3.

Player becomes (10,8).

A becomes (3,8)..(9,8).

U,L,D boards right handle (9,8) from above.

L×7 reaches goal cell (2,8).

Length: 20 directional actions.

Necessity and anti-cleanliness

One left shift leaves beam endpoint (4,8), still separated from the goal floor by (3,8).

Exactly two shifts are needed.

A third shift is legal but strategically losing:

A becomes (2,8)..(8,8);

the right handle lies in the sleeve;

its perpendicular neighbors are walls;

the next axial action is an invalid fourth push rather than a walk onto the beam.

Thus the same shift is first a tool and then a liability without changing its rule.

Level 3 — Borrowed Footing
Player-facing discovery

Two beams lie along the same row but do not touch. Beam A reaches the central floor; Beam B reaches the eastern platform.

The player can walk to Beam B and operate it from the east, but Beam A’s right-side control cell (8,8) is blue void. Moving B left one cell places B’s left handle exactly in that void. B is now not merely a bridge: it is the standing surface from which A can be pushed.

Trying to move B one more cell would overlap A and is rejected with the red collision preview.

State-machine instance
Floor =
    Rect(x=3..3, y=9..11)
  ∪ Cells{(3,8)}
  ∪ Rect(x=0..1, y=6..8)
  ∪ Rect(x=6..9, y=9..11)
  ∪ Cells{(6,8), (7,8)}
  ∪ Rect(x=12..15, y=7..11)

Walls =
    (Rect(x=4..5, y=3..13)
     \ Cells{(4,8), (5,8)})
  ∪ (Rect(x=10..11, y=3..13)
     \ Cells{(10,8), (11,8)})

Goal =
    Rect(x=0..1, y=6..7)

Player start = (3,9)
Selection/cursor = none
max_steps = 320

Beam A:

orientation = H
row = 8
length = 5
sleeve x = 4..5
initial start = 3
initial cells = (3,8)..(7,8)

Beam B:

orientation = H
row = 8
length = 5
sleeve x = 10..11
initial start = 9
initial cells = (9,8)..(13,8)
Verified reference replay
U, R×3, D, R×3, U, R×5,
L, U, L, D, L×4,
L,
D, L, U, L×5, U

Interpretation:

U,R×3,D,R×3,U moves from A to B’s left handle.

R×5 crosses B and reaches (14,8).

L shifts B from start 9 → 8.

U,L,D boards shifted B from above.

L×4 reaches B’s left handle (8,8).

The next L pushes A from start 3 → 2.

D,L,U boards shifted A from below.

L×5,U reaches goal cell (1,7).

Length: 32 directional actions.

Necessity and anti-cleanliness

A cannot be pushed left while (8,8) is void.

B must shift exactly once to supply that control cell.

Locking either beam at its initial position makes the level unsolvable.

The dependency is cyclic rather than a one-way list:

A gives access to B;

B supplies control of A;

shifted A becomes the final route.

The two beams are not encountered in their final use order; A is used, left behind, and then revisited.

Level 4 — The Extra Cell
Player-facing discovery

Beam A begins in a control pocket. One right push appears sufficient because its far half already lies over the eastern dock. However, after that push, the avatar is trapped outside the new left handle: the visible boarding alcove is one cell farther right.

The player must push A right again, apparently making the span worse on the goal side. That second shift aligns the left handle with the lower gray spur, allowing the player to board and cross.

On the far side, Beam B is repositioned beneath the eventual right endpoint of A. A can then be retracted four times, extending beyond its original western reach. B becomes the perpendicular boarding surface for the final A position.

State-machine instance
Floor =
    Cells{(3,5), (4,5), (5,5), (5,6), (6,6)}
  ∪ Rect(x=0..1, y=4..6)
  ∪ Rect(x=10..14, y=4..8)
  ∪ Rect(x=17..19, y=4..8)

Walls =
    (Rect(x=8..9, y=1..13)
     \ Cells{(8,5), (9,5), (8,6), (9,6)})
  ∪ (Rect(x=15..16, y=1..13)
     \ Cells{(15,6), (16,6)})

Goal =
    Rect(x=0..1, y=4..5)

Player start = (3,5)
Selection/cursor = none
max_steps = 420

The lower opening at (8..9,6) is a visible service channel through the first wall. It lets shifted B reach beneath A; it is ordinary traversable space, not a new device.

Beam A:

orientation = H
row = 5
length = 8
sleeve x = 8..9
initial start = 4
initial cells = (4,5)..(11,5)

Beam B:

orientation = H
row = 6
length = 9
sleeve x = 15..16
initial start = 10
initial cells = (10,6)..(18,6)
Verified reference replay
R×2, D, R, U,
R×4, D, R×9,
L, U, L, D,
L×3, U,
L×4,
D, L, U,
L×8

Key beam states:

R×2 shifts A from start 4 → 6.

D,R,U uses the newly aligned alcove to board A at (6,5).

R×4,D transfers from A onto B.

R×9 reaches the far-right control cell (19,6).

L shifts B from start 10 → 9.

U,L,D,L×3,U returns over B to (14,5), outside A’s right handle.

L×4 shifts A from start 6 → 2.

D,L,U uses B at (9,6) to enter A’s final right handle (9,5).

L×8 reaches goal cell (1,5).

Length: 42 directional actions.

Necessity and anti-cleanliness

After only one initial A shift, the player cannot reach (5,6) because (4,6) is void.

The second shift moves the player to (5,5), immediately beside that alcove.

Beam B is required to board A after its four-cell retraction.

Once east, B can be positioned before or after A is retracted; both final states are required, so the level is not a single dependency chain.

Shifting B left twice is a plausible but losing plan: its right handle retracts into its sleeve, cutting off re-entry from the right platform.

Locking either A or B makes the level unsolvable.

Level 5 — Corner Relay
Player-facing discovery

This level introduces a vertical beam without introducing a new operation.

Beam A reaches a lower hub. Its right-side control cell (11,12) is void, so it cannot yet be pushed left. Beam B is visible one cell above that position. Climbing B reaches a northern platform and Beam C.

At the top, the cell above B’s top handle is also void. Shifting C left places its handle at (11,4). Standing there and pressing down shifts the entire vertical B downward. B’s bottom endpoint then occupies (11,12), supplying the missing control footing for A.

The new property cannot be explained as ordinary climbing: B must visibly translate downward, losing its old top segment and gaining a bottom segment.

State-machine instance
Floor =
    Rect(x=4..4, y=13..15)
  ∪ Cells{(4,12)}
  ∪ Rect(x=0..2, y=11..14)

  ∪ Rect(x=9..10, y=11..15)
  ∪ Rect(x=12..13, y=11..15)
  ∪ Rect(x=9..13, y=14..15)

  ∪ Rect(x=12..14, y=3..6)
  ∪ Cells{(11,5)}

  ∪ Rect(x=17..19, y=2..6)

Walls =
    (Rect(x=5..6, y=2..15)
     \ Cells{(5,12), (6,12)})

  ∪ (Rect(x=7..15, y=7..8)
     \ Cells{(11,7), (11,8)})

  ∪ (Rect(x=15..16, y=1..9)
     \ Cells{(15,4), (16,4)})

  ∪ Cells{(10,4)}

Goal =
    Rect(x=0..1, y=11..12)

Player start = (4,13)
Selection/cursor = none
max_steps = 600

The lower hub deliberately has a void notch at (11,12..13). B fills the upper cell of that notch only after shifting down.

Beam A:

orientation = H
row = 12
length = 7
sleeve x = 5..6
initial start = 4
initial cells = (4,12)..(10,12)

Beam B:

orientation = V
column = 11
length = 7
sleeve y = 7..8
initial start = 5
initial cells = (11,5)..(11,11)

Beam C:

orientation = H
row = 4
length = 7
sleeve x = 15..16
initial start = 12
initial cells = (12,4)..(18,4)
Verified reference replay
U, R×5, U, R×2,
U×6, R, U,
R×7, L, U, L, D, L×6,
D, R, D, L, D×6,
L, U, L, D, L×8

Key states:

U,R×5,U,R×2 crosses A and enters B’s bottom handle at (11,11) from the side.

U×6,R,U climbs B and enters C at (12,4).

R×7 crosses C to (19,4).

L shifts C from start 12 → 11.

U,L,D,L×6 boards shifted C and reaches (11,4).

D pushes B from start 5 → 6.

R,D,L,D×6 side-enters shifted B and descends to (11,12).

L pushes A from start 4 → 3.

U,L,D boards A at (9,12).

L×8 reaches goal cell (1,12).

Length: 56 directional actions.

Necessity and anti-cleanliness

B in its initial position does not occupy A’s control cell (11,12).

C in its initial position does not occupy B’s control cell (11,4).

The blocker at (10,4) prevents a second C shift and makes its stopping point physical rather than numerical.

A second downward B shift is rejected because old top cell (11,6) has no static support.

A, B, and C are all necessary; fixing any one in place makes the goal unreachable.

B changes role three times without changing rule:

route upward;

object pushed by C;

standing surface used to push A.

Level 6 — Restore Past the Start
Player-facing discovery

Beam A again begins in a control pocket. Two right shifts expose its only boarding alcove and provide temporary access to the large eastern hub.

This time the useful final position is not the initial position. From the east, A must be pushed left four times, finishing two cells farther west than where it began. That reaches the goal island, but leaves the player on the eastern side with no way to board the retracted endpoint.

The B–C relay from Level 5 independently creates the missing final approach. The player may retract A first or configure B and C first. Both branches converge only when B descends directly above A’s final right endpoint.

State-machine instance
Floor =
    Cells{(3,13), (4,13), (5,13), (5,14), (6,14)}
  ∪ Rect(x=0..1, y=12..14)
  ∪ Rect(x=10..14, y=10..15)
  ∪ Rect(x=10..14, y=3..6)
  ∪ Cells{(9,5)}
  ∪ Rect(x=17..19, y=2..6)

Walls =
    (Rect(x=7..8, y=8..16)
     \ Cells{(7,13), (8,13)})

  ∪ (Rect(x=5..13, y=7..8)
     \ Cells{(9,7), (9,8)})

  ∪ (Rect(x=15..16, y=1..9)
     \ Cells{(15,4), (16,4)})

  ∪ Cells{(8,4)}

Goal =
    Rect(x=0..1, y=12..13)

Player start = (3,13)
Selection/cursor = none
max_steps = 760

Beam A:

orientation = H
row = 13
length = 8
sleeve x = 7..8
initial start = 4
initial cells = (4,13)..(11,13)

Beam B:

orientation = V
column = 9
length = 7
sleeve y = 7..8
initial start = 5
initial cells = (9,5)..(9,11)

Beam C:

orientation = H
row = 4
length = 9
sleeve x = 15..16
initial start = 10
initial cells = (10,4)..(18,4)
Verified reference replay
R×2, D, R, U, R×8, L×4,
U×3, L, U×5, R, U,
R×9, L, U, L, D, L×8,
D, R, D, L, D×7,
L×8

Key states:

R×2 stages A from start 4 → 6.

D,R,U boards the exposed left handle.

R×8 crosses to (14,13).

L×4 restores A past its initial position, from start 6 → 2.

U×3,L,U×5 reaches the top of B.

R,U,R×9 enters and crosses C.

L,U,L,D,L×8 shifts C from start 10 → 9 and reaches (9,4).

D shifts B from start 5 → 6.

R,D,L,D×7 re-enters B and descends through its bottom onto A’s final endpoint (9,13).

L×8 reaches goal cell (1,13).

Length: 68 directional actions.

Necessity and anti-cleanliness

One initial A push leaves its boarding alcove inaccessible; two are required.

A third right push is rejected because old handle cell (6,13) has no floor.

The four left pushes use floor cells (13,13), (12,13), (11,13), and (10,13) as successive docking supports.

A fifth left push is rejected because old endpoint (9,13) has no floor beneath it.

A can be fully restored before the B–C branch, as in the reference path, or afterward. This creates two state conditions rather than a prescribed device order.

Fixing A, B, or C makes the level unsolvable.

Level 7 — Two Preparations
Player-facing discovery

The final level splits preparation into two independent branches.

Beam A is staged right exactly as in Level 6. Its far-side control cell (14,13) is now deliberately missing from the floor. Beam D, located through a second lower wall, must shift left so its endpoint occupies that cell. D then becomes the standing surface from which A can be retracted.

Independently, the B–C relay must be completed so that B descends above A’s eventual final endpoint. D enables changing A; B enables boarding the changed A.

After A has been staged, the player may solve the D branch or the B–C branch first. Neither branch alone is sufficient.

State-machine instance
Floor =
    Cells{(3,13), (4,13), (5,13), (5,14), (6,14)}
  ∪ Rect(x=0..1, y=12..14)

  ∪ Rect(x=10..14, y=10..12)
  ∪ Rect(x=10..13, y=13..13)
  ∪ Rect(x=10..14, y=14..15)
  ∪ Cells{(15,14)}

  ∪ Rect(x=10..12, y=3..6)
  ∪ Cells{(9,5)}

  ∪ Rect(x=15..19, y=2..6)
  ∪ Rect(x=17..19, y=12..15)

Walls =
    (Rect(x=7..8, y=8..16)
     \ Cells{(7,13), (8,13)})

  ∪ (Rect(x=5..13, y=7..8)
     \ Cells{(9,7), (9,8)})

  ∪ (Rect(x=13..14, y=1..9)
     \ Cells{(13,4), (14,4)})

  ∪ Cells{(8,4)}

  ∪ (Rect(x=15..16, y=9..16)
     \ Cells{(15,13), (16,13), (15,14)})

Goal =
    Rect(x=0..1, y=12..13)

Player start = (3,13)
Selection/cursor = none
max_steps = 1000

Cell (15,14) is a visible maintenance alcove under D’s left sleeve mouth. It is the only way to enter D from the lower hub before D is shifted.

Beam A:

orientation = H
row = 13
length = 8
sleeve x = 7..8
initial start = 4
initial cells = (4,13)..(11,13)

Beam B:

orientation = V
column = 9
length = 7
sleeve y = 7..8
initial start = 5
initial cells = (9,5)..(9,11)

Beam C:

orientation = H
row = 4
length = 7
sleeve x = 13..14
initial start = 10
initial cells = (10,4)..(16,4)

Beam D:

orientation = H
row = 13
length = 4
sleeve x = 15..16
initial start = 15
initial cells = (15,13)..(18,13)
Verified reference replay
R×2, D, R, U, R×4,
D, R×5, U, R×4,
L, U, L, D, L×3,
L×4,

U×3, L, U×5, R, U,
R×7, L, U, L, D, L×6,
D, R, D, L, D×7,
L×8
D branch and A restoration

R×2 stages A from start 4 → 6.

D,R,U boards A.

R×4,D,R×5,U reaches D’s left handle through the maintenance alcove.

R×4 crosses D and exits at (19,13).

L shifts D from start 15 → 14.

U,L,D,L×3 re-enters D and reaches its new left handle (14,13).

L×4 pushes A from start 6 → 2.

The first of these four left actions is made while standing on D. The remaining three are supported by the eastern docking floor.

B–C branch and final crossing

U×3,L,U×5 climbs B.

R,U,R×7 enters and crosses C.

L shifts C from start 10 → 9.

U,L,D,L×6 returns along C to (9,4).

D shifts B from start 5 → 6.

R,D,L,D×7 descends B onto A’s endpoint (9,13).

L×8 reaches goal cell (1,13).

Length: 78 directional actions.

Necessity and anti-cleanliness

Static floor deliberately omits (14,13), so A cannot be pushed left until D occupies that cell.

D cannot shift left twice while A is staged: the proposed second shift would overlap A’s endpoint (13,13) and is rejected in red.

The B–C branch does not help push A; it only creates final access to A.

The D branch does not provide final access to A; it only enables A’s retraction.

Either preparation branch may be completed first after A is staged.

A, B, C, and D are all necessary. Locking any one at its initial offset leaves no winning state.

No fourth beam behavior or special final-level exception is introduced. The puzzle is entirely a composition of:

conserved shifts;

control-side transfer;

side-entry geometry;

overextension;

temporary support;

beam collision;

restoration.

6. Mechanical verification and design audit

The seven instances were checked under the state machine above.

Level 1 shortest path: 10
Level 2 shortest path: 20
Level 3 shortest path: 32
Level 4 shortest path: 42
Level 5 shortest path: 56
Level 6 shortest path: 68
Level 7 shortest path: 78

For every level, fixing any beam that is intended to move at its initial position makes the level unsolvable.

The progression introduces consequences rather than a stream of unrelated nouns:

Level 1 establishes conserved translation.

Level 2 turns extra extension into loss of control.

Level 3 makes one beam a control surface for another.

Level 4 makes overextension necessary for handle access.

Level 5 applies the identical operation vertically.

Level 6 makes a beam’s temporary and final positions serve opposing purposes.

Level 7 creates two independent preparation branches that converge on one final causeway.

The goal validator remains unchanged across all seven levels: move the avatar onto the green pad.
