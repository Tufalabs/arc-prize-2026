Beam Compass

Exact authored level count: 7

This is a game about turning arrows becoming difficult because the arrows and sockets themselves interrupt the straight lines needed by the rest of the field.

Beam Compass begins as a one-click aiming toy. An embossed arrow rotates clockwise and projects a straight beam toward a visible socket. The complication is entirely spatial: every emitter is also an opaque body, every terminal absorbs the first beam that reaches it, and only empty beam intersections are harmless.

Late levels therefore become line-of-sight assignment puzzles. A terminal that is already lit may be lit by the wrong emitter. An emitter that appears to be pointed uselessly may be occupying the exact location that prevents another beam from passing. Since every authored level contains the same number of emitters and terminals, duplicate coverage necessarily leaves another socket dark, but this follows naturally from the single visible objective rather than from a separate matching rule.

There is no timer, energy limit, move limit, hidden counter, or irreversible action. Four clicks restore an emitter’s orientation, so experimentation is always recoverable.

1. Controls and click affordance

The available action space remains:

UP, DOWN, LEFT, RIGHT, SPACE, and clicking on any cell.

Only clicking is used.

Action	Effect
Click an emitter cell	Rotate that emitter clockwise by exactly 90°, then recompute every beam and terminal.
Click a terminal, wall, beam, empty cell, or canvas margin	No simulation change; show a brief neutral ripple at the clicked location.
UP, DOWN, LEFT, RIGHT, SPACE	Accepted no-ops. The settled frame remains unchanged.

There is no persistent cursor or selected emitter. The click directly acts on the emitter occupying the clicked cell.

Emitters are the only objects rendered as embossed directional arrows. Their entire logical cell is a click target; the player never has to find a tiny handle or infer that a terminal or wall might secretly be interactive.

Clicking belongs to the central mechanic because orientation is the only mutable state. It is not an auxiliary control layered onto a movement game.

2. Rendering contract
2.1 Canvas and logical lattice

Render canvas: 64×64 pixels.

Native gameplay lattice: 14×14 logical cells.

Logical coordinates:

x = 0…13, left to right.

y = 0…13, top to bottom.

Logical cell (x,y) maps to the inclusive pixel rectangle:

pixel_x = 4 + 4x … 7 + 4x
pixel_y = 4 + 4y … 7 + 4y

Thus the playfield occupies pixels (4,4) through (59,59), leaving a four-pixel margin.

No logical grid lines or internal cell borders are rendered. Empty cells form one continuous dark field.

All IDs and coordinates in this specification are implementation labels only. No letters, numbers, or text appear in the rendered game.

2.2 Colors
Purpose	Color ID
Outer canvas margin	5
Empty playfield	4
Wall body	2
Emitter body	9
Emitter arrow tip and beams	10
Unlit terminal ring	11
Lit terminal halo	12
Lit terminal center	11
Beam impact cap	0
Invalid-click ripple	1
Emitter click flash	7
Completion pulse	14

Only integer color IDs 0–15 are used.

2.3 Emitter silhouette

An emitter occupies one logical cell but is rendered as a connected multi-pixel arrow module, not a 1×1 token.

The north-facing 4×4 pattern is:

4 10 10 4
9  9  9 9
4  9  9 4
4  9  9 4

East, south, and west emitters are exact clockwise rotations of this matrix.

The lighter tip connects visually to the emitted beam. During a valid click flash, body pixels of color 9 temporarily use color 7; tip pixels remain color 10.

2.4 Terminals

An unlit terminal is an empty socket:

11 11 11 11
11  4  4 11
11  4  4 11
11 11 11 11

A lit terminal has a filled center and persistent halo:

12 12 12 12
12 11 11 12
12 11 11 12
12 12 12 12

The filled center and orange halo remain visible in every settled frame for as long as at least one beam hits the terminal.

Terminals do not indicate ownership. If two beams hit the same terminal, both approach lines remain visible, but the socket has the same lit appearance.

2.5 Walls and rim

A wall cell is a solid 4×4 block of color 2. Adjacent wall cells merge visually without outlines.

The outer edge of the playfield behaves as an opaque rim made from the same conceptual material, though it is represented by the transition to the black canvas margin rather than by a separate inner border.

2.6 Beams

A beam is a one-pixel-wide color-10 line through the centerline of logical cells:

Horizontal beam pixel row: 4 + 4y + 2.

Vertical beam pixel column: 4 + 4x + 2.

The beam begins at the emitter’s arrow tip and extends toward its first collision.

At an empty crossing, horizontal and vertical segments are both drawn continuously, producing an uninterrupted plus shape. No crossing marker, spark, state change, or collision is added.

A beam that hits a wall or emitter receives a persistent two-pixel color-0 cap perpendicular to the beam at the struck surface. This makes it clear that the physical body, rather than another beam, caused the stop.

A beam reaching a terminal ends at the terminal’s outer ring. The terminal itself supplies the endpoint feedback.

3. State-machine contract
3.1 Stored state
LevelState:
    width = 14
    height = 14

    emitters:
        id
        position: (x,y)
        orientation: one of N,E,S,W

    terminals:
        id
        position: (x,y)

    walls:
        set of occupied logical cells

    selected_emitter = null
    won: boolean
    last_feedback:
        none | invalid_click(position) | emitter_click(id)

The following are derived after every action rather than treated as independently mutable state:

rays[emitter_id]:
    ordered traversed cells
    collision kind: rim | wall | emitter | terminal
    collision object, if any

terminal_hits[terminal_id]:
    set of emitter IDs whose rays terminate there

terminal_lit[terminal_id]:
    terminal_hits[terminal_id] is nonempty

Object positions never overlap.

3.2 Orientation order

Clockwise rotation is exactly:

N → E → S → W → N

Internally, this may be stored as:

N = 0
E = 1
S = 2
W = 3

with click update:

orientation = (orientation + 1) mod 4
3.3 Raycasting

For each emitter, begin at its logical position and repeatedly advance one cell in its current direction.

At each advanced position:

If the position is outside 0…13 × 0…13, stop at the rim.

If it contains a wall, stop at that wall.

If it contains another emitter, stop at that emitter.

If it contains a terminal, record a hit on that terminal and stop.

Otherwise, continue.

The source emitter does not collide with its own beam.

Emitters are opaque in every orientation. An emitter facing away from an arriving beam still blocks it.

Lit and unlit terminals are equally opaque. Lighting a terminal never makes it transparent.

Beams are not placed in the collision map. They never stop, bend, activate, or alter other beams.

Raycasts are conceptually simultaneous and independent. Iteration order over emitter IDs cannot affect the result.

3.4 step() resolution order

For every action:

Save the pre-action orientation and derived beam state for animation.

Resolve the action.

Valid emitter click: rotate that one emitter clockwise.

Any other click or non-click action: do not alter orientations.

Recompute all rays from the complete updated orientation state.

Recompute every terminal’s hit set and lit state.

Set:

won = all terminal_lit values are true

Produce the deterministic visual transition.

Present the settled post-action frame.

Simulation state changes only inside step(). Animation frames are render views and do not perform additional simulation updates.

3.5 Animation

A valid emitter click produces three short display frames:

Turn frame: the clicked emitter flashes color 7; its pre-action beam is still visible.

Growth frame: the arrow is shown in its new orientation, the old beam is absent, and the first half of the new ray is drawn. Terminal halos reflect only unchanged beams during this transitional frame.

Settled frame: the complete new beam, impact endpoint, and post-action terminal states are shown.

For a ray whose endpoint is immediately adjacent, the growth frame shows only the arrow tip; the collision appears in the settled frame.

Animation frames consume no actions, do not increment any budget, and accept no additional input. A headless implementation may omit intermediate frames as long as the settled frame and state transition are identical.

An invalid click produces a two-frame color-1 ripple in the clicked logical cell, then returns to the unchanged settled frame.

On the first transition to won = true, all lit terminal halos pulse color 14 for two display frames before returning to their normal lit appearance. The level is complete on the same step() that lights the final terminal; SPACE is not required.

3.6 Success validator

The only gameplay requirement is:

success ⇔ every terminal is lit after the current action

Relevant:

Whether each terminal has at least one incoming beam.

Ignored:

Which emitter lights which terminal.

Whether a terminal receives multiple beams.

Beams that end at walls, emitters, or the rim.

Total clicks.

Click order.

Intermediate lit configurations.

Unused directions.

Allowed:

Multiple beams converging on one terminal.

Any number of harmless empty-cell beam crossings.

Arbitrarily many exploratory clicks before completion.

Forbidden only by the physical rules:

A beam passing through a wall, emitter, terminal, or rim.

Counterclockwise rotation.

Clicking a non-emitter to alter state.

Every authored level has an equal number of emitters and terminals. Since one emitter can terminate at at most one terminal, any winning state in these levels automatically uses every emitter and gives every terminal at least one distinct emitter. This consequence is not separately checked by the validator.

4. Progression
Level	Main discovery or reinterpretation	Shortest clicks
1	Clicking rotates clockwise; a beam fills a socket and completes the level.	1
2	An emitter body blocks an arriving beam while still projecting its own.	6
3	The first terminal absorbs a beam; a farther collinear terminal remains dark.	6
4	Perpendicular beams safely cross on an empty cell.	12
5	Blocked lines create handoffs and coupled terminal ownership.	21
6	A wall makes terminal approach direction matter inside a branching assignment network.	30
7	Initially lit terminals may all have the wrong owners; the entire field must be reassigned.	42

Ei×k below means clicking any pixel inside emitter Ei’s logical cell k consecutive times. The listed click order is an explanatory replay, not a hidden temporal requirement. The physics is history-independent, so any order reaching the same orientation vector is accepted.

Each level has exactly one winning orientation vector under the specified geometry.

5. Authored levels
Level 1 — One Quarter Turn
Player-facing discovery

The board contains one blue arrow and one empty yellow socket on the same horizontal line. The arrow initially faces north. It is the only object with an embossed directional silhouette and the only plausible clickable module.

Clicking it once rotates the arrow east. The new beam grows directly into the socket, the socket fills, and the green completion pulse appears. This one-action setup supplies positive goal evidence without a written legend.

Initial state

Terminals:

T1 = (10,7)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(3,7)	N	E→T1	E→T1	1

Other state:

walls = {}
selected_emitter = null
won = false
initial terminal hits = {}

The initial north beam reaches the upper rim.

Exact replay
click E1 at (3,7) once

Post-action:

E1 = E
E1 ray terminates at T1
T1 is lit
won = true
Attribution and necessity check

Only one simulation variable changes. The player sees the arrow rotate, the old beam disappear, the new beam travel east, and the socket fill. No unrelated object or simultaneous rule change competes for explanation.

Clicking any non-emitter cell does not solve the level. The one correct emitter click is required.

Level 2 — The Arrow in the Way
Player-facing discovery

E1 is aligned with the right-hand terminal, but E2’s body occupies the line between them. E1 initially faces east, so its beam visibly ends on E2’s west face with a white impact cap. At the same time, E2 faces south and projects its own beam toward the rim.

This creates the first local impasse: E1 cannot illuminate the apparently obvious terminal behind E2. Rotating E2 east shows the handoff—E2 itself can illuminate that terminal. E1 must instead turn north toward the other socket.

The settled frame distinguishes body blockage from beam collision: E1’s beam reaches E2’s physical surface, while E2’s own beam leaves from a different side.

Initial state

Terminals:

T1 = (2,2)
T2 = (12,8)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(2,8)	E	N→T1	N→T1	3
E2	(7,8)	S	E→T2	E→T2	3

Other state:

walls = {}
selected_emitter = null
won = false
initial terminal hits = {}

Important initial endpoints:

E1 east ray → E2 body
E2 south ray → rim
Exact replay
E2×3:
    S → W → N → E
    E2 now lights T2

E1×3:
    E → S → W → N
    E1 now lights T1

Total: 6 clicks.

Attribution and necessity check

E1 has no orientation from which it can reach T2 because E2’s body is fixed in the line of sight. E2 is therefore necessary as both obstruction and useful source.

The level’s unique winning state is:

E1=N, E2=E

Neither terminal can be omitted, and neither emitter can be ignored.

Level 3 — First Socket Wins
Player-facing discovery

T1 and T2 lie on the same row to the east of E1, with T1 closer and no wall or emitter between the two terminals. E1 initially faces north.

The distinguishing experiment is one click on E1. Its east beam reaches T1 and stops. T1 fills, but the farther T2 remains empty even though the cells between T1 and T2 are visibly clear.

This directly rules out the plausible model that a beam continues through a terminal or illuminates every terminal on its line. E2, positioned below T2, must approach T2 from the south. E3 serves the side terminal.

Initial state

Terminals:

T1 = (6,7)    # near terminal
T2 = (12,7)   # farther terminal on the same row
T3 = (3,11)   # side terminal

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(2,7)	N	E→T1	E→T1	1
E2	(12,11)	E	N→T2	N→T2	3
E3	(6,11)	E	N→T1, W→T3	W→T3	2

Other state:

walls = {}
selected_emitter = null
won = false
initial terminal hits = {}

E3’s initial east beam ends on E2’s body, reusing the established emitter-opacity rule.

Exact replay
E1×1:
    N → E
    T1 lights
    T2 remains visibly empty

E2×3:
    E → S → W → N
    T2 lights

E3×2:
    E → S → W
    T3 lights

Total: 6 clicks.

Attribution and necessity check

The one-click E1 experiment isolates terminal absorption:

The beam visibly reaches T1.

T2 is collinear behind T1.

No emitter or wall lies between T1 and T2.

T2 remains dark immediately after the click.

The unique winning state is:

E1=E, E2=N, E3=W

A solution cannot ignore absorption: E1 can only terminate at T1, so E2 is required for T2.

Level 4 — Cross Without Colliding
Player-facing discovery

Two long required beam routes cross at the empty central cell (7,7):

E1 must eventually project east.

E2 must eventually project south.

The remaining two emitters create the assignment pressure that makes those long routes mandatory. E3 can only light T3. Once E3 owns T3, E1 must use its eastern route. E4 must then take T4, forcing E2 south.

A useful experiment is to establish E1’s horizontal beam first, then rotate E2 south. E2’s beam grows through (7,7) without shortening E1’s beam or extinguishing T1. Both lines remain continuous in the settled plus-shaped crossing.

Initial state

Terminals:

T1 = (12,7)
T2 = (7,12)
T3 = (1,2)
T4 = (12,1)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(1,7)	S	N→T3, E→T1	E→T1	3
E2	(7,1)	W	E→T4, S→T2	S→T2	3
E3	(4,2)	N	W→T3	W→T3	3
E4	(12,4)	E	N→T4, S→T1	N→T4	3

Other state:

walls = {}
selected_emitter = null
won = false
initial terminal hits = {}

The mandatory final crossing cell (7,7) contains no object.

Exact replay
E3×3:
    N → E → S → W
    T3 lights

E1×3:
    S → W → N → E
    T1 lights

E4×3:
    E → S → W → N
    T4 lights

E2×3:
    W → N → E → S
    its beam crosses E1 at (7,7)
    T2 lights

Total: 12 clicks.

Dependency and necessity check

The winning assignment is forced:

E3 can only reach T3.

Therefore E1 cannot spend its beam on T3 and must take T1.

Therefore E4 cannot take T1 and must take T4.

Therefore E2 cannot take T4 and must take T2.

Thus every valid solution contains the perpendicular crossing at (7,7). Safe crossing cannot be ignored or replaced by a noncrossing solution.

The unique winning state is:

E1=E, E2=S, E3=W, E4=N
Level 5 — Outward Pair
Player-facing discovery

The central row contains two emitters, E4 and E5, with terminals outside them:

T4 — E4 —— E5 ——— T7

E4’s eastward line ends on E5’s body. E5’s westward line ends on E4’s body. They cannot exchange sides or shoot through one another. Their useful configuration is therefore to point outward:

E4→T4
E5→T7

Initially, however, E4 points north and lights T1, while E5 points south and lights T2. These look productive locally. The rest of the board reveals why they are liabilities: other emitters need to inherit T1 and T2 so the central pair can cover the otherwise unreachable outer sockets.

This reuses opacity in a contrasting role. The blocker is no longer merely an obstacle to work around; each blocker is also the only source capable of completing the line beyond itself.

Initial state

Terminals:

T1 = (4,1)
T2 = (7,5)
T3 = (10,3)
T4 = (3,4)
T5 = (1,3)
T6 = (7,12)
T7 = (11,4)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(2,3)	S	E→T3, W→T5	E→T3	3
E2	(10,1)	N	S→T3, W→T1	W→T1	3
E3	(7,10)	E	N→T2, S→T6	N→T2	3
E4	(4,4)	N	N→T1, W→T4	W→T4	3
E5	(7,4)	S	E→T7, S→T2	E→T7	3
E6	(10,12)	N	N→T3, W→T6	W→T6	3
E7	(1,9)	E	N→T5	N→T5	3

Other state:

walls = {}
selected_emitter = null
won = false

Initial terminal hits:

T1 ← E4
T2 ← E5
T3 ← E6

All other terminals are initially dark.

Exact replay
E7×3 → N → T5
E1×3 → E → T3
E2×3 → W → T1
E6×3 → W → T6
E3×3 → N → T2
E4×3 → W → T4
E5×3 → E → T7

Total: 21 clicks.

Why the solution follows from the field

The assignment is forced by a branching dependency:

E7 can only reach T5, so E7 must take T5.

E1 therefore cannot use T5 and must take T3.

With T3 occupied:

E2 must take T1.

E6 must take T6.

With T6 occupied, E3 must take T2.

With T1 occupied, E4 must turn outward to T4.

With T2 occupied, E5 must turn outward to T7.

The two branches created by T3 rejoin at the central pair. The puzzle is not a spatially ordered sequence; E7, E2, E6, and the central pair are spread across the board, and the click order may be rearranged freely.

Anti-degenerate check

Every emitter is required because seven emitters must cover seven terminals.

E4 and E5 are necessary physical blockers, not decorative devices.

Keeping the three initially lit beams unchanged makes completion impossible.

The central handoff rule is used twice in opposite directions.

Removing either E4 or E5 changes both the obstruction graph and terminal reachability.

The unique winning state is:

E1=E
E2=W
E3=N
E4=W
E5=E
E6=W
E7=N
Level 6 — The Blocked Approach
Player-facing discovery

A single wall post at (1,11) sits directly between E2 at (1,10) and T8 at (1,12). E2 initially faces south, so the first frame already shows its beam capped against the wall with T8 visibly empty behind it.

The wall is made from the same gray material as the established board rim. It is not clickable and introduces no hidden state. Its purpose is to make approach direction matter: T8 cannot be reached from the north, but E3 can approach it horizontally from the east.

The rest of the level combines:

two emitter handoffs,

several near-terminal interceptions,

a wall-created approach asymmetry,

three mandatory empty-cell beam crossings.

Initial state

Terminals:

T1  = (1,5)
T2  = (3,10)
T3  = (5,10)
T4  = (3,1)
T5  = (5,3)
T6  = (5,8)
T7  = (8,1)
T8  = (1,12)
T9  = (8,12)
T10 = (12,5)

Walls:

W1 = (1,11)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(8,5)	W	N→T7, E→T10, S→T9	S→T9	3
E2	(1,10)	S	N→T1, E→T2	E→T2	3
E3	(3,12)	N	N→T2, E→T9, W→T8	W→T8	3
E4	(3,8)	E	N→T4, E→T6, S→T2	N→T4	3
E5	(10,3)	N	W→T5	W→T5	3
E6	(5,5)	N	N→T5, S→T6, W→T1	W→T1	3
E7	(12,10)	N	N→T10, W→T3	W→T3	3
E8	(10,1)	N	W→T7	W→T7	3
E9	(12,1)	W	S→T10	S→T10	3
E10	(10,8)	N	W→T6	W→T6	3

Other state:

selected_emitter = null
won = false

Initial terminal hits:

T2  ← E3
T5  ← E6
T6  ← E4
T10 ← E7

Important initial nonterminal endpoints:

E2 south ray → wall W1 at (1,11)
E1 west ray → E6 body
E9 west ray → E8 body
E10 north ray → E5 body
Exact replay

A reasoning-oriented replay is:

E5×3  → W → T5
E8×3  → W → T7
E9×3  → S → T10
E10×3 → W → T6

E6×3  → W → T1
E1×3  → S → T9
E7×3  → W → T3

E2×3  → E → T2
E4×3  → N → T4
E3×3  → W → T8

Total: 30 clicks.

The final mandatory crossings are:

E1 vertical × E10 horizontal at (8,8)
E1 vertical × E7 horizontal  at (8,10)
E4 vertical × E6 horizontal  at (3,5)

All three cells are empty.

Dependency network

Four emitters provide initial anchors:

E5  can only reach T5
E8  can only reach T7
E9  can only reach T10
E10 can only reach T6

Those anchors force several other assignments:

E6 cannot use T5 or T6, so it must take T1.

E1 cannot use T7 or T10, so it must take T9.

E7 cannot use T10, so it must take T3.

E2 cannot use T1 and cannot reach T8 through the wall, so it must take T2.

E4 cannot use T6 or T2, so it must take T4.

E3 cannot use T2 or T9, so it must approach T8 from the east.

This forms multiple branches rather than one ordered chain.

Contrasting mechanic roles

Emitter opacity acts as a handoff in two places:

E1’s west ray is blocked by E6; E6 takes T1 while E1 turns south to T9.

E9’s west ray is blocked by E8; E8 takes T7 while E9 turns south to T10.

Terminal interception creates different conflicts:

E2 facing east reaches T2 before the farther T3.

E6 facing south reaches T6 before T3.

E7 facing west reaches T3 before T2.

E8 facing west reaches T7 before T4.

The same absorption rule is useful in some assignments and restrictive in others.

Wall necessity

Without wall (1,11), there would be a second winning assignment:

E2 → T8
E3 → T2

The wall eliminates that approach and forces:

E2 → T2
E3 → T8

The wall is therefore structurally necessary, not decoration.

The unique winning state is:

E1=S
E2=E
E3=W
E4=N
E5=W
E6=W
E7=W
E8=W
E9=S
E10=W
Level 7 — Borrowed Light
Player-facing discovery

The final level begins with five terminals lit, but every one of them is lit by an emitter that must eventually relinquish it:

T3  ← E5 and E6
T7  ← E11
T8  ← E13
T9  ← E7
T13 ← E9

This sharpens the game’s central lesson. A lit socket is evidence about the current line of sight, not evidence that its current owner should be preserved.

Several emitters have terminals that only they can ultimately cover:

E1 must inherit T8 from E13.

E3 must inherit T3 from E5 and E6.

E8 must inherit T13 from E9.

E14 must inherit T9 from E7.

E7 must inherit T7 from E11.

A wall post at (6,11) directly blocks E3’s initial eastward line toward T2. The initial frame shows E3’s impact on the wall. E3 must instead turn north toward T3, which in turn releases E6 to serve T2.

The eventual westward beam from E9 becomes a horizontal trunk crossing three established vertical beams. The final arrangement therefore uses every major rule at once without introducing another device.

Initial state

Terminals:

T1  = (11,3)
T2  = (7,11)
T3  = (5,5)
T4  = (1,3)
T5  = (1,12)
T6  = (11,12)
T7  = (9,11)
T8  = (12,7)
T9  = (11,9)
T10 = (1,9)
T11 = (1,1)
T12 = (1,7)
T13 = (9,5)
T14 = (3,3)

Walls:

W1 = (6,11)

Emitters:

ID	Cell	Initial	Reachable terminal directions	Winning direction	CW clicks
E1	(12,1)	W	S→T8	S→T8	3
E2	(1,11)	E	N→T10, S→T5	N→T10	3
E3	(5,11)	E	N→T3	N→T3	3
E4	(3,11)	E	N→T14	N→T14	3
E5	(1,5)	E	N→T4, E→T3, S→T12	N→T4	3
E6	(7,5)	W	E→T13, S→T2, W→T3	S→T2	3
E7	(11,11)	N	N→T9, S→T6, W→T7	W→T7	3
E8	(9,1)	W	S→T13	S→T13	3
E9	(9,7)	N	N→T13, S→T7, W→T12	W→T12	3
E10	(5,1)	N	S→T3, W→T11	W→T11	3
E11	(9,12)	N	N→T7, E→T6, W→T5	W→T5	3
E12	(7,3)	S	E→T1, W→T14	E→T1	3
E13	(12,12)	N	N→T8, W→T6	W→T6	3
E14	(11,7)	W	N→T1, E→T8, S→T9	S→T9	3

Other state:

selected_emitter = null
won = false

Initial terminal hits:

T3  ← E5, E6
T7  ← E11
T8  ← E13
T9  ← E7
T13 ← E9

Important initial endpoint:

E3 east ray → wall W1 at (6,11)
Exact replay

This order makes the ownership transfers legible:

E1×3  → S → T8
E3×3  → N → T3
E4×3  → N → T14
E8×3  → S → T13

E6×3  → S → T2
E5×3  → N → T4
E10×3 → W → T11
E12×3 → E → T1
E13×3 → W → T6
E14×3 → S → T9

E7×3  → W → T7
E11×3 → W → T5
E2×3  → N → T10

E9×3  → W → T12

Total: 42 clicks.

Rotating E9 last makes the final composition especially readable. Its westward beam crosses three already-established vertical beams before reaching T12.

Mandatory final crossings
E3 vertical × E9 horizontal at (5,7)
E4 vertical × E9 horizontal at (3,7)
E6 vertical × E9 horizontal at (7,7)
E8 vertical × E12 horizontal at (9,3)

All four crossing cells are empty and remain uninterrupted.

Handoff structures

Three emitter-body handoffs operate in different regions:

Top handoff

E8’s westward line is blocked by E10.

E10 takes T11 to the west.

E8 turns south to T13.

Central handoff

E12’s southward line is blocked by E6.

E6 continues south to T2.

E12 turns east to T1.

Right handoff

E14’s westward line is blocked by E9.

E9 continues west to T12.

E14 turns south to T9.

The operational rule is identical in all three cases; only geometry and terminal ownership differ.

Terminal interception structures

Several apparent long routes are cut short by nearer terminals:

E2 north reaches T10 before T12.

E5 north reaches T4 before T11.

E5 south reaches T12 before T10.

E7 west reaches T7 before T2.

E12 west reaches T14 before T4.

These are not exceptions or terminal-specific behaviors. Every terminal always absorbs the first arriving beam.

Why the initial lights must be broken

The forced ownership transfers are:

T8:  E13 → E1
T3:  E5/E6 → E3
T13: E9 → E8
T9:  E7 → E14
T7:  E11 → E7

Each old owner has another terminal that would otherwise be uncovered:

E13 must take T6
E6  must take T2
E5  must take T4
E9  must take T12
E7  must take T7
E11 must take T5

Thus the initially lit state is not random misdirection. Every visible wrong beam represents a genuine locally appealing assignment whose global consequence the player has already learned how to diagnose.

Unique-solution reasoning

The wall and singleton reachability first establish:

E1  → T8
E3  → T3
E4  → T14
E8  → T13

Then:

E10 cannot use T3, so E10→T11.

E6 cannot use T13 or T3, so E6→T2.

E12 cannot use T14, so E12→T1.

E13 cannot use T8, so E13→T6.

T4 has no remaining source except E5, so E5→T4.

E14 cannot use T1 or T8, so E14→T9.

E7 cannot use T6 or T9, so E7→T7.

E11 cannot use T6 or T7, so E11→T5.

E2 cannot use T5, so E2→T10.

E9 cannot use T13 or T7, so E9→T12.

The unique winning state is:

E1=S
E2=N
E3=N
E4=N
E5=N
E6=S
E7=W
E8=S
E9=W
E10=W
E11=W
E12=E
E13=W
E14=S
Wall necessity and anti-degenerate check

Without wall (6,11), E3 could point east to T2. The level would then admit a second winning assignment that swaps the responsibilities of E3 and E6:

E3→T2, E6→T3

The wall blocks exactly that line and forces the intended ownership:

E3→T3, E6→T2

The level cannot be reduced to a corridor or ordered checklist:

Dependencies span the left, center, top, and right sides.

The initial lit state must be dismantled rather than preserved.

Three handoffs point in different directions.

Five terminal-interception relationships constrain assignment.

Four safe crossings are required.

Every emitter and terminal participates in the unique solution.

The wall changes the actual solution graph rather than merely lengthening a route.

6. Implementation assertions

A conforming implementation should satisfy the following tests.

Ray and collision tests

Four valid clicks on an emitter restore its original orientation and ray.

An emitter blocks an incoming beam in all four of its orientations.

An emitter may block an incoming beam while projecting its own beam elsewhere.

A terminal always stops a beam, whether lit or unlit.

A farther collinear terminal is not hit after a nearer terminal.

Perpendicular beams sharing an empty cell both continue.

A beam crossing is not inserted into the obstacle map.

A wall blocks a beam before any object behind it.

A beam stops at the board rim when no earlier collision exists.

Interaction tests

Only an emitter cell changes orientation.

Clicking any pixel in the emitter’s 4×4 cell activates it.

Clicking a beam segment does not activate its source.

Clicking a terminal or wall does not alter state.

UP, DOWN, LEFT, RIGHT, and SPACE do not alter state.

No persistent selection exists between steps.

Animation frames do not alter orientation, ray, or terminal state after the single step() update.

Validator tests

A level does not complete while any terminal is unlit.

Duplicate hits on one terminal are allowed but do not compensate for another dark terminal.

Extra beams ending at walls, emitters, or the rim do not independently prevent completion.

Completion occurs immediately after the action that lights the final terminal.

The seven authored levels have shortest winning click counts:

Level 1:  1
Level 2:  6
Level 3:  6
Level 4: 12
Level 5: 21
Level 6: 30
Level 7: 42

Each authored level has exactly one winning orientation vector.

The game contains exactly 7 levels.
