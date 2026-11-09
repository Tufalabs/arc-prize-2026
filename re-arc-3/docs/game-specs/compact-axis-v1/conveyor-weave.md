Worked for 75m 7s
Conveyor Weave

“This is a game about conveyor travel becoming difficult because the route can be configured only from the shores the player can still reach.”

Authored campaign: exactly 7 levels.

Conveyor Weave is a top-down routing puzzle in which ordinary floor is used for preparation and conveyors are used for committed travel. The player can rotate only a neighboring conveyor and only while standing on floor. A journey therefore changes not just the player’s position, but which sections of the route can still be configured.

The game introduces only three persistent object families:

Fixed and rotatable conveyors.

Ordinary floor, including the exit pad.

One-way gates between adjacent cells.

Jams are a visible state of an existing conveyor, not a separate device. No new object type appears after Level 4.

1. Player-facing rules

The player controls a yellow agent with UP, DOWN, LEFT, RIGHT, and SPACE.

Directional actions move the agent one logical cell when movement is possible and always turn its visible facing indicator, even when movement is blocked. SPACE rotates the faced adjacent rotatable conveyor clockwise, but only while the agent is standing on ordinary floor.

Entering a conveyor commits the agent to its visible arrows. Automatic travel continues until the agent:

reaches ordinary floor or the exit;

reaches an impassable wall;

is rejected by a one-way gate;

encounters a conveyor pointing directly back toward the current conveyor;

or would enter an automatic loop.

A blockage leaves the agent visibly jammed on the last conveyor. From a jam, the player may step only onto an adjacent ordinary-floor or exit cell. Every reachable jam state in the seven authored levels has at least one legal local escape.

Rotations persist until changed again. Late puzzles exploit the fact that a conveyor may need one orientation for an outward journey and another for the return.

Clicking is not used. No cell looks clickable, no cursor or selection is rendered, and a click on any cell leaves the state unchanged.

2. Formal implementation contract
2.1 Logical space

The native board is a 12×12 orthogonal lattice.

Coordinates use (row, column):

rows increase downward;

columns increase rightward;

valid coordinates range from (1,1) through (12,12);

there is no diagonal movement or interaction.

Each logical cell renders as a 5×5-pixel region. The 60×60 board begins at rendered pixel (2,2), leaving a two-pixel margin on the 64×64 canvas.

For logical coordinate (r,c), its rendered top-left pixel is:

x = 2 + 5 × (c - 1)
y = 2 + 5 × (r - 1)

Any coordinate not listed as floor, exit, or conveyor is solid void.

Adjacent traversable cells normally share an open seam. A level may override a seam with either:

a solid wall seam, blocking both directions; or

a one-way gate, allowing exactly one listed direction.

There are no decorative grid lines or borders around every logical cell. Only authored wall and gate seams are drawn.

2.2 Static level record
LevelSpec:
    name: string
    floor: set<Coord>
    exit: Coord
    start_position: Coord
    start_facing: U | R | D | L
    fixed_conveyors: map<Coord, Direction>
    rotatable_conveyors: map<Coord, InitialDirection>
    allowed_gate_edges: set<(source Coord, target Coord)>
    solid_wall_seams: set<unordered pair<Coord>>

A gate entry (A,B) means movement from A to B is allowed and movement from B to A is blocked.

A seam cannot simultaneously be a gate and a solid wall.

2.3 Dynamic state
GameState:
    agent_position: Coord
    agent_facing: U | R | D | L
    agent_jammed: bool
    jammed_edge: optional<(source Coord, target Coord)>
    rotatable_directions: map<Coord, Direction>
    complete: bool
    selected_object: null
    cursor_visible: false

The complete settled gameplay state is visible:

agent position is visible;

agent facing is visible;

every conveyor orientation is visible;

a settled agent on a conveyor is necessarily jammed;

the blocked edge is visibly marked;

gates and walls are static and visible.

There is no hidden route progress, phase, inventory, history variable, counter, timer, or random state.

2.4 Action semantics
Directional action while on floor

For action d ∈ {UP, RIGHT, DOWN, LEFT}:

Set agent_facing = d.

Find the orthogonally adjacent target cell.

If the target is void, separated by a solid wall, or rejected by a one-way gate:

keep the current position;

show a brief bump or gate flash;

retain the new facing.

If the target is ordinary floor or the exit:

move one logical cell;

settle immediately.

If the target is a conveyor:

move onto it;

begin automatic conveyor resolution.

The direct movement from floor onto the first conveyor does not perform the head-on-conveyor test, because the source is not a conveyor. Thus, a conveyor pointing back toward the floor can visibly carry the agent back to the same floor in one action.

SPACE while on floor

Inspect the cell one logical unit in agent_facing.

The target is eligible only if it is a rotatable conveyor.

A one-way gate between the agent and the conveyor does not prevent rotation: it blocks traversal, not adjacency.

A solid wall seam does prevent rotation.

If eligible, rotate the conveyor clockwise:

UP → RIGHT → DOWN → LEFT → UP

The agent does not move and does not change facing.

An ineligible SPACE action leaves gameplay state unchanged and briefly displays a small red padlock pictogram at the faced edge.

Directional action while jammed

A jammed agent may move only to an adjacent ordinary-floor or exit cell through a legally traversable seam.

The directional action always changes facing.

An adjacent conveyor cannot be manually entered from a jam.

SPACE is ineligible while jammed.

A failed escape leaves the agent jammed on the same conveyor.

Clicking

CLICK(r,c) is accepted for any logical cell but is an inert action:

next_state = current_state

It consumes one recorded step() call if the surrounding harness counts calls, but there is no gameplay budget and no feedback suggesting that clicking matters.

2.5 Automatic conveyor resolution

After entering a conveyor:

Read its current arrow direction.

Set the agent’s facing to that direction.

Compute the adjacent target cell.

Before crossing the seam, test in this order:

target is traversable;

no solid wall blocks the seam;

any gate permits the attempted direction;

if the target is another conveyor, its arrow does not point directly back toward the current conveyor;

the target conveyor has not already been visited during this same automatic resolution.

If any test fails:

remain on the current conveyor;

set agent_jammed = true;

set jammed_edge to the rejected edge;

end the action.

Otherwise move to the target.

If the target is floor or exit, settle and end the action.

If the target is another conveyor, repeat.

The repeated-conveyor test makes the transition function total even for an accidental loop. The attempted repeated edge becomes a normal visible jam.

2.6 Update order

Each input causes exactly one call to step():

1. Ignore input if complete.
2. Decode click, movement, or SPACE.
3. Update facing when the action is directional.
4. Apply a legal rotation or manual movement.
5. Resolve all forced conveyor travel.
6. Commit final position, facing, jam state, and orientations.
7. Validate completion.
8. Emit the deterministic render-only animation trace.

Animation frames do not invoke step(), change gameplay state, or consume actions.

2.7 Success condition

The level succeeds exactly when the settled agent_position equals the level’s exit coordinate.

Nothing else is checked:

conveyor orientations may be arbitrary;

unused conveyors need not be restored;

prior jams do not matter;

the agent does not need a particular facing;

recovery corridors need not be visited;

no other floor cell needs to be occupied or activated.

The exit is ordinary floor for movement purposes, so automatic travel stops there. Success is validated only after the complete automatic journey settles.

There is no loss condition, energy system, move budget, or gameplay timer. Exploration is unlimited.

3. Visual specification
3.1 Palette
Role	Color ID
Canvas and void	5
Solid structural seams	4
Ordinary floor	1
Fixed conveyor base	9
Fixed conveyor secondary detail	10
Rotatable conveyor base	15
Rotatable circular rim	7
Conveyor arrows and exit inset	0
Exit base	14
Agent body	11
Agent facing nose and gate chevrons	12
One-way gate barrier	13
Jam and invalid-action feedback	8
3.2 Tile silhouettes

All local tile coordinates below use (y,x) from 0 through 4.

Conveyor arrow

The canonical UP arrow uses color 0 at:

(0,2)
(1,1), (1,2), (1,3)
(2,2)
(3,2)
(4,2)

Rotate this mask for the other three directions.

A fixed conveyor has a color-9 base. Two rear-edge pixels may use color 10 to reinforce its direction, but they must rotate with the arrow and must not resemble a separate state.

A rotatable conveyor has a color-15 base and a color-7 rim at:

(0,0), (0,1),        (0,3), (0,4)
(1,0),                      (1,4)

(3,0),                      (3,4)
(4,0), (4,1),        (4,3), (4,4)

The rim is identical in every orientation. Only the white arrow rotates.

Agent

The agent is a connected hollow 3×3 ring of color 11:

(1,1), (1,2), (1,3)
(2,1),        (2,3)
(3,1), (3,2), (3,3)

A color-12 nose is added at:

UP:    (0,2)
RIGHT: (2,4)
DOWN:  (4,2)
LEFT:  (2,0)

The center remains open so the underlying conveyor arrow remains readable.

Exit

The exit fills the cell with color 14 and contains a color-0 hollow 3×3 inset. It does not contain an arrow and is visibly distinct from ordinary floor before the agent reaches it.

Gates

A gate is a two-pixel-thick color-13 barrier centered on the shared seam. A color-12 chevron is drawn on the allowed source side, pointing across the seam toward the allowed target.

No symbol is drawn on the rejected side.

Jam state

The rejected edge flashes color 8, then retains a two-pixel red jam tick while the agent occupies the conveyor. The tick is drawn after the agent so that its location remains clear.

3.3 Animation

Ordinary floor movement and facing changes are immediate.

A conveyor journey receives a deterministic render trace:

one departure frame with the agent nose stretched toward the next cell;

one arrival frame centered on the next cell;

repeat for every automatically crossed seam;

render either the final floor state or the jam flash.

A rotation uses three render frames:

old arrow;

a compact cross-shaped transition mark;

new arrow.

The new orientation is already authoritative after step(); the animation is explanatory only.

Blocked movement uses a two-frame bump. A rejected gate briefly brightens its maroon seam. Completion uses a four-frame green/white pulse and then leaves the final settled frame visible.

Player input is locked until the explanatory frames finish. No animation frame consumes an action.

4. Authored progression
Level	Main discovery or composition	Verified shortest inputs
1. Ride the Arrow	Entering a conveyor causes automatic travel to the visible exit	1
2. Turn Before Boarding	SPACE rotates a faced purple conveyor clockwise	5
3. The Useful Jam	A blocked journey settles visibly and can be escaped onto nearby floor	6
4. Wrong-Side Gate	A route must approach a gate from its chevron side	22
5. Two Shores	Jams and return belts expose different remote configuration shores	40
6. Borrowed Return Belt	One conveyor must first point outward and later point home	42
7. Conveyor Weave	Two nested outward/return belts must preserve settings used by a final route	50

Level 4 deliberately stops at 22 inputs rather than adding an empty approach corridor merely to reach 25. Its gate composition is complete at that point.

In the solution notation below:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

For all levels initially:

agent_jammed = false
jammed_edge = null
complete = false
selected_object = null
cursor_visible = false
5. Authored levels
Level 1 — Ride the Arrow
Player-facing discovery

The yellow agent begins immediately left of a three-cell blue conveyor pointing toward a green exit. There are no other objects.

Pressing RIGHT moves the agent onto the conveyor. The agent is then visibly carried across all three cells and stops on the exit. The completion pulse supplies direct positive evidence for both the travel rule and the objective.

No visual tutorial panel or symbolic reference is needed: the physical direction of travel and the green destination are sufficient.

State-machine instance
Start:
    position = (4,2)
    facing = RIGHT

Ordinary floor:
    (4,2)

Exit:
    (4,6)

Fixed conveyors:
    (4,3): RIGHT
    (4,4): RIGHT
    (4,5): RIGHT

Rotatable conveyors:
    none

Gates:
    none

Solid wall seams:
    none

Every unlisted coordinate is void.

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
R

Automatic trace:

(4,3) → (4,4) → (4,5) → exit (4,6)

Intended depth: 1 player input.

Attribution and necessity check

The only productive direction enters the visibly aligned belt. The action produces one uninterrupted, attributable consequence and immediate success. There are no decoys, alternate mechanisms, or simultaneous state changes.

Level 2 — Turn Before Boarding
Player-facing discovery

A purple-rimmed conveyor separates the western landing from the exit route. It initially points left.

Entering it unchanged visibly carries the agent back to the same western landing. The upper and lower floor pockets ensure that testing the UP or DOWN orientations also ends safely on visible floor rather than introducing jams prematurely.

SPACE immediately rotates the visible arrow. Two presses change it:

LEFT → UP → RIGHT

The rightward setting connects to the fixed conveyor and exit.

This isolates one new causal fact: the faced purple conveyor rotates clockwise and retains its new orientation.

State-machine instance
Start:
    position = (6,2)
    facing = UP

Ordinary floor:
    western approach:
        (6,2), (5,2), (5,3)

    upper safe pocket:
        (4,3), (4,4)

    lower safe pocket:
        (6,3), (6,4)

Exit:
    (5,6)

Rotatable conveyors:
    T = (5,4), initial LEFT

Fixed conveyors:
    (5,5): RIGHT

Gates:
    none

Solid wall seams:
    none

The four possible directions of T have immediately legible outcomes:

LEFT  → western landing (5,3)
UP    → upper pocket (4,4)
RIGHT → fixed conveyor (5,5), then exit
DOWN  → lower pocket (6,4)

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
U R S S R

Resolution:

U R reaches floor (5,3) while facing T.

S S rotates T from left to up to right.

R enters T, then (5,5), then the exit.

Intended depth: 5 player inputs.

Attribution and necessity check

No jam state is reachable in this level. Every visible floor cell makes an orientation experiment reversible. The exit cannot be reached without changing T, and disabling rotation of T makes the level unsolvable.

Level 3 — The Useful Jam
Player-facing discovery

The now-familiar rightward conveyor route visibly ends at a wall. Entering it leaves the agent settled on its final blue cell with a red jam mark on the blocked eastern edge.

A single ordinary-floor cell lies directly above the jam. Pressing UP moves the agent onto it. From that shore, the player faces a familiar rotatable conveyor and can prepare a second journey to the exit.

This teaches exactly what is new:

a blocked conveyor journey does not rewind;

the jammed endpoint remains the real current position;

nearby ordinary floor is a legal escape;

the resulting shore may provide a useful new configuration position.

State-machine instance
Start:
    position = (6,2)
    facing = UP

Ordinary floor:
    (6,2)
    (5,2)
    (5,3)
    jam shore P = (4,6)

Exit:
    (1,6)

Fixed conveyors:
    lower route:
        (5,4): RIGHT
        (5,5): RIGHT
        (5,6): RIGHT
        east of (5,6) is void

    upper route:
        (2,6): UP

Rotatable conveyors:
    T = (3,6), initial LEFT

Gates:
    none

Solid wall seams:
    none

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
U R R U S U

Resolution:

U R reaches (5,3).

R automatically travels through (5,4), (5,5), and (5,6), then jams at the eastern wall.

U escapes to shore (4,6).

S rotates T from left to up.

U enters T, continues through (2,6), and reaches the exit.

Intended depth: 6 player inputs.

Attribution and necessity check

The lower route cannot reach the exit and the upper route cannot be entered from the starting region. The jam is therefore necessary rather than cosmetic. The only new effect during the impasse is the visible stopped state; the rotation rule was already established in Level 2.

Level 4 — Wrong-Side Gate
Player-facing discovery

The obvious route enters rotatable conveyor A, but A cannot cross into neighboring conveyor B. The seam flashes and A retains a red jam mark. The gate chevron visibly points from B toward A, demonstrating that the desired final crossing must happen from the far side.

A second gate below A points from A toward floor shore P. The player can escape downward. Attempting to move back upward is rejected, but visibly turns the agent toward A; SPACE can then rotate A across the gate.

A separate lower launcher C reaches the far side of B. From there, B can be turned west. The final journey crosses the main gate in its permitted direction and then follows A north to the exit.

The same gate rule plays three contextual roles:

the main gate is an initial obstruction;

the service gate is a jam escape;

the service gate later prevents the shore from becoming a shortcut onto A.

State-machine instance
Start:
    position = (5,2)
    facing = RIGHT

Ordinary floor:
    start landing:
        (5,2)

    service shore:
        P = (6,6)
        (7,6)
        (8,6)
        (8,5)

    far landing:
        (8,10)
        (7,10)
        (6,10)
        (5,10)
        (5,9)
        (5,8)

Exit:
    (2,6)

Rotatable conveyors:
    A = (5,6), initial RIGHT
    B = (5,7), initial UP
    C = (8,7), initial LEFT

Fixed conveyors:
    start route:
        (5,3): RIGHT
        (5,4): RIGHT
        (5,5): RIGHT
        then A

    route from A to exit:
        (4,6): UP
        (3,6): UP
        then exit (2,6)

    route from C to far landing:
        (8,8): RIGHT
        (8,9): RIGHT
        then floor (8,10)

Allowed gate edges:
    B (5,7) → A (5,6)
    A (5,6) → P (6,6)

Solid wall seams:
    none

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
R D U S S S D D L R S S R U U U L L S S S L

Phases:

R

The start route jams on A at the wrong side of the main gate.

D U S S S

Escape to P, use the rejected upward move to face A, and rotate A:

RIGHT → DOWN → LEFT → UP
D D L R S S R

Reach the western side of C, rotate it:

LEFT → UP → RIGHT

and launch to (8,10).

U U U L L S S S L

Approach B from the east, rotate it:

UP → RIGHT → DOWN → LEFT

and enter it from (5,8). The automatic route is:

B west → A up → (4,6) → (3,6) → exit

Intended depth: 22 player inputs.

Final relevant orientations:

A = UP
B = LEFT
C = RIGHT
Anti-degenerate check

Every rotatable is required. The main gate cannot be bypassed; the final crossing must use its indicated direction. The lower launcher is not merely an errand destination: it changes the approach side from which B can be configured and entered.

The rejected UP action on shore P is meaningful rather than a no-op: it visibly aims the agent at A, flashes the service gate, and enables the immediately following rotation.

Level 5 — Two Shores
Player-facing discovery

The direct main route initially fails before rotatable conveyor A, which points back toward the incoming belt. A separate upper launcher leads to an intentional jam above the western workshop.

From the western workshop the player can:

turn A rightward so the main route can pass it;

configure return launcher R1;

ride R1 back to the upper starting branch.

The now-usable main route reaches B, whose downward orientation creates a jam that opens the eastern workshop. There the player configures B for the exit and uses a different return launcher, R2, to get home.

Thus a jam has a different strategic role than in Level 3. It is no longer merely recovery from an obvious mistake; it is the authored entrance to a remote configuration shore.

State-machine instance
Start:
    position = (6,2)
    facing = LEFT

Ordinary floor:
    start and upper branch:
        (6,2), (6,1), (5,1), (4,1), (3,1), (3,2)

    western workshop W1:
        (5,5), (4,5), (4,6)

    eastern workshop W2:
        (5,9), (4,9), (4,10)

    lower recovery lane:
        (7,2), (7,4), (7,8)
        (8,2), (8,3), (8,4), (8,5),
        (8,6), (8,7), (8,8)

Exit:
    (6,11)

Rotatable conveyors:
    U  = (3,3), initial DOWN
    A  = (6,5), initial LEFT
    R1 = (4,7), initial RIGHT
    B  = (6,9), initial DOWN
    R2 = (4,11), initial LEFT

Fixed conveyors:
    upper probe:
        U
        (3,4): RIGHT
        (3,5): UP
        north of (3,5) is void

    R1 return highway:
        R1
        (3,7): UP
        (2,7): UP
        (1,7): LEFT
        (1,6): LEFT
        (1,5): LEFT
        (1,4): LEFT
        (1,3): LEFT
        (1,2): DOWN
        (2,2): DOWN
        then floor (3,2)

    R2 branch:
        R2
        (3,11): UP
        (2,11): UP
        (1,11): LEFT
        (1,10): LEFT
        (1,9): LEFT
        (1,8): LEFT
        then merge at fixed conveyor (1,7)

    main route before A:
        (6,3): RIGHT
        (6,4): RIGHT
        then A

    main route between A and B:
        (6,6): RIGHT
        (6,7): RIGHT
        (6,8): RIGHT
        then B

    main route after B:
        (6,10): RIGHT
        then exit (6,11)

Allowed gate edges:
    A (6,5) → W1 (5,5)
    B (6,9) → W2 (5,9)
    (6,4) → recovery floor (7,4)
    (6,8) → recovery floor (7,8)

Solid wall seams:
    none

The recovery gates permit escape from a head-on conflict on (6,4) or (6,8) while preventing the lower lane from boarding the main conveyor in reverse.

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
L U U U R
S S S R
D D
S S
U R
S S S R
L D D D R
R
U D
S S S
U R
S R
L D D D R
R

Unbroken form:

LUUURSSSRDDSSURSSSRLDDDRRUDSSSURSRLDDDRR

Phases:

Reach the upper launcher and turn U right.

Ride to the jam at (3,5).

Escape downward into W1.

Turn A from left to right.

Turn R1 from right to up and ride home.

Return to the start landing.

Ride the main route through A; B points down and jams.

Escape upward into W2.

The rejected DOWN input aims at B; turn it from down to right.

Turn R2 from left to up and ride home.

Return to the start and take the completed main route to the exit.

Intended depth: 40 player inputs.

Final relevant orientations:

U  = RIGHT
A  = RIGHT
R1 = UP
B  = RIGHT
R2 = UP
Necessity and anti-cleanliness check

Disabling rotation of any one of the five rotatables makes the level unsolvable.

The dependencies are not five independent settings:

A must be configured before W2 can be reached.

R1 is needed because configuring A leaves the player away from the main route’s entrance.

B acts as an intentional jam before it acts as part of the exit route.

R2 returns the player without undoing either A or B.

The lower lane exists specifically to recover from setting A or B toward an incoming conveyor; its gates prevent it from becoming a shortcut.

The same conveyor and gate rules therefore serve as obstruction, access method, return route, and preservation constraint.

Level 6 — Borrowed Return Belt
Player-facing discovery

Rotatable conveyor Q begins pointing left toward the start. It must first be changed to point right, carrying the agent to a far landing.

From the far landing, the player can configure conveyor B, but cannot board it because its gate points from B toward the landing. A blocked DOWN action aims at it for rotation.

A westward probe from the far landing creates an intentional jam beside a small workshop. The workshop is the only place from which Q can be changed back to the left. This establishes the central scheduling consequence:

Q = RIGHT to leave
Q = LEFT to return

The later launcher crosses B and reaches a jam at C. That jam exposes another workshop containing both the control position for C and a return launcher. The player returns home, then reuses the already configured final route.

State-machine instance
Start:
    position = (6,3)
    facing = RIGHT

Ordinary floor:
    start and final-launcher path:
        (6,3), (6,2), (5,2), (4,2),
        (3,2), (3,3), (3,4)

    far landing F:
        (6,6), (6,7), (6,8), (6,9)

    Q workshop:
        (5,4), (4,4)

    early-final recovery:
        (7,11)

    C workshop:
        (8,10), (9,10)

Exit:
    (10,7)

Rotatable conveyors:
    Q    = (6,4), initial LEFT
    L    = (3,5), initial DOWN
    B    = (7,9), initial RIGHT
    C    = (8,9), initial DOWN
    Rret = (10,10), initial LEFT

Fixed conveyors:
    Q rightward route:
        (6,5): RIGHT
        then F at (6,6)

    Q downward recovery route:
        (7,4): RIGHT
        (7,5): RIGHT
        (7,6): UP
        then F at (6,6)

    far-landing return probe:
        from floor (6,7), enter
        (5,7): LEFT
        (5,6): LEFT
        (5,5): UP
        north of (5,5) is void

    final launcher route:
        L
        (3,6): RIGHT
        (3,7): RIGHT
        (3,8): DOWN
        (4,8): DOWN
        (5,8): RIGHT
        (5,9): RIGHT
        (5,10): DOWN
        (6,10): DOWN
        (7,10): LEFT
        then B

    route after B and C:
        B pointing DOWN enters C
        C pointing LEFT enters
        (8,8): LEFT
        (8,7): DOWN
        (9,7): DOWN
        then exit (10,7)

    early-final recovery highway:
        from recovery floor (7,11), enter
        (6,11): UP
        (5,11): UP
        (4,11): UP
        (3,11): UP
        (2,11): LEFT
        (2,10): LEFT
        (2,9): LEFT
        (2,8): LEFT
        (2,7): LEFT
        (2,6): LEFT
        (2,5): LEFT
        (2,4): LEFT
        (2,3): DOWN
        then floor (3,3)

    return launcher route:
        Rret pointing DOWN enters
        (11,10): LEFT
        (11,9): LEFT
        (11,8): LEFT
        (11,7): LEFT
        (11,6): LEFT
        (11,5): LEFT
        (11,4): LEFT
        (11,3): UP
        (10,3): UP
        (9,3): UP
        (8,3): UP
        (7,3): UP
        then start (6,3)

Allowed gate edges:
    Q workshop (5,4) → Q (6,4)
    B (7,9) → far landing (6,9)
    final predecessor (7,10) → recovery floor (7,11)
    C (8,9) → C workshop (8,10)

Solid wall seams:
    (3,4) — (4,4)

    (5,8) — (6,8)
    (5,9) — (6,9)
    (6,9) — (6,10)

    (7,10) — (8,10)

The walls keep the far landing and the C workshop from manually boarding pieces of the final route.

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
S S R
R R R D S
L L U
L U D S S D
L U U U R R
S S S R
R L S
D S S S D
L U U U R R R

Unbroken form:

SSRRRRDSLLULUDSSDLUUURRSSSRRLSDSSSDLUUURRR

Phases:

Turn Q from left to right and cross to F.

Walk to (6,9), use the rejected downward input to face B, and rotate B from right to down.

Enter the return probe from (6,7) and jam at (5,5).

Escape left, use the two-cell workshop to face Q, turn Q from right to left, and ride it home.

Reach launcher L, turn it from down to right, and launch.

The route passes through B, then jams at downward-pointing C.

Escape right. A rejected left input aims at C; rotate C from down to left.

Move below Rret, turn it from left to down, and ride home.

Return to launcher L and take the completed route through B and C to the exit.

Intended depth: 42 player inputs.

Final relevant orientations:

Q    = LEFT
L    = RIGHT
B    = DOWN
C    = LEFT
Rret = DOWN
Necessity and anti-cleanliness check

Every rotatable is necessary.

The puzzle is not solved by simply setting all devices and walking forward:

Q must be right before the far landing is reachable.

The same Q must later be left or the player cannot return.

B is configured early but used only after another complete journey.

C first creates access to its own workshop, then becomes a segment of the final route.

Rret returns the player without altering Q, B, C, or L.

If the final route is attempted before B has the correct orientation, the agent jams at (7,10) and can escape through the visible gate to (7,11). The recovery highway returns to the launcher shore. This corridor exists for one concrete mistaken plan and cannot be used to bypass the far landing.

Level 7 — Conveyor Weave
Player-facing discovery

The final level nests two outward/return schedules.

Outer conveyor Q must point right to reach eastern landing E, where B can be configured for the eventual final route. A return probe from E jams beside the outer workshop.

From that workshop, inner conveyor T is turned upward to reach upper landing D. At D, the player must configure:

Tin, which returns to the inner side of T;

C, which will later redirect the final route toward the exit.

Tin returns the player to an inner shore. There, T must be changed from up to down so it can deliver the player back to the outer workshop. Q is then changed from right to left to return to the starting region.

Only after both shared conveyors have served their outward and return roles can the lower final launcher be used. Its route depends on the orientations of B and C, which were configured much earlier from two different remote regions.

The dependency structure is:

Q outward
    → configure B
    → outer return jam
        → T outward
            → configure Tin and C
            → inner return
        → T downward
    → Q homeward
→ final launcher
    → use preserved B and C settings
    → exit
State-machine instance
Start:
    position = (7,3)
    facing = RIGHT

Ordinary floor:
    upper landing D:
        (2,5), (2,6), (2,7), (2,8),
        (2,9), (2,10), (2,11)

    inner workshop:
        (4,3)

    outer workshop:
        (5,4), (6,4)

    eastern landing E:
        (7,6), (7,7), (7,8), (7,9), (7,10)
        (8,9), (8,10)
        (9,9), (9,10)

    start and lower-launcher approach:
        (7,2), (7,3)
        (8,2), (9,2)
        (10,2), (10,3)

    right-side recovery corridor:
        (4,12), (5,12), (6,12), (7,12),
        (8,12), (9,12), (10,12)

    bottom recovery corridor:
        (11,3), (11,4), (11,5), (11,6),
        (11,7), (11,8), (11,9), (11,10),
        (11,11), (11,12)

Exit:
    (4,6)

Rotatable conveyors:
    Q   = (7,4), initial LEFT
    T   = (4,4), initial LEFT
    Tin = (2,12), initial RIGHT
    C   = (3,11), initial DOWN
    B   = (9,11), initial DOWN
    Lf  = (10,4), initial DOWN

Fixed conveyors:
    Q rightward route:
        (7,5): RIGHT
        then E at (7,6)

    Q downward recovery route:
        (8,4): RIGHT
        (8,5): RIGHT
        (8,6): UP
        then E at (7,6)

    outer return probe:
        from E floor (7,7), enter
        (6,7): LEFT
        (6,6): LEFT
        (6,5): UP
        north of (6,5) is void

    T upward route:
        T pointing UP enters
        (3,4): RIGHT
        (3,5): UP
        then D at (2,5)

    T rightward recovery route:
        T pointing RIGHT enters
        (4,5): UP
        (3,5): UP
        then D at (2,5)

    Tin return route:
        Tin pointing UP enters
        (1,12): LEFT
        (1,11): LEFT
        (1,10): LEFT
        (1,9): LEFT
        (1,8): LEFT
        (1,7): LEFT
        (1,6): LEFT
        (1,5): LEFT
        (1,4): LEFT
        (1,3): LEFT
        (1,2): DOWN
        (2,2): DOWN
        (3,2): DOWN
        (4,2): LEFT
        west of (4,2) is void

    final launcher route:
        Lf pointing RIGHT enters
        (10,5): RIGHT
        (10,6): RIGHT
        (10,7): RIGHT
        (10,8): RIGHT
        (10,9): RIGHT
        (10,10): RIGHT
        (10,11): UP
        then B

    final route after B:
        B pointing UP enters
        (8,11): UP
        (7,11): UP
        (6,11): UP
        (5,11): UP
        (4,11): UP
        then C

    final route after C:
        C pointing LEFT enters
        (3,10): LEFT
        (3,9): LEFT
        (3,8): LEFT
        (3,7): LEFT
        (3,6): DOWN
        then exit (4,6)

Allowed gate edges:
    outer workshop (6,4) → Q (7,4)
    inner workshop (4,3) → T (4,4)

    C (3,11) → upper landing (2,11)
    B (9,11) → E floor (9,10)

    final predecessor (4,11) → recovery floor (4,12)

    final lower conveyors → bottom recovery:
        (10,4)  → (11,4)
        (10,5)  → (11,5)
        (10,6)  → (11,6)
        (10,7)  → (11,7)
        (10,8)  → (11,8)
        (10,9)  → (11,9)
        (10,10) → (11,10)
        (10,11) → (11,11)

Solid wall seams:
    separate upper landing D from the Tin return highway:
        (1,5)—(2,5)
        (1,6)—(2,6)
        (1,7)—(2,7)
        (1,8)—(2,8)
        (1,9)—(2,9)
        (1,10)—(2,10)
        (1,11)—(2,11)

    separate D from the final C route:
        (2,6)—(3,6)
        (2,7)—(3,7)
        (2,8)—(3,8)
        (2,9)—(3,9)
        (2,10)—(3,10)

    separate the right recovery corridor from the final route:
        (5,11)—(5,12)
        (6,11)—(6,12)
        (7,11)—(7,12)
        (8,11)—(8,12)
        (9,11)—(9,12)
        (10,11)—(10,12)

    separate E from the final route:
        (7,10)—(7,11)
        (8,10)—(8,11)
        (9,9)—(10,9)
        (9,10)—(10,10)

Selection/cursor state is null and hidden; clicks are inert.

Verified solution
S S R
R R R D D R
S S
U U L L L
U L
U S U
R R R R R R
S S S
D S
R
R
S S R
D S S D
L D D D R
S S S R

Unbroken form:

SSRRRRDDRSSUULLLULUSURRRRRRSSSDSRRSSRDSSDLDDDRSSSR

Phases:

Turn Q from left to right and cross to E.

Reach (9,10) and turn B:

DOWN → LEFT → UP

Return to (7,7) and enter the outer return probe.

Jam at (6,5), escape left to the outer workshop, and turn T from left to up.

Ride T to upper landing D.

Cross D to (2,11).

Turn Tin:

RIGHT → DOWN → LEFT → UP

Use the rejected downward input at the gate to face C; turn C from down to left.

Enter Tin from (2,11). Its return highway jams at (4,2).

Escape right to the inner workshop.

Turn T from up to down and ride it to the outer workshop.

Turn Q from right to left and ride home.

Move to the lower launcher shore.

Turn Lf:

DOWN → LEFT → UP → RIGHT

Launch. The route now passes through:

B, preserved in the up orientation;

C, preserved in the left orientation;

the final downward conveyor into the exit.

Intended depth: 50 player inputs.

Final relevant orientations:

Q   = LEFT
T   = DOWN
Tin = UP
C   = LEFT
B   = UP
Lf  = RIGHT
Recovery behavior and anti-cleanliness check

The two long recovery corridors correspond to specific plausible strategic errors.

Final launch before configuring B

The rising route reaches (10,11), but downward-pointing B opposes the incoming upward conveyor. The agent jams on (10,11) and may escape through the gate to (11,11). The bottom corridor returns to the starting region.

Final launch before configuring C

With B correct, the agent rises to (4,11). Downward-pointing C opposes that route. The agent jams on (4,11) and may escape right to (4,12). The right corridor leads down to the bottom recovery route.

The recovery corridors cannot be used as shortcuts because all adjoining final-route seams are either walls or one-way exits from the conveyor.

Other mistaken settings also have physical consequences:

Q = DOWN sends the player around the lower detour and back to E.

Q = UP jams locally and can be escaped toward the start.

T = RIGHT reaches D by a longer visible route.

T = LEFT cannot simply carry the player back through the inner gate.

Incorrect Tin orientations return to D or create a locally escapable jam.

Every rotatable is necessary. The final launcher alone is useless until the remotely accessible B and C have been prepared. Conversely, preparing B and C does not solve the level unless T and Q are subsequently reversed in the correct order.

6. Verification summary

The authored instances were checked over the finite state representation:

(agent position,
 agent facing,
 jammed/not jammed,
 orientation of every rotatable conveyor)

An exhaustive breadth-first search under the specified action semantics found the following shortest solution lengths:

Level 1:  1
Level 2:  5
Level 3:  6
Level 4: 22
Level 5: 40
Level 6: 42
Level 7: 50

Additional state-space checks established:

Every listed reference sequence reaches its stated exit.

No shorter state-space path exists under this contract.

Preventing rotation of any individual rotatable conveyor makes its level unsolvable.

Across all reachable states in all seven levels, no jammed state lacks a legal adjacent floor escape.

Levels 1–3 contain no gates or unrelated late-game structures.

Level 2 has no reachable jam state, keeping the rotation tutorial causally isolated.

Level 3 cannot be solved without deliberately using the newly introduced jam shore.

No persistent state needed for planning is absent from the settled frame.

No click, timer, hidden condition, exact final configuration, or action history participates in validation.
