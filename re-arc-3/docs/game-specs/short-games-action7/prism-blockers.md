Prism Blockers

Exact authored level count: 7

This is a game about firing straight colored beams becoming difficult because every prism is simultaneously the next obstacle, a recolorable target, and a gate controlling everything behind it.

1. Core experience

Prism Blockers is a deterministic click-only sequencing game. Colored emitters sit in obvious button racks around a sparse orthogonal playfield. Clicking an emitter fires its color inward along one row or column.

A prism whose current central facet matches the beam is transparent. The first prism whose current facet does not match absorbs the beam, changes to that color, and ends the shot. Its persistent outer frame never changes and shows the color it must have at completion.

This gives every shot two meanings:

It changes one visible target.

It changes which future beams can pass through that location.

A prism may therefore need to be fixed, deliberately made wrong, used as a transparent route, left wrong as a shield, and restored later. Nothing is hidden, irreversible, random, or timed.

2. Action space and objective
Available actions

The environment exposes exactly:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(px, py) on any canvas cell

Only direct clicking is used by the puzzle.

UP, DOWN, LEFT, RIGHT, and SPACE produce a one-frame gray perimeter pulse and leave every logical state variable unchanged. They do not move a cursor and are never required by a solution.

There is no persistent selection. An emitter fires immediately when clicked, after which selected_emitter remains null.

Clickable regions

Only the raised emitter buttons around the edge are gameplay-clickable.

Clicking any other pixel—including a prism, rail, empty cell, or gap between emitter buttons—produces a neutral gray ripple at the clicked location and changes no logical state.

This makes clicking part of the core mechanic rather than an auxiliary interaction: the player chooses exactly which colored line to fire.

Success condition

A level succeeds immediately after an action when:

for every prism p:
    p.current_color == p.target_color

Relevant:

Every authored prism, including the already-matched reference prism in Level 1.

Ignored:

Emitter states; emitters have no persistent state.

Shot history.

Number of actions.

Last-click feedback.

Whether the final shot was blocked or fully transmitted.

Allowed:

Any number of temporary mismatches.

Fully transmitted shots.

Invalid clicks and unused action types.

Recoloring a previously correct prism.

There is no energy, timer, move limit, score threshold, failure state, or hidden validator condition. The episode terminates after the settled completion frame.

3. Native geometry and visual contract
Logical playfield

The native gameplay lattice is a sparse 9×9 orthogonal coordinate system:

x ∈ {1, …, 9}
y ∈ {1, …, 9}

Coordinates increase:

Left to right for x.

Top to bottom for y.

Prisms occupy logical coordinates but do not move.

An emitter on the west or east side addresses one row. An emitter on the north or south side addresses one column.

64×64 rendering

Canvas pixel coordinates are:

px ∈ {0, …, 63}
py ∈ {0, …, 63}

Define:

origin(k) = 10 + 5 × (k - 1)

A prism at logical coordinate (x, y) occupies the 4×4 pixel rectangle:

px = origin(x) … origin(x) + 3
py = origin(y) … origin(y) + 3

The unused fifth row and column of every logical pitch create a one-pixel gap between adjacent prism sprites. There are no cell outlines, internal grid borders, coordinate labels, or rendered text.

The background is color 5 black. Only rows and columns containing emitters receive a one-pixel guide rail in color 3 dark gray. Rails are drawn behind prisms and are visual guides, not objects.

Horizontal rail center:

py = origin(y) + 1

Vertical rail center:

px = origin(x) + 1
Logical colors

Only two prism states are needed:

Logical state	Stored value	Render family
R	8	red 8 and maroon 13
B	9	blue 9 and light blue 10

The secondary colors are redundant visual patterns. Beam matching compares the stored logical values 8 and 9, not individual rendered pixels.

Other used colors:

ID	Purpose
0	white arrow and shot highlights
1	light-gray source halo
2	neutral invalid-click ripple
3	dark-gray beam rails
4	very-dark shadows
5	black background
14	terminal completion border

No other color IDs are needed.

Prism sprite

Each prism is a connected 4×4 shape:

FFFF
FCCF
FCCF
FFFF

F is the immutable target frame.
C is the mutable current facet.

For local sprite coordinates (lx, ly) ∈ {0,1,2,3}²:

frame pixel iff lx ∈ {0,3} or ly ∈ {0,3}
facet pixel iff lx ∈ {1,2} and ly ∈ {1,2}

The target pattern is drawn on frame pixels:

pattern(R, lx, ly) = 8  if lx <= 1 else 13
pattern(B, lx, ly) = 9  if ly <= 1 else 10

The same pattern function is applied to the central facet using current_color.

Consequently:

A red state is a vertical red/maroon split.

A blue state is a horizontal blue/light-blue split.

A matched prism has one coherent orientation across frame and center.

A mismatched prism visibly contains different outer and inner orientations.

This reinforces matching without relying only on hue.

Emitter button racks and exact hitboxes

A lane may have a red button, a blue button, or both. Buttons are 4×4 raised arrow modules placed outside the logical board.

The red button is always in the slot nearest the board. The blue button is always in the outer slot. An absent button leaves no outline or fake clickable region.

Let:

span(k) = origin(k) … origin(k) + 3

Exact inclusive hitboxes:

Side	Red button R	Blue button B
West, row y	px=5…8, py=span(y)	px=0…3, py=span(y)
East, row y	px=55…58, py=span(y)	px=60…63, py=span(y)
North, column x	px=span(x), py=5…8	px=span(x), py=0…3
South, column x	px=span(x), py=55…58	px=span(x), py=60…63

Buttons use the same red or blue split pattern as prisms, with a white inward-pointing arrow overlaid. The two buttons in a rack feed one shared logical lane. They are controls outside the collision lattice and never block one another.

Shot feedback and animation

A valid emitter click produces one logical transition and a short deterministic render sequence:

Source frame: the clicked button receives a light-gray halo.

Path frame: the entire traversed portion of the rail flashes in the beam’s color family. Every matching prism passed by receives a white edge highlight.

Impact frame:

If a prism absorbs the shot, its central facet flashes white.

If the shot exits the board, the opposite rail endpoint flashes white.

Settled frame: the absorbing prism, if any, displays its new current pattern.

The logical state is computed atomically by the single step() call. Animation subframes are derived from the old and new states; they do not call step(), consume actions, or accept queued input. The canonical post-action observation is the settled frame.

A non-emitter click produces a two-frame color-2 ripple and no state change.

On success, all prism frames flash white once and the outermost canvas pixels flash color 14 green before the terminal settled frame.

4. Formal state machine
Persistent state
LevelState:
    level_id: integer in 1…7
    prisms: ordered list of Prism
    emitters: ordered list of Emitter
    won: boolean
Prism:
    id: unique implementation identifier
    x: integer 1…9
    y: integer 1…9
    current_color: R or B
    target_color: R or B
Emitter:
    id: unique implementation identifier
    side: W, E, N, or S
    lane: integer 1…9
    color: R or B
    slot: inner for R, outer for B

There is no persistent cursor, emitter selection, beam charge, phase, random seed, or hidden counter.

Ray ordering

For a horizontal emitter, collect all prisms with p.y == emitter.lane.

For a vertical emitter, collect all prisms with p.x == emitter.lane.

Sort them as follows:

Side	Scan order
West	increasing x
East	decreasing x
North	increasing y
South	decreasing y

A rail intersection without a prism has no effect. A prism at a crossing is one shared object, not two stacked objects.

Shot resolution
resolve_shot(emitter, old_state):

    ordered_prisms = prisms_on_emitter_ray_in_scan_order(emitter)

    hit = null
    traversed = []

    for prism in ordered_prisms:
        traversed.append(prism)

        if prism.current_color == emitter.color:
            continue

        hit = prism
        break

    new_state = copy(old_state)

    if hit != null:
        new_state.prism[hit].current_color = emitter.color

    new_state.won =
        all(p.current_color == p.target_color for p in new_state.prisms)

    return new_state, traversed, hit

Crucial update-order rule:

The absorbing prism ends the current shot even though the recoloring leaves it matching the beam.

The beam never resumes beyond that prism during the same action. A second click is required to pass through it.

Target frames never influence propagation. Only the current facet matters.

Each valid shot changes either zero or one prism.

Click resolution
step(CLICK(px, py)):
    if (px, py) lies in exactly one authored emitter hitbox:
        resolve_shot(that emitter)
    else:
        state unchanged

step(UP | DOWN | LEFT | RIGHT | SPACE):
    state unchanged

Hitbox edges are inclusive. Authored emitter hitboxes never overlap.

5. Curriculum and verified depth
Level	Main discovery	Shortest meaningful clicks
1	Emitters recolor the first mismatch; matching frame and center is the goal	1
2	A prism matching the beam transmits it	5
3	A correct prism may need to become temporarily wrong	6
4	A nearer mismatch can deliberately protect a finished suffix	25
5	Two shared gates require ordered reuse and restoration	47
6	Entry direction reverses which prism is near, deep, and protective	54
7	Four shared gates form a restoration loop with no safe line-by-line order	62

For solution notation below:

Emitter-ID × n

means click that emitter exactly n consecutive times.

Every click in every supplied solution changes a prism; none of the listed length comes from no-op shots or movement padding.

6. Authored levels
Level 1 — One Visible Correction

Teaches: the objective, emitter click affordance, absorption, and recoloring.

Player-facing discovery

A small already-matched prism at (3,3) visibly shows a coherent red vertical pattern. A second prism has the same red target frame but a blue horizontal center. The only raised button is a red inward arrow aligned with the unsolved prism.

Clicking the prism itself gives only a neutral ripple. Clicking the obvious arrow causes the beam to terminate at the unsolved prism, whose center becomes visually identical to its frame and to the solved reference. Completion follows immediately.

The reference prism is positive goal evidence, not a decoy.

Exact state

Initial selection: none.

Prism	Coordinate	Initial	Target	Purpose
REF	(3,3)	R	R	Solved visual example; included in validator
P1	(5,5)	B	R	Only unsolved prism

Emitter:

ID	Side/lane	Color	Exact ray order
W5-R	West, row 5	R	[(5,5)]
Intended solution
W5-R

Shortest depth: 1.

Attribution and necessity check

Only one logical value changes. No second object or simultaneous rule can explain completion. The solved reference communicates the target relation before the first action, while the single emitter supplies the distinguishing experiment.

Level 2 — Penetration

Teaches: a prism matching the beam becomes transparent on later shots.

Player-facing discovery

Five identical unsolved red-framed prisms lie on one red rail. The first click recolors only the nearest prism. Repeating the exact same click visibly passes through that now-red prism and recolors the second.

This immediately distinguishes the real rule from plausible alternatives such as:

Every prism always stops a shot.

The emitter always affects the nearest coordinate.

Repeated shots do nothing after the first prism is fixed.

Every subsequent click reaches one prism deeper.

Exact state

Initial selection: none.

Prisms in west-to-east order:

Coordinates	Initial string	Target string
[(1,5),(3,5),(5,5),(7,5),(9,5)]	BBBBB	RRRRR

Emitter:

ID	Side/lane	Color	Exact ray order
W5-R	West, row 5	R	(1,5) → (3,5) → (5,5) → (7,5) → (9,5)
Intended solution
W5-R × 5

Shortest depth: 5.

The five settled states expose progressive penetration:

RBBBB
RRBBB
RRRBB
RRRRB
RRRRR
Attribution and necessity check

Because one shot can recolor at most one prism, five mismatches give a lower bound of five actions. The solution achieves that bound. All five objects are changed and required by the validator.

Level 3 — Borrowed Color

Teaches: a prism’s target color is not necessarily the color it should have immediately.

Player-facing discovery

A blue-target gate is already correct at the front of a row. Four red-target prisms lie behind it. The red beam cannot reach those deeper prisms while the gate remains blue.

The local impasse points directly at the red emitter. Firing it appears to make the board worse: the correct blue gate absorbs the shot and becomes red inside its still-blue frame. But the next red shot now passes through it and reaches the first deep prism.

After all four deeper prisms are red, the perpendicular blue emitter restores the gate. The same gate has acted first as an obstacle, then as a transparent route, and finally as a target.

Exact state

Initial selection: none.

Row in west-to-east order:

Role	Coordinates	Initial	Target
Shared gate and four deep prisms	[(2,5),(4,5),(6,5),(8,5),(9,5)]	BBBBB	BRRRR

Emitters:

ID	Side/lane	Color	Exact ray order
W5-R	West, row 5	R	(2,5) → (4,5) → (6,5) → (8,5) → (9,5)
N2-B	North, column 2	B	[(2,5)]
Intended solution
W5-R × 5
N2-B

Shortest depth: 6.

Resolution:

First red shot temporarily makes (2,5) wrong.

Four more red shots progressively reach all four deeper targets.

The blue shot restores (2,5) and stops there.

Attribution and necessity check

The first red action visibly worsens exactly one correct prism, and only that temporary change enables later progress. The final perpendicular shot is necessary; without it the gate remains mismatched. Five red absorptions plus one blue restoration also provide a six-action lower bound.

Level 4 — Last Shield

Teaches: protective interception and backward construction of a finished suffix.

This is the first level with two-button emitter racks. The paired red and blue arrow buttons are both visibly connected to the same rail; no new click rule is introduced.

Player-facing discovery

The horizontal target alternates colors, so repeatedly sweeping only one color cannot finish it. The player discovers that a deep target pattern must be built from the far end backward: sweep a color through the full row, restore part of the prefix, then use the opposite color again.

The vertical finishing phase forces the protective interpretation. Immediately before completion, the deeper vertical prisms are already correct and red, while the nearest vertical prism is wrong and red inside a blue frame. Firing blue looks as though it might threaten the red prisms behind it, but the near mismatch absorbs the shot and protects them.

Exact state

Initial selection: none.
Every prism initially has current color R.

Horizontal prisms, west to east:

Coordinates	Target
[(1,5),(2,5),(4,5),(5,5),(7,5),(9,5)]	BRRBRB

The shared prism is (4,5).

Vertical-only prisms, north to south:

Coordinates	Target
[(4,1),(4,7),(4,9)]	BRB

Thus the complete vertical target, including the shared prism, is:

(4,1) (4,5) (4,7) (4,9)
   B      R      R      B

Emitters:

ID	Side/lane	Color	Exact ray order
H-R	West, row 5	R	(1,5),(2,5),(4,5),(5,5),(7,5),(9,5)
H-B	West, row 5	B	same order
V-R	North, column 4	R	(4,1),(4,5),(4,7),(4,9)
V-B	North, column 4	B	same order
Intended insight

The horizontal suffix is constructed from right to left. A nearer mismatch is not merely an unfinished target; it determines where the next shot terminates and therefore preserves everything beyond it.

One shortest solution
H-B × 6
H-R × 5
H-B × 4
H-R × 2
H-B × 1
V-B × 3
V-R × 3
V-B × 1

Total: 25 clicks.

Useful checkpoints, written in emitter ray order:

Start horizontal:             RRRRRR
after H-B × 6:                BBBBBB
after H-R × 5:                RRRRRB
after H-B × 4:                BBBBRB
after H-R × 2:                RRBBRB
after H-B × 1:                BRBBRB

At that point the shared prism is blue and still wrong. The vertical phase is:

before vertical work:         RBRR
after V-B × 3:                BBBB
after V-R × 3:                RRRB
after final V-B:              BRRB

The final blue shot changes only the nearest prism (4,1). Its mismatch intercepts the shot before the already-correct red prisms (4,5) and (4,7).

Anti-degenerate and necessity check

The authored shortest path is 25 actions, and every action changes one prism. Every prism is recolored at least once. Removing any of the four emitter buttons makes the target unreachable.

The offset shared gate is essential: the horizontal phase deliberately leaves it blue, and the vertical phase must restore it without destroying the horizontal suffix.

Level 5 — Two-Gate Spine

Teaches: shared-gate ordering across a dependency network rather than one dependency chain.

Player-facing discovery

Two horizontal rows cross one vertical spine. The upper shared prism targets red; the lower shared prism targets blue.

To reach the lower blue targets with the blue spine emitter, the upper gate must temporarily be blue even though its frame demands red. Later, restoring the upper gate changes which vertical shots can reach the lower gate. The two crossings therefore cannot be treated as independent row tasks.

The same operational rule gives contrasting roles:

Upper gate: temporary blue conduit, final red blocker.

Lower gate: temporary red blocker, final blue conduit.

Near spine prisms: alternately routes and shields during restoration.

Exact state

Initial selection: none.
Every prism initially has current color R.

Top horizontal row H1, west to east:

Coordinates	Target
[(1,3),(2,3),(4,3),(5,3),(7,3),(9,3)]	BRRBRB

Bottom horizontal row H2, west to east:

Coordinates	Target
[(1,7),(2,7),(3,7),(4,7),(6,7),(9,7)]	BRBBRB

Shared gates:

G1 = (4,3), target R
G2 = (4,7), target B

Vertical-only prisms, north to south:

Coordinates	Target
[(4,1),(4,4),(4,6),(4,9)]	RBRB

Complete vertical target:

(4,1),(4,3),(4,4),(4,6),(4,7),(4,9)
   R     R     B     R     B     B

Emitters:

ID	Side/lane	Color	Exact ray order
H1-R	West, row 3	R	(1,3),(2,3),(4,3),(5,3),(7,3),(9,3)
H1-B	West, row 3	B	same order
H2-R	West, row 7	R	(1,7),(2,7),(3,7),(4,7),(6,7),(9,7)
H2-B	West, row 7	B	same order
V-R	North, column 4	R	(4,1),(4,3),(4,4),(4,6),(4,7),(4,9)
V-B	North, column 4	B	same order
Intended insight

The lower row can be prepared before the spine, but its shared gate cannot be finalized independently. Sweeping the spine blue opens access to its bottom, while simultaneously sacrificing the upper gate. The endgame must revisit the top row and spine several times so that nearer mismatches shield the gate that should remain untouched.

One shortest solution
H1-B × 6
H1-R × 2

H2-B × 6
H2-R × 5
H2-B × 3
H2-R × 2
H2-B × 1

V-B × 5

H1-R × 3
H1-B × 2
V-R × 3
H1-B × 2
H1-R × 2
H1-B × 1
V-B × 2
V-R × 2

Total: 47 clicks.

Key phase boundary:

After V-B × 5, the lower horizontal row has reached its final target, but the vertical spine is all blue. The remaining sequence does not simply “finish the top row and then finish the spine.” It alternates them because changing G1 changes the stopping point of both vertical colors.

Final states:

H1 west→east:    BRRBRB
H2 west→east:    BRBBRB
V north→south:   RRBRBB
Anti-degenerate and necessity check

Every one of the 16 prisms changes at least once. All six emitter buttons occur in the shortest solution. Removing any one of them makes the target unreachable.

The two crossing depths are uneven: G1 is the second prism encountered from the north, while G2 is the fifth. This asymmetry is what creates ordered reuse; replacing the layout with independent rows removes the puzzle.

Level 6 — Opposite Approaches

Teaches: entry direction reverses the meaning of “near,” “deep,” and “protective.”

Player-facing discovery

Two horizontal rows and two vertical columns form a four-gate rectangle. The rows are not approached from the same side:

Top row fires from the west.

Bottom row fires from the east.

Left column fires from the north.

Right column fires from the south.

A pattern that would be built from left to right on the top row must be built from right to left on the bottom row. Similarly, the bottom prism of the right column becomes its first protective blocker.

The arrows and rails expose this before experimentation. A shot immediately confirms which end is scanned first.

Exact state

Initial selection: none.
Every prism initially has current color R.

Top row H1, listed west to east:

Coordinates	Target
[(1,3),(2,3),(3,3),(5,3),(7,3),(9,3)]	RBRRBB

Bottom row H2, listed west to east:

Coordinates	Target
[(1,7),(3,7),(4,7),(5,7),(7,7),(9,7)]	BBRBRB

Shared gates:

G11 = (3,3), target R
G12 = (7,3), target B
G21 = (3,7), target B
G22 = (7,7), target R

Left-column-only prisms, north to south:

Coordinates	Target
[(3,1),(3,5),(3,8),(3,9)]	RBRB

Complete left-column target:

(3,1),(3,3),(3,5),(3,7),(3,8),(3,9)
   R     R     B     B     R     B

Right-column-only prisms, north to south:

Coordinates	Target
[(7,1),(7,2),(7,5),(7,9)]	BRBB

Complete right-column target from north to south:

(7,1),(7,2),(7,3),(7,5),(7,7),(7,9)
   B     R     B     B     R     B

Emitters:

ID	Side/lane	Color	Exact ray order
H1-R	West, row 3	R	(1,3),(2,3),(3,3),(5,3),(7,3),(9,3)
H1-B	West, row 3	B	same order
H2-R	East, row 7	R	(9,7),(7,7),(5,7),(4,7),(3,7),(1,7)
H2-B	East, row 7	B	same order
V1-R	North, column 3	R	(3,1),(3,3),(3,5),(3,7),(3,8),(3,9)
V1-B	North, column 3	B	same order
V2-R	South, column 7	R	(7,9),(7,7),(7,5),(7,3),(7,2),(7,1)
V2-B	South, column 7	B	same order
Intended insight

The four shared gates have a checkerboard-like final assignment, but this is not a matching checklist. Their positions within four differently oriented ray orders determine which gate can safely be changed at each phase.

The left column must be cycled before the right column. The right-column work then forces revisits to the east-fired bottom row, because G22 lies near the right-column source but near the bottom-row source as well.

One shortest solution
H1-B × 6
H1-R × 2

H2-B × 6
H2-R × 1

V1-B × 4
H1-R × 2
H1-B × 2
H1-R × 1
V1-R × 4
V1-B × 4
V1-R × 2

V2-B × 4
H2-R × 3
H2-B × 1
V2-R × 4
H2-B × 2
H2-R × 1
V2-B × 3
H2-R × 1
H2-B × 1

Total: 54 clicks.

Final strings in each emitter’s scan order are:

H1, west→east:    RBRRBB
H2, east→west:    BRBRBB
V1, north→south:  RRBBRB
V2, south→north:  BRBBRB

The H2 string is reversed relative to its west-to-east visual listing, which is the central directional lesson.

Anti-degenerate and necessity check

All 20 prisms are recolored during the shortest path. All eight buttons are necessary: removing any one makes the target unreachable.

The four gates are encountered at different depths:

G11: second on V1.

G21: fourth on V1.

G22: second on V2 because V2 enters from the south.

G12: fourth on V2.

Those offsets create the dependency network. Making all emitters enter from the same side substantially changes and simplifies the puzzle.

Level 7 — Closed Restoration Loop

Combines: progressive penetration, temporary wrong colors, protective interception, shared-gate ordering, and approach asymmetry.

No new object or exception is introduced.

Player-facing discovery

The final board retains the four-gate rectangle but changes three structural facts:

The top row has seven prisms rather than six.

The top row is approached from the east.

The two vertical gates lie at unequal depths within that longer row.

The extra prism is not padding. It shifts which prefix must be repainted before either vertical can safely use the top gates.

A line-by-line plan fails. The right vertical and top horizontal form a restoration loop:

The right vertical must overwrite a top-row gate to reach deeper targets.

The top row must then repair that gate.

Repairing it changes where the next right-vertical shot stops.

The right vertical must fire again under the new interception pattern.

The final seven run groups alternate these two lines rather than completing either independently.

Exact state

Initial selection: none.
Every prism initially has current color R.

Top row H1, listed west to east:

Coordinates	Target
[(1,3),(2,3),(3,3),(4,3),(6,3),(7,3),(9,3)]	BRRBRBB

H1 is fired from the east, so its emitter-facing target order is BBRBRRB.

Bottom row H2, west to east:

Coordinates	Target
[(1,7),(3,7),(4,7),(5,7),(7,7),(9,7)]	RRBRRB

Shared gates:

G11 = (3,3), target R
G12 = (7,3), target B
G21 = (3,7), target R
G22 = (7,7), target R

Left-column-only prisms, north to south:

Coordinates	Target
[(3,1),(3,5),(3,6),(3,9)]	RBRB

Complete left-column target:

(3,1),(3,3),(3,5),(3,6),(3,7),(3,9)
   R     R     B     R     R     B

Right-column-only prisms, north to south:

Coordinates	Target
[(7,1),(7,2),(7,5),(7,9)]	BRBR

Complete right-column target from north to south:

(7,1),(7,2),(7,3),(7,5),(7,7),(7,9)
   B     R     B     B     R     R

Emitters:

ID	Side/lane	Color	Exact ray order
H1-R	East, row 3	R	(9,3),(7,3),(6,3),(4,3),(3,3),(2,3),(1,3)
H1-B	East, row 3	B	same order
H2-R	West, row 7	R	(1,7),(3,7),(4,7),(5,7),(7,7),(9,7)
H2-B	West, row 7	B	same order
V1-R	North, column 3	R	(3,1),(3,3),(3,5),(3,6),(3,7),(3,9)
V1-B	North, column 3	B	same order
V2-R	South, column 7	R	(7,9),(7,7),(7,5),(7,3),(7,2),(7,1)
V2-B	South, column 7	B	same order
Intended insight

The useful plan is to prepare broad suffix structure while accepting that shared gates are provisional. The left vertical is resolved first, but the right vertical cannot be closed without repeatedly changing the top row.

The top row’s westmost targets are visually far from its east-side controls. Their final colors must be preserved by nearer mismatches while the two shared gates are alternately exposed and shielded.

One shortest solution
H1-B × 7
H1-R × 4

H2-B × 6
H2-R × 4
H2-B × 3
H2-R × 1

V1-B × 4
H1-R × 2
H1-B × 4
H1-R × 1
H2-R × 1
V1-R × 3
V1-B × 3
V1-R × 2

V2-B × 4
H1-R × 2
H1-B × 1
V2-R × 4
H1-B × 1
V2-B × 3
V2-R × 2

Total: 62 clicks.

The closing restoration loop is:

V2-B × 4
H1-R × 2
H1-B
V2-R × 4
H1-B
V2-B × 3
V2-R × 2

Neither H1 nor V2 can be completed and then ignored. Each changes a shared gate required by the other.

Final strings in emitter scan order:

H1, east→west:    BBRBRRB
H2, west→east:    RRBRRB
V1, north→south:  RRBRRB
V2, south→north:  RRBBRB
Anti-degenerate and necessity check

All 21 prisms change at least once. Every click in the 62-action path changes exactly one prism. Removing any of the eight emitter buttons makes the target unreachable.

The seventh top-row prism changes the depths of both shared gates relative to the east emitter. Removing it does not merely shorten a corridor; it changes the restoration dependencies and the safe interception order.

7. Implementation and design verification

The authored instances were checked by exact breadth-first enumeration of their finite color-state graphs. Since every prism is binary and every emitter action is deterministic, the shortest depths are exact:

Level 1:  1
Level 2:  5
Level 3:  6
Level 4: 25
Level 5: 47
Level 6: 54
Level 7: 62

Additional checks:

Every supplied shot changes exactly one prism; there are no solution-padding no-ops.

Every prism in Levels 2–7 is recolored at least once.

The only unchanged prism in the complete progression is Level 1’s solved reference, whose explicit role is communicating the objective.

In Levels 4–7, removing any single emitter makes the authored target unreachable.

No later level introduces a rule exception.

All strategic differences arise from ray direction, order along a ray, shared coordinates, current state, and target placement.

No level requires memorizing an arbitrary symbol mapping.

No target depends on click history, action count, transient animation, or invisible state.

The settled frame always contains all information needed to choose the next experiment.
