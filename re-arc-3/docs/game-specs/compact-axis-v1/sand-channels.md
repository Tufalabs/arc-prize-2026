Sand Channels

This is a game about filling terrain becoming difficult because every successful footstep advances an irreversible flow.

Authored level count: exactly 7.

The player controls a surveyor moving across dry stone. A finite hopper feeds a visible directed network of channel basins. Every successful movement releases one unit of sand into the first unfilled channel cell on the currently selected route. A basin remains completely impassable while even one of its cells is empty; when its final cell fills, the entire basin compacts into walkable terrain.

The central tension is that movement is both locomotion and a flow clock. Walking toward a diverter may feed the wrong branch before the surveyor reaches it. A basin built as a bridge may also carry later sand farther downstream. A route that is wastefully long in one phase may become exactly the right clock path after a fork has been prepared.

No rule is hidden behind a counter. The hopper level, loose sand, compacted basins, fork orientations, current frontier, exit, and all traversable terrain are visible in every settled frame.

1. Native geometry

The native gameplay lattice is 13 rows × 13 columns.

Coordinates are (row, column):

(0,0) is the upper-left logical tile.

Rows increase downward.

Columns increase rightward.

Movement is orthogonal only.

Every movement action attempts to move exactly one logical tile.

All unlisted logical coordinates in a level are void and impassable.

A logical tile is larger than one rendered pixel. The game is not played at pixel resolution.

Render mapping

The canvas is 64×64 pixels.

Logical tile (r,c) occupies:

x = 1 + 4c through 4 + 4c

y = 6 + 4r through 9 + 4r

The logical board therefore occupies pixels x=1…52, y=6…57.

The remaining right edge contains the hopper.

No text, digits, letters, or level labels are rendered.

No global grid lines are rendered.

Adjacent stone tiles merge into continuous stone regions.

A basin has one outer silhouette; its internal logical cells are indicated by recessed pockets and flow grooves, not rectangular cell borders.

The lettered schematics later in this document are implementation diagrams only. Those letters never appear in the game.

2. Action space

The environment accepts exactly:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x,y) on any canvas cell

The game uses movement and SPACE. Clicking is deliberately unused.

Movement

A movement is successful when the destination is currently walkable.

Walkable terrain consists of:

dry stone;

every logical cell belonging to a completely filled basin.

The following are never walkable:

void;

diverters;

empty basin cells;

loose-filled cells in a basin that is not yet complete.

A successful movement:

moves the surveyor;

if the hopper is nonempty, releases exactly one sand unit.

A blocked movement:

leaves the surveyor in place;

does not reduce the hopper;

does not advance sand;

produces a short visible bump.

This makes a blocked move a safe diagnostic action rather than a hidden punishment.

SPACE

SPACE operates only when the surveyor is orthogonally adjacent to exactly one diverter.

A valid SPACE:

toggles that binary diverter to its other branch;

changes its physical paddle orientation;

does not move the surveyor;

does not release sand;

does not reduce the hopper.

An invalid SPACE:

changes no gameplay state;

makes the surveyor’s tool shake for two render frames.

The authored levels never place the surveyor adjacent to two diverters simultaneously. For implementation completeness, if that occurs in a malformed state, SPACE is invalid.

Clicking

No object or region is clickable. No click affordance is drawn.

CLICK(x,y):

produces a two-frame light-gray reticle pulse at the clicked pixel;

changes no gameplay state;

releases no sand;

does not select an object.

All levels begin and settle with:

selected_object = null

cursor_cell = null

Click feedback is render-only and is gone in the settled observation.

3. Visual language

Only integer color IDs 0–15 are used.

ID	Use
0	white glints, surveyor visor, completion flash
1	light-gray flow engravings
2	gray basin lips and stone edge accents
3	dark-gray stone and diverter housing
4	very-dark-gray empty channel recess
5	black void/background
6	magenta success pulse accent, used sparingly
7	pink valid-SPACE highlight and frontier glint
8	red bump feedback and empty-hopper warning
9	blue surveyor pack
10	light-blue surveyor body
11	yellow loose sand and active diverter paddle
12	orange compacted sand and source inlet
13	unused
14	green exit and success halo
15	unused
Stone

A connected stone region is rendered primarily in color 3, with only its exterior silhouette accented in color 2. There are no outlines between logical stone tiles.

Basin beds

A basin is a connected multi-tile shape.

Each basin cell contains:

a recessed center in color 4;

an engraved groove in color 1;

a one-pixel directional tip toward its successor.

Shared edges between consecutive cells have a narrow connecting neck. There is no complete rectangular outline around each internal cell.

The basin’s ordered cell path is therefore visible without requiring the player to infer global graph connectivity.

Loose sand

A filled cell in an incomplete basin has:

a yellow center in color 11;

its flow groove still visible;

the rest of the basin unchanged.

Every cell in that basin, including already loose-filled cells, remains impassable.

Active frontier

The literal next cell that would receive sand has:

a two-pixel yellow ridge on its entry side;

one white glint pixel;

otherwise the appearance of an empty bed.

If the hopper is empty, this ridge becomes red, showing the stranded next target.

If the selected route is already completely filled to its terminal, the ridge appears at the terminal outlet lip. The next unit would spill there.

Compacted basin

When the final cell fills:

all loose pockets in that basin compact simultaneously;

interiors become orange 12;

internal cell separations visually collapse into one continuous slab;

dark flow grooves remain visible;

every basin cell becomes walkable.

The slab is permanently compacted.

Diverter

A diverter occupies one logical tile but is a connected 4×4-pixel module.

It shows:

one inlet arm;

two outlet arms;

a yellow paddle pointing toward the selected outlet;

the unselected arm in gray.

When the surveyor is orthogonally adjacent, the outer corners pulse pink 7. This is the only SPACE affordance.

Surveyor

The surveyor is a connected 3×3-pixel light-blue figure with a blue pack and white visor, centered over the underlying logical tile.

Exit

The exit is a green 3×3 beacon with a white center. It overlays its underlying terrain but does not change that terrain’s walkability.

An exit may therefore be visible on an incomplete basin while remaining inaccessible.

Hopper

The hopper occupies:

outer frame: x=56…62, y=6…57;

inner reservoir: x=58…60, y=8…54.

One horizontal inner pixel row represents one sand unit. The largest authored supply is 45, fitting within the 47-row reservoir.

For h remaining units, rows 55-h…54 are yellow. Five-unit tick notches appear on the left frame. Counting sand is an actual puzzle variable, so these repeated marks are functional rather than decorative.

At zero:

the reservoir is empty;

the bottom frame turns red;

movement remains possible.

4. Sand-flow state machine

Each level defines a rooted directed network containing:

ordered basin cells;

binary diverters;

terminal spill outlets.

Authored networks are acyclic.

A basin with cells

A0, A1, …, Ak

has fixed internal successors:

A0 → A1 → … → Ak.

The final cell then points either to:

a diverter;

another basin’s first cell;

a terminal spill outlet.

Finding the active frontier

Starting from the root:

At a diverter, follow its selected branch.

At an unfilled basin cell, stop. That cell is the frontier.

At a filled basin cell, follow its successor.

At a terminal, return SPILL.

Only one frontier exists. Sand never splits.

Filled cells remain filled when a diverter is turned away from their branch. Returning to that branch resumes from its first still-empty cell.

Downstream fork preview

A downstream diverter may be turned while an earlier basin cell is still empty.

In that case:

the literal frontier remains on the earlier empty cell;

the paddle visibly changes;

the newly selected downstream arm flashes yellow once;

when the upstream cell later fills, the frontier proceeds through the selected arm.

If the turned diverter lies before the current frontier, the frontier ridge visibly relocates immediately.

This preserves the physical “first empty cell” rule while making every valid turn visibly attributable.

5. Exact step() resolution order

All gameplay changes occur only inside step(action). Animation frames never autonomously advance sand.

Movement action

Compute walkability from the pre-action state.

Compute the destination.

If blocked:

do not move;

do not change the hopper;

do not change sand;

set bump feedback;

skip to validation.

If walkable:

move the surveyor.

If hopper_remaining > 0:

compute the active frontier using current fork orientations;

decrement the hopper by exactly one;

if the frontier is a basin cell, mark that cell filled;

if the frontier is SPILL, record one visual spill;

recompute completion for the affected basin.

If a basin has just become complete:

all its cells become walkable for the next action.

Recompute the settled frontier marker.

Validate success.

Movement legality is checked before sand release. The surveyor cannot move onto a basin completed by that same movement action.

SPACE

Find orthogonally adjacent diverters.

If exactly one exists:

toggle its orientation;

recompute the frontier;

do not release sand.

Otherwise:

leave gameplay state unchanged;

show tool-shake feedback.

Validate success.

Click

Record render-only reticle feedback.

Change no gameplay state.

Validate success.

Success

A level is won exactly when:

surveyor_position == exit_coordinate

after the action’s full resolution.

Nothing else is validated:

hopper contents are ignored;

irrelevant basins may be empty or partial;

forks may face either direction;

traps may contain sand;

the exit’s underlying basin need only be walkable enough for the surveyor to occupy it.

A successful movement onto the exit still performs its normal sand release before the win flag is set.

There is no automatic loss when the hopper empties. A dead state remains visible and navigable. The host environment should permit at least 512 submitted actions before truncation; this cap is not rendered and is not a puzzle timer.

6. Animation contract

Animation is sparse and deterministic. Input is not accepted until the settled frame is reached. Animation frames do not consume actions or hopper units.

Sand release

After a successful movement:

The surveyor snaps to the destination.

A yellow pulse traces the selected route.

Up to four graph nodes are traversed per animation frame.

On the final trace frame:

the hopper surface lowers by one row;

the frontier cell fills, or the unit spills from the terminal lip.

If a basin completes, a two-frame orange compaction wave crosses its connected shape.

The new frontier ridge appears.

The gameplay state is committed on the final animation frame. A headless implementation may compute the same post-state atomically.

Diverter rotation

A valid SPACE uses three frames:

old paddle dims;

paddle rotates through the hub;

new arm becomes yellow and the orientation commits.

Blocked movement

Two frames:

the surveyor offsets one rendered pixel toward the obstacle and the contacted edge flashes red;

the surveyor returns.

Completion

After success validation, the exit emits a three-frame green halo. No further sand or movement occurs.

7. Level notation

The documentation schematics use:

# — void

. — stone

@ — surveyor start

X — exit

uppercase letter — basin

lowercase f, g, h — diverter

X replaces the symbol of its underlying terrain in the schematic. The coordinate tables are authoritative.

All basin coordinate lists are given in flow order.

Level 1 — Last Grain
Teaches

A successful footstep advances one visible sand unit.

Filling the last gap compacts the entire basin.

A compacted basin is floor.

There is only one causal unknown. The basin begins one cell short of completion, so the first useful movement produces an immediate and unmistakable transformation.

Player-facing discovery

The exit sits on the far cell of a two-cell basin. A0 already contains loose sand, while A1 has the frontier ridge. The only forward stone step fills A1; both cells then compact together. The surveyor can immediately walk across.

No legend or reference panel is needed because the cause and physical consequence occur side by side.

Schematic
00 #############
01 #############
02 #############
03 #############
04 ###.#########
05 ##@.AX#######
06 #############
07 #############
08 #############
09 #############
10 #############
11 #############
12 #############
   0123456789012
Exact instance
Property	Value
Start	(5,2)
Exit	(5,5), underlying terrain A1
Hopper	12
Selection/cursor	none
Stone	{(4,3),(5,2),(5,3)}
Basin A	[(5,4),(5,5)]
Initially filled	{A0}; basin remains incomplete
Root	A0, source inlet from north
Graph	A0 → A1 → SPILL
Initial frontier	A1
Verified solution
RIGHT, RIGHT, RIGHT

Move to (5,3). A1 fills and A compacts.

Move onto A0.

Move onto A1, which contains the exit.

Minimum solution depth: 3 actions.

The second and third movements spill harmless excess sand because the selected route is already complete. The generous hopper makes this irrelevant.

Attribution and necessity

The first action is the only gameplay change before compaction.

The exit cannot be entered before A completes.

There are no forks, alternate basins, decoys, or simultaneous state changes.

Clicking and SPACE cannot solve the level.

Level 2 — Loose Is Not Floor
Teaches

Sand advances exactly one cell per successful movement.

A partially filled basin is still completely impassable.

A blocked movement does not advance the flow.

Player-facing discovery

The direct-looking route crosses a three-cell basin. Moving right once fills only A0. A second rightward attempt bumps into it and leaves the hopper and frontier unchanged. Stepping around the tiny stone chamber supplies the remaining two units; only then does the whole basin become floor.

The bump is a low-cost distinguishing experiment. It disproves both “a filled cell is individually walkable” and “every submitted arrow action releases sand.”

Schematic
00 #############
01 #############
02 #############
03 #############
04 ##..#########
05 ##@.AAX######
06 #############
07 #############
08 #############
09 #############
10 #############
11 #############
12 #############
   0123456789012
Exact instance
Property	Value
Start	(5,2)
Exit	(5,6), underlying terrain A2
Hopper	16
Selection/cursor	none
Stone	{(4,2),(4,3),(5,2),(5,3)}
Basin A	[(5,4),(5,5),(5,6)]
Initially filled	none
Root	A0, source inlet from north
Graph	A0 → A1 → A2 → SPILL
Initial frontier	A0
Recommended discovery path
RIGHT,
RIGHT [blocked],
UP, LEFT,
DOWN,
RIGHT, RIGHT, RIGHT, RIGHT

Checkpoints:

After the first RIGHT: A0 is loose-filled, hopper 15.

The blocked RIGHT changes nothing.

UP fills A1.

LEFT fills A2; A compacts.

The remaining movements return and cross the basin.

This path submits 9 actions, one of which is the informative bump.

Minimum rule-known solution
UP, DOWN, RIGHT, RIGHT, RIGHT, RIGHT

Minimum solution depth: 6 actions.

Attribution and necessity

The incomplete basin is the only obstruction.

The direct bump isolates blocked-action behavior.

Completion is reused from Level 1; the new evidence is that two loose-filled cells still do not constitute floor.

The hopper is more than five times the required fill, so experimentation is cheap.

Level 3 — The Paddle
Teaches

A highlighted adjacent diverter is operated with SPACE.

A fork sends all future sand down exactly one branch.

Turning the fork immediately moves the frontier when the fork is upstream of it.

Player-facing discovery

The surveyor begins below a visibly fork-shaped diverter. The exit lies beyond basin A, but the yellow paddle and frontier ridge point left into B.

Trying UP bumps against the mechanical fork. The fork is outlined pink, suggesting SPACE. Pressing it rotates the paddle to the right and moves the ridge from B0 to A0.

A wrong exploratory footstep into B is recoverable because the hopper is generous.

Schematic
00 #############
01 #############
02 #############
03 #############
04 #############
05 #BBfAAX######
06 ###@.########
07 #############
08 #############
09 #############
10 #############
11 #############
12 #############
   0123456789012
Exact instance
Property	Value
Start	(6,3)
Exit	(5,6), stone
Hopper	14
Selection/cursor	none
Stone	{(6,3),(6,4),(5,6)}
Basin A	[(5,4),(5,5)]
Basin B	[(5,2),(5,1)]
Fork F	coordinate (5,3)
F branches	branch 0 → A0, branch 1 → B0
Initial F orientation	branch 1, toward B
Root	F, source inlet from north
Terminals	A1 → SPILL, B1 → SPILL
Initial frontier	B0
Verified solution
SPACE,
RIGHT, LEFT,
RIGHT, UP, RIGHT, RIGHT

Compact form: S R L R U R R.

Checkpoints:

SPACE: F points to A; hopper remains 14.

First RIGHT: fills A0.

LEFT: fills A1; A compacts.

Return right, enter A0, cross A1, and step onto the exit.

Minimum solution depth: 7 actions.

Attribution and necessity

SPACE is mandatory: without changing F, no sand reaches A.

The fork is visibly mechanical, highlighted, and physically adjacent.

B is not decorative; it is the visible selected route and receives any diagnostic movement made before turning.

The basin must be crossed after completion rather than merely checked by a validator.

Level 4 — One Step Before the Branch
Teaches and combines

Walking toward a fork consumes the same flow clock the player is trying to control.

A filled upstream basin carries later sand into a downstream fork.

One grain in the wrong branch can be fatal.

A completed basin may need to be traversed backward relative to its flow direction.

Player-facing discovery

The source begins filling the long upstream basin U. The only useful exit route is basin A, but fork F initially points into B.

The surveyor can see every remaining cell in U. Reaching the fork promptly leaves enough time to rotate it before the first post-U grain is sent left into B. The lower stone loop then provides exactly the required movements to finish U and concentrate all remaining sand into A.

The exit is on A0, near the fork, while the surveyor approaches the completed basin from its far end. This forces actual traversal of the newly built terrain.

Schematic
00 #U###########
01 #U####B######
02 #UUUU#B######
03 #UU#UUB######
04 #####UfXAA###
05 ####...##A###
06 ###....##A###
07 ##.....AAA###
08 #..#.#.#..###
09 #....#....###
10 #@###########
11 #############
12 #############
   0123456789012
Exact instance
Property	Value
Start	(10,1)
Exit	(4,7), underlying terrain A0
Hopper	19
Selection/cursor	none
Fork F	(4,6)
F branches	branch 0 → A0, branch 1 → B0
Initial F orientation	branch 1, toward B
Root	U0, source inlet from north
Graph	U → F; F→A or F→B; both branches terminate

Stone coordinates:

{
 (10,1),
 (9,1),(9,2),(9,3),(9,4),(9,6),(9,7),(9,8),(9,9),
 (8,1),(8,2),(8,4),(8,6),(8,8),(8,9),
 (7,2),(7,3),(7,4),(7,5),(7,6),
 (6,3),(6,4),(6,5),(6,6),
 (5,4),(5,5),(5,6)
}

Basins in flow order:

U = [
 (0,1),(1,1),(2,1),(3,1),(3,2),
 (2,2),(2,3),(2,4),(3,4),(3,5),(4,5)
]

A = [
 (4,7),(4,8),(4,9),(5,9),
 (6,9),(7,9),(7,8),(7,7)
]

B = [(3,6),(2,6),(1,6)]

Successors:

U10 → F
A7  → SPILL
B2  → SPILL

Initial frontier: U0.

Verified solution
UP, UP, RIGHT, UP, RIGHT, UP, RIGHT, UP, RIGHT, RIGHT,
SPACE,
DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, UP, LEFT,
UP, RIGHT, UP, UP, UP, LEFT, LEFT

Compact form:

U U R U R U R U R R | S |
D D D D R R R U L |
U R U U U L L

Checkpoints:

After 10 movements: U0…U9 are filled, U10 remains the frontier, hopper 9.

SPACE: F points toward A.

The next DOWN fills U10; U compacts.

The following eight movements fill A0…A7; hopper reaches 0.

The final seven movements traverse A from A6 backward to the exit at A0.

Minimum solution depth: 27 actions.

Necessity and anti-cleanliness check

The total useful capacity is exactly 11 + 8 = 19.

Any grain entering B makes complete A impossible.

The fork must be prepared no later than the moment before the first post-U movement.

The lower loop is not a corridor to the exit; it is a staging path that both advances sand and places the surveyor at the far end of A.

A’s bent shape matters because the exit is at its upstream end.

U functions as a flow delay and conduit rather than merely another bridge.

Level 5 — Bridge, Then Conduit
Teaches and combines

A compacted basin remains transparent to later sand.

Walking across a bridge may fill something elsewhere.

A side branch can first be a sand destination, then become the route needed to prepare a farther fork.

A downstream fork must be set before the final upstream cell fills.

Player-facing discovery

Basin P begins already compacted. This is a deliberate visible reminder of walkable compacted sand.

The frontier is not in P; it lies in branch B beyond P and fork F. Every step across P sends sand through it. The four crossing movements fill all four cells of B, making B another bridge exactly when the surveyor reaches F.

After turning F toward A, the surveyor uses B to reach G. A is still one cell short when G becomes reachable, so G must be set toward C before the next movement. That movement completes A; all subsequent sand passes through it and into C.

The final route reverses over B and P, so earlier construction is reused rather than abandoned.

Schematic
00 ########D####
01 ########D####
02 ###.....gC###
03 ###B#AAAAC###
04 ###B#A###C###
05 ###B#A###C###
06 ##.B#A###C###
07 #PPfAA###C###
08 #P##AA#CCC###
09 #P##CCCC#####
10 #@..C########
11 ####X########
12 #############
   0123456789012
Exact instance
Property	Value
Start	(10,1)
Exit	(11,4), underlying terrain C14
Hopper	30
Selection/cursor	none
Initially filled	every P cell; P starts compacted
Fork F	(7,3), branch 0→A0, branch 1→B0, initially 1
Fork G	(2,8), branch 0→C0, branch 1→D0, initially 1
Root	P0, source inlet from west
Graph	P→F; F→B terminal or F→A→G; G→C or G→D

Stone:

{
 (10,1),(10,2),(10,3),
 (6,2),
 (2,3),(2,4),(2,5),(2,6),(2,7)
}

Basins:

P = [(9,1),(8,1),(7,1),(7,2)]

B = [(6,3),(5,3),(4,3),(3,3)]

A = [
 (7,4),(8,4),(8,5),(7,5),(6,5),(5,5),
 (4,5),(3,5),(3,6),(3,7),(3,8)
]

C = [
 (2,9),(3,9),(4,9),(5,9),(6,9),
 (7,9),(8,9),(8,8),(8,7),
 (9,7),(9,6),(9,5),(9,4),(10,4),(11,4)
]

D = [(1,8),(0,8)]

Successors:

P3  → F
B3  → SPILL
A10 → G
C14 → SPILL
D1  → SPILL

Initial frontier: B0.

Verified solution

Phase 1 — cross P, filling B:

UP, UP, UP, RIGHT

Phase 2 — turn F toward A and reach G through B:

SPACE,
UP, RIGHT, UP, UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT

Phase 3 — turn G toward C, complete A, and return toward the end of C:

SPACE,
LEFT, LEFT, LEFT, LEFT,
DOWN, DOWN, DOWN, DOWN,
LEFT, DOWN, LEFT,
DOWN, DOWN, DOWN,
RIGHT, RIGHT, RIGHT, DOWN

Compact form:

U U U R |
S U R U U U U R R R R |
S L L L L D D D D L D L D D D R R R D

Checkpoints:

Action 4: B completes; hopper 26; surveyor at (7,2) adjacent to F.

Action 5: F points toward A.

Action 15: A0…A9 filled; surveyor at (2,7) adjacent to G.

Action 16: G points toward C; literal frontier remains A10.

Action 17: A10 fills, A compacts, frontier advances to C0.

Action 32: C14 fills, C compacts, hopper 0.

Actions 33–34: enter C13, then the exit at C14.

Minimum solution depth: 34 actions.

Necessity and anti-cleanliness check

P is not a static example: crossing it is what fills B.

B is both the initial sand destination and the only route to G.

Turning F directly toward A is impossible before reaching it; B necessarily completes first.

G is mandatory. One unit in D leaves C incomplete because the hopper exactly equals B4 + A11 + C15.

The return path runs backward across B and P while sand runs forward through A into C.

The level is a dependency network: movement route, current flow route, and future fork access are different structures.

Level 6 — Buy the Shortcut
Teaches and combines

Spending sand on a nearby bridge can save more automatic releases than it costs.

The same long route can be a liability before a fork is prepared and useful afterward.

A completed shortcut changes both movement distance and the timing of a downstream fork.

Player-facing discovery

Fork F initially points into the main trunk M. A short four-cell basin S rises toward an upper stone route. The surveyor begins adjacent to F, making the alternate branch immediately available.

If the player ignores S, the only dry route to G runs around the outer edge. It is long enough for M to finish and many grains to enter W.

Building S costs four grains but creates an eleven-move route to G. This reaches G with M still one cell short, allowing G to be turned toward T.

After that turn, the long outer route reverses strategic meaning. It now supplies exactly the movements needed to fill T while carrying the surveyor to T’s far endpoint.

Schematic
00 #############
01 #######.....#
02 ####....###.#
03 ###..MMgTTT.#
04 ###S#M#W##T.#
05 ###S#M#W##T.#
06 ###S#M#W##T.#
07 ##.S#M#WTTT.#
08 #.@fMM##TTT.#
09 #..#MM##TTT.#
10 #.##MM#.XTT.#
11 #.#####.#TT.#
12 #...........#
   0123456789012
Exact instance
Property	Value
Start	(8,2)
Exit	(10,8), underlying terrain T19
Hopper	36
Selection/cursor	none
Fork F	(8,3), branch 0→M0, branch 1→S0, initially 0
Fork G	(3,7), branch 0→T0, branch 1→W0, initially 1
Root	F, source inlet from south
Graph	F→S terminal or F→M→G; G→T or G→W

Stone:

{
 (8,2),(9,2),(9,1),(8,1),(7,2),

 (3,3),(3,4),
 (2,4),(2,5),(2,6),(2,7),

 (10,1),(11,1),

 every (12,c) for c=1…11,
 every (r,11) for r=1…11,
 every (1,c) for c=7…10,

 (11,7),(10,7)
}

Basins:

S = [(7,3),(6,3),(5,3),(4,3)]

M = [
 (8,4),(9,4),(10,4),(10,5),(9,5),(8,5),
 (7,5),(6,5),(5,5),(4,5),(3,5),(3,6)
]

T = [
 (3,8),(3,9),(3,10),
 (4,10),(5,10),(6,10),(7,10),(8,10),(9,10),(10,10),
 (11,10),(11,9),(10,9),(9,9),(8,9),(7,9),
 (7,8),(8,8),(9,8),(10,8)
]

W = [(4,7),(5,7),(6,7),(7,7)]

Successors:

S3  → SPILL
M11 → G
T19 → SPILL
W3  → SPILL

Initial frontier: M0.

Verified solution

Phase 1 — buy S:

SPACE,
DOWN, LEFT, UP, RIGHT,
SPACE

The four-move square fills S and returns to F.

Phase 2 — route M and cross the new shortcut:

UP, RIGHT,
UP, UP, UP, UP,
RIGHT, UP,
RIGHT, RIGHT, RIGHT,
SPACE

Phase 3 — use the outer route as the correctly sized flow clock:

UP,
RIGHT, RIGHT, RIGHT, RIGHT,
DOWN × 11,
LEFT × 4,
UP, UP,
RIGHT

Compact form:

S D L U R S |
U R U U U U R U R R R S |
U R R R R DDDDDDDDDDD L L L L U U R

Checkpoints:

Action 5: S completes; hopper 32.

Action 6: F returns to M.

Action 17: M0…M10 filled; surveyor at (2,7) adjacent to G.

Action 18: G points toward T.

Action 19: M11 fills and M compacts.

Actions 20–39: fill T0…T19.

Action 39: T compacts; hopper 0.

Action 40: move to (10,7).

Action 41: enter T19 and win.

Minimum solution depth: 41 actions.

Necessity and anti-cleanliness check

The short route through S is 11 movements from F to G.

The unbuilt outer route from the start to G is 31 movements.

Skipping S fills M and sends 19 units into W, leaving only 5 for the 20-cell T.

S costs four units but prevents nineteen wrong-branch releases.

Before G is prepared, the outer route is disastrous; afterward it is the useful route whose length matches T.

S is therefore neither an unconditional key nor an always-good shortcut. Its value depends on flow phase.

Level 7 — Weave the Network
Teaches and combines

A side branch may need to be completed, abandoned as a flow route, and later reused as movement terrain.

A familiar fork may need to be turned twice.

Earlier basins become the return route while sand passes transparently through them toward a distant final basin.

The final walkable path and the final flow path are different traversals of the same network.

Player-facing discovery

The opening resembles Level 6: build S, then use it to reach G before M finishes.

At G, however, the currently selected trunk N leads toward H but is not physically reachable from the surveyor’s side. The small side basin R forms the missing upper bridge. The player must:

turn G away from the apparent main route;

fill R;

turn G back toward N;

walk over R to reach H before N sends anything into Q.

After H is set toward E, the final journey reverses across R, then uses compacted N, M, and S as a continuous return path. Meanwhile, the sand route runs forward through M, N, and H into E.

Thus every constructed basin has at least two contextual roles.

Schematic
00 #############
01 ##########Q##
02 ######RR.#Q##
03 ###...R#.#Q##
04 ##....gN.#Q##
05 ##S#..MN..hE#
06 ##S#MMMN##NE#
07 ##S#MMMN##NE#
08 ..SMM##NNNNE#
09 .@fM#EEEEEEE#
10 #..##E#######
11 ##XEEE#######
12 ##EE#########
   0123456789012
Exact instance
Property	Value
Start	(9,1)
Exit	(11,2), underlying terrain E17
Hopper	45
Selection/cursor	none
Fork F	(9,2), branch 0→M0, branch 1→S0, initially 0
Fork G	(4,6), branch 0→N0, branch 1→R0, initially 0
Fork H	(5,10), branch 0→E0, branch 1→Q0, initially 1
Root	F, source inlet from south
Graph	F→S or F→M→G; G→R or G→N→H; H→E or H→Q

Stone:

{
 (8,0),(8,1),(9,0),(9,1),
 (10,1),(10,2),

 (4,2),(4,3),(4,4),(4,5),
 (5,4),(5,5),
 (3,3),(3,4),(3,5),

 (2,8),(3,8),(4,8),(5,8),(5,9)
}

Basins:

S = [(8,2),(7,2),(6,2),(5,2)]

M = [
 (9,3),(8,3),(8,4),(7,4),(6,4),
 (6,5),(7,5),(7,6),(6,6),(5,6)
]

R = [(3,6),(2,6),(2,7)]

N = [
 (4,7),(5,7),(6,7),(7,7),(8,7),
 (8,8),(8,9),(8,10),(7,10),(6,10)
]

E = [
 (5,11),(6,11),(7,11),(8,11),
 (9,11),(9,10),(9,9),(9,8),(9,7),(9,6),(9,5),
 (10,5),(11,5),(11,4),(11,3),
 (12,3),(12,2),(11,2)
]

Q = [(4,10),(3,10),(2,10),(1,10)]

Successors:

S3  → SPILL
M9  → G
R2  → SPILL
N9  → H
E17 → SPILL
Q3  → SPILL

Initial frontier: M0.

Verified solution
Phase 1 — build S, then route into M
SPACE,
UP, LEFT, DOWN, RIGHT,
SPACE
Phase 2 — cross S and prepare G for R
UP, RIGHT,
UP, UP, UP, UP,
RIGHT, RIGHT, RIGHT,
SPACE
Phase 3 — finish M, fill R, and restore G to N
DOWN, LEFT, UP, RIGHT,
SPACE
Phase 4 — cross R, reach H, and select E
UP, RIGHT, UP, RIGHT, RIGHT,
DOWN, DOWN, DOWN, RIGHT,
SPACE
Phase 5 — complete N, reverse through the built network, and reach E
LEFT,
UP, UP, UP,
DOWN, DOWN, DOWN,
LEFT,
DOWN, DOWN,
LEFT, LEFT, LEFT,
DOWN,
LEFT, LEFT, LEFT,
DOWN, DOWN,
RIGHT, DOWN

Compact form:

S U L D R S |
U R U U U U R R R S |
D L U R S |
U R U R R D D D R S |
L U U U D D D L D D L L L D L L L D D R D
Checkpoints

Action 5: S completes; hopper 41.

Action 6: F points back toward M.

Action 15: M0…M8 filled; surveyor reaches G.

Action 16: G points toward R.

Action 17: M9 fills; M compacts.

Actions 18–20: fill R0…R2; R compacts.

Action 21: G points back toward N.

Actions 22–30: fill N0…N8 while crossing R toward H.

Action 31: H points toward E.

Action 32: N9 fills; N compacts.

Actions 33–50: fill E0…E17; hopper reaches 0.

Action 51: move onto stone (10,2).

Action 52: enter E17 and win.

Minimum solution depth: 52 actions.

Necessity and anti-cleanliness check

S is the only initial connection from the start region to the G side.

R is not optional: without it, the surveyor cannot enter the region from which H is operable, even after N fills.

G must be turned to R and then back to N; leaving it in either orientation makes the exit unreachable or leaves N empty.

H must be turned before the last N cell is followed by a movement. Any unit entering Q leaves E incomplete because the hopper exactly equals:

S4 + M10 + R3 + N10 + E18 = 45

R is first a deliberately selected terminal sand branch, then a bridge to H, then part of the reverse route.

M is first a timed conduit, then a compacted return corridor.

N is first a flow trunk, then the middle of the final walking path.

The final phase revisits earlier terrain in reverse spatial order while sand continues forward through it. It is not a linear device errand.

8. Reference depths and curriculum
Level	Main discovery	Minimum actions
1	One footstep completes and compacts a basin	3
2	Partial fill remains blocked; blocked moves release nothing	6
3	SPACE selects a fork branch	7
4	Reach and prepare a fork before a wrong-branch release	27
5	A bridge is also a conduit; prepare a downstream fork	34
6	Spend sand on a shortcut; route length changes role by phase	41
7	Build, redirect, revisit, and reverse through one flow network	52

The first three levels isolate causal rules. Levels 4–7 add no unrelated device categories: all difficulty comes from basin placement, route length, fork accessibility, irreversible allocation, and the changing spatial role of completed basins.

9. Implementation invariants and test checkpoints

An implementation should satisfy all of the following.

Blocked movement invariant

From Level 2 after one RIGHT, a second RIGHT leaves:

position (5,3);

hopper 15;

filled set {A0};

frontier A1.

Valid SPACE invariant

At the Level 3 initial state, SPACE changes:

F.orientation: 1 → 0;

frontier B0 → A0;

while leaving the hopper at 14.

No double fill

In Level 4, the first movement after rotating F fills only U10. It must not also fill A0.

Compacted pass-through

Level 5 begins with P complete, yet the initial frontier is B0, proving that completed basin cells are skipped by flow traversal.

Downstream preview

In Level 5 at action 16, turning G changes its paddle to C, but the literal frontier remains A10. After action 17 fills A10, the new frontier is C0.

Spill consumption

If a selected terminal branch is complete and the hopper is nonempty, a successful movement decrements the hopper and produces a spill. It does not refill or alter the completed basin.

Zero-hopper movement

Level 6 actions 40–41 and Level 7 actions 51–52 remain legal despite an empty hopper.

Win independence

On Level 3, B may remain empty or partial when the surveyor reaches the exit. It is ignored by the validator.

No autonomous updates

Re-rendering the same settled state without calling step() cannot move sand, compact a basin, rotate a fork, decrement the hopper, or move the surveyor.

Determinism

Given the same level state and submitted action, the resulting gameplay state and animation sequence are identical.
