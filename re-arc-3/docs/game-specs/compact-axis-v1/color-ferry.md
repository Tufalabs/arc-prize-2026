Worked for 44m 1s
Color Ferry

This is a game about transporting one usable color becoming difficult because every recoloring changes which routes exist.

Color Ferry contains exactly 7 authored levels.

The player guides a small ferry cart carrying one pigment stone through a compact network of walls, basins, and color shutters. The carried stone is permanent: it cannot be dropped, exchanged, or stored. Entering a basin automatically recolors it. Every shutter matching the current stone color is traversable; every other shutter is barred.

The sole objective is to enter the visibly green ferry dock. The dock does not require a particular final color.

The game’s depth comes from the fact that recoloring is simultaneously:

a key acquisition;

a route closure;

a possible loss of access to the basin just used;

a commitment to a different region of the map;

and, later, a change in what the same junction or corridor is useful for.

No additional devices are introduced. The complete game uses only walls, floor, the agent and stone, immutable dye basins, color shutters, and the ferry dock.

1. Formal design integrity

Basins are permanent and may be entered any number of times. However, the authored levels do not pretend that revisiting the same basin with no other persistent state can create progress.

The complete dynamic state before completion is:

(agent position, carried color)

Therefore, returning to the same basin and receiving the same resulting color recreates an earlier complete state. Any route available after that revisit was already available on the first visit. Requiring such a loop would be fake depth.

Consequently:

previous colors are recovered from different basins of the same color;

physical junctions are revisited under different carried colors;

a basin may be harmless in one context and dangerous in another;

literal same-basin revisits are legal, but never falsely presented as progress-producing hidden history.

Level 6 uses this fact explicitly: a magenta basin is harmless when crossed while already magenta, but becomes a route-destroying overwrite if approached later while yellow.

2. Action space
Active actions

UP

DOWN

LEFT

RIGHT

Each active action attempts to move the agent by exactly one logical cell.

There are no click targets, no selected object, and no cursor state.

SPACE and cell clicks are not advertised as active actions. If a generic host nevertheless passes either event to step(), the state and frame remain unchanged and no feedback animation plays.

There is no wait action and no autonomous ticking.

3. Objective and termination

A level is completed when the agent successfully enters any logical cell marked D.

Exact validation:

complete := agent_position ∈ dock_cells

The following are irrelevant to victory:

the carried color on arrival;

which shutters are open;

which basins have been visited;

the route taken;

the number of actions used.

The player cannot win by standing adjacent to the dock or touching its rendered sprite. The agent’s logical position must enter a D cell.

After complete becomes true:

movement input is ignored;

the final state is frozen;

the completion animation plays;

no further gameplay state changes occur.

There is no in-world loss condition, timer, energy meter, or action budget. Exploration is unlimited. An external benchmark may impose an episode cap, but that cap is not part of the game state and must not be rendered or used by the validator.

4. Logical board and coordinate system

Every level uses a 17×17 logical board.

Coordinates are written as (row, column):

row 1 is the top;

row 17 is the bottom;

column 1 is the left;

column 17 is the right;

coordinates are 1-indexed.

There is no wrapping.

Authored-map notation

The following characters appear only in this specification. No characters or text are rendered in the game.

Symbol	Meaning
#	Impassable wall
.	Ordinary floor
S	Agent’s initial floor cell
B	Blue shutter
Y	Yellow shutter
M	Magenta shutter
b	Blue dye basin
y	Yellow dye basin
m	Magenta dye basin
D	Ferry dock cell

The maps are the authoritative static layouts. Every non-# cell is traversable subject to shutter collision rules.

5. Dynamic state
Static per-level data
terrain[17][17]
gate_color[cell]   ∈ {BLUE, YELLOW, MAGENTA}, where applicable
basin_color[cell]  ∈ {BLUE, YELLOW, MAGENTA}, where applicable
dock_cells
start_position
start_color
Dynamic data
agent_position
carried_color ∈ {BLUE, YELLOW, MAGENTA}
complete      ∈ {false, true}

There is no:

inventory;

basin depletion;

shutter latch state;

visit history;

movement direction state;

hidden phase;

selected object;

cursor;

timer;

random seed.

Shutter openness is derived rather than independently stored:

gate_is_open(gate) := gate.color == carried_color

An implementation may cache the rendered open state, but its logical value must always equal this expression.

6. Exact action resolution

For an arrow action with displacement delta:

If complete == true, return the unchanged completed state.

Compute target = agent_position + delta.

If target is outside the board or is #, do not move.

If target contains a shutter whose color differs from carried_color, do not move.

Otherwise set agent_position = target.

If target contains a basin, set:

carried_color = basin_color[target]

This assignment occurs even if the stone already has that color, although the value then remains unchanged.

Recompute the visible state of every shutter from the new carried_color.

If target ∈ dock_cells, set complete = true.

Return the final settled observation after any renderer-only transition frames.

Reference pseudocode:

Python
Run
def step(action):
    if state.complete:
        return state

    if action not in {UP, DOWN, LEFT, RIGHT}:
        return state

    target = state.agent_position + DELTA[action]

    if outside_board(target) or terrain[target] == WALL:
        play_wall_bump()
        return state

    if is_gate(target) and gate_color[target] != state.carried_color:
        play_gate_mismatch(target)
        return state

    state.agent_position = target
    old_color = state.carried_color

    if is_basin(target):
        state.carried_color = basin_color[target]

    if state.carried_color != old_color:
        play_recolor_transition(old_color, state.carried_color)
    elif is_basin(target):
        play_same_color_basin_pulse(target)

    if target in dock_cells:
        state.complete = True
        play_completion()

    return state

A closed shutter blocks entry from every direction. It is a shutter occupying a route cell, not a directional wall segment. This makes shutters on corners behave identically to shutters in straight corridors.

7. Rendering contract
Canvas and board placement

Canvas: 64×64 pixels

Canvas background: color 5

Logical cell size: 3×3 pixels

Board origin: pixel (6, 6), with (x, y) ordering

Board footprint: 51×51 pixels

Board pixels therefore occupy x = 6…56, y = 6…56

Logical cell (row, column) maps to:

x0 = 6 + 3 × (column - 1)
y0 = 6 + 3 × (row - 1)

Its base rectangle is:

x = x0…x0+2
y = y0…y0+2

No grid lines or per-cell borders are rendered. Adjacent walls and floors form uninterrupted connected regions.

Restrained palette usage
Role	Color IDs
Outer background	5 black
Walls	4 very dark gray, with occasional connected-edge highlight 3
Floor	1 light gray
Agent body	0 white, shadow 3
Blue pigment	9, motif accent 10
Yellow pigment	11, motif accent 12
Magenta pigment	6, motif accent 7
Ferry dock	14 green
Invalid-shutter pulse	8 red

Colors 13 and 15 are unused.

Color motifs

Color is reinforced by a shared pattern on the carried stone, corresponding basins, and corresponding shutters.

Blue

Base: 9

Horizontal light-blue stripe: 10

9  9  9
10 10 10
9  9  9
Yellow

Base: 11

Vertical orange stripe: 12

11 12 11
11 12 11
11 12 11
Magenta

Base: 6

Descending pink diagonal: 7

7 6 6
6 7 6
6 6 7

Thus a player need not rely on hue alone.

Agent and carried stone

The agent is rendered as a connected 5×5 ferry-cart sprite centered on its logical cell. Its collision footprint remains exactly one logical cell.

The upper central 3×3 portion is the carried pigment stone and uses the complete color motif. White side brackets and a white lower chassis make it clear that the colored region is mounted on the same persistent agent.

Conceptual sprite:

. s s s .
0 s s s 0
0 s s s 0
. 0 0 0 .
. 3 . 3 .

s is the corresponding 3×3 pigment motif.

. is transparent.

Visual extension into neighboring wall pixels has no collision meaning.

The agent is drawn above floor, basins, and open shutters.

When standing on a basin, a basin-colored underfoot halo remains visible around the agent, so the settled frame still identifies the underlying cell.

Basins

A basin is a shallow floor feature, not a blocking object.

It uses a colored 3×3 concave ring.

Its center remains floor-colored.

Its color motif is visible on the rim.

When occupied by the agent, a larger sparse halo remains visible around the cart.

Basins never empty, dim, crack, or change state.

Shutters
Closed shutter

A solid 3×3 colored shutter fills the cell.

Two dark bars cross the motif.

It visually blocks the floor connection.

A closed shutter cannot be entered from any side.

Open shutter

The colored shutter retracts to small side brackets.

The floor-colored center becomes visibly clear.

The color motif remains on the brackets, preserving identity.

The cell is traversable like ordinary floor.

Every matching shutter is visibly open in every settled observation.

Ferry dock

Dock cells form one connected green platform.

Base color: 14

Dark water or recess immediately beyond the platform: 5

A persistent white cart-and-stone silhouette is embedded in the platform.

The silhouette is color-neutral and communicates “bring the cart here,” not “arrive with a particular pigment.”

The level-1 dock is already visible from the starting chamber.

8. Animation and feedback

All logical state changes occur atomically during step(). Animation frames only display the committed transition and never advance gameplay.

Ordinary successful movement

Ordinary movement is rendered directly in the final position. No decorative movement animation is required.

Basin recoloring

When entering a basin whose color differs from the carried color:

The agent appears on the basin.

The basin rim emits one bright pulse using its accent color.

The carried stone immediately adopts the new motif.

Previously matching shutters begin extending.

Newly matching shutters retract.

The final settled frame shows all shutter states updated.

Use three deterministic transient frames:

Frame 1: stone changes; basin pulses
Frame 2: all affected shutters halfway
Frame 3: final open/closed shutter positions

The state is logically committed before frame 1. These frames consume no actions.

Same-color basin entry

If the agent enters a basin matching its existing color:

the agent moves normally;

the basin gives one low-intensity rim pulse;

the stone and shutters remain unchanged;

no full recoloring animation plays.

Closed-shutter collision

A blocked move:

leaves position and color unchanged;

flashes the struck shutter’s outer edge with color 8;

returns it to its ordinary barred appearance after two transient frames.

This pulse is spatially local to the struck shutter.

Wall collision

A wall collision leaves the state unchanged and gives the agent body a one-frame gray compression. Walls do not flash like shutters.

Completion

The logical state becomes complete immediately upon entering D.

Then:

the whole dock flashes green and white;

the agent sprite contracts into the dock’s cart-and-stone emblem;

the final completed frame remains visible.

The completion animation consumes no actions.

9. Progression summary
Level	Main discovery or composition	Verified canonical length
1	Matching stone and shutter colors determine passage	4
2	Entering a basin automatically changes the carried color and every shutter	8
3	Different basins overwrite with their displayed colors; a previous color can be recovered elsewhere	12
4	An accessible basin can be a hazard that must be bypassed	29
5	Magenta disproves a simple two-color toggle hypothesis; a basin becomes a transfer station	45
6	A basin harmless in one phase becomes destructive in another; hubs change meaning with color	56
7	Two interlocked hubs must be traversed under alternating colors	72

In solution strings below, U, D, L, and R mean the four movement actions. Vertical bars separate explanatory chunks and are not actions.

10. Authored levels
Level 1 — Two Shutters
Player-facing discovery

The agent begins with a blue stone between two geometrically identical horizontal shutters:

the blue shutter on the right is visibly retracted;

the yellow shutter on the left is visibly barred.

The ferry dock is visible above the right-hand route.

The player can perform two low-cost contrasting experiments:

LEFT strikes the yellow shutter and produces a red mismatch pulse;

RIGHT enters the blue shutter.

Because both shutters have the same silhouette and orientation, the observation cannot reasonably be explained by gate orientation. The visible difference is their relationship to the carried stone.

Initial state
agent_position = (9, 8)
carried_color = BLUE
complete = false
selection = none
cursor = none

Objects:

Yellow shutter: (9, 7)

Blue shutter: (9, 9)

Dock: (7, 9), (7, 10)

Basins: none

Exact map
#################
#################
#################
#################
#################
#################
########DD#######
#########.#######
#####.YSB.#######
#################
#################
#################
#################
#################
#################
#################
#################
Canonical solution
R | RUU

Full sequence:

RRUU

Resolution:

Enter the open blue shutter at (9, 9).

Move to (9, 10).

Move up to (8, 10).

Enter the dock at (7, 10).

Verified length: 4 actions.

Attribution and necessity check

The yellow shutter is an informational comparison, not decorative clutter.

The two shutters are the same shape and orientation.

There are no basins or simultaneous state changes.

The dock is visible before the first action.

Removing the yellow shutter would make “the colored opening is simply passable” a viable mistaken explanation.

Random movement cannot complete the level directly because the left and downward directions do not lead to the dock.

Level 2 — The Basin Repaints the Stone
Player-facing discovery

The agent begins yellow.

The yellow shutter to the left is visibly open.

The blue shutter on the route to the dock is visibly closed.

A blue basin is directly below the agent.

Known movement and shutter rules produce a local impasse. Stepping down is the cheapest unexplained experiment.

On entering the basin:

the mounted stone becomes blue;

the yellow shutter closes;

the blue shutter opens;

the basin remains visibly full.

The simultaneous swap and visible stone recoloring distinguish this from an ordinary local floor switch.

Initial state
agent_position = (10, 7)
carried_color = YELLOW
complete = false
selection = none
cursor = none

Objects:

Yellow shutter: (10, 6)

Blue shutter: (10, 9)

Blue basin: (11, 7)

Dock: (7, 10), (7, 11)

Exact map
#################
#################
#################
#################
#################
#################
#########DD######
#########.#######
#########.#######
####.YS.B.#######
######b##########
#################
#################
#################
#################
#################
#################
Canonical solution
D | URR | RUUU

Full sequence:

DURRRUUU

Resolution:

D: enter the blue basin at (11, 7).

The stone changes from yellow to blue.

Return upward and cross the newly opened blue shutter.

Follow the upper route into dock cell (7, 10).

Verified length: 8 actions.

Attribution and necessity check

The basin is adjacent to the initial impasse.

Its consequence is immediate and board-wide but visually coherent.

No second unfamiliar object changes simultaneously.

The blue basin is required; without it, the dock is unreachable.

The yellow shutter exists to show that the old color’s routes close at the same moment the new ones open.

The level remains recoverable because the basin is adjacent to the start chamber.

Level 3 — Blue, Yellow, Blue
Player-facing discovery

The player now encounters both basin colors in one route.

The initial blue stone opens the first blue shutter. A yellow basin then changes the active route family. Later, a different blue basin restores blue and makes a final blue shutter passable.

This level establishes that:

a basin sets the stone to its displayed color;

color change is not one-way;

previously used colors can be recovered at later sources;

the active route family may need to alternate.

The final blue shutter makes the second basin’s effect unavoidable and observable.

Initial state
agent_position = (11, 5)
carried_color = BLUE
complete = false
selection = none
cursor = none

Objects:

Blue shutters: (11, 6), (7, 12)

Yellow shutter: (9, 9)

Yellow basin: (9, 7)

Blue basin: (7, 10)

Dock: (6, 13), (7, 13)

Exact map
#################
#################
#################
#################
#################
############D####
#########b.BD####
#########.#######
######y.Y.#######
######.##########
####SB.##########
#################
#################
#################
#################
#################
#################
Canonical solution
R | RUU | RR | RUU | RR | R

Full sequence:

RRUURRRUURRR

Landmarks:

Cross blue shutter (11, 6).

Enter yellow basin (9, 7).

Cross yellow shutter (9, 9).

Enter blue basin (7, 10).

Cross blue shutter (7, 12).

Enter dock (7, 13).

Verified length: 12 actions.

Attribution and necessity check

The yellow and blue basin effects occur at separated points, so each change is attributable.

The final blue shutter prevents a “the second basin is decorative” interpretation.

Both basin colors are necessary.

There are no optional objects or alternate routes.

The level is intentionally linear because its purpose is to establish color alternation before route-choice puzzles begin.

Level 4 — The Basin You Avoid
Player-facing discovery

The agent starts blue and must first enter the yellow basin at (10, 6).

From there, two visible routes point toward the yellow shutter:

a short central route crosses the blue basin at (10, 9);

a longer upper route preserves yellow.

Taking the central route immediately repaints the stone blue. The yellow shutter at (7, 12) visibly bars itself before the player reaches it. The mistake is recoverable: the player can return to the yellow basin and try the upper route.

Beyond the yellow shutter, a second blue basin at (12, 14) is mandatory. The same blue-basin rule therefore plays contrasting roles:

the first blue basin is a premature overwrite to avoid;

the second blue basin is the required transfer for the final blue shutter.

Initial state
agent_position = (13, 5)
carried_color = BLUE
complete = false
selection = none
cursor = none

Objects:

Yellow basin: (10, 6)

Hazardous blue basin: (10, 9)

Yellow shutter: (7, 12)

Required blue basin: (12, 14)

Blue shutter: (12, 11)

Dock: (13, 8), (14, 8)

Exact map
#################
#################
#################
#################
#################
#####......######
#####.####.Y.####
#####.##...#.####
#####.##.###..###
#####y..b####.###
####..#######.###
####.#####B..b###
####S##D...######
#######D#########
#################
#################
#################
Canonical solution
UURU | UUUURRRRRDR | RDDRDDD | LLL | DLLL

Full sequence:

UURUUUUURRRRRDRRDDRDDDLLLDLLL

Landmarks:

UURU: enter the yellow basin at (10, 6).

Take the upper bypass, preserving yellow.

Cross yellow shutter (7, 12).

Descend through the irregular eastern route.

Enter required blue basin (12, 14).

Move left through blue shutter (12, 11).

Turn downward and back left into dock (13, 8).

Verified length: 29 actions.

Anti-cleanliness and necessity check

The shortest-looking geometric route is not the valid color route.

The hazardous basin is not a cosmetic decoy: replacing it with ordinary floor shortens the solution and removes the preservation problem.

The mistake is locally recoverable through the already known yellow basin.

The required eastern blue basin cannot be omitted because the final blue shutter would remain barred.

The dock lies left of the final transfer even though the player must first travel far right.

Objects are not encountered in a simple left-to-right usage order.

Level 5 — The Magenta Transfer
Player-facing discovery

This level introduces magenta as a third value of the already established basin-and-shutter rule.

The player first uses familiar blue and yellow routing. After crossing the second yellow shutter at (7, 11), ordinary yellow movement reaches a local end. The only productive nearby object is the magenta basin at (7, 13).

Entering it produces unmistakable evidence:

the stone becomes magenta rather than returning to blue;

both yellow shutters close;

both magenta shutters open;

the route behind becomes unavailable;

a different route through the lower half of the map becomes usable.

This disproves a simple “basins toggle between blue and yellow” hypothesis.

The magenta basin functions as a transfer station: the player enters its region using yellow and leaves using magenta.

Initial state
agent_position = (14, 4)
carried_color = BLUE
complete = false
selection = none
cursor = none

Objects:

Blue shutters: (12, 4), (13, 12)

Yellow basin: (8, 4)

Yellow shutters: (6, 8), (7, 11)

Magenta basin: (7, 13)

Magenta shutters: (11, 11), (12, 7)

Blue basin: (13, 10)

Dock: (14, 13), (14, 14)

Exact map
#################
#################
#################
#################
#################
####...Y#########
####.##.#.Y.m####
###y.##.#.##.####
###.##....##.####
###.##.#..##.####
###.##.##.M..####
###B##M##########
###.##...b.B.####
###S########DD###
#################
#################
#################
Canonical solution
UU | UUUU | RUURRR | DDDRRUUR | RR
| DDDDLL | LUULLLDDD | DRRR | RR | RD

Full sequence:

UUUUUURUURRRDDDRRUURRRDDDDLLLUULLLDDDDRRRRRRD

Landmarks:

Cross initial blue shutter (12, 4).

Reach yellow basin (8, 4).

Cross yellow shutter (6, 8).

Loop southeast and cross yellow shutter (7, 11).

Enter magenta basin (7, 13).

Descend and cross magenta shutter (11, 11).

Return west and cross magenta shutter (12, 7).

Enter blue basin (13, 10).

Cross final blue shutter (13, 12).

Enter the left dock lip at (14, 13).

Verified length: 45 actions.

There is an equivalent same-length ending through the other connected dock lip.

Anti-cleanliness and necessity check

Magenta is encountered at a local yellow-route impasse rather than introduced by description.

Its effect changes multiple already visible shutters at once.

The player travels east, returns west under magenta, and then travels east again under blue.

The magenta basin is necessary; without it, the lower magenta route is inaccessible.

The final blue basin is necessary because the dock approach contains a blue shutter.

The initial blue shutter and first yellow shutter prevent premature access to later regions and visibly close behind after recoloring.

The board is a set of interlocking loops rather than a list of stations arranged in usage order.

Level 6 — The Harmless Basin Becomes a Trap
Player-facing discovery

This level composes three established ideas:

junctions change meaning when revisited under another color;

a basin can be crossed harmlessly when it matches the current stone;

the same basin can become destructive in a later phase.

The early central loop works as follows:

Start magenta and pass the lower magenta shutter.

Reach yellow basin (10, 4).

Use two yellow-controlled entries to reach blue basin (6, 10).

Return through the central region under blue.

Reach magenta basin (12, 10).

The path from that basin to the yellow basin at (14, 14) crosses a second magenta basin at (12, 14). On this first pass, the stone is already magenta, so the second basin merely gives a small same-color pulse.

After entering the yellow basin, the geometrically short route back upward crosses that same magenta basin again. Doing so would restore the exact earlier magenta state and close the inner yellow shutter at (9, 14). The player must instead use the longer outer route through yellow shutter (14, 15).

The basin therefore changes role entirely through context without changing its operational rule.

Initial state
agent_position = (16, 2)
carried_color = MAGENTA
complete = false
selection = none
cursor = none

Objects:

Magenta shutters: (15, 2), (12, 11)

Yellow basins: (10, 4), (14, 14)

Yellow shutters: (8, 7), (7, 8), (9, 14), (14, 15)

Blue basins: (6, 10), (8, 14)

Blue shutters: (8, 9), (9, 8), (7, 14)

Magenta basins: (12, 10), (12, 14)

Dock: (2, 14), (3, 14), (4, 14)

Exact map
#################
#############D###
#############D###
#######...###D###
#######.#.###.###
#######.#b..#.###
#######Y###.#B###
###...Y.B...#b..#
###.###B#####Y#.#
###y###...###.#.#
###.#####.###.#.#
#...#####mM..m#.#
#.###########.#.#
#.###########yY.#
#M###############
#S###############
#################
Canonical solution
U | UUURRUU | UURRR | RU | UUURRDD
| RRDDLLL | LD | DRRDD | R | RRR
| DD | R | RUUUUUULL | U | UUU

Full sequence:

UUUURRUUUURRRRUUUURRDDRRDDLLLLDDRRDDRRRRDDRRUUUUUULLUUUU

Landmarks:

Cross magenta shutter (15, 2).

Reach yellow basin (10, 4).

Cross yellow shutter (8, 7).

Cross yellow shutter (7, 8).

Reach blue basin (6, 10).

Return through blue shutter (8, 9).

Exit through blue shutter (9, 8).

Reach magenta basin (12, 10).

Cross magenta shutter (12, 11).

Cross same-color magenta basin (12, 14).

Enter yellow basin (14, 14).

Cross the outer yellow shutter (14, 15).

Follow the outer eastern route and enter blue basin (8, 14).

Cross blue shutter (7, 14).

Enter dock (4, 14).

Verified length: 56 actions.

Tempting wrong experiment

From yellow basin (14, 14), moving upward toward the visibly nearer inner yellow shutter requires crossing magenta basin (12, 14).

The immediate result is:

stone changes yellow → magenta;

inner yellow shutter (9, 14) closes;

the shortcut visibly fails.

The player can return downward to the yellow basin and recover, so the experiment is informative rather than arbitrarily fatal.

Anti-cleanliness and necessity check

The second magenta basin is part of the successful route in its harmless role.

It also invalidates the shorter return route after the yellow transfer.

Replacing it with ordinary floor produces a shorter valid solution, so it is mechanically consequential rather than decorative.

The central junction is used first under yellow and later under blue.

The physically short inner approach conflicts with the required color-preserving outer approach.

The final blue basin and shutter make the last color transition necessary.

The dock is visible at the top, but reaching it requires moving down, across, back up, and around an offset bottleneck.

Level 7 — Braided Harbors
Player-facing discovery

The final level contains two unmarked central floor junctions:

west hub at (8, 6);

east hub at (8, 12).

They are ordinary floor, not new object types.

Each hub is encountered twice, under different carried colors. Its accessible arms therefore change without the hub itself changing state.

The required color braid is:

BLUE → YELLOW → MAGENTA → BLUE → YELLOW → MAGENTA → BLUE

The route repeatedly crosses the center in directions that conflict with the visual position of the dock.

Structural phases
First approach to the west hub: yellow

The initial blue route reaches yellow basin (12, 4). Yellow shutter (8, 5) admits the agent to the west hub at (8, 6).

At this time:

the northern yellow arm is open;

the eastern blue arm is closed.

First approach to the east hub: magenta

The northern yellow arm reaches magenta basin (4, 10). A magenta shutter then carries the route into the east hub at (8, 12).

At this time:

the southern magenta arm is open;

the western and eastern yellow arms are closed.

Return to the west hub: blue

The southern magenta route reaches blue basin (12, 8). The player returns to the west hub through blue shutters.

The west hub now has the opposite useful arm:

its eastward blue route is open;

its previously used yellow arms are closed.

Return to the east hub: yellow

The blue route reaches yellow basin (8, 10), immediately west of the east hub. Yellow shutters admit the agent through the hub and out its eastern side.

Final outer transfer

The eastern yellow route reaches magenta basin (4, 14). Magenta opens the far-right descending route to blue basin (12, 16). Blue then opens the final dock approach.

No hub, basin, or shutter has a special-case rule. All role changes arise from position and carried color.

Initial state
agent_position = (16, 2)
carried_color = BLUE
complete = false
selection = none
cursor = none

Objects:

Blue shutters: (15, 2), (11, 8), (8, 7), (13, 16)

Yellow basins: (12, 4), (8, 10)

Yellow shutters: (8, 5), (7, 6), (8, 11), (8, 13)

Magenta basins: (4, 10), (4, 14)

Magenta shutters: (5, 10), (9, 12), (4, 15)

Blue basins: (12, 8), (12, 16)

Dock: (16, 8), (16, 9), (16, 10)

West hub: ordinary floor (8, 6)

East hub: ordinary floor (8, 12)

Exact map
#################
#################
#################
#####....m###mM.#
#####.###M###.#.#
#####.###...#.#.#
#####Y#####.#.#.#
###.Y.B..yY.Y.#.#
###.#.#####M###.#
###.#...###.###.#
###.###B###.###.#
#..y###b....###b#
#.#############B#
#.#############.#
#B#############.#
#S#####DDD......#
#################
Canonical solution
U | UUURR | UUUUR | RU | UUURRRR
| D | DRRDDD | DDDLLLL | U | ULLUUR
| RRR | R | RR | RUUUU | R
| RDDDDDDDD | D | DDDLLLLLL

Full sequence:

UUUURRUUUURRUUUURRRRDDRRDDDDDDLLLLUULLUURRRRRRRRUUUURRDDDDDDDDDDDDLLLLLL

Landmarks:

Cross initial blue shutter (15, 2).

Reach yellow basin (12, 4).

Cross yellow shutter (8, 5).

Leave the west hub through yellow shutter (7, 6).

Reach magenta basin (4, 10).

Cross magenta shutter (5, 10).

Leave the east hub through magenta shutter (9, 12).

Reach blue basin (12, 8).

Cross blue shutter (11, 8).

Return to the west hub through blue shutter (8, 7).

Reach yellow basin (8, 10).

Cross yellow shutter (8, 11) into the east hub.

Cross yellow shutter (8, 13) out of the east hub.

Reach magenta basin (4, 14).

Cross magenta shutter (4, 15).

Descend the far-right route to blue basin (12, 16).

Cross blue shutter (13, 16).

Descend and return left into dock cell (16, 10).

Verified length: 72 actions.

Anti-cleanliness and necessity check

The two central hubs are revisited under different colors, not repeated complete states.

The west hub’s useful exit changes from north to east.

The east hub’s useful exit changes from south to east.

Basins are distributed around the map rather than placed in their usage order.

The dock is near the start in the bottom portion of the board, but no direct floor connection exists.

The solution repeatedly moves away from the dock to obtain the color needed for a later return.

Every basin in the level is on the verified solution.

Every final approach is protected by the color obtained immediately before it.

Removing the central blue or yellow shutters creates major premature shortcuts through the braid.

The final blue shutter makes the far-right blue basin necessary.

The solution is not a checklist: what matters is which arm of each already visited junction is reachable during each color phase.

11. Final implementation invariants

An implementation is conformant only if all of the following hold:

Exactly 7 levels are present.

The maps above are reproduced exactly.

The agent moves exactly one logical cell per arrow action.

Basins recolor automatically on entry.

Basins are permanent and reusable.

Shutter openness depends only on equality with the carried color.

All matching shutters update simultaneously after recoloring.

Closed shutters block entry from every direction.

The previous color’s shutters close immediately after recoloring.

No logical state changes between player actions.

Animation frames do not consume actions.

The settled frame exposes the agent position, current color, basin colors, shutter colors, shutter openness, walls, floor, and dock.

There is no hidden history, randomness, timer, inventory, cursor, selection, or independent shutter state.

Entering any dock cell completes the level regardless of carried color.

SPACE and clicks have no gameplay effect.

Invalid moves never alter position, pigment, shutters, or basin state.

The game has no internal exploration budget or loss state.

Do you like this personality?
