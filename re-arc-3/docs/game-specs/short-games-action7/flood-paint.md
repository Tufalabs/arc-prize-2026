Flood Paint

Identity: This is a game about changing one connected paint mass to a chosen color becoming difficult because every choice reshapes all of its future frontiers at once.

Exact authored level count: 7.

Flood Paint has one manipulated object: a visibly stamped, wet-edged paint region. The player clicks a colored petal on a flower-shaped selector. The whole region adopts that color, then recursively absorbs every orthogonally connected unabsorbed cell of the chosen color that it can reach from any part of its boundary.

The early game asks, “What does a flood include?” The later game asks, “Which fronts should be made ready before I spend this color?”

There is no avatar, movement, inventory, switch network, or hidden state. Difficulty comes entirely from the geometry of the growing region and the fact that one color choice acts globally across all of its fronts.

1. Rule inventory

There are only three gameplay rules.

Choose a color. Click a raised color petal. The current-color petal is depressed and cannot be selected again until the region first changes to another color.

Recolor and flood. Every absorbed cell changes to the selected color. Every matching unabsorbed cell orthogonally connected to any part of the region is then absorbed recursively.

Spend a droplet. Beginning in Level 3, every valid non-current color choice consumes one visible droplet, even if that choice absorbs zero new cells.

Everything else—corridors, multiple fronts, delayed unions, large tempting patches, and split growth—comes from placement and connectivity under these same rules.

2. Inputs

The full provided action space is respected:

UP

DOWN

LEFT

RIGHT

SPACE

click on any rendered cell

Only clicks on visible flower petals are gameplay actions.

Directional and SPACE actions

UP, DOWN, LEFT, RIGHT, and SPACE do not alter gameplay state. On step(), they produce a one-frame gray pulse around the flower center so the input is visibly acknowledged.

They do not consume a droplet.

Click actions

A click is decoded on step().

Click inside a visible, raised petal: commit that color change.

Click the depressed current-color petal: invalid; no state or budget change.

Click the flower center, board, background, droplet row, absent petal slot, or any other pixel: invalid; no state or budget change.

Invalid clicks produce a brief neutral ripple at the clicked location. Thus an unsuccessful interaction is visually distinguishable from a valid zero-growth color change.

There is no persistent cursor or hover state. The only persistent selection state is the depressed current-color petal.

3. Objective and terminal validation

Let paintable be the set of all non-void logical board cells and absorbed the stamped region.

Success

A level succeeds exactly when:

absorbed == paintable

The following are irrelevant to success:

final region color;

remaining droplets;

number of invalid clicks;

color-choice sequence;

animation phase once the settled state is reached.

Completing the board with the last droplet is allowed. Success is checked before exhaustion failure.

Failure

Levels 1 and 2 have no droplet limit and therefore no budget failure.

In Levels 3–7, failure occurs exactly when:

droplets_remaining == 0
and absorbed != paintable

On failure:

the empty droplet row remains visible;

remaining unabsorbed cells remain in their original colors and remain unstamped;

the remaining unabsorbed cells flash maroon once;

further actions are no-ops until the external harness resets the level.

Void cells, selector pixels, HUD pixels, and animation overlays are ignored by validation.

4. Native geometry and rendering
Logical board

Each level defines a rectangular logical lattice with integer coordinates:

x = 0 ... W-1
y = 0 ... H-1

(0,0) is the top-left logical position.

A coordinate is either:

a paintable colored cell, or

void.

Only orthogonal adjacency counts:

(x-1,y), (x+1,y), (x,y-1), (x,y+1)

Diagonal contact never connects paint.

Canvas mapping

The render canvas is exactly 64×64.

The board is centered inside a 45×45 viewport occupying pixels x=1..45, y=1..45.

For a board of size W×H:

board_origin_x = 1 + floor((45 - 3W) / 2)
board_origin_y = 1 + floor((45 - 3H) / 2)

Logical cell (x,y) occupies the connected 3×3 pixel block:

px = board_origin_x + 3x ... board_origin_x + 3x + 2
py = board_origin_y + 3y ... board_origin_y + 3y + 2

Adjacent cells have no gap and no internal outline. There is no rendered logic-grid border.

Void pixels use color 4, very dark gray.

Paint colors

The gameplay uses five saturated colors:

Spec symbol	Color ID	Meaning
B	9	blue
Y	11	yellow
G	14	green
O	12	orange
P	15	purple

The letters occur only in this implementation document. No letters or text are rendered in the game.

Unabsorbed-cell textures

An unabsorbed cell is filled with its color and receives two color-5 texture pixels. Local coordinates are inside its 3×3 block:

Color	Texture pixels
B	(1,0), (1,2)
Y	(0,1), (2,1)
G	(2,0), (0,2)
O	(0,0), (2,0)
P	(0,2), (2,2)

These repeated motifs make same-colored logical cells inspectable without adding grid borders.

Absorbed-region rendering

Every absorbed cell is filled with current_color, regardless of its original color.

Each absorbed cell carries a persistent sprout-shaped membership stamp:

local (1,1) = color 0
local (1,2) = color 0
local (0,2) = color 1
local (2,2) = color 1

Unabsorbed cells never carry this stamp.

Wet boundary

After filling the region, determine which sides of each absorbed cell touch either void, canvas exterior, or an unabsorbed cell.

For every exposed side, draw a one-pixel-thick color-10 line along that complete side of the cell. Draw the membership stamp afterward so it remains visible.

Do not draw boundary lines between two absorbed cells.

The resulting boundary surrounds only the present region, not individual logical cells. It makes every active frontier directly inspectable in the settled frame.

5. Flower selector

The selector is a flower-like control in the right-hand panel. Its center occupies pixels:

x=54..58, y=8..12

The center is filled with current_color and uses the same texture family as that color.

The five petal hit regions are fixed:

Color	Hit region
B / 9	x=55..58, y=2..5
Y / 11	x=60..63, y=7..10
G / 14	x=58..61, y=15..18
O / 12	x=50..53, y=15..18
P / 15	x=48..51, y=7..10

Each visible petal is a connected 4×4 cluster with the corresponding texture enlarged onto it. Neutral color-3 stems connect petals to the center but are inert.

Only colors in the level’s fixed palette are rendered as petals. Missing colors leave background-colored space, not inert fake petals.

The current-color petal is rendered one pixel visually inset, with a color-3 inner rim. Its hit region remains unchanged, but clicking it is invalid.

After a valid choice:

the chosen petal becomes depressed;

the previous petal rises;

the center changes to the chosen color.

Clicking a petal is the direct physical implementation of “make the active region this color.” It is not an auxiliary UI action.

6. Droplet tray

Levels 1 and 2 do not render droplets and use:

droplets_remaining = None

Levels 3–7 render one finite row.

For budget N, droplet i, with i=0...N-1, is placed at:

droplet_x(i) = 2 + i + floor(i / 5)

The extra column after every five droplets visibly groups the count without text.

Each droplet occupies:

tip:  (droplet_x, 52)
body: (droplet_x, 53), (droplet_x, 54)

Filled droplet:

tip color 10;

body color 9.

Empty droplet:

tip color 4;

body color 2.

Droplets empty from right to left, leaving a contiguous prefix of filled droplets. The maximum authored budget is 39, which fits through pixel x=47.

A valid non-current petal click consumes exactly one droplet whether or not any new cell is absorbed. Invalid input does not consume one.

7. Exact state-machine contract
Persistent state
level_id: int
W, H: int
paintable: set[(x,y)]
base_color: map[(x,y) -> one of 9,11,14,12,15]
absorbed: set[(x,y)]
current_color: one of 9,11,14,12,15
level_palette: fixed set of colors
droplets_remaining: int or None
status: ACTIVE | WON | FAILED

base_color never changes. It is consulted only while a cell is unabsorbed.

At level load:

absorbed is exactly the set specified by the level, always one S cell in these authored instances;

no automatic flood runs at initialization;

all authored starts avoid an orthogonally adjacent unabsorbed cell of the initial current color.

Valid transition

For a valid clicked color c != current_color:

Save old_absorbed = absorbed.

Set current_color = c.

If budgeted, decrement droplets_remaining by one.

Initialize a queue with every unabsorbed cell u satisfying:

base_color[u] == c, and

u is orthogonally adjacent to any cell in old_absorbed.

Recursively traverse only unabsorbed cells whose base_color is c.

Add every reached cell to absorbed.

Recompute the wet boundary and petal depression.

Check success.

If not successful, check budget failure.

Equivalent pseudocode:

Python
Run
def apply_color(c):
    assert status == ACTIVE
    assert c in level_palette
    assert c != current_color

    previous_region = set(absorbed)
    current_color = c

    if droplets_remaining is not None:
        droplets_remaining -= 1

    queue = []
    reached = set()

    for p in previous_region:
        for q in orthogonal_neighbors(p):
            if (
                q in paintable
                and q not in previous_region
                and base_color[q] == c
            ):
                reached.add(q)
                queue.append(q)

    while queue:
        p = queue.pop(0)
        for q in orthogonal_neighbors(p):
            if (
                q in paintable
                and q not in previous_region
                and q not in reached
                and base_color[q] == c
            ):
                reached.add(q)
                queue.append(q)

    absorbed |= reached

    if absorbed == paintable:
        status = WON
    elif droplets_remaining == 0:
        status = FAILED

A matching component can therefore be absorbed from several distant boundary faces during the same action. All those cells join the same recolored region.

No other colors chain-react.

8. Animation and feedback

All gameplay state changes occur during the action’s step(). Animation frames do not call additional state updates and do not consume actions or droplets.

Input is locked until the settled frame is reached.

Valid flood animation

The transition is computed atomically, then presented as:

Selection frame: chosen petal depresses, flower center changes, and one droplet drains.

Region frame: all previously absorbed cells visibly recolor together.

Flood-wave frames: newly reached cells are grouped by shortest orthogonal distance through target-colored unabsorbed cells from old_absorbed.

Distance layer 1 stamps first, then layer 2, and so on.

Cells in the same layer animate simultaneously.

The wet boundary is recomputed after every displayed layer.

The final settled frame is the authoritative observable state.

Each distance layer lasts one render frame. A long connected corridor therefore visibly carries the flood around bends.

Valid zero-growth choice

If reached is empty:

the region still recolors;

the petal changes;

a droplet is consumed when applicable;

the wet boundary gives a one-frame color-2 wobble;

no cells gain stamps.

This is visibly different from an invalid click.

Invalid click

A two-frame neutral ripple is centered on the click:

frame 1: color 1 at the center and four orthogonal neighboring pixels;

frame 2: color 2 at the center;

then clear.

No persistent state changes.

Completion

After the final flood wave:

the wet boundary flashes color 0, then color 10;

all membership stamps briefly become color 0;

status remains WON.

Exhaustion failure

When the last droplet is spent without completion:

unabsorbed cells flash color 13 for two frames;

they then return to their normal colors and remain unstamped;

the empty tray remains visible beside them;

status remains FAILED.

The settled frame alone is sufficient to reconstruct region membership, current color, remaining budget, and terminal outcome.

9. Level notation

The following board diagrams are implementation notation only.

S: initially absorbed seed cell.

B, Y, G, O, P: unabsorbed cells of the corresponding color.

.: void, not paintable.

Rows are listed from y=0 downward. Characters within a row increase x from left to right.

Progression summary
Level	Name	Paintable cells	Budget	Verified minimum
1	First Bloom	13	unlimited	2
2	Behind the Bridge	21	unlimited	4
3	The Droplet Tray	84	12	5
4	Small Fronts First	47	19	19
5	Around the Bend	61	25	25
6	Split Bloom	93	33	33
7	All Fronts Wet	109	39	39

“Minimum” counts valid committed color changes. Invalid clicks and unused directional actions are excluded.

10. Authored levels
Level 1 — First Bloom
Purpose

Teach the complete foundational action in a setting where its cause is unambiguous:

petals are the only actionable objects;

clicking a raised petal changes the region’s color;

matching connected cells acquire stamps;

a connected color can flood around a bend;

all stamped cells means completion.

Initial state
W = 5
H = 3
initial absorbed = {(0,1)}
current color = B / 9
level palette = {B,Y}
droplets_remaining = None
status = ACTIVE

The blue petal is depressed. The yellow petal is raised. There is no persistent cursor.

.BBBB
SYYYB
.BBBB
Player-facing discovery

The seed touches the yellow strip directly. The blue paint is visible, but it touches the seed only diagonally and the blue petal is already depressed.

Movement and SPACE merely pulse the selector. The one visibly productive experiment is clicking yellow.

Clicking Y recolors the seed and stamps the complete three-cell yellow strip.

The blue petal then rises.

Clicking B sends a visible wave around the connected U-shaped blue band and completes the board.

The second flood visibly travels around two corners, establishing recursion rather than a one-cell capture rule.

Intended replay
Y B

New cells gained per click:

3, 9

Absorbed counts after each click:

4, 13
Win validation
len(absorbed) == 13

No final color requirement.

Attribution check

Every visible board cell participates in the two-action demonstration. There are no droplets, distant copies, alternative devices, or simultaneous new rules.

The first click has one immediate visible consequence: exactly the yellow component touching the seed becomes stamped.

A mistaken attempt on the already-current blue petal is invalid and costs nothing. A click elsewhere produces only the neutral ripple.

Level 2 — Behind the Bridge
Purpose

Falsify the plausible but incorrect hypothesis:

“Choosing blue absorbs every blue cell on the board.”

The level demonstrates that only orthogonally connected blue paint touching the present region is absorbed.

Initial state
W = 10
H = 3
initial absorbed = {(0,2)}
current color = B / 9
level palette = {B,Y,G}
droplets_remaining = None
status = ACTIVE

The blue petal is depressed. Yellow and green are raised.

..BBB..BBB
.YBBBGGBBB
SYY..GG..B
Player-facing discovery

The seed again has an adjacent yellow component. Green is visible but remote.

Y absorbs the three-cell yellow component.

B absorbs only the six-cell blue patch on the left.

The identically colored seven-cell blue patch on the right remains unstamped, with a green block visibly separating it.

G absorbs the four-cell bridge.

B now absorbs the formerly disconnected blue patch.

The result of the first blue click is immediate and decisive: same color is insufficient without a route to the region.

Intended replay
Y B G B

New cells gained:

3, 6, 4, 7

Absorbed counts:

4, 10, 14, 21
Win validation
len(absorbed) == 21
Attribution check

The two blue patches, the yellow entry component, and the green separator are all necessary.

The far blue patch cannot be reached before green, and green cannot be reached before the near blue patch. No budget or secondary device can explain why the distant blue cells remain unstamped.

An exhaustive component-state search verifies that four committed changes are necessary.

Level 3 — The Droplet Tray
Purpose

Introduce the only resource rule while keeping the underlying board intentionally obvious.

The player already knows how nested connected colors flood. The sole new causal unknown is that every valid color change empties one visible droplet.

Initial state
W = 10
H = 9
initial absorbed = {(4,4)}
current color = P / 15
level palette = {B,Y,G,O,P}
droplets_remaining = 12
status = ACTIVE

The purple petal is depressed. Twelve filled droplets are shown.

OOOOOOOOO.
OGGGGGGGO.
OGBBBBBGO.
OGBYYYBGOP
OGBYSYBGOP
OGBYYYBGOP
OGBBBBBGO.
OGGGGGGGO.
OOOOOOOOO.
Player-facing discovery

The seed is surrounded by the yellow ring. The other rings are visibly farther away.

Clicking yellow:

changes the flower center and region color;

stamps exactly the eight-cell yellow ring;

visibly empties one droplet.

This immediately associates one committed choice with one spent droplet. No other unfamiliar object changes at the same time.

The concentric board then gives a low-risk continuation:

Y → B → G → O → P

The final purple choice absorbs the three-cell lobe on the right.

A premature outer-color choice is a valid zero-growth experiment: the region recolors and a droplet drains, but no stamp appears. Seven droplets beyond the five-move minimum allow substantial experimentation.

Intended replay
Y B G O P

New cells gained:

8, 16, 24, 32, 3

Absorbed counts:

9, 25, 49, 81, 84

Seven droplets remain on the intended route.

Win validation
len(absorbed) == 84

Completion with any positive or zero remaining budget is valid.

Attribution check

The board is deliberately layered rather than strategically irregular. Its role is to make droplet consumption attributable.

No frontier-order insight is required yet. The outer purple lobe participates in the solution and gives the final ring a visible consequence.

An exhaustive state search verifies the five-change minimum.

Level 4 — Small Fronts First
Purpose

Introduce the first tight ordering puzzle and the central strategic inversion:

A color that absorbs less area now can be better because it prepares several fronts for a later shared color.

The operational rules do not change.

Initial state
W = 13
H = 13
initial absorbed = {(6,6)}
current color = P / 15
level palette = {B,Y,G,O,P}
droplets_remaining = 19
status = ACTIVE

Purple is depressed. All other petals are raised.

GBO..........
Y............
POB...BOPYB..
..Y...G...G..
..G...Y...OPB
..P...Y.....Y
..BOYYSBB...O
........B....
........BBB..
..........O..
....GPO...P..
......Y...G..
......GPOBY..
Critical initial geometry

The initial region has three meaningful approaches:

A large bent blue component at
(7,6),(8,6),(8,7),(8,8),(9,8),(10,8).

A two-cell yellow mouth north at
(6,5),(6,4).

A two-cell yellow mouth west at
(5,6),(4,6).

Thus:

choosing B first gains six cells;

choosing Y first gains only four;

but Y acts simultaneously on two separate fronts.

Player-facing discovery

The large blue corridor is visually tempting. The two yellow mouths look modest.

Clicking yellow produces an immediate global consequence: both disconnected yellow mouths stamp during the same flood. The region now has two newly advanced lobes.

Those lobes bring later blue components into phase with the already visible large blue corridor. When blue is finally chosen, it absorbs seven cells in one action rather than requiring one blue choice now and another later.

A blue-first route remains physically coherent but requires at least 20 committed changes. The tray contains 19. Thus the level makes the frontier insight necessary rather than merely score-improving.

Intended replay
Y G B O B P G Y B G O P Y G B Y O P G

New cells gained:

4, 1, 7, 3, 1, 3, 2, 3, 3, 1,
3, 3, 1, 2, 2, 2, 3, 1, 1

Absorbed counts:

5, 6, 13, 16, 17, 20, 22, 25, 28, 29,
32, 35, 36, 38, 40, 42, 45, 46, 47

The final G spends the nineteenth droplet and completes the board, so success wins over exhaustion.

Structural necessity audit

The paintable-cell adjacency graph is a tree with 47 vertices and 46 edges.

Compressing consecutive equal colors along each root-to-leaf route gives:

leaf (2,0):  Y O B P G Y B O P Y G B O
leaf (12,6): Y G B O P Y B G O P B Y O
leaf (4,10): B O P G Y B O P G Y O P G

Every winning color sequence must contain all three routes as subsequences. The intended sequence is a shortest common supersequence of length 19.

Constraining the first choice to B raises the lower bound to 20.

Anti-degenerate note

The large blue component is not a decoy; it must be absorbed. Its placement changes when a blue action should occur.

The two yellow mouths are not independent checklist items. One global yellow choice advances both, altering which later colors can be shared. Flattening the layout into a corridor would destroy this interaction.

Level 5 — Around the Bend
Purpose

Reuse recursive connectivity in a contrasting role.

A long same-colored corridor is now:

a powerful transport mechanism when delayed;

a budget-losing temptation when taken immediately.

Initial state
W = 13
H = 11
initial absorbed = {(6,6)}
current color = P / 15
level palette = {B,Y,G,O,P}
droplets_remaining = 25
status = ACTIVE
GYO...OGYBP..
B.P.......O..
P.B.......G..
..G.......Y..
..YOPBG.OPB..
......Y.G....
PGB.GGSBY....
Y.O.G.O......
O.Y.G.P...GOP
B...G.Y...B.Y
GPYOB.BGOPY.B
Critical initial geometry

The seed has four frontier colors:

B at (7,6);

Y at (6,5);

O at (6,7);

a five-cell green corridor at
(5,6),(4,6),(4,7),(4,8),(4,9).

The green component turns immediately and carries the wave down the west lobe. A green-first action gains five cells, far more than the one-cell blue, yellow, or orange choices.

Player-facing discovery

The visible green corridor invites the established “recursive corridor” interpretation. Clicking green really does send the wave around the bend.

The strategic discovery is that it should not happen yet.

The exact-budget route begins:

B, Y, G

The small blue and yellow actions alter two other frontiers. When green is then chosen, it gains seven cells: the winding corridor plus newly synchronized green fronts elsewhere.

Thus the same green corridor changes role by timing alone:

first action: impressive immediate area, but desynchronizes the global sequence;

third action: transport corridor and multi-front synchronizer in one click.

Any route beginning with G, Y, or O requires at least 26 actions. The tray contains 25.

Intended replay
B Y G B O P B Y B G O P Y G B P O Y P G B O P Y B

New cells gained:

1, 2, 7, 2, 3, 3, 1, 3, 1, 2,
3, 3, 2, 2, 4, 1, 2, 3, 1, 4,
2, 3, 2, 2, 1

Absorbed counts:

2, 4, 11, 13, 16, 19, 20, 23, 24, 26,
29, 32, 34, 36, 40, 41, 43, 46, 47, 51,
53, 56, 58, 60, 61
Structural necessity audit

The board is a 61-vertex, 60-edge tree with four leaves:

leaf (6,0):   B Y G O P B Y G O P B Y G O
leaf (0,2):   Y G B P O Y G B P O Y G B P
leaf (2,8):   G B O Y P G B O Y P G B O Y
leaf (12,10): O P Y B G O P Y B G O P Y B

The listed 25-action route is a shortest common supersequence.

Forced-first minima are:

B first: 25
Y first: 26
G first: 26
O first: 26
Anti-degenerate note

The green corridor is necessary and gives immediate physical feedback, but immediate use is not the solution. Its strategic meaning depends on the states of distant fronts.

The four lobes curve in different directions and are not arranged in use order. Every committed action after the tutorial changes the readiness relation among several visible branch tips.

Level 6 — Split Bloom
Purpose

Show that expansion can create more fronts rather than merely extending existing ones.

The blue component east of the seed leads to a branch point. Absorbing it turns one frontier into two differently colored frontiers while the other three lobes remain active.

No new object is introduced: the split is ordinary authored geometry.

Initial state
W = 15
H = 15
initial absorbed = {(6,6)}
current color = P / 15
level palette = {B,Y,G,O,P}
droplets_remaining = 33
status = ACTIVE
PYBGOPY.YGOPPBY
O.......B.....G
GBY.....P.BGO.O
..P.....O.B.Y.P
..OGBYP.G.P.P.B
......O.Y.Y.B.Y
....GYSBBGO.G.G
....B.G.....O..
....P.BOY...Y..
....O...P...P..
..BGY...G.OGB..
..P.....B......
..O...B.OYP...O
..Y...G...G...B
..GBPOY...BOYPG
Critical initial geometry

Initial frontier cells are:

blue split stem: (7,6),(8,6);

yellow west: (5,6);

green south: (6,7);

orange north: (6,5).

After selecting blue:

the blue stem absorbs;

its endpoint (8,6) joins the region;

a yellow branch becomes exposed at (8,5);

a green branch becomes exposed at (9,6).

The region therefore grows from four fronts to five.

Player-facing discovery

The first blue action gains only two cells, but its wave visibly reaches a junction and exposes two differently colored exits. This gives immediate evidence that frontier count, not area, is the important result.

Later blue choices do not all have the same value:

some advance both descendants of the split;

some advance only one descendant and leave them out of phase;

delaying blue can allow an older lobe to present blue at the same time.

The split is therefore initially an enabling tool and later a synchronization liability.

An optimal route may begin with blue or orange. The example begins with blue so the split’s causal effect is observed immediately. Every successful route must eventually resolve both descendants.

Intended replay
B O Y G B P O Y P B G B O P O Y P B G B O P B Y B G O P B Y G B O

New cells gained:

2, 1, 2, 4, 2, 2, 4, 4, 3, 4, 4,
2, 3, 2, 1, 5, 2, 2, 5, 2, 4, 4,
1, 4, 1, 2, 3, 4, 2, 3, 4, 2, 2

Absorbed counts:

3, 4, 6, 10, 12, 14, 18, 22, 25, 29, 33,
35, 38, 40, 41, 46, 48, 50, 55, 57, 61, 65,
66, 70, 71, 73, 76, 80, 82, 85, 89, 91, 93
Structural necessity audit

The board is a 93-vertex, 92-edge tree with five leaves.

leaf (6,0):   O P Y B G O P Y B G O P Y B G O P Y
leaf (14,6):  B Y G O P B Y G O P B Y G O P B Y G
leaf (10,10): B G O Y P B G O Y P B G O Y P B G O
leaf (6,12):  Y G B P O Y G B P O Y G B P O Y G B
leaf (14,12): G B O Y P G B O Y P G B O Y P G B O

The example route is a shortest common supersequence of length 33.

A zero-growth color choice, or a premature choice that advances no leaf head, makes the 33-droplet budget impossible.

Anti-degenerate note

The board is not five independent lanes. Two leaves share the initial blue stem and diverge only after it is absorbed. Their shared prefix and later independent colors create a dependency network.

The other three lobes curl through separate parts of the board and repeatedly align with one or both split descendants. Removing any lobe changes the shortest global color interleaving.

Level 7 — All Fronts Wet
Purpose

Compose all established consequences without adding another mechanic:

immediate area versus frontier preparation;

recursive bends;

delayed same-color unions;

one color acting along several fronts;

a split that creates two descendants;

exact droplet pressure;

forced revisits to colors whose usefulness changes over time.

Initial state
W = 15
H = 15
initial absorbed = {(6,6)}
current color = P / 15
level palette = {B,Y,G,O,P}
droplets_remaining = 39
status = ACTIVE
YOP...OPYBGOPYB
G.B...G.......G
B.BGY.B...GOY.O
P...O.Y...B.P.P
O...P.P.OYP.B.P
Y...B.O.G...G.Y
GBP.GYSBBYG.O.B
..O...G...O.Y.G
..Y...B.YBP.P.O
......O.G...B..
OOB...Y.O...G..
Y.G...P.P...O..
P.PYOBG.B.BPY.B
G.......Y.....P
BOYPG...GOPBYGO
Critical initial geometry

The seed again has:

a two-cell blue split stem at (7,6),(8,6);

a yellow west front at (5,6);

a green south front at (6,7);

an orange north front at (6,5).

After the blue stem is absorbed, its two descendants expose different colors in different directions.

Unlike Level 6, all four initially exposed non-purple colors can begin some minimum-length solution. There is no single “always take this first” cue. The puzzle is to preserve a globally compatible frontier ordering over 39 choices.

Player-facing structure

Early clicks visibly create several kinds of progress:

a two-cell split;

one-cell mouth advances;

bent components;

synchronized floods that stamp four or five cells at once.

Large gains are not always correct, and small gains are not always setup. Their value depends on which colors become exposed simultaneously afterward.

Along the provided route:

after six actions, 14 of 109 cells are absorbed;

after twenty actions, 56 are absorbed;

after thirty-three actions, 94 are absorbed;

the final six actions resolve five uneven branch tips rather than traversing a final corridor.

A locally plausible wrong interleaving usually leaves one lobe exactly one color behind when the tray empties. The remaining unstamped tip and empty tray provide a concrete explanation of failure.

Intended replay
B O P Y B G O P B Y P B G O Y P B Y G B P O Y P G B P O Y P G B O Y P B G O Y

New cells gained:

2, 1, 1, 3, 1, 5, 3, 2, 3, 3,
2, 2, 3, 5, 3, 4, 2, 2, 4, 4,
1, 4, 3, 3, 3, 4, 1, 4, 3, 3,
4, 2, 3, 3, 4, 3, 2, 2, 1

Absorbed counts:

3, 4, 5, 8, 9, 14, 17, 19, 22, 25,
27, 29, 32, 37, 40, 44, 46, 48, 52, 56,
57, 61, 64, 67, 70, 74, 75, 79, 82, 85,
89, 91, 94, 97, 101, 104, 106, 108, 109

The final yellow action uses the last droplet and completes the board.

Structural necessity audit

The board is a 109-vertex, 108-edge tree with five leaves:

leaf (2,8):
Y G B P O Y G B P O Y G B P O Y G B P O Y

leaf (14,8):
O P Y B G O P Y B G O P Y B G O P Y B G O

leaf (10,12):
B G O Y P B G O Y P B G O Y P B G O Y P B

leaf (14,12):
B Y G O P B Y G O P B Y G O P B Y G O P B

leaf (4,14):
G B O Y P G B O Y P G B O Y P G B O Y P G

The listed sequence is a shortest common supersequence of length 39. Therefore:

all 39 droplets are structurally necessary;

any valid zero-growth choice makes completion impossible;

every leaf contributes a constraint to the optimum;

no lobe can be ignored.

Multiple minimum sequences exist, which is intentional. The validator checks physical completion, not conformity to the example route.

Anti-degenerate note

The final board does not introduce a new noun or validator condition. Its difficulty comes from the established flood rule acting on a larger dependency network.

The split descendants, west lobe, north lobe, and south lobe are interleaved spatially rather than arranged in execution order. Colors repeatedly change roles:

blue opens the split, later synchronizes, and later threatens desynchronization;

purple begins as the current inactive choice, later becomes a multi-front bridge;

small yellow and orange gains sometimes prepare four-front floods;

a visually large component may be postponed because another copy of its color is nearly exposed elsewhere.

The geometry cannot be replaced by a counter or checklist without removing the simultaneous-front behavior that defines the puzzle.

11. Verification and implementation safeguards
Foundational attribution

Level 1: one clickable operation and immediate connected expansion.

Level 2: one new causal distinction—connected versus merely same-colored.

Level 3: one new causal distinction—valid choices consume droplets.

The first three levels have no irrelevant paint, fake petals, hidden dependencies, or tight exploratory punishment.

Mature-level dependency structure

Levels 4–7 use acyclic paintable-cell graphs. On such a board, every root-to-leaf compressed color route must occur as a subsequence of a winning click sequence. This provides a direct lower-bound and replay check without adding a hidden runtime rule.

Their authored minima were verified as:

Level 4: 19
Level 5: 25
Level 6: 33
Level 7: 39

Budgets exactly match those minima.

Required invariants

An implementation should assert:

absorbed is nonempty
absorbed is orthogonally connected
absorbed is a subset of paintable
current_color is in level_palette
all base colors are in level_palette
droplets_remaining is None or >= 0
status == FAILED implies droplets_remaining == 0
status == WON implies absorbed == paintable
No special cases

The flood transition is identical in all seven levels.

There are no exceptions for:

corridors;

large components;

branch points;

the last droplet;

multiple matching fronts;

returning to the initial color;

components that touch the region at several locations.

All strategic differences arise from visible color, connectivity, region geometry, and remaining droplets.
