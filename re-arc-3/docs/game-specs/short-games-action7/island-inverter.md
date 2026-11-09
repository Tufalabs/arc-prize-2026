Island Inverter

Exact authored level count: 7

This is a game about flipping a connected island becoming difficult because every flip redraws both the island and which pieces still touch the shore.

The player never flips an individual tile. They choose a currently outlined, boundary-accessible component, and every tile in that component turns over simultaneously. Because different tiles in one component can have different reverse colors, the action may split the component, merge it with neighboring components, extend access inland, or withdraw access from something that was previously usable.

The entire relevant future is visible: current faces determine components, reverse swatches predict the result of a flip, outlines show the exact affected region, and boundary glow shows legality.

1. Player actions

The environment accepts the required action space:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x, y) on any rendered cell

Only direct clicking is used by this game.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic no-ops. They increment the harness step counter but change no puzzle state and produce no animation. There is no avatar, cursor, or persistent selection.

A click anywhere inside a rendered tile—including its reverse-color swatch or component outline—targets that tile. Clicking outside the logical board is a no-op.

Direct clicking belongs to the central mechanic because the strategic choice is precisely which visible component to turn over. There are no arbitrary hidden hotspots.

2. Exact objective

Each level contains one marked tile with two orientations:

Orientation 0: its closed seal side is face-up.

Orientation 1: its bright star side is face-up.

All levels begin with the marked tile at orientation 0.

A level succeeds immediately after a legal component flip if:

orientation[special_tile] == 1

Everything else is ignored by validation:

Other tile orientations do not matter.

The star’s resulting component may be large or small.

It may split away from the shore in the settled winning frame.

It need not remain boundary-accessible after the winning flip.

No exact color arrangement, component count, or secondary condition is checked.

The important legality test is performed on the pre-flip component. If that component touches the outer boundary, the flip is legal and may reveal the star even if the resulting star component is inland.

The game becomes terminal immediately on success. Later actions are ignored.

3. Tile and component rules
3.1 Two-sided tiles

Every logical cell is an active tile with two fixed side colors:

side_0[r,c]
side_1[r,c]

Each tile stores one orientation bit:

orientation[r,c] ∈ {0,1}

Its displayed colors are:

current_face = side_orientation
reverse_face = side_(1-orientation)

Turning a tile over toggles its orientation bit. There are no additional tile states.

The seal and star are overlays on the marked tile. They do not affect connectivity.

3.2 Components

A component is a maximal set of tiles satisfying both:

They display the same current-face color.

They are connected through orthogonal adjacency.

Diagonal contact never connects components. There is no wrapping.

There are no inactive cells, holes, or internal board edges. Every coordinate in a level matrix is a tile.

3.3 Boundary accessibility

A component is accessible if at least one of its cells satisfies:

r == 0
or r == height-1
or c == 0
or c == width-1

Touching another accessible component does not confer access unless the two have actually merged into one same-colored component.

3.4 Legal component flip

For a click targeting tile t:

Compute C, the current component containing t.

Test whether C touches the outer boundary.

If inaccessible, do not change any orientation.

If accessible, simultaneously toggle every orientation bit in C.

Only after all toggles, recompute current colors, components, outlines, and accessibility.

Validate the star objective.

Newly matching neighboring tiles merge after the action. They are not retroactively included in the same flip.

Likewise, if the clicked component reveals several colors, all resulting fragments appear only after the simultaneous swap.

3.5 Inaccessible click feedback

Clicking any tile in an inaccessible component produces a locked-outline pulse around the component as a whole. The clicked tile itself is not singled out, reinforcing that the component—not the individual cell—is the attempted action unit.

The pulse is cosmetic:

orientation bits: unchanged
components: unchanged
step count: +1
4. State and update contract

A complete level state consists of:

orientation[height][width] : bit
special_tile               : (row, column)
terminal                   : bool
step_count                 : integer, logging only
feedback_kind              : NONE | LOCKED | FLIP | WIN
feedback_cells             : set of logical cells
feedback_frames_remaining  : integer
selected_cell              : always NONE

Current colors, reverse colors, components, component outlines, and accessibility are derived state.

All authored levels reset with:

orientation = all zeroes
terminal = false
step_count = 0
feedback_kind = NONE
selected_cell = NONE

There is no energy meter, timer, or player-visible move limit. A host harness may use a non-rendered safety cap of 10,000 calls to prevent hung episodes; authored play never approaches it.

5. Animation and causality feedback

A legal component flip uses a short deterministic animation because one action can alter many cells and redraw the component partition.

The underlying state transition still occurs only inside step().

Legal flip presentation

The renderer retains pre-step and post-step snapshots:

Frame 0: old component receives a white click flash.

Frame 1: all affected faces compress horizontally to a one-pixel vertical strip.

Frame 2: orientation bits are represented by the new side colors; the new faces begin expanding.

Frame 3: new faces and reverse swatches are fully visible.

Frame 4: new component outlines and accessibility indicators appear.

All affected cells animate synchronously.

The state is committed atomically during the step() call. The ordinary environment observation returned to an agent is Frame 4, the settled state. Presentation clients may display Frames 0–4 without accepting another action between them.

Locked click

A locked component alternates its outline:

normal → white → red → normal

No tile face compresses or changes.

Completion

After the winning settled frame:

The star expands to its full marked-tile glyph.

A yellow-and-white halo pulses twice around the star component.

Input freezes.

Animation frames never consume actions or budget.

6. Rendering contract
6.1 Canvas and cell geometry

The render canvas is exactly 64×64, using only integer color IDs 0–15.

For a board of height H and width W:

pitch = min(10, floor(56 / max(H,W)))
board_pixel_width  = W * pitch
board_pixel_height = H * pitch

origin_x = floor((64 - board_pixel_width) / 2)
origin_y = floor((64 - board_pixel_height) / 2)

Logical cell (r,c) occupies:

x = origin_x + c*pitch ... origin_x + (c+1)*pitch - 1
y = origin_y + r*pitch ... origin_y + (r+1)*pitch - 1

The outside canvas is color 5.

There is no global grid border and no line between two cells belonging to the same component. The visible partition is the current component partition, not a permanently bordered logic grid.

6.2 Faces and reverse swatches

Each logical cell is filled with its current-face color.

A reverse-color swatch is persistently rendered near its lower-right corner:

swatch_size = 2×2 when pitch >= 5
swatch_size = 1×1 when pitch == 4

For a 2×2 swatch, one black perimeter pixel may be used where space permits. Clicking the swatch still targets its parent tile.

6.3 Component outlines

For each cell edge whose orthogonal neighbor is outside the board or belongs to a different component, draw a one-pixel outline inside that component’s cell region.

Inaccessible component outline: color 4.

Accessible component outline: color 11.

No outline is drawn on an edge shared by two cells in the same component.

For every maximal run where an accessible component touches the logical board boundary, add a small outward-facing color-11 rim tab at the run’s midpoint. This gives direct local evidence for why the component is clickable.

After every legal flip, all outlines and tabs are recomputed from scratch.

6.4 Marked tile

When orientation 0 is face-up:

Draw a closed seal using colors 5 and 13.

Draw a miniature yellow star inside its reverse swatch.

When orientation 1 is face-up:

Draw a yellow 11 star over the current face.

Draw a miniature seal inside the reverse swatch.

For the smallest pitch=4 rendering, the star may use a center pixel plus available cardinal pixels; the reverse-swatch star is a single yellow pixel.

The marked icon never changes the underlying face color.

7. Matrix encoding and palette

The authored levels below use character matrices as implementation notation only. These characters are not rendered as text.

Each level provides:

FACE: the initial side_0 color.

BACK: the initial side_1 color.

A marked-tile coordinate.

All orientation bits initially equal zero.

Code	Color ID	Visual color
J	1	light gray
H	2	gray
I	3	dark gray
D	6	magenta
Y	7	pink
X	8	red
A	9	blue
G	10	light blue
B	12	orange
Z	13	maroon
C	14	green
E	15	purple

Render-only colors:

ID	Use
0	white flash and completion accent
4	inaccessible component contour
5	canvas background, black icon detail
11	accessible outline, rim tabs, star

H/I tiles are ordinary two-sided tiles, not immovable walls. Their reserved neutral colors ensure that they do not merge with the active color networks in these authored levels. Their placement is nevertheless necessary: they fill the rectangular lattice, keep inland components inland, and spatially separate the active branches.

Coordinates are zero-indexed (row,column), with (0,0) at the top-left.

8. Progression summary
Level	Name	Principal discovery or composition	Minimum legal clicks
1	First Turn	A click flips the whole outlined component; reveal the star	1
2	Borrowed Shore	An inland component becomes clickable by merging with a shore component	2
3	Castaway Fragment	Mixed reverse colors fracture a component and can strand the useful fragment	3
4	Prepare the Inlet	Prepare one locked branch and anchor it through another branch	6
5	Three Strands	Three independently produced fragments must form one shore-to-seal chain	11
6	The Shore Closes	Advancing through nested gates destroys the bridge behind the player	30
7	Nested Tides	Three nested gates repeatedly close all earlier gates, forcing planned rebuilds	56

Locked experimental clicks are optional and are not included in these depths.

9. Authored levels
Level 1 — First Turn
Player-facing discovery

The marked tile is part of a four-tile blue component touching the boundary. Its reverse swatch contains a miniature star. The entire component has one yellow accessible outline.

Clicking any of its four cells changes all four faces at once. Their differing reverse swatches become full faces, visibly demonstrating that the action affected the component rather than only the clicked tile. The marked tile reveals its star and completes the level.

This is a one-action positive example. No symbolic goal explanation is required: the star is shown before the action in miniature and after the action at full size.

Initial state
size: 2 rows × 3 columns
special_tile: (0,0)
selected_cell: NONE

FACE

AAB
AAB

BACK

JDA
CDA

Initial blue component:

{(0,0),(0,1),(1,0),(1,1)}
Reference solution
CLICK(0,0)

Any coordinate in the initial A component is equivalent.

Expected pre-click component trace:

A, size 4, accessible
Attribution and necessity check

Only the marked component can toggle the marked tile. The second orange component provides a clear contrasting component boundary but does not contain the goal.

The star-bearing reverse swatch identifies the productive click without random search. No unrelated rule changes during the action.

Level 2 — Borrowed Shore
Player-facing discovery

The seal begins in a one-tile orange component at (2,3). Its outline is dark because it does not touch the board edge. Clicking it produces the locked-component pulse.

A blue finger reaches from the upper-left boundary to (2,2). Every tile in that finger visibly has orange on its reverse. Flipping it turns the whole finger orange. Its outline merges with the seal’s outline, and the new combined component inherits the finger’s boundary contact.

The formerly inland seal is now clickable.

This introduces one causal rule: access belongs to the whole current component, so merging with a shore component lends access inland.

Initial state
size: 5×5
special_tile: (2,3)
selected_cell: NONE

FACE

ACCCC
ACCCC
AAABC
CCCCC
CCCCC

BACK

BDDDD
BDDDD
BBBJD
DDDDD
DDDDD

The initial blue finger is:

{(0,0),(1,0),(2,0),(2,1),(2,2)}

The marked orange tile (2,3) is initially inaccessible.

Reference solution
CLICK(0,0)
CLICK(2,3)

Expected component trace:

A, size 5, accessible
B, size 6, accessible

After the first click, the five changed finger tiles and the marked tile form the six-tile orange component.

Attribution and necessity check

The marked tile cannot be legally flipped in the initial state. The only authored color transformation that creates a same-colored boundary path to it is the blue finger’s A→B transformation.

The large green field is structurally necessary to enclose the marked tile. Its C↔D colors never match the blue/orange route and therefore cannot create a shortcut.

Level 3 — Castaway Fragment
Player-facing discovery

The blue component at the top contains visibly mixed reverse swatches:

Its vertical trunk will become orange.

Its rightmost lobe at (2,3) will become green.

The marked tile at (2,4) is already green but inland.

Flipping the blue component does not create one uniformly changed component. It fractures:

The orange trunk remains associated with the shore.

The green lobe merges with the marked tile.

That green seal component is still inland and therefore remains dark-outlined.

A magenta component enters from the upper-right boundary. Its reverse swatches are green. Flipping it creates a second green route that meets the stranded green lobe and lends it the shore.

The player must therefore observe the property that distinguishes this mechanic from a uniform recoloring rule: the clicked component splits into differently colored fragments with different accessibility.

Initial state
size: 5×6
special_tile: (2,4)
selected_cell: NONE

FACE

HHAHHD
HHADDD
HHAACH
HHHHHH
HHHHHH

BACK

IIBIIC
IIBCCC
IIBCJI
IIIIII
IIIIII
Reference solution
CLICK(0,2)
CLICK(0,5)
CLICK(2,4)

Expected component trace:

A, size 4, accessible
D, size 4, accessible
C, size 6, accessible
Visible causal sequence

The four-cell A component fractures into an orange trunk and green lobe.

The green lobe joins the seal but remains locked.

The boundary D component turns green and joins the inland green component.

Clicking the combined green component reveals the star.

Attribution and necessity check

The second boundary bridge is required precisely because the useful green fragment becomes detached from the original shore. If the initial blue component merely recolored uniformly, this intermediate locked green outline would not appear and the magenta bridge would be unnecessary.

The rule is immediately reused: first fracture strands a fragment; the subsequent merge restores access to it.

Level 4 — Prepare the Inlet
Player-facing discovery

A large blue fork touches the top boundary. Its reverse swatches predict three outcomes:

A green upper branch.

An orange lower-left branch.

A magenta interior lobe.

The first flip produces two accessible branches and one locked magenta network.

The green and orange branches are not independent errands. They prepare different parts of the same future magenta component:

Flipping the green branch produces a necessary purple connector near the seal and adds magenta tiles to the locked interior.

Flipping the orange branch converts its lower-left shore route to magenta, anchoring the prepared interior network.

Only after both preparations does the magenta component both contain the required interior structure and touch the shore. Flipping it produces a purple route near the seal. A light-blue shore component then joins that purple route, allowing the final star flip.

The green and orange preparations may be performed in either order, but both must precede the magenta flip. This is the first dependency network rather than a single forced chain.

Initial state
size: 7×7
special_tile: (4,4)
selected_cell: NONE

FACE

HHAHHHH
HHACCCH
HAAACCG
HAAACDG
HHBAEAH
BBBAAAH
HHHHHHH

BACK

IICIIII
IICDDDI
IBCCDDE
IBBDEEE
IIDDJDI
DDDDDDI
IIIIIII
Reference solution
CLICK(0,2)
CLICK(1,3)
CLICK(4,2)
CLICK(3,5)
CLICK(2,6)
CLICK(4,4)

Expected pre-click colors and component sizes:

A13 → C10 → B7 → D16 → G2 → E5
Phase interpretation

A13: fracture the blue fork.

C10: prepare the interior connector and magenta branch.

B7: anchor the magenta network from the lower-left shore.

D16: flip the fully prepared network, producing the purple internal route.

G2: turn the right-side light-blue shore purple.

E5: flip the purple component containing the seal.

Anti-degenerate and necessity check

The green branch is necessary because it creates the purple connector that ultimately reaches the marked tile. The orange branch is necessary because it provides the magenta network’s boundary contact. Omitting either makes the relevant interior component unusable.

The light-blue component is visible from the beginning but has no useful merge until the magenta network has been flipped. Its strategic meaning therefore depends on timing rather than being encountered exactly when needed.

The irregular U-shaped interior placement matters: replacing the board with a linear list would lose the distinction between the locked prepared branch, the lower-left anchor, and the right-side final shore.

Level 5 — Three Strands
Player-facing discovery

Three relay colors—red X, pink Y, and maroon Z—sit in a row before the purple sealed tile. None touches the shore.

Three asymmetric components approach from different parts of the board:

An upper-left blue fork can make the red relay accessible.

A lower-left green fork can make the pink relay accessible.

An upper-right magenta fork can make the maroon relay accessible.

Each branch uses the already established grammar:

Flip a boundary fork, fracturing off an inland relay-colored tail.

Flip a boundary orange component whose reverse matches that tail.

Flip the resulting relay component, changing its one central relay tile to purple.

Once all three relay tiles are purple, they form one continuous inland chain to the seal. The light-blue component at the left can then turn purple and lend the whole chain the shore.

The three producer modules can be completed in any order. The reference path uses left, lower, then right.

The player may pre-flip the light-blue shore early, but clicking the resulting purple component before all three outputs exist simply turns completed relay tiles back to their relay colors. Thus the same merge is a useful final connection and a liability when used prematurely.

Initial state
size: 11×11
special_tile: (5,6)
selected_cell: NONE

FACE

BHHAHHHDHHH
BHHAHHHDHHH
BHAAHHDDHHH
BHAAHDDHHHH
BBBAHDBBBBB
GGGXYZEHHHH
BBBBCHHHHHH
BHHHCCHHHHH
BHHHCCHHHHH
BHHHCHHHHHH
BHHHCHHHHHH

BACK

XIIBIIIBIII
XIIBIIIBIII
XIBBIIBBIII
XIBBIBBIIII
XXXXIZZZZZZ
EEEEEEJIIII
YYYYYIIIIII
YIIIBBIIIII
YIIIBBIIIII
YIIIBIIIIII
YIIIBIIIIII

The central logical row initially contains:

columns 0–6: G G G X Y Z E(seal)
Reference solution
CLICK(0,3)
CLICK(0,0)
CLICK(5,3)

CLICK(10,4)
CLICK(10,0)
CLICK(5,4)

CLICK(0,7)
CLICK(4,10)
CLICK(5,5)

CLICK(5,0)
CLICK(5,6)

Expected component trace:

A7
B13
X9

C7
B8
Y10

D7
B11
Z7

G3
E7
Structural interpretation

The three groups of three clicks independently produce purple at (5,3), (5,4), and (5,5). Those cells are all required to bridge the final purple shore at columns 0–2 to the marked purple tile at (5,6).

Anti-degenerate and necessity check

Each relay cell is a one-cell break in the final purple component. No pair of modules can bypass the third because four-neighbor connectivity requires the entire row segment.

Each output requires three distinct component flips:

expose the locked tail,

borrow a shore of the matching relay color,

flip the relay component.

No single click can advance two modules because the relay colors and geometries remain in separate components.

The lower bound is therefore:

3 modules × 3 clicks
+ 1 light-blue-to-purple shore click
+ 1 winning purple click
= 11

The surrounding producer geometry, not merely the central row, determines when each relay is accessible. The board is not traversed as a corridor.

Level 6 — The Shore Closes
Player-facing discovery

Two producer modules generate purple output tiles at (5,3) and (5,4). Beyond them are two nested gate pairs:

(E/X, X/E)
(E/Y, Y/E)

The seal lies beyond both.

The upper blue shore can be prepared as red X. The lower green shore can be prepared as pink Y.

After the two outputs are built and the light-blue shore becomes purple, the accessible purple component reaches the first E/X cutter but stops at the neighboring red gate.

Ordinary joining no longer makes progress. The local useful experiment is to flip the accessible purple component itself.

That action visibly does two things:

It turns the prepared output tiles back to red and pink, destroying the route behind the player.

It turns the first cutter red, joining it to the red gate and prepared red shore.

Clicking that red component turns both gate tiles purple and opens the first gate.

Reaching the second cutter requires rebuilding the two purple outputs. Flipping through the second cutter closes the first gate again, opens a pink gate opportunity, and destroys the outputs a second time. Both earlier gates must be reopened before the final rebuild.

No new object has appeared. The new strategic consequence comes entirely from established two-sided tiles and simultaneous component flipping.

Initial state
size: 11×11
special_tile: (5,9)
selected_cell: NONE

FACE

HHAHBHAHHHH
HHAHBHAHHHH
HHAHBHAHHHH
HHAABHAHHHH
HHHABAAHHHH
GGGXYEXEYEH
HHHHCBHCHHH
HHCCCBHCHHH
HHCHHBHCHHH
HHCHHBHCHHH
HHCHHBHCHHH

BACK

IIBIXIXIIII
IIBIXIXIIII
IIBIXIXIIII
IIBBXIXIIII
IIIXXXXIIII
EEEEEXEYEJI
IIIIYYIYIII
IIBBBYIYIII
IIBIIYIYIII
IIBIIYIYIII
IIBIIYIYIII

The central gate spine is:

G G G | X Y | E/X X/E | E/Y Y/E | E(seal)

The slash notation above is explanatory only: current/reverse.

Resettable output macro

Define:

BUILD2 = [
  CLICK(0,2),   # A fork
  CLICK(0,4),   # B→X shore
  CLICK(5,3),   # X relay, output 1 becomes E

  CLICK(10,2),  # C fork
  CLICK(10,5),  # B→Y shore
  CLICK(5,4)    # Y relay, output 2 becomes E
]

Expected pre-click colors during every execution of BUILD2:

A, B, X, C, B, Y

The module faces return to their reset arrangement after each relay click, except for the relay output itself. A later purple cutter flip returns that output to X or Y, making the same macro usable again without any special reset rule.

Exact 30-click reference solution
BUILD2

CLICK(0,6)    # prepare upper A/X shore as X
CLICK(10,7)   # prepare lower C/Y shore as Y
CLICK(5,0)    # G shore becomes E
CLICK(5,5)    # E cut through gate 1; outputs are erased
CLICK(5,5)    # same coordinate now targets the X gate component

BUILD2

CLICK(0,6)    # replenish the consumed X shore
CLICK(5,0)    # G shore becomes E again
CLICK(5,7)    # E cut through gate 2; gate 1 closes
CLICK(5,5)    # reopen gate 1 with X
CLICK(5,7)    # open gate 2 with Y

BUILD2

CLICK(5,0)    # final E shore
CLICK(5,9)    # flip the complete E component and reveal the star

Grouped component-color trace:

BUILD2: A B X C B Y
phase 1: A C G E X

BUILD2: A B X C B Y
phase 2: A G E X Y

BUILD2: A B X C B Y
final:   G E
Necessity and 30-click lower bound

The two output modules must be built three times:

To reach cutter 1.

To reach cutter 2 after cutter 1 erased the outputs.

To reach the star after cutter 2 erased the outputs.

Each build requires six clicks:

3 builds × 6 = 18

The red shore is consumed twice and must be prepared twice. The pink shore is consumed once:

shore preparation = 3

The light-blue shore must become purple once for each cut and once for the final route:

G→E activations = 3

There are two destructive purple cuts and one winning purple flip:

E flips = 3

Opening the nested gates requires one gate click after the first cut and two after the second:

gate openings = 1 + 2 = 3

Therefore:

18 + 3 + 3 + 3 + 3 = 30

A premature purple flip is recoverable but visibly loses output progress. The difficulty comes from forced revisits and state dependency, not an action cap.

Level 7 — Nested Tides
Player-facing discovery

The final level extends the same system to three output modules and three nested gate pairs:

(E/X, X/E)
(E/Y, Y/E)
(E/Z, Z/E)

Three separate boundary components can be pre-staged as red, pink, and maroon shores. A staged shore remains available until it participates in a gate-opening flip.

The critical recursive consequence is now fully exposed:

Cutting through gate 1 closes no earlier gate.

Cutting through gate 2 closes gate 1.

Cutting through gate 3 closes gates 1 and 2.

Every purple cut also erases all three producer outputs.

Thus each deeper advance destroys a larger portion of the route behind it. Earlier gate shores are consumed when used and must be replenished before those gates can be reopened.

The board supports experimentation:

All reverse swatches show exactly which gate colors will appear.

Prepared shore fragments remain visibly colored.

A failed or premature cut does not cause an irreversible loss.

There is no timer or scarce numerical budget.

The insight is not “set six independent variables.” It is to understand a nested physical recurrence: build the route, cut at the frontier, reopen every gate the cut closed, and then rebuild the route in front of the now-advanced frontier.

Initial state
size: 14×14
special_tile: (6,12)
selected_cell: NONE

FACE

HHAHBHHHDHHHHB
HHAHBHHHDHHHHB
HHAHBHHHDHHHHB
HHAHBHHDDHHHHB
HHAABHHDBBBBBB
HHHABDDDBHHHHH
GGGXYZEXEYEZEH
HHHHCBAHCHDHHH
HHHCCBAHCHDHHH
HHHCHBAHCHDHHH
HHHCHBAHCHDHHH
HHHCHBAHCHDHHH
HHHCHBAHCHDHHH
HHHCHBAHCHDHHH

BACK

IIBIXIIIBIIIIZ
IIBIXIIIBIIIIZ
IIBIXIIIBIIIIZ
IIBIXIIBBIIIIZ
IIBBXIIBZZZZZZ
IIIXXZZZZIIIII
EEEEEEXEYEZEJI
IIIIYYXIYIZIII
IIIBBYXIYIZIII
IIIBIYXIYIZIII
IIIBIYXIYIZIII
IIIBIYXIYIZIII
IIIBIYXIYIZIII
IIIBIYXIYIZIII

The central gate spine is:

G G G
| X Y Z
| E/X X/E
| E/Y Y/E
| E/Z Z/E
| E(seal)

In the actual matrix these cells occupy row 6, columns 0–12. The producer components enter that spine asymmetrically from the upper-left, lower-left, and upper-right; the gate shores descend through offset columns below it.

Resettable producer macro

Define:

BUILD3 = [
  CLICK(0,2),   # A fork
  CLICK(0,4),   # B→X shore
  CLICK(6,3),   # X output becomes E

  CLICK(13,3),  # C fork
  CLICK(13,5),  # B→Y shore
  CLICK(6,4),   # Y output becomes E

  CLICK(0,8),   # D fork
  CLICK(0,13),  # B→Z shore
  CLICK(6,5)    # Z output becomes E
]

Expected component colors and sizes for every BUILD3 execution:

A7
B12
X8

C8
B14
Y9

D9
B17
Z15
Exact 56-click reference solution
Phase 1 — reach and open the first gate
BUILD3

CLICK(13,6)   # prepare A/X shore as X
CLICK(13,8)   # prepare C/Y shore as Y
CLICK(13,10)  # prepare D/Z shore as Z

CLICK(6,0)    # G shore becomes E
CLICK(6,6)    # E cut 1; outputs become X/Y/Z, cutter 1 becomes X
CLICK(6,6)    # open gate 1 with the prepared X shore

The second click at (6,6) targets a different current component from the first because the first action changed the cutter from E to X.

Phase 2 — reach the second gate
BUILD3

CLICK(13,6)   # replenish the X shore consumed in phase 1

CLICK(6,0)    # rebuild shore-to-frontier E access
CLICK(6,8)    # E cut 2; gate 1 closes and cutter 2 becomes Y
CLICK(6,6)    # reopen gate 1 with X
CLICK(6,8)    # open gate 2 with the preserved Y shore

The initial Y shore prepared in Phase 1 has remained pink and unused until this point.

Phase 3 — reach the third gate
BUILD3

CLICK(13,6)   # replenish X
CLICK(13,8)   # replenish Y

CLICK(6,0)    # rebuild E access
CLICK(6,10)   # E cut 3; gates 1 and 2 close, cutter 3 becomes Z
CLICK(6,6)    # reopen gate 1 with X
CLICK(6,8)    # reopen gate 2 with Y
CLICK(6,10)   # open gate 3 with the preserved Z shore

The Z shore prepared before the first cut has remained visibly staged for all three phases.

Final phase — rebuild once more and reveal the star
BUILD3

CLICK(6,0)    # final G→E shore connection
CLICK(6,12)   # flip the complete E component containing the seal
Full grouped component-color trace
BUILD3: A B X C B Y D B Z
phase 1: A C D G E X

BUILD3: A B X C B Y D B Z
phase 2: A G E X Y

BUILD3: A B X C B Y D B Z
phase 3: A C G E X Y Z

BUILD3: A B X C B Y D B Z
final:   G E
Necessity and 56-click lower bound

Every destructive purple cut returns the three output cells to X, Y, and Z. Three cuts are needed to advance three nested gates, followed by one final build. The producer network must therefore be built four times:

4 builds × 3 modules × 3 clicks = 36

The light-blue shore must be activated once for each cut and once for the final flip:

4 G→E clicks

There are three frontier cuts and one final star flip:

4 E-component clicks

Gate opening costs grow with depth because every deeper cut closes every prior gate:

1 + 2 + 3 = 6 gate-opening clicks

Opening a gate consumes its matching shore. Required shore preparations are:

X shore: initial + before phase 2 + before phase 3 = 3
Y shore: initial + before phase 3                 = 2
Z shore: initial                                  = 1

total shore preparations = 6

Therefore:

36 producer clicks
+ 4 shore-to-purple clicks
+ 4 purple cut/final clicks
+ 6 gate-opening clicks
+ 6 shore-preparation clicks
= 56
Anti-cleanliness check

The central gate spine cannot be solved by clicking along it in order. It is initially inland and mostly inaccessible. Every advance depends on three spatially separate producer modules and three offset shore components.

Completed producer outputs are deliberately destroyed three times. Gate shores act in contrasting roles:

inert boundary components before preparation,

stored future access after preparation,

active anchors when their cutter color appears,

consumed fragments after opening a gate,

resources that must be regenerated for deeper cuts.

Removing the upper-left, lower-left, or upper-right producer breaks every later phase. Removing any gate shore prevents reopening its corresponding gate. Moving a shore endpoint beside the gate rather than the cutter would create an early shortcut, which is why each endpoint touches only its cutter cell.

The solution is long because the component partition recursively invalidates earlier progress, not because of corridors, repeated no-op movement, or a hidden counter.

10. Epistemic-solvability audit
Foundational attribution

Level 1: One marked, glowing component; one click; all affected cells visibly change. The star inset supplies positive goal evidence.

Level 2: The marked component visibly lacks a boundary glow. A locked click confirms the restriction. The nearby blue finger’s orange reverse swatches predict the only productive merge.

Level 3: Mixed reverse swatches inside one outlined component predict a fracture. The useful green fragment visibly remains locked, and the boundary magenta component visibly advertises green backs.

No foundational level changes two unrelated systems at once.

Reuse in contrasting roles

A shore merge is:

the direct solution in Level 2,

a recovery mechanism for a stranded fracture in Level 3,

one prerequisite in a dependency network in Level 4,

a final bridge that should not be flipped prematurely in Level 5,

a consumable staged resource in Levels 6 and 7.

A fracture is:

evidence about heterogeneous backs in Level 3,

a way to prepare several branches in Level 4,

a relay-production mechanism in Level 5,

the reason an advancing frontier destroys its own route in Levels 6 and 7.

The operational rules never change. Only geometry, timing, current orientation, and adjacency change their strategic meaning.

No hidden information

At every step the player can see:

The exact current component partition.

Which components are legal.

Every tile’s reverse color.

Which fragment will remain on a board edge after a split.

Which neighboring components will merge after a flip.

Whether a staged gate shore has already been consumed.

No earlier frame must be remembered to predict the next state.

Recovery

There are no intended dead ends. Premature flips may erase progress or consume a staged shore, but the relevant boundary forks and shore components remain available to reconstruct it. With no move budget, the player can safely test a hypothesis, observe the resulting partition, and revise it.
