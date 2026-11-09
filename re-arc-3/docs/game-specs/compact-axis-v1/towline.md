Worked for 46m 19s
Towline

This is a game about pulling cargo becoming difficult because every unsupported crate settles downward immediately after each successful action.

Towline contains exactly 7 authored levels.

The player controls a hovering tug. The tug can fly freely through air, but cargo can only be moved by attaching a horizontal towline and moving away from it. Cargo cannot be pushed, lifted, or towed vertically. Whenever a successful action ends, unsupported cargo immediately falls until it reaches terrain, the bottom of the board, or another cargo block.

The interesting tension is that cargo is simultaneously:

something that must reach a dock;

a movable landing surface for other cargo;

a load that prevents the crate beneath it from moving;

a route blocker that can deny the tug the side needed for a later pull.

No new device categories are introduced after the foundational curriculum. All later depth comes from gravity, stacking, pulling-side access, dock geometry, and the irreversible loss of height.

1. Native geometry and rendering
Logical board

Native board: 14 columns × 12 rows.

Coordinates: (x, y).

x = 0…13, left to right.

y = 0…11, top to bottom.

One arrow action moves the tug by exactly one logical cell.

The bottom edge below y=11 is a fixed supporting boundary. Cargo in row 11 does not fall out of the game.

The tug and horizontally moved cargo may not leave the board.

Pixel mapping

Canvas: 64×64.

Each logical cell renders as 4×4 pixels.

Board origin: pixel (4, 8).

Logical cell (x,y) occupies pixels:

left = 4 + 4x
top  = 8 + 4y
width = height = 4

The board field is color 1.

The canvas outside the board field is color 0.

No logical grid lines or internal cell borders are drawn.

A one-pixel dark bottom lip makes the board’s supporting lower boundary visible.

Palette use
Role	Color IDs
Outside canvas / sprite transparency	0
Air field	1
Terrain exposed surface	3
Terrain body	4
Dark sprite detail	5
Towline	6
Invalid pull / impact	8
Tug body	9
Tug highlights and tow ports	10
Dock frame and cargo tow eyes	11
Cargo body	12
Loaded tow-eye dimming	13
Occupied dock / completion	14

Adjacent terrain cells merge into continuous shapes. Only exposed upper terrain edges receive the color-3 cap; there are no per-cell outlines.

Multi-pixel entity silhouettes

The following matrices show local 4×4 color IDs. . means the board background remains visible.

Cargo

12 12 12 12
11  5  5 11
11  5  5 11
12 12 12 12

The color-11 pixels are visible side tow eyes. When another cargo is directly above this cargo, those four pixels become color 13, providing a derived visual “under load” cue.

Tug

 . 10 10  .
 9  9  9  9
10  5  5 10
 .  9  9  .

Empty dock

11  .  . 11
11  .  . 11
11  .  . 11
11 11 11 11

A dock is a marking behind the cell, not solid terrain. Its frame becomes color 14 while a settled cargo occupies that cell. The cargo remains visibly towable; docks never lock cargo.

2. Player actions

Towline uses the following subset of the supplied action space:

UP

DOWN

LEFT

RIGHT

SPACE

There are no clickable objects or clickable regions. A click on any cell is an inert no-op: no state changes, gravity does not run, and no click affordance is rendered.

For compact solution traces:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE
3. Complete state

The settled gameplay state is:

terrain: immutable set<Cell>
docks: immutable set<Cell>
cargo[id].position: Cell for each cargo
tug_position: Cell
attached_id: cargo id or null
mode: PLAYING, COMPLETE, or CRUSHED
wreck_cell: Cell or null

Cargo IDs are implementation handles only. Cargo blocks are visually and functionally interchangeable, and the validator ignores their identities.

There is no cursor state, inventory, hidden timer, action history, random state, or latent mode. The persistent towline makes attached_id visible.

All seven levels begin with:

attached_id = null
mode = PLAYING
wreck_cell = null

All supplied initial states are already gravity-stable.

4. Exact action semantics
4.1 Tug movement while detached

For an arrow input with attached_id = null:

Compute the adjacent destination cell.

The move succeeds only if the destination:

is inside the board;

is not terrain;

contains no cargo.

On success, move the tug one cell.

Run gravity settlement.

Validate completion.

Walking into cargo never pushes it.

4.2 Attaching

For SPACE while detached:

Find cargo blocks immediately horizontally adjacent to the tug:

(tug.x - 1, tug.y)

(tug.x + 1, tug.y)

If exactly one such cargo exists, attach to it.

If zero or two such cargos exist, the action is invalid and nothing changes.

A successful attachment runs gravity and validation, although authored states are already stable.

The line is drawn from the corresponding tug tow port to the adjacent cargo tow eye.

4.3 Detaching

SPACE while attached clears attached_id.

Detaching does not otherwise move the tug or cargo. Gravity and validation then run.

4.4 Pulling

Suppose the tug is attached to cargo C.

Let:

away = tug_position - C.position

Because attachment is horizontal, away is either LEFT or RIGHT.

An arrow input performs a pull only when all of the following hold:

The arrow equals away: the tug is moving directly away from the cargo.

The tug’s destination is inside the board.

The tug’s destination contains no terrain or cargo.

No cargo is directly above C.

On a successful pull:

C.position     = old tug_position
tug_position   = old tug_position + away
attached_id    remains C

The cargo therefore moves into the tug’s former cell.

4.5 Loaded cargo

A cargo is loaded whenever another cargo occupies (x, y-1) directly above it.

A loaded cargo may be attached, but it cannot be pulled. An attempted away-arrow:

does not move either object;

does not run gravity;

turns the line red for the feedback animation;

makes the visible stack bob by one render pixel and return.

This behavior has no stored “loaded” flag; it is derived entirely from visible cargo positions.

4.6 Incompatible movement while attached

While attached, all of the following are invalid:

moving toward the cargo;

moving vertically;

moving away when the tug’s destination is blocked;

pulling a loaded cargo.

The tug does not move. SPACE remains available to detach.

5. Gravity settlement and update order

Gravity runs after every successful movement, pull, attachment, or detachment.

It does not run after clicks or invalid actions.

5.1 Simultaneous fall microticks

At each gravity microtick, compute the least set F of falling cargo blocks:

A cargo with terrain or the board’s bottom boundary directly beneath it is not in F.

A cargo with empty air directly beneath it is in F.

A cargo with another cargo beneath it is in F exactly when that lower cargo is also in F.

This makes an unsupported vertical stack fall together without separating.

Then every cargo in F moves down by one logical row simultaneously. Repeat until F is empty.

5.2 Towline during a fall

If the attached cargo belongs to the first nonempty fall set:

the line flashes red for one frame;

attached_id becomes null;

the line visibly retracts to the tug;

the cargo continues falling normally.

The line never supports cargo.

5.3 Collision with the tug

The tug is not a cargo support.

If a falling cargo’s next destination equals the tug’s cell:

Apply that fall microtick.

Remove the tug.

Set wreck_cell to the collision cell.

Clear the towline.

Set mode = CRUSHED.

Stop further settlement.

The failed settled frame shows the cargo with blue tug debris and a red impact mark beneath it. No authored solution uses this failure.

5.4 Resolution order

Every successful action follows this exact order:

1. Apply attachment, detachment, tug move, or horizontal pull.
2. Resolve all gravity microticks.
3. Recompute dock occupancy.
4. Test the success condition.
5. Return the final settled observation.

A cargo passing through a dock during an intermediate gravity frame does not count as docked.

6. Success condition and validator behavior

Each level has the same number of cargo blocks and docks.

After gravity has completely settled, the level succeeds exactly when:

mode == PLAYING
and set(cargo positions) == set(dock positions)

Consequences:

Cargo identities do not matter.

Every dock is occupied exactly once when complete.

A cargo that starts in a dock already counts.

A cargo may be pulled out of a dock.

A dock does not prevent falling.

Tug position is irrelevant.

Attachment state is irrelevant.

The tug may occupy a dock while it is empty.

Completion is not checked midway through a fall.

No particular action count or cargo-to-dock assignment is required.

On completion, all dock frames turn color 14; after a six-frame pulse, the level enters COMPLETE and ignores further input.

7. Animation and feedback contract

Animation frames never consume player steps. New input is accepted only after the final settled frame.

Ordinary tug movement

The tug updates immediately to the neighboring cell. No ornamental movement animation is required.

Attachment and detachment

Two frames.

The line extends or retracts between the adjacent sprites.

The authoritative attachment state changes at the start of the animation.

Horizontal pull

Three frames:

taut line at the starting cells;

tug and cargo interpolated two rendered pixels;

both at their new logical cells.

Gravity begins after the third frame.

Falling

For every one-cell gravity microtick:

render the falling set halfway down, offset by two pixels;

render it at the next logical row;

commit that microtick and recompute the next falling set.

All falling cargo columns animate simultaneously.

Invalid actions

Blocked tug movement: tug shakes one pixel toward the obstruction and returns.

Loaded or incompatible pull: line becomes color 8, stretches one pixel, and returns.

Invalid SPACE: both tug tow ports briefly flash color 8.

Clicks produce no feedback because nothing is visually clickable.

Exploration budget

There is:

no energy;

no timer;

no move limit;

no score penalty for invalid experiments.

A level can become geometrically unrecoverable after a bad drop because cargo cannot be lifted. Resetting is an external environment operation, not a game action.

8. Progression
Level	Central discovery or composition	Canonical successful actions
1. First Tow	Attach once and maintain a persistent pull	4
2. Missing Ledge	Unsupported cargo falls and breaks the line	10
3. Living Landing	Cargo visibly supports other cargo	13
4. Borrowed Berth	Loaded cargo cannot move; docks do not lock cargo	32
5. Relay Crates	Cargo exchanges payload and support roles	45
6. Last Door	Build a tower, stage its top, and delay a terminal dock	91
7. Vacated Berth	Repurpose a support, stage a second payload, and reassign a completed dock	102

Levels 1–6 were state-space checked at the listed shortest successful-action lengths. The Level 7 trace is mechanically replay-verified but is not asserted to be globally shortest.

9. Authored levels

The ASCII sketches below are implementation documentation only. They are not rendered in the game.

. = air
# = terrain
d = empty dock
O = cargo occupying a dock
C = cargo
T = tug

Cargo labels such as A, B, or Base appear only in the written specification.

Level 1 — First Tow
Player-facing discovery

One orange crate rests on a long floor. Its yellow dock is several cells to the left. The tug begins immediately left of the crate.

Walking right merely bumps the crate. SPACE produces the only new visible event: a bright line joins the tug to the crate. Repeated movement away from the crate then moves both objects while preserving the line.

Nothing can fall in this level, so the causal effect of attachment is isolated.

Initial map
   01234567890123
00 ..............
01 ..............
02 ..............
03 ..............
04 ..............
05 ..............
06 ..............
07 ...d.TC.......
08 .#########....
09 ..............
10 ..............
11 ..............
State-machine instance
Terrain:
  y=8, x=1..9

Docks:
  (3,7)

Cargo:
  A = (6,7)

Tug:
  (5,7)

Attachment:
  null
Canonical solution
S L L L

Compact replay string:

SLLL

Resolution:

S: attach to A.

First L: tug (5,7)→(4,7), cargo (6,7)→(5,7).

Second L: tug (4,7)→(3,7), cargo (5,7)→(4,7).

Third L: tug (3,7)→(2,7), cargo enters dock (3,7).

Final cargo set:

{(3,7)}
Attribution and necessity check

Every visible component participates in the solution.

Cargo remains supported throughout, so gravity cannot explain its movement.

Arrow inputs without attachment cannot move the crate.

The three consecutive pulls demonstrate that attachment persists rather than applying to only one action.

No click, detour, decoy, or recovery path is present.

Level 2 — Missing Ledge
Player-facing discovery

The familiar cargo begins on an elevated platform. Its dock is lower and one column left of the obvious drop column.

The first left pull behaves exactly like Level 1 because terrain remains beneath the crate. The second left pull moves the crate beyond the platform edge. It immediately descends several rows, and the line visibly retracts.

The cargo lands safely beside the dock, allowing the player to descend, reattach, and finish without punishment.

Initial map
   01234567890123
00 ..............
01 ..............
02 ..............
03 .....TC.......
04 .....#####....
05 ..............
06 ..............
07 ..............
08 ...d..........
09 ..#####.......
10 ..............
11 ..............
State-machine instance
Terrain:
  y=4, x=5..9
  y=9, x=2..6

Docks:
  (3,8)

Cargo:
  A = (6,3)

Tug:
  (5,3)

Attachment:
  null
Canonical solution
S L L D D D D D S L

Compact replay string:

SLLDDDDDSL

Resolution:

Attach to A.

First L: A moves to (5,3) and remains supported by (5,4).

Second L: A moves to (4,3), finds no support beneath column 4, and falls to (4,8).

The line auto-detaches during the first fall microtick.

D×5: tug moves from (3,3) to (3,8), immediately left of A.

S,L: reattach and pull A into dock (3,8).

Attribution and necessity check

The only structural difference from Level 1 is the missing platform cell beneath column 4 and the forgiving floor below.

The first pull remains supported while the second falls, identifying the exact condition that triggers gravity.

The dock is deliberately offset, so the player must notice the broken line and reattach rather than winning accidentally from the fall alone.

No other cargo or simultaneous state change can explain the descent.

Level 3 — Living Landing
Player-facing discovery

A second cargo rests on a plain lower floor between two empty docks. The elevated cargo can again be pulled beyond a ledge.

This time the falling cargo stops one row above the lower crate rather than reaching the floor. Pulling the upper cargo sideways off the lower one immediately makes it fall one additional row into a dock. The before-and-after contrast visibly establishes that cargo itself is a support.

The lower crate is then pulled into the remaining dock.

Initial map
   01234567890123
00 ..............
01 ..............
02 .....TC.......
03 ......####....
04 ..............
05 ..............
06 ....dCd.......
07 ...#####......
08 ..............
09 ..............
10 ..............
11 ..............
State-machine instance
Terrain:
  y=3, x=6..9
  y=7, x=3..7

Docks:
  (4,6)
  (6,6)

Cargo:
  A = (6,2)
  B = (5,6)

Tug:
  (5,2)

Attachment:
  null
Canonical solution
S L
D D D
S L
R R R D
S R

Compact replay string:

SLDDDSLRRRDSR

Resolution:

S,L: pull A from (6,2) to column 5.

A falls and stops at (5,5), directly atop B=(5,6).

D×3: tug reaches (4,5), left of A.

S,L: pull A to (4,5).

With B no longer beneath it, A falls one row to dock (4,6).

R×3,D: tug reaches (6,6), right of B.

S,R: pull B into dock (6,6).

Attribution and necessity check

The support observation is contrastive:

A first stops at row 5 while directly touching B.

The same cargo immediately falls to row 6 when moved one column away.

Both final docks are backed by ordinary floor terrain, while B initially occupies neither dock.

Without B, A would fall too low on its first descent and could never be lifted back to the upper row.

The level does not require any new device or symbolic inference.

Level 4 — Borrowed Berth
Player-facing discovery

The lower crate begins visibly docked, with another crate resting on it. The tug begins immediately to its left.

The natural low-cost probe is:

S, L

The line attaches to the lower crate, but the away-arrow only strains the line because the crate is loaded. No other object changes, making the failed pull easy to attribute.

The player must move the upper crate aside, pull the lower crate out of its completed dock, and send it over the platform edge. The upper crate then takes the vacated dock, while the former lower support is finished in the offset lower bay.

A two-cell overhang prevents the upper crate from simply being pulled right into the lower route.

Initial map
   01234567890123
00 ..............
01 ..............
02 ..............
03 ..............
04 ..............
05 ........C##...
06 .......TO.....
07 ..#########...
08 ..............
09 ............d.
10 ..........####
11 ..............
State-machine instance
Terrain:
  y=5, x=9..10
  y=7, x=2..10
  y=10, x=10..13

Docks:
  (8,6)
  (12,9)

Cargo:
  Top   = (8,5)
  Lower = (8,6), initially occupying dock (8,6)

Tug:
  (7,6)

Attachment:
  null
Canonical solution

Phases:

U S L
U R R R R R D D L L
S R R R
L L L L S R
S R R D D R D S R

Compact replay string:

USLURRRRRDDLLSRRRLLLLSRSRRDDRDSR

Resolution:

U,S,L: pull the top crate left. It falls to (7,6).

Route above and around the overhang to (9,6), right of the lower crate.

S,R×3: pull the docked lower crate through (9,6) and (10,6), then beyond the platform edge. It falls to (11,9).

Move left to (8,6), attach to the staged top crate at (7,6), and pull it right into the vacated dock (8,6).

Detach and reach (12,9).

Pull the lower crate from (11,9) into dock (12,9).

Necessity and anti-cleanliness check

The lower crate cannot move while the upper crate remains on it.

The overhang at (9,5)…(10,5) removes the direct top-crate shortcut to the right.

The starting dock is not merely decorative: its cargo must leave, and the upper crate must later refill it.

The lower floor is offset, so the player must commit to a drop column and later reapproach at the new height.

The puzzle is not a list of two deliveries; moving one crate changes whether the other is movable and which dock remains available.

Level 5 — Relay Crates
Player-facing discovery

A middle-height crate initially looks like the obvious support for the highest cargo. That is true—but it is not the whole plan.

After the highest cargo uses it and reaches the right dock, the lower-left cargo is deliberately dropped too low. It becomes a new support. The original support crate is then pulled off its platform and lands on this new support, changing from support to payload. Finally, the remaining lower crate becomes ordinary cargo again and reaches the bottom dock.

The two movable crates exchange strategic roles without any change to their rules or appearance.

Initial map
   01234567890123
00 ..............
01 ..............
02 ......CT......
03 ...####.......
04 ....C.........
05 ..###....d....
06 .......C##....
07 ...d..##......
08 ...##..d......
09 ...######.....
10 ..............
11 ..............
State-machine instance
Terrain:
  y=3, x=3..6
  y=5, x=2..4
  y=6, x=8..9
  y=7, x=6..7
  y=8, x=3..4
  y=9, x=3..8

Docks:
  (9,5)
  (3,7)
  (7,8)

Cargo:
  A = (6,2)
  B = (7,6)
  C = (4,4)

Tug:
  (7,2)

Attachment:
  null
Canonical solution

Phases:

S R
D D D S R R
S
U L L L L L S R
D D S L L
D S L L
S
D D D R R R R R R R U U L L L S R R

Compact replay string:

SRDDDSRRSULLLLLSRDDSLLDSLLSDDDRRRRRRRUULLLSRR

Resolution:

S,R: pull A into column 7. It falls to (7,5) atop B=(7,6).

Reach (8,5) and pull A right twice into dock (9,5).

Detach.

Reach (5,4), attach to C=(4,4), and pull right. C falls to (5,8).

Reach (6,6), left of B.

Pull B left once to (6,6), still supported by terrain.

Pull left again to column 5. B falls and stops at (5,7) atop C.

Descend to (4,7) and pull B left twice into dock (3,7).

Detach and route beneath the shelves to (6,8), right of C.

Pull C right twice into dock (7,8).

Necessity and anti-cleanliness check

A needs B to stop at row 5; otherwise the high dock is unreachable.

B cannot reach the row-7 dock by falling onto ordinary floor. C must first occupy (5,8) and catch it at (5,7).

C is not an independent delivery errand: its location directly determines B’s reachable height.

The crate that begins as the main support later becomes the upper payload.

The crate that begins as an apparent payload becomes support, then returns to being cargo.

Removing any cargo eliminates one required landing height.

Level 6 — Last Door
Player-facing discovery

Three elevated crates and one low base can form a four-crate tower in column 7. The tower is necessary because the docks lie at four different heights.

The top crate should not immediately be delivered to its left dock. That dock fronts a one-cell pocket at (3,5). A cargo pulled into (4,5) leaves the tug at (3,5), surrounded by terrain above, below, and left and by the cargo on the right. Because the tug cannot push and lacks backing space for another pull, entering the dock is a permanent commitment.

The correct plan is therefore:

construct the tower;

move the top crate only one cell left and leave it staged;

deliver the other three tower levels;

return and enter the terminal dock last.

Initial map
   01234567890123
00 .......TC.....
01 ........####..
02 ......C.......
03 ...####.......
04 ..##....C.....
05 ..#.d...###...
06 ..#####...d...
07 ....d...###...
08 ....###C..d...
09 .......####...
10 ..............
11 ..............
State-machine instance
Terrain:
  y=1, x=8..11
  y=3, x=3..6
  y=4, x=2..3
  y=5, x=2
  y=5, x=8..10
  y=6, x=2..6
  y=7, x=8..10
  y=8, x=4..6
  y=9, x=7..10

Docks:
  (4,5)
  (10,6)
  (4,7)
  (10,8)

Cargo:
  A    = (8,0)
  B    = (6,2)
  C    = (8,4)
  Base = (7,8)

Tug:
  (7,0)

Attachment:
  null
Canonical solution

Phases:

SL
DRDSR
DLDSL
DSLS

URRRRRRDDLLLSRRRS

DDDDLLLLLULLLUURRRSLLLS

DDDRRRRRRRRUULLLSRRRS

UUUULLLLLLDSLL

Compact replay string:

SLDRDSRDLDSLDSLSURRRRRRDDLLLSRRRSDDDDLLLLLULLLUURRRSLLLSDDDRRRRRRRRUULLLSRRRSUUUULLLLLLDSLL
Phase resolution
Construct the tower

S,L: A enters column 7 and falls to (7,7) atop Base.

D,R,D,S,R: B enters column 7 and falls to (7,6).

D,L,D,S,L: C enters column 7 and falls to (7,5).

The visible tower is now:

(7,5) C
(7,6) B
(7,7) A
(7,8) Base
Stage the terminal cargo
D S L S

Reach (6,5).

Pull C left to (6,5), supported by the left shelf.

Detach instead of continuing toward the terminal dock.

Deliver the second level right
U R×6 D×2 L×3 S R×3 S

This reaches the right of B=(7,6), pulls it into dock (10,6), and detaches.

Deliver the third level left
D×4 L×5 U L×3 U×2 R×3 S L×3 S

The route goes below the alternating shelves, returns to the left of A=(7,7), pulls it into (4,7), and detaches.

Deliver the base right
D×3 R×8 U×2 L×3 S R×3 S

The base reaches dock (10,8).

Commit to the terminal dock
U×4 L×6 D S L×2

The tug returns to the left of the staged crate at (6,5), attaches, and pulls it through (5,5) into (4,5).

The final pull leaves the tug at (3,5), where it is permanently enclosed. All cargo is now docked, so this commitment completes the level.

Necessity and anti-cleanliness check

The four dock heights require four stacked cargo levels.

Only the top crate of the tower can move at each dismantling stage.

Pulling the top crate directly into (4,5) before the other deliveries makes completion impossible because the tug cannot leave (3,5).

The terminal pocket is formed entirely by ordinary terrain, cargo collision, and the no-push rule; it is not a special locking dock.

The solution contains a genuine branch and revisit: the top delivery is begun, suspended, and finished only after three dependent deliveries elsewhere.

Every shelf controls either a landing height or a necessary route around an occupied cargo.

Level 7 — Vacated Berth
Player-facing discovery

The final level begins with its base crate already glowing green inside dock (7,10). Treating it as permanently finished is a trap in reasoning, not a hidden rule.

The crate must first serve as the base of a central tower. An upper support from that tower is then pulled away and dropped to a second location, where it catches another payload. While that second payload remains staged, the remaining central tower is dismantled.

The original base must eventually leave its starting dock and occupy (5,10). Only then can the repurposed support be pulled left into the now-vacant dock (7,10).

Thus the final assignment is not based on cargo identity:

the starting docked cargo leaves;

a former tower support takes its berth;

another support becomes the foundation of a second stack;

the side payload is staged while an independent lower branch is rearranged.

Initial map
   01234567890123
00 .......TC.....
01 ........####..
02 ......C.......
03 ...####..C....
04 ......C..####.
05 ...####.......
06 ..............
07 ...d..........
08 ...####.......
09 ..........d...
10 ..d..d.O.##...
11 .##########...
State-machine instance
Terrain:
  y=1, x=8..11
  y=3, x=3..6
  y=4, x=9..12
  y=5, x=3..6
  y=8, x=3..6
  y=10, x=9..10
  y=11, x=1..10

Docks:
  (3,7)
  (10,9)
  (2,10)
  (5,10)
  (7,10)

Cargo:
  M    = (8,0)
  U    = (6,2)
  Q    = (9,3)
  P    = (6,4)
  Base = (7,10), initially occupying dock (7,10)

Tug:
  (7,0)

Attachment:
  null
Canonical solution

Phases:

SL
RDDSR
LDDSR
LDDLDSLLLL
S

URRRRRRDDSR
LUUUUUSL

DDDLLLLLLDDDDRRRRRUSL
DSLLLL

SURRRRRDSLL

SURRRURRDSRR

SULLLLDDSL

Compact replay string:

SLRDDSRLDDSRLDDLDSLLLLSURRRRRRDDSRLUUUUUSLDDDLLLLLLDDDDRRRRRUSLDSLLLLSURRRRRDSLLSURRRURRDSRRSULLLLDDSL
Phase resolution
Build the central tower

S,L: pull M into column 7; it falls to (7,9) atop Base.

R,D,D,S,R: pull U into column 7; it falls to (7,8).

L,D,D,S,R: pull P into column 7; it falls to (7,7).

The stack is:

(7,7)  P
(7,8)  U
(7,9)  M
(7,10) Base, still in its original dock
Deliver the high payload
L D D L D S L×4

Reach the left of P and pull it across the supported shelf into dock (3,7).

Then detach.

Repurpose the upper support
U R×6 D×2 S R

Reach (8,8), right of U=(7,8), and pull right.

U enters column 8, loses its central support, and falls to (8,10).

Create the second stack
L U×5 S L

Reach (8,3), left of Q=(9,3), and pull left.

Q falls down column 8 and stops at (8,9) atop U=(8,10).

Do not deliver Q yet.

Dismantle the central branch
D×3 L×6 D×4 R×5 U S L

Reach (6,9), left of M=(7,9), attach, and pull left.

M enters (6,9), loses the base beneath it, and falls to (6,10).

Then:

D S L×4

Pull M left into dock (2,10).

Vacate the original dock
S U R×5 D S L×2

Detach, reach (6,10), and pull Base from (7,10) to (5,10).

The dock at (7,10) is now empty for the first time.

This move could not happen earlier because M was resting on the base.

Deliver the staged side payload
S U R×3 U R×2 D S R×2

Detach, route above the adjacent lower cargo, reach (9,9), and pull Q right twice into dock (10,9).

Give the old berth to the repurposed support
S U L×4 D×2 S L

Detach and reach (7,10), immediately left of U=(8,10).

Pull U left into the dock that the base originally occupied.

Final cargo positions:

(3,7)
(10,9)
(2,10)
(5,10)
(7,10)
Necessity and anti-cleanliness check

P cannot reach row 7 without the full central support stack.

Q cannot reach row 9 unless U is moved to (8,10) first.

M must leave the central stack before the base can move, because the base is loaded.

The base must vacate (7,10) before U can be pulled left:

U is at (8,10);

a left pull requires the tug to occupy (7,10);

the starting base otherwise blocks that position.

The original docked crate is not “the crate assigned to that dock.” Cargo identity is irrelevant.

Q can remain staged while the lower branch is rearranged, so the mature puzzle is a dependency network rather than a single encounter-order recipe.

The bottom-right terrain prevents U from escaping to the right, making the vacated central dock its only viable final berth.

Every cargo changes role at least once: tower member, support, staged payload, route blocker, or final delivery.

10. Implementation assertions

An implementation should include tests for the following invariants:

Every supplied initial cargo configuration is stable under gravity.

No authored initial state contains overlapping cargo, cargo in terrain, or tug in terrain.

All levels begin detached.

All clicks leave the complete state unchanged.

A cargo may be pulled out of a dock.

A dock does not support cargo without terrain, board boundary, or cargo beneath it.

Pulling a loaded crate is invalid.

Pulling a top crate away may immediately expose a new fall.

An unsupported vertical stack falls as one unit.

Attachment breaks on the first vertical fall microtick of the attached cargo.

Completion is checked only after complete settlement.

Cargo IDs are ignored by the validator.

Replaying each canonical action string reaches exactly the listed dock set.

In Level 6, entering dock (4,5) leaves the tug trapped at (3,5).

In Level 7, the cargo that starts in dock (7,10) ends in (5,10), while a different cargo ends in (7,10).

The result is one focused formal system: every late puzzle is still about choosing where to pull a crate before gravity decides its future height, while preserving the space from which the next necessary pull can be made.
