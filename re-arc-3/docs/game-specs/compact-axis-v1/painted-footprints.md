Painted Footprints

This is a game about painting routes becoming difficult because every courier step—and every push—rewrites floor that later parcels depend on.

Authored level count: exactly 7.

The courier carries one visible pot. Walking paints ordinary floor with that pot’s color. Parcels can move only onto floor already matching their own color. This makes travel inseparable from route construction: walking somewhere is also editing it, and pushing a parcel paints the route behind it with whatever the courier is currently carrying.

The game begins as a clear “paint first, push second” puzzle. It develops into a compact logistics system involving destructive shortcuts, temporary parcel parking, time-shared crossings, and deliberate brushback conversion.

1. Native board and rendering
Logical geometry

Every level uses the same 12×10 logical lattice.

Coordinates are (x,y).

x=1 is the left column; x=12 is the right column.

y=1 is the top row; y=10 is the bottom row.

Movement is cardinal and advances exactly one logical cell.

Any coordinate not explicitly designated as floor, pad, or station is a wall.

64×64 canvas

HUD: pixel rows 0..9.

Empty separator: rows 10..11.

Board origin: pixel (2,12).

Logical tile size: 5×5 pixels.

Logical cell (x,y) occupies:

px = 2 + 5·(x-1) ... 6 + 5·(x-1)
py = 12 + 5·(y-1) ... 16 + 5·(y-1)

The board therefore occupies pixels x=2..61, y=12..61.

There are no grid lines or internal cell borders. Adjacent walls and floors render as contiguous regions.

Restrained palette use
Purpose	Color IDs
Background	5
Walls	4
Bare floor	1
Courier body	0, 2, 3
Blue family	9, accent 10
Red family	8, accent 13
Yellow family	11, accent 12
Correct-pad border	14
Collision/error flash	0, then settled underlying color

Colors 6, 7, and 15 are not required.

Color-pattern families

Patterns reinforce color without introducing symbolic mappings.

For a painted ordinary 5×5 floor cell, using local coordinates 0..4:

Blue: fill with 9; local column x=2 uses 10.

Red: fill with 8; local diagonal x=y uses 13.

Yellow: fill with 11; local row y=2 uses 12.

The same vertical, diagonal, or horizontal motif appears on parcels, pads, stations, the carried pot, and the HUD pot indicator.

Courier

The courier is a connected multi-pixel figure inside one logical cell:

Body: a 2×3 block in colors 0 and 2.

Pot attachment: a connected 2×2 colored cluster on the upper-right side.

The pot always shows the current pot_color.

There is no facing variable. The next directional action alone determines movement or push direction.

The floor remains visible around the figure.

A larger 7×7 pot icon in the HUD repeats the same color and motif.

Parcels

A parcel is a centered 3×3 color motif with four dark corner brackets in its
5×5 cell. The brackets distinguish the movable parcel from painted floor and
the courier while leaving the supporting-floor ring visible.

Blue parcel: colors 9/10, vertical motif.

Red parcel: colors 8/13, diagonal motif.

Yellow parcel: colors 11/12, horizontal motif.

When the courier is immediately behind a parcel, the parcel’s destination edge
turns green if the next directional action is a legal push. A dark edge means
the destination support is absent, blocked, or the wrong color.

Pads

A pad is fixed, traversable by the courier, and non-paintable.

Its outer ring uses its color family.

Its center uses the corresponding motif.

Its logical support color is always its authored pad color.

A parcel of another color therefore cannot enter it.

When correctly occupied, its outer ring becomes color 14.

The green border disappears if the parcel is subsequently pushed off.

Paint stations

Stations occupy a full 5×5 logical cell.

Top and left rim: color 0.

Bottom and right rim: color 3.

Interior 3×3: alternating family base/accent stripes.

They look raised and materially different from floor or pads.

The courier may occupy a station.

Parcels may not occupy or cross a station.

Stations cannot be painted.

There are no clickable objects.

2. State-machine contract
Static level data
floor_cells: set[(x,y)]
walls: every board coordinate not in floor_cells, pads, or stations
stations: dict[(x,y) -> B | R | Y]
pads: dict[(x,y) -> B | R | Y]
step_cap: integer
initial_state
Dynamic state
courier_pos: (x,y)
pot_color: B | R | Y
parcel_pos[id]: (x,y)
parcel_color[id]: B | R | Y       # static after initialization
paint[(x,y)]: NONE | B | R | Y    # ordinary floor only
step_count: integer
status: PLAYING | WON | FAILED

There is no selected object, cursor state, facing state, hidden inventory, queue, timer phase, or random state.

The starting courier cell is not automatically painted. Painting occurs only when the courier enters an ordinary floor cell during a directional step.

3. Action semantics

The environment exposes exactly:

UP, DOWN, LEFT, RIGHT, SPACE, CLICK(x,y)
Directional action

A directional action increments step_count by one, including a blocked move or failed push.

Let:

adjacent = courier_pos + direction
Case A: wall or board edge

Nothing in the logical state changes except step_count.

Render a brief courier bump.

Case B: empty ordinary floor

Move the courier to adjacent.

Set paint[adjacent] = pot_color, replacing any previous paint.

Case C: empty pad

Move the courier to the pad.

Do not change the pad or pot.

The pad remains fixed support of its authored color.

Case D: empty station

Move the courier onto the station.

Set pot_color to the station’s color.

Do not paint the station.

The pot swap happens in the same step() as the movement.

Case E: adjacent parcel

Let:

destination = adjacent + direction

A push succeeds only when all of the following are true:

destination is ordinary floor or a pad.

It is not a station, wall, board edge, or another parcel.

Its support color equals the parcel’s color:

support(destination) =
    paint[destination]     if ordinary floor
    pad_color[destination] if pad

On a successful push:

Move the parcel from adjacent to destination.

Move the courier into the parcel’s former cell.

If the former parcel cell is ordinary floor, set its paint to the courier’s current pot color.

If the former parcel cell is a pad, leave the pad unchanged.

Do not paint the parcel’s new destination while it is occupied.

Only the destination support matters. The color under the parcel’s current position does not constrain departure.

There are no chain pushes. A second parcel directly behind the first blocks the push.

On a failed push:

No logical state changes except step_count.

The parcel shudders toward the attempted destination.

The destination receives a temporary white outline.

Its settled visible paint or wall appearance remains unchanged, explaining the failure.

SPACE

SPACE restarts the current authored level atomically:

Restore all initial positions, paint, pot, and status.

Set step_count=0.

It does not consume an arrow-step budget.

This is a recovery control, not a puzzle mechanic.

Click

CLICK(x,y) is ignored for every coordinate.

No cell is clickable.

No selection or cursor appears.

Clicks do not increment step_count.

There is no hidden click affordance to discover.

4. Update and validation order

For every directional step():

Increment step_count.

Resolve collision, movement, station exchange, or parcel push.

Commit all position, pot, and paint changes.

Recompute correct-pad borders.

Check success.

If not successful and step_count >= step_cap, set status=FAILED.

Success takes precedence if the final valid action exactly reaches the cap.

Win condition

A level is won exactly when:

Every parcel occupies a pad of its own color, and

Every authored pad is occupied by its matching parcel.

The following are ignored by the win validator:

Courier position.

Current pot color.

Paint on ordinary floor.

Number or location of non-goal painted cells.

Whether a parcel previously occupied and left a pad.

All pads must be correct simultaneously.

Step HUD

A quiet bar at HUD pixels x=14..61, y=3..5 shows remaining directional budget.

Remaining portion: color 2.

Spent portion: color 4.

Remaining portion changes to 12 below 15%.

No digits or text are rendered.

5. Animation contract

Animation frames never consume steps and never accept input.

Ordinary movement and painting

Render the committed next state immediately. No ornamental walking animation is required.

Successful push

Use a short deterministic three-frame translation:

Parcel at one-third displacement; courier remains at source.

Parcel at two-thirds displacement; courier begins entering the vacated cell.

Both occupy their committed logical cells; vacated-cell paint is visible.

The logical post-state is computed once inside step(). The three frames are a pure visualization of that transition.

Station exchange

After the committed movement:

Pulse the station interior once.

Replace the pot colors immediately.

Use two rendering frames, with no additional state update.

Failed push

Use three frames:

Parcel displaced one rendered pixel toward destination.

Parcel restored; destination outlined with 0.

Outline disappears, revealing the unchanged destination.

Completion

All correct-pad borders pulse 14 for six rendering frames, then the level reports WON.

Budget exhaustion

The outer canvas edge flashes 13. The board then remains frozen until SPACE.

6. Documentation-map notation

The following maps are documentation only. These characters are never rendered in the game.

#  wall
.  ordinary paintable floor
@  courier
B  blue parcel
R  red parcel
Y  yellow parcel
b  blue pad
r  red pad
y  yellow pad
1  blue station
2  red station
3  yellow station

Initial paint beneath parcels is specified separately because the documentation glyph shows the parcel rather than its supporting floor.

Action replays use U/D/L/R and multiplication notation such as R×5.

Level 1 — The Wrong-Colored Step

Teaches: parcels enter only matching support; matching pads are the visible objective.
Intended solution: 4 directional actions.
Step cap: 64.

x→ 123456789012
1  ############
2  ############
3  #####b######
4  #####.######
5  ####@B.#####
6  ####..######
7  ############
8  ############
9  ############
10 ############
Initial state
Courier: (5,5)
Pot: B
Blue parcel: (6,5)
Blue pad: (6,3)

Initial blue ordinary floor:
(5,5), (5,6), (6,6), (6,5), (6,4)

Initial red ordinary floor:
(7,5)

All other ordinary floor: NONE
Selection/cursor: none
Player-facing discovery

The courier begins directly west of the parcel. The most immediate experiment is RIGHT.

That push fails because the destination (7,5) is visibly red. The parcel shudders and the red destination is outlined. A blue support cell is visible immediately north of the parcel, with a matching blue pad above it.

The southern blue cells let the player change pushing direction without introducing any other rule.

All cells traversed by the intended solution are already blue, so walking produces no visible repainting. This keeps footprint painting out of the first causal lesson.

Exact replay
D, R, U×2

Checkpoints:

D,R: courier reaches (6,6), below the parcel.

First U: parcel moves (6,5) → (6,4).

Second U: parcel moves onto blue pad (6,3).

Final state:

Blue parcel: (6,3)
Courier: (6,4)
step_count: 4
Attribution and necessity

The red comparison cell is necessary evidence, not a decorative decoy.

The successful and failed pushes differ only in destination support color.

There are no stations, bare route cells, second parcels, budgets under pressure, or simultaneous state changes.

Removing the color requirement would make the immediate RIGHT push succeed and invalidate the lesson.

Level 2 — Paint, Then Circle

Teaches: entering ordinary floor paints it with the carried pot; route preparation leaves the courier on the wrong side.
Intended solution: 10 directional actions.
Step cap: 96.

x→ 123456789012
1  ############
2  ############
3  ############
4  ###...######
5  ###@B.b#####
6  ###...######
7  ############
8  ############
9  ############
10 ############
Initial state
Courier: (4,5)
Pot: B
Blue parcel: (5,5)
Blue pad: (7,5)

Initial blue ordinary floor:
(5,5)

All other ordinary floor: NONE
Selection/cursor: none
Player-facing discovery

RIGHT attempts to push the parcel into bare cell (6,5) and fails. The destination is neither blue nor another obstruction.

The small upper and lower loops expose (6,5) from the side. Walking around causes every entered cell to turn visibly blue. Entering (6,5) paints the exact failed destination, but leaves the courier in front of the parcel. The player must circle back behind it.

This distinguishes the intended rule from simpler mistaken hypotheses such as “bare cells eventually activate” or “touching the parcel changes its route.”

Exact replay
U, R×2, D, U, L×2, D, R×2

Checkpoints:

U,R,R,D: enter (6,5); it becomes blue.

U,L,L,D: return to (4,5) behind the parcel.

First R: parcel moves to newly painted (6,5).

Second R: parcel enters blue pad (7,5).

Final state:

Blue parcel: (7,5)
Courier: (6,5)
step_count: 10
Attribution and necessity

The only new causal effect is footprint painting.

The gap begins bare and must be entered before the parcel can use it.

The loop is required both to paint the destination and to recover pushing position.

There is no station whose effect could explain the color change.

Level 3 — Exchange the Pot

Teaches: entering a raised station automatically replaces the carried pot.
Intended solution: 12 directional actions.
Step cap: 128.

x→ 123456789012
1  ############
2  ############
3  ############
4  ###2..######
5  ###.#.######
6  ###@R.r#####
7  ###...######
8  ############
9  ############
10 ############
Initial state
Courier: (4,6)
Pot: B
Red parcel: (5,6)
Red pad: (7,6)
Red station: (4,4)

Initial red ordinary floor:
(5,6)

All other ordinary floor: NONE
Selection/cursor: none
Player-facing discovery

Known play suggests painting (6,6) and returning behind the red parcel. Doing so with the visible blue pot produces blue support, and the red parcel still refuses to move.

The raised red-striped station at (4,4) is adjacent to the route and is the only unfamiliar object. Entering it immediately replaces the visible pot with red. Subsequent footprints are red, and the same route preparation now works.

Exact replay
U×2, R×2, D×3, L×2, U, R×2

Checkpoints:

First U: (4,5) is painted blue.

Second U: enter station (4,4); pot becomes red.

R,R,D,D: reach (6,6) and paint it red.

D,L,L,U: circle behind the parcel.

R,R: push onto (6,6) and then red pad (7,6).

Final state:

Red parcel: (7,6)
Courier: (6,6)
Pot: R
step_count: 12
Attribution and necessity

The station is the sole new object and sole new causal unknown.

Its effect is visible before another consequential action occurs.

Blue and red attempts operate on the same destination, ruling out a positional explanation.

The station is necessary because no other red source exists.

Level 4 — One Crossing, Two Times

Combines: destructive walking, two colors, and a time-shared crossing.
Intended solution: 37 directional actions.
Step cap: 192.

x→ 123456789012
1  ############
2  #####r######
3  #####..1####
4  #......@####
5  #.B....b####
6  #####...####
7  #####R.2####
8  #####..#####
9  ############
10 ############
Initial state
Courier: (8,4)
Pot: R

Blue parcel: (3,5)
Blue pad: (8,5)
Blue station: (8,3)

Red parcel: (6,7)
Red pad: (6,2)
Red station: (8,7)

Initial blue ordinary floor:
(3,5)

Initial red ordinary floor:
(6,7)

All other ordinary floor: NONE
Selection/cursor: none

The horizontal blue route and vertical red route share ordinary floor cell:

Crossing: (6,5)
Player-facing discovery

The first three levels might suggest that routes can simply be prepared independently. Here that fails: whichever color is painted last at (6,5) disables the other parcel.

The crossing must be used at different times:

Make it blue and send the blue parcel across.

Later repaint it red and send the red parcel through it.

Walking is no longer merely route creation. A shortcut taken with the wrong pot visibly erases useful work.

Exact replay
Prepare and deliver blue
U, L, D×2, L×3
U, L×2, D
R×5

Effects:

Enter blue station (8,3).

Paint horizontal cells (7,5) through (4,5) blue, including crossing (6,5).

Circle through the upper-left loop to (2,5).

Push the blue parcel from (3,5) onto pad (8,5).

Change to red and prepare the vertical route
D, R, D
L, U×4, L, D×3

Effects:

Enter red station (8,7).

Approach the red route from above.

Paint (6,3), (6,4), crossing (6,5), and (6,6) red.

Deliver red
R, D×2, L
U×5

Final state:

Blue parcel: (8,5)
Red parcel: (6,2)
Courier: (6,3)
step_count: 37
Anti-cleanliness and necessity

Removing crossing (6,5) would reduce the level to two independent deliveries.

The upper and lower circulation paths are needed to approach parcels from their pushing sides without casually traversing the active lane.

A plan that prepares both complete routes before moving either parcel necessarily leaves the crossing wrong for one of them.

Blue-first and carefully staged red-first families may both exist; every valid family must nevertheless change the crossing’s support over time.

Level 5 — Passing Bay

Combines: a shared corridor, temporary parcel staging, route repainting, and station revisits.
Intended solution: 57 directional actions.
Step cap: 256.

x→ 123456789012
1  ############
2  ############
3  ###..@...###
4  #.1.......##
5  #rB.....Rb##
6  #.........##
7  ##2#########
8  ############
9  ############
10 ############
Initial state
Courier: (6,3)
Pot: B

Blue parcel: (3,5)
Blue pad: (10,5)

Red parcel: (9,5)
Red pad: (2,5)

Blue station: (3,4)
Red station: (3,7)

Initial blue ordinary floor:
(3,5)

Initial red ordinary floor:
(9,5)

All other ordinary floor: NONE
Selection/cursor: none

The passing bay is ordinary floor cell:

Bay: (6,4)
Player-facing discovery

The two parcels face each other in a one-cell transport corridor. They cannot pass, and the opposite-color endpoint pads prevent simply pushing either parcel outward onto the wrong endpoint.

The visible cell above the corridor’s midpoint provides a physical answer: park one parcel in the bay, let the other pass, then return the parked parcel.

The corridor consequently has three roles:

Blue approach route.

Red transit route.

Blue route again after the red parcel has passed.

Exact replay
Paint the bay and stage the blue parcel
D×2, L×2
U, L×2, D
R×3
D, R, U

Effects:

Paint bay (6,4) and corridor (6,5)..(4,5) blue.

Reach red pad (2,5), which is traversable by the courier.

Push blue parcel from (3,5) to (6,5).

Circle below and push it upward into bay (6,4).

Checkpoint:

Blue parcel: (6,4)
Red parcel: (9,5)
Main corridor clear
Repaint the corridor red and send red left
D, L×3, D
U×2, R×5
D, R×2, U
L×7

Effects:

Enter red station (3,7).

Paint corridor (3,5)..(8,5) red.

Circle below to blue pad (10,5), east of the red parcel.

Push the red parcel left onto red pad (2,5).

Restore blue support, recover the staged parcel, and deliver it
U
D, R×6
U×2, L×3
D
L, D
R×4

Effects:

Re-enter blue station (3,4).

Paint (3,5)..(9,5) blue.

Move to (6,3) and push the staged parcel down to (6,5).

Circle to (5,5).

Push it right onto blue pad (10,5).

Final state:

Blue parcel: (10,5)
Red parcel: (2,5)
Courier: (9,5)
step_count: 57
Anti-cleanliness and necessity

Without bay (6,4), the parcels cannot pass.

The endpoint pads actively reject the wrong parcel rather than acting as neutral corridor extensions.

The blue station is encountered early but must be deliberately revisited late.

The corridor is not consumed in the same order objects appear: it must be prepared blue, overwritten red, and restored blue.

A mirrored strategy that stages red instead of blue may be allowed, but still requires the same passing-bay and time-sharing principles.

Level 6 — Brushback Spine

Introduces as a required planning tool: pushing with another color deliberately converts the completed route behind a parcel.
Intended solution: 58 directional actions.
Step cap: 72.

x→ 123456789012
1  ########b###
2  ........@###
3  .#######.###
4  .######..###
5  .R.B.....###
6  ##.......###
7  ####2....###
8  ########.###
9  ########r###
10 ############
Initial state
Courier: (9,2)
Pot: B

Blue parcel: (4,5)
Blue pad: (9,1)

Red parcel: (2,5)
Red pad: (9,9)

Red station: (5,7)

Initial blue ordinary floor:
(4,5), (9,2)

Initial red ordinary floor:
(2,5)

All other ordinary floor: NONE
Selection/cursor: none

Shared horizontal spine:

(3,5)..(9,5)

Blue exit branch:

(9,4), (9,3), (9,2), blue pad (9,1)

Red exit branch:

(9,6), (9,7), (9,8), red pad (9,9)
Player-facing discovery

The blue parcel must traverse the horizontal spine and turn north. The red parcel waits behind it and later needs the same spine before turning south.

Painting the spine blue lets the blue parcel move but leaves no economical opportunity to rebuild the entire route red afterward. The red station is positioned beside the red exit branch before the blue parcel is moved.

The productive experiment is to switch to red before pushing blue. On the first successful push, the blue parcel advances over blue support while its vacated cell immediately becomes red. Repeated pushes reveal an unmistakable red trail growing behind the blue parcel.

The blue parcel uses the color in front of it. The red parcel benefits from the color left behind it.

Exact replay
Prepare blue support
D×3, L×4

This paints:

Blue branch: (9,3), (9,4), (9,5)
Blue spine:  (8,5), (7,5), (6,5), (5,5)

The courier ends at (5,5), immediately in front of the blue parcel.

Acquire red and prepaint the red exit branch
D×2
R×4, D, U×2
L×6, U

Effects:

Enter red station (5,7).

Paint (9,8), (9,7), and (9,6) red.

Return through the lower circulation space to (3,5), behind the blue parcel.

Entering (3,5) paints the red follower’s first destination.

Push blue while carrying red
R×5
D, R
U×4

Effects:

Blue parcel moves (4,5) → (9,5).

The vacated spine cells become red one by one.

Reposition at (9,6).

Push blue north onto pad (9,1) using the still-blue vertical branch.

The vacated junction (9,5) becomes red.

Checkpoint:

Blue parcel: (9,1)
Spine (3,5)..(9,5): red
Red branch (9,6)..(9,8): red
Reach the follower’s west side and deliver it
L×8, D×3
R×7
U, R
D×4

Effects:

Use the upper bypass to reach (1,5).

Push red parcel east to junction (9,5).

Move above it to (9,4).

Push it south onto red pad (9,9).

Final state:

Blue parcel: (9,1)
Red parcel: (9,9)
Courier: (9,8)
step_count: 58
Budget and anti-degenerate check

The 72-step cap is visible from the first frame. The intended replay leaves 14 directional actions for local experiments and failed pushes.

A blue-pot-first delivery creates the wrong-colored spine. The direct repair family must then:

Reach the only red station after blue delivery.

Repaint the shared spine.

Repaint the southern branch.

Reach the red parcel’s west side through the upper bypass.

Perform its eastward and southward pushes.

A regression search over the intended parcel route gives 74 actions for that family, so it fails the authored cap. Switching before the push and using brushback is therefore part of validation rather than merely a shortcut.

The cap is not used to teach an invisible timer; it distinguishes two visibly different physical plans that otherwise converge to similar paint states.

Level 7 — Three-Color Relay

Final composition: future-blocked stations, route preservation, two brushback conversions, a shared crossing, and three parcels.
Intended solution: 92 directional actions.
Step cap: 192.

x→ 123456789012
1  ############
2  #####y##2###
3  ........b###
4  .##.#.##@.##
5  .#........##
6  .R.B.....r.#
7  #####......#
8  #####.##.#3#
9  #####Y##.###
10 #####....###
Initial state
Courier: (9,4)
Pot: B

Blue parcel: (4,6)
Blue pad: (9,3)

Red parcel: (2,6)
Red pad: (10,6)

Yellow parcel: (6,9)
Yellow pad: (6,2)

Red station: (9,2)
Yellow station: (11,8)

Initial blue ordinary floor:
(4,6), (9,4)

Initial red ordinary floor:
(2,6)

Initial yellow ordinary floor:
(6,9)

All other ordinary floor: NONE
Selection/cursor: none
Critical authored relationships
Red station pocket

The red station (9,2) has only one entry, through blue pad (9,3).

Before blue delivery:

courier may traverse (9,3) → (9,2)

After blue delivery:

blue parcel at (9,3) blocks the station

The parcel cannot be pushed off afterward:

(9,4) has been converted red.

(8,3) has been walked over with red.

(10,3) is a wall.

Yellow station pocket

The yellow station (11,8) is reached only through:

red pad (10,6) → (11,6) → (11,7) → station (11,8)

After red delivery, the red parcel at (10,6) blocks this pocket.

It also cannot be pushed away:

(9,6), (10,5), (10,7), and (11,6) are yellow by then or are unsuitable.

The remaining sides are walls.

Shared crossing
Crossing: (6,6)

It must successively be:

Blue for the blue parcel.

Red for the red parcel.

Yellow for the yellow parcel.

The last change occurs specifically when the red parcel passes it while the courier carries yellow.

Player-facing discovery

The starting blue pot has no replacement station. The blue route must therefore be completely prepared before entering the visible red station.

The red station will be blocked by the delivered blue parcel, so red must be acquired first. Pushing blue with that red pot converts the horizontal route for the red follower.

The yellow station is similarly placed behind the future red delivery. Yellow must be acquired before red reaches its pad.

However, painting the central crossing yellow before red passes would strand the red parcel, and the blocked red station would prevent repair. The correct timing is to prepare the yellow route above and below the crossing while leaving the crossing red. Then push the red parcel while carrying yellow. Its destination remains red, while its vacated cells—including the crossing—become yellow.

The three parcels therefore form a paint relay:

Blue parcel leaves red support.
Red parcel leaves yellow support.
Yellow parcel uses the final converted crossing.
Exact replay
Phase A — Prepare blue before blue becomes unavailable
D×2, L×4

Paints blue support:

(9,5), (9,6), (8,6), (7,6), (6,6), (5,6)

The courier ends at (5,6), in front of the blue parcel.

Phase B — Acquire red without damaging the blue lane
U, R×4, U×3

This reaches:

(5,5) → ... → (9,5) → (9,4) → blue pad (9,3) → red station (9,2)

The pot becomes red.

Return through the upper-left route rather than walking over the prepared horizontal lane:

D, L×5, D×2, L, D

The courier reaches (3,6), behind the blue parcel. Cell (3,6) becomes red.

Phase C — Blue transports red paint
R×5
D, R
U×3

Effects:

Push blue parcel from (4,6) to (9,6).

Cells (4,6)..(8,6) become red.

Reposition at (9,7).

Push blue north through (9,5), (9,4), and onto blue pad (9,3).

Junction (9,6) becomes red.

Checkpoint:

Blue parcel: (9,3)
Red station: blocked
Horizontal support (3,6)..(9,6): red
Courier: (9,4)
Pot: R
Phase D — Acquire yellow before red blocks it
R, D×2, R, D×2

Path:

(9,4) → (10,4) → (10,5) → red pad (10,6)
→ (11,6) → (11,7) → yellow station (11,8)

The pot becomes yellow.

Phase E — Prepare the two yellow route halves without touching the red crossing

Prepare the lower half:

U×2, L, D, L×4, D

This paints:

(6,7), (6,8)

The crossing (6,6) remains red.

Use the right-side path and fixed red pad to reach the upper half:

U, R×4, U×2, L×4, U×2

This paints:

(6,5), (6,4), (6,3)

The courier never enters (6,6).

Then reach the red parcel’s west side through the upper-left route:

L×5, D×3

The courier arrives at (1,6).

Checkpoint:

Yellow route above crossing: prepared
Yellow route below crossing: prepared
Crossing (6,6): still red
Pot: Y
Phase F — Red transports yellow paint
R×8

The red parcel moves from (2,6) onto red pad (10,6).

Each destination ahead is red when entered. Each vacated cell becomes yellow. In particular:

When red moves (6,6) → (7,6),
crossing (6,6) becomes yellow.

Checkpoint:

Red parcel: (10,6)
Yellow station: blocked
Horizontal cells (2,6)..(9,6): yellow
Crossing (6,6): yellow
Courier: (9,6)
Pot: Y
Phase G — Deliver yellow through the converted crossing

Reach the south side of the yellow parcel:

D×4, L×3

The courier reaches (6,10).

Then:

U×7

The yellow parcel traverses:

(6,9)
→ (6,8)
→ (6,7)
→ converted crossing (6,6)
→ (6,5)
→ (6,4)
→ (6,3)
→ yellow pad (6,2)

Final state:

Blue parcel: (9,3)
Red parcel: (10,6)
Yellow parcel: (6,2)
Courier: (6,3)
Pot: Y
step_count: 92
Anti-cleanliness and necessity

Preparing yellow crossing (6,6) before red passes makes the red route unusable.

After blue delivery, there is no reachable red station.

After red delivery, there is no reachable yellow station.

Thus neither crossing mistake can be repaired by another station errand.

The upper and lower yellow route halves are spatially separated specifically so they can be prepared without touching the crossing.

Blue’s delivery position is also the blocker for the red station; red’s delivery position is the blocker for the yellow station.

The delivered parcels are not abstract switches. Their ordinary collision bodies physically occupy the only station entrances.

Removing either station pocket, the crossing, or the separated yellow approach routes collapses the dependency network into a simple delivery chain.

7. Progression summary
Level	Main evidence or composition	Verified replay	Cap
1	Matching destination support and visible pad goal	4	64
2	Footprint painting; paint, circle, push	10	96
3	Automatic pot exchange	12	128
4	Destructive travel and time-shared crossing	37	192
5	Shared corridor and temporary parcel bay	57	256
6	Deliberate brushback conversion	58	72
7	Chained brushback, blocked stations, three-color crossing	92	192
8. Reuse and contrasting roles

The game gains depth without adding unrelated devices:

Mechanic	Early role	Later contrasting roles
Walking paint	Construct one missing destination	Destroy a prepared lane, preserve a route by detouring, prepare separated route halves
Parcel	Cargo to a pad	Temporary corridor blocker, object parked in a bay, moving paint applicator, future station blocker
Station	Simple color source	Irreversible order constraint once a parcel blocks its entrance
Pad	Visible destination	Fixed mismatched endpoint, traversable courier bridge, future station doorway
Shared floor	One parcel’s route	Time-shared crossing, three-phase corridor, brushback inheritance
Pot choice during push	Initially irrelevant because colors agree	Determines the next parcel’s route behind the current parcel

The operational rules never change. Every role reversal comes from position, pot timing, route geometry, or parcel occupancy.
