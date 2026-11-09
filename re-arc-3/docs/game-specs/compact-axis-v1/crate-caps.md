Crate Caps

Exact authored level count: 7

This is a game about stacking pieces becoming difficult because only completed stacks can support movement.

The courier moves through a broken tiled ruin. Loose discs can be carried one at a time and placed into circular pits. One disc remains visibly below floor level and cannot support the courier; two discs form a bright, flush cap that becomes traversable terrain. Every repaired crossing can later be dismantled, so a bridge is simultaneously a route, a store of material, and a future commitment.

The player’s main insight is not merely “put two discs in every pit.” It is deciding which completed stack must remain terrain, which can safely become material, and from which side a disc will still be recoverable.

1. Logical world
1.1 Board geometry

The native game board is an 11×11 logical lattice.

Coordinates are (row, column).

Rows increase downward.

Columns increase rightward.

Coordinates are 1-indexed.

Each action moves the courier by at most one logical cell.

A level lists its active floor cells and pits explicitly.

Every unlisted coordinate is VOID.

VOID is impassable.

There are no internal grid-line borders.

A level may use only a small irregular subset of the 11×11 lattice. This negative space is part of the puzzle geometry.

1.2 Terrain and objects

Each logical coordinate contains at most one terrain type:

FLOOR: always traversable unless occupied by a loose disc.

EXIT: a green overlay on a floor cell; traversable.

PIT: stores exactly 0, 1, or 2 discs.

VOID: not traversable.

A loose disc occupies a floor cell and blocks entry until picked up.

Pit passability is:

pit_count = 0  -> impassable
pit_count = 1  -> impassable
pit_count = 2  -> traversable

Disc count is conserved. No action creates or destroys a disc.

1.3 Dynamic state
player_position : (row, column)
facing          : UP | DOWN | LEFT | RIGHT
carrying        : false | true
loose_discs     : set of floor coordinates
pit_counts      : map from pit coordinate to 0 | 1 | 2
steps           : nonnegative integer
status          : PLAYING | WON | FAILED
selected        : None
cursor          : None

Discs have no identity once placed. Labels such as A, B, and P below are implementation labels only and are never rendered.

2. Exact action semantics

The game uses UP, DOWN, LEFT, RIGHT, and SPACE.

Clicks are deliberately unused. There are no visually clickable objects.

2.1 Direction action

For a direction action:

Increment steps.

Set facing to the chosen direction, even if movement will fail.

Let target be the adjacent cell in that direction.

Move into target if and only if:

target is an unoccupied floor or exit cell; or

target is a pit with pit_count == 2.

Otherwise remain in place.

If the courier moved onto the exit, set status = WON.

Consequently, pressing toward a loose disc or incomplete pit both faces the object and produces a visible bump. This is the standard way to prepare a subsequent SPACE.

Pressing toward a completed pit moves onto it. There is no independent turn-in-place action. If the player wants to retrieve a completed pit’s cap, the geometry must allow the courier to enter the current cell from the opposite side, thereby facing the pit without stepping onto it.

2.2 SPACE

SPACE acts only on the adjacent cell in facing.

If the courier is not carrying:

Faced loose disc:

Remove the loose disc.

Set carrying = true.

Faced pit with count 1 or 2:

Decrement its count by one.

Set carrying = true.

Anything else:

Invalid action; no gameplay state changes.

If the courier is carrying:

Faced pit with count 0 or 1:

Increment its count by one.

Set carrying = false.

Anything else:

Invalid action; no gameplay state changes.

Additional consequences:

A disc cannot be left on ordinary floor.

A loose disc cannot be picked up while already carrying.

A full pit cannot accept a third disc.

SPACE never targets the courier’s own cell.

Therefore, the courier cannot remove a disc from the completed pit on which it currently stands.

Removing a disc from a two-disc pit immediately changes it from traversable to impassable.

A one-disc pit can be emptied again.

2.3 Click action

A click on any logical cell is accepted by step() but is a deliberate no-op:

steps increments.

Position, facing, carry state, loose discs, pits, selection, and cursor do not change.

There is no hover state or click animation.

This avoids mixing an arbitrary click vocabulary into the movement system.

2.4 Update order

Every input is resolved by exactly one call to step():

1. If status is terminal, ignore the action.
2. Increment steps.
3. Resolve the direction, SPACE, or click action atomically.
4. Check for exit entry.
5. If exit was entered, WON takes precedence.
6. Otherwise, if steps reached the level cap, set FAILED.
7. Produce the settled observation and any transition frames.

Animation frames never invoke another state update and never consume steps.

3. Objective and validator

The sole success condition is:

The courier enters the visible green exit cell through a valid movement action.

Everything else is ignored at success:

The courier may be carrying a disc.

Pits may be empty, partial, or complete.

Old bridges may have collapsed.

Loose discs may remain, although the authored shortest solutions after Level 1 consume every loose disc.

Facing direction is irrelevant.

The following do not trigger success:

Clicking the exit.

Facing the exit.

Using SPACE on the exit.

Merely making a route to the exit without entering it.

No level has a hidden configuration predicate or “complete all pits” requirement.

4. Visual contract
4.1 Canvas mapping

Render onto a 64×64 canvas.

For logical cell (r,c):

cell_left = 4 + 5 × (c - 1)
cell_top  = 3 + 5 × (r - 1)

Each logical cell occupies a 5×5 pixel block.

The largest cell ends at pixel (58,57), leaving negative space around the board and room for an edge HUD.

4.2 Paint order

Fill canvas with color 5.

Paint active floor cells.

Paint pits and loose discs.

Paint exit marker.

Paint courier and carried disc.

Paint transient feedback.

Paint bottom step bar.

Adjacent floor cells merge visually. Do not draw borders between logical cells.

4.3 Palette usage

The renderer uses only these integer color IDs:

ID	Use
0	bright highlight
1	ordinary floor
3	pit rim
4	depleted HUD track
5	void and deep pit recess
8	invalid-action or failure pulse
9	courier body
10	courier facing marker and remaining HUD
11	disc highlight
12	disc body
14	exit
4.4 Exact 5×5 object masks

Floor is five rows of color 1.

Empty pit, count 0

1 3 3 3 1
3 5 5 5 3
3 5 5 5 3
3 5 5 5 3
1 3 3 3 1

Sunken pit, count 1

1 3 3 3 1
3 5 5 5 3
3 5 5 5 3
3 12 11 12 3
1 3 12 3 1

The large dark recess above the lower layer must remain visible.

Completed pit, count 2

1 12 12 12 1
12 11 11 11 12
12 11 0 11 12
12 11 11 11 12
1 12 12 12 1

This state is bright, visually flush, and unmistakably unlike the sunken one-disc state.

Loose disc

1 1 12 1 1
1 12 11 12 1
12 11 11 11 12
1 12 11 12 1
1 1 12 1 1

Exit

1 1 14 1 1
1 14 14 14 1
14 14 0 14 14
1 14 14 14 1
1 1 14 1 1
4.5 Courier

Without a carried disc:

Paint a centered 3×3 body using color 9.

Paint one color-10 nose pixel at:

UP: relative (0,2)

DOWN: relative (4,2)

LEFT: relative (2,0)

RIGHT: relative (2,4)

While carrying:

Paint the carried disc as [12,11,12] on relative row 0, columns 1–3.

Paint the courier body as a 2×3 color-9 block on rows 2–3, columns 1–3.

Paint the facing nose at:

UP: (1,2)

DOWN: (4,2)

LEFT: (3,0)

RIGHT: (3,4)

The courier is painted over the underlying floor or completed cap.

4.6 HUD

Pixels x=4..59, y=60..61 form a horizontal action-limit bar:

Depleted portion: color 4.

Remaining portion: color 10.

At failure: entire bar becomes color 8.

No numbers or text appear in the grid.

5. Animation and feedback

Ordinary successful movement is immediate.

Blocked movement

Three visual frames:

Settled state.

Courier sprite displaced one rendered pixel toward the obstacle; obstacle rim or boundary pulses color 10.

Return to settled state.

Facing changes on the action even though position does not.

Disc pickup, deposit, or retrieval

Four visual frames generated by one step():

Old settled state.

Disc at the source edge.

Disc midway between cells.

New settled state.

For a 1 → 2 deposit, the disc rises into the bright cap mask on frame 4.

For a 2 → 1 retrieval, the cap visibly lowers and exposes the dark recess on frame 4.

The logical post-state is authoritative at frame 4. Input is locked during the four frames. The sequence consumes one action total.

Invalid SPACE

The faced object pulses color 8 for two frames.

If the faced cell is void or ordinary floor, the courier’s nose pulses color 8.

When the invalid action is an attempted second pickup while carrying, both the loose disc and the carried disc pulse. This makes the one-disc carrying limit particularly clear in Level 5.

Completion

Entering the exit immediately sets WON. The exit alternates between colors 14 and 0 for four frames, then the final state freezes.

6. Authored progression

Documentation maps use:

#  void
.  floor
S  start
E  exit
o  loose disc
0  empty pit
1  one-disc pit
2  completed pit

These characters are documentation only and are never rendered.

In solution strings:

U = UP
D = DOWN
L = LEFT
R = RIGHT
X = SPACE

Column headers use their final decimal digit, so column 10 is shown as 0 and column 11 as 1.

Level	Main discovery	Verified shortest length	Action cap
1	Pick up and visibly carry a blocking disc	4	128
2	One disc stays low; two discs form terrain	10	256
3	A completed bridge can be peeled and reused	11	256
4	Concentrate material, then carry the final donor across	32	640
5	Preload a pit as temporary storage; later reuse it as terrain and material	41	768
6	Preserve the foothold that exposes deeper material	60	1024
7	Complete a preloaded crossing from its opposite bank, then compose all earlier roles	69	1280

A breadth-first enumerator over the full logical state found the listed action counts as minimum lengths. The supplied sequences are canonical shortest replays; shortest paths need not be unique.

7. Exact authored levels
Level 1 — Clear Hands
Initial state
start       = (6,4)
facing      = RIGHT
carrying    = false
exit        = (6,7)
steps       = 0
selected    = None
cursor      = None
action_cap  = 128

floor = {
  (6,4), (6,5), (6,6), (6,7)
}

loose_discs = {
  d0: (6,5)
}

pits = {}
    4567
 6  So.E
Player-facing discovery

The green exit is visible at the other end of a four-cell strip, but the orange disc blocks the only route. The courier’s nose already points at it. SPACE immediately lifts the disc above the courier and exposes walkable floor.

This isolates one causal fact: SPACE picks up a faced loose disc. No pit, alternate path, or second device can explain the change.

The player reaches the exit while still carrying the disc, visibly demonstrating that carried material does not need to be disposed of before success.

Canonical shortest replay
X R R R
Checkpoints
After action 1:
  carrying = true
  loose_discs = {}

After action 4:
  position = (6,7)
  carrying = true
  status = WON
Attribution check

The disc is the only obstacle.

The exit cannot be reached without removing it.

The carried sprite distinguishes “picked up” from “deleted.”

Clicking cannot solve the level.

There is no irrelevant visible component.

Level 2 — Two Layers
Initial state
start       = (6,3)
facing      = RIGHT
carrying    = false
exit        = (6,7)
steps       = 0
selected    = None
cursor      = None
action_cap  = 256

floor = {
  (6,3), (6,4), (7,4), (6,6), (6,7)
}

loose_discs = {
  d0: (6,4),
  d1: (7,4)
}

pits = {
  A: (6,5) = 0
}
    34567
 6  So0.E
 7  #o###
Player-facing discovery

The known action picks up d0. Depositing it into A creates a visible lower layer, but the route to the exit remains blocked. The second loose disc lies immediately below the courier.

After picking up d1, pressing right visibly bumps against the still-sunken one-disc pit. Depositing the second disc changes the same tile into a bright cap, and the next right action walks onto it.

The single new causal rule is the two-state completion threshold:

A first disc changes the pit but does not support movement; the second disc makes it terrain.

Canonical shortest replay
X R X D X R X R R R
Checkpoints
After action 3:
  A = 1
  carrying = false

Action 6:
  RIGHT is blocked by A = 1
  facing becomes RIGHT

After action 7:
  A = 2
  carrying = false

After action 8:
  position = (6,5), standing on A
Attribution check

The first and second deposits affect the same object.

The one-disc state is explicitly tested by a blocked movement before completion.

Nothing else changes when the second disc is deposited.

Both loose discs are necessary.

Final state: A=2, no loose discs.

Level 3 — Take the Road With You
Initial state
start       = (6,2)
facing      = RIGHT
carrying    = false
exit        = (6,9)
steps       = 0
selected    = None
cursor      = None
action_cap  = 256

floor = {
  (6,2), (6,3), (6,5), (6,6), (6,8), (6,9)
}

loose_discs = {}

pits = {
  A: (6,4) = 2,
  B: (6,7) = 1
}
    23456789
 6  S.2..1.E
Player-facing discovery

The courier can cross completed pit A, reaching stable floor at (6,6). Pit B is visibly one layer short, and there are no loose discs anywhere.

Known rules do not provide a forward action. The only visible stock of material is the completed bridge just crossed.

Stepping left to (6,5) leaves the courier facing A. SPACE lifts its cap, immediately turning A dark and impassable. The carried cap can then complete B.

The new causal rule is forced:

Completed terrain is reversible material.

Canonical shortest replay
R R R R L X R X R R R
Checkpoints
After action 4:
  position = (6,6)
  A = 2
  B = 1

After action 6:
  A = 1
  carrying = true

After action 8:
  A = 1
  B = 2
  carrying = false
Attribution check

No loose disc can explain the acquired material.

B is already in the familiar one-disc state.

The bridge visibly collapses on retrieval.

The courier stands on stable floor when removing A; removal from under the courier is not possible.

Final state: A=1, B=2.

Level 4 — One Bridge, Then Another
Initial state
start       = (7,5)
facing      = LEFT
carrying    = false
exit        = (2,5)
steps       = 0
selected    = None
cursor      = None
action_cap  = 640

floor = {
  (7,4), (7,5), (7,6),
  (8,5), (6,5),
  (7,8), (7,9),
  (4,5), (2,5)
}

loose_discs = {
  west:  (7,4),
  south: (8,5),
  east:  (7,9)
}

pits = {
  A: (5,5) = 0,
  B: (7,7) = 0,
  C: (3,5) = 1
}
    456789
 2  #E####
 3  #1####
 4  #.####
 5  #0####
 6  #.####
 7  oS.0.o
 8  #o####
Player-facing discovery

Both A and B are empty. The west and south discs are immediately reachable.

Putting one disc in each pit creates two dark half-stacks and no foothold. Concentrating both in B instead opens the eastern branch and exposes the third loose disc.

The eastern disc can seed A. The player then peels both discs from temporary bridge B:

The first completes A.

The second is carried across A and completes preloaded C.

This establishes three consequences of the same reversible-stack rule:

A half-stack is storage but not progress.

A completed bridge can be emptied after it has served its route.

The final disc must sometimes be carried across the bridge it will eventually help replace.

Canonical shortest replay
D X R X L X R X
R R X L L L U X
D R X L U X
D R X L U U U X U U

Equivalent ungrouped sequence: 32 actions.

State checkpoints
After action 8:
  A = 0
  B = 2
  C = 1

After action 16:
  A = 1
  B = 2
  C = 1

After action 22:
  A = 2
  B = 1
  C = 1

After action 25:
  A = 2
  B = 0
  C = 1
  carrying = true

After action 30:
  A = 2
  B = 0
  C = 2
Why the spatial layout matters

The floor at (4,5) is a narrow landing between A and C.

Before C is complete, there is no cell from which the courier can face downward toward A while remaining on stable floor. Therefore, the courier cannot cross A, reclaim its cap, and use that cap for C. The courier must arrive above A already carrying the final disc taken from B.

Necessity and anti-cleanliness check

The three loose discs plus the initial disc in C total four discs.

The successful final support state needs exactly A=2 and C=2.

B must therefore end empty.

Splitting the first two loose discs between A and B visibly produces no route.

The east branch is not a decorative detour; its disc is required.

The solution revisits the center with different bridge states rather than following a one-way checklist.

Level 5 — Set It Down Now
Initial state
start       = (8,7)
facing      = UP
carrying    = false
exit        = (4,10)
steps       = 0
selected    = None
cursor      = None
action_cap  = 768

floor = {
  (8,7),
  (7,7), (7,6), (7,5),
  (5,5), (5,7),
  (4,7),
  (3,7), (3,6),
  (6,8), (6,10),
  (4,10)
}

loose_discs = {
  starter: (7,6),
  landing: (5,5),
  north:   (3,6)
}

pits = {
  A: (6,5)  = 2,
  B: (5,6)  = 1,
  P: (6,7)  = 0,
  D: (6,9)  = 0,
  E: (5,10) = 1
}
    567890
 3  #o.###
 4  ##.##E
 5  o1.##1
 6  2#0.0.
 7  .o.###
 8  ##S###
Player-facing discovery

The direct northern pit P is empty. A loose starter disc lies left of the first junction. Farther along that left route, completed bridge A leads to another loose disc at (5,5).

The courier can carry the starter across A, but then cannot pick up the landing disc. The invalid pickup produces an immediate pulse on both discs. The only reachable non-full receptacle before that choke is P.

The productive experiment is therefore to place the starter in P, even though P=1 is still unusable. The deposited disc persists while the courier walks around through A, picks up the landing disc, returns, and completes P.

P subsequently changes roles twice:

Temporary storage that frees the courier’s hand.

A route into the northern branch.

A donor whose cap completes final bridge E.

The preloaded disc is not an abstract requirement: it solves a visible carrying impasse.

Canonical shortest replay
U L X U X
L L U X D R R U X
U U U U L X D D D R X
L U L X D R X
R L X R R U X U U

Equivalent ungrouped sequence: 41 actions.

State checkpoints
After action 5:
  P = 1
  carrying = false

After action 9:
  carrying = true
  landing loose disc has been picked
  P = 1

After action 14:
  P = 2
  carrying = false

After action 25:
  D = 1

After action 29:
  B = 0
  carrying = true

After action 32:
  D = 2

After action 35:
  P = 1
  carrying = true

After action 39:
  E = 2
Necessity

A constrained state-space check finds no winning state that reaches the northern bank (5,7) without first changing P from 0 to 1.

The reason is physical:

Carrying the starter prevents picking the landing disc.

A is already full.

B cannot be approached until the landing cell is cleared.

P is the only available place to set the starter down.

Every loose disc is used. The final pit state is:

A = 2
B = 0
P = 1
D = 2
E = 2

A is deliberately preserved as terrain, whereas the initially partial B is later harvested as material. The two pits use the same rule but have contrasting strategic roles.

Level 6 — Keep the Middle
Initial state
start       = (9,2)
facing      = RIGHT
carrying    = false
exit        = (2,11)
steps       = 0
selected    = None
cursor      = None
action_cap  = 1024

floor = {
  (9,2), (9,3), (9,6), (9,7),
  (7,6), (7,7), (7,9), (7,10),
  (6,8), (6,9),
  (4,8), (4,9), (4,11),
  (3,9),
  (2,11)
}

loose_discs = {
  q1: (6,8),
  q2: (7,10),
  q3: (3,9),
  q4: (4,8)
}

pits = {
  A: (9,4)  = 2,
  B: (9,5)  = 2,
  C: (8,6)  = 0,
  D: (7,8)  = 0,
  E: (5,9)  = 0,
  F: (4,10) = 1,
  G: (3,11) = 0
}
    2345678901
 2  #########E
 3  #######o#0
 4  ######o.1.
 5  #######0##
 6  ######o.##
 7  ####..0.o#
 8  ####0#####
 9  S.22..####
Player-facing discovery

The courier first crosses completed pits A and B to reach (9,6). Empty pit C blocks the northern route.

From (9,6), B can be retrieved directly. That is the tempting cap. But removing it changes B to a dark one-disc pit and removes the only tile on which the courier can stand to target deeper pit A.

The useful choice is to step back onto B, face A, and peel A first. B remains the working foothold while both discs from A are transferred into C.

After C is complete, B changes role completely: it is no longer a foothold to preserve and can be emptied into D.

The second half repeats the same contextual reversal:

Loose discs q1 and q2 complete E.

q3 must complete preloaded F, because the courier is carrying it and cannot simultaneously retrieve E.

Once F is solid, E is obsolete; its cap becomes the first disc in G.

q4 supplies the final cap.

Canonical shortest replay
R R R R L X R U X L X R U X
R L X U U R X L D D L X U U R X
R R X U X L X U X
U U X R X U D X R R U X
L L X R R U X U U

Equivalent ungrouped sequence: 60 actions.

State checkpoints
After action 14:
  A = 0
  B = 2
  C = 2
  D = 0

After action 21:
  B = 1
  D = 1

After action 30:
  B = 0
  D = 2

After action 39:
  E = 2

After action 44:
  F = 2

After action 47:
  E = 1
  carrying = true

After action 51:
  G = 1

After action 58:
  G = 2
Productive versus unproductive cap choice

Moving a cap from B into C before harvesting A does not create lasting progress. B=1 strands both discs in A. A winning continuation must first restore B before A can be reached again.

The visible consequence is immediate rather than validator-based:

The bright middle foothold becomes a dark recess.

The deeper cap remains visibly full but physically unreachable.

Restoring the misplaced cap visibly restores access.

Later, removing both discs from B is correct because C has replaced its route function.

Necessity and anti-cleanliness check

All four loose discs are consumed.

Final pit state:

A = 0
B = 0
C = 2
D = 2
E = 1
F = 2
G = 2

A and B are not interchangeable despite identical rules; their positions determine which one is currently support and which is material.

q1 and q2 are reached from orthogonal branches and both feed E.

Carrying q3 prevents an early retrieval of E, forcing F to be completed first.

Completing F creates the safe route needed after E collapses.

The level contains several revisits under altered terrain rather than a sequence of devices encountered once.

Level 7 — The Far-Side Cap
Initial state
start       = (8,7)
facing      = UP
carrying    = false
exit        = (1,11)
steps       = 0
selected    = None
cursor      = None
action_cap  = 1280

floor = {
  (8,7),
  (7,5), (7,6), (7,7),
  (5,5), (4,5),
  (5,7), (4,7),
  (3,6), (3,7), (3,8),
  (6,8), (6,10),
  (4,10), (4,11),
  (5,11),
  (2,9), (2,10),
  (1,11)
}

loose_discs = {
  opposite: (4,7),
  upper_l:  (3,6),
  upper_r:  (3,8),
  far_low:  (5,11),
  far_high: (4,11),
  final:    (2,9)
}

pits = {
  A: (6,5)  = 2,
  B: (5,6)  = 0,
  P: (6,7)  = 1,
  D: (6,9)  = 0,
  E: (5,10) = 1,
  G: (3,10) = 0,
  H: (2,11) = 0
}
    5678901
 1  ######E
 2  ####o.0
 3  #o.o#0#
 4  .#o##.o
 5  .0.##1o
 6  2#1.0.#
 7  ...####
 8  ##S####
Player-facing discovery

The direct pit P is visibly preloaded with one disc but still blocks the northern route. There is no loose disc on the starting bank with which to complete it.

The only available route turns left and crosses completed pit A. At the landing, empty pit B blocks the next step. The only available material is the bridge just crossed.

The courier must transfer both discs from A into B. This completes the alternate crossing but permanently collapses the original route. On the new bank, the loose disc at (4,7) completes P from its opposite side. The preloaded half-stack becomes the replacement route back through the center.

The rest of the level composes the complete vocabulary:

The two uneven upper branches supply D.

The cap of P completes preloaded E.

Two far-side loose discs complete G.

Only after G is complete can the courier step through it, return facing downward, and retrieve the cap of E.

That cap seeds H.

The final side disc completes H and exposes the exit.

Canonical shortest replay
U L L U U U D X R X D X R X R R
U X D X
U U L X D D D R X
L U U U R X D D D R X
R L X R R U X
U R X U X R X U X
U D X U U R X L X R X R U

Equivalent ungrouped sequence: 69 actions.

State checkpoints
After action 16:
  A = 0
  B = 2
  P = 1

After action 20:
  P = 2

After action 29:
  D = 1

After action 40:
  D = 2

After action 43:
  P = 1
  carrying = true

After action 47:
  E = 2

After action 56:
  G = 2

After action 59:
  E = 1
  carrying = true

After action 63:
  H = 1

After action 67:
  H = 2
Why opposite-side completion is forced

Before reaching (5,7):

There are no reachable loose discs.

P starts at one disc.

Removing a disc from A while standing on its far landing immediately makes A impassable.

Both discs from A are required to construct B.

Thus P cannot be completed on its starting side. Its second disc must come from the loose disc at (4,7) after the alternate route has reached the opposite bank.

A state-space check finds no winning state in which P never reaches count 2.

Why the final donor order is forced

G lies above (4,10), while E lies below it.

To retrieve E from (4,10), the courier must enter (4,10) from the north so that it faces downward. The only northern cell is completed pit G.

Therefore:

G must be completed first.

The courier crosses onto G.

It moves back down to (4,10), now facing E.

Only then can E be safely peeled and reused in H.

The final loose disc at (2,9) supplies the second layer of H. It cannot replace either far-side disc used in G, because it is inaccessible until after G has opened the upper region.

Necessity and anti-cleanliness check

All six loose discs are consumed. The final pit state is:

A = 0
B = 2
P = 1
D = 2
E = 1
G = 2
H = 2

The initial ten discs are conserved exactly.

The dependency structure is not a linear object checklist:

A is first terrain, then material.

B is first an obstacle, then the replacement route.

P is first an unusable partial stack, then a shortcut, then a donor.

E is first a target, then terrain, then material.

G is both a bridge and the turning foothold required to expose E.

The two branches feeding D and the two branches feeding G must be revisited from different approach directions.

Collapsed routes visibly explain why premature retrievals fail.

8. Curriculum and epistemic audit
Levels 1–3: attribution curriculum
Level 1

A single visible obstacle teaches pickup and carry. One action produces one unmistakable result.

Level 2

The same pit shows both incomplete and complete states. The player physically bumps into the one-disc state before completing it.

Level 3

There are no loose discs, so the only plausible source for the missing cap is the completed bridge behind the courier. Its collapse and the appearance of the carried disc happen together.

No foundational level contains:

a decoy;

an optional branch;

a tight budget;

a hidden dependency;

a role reversal;

an unrelated object.

Levels 4–7: systemic development

The later levels add no new object category. Difficulty comes entirely from stronger consequences of the same rules:

allocating discs between simultaneous gaps;

carrying a future disc over a bridge before collapsing it;

using an incomplete pit as storage;

preserving a pit because its position exposes another pit;

completing a partial stack from a different bank;

using a newly completed route to approach an older bridge from the retrieval side;

replacing one route before dismantling another.

The board cannot be replaced by counters or a checklist. Approach direction, safe landing cells, branch position, and the absence of turning space all affect which caps can be recovered without destroying access.

9. Replay validation summary

Using pit labels in their documented order:

Level	Shortest actions	Canonical final carry	Canonical final pit counts
1	4	1	none
2	10	0	A=2
3	11	0	A=1, B=2
4	32	0	A=2, B=0, C=2
5	41	0	A=2, B=0, P=1, D=2, E=2
6	60	0	A=0, B=0, C=2, D=2, E=1, F=2, G=2
7	69	0	A=0, B=2, P=1, D=2, E=1, G=2, H=2

These final configurations are consequences of the canonical solutions, not additional win requirements.
