Sand Ferry

This is a game about carrying individual blocks of sand, becoming difficult because the same material that blocks one corridor must fill the pits needed elsewhere.

Campaign length: exactly 7 authored levels.

Sand Ferry uses only one material and one tool. Every sand unit obeys the same rule everywhere: on floor it is an obstacle, in the bucket it is cargo, and in a pit it becomes walkable ground. Depth comes from deciding where each unit must be at each phase, not from accumulating devices or validator conditions.

The player can move, face a neighboring cell, and use SPACE to shovel. Clicking is deliberately unused.

1. Native space and rendering
Logical board

Every level uses a 10×9 logical lattice.

Coordinates are one-indexed.

(1,1) is the upper-left cell.

x increases rightward.

y increases downward.

Every coordinate not listed as ordinary floor or pit is a solid wall.

Sand-pile coordinates are also ordinary-floor coordinates.

The exit is an overlay on an ordinary-floor coordinate.

One directional action moves the player by at most one logical cell.

64×64 rendering

The logical board is rendered using 5×5-pixel cells:

Board origin: pixel (7,3).

Logical cell (x,y) occupies pixels
px = 7 + 5(x−1) ... 11 + 5(x−1) and
py = 3 + 5(y−1) ... 7 + 5(y−1).

The board therefore occupies a 50×45-pixel region.

The bottom strip, approximately y=52...62, shows only the bucket state.

There are no internal grid lines or cell borders. Adjacent floor cells merge into continuous regions.

Palette

Only integer color IDs 0–15 are used.

ID	Use
0	White highlights
1	Ordinary floor
2	Blocked-movement bump feedback
4	Wall mass
5	Canvas background and pit depth
6	Player body
7	Bucket and facing attachment
8	Invalid shovel feedback
9	Empty-pit interior
10	Pit rim and pit-cell counting notch
11	Sand highlight
12	Main sand color
13	Sand shadow
14	Exit
3, 15	Unused
Entity silhouettes

Player. A connected magenta body occupying roughly 3×3 pixels, with a pink bucket attachment on the facing side. The attachment makes facing visible even when movement is blocked.

Empty bucket. Pink outline with a white center, both beside the player and in the bottom HUD.

Full bucket. The same outline with an orange center.

Loose sand pile. A connected mound filling most of its logical cell:

narrow yellow/orange top;

wide orange middle;

maroon lower shadow.

It must not resemble a flat filled pit.

Empty pit. Blue cell with a dark center and a small light-blue notch. Adjacent pit cells visually merge into a trench, but each 5×5 logical pit retains one notch so its length can be counted.

Filled pit. A flat orange slab with a narrow blue rim still visible. It is visibly different from both ordinary floor and a loose pile.

Exit. A green arch-like 5×5 shape over ordinary floor. It remains visible from the first frame.

2. Formal state

A nonterminal game state contains:

player_position : coordinate
facing          : UP | DOWN | LEFT | RIGHT
bucket_full     : boolean
floor_sand      : set of ordinary-floor coordinates
filled_pits     : set of pit coordinates
terminal        : boolean
feedback_event  : none | blocked_move | invalid_space | scoop | deposit | win
selected        : null
cursor          : not rendered

Static level data contains:

ordinary_floor
pit_cells
exit_coordinate
initial_player_position
initial_facing
initial_floor_sand

Every level starts with:

bucket_full = false
filled_pits = {}
terminal = false
selected = null
cursor = hidden

Sand conservation is exact:

|floor_sand| + |filled_pits| + int(bucket_full)
    = number of initial sand units

No sand is created, destroyed, stacked, or merged.

3. Action semantics

The available action space is exactly:

UP, DOWN, LEFT, RIGHT, SPACE, click any cell

The game uses the four directions and SPACE. All clicks are no-ops.

Directional actions

For a direction d:

Set facing = d.

Let target be the cardinally adjacent cell in direction d.

Move into target only if it is traversable.

A cell is traversable exactly when it is:

ordinary floor without a loose pile; or

a filled pit.

The following are not traversable:

walls;

empty pits;

ordinary floor containing loose sand.

When blocked, the player remains in place but still turns to face the requested direction. This is important: pressing toward a sand pile or empty pit is a legitimate way to face it before pressing SPACE.

A blocked movement produces a one-frame color-2 bump at the contacted edge. There is no penalty.

SPACE

SPACE acts on the cell directly ahead.

Bucket empty

If the faced cell contains:

a loose floor pile: remove it from floor_sand and set bucket_full=true;

a filled pit: remove it from filled_pits and set bucket_full=true.

Removing a filled-pit slab immediately restores the underlying empty, impassable pit.

Any other target is invalid.

Bucket full

If the faced cell is:

an empty ordinary-floor cell: add a loose pile there and set bucket_full=false;

an empty pit: add that coordinate to filled_pits and set bucket_full=false.

Depositing onto the exit is legal because the exit is ordinary floor. The resulting pile blocks the exit until recovered.

Depositing onto a wall, occupied floor, loose pile, or already-filled pit is invalid.

Important spatial consequence

SPACE never affects the player’s own cell. Therefore:

the player cannot scoop the filled pit currently underfoot;

to reclaim a bridge slab, the player must stand on a cardinally adjacent cell and face it;

approach direction and turning space matter.

Click actions

Clicking any logical cell:

does not select anything;

does not change facing;

does not move the player;

does not alter sand;

does not produce an affordance or feedback flash.

There are no clickable objects and no cursor-based puzzle state.

4. Update order

Every transition occurs only inside step(action):

Clear the previous transient feedback event.

Read the action.

Apply facing and movement, or apply the shovel interaction.

Commit all resulting logical state atomically.

Check the win condition.

Create the corresponding visual-feedback event.

Return the observation.

There are no autonomous updates, gravity, timers, enemy phases, or between-step simulation.

Animation

Ordinary movement is rendered immediately.

A valid scoop or deposit has a deterministic three-frame presentation:

The source sand contracts toward the bucket.

An orange two-pixel connector shows the transfer direction.

The final bucket, pile, or slab appears.

The logical state has already been committed by step(). Inputs are ignored until the three presentation frames finish, so animation cannot create an intermediate interactive state. Animation frames do not consume actions.

For filled-pit removal, the blue pit is revealed on the final presentation frame. For pit filling, the blue surface becomes an orange slab on the final presentation frame.

Invalid SPACE produces two presentation frames:

the faced cell flashes color 8;

the bucket rim flashes color 8.

No logical state changes.

Winning produces a four-frame green pulse and then a terminal observation. These frames consume no actions.

5. Success validator

A level succeeds exactly when, after resolving an action:

player_position == exit_coordinate

Relevant:

the player must physically enter the green exit cell.

Ignored:

bucket state;

remaining loose piles;

filled or unfilled pits;

whether temporary bridges were reclaimed;

the number of actions;

where unused sand remains.

Allowed:

arriving with a full bucket;

abandoning inaccessible sand;

leaving non-goal pits filled;

leaving earlier routes destroyed.

Forbidden:

merely standing adjacent to the exit;

having a sand pile on the exit, because then the player cannot enter it.

There is no energy, timer, move limit, or failure budget. Exploration is unrestricted, including in the early levels.

6. Campaign progression
Level	Main discovery	Verified shortest replay
1	A pile can become a bridge	8 actions
2	Sand can be cached on floor	20 actions
3	A filled pit can be reclaimed and reused	27 actions
4	Cache location matters; contiguous pits need simultaneous material	33 actions
5	A non-goal pit can provide rear access to a sand wall	58 actions
6	A trailing bridge slab can be ferried forward	72 actions
7	An elbow-shaped trench changes how many slabs are required	98 actions

Replay notation below uses:

U = UP
D = DOWN
L = LEFT
R = RIGHT
S = SPACE

Blocked direction presses used to turn are included in the action counts.

Level 1 — First Crossing
Player-facing discovery

The green exit is visible across one blue pit. An orange pile blocks the only initial route. Walking cannot make progress, while the player’s bucket visibly points directly at the pile.

Pressing SPACE immediately moves the pile into the bucket. Depositing it into the pit visibly changes the pit from a blue hole into flat orange ground.

No symbolic goal example is needed: the exit, obstacle, pit, and physical transformation are all visible.

Exact initial state
Ordinary floor:
{(1,2), (2,2), (3,2), (5,2), (1,3), (1,4)}

Pits:
{(4,2)}

Loose sand:
{(1,3)}

Player:
position = (1,4)
facing = UP
bucket_full = false

Exit:
(5,2)

UI:
selected = null
cursor = hidden
filled_pits = {}
Verified shortest replay: 8 actions
S U U R R S R R

Resolution:

S: scoop (1,3).

Move to (3,2).

S: fill (4,2).

Cross (4,2) and enter (5,2).

Necessity and anti-degeneracy

The initial pile blocks the only path out of the start.

The pit blocks the only approach to the exit.

The single sand unit must serve both as removed obstacle and bridge material.

Random movement cannot solve the level because two correctly oriented SPACE actions are indispensable.

Level 2 — Staging Bay
Player-facing discovery

The first pile can be scooped, but a second pile later blocks the corridor while the bucket is already full. The player cannot scoop the second pile and cannot yet reach either pit.

A small lower loop lets the player approach the floor cell (4,2) from below and deposit the first unit there. The newly created floor pile is visibly solid and remains recoverable.

The first pit and second pit are separated so the filling of the first cannot simply be reclaimed from the far side: the player lacks a cell from which to approach the far bank facing backward. The cached unit is therefore genuinely necessary.

Exact initial state
Ordinary floor:
{
  (4,2),
  (1,3), (2,3), (3,3), (4,3), (5,3), (7,3), (9,3),
  (3,4), (4,4)
}

Pits:
{(6,3), (8,3)}

Loose sand:
{(2,3), (5,3)}

Player:
position = (1,3)
facing = RIGHT
bucket_full = false

Exit:
(9,3)

UI:
selected = null
cursor = hidden
filled_pits = {}
Verified shortest replay: 20 actions
S R R D R U S R S R S L U S R R R S R R

Key checkpoints:

Actions 1–7: scoop (2,3) and cache it at (4,2).

Actions 8–11: face and scoop (5,3), then fill (6,3).

Actions 12–14: return to (4,3) and recover the cached unit.

Actions 15–18: carry it across (6,3) and fill (8,3).

Actions 19–20: cross to the exit.

Necessity and anti-degeneracy

The second pile cannot be scooped while the first unit remains in the bucket.

The lower loop is what makes the floor cache physically possible.

Depositing in the corridor instead would block passage.

The fill at (6,3) cannot be recovered from (7,3) because there is no reachable cell to the right of (7,3) before (8,3) is filled; the player cannot stand at (7,3) facing left without stepping back onto the slab.

Both sand units and both pits are necessary.

Level 3 — Handoff
Player-facing discovery

The first sand pile is collected, but the second pile at (5,5) blocks the route toward a pit. The bucket must be emptied.

The useful cache is unusual: the player deposits the first unit at (3,5), directly onto the route. That pile now blocks the straight path, but the offset cells (4,6) and (4,5) let the player walk around it and reach the second pile from the side.

Later, after one sand unit has been used at (7,3), the player reaches another visible shortage. The earlier filled pit (6,5) remains visibly orange. The loop through (8,5) lets the player approach it facing left and scoop it back out. The bridge disappears behind the player, but its material becomes the final bridge.

This is the first forced demonstration that pit filling is a reversible placement, not a permanent completion marker.

Exact initial state
Ordinary floor:
{
  (7,2), (9,2),
  (7,4),
  (3,5), (4,5), (5,5), (7,5), (8,5),
  (3,6), (4,6),
  (3,7)
}

Pits:
{(8,2), (7,3), (6,5)}

Loose sand:
{(5,5), (3,6)}

Player:
position = (3,7)
facing = UP
bucket_full = false

Exit:
(9,2)

UI:
selected = null
cursor = hidden
filled_pits = {}
Dependency network

The level is not a simple corridor chain:

sand A must be cached to free the bucket;

the cache blocks the direct route, making the side detour necessary;

sand B opens the horizontal bridge;

that bridge transports sand A;

sand A opens the upper route;

the horizontal bridge is then reclaimed as sand B and becomes the final fill.

The two units repeatedly exchange infrastructure and cargo roles.

Verified shortest replay: 27 actions
S U S R U R S R S L S R R R U S D R L S U U U R S R R

Key checkpoints:

Action 3: sand A is deposited at (3,5).

Action 7: sand B is scooped from (5,5).

Action 9: sand B fills (6,5).

Action 11: sand A is recovered from (3,5).

Action 16: sand A fills (7,3).

Action 20: sand B is reclaimed from (6,5), destroying the old bridge.

Action 25: sand B fills (8,2).

Necessity and anti-degeneracy

Carrying sand A directly toward the exit is impossible because sand B blocks (5,5).

The cache at (3,5) is not decorative: it is the only floor deposit in the shortest solution and changes the route topology.

The bridge at (6,5) must first be preserved and later destroyed.

There are three pits and only two sand units, so at least one filled pit must be reclaimed.

The loop cell (8,5) exists specifically to make the backward-facing recovery possible.

Level 4 — Two-Cell Crossing
Player-facing discovery

The starting unit again has to be cached before the pile at (5,5) can be collected. This time two plausible cache locations are visible.

The tempting placement at (4,5) immediately blocks the only cell from which (5,5) can be scooped. The player can recover the mistake, but ordinary progress stops right there. The safe cache is (4,6), reached from (3,6).

The upper route ends in two contiguous pit cells, (8,2) and (9,2). A single unit cannot cross both:

after filling (8,2), the player can stand on it;

the bucket is empty;

the slab underfoot cannot be scooped;

(9,2) still needs another unit.

This is a forced, visible distinction between separated pits and a continuous trench.

Exact initial state
Ordinary floor:
{
  (7,2), (10,2),
  (7,3),
  (7,4),
  (2,5), (3,5), (4,5), (5,5), (7,5), (8,5),
  (2,6), (3,6), (4,6),
  (2,7)
}

Pits:
{(8,2), (9,2), (6,5)}

Loose sand:
{(5,5), (2,6)}

Player:
position = (2,7)
facing = UP
bucket_full = false

Exit:
(10,2)

UI:
selected = null
cursor = hidden
filled_pits = {}
Verified shortest replay: 33 actions
S U R S U R S R S L D S R R R U U U R S D D D R L S U U U R S R R

Key checkpoints:

Action 4: the first unit is safely cached at (4,6).

Action 7: the second unit is scooped from (5,5).

Action 9: it fills temporary bridge (6,5).

Action 12: the cached unit is recovered.

Action 20: it fills the first trench cell (8,2).

Action 26: the bridge at (6,5) is reclaimed.

Action 31: that unit fills (9,2) while the player stands on (8,2).

Wrong-plan evidence

A natural failed experiment is:

scoop (2,6);

travel through (2,5) to (3,5);

deposit at (4,5).

The pile then occupies the only standing cell adjacent to (5,5). The obstruction is local and visible; no hidden validator rejects the plan.

Necessity and anti-degeneracy

The safe cache and dangerous cache use identical deposit rules; only their positions differ.

Both sand units are required to cross the two-cell trench.

The temporary bridge at (6,5) must remain filled while the first unit is carried, then be reclaimed for the second trench cell.

The loop through (8,5) is required to face left toward the temporary bridge.

Level 5 — Rear Wall
Player-facing discovery

The exit is visible beyond a three-cell trench, but the near bank is inaccessible. The only immediate experiment is to use the starter sand at the side pit (4,7). Filling it opens a route that does not point toward the exit; it instead reaches the rear side of a two-pile sand wall.

From the rear:

The nearer wall pile at (6,5) can be scooped.

The farther pile at (5,5) remains blocked while the bucket is full.

The nearer pile must therefore be cached at (7,7).

That cache blocks the return route toward the side pit.

Removing (5,5) opens the staging corridor to the final trench.

Recovering the cache reopens the return route.

The side-pit fill is finally reclaimed and used as the third trench slab.

Thus the side pit is useful infrastructure, but not part of the final path.

Exact initial state
Ordinary floor:
{
  (5,3), (9,3), (10,3),
  (5,4),
  (5,5), (6,5), (7,5),
  (2,6), (7,6),
  (2,7), (3,7), (5,7), (6,7), (7,7)
}

Pits:
{(6,3), (7,3), (8,3), (4,7)}

Loose sand:
{(5,5), (6,5), (2,6)}

Player:
position = (2,7)
facing = UP
bucket_full = false

Exit:
(10,3)

UI:
selected = null
cursor = hidden
filled_pits = {}
Dependency network

Three material branches converge on the final trench:

starter sand enables rear access through (4,7);

the nearer wall unit must be cached before the farther unit can be removed;

the farther unit opens the route and becomes the first final slab;

the cached unit becomes the second slab;

reclaiming the side bridge supplies the third slab.

The cached unit and side-bridge unit become independently available only after the wall is removed.

Verified shortest replay: 58 actions
S R S R R R R U U L S D S U L S L U U R S D D R R D S U L L U U R S L D D R R D D L L S R R U U L L U U R R S R R R

Key checkpoints:

Action 3: starter sand fills side pit (4,7).

Action 11: nearer wall unit (6,5) is collected.

Action 13: it is cached at (7,7).

Action 16: farther wall unit (5,5) is collected, opening the corridor.

Action 21: it fills final trench cell (6,3).

Action 27: the cached unit is recovered from (7,7).

Action 34: it fills (7,3).

Action 44: the side bridge at (4,7) is reclaimed.

Action 55: that unit fills (8,3).

Why the three-cell trench needs three units here

After only (6,3) and (7,3) are filled, the player stands on or beside the first slab. The empty third pit prevents reaching a cell from which the player could approach (7,3) facing left and recover (6,3). There is no side turning floor around the trench.

The third sand unit is therefore required by the authored geometry, not by an abstract count condition.

Necessity and anti-degeneracy

The wall piles cannot be reached from the starting region.

The side pit is the only access to their rear side.

The nearer wall pile must be unloaded somewhere before the farther pile can be scooped.

Caching at (7,7) creates a real future obstruction and must be reversed.

Leaving (4,7) filled leaves only two available units for a trench that requires three in this geometry.

Every sand unit and every pit participates in the shortest solution.

Level 6 — Moving the Bridge
Player-facing discovery

Three sand units are needed to form the lower three-cell trench. The two upper-left piles can be collected in either local order, but both must eventually be transported to the trench.

After crossing, the player discovers a stronger use of a filled trench:

with all three cells filled, the player can walk from the third slab back onto the second;

this approach leaves the player facing the first slab;

the first slab can now be scooped without collapsing the player’s current foothold;

the recovered unit can be carried to the upper trench.

This was impossible in Level 5 after only two slabs were filled because there was no third foothold from which to approach the second slab facing backward.

The same operation is later used on the upper trench: three filled cells create enough standing structure to recycle the trailing slab into a fourth cell.

Exact initial state
Ordinary floor:
{
  (5,3), (10,3),
  (5,4),
  (2,5), (3,5), (5,5), (6,5), (7,5), (8,5),
  (2,6), (3,6), (7,6), (8,6), (9,6),
  (2,7), (3,7), (7,7),
  (2,8)
}

Pits:
{
  (6,3), (7,3), (8,3), (9,3),
  (4,7), (5,7), (6,7)
}

Loose sand:
{(2,5), (3,5), (9,6), (2,7)}

Player:
position = (2,8)
facing = UP
bucket_full = false

Exit:
(10,3)

UI:
selected = null
cursor = hidden
filled_pits = {}
Dependency network

The level has two interacting material networks:

Lower network

the near pile and the two offset upper-left piles converge to form the three-cell temporary trench;

after crossing, its first and second slabs are successively removed and transported upward;

the third lower slab remains as abandoned infrastructure.

Upper network

two recovered lower slabs form the first two cells;

the far-room pile at (9,6) forms the third;

the third slab provides the turning foothold needed to reclaim the first slab;

that reclaimed slab is ferried forward into the fourth cell.

Verified shortest replay: 72 actions
S U R S U S D R S L L U S D R R R S R L S R R U U L L U U R S D D R R D D L S R U U L L U U R S L D D R R D R S U L L L U U R R S R L S R S R R

Major checkpoints:

Action 4: fill lower (4,7).

Action 9: fill lower (5,7).

Action 18: fill lower (6,7).

Action 21: while standing on (5,7), recover trailing slab (4,7).

Action 31: use it at upper (6,3).

Action 39: recover lower slab (5,7) while (6,7) remains a foothold.

Action 48: use it at upper (7,3).

Action 56: collect far pile (9,6).

Action 65: fill upper (8,3).

Actions 66–68: use (8,3) to approach (7,3) facing left and recover trailing slab (6,3).

Action 70: move that slab forward into (9,3).

Necessity and anti-degeneracy

All three starting-region units are required to cross the lower three-cell trench.

The lower trench cannot be replaced by a corridor: its filled cells are later mined as material.

The cell (8,7) is deliberately absent. Consequently, the surviving lower slab at (6,7) cannot be approached from the far bank while facing left; it cannot replace the far pile.

The pile at (9,6) is therefore necessary to create the third upper foothold.

The fourth upper pit cannot be filled merely by carrying a fourth unit from the start. It is solved by understanding how to recycle the trailing slab.

Level 7 — Elbow Exchange
Player-facing discovery

The final level combines two side acquisitions, floor caches, reversible bridges, and trench ferrying around a new spatial consequence: a four-cell elbow trench.

Documentation-only topology sketch:

                         far bank (7,4)
                               |
              p2 (4,4) — p3 (5,4) — p4 (6,4)
                 |
              p1 (4,5)
                 |
                         near bank (4,6)

The first two pit cells run upward; the next two turn right.

After only p1, p2, and p3 are filled, the player cannot recycle p1:

recovering p1 requires standing on p2=(4,4) while facing down;

the cell (4,3) is a wall, so the player cannot approach p2 from above;

approaching p2 from p3 makes the player face left, not down;

stepping back from p2 onto p1 does not help.

A fourth sand unit is therefore required to complete p4. This is not a new rule—only a stronger consequence of approach direction and authored geometry.

Exact initial state
Ordinary floor:
{
  (1,2), (7,2), (8,2),
  (7,3), (8,3),
  (7,4), (8,4),
  (4,6), (5,6),
  (2,7), (4,7), (5,7), (6,7), (7,7), (9,7),
  (2,8), (4,8), (5,8), (6,8), (7,8), (9,8),
  (5,9)
}

Pits:
{
  final trench:
    (2,2), (3,2), (4,2), (5,2), (6,2),

  elbow trench:
    (4,5), (4,4), (5,4), (6,4),

  side pits:
    (3,8), (8,8)
}

Loose sand:
{
  (5,8),   initial unit
  (6,7),   hub unit
  (2,7),   left-branch unit
  (9,7),   right-branch unit
  (8,4)    far-basin unit
}

Player:
position = (5,9)
facing = UP
bucket_full = false

Exit:
(1,2)

UI:
selected = null
cursor = hidden
filled_pits = {}
Dependency network

The level deliberately interleaves its branches instead of placing them in use order:

The initial unit opens the left side pit.

The left-branch unit is recovered and immediately becomes elbow slab p1.

The initial unit is reclaimed from the left pit and becomes p2.

The hub unit opens the right branch.

The right-branch unit must be cached at (5,8).

The hub unit is reclaimed and cached again at (4,7).

The right-branch unit becomes p3.

The cached hub unit becomes p4.

Elbow slabs are selectively reclaimed to seed the final trench.

The far-basin pile supplies the third final foothold.

The final footholds are ferried leftward toward the exit.

The branches feed the elbow at different times, while their fills and caches repeatedly block or enable the same central cells.

Verified shortest replay: 98 actions
S U L S L L U S R R U U S D D R L S U U U S D D R S D R R S R R U S L L L S R S U L L S D S U L U U R S D D S U U R S R L S R R U U L S D D L S R U U L S R R D S U L L L S L R S L S L R S L S L L
Phase-by-phase resolution
Phase A — Left branch and first two elbow slabs

Action 4: initial unit fills side pit (3,8).

Action 8: collect left-branch sand (2,7).

Action 13: use it as elbow slab (4,5).

Action 18: reclaim the initial unit from (3,8).

Action 22: use it as elbow slab (4,4).

The left branch is not an errand: its entry bridge is mined again and becomes structural material elsewhere.

Phase B — Right branch and two contextual caches

Action 26: collect the hub pile (6,7).

Action 30: fill right side pit (8,8).

Action 34: collect right-branch sand (9,7).

Action 38: cache it at (5,8).

Action 40: reclaim the hub unit from (8,8).

Action 44: cache that unit at (4,7).

Action 46: recover the first cache from (5,8).

The two caches use the same floor-deposit rule but have different route effects:

(5,8) preserves access to the right pit;

(4,7) temporarily blocks the lower-left route but remains recoverable from (4,6).

Phase C — Completing the elbow

Action 52: right-branch sand fills (5,4).

Action 55: recover the cached hub unit from (4,7).

Action 59: fill elbow cell (6,4).

Action 60: cross onto (6,4).

All four elbow cells were necessary. Three units cannot be ferried around the turn because the required downward-facing approach to (4,4) does not exist.

Phase D — Elbow material becomes final material

Actions 61–62: approach (4,4) from (5,4) and scoop it.

Action 68: deposit that unit into final cell (6,2).

Action 72: reclaim elbow cell (6,4) from far-bank cell (7,4).

Action 77: deposit it into final cell (5,2).

The elbow is partially dismantled after it has served its crossing purpose.

Phase E — Far sand creates the ferry foothold

Action 81: collect far-basin sand (8,4).

Action 86: fill final cell (4,2).

The three adjacent final fills (4,2), (5,2), and (6,2) create the turning structure needed for forward recycling.

Phase F — Ferry toward the exit

Action 89: while standing at (5,2) facing right, reclaim trailing fill (6,2).

Action 91: move it forward to (3,2).

Action 94: reclaim (5,2).

Action 96: move it forward to (2,2).

Actions 97–98: cross (2,2) and enter the exit at (1,2).

Necessity and anti-degeneracy

Skipping the left branch leaves too little material for the elbow.

Skipping the right branch likewise leaves too little material.

The hub pile at (6,7) is necessary both as branch access and as the fourth elbow slab.

The two floor caches are required because the bucket must be emptied before reclaiming the right-side bridge and before transporting the next elbow unit.

The elbow cannot be reduced to a straight trench without changing its material requirement.

The far pile at (8,4) is the only third free unit on the far side after two elbow slabs have been transferred. Without it, the final trench never gains the three-cell foothold needed for ferrying.

The final five-cell span is not solved by permanently filling five cells. It is solved by repeatedly moving the trailing slab forward.

The exit lies opposite the direction of the elbow crossing, forcing the material to be reorganized rather than merely carried along a single corridor.

7. Implementation invariants

An implementation should assert the following after every nonterminal step():

1. player_position is traversable.
2. player_position is not in floor_sand.
3. player_position is not an empty pit.
4. floor_sand is a subset of ordinary_floor.
5. filled_pits is a subset of pit_cells.
6. floor_sand and filled_pits are disjoint by base-cell type.
7. Total sand count is conserved.
8. bucket_full is exactly boolean.
9. No coordinate contains more than one sand unit.
10. terminal is true iff the player has reached the exit on this or an earlier step.

The authored replays above have shortest action counts of:

Level 1:  8
Level 2: 20
Level 3: 27
Level 4: 33
Level 5: 58
Level 6: 72
Level 7: 98

These counts include blocked directional presses that deliberately set facing, but exclude animations because animation frames are not actions.
