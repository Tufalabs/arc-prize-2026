Floodline

This is a game about taking one dry step becoming difficult because every move—or dam—gives every visible water front one step too.

Floodline contains exactly 7 authored levels.

The player races toward one raised exit while orthogonally spreading water converts dry ground into permanent, impassable terrain. A dam can stop or redirect an entire branch, but placing it consumes the same unit of time as moving, and the finished dam also becomes a wall for the player.

The interesting decisions are therefore spatial rather than numerical:

Is this bracket a genuine chokepoint or merely one cell of a broad front?

Must the player cross a bracket before turning it into a wall?

Does a dam actually delay the water, or merely spend an action?

Which route will the redirected water take afterward?

Will a route needed during the return phase still exist?

There are no keys, switches, inventories, hidden speeds, random choices, or invisible triggers. Every settled frame completely describes the state.

1. Gameplay contract
1.1 Actions

Floodline uses this subset of the supplied action space:

UP

DOWN

LEFT

RIGHT

SPACE

Cell clicking is not used. No cell is rendered as clickable, and pointer events are discarded by the input layer without calling step(). Reset is supplied externally by the host and is not an in-game action.

There is no persistent cursor or selected object. The only transient selection state is the automatically highlighted construction bracket adjacent to the agent.

1.2 Logical movement

The board is an orthogonal logical-cell lattice independent of rendered pixels.

A cardinal action attempts to move the agent exactly one logical cell.

A destination is enterable only if it is:

inside the native board;

ground, an exit cell, or an unbuilt construction cell;

not water;

not a completed dam.

A blocked move leaves the agent in place. It still consumes the action and advances the flood.

1.3 Construction cells

An unbuilt construction cell is ordinary dry, traversable ground marked by unmistakable yellow corner brackets.

A construction cell may be:

walked through while dry;

flooded;

converted permanently into a dam.

At the start of each SPACE action, collect every orthogonally adjacent construction cell that is:

unbuilt;

dry;

not already a dam.

If exactly one exists, it becomes a dam.

If zero exist, SPACE is invalid.

The authored states never place the agent adjacent to more than one eligible bracket. For defensive implementation, if more than one somehow exists, SPACE is invalid rather than choosing arbitrarily.

An invalid SPACE still consumes the action and advances the flood.

The agent cannot build the construction cell it is currently standing on; the cell must be adjacent.

1.4 Dam behavior

A completed dam:

blocks water;

blocks the agent;

never moves;

never disappears;

takes effect before the flood expansion caused by the same SPACE action.

Thus a player may place a dam on the exact action during which water would otherwise enter that cell.

1.5 Water expansion

Water is represented as a set of occupied logical cells.

After every valid or invalid gameplay action:

expanded_water =
    current_water
    union
    every orthogonal neighbor of current_water
    that is not rock and not a completed dam

The expansion is simultaneous.

Consequences:

Every exposed branch advances during the same action.

Two fronts may merge.

A cell is filled at most once.

Water has no directional preference.

Water does not choose one branch over another.

Water never recedes.

The agent, exit, and unbuilt brackets do not block water.

The cell vacated by the agent may flood immediately behind it.

1.6 Exact step() order

For each nonterminal gameplay action:

Clear the previous one-frame feedback marker.

Resolve the player action.

Move, remain in place, place a dam, or reject SPACE.

Commit a newly placed dam immediately.

Compute one simultaneous flood expansion from the pre-expansion water set.

Commit the expanded water set.

Resolve terminal state:

If the agent’s cell is water, set status = LOST.

Otherwise, if the agent occupies the exit, set status = WON.

Otherwise remain RUNNING.

Increment step_count once.

Loss is checked before victory. Moving onto the exit while water reaches it on the same action is a loss.

Once won or lost, later gameplay actions are ignored and do not alter the settled board.

1.7 Exit and validator

Each level has exactly one exit cell.

A level is won iff, after the flood expansion of an action:

agent_position == exit_position
and
exit_position not in water

The validator ignores:

how many unused brackets remain;

how many dams were built;

which routes remain accessible;

whether another region is flooded;

step count, except through the simulated flood state.

The following do not automatically end the level:

the exit becoming flooded;

the exit being sealed behind a dam;

the agent entering a settled but unwinnable region;

all useful construction cells becoming inaccessible.

Such states remain visible until external reset. This lets the player inspect why the plan failed.

2. Visual contract
2.1 Colors

Only integer color IDs 0–15 are used.

ID	Use
0	white exit inset, small agent highlight
1	dry ground
2	optional inactive completion flash
3	outer bevel on connected rock masses
4	rock
5	canvas background outside the native board
6	agent
8	invalid-action mark and loss flash
9	water body
10	foam/front edges
11	inactive construction brackets
12	active bracket and dam core
13	completed dam body
14	dry exit platform
15	victory pulse accent

Colors 7 are unused.

2.2 Cell rendering

No internal grid lines are rendered.

Connected rock cells appear as continuous color-4 masses. Color 3 may be used only along the outside silhouette of a connected rock mass; it must not outline every logical cell.

Dry traversable cells use color 1.

Agent

The agent is a connected magenta shape using color 6. A single color-0 pixel may distinguish it from construction marks. The agent has no facing state.

Exit

A dry exit is a green color-14 platform with a high-contrast color-0 inset.

If flooded, the cell is primarily water color 9 but retains two small color-14 corner pixels. This preserves recognition that the submerged cell was the exit.

Construction bracket

An unbuilt site uses color-11 corner brackets over color-1 ground.

When it is the unique adjacent eligible site, its brackets change to color 12.

This derived highlight is recomputed every frame; it is not stored as gameplay state.

Dam

A dam fills its logical cell with a thick color-13 body and a central color-12 bar. It must look materially heavier than a bracket and clearly non-traversable.

Water and foam

Water cells are color 9.

For every water edge adjacent to a non-water cell:

draw a one-rendered-pixel strip of color 10 on the water side of that edge;

retain foam against dry ground and dams;

foam against rock is optional.

Foam against a completed dam is important: it shows that the front is pressing against the barrier rather than disappearing.

2.3 Native board sizes

Level 1: 15×9, rendered at 4×4 pixels per logical cell, origin (2,14).

Levels 2–3: 12×12, rendered at 5×5 pixels per logical cell, origin (2,2).

Level 4: 14×14, rendered at 4×4 pixels per logical cell, origin (4,4).

Levels 5–7: 31×31, rendered at 2×2 pixels per logical cell, origin (1,1).

The full output is always 64×64.

2.4 Feedback and animation

Ordinary one-cell movement is immediate.

A blocked move shows a color-8 mark on the blocked edge and a one-rendered-pixel bump toward it. This feedback lasts one animation frame.

An invalid SPACE shows a color-8 X over the agent and any adjacent inactive bracket. It does not alter gameplay state beyond consuming the action.

A successful dam placement is shown before flood movement.

Each flood expansion uses three deterministic visual frames:

Current water and foam remain settled.

Foam extends halfway into all cells that will be filled.

New water cells are committed and foam is recomputed.

The gameplay state changes only on frame 3. Animation frames consume no additional actions.

On contact loss:

the agent briefly becomes color 8;

a color-10 splash appears on its four edges;

the final flooded state then freezes.

On victory:

the exit alternates once between colors 14 and 15;

the settled winning board freezes.

3. Authored-map notation

For Levels 1–4, the specification uses ASCII only as documentation:

# rock
. dry ground
~ initial water
A agent
E exit
b construction bracket

These characters are never rendered.

Coordinates are zero-indexed (x,y), with (0,0) at the upper-left.

3.1 Coarse graph notation for Levels 5–7

Levels 5–7 use a 31×31 fine logical grid and a sparse corridor graph.

A coarse node (i,j), where 0 ≤ i,j ≤ 14, maps to fine cell:

N(i,j) = (2i + 1, 2j + 1)

A named coarse edge between orthogonally adjacent nodes opens:

both endpoint node cells;

the fine cell exactly between them.

A coarse edge therefore takes two cardinal actions to traverse.

chain[(a),(b),...] expands each axis-aligned segment through every intermediate coarse node and opens every intervening edge. Consecutive waypoints must share either x or y.

Only explicitly listed edges are open. Two adjacent coarse nodes are not connected unless their edge belongs to a listed chain.

A construction site on a coarse edge replaces its fine midpoint.

All unlisted fine cells are rock.

Initial channel prefixes

Some late-level water channels are long enough to begin partially filled.

For an ordered fine-cell path ending at a target construction cell:

front_distance = d

means the initial water prefix ends exactly d flood expansions before the target cell. The entire occupied prefix is rendered blue, so the timing is visible rather than hidden.

4. Level progression
Level 1 — Three Dry Steps
Teaches

The green platform is the objective.

Every action advances water.

Water expands one orthogonal cell through every open row.

Contact is resolved after movement.

No construction sites appear.

Authored map

Native size: 15×9.

###############
###############
##~..........##
##~......A..E##
##~..........##
##~..........##
###############
###############
###############

Initial state:

Agent: (9,3)

Exit: (12,3)

Water:

(2,2)

(2,3)

(2,4)

(2,5)

Construction sites: none

Dams: none

Active bracket: none

Status: RUNNING

Player-facing discovery

The only salient relation is a broad blue wall to the left, the magenta agent, and the raised green platform three cells to the right.

Pressing RIGHT produces two simultaneous and easily attributable changes:

the agent moves one cell right;

the entire straight foam edge moves one cell right.

Nothing else changes.

The player may safely spend up to six actions experimenting with invalid SPACE presses before making the three-move escape. This gives substantial room to confirm that invalid actions also advance the water.

Intended solution
RIGHT, RIGHT, RIGHT

Length: 3 actions.

The player reaches (12,3) on action 3. The nearest initial water cell is ten steps from the exit, so it remains dry.

Attribution and necessity checks

Six leading invalid SPACE actions followed by RIGHT×3 still win on action 9.

Seven leading invalid actions lose on action 7 when water reaches the stationary agent.

Moving left reduces separation and cannot improve the result.

There are no unrelated objects or simultaneous mechanic changes.

Level 2 — The Plug
Teaches

A highlighted bracket is the target of SPACE.

A newly built dam blocks the ensuing flood expansion immediately.

Foam pressing against a dam means the water remains present but contained.

Authored map

Native size: 12×12.

############
############
############
##~~########
##~~bA.#####
####.#.#####
####.#.#####
####E#.#####
####...#####
############
############
############

Initial state:

Agent: (5,4)

Exit: (4,7)

Water:

(2,3), (3,3)

(2,4), (3,4)

Construction site: (4,4)

Active site: (4,4)

Dams: none

Player-facing discovery

The agent cannot reach the exit by simply moving toward it:

the bracket cell is immediately left;

entering it lets the adjacent water enter the same cell during that action;

the safe-looking route to the right is longer than the water route through the bracket.

The bracket is the only saturated nonterrain marking and is already highlighted. SPACE converts it into a thick barrier. On that same action, foam appears against the new dam while no water appears beyond it.

The resulting region is permanently safe, so after discovering the plug the player can inspect movement without further timing pressure.

Intended solution
SPACE,
RIGHT,
DOWN, DOWN, DOWN, DOWN,
LEFT, LEFT,
UP

Compact form:

S R D^4 L^2 U

Length: 9 actions.

Final position: (4,7).

State-machine checkpoints

After action 1:

dam at (4,4);

water remains entirely in the upper-left pocket;

agent remains (5,4).

After action 5:

agent is (6,8);

water still cannot leave its pocket.

After action 9:

agent is on the dry exit.

Necessity

Without the first dam, the water crosses (4,4), reaches the lower route, and makes the exit unreachable before the agent can complete the detour.

A direct LEFT on action 1 moves onto (4,4) and loses because water enters that cell during the same flood update.

Attribution check

Only one causal unknown is introduced: what SPACE does to the highlighted bracket. The consequence is immediate, local, and not explainable by movement alone.

Level 3 — The Wall You Made
Teaches

A dam blocks the agent exactly as it blocks water.

A construction cell is traversable before construction.

Choosing to build changes the future route, not merely the flood timing.

Authored map

Native size: 12×12.

############
############
############
#####E..####
#####.#.####
###~~bA.####
############
############
############
############
############
############

Initial state:

Agent: (6,5)

Exit: (5,3)

Water: (3,5), (4,5)

Construction site: (5,5)

Active site: (5,5)

Player-facing discovery

The older hypothesis “a dam makes the adjacent route safe” is insufficient here.

After SPACE, trying LEFT does not enter the protected cell. The newly thickened barrier produces a bump and remains stationary while the contained foam advances against it.

The only remaining path is the small right-hand loop.

This forces observation of the dam’s shared impassability. The level cannot be explained as merely “water chooses not to cross yellow cells,” because the agent is visibly rejected by the completed object.

Intended solution
SPACE,
RIGHT,
UP, UP,
LEFT, LEFT

Compact form:

S R U^2 L^2

Length: 6 actions.

Necessity

LEFT on action 1 enters the dry bracket, after which water enters it on the same action: immediate loss.

Taking the right loop without building is too slow; water reaches the exit-side junction first.

After building, attempting to pass through the dam is impossible.

The right loop is therefore not decorative; it exists specifically because construction removes the direct route.

Contrasting role

In Level 2, the dam simply creates safety.

Here, the same operational rule creates safety and destroys the shortest player path.

Level 4 — Around the World
Teaches

One dam beside a rock mass can multiply the flood’s route length.

Dam value is measured by added path distance, not by the number of cells directly covered.

The player and water may be forced around different sides of the same structure.

Authored map

Native size: 14×14.

##############
#............#
#.##########.#
#.##########.#
#.##########.#
#A##########.#
#b......E....#
#~##########.#
#.##########.#
#.##########.#
#.##########.#
#.##########.#
#............#
##############

Initial state:

Agent: (1,5)

Construction site: (1,6)

Initial water: (1,7)

Exit: (8,6)

Active site: (1,6)

Geometry

There are two connected routes between the left and right sides:

The short horizontal cut at y=6.

The long outer ring along the top, right, and bottom edges.

The bracket is the only cell connecting the water directly to the short cut.

Player-facing discovery

Without construction, the foam moves upward through the bracket and then races horizontally toward the exit. The exit floods after only eight actions.

Building the single dam does not freeze the water. The foam visibly turns downward, follows the bottom ring, climbs the right side, and then travels left.

The player is also unable to use the dammed cut and must take the top-right route. The board therefore shows both consequences of the same cell:

it adds eighteen actions to the water route;

it removes the player’s direct route.

Intended solution
SPACE,
UP^4,
RIGHT^11,
DOWN^5,
LEFT^4

Length: 25 actions.

Exact timing

Without a dam:

water-to-exit shortest distance: 8;

the exit floods after action 8.

With the dam:

water path is:

down to the bottom ring: 5 cells;

right across the bottom: 11;

up the right side: 6;

left to the exit: 4;

water reaches the exit on action 26.

The player reaches it on action 25.

Necessity and anti-cleanliness

The one-action margin is produced by the ring geometry, not by an abstract timer.

Removing any of these structural segments changes the puzzle:

removing the top route strands the player;

removing the bottom route contains the water completely and makes timing irrelevant;

moving the exit left removes the one-step race;

moving the bracket off the articulation eliminates the detour multiplication.

The long traversal contains one initial commitment but remains meaningful because the redirected foam is continuously visible around the opposite side of the rock mass.

5. Late-board graph levels

The following three levels use the coarse-node construction defined earlier. The chain lists are normative. All unlisted fine cells are rock.

For each level, the listed action sequence is also a regression test: the level must win on exactly the stated action and produce the stated dam set.

Level 5 — Seal, Work, Seal
Teaches and combines

First explicit seal behind yourself.

A rear seal can buy time for a later construction action.

A second crossing must remain open until it has been used.

A route that is safe before a dam may disappear afterward.

Several visible fronts can impose separate deadlines without having different speeds.

There are three construction sites, but they are not independent checklist items:

A prevents the first front from following the player during the middle detour.

B is crossed and sealed while a second front reaches the vacated side.

C is another rear seal approached from the opposite direction.

Omitting an earlier seal changes whether a later seal can still be placed.

Native board

31×31 fine logical cells.

Open player route
chain[
  (0,2), (5,2)
]

chain[
  (5,2),
  (6,2),
  (6,3),
  (7,3),
  (7,4),
  (8,4),
  (8,3),
  (10,3),
  (10,5),
  (11,5),
  (11,7),
  (9,7),
  (8,7),
  (8,9),
  (5,9)
]
Construction sites
Site	Coarse edge	Fine cell	Strategic role
A	(4,2)–(5,2)	(10,5)	first rear seal
B	(7,4)–(8,4)	(16,9)	middle rear seal
C	(9,7)–(8,7)	(18,15)	reverse-direction rear seal
Initial entities
Entity	Coarse/fine location
Agent	N(2,2) = (5,5)
Exit	N(5,9) = (11,19)
Front α	N(0,2) = (1,5)
Front β	first occupied cell of the β channel below
Front γ	N(14,14) = (29,29)

No site is initially adjacent to the agent.

Pressure channel β
chain[
  (2,8),
  (5,8),
  (5,7),
  (6,7),
  (6,6),
  (7,6),
  (7,4)
]

The ordered fine path continues through site B.

Its initial water prefix is chosen with:

front_distance_to_B = 18

Thus:

the player arrives at the near side of B on action 15;

crosses on actions 16–17;

seals on action 18;

the front reaches the vacated near-side node on action 17;

the dam blocks its action-18 entry into the midpoint.

The water-side corridor is already visibly blue when the agent reaches B, so moving back toward it is not a plausible alternative.

Pressure channel γ
chain[
  (14,14),
  (11,14),
  (11,12),
  (14,12),
  (14,9),
  (12,9),
  (12,10),
  (9,10),
  (9,7)
]

The path has 20 coarse edges from the source to the near-side node of C.

Therefore:

γ reaches the near-side node of C on action 40;

it would enter C’s midpoint on action 41;

the intended action-41 dam blocks it.

Front α

Front α begins at N(0,2) and advances along the opening chase corridor.

It would first enter site A on action 9. The intended rear seal is completed on action 7.

If A remains open, α reaches the central crossing network while the player is spending the action at B. The combined α/β front makes the C crossing wet before it can be sealed. This is a physical dependency: B is not an independent errand that can simply substitute for A.

Intended solution
RIGHT^6,
SPACE,             # seal A behind

RIGHT^2,
DOWN^2,
RIGHT^2,
DOWN^2,
RIGHT^2,
SPACE,             # seal B behind

UP^2,
RIGHT^4,
DOWN^4,
RIGHT^2,
DOWN^4,
LEFT^6,
SPACE,             # seal C behind

DOWN^4,
LEFT^6

Length: 51 actions.

Dam completion actions:

A: action 7

B: action 18

C: action 41

Victory: action 51.

Player-facing discoveries

At A, the bracket is initially walked across rather than built. Once the agent reaches its far side, the bracket highlights. SPACE then visibly strands α foam behind the player.

At B, the player approaches a bracket with blue foam moving toward the same crossing from below. Building before crossing would put a wall directly in the intended route. Crossing first and sealing produces immediate, readable separation.

At C, the orientation is reversed. The same rule must be applied from right to left, preventing the player from memorizing a screen-space recipe such as “always pass a bracket from left to right.”

Necessity regressions

The implementation should satisfy these tests:

Intended sequence: win on action 51.

Replace action 7 with RIGHT: C’s near-side cell is flooded before action 39; no later dam can recover the route.

Omit action 18: the far side of C becomes water during the attempted crossing.

Build B before crossing it: the exit component is disconnected from the agent.

Omit action 41: γ reaches the exit on action 50; the agent’s earliest arrival is 50 and therefore loses during post-action validation.

Add one invalid action after crossing C: loss by contact or flooded exit.

Entering either pressure channel does not provide a shortcut; its next node is already water by the earliest possible entry action.

Anti-cleanliness check

The three sites are not encountered as independent stations:

A changes whether B can be safely serviced.

B changes whether C can be crossed.

C must be delayed despite being visible from the middle loop.

The final route uses dry ground preserved by both earlier seals.

Removing the middle loop turns the level into a trivial race. Removing either pressure channel makes one of the rear seals optional.

Level 6 — One Before the Fork
Teaches and combines

Barrier economy: one upstream dam can replace two downstream dams.

A dam after a fork stops only one branch.

Building both downstream dams may contain the water but lose on time and destroy useful routes.

An unbuilt low-value bracket can later be used as a crossing.

A final rear seal must be delayed until after crossing.

Native board

31×31.

Main player route
chain[
  (5,5),
  (5,7),
  (2,7),
  (2,9),
  (3,9),
  (3,8),
  (4,8),
  (4,6),
  (8,6),
  (8,9),
  (11,9),
  (12,9),
  (12,10),
  (10,10),
  (10,12),
  (12,12)
]
Basin geometry

The basin has one stem leading to a fork.

source stem:
chain[
  (4,14),
  (4,12),
  (1,12),
  (1,10),
  (2,10)
]

fork and left outlet:
chain[
  (2,9),
  (1,9),
  (1,8),
  (1,7),
  (2,7)
]

fork and right outlet:
chain[
  (2,9),
  (3,9),
  (3,8),
  (3,7),
  (2,7)
]
Sites
Site	Edge	Fine coordinate	Role
K	(2,9)–(2,10)	(5,20)	high-leverage stem dam
L	(1,9)–(1,8)	(3,18)	low-value left outlet
R	(3,9)–(3,8)	(7,18)	low-value right outlet
G	(11,9)–(12,9)	(24,19)	rear gate

The agent is never adjacent to two of these sites at once.

Initial entities

Agent: N(5,5) = (11,11)

Exit: N(12,12) = (25,25)

Basin water source: N(4,14) = (9,29)

Rear pressure channel: described below

Dams: none

Basin timing

The stem’s initial water prefix has:

front_distance_to_K = 17

The intended K dam is built on action 15, leaving two actions of visible slack.

If K is not built:

water enters the fork;

both outlets advance simultaneously;

a dam at L does not affect R;

a dam at R does not affect L.

The foam motion makes this immediately visible. A low dam produces a satisfying local stop on one side while the opposite foam edge continues toward the shared hub.

Rear pressure channel
chain[
  (14,0),
  (9,0),
  (9,2),
  (14,2),
  (14,5),
  (12,5),
  (12,7),
  (14,7),
  (14,11),
  (13,11),
  (13,8),
  (11,8),
  (11,9)
]

The initial water prefix is set so that:

front_distance_to_G = 48

The player:

arrives at the near-side node of G on action 45;

enters the midpoint on action 46;

reaches the far-side node on action 47;

seals G on action 48.

The pressure front reaches the vacated near-side node on action 47 and would enter the midpoint on action 48.

Intended solution
DOWN^4,
LEFT^6,
DOWN^4,
SPACE,             # K: stop water before the fork

RIGHT^2,
UP^2,              # cross the still-unbuilt R site
RIGHT^2,
UP^4,
RIGHT^8,
DOWN^6,
RIGHT^6,
RIGHT^2,           # cross G
SPACE,             # seal G behind

DOWN^2,
LEFT^4,
DOWN^4,
RIGHT^4

Length: 62 actions.

Dam actions:

K: action 15

G: action 48

The low sites L and R remain unbuilt.

Player-facing discovery

The player first reaches the central stem bracket K. Two similarly styled brackets are visible off to the left and right, but neither is adjacent.

Trying a low bracket yields unambiguous evidence:

that outlet stops;

the other outlet continues in the same flood action;

the shared hub remains threatened.

Returning and building K stops both outlet fronts because the split never receives water.

Later, the intended route crosses the still-unbuilt R bracket. This reinterprets the same low-value site:

as a poor dam before the fork is understood;

as useful traversable ground after K has made the basin safe;

as a liability if it was needlessly converted into a wall.

Why two low dams are not an alternate checklist solution

Building both L and R is physically possible, but it is not equivalent:

it consumes one additional SPACE;

moving between their staging cells costs additional actions;

both short outlet crossings become walls;

the player must use a longer return loop;

the rear pressure front then reaches G before the player can cross it.

Normative timing:

earliest K solution reaches G’s far side on action 47;

earliest L+R containment reaches it on action 55;

G’s midpoint floods on action 48 unless sealed.

Thus the high dam is required by the interaction of geometry, movement obstruction, and the independent rear front—not by a hidden dam budget.

Necessity regressions

One low dam: the opposite outlet floods (2,7) before action 24.

Both low dams: the basin is contained, but G becomes unusable before arrival.

Build R and then attempt the intended route: the direct transition (3,9)→(3,8) is blocked.

Build G before crossing: the agent is trapped on the wrong side of the only route to the exit loop.

Omit G: the rear front takes the six-edge inner return and floods the exit before action 62.

Intended sequence wins exactly on action 62.

Anti-cleanliness check

The mature dependency structure is not K then G then exit in a simple corridor:

K changes which of three nearby routes remain meaningful.

R changes role from tempting dam to required floor.

The route passes the fork, leaves it, and later depends on the route choice made there.

G must remain open during approach but closed during the final loop.

The low-dam alternative changes both flood topology and player topology.

Level 7 — The Floodline
Purpose

No new object or operational rule is introduced.

The final level integrates:

upstream leverage;

tempting downstream patches;

crossing before sealing;

different fronts sharing later junctions;

a dam that protects one phase while blocking the next-shortest player route;

a forced return through terrain whose meaning has changed;

a one-action final race against redirected water.

Native board

31×31.

Main route network
chain[
  (2,12),
  (2,10),
  (3,10),
  (5,10),
  (5,7),
  (7,7),
  (7,8),
  (8,8),
  (10,8),
  (10,10),
  (11,10),
  (11,9),
  (13,9),
  (13,12),
  (10,12),
  (10,10),
  (7,10),
  (7,6),
  (10,6)
]

The repeated node (10,10) is intentional. It is a phase junction:

first encountered while approaching the east dam;

revisited from below after the central rear seal has altered the direct route.

West basin and candidate sites

High stem:

chain[
  (3,4),
  (1,4),
  (1,8),
  (3,8),
  (3,9)
]

Fork after the high site:

left outlet:
chain[
  (3,10),
  (2,10),
  (2,9),
  (2,8),
  (5,8)
]

right outlet:
chain[
  (3,10),
  (4,10),
  (4,9),
  (5,9),
  (5,7)
]

Sites:

Site	Edge	Fine cell	Role
W	(3,10)–(3,9)	(7,20)	high west choke
WL	(2,10)–(2,9)	(5,20)	low left patch
WR	(4,10)–(4,9)	(9,20)	low right patch
G	(7,8)–(8,8)	(16,17)	central rear seal
E	(11,9)–(12,9)	(24,19)	east choke
Initial entities

Agent: N(2,12) = (5,25)

Exit: N(10,6) = (21,13)

West basin: partially filled along its listed channel

Central pressure front: partially filled along the β channel

East basin: partially filled along the ε channel

Dams: none

West-basin timing

The west channel’s visible initial prefix is set to:

front_distance_to_W = 9

The intended W dam is built on action 7.

If W remains open, water reaches the fork and both low outlets. The right outlet reaches the central-gate approach before the player.

Building only WL or WR leaves the other outlet active.

Building both adds enough travel and construction time that the central front reaches G first. WR also walls off part of the route used to reach (5,7).

Central rear front
chain[
  (0,0),
  (6,0),
  (6,2),
  (1,2),
  (1,5),
  (4,5),
  (4,8),
  (7,8)
]

Its initial water prefix is set to:

front_distance_to_G = 26

The player crosses G on actions 24–25 and seals it on action 26.

At that moment:

the player is on the east side;

the central front reaches the west-side node;

the newly built dam blocks its midpoint entry.

Building G while still on its west side disconnects the east basin and the exit from the agent.

East basin
chain[
  (14,14),
  (8,14),
  (8,11),
  (14,11),
  (14,7),
  (12,7),
  (12,9)
]

The initial prefix is set to:

front_distance_to_E = 40

The player reaches the E staging node on action 38 and builds on action 39. The east water-side node is visibly wet at that time, so stepping through E is not a plausible alternative.

E blocks a short east-basin route to the final exit. The completed dam also removes the direct east-to-north player route, forcing the south return loop.

Intended solution
UP^4,
RIGHT^2,
SPACE,             # W: stop the basin before its fork

RIGHT^4,
UP^6,
RIGHT^4,
DOWN^2,
RIGHT^2,           # cross G
SPACE,             # seal the central front behind

RIGHT^4,
DOWN^4,
RIGHT^2,
UP^2,
SPACE,             # E: close the east shortcut

RIGHT^4,
DOWN^6,
LEFT^6,
UP^4,
LEFT^6,
UP^8,
RIGHT^6

Length: 79 actions.

Dam actions:

W: action 7

G: action 26

E: action 39

Victory: action 79.

The final redirected west front reaches the exit on action 80.

Player-facing progression inside the level
Phase 1: upstream or downstream

The agent initially passes near WL before reaching W. Building WL gives immediate but incomplete feedback: foam remains active in the right outlet. The previously learned fork lesson identifies W as the only one-action containment.

Phase 2: preserve the crossing

G is visible from the west approach. Building it immediately is legal but physically seals the only route to E. The impasse is local and visible: ordinary movement cannot reach the east side once the wall exists.

Crossing first and using SPACE from the far side strands the central foam.

Phase 3: the useful dam becomes a wall

At E, the blue-side neighbor is already wet. SPACE blocks the short east-water path to the exit.

The completed E dam also prevents the player from taking the direct north route. The only remaining dry route descends to (13,12), moves west, revisits (10,10) from below, and climbs the left side.

Thus E solves the immediate water problem while creating the final navigation problem.

Phase 4: altered return

During the return, (7,10) is safe only because W was placed before the fork. A downstream low-dam plan either leaves this cell flooded or walls off the approach to it.

The final climb reaches the exit one action before the redirected west front.

Necessity and order regressions

The implementation must satisfy all of the following:

Intended sequence wins on action 79.

Omit W:

the right west-basin outlet reaches the G approach on action 22;

the player’s earliest arrival is action 23.

Build only WL:

the WR branch remains active and produces the same failure.

Build only WR:

the WL branch floods the later return node (7,10).

Build both WL and WR:

containment succeeds;

the added actions make the earliest exit arrival action 85;

the final front reaches the exit on action 80.

Build G before crossing:

E and the exit lie in a different connected dry component;

no automatic loss occurs, but the settled state is unwinnable.

Omit G:

the central front enters the lower return loop and floods (7,10) before the player revisits it.

Omit E:

the east basin uses its short route and floods the exit on action 71.

Build E and attempt to move through it:

movement is blocked;

the flood still advances.

Add one invalid action after E:

the final arrival ties the redirected front and therefore loses.

Reverse the order of G and W:

the west outlet reaches G’s west-side staging region during the crossing.

Reverse the order of G and E:

E is unreachable without first traversing G.

Anti-cleanliness check

The level cannot be reduced to a list of three required dams:

W determines whether the future return route is dry.

G determines which side of the central region the player occupies.

E changes the path the player must use after it is built.

(10,10) is visited in two different phases with different nearby water states.

The low sites are not cosmetic decoys; they instantiate the exact mistaken strategy “patch each visible leak.”

No site is encountered in the simple order in which its consequence is finally felt.

The final timing margin is generated by the combined detours of all three fronts.

6. Progression summary
Level	Central discovery	Intended length
1	every action advances every front	3
2	SPACE creates an immediate water-blocking dam	9
3	the same dam also blocks the player	6
4	one rock-backed dam creates a large detour	25
5	cross first, then seal; earlier seals enable later work	51
6	one dam before a fork beats two dams after it	62
7	crossing order, leverage, redirection, and forced return	79

The progression introduces no new gameplay noun after Level 2. Later difficulty comes from increasingly strong consequences of the same objects:

a bracket is first a plug;

then a future wall;

then a route-length multiplier;

then a crossing that must remain open temporarily;

then a poor or excellent investment depending on its position relative to a fork;

finally, a choice that determines the topology of a later return.

7. Implementation invariants

An implementation is conformant only if all of these hold:

Water expansion is simultaneous and orthogonal.

Every valid or invalid gameplay action advances water exactly once.

A newly built dam blocks that same action’s expansion.

Water, rock, and completed dams block movement.

Dry unbuilt construction cells do not block movement.

Water may flood construction cells and the exit.

Loss is resolved before victory.

Flooding or sealing the exit does not automatically declare loss.

No gameplay state changes during visual-only animation frames.

No hidden timer, random branch choice, or front priority exists.

Foam is derived entirely from the settled water set.

Clicking has no gameplay binding.

The game contains exactly seven levels.

Each authored solution produces the exact completion step listed above.

All colors written to the 64×64 render belong to integer IDs 0–15.
