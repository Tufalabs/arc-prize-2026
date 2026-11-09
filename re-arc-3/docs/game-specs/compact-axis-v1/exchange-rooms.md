Worked for 43m 33s
Exchange Rooms

Exact authored level count: 7
Native action subset: UP, DOWN, LEFT, RIGHT
Unused actions: SPACE and clicking any cell are inert
Identity:

This is a game about swapping places becoming difficult because every swap also exchanges the access property needed for future routes.

The agent can never push a statue forward. Moving into an adjacent statue exchanges the agent and statue: each takes the other’s position and visible emblem. Consequently, every useful interaction simultaneously changes:

where the agent can walk next;

where the statue can later be approached;

which colored gates the agent can cross;

which emblem has been left behind for a pad.

The rules remain unchanged throughout all seven levels. Later difficulty comes from approach direction, color circulation, gates that cannot receive statues, temporary unsolving, delayed pad commitment, and forced revisits.

1. Player-facing rules
Movement and swapping

UP, DOWN, LEFT, and RIGHT target the orthogonally adjacent logical cell.

Moving toward empty floor or an empty pad moves the agent one cell.

Moving toward an adjacent statue performs a simultaneous exchange:

the agent enters the statue’s cell;

the statue enters the agent’s former cell;

the agent receives the statue’s former emblem;

the statue receives the agent’s former emblem.

A useful equivalent invariant is:

During the instant of a swap, the emblem displayed in each of the two occupied cells stays in that cell; the agent and statue exchange which body carries it.

Ordinary walking then carries the agent’s current emblem through space.

An immediately reversed swap restores both positions and emblems, provided the first swap did not begin from a gate. Early experimentation is therefore safe.

Gates

A gate is colored and carries the same pattern vocabulary as movable emblems.

Only the agent can occupy a gate cell.

The agent can enter a gate only when its current emblem matches that gate.

A statue can never occupy a gate.

Therefore, a swap is invalid when the agent is currently standing on a gate: the statue would otherwise have to land on that gate cell.

Adjacent gate cells of the same color form a contiguous gate strip. The agent checks the match independently when entering each cell.

Gates never open, close, or remember previous crossings.

Pads and completion

Each pad has a fixed target emblem.

A pad is correct exactly when:

it is occupied by a statue; and

that statue’s current emblem matches the pad.

Original statue identity is irrelevant. The agent never satisfies a pad.

The level succeeds immediately after an action when every pad is simultaneously correct. Agent position, agent emblem, gate occupancy, statue identity, and move count are ignored by the validator.

Every level has the same number of statues and pads, so success also implies that every statue is on a pad.

Exploration and unused actions

There is no gameplay timer, energy, life system, or move budget. Invalid tests are harmless.

SPACE and clicking any cell cause no gameplay-state change. There are no clickable objects, selected objects, cursor states, or hidden hit regions.

2. Visual contract
Canvas and logical lattice

The render canvas is exactly 64×64.

Each level defines a W×H logical lattice. Coordinates use:

x=1 at the left;

y=1 at the top.

For a level of size W×H:

pitch = min(8, floor(64 / W), floor(64 / H))
origin_x = floor((64 - W * pitch) / 2)
origin_y = floor((64 - H * pitch) / 2)

Logical cell (x,y) occupies:

[origin_x + (x-1)*pitch, origin_x + x*pitch)
×
[origin_y + (y-1)*pitch, origin_y + y*pitch)

There are no decorative grid lines and no borders around individual floor cells. Connected wall cells render as one continuous wall mass.

Used palette
Role	Color ID
White feedback and correct-pad highlight	0
Wall	2
Statue plinth	3
Floor	4
Canvas outside the board	5
Red emblem	8
Blue emblem	9
Yellow emblem	11
Green emblem	14
Purple emblem	15

The game never renders text.

Emblem redundancy

Every gameplay color has a fixed 2×2 white motif, repeated on agents, statues, and gates.

Abbreviation	Color	ID	Motif
B	Blue	9	10 / 10
Y	Yellow	11	11 / 00
R	Red	8	10 / 01
G	Green	14	01 / 10
P	Purple	15	10 / 11

A motif bit of 1 is white; a 0 remains the emblem color. Thus the five properties remain distinguishable without relying only on hue.

Occupant silhouettes

Sprites use a normalized 4×4 mask. For larger logical cells, scale the mask by the largest integer factor that fits and center it.

C means the current emblem color. The four M positions contain the emblem motif.

Agent:

.CC.
CMMC
CMMC
.CC.

Statue:

.CC.
CMMC
CMMC
.DD.

D is color ID 3. The dark plinth makes statues immediately distinguishable from the agent.

A gate fills its logical cell with its gate color and displays the same centered motif.

Pads

An empty pad is a floor inset:

at pitch≥5, draw a one-pixel target-colored perimeter;

at pitch=4, draw the four target-colored corner pixels;

draw the target motif in the center using the target color.

After drawing an occupant, redraw the pad’s target-colored corners over it. Therefore, an occupied pad remains identifiable.

A correct statue gains persistent white edge-midpoint highlights while the target-colored corners remain visible. A wrong-colored statue on a pad shows:

the statue’s current emblem;

the pad’s different target-colored corners;

no white correctness highlight.

The agent never receives the correctness highlight.

3. Animation and feedback

All transitions occur only in response to step(action).

Ordinary movement

One-cell movement is immediate. No ornamental walking animation is required.

Swap animation

A valid swap uses four deterministic frames:

old settled positions;

both silhouettes shift toward their shared edge while their emblem panels remain visually anchored to their old cells;

silhouettes overlap at the shared edge;

final exchanged positions and emblems appear, followed by pad-highlight recomputation.

The gameplay state commits at frame 4. No input is accepted during these frames, and animation frames consume no actions.

If an implementation exposes only one observation per step(), it returns frame 4.

Gate movement

A successful gate entry uses a two-frame brightening of the entered gate, then returns the committed settled frame.

Invalid action

For a wall, mismatched gate, or forbidden gate-to-statue swap:

gameplay positions and emblems remain unchanged;

the agent shifts one rendered pixel toward the obstruction and returns;

the obstructing cell retains a white edge highlight in the resulting observation until the next action.

For a forbidden swap from a gate, both the current gate and adjacent statue receive the highlight. This makes the reason for failure spatially explicit.

Completion

When the validator becomes true:

all correct-pad white highlights pulse once;

terminal=true;

the final settled configuration remains visible;

subsequent actions are inert;

the host advances to the next level without requiring SPACE.

4. State-machine contract
Gameplay state
terrain[x,y] ∈ {WALL, FLOOR, GATE(color), PAD(color)}
agent = {position, emblem}
statues[i] = {position, emblem}
terminal ∈ {false, true}
feedback ∈ {none, bump(cell), blocked_swap(gate, statue), gate(cell), swap(edge), complete}

Statue IDs may be used internally for deterministic storage, but IDs:

are not rendered;

do not affect movement;

are ignored by the validator.

There is no other persistent state.

Invariants

Every entity occupies exactly one logical cell.

The agent and statues never share a cell.

Statues occupy only floor or pad cells.

The agent may occupy floor, pad, or matching gate cells.

Gates and pads never overlap.

The multiset of emblems carried by the agent and statues is conserved.

Nothing moves spontaneously.

No rule depends on move history.

Exact transition order
step(action):
    if terminal:
        return unchanged terminal observation

    clear previous feedback

    if action is SPACE or a click:
        return unchanged gameplay state

    q = agent.position + directional_delta(action)

    if q is outside the lattice or terrain[q] is WALL:
        feedback = bump(q)
        return

    if a statue s occupies q:
        if terrain[agent.position] is GATE:
            feedback = blocked_swap(agent.position, q)
            return

        old_agent_position = agent.position
        old_agent_emblem = agent.emblem
        old_statue_emblem = s.emblem

        agent.position = q
        agent.emblem = old_statue_emblem
        s.position = old_agent_position
        s.emblem = old_agent_emblem
        feedback = swap(old_agent_position, q)

    else if terrain[q] is GATE(gate_emblem):
        if agent.emblem != gate_emblem:
            feedback = bump(q)
            return

        agent.position = q
        feedback = gate(q)

    else:
        agent.position = q

    recompute every pad's correctness

    if every pad is correct:
        terminal = true
        feedback = complete
Validator
success = true

for each pad p:
    s = statue occupying p.position
    if s does not exist or s.emblem != p.target_emblem:
        success = false

Nothing else is checked.

5. Authored-level notation

Terrain maps below do not show occupants. Occupants are listed separately.

#  wall
.  ordinary floor
B  blue gate
Y  yellow gate
R  red gate
G  green gate
P  purple gate
b  blue pad
y  yellow pad
r  red pad
p  purple pad

In solution traces:

U = UP
D = DOWN
L = LEFT
R = RIGHT

Groups separated by | are concatenated directly.

Each listed trace has been replay-checked and is shortest among state-changing successful traces for that level. Suggested bump experiments are excluded because they leave the gameplay state unchanged.

Level	Main discovery or composition	Successful trace length
1	A statue exchanges positions with the agent	1
2	Emblems exchange even away from pads	9
3	Gate entry depends on the agent’s current emblem	6
4	A two-gate color circuit and temporary color reversal	28
5	A gate cell cannot receive a statue	40
6	A solved statue can be a blocker that must be unsolved	54
7	A tempting pad must remain empty until a later revisit	85
Level 1 — Make Room
Player-facing discovery

The agent begins standing on a blue pad, with a blue statue immediately to its right. Every other neighboring cell is a wall.

Pressing RIGHT does not push the statue forward. Instead, the agent and statue exchange positions. The statue appears on the agent’s former pad, the pad gains its white correctness highlight, and the level completes.

Because both participants are blue, this level introduces only positional exchange. Emblem transfer is deliberately not yet observable.

The physical meaning of “stand where the statue should go” is demonstrated in one action.

State-machine instance

Board: 5×5

#####
#####
#b.##
#####
#####

Initial state

Agent:  position (2,3), emblem B
S1:     position (3,3), emblem B
Pads:   B at (2,3)
Gates:  none
terminal=false
feedback=none
selection/cursor=none
timer/budget=none

Reference trace — 1 action

R

Resolution

Before:
    Agent (2,3,B)
    S1    (3,3,B)

After R:
    Agent (3,3,B)
    S1    (2,3,B)
    pad (2,3,B) correct
    terminal=true
Attribution and necessity check

The only useful directional action is RIGHT.

The statue must enter the agent’s old cell; ordinary movement cannot satisfy the pad.

No color changes, gates, alternate paths, or unrelated objects obscure the result.

Both traversable cells participate directly in the experiment and solution.

Level 2 — The Emblem Stays Behind
Player-facing discovery

The blue agent again begins adjacent to a statue, but the statue is yellow and both are far from the blue pad.

The immediate known experiment is RIGHT. It occurs on ordinary floor:

the yellow agent appears in the statue’s former cell;

a blue statue appears in the agent’s former cell;

the distant pad does nothing.

This directly rules out the mistaken hypotheses that pads recolor statues or that statues retain permanent colors.

The rest of the room makes the same statue alternate roles:

first it supplies yellow to the agent and receives blue;

the second swap repositions it but temporarily returns yellow to it;

the third swap puts blue back on it exactly as it enters the pad.

State-machine instance

Board: 5×6

#####
#b.##
#..##
#..##
#####
#####

Initial state

Agent:  position (2,4), emblem B
S1:     position (3,4), emblem Y
Pads:   B at (2,2)
Gates:  none
terminal=false
feedback=none
selection/cursor=none
timer/budget=none

Reference trace — 9 actions

R | ULD | RUULD

Concatenated:

RULDRUULD

Replay

R
Agent becomes Y at (3,4); S1 becomes B at (2,4).
This swap is on ordinary floor.

ULD
The agent reaches (2,3) and swaps down.
Agent becomes B at (2,4); S1 becomes Y at (2,3).

RUULD
The agent walks around to the blue pad at (2,2) and swaps down.
S1 becomes B on (2,2); the agent becomes Y at (2,3).

Attribution and necessity check

The first emblem exchange occurs two cells away from the pad, so pad effects cannot explain it.

Three swaps are necessary: the statue must travel two cells upward while ending with blue, requiring odd swap parity.

Every traversable cell in the small room is used by the reference solution.

Every swap remains immediately reversible while being tested.

Level 3 — Same Color, Same Door
Player-facing discovery

The yellow agent begins immediately below a blue gate and immediately left of a blue statue.

A blind player’s cheapest gate experiment is UP:

the agent remains in place;

the blue gate flashes white.

The adjacent statue then supplies the missing blue emblem. After the swap, the agent approaches the same gate, which now allows passage. The route through that gate is the only way to reach the destination side of the statue.

This distinguishes gate matching from ordinary collision or a permanently closed wall.

State-machine instance

Board: 7×6

#######
#######
#.B.###
#b..###
#######
#######

Initial state

Agent:  position (3,4), emblem Y
S1:     position (4,4), emblem B
Pads:   B at (2,4)
Gates:  B at (3,3)
terminal=false
feedback=none
selection/cursor=none
timer/budget=none

Recommended discovery experiment

U

This bumps the blue gate while the agent is yellow. The gameplay state remains unchanged.

Reference successful trace — 6 state-changing actions

R | ULLD | R

Concatenated:

RULLDR

Replay

R
Agent becomes B at (4,4); S1 becomes Y at (3,4).

U
Agent moves to (4,3).

L
Agent enters the blue gate at (3,3).

LD
Agent reaches the blue pad at (2,4).

Final R
S1 moves from (3,4) to the pad (2,4) and receives blue.
Agent receives yellow at (3,4).

Attribution and necessity check

The same gate visibly rejects yellow and admits blue.

The gate lies on the only route to the pad side of the statue.

The gate cannot be ignored or bypassed.

No second causal mechanism changes between the failed and successful gate attempts.

Level 4 — Two-Gate Circuit
Player-facing discovery

The agent begins red in the left room. A yellow statue is adjacent, while a blue gate and a yellow gate connect the two sides of the map at different heights.

The level establishes a complete color circuit:

take yellow from the left statue;

enter the right room through the yellow gate;

route the blue statue onto the yellow pad using three swaps;

finish those swaps holding blue;

return through the blue gate;

leave blue on the left pad.

The right statue is not solved by a single delivery. Its three swaps alternate its color:

B → Y → B → Y

Its middle swap deliberately makes it temporarily wrong-colored so that it can be approached from the final side. This is the first contrasting use of emblem exchange: the same alternation that opens routes also complicates positional routing.

State-machine instance

Board: 10×8

##########
#...B....#
#...#....#
#b..#.y..#
#...#....#
#...#....#
#...Y....#
##########

Initial state

Agent:  position (2,5), emblem R

S1:     position (3,5), emblem Y
S2:     position (8,6), emblem B

Pads:
    B at (2,4)
    Y at (7,4)

Gates:
    B at (5,2)
    Y at (5,7)

terminal=false
feedback=none
selection/cursor=none
timer/budget=none

Reference trace — 28 actions

R | DDRRR | UURRD | LUR | ULD | LUUULL | DDLLD

Concatenated:

RDDRRRUURRDLURULDLUUULLDDLLD

Replay by phase

R

S1 moves to (2,5) and receives red.

Agent moves to (3,5) and receives yellow.

DDRRR

Agent reaches and crosses the yellow gate at (5,7).

Agent enters the right room at (6,7).

UURRD

Agent stands at (8,5) and swaps down with S2.

S2 becomes yellow at (8,5).

Agent becomes blue at (8,6).

LUR

The second swap moves S2 to (7,5).

S2 becomes blue again.

Agent becomes yellow.

ULD

Agent reaches the yellow pad at (7,4) and swaps down.

S2 becomes yellow on the pad.

Agent becomes blue at (7,5).

LUUULL

Agent crosses the blue gate at (5,2) and re-enters the left room.

DDLLD

Agent reaches the blue pad (2,4) and swaps down with S1.

S1 becomes blue on the pad.

Agent recovers red.

Anti-degenerate check

The yellow gate is required to enter the right room after taking yellow.

The blue gate is required for the return after routing S2.

S1 must remain unresolved until the return trip.

S2’s three positional moves require three swaps; its temporary wrong color is not decorative.

The solution revisits the left room rather than consuming objects in encounter order.

Removing either gate, statue, or pad collapses the circuit.

Level 5 — No Statue on the Threshold
Player-facing discovery

This level introduces the remaining gate consequence: a gate is traversable floor for the matching agent, but it is not a legal destination for a statue.

The green agent first takes blue from the left statue. A wall forces entry through the upper cell of a two-cell blue gate strip. On that gate cell, the agent faces a yellow statue immediately to the right.

Under the older swap rule, RIGHT would appear useful. Instead:

the current gate flashes;

the statue flashes;

neither moves.

The lower blue gate cell remains accessible. Moving down, right onto the blue pad, and then up approaches the same statue from ordinary pad terrain. The swap succeeds.

The rest of the level extends the color circuit to three colors:

Blue → Yellow → Red
State-machine instance

Board: 13×11

#############
#....#......#
#....B......#
#...#Bb.....#
#....#......#
#r...###YY###
#....#......#
#....R..y...#
#....R......#
#....#......#
#############

Initial state

Agent:  position (2,7), emblem G

S1:     position (3,7), emblem B
S2:     position (7,3), emblem Y
S3:     position (11,9), emblem R

Pads:
    R at (2,6)
    B at (7,4)
    Y at (9,8)

Gates:
    B at (6,3), (6,4)
    Y at (9,6), (10,6)
    R at (6,8), (6,9)

terminal=false
feedback=none
selection/cursor=none
timer/budget=none

The wall at (5,4) prevents entering the lower blue gate directly from the left. This guarantees arrival at (6,3) facing S2.

Recommended local experiment

After executing:

R UUUURRR

the agent is blue on gate (6,3). Trying RIGHT produces the visible forbidden-swap bump.

Reference successful trace — 40 actions

R | UUUURRR | DRU | RDDRDD | DDRR | ULD | LUR | ULLDLLL | UULLLD

Concatenated:

RUUUURRRDRURDDRDDDDRRULDLURULLDLLLUULLLD

Replay by phase

R

S1 becomes green at (2,7).

Agent becomes blue at (3,7).

UUUURRR

Agent reaches blue gate (6,3).

DRU

Agent moves down along the blue gate strip to (6,4).

Agent steps right onto blue pad (7,4).

UP swaps with S2:

S2 becomes blue on (7,4);

agent becomes yellow at (7,3).

RDDRDD

Agent reaches and crosses yellow gate (9,6).

Agent enters the lower-right region at (9,7).

DDRR

Agent approaches S3 from (10,9) and swaps right.

S3 becomes yellow at (10,9).

Agent becomes red at (11,9).

ULD

S3 moves to (10,8) and temporarily becomes red.

Agent becomes yellow.

LUR

S3 moves to yellow pad (9,8) and becomes yellow.

Agent becomes red.

ULLDLLL

Agent reaches and crosses red gate (6,8).

Agent returns to the left side at (5,8).

UULLLD

Agent reaches red pad (2,6) and swaps down with S1.

S1 becomes red on the pad.

Agent recovers green.

Anti-degenerate check

S2’s placement beside the blue gate is necessary to expose the forbidden landing rule.

The wall at (5,4) forces the informative approach through the upper gate cell.

The lower gate cell and blue pad provide a nearby physically meaningful alternative.

All three carried colors are needed for different access phases.

S3 must be moved three times; it alternates between tool and obstruction.

The red return gate and left revisit prevent the room from becoming a one-way checklist.

Level 6 — Unsolve to Proceed
Player-facing discovery

One statue begins correctly matched on its blue pad, with the persistent white correctness highlight already visible.

That statue occupies the only opening in a vertical wall. The agent cannot reach any other region without disturbing it.

Swapping right:

turns off the pad’s correctness highlight;

moves the statue behind the agent;

gives the agent blue;

opens the route through the blue gate.

The level therefore establishes that correct statues are not locked prizes. A solved statue may be a temporary door, color source, and future obligation.

The middle statue is then staged with blue but not yet solved. After the lower statue supplies red, the player returns to solve the middle red pad, recovers blue, and finally restores the initially correct statue.

State-machine instance

Board: 14×12

##############
#..#.#.......#
#..#.#..r....#
#..#.B.......#
#..b.B.......#
#..#.##YY##RR#
#..#.#.......#
#..#.#..y....#
#..#.#.......#
#..#.#.......#
#..#.#.......#
##############

Initial state

Agent:  position (3,5), emblem G

S1:     position (4,5), emblem B
S2:     position (11,5), emblem Y
S3:     position (12,10), emblem R

Pads:
    B at (4,5), initially correct under S1
    R at (9,3)
    Y at (9,8)

Gates:
    B at (6,4), (6,5)
    Y at (8,6), (9,6)
    R at (12,6), (13,6)

terminal=false
feedback=none
selection/cursor=none
timer/budget=none

The internal wall at x=4 occupies every interior row except (4,5). S1 is therefore the only passage.

Reference trace — 54 actions

R
| RRRRRRR
| ULD
| LUR
| DLDD
| DDDRRR
| ULD
| LUR
| ULD
| LUR
| URRUU
| UULLLD
| DLLLLLL

Concatenated:

RRRRRRRRULDLURDLDDDDDRRRULDLURULDLURURRUUUULLLDDLLLLLL

Replay by phase

Initial R

S1 moves to (3,5) and becomes green.

The blue pad (4,5) becomes incorrect.

Agent enters (4,5) and becomes blue.

Pressing LEFT immediately would restore the original state, making this a safe experiment.

RRRRRRR

Agent crosses blue gate (6,5).

The final R swaps with S2:

S2 becomes blue at (10,5);

agent becomes yellow at (11,5).

ULD

S2 moves to (10,4) and becomes yellow.

Agent becomes blue.

LUR

S2 moves to (9,4) and becomes blue.

Agent becomes yellow.

S2 is now staged directly below the red pad (9,3).

DLDD

Agent crosses the yellow gate and enters the lower-right region.

DDDRRR

Agent reaches S3 and performs its first swap.

S3 becomes yellow at (11,10).

Agent becomes red.

The next four groups route S3 along an offset staircase:

ULD   -> S3 at (11,9), emblem R
LUR   -> S3 at (10,9), emblem Y
ULD   -> S3 at (10,8), emblem R
LUR   -> S3 at (9,8),  emblem Y, pad correct

The agent ends red at (10,8).

URRUU

Agent crosses the red gate at (12,6) and returns to the upper region.

UULLLD

Agent stands on red pad (9,3) and swaps down with staged S2.

S2 becomes red on the pad.

Agent takes blue.

DLLLLLL

Agent crosses the blue gate in the reverse direction.

Agent reaches blue pad (4,5) and swaps left with S1.

S1 becomes blue on its original pad.

Agent recovers green.

Anti-degenerate check

Keeping the initially solved statue untouched makes every other objective unreachable.

The initial correct border turning off is immediate evidence that solved statues remain ordinary movable statues.

S2 must be staged and revisited; it cannot be independently completed during its first encounter.

S3 requires five swaps and alternating approach sides.

The final blue obtained from S2 is needed both for access and for restoring S1.

The level contains a dependency loop:

unsolve S1
→ stage S2
→ solve S3
→ solve S2
→ restore S1

This cannot be reduced to a corridor or a list of independent targets.

Level 7 — Leave the Blue Pad Empty
Player-facing discovery

The final level composes every established consequence into a four-statue dependency network.

Initial latch

A blue statue occupies the sole opening in an internal wall. The green agent begins immediately to its left and cannot reach either eastern gate route.

Swapping right:

moves the statue to (4,4) with green;

puts the agent at (5,4) with blue;

opens the wall passage.

The blue gate can now be used. The nearby green gate is visible but mismatched.

The tempting blue pad

In the upper-right room, the yellow statue must be moved leftward and left blue at (11,5).

Its blue pad at (10,6) is directly above the only yellow gate at (10,7). Filling that pad during the first visit would put a statue on the only cell from which the yellow gate can be entered.

Thus the visibly attractive early completion is strategically premature. The statue must remain staged while the pad stays empty.

The outer color circuit

The agent carries yellow through the yellow gate, solves the lower-right statue on the yellow pad, and acquires red.

Red crosses to the lower-left room. Routing the purple statue onto the red pad gives the agent purple.

Purple returns to the initial room and solves the first statue on the purple pad. That exchange returns green to the agent and leaves the wall opening clear.

Only then can the agent take the green gate back into the upper-right room.

Final two-swap delivery

The staged statue still holds blue.

The green agent first swaps with it to pick up blue and place green at (10,5). The agent then stands on the previously preserved blue pad (10,6) and swaps upward, leaving blue on the statue and completing the level.

The upper-right statue is therefore:

a yellow color source;

a blue color store;

an object that must deliberately remain unsolved;

the final target.

Its operational rule never changes.

State-machine instance

Board: 16×14

################
#...#..#.......#
#..p#..G.......#
#......G.......#
#...#..#.......#
#...#..B.b.....#
##PP#####Y######
#......#.......#
#...r..#.......#
#......R...y...#
#......R.......#
#......#.......#
#......#.......#
################

Initial state

Agent:  position (4,4), emblem G

S1:     position (5,4),  emblem B
S2:     position (14,5), emblem Y
S3:     position (14,11), emblem R
S4:     position (2,11), emblem P

Pads:
    P at (4,3)
    B at (10,6)
    Y at (12,10)
    R at (5,9)

Gates:
    G at (8,3), (8,4)
    B at (8,6)
    Y at (10,7)
    R at (8,10), (8,11)
    P at (3,7), (4,7)

terminal=false
feedback=none
selection/cursor=none
timer/budget=none
Structural facts encoded by the map

The internal wall at x=5, rows 2,3,5,6, has only one opening: (5,4), initially occupied by S1.

The horizontal wall at y=7 separates upper and lower rooms. Its only right-side opening is yellow gate (10,7).

Blue pad (10,6) is the sole cell immediately above that yellow gate.

S2 occupying (10,6) therefore blocks access to the lower-right room.

Green gates are unreachable before moving S1 because the internal wall blocks the eastern half of the initial room.

Reference trace — 85 actions

RRRDDRR
| URRRRR
| ULLDR
| ULLDR
| DLLDD
| DDDRRRR
| ULD
| LUR
| DLLLLLL
| LLLLL
| URD
| RUL
| URD
| RUL
| UUU
| LUUURD
| RRRRR
| DRR
| DLU

Concatenated:

RRRDDRRURRRRRULLDRULLDRDLLDDDDDRRRRULDLURDLLLLLLLLLLLURDRULURDRULUUULUUURDRRRRRDRRDLU
Replay by dependency phase
Phase A — Open the initial latch and enter with blue

RRRDDRR

First R swaps with S1:

S1 becomes green at (4,4);

agent becomes blue at (5,4).

The agent walks through the opened wall gap.

The agent reaches and crosses blue gate (8,6).

The agent enters the upper-right room at (9,6).

Phase B — Stage S2 without filling its pad

URRRRR

The agent approaches S2 from (13,5) and swaps right.

S2 becomes blue at (13,5).

Agent becomes yellow at (14,5).

First ULLDR

S2 moves to (12,5) and becomes yellow.

Agent becomes blue.

Second ULLDR

S2 moves to (11,5) and becomes blue.

Agent becomes yellow at (12,5).

The blue pad (10,6) deliberately remains empty.

Phase C — Preserve the choke and cross yellow

DLLDD

Agent moves through (10,6) while it is still empty.

Agent enters yellow gate (10,7).

Agent reaches the lower-right room at (10,8).

Had S2 already occupied (10,6), this crossing would be impossible. Because an agent standing on the yellow gate cannot swap the statue off the pad, the early placement is a genuine positional commitment, not an abstract prohibition.

Phase D — Solve the lower-right statue and obtain red

DDDRRRR

Agent reaches (13,11) and swaps right with S3.

S3 becomes yellow at (13,11).

Agent becomes red at (14,11).

ULD

S3 moves to (13,10) and becomes red.

Agent becomes yellow.

LUR

S3 moves to yellow pad (12,10) and becomes yellow.

Agent becomes red.

Phase E — Cross red and route the purple statue

DLLLLLL

The occupied yellow pad at (12,10) blocks the direct left route.

Agent must descend to row 11 and cross red gate (8,11).

Agent enters the lower-left room at (7,11).

LLLLL

Agent reaches and swaps left with S4.

S4 becomes red at (3,11).

Agent becomes purple at (2,11).

The next four groups route S4 through a staircase:

URD   -> S4 at (3,10), emblem P
RUL   -> S4 at (4,10), emblem R
URD   -> S4 at (4,9),  emblem P
RUL   -> S4 at (5,9),  emblem R, pad correct

Agent ends purple at (4,9).

Phase F — Return through purple and solve the initial statue

UUU

Agent crosses purple gate (4,7) and enters the initial region at (4,6).

LUUURD

Agent routes around the staged green S1.

Agent stands on purple pad (4,3).

Final D swaps with S1 at (4,4):

S1 becomes purple on (4,3);

agent becomes green at (4,4).

Cell (5,4) is now empty, so the internal wall passage is open.

Phase G — Use the newly available green return route

RRRRR

Agent passes through the opened wall gap.

Agent crosses green gate (8,4).

Agent re-enters the upper-right room at (9,4).

This route was physically unreachable in the initial state because S1 occupied the wall opening.

Phase H — Recover blue and commit the final pad

DRR

Agent reaches (10,5) and swaps right with staged S2.

Agent becomes blue at (11,5).

S2 becomes green at (10,5).

DLU

Agent moves down to (11,6).

Agent moves left onto blue pad (10,6).

UP swaps with S2:

S2 becomes blue on (10,6);

agent becomes green at (10,5);

all four pads become correct;

terminal=true.

Anti-cleanliness and necessity check

S1 is not merely a key station: its physical body blocks the only opening, stores green, becomes a late purple target, and determines when the green route becomes reachable.

S2 is encountered early but solved last.

Blue pad (10,6) is both an objective and a route cell. Its meaning changes solely because of timing and occupancy.

Solving S2 early creates a concrete obstruction at the only yellow-gate approach, rather than failing an abstract ordering predicate.

S3’s solved position forces the lower red-gate approach through row 11.

S4 requires five directional swaps and alternating approach sides.

Every gate color is used:

blue for the first eastern entry;

yellow for descent;

red for the lower crossing;

purple for return;

green for the final revisit.

Removing any statue, pad, gate, or named internal wall opening breaks a dependency rather than merely shortening travel.

The mature dependency graph is:

move S1 to obtain B and open east
            │
            ▼
stage S2 with B, but preserve its pad
            │
            ▼
solve S3 to obtain R
            │
            ▼
solve S4 to obtain P
            │
            ▼
solve S1 to recover G and expose green route
            │
            ▼
revisit and solve S2 last

Although the phases form a solution, the visible objects are not encountered in completion order, and the first available target must be deliberately postponed. The level therefore tests future reachability and commitment rather than execution of a spatial checklist.
