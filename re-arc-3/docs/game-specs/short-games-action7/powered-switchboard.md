Powered Switchboard

“This is a game about toggling every cable around a powered switch becoming difficult because the click can cut off the controls needed to repair what it changed.”

Exact authored level count: 7.

Powered Switchboard is a deterministic click-only graph puzzle. The switches are simultaneously the controls and the electrical network being controlled: clicking a switch changes every cable touching it, but a switch can only be clicked while it is connected to the source.

The objective is physically visible rather than symbolically checked: illuminate every terminal bulb at the same time.

The game’s depth comes from access:

A switch can open its own last live cable and become unusable.

A neighboring powered switch can restore a shared cable from the opposite endpoint.

A switch can exchange one live feed for another and remain powered.

A cross-connection can move the useful power frontier between branches.

A finished section can be left disconnected when repairing it would disturb the remaining bulbs.

No hidden graph state is required. Closed but unpowered cables remain solid and dark, open cables always retain visible separated stubs, and the complete source-connected component is brightly highlighted in every settled frame.

1. Controls and objective
Action space

The environment accepts exactly:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x, y) on any integer canvas cell

Only CLICK has gameplay semantics.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic no-ops. They increment the environment step counter but do not alter puzzle state or play an animation.

There is no avatar, cursor, persistent selection, energy meter, timer, or action limit. All seven levels allow unrestricted experimentation. An environment-level reset() restores the authored initial state but is not a player action and is not part of the action space.

Clickable region

Each switch occupies a clearly visible 3×3 plate. Clicking any of those nine cells targets that switch.

Only a switch that is powered before the click accepts the click. The source-mounted root switch is always powered.

Clicking any of the following has no logical effect:

An unpowered switch

A bulb

A cable

The source pad outside the central root switch

Empty background

There is no persistent cursor:

selected_switch = null
cursor_position = null

A reference trace clicks the center cell of each named switch.

Success condition

After every accepted switch click and the subsequent power recomputation:

success = every terminal bulb is connected to the source

The following are irrelevant to success:

Which switches remain powered

Which non-terminal cables remain open or closed

Whether unused controls have been stranded

The number of clicks taken

Extra powered switches and closed loops are allowed. The only forbidden final condition is having one or more bulbs unlit.

When success becomes true, done = true immediately. Further actions are ignored.

2. Visual and spatial contract
Canvas

Render canvas: 64×64

Integer coordinates: (x, y)

Origin: top-left

x increases rightward

y increases downward

Every rendered cell contains one integer color ID from 0–15

No text, numbers, arrows, legends, or internal grid borders are rendered

Empty background is color 4

Internal IDs such as A3 and N7 appear only in this specification.

Restrained color use
Purpose	Color ID
Hot powered center / bulb center	0
Switch corner plate	1
Unpowered closed cable and open stubs	2
Unpowered switch body	3
Background and physical gaps	4
Dark center	5
Accepted-click outline	10
Powered switch, energized cable, lit bulb	11
Source pad	12
Unlit bulb	13
Completion halo	14

Colors 6–9 and 15 are unused in settled gameplay frames.

Switch silhouette

A switch centered at (cx, cy) occupies the full 3×3 box from (cx−1, cy−1) through (cx+1, cy+1). All nine cells form its click hitbox.

Powered switch:

1  11  1
11  0 11
1  11  1

Unpowered switch:

1  3 1
3  5 3
1  3 1

The fixed light-gray corners preserve the identity of a dark switch without making it look powered.

Source-mounted root switch

Every level has exactly one root switch R.

A Manhattan-radius-3 diamond of color 12 is rendered behind R. The ordinary 3×3 powered switch is drawn over its center. This communicates a direct physical mounting to the source.

The source contact is not a cable and is never toggled. Clicking R still toggles every visibly drawn cable incident to R.

Only the central 3×3 switch plate is clickable; the surrounding orange source pad is not.

Terminal bulb silhouette

A bulb centered at (cx, cy) is a five-cell diamond:

. X .
X X X
. X .

Off:

Four outer cells: 13

Center: 5

Lit:

Four outer cells: 11

Center: 0

Bulbs are never clickable. Every bulb has exactly one incident cable and therefore never conducts power onward.

Cable routes

Every listed cable follows a straight, deterministic 8-connected Bresenham centerline between its endpoint centers. Cells inside either endpoint’s module footprint are removed from the route.

The implementation must use the same tie-break throughout: when the Bresenham error is exactly tied, advance in x before y.

The authored coordinates guarantee:

At least three route cells between endpoint footprints

No cable crossings away from a switch

No route passing through an unrelated module

Cables are drawn before modules, so modules cover their connection endpoints.

Closed cable

Every centerline cell is rendered.

If the cable belongs to the source-connected component: color 11

Otherwise: color 2

A closed unpowered cable remains visibly solid and dark. Its state is never hidden merely because it lacks electricity.

Open cable

A central contiguous region is omitted:

gap_length = min(3, route_length - 2)

This always leaves at least one visible stub cell at each endpoint.

The endpoint-side stubs are colored independently:

Stub touching a powered endpoint: 11

Stub touching an unpowered endpoint: 2

Missing central cells: background 4

Thus an open cable can visibly have one bright side and one dark side.

3. Exact state machine
Persistent state
edge_closed[e] : bool for every authored cable
done           : bool
step_count     : nonnegative integer

The following values are derived and need not be stored:

powered_vertices
powered_switches
bulb_lit
energized_cables

There are no hidden phases, random values, timers, queues, or delayed triggers.

Graph interpretation

Vertices consist of:

The root switch

Ordinary switches

Terminal bulbs

The root switch is the sole source seed.

A vertex is powered exactly when it belongs to the connected component of R using closed cables.

powered_vertices =
    connected_component(
        source = R,
        edges = {e | edge_closed[e] == true}
    )

A bulb is lit exactly when its bulb vertex is in powered_vertices.

Accepted switch click

Click eligibility is evaluated from the pre-click powered component.

If switch s is powered:

for every cable e incident to s:
    edge_closed[e] = not edge_closed[e]

All incident cables toggle atomically. Power is not recomputed between individual cable changes.

A cable between two switches can therefore be toggled from either endpoint. A switch-to-bulb cable can only be toggled from its switch endpoint.

After all bits are flipped:

Recompute the source-connected component.

Update switch and cable highlighting.

Update bulb states.

Evaluate success.

A switch that disconnects itself still completes its entire click because eligibility was established from the pre-click state.

Reference pseudocode
Python
Run
def step(action):
    if state.done:
        return render_settled(state), True

    state.step_count += 1

    if action.kind != "CLICK":
        return render_settled(state), False

    switch = switch_hit_test(action.x, action.y)
    pre_power = compute_powered_component(state.edge_closed)

    if switch is None or switch not in pre_power:
        play_invalid_feedback(action.x, action.y)
        return render_settled(state), False

    old_edges = state.edge_closed.copy()

    for edge in incident_edges[switch]:
        state.edge_closed[edge] ^= True

    post_power = compute_powered_component(state.edge_closed)
    play_toggle_animation(
        clicked_switch=switch,
        old_edges=old_edges,
        new_edges=state.edge_closed,
        final_power=post_power,
    )

    state.done = all(
        bulb in post_power
        for bulb in level_terminal_bulbs
    )

    if state.done:
        play_completion_animation()

    return render_settled(state), state.done
Feedback and animation
Accepted click

Each accepted click uses a fixed five-frame presentation animation. These frames do not call step(), do not consume actions, and cannot accept input.

A 5×5 outline of color 10 appears around the clicked switch.

Every incident cable begins retracting or extending simultaneously.

All incident cables are shown halfway between their old and new geometries, using structural color 2.

Final open/closed geometries appear, still using structural color 2.

The new source-connected component appears in color 11; disconnected parts remain dark.

The logical transition is atomic. No intermediate animation frame is a valid electrical state. A benchmark implementation that exposes only one observation returns frame 5.

Invalid click

An unpowered switch receives a three-frame dull crossed ripple in color 2. Clicking another invalid region places the same ripple around the clicked cell.

No persistent state changes.

Completion

When all bulbs are lit, each receives a 5×5 outline pulse in color 14 for six frames. The circuit itself remains in its final logical state. done is already true during this presentation.

4. Progression overview
Level	Main discovery or composition	Verified minimum accepted clicks
1. First Contact	Clicking a powered switch toggles its cable; a connected bulb lights	1
2. Cut Your Own Lead	Closing the bulb cable is insufficient if the switch disconnects; repair from the opposite endpoint	2
3. Hand-Off	One feed can open while another closes, preserving access	3
4. Branch Repair	Opposite-end repairs propagate as a coupled wave through an irregular tree	26
5. Alternating Clutch	A hub can be operated repeatedly by swapping between two live feeds; shallow work must be delayed	35
6. Offset Bridge	A cross-rail cable resolves a genuine parity conflict between two otherwise incompatible repair waves	45
7. Finished Side	Two bridges have different roles; the shortest solution deliberately abandons a completed upstream section	53

The minimum depths were verified by exhaustive breadth-first search over the directed edge-bit state graph, including pre-click power eligibility.

Invalid clicks and unused action types are not included in these depths.

5. Authored levels
Level 1 — First Contact
Player-facing discovery

The board contains one glowing source-mounted switch, one dark bulb, and one visibly open cable between them. There are no unrelated objects.

Directional inputs and SPACE do nothing. The root is the only visibly live control and the only productive click experiment. Clicking it visibly closes the gap and lights the bulb.

This establishes:

Powered switches are the click targets.

A click changes a visibly incident cable.

A bulb glows while connected.

The objective is to illuminate the bulbs.

No separate reference panel is needed because the physical cause and effect are immediate and unambiguous.

State-machine detail
Modules
ID	Kind	Center
R	Source-mounted switch	(18, 32)
G	Terminal bulb	(46, 32)
Cables
Cable	Initial state
R–G	Open

No other cable exists.

Initial derived state
powered switches = {R}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution

Click the center of R:

R

After the click:

R–G is closed and energized.

G is lit.

Success occurs on action 1.

Attribution and necessity check

The only changed relation is the one cable touching the clicked switch. The bulb lights in the same settled frame. There is no alternate explanation involving another device or simultaneous rule.

The level is not solvable without clicking R, because no other action changes R–G.

Level 2 — Cut Your Own Lead
Player-facing discovery

A powered ordinary switch A sits between the source-mounted root and a dark bulb.

The source cable is closed. The bulb cable is open. Clicking the switch next to the bulb appears to be the obvious continuation of Level 1.

That click produces two simultaneous local changes:

A–G closes.

R–A opens.

The bulb remains dark because the newly closed bulb cable belongs to a disconnected dark component. A also darkens and rejects a second click. The still-glowing root sits at the opposite end of the open shared cable, making the recovery experiment visually obvious.

Clicking R closes R–A without disturbing A–G, restoring power to both the switch and bulb.

This refutes two plausible mistaken rules:

“A closed bulb cable lights the bulb regardless of source connectivity.”

“A switch remains clickable after it has been used.”

State-machine detail
Modules
ID	Kind	Center
R	Source-mounted switch	(12, 32)
A	Switch	(34, 32)
G	Terminal bulb	(54, 32)
Cables
Cable	Initial state
R–A	Closed
A–G	Open
Initial derived state
powered switches = {R, A}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution
A → R
Action 1: A
R–A : closed → open
A–G : open   → closed

Post-click:

powered switches = {R}
bulb G            = off

The dark A–G cable remains visibly solid.

Action 2: R
R–A : open → closed

Power crosses the repaired shared cable, reaches A, and continues through the already closed A–G.

Success occurs on action 2.

Attribution and necessity check

A must be clicked because it is the only endpoint capable of closing the bulb cable. That same required click necessarily opens its sole feed. R must then be clicked because it is the only powered endpoint of the open R–A cable.

An exploratory click on R before A is recoverable: a second click on R restores the initial state.

Level 3 — Hand-Off
Player-facing discovery

Level 2 established that opening a switch’s last feed darkens it. This level presents a controlled contrast.

Hub H is currently fed through U–H. A second cable V–H is visibly open, but V itself is already powered by the source. Downstream switch Q and its bulb are dark.

Clicking H simultaneously:

Opens U–H

Closes V–H

Closes H–Q

Unlike the switch in Level 2, H stays bright because the alternate feed becomes live at the same moment. The whole downstream switch Q becomes powered.

Clicking Q then closes its bulb cable while opening H–Q, stranding Q. H remains usable and can restore H–Q; that same click exchanges the hub feed back from V to U.

The level forces the player to observe the distinguishing property: the clicked switch survives because another already-powered route closes simultaneously.

State-machine detail
Modules
ID	Kind	Center
R	Source-mounted switch	(8, 32)
U	Switch	(22, 16)
V	Switch	(22, 48)
H	Switch	(38, 32)
Q	Switch	(50, 32)
G	Terminal bulb	(60, 32)

All cables use straight routes. The four cables R–U, R–V, U–H, and V–H form a visible diamond without crossings.

Cables
Cable	Initial state
R–U	Closed
R–V	Closed
U–H	Closed
V–H	Open
H–Q	Open
Q–G	Open
Initial derived state
powered switches = {R, U, V, H}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution
H → Q → H
After first H
U–H = open
V–H = closed
H–Q = closed

Powered switches:

{R, U, V, H, Q}
After Q
H–Q = open
Q–G = closed

Q and G are dark.

After second H
U–H = closed
V–H = open
H–Q = closed
Q–G = closed

The original hub feed is restored, and power reaches Q and G.

Success occurs on action 3.

Attribution and necessity check

Q cannot be clicked initially, so the first H click is required. The Q click necessarily reopens H–Q, requiring the second H click.

Both feed sides are structurally necessary to the three-click explanation. Without V–H, the first H click would strand H and make the final repair impossible from H.

Level 4 — Branch Repair
Player-facing discovery

Only R begins powered. Every visible cable is open.

Clicking R powers junction J. Clicking J closes both of its outward branches but reopens R–J, leaving the new branch switches in a dark island. The bright root remains beside the newly opened cable, so the Level 2 repair action is immediately suggested.

Repeating this behavior reveals a stronger consequence: opposite-end repair is not merely an undo. It propagates a usable frontier through the tree. At a branch point, repairing an ancestor powers several downstream controls simultaneously, but later clicks at that ancestor can also reopen paths that appeared complete.

The tree is deliberately irregular:

Two bulbs sit at depth 3.

One bulb sits on another depth-3 branch.

One bulb lies at depth 5.

The long branch is not spatially ordered after the shallow branches.

The intended trace works on part of the deep branch before finishing the visually nearer leaves.

No new object type is introduced. The challenge comes entirely from authored graph structure.

State-machine detail
Modules
ID	Kind	Center
R	Source-mounted switch	(5, 34)
J	Switch	(15, 34)
K	Switch	(25, 44)
U	Switch	(25, 20)
M	Switch	(35, 44)
P	Switch	(37, 12)
N	Switch	(44, 44)
Q	Switch	(37, 28)
T	Switch	(54, 44)
S	Switch	(35, 56)
GP	Terminal bulb	(50, 12)
GQ	Terminal bulb	(50, 28)
GT	Terminal bulb	(61, 44)
GS	Terminal bulb	(49, 56)
Switch graph
R–J
J–K
J–U
K–M
U–P
M–N
U–Q
N–T
K–S
Bulb cables
P–GP
Q–GQ
T–GT
S–GS

Every listed cable is initially open.

Initial derived state
powered switches = {R}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution
R → J → R
K → J → R
M → K → J → R
U → N → M → K → J → R
P → Q → U
T → N → M
S → K → J → R

Flattened replay sequence:

R, J, R, K, J, R, M, K, J, R,
U, N, M, K, J, R,
P, Q, U,
T, N, M,
S, K, J, R

Accepted-click depth: 26.

The final R click closes the remaining source-side repair. Every switch edge and every bulb cable is then closed, so all four bulbs light simultaneously.

Important visible phases

After J is first clicked, K and U become connected to each other through J but are dark because R–J opened. This visibly demonstrates that a solid local network can still be electrically isolated.

Later, clicks at M, N, and T temporarily advance the deep branch while repeatedly reopening ancestor edges. The short bulbs cannot simply be completed and forgotten because shared ancestors continue to change.

Necessity and anti-cleanliness check

This switch graph is a tree, and every leaf switch has a bulb. Every switch edge lies on the unique source route to at least one bulb, so no branch is optional.

The solution is not a set of independent bulb errands:

J jointly controls both major branches.

K jointly controls the long branch and S.

U jointly controls P and Q.

Repairs must revisit shared ancestors.

A completed bulb route can become dark again when an ancestor is used for another branch.

Exhaustive search confirms no accepted-click solution shorter than 26.

Level 5 — Alternating Clutch
Player-facing discovery

The source side forms the familiar U–H–V hand-off diamond from Level 3, but H now controls an irregular downstream tree.

Initially:

H is powered through U–H.

V–H is open.

Both outward tree cables from H are open.

A short bulb branch through N9 is visibly close to H.

A deeper branch through N1 contains two nested forks.

The first H click closes both outward cables while exchanging its feed from U to V. Later H clicks exchange the feed back and forth. This lets H act as a repeatedly usable clutch rather than requiring a repair wave all the way to R.

The shallow branch through N9 is intentionally not completed when first reachable. Further hub clicks would repeatedly disturb its source connection. It must be delayed until the deep branch has been staged into the correct phase.

This gives established mechanics contrasting roles:

The alternate feed is a rescue in Level 3.

Here it is a repeatedly used operating resource.

The short branch looks convenient but is strategically premature.

The deep branch determines when the shallow branch can safely be completed.

State-machine detail
Source and hand-off region
ID	Kind	Center
R	Source-mounted switch	(5, 32)
U	Switch	(15, 15)
V	Switch	(15, 49)
H	Switch	(26, 32)
Downstream switches
ID	Center	ID	Center
N1	(34, 22)	N7	(56, 12)
N2	(42, 22)	N8	(56, 56)
N3	(49, 12)	N9	(34, 46)
N4	(49, 28)	N10	(38, 56)
N5	(49, 38)	N11	(56, 44)
N6	(49, 48)
Bulbs
Bulb	Attached switch	Center
G7	N7	(62, 12)
G8	N8	(62, 56)
G10	N10	(38, 62)
G11	N11	(62, 44)
Source-side cables
Cable	Initial state
R–U	Closed
R–V	Closed
U–H	Closed
V–H	Open
Downstream switch graph
H–N1
N1–N2
N2–N3
N2–N4
N4–N5
N5–N6
N3–N7
N6–N8
H–N9
N9–N10
N6–N11
Bulb cables
N7–G7
N8–G8
N10–G10
N11–G11

Every downstream switch edge and every bulb cable is initially open.

Initial derived state
powered switches = {R, U, V, H}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution
H → N1 → H
N2 → N1 → H
N3 → N4 → N2 → N1 → H
N5 → N4 → N2 → N1 → H
N6 → N5 → N4 → N2 → N1
N9 → H
N7 → N3
N8
N10 → N9
N11 → N6 → N5 → N4 → N2 → N1 → H

Flattened replay sequence:

H, N1, H, N2, N1, H,
N3, N4, N2, N1, H,
N5, N4, N2, N1, H,
N6, N5, N4, N2, N1,
N9, H,
N7, N3,
N8,
N10, N9,
N11, N6, N5, N4, N2, N1, H

Accepted-click depth: 35.

Final state

Every downstream tree edge and every bulb cable is closed.

Source-side state:

R–U = closed
R–V = closed
U–H = open
V–H = closed

All switches are powered, and all four bulbs are lit.

Why the shallow branch is delayed

N9 and N10 are close to H, but the cable H–N9 is toggled on every hub click. Completing that branch immediately does not preserve it through the remaining deep-tree repairs.

The reference path waits until the deep structure has been staged, then uses:

N9 → H

to place the short branch in the phase needed for its later terminal clicks.

Necessity and anti-cleanliness check

All tree leaves have bulbs, so every downstream branch belongs to a unique required source-to-bulb route.

The source diamond is not decorative. H is clicked seven times. Each H click opens its current feed and closes the other, leaving it powered. Without the alternate feed, the displayed 35-click trace becomes illegal immediately after a hub click.

The board is not ordered as a recipe:

The nearby N9 branch is delayed.

The deep route splits at N2 and again at N6.

N3 and N4 are siblings but have very different downstream consequences.

Work returns through N1, N2, N4, and N5 multiple times.

The geometrically nearest goal is not the first goal completed.

Exhaustive search confirms 35 is the minimum accepted-click depth.

Level 6 — Offset Bridge
Player-facing discovery

Two long rails leave the root and terminate in separate bulbs. A single cross-cable connects the fourth switch of each rail.

Nearly every cable is open, but A4–A5 begins visibly closed and dark in the middle of the upper rail. This isolated closed segment puts the upper rail one toggle phase out of alignment with the lower rail.

The first part resembles two parallel repair waves. Once both rails reach A4 and B4, treating the rails independently stops making coherent progress: the root parity needed to finish one rail conflicts with the root parity needed to finish the other.

The open cross-cable A4–B4 is the only unfamiliar relation near the impasse. Clicking either endpoint immediately closes the rung while changing that rail’s adjacent contacts, visibly transferring access between the rails.

The same rung then plays contrasting roles:

While advancing the upper bulb, it is repeatedly opened and closed as a temporary powered repair route.

After the upper bulb is stable, B4 uses it as the feed from which the lower half is constructed.

In the final state, both sides of an open rail gap are powered through different source routes.

State-machine detail
Coordinates
R  = (3, 32)

A_i = (7 + 7·(i−1), 17), for i = 1..8
B_i = (7 + 7·(i−1), 47), for i = 1..8

GA = (62, 17)
GB = (62, 47)

Thus:

A4 = (28, 17)
B4 = (28, 47)
A8 = (56, 17)
B8 = (56, 47)

The rung A4–B4 is a straight vertical cable.

Switch graph
R–A1
R–B1

A1–A2–A3–A4–A5–A6–A7–A8
B1–B2–B3–B4–B5–B6–B7–B8

A4–B4

The chain notation expands into one cable between each consecutive pair.

Bulb cables
A8–GA
B8–GB
Initial cable states

Only one cable is initially closed:

A4–A5 = closed

Every other listed cable is initially open, including:

R–A1
R–B1
A4–B4
A8–GA
B8–GB
Initial derived state

The closed A4–A5 segment is an isolated dark component.

powered switches = {R}
lit bulbs        = {}
selected_switch  = null
cursor_position  = null
done             = false
step_count       = 0
Reference solution
R → A1 → B1 → R
A2 → A1 → B2 → B1 → R
A3 → A2 → A1 → B3 → B2 → B1 → R

A5 → A4
A6 → A5 → A4
A7 → A6 → A5 → A4
A8 → A7 → A6 → A5 → A4

B4
B5 → B4
B6 → B5 → B4
B7 → B6 → B5 → B4
B8 → B7 → B6 → B5 → B4

Flattened replay sequence:

R, A1, B1, R,
A2, A1, B2, B1, R,
A3, A2, A1, B3, B2, B1, R,
A5, A4,
A6, A5, A4,
A7, A6, A5, A4,
A8, A7, A6, A5, A4,
B4,
B5, B4,
B6, B5, B4,
B7, B6, B5, B4,
B8, B7, B6, B5, B4

Accepted-click depth: 45.

Key checkpoints

After action 16, both rail prefixes through A4 and B4 are powered. The initially closed A4–A5 also makes A5 powered.

A5 is clicked before A4. This advances the upper frontier but cuts A4–A5. Clicking A4 repairs that cut while also changing:

A3–A4
A4–A5
A4–B4

That click visibly closes the rung and powers the next upper segment through a different route.

After the upper bulb is complete, B4 becomes the repeated repair anchor for the lower suffix. Its final click lights GB.

Final state

Every listed cable is closed except:

B3–B4 = open

The rung A4–B4 is closed.

Both sides of B3–B4 are nevertheless powered:

B1–B3 are powered through R–B1.

B4–B8 are powered through the upper rail and the rung.

All switches and both bulbs are powered.

Formal necessity of the rung

Without A4–B4, the two rails impose incompatible final click parity on the shared root.

For the all-open lower rail, satisfying its bulb requires:

root click parity = odd

The initially closed A4–A5 reverses one parity relation on the upper rail, making its bulb require:

root click parity = even

A single root cannot satisfy both. The rung introduces the missing coupling equation and makes simultaneous completion possible.

Removing A4–B4 therefore makes this exact initial state unsolvable, rather than merely making the reference route longer.

Anti-cleanliness check

The rung is neither an optional shortcut nor a decorative loop. The initial offset creates a genuine dependency between the rails.

The second half is not “finish upper, then independently finish lower.” The lower suffix is powered through a route constructed by upper-rail operations, and the final open rail gap is safe only because the rung supplies the far side.

Exhaustive search confirms 45 is the minimum accepted-click depth.

Level 7 — Finished Side
Player-facing discovery

The final board has two nine-switch rails and two cross-cables:

An early rung at A2–B2

A later rung at A5–B5

The source initially feeds A1, while A5–A6 is a separate closed dark segment. The two rungs begin open.

The root’s first click visibly hands source power from the upper rail to the lower rail. The early rung is used to exchange power between the upper prefixes while building far enough to reach the second rung.

The two rungs eventually acquire different meanings:

A2–B2 is a temporary working bridge. It is useful while staging the upper prefixes but must end open.

A5–B5 becomes the permanent final feed for the lower half of the upper rail.

The crucial late insight is to stop repairing A1–A4. Their downstream work has already been transferred across A5–B5. Reconnecting them would require additional ancestor operations that disturb the useful phase of the lower rail.

The shortest completion therefore leaves A1–A4 visibly dark behind two unmistakable gaps while both bulbs remain powered.

State-machine detail
Coordinates
R = (3, 32)

A_i = (7 + 6·(i−1), 17), for i = 1..9
B_i = (7 + 6·(i−1), 47), for i = 1..9

GA = (62, 17)
GB = (62, 47)

Important centers:

A2 = (13, 17)
B2 = (13, 47)

A5 = (31, 17)
B5 = (31, 47)

A9 = (55, 17)
B9 = (55, 47)

Both rungs are straight vertical cables.

Switch graph
R–A1
R–B1

A1–A2–A3–A4–A5–A6–A7–A8–A9
B1–B2–B3–B4–B5–B6–B7–B8–B9

A2–B2
A5–B5
Bulb cables
A9–GA
B9–GB
Initial cable states

Initially closed:

R–A1
A5–A6

Every other listed cable is initially open, including:

R–B1
A2–B2
A5–B5
A9–GA
B9–GB
Initial derived state
powered switches = {R, A1}
lit bulbs        = {}

A5–A6 is solid but dark.

selected_switch = null
cursor_position = null
done            = false
step_count      = 0
Reference solution
R → B1 → R → A1 → B2 → R
B3 → B2 → R → B4 → B3 → B2 → R
B5 → B4 → B3 → B2 → A2 → R → A3 → A2

A6 → A5 → B6 → B5 → R

A7 → A6 → B7 → B6 → B5 → A5 → R

A8 → A7 → A6 → A5
B8 → B7 → B6 → B5 → R

A9 → A8 → A7 → A6
B9 → B8 → B7 → B6 → B5 → A5 → R

Flattened replay sequence:

R, B1, R, A1, B2, R,
B3, B2, R, B4, B3, B2, R,
B5, B4, B3, B2, A2, R, A3, A2,
A6, A5, B6, B5, R,
A7, A6, B7, B6, B5, A5, R,
A8, A7, A6, A5, B8, B7, B6, B5, R,
A9, A8, A7, A6, B9, B8, B7, B6, B5, A5, R

Accepted-click depth: 53.

Role of the first rung

During the opening phase, clicks at B2 alternately close and open A2–B2. This allows the powered prefix to move between the rails without completing a full repair wave through both source leads.

At actions 18–21:

A2 → R → A3 → A2

the first rung is used to stage the upper prefix and is then opened. It remains open in the final state.

Insisting on retaining this early bridge produces an incorrect phase at the later rung.

Role of the second rung

B5 first closes A5–B5, allowing the isolated closed segment A5–A6 to become useful.

From that point onward, A5 and B5 alternate as repair endpoints. The second rung repeatedly changes state while longer suffixes are staged.

The final pair:

A5 → R

leaves A5–B5 closed and changes the source feed to the lower rail. Power then travels:

R
→ B1 → B2 → B3 → B4 → B5
→ A5 → A6 → A7 → A8 → A9 → GA

while the lower rail continues directly to GB.

Exact final state

Closed cables:

R–B1

A1–A2
A2–A3
A3–A4

B1–B2
B2–B3
B3–B4
B4–B5

A5–A6
A6–A7
A7–A8
A8–A9

B5–B6
B6–B7
B7–B8
B8–B9

A5–B5

A9–GA
B9–GB

Open cables:

R–A1
A4–A5
A2–B2

Powered switches:

{R,
 B1, B2, B3, B4, B5, B6, B7, B8, B9,
 A5, A6, A7, A8, A9}

Dark switches:

{A1, A2, A3, A4}

Both bulbs are lit.

The dark upper prefix remains visibly coherent: A1–A4 contains solid closed internal cables, but the section is isolated by the broad gaps at R–A1 and A4–A5.

Necessity and anti-degenerate checks

The two rungs are not equivalent decorations.

For this exact initial state:

With A2–B2 removed, the minimum solution increases from 53 to 80 accepted clicks.

With A5–B5 removed, the minimum solution increases from 53 to 59 accepted clicks.

Reaching a goal in which every switch is powered requires at least 57 accepted clicks.

Therefore every 53-click optimum must accept some stranded controls. Deliberate abandonment is not cosmetic in the intended solution; it is what avoids four or more additional repair clicks.

The board forms a dependency network rather than a chain:

Source parity moves between the rails.

The first rung influences access to the second.

The second rung feeds a suffix whose original prefix is later discarded.

The initial closed A5–A6 segment is spatially separated from the source and matters only after a later route reaches it.

A short-term restoration of the upper prefix is strategically harmful late in the solution.

The two bulbs are not completed by independent routes.

Exhaustive search over all 2^22 = 4,194,304 cable states confirms 53 as the minimum accepted-click depth.

6. Reference validation table

The implementation should replay center-cell clicks for each trace and assert that success first occurs on the final listed action.

Level	Cable count	Initial closed cables	First success	Final open cables
1	1	None	1	None
2	2	R–A	2	None
3	6	R–U, R–V, U–H	3	V–H
4	13	None	26	None
5	19	R–U, R–V, U–H	35	U–H
6	19	A4–A5	45	B3–B4
7	22	R–A1, A5–A6	53	R–A1, A4–A5, A2–B2

Additional validator assertions:

Eligibility is based on the pre-click component.

Every incident cable, including bulb cables, flips exactly once on an accepted click.

The source mounting under R never flips.

Power is recomputed once after the complete incident-edge XOR.

A rejected click changes no edge bit.

Closed dark cables retain their continuous geometry.

Open cables retain endpoint stubs even when both endpoints are dark.

No authored cable route crosses another route away from a switch.

No level begins in a success state.

No reference trace reaches success before its final action.

Success ignores the powered or unpowered status of non-bulb switches.

The game contains exactly seven levels.
