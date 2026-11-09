Lockstep Doors

Exact authored level count: 7

“This is a game about rotating a matching-colored tile becoming difficult because each rotation replaces the routes that make later rotations legal.”

Lockstep Doors is a click-only network puzzle. A fixed source sends power through a visible directed cable network. Each rotary tile shows its current color, its cyclic sequence of future colors, and the alternate cable that belongs to every state. A tile may relay power whenever it is connected, but it may rotate only when at least one glowing incoming cable matches its current face.

The puzzle is therefore not about discovering hidden connectivity. Every possible cable is physically visible. The challenge is preserving, abandoning, and rebuilding the particular routes that permit future rotations.

There are only four gameplay object categories:

One fixed power source.

Colored rotary tiles.

Directed colored cables and their recessed alternate channels.

One locked exit door.

No switches, keys, counters, inventories, timers, movable avatars, or hidden predicates are added.

1. Action space

The environment accepts exactly:

UP

DOWN

LEFT

RIGHT

SPACE

CLICK(x, y) for any cell on the 64×64 canvas

Only clicking is used.

UP, DOWN, LEFT, RIGHT, and SPACE are deterministic no-ops. They do not change selection, power, animation, or state.

A click is resolved as follows:

Clicking any cell in a rotary tile’s module-sized hit region attempts to rotate that tile.

Clicking any cell in the exit door’s hit region attempts to open it.

Clicking the source, a cable, a recessed channel, or background does nothing.

There is no persistent cursor, selected tile, control mode, or hover-dependent state.

The documentation letters A–H and X below are never rendered in the grid.

2. Native geometry and rendering
Canvas and coordinates

The observation is a 64×64 array of integer color IDs. The origin is the upper-left corner. Coordinates are (x, y).

The gameplay is a schematic graph embedded in pixel space, not a tile grid. No grid lines or internal logic-grid borders are drawn.

Across all levels:

Source center: (3, 32)

Exit door hitbox: x=57..63, y=27..37

Recommended exit click: (60, 32)

Background: color 5

Rotary centers vary by level and are listed in the level tables.

Logical colors

Only these four colors have state-machine meaning:

ID	Meaning
6	magenta
9	blue
11	yellow
14	green

The remaining rendering colors are used consistently:

ID	Rendering role
0	white ready-collar, power highlight, open-door interior
1	powered tile shell
2	source frame, structural bridge at cable crossings
3	unpowered tile shell, closed door panels
4	recessed channel
5	background
8	rejected-click and terminal-brownout feedback

Colors 7, 10, 12, 13, and 15 are unused.

Rotary tile silhouette

Each rotary is a connected approximately 7×7 module:

A 5×5 shell.

A central 3×3 selected face.

Two, three, or four attached colored state petals.

A state selector connecting the central face to the currently selected petal.

Conceptual sketch, not a rendered legend:

             state 0 petal
                   │
     state 3 ── current face ── state 1
                   │
             state 2 petal

For four-state rotors, state order is clockwise: north, east, south, west. Three-state rotors use north, east, and south. Two-state rotors use opposite petals. The exact order is always visible from the petal arrangement and selector.

Every state petal has:

A logical color.

A recessed outgoing channel leading continuously to its target.

A target-end arrowhead showing cable direction.

The central 3×3 face always has the color of the selected state.

Click hit region

A click on the 5×5 body or any attached state petal targets the rotor. The player never has to hit an individual petal, socket, or cable pixel.

Powered versus ready

These are deliberately separate visible properties:

Powered tile: shell is color 1.

Unpowered tile: shell is color 3.

Ready tile: at least one matching glowing input terminates in a white color-0 collar, and the shell corners gain a white highlight.

Powered but mismatched tile: shell is light, but there is no white ready-collar.

Thus a tile can be visibly powered and relaying while still being illegal to rotate.

Cable rendering

Every possible state cable is visible from the first frame.

A cable occupies a three-cell-wide recessed trough:

Inactive alternate channel: color 4.

Selected but unpowered cable: a one-cell centerline in its logical color, with color 4 on either side.

Selected and powered cable: the full trough fills with the logical color.

Direction is shown by a broad arrow-shaped termination at the receiving module.

When multiple state channels connect the same ordered pair of modules, they occupy parallel lanes. They never collapse into an ambiguous single line.

Where two cable routes cross, one receives a small color-2 bridge and the lower cable has a black interruption. Crossings are never graph junctions.

The target tables below are mechanically authoritative. Exact curvature between the given module centers is presentational: a conforming renderer may use straight or orthogonal curves provided that every channel remains continuous, directionally unambiguous, and does not touch an unintended input socket. Upper-branch channels should stay primarily above the horizontal centerline, lower-branch channels below it, and central-return cables near y=32.

Source

The source is a connected gray-and-white block occupying approximately x=0..6, y=28..36.

It is permanently powered.

It is not clickable.

It may have multiple fixed colored output sockets, but remains one source object.

Every source cable is continuously selected and powered.

Fixed source cables never retract or change color.

Exit door

The exit is a conspicuous 7×11 door at the right edge.

It contains:

A dark two-panel frame.

A central 3×3 demand face in the required color.

One or more incoming cable sockets on its left edge.

A white collar when a matching powered cable is present.

The entire door is clickable. The player is never asked to click a tiny latch.

The exit does not relay power and does not rotate. It opens only when legally clicked.

3. Exact state machine
Static level data

Each rotary state is a pair:

(face_color, outgoing_target)

The selected outgoing cable’s color is always equal to face_color.

A level contains:

rotor_states[tile][state_index]
source_edges = [(color, target), ...]
exit_demand_color
initial_state_index[tile]

All authored initial rotor indices are 0.

No authored rotor has two identical states. In particular, consecutive states never preserve both color and target. Every successful rotation therefore produces at least one immediately visible change:

The face color changes;

the cable target changes;

or both change.

Several later rotors have consecutive states with the same color but different targets. These are intentional same-color redirects, not invisible counter clicks.

Derived power graph

The source is considered powered.

Power is recomputed to a least fixed point:

Every fixed source edge glows.

A rotary reached by any glowing edge becomes powered, regardless of cable color.

The selected outgoing edge of every powered rotary glows.

Its target rotary, if any, becomes powered.

Repeat until no new rotary becomes powered.

Directed cycles are handled by the same reachability computation. A cycle glows only if some member is reachable from the source.

Color mismatch never blocks power propagation. Color controls permission to rotate, not conductivity.

For a rotary r:

incoming_colors(r) =
    colors of all glowing edges whose target is r

ready(r) =
    powered(r)
    and current_face_color(r) is in incoming_colors(r)

For the exit:

exit_ready =
    exit_demand_color is among the colors
    of glowing edges targeting the exit

If several incoming links glow, matching any one of them is sufficient.

step() resolution order

All gameplay state changes occur only during step(action).

For a click action:

Clear the previous nonterminal rejection marker.

Hit-test the pre-action frame.

Compute readiness from the pre-action state.

Resolve the clicked object:

Legal exit click: set door_open=true.

Legal rotor click: advance only that rotor to (index+1) mod state_count.

Illegal rotor or exit click: leave all mechanical state unchanged and set its rejection marker.

Noninteractive click: leave state unchanged without a misleading rejection outline.

After a legal rotor rotation, recompute the full powered graph and ready set.

If the exit is not ready and no rotary is ready, set terminal_failure=true.

Render the settled post-step observation.

Only the clicked rotary changes state. Power propagation never automatically rotates another tile.

A legal click remains legal even if its own consequences immediately remove the cable that permitted it.

Rejected click feedback

Clicking an unpowered or mismatched rotary, or a non-ready exit:

Does not change its state.

Produces a four-frame one-pixel shake.

Replaces its outer shell with color 8.

Leaves the color-8 outline visible in the settled observation until the next step().

This ensures that an agent receiving only settled frames can still observe rejection.

Terminal brownout

After a successful rotation, a state is terminal failure when:

not exit_ready
and no ready rotor exists

Because rotors change only through legal clicks, such a state has no possible future transition. Detecting it is therefore not an arbitrary punishment or hidden solver judgment.

The terminal frame shows:

Source core changing to color 8.

Previously glowing cables collapsing to their thin unpowered centerlines.

Tile shells returning to color 3.

Levels 1–3 have no reachable brownout under any sequence of legal rotor clicks. Brownouts first become possible in Level 4, after the foundational causal rules have been safely demonstrated.

An external reset(level) is part of the harness, not an in-grid action.

Success, reward, and validation

Success requires a legal click on the exit.

Merely supplying the correct cable does not finish the level.

On legal exit click:

door_open = true
terminated = true
success = true
reward = 1

All other steps return reward 0. There is no step penalty and no penalty for rejected experiments.

Terminal brownout returns:

terminated = true
success = false
reward = 0

Other rotor states, powered links, or ready controls are ignored once the exit has been legally clicked. There are no additional hidden requirements.

The non-gameplay harness truncation limit is 2048 calls to step(), providing extensive exploration relative to the longest 56-action solution. It is not displayed as a timer or budget.

4. Animation contract

Animations clarify cable replacement and power propagation but do not introduce intermediate game states.

Successful rotor rotation

One successful click produces a deterministic six-to-eight-frame presentation:

Old cable glow contracts toward the clicked tile.

Old selected cable retracts into its channel.

Selector moves to the next petal; central face changes.

New cable extends to its target.

Glow changes propagate outward from the source in breadth-first distance order.

Newly ready collars appear; lost collars disappear.

The logical post-state is computed atomically during the action’s step(). Animation frames are visual interpolation between pre-state and post-state, are not independently actionable, and consume no steps.

Exit opening

On a legal exit click:

The two dark panels separate.

A connected white interior appears.

Matching input cable remains glowing for the completion frame.

door_open=true is committed by the click’s step().

Benchmark observation

An interactive renderer may display all animation frames. A benchmark implementation may return only the settled final frame. All necessary state information is present in the settled frame.

5. Curriculum overview
Level	Title	Main discovery or composition	Verified total actions
1	Live Latch	A matching powered cable makes the exit clickable	1
2	One Turn	A legal rotation replaces the active cable; powered and ready are different	2
3	Two Handoffs	A new-colored route can restore access to the same rotary	6
4	Borrowed Return	Same-color redirects and configuring a return before disconnecting it	28
5	Leave by Another Side	Deliberately abandoning a hub and rebuilding access through another branch	42
6	False Finish	The door-side rotor is first used as a return control, not as the final move	46
7	Full Lockstep	Two interdependent lobes, a crossbar, and a reciprocal final color handoff	56

Action counts include the final exit click and exclude optional rejected experiments.

6. Authored levels

In all tables, a state entry such as 14→D means:

face color 14
selected outgoing cable color 14
outgoing target tile D

X means the exit door.

All tile letters are documentation-only.

Level 1 — Live Latch
Player-facing discovery

The first frame contains only the source, one bright yellow cable, and the exit.

The source’s yellow cable enters a yellow demand face. The exit has a white ready-collar. Clicking the large door immediately separates its panels.

This is positive goal evidence rather than a symbolic instruction: matching power visibly unlocks a physical door.

There is no unrelated object, alternate route, failure state, or misleading affordance.

Exact instance

Source center: (3,32)

Exit hitbox: x=57..63, y=27..37

Fixed source edge: 11→X

Exit demand: 11

Rotary tiles: none

door_open=false

terminal_failure=false

Persistent selection: none

Rejection marker: none

Verified solution
X

Recommended concrete action:

CLICK(60, 32)

Verified depth: 1 action.

Attribution check

The only interactive shape is the door. The action produces one immediate physical consequence: the door opens. No other component changes simultaneously.

Level 2 — One Turn
Player-facing discovery

A single rotary A sits between a blue source and a yellow exit.

Initially:

A receives a glowing blue cable and has a blue face, so it is ready.

A sends a glowing blue cable to the exit.

The exit is powered but has a yellow demand face, so it has no white ready-collar.

Clicking the conspicuous exit at this point gives a harmless red rejection outline. This distinguishes “powered” from “legally clickable.”

Clicking A then causes one clean transition:

Its blue cable retracts.

Its face becomes yellow.

Its alternate yellow cable extends to the exit.

A remains powered by the blue source but loses its ready-collar.

Its yellow outgoing cable nevertheless remains brightly powered.

The exit gains its white ready-collar.

This isolates the rotary transition without any second control or route-order problem.

Exact instance

Source edge: 9→A

Exit demand: 11

Initial vector: (0)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1
A	(32,32)	9→X	11→X

The two state channels occupy separate parallel lanes between A and the exit.

One verified shortest solution
A X

Concrete center clicks:

CLICK(32, 32)
CLICK(60, 32)

Verified depth: 2 actions.

An exploratory player may first click X, observe rejection, and then perform the two successful actions.

Necessity and attribution

Without rotating A, the only exit cable is blue and cannot open the yellow exit. The level cannot be solved by the mistaken rules “any power opens the exit” or “a powered tile may be clicked repeatedly.”

Level 3 — Two Handoffs
Player-facing discovery

Tile A has four faces and always points to the exit. It must advance:

blue → yellow → green → magenta

The source supplies blue to A, so A can make its first rotation. After that rotation, A is still powered but its yellow face no longer matches its blue source cable.

Two nearby helpers provide the missing future approaches:

Upper helper B can change from blue to yellow and feed A.

Lower helper C can change from blue to green and feed A.

The first local impasse is deliberately clean:

Rotate A to yellow.

Repeatedly clicking A now rejects.

B remains visibly blue-ready.

Rotating B immediately lights a yellow cable into A.

A regains its white ready-collar.

The same rule is then reused with a different color and spatial approach through C.

Both helpers are genuine future controls, not decoys. Clicking the green helper early is harmless but does not solve the immediate yellow mismatch.

Exact instance

Source edges: 9→A, 9→B, 9→C

Exit demand: 6

Initial vector: (0,0,0) in order (A,B,C)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1	State 2	State 3
A	(40,32)	9→X	11→X	14→X	6→X
B	(22,16)	9→A	11→A	—	—
C	(22,48)	9→A	14→A	—	—

All three source cables are fixed outputs from the same source object.

One verified shortest solution
A B A C A X

Verified depth: 6 actions.

Final pre-exit state vector:

(3,1,1)

At that point A is powered, has a magenta face, and sends a glowing magenta cable to the magenta exit.

Necessity and attribution

Forbidding clicks on A, B, or C makes success unreachable.

B is the only possible yellow approach to A.

C is the only possible green approach to A.

There is no reachable terminal brownout under any legal click sequence.

The immediate change after each helper click is a newly colored physical cable entering A; no unrelated state changes.

Level 4 — Borrowed Return
Player-facing discovery

This is the first composed network and the first level that permits terminal brownout.

The initial green route forms a powered return loop among A, D, and B. Several controls may look usable, but their outgoing links are also the links maintaining access to one another.

The opening three actions demonstrate the new interpretation of an established rule:

Rotating A changes it from green to yellow while still pointing to D.

Only B can now restore a yellow input to A.

Rotating A again keeps its face yellow but moves its cable from D to C.

The third action is a same-color redirect. The face remains yellow, but the old cable visibly retracts and a different yellow cable extends. Because the current feed is still yellow, the tile may remain usable. Nothing about the legality rule changes.

The strategic pressure is to use a return cable before rotating the tile that supplies it. A tempting early redirect can leave a tile illuminated but mismatched, with no route capable of restoring its needed color.

Exact instance

Source edge: 14→A

Exit demand: 11

Initial vector: (0,0,0,0,0) in order (A,B,C,D,E)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1	State 2	State 3
A	(11,32)	14→D	11→D	11→C	14→B
B	(23,16)	14→A	11→A	9→D	9→E
C	(23,48)	11→D	14→E	11→A	11→E
D	(36,32)	14→B	14→C	11→A	9→E
E	(50,32)	11→D	9→C	11→B	11→X

Upper routes involving B stay above the centerline. Routes involving C use the lower half. D is the central crossing relay and E is the door-side rotor.

One verified shortest solution
A B A C A A D C D A A C D E C E C A A B D D A C A C E X

Verified depth: 28 actions.

Implementation checkpoints:

after  3 actions: (2,1,0,0,0)
after 17 actions: (2,1,1,3,2)
after 27 actions: (2,2,3,1,3)

The final vector selects E state 3, producing a powered yellow cable to the yellow exit.

Player-facing phase outline

Borrow the first return.
A changes to yellow and loses its source match. B must first send yellow back to it.

Redirect without changing color.
A then moves the yellow route from the center to the lower branch. This is the forced same-color redirect tutorial.

Build the lower-center loop.
C and D alternately return yellow and green power. Rotating either at the wrong moment can leave the other on a color for which no powered approach remains.

Stage the door rotor.
E is powered well before completion, but its early states send power back into the network. It must be advanced while the appropriate return routes still exist.

Necessity and anti-degeneracy

Every one of the five rotors must be clicked in every successful plan; forbidding any one makes the level unsolvable.

Forbidding all same-color redirects makes the level unsolvable.

No two states of a rotor have the same (color,target) pair.

The exit-side tile is encountered before it can finish the level and later functions as a return relay.

The level cannot be reduced to a list of colors: which tile receives each cable determines whether the control network remains actionable.

Level 5 — Leave by Another Side
Player-facing discovery

The network now has two side branches, a central relay, an east relay, and a door rotor.

The source supplies green to A, but A initially has a blue face. It is actionable only because the initial blue route forms a visible loop:

A → C → A

This makes the first frame a mature application of the already learned distinction:

A is source-powered.

Its source cable mismatches.

A returning blue cable from C makes it ready.

After its first rotation, A receives matching green directly from the source. It then has several green states pointing to different branches. Those same-color redirects let it deliberately leave one side of the graph, but eventually it wraps back to blue and must recover the lower return route from C.

A branch that is temporarily dark is therefore not necessarily lost. The player can re-enter it later through another color and another side.

The door rotor F is first useful as a relay back toward D and E; treating its first powered appearance as a finish strands the controls needed to reach its final yellow state.

Exact instance

Source edge: 14→A

Exit demand: 11

Initial vector: (0,0,0,0,0,0) in order (A,B,C,D,E,F)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1	State 2	State 3
A	(11,32)	9→C	14→B	14→C	14→D
B	(22,15)	11→A	9→A	14→A	11→D
C	(22,49)	9→A	9→E	14→E	9→D
D	(34,32)	11→E	14→C	9→C	9→F
E	(45,32)	11→B	11→F	14→B	9→B
F	(52,32)	9→D	11→D	9→E	11→X

The long B↔E routes use the upper corridor. C↔E uses the lower corridor. The D↔F cable bends around E rather than passing through it.

One verified shortest solution
A A A B E A C C E A A E B C A A C A B A A D D A A C A E C C D F D A E F E D D D F X

Verified depth: 42 actions.

Implementation checkpoints:

after 13 actions: (2,2,2,0,3,0)
after 31 actions: (2,3,3,3,0,0)
after 41 actions: (3,3,3,3,2,3)

The final state selects F state 3, supplying yellow to the yellow exit.

Player-facing phase outline

Move A across three green routes.
After obtaining source-green access, A successively points to B, C, and D. Its face remains green while the illuminated region changes.

Recover blue from below.
When A wraps to blue, the source no longer permits a click. C must be brought back to a blue return state before A can continue.

Cross-couple the branches.
B feeds D; D feeds C or E; E alternates between the upper branch and the door rotor. These dependencies form a network rather than an ordered sequence of stations.

Use F as a relay before using it as the exit driver.
Its first three states send power back into the graph. Only the fourth state points to X.

Necessity and anti-cleanliness

All six rotors must be clicked; forbidding any one leaves no solution.

Same-color redirects are necessary.

Every successful rotor action changes color, target, or both.

A, C, D, and E are revisited in different power contexts rather than encountered once in use order.

Several legal-looking early choices remove the only future return of a required color.

The lower and upper routes are not interchangeable because they enter different faces at different phases.

Level 6 — False Finish
Player-facing discovery

The door-side rotor G is now part of the control network itself.

Its states are:

blue back to E
magenta back to E
magenta back to F
green to the exit

The first time G becomes actionable, clicking it does not expose the exit. It sends power backward into the network and leaves itself on a color that must later be supplied by a different route.

This turns the familiar door-side object into a temporary return tool. The player must preserve enough of the west and central network to revisit G twice more.

The opening also reinforces downstream staging. Initially A and C are both ready, but rotating A first makes eventual success unreachable. C must first change which east-side control it powers, after which A can safely leave its blue state.

Exact instance

Source edge: 9→A

Exit demand: 14

Initial vector: (0,0,0,0,0,0,0) in order (A,B,C,D,E,F,G)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1	State 2	State 3
A	(11,32)	9→C	6→B	9→D	11→C
B	(21,15)	6→E	6→A	11→D	6→D
C	(21,49)	9→E	11→E	11→A	14→E
D	(32,32)	6→E	11→A	11→E	11→F
E	(42,16)	11→B	14→D	6→G	6→F
F	(42,48)	6→E	6→D	11→D	9→G
G	(52,32)	9→E	6→E	6→F	14→X

Upper cables through B and E stay above the center. Lower cables through C and F stay below. D is the central cross-coupler.

One verified shortest solution
C A B B B D B B A A E C C A E A B D D D E F D D F D F D G E B D B B B D B B B A E A E G G X

Verified depth: 46 actions.

Implementation checkpoints:

after 20 actions: (1,2,3,0,2,0,0)
after 29 actions: (1,2,3,0,3,3,1)
after 45 actions: (3,1,3,2,2,3,3)

The first click on G occurs well before completion. The final two clicks on G select its magenta-return and green-exit states in the required contexts.

Player-facing phase outline

Prepare C before moving A.
The initial alternative is visible and consequential: changing A first removes the route needed to establish its later magenta access.

Build the upper-center return.
B is advanced through several incoming contexts. Its magenta states point to different targets, so preserving the matching input does not imply preserving the same powered region.

Stage both east relays.
E and F must be configured before the first G click. Otherwise G sends power back to a branch that cannot return the needed color.

Use G, leave it, and rebuild access.
The first G rotation is a commitment, not a finish. The network must return through E, B, and D before G can be advanced again.

Necessity and anti-cleanliness

Every rotor A–G must be clicked in any successful solution.

Same-color redirects are required.

G changes strategic role: return relay, stranded liability, and final exit producer.

The exit-facing region becomes powered before the correct exit cable exists.

The solution repeatedly crosses between upper and lower branches; it is not a left-to-right chain.

Level 7 — Full Lockstep
Player-facing discovery

The final level uses a complete two-diamond mesh:

West hub A

Upper-left and upper-right rotors B and C

Lower-left and lower-right rotors D and E

Central crossbar F

East hub G

Door rotor H

No new object or operational exception is introduced.

The initial state already presents a meaningful choice:

A is ready from the yellow source.

F is also ready through A.

Clicking A immediately makes the final door configuration unreachable.

Clicking F first preserves the yellow route while moving power into the correct lower lobe.

This is not a hidden trap. A’s outgoing route visibly retracts if clicked, and the return cable that would be lost can be traced directly.

The upper-right rotor C has three consecutive blue faces, each pointing somewhere different. It functions as a route selector only because the matching blue supply has been carefully retained.

The final three rotor actions form a reciprocal handoff:

H rotates from magenta to blue and sends blue back to G.

G rotates to its blue cable into H.

H, now receiving matching blue, rotates to green and exposes the green exit cable.

The exit then still requires its own final click.

Exact instance

Source edge: 11→A

Exit demand: 14

Initial vector: (0,0,0,0,0,0,0,0) in order (A,B,C,D,E,F,G,H)

Selection/cursor: none

door_open=false

terminal_failure=false

Tile	Center	State 0	State 1	State 2	State 3
A	(11,32)	11→F	9→B	9→F	9→D
B	(21,15)	9→C	9→A	6→F	11→F
C	(35,15)	9→B	9→G	9→F	11→F
D	(21,49)	9→F	6→A	6→E	9→A
E	(35,49)	6→G	11→G	6→D	9→D
F	(28,32)	11→E	9→E	6→E	6→D
G	(44,32)	11→E	11→C	6→H	9→H
H	(52,32)	6→G	9→G	14→X	11→G

The intended visual topology is:

             B ───── C
            / \     / \
source ── A ─── F ─── G ── H ── exit
            \ /     \ /
             D ───── E

This diagram documents connectivity only; letters and lines are not rendered as text. Every individual state channel remains separately traceable in the actual frame.

One verified shortest solution
F A C C F C B B E E E D F D D F A A A F A A D D E F E G C F D F G E D A A F A A F A A A E F F A A A F E H G H X

Verified depth: 56 actions.

Implementation checkpoints:

after 16 actions: (1,2,3,3,3,0,0,0)
after 28 actions: (2,2,3,1,1,2,1,0)
after 52 actions: (0,2,0,3,0,1,2,0)
after 55 actions: (0,2,0,3,0,1,3,2)

At the final pre-exit vector, H is in state 2, sending a powered green cable to the green exit.

Player-facing phase outline

Configure the crossbar before moving the west hub.
F must redirect the initial yellow supply. An immediate A rotation discards the only route that can perform this setup.

Stage the upper selector.
C is advanced through distinct blue targets. B then changes the color entering the crossbar, allowing the lower lobe to become useful.

Prepare the lower lobe before redirecting the crossbar.
D and E must be positioned while their magenta routes are still illuminated. The same links that help them also keep A and F accessible.

Transfer control to the east hub.
G is reached through both the upper and lower sides at different times. A cable that enables G in one phase becomes a liability after G changes face.

Return west instead of finishing early.
The first door-side access is used to power another control, after which the solution revisits A, D, E, and F.

Perform the reciprocal H–G–H handoff.
The door rotor supplies the color that prepares its own next incoming route. Only then can it expose green to the exit.

Necessity and anti-cleanliness

Every rotor A–H must be clicked; forbidding any one makes success unreachable.

Same-color redirects are necessary.

Every successful rotation changes the visible face, active cable target, or both.

The central crossbar is a tool early, a route bottleneck in the middle, and a liability when rotated at the wrong time.

H is not a one-use key. It first redirects power backward and only later produces the exit cable.

Upper and lower lobes are entered, abandoned, and revisited in different color states.

The graph contains interacting cycles and alternate approaches; replacing the spatial network with a list of target colors would remove the actual puzzle.

7. Epistemic-solvability audit
Foundational attribution
Level 1

The only clickable object is already receiving a matching glowing cable. Its click produces immediate opening feedback.

Level 2

Only one rotary exists. Its click changes exactly its selected face and cable. The initially powered-but-mismatched exit provides a direct contrast without introducing another device.

Level 3

After each target rotation, the next required face matches exactly one helper’s visibly available future cable. Clicking that helper produces an immediate matching input at the stranded target. Legal experimentation cannot brown out the board.

Mature-level evidence

From Level 4 onward:

All possible links remain visibly recessed.

Powered status and ready status are separately rendered.

Return routes can be traced before committing.

Same-color states always point to different targets.

Wrong choices fail by visibly darkening or mismatching the relevant control network.

There are no hidden dependencies, secret click regions, random transitions, or validator-only predicates.

The player may not know which legal click is strategically correct, but they can always see what cable it will replace and which modules that cable currently powers.

Anti-padding and necessity checks

The authored instances satisfy the following structural checks:

Levels 4–7 contain no duplicate (color,target) state within a rotor.

Every successful click changes an immediately visible property.

Forbidding all same-color redirects makes each of Levels 4–7 unsolvable.

Forbidding clicks on any listed rotary makes its level unsolvable.

No level uses a timer, action budget, energy meter, or long traversal sequence.

Difficulty comes from route dependency, timing of disconnection, future color access, and forced revisits.

The exit condition is one natural physical event: legally click the matching powered door.

There are exactly 7 levels.
