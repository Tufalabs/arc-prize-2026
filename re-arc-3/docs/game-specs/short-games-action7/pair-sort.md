Pair Sort

Identity: This is a game about swapping neighboring tokens becoming difficult because every local fix displaces another token through a shared network of pair goals.

Pair Sort contains exactly 7 authored levels.

The player sees token slots grouped into heavy two-slot frames. Bright double-arrow connectors lie between some orthogonally adjacent slots. Clicking a connector swaps the two tokens at its endpoints. A frame glows when its two tokens are exactly the same type. The level ends when every frame glows simultaneously.

The rules remain almost unchanged across the full game. Depth comes from the exchange topology:

A matched frame can be opened again.

A token entering a useful slot always ejects another token.

A side branch can be a final destination, temporary reservoir, or three-token sorting buffer.

Two complementary mismatches can be deliberately aligned so one final swap completes both.

There is no timer, energy system, move penalty, locking, token creation, or hidden state. Every swap is reversible, so experimentation remains affordable even in the final level.

1. Global implementation contract
1.1 Canvas and logical geometry

Render onto a 64×64 integer-color canvas.

The native board is a sparse subset of a fixed 6×6 logical slot lattice. Logical coordinates are (r,c), with rows increasing downward and columns increasing rightward, both in 0…5.

A slot at (r,c) occupies the inclusive 5×5 pixel rectangle

x = 4 + 10c ... 8 + 10c
y = 4 + 10r ... 8 + 10r

Unused logical coordinates are not drawn. There is no underlying grid and no internal grid border.

Connector hitboxes

For horizontally adjacent slots (r,c) and (r,c+1), the connector hitbox is

x = 9 + 10c ... 13 + 10c
y = 4 + 10r ... 8 + 10r

For vertically adjacent slots (r,c) and (r+1,c), the connector hitbox is

x = 4 + 10c ... 8 + 10c
y = 9 + 10r ... 13 + 10r

Every connector therefore has a full 5×5 clickable region. Clicking any cell in that region activates the connector; the player never has to hit an individual arrow pixel.

The 5×5 square where four lattice cells would meet is never a connector and cannot overlap a connector hitbox.

1.2 Pair frames

Every rendered slot belongs to exactly one fixed pair frame. The two slots of a pair are always orthogonally adjacent.

To render a pair:

Find the bounding box containing both 5×5 slot pads and the gap between them.

Expand that box by two pixels on every side.

Draw a two-pixel-thick outer outline.

Do not draw a divider between the two partner slots.

Thus the board is not rendered as a bordered logic grid. Only semantically meaningful two-slot frames are outlined.

Frame state:

State	Rendering
Unmatched	Outline in color 2
Newly matched	One brief flash in color 11, then matched rendering
Matched	Outline in color 10, plus a small check-shaped corner mark in color 14
Newly broken	One brief flash in color 8, then unmatched rendering

The check is placed at the outer corner nearest the slot with no incident connector, keeping it away from the exchange network. This placement is cosmetic and has no game-state significance.

A matched frame never locks.

1.3 Slot and connector rendering

Use:

Element	Color ID
Background	4
Slot pad	1
Unmatched frame	2
Marked connector plate	3
Marked connector arrows	11
Matched frame glow	10
Completion check	14
Invalid click pulse	8

A marked connector is drawn over frame outlines so its yellow double-arrow remains unmistakable even between two glowing frames.

An unmarked gap remains background-colored. It has no faint seam, marker, or hover affordance.

1.4 Token types

A token type is the exact combination of a render color and silhouette. There is no text or symbolic label in the rendered board.

Two connected 5×5 silhouettes are used:

DIAMOND          RING

..X..            .XXX.
.XXX.            XX.XX
XXXXX            X...X
.XXX.            XX.XX
..X..            .XXX.

X is filled with the token’s color ID. . shows the slot-pad color 1.

The twelve possible internal token types are:

T0  = color 9,  diamond      T6  = color 9,  ring
T1  = color 11, diamond      T7  = color 11, ring
T2  = color 6,  diamond      T8  = color 6,  ring
T3  = color 14, diamond      T9  = color 14, ring
T4  = color 12, diamond      T10 = color 12, ring
T5  = color 15, diamond      T11 = color 15, ring

Matching compares the complete internal type. For example, T0 and T6 share blue color 9 but do not match because one is a diamond and the other a ring.

Levels 1–3 use only differently colored diamonds. Level 4 introduces the first same-color/different-silhouette comparison through direct visible evidence.

1.5 Controls

Only cell clicks are meaningful.

Valid click

A click is valid only when its pixel lies inside a marked connector’s 5×5 hitbox.

The two endpoint tokens are exchanged atomically.

Invalid click

Clicking any of the following changes no persistent game state:

a token slot;

a frame outline;

an unmarked gap;

empty background;

the central gap between four lattice positions.

The clicked location displays a two-frame red crossed-out pulse in color 8.

Other actions

UP, DOWN, LEFT, RIGHT, and SPACE are accepted by the action interface but are no-ops. They produce a brief red pulse around the outer canvas edge and no board change.

There is no persistent cursor, selected token, selected frame, or selected connector. A connector click executes immediately.

1.6 State representation

A sufficient implementation state is:

token_at: map[slot_coordinate -> token_type]
frames: ordered list[(slot_a, slot_b)]
marked_edges: ordered list[(slot_u, slot_v, hitbox)]
matched: list[bool]
step_count: integer
won: bool

transient_feedback:
    none
    invalid_click(x,y)
    swap_animation(old_u, old_v, edge)
    completion_animation

The pair relation and marked-edge relation are independent. A marked edge always performs the same transposition, regardless of which frames own its endpoints.

For level notation below:

a_i means the endpoint of frame F_i that has no incident marked connector in that authored level.

v_i means the connector-bearing endpoint of F_i.

This is documentation shorthand only. The two slots render identically and have identical general behavior. There is no special “anchor” object or immovable-token rule. The token at a_i remains stationary solely because no authored connector touches that slot.

Every authored level uses:

F_i = {a_i, v_i}
token_at[a_i] = T_i

An initial port vector

p = [p0, p1, ..., pn-1]

means

token_at[v_i] = T_pi

Consequently, every active token type appears exactly twice: once at its quiet frame endpoint and once somewhere in the connector network.

1.7 step() update order

For every action:

Increment step_count.

Clear the previous transient invalid-action pulse.

Decode the action.

If the action is not a cell click, leave persistent state unchanged.

If the clicked cell is not inside a marked connector, leave persistent state unchanged.

Otherwise, identify connector endpoints u and v from the pre-action state.

Atomically exchange token_at[u] and token_at[v].

Recompute every frame flag:

matched[i] = token_at[frames[i].slot_a]
             == token_at[frames[i].slot_b]

Set:

won = all(matched)

Return reward 1 only on the first transition to won=true; otherwise return reward 0.

Terminate the episode after the completion feedback has rendered.

There are no partial rewards for individual matches and no penalties for invalid actions.

The validator does not inspect the route, number of clicks, order of completion, or which token type occupies which frame. It checks only exact simultaneous equality within every frame. In these authored levels, the quiet copies make that predicate equivalent to token_at[v_i] == T_i for all i, but that equivalent vector must not be hard-coded into the general validator.

1.8 Swap animation

A connector moves both tokens one logical slot, so every valid swap receives a short deterministic animation.

The logical swap and new frame flags are computed during the single step() call. The renderer may then emit four interstitial display frames before returning the settled observation:

Both endpoint pads flash color 11.

Both token sprites translate three pixels toward the connector.

They cross on two one-pixel-offset parallel lanes. The token from the lexicographically smaller logical coordinate uses the upper or left lane.

Both sprites settle in their new slots; frame-status transitions appear.

The animation is based on the saved pre-swap state and committed post-swap state. Intermediate sprite positions are not game states, cannot be clicked, and consume no additional steps.

The settled frame always contains the entire state needed to continue.

If the swap produces global completion, all matched outlines pulse between colors 10 and 0 for six display frames. No text is shown.

1.9 Exploration allowance

There is no authored move budget, timer, energy, or loss condition.

A framework requiring an episode-safety cap should use at least 2048 actions per level. That cap is not a puzzle mechanic and should not be rendered as a countdown.

2. Authored levels
Level 1 — Two at Once
Player-facing discovery

The upper frame already contains two identical blue diamonds and therefore glows. This is positive goal evidence: identical tokens inside one frame are desirable.

Below it are two plain frames containing complementary mismatches. There is exactly one bright connector between them. Clicking it visibly exchanges both endpoint tokens, causes both lower frames to glow, and immediately completes the level.

The player does not have to infer the objective from prose, experiment with arbitrary cells, or distinguish multiple simultaneous rules.

State-machine instance
Frames: 3
Types used: T0 ... T2

a = [
  (1,2),
  (2,2),
  (3,2)
]

v = [
  (1,3),
  (2,3),
  (3,3)
]

p = [0, 2, 1]

Thus:

F0 = T0 | T0    initially matched
F1 = T1 | T2
F2 = T2 | T1

Marked connector:

x = (v1, v2)

x is a vertical connector between (2,3) and (3,3).

Selection/cursor state:

none / null

Win validation:

match(F0) AND match(F1) AND match(F2)
Exact reference solution
x

Intended and minimum depth: 1 valid swap.

Attribution and necessity check

The pre-glowing frame explains the objective before the action. The only marked connector is the only productive control. Its activation moves both visible tokens and changes exactly the two previously unmatched frames.

A mistaken “the clicked token moves alone” hypothesis is contradicted by the simultaneous endpoint motion. A mistaken “frames prescribe a fixed color” hypothesis is contradicted because the two lower neutral frames accept different types.

The upper pair is not a decoy: it supplies the initial positive goal example and participates in the all-frames validator.

Level 2 — Borrow the Glow
Player-facing discovery

A central frame begins matched and glowing. Three mismatched frames surround its connector-bearing slot. Every marked connector touches the glowing frame; there is no leaf-to-leaf connector.

Known actions cannot move the three wrong leaf tokens without involving the matched center. Clicking any connector therefore provides a cheap, decisive experiment: the center’s token swaps normally and its glow immediately disappears.

This teaches that a completed pair is not locked. It can act as a shared reservoir and later be rebuilt.

State-machine instance
Frames: 4
Types used: T0 ... T3

a = [
  (1,2),
  (1,1),
  (1,3),
  (3,3)
]

v = [
  (2,2),
  (2,1),
  (2,3),
  (3,2)
]

p = [0, 2, 3, 1]

The connector-bearing slot v0 is the center of a three-leaf star.

Marked connectors:

L = (v0, v1)
R = (v0, v2)
D = (v0, v3)

Initial matched frames:

F0

Selection/cursor state:

none / null
Exact reference solution
L R D L

Port states along the route:

start       [T0, T2, T3, T1]
after L     [T2, T0, T3, T1]
after R     [T3, T0, T2, T1]
after D     [T1, T0, T2, T3]
after L     [T0, T1, T2, T3]

Intended and minimum depth: 4 valid swaps.

Attribution and necessity check

The first swap has one new causal implication and one unambiguous result: a glowing pair can be broken.

The three leaf tokens form a three-cycle. Because leaves have no direct connectors, any solution must use the central port as a buffer and must temporarily destroy F0. Four center-leaf transpositions are necessary to realize the leaf cycle while restoring the central token.

No visible connector is optional.

Level 3 — Round the Supply
Player-facing discovery

Six connector-bearing slots form a visible loop. Each frame’s quiet endpoint shows which exact token belongs at its port, but every port initially contains the token from the opposite side of the loop.

There is no direct “take this token to its frame” move. Each swap advances one token while displacing another around the cycle. The player now has enough evidence to reason about the unwanted token as part of the same move.

No new object or special-case rule appears. This is the first clean composition of the established swap rule.

State-machine instance

Ports are ordered clockwise.

Frames: 6
Types used: T0 ... T5

a = [
  (0,1),
  (0,2),
  (0,3),
  (2,4),
  (3,2),
  (2,0)
]

v = [
  (1,1),
  (1,2),
  (1,3),
  (2,3),
  (2,2),
  (2,1)
]

p = [3, 4, 5, 0, 1, 2]

Marked connectors:

c0 = (v0, v1)
c1 = (v1, v2)
c2 = (v2, v3)
c3 = (v3, v4)
c4 = (v4, v5)
c5 = (v5, v0)

Initial matched frames:

none

Selection/cursor state:

none / null
Exact reference solution
c0 c1 c3 c2 c5 c0 c1 c4 c5

Intended and minimum depth: 9 valid swaps.

The route uses all six connectors. Other symmetric nine-swap solutions exist, which is appropriate here: the cycle is teaching reversible circulation rather than a hidden prescribed direction.

Attribution and necessity check

Every visible arrow belongs to the same already-established interaction class. There is no new causal unknown hidden among the six connectors.

The half-turn permutation cannot be solved by repeatedly chasing only one token: doing so pushes its displaced counterpart away around the other side. The closed topology makes both directions visible and allows the player to compare them.

Exhaustive permutation search verifies that fewer than nine swaps cannot solve this instance.

Level 4 — The Quiet Bay
Player-facing discovery

The exchange network now bends around a side bay. The token required by the bay begins near the far end of the trunk, while the only connector into the bay is attached near the beginning.

Frame F1, located at that junction, begins matched. It must eventually be opened so the bay token can pass through.

This level also sharpens token identity without changing the rule. F6 visibly contains a blue ring at its quiet endpoint and a blue diamond at its port. Despite sharing color 9, the frame is plain rather than glowing. Nearby F1 contains an exact pair and does glow. The initial settled board therefore directly disproves a plausible “same color is enough” hypothesis before the player has to commit to a plan.

State-machine instance

v0…v7 form the bent trunk. v8 is the side bay.

Frames: 9
Types used: T0 ... T8

a = [
  (0,0),
  (0,1),
  (0,2),
  (0,3),
  (2,4),
  (4,3),
  (4,2),
  (4,1),
  (2,0)
]

v = [
  (1,0),
  (1,1),
  (1,2),
  (1,3),
  (2,3),
  (3,3),
  (3,2),
  (3,1),
  (2,1)
]

p = [7, 1, 5, 4, 3, 2, 0, 8, 6]

Marked connectors:

e0 = (v0, v1)
e1 = (v1, v2)
e2 = (v2, v3)
e3 = (v3, v4)
e4 = (v4, v5)
e5 = (v5, v6)
e6 = (v6, v7)

b  = (v1, v8)

Initial matched frames:

F1

Distinctive same-hue mismatch:

F6 quiet token = T6, blue ring
F6 port token  = T0, blue diamond
F6 is unmatched

Selection/cursor state:

none / null
Solving insight

T8, required by bay frame F8, begins at v7. It must travel left along the trunk and then enter through b. This forces traffic through the initially glowing junction F1.

Once T8 is deposited in the bay, the remaining trunk permutation must be unwound without pulling it back out. The final e0 exchange deliberately aligns complementary tokens at v0 and v1, completing both final frames at once.

Exact reference solution
e6 e5 e4 e3 e2 e0 e1 b
e2 e3 e4 e5 e6
e1 e2 e3 e4 e5
e1 e2 e3 e4
e1 e2 e3
e1 e2
e1 e0

Expanded length: 29 valid swaps.

Exhaustive search over the nine-port permutation state verifies that 29 is minimal.

Anti-degenerate check

Deleting any connector partitions this tree. The number of tokens initially on the wrong side of each connector cut is:

e0 e1 e2 e3 e4 e5 e6 b
 1  2  3  4  3  2  1 1

Every connector must therefore be crossed by every solution.

The transport lower bound is 17 swaps, while the true minimum is 29 because tokens competing for the same narrow trunk must pass one another in a particular order. Difficulty comes from authored placement and ordering, not a move budget or decorative density.

Level 5 — Use the Bay Twice
Player-facing discovery

Two frames, F1 and F4, begin matched. The side bay needs T9, which begins at the far end of the trunk.

A greedy interpretation of the bay as “the place where T9 should immediately go” fails: T9 is not initially at its entrance. The bay must first accept a temporary token so that several trunk tokens can change order. Much later, the bay is reopened and receives its final token.

Thus the exact same branch connector has two contrasting roles:

temporary storage that breaks nearby progress;

final destination that can then be preserved.

State-machine instance
Frames: 10
Types used: T0 ... T9

a = [
  (0,0),
  (0,1),
  (0,2),
  (0,3),
  (0,4),
  (2,5),
  (4,4),
  (4,3),
  (4,2),
  (2,0)
]

v = [
  (1,0),
  (1,1),
  (1,2),
  (1,3),
  (1,4),
  (2,4),
  (3,4),
  (3,3),
  (3,2),
  (2,1)
]

p = [8, 1, 6, 5, 4, 3, 2, 0, 9, 7]

Marked connectors:

e0 = (v0, v1)
e1 = (v1, v2)
e2 = (v2, v3)
e3 = (v3, v4)
e4 = (v4, v5)
e5 = (v5, v6)
e6 = (v6, v7)
e7 = (v7, v8)

b  = (v1, v9)

Initial matched frames:

F1, F4

Selection/cursor state:

none / null
Solving insight

The first five trunk swaps bring useful supply toward the junction. The first b click does not complete the bay; it parks T1 there and opens F1.

After the left end is repaired, the second b click places T9 in its final bay and simultaneously restores the junction supply. The remaining permutation is then settled from left to right. The final e7 swap completes F7 and F8 together.

Exact reference solution
e6 e5 e4 e3 e2 b e1 e0
e7 e6 e5 e4 e3 e2 e1 b
e7 e6 e5 e4 e3 e2
e7 e6 e5 e4 e3
e7 e6 e5 e4
e7 e6 e5
e7 e6
e7

Expanded length: 37 valid swaps.

Exact A* search with an admissible token-distance heuristic verifies that 37 is minimal.

Useful checkpoints

After the first branch click, no frame is matched:

[T8,T7,T0,T6,T5,T4,T3,T2,T9,T1]

After the second branch click:

[T0,T1,T8,T7,T6,T5,T4,T3,T2,T9]

At that point F0, F1, F5, and the bay F9 glow. The bay has changed from temporary reservoir to preserved completion without any operational rule changing.

Anti-degenerate check

Connector cut demands are:

e0 e1 e2 e3 e4 e5 e6 e7 b
 1  2  3  4  4  3  2  1 1

Every connector is necessary. The transport lower bound is 21 swaps, and ordering conflicts raise the exact minimum to 37.

The reference route never revisits a complete board state.

Level 6 — Rotate Through the Bay
Player-facing discovery

The trunk is longer, but the meaningful development is not size. The side bay is now used as a true three-token sorting buffer.

F1 starts matched at the bay junction. The bay requires T10, which begins at the far endpoint v9. The far endpoint itself initially contains T10, so it looks locally valuable even though that token ultimately belongs elsewhere.

During the solution, the far-end frame can be completed and preserved while the side bay remains unresolved. Later, two branch clicks separated by one trunk swap rotate three tokens through the junction:

branch → trunk edge → branch

The bay first receives a temporary token and then its actual partner. This is the same transposition rule, used as a local permutation primitive.

State-machine instance

v0…v9 form a C-shaped trunk. v10 is the inward bay.

Frames: 11
Types used: T0 ... T10

a = [
  (2,1),
  (0,2),
  (0,3),
  (0,4),
  (2,5),
  (3,5),
  (5,4),
  (5,3),
  (5,2),
  (3,1),
  (2,3)
]

v = [
  (1,1),
  (1,2),
  (1,3),
  (1,4),
  (2,4),
  (3,4),
  (4,4),
  (4,3),
  (4,2),
  (4,1),
  (2,2)
]

p = [9, 1, 7, 6, 5, 4, 3, 2, 0, 10, 8]

Marked connectors:

e0 = (v0, v1)
e1 = (v1, v2)
e2 = (v2, v3)
e3 = (v3, v4)
e4 = (v4, v5)
e5 = (v5, v6)
e6 = (v6, v7)
e7 = (v7, v8)
e8 = (v8, v9)

b  = (v1, v10)

Initial matched frames:

F1

Selection/cursor state:

none / null
Solving insight

The first major phase establishes the two trunk ends without treating every newly glowing middle frame as permanent. By action 22, F9 is complete and can be left alone.

At the branch junction, the local sequence

b e1 b

uses the bay to rotate three tokens rather than merely storing one. After that rotation, both F1 and F10 glow, but several middle tokens still have to pass through previously matched positions.

The last action, e4, swaps complementary T4/T5 tokens and completes both central frames simultaneously.

Exact replayable reference solution
e5 e4 e2 e3 e4 e5
e7 e6 e5 e4 e3 e2
e0 e1 e0
e2 e3 e4 e5 e6 e7 e8
e7 e6 e5 e4 e3 e2
b e1 b
e2 e3 e4 e5 e6 e7
e6 e5 e4
e2 e3 e2
e4 e5 e4

Expanded length: 46 valid swaps.

This is a verified solution route, not a requirement that the implementation reject shorter valid routes.

Anti-degenerate check

Connector cut demands are:

e0 e1 e2 e3 e4 e5 e6 e7 e8 b
 1  2  3  4  5  4  3  2  1 1

Every connector separates at least one token from its required frame. Removing any connector makes the initial instance unsolvable.

The transport lower bound is 26 swaps. The 46-step reference is state-self-avoiding: no complete token arrangement occurs twice.

The bay’s role contrasts with Level 5. In Level 5 it is a long-term store and eventual destination; here it is briefly used to alter the order of two trunk tokens.

Level 7 — Crossed Reservoirs
Player-facing discovery

The final board has two side bays attached near opposite ends of a bent trunk.

Their desired tokens begin almost maximally crossed:

Left bay F10 needs T10; T10 begins at the far-right trunk endpoint v9.

Right bay F11 needs T11; T11 begins at the far-left endpoint v0.

Meanwhile, F7 begins matched near the right bay. Preserving it blindly obstructs the route required by T11.

The two bays must be used at different times and in different ways:

the right bay is opened early, used as a long-lived reservoir, and finalized later;

the left bay is handled late with a short three-token rotation;

already-glowing trunk frames are sometimes preserved and sometimes intentionally reopened depending on which crossed token still needs passage.

Nothing is hidden. Both destinations, both travelling tokens, every connector, and every match transition remain visible.

State-machine instance
Frames: 12
Types used: T0 ... T11

a = [
  (2,1),
  (0,2),
  (0,3),
  (0,4),
  (2,5),
  (3,5),
  (5,4),
  (5,3),
  (5,2),
  (3,1),
  (2,3),
  (3,3)
]

v = [
  (1,1),
  (1,2),
  (1,3),
  (1,4),
  (2,4),
  (3,4),
  (4,4),
  (4,3),
  (4,2),
  (4,1),
  (2,2),
  (3,2)
]

p = [11, 6, 8, 5, 1, 4, 3, 7, 2, 10, 9, 0]

Marked connectors:

e0 = (v0, v1)
e1 = (v1, v2)
e2 = (v2, v3)
e3 = (v3, v4)
e4 = (v4, v5)
e5 = (v5, v6)
e6 = (v6, v7)
e7 = (v7, v8)
e8 = (v8, v9)

bL = (v1, v10)
bR = (v8, v11)

Initial matched frames:

F7

Selection/cursor state:

none / null
Solving insight

The first phase moves T11 toward the right while breaking the initially matched F7 only when its slot is needed. The first bR click uses the right bay as storage rather than completing it.

The second bR click, much later, installs T11 permanently. The left bay is then solved through:

bL e1 bL

which rotates the bay token, the junction token, and its right neighbor. Only after both bays are stable can the center be closed without reopening either branch.

The last e2 swap exchanges complementary tokens at v2 and v3, completing the final two frames together—the introductory Level 1 insight now appearing at the end of a global routing problem.

Exact replayable reference solution
e3 e2 e1 e3 e4 e5 e6 bR
e7 e6 e4 e3 e5 e4 e3 e2
e0 e1 e0
e2 e3 e4 e5 e4 e6
e8 e7 bR
e6 e2 e3 e5 e4 e3 e2
bL e1 bL
e2 e3 e4 e5 e6 e7 e6 e8
e4 e5 e4 e3 e2

Expanded length: 51 valid swaps.

The branch timings in this route are:

Action 8:  first bR use, temporary reservoir
Action 28: second bR use, right bay finalized
Action 36: first bL use
Action 38: second bL use, left bay finalized
Anti-degenerate check

Connector cut demands are:

e0 e1 e2 e3 e4 e5 e6 e7 e8 bL bR
 1  3  4  5  5  4  3  3  1  1   1

Every connector is necessary in every solution: deleting any one strands at least one token on the wrong side of its matching frame.

The transport lower bound is 31 swaps. The 51-step reference contains no repeated complete board state and uses both branches in distinct temporal roles.

The layout is not ordered by use:

the right branch is touched before the left branch;

the left branch’s target begins at the opposite endpoint;

the initially matched frame is neither simply preserved nor immediately discarded;

the middle is closed only after both spatially separate branch dependencies are resolved.

3. Progression and epistemic audit
Level	Evidence introduced or reinforced	Strategic role of a match	Reference depth
1	Bright connector swaps both endpoints; exact pair equality causes glow	Visible objective and simultaneous double completion	1
2	Glowing frames remain swappable	Required central reservoir	4
3	Displaced tokens must be routed around visible topology	Intermediate progress that may move around the cycle	9
4	Silhouette is part of exact type; branch token must pass through a match	Matched articulation becomes a liability	29
5	One bay can be temporary storage and later a final home	Two matches are opened at different times	37
6	A side bay can implement a three-token rotation	Stable endpoint versus temporary branch buffer	46
7	Two reservoirs require offset timing and crossed routing	Tool, obstruction, temporary store, and preserved endpoint	51
Foundational attribution check

The first three levels contain no decoys, hidden dependencies, tight budgets, or decorative objects.

Level 1 provides a pre-solved example and one causal action.

Level 2 introduces exactly one new causal fact: matches do not lock.

Level 3 introduces no new operational rule; it composes the known swap into visible circulation.

Invalid actions are immediately identifiable, reversible valid actions have no penalty, and the settled frame always shows every relevant state variable.

Mature-level anti-cleanliness check

Levels 4–7 use connector trees rather than a sequence of independent pair adjustments. Every tree edge has positive cut demand, so every marked connector participates in the actual token dependency structure.

The mature solutions are not “visit device A, then B, then C” recipes:

branches are used before their final tokens arrive;

matched frames are opened according to future traffic rather than local correctness;

distant tokens travel in opposite directions through shared junctions;

the final complementary swap must be created by prior routing;

all provided reference routes are state-self-avoiding.

No later level introduces a new button, lock, checker, counter, direction rule, token generator, or exception. The increasing difficulty comes entirely from stronger consequences of the same visible transposition system.
