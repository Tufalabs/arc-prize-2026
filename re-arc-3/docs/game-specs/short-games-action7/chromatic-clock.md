Chromatic Clock

This is a game about advancing one clock becoming difficult because every clock showing the same sector moves with it, and collisions permanently coalesce their future behavior.

Authored level count: exactly 7.

The player clicks large clock-like dials. A click does not affect only the chosen dial: it advances every dial whose hand currently indicates the same color-symbol sector. Consequently, the controllable groups continually change. Empty sectors act as maneuvering space, identical-target groups may be merged deliberately, and different-target groups must never collide.

There are no hidden mappings, random events, movable cursors, splitters, switches, inventories, or secondary devices. All depth comes from one global operation on an eight-position cycle.

1. Player-visible system
1.1 Shared clock cycle

Every dial contains the same eight sectors in the same orientation:

Phase	Direction	Color ID	Geometric sector symbol	Successor
0	north	8 red	center dot	1
1	northeast	14 green	vertical bar	2
2	east	9 blue	horizontal bar	3
3	southeast	11 yellow	\ diagonal	4
4	south	15 purple	/ diagonal	5
5	southwest	12 orange	L-corner	6
6	west	6 magenta	plus	7
7	northwest	10 light blue	X	0

The progression is clockwise. Phase 7 wraps to phase 0.

The symbols are geometric marks, not text. Color, symbol, and direction all redundantly identify the same phase. Thus two distant dials with matching hands are visibly members of the same current cohort.

1.2 Objective

Each dial has one fixed pink target notch attached to one of its eight sectors.

A dial is currently satisfied exactly when:

current_phase[dial] == target_phase[dial]

A satisfied dial displays a green-centered white check inset. The check persists in settled frames while the dial remains satisfied, but it is not a lock. The dial can be clicked or moved by another matching dial, after which the check disappears.

The level succeeds when every authored dial is satisfied simultaneously after a step settles.

Relevant to validation:

Every active dial must match its own target.

Dial positions on the 64×64 canvas do not enter validation.

Step count does not enter validation.

Compatible dials may be merged.

Extra safe revolutions are allowed.

A dial that was previously correct need not have stayed correct.

Inactive module slots and background pixels are ignored.

1.3 Click affordance

Every dial is a large, connected octagonal module. Clicking any rendered cell of its plate activates that dial. Clicking its hub, hand, sector pad, rim, or target notch is equivalent.

The exact subcell clicked does not select that sector. The clicked dial’s current hand determines which cohort advances.

This makes the core decision:

“Which currently visible equivalence class should advance now?”

Clicking a background cell produces a neutral ripple and changes no logical state.

UP, DOWN, LEFT, RIGHT, and SPACE produce a neutral gray edge pulse and change no logical state. Only dial clicks are used to solve levels.

2. Native geometry and rendering contract
2.1 Canvas coordinates

Render canvas: 64×64.

Coordinates: (x,y), with (0,0) at the upper-left.

x increases rightward.

y increases downward.

Background: color 5.

No internal grid, slot border, legend panel, or text is drawn.

The native puzzle geometry is the shared eight-sector cycle. The floating dial arrangement is a schematic display of the objects occupying that cycle.

2.2 Module slots

Documentation IDs are not rendered in the game.

Module	Origin (ox,oy)	Center
A	(1,1)	(8,8)
B	(17,1)	(24,8)
C	(33,1)	(40,8)
D	(49,1)	(56,8)
E	(1,21)	(8,28)
F	(17,21)	(24,28)
G	(33,21)	(40,28)
H	(49,21)	(56,28)
I	(1,41)	(8,48)
J	(17,41)	(24,48)
K	(33,41)	(40,48)
L	(49,41)	(56,48)

Each slot has a 15×15 local coordinate system (u,v), where:

global_x = ox + u
global_y = oy + v
0 <= u,v <= 14

Inactive slots render as plain background and are not clickable.

2.3 Dial silhouette and click mask

The dial plate is a connected octagon. Its clickable mask contains:

v = 1 or 13: u = 5..9
v = 2 or 12: u = 3..11
v = 3 or 11: u = 2..12
v = 4..10:   u = 1..13

All other local cells remain background.

Rendering order:

Plate shell.

Plate interior.

Eight sector pads.

Sector symbols.

Fixed target notch.

Hub and current hand.

Match, animation, failure, or completion overlays.

Plate colors:

Outer one-cell shell: color 2.

Interior: color 4.

A shell pixel is any plate pixel with at least one four-neighbor outside the plate mask.

2.4 Sector pads

Each sector is a 3×3 colored pad:

Phase	Local pad rectangle
0	u=6..8, v=1..3
1	u=10..12, v=3..5
2	u=11..13, v=6..8
3	u=10..12, v=9..11
4	u=6..8, v=11..13
5	u=2..4, v=9..11
6	u=1..3, v=6..8
7	u=2..4, v=3..5

Symbols use color 5 and are drawn relative to the upper-left of the pad:

Phase	Symbol pixels
0	(1,1)
1	(1,0),(1,1),(1,2)
2	(0,1),(1,1),(2,1)
3	(0,0),(1,1),(2,2)
4	(2,0),(1,1),(0,2)
5	(0,0),(0,1),(0,2),(1,2),(2,2)
6	(1,0),(0,1),(1,1),(2,1),(1,2)
7	(0,0),(2,0),(1,1),(0,2),(2,2)
2.5 Target notch

The target notch uses color 7. It replaces these three plate pixels for target phase t:

t	Notch pixels
0	(6,1),(7,1),(8,1)
1	(11,3),(12,3),(12,4)
2	(13,6),(13,7),(13,8)
3	(12,10),(12,11),(11,11)
4	(6,13),(7,13),(8,13)
5	(2,10),(2,11),(3,11)
6	(1,6),(1,7),(1,8)
7	(2,3),(3,3),(2,4)

The notch remains fixed throughout the level.

2.6 Hub and hand

The hub is the local 3×3 block:

u = 6..8
v = 6..8

Its ordinary color is 1.

The hand consists of a white hub-edge marker and a two-cell ray:

Phase	Hub marker	Outer ray
0	(7,6)	(7,5),(7,4)
1	(8,6)	(9,5),(10,5)
2	(8,7)	(9,7),(10,7)
3	(8,8)	(9,9),(10,9)
4	(7,8)	(7,9),(7,10)
5	(6,8)	(5,9),(4,9)
6	(6,7)	(5,7),(4,7)
7	(6,6)	(5,5),(4,5)

All hand pixels use color 0. The ray ends before the sector’s central symbol, leaving that symbol readable.

When matched:

Hub base becomes color 14.

White check pixels are drawn at local coordinates
(6,7),(7,8),(8,7),(8,6).

When jammed:

Hub base becomes color 13.

A white X occupies
(6,6),(8,6),(7,7),(6,8),(8,8).

3. State-machine contract
3.1 Persistent state

For every active dial d:

origin[d]          fixed module origin
phase[d]           integer in 0..7
initial_phase[d]   fixed reset value
target[d]          integer in 0..7

Global state:

mode ∈ {PLAY, JAMMED, COMPLETE}
step_count
jammed_phases      set of phase indices, empty unless JAMMED

Transient render state:

selected_dial
old_phase
affected_dials
animation_frame
last_background_click

Every level begins with:

mode = PLAY
step_count = 0
selected_dial = null
affected_dials = empty
animation_frame = 0
jammed_phases = empty

There is no persistent cursor or selected cohort in a settled frame.

3.2 Derived cohorts

A cohort is not stored as a hidden object. It is derived entirely from visible phases:

cohort(k) = {d | phase[d] == k}

This is important: once two dials reach the same phase, they necessarily receive exactly the same future updates. They therefore cannot split again.

3.3 Dial-click transition

For a click on dial s:

k = phase[s]
affected = {d | phase[d] == k}
destination = (k + 1) mod 8

All affected dials advance simultaneously:

for d in affected:
    next_phase[d] = destination

for d not in affected:
    next_phase[d] = phase[d]

Then all phase values are replaced atomically by next_phase.

Important resolution rules:

Membership is tested from the pre-action state.

The destination cohort does not advance during the same action.

There is no cascading movement.

Every affected dial advances exactly one phase.

Phase 7 wraps directly to phase 0.

Clicking any member of a cohort produces the same logical action.

3.4 Coalescence and jam detection

After movement, collect the target phases represented at each occupied current phase:

targets_at_phase[p] =
    {target[d] | phase[d] == p}

If any occupied phase contains more than one distinct target:

exists p such that len(targets_at_phase[p]) > 1

the level enters JAMMED.

This is not an arbitrary punishment. Every dial in such a cohort will remain synchronized forever, but its members require different final phases. The level is therefore mathematically unsatisfiable from that state.

A safe merge occurs when the moving cohort enters an occupied phase and all members of the resulting cohort share one target. The merge receives a brief pink hub pulse but creates no additional stored state.

Jam recovery

A jammed settled frame retains the actual merged hand positions and displays maroon X hubs on the incompatible cohort. This lets the player see exactly what collided.

The next action of any kind:

Restores every phase[d] to initial_phase[d].

Sets mode=PLAY.

Resets step_count=0.

Does not process the supplied action normally.

Thus a risky experiment costs one click plus one recovery action, not an entire run.

3.5 Success validation

After an ordinary dial transition:

Detect incompatible cohorts.

If incompatible, enter JAMMED.

Otherwise recompute every match inset.

If all dials satisfy phase[d] == target[d], enter COMPLETE.

A level does not complete merely because all checks appeared at some earlier time. They must coexist in the current settled state.

3.6 Non-dial and keyboard actions

In PLAY:

Background click: no logical change; neutral local ripple.

UP, DOWN, LEFT, RIGHT, or SPACE: no logical change; neutral bottom-edge pulse.

These actions increment step_count.

In JAMMED, any action performs the reset described above.

3.7 Exploration allowance

There is no strategic energy, timer, or score.

Use:

max_steps_per_attempt = 512

for every level. This is far above the longest verified solution of 52 clicks. If the cap is reached, show a neutral gray perimeter pulse and reset the attempt; timeout is not used as a puzzle mechanic.

Animation frames never consume steps.

4. Animation and causal feedback

A valid dial click produces a deterministic four-stage animation inside one step():

Cohort identification, one frame: all affected dial faces are rendered one pixel left of their ordinary origin, still showing the old hand.

Rock and sweep, one frame: affected faces render one pixel right; the old ray fades to color 1 while the new ray appears in color 0.

Commit, one frame: faces return to their fixed origins with only the new hand.

Settle, two frames: match checks, safe-merge pulse, jam overlay, or completion pulse appears.

The logical next state is computed atomically when step() is called. The visual offsets are interpolation only. Input is disabled until the final settled frame.

Additional feedback:

Safe merge: both newly coalesced subcohorts’ hubs pulse color 7 once.

Jam: affected hubs become color 13 with white X marks; canvas perimeter flashes color 13.

Success: canvas perimeter pulses color 14 for six render-only frames, then the all-checked board remains visible.

Background click: a color-2 Manhattan ring expands to radius 1 and then 2 before disappearing.

Keyboard no-op: three centered cells on the bottom edge flash color 2.

The settled state alone is sufficient to infer every important consequence.

5. Discovery curriculum
Level	New causal evidence	Distinguishing experiment
1	Clicking advances a hand; hand-on-notch is the goal	One dial is one step before its notch
2	Every dial on the clicked phase advances, but other phases do not	Two distant matching dials move while a third dial remains still
3	Entering an occupied phase coalesces cohorts; incompatible targets make this irreversible and fatal	Clicking the rear dial visibly produces a mixed-target jam
4	A collision is useful when both cohorts share a target	Same-direction target notches predict a safe merge
5	A green check is feedback, not a lock	The only initially safe moves are clicks on already-correct dials
6	Compatible cohorts may need different rendezvous plans	One pair is adjacent; another must first occupy an empty intermediate phase
7	No new rule; all prior rules form a dependency network	A fully occupied clock can be opened only by intentional coalescence

Contrasting uses are entirely contextual:

A merge is fatal in Level 3, useful in Level 4, staged in Level 6, and the only source of space in Level 7.

A satisfied dial is an endpoint in Level 1, a movable blocker in Level 5, and part of a temporary almost-solved trap in Level 7.

An empty phase is a destination to protect in Level 3, a circulating buffer in Level 5, a rendezvous staging point in Level 6, and initially absent in Level 7.

6. Authored levels

In every replay sequence below, click the center coordinate of the named module. Commas separate individual step() calls. After a merge, the listed module is simply a representative; clicking another dial in that cohort is equivalent.

The stated depths are verified shortest legal solution depths under the transition rules above.

Level 1 — One Tick

Teaches: click affordance, one-step clockwise movement, target alignment, completion feedback.
Shortest depth: 1 click.

Player-facing discovery

A single central dial has its hand on the red north sector and its pink notch on the immediately following green northeast sector.

There is no other visible object. Clicking the large dial advances the hand into the notch, produces the persistent check, and triggers completion. The result is attributable to one action and one visible consequence.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start phase	Target phase	Initial check
F	(24,28)	0	1	no

Initial tuple, in dial order (F):

phase  = (0)
target = (1)

Exact replay:

F

Final tuple:

(1)
Attribution and necessity check

The dial itself communicates both interaction and objective.

No legend or arbitrary pattern interpretation is required.

No simultaneous rule change occurs.

The one-action completion is intentional positive goal evidence for the rest of the game.

Level 2 — Distant Echo

Teaches: clicks affect the entire current phase cohort, not only the selected dial and not all dials.
Shortest depth: 6 clicks.

Player-facing discovery

B and K are far apart but both show phase 1. E shows phase 7.

Clicking B causes both B and K to rock and advance to phase 2. E does not move. This single observation distinguishes the intended rule from:

selected-dial-only movement;

physical-neighbor movement;

every-dial movement.

E must later be advanced separately, so it is an active causal control rather than a decoy.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target	Initial cohort
B	(24,8)	1	6	{B,K}
K	(40,48)	1	6	{B,K}
E	(8,28)	7	0	{E}

Tuple order (B,K,E):

phase  = (1,1,7)
target = (6,6,0)

Representative exact replay:

B, B, B, B, B, E

State changes:

B/K: 1 → 2 → 3 → 4 → 5 → 6
E:   7 → 0

The E click may be inserted anywhere without changing correctness.

Attribution and necessity check

B and K are both necessary to reveal global matching.

E is necessary to reveal selectivity and is itself required for success.

B/K never enter E’s phase during the solution, so no collision rule is introduced yet.

There is exactly one new causal unknown: which dials respond to a click.

Level 3 — Vacate Seven

Teaches: destination occupancy, irreversible coalescence, harmful merge, and vacating the destination.
Shortest depth: 6 clicks.

Player-facing discovery

F shows phase 6 and wants phase 1. G shows phase 7 and wants phase 2.

F’s immediately next sector is already occupied by G. The tempting click on F is a low-cost distinguishing experiment:

before: F=6, G=7
click F
after:  F=7, G=7

Their targets differ, so both hubs become maroon X marks and the level jams. This visibly rules out blocking, bouncing, swapping, or harmless overlap.

After reset, moving G ahead first leaves phase 7 empty. F can then follow without catching it.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target
F	(24,28)	6	1
G	(40,28)	7	2

Tuple order (F,G):

phase  = (6,7)
target = (1,2)

Wrong experiment:

F

Result:

phase = (7,7)
targets at phase 7 = {1,2}
mode = JAMMED

Exact successful replay:

G, G, G, F, F, F

Path:

G: 7 → 0 → 1 → 2
F: 6 → 7 → 0 → 1

Final tuple:

(1,2)
Attribution and necessity check

Only two dials are present.

The wrong action changes only the collision relation being taught.

The failure is immediately visible and automatically recoverable.

F cannot safely advance until G has vacated phase 7.

The same universal movement rule from Level 2 explains why equality is permanent; no special collision exception is added.

Level 4 — Safe Collision

Teaches: intentional compatible merging and treating a merged cohort as one controllable unit.
Shortest depth: 27 clicks.

Player-facing discovery

A and B begin on adjacent phases 0 and 1, but both target phase 7. Their pink notches point in the same direction.

After Level 3, entering an occupied phase appears dangerous. Here the identical target notches provide a principled reason to test the collision. Clicking A moves it into B’s phase:

A=0, B=1  →  A=1, B=1

No jam occurs. On the next click of A later in the path, both A and B advance together, confirming permanent coalescence.

The merge is not optional: both dials must finish at phase 7, so they must eventually become one cohort.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target
A	(8,8)	0	7
B	(24,8)	1	7
D	(56,8)	3	0
F	(24,28)	5	1
I	(8,48)	6	3
L	(56,48)	7	5

Tuple order (A,B,D,F,I,L):

phase  = (0,1,3,5,6,7)
target = (7,7,0,1,3,5)

Initially empty phases:

{2,4}

Opening checkpoint:

click A
phase = (1,1,3,5,6,7)

A and B now form one compatible cohort.

Exact replay:

A, L, A, L, I, D, A, I, L,
F, D, A, L, I, F, D, A, F,
L, I, D, A, F, D, L, I, A

When A occurs after the first step, both A and B move.

Final tuple:

(7,7,0,1,3,5)
Anti-degenerate and necessity check

A/B’s common target makes their merge necessary rather than decorative.

The empty phases at 2 and 4 are unevenly placed; they must be circulated rather than consumed in one linear sweep.

D, F, I, and L each occupy phases that alternately block and release the merged A/B cohort.

Removing B eliminates the safe-merge lesson.

Removing any other dial supplies enough extra space to collapse the authored leapfrogging pressure.

The same collision operation remains fatal for every pair with differing notch directions.

Level 5 — Break the Checks

Teaches: a satisfied dial remains movable; correct states may need to be deliberately broken.
Shortest depth: 44 clicks.

Player-facing discovery

B and L begin checked:

B is at phase 1 with target 1.

L is at phase 7 with target 7.

The occupied phases are:

{1,3,4,5,6,7}

so the only empty phases are 0 and 2.

The only initially safe advances are:

L from 7 into empty phase 0;

B from 1 into empty phase 2.

Both require clicking a currently correct dial. Every other click immediately enters an occupied phase belonging to an incompatible target and jams.

Thus ordinary preservation makes no progress. The local impasse directly suggests clicking one of the conspicuously checked modules. Its check disappears, an empty destination moves elsewhere, and the meaning of the check becomes unambiguous: it reports current correctness but grants no protection.

No targets are duplicated in this level, so every merge is harmful.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target	Initially checked
B	(24,8)	1	1	yes
C	(40,8)	3	2	no
E	(8,28)	4	3	no
H	(56,28)	5	4	no
I	(8,48)	6	5	no
L	(56,48)	7	7	yes

Tuple order (B,C,E,H,I,L):

phase  = (1,3,4,5,6,7)
target = (1,2,3,4,5,7)

Initial legal non-jamming classes:

phase 7: L, because phase 0 is empty
phase 1: B, because phase 2 is empty

Opening state changes in the listed replay:

click L: (1,3,4,5,6,0)
click B: (2,3,4,5,6,0)
click L: (2,3,4,5,6,1)

Exact replay:

L, B, L, I, H, E, C, B, I, L, H,
E, C, B, I, H, L, E, C, B, I, H,
E, L, C, B, I, H, E, C, L, B, I,
H, E, C, B, L, L, I, H, E, C, B

Final tuple:

(1,2,3,4,5,7)
Anti-degenerate and necessity check

The only two initially productive experiments both remove visible checks.

A mistaken hypothesis that checked dials are locked or should never move cannot solve the opening.

Every target is distinct, so intentional merging from Level 4 cannot substitute for vacancy management.

The two gaps begin on opposite sides of an uneven six-cohort arrangement and change relative position repeatedly.

The replay contains no long uniform lap: the order of L, I, and the central four cohorts changes as the vacancies migrate.

Removing either checked boundary dial creates a conventional un-checked opening and destroys the central lesson.

Level 6 — Two Rendezvous

Teaches: compatible classes can require different approaches before merging; an empty phase may be a staging location rather than merely free space.
Shortest depth: 47 clicks.

Player-facing discovery

There are two required target pairs:

C and D both target phase 3.

H and J both target phase 5.

Their initial geometry on the shared cycle differs.

C and D occupy adjacent phases 1 and 2. Clicking C immediately produces a safe merge at phase 2.

H and J occupy phases 4 and 6, with empty phase 5 between them. Clicking H once does not merge anything: H merely enters phase 5. It must later advance from 5 into J’s phase 6.

This forces the player to generalize the compatible-merge rule. “Matching targets mean click once” is insufficient; the actual rule remains ordinary one-step clockwise motion, and target-compatible coalescence occurs only when the physical destination is occupied.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target	Required final group
A	(8,8)	0	2	singleton
C	(40,8)	1	3	{C,D}
D	(56,8)	2	3	{C,D}
F	(24,28)	3	4	singleton
H	(56,28)	4	5	{H,J}
J	(24,48)	6	5	{H,J}
L	(56,48)	7	0	singleton

Tuple order (A,C,D,F,H,J,L):

phase  = (0,1,2,3,4,6,7)
target = (2,3,3,4,5,5,0)

Initial empty phase:

{5}

Key checkpoints:

click C:
(0,2,2,3,4,6,7)
C/D safely merge at phase 2.

after C, A, L, H:
(1,2,2,3,5,6,0)
H is staged at phase 5 but has not merged.

after the first nine listed clicks:
(2,3,3,4,6,6,1)
H/J have now merged at phase 6.

Exact replay:

C, A, L, H, F, C, A, L, H, F, C, A,
L, H, F, C, A, H, L, F, C, A, H, F,
L, C, A, H, F, C, L, A, H, F, C, A,
L, L, H, F, C, A, L, H, F, C, A

After merging:

C is the representative click for cohort {C,D}.

H is the representative click for cohort {H,J}.

Final tuple:

(2,3,3,4,5,5,0)
Anti-degenerate and necessity check

C/D must merge, because both require phase 3.

H/J must merge, because both require phase 5.

C/D begin adjacent; H/J are separated by an authored one-sector gap.

Phase 5 first serves as a rendezvous staging point, then later becomes part of the rotating vacancy structure.

A, F, and L prevent the two pairwise tasks from being independent errands; moving one rendezvous changes when the other can advance safely.

Removing J turns H’s staging move into a simple target arrival.

Removing D eliminates the immediate-merge contrast.

The level therefore cannot be reduced to two disconnected target checks.

Level 7 — No Empty Hour

Teaches: full composition with no initial vacancy; intentional merge as space creation; completed cohorts and checked singletons as temporary liabilities.
Shortest depth: 52 clicks.

Player-facing discovery

Every one of the eight phases is occupied.

There is initially no empty destination, so every click creates a collision. Six of the eight possible current-phase clicks would merge incompatible targets and jam.

Only two opening actions are safe:

E at phase 3 can enter G at phase 4 because both target phase 4.

H at phase 5 can enter J at phase 6 because both target phase 5.

The intended replay begins with E/G. This creates the first vacancy at phase 3.

The next three actions place D, B, and A on their targets. At that point the board looks almost solved:

A=1, B=2, D=3, E/G=4, H=5, J=6, L=7

Only J is visibly wrong. Nevertheless, J cannot move backward into H. H must leave its own check, merge forward with J, and then travel around the cycle. L must also leave its check to vacate the wraparound destination. Even the completed E/G cohort must move away from phase 4 to let the newly merged H/J cohort circulate.

The apparent near-solution is therefore a physically grounded trap, not a hidden validator trick.

State-machine instance

Initial control state:

mode = PLAY
selected_dial = null
step_count = 0
Dial	Center	Start	Target	Initially checked
A	(8,8)	0	1	no
B	(24,8)	1	2	no
D	(56,8)	2	3	no
E	(8,28)	3	4	no
G	(40,28)	4	4	yes
H	(56,28)	5	5	yes
J	(24,48)	6	5	no
L	(56,48)	7	7	yes

Tuple order (A,B,D,E,G,H,J,L):

phase  = (0,1,2,3,4,5,6,7)
target = (1,2,3,4,4,5,5,7)

Initial occupancy:

every phase 0..7 occupied

Initial safe collisions:

E: phase 3 → phase 4, merging with G; both target 4
H: phase 5 → phase 6, merging with J; both target 5

All other initial clicks jam.

Opening checkpoints for the intended replay:

start:
(0,1,2,3,4,5,6,7)

after E:
(0,1,2,4,4,5,6,7)

after E, D, B, A:
(1,2,3,4,4,5,6,7)

after L:
(1,2,3,4,4,5,6,0)
L has deliberately left target 7.

after H:
(1,2,3,4,4,6,6,0)
H/J have merged forward at phase 6.

after the next E:
(1,2,3,5,5,6,6,0)
The completed E/G cohort has deliberately left target 4.

Exact replay:

E, D, B, A, L, H, E, D, B, A, L, H, E,
D, B, A, H, L, E, D, B, A, H, E, L, D,
B, A, H, E, D, L, B, A, H, E, D, B, L,
A, H, E, D, B, A, L, L, H, E, D, B, A

After merging:

E represents cohort {E,G}.

H represents cohort {H,J}.

Final tuple:

(1,2,3,4,4,5,5,7)
Anti-degenerate and necessity check

With no initial vacancy, compatible merging is the only way to begin.

E/G merge directly onto their target and initially appear finished.

H begins finished but must move forward to collect J.

L begins finished but must vacate phase 7 so the wraparound chain can continue.

The completed E/G cohort later becomes a blocker and must also leave its target.

Removing any one dial creates an initial empty phase and bypasses the defining opening dependency.

The two target pairs, three advancing singletons, and checked wraparound singleton form a dependency network rather than an ordered device list.

No new object or exception appears in the final level; every consequence follows from the already observed transition rule.

7. Implementation and verification summary

The complete game uses:

one object category: dial;

one meaningful player action: click a dial;

one cyclic spatial operation: advance the selected current-phase cohort by one sector;

one derived irreversible consequence: cohorts that meet can never split;

one exact objective: every hand aligned with its own attached notch.

Verified shortest solution depths:

Level	Depth
1	1
2	6
3	6
4	27
5	44
6	47
7	52

The first three levels isolate their causal evidence. Levels 4–7 reuse the same rules under increasingly different occupancy, target-grouping, vacancy, timing, and state-preservation pressures. No level requires reading hidden attributes, visiting independent stations, satisfying an arbitrary checklist, or learning a late one-off device.
