# Valvecheck

## Core Game

Valvecheck is a click-only fluid transfer puzzle. The board is built from black reservoirs, gray separator walls, blue valve buttons, colored target bars, and movable colored "curl" pieces. There is no avatar and no directional movement. The only player action is clicking a visible blue valve button or an active gate.

Each level has a gravity direction. Blue valves transfer a fixed slab of black reservoir space between two linked reservoirs along that gravity axis. The transfer amount is the magnitude of the level gravity: 2 pixels in the small tutorial levels and 3 pixels in the larger gate levels. For example, with rightward gravity, a transfer from reservoir A to reservoir B removes 2 or 3 pixels from A's upstream edge and adds the same amount to B's upstream edge. The reverse button moves the same amount back.

Movable curls are gray bracket-like pieces with a colored strip. A curl is always attached to the upstream edge of one black reservoir:

- With rightward gravity, the curl sits just left of the reservoir.
- With leftward gravity, the curl sits just right of the reservoir.
- With downward gravity, the curl sits just above the reservoir.
- With upward gravity, the curl sits just below the reservoir.

When a blue valve transfers volume out of the reservoir that carries a curl, the curl moves one transfer step with that reservoir in the gravity direction. When a valve transfers volume into the reservoir that carries a curl, the curl moves one transfer step against gravity. This is the main way to position curls on their target markers.

The win condition is color and alignment based. Every movable curl must match a fixed target bar of the same color. A curl is satisfied when:

- it is attached to a black reservoir;
- the target's gray wall is one of the walls adjacent to that reservoir;
- the curl and target have the same coordinate along the gravity axis;
- the target color appears in the curl's colored strip.

Horizontal levels compare x-coordinates. Vertical levels compare y-coordinates. Multiple curls must all be satisfied at the same time before the level advances.

## Shared Objects

Use the level's native grid coordinates for reconstruction. The renderer scales the board to the display; the game logic uses the native coordinates listed below.

Common visual objects:

- Black reservoirs: mutable black rectangles. Their position and length change when blue valves are clicked.
- Gray walls: solid separator or stop pieces. Some longer walls have transparent gaps cut out of them.
- Blue valve buttons: small blue 2x2 or 3x3 click targets. Each one has a fixed directed reservoir pair.
- Fixed target bars: small colored bars embedded in or next to a gray wall.
- Movable curls: gray bracket shapes with a colored strip. Small curls are 3x3; large curls are 6x6.
- Gates: click targets embedded in gray separators. They are inactive until black reservoirs touch both sides.

Colors used by gameplay objects:

```text
black    reservoir / active channel
gray     solid walls and curl body
blue     clickable valve buttons
yellow   target/curl color
green    target/curl color
purple   target/curl color
```

The UI also shows a top step bar. Every click consumes one step, including clicks that do not change the board. Gate animation frames do not consume extra steps. If the step counter reaches zero before the level is solved, the level is lost. Solving on the click that empties the step bar still advances, because the win check runs before the loss check. Reset is the normal engine reset; there are no lives, pickups, or automatic partial resets.

## Transfer Rules

Each blue button is directional. A level usually places a pair of blue buttons at one connection: one transfers A to B, and the other transfers B to A. The destination reservoir must have room before the nearest gray stop or bounded track. A click that cannot transfer because the source is empty or the destination is full leaves the board unchanged but still costs one step.

When a reservoir with a curl moves, the curl remains attached to that reservoir's upstream edge. This means the player can push a curl in either direction along a lane by choosing which linked reservoir receives volume:

- To move a curl with gravity, transfer out of its reservoir.
- To move a curl against gravity, transfer into its reservoir.

Black reservoir area is conserved across a blue-button transfer: one reservoir shrinks by the transfer amount and the linked reservoir grows by the same amount.

## Gate Rules

Gates are used in the later levels to move curls across gray separators. A gate is active only when a black reservoir touches it on both sides at the same along-axis position. Active gates are visibly highlighted and outlined. Inactive gates are closed; clicking them does nothing except spend a step.

Clicking an active gate performs an animation:

1. The gate slides one gate length upstream, opening the separator.
2. Every curl attached to either adjacent reservoir moves across the separator to the corresponding position on the opposite reservoir.
3. The gate slides back into place.
4. Gate highlights are recalculated for the new reservoir positions.

A gate does not transfer black reservoir volume. It only transfers attached curls across the separator. If curls are attached on both sides, they swap sides during the same gate animation.

## Wall Masks

Most gray walls are solid rectangles. A few long separator walls contain transparent gaps. A gap means that those cells are visually empty and do not serve as a solid gray stop for target-wall matching.

For vertical gray walls, gap coordinates are inclusive y-ranges inside the wall's rectangle. For horizontal gray walls, gap coordinates are inclusive x-ranges inside the wall's rectangle.

```text
Solid:     entire listed rectangle is gray.
V gap:     vertical wall, gray except listed y-ranges.
H gap:     horizontal wall, gray except listed x-ranges.
```

## Level 1

```text
Grid: 32x32
Gravity: right, transfer step 2
Step budget: 50
Targets: 1 yellow
Gates: none
```

Initial objects:

```text
Reservoir A  (16, 0, 16, 14)
Reservoir B  (26, 16, 6, 16)

Gray wall    (10, 14, 22, 2) solid

Blue valve   (30, 12, 2, 2): Reservoir B -> Reservoir A
Blue valve   (30, 16, 2, 2): Reservoir A -> Reservoir B

Yellow target bar  (17, 14, 3, 2), on the gray wall
Yellow curl        (23, 22, 3, 3), attached to Reservoir B
```

This is the one-curl tutorial. The yellow curl starts on the lower reservoir, to the left of its black segment. The fixed yellow marker is on the gray separator above that lower reservoir. Because gravity is rightward, the curl must end with x-coordinate 17 while attached to a reservoir adjacent to the target wall.

The lower blue valve transfers black reservoir area from Reservoir A into Reservoir B. Since the curl is attached to Reservoir B, each useful click on that lower valve expands Reservoir B upstream and pulls the curl 2 pixels left. Three clicks move the curl from x=23 to x=17 and solve the level.

The upper valve transfers the other way. It pushes the lower curl rightward if used from the initial state, so it moves the curl away from the target.

Important reconstruction details:

- The target is not a separate destination cell. It is a wall marker; the curl aligns with it along the gravity axis.
- The curl remains below the wall. It does not need to enter the exact pixels occupied by the fixed target bar.
- The two blue buttons are a reversible pair for the same connection.

## Level 2

```text
Grid: 32x32
Gravity: left, transfer step 2
Step budget: 50
Targets: 1 green
Gates: none
```

Initial objects:

```text
Reservoir A  (0, 0, 26, 10)
Reservoir B  (0, 12, 6, 8)
Reservoir C  (0, 22, 4, 10)

Gray wall    (0, 10, 28, 2) solid
Gray wall    (0, 20, 22, 2) solid

Blue valve   (0, 8, 2, 2):  Reservoir B -> Reservoir A
Blue valve   (0, 12, 2, 2): Reservoir A -> Reservoir B
Blue valve   (0, 18, 2, 2): Reservoir C -> Reservoir B
Blue valve   (0, 22, 2, 2): Reservoir B -> Reservoir C

Green target bar  (14, 20, 3, 2), on the lower gray wall
Green curl        (4, 26, 3, 3), attached to Reservoir C
```

This level introduces multiple reservoirs stacked vertically. The green curl starts on the lowest reservoir. Its target is on the separator between the middle and lowest reservoirs, at x=14. With leftward gravity, a curl attached to the right edge of a reservoir must be moved rightward to reach a larger x-coordinate.

The connection between Reservoir B and Reservoir C is the connection that moves the curl. Clicking the valve at `(0, 22)` transfers black area from Reservoir B into Reservoir C, so the attached green curl moves 2 pixels right against gravity. Reservoir B does not initially contain enough volume for all required lower transfers, so the player first has to use the valve at `(0, 12)` to transfer black area from Reservoir A into Reservoir B. After replenishing the middle reservoir, repeated lower transfers move the curl from x=4 to x=14 and satisfy the target.

The upper reservoir and upper valve pair therefore act as a supply stage for the middle reservoir. The level is solved by routing volume down the chain before pushing the green curl to the target coordinate.

Important reconstruction details:

- The left-facing orientation reverses the tutorial intuition: adding volume to the curl's reservoir moves the curl to the right.
- Reservoirs A, B, and C are separate black mutable rectangles. They do not merge visually unless a transfer makes one extend to its stop.
- The green target is adjacent to Reservoir B and Reservoir C; the curl can satisfy it while attached to the lower reservoir.

## Level 3

```text
Grid: 52x52
Gravity: down, transfer step 2
Step budget: 75
Targets: yellow, green, purple
Gates: none
```

Initial objects:

```text
Reservoir A  (0, 48, 8, 4)
Reservoir B  (10, 50, 10, 2)
Reservoir C  (22, 48, 8, 4)
Reservoir D  (32, 46, 10, 6)
Reservoir E  (44, 24, 8, 28)

Gray wall    (8, 32, 2, 20) solid
Gray wall    (20, 46, 2, 6) solid
Gray wall    (30, 30, 2, 22) solid
Gray wall    (42, 24, 2, 28) solid

Blue valve   (6, 50, 2, 2):  Reservoir B -> Reservoir A
Blue valve   (10, 50, 2, 2): Reservoir A -> Reservoir B
Blue valve   (22, 50, 2, 2): Reservoir B -> Reservoir C
Blue valve   (18, 50, 2, 2): Reservoir C -> Reservoir B
Blue valve   (32, 50, 2, 2): Reservoir C -> Reservoir D
Blue valve   (28, 50, 2, 2): Reservoir D -> Reservoir C
Blue valve   (44, 50, 2, 2): Reservoir D -> Reservoir E
Blue valve   (40, 50, 2, 2): Reservoir E -> Reservoir D

Green target bar   (8, 33, 2, 3), on the left wall
Purple target bar  (30, 31, 2, 3), on the middle wall
Yellow target bar  (42, 39, 2, 3), on the right wall

Green curl   (2, 45, 3, 3), attached to Reservoir A
Purple curl  (35, 43, 3, 3), attached to Reservoir D
Yellow curl  (46, 21, 3, 3), attached to Reservoir E
```

This level is the first simultaneous multi-target puzzle. The reservoirs form a row of vertical channels along the bottom, with one tall right channel. Blue valves at the bottom transfer black volume between adjacent channels.

Because gravity is downward, the compared coordinate is y. Curls attach above their reservoirs. Transferring volume into a curl's reservoir pulls that curl upward; transferring volume out pushes it downward. The player must place all three curls on their same-colored y positions at once:

```text
green   y=33
purple  y=31
yellow  y=39
```

The green and purple curls start above low reservoirs, while the yellow curl starts above the tall right reservoir. Moving one channel can change the fill lengths available to neighboring channels, so the order matters. A valid solution balances the bottom transfers so each curl's final y-coordinate matches its target while preserving enough black reservoir length to keep the curl attached.

Important reconstruction details:

- There is no gate in this level. All positioning is done by bottom blue valves.
- All three target checks are simultaneous; reaching one target does not lock that curl in place.
- The rightmost tall reservoir is needed to move the yellow curl down to the target range, while the lower reservoirs handle the green and purple curls.
- The target walls are the four vertical gray separators. A curl satisfies a target if attached to either reservoir adjacent to that separator.

## Level 4

```text
Grid: 64x64
Gravity: down, transfer step 3
Step budget: 50
Targets: 1 yellow
Gates: 2
```

Initial objects:

```text
Reservoir A  (0, 49, 12, 15)
Reservoir B  (15, 61, 12, 3)
Reservoir C  (30, 55, 12, 9)
Reservoir D  (45, 58, 9, 6)
Reservoir E  (57, 46, 7, 18)

Gray wall    (12, 31, 3, 33), V gap y=43..54
Gray wall    (27, 22, 3, 42), V gap y=34..45
Gray wall    (42, 25, 3, 39) solid
Gray wall    (54, 34, 3, 30) solid

Blue valve   (15, 61, 3, 3): Reservoir A -> Reservoir B
Blue valve   (9, 61, 3, 3):  Reservoir B -> Reservoir A
Blue valve   (45, 61, 3, 3): Reservoir C -> Reservoir D
Blue valve   (39, 61, 3, 3): Reservoir D -> Reservoir C
Blue valve   (57, 61, 3, 3): Reservoir D -> Reservoir E
Blue valve   (51, 61, 3, 3): Reservoir E -> Reservoir D

Gate         (12, 43, 3, 12), between Reservoir A/B side and the next lane
Gate         (27, 34, 3, 12), between the middle lanes

Yellow target bar  (42, 25, 3, 6), on the right target wall
Yellow curl        (3, 43, 6, 6), attached to Reservoir A
```

This level introduces gates. The yellow curl starts above the left reservoir at y=43. The target is much higher, at y=25, on the wall around the right-side lanes. Blue valves can raise or lower a curl within its current reservoir, but they cannot by themselves move the curl across the vertical separators. The two gates provide the lateral crossings.

A gate becomes usable only when a black reservoir touches both sides of it. At the start both gates are closed. The player must use blue transfers to bring black reservoir edges up to the gate openings. Once a gate is highlighted, clicking it moves the yellow curl across the separator without moving black volume. The curl can then be raised or lowered in the next lane by using the appropriate blue valve pair.

To solve the level, the yellow curl must be transported from the left reservoir system through the gate at x=12 and then through the gate at x=27, and finally positioned with y-coordinate 25 while attached to a reservoir adjacent to the wall at x=42. The right-side channels and the D/E valve pair provide the vertical adjustment near the target.

Important reconstruction details:

- Gate openings are the transparent gaps in the two left gray walls.
- The gray target wall at x=42 is solid and carries the large yellow target bar near its top.
- Gates do not stay open after being clicked. They animate open, move the curl across, and close again.
- A closed gate remains clickable by the engine, but a closed-gate click has no board effect and still costs a step.

## Level 5

```text
Grid: 64x64
Gravity: right, transfer step 3
Step budget: 200
Targets: yellow and green
Gates: 3
```

Initial objects:

```text
Reservoir A  (16, 0, 48, 14)
Reservoir B  (58, 17, 6, 15)
Reservoir C  (46, 35, 18, 14)
Reservoir D  (43, 52, 21, 12)

Gray wall    (16, 14, 48, 3), H gaps x=28..39
Gray wall    (25, 32, 39, 3), H gap  x=40..51
Gray wall    (10, 49, 54, 3), H gaps x=25..36

Blue valve   (61, 17, 3, 3): Reservoir A -> Reservoir B
Blue valve   (61, 11, 3, 3): Reservoir B -> Reservoir A
Blue valve   (61, 29, 3, 3): Reservoir C -> Reservoir B
Blue valve   (61, 35, 3, 3): Reservoir B -> Reservoir C
Blue valve   (61, 46, 3, 3): Reservoir D -> Reservoir C
Blue valve   (61, 52, 3, 3): Reservoir C -> Reservoir D

Gate         (28, 14, 12, 3), between the top and upper-middle reservoirs
Gate         (40, 32, 12, 3), between the middle reservoirs
Gate         (25, 49, 12, 3), between the lower-middle and bottom reservoirs

Yellow target bar  (16, 14, 6, 3), on the top separator
Green target bar   (10, 49, 6, 3), on the bottom separator

Green curl         (10, 5, 6, 6), attached to Reservoir A
Yellow curl        (37, 55, 6, 6), attached to Reservoir D
```

This level combines multiple gates with multiple simultaneous targets. The reservoirs are horizontal lanes stacked from top to bottom, and the blue valve controls sit on the right edge. Gravity is rightward, so the target coordinate is x.

The green curl begins in the top lane already near the green target's x-coordinate, but the green target is on the bottom separator. The yellow curl begins in the bottom lane, while the yellow target is on the top separator. The player must use the three gates to exchange curls between lanes and the right-edge valves to tune their x-coordinates.

The intended structure is a vertical ladder:

```text
top lane          Reservoir A, green curl starts here, yellow target wall below
upper-middle      Reservoir B
lower-middle      Reservoir C
bottom lane       Reservoir D, yellow curl starts here, green target wall above
```

Each gate is usable only after the adjacent reservoirs have been shifted until black space touches both sides of that separator opening. The two curls can pass through the gate ladder one gate at a time. Since targets are checked simultaneously, moving one curl into place is not enough; both the yellow curl at x=16 and the green curl at x=10 must be satisfied at the same time on walls adjacent to their current reservoirs.

Important reconstruction details:

- The three gray separators have transparent horizontal gaps exactly where their gates sit.
- The top target is yellow and belongs to the top separator; the bottom target is green and belongs to the bottom separator.
- The two curls are large 6x6 pieces. Their x-coordinate, not their center, is what is matched to target x.
- The long 200-step budget allows experimentation with gate timing and repeated transfers.

## Level 6

```text
Grid: 64x64
Gravity: left, transfer step 3
Step budget: 50
Targets: 1 yellow
Gates: 2
```

Initial objects:

```text
Fixed gray stop  (21, 0, 3, 30)

Reservoir A      (0, 0, 3, 30)
Reservoir B      (0, 33, 18, 31)
Reservoir C      (24, 0, 15, 30)

Gray wall        (0, 30, 54, 3), H gaps x=6..17 and x=30..41

Blue valve       (0, 27, 3, 3):  Reservoir B -> Reservoir A
Blue valve       (0, 33, 3, 3):  Reservoir A -> Reservoir B
Blue valve       (24, 27, 3, 3): Reservoir B -> Reservoir C
Blue valve       (24, 33, 3, 3): Reservoir C -> Reservoir B

Gate             (6, 30, 12, 3), left separator opening
Gate             (30, 30, 12, 3), right separator opening

Yellow target bar  (48, 30, 6, 3), on the long horizontal separator
Yellow curl        (3, 12, 6, 6), attached to Reservoir A
```

This level uses the same gate mechanism in an indirect layout. The upper-left and upper-right reservoirs are not next to each other. The lower reservoir is the transfer bridge between them. The player must use the left valve pair to connect Reservoir A to Reservoir B, then use Reservoir B to affect Reservoir C through the right valve pair.

Gravity is leftward, so the yellow curl begins on the right edge of Reservoir A and must eventually reach x=48, far to the right. Moving a left-gravity curl rightward requires transferring black volume into its current reservoir. Once the curl is positioned at an active gate, clicking that gate moves it across the horizontal separator into the lower or upper reservoir on the other side. The process then repeats through the right-side gate.

The target is at the far right end of the long separator. To solve the level, the yellow curl must be carried through the indirect lower-reservoir bridge and positioned with x-coordinate 48 while attached to a reservoir adjacent to the target wall.

Important reconstruction details:

- The gray stop at `(21, 0, 3, 30)` limits upper transfer and creates the indirect route.
- The two blue valve pairs are at different x-columns, so the lower reservoir is the only shared transfer medium.
- The horizontal separator has gate openings under the left and right upper reservoirs plus a target area at the far right.
- The 50-step limit is tight compared with the amount of setup needed for both gates.

## Level 7

```text
Grid: 48x48
Gravity: up, transfer step 2
Step budget: 200
Targets: yellow, green, purple
Gates: 3
```

Initial objects:

```text
Fixed gray stop  (0, 22, 14, 2)
Fixed gray stop  (34, 22, 14, 2)

Reservoir A      (0, 0, 14, 20)
Reservoir B      (0, 24, 14, 8)
Reservoir C      (16, 0, 16, 8)
Reservoir D      (34, 0, 14, 6)
Reservoir E      (34, 24, 14, 10)

Gray wall        (14, 0, 2, 48), V gap y=30..37
Gray wall        (32, 0, 2, 48), V gaps y=8..15 and y=30..37

Blue valve       (16, 0, 2, 2):  Reservoir A -> Reservoir C
Blue valve       (12, 0, 2, 2):  Reservoir C -> Reservoir A
Blue valve       (16, 24, 2, 2): Reservoir B -> Reservoir C
Blue valve       (12, 24, 2, 2): Reservoir C -> Reservoir B
Blue valve       (34, 0, 2, 2):  Reservoir C -> Reservoir D
Blue valve       (30, 0, 2, 2):  Reservoir D -> Reservoir C
Blue valve       (34, 24, 2, 2): Reservoir C -> Reservoir E
Blue valve       (30, 24, 2, 2): Reservoir E -> Reservoir C

Gate             (14, 30, 2, 8), lower-left wall opening
Gate             (32, 8, 2, 8), upper-right wall opening
Gate             (32, 30, 2, 8), lower-right wall opening

Green target bar   (14, 10, 2, 3), on the left vertical wall
Purple target bar  (14, 42, 2, 3), on the left vertical wall
Yellow target bar  (32, 18, 2, 3), on the right vertical wall

Yellow curl        (6, 32, 3, 3), attached to Reservoir B
Green curl         (40, 34, 3, 3), attached to Reservoir E
Purple curl        (40, 6, 3, 3), attached to Reservoir D
```

The final level is the most constrained version of the transfer-and-gate system. The reservoirs are arranged around a central top reservoir. The left and right vertical gray walls have gate openings and target bars, and three curls must all finish on matching colored target coordinates.

Gravity is upward, so curls attach below their reservoirs and the compared coordinate is y. Transferring volume into a curl's reservoir moves that curl downward; transferring out of it moves the curl upward. The blue valves connect the central reservoir to the four side reservoirs:

```text
left top     Reservoir A <-> Reservoir C
left bottom  Reservoir B <-> Reservoir C
right top    Reservoir D <-> Reservoir C
right bottom Reservoir E <-> Reservoir C
```

The fixed gray stops at y=22 divide the left and right sides into top and bottom regions. The central reservoir is the indirect bridge. Gates in the vertical walls move curls laterally between side reservoirs and the center/right-side structure once black reservoirs touch both sides of the gate opening.

To solve the level, all three curls must be positioned at once:

```text
green   y=10, on a reservoir adjacent to the left wall
yellow  y=18, on a reservoir adjacent to the right wall
purple  y=42, on a reservoir adjacent to the left wall
```

The puzzle requires repeated transfer through the central reservoir, opening gates only after the neighboring black reservoirs have been shifted into contact, and preserving enough reservoir length to keep each curl attached when its target y-coordinate is reached.

Important reconstruction details:

- Fog or hidden information is not part of this implementation; the whole 48x48 logical board is present and visible.
- The three gates are narrow vertical bars, 2 pixels wide by 8 pixels high.
- The left wall carries both the green and purple targets at different y-coordinates; the right wall carries the yellow target.
- The side reservoirs are disconnected except through the central reservoir and gates, making transfer order the main challenge.

## Implementation Checklist

- Use click action only.
- Track a per-level click budget and decrement it once per click.
- Store each black reservoir as a mutable rectangle with position and length along the gravity axis.
- Blue valves have fixed directed reservoir pairs.
- Blue-valve clicks conserve black reservoir area by shrinking the source and growing the destination by the level transfer step.
- Curls attach to the upstream edge of exactly one reservoir and move with reservoir edge changes.
- Target checks are simultaneous for all curls.
- A target matches by color, same along-axis coordinate, and adjacency between the curl's current reservoir and the target's wall.
- Gates are active only when black reservoirs touch both sides of the gate opening.
- Active gate clicks move curls across the separator but do not transfer reservoir volume.
- Recalculate gate highlighting after every reservoir transfer and after each gate animation.
