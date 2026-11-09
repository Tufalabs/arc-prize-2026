# LF52 - Leap Frog

## Core Game

Leap Frog is a discrete jump puzzle played on a light-blue background with white square floor tiles, black rail tracks, yellow movable platforms, colored buds, and purple fixed posts. The player alternates between two kinds of actions:

- Click a bud to select it, then click one of its highlighted legal landing cells to jump.
- Press a direction to move every movable platform one rail cell in that direction.

The goal is to reduce the counted buds to the level's target count. A level is checked for completion after a jump, not after a platform move. Winning advances to the next level.

## Jump Rules

All coordinates below use `(x,y)`, where `x` is the column from left to right and `y` is the row from top to bottom.

A selectable bud may jump exactly two cells in one cardinal direction. The cell one step away must contain a jumpable object, and the landing cell two steps away must be empty landable terrain.

Jumpable midpoint objects are:

- green buds
- red buds
- blue buds
- purple posts

Landable destination cells are:

- an empty white floor tile
- an empty yellow movable platform

Rail-only cells are not landable. A platform with a bud or purple post on it is occupied, so it is not a legal landing until the occupant moves away.

When a jump is executed, the selected bud moves to the landing cell. The midpoint object is removed only when it is a non-blue bud of the same color/type as the jumping bud. This gives the important color behavior:

- Green over green removes the jumped green bud.
- Red over red would remove the jumped red bud, although the official levels use red mainly as a reusable piece.
- Red over green, green over red, blue over anything, and anything over blue do not remove the jumped bud.
- Purple posts are never removed.

Blue buds are movable and jumpable, but they are never removed by being jumped over and are ignored by the win count in the blue-bud levels.

## Platform Rules

Yellow platforms live on black rails. A direction action moves all platforms one cell in that direction if the next cell contains rail and is not occupied by another platform. Platform chains move in travel order, so adjacent platforms can move together when the leading platform has room.

A moving platform carries every non-rail object currently on its cell: buds, purple posts, and the platform tile itself. This is how the game transports buds and posts between otherwise disconnected floor islands.

Selecting a bud shows gray landing markers around that bud. Any platform move clears the current selection. Clicking a selected landing marker performs the jump. Clicking elsewhere usually clears the selection or makes the selected object wobble without changing the puzzle state.

## Reset, Undo, And Failure

The puzzle supports undo. Undo restores the previous stored game state, including removed buds, platform positions, carried objects, the camera position, and the current selection state.

Reset restores the level to its initial state. Several known dead positions also show a reset key at the bottom-left of the interface; clicking that key resets the level. When this prompt appears, the remaining active buds briefly turn gray.

For exact runtime behavior, there is also an internal action cap:

- Level 1 loses after 64 counted actions.
- Levels 2 through 5 lose after 320 counted actions.
- Levels 6 through 10 lose after 640 counted actions.

The maps below are logical cell maps. The rendered game shows a 64x64 viewport and pans the board in later levels, but the full logical map should always exist.

## Map Legend

```text
space  void / no cell
.      fixed white floor
x      green bud on floor
r      red bud on floor
b      blue bud on floor
p      purple fixed post on floor
,      empty movable platform on horizontal rail
;      empty movable platform on vertical rail
?      empty movable platform on vertical rail
P      movable platform carrying a purple post
D      movable platform carrying a purple post on vertical rail
7      movable platform carrying a blue bud on vertical rail
-      horizontal rail
|      vertical rail
< >    rail end/cap graphics
L 3    rail corner graphics
T t    rail junction graphics
```

The rail glyphs are visual track cells. They become jump landings only when an empty platform is currently on that rail cell.

## Win Counts

Levels 1 through 5 are won when only one counted bud remains. Levels 6 and 7 are won when two counted buds remain, normally the red bud and one green bud. Levels 8 through 10 are won when only one green bud remains; blue buds do not count toward this target.

## Level 1

```text
x:  0123456
00 |       |
01 |.......|
02 |.xx.x..|
03 |.....x.|
04 |    ...|
05 |    .x.|
06 |    ...|
07 |    ...|
```

Level 1 is the basic jump-elimination tutorial. It is a 7x8 L-shaped white-floor board with five green buds at `(1,2)`, `(2,2)`, `(4,2)`, `(5,3)`, and `(5,5)`. There are no rails, platforms, purple posts, red buds, or blue buds.

The player clicks a green bud, sees every legal jump landing two cells away, and clicks a highlighted landing. Each green-over-green jump removes the jumped bud. The level is won by reducing the five green buds to one green bud.

Important reconstruction details:

- Only same-color green jumps remove pieces.
- The first level gives extra click feedback and reset hints if the player gets stuck or clicks around before understanding selection.
- Landing on some early dead cells triggers the reset prompt, especially landings at `(0,2)`, `(2,2)`, and `(5,1)`.

## Level 2

```text
x:  012345678
00 |.......  |
01 |.xx.x.x->|
02 |....... ||
03 |        ||
04 | <--,---3|
05 | |       |
06 | |    .. |
07 | L----x. |
```

Level 2 adds one movable platform. The upper white-floor island contains four green buds at `(1,1)`, `(2,1)`, `(4,1)`, and `(6,1)`. A small lower-right island contains one green bud at `(6,7)`. The empty platform starts at `(4,4)` on a rectangular rail loop connecting the top area to the lower island.

The player must use directional movement to slide the platform along the rail, jump a bud onto or across the platform, and use the platform as a transport bridge between separated islands. The level is won by reducing the five green buds to one.

Important reconstruction details:

- Direction actions move the platform along the black rail. The platform is the only landable rail cell.
- A bud riding on the platform moves with it.
- The upper-row positions `(0,1)`, `(2,1)`, and `(4,1)` are treated as dead landings and can trigger the reset prompt.

## Level 3

```text
x:  01234567890123
00 |.. ..     ..  |
01 |.x .x-,--x..  |
02 |.x.x      .xx.|
03 |..x.      ..  |
04 |          .xx.|
05 |      <-> ..  |
06 |      | | x.. |
07 |.x.x-,3 L-.x  |
08 |...       ..  |
```

Level 3 expands the board and adds a second platform. The green buds start at `(1,1)`, `(4,1)`, `(9,1)`, `(1,2)`, `(3,2)`, `(11,2)`, `(12,2)`, `(2,3)`, `(11,4)`, `(12,4)`, `(10,6)`, `(1,7)`, `(3,7)`, and `(11,7)`. The two platforms start at `(6,1)` and `(5,7)`.

Both platforms respond to every direction action. The player has to align them with their rails to bridge the upper-left group, lower-left group, and right-side group. The basic rule is still green-over-green elimination, but platform timing matters because the islands are disconnected.

The level is won by reducing fourteen green buds to one.

Important reconstruction details:

- The upper platform travels on the rail segment across row 1, while the lower platform travels around the lower rail segment that reaches toward the right-side cluster.
- Direction actions attempt to move both platforms at once; one may move while the other stays blocked.
- Several right-edge or top-edge landings are known dead states and show the reset prompt: `(1,0)`, `(0,3)`, `(10,0)`, `(13,2)`, `(13,4)`, plus `(10,2)` and `(10,4)` once enough greens have already been removed.

## Level 4

```text
x:  012345678901234567
00 |                  |
01 |                  |
02 |.......           |
03 |.xp.p.x-,---x...x.|
04 |.......     .p..p.|
05 |            ......|
06 |            .p..p.|
07 |        <---,..p..|
08 |        |         |
09 |    .x. |         |
10 |   .p p.p..       |
11 |   .----,-.       |
12 |   .p p....       |
13 |    .x.           |
```

Level 4 introduces purple posts. The six green buds are at `(1,3)`, `(6,3)`, `(12,3)`, `(16,3)`, `(5,9)`, and `(5,13)`. Twelve purple posts sit on floor cells throughout the top-right and lower-left regions. The platforms start at `(8,3)`, `(12,7)`, and `(8,11)`.

Purple posts are permanent jump hurdles. They can be jumped over to reposition a bud, but they do not disappear. This makes the level less about simply consuming every adjacent piece and more about using posts and platforms to create legal two-cell jumps.

The level is won by reducing the six green buds to one.

Important reconstruction details:

- Purple posts are not selectable pieces. They are obstacles/supports for jumps.
- A platform can be moved under or with objects and can be reused multiple times to connect the top strip, right room, and lower room.
- Some successful jumps pan the viewport to reveal more of the wide board, especially jumps that land on the platform corridor around `(7,3)` and `(12,7)`.

## Level 5

```text
x:  012345678901234567890
00 | -,-T-P-      ..p..x |
01 |    |        <---T-P>|
02 | ...|  <--> .|...|. ||
03 | xp.|.x|  Lx.|.p.|..||
04 | ...|  |    .|..p|x.||
05 |    |  |     L---t--3|
06 |  <-t--3       .     |
07 |  |            p     |
08 |  |           ..x    |
```

Level 5 uses a larger rail network and puts jumpable posts on platforms. The six green buds are at `(19,0)`, `(1,3)`, `(6,3)`, `(11,3)`, `(18,4)`, and `(16,8)`. There are seven purple posts: five fixed on floor cells and two initially riding platforms. The platforms start at `(2,0)`, `(6,0)`, and `(19,1)`. The platforms at `(6,0)` and `(19,1)` begin with purple posts on them, so they are moving jump hurdles rather than empty landing cells.

The player must manipulate the rail network so platforms and post-carrying platforms create useful midpoints and landing cells. The puzzle connects a left floor island, upper rail corridor, center-right rooms, and the bottom-right green bud.

The level is won by reducing the six green buds to one.

Important reconstruction details:

- A platform carrying a purple post moves both the platform and the post.
- A platform with a post is not a legal landing, but the post can be jumped over.
- The camera pans after important jumps near `(7,3)` and `(12,3)`, allowing the player to continue across the larger horizontal board.

## Level 6

```text
x:  0123456789012345678901234567
00 |                            |
01 | ....         ....          |
02 | .r..         .x.........>. |
03 | .x..         p..p     p |x |
04 | ....         |  |    .?.|. |
05 | ......       |  |    x| |  |
06 | ......,,-----t--3  ...L-3  |
07 | x.....             x       |
08 |   x                .       |
```

Level 6 introduces the red bud. The red bud starts at `(2,2)`. The green buds start at `(15,2)`, `(2,3)`, `(26,3)`, `(22,5)`, `(1,7)`, `(20,7)`, and `(3,8)`. Purple posts start at `(14,3)`, `(17,3)`, and `(23,3)`. The platforms start as a two-platform pair at `(7,6)` and `(8,6)`, plus a vertical-rail platform at `(23,4)`.

The red bud is a reusable mover because red-over-green jumps do not remove green buds. This lets the player cross gaps and set up green-over-green removals without consuming every midpoint. The level is not won by reaching one counted bud; it is won when two counted buds remain, normally the red bud and one green bud.

Important reconstruction details:

- Red and green are different removal types. Jumping one over the other moves the jumper but leaves the midpoint bud intact.
- The adjacent platforms on row 6 can travel as a coupled pair along the long rail and carry pieces into the central/right sections.
- A green landing at `(16,2)` is treated as a dead state if the red bud is sitting at `(6,6)`.
- Jumps around `(7,6)` and `(18,2)` pan the viewport to expose later parts of the wide map.

## Level 7

```text
x:  01234567890123456789012
00 |                 <---> |
01 |x     r    <-->  |   | |
02 |p     p    |  ; <tTPTt>|
03 |L--T--3  <-t-p. | D D ||
04 |   |     |    .   | | ;|
05 | <-t-,> <t->  .p.p. | .|
06 | |    | |  |  .p.p.   x|
07 | p    p L->p           |
08 | .p.p.....3.           |
```

Level 7 is a dense rail-and-post puzzle. The red bud starts at `(6,1)`. The two green buds start at `(0,1)` and `(22,6)`. There are fifteen purple posts across the left, lower, and right regions. Six platforms start at `(14,2)`, `(19,2)`, `(18,3)`, `(20,3)`, `(22,4)`, and `(5,5)`. Three of those platforms, at `(19,2)`, `(18,3)`, and `(20,3)`, begin with purple posts on them.

Only one green bud needs to be removed because the target count for this level is two counted buds. The challenge is not the final count; it is moving the rail platforms and carried posts into a configuration where the red bud and the two green buds can interact.

Important reconstruction details:

- The right-side rail loop has multiple platforms and post carriers. Direction actions can move several of them at once.
- The lower-left and lower-middle rails are used to shuttle posts and create jump midpoints.
- A jump landing at `(8,8)` pans the board from the initial view toward the right. Later, landings with `x = 16` pan again when the board has already shifted.
- The final portion requires arranging the movable posts/platforms before the last green-over-green removal is possible.

## Level 8

```text
x:  012345678
00 |         |
01 |         |
02 | ........|
03 | xp...p..|
04 | ......p.|
05 |<-p...p..|
06 ||...b....|
07 ||...b...x|
08 ||       ||
09 |L-,>   <3|
10 |   |   ; |
11 | ......bb|
12 | L--P,P-3|
```

Level 8 introduces blue buds as non-counted pieces. The green buds start at `(1,3)` and `(8,7)`. The blue buds start at `(4,6)`, `(4,7)`, `(7,11)`, and `(8,11)`. Purple posts start at `(2,3)`, `(6,3)`, `(7,4)`, `(2,5)`, `(6,5)`, `(4,12)`, and `(6,12)`. Five platforms start at `(2,9)`, `(7,10)`, `(4,12)`, `(5,12)`, and `(6,12)`.

The obvious route across the main white room is a bait route; the green bud on the far right is not reached cleanly by going straight at it. The player must use the left and bottom rail loop, the platforms, and blue buds as persistent supports to approach indirectly.

The level is won when only one green bud remains. The blue buds may move and may be jumped over, but they never count toward the win target and are not removed.

Important reconstruction details:

- Blue buds are selectable and movable like other buds, but blue is non-removing.
- The bottom platforms include post-carrying platforms, so the bottom rail can alternate between being a bridge, a blocker, and a jump-hurdle generator.
- Direction movement on this level can pan the board vertically when a green bud rides a platform.

## Level 9

```text
x:  01234567890123456789012
00 |                       |
01 |           x..p.p......|
02 |  ..b..    .........bb.|
03 |  ...b.    .p.....p.p..|
04 |  .....              ..|
05 |  ....,--------------..|
06 |  xb..                 |
07 |  .b..x                |
```

Level 9 hides much of the useful route off to the right at the start. The three green buds are at `(11,1)`, `(2,6)`, and `(6,7)`. The six blue buds are at `(4,2)`, `(20,2)`, `(21,2)`, `(5,3)`, `(3,6)`, and `(3,7)`. Purple posts are at `(14,1)`, `(16,1)`, `(12,3)`, `(18,3)`, and `(20,3)`. The single platform starts at `(6,5)` on a long horizontal rail extending rightward.

The player must use the platform to reveal and access the right side of the map. The puzzle then requires recovering or repositioning a piece so that the green buds can be reduced while the blue buds remain as reusable supports.

The level is won when only one green bud remains.

Important reconstruction details:

- The platform at `(6,5)` is the main camera and transport trigger. A jump landing on it pans the board toward the right.
- Blue buds on the left cluster and far-right cluster should persist and can be used as midpoints without changing the green win count.
- Because there is only one platform, moving it at the wrong time can strand the route and require undo or reset.

## Level 10

```text
x:  012345678
00 |   .x.   |
01 |<-T-T-T->|
02 || ; ; ; ||
03 || L-t-3 ||
04 ||       ||
05 || ...bb ||
06 |L--b... ||
07 |  ..b.. ||
08 |  b.... ||
09 |  ....x 7|
10 |        7|
11 |        7|
12 |        7|
13 |        7|
```

Level 10 is the final large platform puzzle. The two green buds start at `(4,0)` and `(6,9)`. Ten blue buds are present: five in the central/lower white-floor room at `(5,5)`, `(6,5)`, `(3,6)`, `(4,7)`, and `(2,8)`, plus five blue buds riding the vertical right-side platform column at `(8,9)` through `(8,13)`. The empty top platforms start at `(2,2)`, `(4,2)`, and `(6,2)`. The right-side blue platform column starts at `(8,9)`, `(8,10)`, `(8,11)`, `(8,12)`, and `(8,13)`.

The main problem is arranging the top platforms and the right-side blue-carrying platform column so the two green buds can be brought into a valid green-over-green jump relationship. The blue pieces are blockers and supports, not targets.

The level is won when only one green bud remains.

Important reconstruction details:

- The right-side vertical rail has a full column of platforms carrying blue buds. Direction movement shifts that column as a group when the leading platform has rail space.
- The top three platforms provide temporary landings and can also block the central route if left in the wrong positions.
- Once the platforms and blue supports are out of the way, the final win condition is the same as the other green-count levels: one green bud must be removed by a green-over-green jump.
