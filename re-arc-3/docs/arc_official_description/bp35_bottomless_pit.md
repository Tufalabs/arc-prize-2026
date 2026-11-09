# Bottomless Pit

## Core Game

Bottomless Pit is a side-view falling maze. The player occupies one cell and can move only left or right. After every horizontal move into a passable cell, gravity is resolved immediately: the player keeps falling in the current gravity direction until the next blocking cell, the goal, or a trap. The initial gravity direction is downward in all levels. Red gravity switches reverse it between downward and upward.

The player wins a level by reaching the goal tile. The player loses by falling onto a trap mouth, by running out of the level's turn meter, or in the early levels by letting the descending ceiling band catch up. Reaching the goal advances to the next level.

The visible playfield is a 64x64 render area over a taller 11-column map. Cells are 6x6 pixels. The camera is fixed horizontally and scrolls vertically after falls and gravity changes. Because the map is taller than the viewport, later levels require remembering terrain and interactable blocks that have scrolled off-screen.

## Objects and Actions

The player has four meaningful actions: move left, move right, click a visible special block, and undo. Reset returns the current level to its initial state and clears the turn meter.

Movement works as follows:

1. Choose the adjacent left or right cell.
2. If that cell is solid, the player bumps in place.
3. If that cell is air or an inactive orange marker, the player enters it.
4. If the cell beyond the player in the current gravity direction is air or an inactive orange marker, the player falls through consecutive passable cells.
5. If the fall reaches the goal, the level is won.
6. If the fall reaches a trap, the level is lost.
7. Otherwise the player stops on the last passable cell before the blocking support.

Clicks affect visible special blocks at their clicked map cell. A click consumes a turn even if it does not produce useful movement.

- Green breakable block: solid until clicked. A click shrinks and removes the block permanently for the current attempt.
- Active orange block: solid until clicked. A click turns it into an inactive orange marker.
- Inactive orange marker: passable as air. A click turns it back into a solid active orange block.
- Red gravity switch: solid and one-use. A click removes the switch and reverses gravity. After the reversal, the player immediately falls if no support exists in the new gravity direction.
- Purple builder block: solid until clicked. A click removes the clicked block and creates new purple builder blocks in each empty orthogonal neighbor cell. It never fills diagonals and does not overwrite occupied cells.
- Goal: entering it by movement or fall wins.
- Trap mouth: falling onto it loses. Trap orientation is visual; upward- and downward-facing traps are both lethal.

After clicking a non-switch block, gravity is rechecked only if the clicked cell was immediately in the current gravity direction from the player. Remote edits are allowed, but they do not move the player until a later move or support-removal click makes the player unsupported.

Undo restores the previous full level state, including player position, gravity, removed blocks, toggled orange blocks, generated purple blocks, consumed switches, and camera position. Undo itself still consumes one turn.

## Map Legend

Rows are printed top to bottom, and columns are printed left to right. Initial gravity points toward larger row numbers.

```text
# solid cyan terrain
. empty air
P player start
G goal
B green breakable block
A active orange solid block
a inactive orange pass-through marker
R red one-use gravity switch
Y purple builder block
V downward-facing trap mouth
U upward-facing trap mouth
W upper descending ceiling band
M lower descending ceiling band
```

Levels 1-3 include the `W`/`M` ceiling hazard. When the turn count is even and a horizontal action is resolved without a fall, including a blocked bump, the ceiling bands move one row downward. If the lower band reaches the player's row, the level fails. These levels also have the normal 64-turn meter, so they reward opening a clean route rather than repeatedly probing.

Levels 1-6 have a 64-turn limit. Levels 7-9 have a 128-turn limit. The turn meter advances on horizontal moves, clicks, and undo.

## Level 1

Level 1 teaches horizontal movement, gravity, remote block breaking, and the goal. The player starts at column 3, row 12. The goal is far below at column 3, row 28. There are no traps or gravity switches. The only mutable terrain is the green breakable blocks.

To win, the player must move across supports, break green blocks to open vertical drops, and use the offset landings to line up with the goal chamber. Breaking the wrong support can drop the player past useful landings, but there is no lethal trap other than the descending ceiling and turn limit.

```text
    01234567890
00  WWWWWWWWWWW
01  WWWWWWWWWWW
02  WWWWWWWWWWW
03  WWWWWWWWWWW
04  WWWWWWWWWWW
05  MMMMMMMMMMM
06  ##.......##
07  ##.......##
08  ##.......##
09  ##.......##
10  ##.......##
11  ##.......##
12  ##.P.....##
13  #######.###
14  ##.......##
15  ##.......##
16  #####BBBB##
17  #####....##
18  ##BBB....##
19  ##BBB....##
20  ##BBB######
21  ##.......##
22  ##.......##
23  ##..BBB..##
24  ##.......##
25  ##.......##
26  #####BBB###
27  ##.......##
28  ##.G.....##
29  ###########
30  ###########
31  ###########
32  ###########
33  ###########
34  ###########
35  ###########
```

## Level 2

Level 2 adds trap mouths. The player again starts at column 3, row 12, with the goal near the bottom at column 5, row 42. Green breakable blocks form many tempting vertical shafts, but several shafts end on `V` traps.

To win, the player must break a route that lands on solid cyan terrain or safe empty cells, not on a trap row. The dangerous lesson is that an open vertical tunnel is not automatically a path; if the first blocker in the gravity direction is a trap, the fall kills the player.

```text
    01234567890
00  WWWWWWWWWWW
01  WWWWWWWWWWW
02  WWWWWWWWWWW
03  WWWWWWWWWWW
04  WWWWWWWWWWW
05  MMMMMMMMMMM
06  ##.......##
07  ##.......##
08  ##.......##
09  ##.......##
10  ##.......##
11  ##.......##
12  ##.P.....##
13  ##BBBBBBB##
14  ##BBBBBBB##
15  ##...#...##
16  ##...#...##
17  ##VVV#...##
18  ######...##
19  ##BBBB...##
20  ##BBBB...##
21  ##B###BBB##
22  ##.......##
23  ##.......##
24  ##....VVV##
25  #####B#####
26  #####B#####
27  ##.......##
28  ##......V##
29  ##BBB######
30  ##.......##
31  ##V.VVVVV##
32  ###B#######
33  ###BBBBBB##
34  ###B####B##
35  ###B####B##
36  ##.......##
37  ##VVV....##
38  #####....##
39  ##.....BB##
40  ##BBBBBBB##
41  ##.......##
42  ##...GVVV##
43  ###########
44  ###########
45  ###########
46  ###########
47  ###########
48  ###########
49  ###########
```

## Level 3

Level 3 introduces two-state orange blocks. The player starts at column 3, row 12, and the goal is at column 7, row 33. `A` cells are solid orange supports. `a` cells are inactive markers that can be passed through and clicked back into solid supports.

The level requires both removing and restoring platforms. Some orange blocks must be turned off to open a fall. Other inactive orange markers must be turned on to create landings above trap rows. The green blocks at row 13 are still simple breakable blockers.

```text
    01234567890
00  WWWWWWWWWWW
01  WWWWWWWWWWW
02  WWWWWWWWWWW
03  WWWWWWWWWWW
04  WWWWWWWWWWW
05  MMMMMMMMMMM
06  ##.......##
07  ##.......##
08  ##.......##
09  ##.......##
10  ##.......##
11  ##...A...##
12  ##.P.A...##
13  ######BBB##
14  ##.......##
15  ##.......##
16  ##.......##
17  ##.aaa#####
18  ##.......##
19  ##.VVVV..##
20  ##.####..##
21  ##.......##
22  ###..AA..##
23  #####aa..##
24  #####VV..##
25  #######..##
26  ##.......##
27  ##.......##
28  ##AAaaa####
29  ##.....####
30  ##..VVV####
31  ##..#######
32  ##...A...##
33  ##...A.G.##
34  ##AAAAAAA##
35  ##.......##
36  ##VVVVVVV##
37  ###########
38  ###########
39  ###########
40  ###########
```

## Level 4

Level 4 removes the descending ceiling and introduces one-use gravity switches. The player starts at column 4, row 24. The goal is above the start at column 4, row 11, so the player cannot finish with downward gravity alone. Red switches appear at rows 7, 15, and 31.

To win, the player must click red switches at the right times to alternate between falling downward and falling upward. Every red switch disappears after use, so an early or unnecessary reversal can strand the player. Green blocks still open routes, and the trap pair at row 24 punishes falling into the start-side shaft incorrectly.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  ###########
05  ###########
06  ###########
07  ####R######
08  ###########
09  ##.......##
10  ##.......##
11  ##..G....##
12  ##.......##
13  ##VV.VV..##
14  #######BB##
15  ###R#R#BB##
16  #######..##
17  ##....#..##
18  ##.......##
19  ##...BBBB##
20  ##.......##
21  ##BB#######
22  ##.......##
23  ##.......##
24  ##VVP....##
25  ######...##
26  #######..##
27  ########.##
28  ##.......##
29  ##.......##
30  ###########
31  #####R#####
32  ###########
33  ###########
34  ###########
35  ###########
36  ###########
37  ###########
38  ###########
```

## Level 5

Level 5 is a larger gravity-switch route. The player starts at column 3, row 26, next to a red switch at column 8, row 26. A second red switch is near the upper-right at column 8, row 9. The goal is below the start-side route at column 5, row 31.

The level combines downward traps (`V`), upward traps (`U`), inactive orange markers, breakable green supports, and two distant gravity switches. Winning requires traveling right to use the nearby switch, navigating upward and downward through the central chambers, reaching the distant switch, and returning through a reshaped path to the goal. The two trap orientations matter visually because the player will be moving under both gravity directions.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  ###########
05  ###########
06  ###########
07  ###########
08  ###########
09  ########R##
10  ###UU######
11  ###..######
12  ###BB######
13  ###..######
14  ###.....###
15  ##........#
16  ##........#
17  ##.aa###BB#
18  ##.VV#BB..#
19  ##.###....#
20  ##.###..VV#
21  ##......###
22  ######BB###
23  ######..###
24  ##UUUU..###
25  ##......###
26  ##.P....R.#
27  #########.#
28  #########.#
29  ####...BBB#
30  ####...BBB#
31  ####.G.VVV#
32  ###########
33  ###########
34  ###########
35  ###########
36  ###########
37  ###########
38  ###########
```

## Level 6

Level 6 hides the goal and several required interactables outside the initial viewport. The player starts at column 3, row 15. The goal is near the upper-left at column 2, row 7. Red switches appear at column 4 row 7, column 6 row 16, and column 8 row 37.

There are no green breakable blocks in this level. The route is instead built around gravity reversals and orange toggles. The large inactive-orange cluster around row 25 is a timing and platform-planning puzzle: toggling the wrong cells can remove the supports needed to reverse direction safely. The bottom switch at row 37 forces the player to traverse far from the visible goal before the final approach.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  ###########
05  ###########
06  ###########
07  #.G.R....##
08  #...#....##
09  ########.##
10  ########.##
11  ########.##
12  ##UUUUUU.##
13  ##....aa.##
14  ##.......##
15  ##.P.....##
16  ######R####
17  ##.......##
18  ##aaa###.##
19  ##.....#.##
20  ##VVV....##
21  ######.####
22  ######.####
23  ##UUUU...##
24  ##.......##
25  ##aaaaAaa##
26  ##.......##
27  ##.......##
28  ####.VVVV##
29  ####.######
30  ####.....##
31  #######..##
32  ###########
33  ###########
34  ###########
35  ###########
36  ###########
37  ########R##
38  ###########
```

## Level 7

Level 7 is a compact maze with a long column of one-use red gravity switches along the left side. The player starts at column 3, row 12. The goal is directly above in the same column at row 6, but walls, orange toggles, and traps block the direct route.

The switch column lets the player flip gravity repeatedly, but each cell in the column can be consumed only once. The central maze has many inactive orange markers that can be restored into supports, plus a few active orange blocks that can be removed. To win, the player must alternate gravity, reshape orange platforms, and avoid dropping into the `U` and `V` traps in the center and lower-right.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  R#...######
05  R#...##...#
06  R#.G.##.#.#
07  R#......U.#
08  R#........#
09  R#######..#
10  R#....a#..#
11  R#........#
12  R#.P..a#..#
13  R#######..#
14  R#.UaaaU..#
15  R#..aaa...#
16  R#..aaa...#
17  R#.#aaa.#.#
18  R#.#aaa.#.#
19  R#.######.#
20  R#.a.U.U#.#
21  R#.a.A.a#.#
22  R#.aa.Aa#.#
23  R#.aV.Va#.#
24  R######a..#
25  R######a..#
26  R######a#.#
27  #######a#V#
28  ###########
29  ###########
30  ###########
31  ###########
```

## Level 8

Level 8 introduces purple builder blocks. The player starts at column 3, row 6. The goal is at column 9, row 19. A red gravity switch is far below at column 5, row 36. Purple builders start at column 2 row 9 and column 3 row 20.

The first builder teaches that clicking a purple block creates new purple blocks in empty adjacent cells. The lower half of the map contains a wide trap funnel and no safe direct descent to the red switch. The intended reconstruction challenge is to use builder blocks to make protective landings and eventually a staircase or chain of supports that reaches the bottom switch. After the gravity reversal, the player can use the rebuilt route and orange platforms to approach the goal from the correct side.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  #.........#
05  #.........#
06  #..P......#
07  #####...###
08  #.........#
09  #.Y.......#
10  #.........#
11  #VVVVVVV..#
12  ########..#
13  #......A..#
14  #......####
15  #.........#
16  #AAAAAAAAA#
17  #.........#
18  #......####
19  #......#.G#
20  #..Y...#A##
21  #.........#
22  #.........#
23  #.........#
24  #.........#
25  #.........#
26  #.........#
27  #.........#
28  #V.......V#
29  ##V.....V##
30  ###V...V###
31  ####...####
32  ####VVV####
33  ###########
34  ###########
35  ###########
36  #####R#####
37  ###########
38  ###########
```

## Level 9

Level 9 combines all late-game mechanics. The player starts at column 3, row 10. The goal is near the top-left at column 2, row 5. Red switches are scattered on the left edge and as a long strip near the bottom at row 44. Purple builders are at column 6 row 14 and column 7 row 37. Green breakable blocks appear at rows 19, 32, and 37. A long active orange bridge spans row 30.

Winning requires building and deleting supports across multiple vertical passes. The player must use red switches to change gravity several times, use the purple builders to create safe landings near the lower passages, toggle or preserve the orange bridge as needed, and break green blocks only when the opened fall will land safely. The final approach to the goal happens near the top-left, so the bottom switch strip and lower builder are preparation for returning upward without falling into the trap rows at 18, 20, 26, 28, and 39.

```text
    01234567890
00  ###########
01  ###########
02  ###########
03  ###########
04  ###########
05  ..G.#######
06  ....#######
07  .##########
08  .#........#
09  .#........#
10  R#.P......#
11  .####...###
12  R#........#
13  .#........#
14  .#....Y...#
15  .#........#
16  .#........#
17  .#........#
18  .#...VVVVV#
19  .#BBB######
20  R#...#UUUU#
21  .#...#....#
22  .#...#....#
23  .#...#..#.#
24  R#......#.#
25  .#......#.#
26  R#VVVVVV#.#
27  .########.#
28  .#UUUUUUU.#
29  .#........#
30  R#AAAAAAAA#
31  .#........#
32  .#..B.....#
33  .#........#
34  .#........#
35  .#........#
36  .#........#
37  .B.....Y..#
38  .#........#
39  ##VVVVVVVV#
40  ###########
41  ###########
42  ###########
43  ###########
44  #RRRRRRRRR#
45  ###########
```

## Rebuild Checklist

- Use a one-cell player on an 11-column, variable-height grid with a 6x6 pixel tile size.
- Start every level with downward gravity.
- Resolve gravity immediately after horizontal moves and after gravity-switch clicks.
- Allow clicks on visible special blocks, not only adjacent blocks.
- Persist all terrain mutations until undo or reset.
- Make red switches one-use and remove them when clicked.
- Make inactive orange markers passable but still clickable.
- Make purple builders copy themselves into empty orthogonal neighbors only.
- Treat both trap orientations as lethal landing surfaces.
- Include the descending ceiling behavior only in levels 1-3.
- Enforce a 64-turn limit in levels 1-6 and a 128-turn limit in levels 7-9.
- Implement undo as a full state rewind that still consumes a turn.



  