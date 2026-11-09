# Escape Velocity

*(The name is the game's one law hiding in plain sight — **you leave a portal at the velocity you entered it**, a step stays a step, a flight stays a flight — and its recurring deed: every level is an escape.)*

> This is a game about walking through doorways you place yourself, becoming difficult because portals only stick to white walls and you exit a portal only as fast as you entered it.

A **top-down**, grid-based puzzle game. The player carries a visible portal gun and walks in four directions, but cannot cross water — stepping in means sinking with a splash. The only way across is to link two points in space with a blue/orange portal pair, fired from the player's own body: a click sends a bolt **at exactly the angle needed to land where you clicked**, and the agent visibly turns to face the shot. Level 2 adds an accelerator pad — flight entering a portal exits as flight and sails over the water, while walking in exits as a single step — and a pressure button that holds a door open only while something stands on it. A time bar at the bottom edge drains by one unit per action; it is generous (6× the intended solution), but it makes dithering expensive. Level 3 composes everything: the box doubles as a portable portal surface — and, dropped into a flight line with portals on two perpendicular faces, as a **flight elbow** that turns the pad's ride in mid-air. Level 4 turns the systems on their heads: doors can be mid-level **gates** (and bolts pass open doors, so a button becomes a *firing position*), and pads launch **whatever rests on them** — including the box, fired through a portal pair as artillery. Level 5: **escape a sealed cell through your own parcel**, then send the box on a long pinball journey across the map — and mail yourself after it. Level 6 is the graduation exam: every system at once, across five zones, ending with the player **holding a gate open for their own parcel's flight**.

**This game has exactly 8 levels.**

---

## 1. Canvas, grid, and coordinate system

- Render canvas: **64×64 pixels**.
- Logical gameplay grid: **16×16 cells**, each cell rendered as a **4×4 pixel block**. Top-down view: row 0 is the north edge, +y is south.
- Coordinates written `(col, row)`: `col` 0–15 west→east, `row` 0–15 north→south. Cell `(c, r)` occupies pixels `x ∈ [4c, 4c+3]`, `y ∈ [4r, 4r+3]`.
- The player, the box, portals, the pad, the button, and door segments each occupy exactly one logical cell (rendered as detailed 4×4 sprites, not flat squares — see §8).

### Tile types

| Tile | Color ID | Category | Portalable |
|---|---|---|---|
| Water | 10 (light blue) | sink = death (splash) | — |
| Floor | 2 (gray) | walkable | no (bolts pass over it) |
| White wall | 1 (light gray) | solid | **yes** |
| Dark wall | 4 (very dark gray) | solid | **no** |
| Accelerator pad (floor marking) | 6 (magenta) + white chevrons | walkable, triggers launch | no (bolts pass over it) |
| Pressure button (floor marking) | 8 (red) + maroon corners | walkable, holds the door open | no (bolts pass over it) |
| Door segment (closed) | 9 (dark blue), solid fill + keyhole | solid | no (bolts fizzle) |
| Door segment (open) — also the level-1 exit | 9 (dark blue) frame with a dark opening | walkable (enter = win) | — |
| Box | yellow core in a light-gray rim (the portalable gray) | walkable (stand on it; SPACE lifts it); **blocks bolts** | **yes — per face, fully two-way** |
| Portal (either color) | 9 (blue) / 12 (orange) | solid wall cell | — |

Three visual rules taught by play: **white walls take portals, dark walls refuse them**; **light-blue cells are water — step in and you sink**; **the dark-blue door is the goal — open shows a dark opening you can walk into, closed shows a solid slab with a keyhole**. The exit door is the *same object in both levels*: level 1's stands open, level 2's starts closed. The wall tones are deliberately far apart in the palette (1 vs 4) so the distinction reads at a glance; floor (2) sits between them and is never a bolt target. Floor markings (pad, button) are walkable and never block bolts; neither do open door cells.

---

## 2. Action space

Uses the full movement set plus SPACE and click: `UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, **click**. **Every action costs one unit of the time bar (§6), including bumps, toggles, and missed shots.**

| Action | Semantics |
|---|---|
| `UP` / `DOWN` / `LEFT` / `RIGHT` | Walk one cell north / south / west / east, and turn to face that direction. Walking into a solid cell bumps (flash, no move) — except a linked portal's **marked face**, which is entered by moving against its normal (§4). Walking onto the box just stands on it (no auto-pickup). Walking into a water cell means sinking: splash, death, respawn (§6). |
| `SPACE` | Context-sensitive, in priority order: **carrying the box → set it down**: if **any adjacent cell is a pad**, the box is fed onto it and launches immediately (the faced pad wins if several are adjacent, then N/E/S/W order); otherwise it is set underfoot on the cell you stand on; **standing on the box → pick it up**; **otherwise → toggle the armed portal color** (blue ↔ orange; level start: blue) — *if the level grants the gun*. The armed color shows **only on the player's gun** — the 2-pixel barrel centered on the sprite's facing edge (§8); gun-less tutorial levels render no barrel and SPACE is purely the box verb. Consequence: you cannot switch portal colors while holding or standing on the box — color management happens before you lift it. |
| Click on cell `(cx, cy)` | *(only in levels with the gun)* Fire a portal bolt of the armed color **from the player's cell, at exactly the angle toward the clicked cell** (§3). The player does not move, but **turns to face the shot** — shooting behind you spins the agent around, gun pointing in the bolt's general direction. The bolt is animated as a traveling projectile (§3.6). |

State updates only inside `step()` after an action. Each action resolves its **entire physics cascade** (bolt travel, launch → portal transit → landing, splash → respawn, door open/close, §7) deterministically within that one step; the cascade is rendered as an animation, and any input arriving mid-cascade is dropped.

---

## 3. Firing portal bolts

**A portal appears exactly where your aim line meets a portalable surface — never past it, never beside it.** Predictability is the contract (player feedback drove this rule, twice).

1. The bolt travels along the straight line from the player's cell center toward the clicked cell center, visiting exactly the cells the segment crosses (deterministic DDA; exact corner crossings step diagonally). It passes freely over floor, water, pads, buttons, goal tiles, and open-door cells.
2. Clicking the player's own cell is a no-op (still costs 1 time unit).
3. The bolt stops at the **first portalable or solid cell on the line, up to and including the clicked cell**:
   - **White wall or the box** → the armed portal is placed there (relocating that color's previous portal; at most one of each exists). Clicking the wall cell itself or clicking floor *behind* it along the same line both land the portal where the line meets the surface — forgiving aim, never surprising placement.
   - **Dark wall or closed door** → the bolt visibly flies there and **fizzles** (2-frame outline flash in the bolt's color): the animation *is* the "no line of sight" feedback. Nothing is placed.
   - **Open clicked cell** (floor/water) with nothing in the way → the bolt flies to the click and fizzles there.
4. **The struck face is the face the bolt actually enters through** (the DDA entry side) — shoot a block's left side and the portal is on its left face, no matter the angle. A perfectly diagonal corner entry falls back to the vertical-preferred reverse-component rule. Occupancy is **per face**: only a face already holding the *other* color refuses (fizzle).
   **Edge-pixel aiming**: clicking one of the cell's border pixels *names the face you want* — bottom edge for the belly, left edge for the west side. The named face is honored whenever it plausibly looks toward the shooter (its direction opposes the bolt's travel) and is open and unoccupied; center clicks keep the automatic entry-face rule. This makes side-and-belly elbow shots forgiving: any position with line of sight works, no exact diagonals required.
5. **Same-surface skim**: when a shallow aim line clips a *neighboring cell of the very wall being clicked* (a rasterization artifact, not a real obstruction), the bolt skims along that surface and lands **on the clicked cell** — clicking a wall never puts the portal one cell off. If the clicked cell has no strikable face from your side (e.g. the far column of a thick block), the portal falls back to where the line first met the surface, with that entry face.
6. **Bolt animation**: a 2×2 dot of the armed color with a white glint, 2 cells per rendered frame; the player turns to face the shot on the input frame; placements appear (their face-strip *is* the feedback), fizzles flash. Inputs are dropped mid-animation.

## 4. Portals: linking, entering, exiting

- A portal renders as a **colored strip on the struck face only** — half the wall cell turns the portal's color on its normal side, the rest of the cell stays white. The whole block never recolors; on a free-standing block, only the side you shot (the side you will pass through) is marked.
- **Both portals may share one wall cell on different faces** (e.g. blue on the block's front, orange on its back). The motion direction selects which portal a move enters. With the pair on opposite faces of a 1-cell wall, the wall becomes a walkable pass-through — step in one side, step out the other.
- **The box takes portals too** — its light-gray rim is the same gray as portalable walls, and occupancy is per face like any wall (two adjacent faces can hold both colors; the shared corner splits the colors half-and-half, §8). Box portals are **fully two-way**, exactly like wall portals. One rule keeps it sane: **lifting the box wipes its portals** (surfaces don't travel). Two consequences worth knowing: the payoff is a *carried doorway* — drop the box anywhere you can stand, shoot it, and walk through in both directions; and the limitation is that **the box can never ferry itself** — to carry it through a portal you must lift it first, which destroys any portal riding on it.
- A portal whose partner color is not placed is **inert**: the strip alone, and its wall cell behaves as ordinary solid. When both portals exist they are **linked**: a small white pip lights on each strip's outer edge.

**Entering** a linked portal = moving into its wall cell **through the marked face only** — walking or flying against its normal. The colored strip shows exactly the enterable side; from every other side the cell is plain solid wall. A portal on one side of the center block cannot be entered from the other side.

**Exiting** is always through the partner's **normal side**, and preserves the entry speed class — this is the core rule:

- **Walk entry → step-out exit.** The player is placed in the exit portal's *front cell* (portal cell + normal), standing. One cell of free travel, no more. If the front cell is **water**, the player steps out over it and sinks: splash, death (§6).
- **Flight entry (pad launch, §5) → flight exit.** The player appears in the exit portal's front cell already flying along the exit normal, and the flight continues (over water, until it stops, §5).
- **Blocked exit:** if the exit portal's front cell is solid, the pair is blocked — entering does nothing (treated as solid wall, brief flash on both portals).

Portals are two-way and persistent: they survive player death and stay until relocated by a new bolt of their color. Teleporting plays a 2-frame flash on both portals; the state change commits on the first flash frame.

---

## 5. Objects

### Accelerator pad
A floor marking (magenta, two white chevron pixels showing its direction — magenta, not red, so it can never be confused with the button). When the player **walks onto it** (or a flight comes to rest on it), they are immediately **launched**: flight in the chevron direction, 1 cell per frame. Flight passes freely over floor, water, and markings; it continues until the next cell is solid — the player stops in the current cell — or the next cell is a linked portal entered through its marked face (flight continues from the partner's front cell along the partner's normal). When flight stops, the arrival cell is evaluated like a normal arrival (§7): stopping over water means sinking. The pad is irresistible — the player can never stand on it. A flight that chains more than 8 teleports is stopped as if blocked (engine safety; unreachable in the authored levels).

### Box
A 1-cell crate — a symmetric yellow core in a **light-gray rim, the same gray as portalable walls** (the rim is the hint: you can shoot it), clearly distinct from the flat red button. A level may contain **several boxes**: you carry one at a time, any box weights a button, and a flying box treats other boxes as solid (it parks beside them, never stacks). It is a *low* object: the player can walk onto it and stand on it, bolts and flights pass over it, and nothing happens automatically. **SPACE while standing on it picks it up**; **SPACE while carrying puts it down** on the cell the player occupies (§2). The carried box renders as a yellow bundle on the player sprite and travels through portals and flights. A grounded box can always be picked back up — placing it is never final. If the player dies while carrying it, the box resets to its level-start cell; a grounded box stays where it was left.

### Goal tiles, switch-gates, and buttons

**The goal and the doors are different things** (player feedback drove this split):

- **Goal tiles** (purple, white center) are the level's exit: step on one and the level is won. They are *never* switch-controlled and always visible — when a level wants to gate the goal, it puts a **door in front of it**, goal glowing behind.
- **Doors** are switch-gates only: a contiguous run of dark-blue wall cells, each wired to one red **button**. The circuit is live, re-evaluated after every arrival and every SPACE pickup/drop:

> **A door is open iff its button is weighted** — by the player standing on it, or by the box sitting on it.

- Open = blue frame around a dark opening, walkable (and bolts and flights pass through). Closed = solid blue slab (bolts fizzle on it). The opening/closing wipes play inside the move's cascade — the player sees the door react to their own weight.
- Carry the box onto a button and press **SPACE**: the box weights it for as long as it sits there. Nothing is permanent — lifting the box closes the door again.
- Buttons sit ≥2 cells from their door, and the circuit updates before any further movement, so "sprint through before it shuts" is structurally impossible.
- **The keyhole consequence**: bolts and flights pass through *open* doors — a gate you cannot keep open for your own walk can still be shot or flown through while something weights its button.

### Pre-placed portals and gun-less levels

A level may start with portals already on its walls (`portals` in the level file) and may withhold the portal gun entirely (`gun: false`): clicks then do nothing, SPACE never toggles, and the player sprite renders without the barrel. This is how the first levels teach portals as *doors you walk through* before handing over the gun that makes them.

## 6. Time bar, death, and respawn

### Time bar
- Each level starts with a **time budget**, graded by what the player already knows: tutorial levels get **3×** their intended solutions, the late insight levels **1.5×–2×**. Current ladder (budget / intended): 78/13 (the very first level gets 6× — room to simply get used to moving), 60/20, 120/40, 108/30, 201/134, 74/49, 66/33, 290/145 (the finale gets 2x: long, but every beat has been taught).
- **Every player action costs exactly 1 unit** — moves, bumps, toggles, shots, deaths included. Animation frames cost nothing.
- The bar is drawn along the bottom edge over the south wall (§8): a green fill on a black track that drains left to right. When it empties, the level **fails** (game over); resetting restores the level with a full bar. The bar also refills on level entry. Winning an action that lands exactly on 0 still counts as a win.

### Death and respawn
- Arriving in a water cell — by walking in, stepping out of a portal whose front cell is water, or a flight stopping over it — means **sinking**: a 3-stage splash animation plays on that cell (§8) while the player is hidden, then the player reappears at the level's spawn cell with a brief flash. **No lives — unlimited respawns**; only the time bar is a global constraint. Death is the cheap reset button.
- On respawn: portals **persist**, a grounded box stays where it was left (including on the button), but a **carried** box resets to its level-start cell. The time bar is *not* refilled by death. The door re-evaluates (nobody is on the button after a respawn, so a door weighted only by the player shuts).

---

## 7. `step()` resolution order

1. Deduct 1 time unit; if this was the last unit, the action still resolves fully and then the level fails unless it won.
2. Apply the action: `SPACE` drops the carried box / lifts a stood-on box / otherwise flips `armed_color` (gun recolors), re-evaluating the door when the box moves (§2, §5); click sets facing and launches the bolt (§3); a move action sets facing and attempts one cell — bump (solid), teleport (linked portal's marked face), or step.
3. Resolve the cascade loop, one frame at a time:
   a. **Bolt**: the projectile advances 2 cells per frame along the precomputed ray; on arrival the portal is placed or the impact flashes.
   b. **Flight**: advance 1 cell along the flight direction; next cell a linked portal's marked face → teleport to the partner's front cell, continue along the partner's normal; next cell solid (or off-grid, or >8 teleports) → flight ends in the current cell.
   c. **Arrival checks** when the player comes to rest in a cell, in order: **door re-evaluation** (open iff button weighted — wipe animation if it changed); water → splash animation, then respawn (§6); pad → launch (back to b); open door cell → **win level**. A winning arrival drops any pending flash frames and ends the action immediately (the engine's level transition must never replay the input on the next level).
4. A win on level 1 advances to level 2; a win on level 2 completes the game.

The cascade is finite by construction in both authored levels (no portal pair faces itself along any reachable flight line; the teleport cap is a backstop only).

---

## 8. Visual design

Palette discipline: water 10 (light blue), floor 2 (gray), white walls 1 (light gray), dark walls 4 (very dark gray), and only a few saturated gameplay colors at once — blue 9 / orange 12 (portals, bolts, gun), dark blue 9 doubling as the door, magenta 6 (pad), red 8 (button), yellow 11 (box), maroon 13 (button corners), green 14 (player + time bar). Dark gray 3 appears only as small pips (box corners). There is **no armed-color HUD element** — the gun barrel on the player is the sole armed-color indicator.

Sprites (4×4 px in-cell sketches; `.` = underlying tile; `A` = armed color; `K` = black). When both portals share a cell, each strip keeps its own half: opposite faces tile the cell exactly; **perpendicular faces split the shared 2×2 corner along the diagonal** — each color takes one diagonal pair of pixels, woven so neither dominates:

```
Player (green 14, no face — just the gun, a 2-px barrel of the armed
color centered on the facing edge; it snaps to the walk direction or
the last shot direction):

 facing E   facing W   facing N   facing S   carrying (E shown):
  GGGG       GGGG       GAAG       GGGG       yellow 2x2 bundle
  GGGA       AGGG       GGGG       GGGG       in the center:
  GGGA       AGGG       GGGG       GGGG        GGGG
  GGGG       GGGG       GGGG       GAAG        GYYA
                                               GYYA
                                               GGGG

Portal: only the STRUCK FACE of the white cell turns the portal's
color — a half-cell strip on the normal side; the rest stays white.
Linked: a white pip lights on the strip's outer edge. (W = white wall.)

 normal W   normal E   normal N   normal S
  CCWW       WWCC       CCCC       WWWW
  0CWW       WC0W ->    C00C       WWWW      (0 = white pip,
  0CWW       WC0W       WWWW       WWWW       only when linked)
  CCWW       WWCC       WWWW       C00C
                                   CCCC  <- strip rows

Pad (magenta 6, white     Button (red 8, maroon    Box (symmetric: yellow 11
 chevron POINTS the        13 corners; white 2x2     core, rim = portalable
 firing direction —        pip while weighted):      light gray 1):
 N and E shown):            DRRD                      WWWW
  MMMM     MWMM             RRRR                      WYYW
  MWWM     MMWM             RRRR                      WYYW
  WMMW     MMWM             DRRD                      WWWW
  MMMM     MWMM

Door segment, closed     Door segment, open
 (dark blue 9, keyhole    (blue frame, dark
 yellow 11 mid seg):       opening):
  BBBB                      BBBB
  BBYB                      BKKB
  BBBB                      BKKB
                            BBBB

Bolt projectile (armed color, white glint), centered in its cell:
  .CC.
  .WC.
```

**HUD**: a single element — the **time bar**, a 2-px-tall green bar on a black track along the bottom wall, pixels `x ∈ [4, 59]`, `y ∈ [61, 62]`. Fill = `remaining / budget` of the width, draining as actions are spent.

Animations (all deterministic, frames consume no time units):
- **Walk / turn** = instant (sprite redraws with new position/facing).
- **Portal bolt** = projectile travels 2 cells/frame from the player to the impact (§3.6); on placement the portal ring simply appears in its color (no white flash), on a fizzle the impact cell flashes a 2-frame outline in the bolt's color.
- **Flight** = 1 cell per frame; **teleport** = 2-frame flash on both portals.
- **Bump / blocked pair / teleport** flashes on a cell that hosts a portal highlight **only the portal's face edge** (a 1-px white line along the marked face), never the whole cell — the flicker stays on the portal area. Non-portal cells flash a full outline as before.
- **Door open / close** = 2-frame wipe (solid slab → dark opening on open, reverse on close, with a white flash on the door cells), played inside the move's cascade whenever the button's weighted state changes.
- **Box flight** = 1 cell per frame like the player's, with the same portal flashes; a box sinking plays the same splash (player visible) and reappears at its start cell.
- **Splash** (death) = 3 stages on the water cell while the player is hidden — burst (white core with light-blue diamond is replaced by white core with **blue 9** diamond, since water is now light blue: burst = white core + blue ring), ring (blue cell outline), dissipation (blue corner droplets) — then the player reappears at spawn with a 2-frame white flash.
- **Win** = the **exit door flickers** (4 alternating-outline frames), then a **fullscreen transition**: a dark-blue curtain wipes left-to-right across the whole canvas over 8 frames (~1 second) before the next level appears.
- **Final win** = after the flicker, a 28-frame **homecoming cutscene** (an unhurried ~3.5 s): a quiet dark field, a little house with an open blue door, and the agent (gun and all) walking across to it; the agent steps inside, the door closes, the window lights up yellow. Then the game is won.
- **Flashes never linger**: every flash plays its frames and then the action ends on a clean settled frame — the idle state between actions shows no outlines, only true world state (portals in their colors, the door in its current open/closed look).

---

## 9. Level files — one JSON per level

Levels are **data, not code**. Each level lives in its own JSON file under the game folder:

```
re_arc/environment_files/portal/0001/levels/level_01.json
re_arc/environment_files/portal/0001/levels/level_02.json
```

The engine loads every `level_*.json` in filename order at startup; the number of files **is** the number of levels. Everything about a level — terrain, objects, spawn, budget — is in its file; the game code contains no level-specific constants. To edit a level, edit its JSON; to add one, add a file.

### Schema

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Level title shown by the engine. |
| `time_budget` | int | Starting time units (§6). |
| `spawn` | `[col, row]` | Player start cell; must be plain floor. |
| `spawn_facing` | `"N"|"S"|"E"|"W"` | Initial facing. |
| `tiles` | 16 strings × 16 chars | Terrain map, **terrain characters only**: `#` dark wall, `%` white wall, `.` floor, `~` water. |
| `pads` | list of `{"cell": [c,r], "dir": "N"|"S"|"E"|"W"}` | Accelerator pads (§5). |
| `exit` | list of `[col, row]`, **required** | Goal tiles (purple): always walkable, never gated; entering one wins. Replace dark-wall cells. |
| `doors` | list of `{"cells": [[c,r],…], "button": [c,r]}` | Switch-gates only. One contiguous horizontal/vertical run of dark-wall cells each; `button` is **required** (≥2 cells from its door). |
| `portals` | `{"blue": {"cell": [c,r], "normal": "N|S|E|W"}, "orange": {…}}` | Optional pre-placed portals (on white walls). |
| `gun` | bool, default `true` | `false` = no portal gun: clicks no-op, SPACE never toggles, sprite has no barrel. |
| `boxes` | list of `[col, row]` (legacy single `box` also accepted) | Box start cells. |

### Layering and validation rules (fail fast on load)

- `tiles` must be exactly 16 rows of 16 characters drawn from `#`, `%`, `.`, `~`.
- Objects are **layered on top of terrain** and are authoritative for behavior: `spawn`, `pads`, `button`, and `box` cells must sit on `.` floor cells; `door` cells must sit on `#` dark-wall cells, which they replace (closed = solid dark-blue slab, open = walkable dark-blue doorway).
- A cell may host at most one object; the spawn may not sit on a pad or button.
- Portals attach at runtime only to `%` cells; nothing about portals appears in the file.
- Win cells = the `exit` goal tiles, always. Doors only ever gate routes (including the route to the goal).

---

## 10. The eight levels

Authoritative geometry lives in `levels/level_0*.json`; this lineup records each level's lesson and its load-bearing design facts. Pacing follows the playtest doctrine: **introduce each mechanic in a tiny room where nothing else is possible, then open into a space that demands it.**

| # | Name | New lesson | Intended / budget |
|---|------|-----------|-------------------|
| 1 | First Steps | portals are doors you walk through; SPACE lifts and drops the box; buttons open gates; purple = goal. **No gun.** | 13 / 78 |
| 2 | Two Gaps | the gun: click a white wall to place a portal; one of each color; relocation | 20 / 60 |
| 3 | Special Delivery | antechamber: boosters carry you over water; then walls vs. water vs. struck-face normals, and the box ferried home | 40 / 120 |
| 4 | Loading Dock | **escape the jail**: penned by dark walls + a moat, build an elbow on the one reachable block (two adjacent faces) and ride the pad through it out; then repeat with the box as artillery | 30 / 108 |
| 5 | Round Trip | the box as flight elbow; pass-through (both faces of one cell); ferry discipline, twice | 134 / 201 |
| 6 | Air Mail | the keyhole shot (gate buttons are firing positions); box artillery through a portal bend | 49 / 74 |
| 7 | Return to Sender | escape a sealed cell through your own parcel; fire while carrying; mail yourself | 33 / 66 |
| 8 | Grand Tour | the final exam — keyhole, a sealed nook heist for a second box, shielded crossing, pocket, elbow ride, re-entry, ferry, then a player-built box elbow feeds the parcel onto the islet | 145 / 290 |

**Level 1 — First Steps.** A 3×3 cell holding the player and the box; the only way out is a pre-placed blue portal in the wall (its orange twin sits outside). Lift the box (SPACE on it — the box verb is learned before the gun exists), carry it through, drop it on the button (gate stays open under its weight), walk through the gate to the goal. Nothing else is clickable, holdable, or possible.

**Level 2 — Two Gaps.** The old opener: two water channels, a free-standing white block, the goal corridor in the east mass. The block's struck face decides the exit; the goal brick `(15,7)` is occluded from platform 1 (DDA-verified), so two pair placements with a double relocation are forced. The corridor cells are plain goal tiles.

**Level 3 — Special Delivery.** Starts in a sealed NW antechamber: a booster below, a water curtain beside — one ride teaches that pads carry you over water (a rock at `(8,3)` parks the flight). Then the classic: dark floors refuse portals, the block `(7,7)` must be struck from the water's edge for an east-facing exit, the pad-flight is the only way to the island, the box waits in a water inlet (every start-side bolt gives it a water-fronting face), and the post `(11,1)` is the way home. The goal `(0,5)-(0,7)` glows behind a gate `(1,5)-(1,7)` wired to the button `(4,6)` — weight it with the box.

**Level 4 — Loading Dock.** Two beats that teach the flight-elbow twice — first with *you* as the payload, then with the box. Beat 1 is a **jailbreak**: the player is penned in a cell walled by dark (non-portalable) tiles on three sides and a water moat on the fourth, with a launch pad `(1,5)` inside. The *only* portalable surface in reach is the lone white block `(7,5)` across the moat — the box is sealed behind dark walls and nothing else can be struck. Build an elbow on that one block by striking **two adjacent faces** — blue on its **south belly** (a bottom-edge shot from the south-west, over the moat) and orange on its **west face** head-on — then step onto the pad: the flight enters the west face, turns the corner, and sails **down** out the belly, carrying you out of the jail onto the work platform `(7,9)`. A misfire sinks in the moat and resets — cheap retry. Beat 2 repeats the trick with the box as artillery: rebuild the same elbow on the east wall `(14,5)` (west face head-on, south belly from `(10,9)`), lift the box `(9,7)` and feed it to the pad `(10,5)`; it flies east over the water, turns the corner, and drops onto the button `(14,9)`. Gate opens, goal `(9,3)` behind. This is exactly the flight-elbow Round Trip demands one level later.

**Level 5 — Round Trip.** The marathon (old level 3) unchanged in body: extract the box from the sealed pocket via the water-hole shot and the both-faces pass-through cell `(2,13)`, elbow the pad flights over two waterways (the catcher rock marks the column), ferry with the evacuate-before-lifting discipline. The goal now sits at the top wall behind a gate `(12,1)-(14,1)`; the second elbow flight parks against the closed gate at `(12,2)` — one cell shy of where it used to land.

**Level 6 — Air Mail.** The keyhole: the gate's button is a firing position — bolt through the open doorway onto the block beyond, portal past the gate. Then artillery: the box fed to the pad flies through the X→Y bend onto the islet button. The goal row 0 glows behind a second gate `(10,1)-(12,1)` wired to that islet button.

**Level 7 — Return to Sender.** Sealed in a cell where the box is the only enterable portal surface: the 45° bolt threads the water slit onto the A-block, walk out through your own parcel; X planted outside doubles as the re-entry door; Y is fired from inside while carrying; feed the launcher and the box pinballs Y→X→S2→S3 onto the islet button — then mail yourself after it and land on the parcel. Gate `(14,6),(14,7)` before the goal.

**Level 8 — Grand Tour.** The exam: keyhole out of the start cell; **the nook heist** — a second box waits in a sealed two-cell nook center-left (dark walls, a water slit at `(3,6)`, a splash valve at `(1,7)`), and the only way in is to realize the nook's inner ceiling face `(2,5)` can be struck by a **45° bolt threaded through the slit** (or named with a bottom-edge click); pair it with A1, walk through your own portal into the nook, lift the parcel, and walk back out the same way. Park it at `(12,1)` — where a bounce pad used to be, because the down-launcher is *gone* and the player must rebuild it from the Loading Dock lesson: **both portals on the parked box, adjacent faces** (west flank from `(11,1)`, belly from `(12,2)`), a hand-built elbow. Then the campaign: the shielded crossing (wall row 9's two slits make hall and zone C mutually bolt-proof; A1 is dark-flanked, A2 fronts its pond); the pocket (slit shot + pass-through); the forced empty-handed elbow ride; re-entry through the abandoned parcel; the post ferry; and the finale — stand on the gate button beside the feed pad so the first parcel flies through the door you are holding, through the elbow you built, and **down** onto the islet button. Gate `(14,7),(14,8)` before the goal. Two boxes are both essential: one to be the elbow, one to be the payload.

## 11. State variables (per level)

```
player_cell, facing ∈ {N,S,E,W}, armed_color ∈ {blue, orange}
blue_portal:  {cell, normal} | none
orange_portal:{cell, normal} | none
carrying_box: bool
box_cells: list (per box; none while carried/flying)
door_open: list[bool]        (per door; player or box on its button)
remaining_time: int          (budgets 78/60/120/108/201/74/66/290; −1 per action; 0 = level fails)
in_flight: {dir} | none              (transient within a cascade)
box_flight: {dir} | none             (transient; the box flying after being fed to a pad)
bolt_anim: {cells, idx, outcome} | none   (transient)
splash_anim: {cell, stage} | none         (transient)
level_index ∈ {0, 1}         (exactly 2 levels)
```

Static per-level data is **not** hard-coded: it is loaded from the level's JSON file (§9) — terrain map, time budget, pad cells + directions, button cell, door cells, exit cells, spawn cell + facing, box start cell.
