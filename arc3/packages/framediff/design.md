# Design

*Register: as-built — updates with the code.*

How framediff turns two frames into a short block of numpy. The code says
what each transform is and what the search does; this file carries what it
cannot: why the costs are the numbers they are, what the search may and
may not do, and the measurements the shape rests on. Open items are in
`roadmap.md`; what the seal does with the blocks is `seal.frame_delta`.

## Why a block of code at all

Two readings of the seal's 4×25 world-model run (`seal_wm_full`, 100 game
runs, 2026-09-17):

- **The model's own renders are frame copies plus slice assignments.** Of
  466 `world_model.py` versions with a readable `render`, 62% start from a
  copy of `history[0].current_frame.grid` and paint movers, bars and
  toggles on top, 21% rebuild from `np.full` with parameters read off the
  frame, 12% are hard-coded. Code of the form `g[dst] = a[src]` and
  `g[r0:r1, c0:c1] = c` is the model's own idiom, not a new one to teach.
- **Reading the board is where the turns go.** In one game's trace 22% of
  all reasoning was decoding pixels and diffs by hand, with repeated
  mis-indexing of its own printed rows; one turn (a red block moved six
  columns onto the cell the player entered, one indicator cell recoloured)
  cost 24k characters of hypotheses. The same fact is three lines of
  numpy.

The concepts come from `arc3-scene-decomposition`'s warm transition —
reconcile with the cheapest edit, bit costs as a prior, emitted code that
self-tests — with the sprite library left out. There is no scene: the
vocabulary is slices of the frame.

## Transforms

`Transform` has three public methods and one rule each: `transform(frame)`
performs the edit and returns a new frame, `to_code(a, b, g)` writes it as
a block, `bit_cost()` prices it. Subclasses implement `_apply`,
`_code_lines` and `_bit_cost`; the superclass owns the block's header
`g = a.copy()` and its closing assert, so any transform alone is a
complete block and `Composite` only concatenates lines. `NoOp` is the
exception: equal frames need no board, so its block is the comparison
alone.

`to_code` runs its own text before returning it, on two probe frames (a
seeded random one, where a wrong shift, turn or mirror shows almost
everywhere; a banded one, where every 4×4 window holds every colour, so a
recolour finds its colour). A block that does not reach what `transform`
reaches, or that writes to its start frame, is never emitted. This is why
the search can hand a block to a model without a human reading it.

A block's names are the caller's expressions: `a` and `b` may be any
attribute, subscript or call chain, `g` a plain name. A block may call
`from_ascii_grid` and `bresenham_line`, which it takes from where it runs;
`block_namespace()` binds framediff's own outside a REPL that preloads
them.

**Wildcards.** A start frame may hold `-1` cells, meaning "any colour": a
model's render says that where it declines to predict. A wildcard stays a
wildcard through every transform, costs nothing in the search, and the
block resolves them from the target before its assert. A target never
holds one.

## The costs

Declared per kind as class constants, not derived from an encoding. Their
only job is the ordering that makes the search reach for the edit a person
would name:

| transform | cost |
|---|---|
| one cell | 4 |
| fill a rectangle | 6 |
| ascii-grid paste, H×W | 8 + H·W |
| recolour one colour to another inside a box | 10 |
| block copy, or a move (the same, leaving a colour behind) | 10 |
| either of those mirrored or turned | 18 |
| an object moved alone (`MaskedMove`, with `object_moves`), and mirrored or turned | 10 + 4, 18 + 4 |
| line | 12 |

There is no residual transform: whatever the search leaves is single
cells at 4 each, which is the fallback everything else is priced against.

**Two of the numbers have a reason beyond taste.** A turned copy costs 18,
above a move and a fill together: at 14 a sprite moving by its own size
inside a strip was cheaper to describe as that strip turned in place than as
a move. A move costs what a copy costs, because a game's
mover leaves background behind and a price for that would push the search
back to the strip turns.

**A light cost on distance** (`Decomposer.move_distance_cost`). A copy or move can also pay bits per cell it travels, rows plus
columns. It exists for one tie: a one-colour block moved left or up by less
than its own size is written exactly by the whole block moving and by its
trailing strip jumping across it, at one price, and the enumeration order
picked the strip for those two directions where it picked the block for
right and down. Priced by distance, the shorter move wins in every
direction. Kept light so it decides between moves and nothing else; on the
reference run's corpus (`packages/seal/tools/framediff_corpus.py`) it made no
block costlier, and it ships on (2026-10-08).
A fraction of a bit can still change an early greedy step and lead a run to a
costlier script, so with the price on the runs are made with it and then
without it, every script priced by distance and the priced run's kept at a
tie: the result never costs more than the unpriced search's script priced the
same way.

**The search may only prefer a lower cost.** No part of it may sidestep
the cost to reach a script that reads better; when the cheapest script is
the wrong one, the costs change, not the search. Everything the search
does is about *finding* cheaper scripts: which candidates exist, and which
get scored exactly.

## The search

Notation: `a` the start frame, `b` the target, `g` the board so far
(starting as `a`). A candidate's gain is the cells it makes right minus
the cells it makes wrong, wildcards counting for neither. Every candidate
is scored on `cost − 4·gain`, the change in the whole script's length with
the single-cell fallback priced in — so breakage costs exactly what
repairing it costs, and the stopping rule falls out: greedy takes the
lowest score each step and stops when none is below zero.

A candidate is a short *list* of transforms, because what a person weighs
as one edit has to be scored as one.

- **Gain maps and their best rectangle.** A fill, a recolour, a paste and
  a copy at a given shift each write one value per cell inside some
  rectangle, so their gain is a per-cell map and the best rectangle is its
  maximum-sum subrectangle (2D Kadane, exact: every row span summed into a
  strip, the column scan vectorised over all strips of all maps). Each map
  is cropped to the bounding box of its positive cells first, which is
  what makes a small change cheap. A copy's map bars cells whose source
  would fall outside the frame, so a scroll under a fixed bar comes out as
  a copy of the world rectangle alone.
- **Shifts by correlation.** How many wrong cells a shift would fix is,
  for all 4096 shifts at once, the cross-correlation of each colour's
  plane of `a` with the same colour's plane of the still-wrong cells of
  `b`, by FFT on zero-padded planes. The eight mirrored and turned
  variants are the same search on the transformed frame, since a turned
  sub-block is a sub-block of the turned frame. Four shortlists feed the
  rectangle search: the best shifts overall, more of them for the plain
  variant than for the turned ones; the best shift for each large
  component of wrong cells, correlated over the component's whole bounding
  box; and the best shifts for each *hole*, a cheap fill's rectangle whose
  content is looked for among the wrong cells. The hole pairing is what
  finds a mover on a frame whose background matches at dozens of shifts. A
  shortlist can miss a cheaper copy, but it never chooses a costlier one,
  since every shortlisted shift is scored exactly.
- **Moves.** Every shortlisted copy is also offered as a move, clearing
  its old place to the colour that makes the most of that place right,
  whenever clearing gains.
- **Carved pastes.** A paste is the one transform whose price grows with
  its area, and a cell inside it costs 1 where a cell on its own costs 4,
  so a paste absorbs whatever lies beside the content it is for — a
  filled rectangle next to a complex shape, or, on two unrelated frames,
  everything. Once taken it cannot be undone, so it is carved before it is
  proposed: the cheapest other transforms inside its rectangle are tried,
  each followed by pastes of what they leave, and the cheapest of those
  replaces it. Every paste that results is carved in turn.
- **Lines from components.** An 8-connected component of wrong cells that
  the target paints in one colour is tried as a line between its extreme
  cells, each way round, since a line and its reverse can differ by a cell
  on some slopes.
- **Single cells** for whatever remains.
- **Objects matched across the frames**, with `Decomposer.object_moves`
  (`framediff.objects`). A 4-connected piece of one colour that
  touches a changed cell is looked for in the other frame with the same
  colour and shape, or turned; identical pieces are paired nearest first,
  and touching pieces that moved by one offset are proposed as one object
  of several colours. Each match that moved proposes a `MaskedMove`: its
  cells picked by colour inside its box, which only a box holding no other
  cell of those colours can do exactly, so only such boxes are proposed.
  The move is scored like any candidate, on the board as it stands. Where
  the box around an object holds a floor the move cannot clear, a copy of
  the box carries the floor along and reads as a strip; the masked move
  carries the object alone, for the 4 its mask line costs 
  (2026-10-08: a first price). A floor that repeats at the
  shift keeps the copy, which is exact and cheaper.

**Two runs.** The greedy runs once with plain shifts only and once with
every variant, and the cheaper script is returned, the plain one at a tie.
A turned copy that fixes two things at once wins a step and can lead a run
to a costlier script than the plain moves it displaced; the minimum of two
runs is never costlier than either. With `object_moves` a third run adds
the matched objects to every variant, for the same reason: a script is
never costlier with them than without. With `move_distance_cost` the runs
are made priced and unpriced (above).

**Two tie-breaks**, the only ones, each between readings that write the
same cells at the same cost. A one-colour block moving onto a one-colour
place reads as either block moving, so the one leaving the more common of
the two colours behind wins. A plain copy of a strip shifted by less than
its height writes what lay beyond a sprite onto the sprite's old place,
which is also a move of the sprite alone, so the smallest such move
replaces it.

Otherwise there are no tie rules: ties fall where the enumeration order
puts them, and the whole search is a deterministic function of the two
frames.

**Lookahead for settled frames.** `Decomposer.lookahead` tries a step's
best two candidates one step further before choosing 
(2026-10-09): on the reference run's corpus it makes 408 of 6,978 settled
blocks cheaper (median 6 bits, at most 99; 17 costlier, at most 8), 1.3%
of all their bits, for about a quarter more of a cheap search. Greedy's
first step, a carved move that fixes many cells, can lead past a recolour
that ends cheaper (sp80: 30 bits against 20). Animations keep plain greedy
(`animation_lookahead`): there it buys 0.7% for as much more time on far
longer searches, and pushes the longest past the time limit.

**Considered and rejected.** Ranking the shift shortlist by colour frequency was
tried and dropped: it made the shortlist better by a heuristic that is not
about cost, where scoring more plain shifts exactly does the same thing
honestly.

## Animations

A move's animation is explained as one block that rebuilds it into a dict
keyed by frame index, a piece per frame, closed by one assert over the
whole dict (`framediff.animation` has the shape). The unit is still one
frame: each frame is decomposed from a source, and the pieces are strung
together.

- **A stride, and a character limit** (2026-09-25). A stride
  builds every stride-th frame, from the frames built before it; the dict
  keeps the frames' own indices, so a source is always named by the frame
  it is. Without a stride the block is tried at 1, 2, 3, 4 and 6 in turn
  until it fits the character limit, each attempt abandoned the moment it
  passes the limit rather than once it is whole, and the last written
  whole. With a stride given the block is written whole, and cutting it
  is the caller's business.
- **The board holds the newest built frame throughout.** Every piece ends
  by storing a copy of the board, so a frame built from the one before it
  needs no line to start, and any other source starts with a copy of that
  source. A piece reads its start frame under the source's own name — the
  dict entry, never the board it is editing — so a copy reads what the
  source held, whatever the piece has written so far.
- **An exact repeat is a copy, found first.** A frame that repeats an
  earlier built frame exactly, from any distance, is a copy of the most
  recent such, or of the frame before the move if it repeats only that. A
  copy costs nothing and nothing costs less, so nothing else is searched.
- **Otherwise the cheapest script wins.** The candidates are the three
  built frames just before, most recent first, then the frame before the move
  (2026-09-24); the order only breaks ties.
- **Shortcuts that never change the choice.** A candidate's search stops
  once its script so far is over budget or no cheaper than the best so far
  (a script's cost only grows as the search appends to it, and a tie goes
  to the earlier candidate anyway), and a candidate whose frame repeats an
  earlier candidate's is not searched. Most frames are a small step from
  the one before, so most frames decompose once.
- **The budget is per frame**, `animation_max_bits`, lower than a settled
  frame's: an animation is read a frame at a time. A frame whose script
  exceeds it is taken as it is, a copy of the frame under its own name,
  and the frames after it build on it as on any other.
- **Checked twice.** Every script in the block passes its own self-test on
  probe frames, and the whole block is run under plain names against the
  frames it was given before it is returned.
- **Runs written as loops**, with `Decomposer.animation_patterns`
  (`framediff.patterns`). Once every frame has its piece, three kinds of run
  are looked for, each one loop: frames that each repeat the frame some
  distance back (a blink); frames each built from the one before by the
  same edit, its numbers moving by a fixed step (a bar growing, a slide on a
  plain floor), found in the pieces' own text; and one shape of any outline
  moving by a fixed offset over a background that stays put, for at least
  three frames, found in the frames, the shape matched across two of them
  as `object_moves` matches objects and the background read from the frames
  around it. The runs written are those that save the most characters
  together, none overlapping, so a block is never longer for them. A
  stride whose pieces pass the length limit is not given up at once, as it
  is without runs: its frames so far are written with their best runs, and
  it is given up once that block passes the limit by more than the pieces
  of its last three frames, which a run reaching past them could still take
  into its loop. Building every stride whole instead made a long animation
  search each stride in full. The loops are checked with the rest of the
  block.

## What it costs to run

Over 300 transitions sampled from `seal_wm_full` (a game, then a level,
then a move, uniformly):

| start frame | script cost, median | ms median | ms p90 | ms max |
|---|---|---|---|---|
| the frame before the move | 16 | 53 | 216 | 698 |
| the frame the level opened with | 50 | 199 | 581 | 1774 |

The previous frame is the cheaper description for 92% of transitions. The
worst cases are rebuilt boards and scrolls with new content, whose cost is
in pastes and cells nothing else describes. A level-start comparison on a
rebuilt board is where the time goes.

**A time limit, not a bound by construction** (2026-09-25). One
`explain` or `explain_animation` call searches for at most
`Decomposer.time_limit_s`, every source and stride included, or until a
deadline the caller passes if that comes first. The clock is read where
the search already asks whether to give up, after every greedy step, so a
call ends within a step of its deadline. A call that reaches it says
`timed_out` and keeps the best it found by then (2026-10-08): a
script the clock did not stop is exact, if perhaps costlier than a whole
search would have found. `explain` chooses among the sources whose search
came back with a script, and with none comes back over budget with no
script; `explain_animation` keeps the frames it built at the stride under
way, the frame under way from the best script its search had found, and
takes the rest as they are. What a timeout means is the caller's: the seal
shows the block as any other, the timeout on its record only, and ends the
game instead under `SealAgent.crash_on_framediff_timeout`, which no column
turns on.
