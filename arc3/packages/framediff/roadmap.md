# framediff roadmap

*Register: living.*

Open work on framediff. How it works today is `design.md`; what the seal
does with its blocks is `seal.frame_delta` and `packages/seal/design.md`.

- **What the search spends in a full run.** A time limit holds every
  call (`design.md`), and every seal record carries `duration_s`:
  read from a full run which calls come near it, and which of the
  generators the tail comes from, before tightening either.
- **Cost-table candidates, all deferred** (2026-09-17) so that
  the runs compare one change at a time. Each is a cost, never search
  logic:
  - a light cost on a move's area, so a smaller block wins at equal cost
    (its distance is priced already, `Decomposer.move_distance_cost`);
  - a light cost on a move that leaves anything but the frame's most
    common colour behind, which would settle the one-colour swap by price
    instead of by the tie-break it has now;
  - colours priced differently, the general form of the two above.
- **Scrolling** (2026-10-08: nothing built for it yet). A scroll
  is one copy of the rectangle that moved (a copy's gain map bars cells
  whose source falls outside the frame, so a fixed bar stays out of it)
  and pastes for what scrolled in. In an animation every frame repeats
  that, the incoming strip a fresh paste each time, which no loop catches.
  Candidates: a scroll as an edit of its own, the shift and the strip that
  came in; and for an animation, its frames read as windows onto one larger
  picture, built once and sliced frame by frame.
- **Copies that explain by coincidence** (2026-10-09: a general
  weakness). The search takes any copy whose cells match, from anywhere on
  the board, mirrored or turned; a copy costs the same however unrelated
  its source. So a script can rebuild cells from a region that only happens
  to look the same, and it reads as that region having moved or turned. In
  round 1 of seal sweep 2 a sliding run's background in escape_velocity was
  built as `np.rot90(anim[0][48:52, 24:28], 2)`, a turned copy of an
  unrelated region, where the cells were the floor. Colour changes read as
  copies (below) and strip copies are the same weakness. The fix is in the
  costs: a copy should pay for being far-fetched (a turned copy already
  pays more), or the plain readings (fills, recolours) should win a tie.
- **Colour changes read as copies** (2026-10-08: accepted for
  now). Squares that only change colour are often written as copies or
  moves from elsewhere: a copy at 10 bits fixes every colour in its
  rectangle at once, where a recolour at 10 fixes one, so a patterned
  square that changes several colours, or a row of squares, reads as
  content copied in from wherever it already looks that way. Pricing
  colours (the cost-table candidates above) is one route; a recolour of
  several colours in one edit is another.
- **Several objects moving at once, each on its own path** 
  (2026-10-08: not supported). A sliding run follows one shape, or touching
  pieces that move by one offset, over a background that stays put; a
  second shape moving on its own breaks that, so neither is a run (two at
  one velocity each can still come out as a stepped loop). The general
  form: one background, and each frame drawn as every shape at its place
  that frame.
- **Sliding shapes colour cannot cut out** (2026-10-08). A sliding
  run finds its shape as one-colour pieces that moved, and picks it out of
  its box by colour. A shape over a floor of one of its own colours is
  neither: in cd82 (pass 3, level 1, history[137]) a diamond with a black
  interior slides back over a black rectangle, a pure movement over a board
  that stays put, and is written a frame at a time because its interior
  merges with the rectangle. The general fix: find the shape by motion (the
  offset from the part that moves cleanly, the shape every cell that keeps
  its colour at the shifted places while the rest stays put), and write an
  explicit mask (`from_ascii_grid`) where colour cannot pick it out.
- **Animation frames explained** (2026-09-24).
  `Decomposer.explain_animation` writes one block that rebuilds a turn's
  animation frame by frame (`framediff.animation`), and with
  `animation_patterns` writes runs of frames as loops (`design.md`). Open: a budget for
  long animations as a whole, since the per-frame budget alone lets a
  300-frame animation print 300 pieces, and whether a world model should
  then predict animations too.
- **Blocks the model can manipulate, not only read** 
  (2026-09-24). A frame delta or an animation block is printed text; a
  result like `segment()`'s, whose parts the model can index, filter and
  re-run, would let it take a block apart instead of reading it. That
  means the transform library in the REPL, as objects the model holds,
  with their source still blocked.
- **Cold decomposition** (2026-09-17): a block from an empty frame
  to a level's opening board, describing the board itself rather than a
  transition. Needs more vocabulary than the transforms here have —
  upscaled copies and upscaled ascii-grid pastes at least, since ARC
  boards are drawn as tiles.
- **Copy from the board, and integer scaling.** `BlockCopy(source='g')`
  exists and is tested but the search never proposes it: its lines are
  order-dependent, and no recorded transition has needed it. Integer
  scaling is not implemented. Both wait for a trace that asks.
