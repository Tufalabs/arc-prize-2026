# framediff

*Register: living.*

What changed between two 64×64 frames, as a short numpy edit script. Given
a start frame `a` and a target frame `b`, `Decomposer().decompose(a, b)`
searches for a cheap sequence of edits that turns one into the other, and
`Transform.to_code()` writes that sequence as code a model can read and a
harness can run, ending in `assert (g == b).all()`.

The package holds no agent and no game: two frames in, a block of numpy
out. The seal uses it to tell its model what changed instead of "N pixels
differ" (`seal.frame_delta`); nothing else depends on it.

- `framediff.transforms` — the edits, each of which performs itself on a
  frame, writes itself as numpy, and prices itself.
- `framediff.decompose` — the search; `Decomposer.explain`, which takes
  named start frames (`Source`, each with a handicap), returns the cheapest
  one's script as a block, and flags it when it costs more than the
  decomposer's `max_bits` or ran out of its `time_limit_s`; and
  `explain(a, b, ...)`, the one-source shorthand.
- `framediff.animation` — `Decomposer.explain_animation`: one block that
  rebuilds an animation frame by frame, each frame from one of the frames
  just before it or from the frame before the move.
- `framediff.grid` — frames as int8 arrays and their ascii-grid notation
  (`to_ascii_grid`, `from_ascii_grid`).
- `framediff.drawing` — `bresenham_line`, the line the ARC games
  hand-roll.
- `framediff.diagnostics` — notebook views; nothing on the path from two
  frames to a block.
- `notebooks/demo_transforms.ipynb` — every transform and the code it
  writes, on recorded frames. `notebooks/demo_decomposer.ipynb` — the
  search on transitions sampled from a recorded run, both baselines and
  the worst cases (`run_transitions.py` reads the run).
  `notebooks/demo_animation.ipynb` — the first and the longest animation
  of every game in a recorded run, as the seal's kernel keeps them, with
  their blocks (`run_animations.py` reads the run).

Docs: `design.md` is how it works and why the costs are what they are;
`roadmap.md` is the open work.
