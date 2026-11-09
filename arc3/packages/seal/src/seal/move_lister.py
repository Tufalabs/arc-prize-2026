"""Per-state action enumeration for the planner.

A single long-lived ``MoveLister`` serves every state: ``list_moves`` takes
``available_actions`` and ``frame`` per call, and the planner calls it once
per node it expands.

The instance holds no random state — every call takes the
``random.Random`` it should consume, derived by the caller (:mod:`seal.rng`).
Two calls given equal streams produce equal output, which is what makes a
search reproducible from its seed alone. The RNG draws below are therefore
load-bearing in their exact order; ``tests/data/rng_contract_vector.json``
pins them.
"""

from __future__ import annotations

import random
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from platypus.contract import Action

#: Never enumerated as a candidate. RESET is issued by policy, never by
#: search (from any non-root state it lands back at ``initial_state()``);
#: UNDO has no world model.
_EXCLUDED = ("RESET", "ACTION7")
_CLICK = "ACTION6"

#: 4-connectivity: components are orthogonally contiguous, diagonals do not
#: join. Read-only, never mutated.
_STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _label(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int32], int]:
    """4-connected labelling of ``mask``.

    Exists only to pin a type: ``ndimage.label`` is overloaded on its
    ``output`` argument and its stub does not let a checker destructure the
    two-element return for the ``output=None`` call we make.
    """
    return cast("tuple[npt.NDArray[np.int32], int]", ndimage.label(mask, structure=_STRUCTURE))


class MoveLister:
    """List candidate actions for a state.

    Behavior:
    - Filters RESET and UNDO unconditionally.
    - For each remaining simple action: emits one ``Action(name)``.
    - For CLICK: enumerates maximal monochromatic 4-connected regions in
      ``frame`` and takes a representative pixel per region (centroid
      snapped into the region). When ``max_clicks_per_color`` is set, keeps
      only that many biggest components per colour (ties broken by ``rng``
      jitter). The final click list is then capped globally at
      ``max_clicks``.
    - The final move list is a uniform shuffle of (clicks + simple actions).
    """

    def __init__(self, max_clicks: int = 100, max_clicks_per_color: int | None = None) -> None:
        self.max_clicks = max_clicks
        self.max_clicks_per_color = max_clicks_per_color
        #: Over the boards clicks were listed for since the last
        #: :meth:`reset_counts`: the most click regions on one, and how many
        #: had more than ``max_clicks``, so some of their regions went
        #: unclicked. What a search's report says about its clicks.
        self.most_regions = 0
        self.boards_over_cap = 0

    def reset_counts(self) -> None:
        """Start the click counts over, for a new search."""
        self.most_regions = 0
        self.boards_over_cap = 0

    def list_moves(
        self,
        available_actions: Sequence[str],
        frame: npt.NDArray[np.int8] | None,
        rng: random.Random,
    ) -> list[Action]:
        """Candidate actions from one state, in the order they should be
        tried. Consumes ``rng``: one draw per component when CLICK is
        available, then one shuffle."""
        clicks: list[Action] = []
        simple: list[Action] = []
        for name in available_actions:
            if name in _EXCLUDED:
                continue
            if name == _CLICK:
                if frame is None:
                    raise ValueError("MoveLister: CLICK is in available_actions but no frame was provided")
                clicks = self._click_actions(frame, rng)
            else:
                simple.append(Action(name))
        out = clicks + simple
        rng.shuffle(out)
        return out

    def _components_by_color(
        self,
        frame: npt.NDArray[np.int8],
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Connected components of ``frame`` grouped by colour:
        ``{colour: [(size, row, col), ...]}`` where (row, col) is a
        representative pixel snapped into the component. Unordered within a
        colour. Iterates ``np.unique(frame)`` in ascending colour order, so
        the returned dict's insertion order is deterministic — which is what
        makes callers' rng draws reproducible, since they draw per component
        while iterating it. Consumes no rng itself."""
        comps_by_color: dict[int, list[tuple[int, int, int]]] = {}
        for color in np.unique(frame):
            mask = frame == color
            labels, n = _label(mask)
            if n == 0:
                continue
            sizes = np.bincount(labels.ravel())  # index 0 is "not in mask"
            bucket = comps_by_color.setdefault(int(color), [])
            for lbl in range(1, n + 1):
                rows, cols = np.where(labels == lbl)
                crow, ccol = int(rows.mean()), int(cols.mean())
                if labels[crow, ccol] != lbl:
                    idx = int(np.argmin((rows - crow) ** 2 + (cols - ccol) ** 2))
                    crow, ccol = int(rows[idx]), int(cols[idx])
                bucket.append((int(sizes[lbl]), crow, ccol))
        return comps_by_color

    def _click_actions(
        self,
        frame: npt.NDArray[np.int8],
        rng: random.Random,
    ) -> list[Action]:
        """Click targets for a frame: the centroids of the largest monochromatic
        components, with the per-component random tie-break the vector in ``tests/data`` pins.
        """
        comps_by_color = self._components_by_color(frame)

        # Per-colour cap: keep the ``max_clicks_per_color`` biggest
        # components for each colour, tie-broken by rng jitter so the cap
        # cutoff doesn't always pick the same component when several share
        # the same size. One draw per component, in bucket order.
        flat: list[tuple[int, float, int, int]] = []  # (size, jitter, row, col)
        for bucket in comps_by_color.values():
            keyed = [(-size, rng.random(), row, col) for size, row, col in bucket]
            keyed.sort()  # biggest first within colour
            if self.max_clicks_per_color is not None:
                keyed = keyed[: self.max_clicks_per_color]
            for neg_size, jitter, row, col in keyed:
                flat.append((-neg_size, jitter, row, col))

        # Descending by size across colours, so the global ``max_clicks`` cap
        # keeps the biggest blobs and tosses the dust.
        flat.sort(key=lambda t: (-t[0], t[1]))
        self.most_regions = max(self.most_regions, len(flat))
        self.boards_over_cap += len(flat) > self.max_clicks

        return [Action(_CLICK, row=row, col=col) for _, _, row, col in flat[: self.max_clicks]]
