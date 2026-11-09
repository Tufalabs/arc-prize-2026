"""The duck's typed per-game state.

One subject: what the model has put in ``test_out`` and whether it can be
submitted. Everything else the ARC3 duck carried here — the frame timeline,
pending level wins, batched actions — has no counterpart in a static task.

The split with :mod:`duck.prompts` is deliberate. This module produces short
factual phrases about a value ("has shape (2, 3, 4); a grid is 2-D"), which
are facts, not wording; the prompt class turns a list of them into the
sentence the model reads. So a phrase can be quoted in a diagnostic or a
test without dragging prompt wording along.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

import taaf.game
import taaf.grid
from duck.kernel_code import grid_from_rows

SlotState = Literal["ok", "missing", "bad"]


@dataclass(frozen=True)
class AnswerSlot:
    """One attempt at one question, as read out of the kernel."""

    state: SlotState
    grid: taaf.grid.Grid | None = None
    #: Why this slot is unusable, as a phrase completing "attempt N ...".
    #: Empty when the slot is fine or simply absent.
    why: str = ""


@dataclass(frozen=True)
class AnswerReading:
    """One read of ``test_out``: a slot per attempt per question.

    ``bound`` is false when ``test_out`` is not a list at all — the model
    rebound or deleted it — in which case ``rows`` is empty and ``why`` says
    what it found instead.
    """

    bound: bool
    rows: tuple[tuple[AnswerSlot, ...], ...] = ()
    why: str = ""

    @property
    def usable(self) -> tuple[taaf.grid.Grid | None, ...]:
        """The first usable grid for each question, or ``None`` where there
        is none. Attempt order is preference order: a model that filled both
        slots meant the first one first."""
        return tuple(next((s.grid for s in row if s.state == "ok"), None) for row in self.rows)

    def is_complete(self, n_questions: int) -> bool:
        """Whether every question has an answer the model actually produced.

        This — not :meth:`submission` — is what ends a game. Submission fills
        gaps with placeholders so that partial work still scores, and using
        that as the stop condition would end the game the moment the model
        answered its first question of several.
        """
        return len(self.rows) == n_questions and all(grid is not None for grid in self.usable)

    def submission(self, questions: tuple[taaf.grid.Grid, ...]) -> taaf.game.Submission | None:
        """The best answer TAAF can score right now, or ``None`` if the model
        has produced nothing usable at all.

        A question answered once is answered twice: the single grid is
        duplicated into both attempts. ARC-AGI-2 scores the two
        independently, so a second distinct hypothesis is free score — but a
        model that offers one is not made to invent another, and a duplicated
        attempt costs nothing.

        **Unanswered questions are filled with the question itself.** TAAF's
        ``Submission`` covers every pair or none, so without this a model
        that answered one of two questions and then hit the deadline would
        score zero rather than a half — and on ``arc2-eval`` more than a
        third of tasks have several questions, which is exactly where a hard
        cutoff costs the most. The placeholder is the input grid unchanged:
        always a valid grid, because it came from the task, and wrong in the
        ordinary way rather than malformed. :meth:`n_placeholders` says how
        many went in, so a filled slot is never mistaken for a choice.
        """
        if len(self.rows) != len(questions):
            return None
        usable = self.usable
        if all(grid is None for grid in usable):
            return None
        attempts: list[tuple[taaf.grid.Grid, ...]] = []
        for index, row in enumerate(self.rows):
            good = [slot.grid for slot in row if slot.state == "ok" and slot.grid is not None]
            if not good:
                good = [questions[index]]
            filled = (good + [good[0]] * taaf.game.N_ATTEMPTS)[: taaf.game.N_ATTEMPTS]
            attempts.append(tuple(filled))
        return taaf.game.Submission(attempts=tuple(attempts))

    def produced(self) -> tuple[tuple[bool, ...], ...]:
        """Which grids of :meth:`submission` the model wrote itself, indexed
        like its attempts: its usable slots in order, then ``False`` for each
        copy and placeholder that fills the rest."""
        flags: list[tuple[bool, ...]] = []
        for row in self.rows:
            written = sum(1 for slot in row if slot.state == "ok" and slot.grid is not None)
            written = min(written, taaf.game.N_ATTEMPTS)
            flags.append((True,) * written + (False,) * (taaf.game.N_ATTEMPTS - written))
        return tuple(flags)

    def n_placeholders(self, n_questions: int) -> int:
        """How many questions a submission would have to fill in for."""
        if len(self.rows) != n_questions:
            return n_questions
        return sum(1 for grid in self.usable if grid is None)

    def problems(self, n_questions: int) -> list[str]:
        """Everything wrong with this reading, one phrase per problem.

        Empty means the answer is submittable. A question with a usable
        first attempt and a broken second is NOT a problem: the model has
        answered, and nagging it about a spare slot spends a round on
        nothing.
        """
        if not self.bound:
            return [self.why or "test_out is not a list"]
        if len(self.rows) != n_questions:
            return [f"test_out has {len(self.rows)} entr{'y' if len(self.rows) == 1 else 'ies'}, not {n_questions}"]
        out: list[str] = []
        for index, row in enumerate(self.rows):
            if any(slot.state == "ok" for slot in row):
                continue
            broken = [slot for slot in row if slot.state == "bad"]
            if not broken:
                out.append(f"test_out[{index}] is still unset")
                continue
            for attempt, slot in enumerate(row):
                if slot.state == "bad":
                    out.append(f"test_out[{index}][{attempt}] {slot.why}")
        return out


def _slot_from_payload(entry: object) -> AnswerSlot:
    """One kernel slot report, with the ARC-AGI-2 rules applied.

    The kernel decides what is decidable from the value alone; the palette
    and the size limit are checked here, where ``taaf.grid`` defines them.
    """
    if not isinstance(entry, dict):
        return AnswerSlot(state="bad", why="could not be read out of the kernel")
    slot = cast("dict[str, Any]", entry)
    state = slot.get("state")
    if state == "missing":
        return AnswerSlot(state="missing")
    if state != "ok":
        return AnswerSlot(state="bad", why=str(slot.get("why", "is not a usable grid")))
    arr = grid_from_rows(slot.get("grid"))
    if arr is None:
        return AnswerSlot(state="bad", why="is not a rectangular grid of whole numbers")
    height, width = arr.shape
    if not (1 <= height <= taaf.grid.MAX_SIDE and 1 <= width <= taaf.grid.MAX_SIDE):
        return AnswerSlot(state="bad", why=f"is {height}x{width}; each side must be 1..{taaf.grid.MAX_SIDE}")
    if not np.all((arr >= 0) & (arr < taaf.grid.N_COLORS)):
        return AnswerSlot(state="bad", why=f"holds values outside 0..{taaf.grid.N_COLORS - 1}")
    return AnswerSlot(state="ok", grid=taaf.grid.Grid(arr.astype(np.int8)))


def reading_from_payload(payload: dict[str, object] | None) -> AnswerReading:
    """Turn one kernel answer payload into a reading.

    ``None`` — no marker line, so the kernel died or timed out — reads as an
    unbound answer rather than a malformed one: nothing was learned about
    what the model had written, and reporting it as broken would send a
    repair prompt about a slot that may well be fine.
    """
    if payload is None:
        return AnswerReading(bound=False, why="the kernel produced no answer this round")
    if not payload.get("bound"):
        return AnswerReading(bound=False, why=str(payload.get("why", "test_out is not a list")))
    raw = payload.get("rows")
    rows: list[Any] = cast("list[Any]", raw) if isinstance(raw, list) else []
    return AnswerReading(
        bound=True,
        rows=tuple(_row_from_payload(row) for row in rows),
    )


def _row_from_payload(row: Any) -> tuple[AnswerSlot, ...]:
    """One question's attempt slots. A row that is not a list at all means the
    model replaced an entry of ``test_out`` with something else, which is
    reported as no usable attempt rather than raised."""
    if not isinstance(row, list):
        return ()
    return tuple(_slot_from_payload(entry) for entry in cast("list[Any]", row))


__all__ = ["AnswerReading", "AnswerSlot", "SlotState", "reading_from_payload"]
