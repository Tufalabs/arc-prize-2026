"""The duck in takes: ``platypus.multi_take`` over ducks — two takes, or up to
ten queued by how far they agree.

Two things here are the duck's own. The build, which lives in this package
because platypus cannot import the duck. And what a duck counts per game
rather than per take, carried across the take boundary: its token budget,
read off the game's one backend meter, and its history epoch, which the
trace collector nests a game's steps under.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import ullm.backend

import taaf.game
from duck import presets
from duck.agent import LR_REFLECTION, DuckAgent, PuzzleLayout
from duck.prompts import DuckPrompts, Prompts
from platypus.contract import GiveUp
from platypus.llm import LLMAgent, LLMMoveContext
from platypus.multi_take import MultiTakeAgent, TieredTakeAgent

#: What the second take's audit ranks behind its spend from its first compaction on
#: (``DuckAgent.audit_handicap_tokens``; 2026-10-05): ten million, more than
#: any game spends, so the rest of the audit waits for capacity no game that has not
#: deferred wants. The take-gap simulation read it +0.07 at 12 h over the audit on
#: the first take only, and +0.16 over both undeferred (arc2-experiments
#: ``take_gap/audit_arms.py``).
SECOND_TAKE_AUDIT_HANDICAP_TOKENS = 10_000_000

#: The second take's sampling temperature (2026-10-06), on its play
#: profile, which its audit and summaries sample on too; it also plays without
#: the file tools. Of sweep 9's knobs, the play's reasoning temperature and the
#: file tools were the two besides the checkpoint (which both takes share) whose
#: effect differed from task to task more than chance (p=0.004 and 0.003, the
#: knob permuted over arms). As a second take's change alone the sweep read
#: neither moving pass@2 (-0.04 and -0.30 points; ``evidence.md``, "Duck sweeps
#: 8 and 9").
SECOND_TAKE_TEMPERATURE = 0.5


@dataclass(kw_only=True)
class MultiTakeDuck(MultiTakeAgent):
    """A ``MultiTakeAgent`` whose takes are ducks.

    Each take's ``max_tokens_per_game`` bounds that take: the meter it reads
    is the game's, so the next take's allowance starts from what the game
    has spent (``_start_take``).
    """

    def __post_init__(self) -> None:
        """Refuse a take that is not a duck, then construct as the base does."""
        for take in self.takes:
            if not isinstance(cast(object, take), DuckAgent):
                raise TypeError(f"a {type(self).__name__} take must be a DuckAgent; got {type(take).__name__}")
        super().__post_init__()

    def prepare_for_backend(self, *, settle_engine: bool = True) -> None:
        """Refuse what a later take would get wrong, then prepare as the base
        does. Here rather than at construction alone, because a take is
        normally changed through ``takes[i]`` after it."""
        for index, take in enumerate(self.takes[1:], start=1):
            self._refuse_game_meter_readers(index, cast(DuckAgent, take))
        for index, take in enumerate(self.takes):
            self._refuse_per_game_observers(index, cast(DuckAgent, take))
        super().prepare_for_backend(settle_engine=settle_engine)

    @staticmethod
    def _refuse_per_game_observers(index: int, duck: DuckAgent) -> None:
        """Refuse the observers whose contract is the whole game.

        A ``ContextObserver`` is one per game and is told of every history
        rewrite; a take is a copy with its own observer, and the second take
        opens a new conversation that no rewrite declares. So an observer on
        any take would see half a game and an undeclared epoch jump; the
        response observer, copied with its take, would split the game's
        responses the same way.
        """
        refused: list[str] = []
        if duck.context_observer is not None:
            refused.append("context_observer observes one game's history, and each take is a copy with its own")
        if duck.response_observer is not None:
            refused.append("response_observer would be copied into each take and see half the game's responses")
        if refused:
            raise ValueError(f"take {index + 1} carries a per-game observer: " + "; ".join(refused) + ".")

    @staticmethod
    def _refuse_game_meter_readers(index: int, duck: DuckAgent) -> None:
        """Refuse the settings that read the game's token meter as the take's.

        In any take after the first the meter already holds the takes before
        it, which the allowance carried across (``_start_take``) accounts for
        and these do not.
        """
        refused: list[str] = []
        if duck.first_response_max_tokens is not None:
            refused.append("first_response_max_tokens caps a reply only while the game has generated nothing")
        if getattr(duck.prompts, "budget_line", False):
            refused.append("prompts.budget_line prints the allowance raised by the takes before")
        if getattr(duck.prompts, "cost_model", "none") == "explicit":
            refused.append('prompts.cost_model="explicit" states that raised allowance in the system prompt')
        if refused:
            raise ValueError(f"take {index + 1} reads the game's token meter as its own: " + "; ".join(refused) + ".")

    @classmethod
    def build(
        cls,
        *,
        engine: str = presets.DEFAULT_ENGINE,
        environment: str = presets.DEFAULT_ENVIRONMENT,
        serving: Literal["direct", "server"] = "direct",
        backend_factory: Any = None,
        **overrides: Any,
    ) -> Self:
        """Two takes of the duck ``DuckAgent.build`` makes from the same
        arguments, each asked to put its likeliest answer first.

        Everything passes through to that one build; the first take is that
        duck as built, and the second differs from it in what is chosen to
        move WHICH tasks it solves rather than how many. Three from sweep 7
        (2026-09-21): the digit rows in its task prompt, which the
        first leaves out, so the no-hand-arithmetic caveat takes its
        digit-rows wording there; the shorter brevity ask; and the full latest
        response kept after a summary. Each read as no worse than the first
        take's value in the sweep and none resolved. Two from sweep 9
        (2026-10-06): no file tools, and its play profile at
        ``SECOND_TAKE_TEMPERATURE``. Its audit, too, waits from its first
        compaction behind every game that has not deferred
        (``SECOND_TAKE_AUDIT_HANDICAP_TOKENS``), where the first take's ranks
        as its play does.
        Anything else a caller wants between them is set on ``takes[i]``
        afterwards.
        """
        duck = DuckAgent.build(
            engine=engine,
            environment=environment,
            serving=serving,
            backend_factory=backend_factory,
            **overrides,
        )
        duck.prompts = _likeliest_first(duck.prompts)
        built = cls(takes=(duck, duck))
        second = cast(DuckAgent, built.takes[1])
        second.text_grids = True
        second.summary_tail = "full"
        second.prompts = dataclasses.replace(cast(DuckPrompts, second.prompts), brevity_line=True, brevity="half")
        # Its audit, from its first compaction on, waits behind every game that has
        # not deferred (2026-10-05).
        second.audit_handicap_tokens = SECOND_TAKE_AUDIT_HANDICAP_TOKENS
        second.file_tools = "none"
        # On its own configuration: the takes share the engine, not the sampling.
        play = second.llm_config.sampling_profiles[second.profile]
        if not isinstance(play, ullm.backend.GenericSamplingParameters):
            raise TypeError(f"the second take's profile {second.profile!r} has no temperature to set")
        second.llm_config.sampling_profiles = {
            **second.llm_config.sampling_profiles,
            second.profile: dataclasses.replace(play, temperature=SECOND_TAKE_TEMPERATURE),
        }
        return built

    def _start_take(self, take: LLMAgent, *, previous: LLMAgent, spent_tokens: int) -> None:
        """Carry the game's counts into the next take: its token allowance is
        its own on top of what the game has spent, and its history opens an
        epoch past every one the previous take declared, so no step of it
        reads as a continuation of the previous take's conversation."""
        duck = cast(DuckAgent, take)
        if duck.max_tokens_per_game is not None:
            duck.max_tokens_per_game += spent_tokens
        duck.history_epoch = max(duck.history_epoch, cast(DuckAgent, previous).history_epoch + 1)

    async def _do_move(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Coordinate opted-in takes' layouts before playing them on the shared game.

        The first take is unchanged. Each dimension exhausts its possibilities
        before reuse, counting the first take's identity layout as already used.
        Disabled later takes keep their original layout and consume no draws.
        """
        rng = random.Random(f"multi_take_augmentation:{ctx.rng.getstate()!r}")
        train_identity = tuple(range(len(view.given)))
        test_identity = tuple(range(len(view.questions)))
        used_train = {train_identity}
        used_test = {test_identity}
        orientations = list(range(1, 8))
        for index, take in enumerate(self.takes):
            duck = cast(DuckAgent, take)
            if index == 0:
                duck._take_layout = PuzzleLayout(transform=0, train_order=train_identity, test_order=test_identity)
            elif duck.use_dihedral_and_order_permutation:
                if not orientations:
                    orientations = list(range(8))
                transform = LR_REFLECTION if index == 1 else rng.choice(orientations)
                orientations.remove(transform)
                duck._take_layout = PuzzleLayout(
                    transform=transform,
                    train_order=_draw_permutation(len(view.given), rng, used_train),
                    test_order=_draw_permutation(len(view.questions), rng, used_test),
                )
        return await super()._do_move(ctx, view)


@dataclass(kw_only=True)
class TieredTakeDuck(MultiTakeDuck, TieredTakeAgent):
    """A ``TieredTakeAgent`` whose takes are ducks, every one of them
    ``MultiTakeDuck``'s first take."""

    @classmethod
    def build(
        cls,
        *,
        engine: str = presets.DEFAULT_ENGINE,
        environment: str = presets.DEFAULT_ENVIRONMENT,
        serving: Literal["direct", "server"] = "direct",
        backend_factory: Any = None,
        **overrides: Any,
    ) -> Self:
        """The first take ``MultiTakeDuck.build`` makes from the same arguments,
        played by every take, with its per-take net: no take after the first
        changes its sampling or its prompts (2026-10-27)."""
        pair = MultiTakeDuck.build(
            engine=engine,
            environment=environment,
            serving=serving,
            backend_factory=backend_factory,
            **overrides,
        )
        return cls(takes=(pair.takes[0],) * cls.n_takes, max_runtime_s_per_take=pair.max_runtime_s_per_take)


def _draw_permutation(size: int, rng: random.Random, used: set[tuple[int, ...]]) -> tuple[int, ...]:
    """Draw without replacement until all orders are used, storing only drawn orders."""
    if len(used) == math.factorial(size):
        used.clear()
    while True:
        order = list(range(size))
        rng.shuffle(order)
        result = tuple(order)
        if result not in used:
            used.add(result)
            return result


def _likeliest_first(prompts: Prompts) -> DuckPrompts:
    """``prompts`` with the likeliest-first line on, in its own class."""
    if not isinstance(prompts, DuckPrompts):
        raise TypeError(
            f"a two-take duck asks for the likeliest answer first through DuckPrompts; "
            f"{type(prompts).__name__} has no such line"
        )
    return dataclasses.replace(prompts, likeliest_first_line=True)


__all__ = ["SECOND_TAKE_AUDIT_HANDICAP_TOKENS", "SECOND_TAKE_TEMPERATURE", "MultiTakeDuck", "TieredTakeDuck"]
