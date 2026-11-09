"""The shipped reference agents.

``FlipAgent`` is free and deterministic — the end-to-end check for every
deployment path, since what it should score is known before it runs.
``PromptedFlipAgent`` is the same player with one trivial LLM turn in front
of each answer, which makes it the check on everything the LLM layer wires
up: backend, socket, sampling profile, capture, and token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ullm.conv

from platypus.agent import Agent
from platypus.context import MoveContext
from platypus.contract import GiveUp
from platypus.llm import LLMAgent, LLMMoveContext, grid_image
from taaf.game import N_ATTEMPTS, GameView, Submission
from taaf.grid import Grid

PROMPT_TEMPLATE = "What is {a}+{b}? Answer with just the number."


def flip_lr(view: GameView) -> Submission:
    """Answer every question with its left-right mirror.

    Attempt 2 is the question unchanged rather than a copy of attempt 1, so
    the two-attempt machinery is genuinely exercised end to end instead of
    being duplicated past.
    """
    return Submission(attempts=tuple((Grid(np.fliplr(q.data)), q) for q in view.questions))


def all_black(view: GameView) -> Submission:
    """Answer every question with an all-zero grid of its shape.

    A deliberately wrong answer that is still well-formed, so a demo can show
    an answer being replaced without either one being a shape error.
    """
    return Submission(attempts=tuple((Grid(np.zeros_like(q.data)),) * N_ATTEMPTS for q in view.questions))


#: The answers the reference agents know how to build, by name.
ANSWERS = {"flip": flip_lr, "black": all_black}

#: What they submit, in order: two changes of mind before settling on the
#: mirror. The mirror decides the score; the rest is what exercises a lane
#: that steps back down, and a cancellation landing between two answers.
DEMO_ANSWERS = ("black", "flip", "black", "flip")


@dataclass
class FlipAgent(Agent):
    """Submits ``answers`` in order, settling on the left-right mirror.

    Deterministic and free, which makes it the end-to-end check for every
    deployment path: across ARC-AGI-2's public training set exactly one task
    (``67a3c6ac``) is a left-right flip throughout, and none of the public
    evaluation set is, so the run's outcome is known before it starts.
    """

    answers: tuple[str, ...] = DEMO_ANSWERS

    async def _do_move(self, ctx: MoveContext, view: GameView) -> GiveUp | None:
        """Submit every answer in ``answers``, in order."""
        for name in self.answers:
            ctx.submit(ANSWERS[name](view))
        return None


@dataclass
class PromptedFlipAgent(LLMAgent):
    """``FlipAgent`` with one trivial LLM turn before each answer.

    The turn asks a small random sum (drawn from ``ctx.rng``, so it is
    deterministic per seed) through the per-use turn-runner and hard-fails
    unless the right answer comes back — a real end-to-end check of backend,
    socket, profiles and token accounting, with no agent logic in the way.

    ``randomize_sum=False`` pins the question to 3+5 for scripted backends
    that cannot do arithmetic. ``max_llm_calls`` caps the seam checks per
    game (``None`` = one before every answer), so a run can show tokens
    landing on the answers that cost them and none on the rest.
    ``attach_grid_image=True`` also sends the first question as an RGB
    image, extending the check to the multimodal path — proxy wire, capture
    blocks, replay pixel verification. ``keep_history=True`` runs every
    seam check in ONE growing conversation rather than a fresh one each,
    which is what gives a continuation-capable backend served history to
    extend — the shape a traced game has to have for its steps to nest.
    """

    answers: tuple[str, ...] = DEMO_ANSWERS
    profile: str = "play"  # sampling profile name; must exist in llm_config
    randomize_sum: bool = True
    max_llm_calls: int | None = None
    attach_grid_image: bool = False
    image_scale: int = 8
    keep_history: bool = False

    # -- state --
    llm_calls_made: int = 0
    _seam_conversation: ullm.conv.Conversation | None = None

    async def _do_move(self, ctx: LLMMoveContext, view: GameView) -> GiveUp | None:
        """Submit every answer in ``answers``, each preceded by a seam check
        while the cap allows one."""
        for name in self.answers:
            if self.max_llm_calls is None or self.llm_calls_made < self.max_llm_calls:
                self.llm_calls_made += 1
                await self._check_llm_seam(ctx, view)
            ctx.submit(ANSWERS[name](view))
        return None

    async def _check_llm_seam(self, ctx: LLMMoveContext, view: GameView) -> None:
        """Run one trivial LLM turn and require the right answer back.

        Raising here ends the game as an agent error, which is the point:
        the check exists so that a broken backend, socket, profile or
        capture path fails loudly rather than scoring as a bad player.
        """
        # The free-form agent span: a producer-named span with kv and a grid
        # artifact, nesting under infra's span with the llm/turn span nesting
        # under it in turn. No chunk class registers this kind — the viewer's
        # generic fallback renders it.
        with ctx.diag.span("agent/seam_check") as span:
            a, b = (ctx.rng.randint(1, 9), ctx.rng.randint(1, 9)) if self.randomize_sum else (3, 5)
            prompt = PROMPT_TEMPLATE.format(a=a, b=b)
            span.kv(question=prompt, expected=a + b)
            if view.questions:
                span.artifact("question", view.questions[0].data, codec="png-grid")
            turn_manager = self.new_turn_manager(
                ctx,
                profile=self.profile,
                system_prompt="You are a calculator. Answer with just the number.",
                conversation=self._seam_conversation,
            )
            if self.keep_history:
                self._seam_conversation = turn_manager.conversation
            content: ullm.conv.Content = ullm.conv.StringContent(text=prompt)
            if self.attach_grid_image and view.questions:
                content = ullm.conv.ListContent(
                    elements=[
                        content,
                        grid_image(view.questions[0], scale=self.image_scale, alt_text="the first question"),
                    ]
                )
            turn_manager.conversation.elements.append(ullm.conv.UserPrompt(content=content))
            await turn_manager.run()
            answer = turn_manager.conversation.last_turn.last_response.final_content.to_string()
            # Containment, not equality: a thinking model legitimately carries
            # prose around the number. What makes the loose test sufficient is
            # repetition — with randomize_sum the summands are redrawn from
            # ctx.rng every call, so a backend that merely echoes something
            # digit-shaped has to survive every draw to stay hidden. This is a
            # liveness probe for backend, socket, profile and capture wiring,
            # not a marker.
            if str(a + b) not in answer:
                raise RuntimeError(
                    f"LLM seam check failed: expected {a + b!r} in the response to {prompt!r}, got {answer!r}"
                )


__all__ = [
    "ANSWERS",
    "DEMO_ANSWERS",
    "PROMPT_TEMPLATE",
    "FlipAgent",
    "PromptedFlipAgent",
    "all_black",
    "flip_lr",
]
