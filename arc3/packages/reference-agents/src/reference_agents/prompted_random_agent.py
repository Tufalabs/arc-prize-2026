"""``PromptedRandomAgent``: proves the LLM seam with near-zero agent
logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ullm.conv

import platypus.diagnostics
from platypus import Action, GiveUp, LLMAgent, LLMMoveContext, Observation, frame_image
from reference_agents.random_agent import pick_random_action

PROMPT_TEMPLATE = "What is {a}+{b}? Answer with just the number."


@dataclass
class PromptedRandomAgent(LLMAgent):
    """Acts exactly like ``RandomAgent``, but decision moves first run one
    trivial LLM turn (a small random sum, drawn from ``ctx.rng`` so it is
    deterministic per seed) through the per-use turn-runner and hard-fail
    unless the right answer comes back — a real end-to-end check of
    backend, socket, profiles, and token accounting, with no agent logic
    in the way. ``randomize_sum=False`` pins the question to 3+5 for
    scripted backends that cannot do arithmetic (the stub tier).
    ``max_llm_calls`` caps the seam checks per game (``None`` = every
    decision move). Delivery-only calls (``level_won``) make no LLM
    call. ``attach_frame_image=True`` attaches the current frame as an
    RGB image to every seam-check prompt (scaled by ``image_scale``),
    extending the seam check to the multimodal path — proxy wire, capture
    blocks, replay pixel verification."""

    max_moves: int | None = 100
    click_on_color: bool = False
    profile: str = "play"  # sampling profile name; must exist in llm_config
    max_llm_calls: int | None = None
    randomize_sum: bool = True
    attach_frame_image: bool = False
    # Keep the seam-check conversation across moves (one growing history the
    # backend can continue by tokens) instead of a fresh one per call.
    keep_history: bool = False
    _history: list[Any] = field(default_factory=lambda: list[Any]())
    image_scale: int = 1

    # -- state --
    moves_made: int = 0
    llm_calls_made: int = 0

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp:
        """One move: as :class:`RandomAgent`, preceded by a seam check on the
        decision moves that are still under ``max_llm_calls``."""
        if obs.outcome == "level_won":
            return Action("RESET")  # delivery-only call; the return is discarded
        if self.max_moves is not None and self.moves_made >= self.max_moves:
            return GiveUp(f"reached max_moves={self.max_moves}")
        self.moves_made += 1
        if obs.outcome == "lost":
            platypus.diagnostics.current_span().kv(trivial=True)  # reflex RESET, no LLM
            return Action("RESET")
        if self.max_llm_calls is None or self.llm_calls_made < self.max_llm_calls:
            self.llm_calls_made += 1
            await self._check_llm_seam(ctx, obs)
        else:
            platypus.diagnostics.current_span().kv(trivial=True)  # random pick, no LLM
        return pick_random_action(ctx.rng, obs, self.click_on_color)

    async def _check_llm_seam(self, ctx: LLMMoveContext, obs: Observation) -> None:
        # The free-form agent span (I7 worked example): a producer-named
        # span with kv + a frame artifact, nesting under infra's move span
        # with the llm/turn span nesting under it in turn. No chunk class
        # registers this kind — the viewer's generic fallback renders it.
        """Run one trivial LLM turn and require the right answer back.

        Raising here ends the game as an agent error, which is the point:
        the check exists so that a broken backend, socket, profile or
        capture path fails loudly rather than scoring as a bad player."""
        with ctx.diag.span("agent/seam_check") as span:
            a, b = (ctx.rng.randint(1, 9), ctx.rng.randint(1, 9)) if self.randomize_sum else (3, 5)
            prompt = PROMPT_TEMPLATE.format(a=a, b=b)
            span.kv(question=prompt, expected=a + b)
            span.artifact("frame", obs.frame.data, codec="png-grid")
            turn_manager = self.new_turn_manager(
                ctx, profile=self.profile, system_prompt="You are a calculator. Answer with just the number."
            )
            if self.keep_history:
                turn_manager.conversation.elements.extend(self._history)
            content: ullm.conv.Content = ullm.conv.StringContent(text=prompt)
            if self.attach_frame_image:
                content = ullm.conv.ListContent(
                    elements=[content, frame_image(obs.frame, scale=self.image_scale, alt_text="current game frame")]
                )
            turn_manager.conversation.elements.append(ullm.conv.UserPrompt(content=content))
            await turn_manager.run()
            if self.keep_history:
                self._history = list(turn_manager.conversation.elements)
            answer = turn_manager.conversation.last_turn.last_response.final_content.to_string()
            # Containment, not equality: the profile is a thinking model with room
            # to reason, so a correct reply legitimately carries prose around the
            # number. What makes the loose test sufficient is repetition — with
            # randomize_sum the summands are redrawn from ctx.rng every call, so a
            # backend that merely echoes something digit-shaped has to survive
            # max_llm_calls independent draws to stay hidden. This is a liveness
            # probe for backend, socket, profile and capture wiring, not a marker.
            if str(a + b) not in answer:
                raise RuntimeError(
                    f"LLM seam check failed: expected {a + b!r} in the response to {prompt!r}, got {answer!r}"
                )
