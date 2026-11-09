"""A summary written inside a thinking block the model never closed is still a summary.

With ``summary_thinking`` on, the summary request opens on the ordinary
generation prompt and the model reasons first. A model can then write the whole
summary as its thinking and stop there: the reply's final text is empty and the
summary is all reasoning. The backend rejects an empty reply as malformed by
default, which cost a game in sweep 12's pilot (rs902, tr87); the summary
request asks for such a reply to be kept (``keep_empty_reply``), and the duck
reads the summary from the reasoning. Played through a scripted stub on real
kernels, over the proxy, so the ask has to travel with the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arcengine
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent

SUMMARY_TEXT = "SUMMARY FROM THINKING: UP advances; keep pressing UP."
WRAPPER = "[context compacted — the following is your own summary of your work so far]"

#: What the stub saw, for the test to read back: the process runs in-process.
SEEN: dict[str, list[str]] = {"summary_requests": [], "later_prompts": []}


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running one snippet."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})]
        )
    )


@dataclass
class _ThinkingSummaryStub(ullm.backends.stub.StubBackend):
    """Plays the script, and answers a summary request with the summary as
    reasoning and no final text -- the reply the pilot's model gave."""

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        users = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        last = users[-1].content.to_string() if users else ""
        if last.startswith("Your context window is nearly full"):
            SEEN["summary_requests"].append(last)
            return ullm.conv.LLMResponse(
                final_content=ullm.conv.EmptyContent(),
                reasoning_content=ullm.conv.StringContent(text=SUMMARY_TEXT),
            )
        SEEN["later_prompts"].extend(u.content.to_string() for u in users)
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        if llm_output.reasoning_content.to_string() == SUMMARY_TEXT:
            # Not a scripted response: fill it without advancing the script.
            self._fill_into(llm_output, element, conversation)
            return
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


async def test_a_summary_left_in_the_thinking_is_installed_and_play_goes_on(tmp_path: Path) -> None:
    SEEN["summary_requests"].clear()
    SEEN["later_prompts"].clear()
    script = [_python(f"print('look {i} ' * 1500)\nsubmit_action(['UP'])\n") for i in range(16)]
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=_ThinkingSummaryStub(responses=script)),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        context_window_tokens=40000,
        reply_cap_tokens=2000,
        engine_ceiling_tokens=262144,
        compaction_strategy="summarize",
        summary_thinking=True,
        max_tokens_per_level=None,
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 16], label="thinking_summary")],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=tmp_path,
    )
    await benchmark.run()

    (run,) = benchmark.game_runs
    assert run.state == "won", run.state
    assert SEEN["summary_requests"], "no summary fired: the window is too wide for the script"
    # The summary the model left in its thinking is what the rebuilt history carries.
    assert f"{WRAPPER}\n\n{SUMMARY_TEXT}" in SEEN["later_prompts"]
