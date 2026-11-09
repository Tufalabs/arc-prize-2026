"""The Kimi student duck (``DuckAgent.build(kimi_student=True)``): the
settings the Kimi K3.1 Pro teacher traces were played with, on any engine
coordinate, an override still winning, and nothing of the API teacher's
own."""

from __future__ import annotations

from duck import DuckAgent
from duck.agent import KIMI_STUDENT


def test_kimi_student_sets_every_teacher_value_and_an_override_still_wins() -> None:
    agent = DuckAgent.build(kimi_student=True, summary_thinking=False)
    for name, value in KIMI_STUDENT.items():
        assert getattr(agent, name) == (False if name == "summary_thinking" else value), name
    plain = DuckAgent.build()
    assert any(getattr(plain, name) != value for name, value in KIMI_STUDENT.items()), "kimi_student is the default"


def test_the_student_keeps_what_only_the_api_teacher_lacks() -> None:
    """The API teacher's own settings are not the student's: it counts by
    tokens, bans the tool-call token in a summary, and keeps the column's
    reasoning level, the one the traces are rendered at in training."""
    student = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200", kimi_student=True)
    column = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    assert student.compaction_forbidden_token_ids == column.compaction_forbidden_token_ids != ()
    assert student.token_native_context and student.usage_window_tokens is None
    assert student.reasoning_effort == column.reasoning_effort == "xhigh"
