"""Parameter sweeping: one run covering several agent configurations.

``SweepAgent`` / ``LLMSweepAgent`` wrap another agent and configure it per
pass, so a benchmark's passes become the sweep's arms; ``collect`` turns the
finished job dirs back into rows, at the unit each knob was drawn at.
What varies is declared as a :class:`~sweeper.knobs.Design` — the dedicated
DoE setup the contract asks for (`requirements.md` A6), not a registry of
every settable field.

``example_analysis`` sweeps the random reference agent end to end.
"""

from sweeper.agent import SWEEP_LINE, LLMSweepAgent, SweepAgent, SweepCore
from sweeper.collect import (
    DEFAULT_BUDGET_TOKENS,
    MAX_CANCELLED_FRACTION,
    MAX_CRASHED_FRACTION,
    UNCACHED_INPUT_EXCHANGE_RATE,
    UNCACHED_INPUT_WEIGHT,
    LoadedJob,
    SweepRow,
    UnitRow,
    by_arm,
    by_job,
    collect_rows,
    load_job,
    load_jobs,
    parse_sweep_lines,
    rows_from_jobs,
    score_by_arm,
    unit_for,
    verify_arms,
)
from sweeper.jobs import (
    DESIGN_STAMP,
    JobStatus,
    check_reusable,
    design_moved,
    finished_job_dirs,
    is_finished,
    job_status,
    write_design_stamp,
)
from sweeper.knobs import (
    RECORD,
    Choice,
    Derive,
    Design,
    Field,
    Knob,
    LogUniformInt,
    Record,
    Uniform,
    UniformInt,
    apply_values,
    draw_values,
)
from sweeper.seeds import DEFAULT_PLAY_SEED_STRIDE, arm_seed_block, play_seed

__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "UNCACHED_INPUT_EXCHANGE_RATE",
    "UNCACHED_INPUT_WEIGHT",
    "MAX_CANCELLED_FRACTION",
    "MAX_CRASHED_FRACTION",
    "DEFAULT_PLAY_SEED_STRIDE",
    "RECORD",
    "DESIGN_STAMP",
    "SWEEP_LINE",
    "Choice",
    "Derive",
    "Design",
    "Field",
    "JobStatus",
    "Knob",
    "LLMSweepAgent",
    "LoadedJob",
    "LogUniformInt",
    "UniformInt",
    "Record",
    "SweepAgent",
    "SweepCore",
    "SweepRow",
    "Uniform",
    "UnitRow",
    "apply_values",
    "arm_seed_block",
    "by_arm",
    "by_job",
    "collect_rows",
    "draw_values",
    "check_reusable",
    "design_moved",
    "finished_job_dirs",
    "is_finished",
    "job_status",
    "load_job",
    "load_jobs",
    "parse_sweep_lines",
    "rows_from_jobs",
    "play_seed",
    "score_by_arm",
    "unit_for",
    "verify_arms",
    "write_design_stamp",
]
