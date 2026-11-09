from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, model_validator

ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


class PipelineRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idea_json: str | None = None
    official_description_paths: tuple[str, ...] = ()
    official_description_num_levels: int = 6
    implementation_prompt_path: str
    spec_instructions_path: str
    implementation_backend: Literal["codex", "claude"] = "codex"
    executor: Literal["local", "slurm"] = "local"
    slurm_partition: str = "llm"
    slurm_time: str = "01:30:00"
    slurm_mem: str = "8G"
    slurm_ntasks: int = 1
    slurm_cpus_per_task: int = 2
    model: str = "gpt-5.4"
    reasoning_effort: ReasoningEffort = "high"
    codex_reasoning_effort: ReasoningEffort | None = None
    spec_background: bool = False

    @model_validator(mode="after")
    def _validate_spec_source(self) -> PipelineRunConfig:
        has_ideas = bool(str(self.idea_json or "").strip())
        has_official_descriptions = bool(self.official_description_paths)
        if has_ideas == has_official_descriptions:
            raise ValueError("Set exactly one of `idea_json` or `official_description_paths`.")
        if has_official_descriptions and len(self.official_description_paths) < 2:
            raise ValueError("`official_description_paths` must contain at least two files.")
        if self.official_description_num_levels <= 0:
            raise ValueError("`official_description_num_levels` must be positive.")
        return self


def _load_yaml(path: Path) -> object:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    return payload


def load_run_config(path: Path | str) -> PipelineRunConfig:
    config_path = Path(path)
    return PipelineRunConfig.model_validate(_load_yaml(config_path))
