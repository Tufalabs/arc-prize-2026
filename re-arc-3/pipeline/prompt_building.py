from __future__ import annotations

from pathlib import Path


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip()


def load_spec_generation_prompt(*, instructions_path: Path) -> str:
    return load_prompt_template(instructions_path)


def load_spec_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip()


def build_implementation_prompt(*, template: str, spec_text: str) -> str:
    return f"{template.rstrip()}\n\nGame spec:\n{spec_text}\n"
