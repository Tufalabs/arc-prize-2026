from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from .commons import read_json


def _slugify_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "idea"


class IdeaSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    description: str
    num_levels: int
    source_official_descriptions: tuple[dict[str, str], ...] = ()
    _idea_id: str = PrivateAttr(default="")

    @property
    def idea_id(self) -> str:
        return self._idea_id


def load_idea_specs(path: Path) -> list[IdeaSpec]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Ideas file must be a JSON array.")
    ideas: list[IdeaSpec] = []
    seen_counts: dict[str, int] = {}
    for raw in payload:
        try:
            idea = IdeaSpec.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        base_slug = _slugify_title(idea.title)
        count = seen_counts.get(base_slug, 0) + 1
        seen_counts[base_slug] = count
        object.__setattr__(idea, "_idea_id", base_slug if count == 1 else f"{base_slug}-{count:02d}")
        ideas.append(idea)
    return ideas


def _source_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.stem


def _load_official_description_source(path: Path, *, root: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Official description file is empty: {path}")
    try:
        relative_path = str(path.relative_to(root))
    except ValueError:
        relative_path = str(path)
    return {"name": _source_name(path), "path": relative_path, "text": text}


def load_official_description_pair_specs(paths: list[Path], *, root: Path, num_levels: int) -> list[IdeaSpec]:
    if len(paths) < 2:
        raise ValueError("official_description_paths must contain at least two files.")

    sources = [_load_official_description_source(path, root=root) for path in paths]
    ideas: list[IdeaSpec] = []
    seen_counts: dict[str, int] = {}
    for left, right in combinations(sources, 2):
        title = f"{left['name']} + {right['name']}"
        idea = IdeaSpec(
            title=title,
            description=(
                "Generate a new implementation-ready game spec by combining and recomposing "
                f"the official descriptions {left['name']!r} and {right['name']!r}."
            ),
            num_levels=num_levels,
            source_official_descriptions=(left, right),
        )
        base_slug = _slugify_title(title)
        count = seen_counts.get(base_slug, 0) + 1
        seen_counts[base_slug] = count
        object.__setattr__(idea, "_idea_id", base_slug if count == 1 else f"{base_slug}-{count:02d}")
        ideas.append(idea)
    return ideas
