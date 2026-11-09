from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any, cast

from arc_agi.local_wrapper import LocalEnvironmentWrapper  # type: ignore[import-untyped]
from arcengine import ARCBaseGame


def _to_snake_case(name: str) -> str:
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1).lower()


def _patched_load_game_class(self: LocalEnvironmentWrapper, seed: int = 0) -> None:
    if self.environment_info.local_dir is None:
        return

    local_dir = Path(self.environment_info.local_dir)
    class_name = self.environment_info.class_name

    if not class_name:
        self.logger.error("Cannot load game class: class_name is not set for %s", self.environment_info.game_id)
        return

    snake_case_name = _to_snake_case(class_name)
    candidates = [
        local_dir / f"{class_name.lower()}.py",
        local_dir / f"{class_name}.py",
        local_dir / f"{snake_case_name}.py",
    ]

    game_file = next((path for path in candidates if path.exists()), None)
    if game_file is None:
        self.logger.error("Game source file not found. Looked in: %s", ", ".join(str(path) for path in candidates))
        return

    try:
        source_code = game_file.read_text(encoding="utf-8")
    except Exception as exc:
        self.logger.exception("Failed to read game source from %s: %s", game_file, exc)
        return

    module_name = f"arc_agi_3.{self.environment_info.game_id}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if spec is None:
        self.logger.error("Could not create module spec for %s", module_name)
        return

    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(game_file)
    sys.modules[module_name] = module
    try:
        exec(source_code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        self.logger.exception("Error executing game source for %s: %s", self.environment_info.game_id, exc)
        return

    cls: Any | None = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        self.logger.error("Expected class `%s` not found in %s", class_name, game_file)
        return

    if not issubclass(cls, ARCBaseGame):
        self.logger.error("Class `%s` exists but is not a subclass of ARCBaseGame", class_name)
        return

    signature = inspect.signature(cls)
    kwargs = {"seed": seed} if seed is not None and "seed" in signature.parameters else {}
    game_class = cast(Any, cls)
    self._game = game_class(**kwargs)
    self.logger.info("Successfully loaded game class %s from %s", class_name, game_file)


def patch_arc_agi_local_wrapper() -> None:
    current = getattr(LocalEnvironmentWrapper._load_game_class, "__module__", "")
    if current == __name__:
        return
    LocalEnvironmentWrapper._load_game_class = _patched_load_game_class
