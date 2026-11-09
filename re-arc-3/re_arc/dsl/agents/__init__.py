from __future__ import annotations

import importlib
import logging
import pkgutil

from ..core import DslAgent

__all__: list[str] = []
_LOGGER = logging.getLogger(__name__)

for module_info in pkgutil.iter_modules(__path__):
    module_name = module_info.name
    if module_name.startswith("_"):
        continue
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        # Some generated agents depend on environment modules that may not
        # exist in the current checkout. Skip those agents rather than
        # breaking package import for unrelated games.
        if exc.name and exc.name.startswith("re_arc.environment_files."):
            _LOGGER.debug(
                "Skipping DSL agent module %s due to missing environment dependency %s", module_name, exc.name
            )
            continue
        raise
    for symbol, obj in vars(module).items():
        if (
            isinstance(obj, type)
            and issubclass(obj, DslAgent)
            and obj is not DslAgent
            and obj.__module__ == module.__name__
        ):
            globals()[symbol] = obj
            __all__.append(symbol)

__all__ = sorted(set(__all__))
