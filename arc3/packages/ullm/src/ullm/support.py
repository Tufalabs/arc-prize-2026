import asyncio
import sys
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

import typeguard

# Global variable to control debugging mode, can be 0, 1, 2, or 3, which each mode more strict than the previous
debugging_mode: int = 2


@dataclass
class BaseClass:
    """Base dataclass with frozen attributes and runtime type checking.

    After construction, new attributes cannot be added. When debugging_mode > 0,
    check_constraints() validates field types and subclass-specific constraints.

    Note:
        Subclasses should override _check_constraints() for custom validation.
    """

    _is_frozen: bool = field(default=False, init=False, repr=False)  # Freeze flag

    def check_constraints(self, debugging_mode_offset: int = 0) -> None:
        """Validate types and constraints if debugging_mode > 0.

        Args:
            debugging_mode_offset: Temporarily adjust global debugging_mode.
        """
        global debugging_mode
        if debugging_mode == 0:
            return
        debugging_mode = debugging_mode + debugging_mode_offset
        try:
            self._check_types()
            self._check_constraints()
            if debugging_mode >= 2:
                self._check_nested_constraints()
        finally:
            debugging_mode = debugging_mode - debugging_mode_offset

    def _check_nested_constraints(self) -> None:
        """Recursively check constraints on nested BaseClass instances."""
        for field_info in fields(self):
            value = getattr(self, field_info.name)

            if isinstance(value, BaseClass):
                value.check_constraints()
            elif isinstance(value, list):
                for item in value:  # type: ignore
                    if isinstance(item, BaseClass):
                        item.check_constraints()
            elif isinstance(value, dict):
                for dict_value in value.values():  # type: ignore
                    if isinstance(dict_value, BaseClass):
                        dict_value.check_constraints()

    def _check_constraints(self) -> None:
        """Override in subclasses for custom constraint validation."""
        pass

    def _check_types(self) -> None:
        """Verify all fields match their declared type hints.
        Builds a namespace from the class's module globals so that forward
        references (e.g. TYPE_CHECKING imports) resolve at runtime. An
        annotation that still will not resolve costs only its own field: that
        one goes unchecked and every other field is still verified."""
        module = sys.modules.get(self.__class__.__module__, None)
        globalns = dict(vars(module)) if module else {}
        try:
            type_hints = typing.get_type_hints(self.__class__, globalns=globalns)
        except (NameError, AttributeError):
            # Whole-class resolution is all-or-nothing; retry field by field
            type_hints = _resolvable_type_hints(self.__class__, globalns)
        for field_info in fields(self):
            field_name = field_info.name
            if field_name not in type_hints:  # unresolvable annotation
                continue
            actual_value = getattr(self, field_name)
            typeguard.check_type(actual_value, type_hints[field_name])

    def __post_init__(self) -> None:
        object.__setattr__(self, "_is_frozen", True)

    def __setattr__(self, key: str, value: Any) -> None:
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)


def _resolvable_type_hints(cls: type, globalns: dict[str, Any]) -> dict[str, Any]:
    """Type hints for the fields of cls whose annotations resolve, one at a time.

    typing.get_type_hints on a class is all-or-nothing: one annotation naming a
    module nobody imported takes the whole class's hints down with it. Walking
    the MRO and resolving each annotation on its own confines that to the field
    that caused it.

    Args:
        cls: The class whose annotations to resolve.
        globalns: Namespace to resolve against, layered over each defining
            class's own module globals.
    """
    hints = dict[str, Any]()
    # Reversed MRO: a subclass re-annotating an inherited field wins.
    for base in reversed(cls.__mro__):
        base_module = sys.modules.get(base.__module__, None)
        base_globalns = {**(vars(base_module) if base_module else {}), **globalns}
        for name, annotation in base.__dict__.get("__annotations__", {}).items():
            # get_type_hints resolves nested forward refs (list["Foo"]) that a
            # bare eval of the annotation would not, so route one through it.
            def _probe() -> None: ...

            _probe.__annotations__ = {"field": annotation}
            try:
                hints[name] = typing.get_type_hints(_probe, globalns=base_globalns)["field"]
            except (NameError, AttributeError):
                hints.pop(name, None)  # a base resolved it, this subclass does not
    return hints


def diagnose_pickle(obj: Any, path: str = "root", _seen: set[int] | None = None) -> list[str]:
    """Recursively try to pickle each field of a dataclass, returning dotted paths to unpicklable leaves."""
    import dataclasses
    import pickle

    if _seen is None:
        _seen = set()

    # Avoid infinite recursion on circular references
    obj_id = id(obj)
    if obj_id in _seen:
        return []
    _seen.add(obj_id)

    # Try pickling the whole object first; if it works, nothing to report
    try:
        pickle.dumps(obj)
        return []
    except Exception:
        pass

    # If it's a dataclass, recurse into fields
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        problems = list[str]()
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            problems.extend(diagnose_pickle(val, f"{path}.{f.name}", _seen))
        # If no field-level problems found, the object itself is the issue (e.g. __getstate__)
        if not problems:
            try:
                pickle.dumps(obj)
            except Exception as e:
                problems.append(f"{path}: {type(obj).__name__} — {e}")
        return problems

    # For lists/tuples, check elements
    if isinstance(obj, (list, tuple)):
        problems = list[str]()
        for i, item in enumerate(obj):
            problems.extend(diagnose_pickle(item, f"{path}[{i}]", _seen))
        return problems

    # For dicts, check keys and values
    if isinstance(obj, dict):
        problems = list[str]()
        for k, v in obj.items():
            problems.extend(diagnose_pickle(v, f"{path}[{k!r}]", _seen))
        return problems

    # Leaf that can't be pickled
    try:
        pickle.dumps(obj)
        return []
    except Exception as e:
        return [f"{path}: {type(obj).__name__} — {e}"]


@dataclass
class Processor(BaseClass):
    """Base for a one-shot async unit of work: subclasses implement ``_run``,
    callers ``await run()`` once from the ``ready`` state. The state field is
    the record of what happened, and ``exception`` is set exactly when it is
    ``errored`` (enforced by the constraint check). Tools, ``ResponseManager``
    and ``TurnManager`` are all Processors; ``rearm()`` returns a completed one
    to ``ready`` for reuse."""

    state: Literal["ready", "running", "completed", "cancelled", "errored"] = field(default="ready", init=False)
    exception: Optional[BaseException] = field(default=None, init=False, repr=False)

    def _check_constraints(self) -> None:
        """``exception`` is set exactly when the processor errored."""
        if self.state == "errored":
            assert self.exception is not None, "Exception must be set when state is 'errored'"
        else:
            assert self.exception is None, "Exception must be None unless state is 'errored'"
        super()._check_constraints()

    async def run(self) -> None:
        """Run the processor"""
        self.check_constraints()
        assert self.state == "ready", "Processor must be in 'ready' state to run"
        self.state = "running"
        try:
            await self._run()  # to be implemented by subclass
            self.state = "completed"
        except BaseException as e:
            self.exception = e
            if isinstance(e, asyncio.CancelledError):
                self.state = "cancelled"
            else:
                self.state = "errored"
            raise
        self.check_constraints()

    def run_blocking(self) -> None:
        """Run the processor in a blocking manner; typically the external entry point"""
        self.check_constraints(debugging_mode_offset=1)  # Heavier check on external interface
        asyncio.run(self.run())

    def rearm(self) -> None:
        """Reset after completion to run again. This does not necessarily mean cleaning the whole state!"""
        self.check_constraints()
        assert not self.state == "running", "Cannot rearm while running; this is meant for restarting after completion"
        self._rearm()
        self.state = "ready"
        self.exception = None
        self.check_constraints()

    def peek(self) -> None:
        """Update intermediate outputs; default behavior is to do nothing"""
        assert self.state == "running", "Peek can only be called while running"
        self._peek()

    def _rearm(self) -> None:
        """Rearm behavior to be implemented by subclasses; default is to do nothing."""
        pass

    def _peek(self) -> None:
        """Peek behavior to be implemented by subclasses; default is to do nothing."""
        pass

    async def _run(self) -> None:
        """The actual processing logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _run method")


@dataclass
class ProcessorMulti(Processor):
    """Runs multiple Processors in parallel; cancels all if any fails."""

    children: list[Processor] = field(default_factory=lambda: list[Processor]())

    async def _run(self) -> None:
        """Run all children in parallel using TaskGroup."""
        async with asyncio.TaskGroup() as tg:
            for child in self.children:
                tg.create_task(child.run())

    def _rearm(self) -> None:
        """Rearm all children."""
        for child in self.children:
            child.rearm()

    def _peek(self) -> None:
        """Peek into all running children."""
        for child in self.children:
            if child.state == "running":
                child.peek()
