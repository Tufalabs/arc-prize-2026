from __future__ import annotations

import copy
import difflib
import json
import logging
import os
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from arc_agi import Arcade, OperationMode  # type: ignore[import-untyped]
from arcengine import GameAction

from .arc_agi_compat import patch_arc_agi_local_wrapper
from .types import Action, ActionData, ActionValue, Environment, Frame, FrameLayer, IntCoercible, Observation, Renderer

if TYPE_CHECKING:
    from .reward_env import TransitionRewardEnv

PACKAGE_ENVIRONMENTS_DIR = Path(__file__).resolve().parent / "environment_files"
PACKAGE_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

patch_arc_agi_local_wrapper()


_MOVE_ACTION_TO_DELTA = {
    1: (0, -1),  # up
    2: (0, 1),  # down
    3: (-1, 0),  # left
    4: (1, 0),  # right
}
_DELTA_TO_MOVE_ACTION = {delta: action_id for action_id, delta in _MOVE_ACTION_TO_DELTA.items()}


def default_environments_dir() -> str:
    return str(PACKAGE_ENVIRONMENTS_DIR)


def default_datasets_dir() -> str:
    return str(PACKAGE_DATASETS_DIR)


def _coerce_log_level(level: object, default: int = logging.WARNING) -> int:
    if isinstance(level, int):
        return int(level)
    text = str(level or "").strip().upper()
    if not text:
        return int(default)
    resolved = getattr(logging, text, None)
    if isinstance(resolved, int):
        return int(resolved)
    return int(default)


def _build_arcade_logger(level: object, *, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    resolved = _coerce_log_level(level)
    logger.setLevel(resolved)
    for existing_handler in logger.handlers:
        existing_handler.setLevel(resolved)
    return logger


def _normalize_id(value: str) -> str:
    return str(value).strip().lower()


def _normalize_tag(value: str) -> str:
    return str(value).strip().lower()


def _base_game_id(game_id: str) -> str:
    return _normalize_id(game_id).split("-", 1)[0]


def _normalize_specs(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [_normalize_id(values)]
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = _normalize_id(value)
        if normalized:
            out.append(normalized)
    return out


def _normalize_dataset_specs(values: Sequence[str | Path] | str | Path | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        text = str(values).strip()
        return [text] if text else []
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _discover_game_metadata_from_dir(environments_dir: str | Path) -> dict[str, set[str]]:
    base = Path(environments_dir).expanduser()
    if not base.exists() or not base.is_dir():
        return {}

    metadata_by_game_id: dict[str, set[str]] = {}
    for metadata_file in base.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_file.read_text())
        except Exception:
            continue
        game_id = metadata.get("game_id")
        if game_id:
            normalized_game_id = _normalize_id(game_id)
            raw_tags = metadata.get("tags", [])
            metadata_by_game_id[normalized_game_id] = {
                _normalize_tag(tag) for tag in raw_tags if isinstance(tag, str) and _normalize_tag(tag)
            }
    return metadata_by_game_id


@dataclass(frozen=True)
class _DatasetDefinition:
    path: Path
    names: frozenset[str]
    tags: frozenset[str]
    game_ids: tuple[str, ...]


def _dataset_list_from_payload(payload: object, *, path: Path) -> list[object]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset {path} must be a JSON list or object.")

    for key in ("game_ids", "games", "ids"):
        raw_game_ids = payload.get(key)
        if raw_game_ids is None:
            continue
        if not isinstance(raw_game_ids, list):
            raise ValueError(f"Dataset {path} field `{key}` must be a list.")
        return raw_game_ids

    raise ValueError(f"Dataset {path} must include a `game_ids`, `games`, or `ids` list.")


def _extract_dataset_game_ids(payload: object, *, path: Path) -> tuple[str, ...]:
    raw_entries = _dataset_list_from_payload(payload, path=path)
    game_ids: list[str] = []

    for entry in raw_entries:
        raw_game_id: object
        if isinstance(entry, str):
            raw_game_id = entry
        elif isinstance(entry, dict):
            raw_game_id = entry.get("game_id")
        else:
            raise ValueError(f"Dataset {path} entries must be game id strings or objects with `game_id`.")

        if not isinstance(raw_game_id, str):
            raise ValueError(f"Dataset {path} has an entry without a string `game_id`.")
        normalized = _normalize_id(raw_game_id)
        if normalized:
            game_ids.append(normalized)

    if not game_ids:
        raise ValueError(f"Dataset {path} does not contain any game ids.")
    return tuple(dict.fromkeys(game_ids))


def _extract_dataset_labels(payload: object, *, path: Path) -> tuple[frozenset[str], frozenset[str]]:
    names = {_normalize_tag(path.stem)}
    tags: set[str] = set()

    if isinstance(payload, dict):
        for key in ("name", "id", "slug"):
            raw_label = payload.get(key)
            if isinstance(raw_label, str):
                normalized = _normalize_tag(raw_label)
                if normalized:
                    names.add(normalized)

        raw_tags = payload.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, str):
                    normalized = _normalize_tag(tag)
                    if normalized:
                        tags.add(normalized)

    return frozenset(names), frozenset(tags)


def _load_dataset_definition(path: Path) -> _DatasetDefinition:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read dataset {path}: {e}") from e

    names, tags = _extract_dataset_labels(payload, path=path)
    game_ids = _extract_dataset_game_ids(payload, path=path)
    return _DatasetDefinition(path=path, names=names, tags=tags, game_ids=game_ids)


def _discover_dataset_definitions(datasets_dir: str | Path | None) -> list[_DatasetDefinition]:
    base = Path(datasets_dir).expanduser() if datasets_dir is not None else PACKAGE_DATASETS_DIR
    if not base.exists() or not base.is_dir():
        return []

    definitions: list[_DatasetDefinition] = []
    for dataset_file in sorted(base.rglob("*.json")):
        try:
            definitions.append(_load_dataset_definition(dataset_file))
        except ValueError:
            continue
    return definitions


def _dataset_file_candidates(spec: str, *, datasets_dir: Path) -> list[Path]:
    text = str(spec).strip()
    if not text:
        return []

    raw_path = Path(text).expanduser()
    candidates: list[Path] = []
    if raw_path.is_file():
        candidates.append(raw_path)

    if raw_path.is_absolute() or len(raw_path.parts) > 1:
        if raw_path.suffix != ".json":
            with_json = raw_path.with_suffix(".json")
            if with_json.is_file():
                candidates.append(with_json)
        return sorted(dict.fromkeys(candidates))

    named_path = datasets_dir / text
    if named_path.suffix != ".json":
        named_path = named_path.with_suffix(".json")
    if named_path.is_file():
        candidates.append(named_path)

    return sorted(dict.fromkeys(candidates))


def _available_dataset_labels(definitions: Sequence[_DatasetDefinition]) -> list[str]:
    labels: set[str] = set()
    for definition in definitions:
        labels.update(definition.names)
        labels.update(definition.tags)
    return sorted(labels)


def _resolve_dataset_game_ids(
    specs: Sequence[str], *, datasets_dir: str | Path | None, field_name: str
) -> tuple[str, ...]:
    if not specs:
        return ()

    resolved_datasets_dir = Path(datasets_dir).expanduser() if datasets_dir is not None else PACKAGE_DATASETS_DIR
    discovered_definitions = _discover_dataset_definitions(resolved_datasets_dir)
    selected_definitions: list[_DatasetDefinition] = []
    unmatched: list[str] = []

    for spec in specs:
        path_matches = _dataset_file_candidates(spec, datasets_dir=resolved_datasets_dir)
        if path_matches:
            selected_definitions.extend(_load_dataset_definition(path) for path in path_matches)
            continue

        normalized = _normalize_tag(spec)
        tag_matches = [
            definition
            for definition in discovered_definitions
            if normalized in definition.names or normalized in definition.tags
        ]
        if tag_matches:
            selected_definitions.extend(tag_matches)
            continue

        unmatched.append(spec)

    if unmatched:
        available = _available_dataset_labels(discovered_definitions)
        suggestions = {
            spec: difflib.get_close_matches(_normalize_tag(spec), available, n=3, cutoff=0.6) for spec in unmatched
        }
        raise ValueError(
            f"Unknown {field_name}: {unmatched}. datasets_dir={resolved_datasets_dir} "
            f"available_datasets={_format_values(available)} dataset_suggestions={suggestions}"
        )

    game_ids: list[str] = []
    for definition in selected_definitions:
        game_ids.extend(definition.game_ids)
    return tuple(sorted(dict.fromkeys(game_ids)))


def _canonicalize_game_ids(game_ids: Iterable[str]) -> list[str]:
    groups: dict[str, set[str]] = {}
    for game_id in game_ids:
        normalized = _normalize_id(game_id)
        if not normalized:
            continue
        base = _base_game_id(normalized)
        groups.setdefault(base, set()).add(normalized)

    selected: list[str] = []
    for base, variants in groups.items():
        if base in variants:
            selected.append(base)
            continue
        selected.append(sorted(variants)[-1])
    selected.sort()
    return selected


def _matches_filter(game_id: str, spec: str) -> bool:
    normalized = _normalize_id(game_id)
    wanted = _normalize_id(spec)
    return wanted == normalized or wanted == _base_game_id(normalized)


def _matches_tag_filter(tags: set[str], spec: str) -> bool:
    return _normalize_tag(spec) in tags


def _format_values(values: Sequence[str], *, max_items: int = 50) -> str:
    items = [str(value) for value in values if str(value).strip()]
    if not items:
        return "[]"

    if len(items) <= max_items:
        return str(items)

    shown = items[:max_items]
    hidden_count = len(items) - len(shown)
    return f"{shown} (+{hidden_count} more)"


def _unmatched_specs(specs: Sequence[str], candidates: Sequence[str]) -> list[str]:
    unmatched: list[str] = []
    for spec in specs:
        if not any(_matches_filter(candidate, spec) for candidate in candidates):
            unmatched.append(spec)
    return unmatched


def _build_filter_targets(canonical_ids: Sequence[str]) -> list[str]:
    values = set()
    for game_id in canonical_ids:
        normalized = _normalize_id(game_id)
        if not normalized:
            continue
        values.add(normalized)
        values.add(_base_game_id(normalized))
    return sorted(values)


def _build_no_match_message(
    *,
    include_specs: Sequence[str],
    exclude_specs: Sequence[str],
    dataset_specs: Sequence[str],
    exclude_dataset_specs: Sequence[str],
    include_tag_specs: Sequence[str],
    exclude_tag_specs: Sequence[str],
    available_ids: Sequence[str],
    available_tags: Sequence[str],
) -> str:
    message_parts = ["EnvSampler has no matching environments after include/exclude filtering."]
    message_parts.append(f"include={list(include_specs) or None}")
    message_parts.append(f"exclude={list(exclude_specs) or None}")
    message_parts.append(f"datasets={list(dataset_specs) or None}")
    message_parts.append(f"exclude_datasets={list(exclude_dataset_specs) or None}")
    message_parts.append(f"include_tags={list(include_tag_specs) or None}")
    message_parts.append(f"exclude_tags={list(exclude_tag_specs) or None}")

    if not available_ids:
        message_parts.append("No environments were discovered from the configured environments_dir.")
        return " ".join(message_parts)

    include_targets = _build_filter_targets(available_ids)

    unmatched_include = _unmatched_specs(include_specs, available_ids)
    if unmatched_include:
        suggestions: dict[str, list[str]] = {}
        for spec in unmatched_include:
            suggestions[spec] = difflib.get_close_matches(spec, include_targets, n=3, cutoff=0.6)
        message_parts.append(f"unmatched_include={unmatched_include}")
        message_parts.append(f"include_suggestions={suggestions}")

    message_parts.append(f"valid_include_names={_format_values(include_targets)}")
    message_parts.append(f"canonical_env_ids={_format_values(available_ids)}")
    if available_tags:
        unmatched_include_tags = [spec for spec in include_tag_specs if _normalize_tag(spec) not in available_tags]
        if unmatched_include_tags:
            suggestions = {
                spec: difflib.get_close_matches(_normalize_tag(spec), available_tags, n=3, cutoff=0.6)
                for spec in unmatched_include_tags
            }
            message_parts.append(f"unmatched_include_tags={unmatched_include_tags}")
            message_parts.append(f"include_tag_suggestions={suggestions}")
        message_parts.append(f"available_tags={_format_values(available_tags)}")
    return " ".join(message_parts)


def _validate_dataset_game_ids(
    *, field_name: str, dataset_game_ids: Sequence[str], available_ids: Sequence[str]
) -> None:
    if not dataset_game_ids:
        return

    unmatched = _unmatched_specs(dataset_game_ids, available_ids)
    if not unmatched:
        return

    include_targets = _build_filter_targets(available_ids)
    suggestions = {
        spec: difflib.get_close_matches(_normalize_id(spec), include_targets, n=3, cutoff=0.6) for spec in unmatched
    }
    raise ValueError(
        f"{field_name} reference unknown game ids: {_format_values(unmatched)} "
        f"suggestions={suggestions} canonical_env_ids={_format_values(available_ids)}"
    )


@dataclass
class _EpisodeTransform:
    color_map: np.ndarray
    rot_k: int
    flip_lr: bool


@dataclass(frozen=True)
class AugmentationConfig:
    """
    Per-episode augmentation switches used by `EnvSampler`.

    All options default to `True` and are only applied when `augment=True`.
    """

    color_permutation: bool = True
    rotation: bool = True
    flip_lr: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "color_permutation", bool(self.color_permutation))
        object.__setattr__(self, "rotation", bool(self.rotation))
        object.__setattr__(self, "flip_lr", bool(self.flip_lr))

    @classmethod
    def coerce(cls, value: AugmentationConfig | dict[str, bool] | None) -> AugmentationConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            valid_keys = {"color_permutation", "rotation", "flip_lr"}
            unknown = sorted(str(key) for key in value if key not in valid_keys)
            if unknown:
                raise ValueError(f"Unknown augmentation_config keys: {unknown}. Valid keys are: {sorted(valid_keys)}.")
            return cls(
                color_permutation=value.get("color_permutation", True),
                rotation=value.get("rotation", True),
                flip_lr=value.get("flip_lr", True),
            )
        raise TypeError("augmentation_config must be an AugmentationConfig, a dict, or None.")

    def any_enabled(self) -> bool:
        return bool(self.color_permutation or self.rotation or self.flip_lr)


class _ObservationAugmenter:
    def __init__(self, rng: random.Random, enabled: bool, config: AugmentationConfig | dict[str, bool] | None = None):
        self._rng = rng
        self._enabled = bool(enabled)
        self._config = AugmentationConfig.coerce(config)
        self._transform = _EpisodeTransform(color_map=np.arange(16, dtype=np.int16), rot_k=0, flip_lr=False)

    @property
    def is_effective(self) -> bool:
        return bool(self._enabled and self._config.any_enabled())

    def new_episode(self) -> None:
        if not self.is_effective:
            self._transform = _EpisodeTransform(color_map=np.arange(16, dtype=np.int16), rot_k=0, flip_lr=False)
            return

        if self._config.color_permutation:
            colors = list(range(16))
            self._rng.shuffle(colors)
        else:
            colors = list(range(16))
        self._transform = _EpisodeTransform(
            color_map=np.array(colors, dtype=np.int16),
            rot_k=int(self._rng.randrange(4)) if self._config.rotation else 0,
            flip_lr=bool(self._rng.randrange(2)) if self._config.flip_lr else False,
        )

    def _transform_layer(self, layer: FrameLayer) -> np.ndarray:
        arr = np.asarray(layer)
        if arr.ndim < 2:
            return arr

        transformed = np.array(arr, copy=True)
        if self._transform.rot_k:
            transformed = np.rot90(transformed, self._transform.rot_k)
        if self._transform.flip_lr:
            transformed = np.fliplr(transformed)

        mask = (transformed >= 0) & (transformed < 16)
        if np.any(mask):
            transformed = np.array(transformed, copy=True)
            transformed[mask] = self._transform.color_map[transformed[mask]]
        return transformed

    def _transform_frame(self, frame: Frame) -> Frame:
        if isinstance(frame, list):
            out: list[np.ndarray] = []
            for layer in frame:
                out.append(self._transform_layer(layer))
            return out
        return self._transform_layer(frame)

    def transform_observation(self, observation: Observation | None) -> Observation | None:
        if observation is None or not self.is_effective:
            return observation
        if not hasattr(observation, "frame"):
            return observation

        transformed = copy.copy(observation)
        frame = observation.frame
        if isinstance(frame, list):
            transformed.frame = cast(list[np.ndarray], self._transform_frame(frame))
        else:
            transformed.frame = cast(np.ndarray, self._transform_frame(frame))
        return transformed

    def inverse_point(self, x: int, y: int, *, height: int, width: int) -> tuple[int, int]:
        """
        Map a point from transformed (augmented) coordinates back to base coordinates.
        """
        x_t = int(x)
        y_t = int(y)
        rot_k = int(self._transform.rot_k) % 4
        flip_lr = bool(self._transform.flip_lr)

        rot_width = int(width) if rot_k % 2 == 0 else int(height)

        # Undo optional left-right flip performed after rotation.
        x_r = (rot_width - 1 - x_t) if flip_lr else x_t
        y_r = y_t

        if rot_k == 0:
            x_base = x_r
            y_base = y_r
        elif rot_k == 1:
            x_base = int(width) - 1 - y_r
            y_base = x_r
        elif rot_k == 2:
            x_base = int(width) - 1 - x_r
            y_base = int(height) - 1 - y_r
        else:
            x_base = y_r
            y_base = int(height) - 1 - x_r

        return int(x_base), int(y_base)

    def inverse_delta(self, dx: int, dy: int) -> tuple[int, int]:
        """
        Map a direction vector from transformed coordinates back to base coordinates.
        """
        dx_t = int(dx)
        dy_t = int(dy)

        # Undo optional left-right flip performed after rotation.
        dx_r = -dx_t if bool(self._transform.flip_lr) else dx_t
        dy_r = dy_t

        rot_k = int(self._transform.rot_k) % 4
        if rot_k == 0:
            dx_base, dy_base = dx_r, dy_r
        elif rot_k == 1:
            dx_base, dy_base = -dy_r, dx_r
        elif rot_k == 2:
            dx_base, dy_base = -dx_r, -dy_r
        else:
            dx_base, dy_base = dy_r, -dx_r

        return int(dx_base), int(dy_base)

    def forward_delta(self, dx: int, dy: int) -> tuple[int, int]:
        """
        Map a direction vector from base coordinates into transformed coordinates.
        """
        dx_base = int(dx)
        dy_base = int(dy)
        for dx_t, dy_t in _DELTA_TO_MOVE_ACTION:
            if self.inverse_delta(dx_t, dy_t) == (dx_base, dy_base):
                return int(dx_t), int(dy_t)
        return dx_base, dy_base


def _final_layer(frame: Frame | None) -> FrameLayer | None:
    if frame is None:
        return None
    if isinstance(frame, list):
        for layer in reversed(frame):
            return layer
        return None
    return frame


def _observation_size(observation: Observation | None) -> tuple[int, int] | None:
    if observation is None:
        return None
    layer = _final_layer(observation.frame)
    if layer is None:
        return None
    arr = np.asarray(layer)
    if arr.ndim < 2:
        return None
    return int(arr.shape[0]), int(arr.shape[1])


class _AugmentedEnv:
    def __init__(
        self,
        env: Environment,
        augmenter: _ObservationAugmenter,
        initial_transform_ready: bool = False,
        game_id: str | None = None,
    ) -> None:
        self._env = env
        self._augmenter = augmenter
        self._last_base_size: tuple[int, int] | None = None
        self._initial_transform_ready = bool(initial_transform_ready)
        self._game_id = _normalize_id(game_id) if game_id is not None else ""

    @property
    def action_space(self) -> Sequence[Action]:
        if not self._augmenter.is_effective:
            return self._env.action_space

        remapped: list[Action] = []
        for action in self._env.action_space:
            remapped.append(self._augment_action(action))
        return tuple(remapped)

    @property
    def game_id(self) -> str:
        if self._game_id:
            return self._game_id
        return str(getattr(self._env, "game_id", None) or "").strip().lower()

    @property
    def name(self) -> str:
        env_name = getattr(self._env, "name", None)
        if env_name:
            return str(env_name)
        return self.game_id

    def _sync_game_id(self, observation: Observation | None) -> None:
        observed = str(getattr(observation, "game_id", None) or "").strip().lower()
        if observed:
            self._game_id = observed

    def _augment_action(self, action: Action) -> Action:
        action_id = int(getattr(action, "value", -1))
        move_delta = _MOVE_ACTION_TO_DELTA.get(action_id)
        if move_delta is None:
            return action
        dx_t, dy_t = self._augmenter.forward_delta(move_delta[0], move_delta[1])
        transformed_action_id = _DELTA_TO_MOVE_ACTION.get((dx_t, dy_t), action_id)
        if transformed_action_id == action_id:
            return action
        return GameAction.from_id(transformed_action_id)

    def reset(self, *, preserve_transform: bool = False) -> Observation:
        if self._initial_transform_ready:
            self._initial_transform_ready = False
        elif not preserve_transform:
            self._augmenter.new_episode()
        observation = self._env.reset()
        self._sync_game_id(observation)
        self._last_base_size = _observation_size(observation)
        transformed = self._augmenter.transform_observation(observation)
        return cast(Observation, transformed)

    def step(self, action: Action, data: ActionData | None = None) -> Observation:
        payload: dict[str, ActionValue] = dict(data or {})
        mapped_action = action

        action_id = int(getattr(action, "value", -1))
        move_delta = _MOVE_ACTION_TO_DELTA.get(action_id)
        if move_delta is not None:
            base_delta = self._augmenter.inverse_delta(move_delta[0], move_delta[1])
            mapped_action_id = _DELTA_TO_MOVE_ACTION.get(base_delta)
            if mapped_action_id is not None and mapped_action_id != action_id:
                for candidate in self._env.action_space:
                    if int(getattr(candidate, "value", -1)) == mapped_action_id:
                        mapped_action = candidate
                        break

        if (
            getattr(action, "is_complex", lambda: False)()
            and "x" in payload
            and "y" in payload
            and self._last_base_size is not None
        ):
            try:
                x_t = int(cast(IntCoercible, payload["x"]))
                y_t = int(cast(IntCoercible, payload["y"]))
                height, width = self._last_base_size
                x_base, y_base = self._augmenter.inverse_point(x_t, y_t, height=height, width=width)
                payload["x"] = int(x_base)
                payload["y"] = int(y_base)
            except (TypeError, ValueError):
                pass

        observation = self._env.step(mapped_action, data=payload)
        self._sync_game_id(observation)
        self._last_base_size = _observation_size(observation)
        transformed = self._augmenter.transform_observation(observation)
        return cast(Observation, transformed)

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)


def _build_arcade_and_game_ids(
    *,
    include: Sequence[str] | str | None = None,
    exclude: Sequence[str] | str | None = None,
    environments_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
    log_level: str | int = logging.WARNING,
    logger_name: str = "re_arc.arcade.sampler",
    include_tags: Sequence[str] | str | None = None,
    exclude_tags: Sequence[str] | str | None = None,
    datasets: Sequence[str | Path] | str | Path | None = None,
    exclude_datasets: Sequence[str | Path] | str | Path | None = None,
    datasets_dir: str | Path | None = None,
) -> tuple[Arcade, tuple[str, ...]]:
    include_specs = _normalize_specs(include)
    exclude_specs = _normalize_specs(exclude)
    include_tag_specs = _normalize_specs(include_tags)
    exclude_tag_specs = _normalize_specs(exclude_tags)
    dataset_specs = _normalize_dataset_specs(datasets)
    exclude_dataset_specs = _normalize_dataset_specs(exclude_datasets)
    resolved_environments_dir = (
        Path(environments_dir).expanduser() if environments_dir is not None else PACKAGE_ENVIRONMENTS_DIR
    )

    operation_mode = OperationMode.ONLINE if os.environ.get("ARC_BASE_URL") else OperationMode.OFFLINE
    arcade_kwargs = {"operation_mode": operation_mode, "environments_dir": str(resolved_environments_dir)}
    arcade_kwargs["logger"] = logger or _build_arcade_logger(log_level, name=logger_name)
    arcade = Arcade(**arcade_kwargs)

    discovered = _discover_game_metadata_from_dir(resolved_environments_dir)

    canonical_ids = _canonicalize_game_ids(discovered.keys())
    available_ids = list(canonical_ids)
    available_tags = sorted({tag for game_id in canonical_ids for tag in discovered.get(game_id, set())})

    dataset_game_ids = _resolve_dataset_game_ids(dataset_specs, datasets_dir=datasets_dir, field_name="datasets")
    exclude_dataset_game_ids = _resolve_dataset_game_ids(
        exclude_dataset_specs, datasets_dir=datasets_dir, field_name="exclude_datasets"
    )
    _validate_dataset_game_ids(field_name="datasets", dataset_game_ids=dataset_game_ids, available_ids=available_ids)
    _validate_dataset_game_ids(
        field_name="exclude_datasets", dataset_game_ids=exclude_dataset_game_ids, available_ids=available_ids
    )

    if dataset_game_ids:
        canonical_ids = [
            game_id for game_id in canonical_ids if any(_matches_filter(game_id, spec) for spec in dataset_game_ids)
        ]
    if exclude_dataset_game_ids:
        canonical_ids = [
            game_id
            for game_id in canonical_ids
            if not any(_matches_filter(game_id, spec) for spec in exclude_dataset_game_ids)
        ]
    if include_specs:
        canonical_ids = [
            game_id for game_id in canonical_ids if any(_matches_filter(game_id, spec) for spec in include_specs)
        ]
    if exclude_specs:
        canonical_ids = [
            game_id for game_id in canonical_ids if not any(_matches_filter(game_id, spec) for spec in exclude_specs)
        ]
    if include_tag_specs:
        canonical_ids = [
            game_id
            for game_id in canonical_ids
            if any(_matches_tag_filter(discovered.get(game_id, set()), spec) for spec in include_tag_specs)
        ]
    if exclude_tag_specs:
        canonical_ids = [
            game_id
            for game_id in canonical_ids
            if not any(_matches_tag_filter(discovered.get(game_id, set()), spec) for spec in exclude_tag_specs)
        ]

    if not canonical_ids:
        raise ValueError(
            _build_no_match_message(
                include_specs=include_specs,
                exclude_specs=exclude_specs,
                dataset_specs=dataset_specs,
                exclude_dataset_specs=exclude_dataset_specs,
                include_tag_specs=include_tag_specs,
                exclude_tag_specs=exclude_tag_specs,
                available_ids=available_ids,
                available_tags=available_tags,
            )
        )

    return arcade, tuple(sorted(canonical_ids))


class EnvSampler:
    """
    Random environment sampler with optional per-episode augmentation.

    EnvSampler is offline-only and loads games from local `environments_dir`.

    Augmentation is episode-consistent:
    - Optional random color permutation for colors 0..15
    - Optional random geometric transform (rotation + optional flip)
    The same transform is applied to every frame in the episode.

    Use `augmentation_config` to choose which augmentation components are active.

    Transition reward wrapping is always enabled:
    - each positive `levels_completed` transition returns `delta_levels / win_levels`
    - a terminal `WIN` step is reconciled so total episode reward is exactly `1.0`
    When a base `seed` is provided and `make(..., seed=None)` is used, each call
    receives a deterministic rolling seed derived from that base seed.
    """

    def __init__(
        self,
        include: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str | None = None,
        augment: bool = False,
        augmentation_config: AugmentationConfig | dict[str, bool] | None = None,
        environments_dir: str | Path | None = None,
        seed: int | None = None,
        logger: logging.Logger | None = None,
        log_level: str | int = logging.WARNING,
        include_tags: Sequence[str] | str | None = None,
        exclude_tags: Sequence[str] | str | None = None,
        datasets: Sequence[str | Path] | str | Path | None = None,
        exclude_datasets: Sequence[str | Path] | str | Path | None = None,
        datasets_dir: str | Path | None = None,
    ):
        self._rng = random.Random(seed)
        self._augment = bool(augment)
        self._augmentation_config = AugmentationConfig.coerce(augmentation_config)
        self._rolling_seed_rng = random.Random(seed) if seed is not None else None
        self._arcade, self._game_ids = _build_arcade_and_game_ids(
            include=include,
            exclude=exclude,
            environments_dir=environments_dir,
            logger=logger,
            log_level=log_level,
            logger_name="re_arc.arcade.sampler",
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            datasets=datasets,
            exclude_datasets=exclude_datasets,
            datasets_dir=datasets_dir,
        )

    @property
    def game_ids(self) -> tuple[str, ...]:
        return self._game_ids

    @classmethod
    def list_game_ids(
        cls,
        include: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str | None = None,
        environments_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
        log_level: str | int = logging.WARNING,
        include_tags: Sequence[str] | str | None = None,
        exclude_tags: Sequence[str] | str | None = None,
        datasets: Sequence[str | Path] | str | Path | None = None,
        exclude_datasets: Sequence[str | Path] | str | Path | None = None,
        datasets_dir: str | Path | None = None,
    ) -> tuple[str, ...]:
        _, game_ids = _build_arcade_and_game_ids(
            include=include,
            exclude=exclude,
            environments_dir=environments_dir,
            logger=logger,
            log_level=log_level,
            logger_name="re_arc.arcade.discovery",
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            datasets=datasets,
            exclude_datasets=exclude_datasets,
            datasets_dir=datasets_dir,
        )
        return game_ids

    def __len__(self) -> int:
        return len(self._game_ids)

    def __iter__(self) -> Iterator[TransitionRewardEnv]:
        return self

    def __next__(self) -> TransitionRewardEnv:
        return self.sample()

    def sample_game_id(self) -> str:
        return self._rng.choice(self._game_ids)

    def _resolve_seed(self, seed: int | None) -> int | None:
        if seed is not None:
            return int(seed)
        if self._rolling_seed_rng is None:
            return None
        return int(self._rolling_seed_rng.randrange(2**31))

    def _resolve_game_id(self, game_id: str | None) -> str:
        if game_id is None:
            return self.sample_game_id()

        wanted = _normalize_id(game_id)
        for candidate in self._game_ids:
            if _matches_filter(candidate, wanted):
                return candidate
        raise ValueError(f"Requested game `{game_id}` is not in this sampler. Available: {list(self._game_ids)}")

    def make(
        self, game_id: str | None = None, seed: int | None = None, renderer: Renderer | None = None
    ) -> TransitionRewardEnv:
        selected = self._resolve_game_id(game_id)
        resolved_seed = self._resolve_seed(seed)
        # Use a per-call RNG so concurrent make() calls don't interleave draws on a shared
        # state. Seeding with resolved_seed also makes augmentation a deterministic function
        # of the env seed.
        augmenter = _ObservationAugmenter(
            random.Random(resolved_seed), enabled=self._augment, config=self._augmentation_config
        )
        augmentation_active = augmenter.is_effective
        initial_transform_ready = False
        if augmentation_active and renderer is not None:
            # Some environments emit renderer frames before the first reset.
            # Pre-sample the first episode transform so those frames are augmented too.
            augmenter.new_episode()
            initial_transform_ready = True

        wrapped_renderer: Renderer | None = renderer
        if renderer is not None and augmentation_active:

            def _renderer(steps: int, frame_data: Observation) -> None:
                transformed = augmenter.transform_observation(frame_data)
                renderer(steps, cast(Observation, transformed))

            wrapped_renderer = _renderer

        env = cast(Environment | None, self._arcade.make(selected, seed=resolved_seed, renderer=wrapped_renderer))
        if env is None:
            raise RuntimeError(f"Failed to create environment: {selected}")

        wrapped_env: Environment | _AugmentedEnv | TransitionRewardEnv = env
        if augmentation_active:
            wrapped_env = _AugmentedEnv(
                env, augmenter, initial_transform_ready=initial_transform_ready, game_id=selected
            )

        from .reward_env import TransitionRewardEnv

        if not isinstance(wrapped_env, TransitionRewardEnv):
            wrapped_env = TransitionRewardEnv(wrapped_env, game_id=selected, seed=resolved_seed)

        return wrapped_env

    def sample(self, seed: int | None = None, renderer: Renderer | None = None) -> TransitionRewardEnv:
        return self.make(game_id=None, seed=seed, renderer=renderer)


def list_game_ids(
    include: Sequence[str] | str | None = None,
    exclude: Sequence[str] | str | None = None,
    environments_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
    log_level: str | int = logging.WARNING,
    include_tags: Sequence[str] | str | None = None,
    exclude_tags: Sequence[str] | str | None = None,
    datasets: Sequence[str | Path] | str | Path | None = None,
    exclude_datasets: Sequence[str | Path] | str | Path | None = None,
    datasets_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """
    Discover available game ids without creating an EnvSampler instance.
    """
    return EnvSampler.list_game_ids(
        include=include,
        exclude=exclude,
        environments_dir=environments_dir,
        logger=logger,
        log_level=log_level,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        datasets=datasets,
        exclude_datasets=exclude_datasets,
        datasets_dir=datasets_dir,
    )
