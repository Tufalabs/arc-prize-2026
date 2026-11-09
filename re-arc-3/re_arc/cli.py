import argparse
import contextlib
import errno
import http.server
import importlib
import json
import logging
import os
import random
import re
import sys
import threading
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, quote, urlparse

from arc_agi import Arcade, OperationMode  # type: ignore[import-untyped]
from arcengine import GameAction, GameState

from .dsl import create_dsl_agent, run_dsl_episode
from .dsl import is_terminal as _dsl_is_terminal
from .dsl import unpack_step_result as _dsl_unpack_step_result
from .env_sampler import (
    AugmentationConfig,
    EnvSampler,
    _build_arcade_logger,
    _coerce_log_level,
    default_environments_dir,
)
from .env_sampler import list_game_ids as _sampler_list_game_ids
from .reward_env import TransitionRewardEnv

COLOR_MAP = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}

_CLI_OVERRIDE_KEYS = "__CLI_OVERRIDE_KEYS__"


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Any) -> Any:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    config: dict[str, str] = {}
    config["__CONFIG_PATH__"] = str(config_path.resolve())
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip().strip('"').strip("'")
        if value:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key == "GAME_ID" and value:
            match = re.match(r"^(?P<gid>[^@:\s]+)[@:](?P<port>\d+)$", value)
            if match:
                value = match.group("gid")
                if not config.get("WEB_PORT"):
                    config["WEB_PORT"] = match.group("port")
        config[key] = value
    return config


def _split_list_value(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "null"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _cfg_list(config: Any, *keys: str) -> list[str] | None:
    for key in keys:
        raw = config.get(key)
        values = _split_list_value(raw)
        if values:
            return values
    return None


def _mark_cli_override(config: Any, key: str) -> None:
    overrides = set(config.get(_CLI_OVERRIDE_KEYS, set()))
    overrides.add(key)
    config[_CLI_OVERRIDE_KEYS] = overrides


def _has_cli_override(config: Any, *keys: str) -> bool:
    overrides = set(config.get(_CLI_OVERRIDE_KEYS, set()))
    return any(key in overrides for key in keys)


def _env_or_cfg_list(config: Any, *keys: str) -> list[str] | None:
    if _has_cli_override(config, *keys):
        return _cfg_list(config, *keys)
    for key in keys:
        values = _split_list_value(os.environ.get(key))
        if values:
            return values
    return _cfg_list(config, *keys)


def _env_or_cfg_bool(config: Any, *keys: str, default: bool = False) -> bool:
    if _has_cli_override(config, *keys):
        for key in keys:
            if key in config:
                return _cfg_bool(config, key, default)
        return bool(default)
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None:
            return _cfg_bool({key: raw}, key, default)
    for key in keys:
        if key in config:
            return _cfg_bool(config, key, default)
    return bool(default)


def _augmentation_config_from_config(config: Any) -> AugmentationConfig:
    return AugmentationConfig(
        color_permutation=_env_or_cfg_bool(config, "AUGMENT_COLOR_PERMUTATION", default=True),
        rotation=_env_or_cfg_bool(config, "AUGMENT_ROTATION", default=True),
        flip_lr=_env_or_cfg_bool(config, "AUGMENT_FLIP_LR", default=True),
    )


def _augment_enabled(config: Any) -> bool:
    return _env_or_cfg_bool(config, "AUGMENT", default=False)


def _sampler_augmentation_active(config: Any) -> bool:
    return bool(_augment_enabled(config) and _augmentation_config_from_config(config).any_enabled())


def _env_or_cfg_value(config: Any, key: str) -> Any:
    if _has_cli_override(config, key):
        return config.get(key)
    return os.environ.get(key) or config.get(key)


def _load_runtime_config(args: Any) -> Any:
    config = _load_config(args.config)

    datasets = getattr(args, "dataset", None)
    if datasets:
        config["DATASETS"] = ",".join(str(value) for value in datasets if str(value).strip())
        _mark_cli_override(config, "DATASETS")

    exclude_datasets = getattr(args, "exclude_dataset", None)
    if exclude_datasets:
        config["EXCLUDE_DATASETS"] = ",".join(str(value) for value in exclude_datasets if str(value).strip())
        _mark_cli_override(config, "EXCLUDE_DATASETS")

    datasets_dir = getattr(args, "datasets_dir", None)
    if datasets_dir:
        config["DATASETS_DIR"] = str(datasets_dir)
        _mark_cli_override(config, "DATASETS_DIR")

    return config


def _cfg_int(config: Any, key: Any, default: Any) -> Any:
    raw = config.get(key)
    if raw is None:
        return default
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return default
    return int(text)


def _cfg_bool(config: Any, key: Any, default: Any = False) -> Any:
    raw = config.get(key)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    return bool(default)


def _resolve_game_id(args: Any, config: Any) -> Any:
    game_id = args.game or config.get("GAME_ID") or os.environ.get("ARC_GAME_ID")
    if not game_id:
        raise ValueError(
            "GAME_ID is required (set in config or --game). "
            "Tip: run `make games` or use --list-games to see available options."
        )
    return game_id


def _arcade_kwargs_from_config(config: Any) -> Any:
    mode = (os.environ.get("OPERATION_MODE") or config.get("OPERATION_MODE") or "").strip().upper()
    arc_kwargs = {}
    if mode:
        try:
            arc_kwargs["operation_mode"] = OperationMode[mode]
        except KeyError as e:
            valid = ", ".join(m.name for m in OperationMode)
            raise ValueError(f"Invalid OPERATION_MODE={mode!r}. Valid values: {valid}.") from e
    arc_api_key = os.environ.get("ARC_API_KEY") or config.get("ARC_API_KEY")
    if arc_api_key:
        arc_kwargs["arc_api_key"] = arc_api_key
    environments_dir = os.environ.get("ENVIRONMENTS_DIR") or config.get("ENVIRONMENTS_DIR")
    if environments_dir:
        arc_kwargs["environments_dir"] = environments_dir
    recordings_dir = os.environ.get("RECORDINGS_DIR") or config.get("RECORDINGS_DIR")
    if recordings_dir:
        arc_kwargs["recordings_dir"] = recordings_dir
    arc_base_url = os.environ.get("ARC_BASE_URL") or config.get("ARC_BASE_URL")
    if arc_base_url:
        arc_kwargs["arc_base_url"] = arc_base_url
    arc_log_level = os.environ.get("ARC_LOG_LEVEL") or config.get("ARC_LOG_LEVEL") or logging.WARNING
    arc_kwargs["logger"] = _build_arcade_logger(
        _coerce_log_level(arc_log_level, default=logging.WARNING), name="re_arc.arcade.cli"
    )
    return arc_kwargs


def _build_arcade(config: Any) -> Any:
    return Arcade(**_arcade_kwargs_from_config(config))


def _build_env_sampler(
    config: Any,
    seed: Any = None,
    augment: Any = False,
    include: list[str] | tuple[str, ...] | None = None,
    exclude: list[str] | tuple[str, ...] | None = None,
    apply_dataset_filters: bool = True,
) -> Any:
    kwargs = _arcade_kwargs_from_config(config)
    datasets = _env_or_cfg_list(config, "DATASETS", "DATASET") if apply_dataset_filters else None
    exclude_datasets = (
        _env_or_cfg_list(config, "EXCLUDE_DATASETS", "EXCLUDE_DATASET") if apply_dataset_filters else None
    )
    include_tags = _env_or_cfg_list(config, "INCLUDE_TAGS", "INCLUDE_TAG")
    exclude_tags = _env_or_cfg_list(config, "EXCLUDE_TAGS", "EXCLUDE_TAG")
    return EnvSampler(
        include=include,
        exclude=exclude,
        augment=bool(augment),
        augmentation_config=_augmentation_config_from_config(config),
        seed=seed,
        environments_dir=kwargs.get("environments_dir", default_environments_dir()),
        logger=kwargs.get("logger"),
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        datasets=datasets,
        exclude_datasets=exclude_datasets,
        datasets_dir=_env_or_cfg_value(config, "DATASETS_DIR"),
    )


def _is_random_game_id(game_id: Any) -> Any:
    return str(game_id).strip().lower() == "random"


def _selected_game_id_from_observation(observation: Any, fallback: Any) -> Any:
    game_id = getattr(observation, "game_id", None) if observation is not None else None
    if game_id:
        return str(game_id)
    return str(fallback)


def _wrap_env_with_transition_rewards(env: Any, game_id: Any, seed: Any) -> Any:
    if isinstance(env, TransitionRewardEnv):
        return env
    return TransitionRewardEnv(env, game_id=game_id, seed=seed)


def _create_env_and_initial_observation(config: Any, game_id: Any, seed: Any = None, renderer: Any = None) -> Any:
    augment = _augment_enabled(config)
    if _is_random_game_id(game_id):
        sampler = _build_env_sampler(config, seed=seed, augment=augment)
        base_env = sampler.sample(seed=seed, renderer=renderer)
        if base_env is None:
            raise RuntimeError("Failed to create random environment.")
        env = _wrap_env_with_transition_rewards(env=base_env, game_id=game_id, seed=seed)
        observation = env.reset()
        selected_game_id = _selected_game_id_from_observation(observation, game_id)
        return env, observation, selected_game_id

    if _sampler_augmentation_active(config):
        sampler = _build_env_sampler(
            config, seed=seed, augment=augment, include=[str(game_id)], apply_dataset_filters=False
        )
        base_env = sampler.make(game_id=str(game_id), seed=seed, renderer=renderer)
        env = _wrap_env_with_transition_rewards(env=base_env, game_id=game_id, seed=seed)
        observation = env.reset()
        selected_game_id = _selected_game_id_from_observation(observation, game_id)
        return env, observation, selected_game_id

    arc = _build_arcade(config)
    base_env = arc.make(game_id, seed=seed, renderer=renderer)
    if base_env is None:
        raise RuntimeError(f"Failed to create environment: {game_id}")
    env = _wrap_env_with_transition_rewards(env=base_env, game_id=game_id, seed=seed)
    observation = env.reset()
    selected_game_id = _selected_game_id_from_observation(observation, game_id)
    return env, observation, selected_game_id


def _list_game_ids(config: Any, apply_dataset_filters: bool = True) -> Any:
    environments_dir = config.get("ENVIRONMENTS_DIR") or default_environments_dir()
    return list(
        _sampler_list_game_ids(
            environments_dir=environments_dir,
            log_level=logging.WARNING,
            include_tags=_env_or_cfg_list(config, "INCLUDE_TAGS", "INCLUDE_TAG"),
            exclude_tags=_env_or_cfg_list(config, "EXCLUDE_TAGS", "EXCLUDE_TAG"),
            datasets=_env_or_cfg_list(config, "DATASETS", "DATASET") if apply_dataset_filters else None,
            exclude_datasets=_env_or_cfg_list(config, "EXCLUDE_DATASETS", "EXCLUDE_DATASET")
            if apply_dataset_filters
            else None,
            datasets_dir=_env_or_cfg_value(config, "DATASETS_DIR"),
        )
    )


def _ensure_package_path(module_name: str, package_path: Path) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return
    package_path_str = str(package_path)
    if package_path_str not in module_path:
        module_path.append(package_path_str)


def _register_worker_repo_packages(worker_repo_path: Path) -> None:
    worker_re_arc = worker_repo_path / "re_arc"
    if not worker_re_arc.exists():
        return
    _ensure_package_path("re_arc", worker_re_arc)
    _ensure_package_path("re_arc.dsl", worker_re_arc / "dsl")
    _ensure_package_path("re_arc.dsl.agents", worker_re_arc / "dsl" / "agents")


def _frame_to_image(frame_grid: Any, scale: Any, separator: Any) -> Any:
    from PIL import Image

    grid = frame_grid
    if grid is None:
        return None
    if isinstance(grid, list):
        if len(grid) == 0:
            return None
        first_layer = grid[0]
    else:
        grid = [grid]
        first_layer = grid[0]
    if hasattr(first_layer, "size"):
        if first_layer.size == 0:
            return None
    elif len(first_layer) == 0:
        return None

    first_layer = grid[0]
    if hasattr(first_layer, "shape") and len(first_layer.shape) >= 2:
        height = int(first_layer.shape[0])
        width = int(first_layer.shape[1])
    else:
        height = len(first_layer)
        width = len(first_layer[0]) if height else 0
    layers = len(grid)

    sep = separator if layers > 1 else 0
    total_width = (width * layers) + (sep * (layers - 1))

    image = Image.new("RGB", (total_width, height), "white")
    pixels = image.load()
    if pixels is None:
        return None

    for i, grid_layer in enumerate(grid):
        if len(grid_layer) != height or len(grid_layer[0]) != width:
            continue
        offset_x = i * (width + sep)
        for y in range(height):
            row = grid_layer[y]
            for x in range(width):
                color_index = int(row[x])
                pixels[x + offset_x, y] = COLOR_MAP.get(color_index, (0, 0, 0))

    if scale != 1:
        resample = getattr(Image, "Resampling", Image).NEAREST
        image = image.resize((total_width * scale, height * scale), resample=resample)
    return image


def _write_gif(frames: Any, gif_path: Any, fps: Any, scale: Any, separator: Any) -> Any:
    from PIL import Image

    images = []
    for captured in frames:
        raw = getattr(captured, "frame", None)
        if raw is None:
            continue
        if isinstance(raw, list):
            # Raw wrapper responses store a list of temporal frames for each action.
            for frame_grid in raw:
                image = _frame_to_image(frame_grid, scale, separator)
                if image is not None:
                    images.append(image)
        else:
            image = _frame_to_image(raw, scale, separator)
            if image is not None:
                images.append(image)

    if not images:
        raise ValueError("No frames captured for GIF.")
    base_size = images[0].size
    resample = getattr(Image, "Resampling", Image).NEAREST
    for idx, image in enumerate(images):
        if image.size != base_size:
            images[idx] = image.resize(base_size, resample=resample)

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = max(10, round(1000 / max(1, int(fps))))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved GIF to {gif_path}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _replay_path_for_gif(gif_path: Any) -> Path:
    return Path(gif_path).with_suffix(".replay.json")


def _action_name_from_id(action_id: Any) -> str:
    human_names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "CLICK"}
    with contextlib.suppress(TypeError, ValueError):
        action_value = int(action_id)
        if action_value in human_names:
            return human_names[action_value]
    try:
        return str(GameAction.from_id(int(action_id)).name)
    except Exception:
        return f"ACTION_{action_id}"


def _write_replay_trace(trace: Any, replay_path: Any) -> Any:
    replay_path = Path(replay_path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Saved replay trace to {replay_path}")


_WEB_INDEX_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ARC-AGI Web Player</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #0b0f17;
        --panel: #0f172a;
        --muted: #94a3b8;
        --text: #e2e8f0;
        --border: #1e293b;
        --accent: #60a5fa;
        --danger: #f87171;
      }

      html,
      body {
        height: 100%;
        margin: 0;
        background: radial-gradient(900px circle at 20% 10%, #111827, var(--bg));
        color: var(--text);
        font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto,
          Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji",
          "Segoe UI Emoji";
      }

      .wrap {
        max-width: 1120px;
        margin: 0 auto;
        padding: 16px 14px 24px;
      }

      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
      }

      h1 {
        margin: 0;
        font-size: 15px;
        letter-spacing: 0.2px;
        font-weight: 650;
      }

      .meta {
        color: var(--muted);
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 60vw;
      }

      .grid {
        display: grid;
        grid-template-columns: 1fr 300px;
        gap: 12px;
      }

      .panel {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      }

      .viewer {
        padding: 12px;
        overflow: auto;
      }

      .controls {
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 9px;
      }

      .row {
        display: flex;
        gap: 8px;
        align-items: center;
      }

      .kbd-grid {
        display: grid;
        grid-template-columns: repeat(3, 40px);
        grid-template-rows: repeat(2, 40px);
        gap: 7px;
      }

      .kbd-btn {
        padding: 0;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        user-select: none;
      }

      .kbd-wide {
        width: 100%;
      }

      .kbd-stack {
        flex: 1;
        display: grid;
        gap: 8px;
      }

      .level-buttons {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(44px, 1fr));
        gap: 6px;
      }

      .level-buttons button {
        min-height: 34px;
        padding: 6px 8px;
      }

      .level-buttons button.active {
        border-color: rgba(96, 165, 250, 0.95);
        background: rgba(96, 165, 250, 0.2);
      }

      label {
        color: var(--muted);
        font-size: 11px;
      }

      input,
      select,
      button,
      textarea {
        background: rgba(2, 6, 23, 0.75);
        border: 1px solid rgba(30, 41, 59, 0.8);
        color: var(--text);
        border-radius: 10px;
        padding: 8px 10px;
        font: inherit;
      }

      input[type="number"] {
        width: 92px;
      }

      input[type="range"] {
        width: 100%;
      }

      button {
        cursor: pointer;
        font-weight: 600;
      }

      button.primary {
        border-color: rgba(96, 165, 250, 0.9);
        background: rgba(96, 165, 250, 0.16);
      }

      button.danger {
        border-color: rgba(248, 113, 113, 0.9);
        background: rgba(248, 113, 113, 0.12);
      }

      button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }

      .kbd-btn:disabled {
        opacity: 0.22;
        border-style: dashed;
        border-color: rgba(148, 163, 184, 0.35);
        background: rgba(2, 6, 23, 0.35);
        color: var(--muted);
      }

      .hint {
        color: var(--muted);
        font-size: 11px;
      }

      .error {
        color: var(--danger);
        font-size: 11px;
        white-space: pre-wrap;
      }

      .layers {
        display: grid;
        gap: 12px;
      }

      .viewer-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(220px, 320px);
        gap: 12px;
        align-items: start;
      }

      .layer {
        display: grid;
        gap: 6px;
      }

      .layer-title {
        color: var(--muted);
        font-size: 12px;
      }

      canvas {
        display: block;
        border-radius: 10px;
        border: 1px solid rgba(30, 41, 59, 0.8);
        background: #000;
        image-rendering: pixelated;
        cursor: crosshair;
      }

      @media (max-width: 960px) {
        .grid {
          grid-template-columns: 1fr;
        }
        .viewer-grid {
          grid-template-columns: 1fr;
        }
        .meta {
          max-width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <h1>ARC-AGI Web Player</h1>
        <div class="meta" id="meta">Connecting…</div>
      </header>

      <div class="grid">
        <div class="panel viewer">
          <div class="viewer-grid">
            <div class="layers" id="layers"></div>
          </div>
        </div>

        <div class="panel controls">
          <div class="row">
            <button id="undoBtn">Undo (Z)</button>
            <button class="danger" id="resetLevelBtn">Reset level (R)</button>
            <button id="restartBtn">Restart</button>
          </div>

          <div class="row" style="justify-content: space-between">
            <label>Quick play</label>
            <span class="hint" id="levelLabel">LEVEL: -</span>
          </div>
          <div class="level-buttons" id="levelButtons"></div>

          <div class="row" style="align-items: flex-start">
            <div class="kbd-grid" aria-label="WASD">
              <div></div>
              <button class="kbd-btn" id="wBtn" type="button">W</button>
              <div></div>
              <button class="kbd-btn" id="aBtn" type="button">A</button>
              <button class="kbd-btn" id="sBtn" type="button">S</button>
              <button class="kbd-btn" id="dBtn" type="button">D</button>
            </div>
            <div class="kbd-stack">
              <button class="kbd-btn kbd-wide" id="spaceBtn" type="button">
                Space
              </button>
            </div>
          </div>

          <div class="row">
            <span class="hint" id="mouseLabel">Mouse: unavailable</span>
          </div>

          <div class="row" style="justify-content: space-between">
            <label for="seed">Seed (optional)</label>
            <span class="hint">Reset to apply</span>
          </div>
          <input id="seed" type="number" placeholder="(keep)" />

          <div class="row" style="justify-content: space-between">
            <label for="gameSelect">Environment</label>
            <span class="hint">Switch and restart</span>
          </div>
          <select id="gameSelect"></select>

          <div class="row">
            <button id="recordBtn" type="button">Record</button>
            <button id="useReplayBtn" type="button" style="display:none">Use recording for DSL agent</button>
            <button id="copyAsciiBtn" type="button">Copy ASCII</button>
            <button id="refreshBtn" type="button">Refresh</button>
            <span class="hint" id="reloadHint"></span>
          </div>

          <div class="error" id="error"></div>

          <div class="hint">
            Keys: WASD, Space, Z undo, R reset level.
            Click the game to send a click action.
          </div>

          <div class="hint">
            Tip: use <code>make games</code> to list game ids, set <code>GAME_ID</code>
            in <code>config.env</code>, then start the UI with <code>make play</code>.
          </div>
        </div>
      </div>
    </div>

    <script>
      const COLOR_MAP = __COLOR_MAP_JSON__;
      const API_BASE = "__API_BASE_JSON__";

      function clamp(n, lo, hi) {
        return Math.max(lo, Math.min(hi, n));
      }

      function apiUrl(path, extraParams) {
        const url = new URL(`${API_BASE}${path}`, window.location.origin);
        const current = new URLSearchParams(window.location.search || "");
        current.forEach((value, key) => url.searchParams.append(key, value));
        if (extraParams) {
          Object.entries(extraParams).forEach(([key, value]) => {
            if (value !== null && value !== undefined && value !== "") {
              url.searchParams.set(key, String(value));
            }
          });
        }
        return url.toString();
      }

      async function fetchJson(url, options) {
        const resp = await fetch(url, options);
        const text = await resp.text();
        let data = null;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          data = null;
        }
        if (!resp.ok) {
          const msg = (data && data.error) ? data.error : text || resp.statusText;
          throw new Error(msg);
        }
        return data;
      }

      function cellRgb(v) {
        const key = String(v);
        return COLOR_MAP[key] || [0, 0, 0];
      }

      function drawLayer(canvas, layer, scale) {
        const w = layer.width;
        const h = layer.height;
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = `${w * scale}px`;
        canvas.style.height = `${h * scale}px`;

        const ctx = canvas.getContext("2d", { alpha: false });
        const img = ctx.createImageData(w, h);
        const data = img.data;
        let i = 0;
        for (let y = 0; y < h; y++) {
          const row = layer.cells[y];
          for (let x = 0; x < w; x++) {
            const [r, g, b] = cellRgb(row[x]);
            data[i++] = r;
            data[i++] = g;
            data[i++] = b;
            data[i++] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
      }

      function colorToAscii(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return "0";
        return "0123456789ABCDEF"[clamp(Math.trunc(n), 0, 15)] || "0";
      }

      function layerToAscii(layer) {
        if (!layer || !Array.isArray(layer.cells)) return "";
        return layer.cells
          .map((row) => (Array.isArray(row) ? row.map(colorToAscii).join("") : ""))
          .join("\n");
      }

      function setError(msg) {
        document.getElementById("error").textContent = msg || "";
      }

      async function main() {
        const meta = document.getElementById("meta");
        const levelLabel = document.getElementById("levelLabel");
        const levelButtonsEl = document.getElementById("levelButtons");
        const mouseLabel = document.getElementById("mouseLabel");
        const layersEl = document.getElementById("layers");
        const seedEl = document.getElementById("seed");
        const gameSelectEl = document.getElementById("gameSelect");
        const undoBtn = document.getElementById("undoBtn");
        const resetLevelBtn = document.getElementById("resetLevelBtn");
        const restartBtn = document.getElementById("restartBtn");
        const recordBtn = document.getElementById("recordBtn");
        const useReplayBtn = document.getElementById("useReplayBtn");
        const copyAsciiBtn = document.getElementById("copyAsciiBtn");
        const refreshBtn = document.getElementById("refreshBtn");
        const reloadHint = document.getElementById("reloadHint");
        const wBtn = document.getElementById("wBtn");
        const aBtn = document.getElementById("aBtn");
        const sBtn = document.getElementById("sBtn");
        const dBtn = document.getElementById("dBtn");
        const spaceBtn = document.getElementById("spaceBtn");

        let actions = [];
        let state = null;
        let busy = false;
        let uiBusy = false;
        let uiBusyTimer = null;
        let reloadAvailable = false;
        let recording = false;
        let recordedActions = [];
        let recordedLevels = [];
        let animTimer = null;
        let animFrameIndex = 0;
        let currentAsciiFrame = "";
        // resolves when the current animation finishes; runOp awaits it so a new
        // action can't interrupt (skip) an animation that is still playing.
        let animationDone = Promise.resolve();
        let animationResolve = null;
        const quickActionButtons = new Map([
          [1, wBtn],
          [2, sBtn],
          [3, aBtn],
          [4, dBtn],
          [5, spaceBtn],
        ]);

        function cellFromEvent(evt, layer) {
          const rect = evt.target.getBoundingClientRect();
          const x = Math.floor(((evt.clientX - rect.left) / rect.width) * layer.width);
          const y = Math.floor(((evt.clientY - rect.top) / rect.height) * layer.height);
          const cx = clamp(x, 0, layer.width - 1);
          const cy = clamp(y, 0, layer.height - 1);
          return { x: cx, y: cy };
        }

        function setActions(newActions) {
          actions = Array.isArray(newActions) ? newActions : [];
          syncControls();
        }

        function clickActionId() {
          const a = actions.find((x) => x.complex);
          return a ? Number(a.id) : null;
        }

        function setBusy(value) {
          busy = value;
          if (value) {
            // Defer the disabled styling so quick steps don't flash the
            // controls; only ops that actually take a while grey them out.
            if (uiBusyTimer === null) {
              uiBusyTimer = setTimeout(() => {
                uiBusyTimer = null;
                uiBusy = true;
                syncControls();
              }, 150);
            }
          } else {
            if (uiBusyTimer !== null) {
              clearTimeout(uiBusyTimer);
              uiBusyTimer = null;
            }
            uiBusy = false;
            syncControls();
          }
        }

        async function runOp(fn) {
          if (busy) return;
          setBusy(true);
          try {
            await fn();
            await animationDone;  // stay busy until the animation finishes -> no skip
          } catch (e) {
            setError(String(e && e.message ? e.message : e));
          } finally {
            setBusy(false);
          }
        }

        function actionAvailable(actionId) {
          if (!actions.some((a) => Number(a.id) === Number(actionId))) return false;
          const avail = state ? state.available_actions : null;
          if (Array.isArray(avail)) {
            return avail.some((id) => Number(id) === Number(actionId));
          }
          return true;
        }

        function resetAvailable() {
          return actions.some((a) => String((a && a.name) || "").toUpperCase() === "RESET");
        }

        function mouseAvailable() {
          const id = clickActionId();
          return id !== null && actionAvailable(id);
        }

        function setReloadAvailable(value) {
          reloadAvailable = Boolean(value);
          refreshBtn.classList.toggle("primary", reloadAvailable);
          refreshBtn.textContent = reloadAvailable ? "Refresh available" : "Refresh";
          refreshBtn.title = reloadAvailable ? "New game files are available. Press to reload." : "";
          reloadHint.textContent = reloadAvailable ? "New game files available" : "";
        }

        function syncControls() {
          const disableInteractive = uiBusy || !state;
          quickActionButtons.forEach((button, actionId) => {
            const available = actionAvailable(actionId);
            button.disabled = disableInteractive || !available;
            button.title = !available && state ? "Action not available right now" : "";
          });
          resetLevelBtn.disabled = disableInteractive || !resetAvailable();
          undoBtn.disabled = uiBusy;
          restartBtn.disabled = uiBusy;
          recordBtn.disabled = uiBusy;
          useReplayBtn.style.display = recording ? "" : "none";
          useReplayBtn.disabled = uiBusy || !recording || recordedActions.length === 0;
          copyAsciiBtn.disabled = disableInteractive || !currentAsciiFrame;
          refreshBtn.disabled = uiBusy;
          gameSelectEl.disabled = uiBusy;
          seedEl.disabled = uiBusy;
          levelButtonsEl.querySelectorAll("button").forEach((button) => {
            button.disabled = disableInteractive;
          });

          const pointerEnabled = !disableInteractive && mouseAvailable();
          mouseLabel.textContent = `Mouse: ${pointerEnabled ? "available" : "unavailable"}`;
          layersEl.querySelectorAll("canvas").forEach((canvas) => {
            canvas.style.cursor = pointerEnabled ? "crosshair" : "not-allowed";
          });
        }

        function recordAction(actionId, data) {
          if (!recording) return;
          recordedActions.push([Number(actionId), { ...(data || {}) }]);
          recordedLevels.push(currentLevelNumber());
        }

        function currentLevelNumber() {
          if (!state) return null;
          const completed = Number.isFinite(state.levels_completed)
            ? Number(state.levels_completed)
            : 0;
          const total = Number.isFinite(state.win_levels)
            ? Number(state.win_levels)
            : 0;
          return total ? clamp(completed + 1, 1, total) : completed + 1;
        }

        function syncRecordingButton() {
          recordBtn.classList.toggle("primary", recording);
          recordBtn.textContent = recording ? `Stop recording (${recordedActions.length})` : "Record";
          syncControls();  // refresh the "use recording" button's visibility/enabled state
        }

        async function useRecordingForDsl() {
          if (!recordedActions.length) {
            setError("No actions recorded yet — record a full solution first.");
            return;
          }
          const result = await fetchJson(apiUrl("/use-recording"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              game_id: state && state.game_id ? String(state.game_id) : "",
              levels: recordedLevels,
              actions: recordedActions,
            }),
          });
          recording = false;
          syncRecordingButton();
          const made = result && result.agent_created ? "created agent" : "updated agent";
          setError(
            `Saved DSL replay: ${result.actions} actions / ${result.levels} levels -> ${result.replay_path} ` +
            `(${made}). ` +
            `Next: run \`make precompute_dsl_actions\` to bake & verify the trace — the DSL agent test ` +
            `stays red until that precomputed file exists. After a re-record or level change, force-refresh with ` +
            `\`python -m re_arc.dsl.precomputed_actions --game ${result.game_id} --sync-baseline-actions --force\`.`
          );
        }

        async function toggleRecording() {
          setError("");
          if (!recording) {
            recordedActions = [];
            recordedLevels = [];
            recording = true;
            syncRecordingButton();
            return;
          }

          recording = false;
          syncRecordingButton();
          await fetchJson(apiUrl("/recording"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              game_id: state && state.game_id ? String(state.game_id) : "",
              levels: recordedLevels,
              actions: recordedActions,
            }),
          });
        }

        async function copyAsciiFrame() {
          setError("");
          if (!currentAsciiFrame) {
            throw new Error("No ASCII frame is available to copy.");
          }
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(currentAsciiFrame);
          } else {
            const scratch = document.createElement("textarea");
            scratch.value = currentAsciiFrame;
            scratch.style.position = "fixed";
            scratch.style.left = "-9999px";
            document.body.appendChild(scratch);
            scratch.focus();
            scratch.select();
            document.execCommand("copy");
            scratch.remove();
          }
          copyAsciiBtn.textContent = "Copied";
          window.setTimeout(() => {
            copyAsciiBtn.textContent = "Copy ASCII";
          }, 900);
        }

        function triggerAction(actionId) {
          if (busy || !actionAvailable(actionId)) return;
          runOp(() => stepActionId(actionId));
        }

        async function stepActionId(actionId, extra = {}) {
          if (actionId === null || actionId === undefined) {
            throw new Error("No action selected.");
          }
          if (!actionAvailable(actionId)) {
            throw new Error(`Action ${actionId} is not available right now.`);
          }
          state = await fetchJson(apiUrl("/step"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action_id: Number(actionId), ...extra }),
          });
          recordAction(actionId, extra);
          syncRecordingButton();
          if (state && state.actions) setActions(state.actions);
          render();
        }

        function stopAnimation() {
          if (animTimer !== null) {
            clearInterval(animTimer);
            animTimer = null;
          }
          if (animationResolve !== null) {
            animationResolve();
            animationResolve = null;
          }
        }

        function renderFrame(layer) {
          const existing = layersEl.querySelector("canvas");
          if (existing && layer) {
            const available = Math.max(1, layersEl.clientWidth || 1);
            const scale = Math.max(1, Math.floor(available / layer.width));
            drawLayer(existing, layer, scale);
          }
          currentAsciiFrame = layerToAscii(layer);
        }

        function render() {
          stopAnimation();
          currentAsciiFrame = "";
          if (!state) {
            layersEl.innerHTML = "";
            levelButtonsEl.innerHTML = "";
            return;
          }

          const game = state.game_id ? `game=${state.game_id}` : "game=(unset)";
          const seed = (
            state.seed !== null && state.seed !== undefined
              ? `seed=${state.seed}`
              : "seed=NA"
          );
          const status = state.state ? `state=${state.state}` : "state=NA";
          const stepReward = Number.isFinite(state.last_reward)
            ? Number(state.last_reward).toFixed(4)
            : "NA";
          const totalReward = Number.isFinite(state.episode_reward)
            ? Number(state.episode_reward).toFixed(4)
            : "NA";
          const completed = Number.isFinite(state.levels_completed)
            ? Number(state.levels_completed)
            : 0;
          const total = Number.isFinite(state.win_levels)
            ? Number(state.win_levels)
            : 0;
          const current = total ? clamp(completed + 1, 1, total) : completed + 1;
          levelLabel.textContent = total
            ? `LEVEL: ${current} / ${total}`
            : `LEVEL: ${current}`;
          renderLevelButtons(current, total);
          meta.textContent = (
            `${game}  ${seed}  steps=${state.steps}  r_t=${stepReward}  ` +
            `R=${totalReward}  ${status}  ${levelLabel.textContent}`
          );

          const animFrames = state.animation_frames || [];
          const displayLayers = animFrames.length > 0 ? [animFrames[0]] : (state.layers || []);
          const asciiLayer = displayLayers.length > 0 ? displayLayers[0] : null;
          currentAsciiFrame = layerToAscii(asciiLayer);

          // Reuse existing canvases between renders so the board does not
          // flash; only rebuild when the layer count changes.
          let existingCanvases = layersEl.querySelectorAll("canvas");
          if (existingCanvases.length !== displayLayers.length) {
            layersEl.innerHTML = "";
            displayLayers.forEach((layer, idx) => {
              const wrap = document.createElement("div");
              wrap.className = "layer";
              if (displayLayers.length > 1) {
                const title = document.createElement("div");
                title.className = "layer-title";
                title.textContent = `Layer ${idx}`;
                wrap.appendChild(title);
              }
              const canvas = document.createElement("canvas");
              canvas.dataset.layerIndex = String(idx);
              canvas.addEventListener("click", (e) => {
                if (busy || !mouseAvailable()) return;
                const dims = {
                  width: Number(canvas.dataset.gridWidth) || 1,
                  height: Number(canvas.dataset.gridHeight) || 1,
                };
                const pos = cellFromEvent(e, dims);
                const actionId = clickActionId();
                if (actionId === null) return;
                runOp(() => stepActionId(actionId, { x: pos.x, y: pos.y }));
              });
              wrap.appendChild(canvas);
              layersEl.appendChild(wrap);
            });
            existingCanvases = layersEl.querySelectorAll("canvas");
          }
          displayLayers.forEach((layer, idx) => {
            const canvas = existingCanvases[idx];
            canvas.dataset.gridWidth = String(layer.width);
            canvas.dataset.gridHeight = String(layer.height);
            const available = Math.max(1, layersEl.clientWidth || 1);
            const scale = Math.max(1, Math.floor(available / layer.width));
            drawLayer(canvas, layer, scale);
          });

          if (animFrames.length > 1) {
            const delay = 50;
            animFrameIndex = 0;
            animationDone = new Promise((resolve) => { animationResolve = resolve; });
            animTimer = setInterval(() => {
              animFrameIndex++;
              if (animFrameIndex >= animFrames.length) {
                stopAnimation();
                return;
              }
              renderFrame(animFrames[animFrameIndex]);
            }, delay);
          } else {
            animationDone = Promise.resolve();
          }
          syncControls();
        }

        function renderLevelButtons(current, total) {
          if (!Number.isFinite(total) || total <= 1) {
            levelButtonsEl.innerHTML = "";
            return;
          }
          // Rebuild only when the level count changes; otherwise just move
          // the active highlight so the panel does not flicker per action.
          if (levelButtonsEl.childElementCount !== total) {
            levelButtonsEl.innerHTML = "";
            for (let i = 0; i < total; i++) {
              const button = document.createElement("button");
              button.type = "button";
              button.textContent = String(i + 1);
              button.addEventListener("click", () => runOp(() => jumpLevel(i)));
              levelButtonsEl.appendChild(button);
            }
          }
          Array.from(levelButtonsEl.children).forEach((button, i) => {
            button.className = (i + 1) === current ? "active" : "";
          });
        }

        async function refresh() {
          setError("");
          if (reloadAvailable) {
            state = await fetchJson(apiUrl("/reload"), {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({}),
            });
            await loadGames();
            setReloadAvailable(false);
          } else {
            state = await fetchJson(apiUrl("/state", { resample_random: 1 }));
          }
          if (state && state.actions) setActions(state.actions);
          render();
        }

        async function checkHotReload() {
          if (busy) return;
          const payload = await fetchJson(apiUrl("/reload-status"));
          setReloadAvailable(Boolean(payload && payload.reload_available));
        }

        async function loadGames() {
          const payload = await fetchJson(apiUrl("/games"));
          const games = Array.isArray(payload && payload.games) ? payload.games : [];
          const requested = (
            payload && payload.requested_game_id
              ? String(payload.requested_game_id)
              : ""
          );

          gameSelectEl.innerHTML = "";
          games.forEach((gid) => {
            const opt = document.createElement("option");
            opt.value = String(gid);
            opt.textContent = String(gid);
            gameSelectEl.appendChild(opt);
          });
          if (requested && games.includes(requested)) {
            gameSelectEl.value = requested;
          }
        }

        async function selectGame(gameId) {
          if (!gameId) return;
          setError("");
          state = await fetchJson(apiUrl("/select-game"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ game_id: String(gameId) }),
          });
          if (state && state.actions) setActions(state.actions);
          render();
          gameSelectEl.blur();
        }

        async function restart() {
          setError("");
          const rawSeed = seedEl.value.trim();
          const payload = {};
          if (rawSeed !== "") payload.seed = Number(rawSeed);
          state = await fetchJson(apiUrl("/reset"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (state && state.actions) setActions(state.actions);
          render();
        }

        async function resetLevel() {
          setError("");
          state = await fetchJson(apiUrl("/reset-level"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          if (state && state.actions) setActions(state.actions);
          render();
        }

        async function jumpLevel(levelIndex) {
          setError("");
          state = await fetchJson(apiUrl("/jump-level"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ level_index: Number(levelIndex) }),
          });
          if (state && state.actions) setActions(state.actions);
          render();
        }

        async function undo() {
          setError("");
          state = await fetchJson(apiUrl("/undo"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          if (state && state.actions) setActions(state.actions);
          render();
        }

        try {
          refreshBtn.addEventListener(
            "click",
            () => runOp(refresh),
          );
          recordBtn.addEventListener(
            "click",
            () => runOp(toggleRecording),
          );
          useReplayBtn.addEventListener(
            "click",
            () => runOp(useRecordingForDsl),
          );
          copyAsciiBtn.addEventListener(
            "click",
            () => runOp(copyAsciiFrame),
          );
          restartBtn.addEventListener(
            "click",
            () => runOp(restart),
          );
          resetLevelBtn.addEventListener(
            "click",
            () => runOp(resetLevel),
          );
          undoBtn.addEventListener(
            "click",
            () => runOp(undo),
          );
          gameSelectEl.addEventListener(
            "change",
            () => runOp(() => selectGame(gameSelectEl.value)),
          );

          wBtn.addEventListener(
            "click",
            () => triggerAction(1),
          );
          aBtn.addEventListener(
            "click",
            () => triggerAction(3),
          );
          sBtn.addEventListener(
            "click",
            () => triggerAction(2),
          );
          dBtn.addEventListener(
            "click",
            () => triggerAction(4),
          );
          spaceBtn.addEventListener(
            "click",
            () => triggerAction(5),
          );

          window.addEventListener("resize", () => {
            if (state) render();
          });

          window.setInterval(() => {
            checkHotReload().catch((e) => {
              setError(String(e && e.message ? e.message : e));
            });
          }, 750);

          document.addEventListener("keydown", (e) => {
            if (e.repeat) return;
            const active = document.activeElement;
            const tag = active && active.tagName ? active.tagName.toUpperCase() : "";
            if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

            const key = String(e.key || "").toLowerCase();
            if (key === "w") {
              e.preventDefault();
              triggerAction(1);
              return;
            }
            if (key === "a") {
              e.preventDefault();
              triggerAction(3);
              return;
            }
            if (key === "s") {
              e.preventDefault();
              triggerAction(2);
              return;
            }
            if (key === "d") {
              e.preventDefault();
              triggerAction(4);
              return;
            }
            if (e.code === "Space") {
              e.preventDefault();
              triggerAction(5);
              return;
            }
            if (key === "z") {
              e.preventDefault();
              runOp(undo);
              return;
            }
            if (key === "r") {
              if (e.ctrlKey || e.metaKey) {
                // Let the browser handle hard/page reload shortcuts.
                return;
              }
              e.preventDefault();
              if (e.shiftKey) runOp(restart);
              else if (resetAvailable()) runOp(resetLevel);
              return;
            }
          });

          syncControls();
          await loadGames();
          await refresh();
        } catch (e) {
          setError(String(e.message || e));
          meta.textContent = "Error";
        }
      }

      main();
    </script>
  </body>
</html>
"""


def _web_index_html(api_base: str = "/api") -> Any:
    color_map_json = json.dumps(
        {str(k): [int(v[0]), int(v[1]), int(v[2])] for k, v in COLOR_MAP.items()}, separators=(",", ":"), sort_keys=True
    )
    return _WEB_INDEX_HTML_TEMPLATE.replace("__COLOR_MAP_JSON__", color_map_json).replace(
        '"__API_BASE_JSON__"', json.dumps(api_base)
    )


_REPLAY_INDEX_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ARC Replay Viewer</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #0b0f17;
        --panel: #0f172a;
        --muted: #94a3b8;
        --text: #e2e8f0;
        --border: #1e293b;
        --accent: #60a5fa;
      }
      body {
        margin: 0;
        background: radial-gradient(900px circle at 20% 10%, #111827, var(--bg));
        color: var(--text);
        font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }
      .app {
        max-width: 1180px;
        margin: 0 auto;
        padding: 20px;
      }
      .grid {
        display: grid;
        gap: 16px;
        grid-template-columns: minmax(340px, 520px) minmax(300px, 1fr);
      }
      .card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
      }
      h1, h2, p {
        margin: 0;
      }
      .meta {
        color: var(--muted);
        margin-top: 6px;
      }
      .topbar {
        display: flex;
        gap: 12px;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
      }
      .picker {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }
      select {
        border: 1px solid var(--border);
        background: #13203a;
        color: var(--text);
        border-radius: 10px;
        padding: 8px 12px;
        min-width: 260px;
      }
      canvas {
        display: block;
        width: min(100%, 480px);
        aspect-ratio: 1;
        margin: 0 auto;
        background: #fff;
        image-rendering: pixelated;
        border: 1px solid var(--border);
        border-radius: 8px;
      }
      .controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
        margin-top: 14px;
      }
      .review-controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
        margin-top: 14px;
      }
      button {
        border: 1px solid var(--border);
        background: #13203a;
        color: var(--text);
        border-radius: 10px;
        padding: 8px 12px;
        cursor: pointer;
      }
      button:hover {
        border-color: var(--accent);
      }
      input[type="range"] {
        width: 100%;
        margin-top: 14px;
      }
      .kv {
        display: grid;
        grid-template-columns: 140px 1fr;
        gap: 6px 12px;
        margin-top: 14px;
      }
      .kv div:nth-child(odd) {
        color: var(--muted);
      }
      pre {
        margin: 14px 0 0;
        padding: 12px;
        background: #08101f;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-size: 12px;
      }
      .pill.accepted {
        background: #10361e;
        color: #9ae6b4;
      }
      .pill.rejected {
        background: #3a1515;
        color: #feb2b2;
      }
      .pill.pending {
        background: #1c2333;
        color: #cbd5e1;
      }
      @media (max-width: 860px) {
        .grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <div class="card" style="margin-bottom: 16px">
        <div class="topbar">
          <div>
            <h1>ARC Replay Viewer</h1>
            <p class="meta" id="runMeta">Loading replay...</p>
            <p class="meta" id="reviewMeta"></p>
            <div class="meta" id="ideaMeta" style="display:none"></div>
          </div>
          <div class="picker">
            <label for="replaySelect">Replay</label>
            <select id="replaySelect"></select>
          </div>
        </div>
      </div>
      <div class="grid">
        <div class="card">
          <canvas id="screen" width="64" height="64"></canvas>
          <input id="scrubber" type="range" min="0" max="0" value="0" />
          <div class="controls">
            <button id="prevBtn">Prev</button>
            <button id="playBtn">Play</button>
            <button id="nextBtn">Next</button>
            <button id="prevTransitionBtn">Prev Transition</button>
            <button id="nextTransitionBtn">Next Transition</button>
          </div>
          <div class="review-controls" id="reviewControls" style="display:none">
            <button id="playGameBtn">Play Game</button>
            <button id="acceptBtn">Accept</button>
            <button id="rejectBtn">Reject</button>
            <button id="submitReviewBtn">Submit Review</button>
            <span id="decisionPill" class="pill pending">pending</span>
          </div>
        </div>
        <div class="card">
          <h2>Step Details</h2>
          <div class="kv" id="details"></div>
          <pre id="payload"></pre>
        </div>
      </div>
    </div>
    <script>
      const COLOR_MAP = __COLOR_MAP_JSON__;
      const REPLAY_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_";
      const runMeta = document.getElementById("runMeta");
      const reviewMeta = document.getElementById("reviewMeta");
      const ideaMeta = document.getElementById("ideaMeta");
      const replaySelect = document.getElementById("replaySelect");
      const screen = document.getElementById("screen");
      const scrubber = document.getElementById("scrubber");
      const details = document.getElementById("details");
      const payload = document.getElementById("payload");
      const reviewControls = document.getElementById("reviewControls");
      const playGameBtn = document.getElementById("playGameBtn");
      const acceptBtn = document.getElementById("acceptBtn");
      const rejectBtn = document.getElementById("rejectBtn");
      const submitReviewBtn = document.getElementById("submitReviewBtn");
      const decisionPill = document.getElementById("decisionPill");
      const ctx = screen.getContext("2d");
      const CELL_SIZE = 12;
      let replay = null;
      let replayList = [];
      let reviewState = { enabled: false, decisions_by_name: {}, counts: {} };
      let index = 0;
      let timer = null;
      let animationTimer = null;

      function layerWidth(layer) {
        return Number(layer && layer.width || 0);
      }

      function layerHeight(layer) {
        return Number(layer && layer.height || 0);
      }

      function layerCell(layer, x, y) {
        if (!layer) return 0;
        const rows = Array.isArray(layer.rows) ? layer.rows : null;
        const row = rows ? String(rows[y] || "") : "";
        const ch = row.charAt(x);
        const idx = REPLAY_DIGITS.indexOf(ch);
        return idx >= 0 ? idx : 0;
      }

      function drawLayer(layer) {
        const width = layerWidth(layer);
        const height = layerHeight(layer);
        screen.width = Math.max(1, width || 64) * CELL_SIZE;
        screen.height = Math.max(1, height || 64) * CELL_SIZE;
        ctx.clearRect(0, 0, screen.width, screen.height);
        if (!layer) return;
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const color = COLOR_MAP[String(layerCell(layer, x, y))] || [0, 0, 0];
            ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
          }
        }
      }

      function drawObservation(observation) {
        const layer = observation && observation.layers && observation.layers[0];
        drawLayer(layer);
      }

      function stopStepAnimation() {
        if (animationTimer) {
          clearInterval(animationTimer);
          animationTimer = null;
        }
      }

      function animationFrames(observation) {
        const frames = observation && observation.animation_frames;
        return Array.isArray(frames) ? frames.filter((frame) => frame && frame.rows) : [];
      }

      function renderAnimatedObservation(record, observation) {
        stopStepAnimation();
        const frames = animationFrames(observation);
        if (frames.length <= 1) {
          drawObservation(observation);
          highlightClick(record, observation);
          return;
        }

        let frameIndex = 0;
        drawLayer(frames[frameIndex]);
        animationTimer = setInterval(() => {
          frameIndex++;
          if (frameIndex >= frames.length) {
            stopStepAnimation();
            drawObservation(observation);
            highlightClick(record, observation);
            return;
          }
          drawLayer(frames[frameIndex]);
        }, 50);
      }

      function highlightClick(record, observation) {
        if (!record || Number(record.action && record.action.id) !== 6) return;
        const data = (record.action && record.action.data) || {};
        const x = Number(data.x);
        const y = Number(data.y);
        const layer = observation && observation.layers && observation.layers[0];
        const width = layer ? layerWidth(layer) : Math.floor(screen.width / CELL_SIZE);
        const height = layer ? layerHeight(layer) : Math.floor(screen.height / CELL_SIZE);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        ctx.save();
        ctx.strokeStyle = clickOutlineColor(observation, x, y);
        ctx.lineWidth = 2;
        ctx.strokeRect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
        ctx.restore();
      }

      function stepRecord(idx) {
        if (!replay) return null;
        if (idx <= 0) return null;
        return replay.steps[idx - 1] || null;
      }

      function observationAt(idx) {
        const record = stepRecord(idx);
        return record ? record.observation : replay.initial_observation;
      }

      function humanActionName(action) {
        const mapping = {
          1: "UP",
          2: "DOWN",
          3: "LEFT",
          4: "RIGHT",
          5: "SPACE",
          6: "CLICK",
        };
        const id = Number(action && action.id);
        if (mapping[id]) return mapping[id];
        return String((action && action.name) || `ACTION_${id}`);
      }

      function clickOutlineColor(observation, x, y) {
        const layer = observation && observation.layers && observation.layers[0];
        const colorId = layerCell(layer, x, y);
        const rgb = COLOR_MAP[String(colorId)] || [255, 255, 255];
        const luminance = (0.299 * rgb[0]) + (0.587 * rgb[1]) + (0.114 * rgb[2]);
        const isRedLike = rgb[0] > 180 && rgb[1] < 120 && rgb[2] < 140;
        if (isRedLike) return "#00e5ff";
        return luminance < 110 ? "#ffffff" : "#111111";
      }

      function render() {
        if (!replay) return;
        const record = stepRecord(index);
        const observation = observationAt(index);
        renderAnimatedObservation(record, observation);
        scrubber.max = String(replay.steps.length);
        scrubber.value = String(index);
        const meta = replay.metadata || {};
        runMeta.textContent =
          `${meta.game_id || "unknown"}  seed=${meta.seed}  ` +
          `steps=${replay.steps.length}  normalized=${meta.normalized_score}`;
        const items = [];
        items.push(["Position", `${index} / ${replay.steps.length}`]);
        items.push(["State", String(observation && observation.state || "UNKNOWN")]);
        items.push([
          "Levels",
          `${observation && observation.levels_completed || 0} / ` +
          `${observation && observation.win_levels || 0}`,
        ]);
        if (record) {
          items.push(["Action", `${humanActionName(record.action)} (${record.action.id})`]);
          items.push(["Reward", String(record.reward)]);
          items.push(["Episode Reward", String(record.episode_reward)]);
          items.push(["Transition", record.info && record.info.level_transition ? "yes" : "no"]);
        } else {
          items.push(["Action", "Initial observation"]);
          items.push(["Reward", "0"]);
          items.push(["Episode Reward", "0"]);
          items.push(["Transition", "no"]);
        }
        details.innerHTML = items.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("");
        payload.textContent = JSON.stringify(record ? record.action.data : {}, null, 2);
        document.getElementById("playBtn").textContent = timer ? "Pause" : "Play";
        renderReviewControls();
      }

      function renderReplayPicker(selectedName) {
        replaySelect.innerHTML = replayList
          .map((item) => {
            const selected = item.name === selectedName ? " selected" : "";
            const suffix = item.decision ? ` [${item.decision}]` : "";
            return `<option value="${item.name.replaceAll('"', '&quot;')}"${selected}>${item.label}${suffix}</option>`;
          })
          .join("");
      }

      function renderReviewControls() {
        if (!reviewState.enabled || !replay) {
          reviewControls.style.display = "none";
          reviewMeta.textContent = "";
          ideaMeta.style.display = "none";
          ideaMeta.innerHTML = "";
          return;
        }
        reviewControls.style.display = "flex";
        playGameBtn.style.display = replay.play_url ? "inline-flex" : "none";
        const currentDecision = String((replay.review && replay.review.decision) || "pending").toLowerCase();
        decisionPill.textContent = currentDecision;
        decisionPill.className = `pill ${currentDecision}`;
        const counts = reviewState.counts || {};
        reviewMeta.textContent =
          `accepted=${counts.accepted || 0} rejected=${counts.rejected || 0} pending=${counts.pending || 0}`;
        submitReviewBtn.disabled = !reviewState.all_decided || !!reviewState.submitted;
        const context = replay.review_context || {};
        const ideaTitle = String(context.idea_title || "").trim();
        const ideaDescription = String(context.idea_description || "").trim();
        const ideaId = String(context.idea_id || "").trim();
        if (ideaTitle || ideaDescription || ideaId) {
          const titleLine = ideaTitle || ideaId || "Unknown idea";
          const idLine = ideaId ? `<div><strong>Idea ID:</strong> ${ideaId}</div>` : "";
          const bodyLine = ideaDescription ? `<div>${ideaDescription}</div>` : "";
          ideaMeta.innerHTML = `<div><strong>Original Idea:</strong> ${titleLine}</div>${idLine}${bodyLine}`;
          ideaMeta.style.display = "block";
        } else {
          ideaMeta.style.display = "none";
          ideaMeta.innerHTML = "";
        }
      }

      function setIndex(next) {
        if (!replay) return;
        index = Math.max(0, Math.min(replay.steps.length, Number(next) || 0));
        render();
      }

      function jumpTransition(direction) {
        if (!replay) return;
        let cursor = index + direction;
        while (cursor >= 1 && cursor <= replay.steps.length) {
          const record = stepRecord(cursor);
          if (record && record.info && record.info.level_transition) {
            setIndex(cursor);
            return;
          }
          cursor += direction;
        }
      }

      function togglePlay() {
        if (timer) {
          clearInterval(timer);
          timer = null;
          render();
          return;
        }
        timer = setInterval(() => {
          if (!replay || index >= replay.steps.length) {
            clearInterval(timer);
            timer = null;
            render();
            return;
          }
          setIndex(index + 1);
        }, 450);
        render();
      }

      async function loadReplay(name) {
        const query = name ? `?name=${encodeURIComponent(name)}` : "";
        const response = await fetch(`/api/replay${query}`, { cache: "no-store" });
        replay = await response.json();
        index = 0;
        renderReplayPicker(replay.selected_replay);
        render();
      }

      async function setDecision(decision) {
        if (!replay) return;
        const response = await fetch("/api/review/decision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: replay.selected_replay, decision }),
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Failed to save review decision.");
        reviewState = body;
        replay.review = { ...(replay.review || {}), decision };
        replayList = replayList.map((item) =>
          item.name === replay.selected_replay ? { ...item, decision } : item
        );
        renderReplayPicker(replay.selected_replay);
        render();
      }

      async function submitReview() {
        const response = await fetch("/api/review/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Failed to submit review.");
        reviewState = body;
        render();
        const promotedCount = Number(body.submission && body.submission.promoted_count || 0);
        reviewMeta.textContent = `${reviewMeta.textContent}  submitted: promoted=${promotedCount}`;
      }

      async function main() {
        const listResponse = await fetch("/api/replays", { cache: "no-store" });
        const listPayload = await listResponse.json();
        replayList = listPayload.replays || [];
        reviewState = listPayload.review || { enabled: false, decisions_by_name: {}, counts: {} };
        await loadReplay(listPayload.selected_replay || (replayList[0] && replayList[0].name) || "");
      }

      scrubber.addEventListener("input", (e) => setIndex(e.target.value));
      replaySelect.addEventListener("change", (e) => loadReplay(e.target.value));
      document.getElementById("prevBtn").addEventListener("click", () => setIndex(index - 1));
      document.getElementById("nextBtn").addEventListener("click", () => setIndex(index + 1));
      document.getElementById("playBtn").addEventListener("click", togglePlay);
      document.getElementById("prevTransitionBtn").addEventListener("click", () => jumpTransition(-1));
      document.getElementById("nextTransitionBtn").addEventListener("click", () => jumpTransition(1));
      acceptBtn.addEventListener("click", () => setDecision("accepted").catch((err) => {
        reviewMeta.textContent = String(err.message || err);
      }));
      rejectBtn.addEventListener("click", () => setDecision("rejected").catch((err) => {
        reviewMeta.textContent = String(err.message || err);
      }));
      submitReviewBtn.addEventListener("click", () => submitReview().catch((err) => {
        reviewMeta.textContent = String(err.message || err);
      }));
      playGameBtn.addEventListener("click", () => {
        if (!replay || !replay.play_url) return;
        window.open(replay.play_url, "_blank", "noopener");
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") setIndex(index - 1);
        if (e.key === "ArrowRight") setIndex(index + 1);
        if (e.code === "Space") {
          e.preventDefault();
          togglePlay();
        }
      });
      main().catch((err) => {
        runMeta.textContent = String(err.message || err);
      });
    </script>
  </body>
</html>
"""


def _replay_index_html() -> Any:
    color_map_json = json.dumps(
        {str(k): [int(v[0]), int(v[1]), int(v[2])] for k, v in COLOR_MAP.items()}, separators=(",", ":"), sort_keys=True
    )
    return _REPLAY_INDEX_HTML_TEMPLATE.replace("__COLOR_MAP_JSON__", color_map_json)


def _to_py_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _action_id_value(action: Any) -> int:
    value = getattr(action, "value", action)
    if isinstance(value, tuple) and value:
        value = value[0]
    return int(value)


def _layer_to_cells(layer: Any) -> Any:
    if layer is None:
        return None
    if hasattr(layer, "tolist"):
        layer = layer.tolist()
    if not isinstance(layer, list):
        return None
    out = []
    for row in layer:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list):
            return None
        out.append([int(v) for v in row])
    return out


_REPLAY_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"


def _encode_replay_row(row: Any) -> Any:
    if not isinstance(row, list):
        return None
    chars = []
    for value in row:
        idx = int(value)
        if not (0 <= idx < len(_REPLAY_DIGITS)):
            raise ValueError(f"Replay cell value out of range: {idx}")
        chars.append(_REPLAY_DIGITS[idx])
    return "".join(chars)


def _final_frame(obs: Any) -> Any:
    if obs is None:
        return None
    frames = getattr(obs, "frame", None)
    if frames is None:
        return None
    if isinstance(frames, list):
        if not frames:
            return None
        return frames[-1]
    return frames


def _serialize_observation(obs: Any) -> Any:
    state = getattr(obs, "state", None) if obs is not None else None
    state_name = getattr(state, "name", None)
    if state_name is None and state is not None:
        state_name = str(state)
    levels_completed = getattr(obs, "levels_completed", None) if obs is not None else None
    win_levels = getattr(obs, "win_levels", None) if obs is not None else None
    full_reset = getattr(obs, "full_reset", None) if obs is not None else None
    available_actions = getattr(obs, "available_actions", None) if obs is not None else None

    layers_out: list[dict[str, Any]] = []
    frame = _final_frame(obs)
    if frame is not None:
        cells = _layer_to_cells(frame)
        if not cells:
            layers_out = []
        else:
            height = len(cells)
            width = len(cells[0]) if height else 0
            layers_out.append({"width": width, "height": height, "cells": cells})

    # Serialize all temporal animation frames (not just the final one).
    animation_frames: list[dict[str, Any]] = []
    raw_frames = getattr(obs, "frame", None) if obs is not None else None
    if isinstance(raw_frames, list):
        for raw_frame in raw_frames:
            cells = _layer_to_cells(raw_frame)
            if cells:
                height = len(cells)
                width = len(cells[0]) if height else 0
                animation_frames.append({"width": width, "height": height, "cells": cells})
    elif raw_frames is not None:
        cells = _layer_to_cells(raw_frames)
        if cells:
            height = len(cells)
            width = len(cells[0]) if height else 0
            animation_frames.append({"width": width, "height": height, "cells": cells})

    done = bool(_dsl_is_terminal(obs))

    return {
        "state": state_name,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "full_reset": full_reset,
        "available_actions": available_actions,
        "done": done,
        "layers": layers_out,
        "animation_frames": animation_frames,
    }


def _serialize_replay_observation(obs: Any) -> Any:
    state = getattr(obs, "state", None) if obs is not None else None
    state_name = getattr(state, "name", None)
    if state_name is None and state is not None:
        state_name = str(state)
    levels_completed = getattr(obs, "levels_completed", None) if obs is not None else None
    win_levels = getattr(obs, "win_levels", None) if obs is not None else None

    layers_out: list[dict[str, Any]] = []
    frame = _final_frame(obs)
    if frame is not None:
        cells = _layer_to_cells(frame)
        if cells:
            height = len(cells)
            width = len(cells[0]) if height else 0
            layers_out.append({"width": width, "height": height, "rows": [_encode_replay_row(row) for row in cells]})

    animation_frames: list[dict[str, Any]] = []
    raw_frames = getattr(obs, "frame", None) if obs is not None else None
    replay_frames = raw_frames if isinstance(raw_frames, list) else [raw_frames] if raw_frames is not None else []
    for raw_frame in replay_frames:
        cells = _layer_to_cells(raw_frame)
        if cells:
            height = len(cells)
            width = len(cells[0]) if height else 0
            animation_frames.append(
                {"width": width, "height": height, "rows": [_encode_replay_row(row) for row in cells]}
            )

    return {
        "state": state_name,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "layers": layers_out,
        "animation_frames": animation_frames,
    }


def _environment_watch_signature(config: Any) -> tuple[int, int]:
    environments_dir = Path(config.get("ENVIRONMENTS_DIR") or default_environments_dir()).expanduser()
    if not environments_dir.exists() or not environments_dir.is_dir():
        return (0, 0)

    watched_suffixes = {".py", ".json"}
    latest_mtime = 0
    file_count = 0
    for path in environments_dir.rglob("*"):
        if not path.is_file() or path.suffix not in watched_suffixes:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        latest_mtime = max(latest_mtime, int(stat.st_mtime_ns))
        file_count += 1
    return (latest_mtime, file_count)


def _purge_environment_modules() -> None:
    for module_name in list(sys.modules):
        if module_name.startswith(("arc_agi_3.", "re_arc.environment_files.")):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _unwrap_attr(obj: Any, attr_name: str) -> Any:
    seen: set[int] = set()
    current = obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = current.__dict__.get(attr_name) if hasattr(current, "__dict__") else None
        if value is not None:
            return value
        current = current.__dict__.get("_env") if hasattr(current, "__dict__") else None
    return None


def _format_recorded_action(action: list[Any]) -> str:
    action_id = int(action[0])
    data = action[1]
    if not data:
        return str(action_id)
    args = ",".join(f"{key}={value}" for key, value in sorted(data.items()))
    return f"{action_id}({args})"


def _normalize_recorded_levels(levels: Any) -> list[int]:
    if not isinstance(levels, list):
        return []
    normalized: list[int] = []
    for value in levels:
        level = _to_py_int(value)
        if level is not None:
            normalized.append(int(level))
    return normalized


def _format_recorded_levels(levels: list[int]) -> str:
    unique = sorted(set(levels))
    if not unique:
        return "level=unknown"
    if len(unique) == 1:
        return f"level={unique[0]}"
    if unique == list(range(unique[0], unique[-1] + 1)):
        return f"levels={unique[0]}-{unique[-1]}"
    return "levels=" + ",".join(str(level) for level in unique)


def _normalize_recorded_actions(actions: Any) -> list[list[Any]]:
    if not isinstance(actions, list):
        raise ValueError("actions must be a list.")
    normalized_actions: list[list[Any]] = []
    for index, entry in enumerate(actions):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"actions[{index}] must be [action_id, data].")
        action_id = _to_py_int(entry[0])
        if action_id is None:
            raise ValueError(f"actions[{index}][0] must be an integer action id.")
        raw_data = entry[1] or {}
        if not isinstance(raw_data, dict):
            raise ValueError(f"actions[{index}][1] must be an object.")
        data: dict[str, int] = {}
        for key, value in raw_data.items():
            int_value = _to_py_int(value)
            if int_value is None:
                raise ValueError(f"actions[{index}][1][{key!r}] must be an integer.")
            data[str(key)] = int_value
        normalized_actions.append([int(action_id), data])
    return normalized_actions


def _print_recorded_actions(game_id: Any, actions: Any, levels: Any = None) -> dict[str, Any]:
    normalized_actions = _normalize_recorded_actions(actions)
    label = str(game_id or "").strip() or "(unknown game)"
    normalized_levels = _normalize_recorded_levels(levels)
    level_label = _format_recorded_levels(normalized_levels)
    action_text = " ".join(_format_recorded_action(action) for action in normalized_actions)
    print(f"\nRecorded solution: game={label} {level_label} actions={len(normalized_actions)}", flush=True)
    print(action_text, flush=True)
    return {"printed": True, "actions": len(normalized_actions)}


def _segment_actions_by_level(actions: list[list[Any]], levels: list[int]) -> list[list[list[Any]]]:
    """Group a flat action list into per-level segments using the parallel level
    numbers the player records. Levels are non-decreasing, so first-seen order
    yields one segment per level; falls back to a single segment if unavailable."""
    if len(levels) != len(actions) or not levels:
        return [list(actions)]
    order: list[int] = []
    by_level: dict[int, list[list[Any]]] = {}
    for action, level in zip(actions, levels, strict=True):
        if level not in by_level:
            by_level[level] = []
            order.append(level)
        by_level[level].append(action)
    return [by_level[level] for level in order]


def _use_recording_for_dsl(game_id: Any, actions: Any, levels: Any) -> dict[str, Any]:
    """Persist a player recording as a human-replay DSL agent, editing the repo
    in place: write re_arc/dsl/human_replays/<base>.json and create the agent
    module. Refuses to clobber an existing non-replay (hand-written) agent."""
    gid = str(game_id or "").strip()
    if not gid:
        raise ValueError("No game is loaded; cannot attach a DSL agent.")
    base = gid.split("-", 1)[0].strip().lower()
    normalized_actions = _normalize_recorded_actions(actions)
    if not normalized_actions:
        raise ValueError("No actions recorded yet — record a full solution first.")
    normalized_levels = _normalize_recorded_levels(levels)

    from re_arc.dsl import core as dsl_core
    from re_arc.dsl.agents._human_replay import REPLAY_DIR, HumanReplayAgent

    try:
        existing = dsl_core.resolve_agent_class(gid)
    except ModuleNotFoundError:
        existing = None
    except ValueError as exc:
        raise ValueError(
            f"{gid}: existing agent module re_arc/dsl/agents/{base}.py is invalid ({exc}); fix it first."
        ) from exc
    if existing is not None and not (isinstance(existing, type) and issubclass(existing, HumanReplayAgent)):
        raise ValueError(
            f"{gid} already has a custom DSL agent {existing.__name__} in re_arc/dsl/agents/{base}.py "
            f"that is not a human replay. Refusing to overwrite it; remove that agent first."
        )

    segments = _segment_actions_by_level(normalized_actions, normalized_levels)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    replay_path = REPLAY_DIR / f"{base}.json"
    replay_payload: dict[str, Any] = {"game_id": gid, "source": {"mode": "player-recording"}, "levels": segments}
    replay_path.write_text(json.dumps(replay_payload, indent=2) + "\n", encoding="utf-8")

    agent_path = Path(dsl_core.__file__).resolve().parent / "agents" / f"{base}.py"
    agent_created = not agent_path.exists()
    if agent_created:
        agent_path.write_text(
            "from ._human_replay import HumanReplayAgent\n\nAGENT_CLASS = HumanReplayAgent\n", encoding="utf-8"
        )

    print(
        f"\nDSL agent from recording: game={gid} actions={len(normalized_actions)} levels={len(segments)}", flush=True
    )
    print(f"  replay -> {replay_path}", flush=True)
    print(f"  agent  -> {agent_path} ({'created' if agent_created else 'already present'})", flush=True)
    return {
        "ok": True,
        "game_id": gid,
        "actions": len(normalized_actions),
        "levels": len(segments),
        "replay_path": str(replay_path),
        "agent_path": str(agent_path),
        "agent_created": agent_created,
    }


class _WebSession:
    def __init__(self: Any, config: Any, game_id: Any, seed: Any) -> None:
        self._lock = threading.Lock()
        self._config = config
        self._requested_game_id = game_id
        self.game_id = game_id
        self._random_game_mode = _is_random_game_id(game_id)
        self.seed = seed
        self.steps = 0
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self._history: list[dict[str, Any]] = []
        self._level_start_history_len = 0
        self._history_base_level = 0
        self._available_game_ids = tuple(_list_game_ids(config, apply_dataset_filters=False))
        if "random" not in self._available_game_ids:
            self._available_game_ids = tuple(sorted({*self._available_game_ids, "random"}))
        self._reload_token = _environment_watch_signature(self._config)

        self._arc = None
        self._sampler = None
        self._reset_backend_for_requested_game_locked()
        self._env = None
        self._obs = None
        with self._lock:
            self._create_env_locked()

    def _reset_backend_for_requested_game_locked(self: Any) -> Any:
        _purge_environment_modules()
        self._random_game_mode = _is_random_game_id(self._requested_game_id)
        if self._random_game_mode or _sampler_augmentation_active(self._config):
            self._arc = None
            self._sampler = _build_env_sampler(
                self._config,
                seed=self.seed,
                augment=_augment_enabled(self._config),
                include=None if self._random_game_mode else [str(self._requested_game_id)],
                apply_dataset_filters=self._random_game_mode,
            )
            return
        self._sampler = None
        self._arc = _build_arcade(self._config)

    def _sample_random_game_id_locked(self: Any, previous_game_id: Any = None) -> Any:
        if self._sampler is None:
            raise RuntimeError("Random game sampler is not initialized.")
        game_ids = tuple(getattr(self._sampler, "game_ids", ()) or ())
        if not game_ids:
            raise RuntimeError("No games available in random game sampler.")

        previous = str(previous_game_id).strip().lower() if previous_game_id else None
        candidate = str(self._sampler.sample_game_id())
        if previous and len(game_ids) > 1 and candidate.strip().lower() == previous:
            attempts = max(4, len(game_ids) * 3)
            for _ in range(attempts):
                candidate = str(self._sampler.sample_game_id())
                if candidate.strip().lower() != previous:
                    break
        return candidate

    def _create_env_locked(self: Any, prefer_new_random_game: Any = False) -> Any:
        if self._sampler is not None:
            if self._random_game_mode:
                previous_game_id = self.game_id if prefer_new_random_game else None
                selected_game_id = self._sample_random_game_id_locked(previous_game_id=previous_game_id)
            else:
                selected_game_id = str(self._requested_game_id)
            base_env = self._sampler.make(game_id=selected_game_id, seed=self.seed, renderer=None)
            if base_env is None:
                raise RuntimeError(f"Failed to create environment: {selected_game_id}")
        else:
            base_env = self._arc.make(self._requested_game_id, seed=self.seed, renderer=None)
            if base_env is None:
                raise RuntimeError(f"Failed to create environment: {self._requested_game_id}")

        self._env = _wrap_env_with_transition_rewards(env=base_env, game_id=self._requested_game_id, seed=self.seed)
        self._obs = self._env.reset()
        self.game_id = _selected_game_id_from_observation(self._obs, self._requested_game_id)
        self.steps = 0
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self._history = []
        self._level_start_history_len = 0
        self._history_base_level = 0

    def _reload_if_changed_locked(self: Any) -> bool:
        signature = _environment_watch_signature(self._config)
        if signature == self._reload_token:
            return False
        self._reload_token = signature
        self._available_game_ids = tuple(_list_game_ids(self._config, apply_dataset_filters=False))
        if "random" not in self._available_game_ids:
            self._available_game_ids = tuple(sorted({*self._available_game_ids, "random"}))
        self._reset_backend_for_requested_game_locked()
        self._create_env_locked(prefer_new_random_game=False)
        return True

    def _reload_available_locked(self: Any) -> bool:
        return _environment_watch_signature(self._config) != self._reload_token

    def _actions_locked(self: Any) -> Any:
        env = self._env
        if env is None:
            return []
        actions = []
        for idx, action in enumerate(env.action_space):
            actions.append(
                {
                    "index": idx,
                    "id": int(action.value),
                    "name": getattr(action, "name", str(action)),
                    "complex": bool(action.is_complex()),
                }
            )
        return actions

    def actions(self: Any) -> Any:
        with self._lock:
            return self._actions_locked()

    def game_options(self: Any) -> Any:
        with self._lock:
            return {"games": list(self._available_game_ids), "requested_game_id": self._requested_game_id}

    def reload_status(self: Any) -> Any:
        with self._lock:
            return {
                "reload_token": list(self._reload_token),
                "reload_available": self._reload_available_locked(),
                "reloaded": False,
                "game_id": self.game_id,
            }

    def reload(self: Any) -> Any:
        with self._lock:
            self._reload_if_changed_locked()
            return self._state_locked()

    def select_game(self: Any, game_id: Any, seed: Any = None) -> Any:
        with self._lock:
            requested = str(game_id or "").strip()
            if not requested:
                raise ValueError("game_id is required.")
            if self._available_game_ids and requested not in self._available_game_ids:
                raise ValueError(f"Unknown game_id: {requested}")

            if seed is not None and seed != self.seed:
                self.seed = seed
            self._requested_game_id = requested
            self._reset_backend_for_requested_game_locked()
            self._create_env_locked(prefer_new_random_game=False)
            return self._state_locked()

    def state(self: Any, resample_random: Any = False) -> Any:
        with self._lock:
            if resample_random and self._random_game_mode:
                self._create_env_locked(prefer_new_random_game=True)
            return self._state_locked()

    def _state_locked(self: Any) -> Any:
        payload = _serialize_observation(self._obs)
        payload.update(
            {
                "game_id": self.game_id,
                "requested_game_id": self._requested_game_id,
                "seed": self.seed,
                "steps": self.steps,
                "level_start_steps": self._level_start_history_len,
                "last_reward": float(self.last_reward),
                "episode_reward": float(self.episode_reward),
                "actions": self._actions_locked(),
                "reload_token": list(self._reload_token),
            }
        )
        return payload

    def _with_only_reset_levels_locked(self: Any, enabled: Any) -> Any:
        key = "ONLY_RESET_LEVELS"
        prev = os.environ.get(key)
        if enabled:
            os.environ[key] = "true"
        else:
            os.environ.pop(key, None)
        return prev

    def _restore_only_reset_levels_locked(self: Any, prev: Any) -> Any:
        key = "ONLY_RESET_LEVELS"
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev

    def _full_reset_env_locked(self: Any) -> Any:
        if self._env is None:
            self._create_env_locked()
            return
        prev = self._with_only_reset_levels_locked(False)
        try:
            obs = self._env.reset()
            if obs is not None and not bool(getattr(obs, "full_reset", False)):
                obs = self._env.reset()
            self._obs = obs
            self.last_reward = 0.0
            self.episode_reward = 0.0
        finally:
            self._restore_only_reset_levels_locked(prev)

    def restart(self: Any, seed: Any = None) -> Any:
        with self._lock:
            if seed is not None and seed != self.seed:
                self.seed = seed
                self._create_env_locked()
                return self._state_locked()

            if self._random_game_mode:
                self._create_env_locked(prefer_new_random_game=True)
                return self._state_locked()

            self._full_reset_env_locked()
            self._history = []
            self._level_start_history_len = 0
            self._history_base_level = 0
            self.steps = 0
            return self._state_locked()

    def reset_level(self: Any) -> Any:
        return self.step(_action_id_value(GameAction.RESET), {})

    def jump_level(self: Any, level_index: Any) -> Any:
        with self._lock:
            if self._env is None:
                self._create_env_locked()
            if self._random_game_mode:
                raise ValueError("Level jumping is not available while GAME_ID=random.")

            target = _to_py_int(level_index)
            if target is None:
                raise ValueError("level_index is required.")

            self._full_reset_env_locked()
            win_levels = _to_py_int(getattr(self._obs, "win_levels", None)) or 1
            if not 0 <= target < win_levels:
                raise ValueError(f"level_index must be in [0, {win_levels}).")

            self._jump_to_level_locked(int(target), int(win_levels))
            self._history = []
            self._level_start_history_len = 0
            self._history_base_level = int(target)
            self.steps = 0
            return self._state_locked()

    def _jump_to_level_locked(self: Any, target: int, win_levels: int) -> Any:
        game = _unwrap_attr(self._env, "_game")
        if game is None or not callable(getattr(game, "set_level", None)):
            raise ValueError("Level jumping is only available for local playable games.")

        game._score = int(target)
        game.set_level(int(target))
        game._state = GameState.NOT_FINISHED

        obs = self._env._level_reset_observation()
        if obs is None:
            raise RuntimeError("Level jump produced no observation.")

        self._env._obs = obs
        self._env._current_level = int(target)
        self._env._actions_in_level = 0
        self._env._episode_reward = float(target) / float(max(1, win_levels))
        self._obs = obs
        self.last_reward = 0.0
        self.episode_reward = float(target) / float(max(1, win_levels))

    def _replay_history_locked(self: Any) -> Any:
        if self._env is None:
            self._create_env_locked()
            return

        self._full_reset_env_locked()
        if self._obs is None:
            raise RuntimeError("Failed to reset environment.")

        win_levels = _to_py_int(getattr(self._obs, "win_levels", None)) or 1
        base_level = max(0, min(int(self._history_base_level), int(win_levels) - 1))
        if base_level > 0:
            self._jump_to_level_locked(base_level, int(win_levels))

        self._level_start_history_len = 0
        self.last_reward = 0.0
        self.episode_reward = float(base_level) / float(max(1, win_levels))
        prev_levels = getattr(self._obs, "levels_completed", 0)

        for idx, record in enumerate(self._history):
            action_id = int(record.get("action_id"))
            data = record.get("data") or {}

            allowed_ids = {int(a.value) for a in self._env.action_space}
            if action_id not in allowed_ids and action_id != _action_id_value(GameAction.RESET):
                raise RuntimeError(
                    f"Action {action_id} not available during replay (available: {sorted(allowed_ids)})."
                )

            action = GameAction.from_id(action_id)
            obs, reward, _, _ = _dsl_unpack_step_result(self._env.step(action, data=data))
            if obs is None:
                raise RuntimeError(f"Step failed for action {action_id}.")
            self._obs = obs
            self.last_reward = float(reward)
            self.episode_reward += float(reward)

            new_levels = getattr(obs, "levels_completed", prev_levels)
            if new_levels is not None and prev_levels is not None and new_levels > prev_levels:
                self._level_start_history_len = idx + 1
            prev_levels = new_levels

    def _game_undo_action_id_locked(self: Any) -> int | None:
        if self._env is None:
            self._create_env_locked()
        allowed_ids = {int(a.value) for a in self._env.action_space}
        return 7 if 7 in allowed_ids else None

    def _game_undo_depth_locked(self: Any, undo_action_id: int) -> int:
        depth = 0
        reset_action_id = _action_id_value(GameAction.RESET)
        for record in self._history:
            action_id = int(record.get("action_id"))
            if action_id == undo_action_id:
                depth = max(0, depth - 1)
            elif action_id != reset_action_id:
                depth += 1
        return depth

    def _step_locked(self: Any, action_id: Any, data: Any) -> Any:
        if self._env is None:
            self._create_env_locked()

        action_id = int(action_id)
        allowed_ids = {int(a.value) for a in self._env.action_space}
        if action_id not in allowed_ids and action_id != _action_id_value(GameAction.RESET):
            raise ValueError(f"Action {action_id} is not available (available: {sorted(allowed_ids)}).")
        action = GameAction.from_id(action_id)

        payload_data = data or {}
        if action.is_complex():
            x = _to_py_int(payload_data.get("x"))
            y = _to_py_int(payload_data.get("y"))
            if x is None or y is None:
                raise ValueError("Complex actions require integer x and y.")
            payload_data = {"x": x, "y": y}
            action.validate_data(payload_data)
        else:
            payload_data = {}

        prev_levels = getattr(self._obs, "levels_completed", None) if self._obs is not None else None
        obs, reward, _, info = _dsl_unpack_step_result(self._env.step(action, data=payload_data))
        if obs is None:
            raise RuntimeError("Step failed.")
        self._obs = obs
        self.last_reward = float(reward)
        self.episode_reward += float(reward)
        reset_ignored = bool((info or {}).get("reset_ignored", False))
        if not reset_ignored:
            self._history.append({"action_id": action_id, "data": payload_data})
            self.steps = len(self._history)

        new_levels = getattr(obs, "levels_completed", None)
        if new_levels is not None and prev_levels is not None and new_levels > prev_levels:
            self._level_start_history_len = len(self._history)

        return self._state_locked()

    def undo(self: Any) -> Any:
        with self._lock:
            game_undo_action_id = self._game_undo_action_id_locked()
            if game_undo_action_id is not None:
                if self._game_undo_depth_locked(game_undo_action_id) > 0:
                    return self._step_locked(game_undo_action_id, {})
                return self._state_locked()
            if not self._history:
                return self._state_locked()
            self._history.pop()
            self.steps = len(self._history)
            self._replay_history_locked()
            return self._state_locked()

    def step(self: Any, action_id: Any, data: Any) -> Any:
        with self._lock:
            return self._step_locked(action_id, data)


class _WebHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self: Any, _fmt: Any, *_args: Any) -> Any:
        return

    def _send_bytes(self: Any, body: Any, status: Any, content_type: Any) -> Any:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self: Any, payload: Any, status: Any = 200) -> Any:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def _read_json(self: Any) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 1_000_000:
            raise ValueError("Request too large.")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self: Any) -> Any:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query or "")
        if parsed.path == "/":
            body = _web_index_html().encode("utf-8")
            self._send_bytes(body, status=200, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/actions":
            self._send_json({"actions": self.server.session.actions()})
            return
        if parsed.path == "/api/games":
            self._send_json(self.server.session.game_options())
            return
        if parsed.path == "/api/reload-status":
            self._send_json(self.server.session.reload_status())
            return
        if parsed.path == "/api/state":
            resample_random = str((query.get("resample_random") or ["0"])[0]).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self._send_json(self.server.session.state(resample_random=resample_random))
            return
        self._send_json({"error": "Not found."}, status=404)

    def do_POST(self: Any) -> Any:
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
        except Exception as e:
            self._send_json({"error": str(e)}, status=400)
            return

        try:
            if parsed.path == "/api/reset":
                seed = data.get("seed") if isinstance(data, dict) else None
                seed_int = _to_py_int(seed) if seed is not None else None
                self._send_json(self.server.session.restart(seed=seed_int))
                return
            if parsed.path == "/api/select-game":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                game_id = str(data.get("game_id") or "").strip()
                if not game_id:
                    raise ValueError("game_id is required.")
                seed = data.get("seed")
                seed_int = _to_py_int(seed) if seed is not None else None
                self._send_json(self.server.session.select_game(game_id=game_id, seed=seed_int))
                return
            if parsed.path == "/api/reset-level":
                self._send_json(self.server.session.reset_level())
                return
            if parsed.path == "/api/jump-level":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(self.server.session.jump_level(data.get("level_index")))
                return
            if parsed.path == "/api/undo":
                self._send_json(self.server.session.undo())
                return
            if parsed.path == "/api/reload":
                self._send_json(self.server.session.reload())
                return
            if parsed.path == "/api/recording":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(_print_recorded_actions(data.get("game_id"), data.get("actions"), data.get("levels")))
                return
            if parsed.path == "/api/use-recording":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(_use_recording_for_dsl(data.get("game_id"), data.get("actions"), data.get("levels")))
                return
            if parsed.path == "/api/step":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                action_id = _to_py_int(data.get("action_id"))
                if action_id is None:
                    action_index = _to_py_int(data.get("action_index"))
                    if action_index is None:
                        raise ValueError("action_id (or action_index) is required.")
                    match = next(
                        (
                            action
                            for action in self.server.session.actions()
                            if int(action.get("index")) == int(action_index)
                        ),
                        None,
                    )
                    if match is None:
                        raise ValueError("Invalid action_index.")
                    action_id = int(match.get("id"))
                self._send_json(self.server.session.step(action_id, data))
                return
        except Exception as e:
            self._send_json({"error": str(e)}, status=400)
            return

        self._send_json({"error": "Not found."}, status=404)


class _ReplaySession:
    def __init__(self: Any, replay_path: Any = None, replay_dir: Any = None, config: Any = None) -> None:
        requested_path = Path(replay_path) if replay_path else None
        requested_dir = Path(replay_dir) if replay_dir else None

        if requested_path is None and requested_dir is None:
            raise ValueError("Replay path or replay directory is required.")

        if requested_dir is None:
            if requested_path is not None:
                requested_dir = requested_path if requested_path.is_dir() else requested_path.parent
        if requested_dir is None or not requested_dir.exists():
            raise FileNotFoundError(f"Replay directory not found: {requested_dir}")

        self.dir = requested_dir
        self._config = dict(config or {})
        config_path_raw = str(self._config.get("__CONFIG_PATH__") or "").strip()
        self._repo_root = Path(config_path_raw).resolve().parent if config_path_raw else None
        self._replays: dict[str, Path] = {}
        self._play_sessions: dict[str, _WebSession] = {}
        for path in sorted(self.dir.glob("*.replay.json")):
            self._replays[path.name] = path
        if not self._replays:
            raise FileNotFoundError(f"No replay traces found in {self.dir}")

        self.selected_name = ""
        if requested_path is not None and requested_path.is_file():
            if requested_path.name not in self._replays:
                raise FileNotFoundError(f"Replay trace not found in directory index: {requested_path}")
            self.selected_name = requested_path.name
        if not self.selected_name:
            self.selected_name = next(iter(self._replays))
        self._review_dir = self.dir.parent
        self._review_manifest_path = self._review_dir / "manifest.json"
        self._review_enabled = self._review_manifest_path.exists()
        self._review_entries_by_name: dict[str, dict[str, Any]] = {}
        if self._review_enabled:
            try:
                manifest = _read_json_file(self._review_manifest_path)
            except Exception:
                manifest = {}
            raw_entries = manifest.get("entries") if isinstance(manifest, dict) else None
            entries = (
                [entry for entry in raw_entries if isinstance(entry, dict)] if isinstance(raw_entries, list) else []
            )
            for entry in entries:
                review_replay_path = str(entry.get("review_replay_path") or "").strip()
                if not review_replay_path:
                    continue
                self._review_entries_by_name[Path(review_replay_path).name] = entry

    def _entry_for_replay(self: Any, name: Any) -> dict[str, Any]:
        replay_name = str(name or "").strip()
        if replay_name not in self._replays:
            raise ValueError(f"Unknown replay: {replay_name}")
        entry = self._review_entries_by_name.get(replay_name)
        if not entry:
            raise ValueError(f"No review entry found for replay: {replay_name}")
        return entry

    def _play_url_for_replay(self: Any, name: Any) -> str:
        replay_name = str(name or "").strip()
        if replay_name not in self._replays:
            return ""
        entry = self._review_entries_by_name.get(replay_name) or {}
        worker_repo_path = (
            Path(str(entry.get("worker_repo_path") or "")).resolve()
            if entry.get("worker_repo_path")
            else Path(str(entry.get("worker_root") or "")).resolve() / "repo"
        )
        environments_dir = worker_repo_path / "re_arc" / "environment_files" if worker_repo_path is not None else None
        game_id = str(entry.get("game_id") or "").strip()
        if not game_id or environments_dir is None or not environments_dir.exists():
            return ""
        return f"/play?name={quote(replay_name)}"

    def _play_session_for_replay(self: Any, name: Any) -> _WebSession:
        replay_name = str(name or "").strip()
        if replay_name in self._play_sessions:
            return self._play_sessions[replay_name]
        entry = self._entry_for_replay(replay_name)
        worker_repo_path = (
            Path(str(entry.get("worker_repo_path") or "")).resolve()
            if entry.get("worker_repo_path")
            else Path(str(entry.get("worker_root") or "")) / "repo"
        )
        _register_worker_repo_packages(worker_repo_path)
        environments_dir = worker_repo_path / "re_arc" / "environment_files"
        if not environments_dir.exists():
            raise FileNotFoundError(f"Playable environments directory not found: {environments_dir}")
        game_id = str(entry.get("game_id") or "").strip()
        if not game_id:
            raise ValueError(f"Missing game_id for replay: {replay_name}")
        config = dict(self._config)
        config["ENVIRONMENTS_DIR"] = str(environments_dir)
        config.setdefault("OPERATION_MODE", "OFFLINE")
        session = _WebSession(config, game_id, seed=None)
        self._play_sessions[replay_name] = session
        return session

    def _review_payload(self: Any) -> dict[str, Any]:
        if not self._review_enabled:
            return {"enabled": False}
        from pipeline.review import load_review_decisions

        decisions = load_review_decisions(review_dir=self._review_dir)
        raw_entries = decisions.get("entries") if isinstance(decisions, dict) else None
        entries = [entry for entry in raw_entries if isinstance(entry, dict)] if isinstance(raw_entries, list) else []
        counts: dict[str, int] = {}
        decisions_by_name: dict[str, str] = {}
        for entry in entries:
            decision = str(entry.get("decision") or "pending")
            counts[decision] = counts.get(decision, 0) + 1
            replay_name = str(entry.get("review_replay_name") or "")
            if replay_name:
                decisions_by_name[replay_name] = decision
        return {
            "enabled": True,
            "submitted": bool(decisions.get("submitted")),
            "counts": counts,
            "decisions_by_name": decisions_by_name,
            "all_decided": counts.get("pending", 0) == 0,
        }

    def set_review_decision(self: Any, name: Any, decision: Any) -> dict[str, Any]:
        if not self._review_enabled:
            raise ValueError("Review decisions are not available for this replay directory.")
        replay_name = str(name or "").strip()
        if replay_name not in self._replays:
            raise ValueError(f"Unknown replay: {replay_name}")
        from pipeline.review import record_review_decision

        record_review_decision(
            review_dir=self._review_dir, review_replay_name=replay_name, decision=str(decision or "")
        )
        return self._review_payload()

    def submit_review(self: Any) -> dict[str, Any]:
        if not self._review_enabled:
            raise ValueError("Review submission is not available for this replay directory.")
        from pipeline.review import submit_review

        submission = submit_review(review_dir=self._review_dir, root=self._repo_root)
        payload = self._review_payload()
        payload["submission"] = submission
        return payload

    def replay_options(self: Any) -> dict[str, Any]:
        review = self._review_payload()
        items = []
        for name, path in self._replays.items():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                meta = payload.get("metadata") or {}
            except Exception:
                meta = {}
            label = str(meta.get("game_id") or name)
            decision = str(review.get("decisions_by_name", {}).get(name) or "")
            review_entry = self._review_entries_by_name.get(name) or {}
            items.append(
                {
                    "name": name,
                    "label": label,
                    "decision": decision,
                    "idea_title": str(review_entry.get("idea_title") or ""),
                }
            )
        return {"replays": items, "selected_replay": self.selected_name, "review": review}

    def payload(self: Any, name: Any = None) -> dict[str, Any]:
        selected = str(name or self.selected_name).strip()
        if selected not in self._replays:
            raise ValueError(f"Unknown replay: {selected}")
        self.selected_name = selected
        payload = json.loads(self._replays[selected].read_text(encoding="utf-8"))
        payload["selected_replay"] = selected
        review = self._review_payload()
        payload["review"] = {
            "enabled": bool(review.get("enabled")),
            "decision": str(review.get("decisions_by_name", {}).get(selected) or "pending"),
            "submitted": bool(review.get("submitted")),
            "all_decided": bool(review.get("all_decided")),
        }
        review_entry = self._review_entries_by_name.get(selected) or {}
        payload["review_context"] = {
            "idea_id": str(review_entry.get("idea_id") or ""),
            "idea_title": str(review_entry.get("idea_title") or ""),
            "idea_description": str(review_entry.get("idea_description") or ""),
        }
        payload["play_url"] = self._play_url_for_replay(selected)
        return payload


class _ReplayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self: Any, _fmt: Any, *_args: Any) -> Any:
        return

    def _send_bytes(self: Any, body: Any, status: Any, content_type: Any) -> Any:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self: Any, payload: Any, status: Any = 200) -> Any:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def do_GET(self: Any) -> Any:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query or "")
        if parsed.path == "/play":
            body = _web_index_html(api_base="/play/api").encode("utf-8")
            self._send_bytes(body, status=200, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/":
            body = _replay_index_html().encode("utf-8")
            self._send_bytes(body, status=200, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/play/api/actions":
            name = str((query.get("name") or [""])[0]).strip()
            self._send_json({"actions": self.server.session._play_session_for_replay(name).actions()})
            return
        if parsed.path == "/play/api/games":
            name = str((query.get("name") or [""])[0]).strip()
            self._send_json(self.server.session._play_session_for_replay(name).game_options())
            return
        if parsed.path == "/play/api/reload-status":
            name = str((query.get("name") or [""])[0]).strip()
            self._send_json(self.server.session._play_session_for_replay(name).reload_status())
            return
        if parsed.path == "/play/api/state":
            name = str((query.get("name") or [""])[0]).strip()
            resample_random = str((query.get("resample_random") or ["0"])[0]).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self._send_json(self.server.session._play_session_for_replay(name).state(resample_random=resample_random))
            return
        if parsed.path == "/api/replays":
            self._send_json(self.server.session.replay_options())
            return
        if parsed.path == "/api/replay":
            replay_name: str | None = str((query.get("name") or [""])[0]).strip() or None
            self._send_json(self.server.session.payload(replay_name))
            return
        if parsed.path == "/api/review":
            self._send_json(self.server.session._review_payload())
            return
        self._send_json({"error": "Not found."}, status=404)

    def _read_json(self: Any) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 1_000_000:
            raise ValueError("Request too large.")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_POST(self: Any) -> Any:
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
        except Exception as e:
            self._send_json({"error": str(e)}, status=400)
            return
        try:
            if parsed.path == "/play/api/reset":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                seed = data.get("seed") if isinstance(data, dict) else None
                seed_int = _to_py_int(seed) if seed is not None else None
                self._send_json(self.server.session._play_session_for_replay(name).restart(seed=seed_int))
                return
            if parsed.path == "/play/api/select-game":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                game_id = str(data.get("game_id") or "").strip()
                if not game_id:
                    raise ValueError("game_id is required.")
                seed = data.get("seed")
                seed_int = _to_py_int(seed) if seed is not None else None
                self._send_json(
                    self.server.session._play_session_for_replay(name).select_game(game_id=game_id, seed=seed_int)
                )
                return
            if parsed.path == "/play/api/reset-level":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                self._send_json(self.server.session._play_session_for_replay(name).reset_level())
                return
            if parsed.path == "/play/api/jump-level":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(self.server.session._play_session_for_replay(name).jump_level(data.get("level_index")))
                return
            if parsed.path == "/play/api/undo":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                self._send_json(self.server.session._play_session_for_replay(name).undo())
                return
            if parsed.path == "/play/api/reload":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                self._send_json(self.server.session._play_session_for_replay(name).reload())
                return
            if parsed.path == "/play/api/recording":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(_print_recorded_actions(data.get("game_id"), data.get("actions"), data.get("levels")))
                return
            if parsed.path == "/play/api/use-recording":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                self._send_json(_use_recording_for_dsl(data.get("game_id"), data.get("actions"), data.get("levels")))
                return
            if parsed.path == "/play/api/step":
                name = str((parse_qs(parsed.query or "").get("name") or [""])[0]).strip()
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                play_session = self.server.session._play_session_for_replay(name)
                action_id = _to_py_int(data.get("action_id"))
                if action_id is None:
                    action_index = _to_py_int(data.get("action_index"))
                    if action_index is None:
                        raise ValueError("action_id (or action_index) is required.")
                    match = next(
                        (action for action in play_session.actions() if int(action.get("index")) == int(action_index)),
                        None,
                    )
                    if match is None:
                        raise ValueError("Invalid action_index.")
                    action_id = int(match.get("id"))
                self._send_json(play_session.step(action_id, data))
                return
            if parsed.path == "/api/review/decision":
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object.")
                replay_name = str(data.get("name") or "").strip()
                decision = str(data.get("decision") or "").strip()
                self._send_json(self.server.session.set_review_decision(replay_name, decision))
                return
            if parsed.path == "/api/review/submit":
                self._send_json(self.server.session.submit_review())
                return
        except Exception as e:
            self._send_json({"error": str(e)}, status=400)
            return
        self._send_json({"error": "Not found."}, status=404)


def _run_web_ui(args: Any) -> Any:
    config = _load_runtime_config(args)
    game_id = _resolve_game_id(args, config)

    seed = args.seed if args.seed is not None else _cfg_int(config, "SEED", None)
    host = args.host or (config.get("WEB_HOST") or "").strip() or "127.0.0.1"
    port = args.port if args.port is not None else _cfg_int(config, "WEB_PORT", 8000)

    session = _WebSession(config, game_id, seed)
    server = _start_http_server(host=host, port=port, handler=_WebHandler, server_name="Web UI")
    cast(Any, server).session = session

    bound_port = int(server.server_address[1])
    url_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    print(f"Web UI running at http://{url_host}:{bound_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _run_replay_ui(args: Any) -> Any:
    config = _load_runtime_config(args)
    replay_path = args.replay or config.get("REPLAY_PATH")
    replay_dir = args.replay_dir if args.replay_dir is not None else config.get("REPLAY_DIR")
    if not replay_path and not replay_dir:
        raise ValueError(
            "Replay path or replay directory is required via --replay, --replay-dir, REPLAY_PATH, or REPLAY_DIR."
        )
    session = _ReplaySession(replay_path=replay_path, replay_dir=replay_dir, config=config)
    host = args.host or (config.get("WEB_HOST") or "").strip() or "127.0.0.1"
    port = args.port if args.port is not None else _cfg_int(config, "WEB_PORT", 8000)
    server = _start_http_server(host=host, port=port, handler=_ReplayHandler, server_name="Replay UI")
    cast(Any, server).session = session

    bound_port = int(server.server_address[1])
    url_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    print(f"Replay UI running at http://{url_host}:{bound_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _start_http_server(
    *, host: str, port: int, handler: type[http.server.BaseHTTPRequestHandler], server_name: str
) -> http.server.ThreadingHTTPServer:
    try:
        return http.server.ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            url_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
            raise RuntimeError(
                f"{server_name} could not bind to http://{url_host}:{port}/ because that address is already in use. "
                "Stop the existing process or choose a different port with --port."
            ) from exc
        raise


def _run_random_policy(args: Any) -> Any:
    config = _load_runtime_config(args)

    game_id = _resolve_game_id(args, config)

    max_actions = args.max_actions if args.max_actions is not None else _cfg_int(config, "MAX_ACTIONS", 80)
    seed = args.seed if args.seed is not None else _cfg_int(config, "SEED", None)
    gif_path = args.gif if args.gif is not None else config.get("GIF_PATH")
    fps = args.fps if args.fps is not None else _cfg_int(config, "FPS", 12)
    scale = args.scale if args.scale is not None else _cfg_int(config, "SCALE", 6)
    separator = args.separator if args.separator is not None else _cfg_int(config, "SEPARATOR", 2)

    frames = []
    renderer_used = False

    def renderer(_steps: Any, frame_data: Any) -> Any:
        nonlocal renderer_used
        renderer_used = True
        frames.append(frame_data)

    rng = random.Random(seed)

    env, obs, selected_game_id = _create_env_and_initial_observation(
        config=config, game_id=game_id, seed=seed, renderer=renderer if gif_path else None
    )
    if gif_path and not renderer_used and obs is not None:
        frames.append(obs)

    actions = 0
    episode_reward = 0.0
    for _ in range(max_actions):
        selectable_actions = list(env.action_space)
        if not selectable_actions:
            print("No actions available; stopping.")
            break
        action = rng.choice(selectable_actions)
        action_data = {}
        if action.is_complex():
            action_data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
        obs, reward, done, _ = _dsl_unpack_step_result(env.step(action, data=action_data))
        actions += 1
        episode_reward += reward
        if gif_path and not renderer_used and obs is not None:
            frames.append(obs)
        if done:
            break

    score = getattr(obs, "score", "NA") if obs else "NA"
    print(
        f"game={selected_game_id} actions={actions} "
        f"state={obs.state.name if obs else 'UNKNOWN'} score={score} "
        f"reward={episode_reward:.6f}"
    )

    if gif_path:
        _write_gif(frames, gif_path, fps, scale, separator)

    return 0


def _run_dsl_policy(args: Any) -> Any:
    config = _load_runtime_config(args)
    game_id = _resolve_game_id(args, config)
    if _is_random_game_id(game_id):
        raise ValueError("`GAME_ID=random` is not supported with `--policy dsl`. Pick a specific game id for DSL runs.")

    max_actions = args.max_actions if args.max_actions is not None else _cfg_int(config, "MAX_ACTIONS", 400)
    seed = args.seed if args.seed is not None else _cfg_int(config, "SEED", None)
    gif_path = args.gif if args.gif is not None else config.get("GIF_PATH")
    replay_path = args.replay if args.replay is not None else (_replay_path_for_gif(gif_path) if gif_path else None)
    fps = args.fps if args.fps is not None else _cfg_int(config, "FPS", 12)
    scale = args.scale if args.scale is not None else _cfg_int(config, "SCALE", 6)
    separator = args.separator if args.separator is not None else _cfg_int(config, "SEPARATOR", 2)

    frames = []
    renderer_used = False

    def renderer(_steps: Any, frame_data: Any) -> Any:
        nonlocal renderer_used
        renderer_used = True
        frames.append(frame_data)

    env, initial_observation, selected_game_id = _create_env_and_initial_observation(
        config=config, game_id=game_id, seed=seed, renderer=renderer if gif_path else None
    )
    agent = create_dsl_agent(selected_game_id)
    replay_steps: list[dict[str, Any]] = []

    def _capture_observation(observation: Any) -> Any:
        if gif_path and not renderer_used and observation is not None:
            frames.append(observation)

    def _record_step(record: Any) -> Any:
        observation = record.get("observation")
        info = record.get("info") or {}
        replay_steps.append(
            {
                "index": int(record.get("index", len(replay_steps) + 1)),
                "action": {
                    "id": int(record.get("action_id", -1)),
                    "name": str(record.get("action_name") or _action_name_from_id(record.get("action_id"))),
                    "data": _json_safe(record.get("action_data") or {}),
                },
                "reward": float(record.get("reward", 0.0)),
                "episode_reward": float(record.get("episode_reward", 0.0)),
                "info": {"level_transition": bool(info.get("level_transition", False))},
                "observation": _serialize_replay_observation(observation),
            }
        )

    result = run_dsl_episode(
        env=env,
        agent=agent,
        max_actions=max_actions,
        initial_observation=initial_observation,
        observation_callback=_capture_observation,
        step_callback=_record_step,
    )
    obs = result.get("observation")
    actions = int(result.get("actions", 0))
    normalized_score = float(result.get("normalized_score", 0.0))
    episode_reward = float(result.get("episode_reward", 0.0))
    solved_levels = result.get("solved_levels")
    total_levels = result.get("total_levels")

    score = getattr(obs, "score", "NA") if obs else "NA"
    normalized_text = f"{normalized_score:.3f}"
    if total_levels:
        normalized_text = f"{normalized_text} ({solved_levels}/{total_levels})"
    print(
        f"game={selected_game_id} actions={actions} "
        f"state={obs.state.name if obs else 'UNKNOWN'} score={score} "
        f"normalized={normalized_text} reward={episode_reward:.6f}"
    )

    if gif_path and frames:
        _write_gif(frames, gif_path, fps, scale, separator)
    if replay_path is not None:
        trace = {
            "version": 2,
            "metadata": {
                "game_id": selected_game_id,
                "seed": seed,
                "policy": "dsl",
                "actions": actions,
                "normalized_score": normalized_score,
                "episode_reward": episode_reward,
                "solved_levels": solved_levels,
                "total_levels": total_levels,
                "gif_path": str(gif_path) if gif_path else None,
            },
            "initial_observation": _serialize_replay_observation(initial_observation),
            "steps": replay_steps,
        }
        _write_replay_trace(trace, replay_path)

    return 0


def _frame_layers(frame_data: Any) -> Any:
    if frame_data is None:
        return []
    grid = getattr(frame_data, "frame", None)
    if grid is None:
        return []
    if isinstance(grid, list):
        return [layer for layer in grid if layer is not None]
    return [grid]


def _layer_dims(layer: Any) -> Any:
    if hasattr(layer, "shape") and len(layer.shape) >= 2:
        return int(layer.shape[0]), int(layer.shape[1])
    height = len(layer)
    width = len(layer[0]) if height else 0
    return height, width


def _cell_to_char(value: Any) -> Any:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "?"
    if v == 0:
        return "."
    if 0 < v < 16:
        return "0123456789ABCDEF"[v]
    return "#"


def _print_frame(frame_data: Any, max_width: Any = 128) -> Any:
    layers = _frame_layers(frame_data)
    if not layers:
        print("(no frame)")
        return

    for idx, layer in enumerate(layers):
        height, width = _layer_dims(layer)
        if width > max_width:
            print(f"(layer {idx}: width {width} > {max_width}; not printing)")
            continue
        if len(layers) > 1:
            print(f"Layer {idx}:")
        for y in range(height):
            row = layer[y]
            print("".join(_cell_to_char(row[x]) for x in range(width)))
        if idx != len(layers) - 1:
            print()


def _action_label(action: Any) -> Any:
    name = getattr(action, "name", None)
    return name if name else str(action)


def _print_actions(action_space: Any) -> Any:
    for idx, action in enumerate(action_space):
        suffix = " (x,y)" if getattr(action, "is_complex", lambda: False)() else ""
        print(f"{idx:>2}: {_action_label(action)}{suffix}")


def _parse_int(value: Any) -> Any:
    return int(value.strip())


def _prompt_action(env: Any) -> Any:
    action_space = env.action_space
    while True:
        raw = input("action (? for list, q to quit)> ").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            return None, None
        if lowered in {"?", "h", "help"}:
            _print_actions(action_space)
            continue

        parts = raw.replace(",", " ").split()
        selector = parts[0]
        extra = parts[1:]

        action = None
        if selector.lstrip("-").isdigit():
            idx = int(selector)
            if not (0 <= idx < len(action_space)):
                print(f"Invalid action index: {idx} (0..{len(action_space) - 1})")
                continue
            action = action_space[idx]
        else:
            needle = selector.strip().upper()
            matches = [a for a in action_space if _action_label(a).strip().upper() == needle]
            if not matches:
                matches = [a for a in action_space if needle in _action_label(a).strip().upper()]
            if len(matches) == 1:
                action = matches[0]
            elif len(matches) > 1:
                print("Ambiguous action; use an index or a more specific name.")
                continue
            else:
                print("Unknown action; use ? to list actions.")
                continue

        data = {}
        if action.is_complex():
            if len(extra) >= 2:
                try:
                    x = _parse_int(extra[0])
                    y = _parse_int(extra[1])
                except ValueError:
                    print("Invalid coordinates; expected: x y")
                    continue
            else:
                coords = input("coords (x y)> ").strip().replace(",", " ").split()
                if len(coords) != 2:
                    print("Invalid coordinates; expected: x y")
                    continue
                try:
                    x = _parse_int(coords[0])
                    y = _parse_int(coords[1])
                except ValueError:
                    print("Invalid coordinates; expected integers.")
                    continue
            data = {"x": x, "y": y}

        return action, data


def _run_human_policy(args: Any) -> Any:
    config = _load_runtime_config(args)
    game_id = _resolve_game_id(args, config)

    max_actions = args.max_actions if args.max_actions is not None else _cfg_int(config, "MAX_ACTIONS", 400)
    seed = args.seed if args.seed is not None else _cfg_int(config, "SEED", None)
    gif_path = args.gif if args.gif is not None else config.get("GIF_PATH")
    fps = args.fps if args.fps is not None else _cfg_int(config, "FPS", 12)
    scale = args.scale if args.scale is not None else _cfg_int(config, "SCALE", 6)
    separator = args.separator if args.separator is not None else _cfg_int(config, "SEPARATOR", 2)

    frames = []
    renderer_used = False

    def renderer(_steps: Any, frame_data: Any) -> Any:
        nonlocal renderer_used
        renderer_used = True
        frames.append(frame_data)

    env, obs, selected_game_id = _create_env_and_initial_observation(
        config=config, game_id=game_id, seed=seed, renderer=renderer if gif_path else None
    )
    if gif_path and not renderer_used and obs is not None:
        frames.append(obs)

    print(f"game={selected_game_id} seed={seed}")
    _print_actions(env.action_space)
    print()

    def _print_status(observation: Any, step_reward: Any = None, total_reward: Any = None) -> Any:
        state = getattr(observation, "state", None)
        state_name = getattr(state, "name", None) or str(state)
        score = getattr(observation, "score", "NA")
        reward_text = ""
        if step_reward is not None and total_reward is not None:
            reward_text = f" reward={step_reward:.6f} total_reward={total_reward:.6f}"
        print(f"state={state_name} score={score}{reward_text}")

    if obs is not None:
        _print_status(obs)
        _print_frame(obs)

    actions = 0
    episode_reward = 0.0
    try:
        for _ in range(max_actions):
            if _dsl_is_terminal(obs):
                break

            action, data = _prompt_action(env)
            if action is None:
                break

            obs, reward, done, _ = _dsl_unpack_step_result(env.step(action, data=data or {}))
            actions += 1
            episode_reward += reward
            if gif_path and not renderer_used and obs is not None:
                frames.append(obs)

            print()
            print(f"action={_action_label(action)} data={data or {}}")
            if obs is not None:
                _print_status(obs, step_reward=reward, total_reward=episode_reward)
                _print_frame(obs)
            print()
            if done:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")

    final_score = getattr(obs, "score", "NA") if obs else "NA"
    final_state = obs.state.name if obs else "UNKNOWN"
    print(
        f"game={selected_game_id} actions={actions} state={final_state} score={final_score} reward={episode_reward:.6f}"
    )

    if gif_path and frames:
        _write_gif(frames, gif_path, fps, scale, separator)

    return 0


def main() -> Any:
    parser = argparse.ArgumentParser(description="Run an ARC-AGI-3 game.")
    parser.add_argument("--config", default="config.env", help="Path to config file.")
    parser.add_argument("--game", help="Game id to play.")
    parser.add_argument("--list-games", action="store_true", help="List available game ids and exit.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Named dataset, dataset tag, or dataset JSON path to sample/list. Repeatable.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Named dataset, dataset tag, or dataset JSON path to remove from sampling/listing. Repeatable.",
    )
    parser.add_argument("--datasets-dir", default=None, help="Directory containing dataset JSON files.")
    parser.add_argument(
        "--policy",
        choices=("random", "human", "web", "dsl", "replay"),
        default="random",
        help="Policy to use: random, human, web, dsl, or replay.",
    )
    parser.add_argument("--host", default=None, help="Web UI host (used with --policy web or replay).")
    parser.add_argument("--port", type=int, default=None, help="Web UI port (used with --policy web or replay).")
    parser.add_argument("--replay", type=str, default=None, help="Replay trace path (used with --policy replay).")
    parser.add_argument(
        "--replay-dir", type=str, default=None, help="Replay directory to browse (used with --policy replay)."
    )
    parser.add_argument("--max-actions", type=int, default=None, help="Max actions.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--gif", type=str, default=None, help="Write frames to a GIF file.")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for GIF.")
    parser.add_argument("--scale", type=int, default=None, help="Scale factor for GIF.")
    parser.add_argument("--separator", type=int, default=None, help="Separator width.")
    args = parser.parse_args()
    if args.list_games:
        config = _load_runtime_config(args)
        try:
            game_ids = _list_game_ids(config)
        except Exception as e:
            print(f"Failed to list games: {e}")
            return 1
        if not game_ids:
            print(
                "No game ids discovered.\n"
                "- For offline/local discovery: set ENVIRONMENTS_DIR to a "
                "folder containing `**/metadata.json` and set OPERATION_MODE=OFFLINE.\n"
                "- For API discovery: set ARC_API_KEY (or ensure network access)."
            )
            return 1
        for gid in game_ids:
            print(gid)
        return 0
    if args.policy == "web":
        return _run_web_ui(args)
    if args.policy == "replay":
        return _run_replay_ui(args)
    if args.policy == "human":
        return _run_human_policy(args)
    if args.policy == "dsl":
        return _run_dsl_policy(args)
    return _run_random_policy(args)


if __name__ == "__main__":
    raise SystemExit(main())
