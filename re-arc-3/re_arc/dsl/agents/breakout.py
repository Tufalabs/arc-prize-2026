from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_breakout_mod = import_module("re_arc.environment_files.breakout.0001.breakout")
A_LEFT = int(_breakout_mod.A_LEFT)
A_RIGHT = int(_breakout_mod.A_RIGHT)
A_SPACE = int(_breakout_mod.A_SPACE)
BreakoutModel = _breakout_mod.BreakoutModel

_MODEL_ACTION_REPEAT = 1


def _move_toward(current: int, target: int) -> int:
    if current < target:
        return A_RIGHT
    if current > target:
        return A_LEFT
    return A_SPACE


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _target_brick_x(model: BreakoutModel) -> int:
    candidates = [brick for brick in model.bricks.values() if (not brick.removed) and brick.kind != "unbreakable"]
    if not candidates:
        return model.paddle_center
    candidates.sort(key=lambda brick: (brick.y, abs((brick.x + 1) - model.ball_x), brick.x))
    target = candidates[0]
    return int(target.x + 1)


def _choose_action(model: BreakoutModel, dock_launch_bias: int = 0) -> int:
    if model.ball_docked:
        target_x = _target_brick_x(model)
        center = model.paddle_center
        if abs(center - target_x) > 1:
            return _move_toward(center, target_x)
        if dock_launch_bias < 0 and model.paddle_x > 1:
            return A_LEFT
        if dock_launch_bias > 0 and model.paddle_x + model.paddle_width < model.width - 1:
            return A_RIGHT
        return A_SPACE

    if model.ball_dy > 0:
        target_x = _target_brick_x(model)
        desired_dx = _sign(target_x - model.ball_x)
        desired_center = int(model.ball_x - desired_dx)
        min_center = 1 + (model.paddle_width // 2)
        max_center = (model.width - 2) - ((model.paddle_width - 1) // 2)
        desired_center = max(min_center, min(max_center, desired_center))
        return _move_toward(model.paddle_center, desired_center)

    target_x = _target_brick_x(model)
    return _move_toward(model.paddle_center, target_x)


def _step_model(model: BreakoutModel, action_id: int) -> None:
    for _ in range(_MODEL_ACTION_REPEAT):
        if model.is_win or model.is_fail:
            break
        model.step(int(action_id))


class BreakoutDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        runtime_model = getattr(getattr(env, "_game", None), "_model", None)
        clone_fn = getattr(runtime_model, "clone", None)
        if callable(clone_fn):
            sim = clone_fn()
        else:
            level_data = dict(getattr(env._game.current_level, "_data", {}) or {})
            sim = BreakoutModel.from_level_data(level_data)
        program: list[tuple[int, dict[str, int]]] = []

        no_progress_steps = 0
        prior_breakables = sim.breakable_remaining
        launch_bias = -1

        def evaluate_action(candidate_action: int) -> float:
            trial = sim.clone()
            start_breakables = trial.breakable_remaining
            start_lives = trial.lives
            start_time = trial.time_left
            _step_model(trial, int(candidate_action))
            if trial.is_fail:
                return -1_000_000.0
            if trial.is_win:
                return 10_000_000.0

            local_bias = launch_bias
            for _ in range(72):
                if trial.is_fail:
                    break
                if trial.is_win:
                    break
                aid = _choose_action(trial, local_bias)
                was_docked = trial.ball_docked
                _step_model(trial, int(aid))
                if was_docked and not trial.ball_docked:
                    local_bias *= -1

            destroyed = start_breakables - trial.breakable_remaining
            life_delta = trial.lives - start_lives
            time_delta = trial.time_left - start_time
            center_bias = -abs(trial.paddle_center - _target_brick_x(trial))
            score = (
                float(destroyed) * 1500.0
                + float(life_delta) * 2000.0
                + float(time_delta) * 1.5
                + float(center_bias) * 0.5
            )
            if trial.is_fail:
                score -= 200_000.0
            if trial.is_win:
                score += 5_000_000.0
            return score

        for _ in range(900):
            if sim.is_win:
                break
            if sim.is_fail:
                break

            candidate_actions = (A_LEFT, A_RIGHT, A_SPACE)
            best_action = A_SPACE
            best_score = float("-inf")
            for action_id in candidate_actions:
                score = evaluate_action(action_id)
                if score > best_score:
                    best_score = score
                    best_action = int(action_id)

            action_id = int(best_action)
            program.append((int(action_id), {}))
            was_docked = sim.ball_docked
            _step_model(sim, int(action_id))

            if sim.breakable_remaining < prior_breakables:
                no_progress_steps = 0
                prior_breakables = sim.breakable_remaining
            else:
                no_progress_steps += 1

            if was_docked and not sim.ball_docked:
                launch_bias *= -1

            if no_progress_steps > 180:
                launch_bias *= -1
                no_progress_steps = 0

        if not sim.is_win and not sim.is_fail:
            for _ in range(4100):
                if sim.is_win or sim.is_fail:
                    break
                aid = int(_choose_action(sim, launch_bias))
                program.append((aid, {}))
                was_docked = sim.ball_docked
                _step_model(sim, aid)
                if was_docked and not sim.ball_docked:
                    launch_bias *= -1
        if not program:
            program = [(A_SPACE, {})]
        return program


AGENT_CLASS = BreakoutDslAgent
