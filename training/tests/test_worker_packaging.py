"""Standalone workers must reach slot validation using only installed packages."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


@pytest.fixture(scope="module")
def installed_worker(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    staging = tmp_path_factory.mktemp("worker-wheel")
    source = staging / "source"
    source.mkdir()
    for package in ("training", "harness_support", "sft", "scripts"):
        shutil.copytree(
            root / package,
            source / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(root / name, source / name)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    installed = staging / "site-packages"
    with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as wheel:
        # This project has a pure Python wheel: unpack its installed layout
        # without pip, dependency downloads, or the test runner's editable install.
        assert not any(name.startswith("scripts/") for name in wheel.namelist())
        wheel.extractall(installed)

    scripts = staging / "launch" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(
        root / "scripts/platypus_rollout_worker.py",
        scripts / "platypus_rollout_worker.py",
    )
    # Only external harness/engine imports and their bridge are stand-ins. The
    # real installed adapters and shared helpers must validate the spec before
    # any model, server, game, or private dataset is needed.
    dependencies = staging / "dependencies"
    for name, body in {
        "platypus.py": "RunConfig = object\n",
        "serving_contract.py": (
            "ContractDuckAgent = ContractLLMAgentSolver = object\n"
            "TrainingContextObserver = object\n"
            "install_contract_backend = object\n"
        ),
        "training_backend.py": (
            "TrainingLLMAgentSolver = object\n"
            "configure_training_agent = object\n"
            "qwen_engine_factory = training_backend_factory = training_sampling = object\n"
            "prompt_injection_settings = object\n"
        ),
        "duck/agent.py": "DuckAgent = object\n",
        "ullm/backends/engine_proxy.py": "EngineProxy = object\n",
        "taaf/benchmark.py": "Benchmark = object\n",
        "taaf/game_api.py": "GameAPI = object\n",
        "taaf/game.py": "ArcGame = object\n",
        "taaf/datasets.py": (
            "from types import SimpleNamespace\n"
            "def load(name):\n"
            "    return [SimpleNamespace(game_id='puzzle', to_answer=[None])]\n"
        ),
        "numpy.py": "# Dataset introspection is not used by these worker specs.\n",
    }.items():
        path = dependencies / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, (installed, dependencies))),
    }
    return scripts / "platypus_rollout_worker.py", env


@pytest.mark.parametrize("environment", ["arc2", "arc3"])
def test_installed_worker_reaches_slot_validation(
    installed_worker, tmp_path, environment
):
    worker, env = installed_worker
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "environment": environment,
                "dataset": "fixture",
                "games": ["puzzle"] * 2,
                "slot_seeds": [11],
                "rng_seed": 0,
                "max_concurrent": 2,
                "max_runtime_s_per_game": None,
            }
        )
    )
    result = subprocess.run(
        # -S excludes editable installs and sitecustomize; only the wheel and
        # explicit dependency fixtures above can supply application packages.
        [sys.executable, "-S", str(worker), "--spec", str(spec)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ValueError: slot_seeds lists 1 seeds for 2 game slots" in result.stderr, (
        result.stderr
    )


def test_installed_arc2_results_preserve_independent_slot_seeds(
    installed_worker, tmp_path
):
    _, env = installed_worker
    program = """
import importlib.util
assert importlib.util.find_spec('scripts') is None
from pathlib import Path
from harness_support.agent_settings import record_agent_settings
from harness_support.arc2.worker import enrich_results
injection = {'inject_text_on_first_prompt': 'We need solve',
             'inject_text_on_later_prompts': 'We need', 'prompt_injection_schedule': 'game'}
record_agent_settings(Path('.'), injection)
spec = dict(dataset='fixture', games=['puzzle'] * 2, slot_seeds=[11, 12],
            job_dir='.', max_runtime_s_per_game=None, malformed_call_penalty=0.05,
            max_episode_penalty=0.5,
            behavior_policy_snapshot='policy', serving_adapter='adapter')
runs = [dict(game_id='puzzle', run_id=f'run{i}', state='won', steps=[]) for i in range(2)]
originals = [dict(game_id=f'puzzle#s{i}', finished=True, final_score=100,
                  result={'pairs': [{'correct': True}]}) for i in range(2)]
enrich_results(spec, runs, originals)
assert [run['rollout_seed'] for run in runs] == [11, 12]
assert [run['rollout_slot_id'] for run in runs] == ['puzzle#s0:run0', 'puzzle#s1:run1']
assert [run['episode_return'] for run in runs] == [1.0, 1.0]
assert [run['agent_settings'] for run in runs] == [injection] * 2
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
