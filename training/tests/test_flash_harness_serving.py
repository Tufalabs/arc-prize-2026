"""Every Flash serving setting of the harness deployment, mirrored or documented.

The harness's Flash engine on slurm-b200 is ullm's (platypus submodule,
``backends/deployment.py``: ``_flashnext_config``, ``build()``'s universal and
slurm-b200 layers and the Flash cell, with the helpers it calls) plus what the
ARC-2 Duck's ``presets.configure`` adds for its Flash column (arc2-platypus
submodule). Both are read here by AST. Each setting is either MIRRORED (the
launcher serves the same value for the recipe's job shape) or DIFFERENT (a
deliberate difference that training/FLASH_RL.md's Serving section states with
its reason). A new harness setting, or a changed harness value, fails until
someone classifies it.
"""

import ast
import json
import os
import subprocess
import sys


from flash_configs import RECIPE, ROOT
from training.config import ExperimentConfig
from training.flash_drafting import FLASH_E307_DRAFTER, HARNESS_SPECULATIVE
from training.job_shape import job_shape

DEPLOYMENT = ROOT / "platypus/packages/ullm/src/ullm/backends/deployment.py"
PRESETS = ROOT / "arc2-platypus/packages/duck/src/duck/presets.py"
FLASH_CELL = "cell == ('qwen3.8-flash-next', 'slurm-b200')"
CHECKPOINTS = "/data/shared/research/checkpoints/"

# The launcher serves exactly these harness values (checked against its argv below).
MIRRORED = {
    "ullm model": CHECKPOINTS + "Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform",
    "ullm revision": None,
    "ullm tokenizer_revision": None,
    # Not a flag: the fork reads the checkpoint's own quantization_config.
    "ullm quantization": "modelopt_fp4",
    "ullm dtype": "bfloat16",
    "ullm max_model_len": 262144,
    "ullm mamba_ssm_cache_dtype": "bfloat16",
    "ullm mamba_cache_mode": "align",
    "ullm speculative_config": HARNESS_SPECULATIVE | {"num_speculative_tokens": 4},
    "ullm speculative_config.model": FLASH_E307_DRAFTER,
    "ullm generation_config": "vllm",
    "ullm enforce_eager": False,
    "ullm enable_prefix_caching": True,
    # Not a flag: under drafting the fork defaults a hybrid model to dense
    # recurrent checkpoints (vllm/engine/arg_utils.py at 3bda1ba09).
    "ullm prefix_cache_retention_interval": None,
    "ullm watermark": 0.10,
    "ullm async_scheduling": True,
    "ullm tensor_parallel_size": 1,
    "ullm env VLLM_USE_V2_MODEL_RUNNER": "1",
    "ullm enable_flashinfer_autotune": False,
    "ullm kernel_config.moe_backend": "marlin",
    "ullm engram_config": "vllm.config.EngramConfig(cpu_offload=True)",
    "duck max_num_batched_tokens": 8192,
}

# Deliberately different: the harness value, and words of FLASH_RL.md's Serving
# section that state the difference and its reason.
DIFFERENT = {
    "ullm kv_cache_dtype": ("fp8_per_token_head", "`--kv-cache-dtype auto` (BF16 KV)"),
    "ullm cpu_offload_params": ({"embed_tokens"}, "no `embed_tokens` offload"),
    "ullm reasoning_parser": ("qwen3", "no `--reasoning-parser`"),
    "ullm prefix_match_unit": (80, "`prefix_match_unit`"),
    "ullm prefix_cache_decode_tail": (True, "`prefix_cache_decode_tail`"),
    "ullm prefix_cache_context_threads": (True, "`prefix_cache_context_threads`"),
    "ullm mm_processor_cache_type": ("shm", '`mm_processor_cache_type="shm"`'),
    "ullm factory.dense_cuda_capture": (True, "`dense_cuda_capture`"),
    "ullm gpu_memory_utilization": (0.9488, "ullm 0.9488"),
    "ullm env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": (
        "0",
        "`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`",
    ),
    "duck gpu_memory_utilization": (0.9174, "the ARC-2 column 0.9174"),
    "duck attention_config.indexer_kv_dtype": (
        "fp8",
        "`attention_config.indexer_kv_dtype`",
    ),
    "duck factory.cpu_kv_offload": (True, "`cpu_kv_offload`"),
    "duck factory.cpu_kv_offload_gib": (96.0, "96 GiB on the ARC-2 column"),
    "duck model": (
        CHECKPOINTS + "Qwen3.8-Flash-Next-NVFP4-arc2-reap307",
        "`E307_ARC2_CHECKPOINT`",
    ),
    "duck speculative_config.model": (
        CHECKPOINTS + "Qwen3.8-Flash-Next-NVFP4-arc2-reap307-drafter",
        "`E307_ARC2_DRAFTER`",
    ),
    "duck run_config.max_concurrent": (230, "230 concurrent games per engine"),
}


def _value(node, names):
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def _module(path):
    tree = ast.parse(path.read_text())
    constants, functions = {}, {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return constants, functions


def _settings(statements, prefixes, names, functions, *, nested=True):
    """Serving setting -> value of every assignment among STATEMENTS whose
    target starts with one of PREFIXES (or sets ``engine_env[...]``), through
    calls to the module's own helpers; ``{**x, "k": v}`` sets ``x.k``."""
    found = {}
    nodes = [n for s in statements for n in (ast.walk(s) if nested else [s])]
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if (
                isinstance(target, ast.Subscript)
                and ast.unparse(target.value) == "engine_env"
            ):
                found[f"env {ast.literal_eval(target.slice)}"] = _value(value, names)
                continue
            if ast.unparse(target) == "engine_env" and ast.unparse(value).startswith(
                "dict("
            ):
                for key, item in names[value.args[0].id].items():
                    found[f"env {key}"] = item
                continue
            name = ast.unparse(target)
            label = next(
                (
                    label + name[len(prefix) :]
                    for prefix, label in prefixes.items()
                    if name.startswith(prefix)
                ),
                None,
            )
            if label is None:
                continue
            if isinstance(value, ast.Dict) and None in value.keys:
                for key, item in zip(value.keys, value.values):
                    if key is not None:
                        found[f"{label}.{ast.literal_eval(key)}"] = _value(item, names)
            else:
                found[label] = _value(value, names)
        elif (
            nested
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in functions
        ):
            helper = functions[node.value.func.id]
            parameters = [argument.arg for argument in helper.args.args]
            bound = {
                **names,
                **{
                    parameter: _value(argument, names)
                    for parameter, argument in zip(parameters, node.value.args)
                },
            }
            found.update(
                _settings(helper.body, {f"{parameters[0]}.": ""}, bound, functions)
            )
    return found


def _branch(function, test):
    """The body of the if/elif branch of FUNCTION whose condition is TEST."""
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == test
    ]
    assert len(matches) == 1, test
    return matches[0].body


def _harness_settings():
    """The ullm Flash slurm-b200 deployment, then what the ARC-2 Duck adds."""
    constants, functions = _module(DEPLOYMENT)
    flash = functions["_flashnext_config"]
    local = {
        **constants,
        **{
            node.targets[0].id: _value(node.value, constants)
            for node in flash.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        },
    }
    engine = next(
        node
        for node in ast.walk(flash)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "vllm.AsyncEngineArgs"
    )
    ullm = {keyword.arg: _value(keyword.value, local) for keyword in engine.keywords}
    build = functions["build"]
    environment = _branch(build, "environment == 'slurm-b200'")
    # build()'s locals as the slurm-b200 branch leaves them (dense_cuda_capture).
    names = dict(constants)
    for statements in (build.body, environment):
        names.update(
            {
                node.targets[0].id: _value(node.value, constants)
                for node in statements
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
            }
        )
    prefixes = {"config.engine_args.": "", "factory.": "factory."}
    for layer in (build.body, environment, _branch(build, FLASH_CELL)):
        ullm.update(
            _settings(layer, prefixes, names, functions, nested=layer is not build.body)
        )

    constants, functions = _module(PRESETS)
    configure = functions["configure"]
    prefixes = {
        "engine_args.": "",
        "agent.llm_config.backend_factory.": "factory.",
        "agent.run_config.": "run_config.",
    }
    duck = {}
    layers = [
        node.body
        for node in configure.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "engine_args is not None"
    ]
    layers += [
        _branch(configure, "engine == 'qwen3.8-flash-next'"),
        _branch(configure, "environment == 'slurm-b200'"),
        _branch(configure, FLASH_CELL),
    ]
    for layer in layers:
        duck.update(_settings(layer, prefixes, constants, {}))
    return {f"ullm {k}": v for k, v in ullm.items()} | {
        f"duck {k}": v for k, v in duck.items()
    }


def _launch(tmp_path, environment):
    """The launcher's argv and engine environment, the native server faked."""
    virtualenv = tmp_path / "venv"
    (virtualenv / "bin").mkdir(parents=True)
    output = tmp_path / "launch.json"
    python = virtualenv / "bin/python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3] == ['-m', 'training.native_vllm_server']:\n"
        f"    Path({str(output)!r}).write_text(json.dumps({{'argv': sys.argv[3:], "
        "'env': dict(os.environ)}))\n"
        "elif 'config_local.set_env' in sys.argv[-1]: print('{}')\n"
        "else: os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        **environment,
        "PLATYPUS_VENV": str(virtualenv),
        "SERVER_MODEL": str(tmp_path),
        "SERVER_PORT": "0",
        "ARC3_RL_CACHE_ROOT": str(tmp_path / "cache"),
    }
    for key in (
        "SERVER_EXTRA_ARGS",
        "SERVER_LOG",
        "SERVER_GPU_MEMORY_UTILIZATION",
        "SERVER_KV_CACHE_DTYPE",
        "SERVER_MAX_NUM_BATCHED_TOKENS",
    ):
        env.pop(key, None)
    subprocess.run(
        ["bash", str(ROOT / "scripts/launch_platypus_server.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    launched = json.loads(output.read_text())
    return launched["argv"], launched["env"]


def test_launcher_serves_the_mirrored_settings(tmp_path):
    """What flash_rl.sbatch hands each server for the recipe, served as the harness serves."""
    shape = job_shape(ExperimentConfig.load(RECIPE), {})
    argv, env = _launch(
        tmp_path,
        {
            "SERVER_WEIGHT_MODE": shape["SERVER_WEIGHT_MODE"],
            "SERVER_MAX_MODEL_LEN": shape["SERVER_CONTEXT_LIST"].split()[0],
            "SERVER_MAX_NUM_SEQS": shape["SERVER_MAX_NUM_SEQS"],
            "SERVER_TENSOR_PARALLEL": "1",
            "SERVER_ENVIRONMENT": shape["SERVER_ENVIRONMENT"],
            "SERVER_MULTIMODAL_CONTEXT": shape["SERVER_MULTIMODAL_CONTEXT"],
            "SERVER_SPECULATIVE_CONFIG": shape["SERVER_SPECULATIVE_CONFIG"],
        },
    )

    def flag(name):
        assert argv.count(name) == 1, name
        return argv[argv.index(name) + 1]

    speculative = json.loads(flag("--speculative-config"))
    engram = json.loads(flag("--engram-config"))
    served = {
        # The job's SERVER_MODEL, passed as --model (tests/test_native_launcher.py).
        "ullm model": shape["SERVER_MODEL"],
        "ullm revision": None if "--revision" not in argv else flag("--revision"),
        "ullm tokenizer_revision": (
            None if "--tokenizer-revision" not in argv else flag("--tokenizer-revision")
        ),
        "ullm quantization": flag("--quantization")
        if "--quantization" in argv
        else "modelopt_fp4",
        "ullm dtype": flag("--dtype"),
        "ullm max_model_len": int(flag("--max-model-len")),
        "ullm mamba_ssm_cache_dtype": flag("--mamba-ssm-cache-dtype"),
        "ullm mamba_cache_mode": flag("--mamba-cache-mode"),
        "ullm speculative_config": {
            k: v for k, v in speculative.items() if k != "model"
        },
        "ullm speculative_config.model": speculative["model"],
        "ullm generation_config": flag("--generation-config"),
        "ullm enforce_eager": "--enforce-eager" in argv,
        "ullm enable_prefix_caching": "--enable-prefix-caching" in argv,
        "ullm prefix_cache_retention_interval": (
            flag("--prefix-cache-retention-interval")
            if "--prefix-cache-retention-interval" in argv
            else None
            if "--speculative-config" in argv
            else 0
        ),
        "ullm watermark": float(flag("--watermark")),
        "ullm async_scheduling": "--async-scheduling" in argv,
        "ullm tensor_parallel_size": int(flag("--tensor-parallel-size")),
        "ullm env VLLM_USE_V2_MODEL_RUNNER": env.get("VLLM_USE_V2_MODEL_RUNNER"),
        "ullm enable_flashinfer_autotune": "--no-enable-flashinfer-autotune"
        not in argv,
        "ullm kernel_config.moe_backend": json.loads(flag("--kernel-config"))[
            "moe_backend"
        ],
        "ullm engram_config": "vllm.config.EngramConfig("
        + ", ".join(f"{key}={value!r}" for key, value in engram.items())
        + ")",
        "duck max_num_batched_tokens": int(flag("--max-num-batched-tokens")),
    }
    assert served == MIRRORED
    # And none of the deliberately different settings is served the harness's way.
    assert flag("--kv-cache-dtype") == "auto"
    assert float(flag("--gpu-memory-utilization")) == 0.95
    assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" not in env
    for absent in (
        "--attention-config",
        "--cpu-offload-params",
        "--reasoning-parser",
        "--prefix-match-unit",
        "--prefix-cache-decode-tail",
        "--prefix-cache-context-threads",
        "--kv-transfer-config",
        "--mm-processor-cache-type",
    ):
        assert not any(argument.startswith(absent) for argument in argv), absent


def test_flash_rl_states_every_deliberate_difference():
    text = (ROOT / "training/FLASH_RL.md").read_text()
    serving = text[text.index("\n## Serving\n") : text.index("\n## Parity gate\n")]
    missing = {
        key: words for key, (_, words) in DIFFERENT.items() if words not in serving
    }
    assert missing == {}
