"""Exercise the launcher without CUDA: only the final native invocation is faked."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("weight_mode", ["lora", "full_weight", "flash"])
@pytest.mark.parametrize("served_name", [None, "explicit-policy-name"])
@pytest.mark.parametrize("bind_host", [None, "0.0.0.0"])
@pytest.mark.parametrize("environment", ["arc3", "arc2"])
@pytest.mark.parametrize("image_mode", ["", "current_grid", "level_start"])
def test_local_model_override_preserves_public_alias(tmp_path, served_name, bind_host, environment, image_mode, weight_mode):
    virtualenv = tmp_path / "venv"
    (virtualenv / "bin").mkdir(parents=True)
    python = virtualenv / "bin" / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:3] == ['-m', 'training.native_vllm_server']:\n"
        "    Path(os.environ['ARC3_LAUNCHER_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "    assert os.environ['SERVER_API_KEY'] == 'launcher-test-secret'\n"
        "    if os.environ['SERVER_WEIGHT_MODE'] == 'flash': assert os.environ['VLLM_USE_V2_MODEL_RUNNER'] == '1'\n"
        "elif 'config_local.set_env' in sys.argv[-1]:\n"
        "    print('{}')\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    model = tmp_path / "merged-checkpoint"
    model.mkdir()
    output = tmp_path / "argv.json"
    env = dict(os.environ)
    for key in ("SERVER_SERVED_NAME", "SERVER_EXTRA_ARGS", "SERVER_LOG", "SERVER_HOST",
                "SERVER_SPECULATIVE_CONFIG", "SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS"):
        env.pop(key, None)
    env.update(
        PLATYPUS_VENV=str(virtualenv), SERVER_MODEL=str(model), SERVER_MAX_NUM_SEQS="40",
        SERVER_REVISION="base-model-revision", SERVER_PORT="0",
        ARC3_RL_CACHE_ROOT=str(tmp_path / "cache"),
        ARC3_COMPILE_CACHE_ROOT=str(tmp_path / "cache" / "compile"),
        ARC3_LAUNCHER_TEST_ARGS=str(output),
        SERVER_API_KEY="launcher-test-secret",
        SERVER_WEIGHT_MODE=weight_mode,
        SERVER_ENVIRONMENT=environment,
        SERVER_MULTIMODAL_CONTEXT=image_mode,
    )
    if weight_mode == "flash":
        env.pop("SERVER_REVISION")
    if served_name is not None:
        env["SERVER_SERVED_NAME"] = served_name
    if bind_host is not None:
        env["SERVER_HOST"] = bind_host
    launcher = Path(__file__).resolve().parents[1] / "scripts/launch_platypus_server.sh"
    completed = subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True, text=True)
    arguments = json.loads(output.read_text())
    assert arguments[:2] == ["-m", "training.native_vllm_server"]
    assert arguments[arguments.index("--model") + 1] == str(model)
    assert arguments[arguments.index("--served-model-name") + 1] == (served_name or "Qwen/Qwen3.8-27B")
    assert "--revision" not in arguments and "--tokenizer-revision" not in arguments
    assert arguments[arguments.index("--host") + 1] == (bind_host or "127.0.0.1")
    assert "--api-key" not in arguments
    assert arguments[arguments.index("--dtype") + 1] == "bfloat16"
    # Every served model shares the Qwen vocabulary: <tool_call> is 248058.
    assert arguments[arguments.index("--summary-forbidden-token-ids") + 1] == "248058"
    limits = json.loads(arguments[arguments.index("--limit-mm-per-prompt") + 1])
    if environment == "arc2" or image_mode:
        assert limits == {"video": 0}
        assert arguments[arguments.index("--mm-encoder-attn-backend") + 1] == "FLASH_ATTN"
    else:
        assert limits == {"image": 0, "video": 0}
    assert "launcher-test-secret" not in output.read_text() + completed.stdout + completed.stderr

    # Flash serves merged policy weights: no LoRA flags of any kind.
    assert ("--enable-lora" in arguments) == (weight_mode == "lora")
    assert ("--max-lora-rank" in arguments) == (weight_mode == "lora")
    assert ("--max-loras" in arguments) == (weight_mode == "lora")
    # Full weights and Flash's merged policy weights both arrive over NCCL.
    assert ("--weight-transfer-config" in arguments) == (weight_mode in ("full_weight", "flash"))
    if weight_mode != "lora":
        assert json.loads(arguments[arguments.index("--weight-transfer-config") + 1]) == {"backend": "nccl"}
    if weight_mode == "flash":
        assert "--additional-config" not in arguments
        assert "--language-model-only" not in arguments
        assert "--enable-chunked-prefill" in arguments
        assert "--no-enable-flashinfer-autotune" in arguments
        assert "--skip-mm-profiling" in arguments
        assert "--async-scheduling" in arguments
        assert json.loads(arguments[arguments.index("--compilation-config") + 1]) == {
            "inductor_compile_config": {"triton.autotune_at_compile_time": False}
        }
        for flag, value in {
            "--kv-cache-dtype": "auto", "--mamba-ssm-cache-dtype": "bfloat16",
            "--mamba-cache-mode": "align", "--kernel-config": '{"moe_backend":"marlin"}',
            "--max-num-seqs": "40", "--max-num-batched-tokens": "8192",
            "--gpu-memory-utilization": "0.95", "--mm-processor-cache-gb": "0",
            "--engram-config": '{"cpu_offload": true}', "--watermark": "0.10",
        }.items():
            assert arguments[arguments.index(flag) + 1] == value
        for flag in ("--offload-backend", "--cpu-offload-gb", "--cpu-offload-params"):
            assert flag not in arguments
    else:
        assert "--cpu-offload-gb" not in arguments
        # The harness's Flash scheduling is the fork's; vLLM 0.25.1 keeps its own.
        assert "--async-scheduling" not in arguments and "--watermark" not in arguments
        assert arguments[arguments.index("--additional-config") + 1] == '{"gdn_prefill_backend": "triton"}'
    # Drafting is opt-in (inference.speculative_decoding): off unless the job sets it.
    assert "--speculative-config" not in arguments


@pytest.mark.parametrize("extra", ["--kv-offloading-size 1", "--cpu-offload-gb 48", "--kv-transfer-config {}",
                                   "--cpu-offload-params embed_tokens", "--engram-config {}"])
def test_flash_rejects_offload_passthrough(tmp_path, extra):
    env = dict(os.environ, PLATYPUS_VENV=str(tmp_path), SERVER_MODEL=str(tmp_path),
               SERVER_PORT="0", SERVER_WEIGHT_MODE="flash", SERVER_EXTRA_ARGS=extra)
    result = subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "GPU-only" in result.stderr


def test_flash_requires_local_checkpoint(tmp_path):
    env = dict(os.environ, PLATYPUS_VENV=str(tmp_path), SERVER_MODEL="Qwen/remote",
               SERVER_PORT="0", SERVER_WEIGHT_MODE="flash")
    result = subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "local checkpoint directory" in result.stderr


@pytest.mark.parametrize("seqs", [None, "", "0", "08", "-4", "lots"])
def test_flash_requires_the_jobs_engine_concurrency(tmp_path, seqs):
    """No default: an engine with fewer slots than its share of a window queues
    the episodes, and collection slows without an error (the job derives it)."""
    env = dict(os.environ, PLATYPUS_VENV=str(tmp_path), SERVER_MODEL=str(tmp_path),
               SERVER_PORT="0", SERVER_WEIGHT_MODE="flash", ARC3_RL_CACHE_ROOT=str(tmp_path / "cache"))
    env.pop("SERVER_EXTRA_ARGS", None)
    env.pop("SERVER_MAX_NUM_SEQS", None)
    if seqs is not None:
        env["SERVER_MAX_NUM_SEQS"] = seqs
    result = subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "flash requires SERVER_MAX_NUM_SEQS, a positive integer" in result.stderr


def test_flash_scheduler_and_kv_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVER_KV_CACHE_DTYPE", "fp8_per_token_head")
    monkeypatch.setenv("SERVER_MAX_MODEL_LEN", "40960")
    # Capture the same real launch path, then verify the explicitly requested KV dtype.
    # Default assertions are covered by the matrix above.
    virtualenv = tmp_path / "venv"
    (virtualenv / "bin").mkdir(parents=True)
    output = tmp_path / "argv.json"
    python = virtualenv / "bin/python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3] == ['-m', 'training.native_vllm_server']:\n"
        f"    Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "elif 'config_local.set_env' in sys.argv[-1]: print('{}')\n"
        "else: os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    env = dict(os.environ, PLATYPUS_VENV=str(virtualenv), SERVER_MODEL=str(tmp_path),
               SERVER_WEIGHT_MODE="flash", SERVER_PORT="0", SERVER_MAX_NUM_SEQS="8",
               SERVER_MAX_NUM_BATCHED_TOKENS="4096", SERVER_MAX_LORA_RANK="32",
               SERVER_MAX_LORAS="3", SERVER_SCORE_CENTERING_TOP_K="64",
               ARC3_RL_CACHE_ROOT=str(tmp_path / "cache"))
    for key in ("SERVER_EXTRA_ARGS", "SERVER_LOG"):
        env.pop(key, None)
    subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                   check=True, capture_output=True, text=True)
    arguments = json.loads(output.read_text())
    for flag, value in {
        "--kv-cache-dtype": "fp8_per_token_head", "--max-model-len": "40960",
        "--max-num-seqs": "8", "--max-num-batched-tokens": "4096",
        "--score-centering-top-k": "64",
    }.items():
        assert arguments[arguments.index(flag) + 1] == value
    # LoRA settings in the environment never enable LoRA on a Flash server.
    assert not {"--enable-lora", "--max-lora-rank", "--max-loras"}.intersection(arguments)


def _flash_arguments(tmp_path, model, **environment):
    """The argv the launcher hands the native server in flash mode."""
    virtualenv = tmp_path / "venv"
    (virtualenv / "bin").mkdir(parents=True)
    output = tmp_path / "argv.json"
    python = virtualenv / "bin/python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3] == ['-m', 'training.native_vllm_server']:\n"
        f"    Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "elif 'config_local.set_env' in sys.argv[-1]: print('{}')\n"
        "else: os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    env = {**os.environ, "PLATYPUS_VENV": str(virtualenv), "SERVER_MODEL": str(model),
           "SERVER_WEIGHT_MODE": "flash", "SERVER_PORT": "0", "SERVER_MAX_NUM_SEQS": "40",
           "ARC3_RL_CACHE_ROOT": str(tmp_path / "cache"), **environment}
    for key in ("SERVER_EXTRA_ARGS", "SERVER_LOG"):
        env.pop(key, None)
    subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                   check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_flash_drafts_with_the_job_speculative_config(tmp_path):
    from training.config import SpeculativeDecodingConfig

    speculative = json.dumps(SpeculativeDecodingConfig(enabled=True).engine_config(), sort_keys=True)
    arguments = _flash_arguments(tmp_path / "drafted", tmp_path, SERVER_SPECULATIVE_CONFIG=speculative)
    assert arguments.count("--speculative-config") == 1
    assert json.loads(arguments[arguments.index("--speculative-config") + 1]) == {
        "method": "mtp", "num_speculative_tokens": 4, "quantization": "modelopt_fp4",
        "draft_sample_method": "probabilistic", "gdn_recoverssm": True,
        "disable_eagle_block_drop": True,
        "model": "/data/shared/research/checkpoints/"
                 "Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform-drafter-v1-nvfp4-dv16k",
    }
    # Everything else is the undrafted flash launch.
    plain = _flash_arguments(tmp_path / "plain", tmp_path, SERVER_SPECULATIVE_CONFIG="")
    index = arguments.index("--speculative-config")
    assert arguments[:index] + arguments[index + 2:] == plain


@pytest.mark.parametrize("ids", ["", "248058,248059"])
def test_summary_forbidden_ids_can_be_replaced(tmp_path, ids):
    arguments = _flash_arguments(tmp_path, tmp_path, SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS=ids)
    assert arguments[arguments.index("--summary-forbidden-token-ids") + 1] == ids


@pytest.mark.parametrize("weight_mode", ["lora", "full_weight"])
def test_speculative_config_is_flash_only(tmp_path, weight_mode):
    env = dict(os.environ, PLATYPUS_VENV=str(tmp_path), SERVER_MODEL=str(tmp_path),
               SERVER_REVISION="r", SERVER_PORT="0", SERVER_WEIGHT_MODE=weight_mode,
               SERVER_SPECULATIVE_CONFIG='{"method": "mtp"}')
    env.pop("SERVER_EXTRA_ARGS", None)
    result = subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "Flash serving setting" in result.stderr


@pytest.mark.parametrize("extra", ["--speculative-config {}", "--speculative_config={}"])
def test_launcher_refuses_speculative_passthrough(tmp_path, extra):
    env = dict(os.environ, PLATYPUS_VENV=str(tmp_path), SERVER_MODEL=str(tmp_path),
               SERVER_PORT="0", SERVER_WEIGHT_MODE="flash", SERVER_EXTRA_ARGS=extra)
    result = subprocess.run(["bash", "scripts/launch_platypus_server.sh"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "inference.speculative_decoding" in result.stderr
