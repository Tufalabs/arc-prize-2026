"""The GPU diagnostic is bounded to allocations and preserves failure evidence."""

import json

from scripts import arc2_vision_preflight as preflight


def test_run_refuses_gpu_work_outside_slurm(monkeypatch):
    """Login and local hosts never start the diagnostic encoder accidentally."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    report = {"runtime": {"versions": {"vllm": "0.25.1"}}}
    try:
        preflight.run("model", "revision", report)
    except RuntimeError as error:
        assert "Slurm" in str(error)
    else:
        raise AssertionError("preflight ran outside an allocation")


def test_failure_is_written_with_stage_and_traceback(monkeypatch, tmp_path):
    """A CuTe failure leaves a concrete durable diagnosis and a failing exit code."""
    output = tmp_path / "vision.json"
    monkeypatch.setattr("sys.argv", ["preflight", "--output", str(output)])
    monkeypatch.setattr(
        preflight, "runtime_metadata", lambda: {"versions": {"vllm": "0.25.1"}}
    )

    def fail(model, revision, report):
        """Raise the diagnostic's representative toolchain error."""
        report["stage"] = "cute_import"
        raise ImportError("CUDA toolkit 13.1 required")

    monkeypatch.setattr(preflight, "run", fail)
    assert preflight.main() == 1
    result = json.loads(output.read_text())
    assert result["ok"] is False
    assert result["stage"] == "cute_import"
    assert "CUDA toolkit 13.1" in result["traceback"]


def test_metadata_only_never_initializes_cuda(monkeypatch, tmp_path):
    """Inspecting dependency identities does not allocate GPU memory."""
    output = tmp_path / "metadata.json"
    monkeypatch.setattr(
        "sys.argv", ["preflight", "--output", str(output), "--metadata-only"]
    )
    monkeypatch.setattr(
        preflight, "runtime_metadata", lambda: {"versions": {"vllm": "0.25.1"}}
    )
    monkeypatch.setattr(
        preflight,
        "run",
        lambda *args: (_ for _ in ()).throw(AssertionError("GPU work")),
    )
    assert preflight.main() == 0
    assert json.loads(output.read_text())["stage"] == "metadata"
