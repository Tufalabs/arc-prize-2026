# type: ignore
"""Tests for taaf.deploy_kaggle."""

import ast
import asyncio
import json
import os
import pickle
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import taaf

#: The runner's mount-layout helpers; every ref is searched through both.
_MOUNT_HELPERS = ("_dataset_mount_candidates", "_kernel_mount_candidates")


def _benchmark(job_dir: Path, solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    return taaf.benchmark.Benchmark(
        label="test_kaggle",
        games=[taaf.game_examples.flip_game()],
        solver=solver or taaf.solver_examples.SolverFlipLR(),
        n_passes=1,
        job_dir=job_dir,
    )


def _fake_repo(tmp_path: Path, name: str = "fake_repo", pkg: str = "fake_pkg") -> Path:
    repo = tmp_path / name
    (repo / "src" / pkg).mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
        f'[tool.setuptools.packages.find]\nwhere = ["src"]\ninclude = ["{pkg}*"]\n'
    )
    (repo / "README.md").write_text("# fake\n")
    (repo / "src" / pkg / "__init__.py").write_text("")
    return repo


async def test_kaggle_deploy_dry_run_writes_kernel_and_source_bundles(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    bm = _benchmark(tmp_path / "job")
    handle = await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True))

    assert isinstance(handle, taaf.deploy_kaggle.KaggleHandle)
    assert handle.kernel_id == "dry-user/taaf-arc-agi2"
    assert handle.dataset_ref == "dry-user/taaf-release-source"
    assert not handle.uploaded
    source = bm.job_dir / "kaggle" / "source-dataset"
    kernel = bm.job_dir / "kaggle" / "kernel"
    assert (source / "taaf-kaggle-bundle.json").is_file()
    assert not (source / "run_kaggle.py").exists()
    assert (source / "src" / "fake_repo" / "src" / "fake_pkg" / "__init__.py").is_file()
    assert (source / "benchmark_initial.pkl").is_file()
    assert (source / "deploy_target.pkl").is_file()
    # Bundled so the runner can drop it into /kaggle/working for diagnostics.html.
    assert (source / "git_status.txt").is_file()
    assert (kernel / "taaf_kaggle_run.ipynb").is_file()
    metadata = json.loads((kernel / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    assert "dry-user/taaf-release-source" in metadata["dataset_sources"]
    notebook_source = (kernel / "taaf_kaggle_run.ipynb").read_text()
    assert "ENABLE_GPU = True" in notebook_source
    assert "LIBRARY_PATH" in notebook_source


async def test_kaggle_deploy_cpu_only_disables_gpu_for_offline_debug(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    bm = _benchmark(tmp_path / "job")
    await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, cpu_only=True))

    metadata = json.loads((bm.job_dir / "kaggle" / "kernel" / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert "machine_shape" not in metadata
    notebook = json.loads((bm.job_dir / "kaggle" / "kernel" / "taaf_kaggle_run.ipynb").read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "ENABLE_GPU = False" in notebook_source


async def test_kaggle_deploy_rejects_kernel_title_slug_mismatch(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    bm = _benchmark(tmp_path / "job")
    with pytest.raises(ValueError, match="kernel_title must slugify"):
        await bm.deploy(
            taaf.deploy_kaggle.KaggleTarget(
                dry_run=True,
                kernel_slug="stable-slug",
                kernel_title="different title",
            )
        )


@dataclass
class _KaggleExtrasSolver(taaf.solver_examples.SolverFlipLR):
    kaggle_dataset_sources: list[str] = field(
        default_factory=lambda: [
            "https://www.kaggle.com/datasets/owner/model-dataset",
            "owner/model-dataset",
        ]
    )
    kaggle_model_sources: list[str] = field(
        default_factory=lambda: [
            "owner/shared-model/pytorch/default/2",
            "owner/solver-model/pytorch/default/1",
        ]
    )
    kaggle_setup_commands: list[str] = field(default_factory=lambda: ["echo solver-setup"])
    kaggle_teardown_commands: list[str] = field(default_factory=lambda: ["echo solver-teardown"])
    kaggle_pip_install_packages: list[str] = field(default_factory=lambda: ["example-package==1.2.3"])
    kaggle_kernel_sources: list[str] = field(
        default_factory=lambda: [
            "https://www.kaggle.com/code/owner/util-script",
            "owner/util-script",
        ]
    )


async def test_kaggle_deploy_includes_solver_declared_assets(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    solver = _KaggleExtrasSolver()
    bm = _benchmark(tmp_path / "job", solver=solver)
    await bm.deploy(
        taaf.deploy_kaggle.KaggleTarget(
            dry_run=True,
            additional_dataset_sources=[
                "https://www.kaggle.com/datasets/owner/static-dataset",
                "owner/static-dataset",
            ],
            additional_model_sources=[
                "owner/target-model/pytorch/default/1",
                " owner/shared-model/pytorch/default/2 ",
            ],
            setup_commands=["echo target-setup"],
            teardown_commands=["echo target-teardown"],
            additional_kernel_sources=[
                "https://www.kaggle.com/code/owner/static-script",
                "owner/static-script",
            ],
        )
    )

    kernel_metadata = json.loads((bm.job_dir / "kaggle" / "kernel" / "kernel-metadata.json").read_text())
    assert kernel_metadata["dataset_sources"] == [
        "dry-user/taaf-release-source",
        "owner/static-dataset",
        "owner/model-dataset",
    ]
    assert kernel_metadata["kernel_sources"] == [
        "owner/static-script",
        "owner/util-script",
    ]
    # Models mount apart from datasets: the target's and the solver's, stripped
    # and deduplicated in that order, and none of them in dataset_sources.
    assert kernel_metadata["model_sources"] == [
        "owner/target-model/pytorch/default/1",
        "owner/shared-model/pytorch/default/2",
        "owner/solver-model/pytorch/default/1",
    ]
    notebook = json.loads((bm.job_dir / "kaggle" / "kernel" / "taaf_kaggle_run.ipynb").read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert '"dry-user/taaf-release-source", "owner/static-dataset", "owner/model-dataset"' in notebook_source
    assert '"owner/static-script", "owner/util-script"' in notebook_source
    assert "TAAF_KAGGLE_INPUT_PATHS" in notebook_source
    setup_commands = json.loads((bm.job_dir / "kaggle" / "source-dataset" / "setup_commands.json").read_text())
    assert "echo target-setup" in setup_commands
    assert "echo solver-setup" in setup_commands
    assert "\"$PYTHON\" -m pip install --no-deps 'example-package==1.2.3'" in setup_commands
    teardown_commands = json.loads((bm.job_dir / "kaggle" / "source-dataset" / "teardown_commands.json").read_text())
    assert "echo target-teardown" in teardown_commands
    assert "echo solver-teardown" in teardown_commands


async def test_kaggle_deploy_refuses_existing_deploy_meta(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")
    job = tmp_path / "job"
    job.mkdir()
    (job / "deploy_meta.json").write_text("{}")
    with pytest.raises(FileExistsError, match="deploy_meta.json"):
        await _benchmark(job).deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True))


async def test_the_source_dataset_goes_up_as_one_archive_the_notebook_unpacks(tmp_path: Path, monkeypatch):
    """A dataset of many files is slow to attach, and Kaggle unpacks an
    uploaded `.zip` into its files: the upload holds the bundle as one archive
    under another name, and the notebook's own helpers unpack it back into
    the bundle, byte for byte."""
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")
    monkeypatch.delenv("TAAF_KAGGLE_BUNDLE_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    bm = _benchmark(tmp_path / "job")
    await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True))

    source = bm.job_dir / "kaggle" / "source-dataset"
    upload = bm.job_dir / "kaggle" / "source-dataset-upload"
    archive = upload / taaf.deploy_kaggle.DATASET_BUNDLE_ARCHIVE
    assert not archive.name.endswith(".zip")
    assert sorted(path.name for path in upload.iterdir()) == sorted([archive.name, "dataset-metadata.json"])

    def files(root: Path) -> dict[Path, bytes]:
        return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    bundled = {name: data for name, data in files(source).items() if name != Path("dataset-metadata.json")}
    assert Path(taaf.deploy_kaggle.DATASET_BUNDLE_MARKER) in bundled
    notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    # The notebook's own helpers and names, executed as written rather than restated here.
    helpers = ast.Module(
        body=[
            node
            for node in ast.parse(code).body
            if (isinstance(node, ast.FunctionDef) and node.name in {"_unpack_bundle", "_find_taaf_bundle"})
            or (
                isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) in {"DATASET_BUNDLE_ARCHIVE", "DATASET_BUNDLE_MARKER"}
            )
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {"Path": Path, "os": os, "tempfile": tempfile, "time": time, "zipfile": zipfile}
    exec(compile(helpers, "taaf_kaggle_run.ipynb", "exec"), namespace)  # noqa: S102
    assert files(namespace["_unpack_bundle"](archive)) == bundled
    # The runner searches its working directory too, where it finds the archive, not a bundle.
    monkeypatch.chdir(upload)
    found = namespace["_find_taaf_bundle"]()
    assert found != upload
    assert files(found) == bundled


def test_kaggle_notebook_template_cells_are_valid_python(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    target = taaf.deploy_kaggle.KaggleTarget(dry_run=True)
    target.source_dataset_ref = "dry-user/taaf-release-source"
    bundle = taaf.deploy_kaggle.package_for_local_debug(_benchmark(tmp_path / "job"), target, tmp_path / "bundle")

    notebook_text = taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False)
    notebook = json.loads(notebook_text)
    code = "\n\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    compile(code, "taaf_kaggle_run.ipynb", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    # Launcher bundles git_status.txt; the runner copies it into job_dir so
    # diagnostics.html can embed the launcher-captured git status.
    assert (bundle / "git_status.txt").is_file()
    assert "git_status.txt" in code
    with open(bundle / "deploy_target.pkl", "rb") as file:
        restored = pickle.load(file)
    assert isinstance(restored, taaf.deploy_kaggle.KaggleTarget)


def test_a_competition_rerun_plays_one_pass_whatever_the_bundle_says():
    """The runner forces ``n_passes = 1`` inside its competition-rerun branch,
    before the submission path is set and the run starts.

    A measurement bundle may carry several passes; a scoring rerun hands the
    same bundle the hidden tasks and must play each exactly once, or the
    file it writes is the first pass of a run that spent its budget on
    replicates. Checked on the runner's own AST rather than by grepping, so
    the assignment has to sit under ``if true_submission:`` itself.
    """
    for run_as_submission in (False, True):
        notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=run_as_submission))
        code = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        tree = ast.parse(code)
        rerun_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "true_submission"
        ]
        assert rerun_branches, "the runner has no `if true_submission:` branch"

        def _forces_one_pass(branch: ast.If) -> bool:
            for node in ast.walk(branch):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "n_passes"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "bm"
                ):
                    return isinstance(node.value, ast.Constant) and node.value.value == 1
            return False

        assert any(_forces_one_pass(branch) for branch in rerun_branches), (
            "the rerun branch does not set bm.n_passes = 1"
        )
        # The forced value has to precede the run itself.
        assert code.index("bm.n_passes = 1") < code.index("bm.submission_path = ") < code.index("await bm.run(")


def test_kaggle_runner_setup_commands_can_persist_environment(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    target = taaf.deploy_kaggle.KaggleTarget(
        dry_run=True,
        dataset_ref="dry-user/taaf-release-source",
        additional_dataset_sources=["owner/model-dataset"],
        additional_kernel_sources=["owner/wheelhouse"],
        setup_commands=[
            '"$PYTHON" -c "import json, os, pathlib; '
            "paths = json.loads(os.environ['TAAF_KAGGLE_INPUT_PATHS']); "
            "assert paths['dry-user/taaf-release-source'] == os.environ['TAAF_KAGGLE_BUNDLE_DIR']; "
            "assert paths['owner/model-dataset'].endswith('/model-dataset'); "
            # Nothing is mounted in a local debug run, so this is the
            # declared kind's own path: the reported fallback.
            "assert paths['owner/wheelhouse'].endswith('/usr/lib/wheelhouse'); "
            "pathlib.Path(os.environ['TAAF_KAGGLE_SETUP_ENV']).write_text("
            "json.dumps({'TAAF_TEST_SETUP_ENV': 'ready'}))\"",
            "\"$PYTHON\" -c \"import os; assert os.environ.get('TAAF_TEST_SETUP_ENV') == 'ready'\"",
        ],
        teardown_commands=[
            '"$PYTHON" -c "import os, pathlib; '
            "pathlib.Path(os.environ['TAAF_KAGGLE_WORKING_DIR'], 'teardown.txt').write_text("
            "os.environ.get('TAAF_TEST_SETUP_ENV', ''))\"",
        ],
    )
    target.source_dataset_ref = "dry-user/taaf-release-source"
    bundle = taaf.deploy_kaggle.package_for_local_debug(_benchmark(tmp_path / "job"), target, tmp_path / "bundle")

    result = taaf.deploy_kaggle.run_source_bundle_locally(bundle, tmp_path / "working", timeout_s=20)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "working" / "teardown.txt").read_text() == "ready"


def test_kaggle_notebook_has_submission_flag_and_intro(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    async def _run() -> taaf.benchmark.Benchmark:
        bm = _benchmark(tmp_path / "job")
        await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, run_as_submission=True))
        return bm

    bm = asyncio.run(_run())
    notebook = json.loads((bm.job_dir / "kaggle" / "kernel" / "taaf_kaggle_run.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "TAAF ARC-AGI2 Kaggle Run" in source
    assert "https://tufalabs.ai/" in source
    assert "RUN_AS_SUBMISSION = True" in source
    assert "KAGGLE_IS_COMPETITION_RERUN" in source
    assert "run_kaggle.py" not in source
    # ARC-AGI2's competition mount is data only, so nothing is installed
    # from it — the ARC3 wheelhouse step must not creep back in.
    assert "wheelhouse" not in source
    assert "competition mount is data-only" in source
    assert "Inline customization hook" in source
    assert "await bm.run(" in source


def test_dataset_last_updated_parses_csv_skipping_warnings(monkeypatch):
    """`kaggle datasets list --csv` prints CLI warnings on stdout before the
    header; the parser must skip to the `ref,` line and match the exact ref."""
    stdout = (
        "Warning: Your Kaggle API key is readable by other users on this system!\n"
        "Warning: Looks like you're using an outdated API Version\n"
        "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating\n"
        "me/taaf-release-source,TAAF,1MB,2026-05-28 10:00:00,0,0,0.1\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
    )
    assert (
        taaf.deploy_kaggle._dataset_last_updated("me/taaf-release-source", username="me", key="k")
        == "2026-05-28 10:00:00"
    )
    assert taaf.deploy_kaggle._dataset_last_updated("me/absent", username="me", key="k") is None


def test_wait_for_dataset_version_change_returns_when_lastupdated_advances(monkeypatch):
    """Returns as soon as the listing's lastUpdated moves past the pre-push value."""
    seq = iter(["2026-05-28 10:00:00", "2026-05-28 10:05:00"])
    monkeypatch.setattr(taaf.deploy_kaggle, "_dataset_last_updated", lambda *a, **k: next(seq))
    monkeypatch.setattr(taaf.deploy_kaggle.time, "sleep", lambda *_a, **_k: None)
    taaf.deploy_kaggle._wait_for_dataset_version_change(
        "me/ds", before="2026-05-28 10:00:00", username="me", key="k", timeout_s=60.0
    )


def test_wait_for_dataset_version_change_warns_and_proceeds_on_timeout(monkeypatch, capsys):
    """If lastUpdated never advances within the budget, warn and proceed rather
    than hang the deploy — the `ready` gate already passed, and some refs
    (private datasets owned by others) aren't observable via `datasets list`."""
    monkeypatch.setattr(taaf.deploy_kaggle, "_dataset_last_updated", lambda *a, **k: "2026-05-28 10:00:00")
    monkeypatch.setattr(taaf.deploy_kaggle.time, "sleep", lambda *_a, **_k: None)
    taaf.deploy_kaggle._wait_for_dataset_version_change(
        "me/ds", before="2026-05-28 10:00:00", username="me", key="k", timeout_s=0.0
    )
    assert "did not advance" in capsys.readouterr().out


def _target_with_refs(*, kernel_id, dataset_ref, public=False):
    target = taaf.deploy_kaggle.KaggleTarget(public=public)
    target.kernel_id = kernel_id
    target.source_dataset_ref = dataset_ref
    return target


def _absent_error():
    """Kaggle masks a missing/own ref as HTTP 403 (see _kaggle_ref_absent)."""
    return RuntimeError("403 Client Error: Forbidden for url: .../GetDatasetMetadata")


def _patch_visibility(monkeypatch, mapping):
    """Run the guard keyless. ``mapping`` is ref -> ``"public"`` / ``"private"``
    / a ``BaseException`` to raise (an absent/forbidden or a transient error)."""
    monkeypatch.setattr(taaf.deploy_kaggle, "_kaggle_api", lambda username, key: object())

    def fake_fetch(api, *, kind, owner, slug):
        outcome = mapping[f"{owner}/{slug}"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome == "private"  # is_private

    monkeypatch.setattr(taaf.deploy_kaggle, "_fetch_is_private", fake_fetch)


def test_resource_is_public_reads_exact_ref(monkeypatch):
    """Public -> True; private -> False; an absent/own ref (403/404) -> False
    (deploy will create it private); any other error -> raises (fail closed)."""

    def check(outcome, *, ref="me/data", username="me"):
        def fake_fetch(api, *, kind, owner, slug):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome == "private"

        monkeypatch.setattr(taaf.deploy_kaggle, "_fetch_is_private", fake_fetch)
        return taaf.deploy_kaggle._resource_is_public(object(), kind="dataset", ref=ref, username=username)

    assert check("public") is True
    assert check("private") is False
    assert check(_absent_error()) is False  # 403 on our own ref -> not created yet
    assert check(RuntimeError("404 Not Found")) is False
    with pytest.raises(RuntimeError, match="could not determine"):
        check(RuntimeError("500 Server Error"))
    # A 403 for a ref we do NOT own is not 'absent' — it's unresolved -> fail closed.
    with pytest.raises(RuntimeError, match="could not determine"):
        check(_absent_error(), ref="someone-else/data", username="me")


def test_assert_targets_not_public_raises_on_public_notebook(monkeypatch):
    """A public destination notebook (dataset private) blocks the upload, and the
    error names the notebook as the public offender — not the private dataset."""
    _patch_visibility(monkeypatch, {"me/nb": "public", "me/ds": "private"})
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds")
    with pytest.raises(RuntimeError, match="public") as exc:
        taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")
    message = str(exc.value)
    assert "me/nb" in message
    assert "datasets/me/ds" not in message  # the private dataset is not blamed


def test_assert_targets_not_public_raises_on_public_dataset(monkeypatch):
    """A public destination dataset (notebook private) blocks the upload."""
    _patch_visibility(monkeypatch, {"me/nb": "private", "me/ds": "public"})
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds")
    with pytest.raises(RuntimeError, match="datasets/me/ds"):
        taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")


def test_assert_targets_not_public_passes_when_both_private(monkeypatch):
    """Both private -> upload proceeds (no raise). Pins the raise in the tests
    above to publicness rather than to the guard simply running."""
    _patch_visibility(monkeypatch, {"me/nb": "private", "me/ds": "private"})
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds")
    taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")


def test_assert_targets_not_public_proceeds_when_targets_absent(monkeypatch):
    """Not-yet-created targets (403/404 on our own refs) don't block the deploy —
    it will create them private by default, so a first deploy proceeds."""
    _patch_visibility(monkeypatch, {"me/nb": _absent_error(), "me/ds": _absent_error()})
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds")
    taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")


def test_assert_targets_not_public_fails_closed_on_unresolved(monkeypatch):
    """A destination whose visibility can't be read for a non-absent reason
    (here a 500) blocks the deploy — the guard fails closed, with a hint."""
    _patch_visibility(monkeypatch, {"me/nb": RuntimeError("500 Server Error"), "me/ds": "private"})
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds")
    with pytest.raises(RuntimeError, match="could not determine"):
        taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")


def test_assert_targets_not_public_blocks_public_flag_before_remote_state(monkeypatch):
    """`public=True` on the non-share bundle is itself refused, short-circuiting
    before any Kaggle query (the api factory here must not even be consulted)."""

    def _boom(username, key):
        raise AssertionError("public=True must be refused before querying Kaggle")

    monkeypatch.setattr(taaf.deploy_kaggle, "_kaggle_api", _boom)
    target = _target_with_refs(kernel_id="me/nb", dataset_ref="me/ds", public=True)
    with pytest.raises(RuntimeError, match="target.public=True"):
        taaf.deploy_kaggle._assert_targets_not_public(target, "me", "key")


def test_kaggle_env_gives_cli_a_per_user_tmpdir(tmp_path: Path, monkeypatch):
    """The kaggle CLI keeps resumable-upload state at {tempdir}/.kaggle/uploads,
    which collides between users on a shared host (EACCES for everyone but the
    directory's creator) — so the CLI subprocess must get its own TMPDIR."""
    import getpass
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    env = taaf.deploy_kaggle._kaggle_env("me", "key")
    tmpdir = Path(env["TMPDIR"])
    assert tmpdir.parent == tmp_path
    assert getpass.getuser() in tmpdir.name
    assert tmpdir.is_dir()


_LIVE_KAGGLE = os.environ.get("TAAF_KAGGLE_LIVE_TESTS") == "1"


@pytest.mark.skipif(
    not _LIVE_KAGGLE, reason="set TAAF_KAGGLE_LIVE_TESTS=1 with Kaggle creds to run live public-resource checks"
)
def test_live_real_public_vs_private_and_absent():
    """Ground-truth against real Kaggle artefacts (others can read but not write
    them): the guard fires on public targets, and does NOT fire on private or
    not-yet-created ones. Excluded from CI (no Kaggle key) via the env-var gate."""
    username, key = taaf.deploy_kaggle._resolve_kaggle_credentials(None)
    assert key is not None
    api = taaf.deploy_kaggle._kaggle_api(username, key)

    def is_public(kind, ref):
        return taaf.deploy_kaggle._resource_is_public(api, kind=kind, ref=ref, username=username)

    # Public -> guard would fire.
    assert is_public("dataset", "some-user/test-dataset") is True
    assert is_public("notebook", "some-user/notebookdeb3476d76") is True
    # Private -> guard does NOT fire.
    assert is_public("dataset", "some-user/my-math-library") is False
    # Absent (403-masked, our own) -> treated as not-yet-created, guard does NOT fire.
    assert is_public("dataset", f"{username}/taaf-guard-nonexistent-xyz") is False
    assert is_public("notebook", f"{username}/taaf-guard-nonexistent-xyz") is False

    # Guard blocks a public target...
    public_target = _target_with_refs(kernel_id="some-user/notebookdeb3476d76", dataset_ref="some-user/test-dataset")
    with pytest.raises(RuntimeError, match="public"):
        taaf.deploy_kaggle._assert_targets_not_public(public_target, username, key)
    # ...and proceeds when the targets are private / not-yet-created.
    safe_target = _target_with_refs(
        kernel_id=f"{username}/taaf-guard-nonexistent-xyz", dataset_ref="some-user/my-math-library"
    )
    taaf.deploy_kaggle._assert_targets_not_public(safe_target, username, key)


def test_submission_stop_resolves_to_runtime_minus_four_minutes():
    """The default: total runtime minus the submission end margin; an
    explicit value wins; a value that would fire at or before t=0 is a
    deploy-time error, because inside the notebook it kills the submission
    the moment the first cell runs.

    The two margins are easy to conflate and must not move together: a
    submission keeps 4 minutes to Kaggle's platform kill, while an ordinary
    run keeps 10 minutes to its own budget — which
    ``test_submission_soft_stop_fires_at_the_stop_line`` pins from the
    notebook's side.
    """
    target = taaf.deploy_kaggle.KaggleTarget()
    assert target.resolved_submission_stop_s() == 12 * 3600.0 - 240.0
    target = taaf.deploy_kaggle.KaggleTarget(max_runtime_s=3600.0, submission_stop_runtime_s=1234.0)
    assert target.resolved_submission_stop_s() == 1234.0
    # Under the margin, so it resolves at or before t=0. The number has to
    # sit below the margin itself, which is why it moved with it.
    with pytest.raises(ValueError, match="TOTAL runtime"):
        taaf.deploy_kaggle.KaggleTarget(max_runtime_s=60.0).resolved_submission_stop_s()


def test_submission_soft_stop_fires_at_the_stop_line(tmp_path: Path):
    """A submission's only exit is the graceful one, and it fires AT the
    stop line — the platform margin already lives inside the stop value,
    so a second buffer here would double-count it, and there is no hard
    fallback behind it (kernel death reads as ERROR whatever the exit
    code, so a hard exit can only destroy a run that would have
    completed). The submission branch is selected by argument."""
    notebook = json.loads(
        taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False, submission_stop_runtime_s=900.0)
    )
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    namespace: dict[str, object] = {}
    env = dict(os.environ)
    os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    try:
        exec("".join(cells[0]["source"]), namespace)  # noqa: S102 — the rendered cell IS the code under test
        # Defined by the bundle-locating cell, which cannot run outside Kaggle.
        namespace["SOFT_DEADLINE_BUFFER_S"] = 600.0
        exec("".join(cells[5]["source"]), namespace)  # noqa: S102 — _soft_end_time
    finally:
        os.environ.clear()
        os.environ.update(env)
    from datetime import datetime, timedelta

    soft_end_time = namespace["_soft_end_time"]
    start = datetime.fromtimestamp(namespace["NOTEBOOK_START_EPOCH"])  # type: ignore[arg-type]
    # Submission: at the stop line itself, no buffer of its own.
    assert soft_end_time(0.0, run_as_submission=True) == start + timedelta(seconds=900)  # type: ignore[operator]
    # Ordinary run: unchanged max_runtime_s - 600 shape.
    assert soft_end_time(1800.0, run_as_submission=False) == start + timedelta(seconds=1200)  # type: ignore[operator]


@dataclass
class _KaggleExcludingSolver(taaf.solver_examples.SolverFlipLR):
    """A solver that names a repo its own worker install supersedes."""

    kaggle_exclude_source_repos: list[str] = field(default_factory=lambda: ["Solver-Excluded"])


def test_kaggle_excludes_source_repos_from_target_and_solver(tmp_path: Path, monkeypatch):
    """A target and its solver can each keep a repo out of the bundle.

    The repo a worker installs by other means has to stay out: the runner
    prepends every bundled repo to sys.path, so a bundled copy shadows the
    installed one rather than merely wasting space in the dataset.
    """
    kept = _fake_repo(tmp_path, name="fake_repo", pkg="fake_pkg")
    by_target = _fake_repo(tmp_path, name="target-excluded", pkg="target_excluded")
    by_solver = _fake_repo(tmp_path, name="solver-excluded", pkg="solver_excluded")
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [kept, by_target, by_solver])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    async def _run() -> taaf.benchmark.Benchmark:
        bm = _benchmark(tmp_path / "job", solver=_KaggleExcludingSolver())
        await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, exclude_source_repos=["target-excluded"]))
        return bm

    bm = asyncio.run(_run())

    src_root = bm.job_dir / "kaggle" / "source-dataset" / "src"
    assert (src_root / "fake_repo").is_dir()
    assert not (src_root / "target-excluded").exists()
    # Matched case-insensitively: the solver spells it "Solver-Excluded".
    assert not (src_root / "solver-excluded").exists()


def test_every_mount_layout_is_probed_whatever_kind_a_ref_was_declared_as():
    """A wheelhouse declared as a utility output but attached as a plain
    dataset still resolves, and vice versa.

    Kaggle has attached the same wheelhouse both ways at different times,
    and getting it wrong fails silently and totally: the runner reports the
    unmounted path, and the setup command installing from it dies before a
    game is played. So the declared kind chooses which path is REPORTED when
    none of them exists; it does not narrow the search.
    """
    for run_as_submission in (False, True):
        notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=run_as_submission))
        code = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )

        # The runner's own helpers, executed as written rather than restated here.
        helpers = ast.Module(
            body=[
                node
                for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef) and node.name in {"_split_ref", *_MOUNT_HELPERS}
            ],
            type_ignores=[],
        )
        namespace: dict[str, Any] = {"Path": Path}
        exec(compile(helpers, "taaf_kaggle_run.ipynb", "exec"), namespace)  # noqa: S102
        searched = {
            path for helper in _MOUNT_HELPERS for path in namespace[helper]("other-user/arc3-vllm-0-27-1-wheels")
        }

        # The two layouts this wheelhouse has actually appeared at.
        assert Path("/kaggle/input/arc3-vllm-0-27-1-wheels") in searched
        assert Path("/kaggle/usr/lib/arc3-vllm-0-27-1-wheels") in searched
        # Both loops search the union, so neither list of refs is confined to
        # the layout its name suggests.
        for source_list in ("DATASET_SOURCES", "KERNEL_SOURCES"):
            loop = code[
                code.index(
                    f"ref in enumerate({source_list})"
                    if source_list == "DATASET_SOURCES"
                    else f"for ref in {source_list}"
                ) :
            ]
            loop = loop[: loop.index("kaggle_input_paths[ref]")]
            assert all(helper in loop for helper in _MOUNT_HELPERS), f"{source_list} does not search both layouts"


@dataclass
class _KagglePayloadSolver(taaf.solver_examples.SolverFlipLR):
    """A solver that ships a file for its kernel's setup commands to read."""

    kaggle_bundle_files: list[str] = field(default_factory=lambda: list[str]())


def test_kaggle_bundle_files_land_in_the_payload_directory(tmp_path: Path, monkeypatch):
    """Target- and solver-supplied files reach the bundle's payload/.

    A payload file is the way to ship an offline artifact the kernel
    installs for itself: nothing puts payload/ on sys.path, so unlike a
    bundled source repo it cannot shadow what the kernel installs.
    """
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")
    from_target = tmp_path / "from_target.tar"
    from_target.write_bytes(b"target payload")
    from_solver = tmp_path / "from_solver.tar"
    from_solver.write_bytes(b"solver payload")

    async def _run() -> taaf.benchmark.Benchmark:
        solver = _KagglePayloadSolver(kaggle_bundle_files=[str(from_solver)])
        bm = _benchmark(tmp_path / "job", solver=solver)
        await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, bundle_files=[from_target]))
        return bm

    bm = asyncio.run(_run())

    payload = bm.job_dir / "kaggle" / "source-dataset" / "payload"
    assert (payload / "from_target.tar").read_bytes() == b"target payload"
    assert (payload / "from_solver.tar").read_bytes() == b"solver payload"
    # Payload files are not sources: they must not appear under src/.
    assert not (bm.job_dir / "kaggle" / "source-dataset" / "src" / "from_solver.tar").exists()


def test_kaggle_bundle_file_missing_fails_the_deploy(tmp_path: Path, monkeypatch):
    """A payload file that does not exist fails on the launcher.

    The alternative is a kernel that installs nothing and runs on
    whatever the wheels happened to give it.
    """
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    async def _run() -> None:
        bm = _benchmark(tmp_path / "job")
        await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, bundle_files=[tmp_path / "absent.tar"]))

    with pytest.raises(RuntimeError, match="payload file missing"):
        asyncio.run(_run())


def test_kaggle_bundle_files_colliding_by_name_fail(tmp_path: Path, monkeypatch):
    """payload/ is flat, so two same-named files would silently overwrite
    one another."""
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")
    first = tmp_path / "a" / "overlay.tar"
    second = tmp_path / "b" / "overlay.tar"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    async def _run() -> None:
        bm = _benchmark(tmp_path / "job")
        await bm.deploy(taaf.deploy_kaggle.KaggleTarget(dry_run=True, bundle_files=[first, second]))

    with pytest.raises(RuntimeError, match="collide by name"):
        asyncio.run(_run())


_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


@dataclass
class _PinnedImageSolver(taaf.solver_examples.SolverFlipLR):
    kaggle_docker_image: str | None = _IMAGE


async def _kernel_metadata(tmp_path: Path, monkeypatch, target, solver=None) -> dict[str, Any]:
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")
    bm = _benchmark(tmp_path / "job", solver=solver)
    await bm.deploy(target)
    return json.loads((bm.job_dir / "kaggle" / "kernel" / "kernel-metadata.json").read_text())


async def test_kaggle_deploy_pins_the_image_the_solver_or_target_names(tmp_path: Path, monkeypatch):
    """A notebook runs on the image named by digest, from the solver (whose
    setup was built for it) or the target; unnamed, Kaggle chooses."""
    by_solver = await _kernel_metadata(
        tmp_path / "a", monkeypatch, taaf.deploy_kaggle.KaggleTarget(dry_run=True), _PinnedImageSolver()
    )
    assert by_solver["docker_image"] == _IMAGE
    by_target = await _kernel_metadata(
        tmp_path / "b", monkeypatch, taaf.deploy_kaggle.KaggleTarget(dry_run=True, docker_image=_IMAGE)
    )
    assert by_target["docker_image"] == _IMAGE
    unpinned = await _kernel_metadata(tmp_path / "c", monkeypatch, taaf.deploy_kaggle.KaggleTarget(dry_run=True))
    assert "docker_image" not in unpinned


async def test_kaggle_deploy_refuses_two_images(tmp_path: Path, monkeypatch):
    """The target and the solver may both name the image only if they agree."""
    agreeing = await _kernel_metadata(
        tmp_path / "a",
        monkeypatch,
        taaf.deploy_kaggle.KaggleTarget(dry_run=True, docker_image=_IMAGE),
        _PinnedImageSolver(),
    )
    assert agreeing["docker_image"] == _IMAGE
    with pytest.raises(ValueError, match="one image"):
        await _kernel_metadata(
            tmp_path / "b",
            monkeypatch,
            taaf.deploy_kaggle.KaggleTarget(dry_run=True, docker_image="gcr.io/kaggle-private-byod/python@sha256:0"),
            _PinnedImageSolver(),
        )


def test_the_notebook_takes_the_earlier_of_the_two_startup_deadlines():
    """One deadline counts from notebook start, one from machine boot: a
    session that sat on its machine for hours is past the second before its
    notebook starts, and with no reading from /proc only the first applies."""
    notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert "startup_deadline=startup_deadline" in code
    # The notebook's own helper, executed as written rather than restated here.
    helper = ast.Module(
        body=[n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "_startup_deadline"],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {"datetime": datetime, "timedelta": timedelta}
    exec(compile(helper, "taaf_kaggle_run.ipynb", "exec"), namespace)  # noqa: S102

    def deadline(machine_uptime_s, run_as_submission=False, **allowances):
        target = taaf.deploy_kaggle.KaggleTarget(dry_run=True, **allowances)
        return namespace["_startup_deadline"](
            target, run_as_submission=run_as_submission, notebook_start=start, machine_uptime_s=machine_uptime_s
        )

    start = 1_790_000_000.0
    both = {"startup_deadline_s": 25 * 60, "startup_deadline_after_boot_s": 40 * 60}
    from_start = datetime.fromtimestamp(start + 25 * 60)
    # A machine booted shortly before: notebook start's deadline comes first.
    assert deadline(120.0, **both) == from_start
    # One that sat for hours: boot's deadline, already past at notebook start.
    assert deadline(24071.0, **both) == datetime.fromtimestamp(start - 24071 + 40 * 60)
    assert deadline(24071.0, **both) < datetime.fromtimestamp(start)
    # Each alone, and neither.
    assert deadline(None, **both) == from_start
    assert deadline(24071.0, startup_deadline_s=25 * 60) == from_start
    assert deadline(1200.0, startup_deadline_after_boot_s=40 * 60) == datetime.fromtimestamp(start + 20 * 60)
    assert deadline(None, startup_deadline_after_boot_s=40 * 60) is None
    assert deadline(600.0) is None
    # By mode: a machine up 40 minutes when a prompt session's notebook starts
    # passes the long allowance outside a submission, and is past the short one in it.
    modes = {**both, "startup_deadline_after_boot_s": 90 * 60, "submission_startup_deadline_after_boot_s": 20 * 60}
    assert deadline(2379.0, **modes) == from_start
    assert deadline(2379.0, run_as_submission=True, **modes) == datetime.fromtimestamp(start - 2379 + 20 * 60)
    assert deadline(2379.0, run_as_submission=True, startup_deadline_after_boot_s=40 * 60) is None
