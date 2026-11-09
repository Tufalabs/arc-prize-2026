#!/usr/bin/env python3
"""Find the prebuilt job environments for a revision of this repository.

scripts/build_env.sbatch publishes each environment as <root>/<name>-<key>.tar.zst
beside its manifest <name>-<key>.json. The key hashes what was installed, so it
is known only after a build; a revision is matched to an artifact through the
identity the builder derives from the revision without building (--print-plan):
the recipe and name, the base interpreter, and the locks and submodule pins
that decide what is installed. A manifest's "source" holds the identity of the
build that first published its key; every later build that installed the same
thing (say, a revision whose lock changed only for another extra) leaves a
receipt, <root>/receipts/<name>-<key>-<job>.json, with its own identity.
With --run-dir, an artifact a run started with (named by the staged
interpreters in its runtime_fingerprint.json) comes first: the job refuses to
resume a run with other environments.

    resolve_envs.py --root DIR --repo REPO --revision REV [--run-dir DIR] [--quiet] RECIPE...

prints "<recipe> <tarball>" for every recipe with a match (the newest when
several match) and exits 3 if any recipe has none. Unless --quiet, stderr names
each such recipe, the identity it needs, and the nearest artifact under the root
(the record differing in the fewest identity fields) with the fields that differ.
Exit 2: the builder could not plan the revision. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RECIPES = ("trainer", "harness", "arc2-worker")
# Plan keys (build_env.sbatch --print-plan) that decide what a build installs;
# the manifest's "source" records each under the same name, except the
# interpreter (plan "python", source "python_source").
COMMON = ("recipe", "python")
IDENTITY = {
    "trainer": ("uv_lock_sha256", "extras", "harness_packages", "harness_revision"),
    "harness": ("package", "harness_revision"),
    "arc2-worker": (
        "project_uv_lock_sha256",
        "harness_revision",
        "arc2_harness_submodule_pin",
    ),
}
SOURCE_NAMES = {"python": "python_source"}
KEY = re.compile(r"[0-9a-f]{16}")
# stage_venv's per-key directory: /cache/users/<uid>/venvs/<name>-<key>/bin/python.
STAGED = re.compile(r"/venvs/([a-z0-9][a-z0-9-]*)-([0-9a-f]{16})/")


def read_plan(builder: Path, recipe: str, repo: Path, revision: str) -> dict[str, str]:
    """The builder's resolved recipe for REVISION of REPO, as key=value fields."""
    result = subprocess.run(
        ["bash", str(builder), recipe, revision, "--repo", str(repo), "--print-plan"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{builder} could not plan {recipe} at {revision}:\n{result.stderr.strip()}"
        )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def _normal(value: object) -> str | None:
    """Plan values are space-joined strings; manifests store lists as lists."""
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _shown(value: str | None) -> str:
    """A normalized value as a refusal prints it."""
    return "unset" if value is None else (value or "empty")


def expected_identity(plan: dict[str, str]) -> dict[str, str | None]:
    """The source fields an artifact built from this plan records."""
    recipe = plan["recipe"]
    if recipe not in IDENTITY:
        raise ValueError(f"unknown recipe {recipe!r}")
    return {
        SOURCE_NAMES.get(key, key): _normal(plan.get(key))
        for key in COMMON + IDENTITY[recipe]
    }


def _load(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def candidates(root: Path, name: str) -> list[tuple[Path, dict, str, Path]]:
    """(tarball, source, created_at, record) for every published artifact named
    NAME under ROOT, once from its manifest and once from each of its receipts
    (RECORD is the file the source was read from)."""
    found = []
    for manifest in sorted(root.glob(f"{name}-*.json")):
        key = manifest.name[len(name) + 1 : -len(".json")]
        tarball = root / f"{name}-{key}.tar.zst"
        record = _load(manifest)
        if KEY.fullmatch(key) and tarball.is_file() and record is not None:
            found.append(
                (
                    tarball,
                    record.get("source") or {},
                    str(record.get("created_at", "")),
                    manifest,
                )
            )
    for receipt in sorted((root / "receipts").glob(f"{name}-*.json")):
        record = _load(receipt)
        if (
            record is None
            or record.get("name") != name
            or not KEY.fullmatch(str(record.get("key", "")))
        ):
            continue
        tarball = root / f"{name}-{record['key']}.tar.zst"
        if (
            tarball.is_file()
            and tarball.with_name(f"{name}-{record['key']}.json").is_file()
        ):
            found.append(
                (
                    tarball,
                    record.get("source") or {},
                    str(record.get("created_at", "")),
                    receipt,
                )
            )
    return found


def started_with(run_dir: Path, root: Path) -> dict[str, Path]:
    """By name, the artifacts under ROOT whose staged interpreters the run in
    RUN_DIR recorded when it started (its runtime_fingerprint.json)."""
    try:
        payload = json.loads((run_dir / "runtime_fingerprint.json").read_text())
    except (OSError, ValueError):
        return {}
    found = {}
    for section in payload.values() if isinstance(payload, dict) else ():
        if not isinstance(section, dict):
            continue
        for field in ("python", "python_resolved"):
            match = STAGED.search(str(section.get(field, "")))
            tarball = match and root / f"{match.group(1)}-{match.group(2)}.tar.zst"
            if (
                tarball
                and tarball.is_file()
                and tarball.with_suffix("").with_suffix(".json").is_file()
            ):
                found[match.group(1)] = tarball
    return found


def differences(expected: dict[str, str | None], source: dict) -> list[str]:
    """The identity fields whose value SOURCE records differently."""
    return [
        field
        for field, value in expected.items()
        if _normal(source.get(field)) != value
    ]


def select(plan: dict[str, str], root: Path) -> Path | None:
    """The newest artifact under ROOT whose recorded identity equals PLAN's."""
    expected = expected_identity(plan)
    matches = [
        (created, tarball.stat().st_mtime, tarball)
        for tarball, source, created, _ in candidates(root, plan["name"])
        if not differences(expected, source)
    ]
    return max(matches)[2] if matches else None


def nearest(plan: dict[str, str], root: Path) -> str:
    """For a refusal: the record under ROOT differing from PLAN's identity in
    the fewest fields (the newest among equals), its key and those fields."""
    expected = expected_identity(plan)
    best = None
    for tarball, source, created, record in candidates(root, plan["name"]):
        differ = differences(expected, source)
        rank = (-len(differ), created, tarball.stat().st_mtime)
        if best is None or rank > best[0]:
            best = (rank, tarball, source, record, differ)
    if best is None:
        return f"there is no {plan['name']}-<key> artifact in {root}"
    _, tarball, source, record, differ = best
    key = tarball.name[len(plan["name"]) + 1 : -len(".tar.zst")]
    fields = "; ".join(
        f"{field} (recorded {_shown(_normal(source.get(field)))}, "
        f"this revision {_shown(expected[field])})"
        for field in differ
    )
    return f"the nearest is key {key} ({record.relative_to(root)}), which differs in {fields}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="artifact directory")
    parser.add_argument(
        "--repo", required=True, type=Path, help="git clone holding the revision"
    )
    parser.add_argument(
        "--revision", required=True, help="commit (or ref) of this repository"
    )
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).resolve().with_name("build_env.sbatch"),
        help="build_env.sbatch to plan with (default: the one beside this script)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="a run to resume: the environments it started with come first",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not report recipes without a match"
    )
    parser.add_argument("recipes", nargs="+", choices=RECIPES, metavar="RECIPE")
    args = parser.parse_args(argv)
    resumed = started_with(args.run_dir, args.root) if args.run_dir else {}
    missing = []
    for recipe in args.recipes:
        try:
            plan = read_plan(args.builder, recipe, args.repo.resolve(), args.revision)
        except RuntimeError as error:
            print(f"resolve_envs: {error}", file=sys.stderr)
            return 2
        tarball = resumed.get(plan["name"]) or select(plan, args.root)
        if tarball is None:
            missing.append((recipe, plan))
        else:
            print(recipe, tarball)
    if not args.quiet:
        for recipe, plan in missing:
            identity = ", ".join(
                f"{key}={value}"
                for key, value in expected_identity(plan).items()
                if value
            )
            print(
                f"resolve_envs: no {recipe} environment in {args.root} matches {plan['revision']} "
                f"({plan['name']}: {identity})\n"
                f"resolve_envs:   {nearest(plan, args.root)}",
                file=sys.stderr,
            )
    return 3 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
