"""Checks and provenance for environments built by scripts/build_env.sbatch.

Run with the interpreter of the environment under test, isolated from the
source tree (``python -I -B scripts/env_checks.py ...`` from a neutral working
directory). Standard library only, plus ``packaging`` from that environment to
evaluate requirement markers.

  lock REQUIREMENTS [--base DIR] [--allow-extra NAME]... [--import-local]
      The installed distributions equal the lock export REQUIREMENTS (uv
      export, requirements-txt format): every requirement whose marker holds
      here is installed at the exported version (git requirements at the
      exported commit; local paths, relative to DIR, as non-editable installs),
      and nothing else is installed except the --allow-extra names.
      --import-local also imports each local (workspace) distribution.
  absent MODULE...
      None of the modules is importable and no distribution named after one
      (MODULE or MODULE-*) is installed.
  imports MODULE...
      Every module imports.
  reports-commit MODULE COMMIT
      MODULE.__version__ carries a +g<abbreviated hash> that abbreviates COMMIT.
  lock-agrees LOCK REFERENCE_LOCK
      Every distribution LOCK (a uv.lock) shares with REFERENCE_LOCK is locked
      at one of the reference's versions, git sources at the reference's
      commit. Local packages (workspace members, path and directory sources)
      are not compared. Needs only the standard library.
  versions NAME...
      Print "NAME VERSION" for each distribution ("NAME -" if not installed),
      so two environments can be compared line for line.
  provenance OUT FIELD...
      Write the provenance JSON stored in the artifact manifest: FIELD is
      key=string or key:=json; installed versions and git commits are added.
"""

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

GIT_COMMIT = re.compile(r"@([0-9a-f]{40})$")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    found = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            found.setdefault(canonical(name), dist)
    return found


def direct_url(dist):
    text = dist.read_text("direct_url.json")
    return json.loads(text) if text else None


def logical_lines(text):
    """Requirement lines with continuations joined and comments removed."""
    joined = re.sub(r"\\\n", " ", text)
    for raw in joined.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if line:
            yield line


def marker_holds(marker):
    try:
        from packaging.markers import Marker
    except ImportError:
        sys.exit(f"cannot evaluate the marker {marker!r}: packaging is not installed in {sys.prefix}")
    return Marker(marker).evaluate()


def project_name(path):
    data = tomllib.loads((path / "pyproject.toml").read_text())
    return data["project"]["name"]


def parse_requirements(path, base):
    """Map canonical name -> (kind, value) for the requirements that apply here."""
    expected = {}
    for line in logical_lines(Path(path).read_text()):
        line = re.sub(r"\s--hash=\S+", "", line)
        editable = line.startswith(("-e ", "--editable "))
        if editable:
            line = line.split(None, 1)[1]
        elif line.startswith("-"):
            continue  # index and other options
        spec, _, marker = line.partition(";")
        spec, marker = spec.strip(), marker.strip()
        if marker and not marker_holds(marker):
            continue
        if spec.startswith((".", "/")):
            name = canonical(project_name((Path(base) / spec).resolve()))
            expected[name] = ("local", spec)
        elif " @ " in spec:
            name, url = (part.strip() for part in spec.split(" @ ", 1))
            expected[canonical(name)] = ("url", url)
        elif "==" in spec:
            name, version = (part.strip() for part in spec.split("==", 1))
            expected[canonical(name)] = ("version", version)
        else:
            sys.exit(f"unpinned requirement in the lock export: {line}")
    return expected


def same_version(a, b):
    try:
        from packaging.version import Version
    except ImportError:
        return a == b
    return Version(a) == Version(b)


def check_lock(args):
    expected = parse_requirements(args.requirements, args.base)
    found = installed()
    allowed = {canonical(name) for name in args.allow_extra}
    problems = []
    counts = {"version": 0, "url": 0, "local": 0}
    for name, (kind, value) in sorted(expected.items()):
        dist = found.get(name)
        if dist is None:
            problems.append(f"{name}: in the lock export, not installed")
            continue
        counts[kind] += 1
        if kind == "version" and not same_version(dist.version, value):
            problems.append(f"{name}: installed {dist.version}, lock export {value}")
        origin = direct_url(dist)
        if kind == "url" and (commit := GIT_COMMIT.search(value)):
            installed_commit = (origin or {}).get("vcs_info", {}).get("commit_id")
            if installed_commit != commit.group(1):
                problems.append(f"{name}: installed from commit {installed_commit}, lock export {commit.group(1)}")
        if kind == "local":
            if not origin or "dir_info" not in origin:
                problems.append(f"{name}: expected a local install of {value}, found {origin}")
            elif origin["dir_info"].get("editable"):
                problems.append(f"{name}: editable install; a packed environment needs a regular one")
    extra = sorted(set(found) - set(expected) - allowed)
    problems += [f"{name} {found[name].version}: installed, not in the lock export" for name in extra]
    if args.import_local:
        for name, (kind, _) in sorted(expected.items()):
            if kind == "local" and name in found:
                module = importlib.import_module(name.replace("-", "_"))
                print(f"import {module.__name__} ok: {module.__file__}")
    if problems:
        sys.exit("lock export check failed:\n  " + "\n  ".join(problems))
    allowed_present = sorted(allowed & set(found))
    print(f"lock export ok: {sum(counts.values())} distributions ({counts['version']} pinned, "
          f"{counts['url']} by URL, {counts['local']} local) installed exactly as exported"
          + (f"; allowed additions: {', '.join(allowed_present)}" if allowed_present else ""))


def check_absent(args):
    found = installed()
    problems = []
    for module in args.modules:
        if importlib.util.find_spec(module) is not None:
            problems.append(f"module {module} is importable")
        root = canonical(module)
        problems += [f"distribution {name} is installed" for name in sorted(found)
                     if name == root or name.startswith(root + "-")]
    if problems:
        sys.exit("forbidden packages present:\n  " + "\n  ".join(problems))
    print(f"absent ok: {' '.join(args.modules)}")


def check_imports(args):
    for name in args.modules:
        module = importlib.import_module(name)
        print(f"import {name} ok: {getattr(module, '__version__', '')} {getattr(module, '__file__', '')}".rstrip())


def check_reports_commit(args):
    version = importlib.import_module(args.module).__version__
    match = re.search(r"\+g([0-9a-f]{7,40})", version)
    if not match or not args.commit.startswith(match.group(1)):
        sys.exit(f"{args.module}.__version__ {version} does not name commit {args.commit}")
    print(f"{args.module} {version} reports commit {args.commit}")


LOCAL_SOURCES = ("editable", "directory", "path", "virtual", "workspace")


def locked_packages(path):
    """Map canonical name -> list of (version, git commit or None) for non-local packages."""
    packages = {}
    for package in tomllib.loads(Path(path).read_text()).get("package", []):
        source = package.get("source", {})
        if any(kind in source for kind in LOCAL_SOURCES):
            continue
        commit = source["git"].rpartition("#")[2] if "git" in source else None
        packages.setdefault(canonical(package["name"]), []).append((package.get("version"), commit))
    return packages


def check_lock_agrees(args):
    locked, reference = locked_packages(args.lock), locked_packages(args.reference)
    shared = sorted(set(locked) & set(reference))
    problems = []
    for name in shared:
        allowed = set(reference[name])
        for version, commit in locked[name]:
            if (version, commit) not in allowed:
                expected = ", ".join(f"{v}{f' @ {c}' if c else ''}" for v, c in sorted(allowed, key=str))
                problems.append(f"{name}: {version}{f' @ {commit}' if commit else ''}; reference {expected}")
    if problems:
        sys.exit(f"{args.lock} disagrees with {args.reference}:\n  " + "\n  ".join(problems))
    only = sorted(set(locked) - set(reference))
    print(f"lock agreement ok: {len(shared)} shared distributions at the reference's versions"
          + (f"; only in {Path(args.lock).parent.name}: {', '.join(only)}" if only else ""))


def print_versions(args):
    found = installed()
    for name in args.names:
        dist = found.get(canonical(name))
        print(f"{canonical(name)} {dist.version if dist else '-'}")


def write_provenance(args):
    record = {}
    for field in args.fields:
        if ":=" in field:
            key, value = field.split(":=", 1)
            record[key] = json.loads(value)
        elif "=" in field:
            key, value = field.split("=", 1)
            record[key] = value
        else:
            sys.exit(f"provenance field must be key=value or key:=json: {field}")
    found = installed()
    record["python_version"] = sys.version.split()[0]
    record["versions"] = {name: found[name].version for name in sorted(found)}
    record["git_commits"] = {
        name: origin["vcs_info"]["commit_id"] for name in sorted(found)
        if (origin := direct_url(found[name])) and "vcs_info" in origin
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("requirements")
    lock.add_argument("--base", default=".")
    lock.add_argument("--allow-extra", action="append", default=[])
    lock.add_argument("--import-local", action="store_true")
    lock.set_defaults(run=check_lock)
    absent = commands.add_parser("absent")
    absent.add_argument("modules", nargs="+")
    absent.set_defaults(run=check_absent)
    imports = commands.add_parser("imports")
    imports.add_argument("modules", nargs="+")
    imports.set_defaults(run=check_imports)
    reports = commands.add_parser("reports-commit")
    reports.add_argument("module")
    reports.add_argument("commit")
    reports.set_defaults(run=check_reports_commit)
    agrees = commands.add_parser("lock-agrees")
    agrees.add_argument("lock")
    agrees.add_argument("reference")
    agrees.set_defaults(run=check_lock_agrees)
    versions = commands.add_parser("versions")
    versions.add_argument("names", nargs="+")
    versions.set_defaults(run=print_versions)
    provenance = commands.add_parser("provenance")
    provenance.add_argument("out")
    provenance.add_argument("fields", nargs="*")
    provenance.set_defaults(run=write_provenance)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
