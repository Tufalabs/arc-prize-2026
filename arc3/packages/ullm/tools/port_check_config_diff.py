"""Diff the engine configuration two runs actually ran with.

A port's first question is what the new engine changed under it, and the
answer is in the configuration each run recorded, not in the connector: a
default that moved is invisible to every counter measuring its consequences.
This reads the ``AsyncEngineArgs`` dump TAAF writes at the head of a run's
stdout and prints only the arguments that differ.

    uv run python packages/ullm/tools/port_check_config_diff.py <job_dir_a> <job_dir_b>
"""

from __future__ import annotations

import pathlib
import re
import sys

# The dump is one long repr; each argument is `name=value` with values that
# may themselves contain commas inside brackets, quotes or braces.
ARG = re.compile(r"(\w+)=('[^']*'|\[[^\]]*\]|\{[^}]*\}|[^,()]+?)(?=, \w+=|\)$|$)")


def engine_args(job_dir: pathlib.Path) -> dict[str, str]:
    """The engine arguments a run recorded, by name."""
    text = (job_dir / "stdout.log").read_text(errors="replace")[:200_000]
    match = re.search(r"AsyncEngineArgs\((.*)", text)
    if match is None:
        raise SystemExit(f"no AsyncEngineArgs dump in {job_dir}/stdout.log")
    out: dict[str, str] = {}
    for name, value in ARG.findall(match.group(1)[:20_000]):
        out.setdefault(name, value.strip())
    return out


def main() -> None:
    """Print the arguments that differ between two runs."""
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a_dir, b_dir = (pathlib.Path(x) for x in sys.argv[1:3])
    a, b = engine_args(a_dir), engine_args(b_dir)
    differing = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print(f"{a_dir.name}  vs  {b_dir.name}")
    print(f"{len(a)} and {len(b)} arguments; {len(differing)} differ\n")
    for key in differing:
        print(f"  {key}")
        print(f"      {a_dir.name}: {a.get(key, '<absent>')}")
        print(f"      {b_dir.name}: {b.get(key, '<absent>')}")


if __name__ == "__main__":
    main()
