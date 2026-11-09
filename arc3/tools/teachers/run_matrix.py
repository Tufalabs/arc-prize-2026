"""Run ullm's live backend matrix against an API teacher
(``teachers.API_TEACHERS``), its key read from the environment.

    uv run python tools/teachers/run_matrix.py --api kimi-k3.1-pro [--images]
"""

from __future__ import annotations

import argparse
import asyncio

import teachers
import ullm.backend_matrix


async def run_api(name: str, images: bool) -> None:
    """The matrix over an API teacher, at its stated cache granularity."""
    teacher = teachers.API_TEACHERS[name]
    backend = teacher.backend()
    try:
        results = await ullm.backend_matrix.run_live_matrix(
            backend, name=name, images=images, cache_slack_tokens=teacher.cache_slack_tokens
        )
    finally:
        await backend.close()
    for row, result in results.items():
        print(f"{row}: {result}", flush=True)


def main() -> None:
    """The command line: the module docstring's usage."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--api", required=True, choices=sorted(teachers.API_TEACHERS), help="the API teacher")
    parser.add_argument("--images", action="store_true", help="the image rows too")
    args = parser.parse_args()
    asyncio.run(run_api(args.api, args.images))


if __name__ == "__main__":
    main()
