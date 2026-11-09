"""Image stress test: the ways a serving stack can lose or swap the images
in a conversation, heavier variants, and an image held only in the cache.

Every image is a solid square of one of four colours the model has to name;
every conversation opens with a fresh random tag, so nothing hits the prefix
cache by accident. The cases:

- A: one image a request, cold, in parallel;
- B: two images in one request, cold, one request at a time;
- C: B's requests again, their prefixes cached through the images;
- D: two images in one request, cold, in parallel;
- E: conversations in parallel, each turn adding a new image and asking its
  colour, past reasoning sent back (a stack that mixes images up names an
  earlier image's colour for the new one);
- F: one conversation of six image turns, then one question naming all six
  colours in order;
- G: conversations in parallel, thinking off: turn 1 shows an image and asks
  a sum, telling the model not to mention the image in its answer (passing
  needs the sum and no colour); turn 2 asks the image's colour, which only
  the cached image can answer. The instruction stops short of "in any way",
  which a model can read as forbidding turn 2's answer too.

Against an API teacher (``teachers.API_TEACHERS``), its key read from the
environment:

    uv run python tools/teachers/image_stress.py --api kimi-k3.1-pro [--parallel N] [--size PX] [--seed N]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import time
from typing import Any, Optional

import numpy as np
import ullm.backend_matrix
import ullm.conv

COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}


#: The squares' side in pixels unless ``--size`` says; the seal renders a
#: 64x64 board at 512.
DEFAULT_SIZE = 128


def square(name: str, size: int) -> ullm.conv.ImageContent:
    """A solid square of colour ``name``, ``size`` pixels a side."""
    data = np.zeros((size, size, 3), dtype=np.uint8)
    data[:, :] = COLOURS[name]
    return ullm.conv.ImageContent(data=data)


def named(answer: str) -> list[str]:
    """The colour words in ``answer``, in order (markdown and punctuation aside)."""
    words = "".join(c if c.isalpha() else " " for c in answer.lower()).split()
    return [w for w in words if w in COLOURS]


def prompt(*parts: Any) -> ullm.conv.UserPrompt:
    """A user prompt of ``parts`` in order: strings as text, the rest as given."""
    elements: list[ullm.conv.Content] = [ullm.conv.StringContent(text=p) if isinstance(p, str) else p for p in parts]
    return ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=elements))


@dataclasses.dataclass
class Target:
    """What the cases run against: a backend for thinking calls, one for
    non-thinking calls (the same one where thinking is closed by injection),
    and that injection."""

    name: str
    backend: Any
    no_thinking_backend: Any
    no_thinking_injection: Optional[str]


class Stress:
    """The cases, each a fresh conversation (or several) against one target."""

    def __init__(self, target: Target, rng: random.Random, max_tokens: int, size: int) -> None:
        self.target = target
        self.rng = rng
        self.max_tokens = max_tokens
        self.size = size

    def square(self, name: str) -> ullm.conv.ImageContent:
        """A square of colour ``name`` at the run's size."""
        return square(name, self.size)

    def manager(self, *, thinking: bool = True) -> Any:
        """A fresh conversation, past reasoning sent back, thinking on or
        closed."""
        backend = self.target.backend if thinking else self.target.no_thinking_backend
        tm = ullm.backend_matrix.make_tm(backend, max_tokens=self.max_tokens)
        tm.response_manager.sampling_params.send_back_reasoning = "all"
        if not thinking and self.target.no_thinking_injection is not None:
            tm.response_manager.sampling_params.response_injection = self.target.no_thinking_injection
        return tm

    async def ask(self, tm: Any, user: ullm.conv.UserPrompt) -> tuple[str, Any]:
        """``user`` into the conversation; the answer's text and its usage."""
        tm.conversation.elements.append(user)
        if tm.state != "ready":
            tm.rearm()
        await tm.run()
        response = tm.conversation.last_turn.last_response
        return response.final_content.to_string().strip(), response.token_usage

    def tag(self) -> str:
        """A random opening, so no conversation shares a cached prefix."""
        return f"Session {self.rng.randint(10**8, 10**9)}."

    async def one(self) -> tuple[bool, str]:
        """Case A: one image, its colour named."""
        colour = self.rng.choice(list(COLOURS))
        answer, _ = await self.ask(
            self.manager(), prompt(f"{self.tag()} What colour is this picture? One word.", self.square(colour))
        )
        return named(answer)[:1] == [colour], f"{colour} -> {answer!r}"

    def two_prompt(self) -> tuple[list[str], ullm.conv.UserPrompt]:
        """Two different colours and the prompt asking both, in order."""
        colours = self.rng.sample(list(COLOURS), 2)
        return colours, prompt(
            f"{self.tag()} Picture A:",
            self.square(colours[0]),
            "Picture B:",
            self.square(colours[1]),
            "Name the colour of picture A, then of picture B, as two words.",
        )

    async def two(self, colours: list[str], user: ullm.conv.UserPrompt) -> tuple[bool, str]:
        """Cases B to D: both colours named, in order."""
        answer, usage = await self.ask(self.manager(), user)
        said = named(answer)
        ok = len(said) >= 2 and [said[0], said[-1]] == colours
        return ok, f"{'/'.join(colours)} -> {answer!r} cached={usage.cached_tokens}/{usage.prompt_tokens}"

    async def turns(self, count: int) -> list[tuple[bool, str]]:
        """Case E: ``count`` turns, each a new image whose colour is asked."""
        tm = self.manager()
        out: list[tuple[bool, str]] = []
        tag = self.tag()
        for turn in range(count):
            colour = self.rng.choice(list(COLOURS))
            text = (
                f"{tag} What colour is this picture? One word."
                if turn == 0
                else "And what colour is this new picture? One word."
            )
            answer, usage = await self.ask(tm, prompt(text, self.square(colour)))
            ok = named(answer)[:1] == [colour]
            out.append((ok, f"turn {turn}: {colour} -> {answer!r} cached={usage.cached_tokens}/{usage.prompt_tokens}"))
        return out

    async def album(self, count: int) -> tuple[bool, str]:
        """Case F: ``count`` image turns, then every colour in order."""
        tm = self.manager()
        colours = [self.rng.choice(list(COLOURS)) for _ in range(count)]
        tag = self.tag()
        for index, colour in enumerate(colours, start=1):
            await self.ask(tm, prompt(f"{tag} This is picture {index}. Reply with just: OK", self.square(colour)))
        answer, usage = await self.ask(
            tm, prompt(f"Name the colours of pictures 1 to {count}, in order, as {count} words separated by spaces.")
        )
        said = named(answer)
        return said == colours, f"{' '.join(colours)} -> {answer!r} cached={usage.cached_tokens}/{usage.prompt_tokens}"

    async def unmentioned(self) -> list[tuple[bool, str]]:
        """Thinking off: a sum beside an image it must not mention, then the image's colour."""
        tm = self.manager(thinking=False)
        colour = self.rng.choice(list(COLOURS))
        a, b = self.rng.randint(11, 49), self.rng.randint(11, 49)
        first, _ = await self.ask(
            tm,
            prompt(
                f"{self.tag()} Here is a picture. Do not mention the picture, its colour or anything about it in "
                f"your answer to this question. What is {a} + {b}? Reply with the number only.",
                self.square(colour),
            ),
        )
        second, usage = await self.ask(tm, prompt("What colour was the picture I showed you? One word."))
        return [
            (str(a + b) in first and not named(first), f"sum {a}+{b} beside {colour} -> {first!r}"),
            (
                named(second)[:1] == [colour],
                f"{colour} -> {second!r} cached={usage.cached_tokens}/{usage.prompt_tokens}",
            ),
        ]


def report(case: str, results: list[tuple[bool, str]]) -> int:
    """Print a case's hits and each miss; the number of misses."""
    hits = sum(ok for ok, _ in results)
    print(f"{case}: {hits}/{len(results)}", flush=True)
    for ok, note in results:
        if not ok:
            print(f"    MISS {note}", flush=True)
    return len(results) - hits


async def run_cases(target: Target, parallel: int, max_tokens: int, seed: int, size: int) -> int:
    """Every case against ``target``; the misses in all."""
    stress = Stress(target, random.Random(seed), max_tokens, size)
    started = time.time()
    print(f"{target.name}: {parallel} in parallel, seed {seed}", flush=True)
    misses = report("A one image, cold, parallel", list(await asyncio.gather(*(stress.one() for _ in range(parallel)))))
    prompts = [stress.two_prompt() for _ in range(5)]
    misses += report("B two images, cold, serial", [await stress.two(c, p) for c, p in prompts])
    misses += report("C B again, cached through the images", [await stress.two(c, p) for c, p in prompts])
    misses += report(
        "D two images, cold, parallel",
        list(await asyncio.gather(*(stress.two(*stress.two_prompt()) for _ in range(parallel)))),
    )
    conversations = await asyncio.gather(*(stress.turns(3) for _ in range(parallel)))
    misses += report(
        f"E a new image each turn, {parallel} conversations in parallel", [r for c in conversations for r in c]
    )
    misses += report("F six image turns, then all six in order", [await stress.album(6)])
    unmentioned = await asyncio.gather(*(stress.unmentioned() for _ in range(parallel)))
    misses += report("G thinking off: the sum without the image", [c[0] for c in unmentioned])
    misses += report("G thinking off: then the unmentioned image's colour", [c[1] for c in unmentioned])
    print(f"{target.name}: misses {misses} ({time.time() - started:.0f} s)", flush=True)
    return misses


async def run_api(name: str, parallel: int, max_tokens: int, seed: int, size: int) -> int:
    """An API teacher (``teachers.API_TEACHERS``): thinking off is the API's own switch."""
    import teachers

    teacher = teachers.API_TEACHERS[name]
    backend, no_thinking = teacher.backend(), teacher.backend(thinking=False)
    try:
        return await run_cases(Target(name, backend, no_thinking, None), parallel, max_tokens, seed, size)
    finally:
        await no_thinking.close()
        await backend.close()


def main() -> None:
    """The command line: the module docstring's usage; exits 1 on any miss."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--api", required=True, help="an API teacher (teachers.API_TEACHERS)")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="the squares' side in pixels")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else time.time_ns() % 10**9
    misses = asyncio.run(run_api(args.api, args.parallel, args.max_tokens, seed, args.size))
    raise SystemExit(1 if misses else 0)


if __name__ == "__main__":
    main()
