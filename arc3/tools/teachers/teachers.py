"""The distillation teacher played through its provider's API: Kimi K3.1 Pro
on Moonshot's API (2026-10-27).

The seal runs it in the Flash-Next B200 column's shape, with every token
figure the harness acts on in Flash-Next's tokens (2026-10-27: the
traces are SFT input for Flash-Next). The prompt side goes through the seal's
estimate-mode bound, calibrated on Flash-Next's tokenizer; the reply side
(the per-call reply cap, the per-level budget) is counted in the teacher's
billed tokens, so it is set to Flash-Next's default divided by
``flash_tokens_per_token``.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class ApiTeacher:
    """A teacher played through its provider's API, with the key from the
    environment."""

    name: str
    server: str  # the backend class speaking the provider's API, a key of ``API_SERVERS``
    model: str  # the API's model id
    key_env: str  # the environment variable holding the API key
    effort: Optional[str]  # the API's reasoning level for a thinking call; None: the API's default
    play_sampling: dict[str, Any]
    images: bool
    # How far short of the previous prompt a call's cache hit may stop: the
    # provider's cache granularity, as measured.
    cache_slack_tokens: int
    max_context_tokens: int  # the API's window for the model (/v1/models)
    # The seal's estimate-mode accounting, in Flash-Next's tokens: the
    # rendered request's characters per Flash-Next token, and a board image's
    # Flash-Next tokens (``SealAgent.bound_chars_per_token``,
    # ``image_tokens_override``).
    bound_chars_per_token: float
    image_tokens: int
    # Flash-Next tokens per teacher-billed token on the teacher's own replies:
    # the reply cap and the level budget, which count billed tokens, are
    # Flash-Next's divided by it.
    flash_tokens_per_token: float
    play_slots: int  # games played at once, under the provider's rate limits

    def backend(self, *, thinking: bool = True, streaming: bool = True) -> Any:
        """A started backend: reasoning on, or off for a direct answer;
        streaming off for a traced run (tracing reads the whole completion).
        The key is read from ``key_env`` when the client is built, never
        stored."""
        module, name = API_SERVERS[self.server].rsplit(".", 1)
        server = getattr(importlib.import_module(module), name)
        return server(
            model=self.model,
            api_key_env=self.key_env,
            reasoning_effort=self.effort if thinking else "none",
            streaming=streaming,
        )


#: The backend class for each provider's API, imported when a backend is built.
API_SERVERS = {"kimi": "ullm.backends.kimi.KimiServer"}


API_TEACHERS: dict[str, ApiTeacher] = {
    # Kimi K3.1 Pro through Moonshot's API, released 2026-10-23; open weights
    # announced for 2026-11-04.
    "kimi-k3.1-pro": ApiTeacher(
        name="kimi-k3.1-pro",
        server="kimi",
        model="kimi-k3.1-pro",
        key_env="KIMI_API_KEY",
        # The API's default, which reasons most: xhigh and max are taken as
        # it, low to high reason less (2026-10-27).
        effort=None,
        play_sampling={},  # fixed by the API: temperature 1, top_p 0.95
        # 200 base64 images a request taken, and images keep the cache
        # (2026-10-27).
        images=True,
        # Hits cover the previous prompt in 2,048-token blocks, never its
        # reply (2026-10-27).
        cache_slack_tokens=2048,
        max_context_tokens=1_048_576,  # /v1/models
        # Flash-Next's own rate on seal traffic (the MiMo smoke game's
        # conversation rendered by its chat template, 6k-57k-token prefixes):
        # 3.67-4.06 bound characters a token, 3.7 at playing lengths; a 512 px
        # board is 256 image pads plus its two marks (2026-10-27). Kimi bills
        # fewer tokens than this for the same text (3.92-4.32), but an image
        # at roughly 560-630 (ball_bounce's last prompt: 93 images, billed
        # 34,525 over the estimate), so image-heavy games bill past the
        # window: the seal holds the bills to the API's (``max_context_tokens``).
        bound_chars_per_token=3.7,
        image_tokens=258,
        # Kimi's replies in its four-game smoke (166, 110,680 billed tokens)
        # are 116,644 Flash-Next tokens rendered as Flash-Next turns: 1.054
        # (median 1.026, p10-p90 0.91-1.09; 2026-10-27).
        flash_tokens_per_token=1.054,
        # Sixteen games at once (2026-10-27). The account takes 50
        # requests at once, 200 a minute and 3M tokens a minute; 20 games
        # sending ~100M prompt tokens an hour drew hours of refusals after an
        # hour and a half, which the backend waits out (README.md, "The
        # API").
        play_slots=16,
    ),
}
