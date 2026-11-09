"""Load TAAF raw traces into TRL-ready datasets.

TAAF (`tufa-arc-agi-framework`) writes each run as a folder: `trace.json`
(metadata) + `steps.jsonl` (one `TraceStep` per line) + `media/` files +
`_SUCCESS`. Its `messages` use a vendored **OpenAI Chat Completions subset** and
hold the *prompt only*; the completion is stored separately as `output_text`
(requirement R5.32). Per step TAAF also stores `input_token_ids` and
`output_token_ids` — these are OPTIONAL and may be empty.

Two training modes:

- **retemplate** (default for vision): normalize each step to an HF-native
  per-action record (`prompt` + single-turn `completion` + `images`) and let the
  target model's chat template re-tokenize. Model-portable.
- **token_exact**: when token ids are present, reproduce the exact run by
  building `input_ids = input_token_ids + output_token_ids` and masking the
  prompt (`labels = [-100]*len(input) + output_token_ids`). No re-templating.
  Vision features are NOT reconstructed (TAAF stores media files + preprocessing
  hashes, not pixel arrays), so token_exact is for text-only steps; mixing it
  with image steps is rejected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sft.dataset import _project_messages, records_to_dataset  # noqa: PLC2701

# How retemplate handles a step whose completion text is empty while the backend
# generated tokens — the fingerprint of a pre-R5.37 trace that dropped the
# completion at capture. Default "error" fails loudly so broken data never
# trains silently again; "skip" drops the step; "keep" restores legacy behavior.
_ON_EMPTY_COMPLETION = os.environ.get("SFT_ON_EMPTY_COMPLETION", "error").strip().lower()
_EMPTY_COMPLETION_POLICIES = frozenset({"error", "skip", "keep"})

# TAAF content-part types (trace_openai_schema.ALLOWED_CONTENT_PART_TYPES,
# schema openai.chat.messages.subset). Media parts carry a relative `path`
# anywhere in the part (TAAF nests it, e.g. image_url={"path": "media/x.png"}).
_IMAGE_PART_TYPES = {"image_url", "input_image", "image"}
_VIDEO_PART_TYPES = {"video_url", "input_video", "video"}
MODES = ("auto", "retemplate", "token_exact")


def _read_steps(trace_dir: Path) -> list[dict[str, Any]]:
    steps_file = trace_dir / "steps.jsonl"
    if not steps_file.is_file():
        raise FileNotFoundError(f"not a TAAF trace folder (no steps.jsonl): {trace_dir}")
    steps = [
        json.loads(line)
        for line in steps_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not steps:
        raise ValueError(f"{steps_file}: no steps found")
    return steps


# --- format normalization (OpenAI subset -> HF-native) -----------------------


def _extract_media_path(obj: Any) -> str | None:
    """First relative media path found anywhere in a content part.

    The path may be nested under a `path` key or, per TAAF's OpenAI-subset
    emission (R5.31: parts conform to the vendored OpenAI image schema, whose
    image part is `image_url.url`), under a relative `url`:
    {"type": "image_url", "image_url": {"url": "media/frame.png"}}. Absolute
    `data:` / `scheme://` urls are inline/remote, not trace files, so skip them.
    """
    if isinstance(obj, dict):
        path = obj.get("path")
        if isinstance(path, str):
            return path
        url = obj.get("url")
        if isinstance(url, str) and not url.startswith("data:") and "://" not in url:
            return url
        for value in obj.values():
            found = _extract_media_path(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _extract_media_path(value)
            if found is not None:
                return found
    return None


def _part_media_kind(part: Any) -> str | None:
    """'image' / 'video' / None, by the part's declared type."""
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if part_type in _IMAGE_PART_TYPES:
        return "image"
    if part_type in _VIDEO_PART_TYPES:
        return "video"
    return None


def _native_part(part: Any) -> Any:
    """Rewrite a TAAF media part to HF-native {"type": image|video, "path": …}."""
    kind = _part_media_kind(part)
    if kind is None:
        return part
    path = _extract_media_path(part)
    if path is None:
        # An unresolvable media source (e.g. an inline data: URL) would pass
        # through un-lifted and silently vanish from the images column.
        raise ValueError(f"{kind} part with no resolvable trace-relative path: {str(part)[:200]}")
    return {"type": "image" if kind == "image" else "video", "path": path}


def _native_message(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        out["content"] = [_native_part(part) for part in content]
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        converted = []
        for call in tool_calls:
            call = dict(call) if isinstance(call, dict) else call
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                fn = dict(fn)
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (TypeError, ValueError):
                    # Best-effort: keep the raw string, but make the resulting
                    # heterogeneous prompt shape visible instead of silent.
                    print(
                        "[taaf_adapter] WARNING: tool-call arguments are not JSON; keeping raw string "
                        f"({str(fn.get('arguments'))[:80]!r})"
                    )
                call["function"] = fn
            converted.append(call)
        out["tool_calls"] = converted
    return out


def _message_media_kinds(messages: list[dict[str, Any]]) -> set[str]:
    """Set of media kinds ('image'/'video') referenced in message content parts."""
    kinds: set[str] = set()
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                kind = _part_media_kind(part)
                if kind:
                    kinds.add(kind)
    return kinds


def _step_has_media(step: dict[str, Any]) -> bool:
    return bool(step.get("media")) or bool(_message_media_kinds(step.get("messages", [])))


def _step_has_tokens(step: dict[str, Any]) -> bool:
    return bool(step.get("input_token_ids")) and bool(step.get("output_token_ids"))


# --- record builders ---------------------------------------------------------
# Input-token fidelity is verified at CAPTURE time, not here: the trace writer
# (ARC3-Inference taaf_trace, R12.02) re-renders each step's messages — with
# the request's tools, which this adapter never sees — through the serving
# template and records input_token_reconstruction_status. A training-side
# re-render can't reproduce the served prompt (no tools), so token_exact
# trusts the stored ids and their recorded status.


def taaf_trace_to_records(trace_dir: str | Path, *, on_empty_completion: str | None = None) -> list[dict[str, Any]]:
    """retemplate mode: TAAF steps -> HF-native per-action records.

    The completion is the step's ``output_text`` — which, for R5.37 (v2) traces,
    is the complete generation (reasoning + tool-call markup), retokenized by the
    target model's chat template. A step with an empty ``output_text`` is a
    dropped completion — the capture-bug class that trains empty targets —
    whether or not token ids were recorded; ``on_empty_completion`` decides what
    to do (default from ``SFT_ON_EMPTY_COMPLETION``: ``error`` | ``skip`` |
    ``keep``). ``skip`` counts and prints what it dropped.
    """
    trace_dir = Path(trace_dir)
    policy = (on_empty_completion or _ON_EMPTY_COMPLETION).strip().lower()
    if policy not in _EMPTY_COMPLETION_POLICIES:
        raise ValueError(
            "on_empty_completion must be one of "
            f"{sorted(_EMPTY_COMPLETION_POLICIES)}, got {policy!r}"
        )
    records: list[dict[str, Any]] = []
    skipped_empty = 0
    for step in _read_steps(trace_dir):
        messages = step.get("messages", [])
        if "video" in _message_media_kinds(messages):
            raise ValueError(
                f"{trace_dir}: step {step.get('step_index')} has video content; "
                "SFT v0 supports text + images only"
            )
        # `or ""`: a JSON-null output_text must not become the literal "None".
        output_text = str(step.get("output_text") or "")
        n_out_tokens = len(step.get("output_token_ids") or [])
        if not output_text.strip():
            where = f"{trace_dir}: step {step.get('step_index')}"
            evidence = f"{n_out_tokens} output_token_ids" if n_out_tokens else "no output_token_ids (token capture off)"
            if policy == "error":
                raise ValueError(
                    f"{where} has an empty completion ({evidence}) — a dropped completion "
                    "trains the model to emit nothing. Repair the dataset before training, or "
                    "set SFT_ON_EMPTY_COMPLETION=skip to drop such steps."
                )
            if policy == "skip":
                skipped_empty += 1
                continue
            # "keep": fall through and emit the empty completion (legacy behavior).
        native_prompt = [_native_message(m) for m in messages]
        images: list[Any] = []
        prompt = _project_messages(native_prompt, base_dir=trace_dir, images_out=images)
        records.append(
            {
                "prompt": prompt,
                "completion": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": output_text}],
                    }
                ],
                "images": images,
            }
        )
    if skipped_empty:
        print(f"[taaf_adapter] {trace_dir}: skipped {skipped_empty} empty-completion step(s) (SFT_ON_EMPTY_COMPLETION=skip)")
    return records


def taaf_trace_to_tokenized_records(trace_dir: str | Path) -> list[dict[str, Any]]:
    """token_exact mode: TAAF steps -> pre-tokenized {input_ids, labels, ...}."""
    trace_dir = Path(trace_dir)
    records: list[dict[str, Any]] = []
    for step in _read_steps(trace_dir):
        if not _step_has_tokens(step):
            raise ValueError(
                f"{trace_dir}: step {step.get('step_index')} lacks token ids; "
                "token_exact mode requires input_token_ids and output_token_ids"
            )
        if _step_has_media(step):
            raise ValueError(
                f"{trace_dir}: step {step.get('step_index')} has media; "
                "token_exact does not reconstruct vision features — use retemplate"
            )
        prompt_ids = [int(t) for t in step["input_token_ids"]]
        completion_ids = [int(t) for t in step["output_token_ids"]]
        input_ids = prompt_ids + completion_ids
        records.append(
            {
                "input_ids": input_ids,
                "labels": ([-100] * len(prompt_ids)) + completion_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return records


# --- mode resolution + dataset build -----------------------------------------


def discover_trace_dirs(root: str | Path) -> list[Path]:
    """Return the trace folders under `root`.

    If `root` is itself a trace folder (`steps.jsonl` present) returns `[root]`;
    otherwise every immediate subfolder finalized with `_SUCCESS`. Both forms
    require `_SUCCESS` — a crashed/partially-written trace must not train.
    """
    root = Path(root)
    if (root / "steps.jsonl").is_file():
        if not (root / "_SUCCESS").is_file():
            raise FileNotFoundError(f"{root} has steps.jsonl but no _SUCCESS marker; not a finalized trace")
        return [root]
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "_SUCCESS").is_file()]
    if not dirs:
        raise FileNotFoundError(f"no finalized TAAF traces under {root}")
    return dirs


def _resolve_mode(dirs: list[Path], mode: str) -> str:
    """Resolve 'auto' by scanning steps trace-by-trace (memory = one trace).

    'auto' picks token_exact only when every step has token ids and none has
    images, so the scan can stop early: the first image (or first step
    without token ids) settles it as retemplate. Explicit retemplate needs no
    scan at all; explicit token_exact fails fast on the first offending step.
    """
    if mode == "retemplate":
        return mode
    all_tokens = True
    any_media = False
    for trace_dir in dirs:
        for step in _read_steps(trace_dir):
            all_tokens = all_tokens and _step_has_tokens(step)
            any_media = any_media or _step_has_media(step)
        if mode == "token_exact":
            if not all_tokens:
                raise ValueError("token_exact requested but some steps lack token ids")
            if any_media:
                raise ValueError(
                    "token_exact requested but some steps contain media; "
                    "vision features cannot be reconstructed from token ids"
                )
        elif any_media or not all_tokens:
            return "retemplate"
    return "token_exact" if (all_tokens and not any_media) else "retemplate"


def iter_trace_records(dirs: list[Path], resolved: str):
    """Yield TRL-ready records trace by trace (never all traces in RAM)."""
    for trace_dir in dirs:
        if resolved == "token_exact":
            yield from taaf_trace_to_tokenized_records(trace_dir)
        else:
            yield from taaf_trace_to_records(trace_dir)


def build_dataset_from_trace_dirs(
    dirs: list[Path], *, mode: str = "auto", work_dir: str | Path | None = None
):
    """Explicit list of TAAF trace folders -> (datasets.Dataset, resolved_mode).

    Mode is resolved globally so every trace is built the same way. With
    `work_dir=None` every record (decoded PIL images included) is materialized
    in RAM before Arrow conversion -- fine for small selections and tests.
    Passing `work_dir` streams the build instead: `Dataset.from_generator`
    writes records to Arrow under `work_dir` one writer-batch at a time and
    returns a memory-mapped dataset, so peak RSS stays near one trace
    regardless of selection size (full-run selections OOM the in-RAM path).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not dirs:
        raise ValueError("no trace folders to build a dataset from")
    resolved = _resolve_mode(dirs, mode)

    if work_dir is None:
        records = list(iter_trace_records(dirs, resolved))
        if not records:
            raise ValueError(
                "selected traces produced zero training records "
                "(all steps may have been skipped as empty completions)"
            )
        return records_to_dataset(records), resolved

    from datasets import Dataset, Image as HFImage, Sequence

    # Schema template from the first non-empty trace: `skip` policy can remove
    # every step in an earlier trace. from_generator otherwise
    # infers features per writer batch, and an image-less first batch would
    # type `images` inconsistently with later PIL-carrying batches.
    probe_records: list[dict[str, Any]] = []
    for trace_dir in dirs:
        probe_records = list(iter_trace_records([trace_dir], resolved))
        if probe_records:
            break
    if not probe_records:
        raise ValueError(
            "selected traces produced zero training records "
            "(all steps may have been skipped as empty completions)"
        )
    probe = records_to_dataset(probe_records)
    features = probe.features.copy()
    if "images" in features:
        # The probe trace itself may be text-only, which would type `images`
        # as Sequence(null) and fail Arrow generation on the first
        # image-bearing trace later in the list. The column always holds PIL
        # images when non-empty -- pin the feature unconditionally.
        features["images"] = Sequence(HFImage())
    try:
        build_num_proc = int(os.environ.get("SFT_BUILD_NUM_PROC", "8"))
    except ValueError as exc:
        raise ValueError("SFT_BUILD_NUM_PROC must be a positive integer") from exc
    if build_num_proc <= 0:
        raise ValueError(f"SFT_BUILD_NUM_PROC must be positive, got {build_num_proc}")
    dataset = Dataset.from_generator(
        iter_trace_records,
        gen_kwargs={"dirs": dirs, "resolved": resolved},
        features=features,
        cache_dir=str(work_dir),
        # Modest batches keep any single Arrow chunk far from the int32
        # offset limit even with many images per row.
        writer_batch_size=100,
        # Shard the trace list across worker processes (the build is CPU+IO
        # bound: media decode + per-step normalization). Serial would take
        # hours on full-run selections; capped by the trace count.
        num_proc=min(build_num_proc, len(dirs)),
    )
    return dataset, resolved
