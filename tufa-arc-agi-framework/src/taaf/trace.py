"""Trace collection for training (R5.*).

``Trace`` / ``TraceStep`` are lightweight, JSON-serializable records a solver
compiles for an LLM-traced run; TAAF owns writing, validating, reading, and
exporting them (R5.01). Storage is one self-contained folder per run keyed by
``GameRun.run_id`` (R5.41). This module is pure-Python on purpose: it never
imports ``vllm`` / ``torch`` / HF / GPU libraries, so it cannot rerun
backend roundtrips — it validates the *contract* only (R5.62).
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import arcengine

from taaf import support

if TYPE_CHECKING:
    import taaf.game

# Fixed format identifiers (R5.21); ``message_format_source`` pins the OpenAI
# OpenAPI subset identifier to an immutable upstream commit (not vendored;
# subset conformance is owned by the capture adapter).
# v2 (R5.37/R5.38): ``output_text`` is the complete completion text (not a
# single parsed channel), ``output_message`` carries the provider's structured
# completion, and ``provider_validation`` carries output reconstruction
# evidence. v1 traces remain readable; validation accepts only v2 writes.
SCHEMA_VERSION = "taaf.trace.v2"
_READABLE_SCHEMA_VERSIONS = frozenset({"taaf.trace.v1", "taaf.trace.v2"})
MESSAGE_FORMAT_NAME = "openai-openapi-chat-subset"
MESSAGE_FORMAT_VERSION = "2.3.0"
OPENAI_OPENAPI_COMMIT = "5162af98d3147432c14680df789e8e12d4891e6b"
MESSAGE_FORMAT_SOURCE = f"openai/openai-openapi@{OPENAI_OPENAPI_COMMIT}"

# What a trace stores (R5.39): both representations, or the exact side alone.
CONTENT_FULL = "full"
CONTENT_IDS = "ids"
_CONTENTS = frozenset({CONTENT_FULL, CONTENT_IDS})

# Provider/backend evidence keys every step must carry (R5.31 / R5.38).
REQUIRED_PROVIDER_VALIDATION_KEYS = (
    "provider_name",
    "provider_version",
    "api_kind",
    "provider_model_id",
    "request_args",
    "validation_method",
    "original_payload_sha256",
    "reconstructed_payload_sha256",
    "input_token_reconstruction_status",
    "output_token_reconstruction_status",
    "media_reconstruction_status",
)

# Reconstruction-status fields; ``failed``/unknown rejected everywhere (R5.62).
# For OUTPUT (R5.38), ``mismatch`` and ``whitespace_mismatch`` are RECORDED
# evidence, not validity failures: ``output_text`` is the exact detokenized
# completion regardless, and ``output_message`` is a parsed view whose
# re-render can legitimately differ (tool-arg reformatting, whitespace).
# For INPUT there is no such authoritative fallback — a reconstructed-input
# mismatch means the trace is not reproducible for SFT, so R12.02 makes it a
# validation failure, never tolerated evidence.
ACCEPTED_RECONSTRUCTION_STATUSES = frozenset({"ok", "whitespace_mismatch", "mismatch", "unavailable", "not_applicable"})
_ACCEPTED_STATUSES_BY_FIELD: dict[str, frozenset[str]] = {
    "input_token_reconstruction_status": frozenset({"ok", "unavailable", "not_applicable"}),
    "output_token_reconstruction_status": ACCEPTED_RECONSTRUCTION_STATUSES,
    "media_reconstruction_status": ACCEPTED_RECONSTRUCTION_STATUSES,
}

# Filenames in a raw trace folder (R5.41).
TRACE_JSON = "trace.json"
STEPS_JSONL = "steps.jsonl"
MEDIA_DIR = "media"
SUCCESS_MARKER = "_SUCCESS"

# Default retained-bundle filename (R5.49).
BUNDLE_FILENAME = "trace_bundle.tar.zst"
MANIFEST_FILENAME = "trace_manifest.json"


# --- action <-> json (mirrors GameRun.to_json_dict's projection) -------------


def _action_to_json(action: arcengine.ActionInput | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {"id": action.id.name, "data": dict(action.data)}


def _action_from_json(d: dict[str, Any] | None) -> arcengine.ActionInput | None:
    if d is None:
        return None
    return arcengine.ActionInput(id=arcengine.GameAction[d["id"]], data=dict(d["data"]))


# --- dataclasses -------------------------------------------------------------


@dataclass
class MediaRef:
    """A media item attached to a step (R5.31 / R5.44).

    Fields:

    - ``media_id``: stable id for this item within the trace.
    - ``path``: location **relative to the trace folder**, also used by the
      corresponding ``messages`` content part.
    - ``role``: why the item is attached (initial roles ``current_frame`` /
      ``animation``; more may be added).
    """

    media_id: str
    path: str
    role: str
    # In-memory only: the exact bytes to write under the trace folder. Excluded
    # from JSON (the on-disk ref carries only id/path/role) and from equality.
    source_bytes: bytes | None = field(default=None, repr=False, compare=False)

    def to_json_dict(self) -> dict[str, Any]:
        return {"media_id": self.media_id, "path": self.path, "role": self.role}

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> MediaRef:
        return cls(media_id=d["media_id"], path=d["path"], role=d["role"])


@dataclass
class TraceStep:
    """One model call or solver decision (R5.31).

    ``messages`` is the portable SFT source (normalized OpenAI OpenAPI subset);
    ``input_token_ids`` / ``output_text`` / ``output_token_ids`` preserve the
    exact prompt/completion boundary actually run (R5.32). ``action`` /
    ``action_index`` link the step back to ``GameRun.history`` when it produced
    a recorded move (both ``None`` for a no-move planning/decision step).
    ``provider_validation`` carries provider provenance + reconstruction
    evidence.

    Fields:

    - ``run_id`` / ``step_index``: authoritative membership + ordering (R5.33).
    - ``messages``: normalized structured chat input.
    - ``input_token_ids`` / ``output_token_ids``: exact backend token ids.
    - ``output_text``: the COMPLETE generated/selected text (R5.32 / R5.37) —
      including reasoning spans and tool-call markup the provider's response
      parsers may have split off, special tokens kept. Never a single parsed
      channel. Non-empty whenever ``output_token_ids`` is non-empty.
    - ``output_message``: the provider's structured completion message
      (role ``assistant``; reasoning/content/tool_calls exactly as returned,
      unnormalized), or ``None`` when the backend returned plain text only
      (R5.37). Lets consumers re-render the completion through another
      model's chat template.
    - ``logprobs``: optional per-output-token alternatives, aligned to
      ``output_token_ids``; ``None`` when unavailable.
    - ``sampling_params`` / ``reward_hints`` / ``extra``: JSON objects.
    - ``media``: media references, empty if none.
    - ``provider_validation``: provider/backend evidence (keys per
      ``REQUIRED_PROVIDER_VALIDATION_KEYS``).
    - ``action_index``: nullable index into ``GameRun.history``.
    - ``action``: ``arcengine.ActionInput`` selected, or ``None``.
    - ``uncached_input_tokens``: uncached input token count, 0 if unknown.
    """

    run_id: str
    step_index: int
    messages: list[dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    input_token_ids: list[int] = field(default_factory=lambda: list[int]())
    output_text: str = ""
    output_message: dict[str, Any] | None = None
    output_token_ids: list[int] = field(default_factory=lambda: list[int]())
    logprobs: list[Any] | None = None
    sampling_params: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    media: list[MediaRef] = field(default_factory=lambda: list[MediaRef]())
    provider_validation: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    action_index: int | None = None
    action: arcengine.ActionInput | None = None
    uncached_input_tokens: int = 0
    reward_hints: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "created_at": self.created_at,
            "messages": self.messages,
            "input_token_ids": list(self.input_token_ids),
            "output_text": self.output_text,
            "output_message": dict(self.output_message) if self.output_message is not None else None,
            "output_token_ids": list(self.output_token_ids),
            "logprobs": self.logprobs,
            "sampling_params": dict(self.sampling_params),
            "media": [m.to_json_dict() for m in self.media],
            "provider_validation": dict(self.provider_validation),
            "action_index": self.action_index,
            "action": _action_to_json(self.action),
            "uncached_input_tokens": self.uncached_input_tokens,
            "reward_hints": dict(self.reward_hints),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> TraceStep:
        return cls(
            run_id=d["run_id"],
            step_index=d["step_index"],
            messages=d["messages"],
            created_at=d.get("created_at", ""),
            input_token_ids=list(d.get("input_token_ids") or []),
            output_text=d.get("output_text", ""),
            output_message=d.get("output_message"),
            output_token_ids=list(d.get("output_token_ids") or []),
            logprobs=d.get("logprobs"),
            sampling_params=dict(d.get("sampling_params") or {}),
            media=[MediaRef.from_json_dict(m) for m in cast("list[dict[str, Any]]", d.get("media") or [])],
            provider_validation=dict(d.get("provider_validation") or {}),
            action_index=d.get("action_index"),
            action=_action_from_json(d.get("action")),
            uncached_input_tokens=d.get("uncached_input_tokens", 0),
            reward_hints=dict(d.get("reward_hints") or {}),
            extra=dict(d.get("extra") or {}),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class Trace:
    """Trace-level metadata + its steps (R5.21).

    ``to_json_dict`` emits only the metadata (written to ``trace.json``); steps
    are written one-per-line to ``steps.jsonl`` (R5.41). Run-level outcomes
    (score, levels, final state) are NOT copied here — they stay on ``GameRun``
    in ``benchmark.json`` and are recovered via ``run_id`` (R5.11).

    Fields:

    - ``run_id``: join key back to the ``GameRun`` (R5.11).
    - ``source_kind``: open string for why the trace was collected (R5.22).
    - ``total_uncached_input_tokens``: sum of uncached input tokens, 0 if
      unknown.
    - ``repo_name`` / ``repo_commit_hash``: source repo identity.
    - model/tokenizer/processor identity block: non-null strings identifying
      the stack that produced the trace.
    - ``message_format_*``: identity of the normalized message schema.
    - ``content``: ``"full"`` stores both representations; ``"ids"`` stores
      the exact side alone, with no ``messages``, ``output_text`` or
      ``output_message`` on any step (R5.39).
    - ``extra``: solver-specific metadata.
    - ``steps``: the ``TraceStep`` rows (in-memory; serialized separately).
    """

    run_id: str
    source_kind: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: str = SCHEMA_VERSION
    content: str = CONTENT_FULL
    total_uncached_input_tokens: int = 0
    repo_name: str = ""
    repo_commit_hash: str = ""
    model_id: str = ""
    model_revision: str = ""
    model_weight_hash: str = ""
    parent_model_revision: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    processor_id: str = ""
    processor_revision: str = ""
    vision_preprocessor_revision: str = ""
    message_format_name: str = MESSAGE_FORMAT_NAME
    message_format_version: str = MESSAGE_FORMAT_VERSION
    message_format_source: str = MESSAGE_FORMAT_SOURCE
    extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    steps: list[TraceStep] = field(default_factory=lambda: list[TraceStep]())

    # Fields carrying the model/tokenizer/processor identity (R5.21). Kept as a
    # tuple so validation can require all of them as non-null strings.
    _IDENTITY_FIELDS = (
        "model_id",
        "model_revision",
        "model_weight_hash",
        "parent_model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "processor_id",
        "processor_revision",
        "vision_preprocessor_revision",
    )

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "source_kind": self.source_kind,
            "n_steps": self.n_steps,
            "total_uncached_input_tokens": self.total_uncached_input_tokens,
            "repo_name": self.repo_name,
            "repo_commit_hash": self.repo_commit_hash,
        }
        for name in self._IDENTITY_FIELDS:
            d[name] = getattr(self, name)
        d["message_format_name"] = self.message_format_name
        d["message_format_version"] = self.message_format_version
        d["message_format_source"] = self.message_format_source
        d["content"] = self.content
        d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any], steps: list[TraceStep] | None = None) -> Trace:
        trace = cls(
            run_id=d["run_id"],
            source_kind=d["source_kind"],
            created_at=d.get("created_at", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            total_uncached_input_tokens=d.get("total_uncached_input_tokens", 0),
            repo_name=d.get("repo_name", ""),
            repo_commit_hash=d.get("repo_commit_hash", ""),
            message_format_name=d.get("message_format_name", MESSAGE_FORMAT_NAME),
            message_format_version=d.get("message_format_version", MESSAGE_FORMAT_VERSION),
            message_format_source=d.get("message_format_source", MESSAGE_FORMAT_SOURCE),
            content=d.get("content", CONTENT_FULL),
            extra=dict(d.get("extra") or {}),
            steps=steps if steps is not None else [],
        )
        for name in cls._IDENTITY_FIELDS:
            setattr(trace, name, d.get(name, ""))
        return trace


# --- validation (R5.62) ------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid trace: {msg}")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x)


def _is_int_list(x: Any) -> bool:
    return isinstance(x, list) and all(_is_int(i) for i in cast("list[Any]", x))


def _is_safe_relpath(x: Any) -> bool:
    if not _is_nonempty_str(x):
        return False
    p = Path(x)
    return not p.is_absolute() and ".." not in p.parts


def _message_media_paths(messages: Any) -> set[str]:
    """Relative media paths referenced by image parts in ``messages`` (R5.44).

    Skips inline ``data:`` URLs and absolute ``scheme://`` URLs — only
    folder-relative paths are trace media.
    """
    paths: set[str] = set()
    if not isinstance(messages, list):
        return paths
    for msg in cast("list[Any]", messages):
        if not isinstance(msg, dict):
            continue
        content = cast("dict[str, Any]", msg).get("content")
        if not isinstance(content, list):
            continue
        for part in cast("list[Any]", content):
            if not isinstance(part, dict):
                continue
            part_d = cast("dict[str, Any]", part)
            if part_d.get("type") != "image_url":
                continue
            image = part_d.get("image_url")
            if not isinstance(image, dict):
                continue
            url = cast("dict[str, Any]", image).get("url")
            if isinstance(url, str) and not url.startswith("data:") and "://" not in url:
                paths.add(url)
    return paths


def validate_trace(trace: Trace, game_run: taaf.game.GameRun) -> None:
    """Validate the trace contract against ``game_run`` (R5.62).

    Checks required fields, JSON-serializability, contiguous ``step_index``,
    matching ``run_id``, copied action fields against ``GameRun.history``,
    referenced media paths, normalized message structure, required
    ``provider_validation`` fields, accepted status values (an input-token
    ``mismatch`` is rejected — R12.02), and media-fingerprint structure when
    fingerprints are recorded. Rejects missing
    evidence and ``failed`` statuses. Does NOT rerun backend roundtrips (no
    ``vllm`` / ``torch`` / HF imports). Raises ``ValueError`` on the first
    violation.

    Validation runs over the JSON projection (``to_json_dict``) so it inspects
    exactly what gets persisted and so building that projection at all proves
    the trace is shaped for serialization (R5.34).
    """
    try:
        trace_d = trace.to_json_dict()
        step_ds = [s.to_json_dict() for s in trace.steps]
    except Exception as exc:  # noqa: BLE001 — any construction failure is a contract violation
        raise ValueError(f"invalid trace: cannot build JSON projection: {exc}") from exc

    _require(trace_d["schema_version"] == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION!r}")
    _require(
        trace_d["run_id"] == game_run.run_id,
        f"trace.run_id {trace_d['run_id']!r} != game_run.run_id {game_run.run_id!r}",
    )
    _require(_is_nonempty_str(trace_d["created_at"]), "trace.created_at must be a non-empty str")
    _require(_is_nonempty_str(trace_d["source_kind"]), "source_kind must be a non-empty str")
    tut = trace_d["total_uncached_input_tokens"]
    _require(_is_int(tut) and tut >= 0, "total_uncached_input_tokens must be a non-negative int")
    for name in Trace._IDENTITY_FIELDS:
        _require(isinstance(trace_d[name], str), f"trace.{name} must be a non-null str")
    for name in ("message_format_name", "message_format_version", "message_format_source"):
        _require(_is_nonempty_str(trace_d[name]), f"trace.{name} must be a non-empty str")
    _require(trace_d["content"] in _CONTENTS, f"trace.content must be one of {sorted(_CONTENTS)}")
    ids_only = trace_d["content"] == CONTENT_IDS

    n_history = len(game_run.history)
    for i, sd in enumerate(step_ds):
        where = f"step[{i}]"
        _require(
            sd["run_id"] == trace_d["run_id"], f"{where}.run_id {sd['run_id']!r} != trace.run_id {trace_d['run_id']!r}"
        )
        _require(sd["step_index"] == i, f"{where}.step_index {sd['step_index']} not contiguous (expected {i})")
        _require(_is_nonempty_str(sd["created_at"]), f"{where}.created_at must be non-empty str")
        msgs = sd["messages"]
        _require(isinstance(msgs, list), f"{where}.messages must be a list")
        for j, msg in enumerate(msgs):
            _require(isinstance(msg, dict) and "role" in msg, f"{where}.messages[{j}] must be a dict with a 'role'")
        _require(_is_int_list(sd["input_token_ids"]), f"{where}.input_token_ids must be a list of ints")
        _require(isinstance(sd["output_text"], str), f"{where}.output_text must be a str")
        _require(_is_int_list(sd["output_token_ids"]), f"{where}.output_token_ids must be a list of ints")
        # R5.37: output_text is the COMPLETE completion (a faithful rendering of
        # output_token_ids), not a single parsed channel. It may legitimately be
        # empty only when the generation itself was empty. We do NOT re-tokenize
        # here (R5.62 forbids backend imports) — full fidelity evidence is the
        # capture adapter's reconstruction statuses (below) — but an empty text
        # alongside non-empty output tokens is always a capture bug.
        if ids_only:
            # R5.39: the portable side is absent on every step, never partly
            # present, so a consumer can tell the two contents apart by the
            # header alone.
            _require(
                not msgs and not sd["output_text"] and sd["output_message"] is None,
                f"{where} carries messages, output_text or output_message in an ids-only trace",
            )
        else:
            _require(
                bool(sd["output_text"]) or not sd["output_token_ids"],
                f"{where}.output_text is empty but output_token_ids is non-empty",
            )
        om = sd["output_message"]
        _require(om is None or isinstance(om, dict), f"{where}.output_message must be None or a dict")
        if isinstance(om, dict):
            role = cast("dict[str, Any]", om).get("role")
            _require(
                role == "assistant",
                f"{where}.output_message.role must be 'assistant', got {role!r}",
            )
        lp = sd["logprobs"]
        _require(lp is None or isinstance(lp, list), f"{where}.logprobs must be None or a list")
        if isinstance(lp, list):
            # R5.31: the outer list aligns one-to-one to output_token_ids
            # (``None``, not an empty list, means alternatives unavailable).
            _require(
                len(cast("list[Any]", lp)) == len(sd["output_token_ids"]),
                f"{where}.logprobs length {len(cast('list[Any]', lp))} != len(output_token_ids) "
                f"{len(sd['output_token_ids'])}",
            )
        _require(isinstance(sd["sampling_params"], dict), f"{where}.sampling_params must be a dict")
        _require(isinstance(sd["reward_hints"], dict), f"{where}.reward_hints must be a dict")
        _require(isinstance(sd["extra"], dict), f"{where}.extra must be a dict")
        _require(_is_int(sd["uncached_input_tokens"]), f"{where}.uncached_input_tokens must be an int")

        # Media structure + path consistency (R5.44).
        ref_paths: set[str] = set()
        media = sd["media"]
        _require(isinstance(media, list), f"{where}.media must be a list")
        for k, m in enumerate(media):
            _require(isinstance(m, dict), f"{where}.media[{k}] must be a media ref dict")
            _require(
                _is_nonempty_str(m.get("media_id")) and _is_nonempty_str(m.get("role")),
                f"{where}.media[{k}] must have media_id and role",
            )
            _require(_is_safe_relpath(m.get("path")), f"{where}.media[{k}].path must be a safe trace-relative path")
            ref_paths.add(m["path"])
        missing = _message_media_paths(msgs) - ref_paths
        _require(not missing, f"{where} messages reference media paths not in media refs: {sorted(missing)}")

        _validate_provider_validation(sd["provider_validation"], where)

        # Action linkage (R5.31 / R5.62).
        ai = sd["action_index"]
        if ai is not None:
            _require(_is_int(ai) and 0 <= ai < n_history, f"{where}.action_index {ai} out of range [0, {n_history})")
            _require(sd["action"] is not None, f"{where} has action_index but action is None")
            idx = cast(int, ai)
            expected = _action_to_json(game_run.history[idx].action)
            _require(sd["action"] == expected, f"{where}.action does not match GameRun.history[{idx}].action")

        # JSON-serializability (R5.34) — must not depend on pickle.
        try:
            json.dumps(sd)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid trace: {where} is not JSON-serializable: {exc}") from exc

    try:
        json.dumps(trace_d)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid trace: trace metadata is not JSON-serializable: {exc}") from exc


def _validate_provider_validation(pv: Any, where: str) -> None:
    _require(isinstance(pv, dict), f"{where}.provider_validation must be a dict")
    for key in REQUIRED_PROVIDER_VALIDATION_KEYS:
        _require(key in pv, f"{where}.provider_validation missing required key {key!r}")
        # "reject missing evidence": required keys must not be None.
        _require(pv[key] is not None, f"{where}.provider_validation[{key!r}] must not be null (missing evidence)")
    for key, accepted in _ACCEPTED_STATUSES_BY_FIELD.items():
        status = pv[key]
        _require(
            status in accepted,
            f"{where}.provider_validation[{key!r}]={status!r} not in {sorted(accepted)} "
            "(a 'failed' or unknown status is rejected; an input-token mismatch is a "
            "validation failure per R12.02, not recorded evidence)",
        )
    _validate_media_fingerprints(pv, where)


def _validate_media_fingerprints(pv: dict[str, Any], where: str) -> None:
    """R5.62: validate media-fingerprint structure when fingerprints are
    recorded. Each record (R5.31) must carry ``shape`` (list of non-negative
    ints) and ``dtype``, plus at least one of ``values`` (exact small values)
    or ``sha256`` (hash of a large dense tensor). Absent/None fingerprints are
    fine — no producer is obliged to record them."""
    fingerprints = pv.get("media_fingerprints")
    if fingerprints is None:
        return
    _require(isinstance(fingerprints, dict), f"{where}.provider_validation.media_fingerprints must be a dict")
    for name, record in cast("dict[str, Any]", fingerprints).items():
        label = f"{where}.provider_validation.media_fingerprints[{name!r}]"
        _require(isinstance(record, dict), f"{label} must be a dict")
        record_d = cast("dict[str, Any]", record)
        shape = record_d.get("shape")
        _require(
            isinstance(shape, list) and all(_is_int(d) and d >= 0 for d in cast("list[Any]", shape)),
            f"{label}.shape must be a list of non-negative ints",
        )
        _require(_is_nonempty_str(record_d.get("dtype")), f"{label}.dtype must be a non-empty str")
        values = record_d.get("values")
        sha256 = record_d.get("sha256")
        _require(values is None or isinstance(values, list), f"{label}.values must be a list when present")
        _require(
            sha256 is None or (isinstance(sha256, str) and len(sha256) == 64),
            f"{label}.sha256 must be a 64-char hex digest when present",
        )
        _require(
            values is not None or sha256 is not None,
            f"{label} must carry at least one of 'values' or 'sha256'",
        )


# --- writing (R5.41 / R5.43 / R5.63) -----------------------------------------


def _plan_content_media(trace: Trace) -> tuple[dict[str, str], dict[str, bytes]]:
    """Where each media path ends up, and what to write there.

    Returns ``(plan, payloads)``: ``plan`` maps every original ref path that
    a byte-backed ref names to its content path ``media/<sha256><ext>`` —
    one path per digest, the extension taken from the first ref carrying
    those bytes (lower-cased), so equal bytes with differing suffixes still
    land on one file; ``payloads`` maps each content path to its bytes.
    A byte-less ref at a planned path follows the plan (it aliased the
    bytes); two payloads claiming one original path is a producer error.
    """
    plan: dict[str, str] = {}
    digest_path: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for step in trace.steps:
        for m in step.media:
            if m.source_bytes is None:
                continue
            digest = hashlib.sha256(m.source_bytes).hexdigest()
            content_path = digest_path.setdefault(digest, f"media/{digest}{Path(m.path).suffix.lower()}")
            payloads.setdefault(content_path, m.source_bytes)
            planned = plan.setdefault(m.path, content_path)
            if planned != content_path:
                raise ValueError(
                    f"trace {trace.run_id}: media path {m.path!r} carries two different payloads "
                    f"({planned} and {content_path})"
                )
    return plan, payloads


def _rewrite_media_paths(messages: Any, plan: dict[str, str]) -> None:
    """Point every image part in ``messages`` at its planned path — each part
    looked up once by its original url (the same walk as
    ``_message_media_paths``), so a rewrite never feeds another."""
    if not isinstance(messages, list):
        return
    for msg in cast("list[Any]", messages):
        if not isinstance(msg, dict):
            continue
        content = cast("dict[str, Any]", msg).get("content")
        if not isinstance(content, list):
            continue
        for part in cast("list[Any]", content):
            if not isinstance(part, dict) or cast("dict[str, Any]", part).get("type") != "image_url":
                continue
            image = cast("dict[str, Any]", part).get("image_url")
            if isinstance(image, dict):
                url = cast("dict[str, Any]", image).get("url")
                if isinstance(url, str) and url in plan:
                    cast("dict[str, Any]", image)["url"] = plan[url]


def _atomic_bytes_dump(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_jsonl_dump(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
    os.replace(tmp, path)


def write_trace(trace: Trace, game_run: taaf.game.GameRun, trace_root: Path) -> Path:
    """Write ``trace.json``, ``steps.jsonl``, referenced media, and ``_SUCCESS``
    under ``trace_root/<run_id>/`` (R5.63). ``_SUCCESS`` is written last, only
    after every other file is flushed (R5.43). Returns the trace folder path.
    """
    folder = Path(trace_root) / trace.run_id
    folder.mkdir(parents=True, exist_ok=True)

    # Media are stored by content (R5.44): a ref that carries bytes is renamed
    # to media/<sha256>.<ext>, the message parts citing it follow, and equal
    # bytes are written once per trace. An agent that attaches the same
    # frame to every prompt otherwise writes it once per step per step.
    # Planned from a snapshot of the original paths and applied once, so a
    # ref shared between steps, a byte-less ref aliasing a renamed path, or
    # a producer name that happens to look like a content name all resolve
    # the same way; the rewritten projection is validated before _SUCCESS.
    plan, payloads = _plan_content_media(trace)
    for step in trace.steps:
        _rewrite_media_paths(step.messages, plan)
        for m in step.media:
            m.path = plan.get(m.path, m.path)
    for content_path, data in payloads.items():
        _atomic_bytes_dump(data, folder / content_path)
    for step in trace.steps:
        dangling = _message_media_paths(step.messages) - {m.path for m in step.media}
        if dangling:
            raise ValueError(
                f"trace {trace.run_id} step {step.step_index}: messages cite media without a ref: {sorted(dangling)}"
            )

    # R5.43: _SUCCESS asserts every referenced media file is flushed. A ref
    # without source_bytes must already exist on disk (e.g. pre-written by the
    # adapter) — otherwise the trace would look complete while dangling.
    missing = sorted({m.path for step in trace.steps for m in step.media if not (folder / m.path).is_file()})
    if missing:
        raise FileNotFoundError(f"trace {trace.run_id}: referenced media files not on disk: {missing}")

    support.atomic_json_dump(trace.to_json_dict(), folder / TRACE_JSON)
    _atomic_jsonl_dump([s.to_json_dict() for s in trace.steps], folder / STEPS_JSONL)
    # _SUCCESS is the only completion marker (R5.42/5.43); write it last.
    (folder / SUCCESS_MARKER).write_text("")
    return folder


# --- reading (R5.64 / R5.65) -------------------------------------------------


def load_trace(path: Path) -> Trace:
    """Read one completed trace folder into ``Trace`` (R5.64). Raises if the
    folder lacks ``_SUCCESS`` (it is not a completed trace, R5.42)."""
    folder = Path(path)
    if not (folder / SUCCESS_MARKER).exists():
        raise FileNotFoundError(f"{folder} has no {SUCCESS_MARKER}; not a completed trace")
    with open(folder / TRACE_JSON) as f:
        meta = json.load(f)
    version = meta.get("schema_version", SCHEMA_VERSION)
    if version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"{folder}: unknown trace schema_version {version!r} (readable: {sorted(_READABLE_SCHEMA_VERSIONS)})"
        )
    steps: list[TraceStep] = []
    with open(folder / STEPS_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(TraceStep.from_json_dict(json.loads(line)))
    # A truncated steps.jsonl inside a _SUCCESS folder is a partial write that
    # slipped past finalization — loading it silently would train on it.
    n_steps = meta.get("n_steps")
    if n_steps is not None and n_steps != len(steps):
        raise ValueError(f"{folder}: {STEPS_JSONL} has {len(steps)} steps but {TRACE_JSON} records n_steps={n_steps}")
    return Trace.from_json_dict(meta, steps=steps)


# Trace-level keys renamed to avoid collisions with step keys when flattened.
_TRACE_ROW_COLLISIONS = {
    "created_at": "trace_created_at",
    "schema_version": "trace_schema_version",
    "extra": "trace_extra",
}


def iter_trace_rows(trace_root: Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per step from completed (``_SUCCESS``) trace folders, with
    trace-level metadata denormalized onto each row (R5.65). The canonical
    dataframe bridge — no pandas/Polars/PyArrow dependency."""
    root = Path(trace_root)
    if not root.exists():
        return
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / SUCCESS_MARKER).exists():
            continue
        trace = load_trace(folder)
        trace_meta = trace.to_json_dict()
        trace_meta.pop("run_id", None)  # identical to the step's run_id
        for old, new in _TRACE_ROW_COLLISIONS.items():
            if old in trace_meta:
                trace_meta[new] = trace_meta.pop(old)
        for step in trace.steps:
            row = dict(trace_meta)
            row.update(step.to_json_dict())  # step fields take precedence
            yield row


# --- retained-bundle export (R5.48–5.51 / R5.66) -----------------------------


def _completed_trace_folders(trace_root: Path) -> list[Path]:
    root = Path(trace_root)
    if not root.exists():
        return []
    return [f for f in sorted(root.iterdir()) if f.is_dir() and (f / SUCCESS_MARKER).exists()]


def export_run_bundle(job_dir: Path, trace_root: Path, out_path: Path, write_manifest: bool = True) -> Path:
    """Create a durable, self-contained run bundle (R5.66 / R5.48–5.51).

    Includes ``job_dir/benchmark.json`` at the bundle root and every
    ``_SUCCESS`` trace folder under ``traces/<run_id>/``, written as a single
    zstd-compressed tar to avoid many small files. Published atomically (write
    to ``.tmp`` then rename). When ``write_manifest`` is set, an inventory
    manifest (R5.50) is published next to the bundle *after* the bundle exists
    (R5.51). Returns ``out_path``.
    """
    import zstandard  # noqa: PLC0415 — keep zstd import local to the export path

    job_dir = Path(job_dir)
    trace_root = Path(trace_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _completed_trace_folders(trace_root)
    benchmark_json = job_dir / "benchmark.json"

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cctx = zstandard.ZstdCompressor()
    with (
        open(tmp, "wb") as raw,
        cctx.stream_writer(raw, closefd=False) as zfh,
        tarfile.open(fileobj=zfh, mode="w|") as tar,
    ):
        if benchmark_json.exists():
            tar.add(benchmark_json, arcname="benchmark.json")
        for folder in completed:
            tar.add(folder, arcname=f"traces/{folder.name}")
    os.replace(tmp, out_path)

    if write_manifest:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "rows": [
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": folder.name,
                    "source_job_dir": str(job_dir),
                    "export_path": out_path.name,
                    "export_member_prefix": f"traces/{folder.name}",
                }
                for folder in completed
            ],
        }
        # Published only after the bundle exists (R5.51); atomic write.
        support.atomic_json_dump(manifest, out_path.parent / MANIFEST_FILENAME)
    return out_path
