"""R5 trace collection — the GameRun-side writer.

One ``TraceCollector`` per run, living in the parent next to the host
backend and the ``GameRun``s. The backend REPORTS (``ullm.tracing``
records via ``Backend.trace_recorder``, one per successful model call,
tagged with the solver-stamped ``run_id``); the collector OWNS step
numbering, media splitting, payload hashes, action linkage, and the taaf
boundary — one ``game_run.register_trace(trace)`` per game at game end
(validate + write + ``_SUCCESS``, all TAAF's).

Action linkage is snapshot-based (branch_notes "Action linkage"): each
step remembers ``len(GameRun.history)`` at record time; at finalize,
step *i* links to history index ``L_i`` iff the next step's snapshot
(or the final history length) exceeds it — the FIRST action the call
authorized, matching the batch-tokens-to-first rule. There is no
execution-path hook, so in-snippet interleaving (respond → execute →
respond within one move) attributes correctly by construction, and the
linkage reads the authoritative ``GameRun.history`` — the same authority
``validate_trace`` re-checks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ullm.tracing

import taaf.game
import taaf.trace

_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def _canonical_sha256(payload: Any) -> str:
    """Hash of a payload in one canonical JSON form, so the same structure
    always hashes the same — the basis of the invertibility proof."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _repo_identity() -> tuple[str, str]:
    """(commit, dirty marker) for the running tree; empty when git cannot answer."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = ""
    return "arc3-platypus", commit


@dataclass
class _PendingStep:
    """A recorded step waiting for the action it authorized. ``pre_history_len``
    is the snapshot that resolves the linkage at finalize: the first action
    appended after it is the one the step's tokens paid for."""

    step: taaf.trace.TraceStep
    pre_history_len: int  # len(GameRun.history) when the step was recorded


class TraceCollector:
    """Per-run trace assembly. ``record`` is the backend's
    ``trace_recorder``; ``finalize`` runs at game end in the parent.

    Slot provenance rides the steps: ``note_grant`` hears every slot the
    host grants, and the granted run's next step carries
    ``extra["after"]`` -- the step whose end freed that slot, ``None`` for
    a slot nobody held. A run's steps from one such step to the next are the
    calls it made in one tenure of a slot, and the links chain those tenures
    into what each slot served in turn (``ullm.traffic`` replays that).
    """

    def __init__(
        self,
        capability: ullm.tracing.TraceCapability,
        *,
        source_kind: str = "platypus",
        content: str = taaf.trace.CONTENT_FULL,
    ):
        """Collect steps for one run under a fixed serving-stack identity;
        ``content`` is what the traces store."""
        self.capability = capability
        self.source_kind = source_kind
        self.content = content
        self._runs: dict[str, taaf.game.GameRun] = {}
        self._pending: dict[str, list[_PendingStep]] = {}
        # Steps each run has recorded, kept past its finalize: a finished
        # run's slot is granted on after its steps are written.
        self._counts: dict[str, int] = {}
        # A grant's link, held until the granted run's next step takes it.
        self._openings: dict[str, dict[str, Any] | None] = {}
        self._repo_name, self._repo_commit = _repo_identity()

    def note_grant(self, run_id: str, freed_by: str | None) -> None:
        """A slot went to ``run_id``, freed by ``freed_by`` (``None``: a slot
        nobody held). A run that gives a slot up without a step since its own
        grant passes that grant's link on, so no tenure is linked to twice."""
        if freed_by is None:
            after = None
        elif freed_by in self._openings:
            after = self._openings.pop(freed_by)
        else:
            count = self._counts.get(freed_by, 0)
            after = {"run_id": freed_by, "step_index": count - 1} if count else None
        self._openings[run_id] = after

    def register_run(self, game_run: taaf.game.GameRun) -> None:
        """Start collecting for a game run; steps arrive keyed by its run_id."""
        if game_run.trace_root is None:
            raise RuntimeError(
                "tracing is enabled but the GameRun has no trace_root — set Benchmark.trace_root "
                "(or job_dir, whose traces/ subdir is the default) so register_trace has a destination"
            )
        self._runs[game_run.run_id] = game_run
        self._pending.setdefault(game_run.run_id, [])

    def record(self, record: ullm.tracing.TraceStepRecord) -> None:
        """Take one backend-reported call: number it, split its media out, and
        hold it until an action links it — the first action it authorized."""
        run = self._runs.get(record.run_id)
        if run is None:
            raise RuntimeError(
                f"trace step reported for unknown run_id {record.run_id!r} — every traced game "
                "must be registered with the collector before its first model call"
            )
        pending = self._pending[record.run_id]
        step_index = len(pending)
        if self.content == taaf.trace.CONTENT_IDS:
            # No messages to rewrite: the images are the step's media as they
            # come, in placeholder order, and the payload reinlines to itself.
            messages: list[dict[str, Any]] = []
            media = [
                self._media_ref(image.png, image.mime, image.role, step_index, k)
                for k, image in enumerate(record.images)
            ]
            markers: list[tuple[int, int]] = []
        else:
            messages, media, markers = self._split_media(record, step_index)

        request_args = {k: v for k, v in record.request_payload.items() if k != "messages"}
        original_sha = _canonical_sha256(record.request_payload)
        reconstructed_sha = _canonical_sha256(
            {**request_args, "messages": self._reinline_media(messages, media, markers)}
        )
        if reconstructed_sha != original_sha:
            # The split must be exactly invertible from what lands on disk;
            # anything else means the stored trace cannot reproduce the
            # payload — fatal, same posture as the token roundtrip.
            raise ullm.tracing.TraceReconstructionError(
                f"payload reconstruction mismatch for run {record.run_id!r} step {step_index}: "
                "re-inlining the split media does not reproduce the request payload"
            )

        provider_validation: dict[str, Any] = {
            "provider_name": self.capability.provider_name,
            "provider_version": self.capability.provider_version,
            "api_kind": self.capability.api_kind,
            "provider_model_id": self.capability.provider_model_id,
            "request_args": request_args,
            "validation_method": self.capability.validation_method,
            "original_payload_sha256": original_sha,
            "reconstructed_payload_sha256": reconstructed_sha,
            "input_token_reconstruction_status": record.input_token_reconstruction_status,
            "output_token_reconstruction_status": record.output_token_reconstruction_status,
            "media_reconstruction_status": record.media_reconstruction_status,
        }
        extra = dict(record.extra)
        if record.run_id in self._openings:
            extra["after"] = self._openings.pop(record.run_id)
        self._counts[record.run_id] = step_index + 1
        measurable = bool(record.input_token_ids) and record.input_token_reconstruction_status != "unavailable"
        if pending and measurable and pending[-1].step.input_token_ids:
            # The nesting invariant: the served history extends the previous
            # step's prompt + sampled ids. Stamped wherever both steps carry
            # a token witness (a hosted API's records carry none, and
            # nothing is claimed for them); enforced only under exact
            # history, within an epoch — an agent may legitimately rewrite
            # history (compaction), and declares it by bumping the epoch.
            previous = pending[-1].step
            nests = ullm.tracing.prompt_extends_previous(
                previous.input_token_ids, previous.output_token_ids, list(record.input_token_ids)
            )
            extra["nests_previous"] = nests
            same_epoch = extra.get("history_epoch") == previous.extra.get("history_epoch")
            if extra.get("exact_history") and same_epoch and not nests:
                # The run promised nesting within an epoch (LLMConfig.exact_history)
                # and the backend let a re-tokenised prompt through: the trace
                # would carry the drift the promise excludes.
                raise ullm.tracing.TraceReconstructionError(
                    f"exact history violated for run {record.run_id!r} step {step_index}: the prompt does not "
                    "extend the previous step's prompt + sampled ids and no history rewrite was declared ("
                    + ullm.tracing.describe_nesting_break(
                        previous.input_token_ids, previous.output_token_ids, list(record.input_token_ids)
                    )
                    + ")"
                )
        step = taaf.trace.TraceStep(
            run_id=record.run_id,
            step_index=step_index,
            messages=messages,
            input_token_ids=list(record.input_token_ids),
            output_text=record.output_text,
            output_message=record.output_message,
            output_token_ids=list(record.output_token_ids),
            # None unless the call asked the engine for them; null is spec-valid.
            logprobs=record.logprobs,
            sampling_params=dict(record.sampling_params),
            media=media,
            provider_validation=provider_validation,
            uncached_input_tokens=record.uncached_input_tokens,
            extra=extra,
        )
        pending.append(_PendingStep(step=step, pre_history_len=len(run.history)))

    def finalize(self, game_run: taaf.game.GameRun) -> Path | None:
        """Link actions from the snapshots, build the ``Trace``, and hand it
        to TAAF (validate + write). Returns the trace folder, or None when
        the game made no model calls."""
        pending = self._pending.pop(game_run.run_id, None)
        self._runs.pop(game_run.run_id, None)
        if not pending:
            return None
        final_len = len(game_run.history)
        for i, item in enumerate(pending):
            next_len = pending[i + 1].pre_history_len if i + 1 < len(pending) else final_len
            if next_len > item.pre_history_len:
                item.step.action_index = item.pre_history_len
                item.step.action = game_run.history[item.pre_history_len].submission
            # Values a backend packed while the step waited go out as they were.
            item.step.extra = ullm.tracing.unpack_extra(item.step.extra)
        capability = self.capability
        trace = taaf.trace.Trace(
            run_id=game_run.run_id,
            source_kind=self.source_kind,
            content=self.content,
            total_uncached_input_tokens=sum(p.step.uncached_input_tokens for p in pending),
            repo_name=self._repo_name,
            repo_commit_hash=self._repo_commit,
            model_id=capability.model_id,
            model_revision=capability.model_revision,
            model_weight_hash=capability.model_weight_hash,
            parent_model_revision=capability.parent_model_revision,
            tokenizer_id=capability.tokenizer_id,
            tokenizer_revision=capability.tokenizer_revision,
            processor_id=capability.processor_id,
            processor_revision=capability.processor_revision,
            vision_preprocessor_revision=capability.vision_preprocessor_revision,
            extra={
                "provider_name": capability.provider_name,
                "api_kind": capability.api_kind,
                # None when no step pair carried a token witness: nesting was
                # never measurable, which is not the same as never broken.
                "nesting_breaks": (
                    sum(1 for p in pending if p.step.extra.get("nests_previous") is False)
                    if any("nests_previous" in p.step.extra for p in pending)
                    else None
                ),
                # Any step served injected text: the ids are faithful (input
                # ids end with the injection, output ids are the sampled
                # continuation), but a consumer that re-tokenises text finds
                # the response opening in no message — such a consumer must
                # refuse the trace rather than train on it.
                "response_injected": any(p.step.extra.get("response_injection") for p in pending),
            },
            steps=[p.step for p in pending],
        )
        return game_run.register_trace(trace)

    def unfinalized_run_ids(self) -> list[str]:
        """Runs with steps still held — a crashed child legitimately leaves some."""
        return sorted(run_id for run_id, steps in self._pending.items() if steps)

    # --- media -------------------------------------------------------------

    def _split_media(
        self, record: ullm.tracing.TraceStepRecord, step_index: int
    ) -> tuple[list[dict[str, Any]], list[taaf.trace.MediaRef], list[tuple[int, int]]]:
        """Move image payloads out of the step, leaving every image part
        naming the file it was written to.

        Data-URL parts (OAI) are rewritten to trace-relative paths;
        marker-style backends (vllmx) ship their bytes via ``record.images``
        in marker order, and their markers are rewritten to the same form.
        Neither the data-URL string nor the bare marker is kept —
        ``_reinline_media`` rebuilds both from what is on disk, which is the
        invertibility the payload-hash pair proves. The third return value is
        the slots that were markers, which is what lets it tell the two
        apart."""
        media: list[taaf.trace.MediaRef] = []
        messages = json.loads(json.dumps(record.messages))  # own copy; already JSON-shaped

        def next_ref(payload: bytes, mime: str, role: str) -> taaf.trace.MediaRef:
            ref = self._media_ref(payload, mime, role, step_index, len(media))
            media.append(ref)
            return ref

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in cast("list[Any]", content):
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    continue
                part_dict = cast("dict[str, Any]", part)
                image_url = cast("dict[str, Any]", part_dict.get("image_url") or {})
                url = str(image_url.get("url", ""))
                if not url.startswith("data:"):
                    continue
                header, _, b64_payload = url.partition(",")
                mime = header[len("data:") :].split(";")[0] or "application/octet-stream"
                ref = next_ref(base64.b64decode(b64_payload), mime, "conversation_image")
                part_dict["image_url"]["url"] = ref.path
        # Marker-style backends (vllmx) put a bare ``{"type": "image"}`` in the
        # messages and ship the bytes out of band, in marker order
        # (ullm.tracing.TraceImage). Rewrite each marker into the same
        # self-describing form the data-URL path produces: a trace whose
        # messages name their own media is readable without knowing which
        # backend wrote it, and the declared message format has no bare image
        # part anyway. Their slots are returned so the payload roundtrip can
        # put the markers back.
        markers: list[tuple[int, int]] = []
        pending = iter(record.images)
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(cast("list[Any]", content)):
                if not (isinstance(part, dict) and part.get("type") == "image"):
                    continue
                image = next(pending, None)
                if image is None:
                    raise ullm.tracing.TraceReconstructionError(
                        f"step {step_index} carries more image markers than images: the backend must "
                        f"supply one TraceImage per marker, in marker order"
                    )
                ref = next_ref(image.png, image.mime, image.role)
                part_dict = cast("dict[str, Any]", part)
                part_dict.clear()
                part_dict.update({"type": "image_url", "image_url": {"url": ref.path}})
                markers.append((message_index, part_index))
        if next(pending, None) is not None:
            raise ullm.tracing.TraceReconstructionError(
                f"step {step_index} carries more images than image markers: the pairing is positional, "
                f"so a surplus image cannot be placed"
            )
        return messages, media, markers

    @staticmethod
    def _media_ref(payload: bytes, mime: str, role: str, step_index: int, k: int) -> taaf.trace.MediaRef:
        """The ref for a step's ``k``-th image, carrying its bytes to the write."""
        return taaf.trace.MediaRef(
            media_id=f"step{step_index}_img{k}",
            path=f"media/step{step_index}_img{k}.{_MIME_EXT.get(mime, 'bin')}",
            role=role,
            source_bytes=payload,
        )

    @staticmethod
    def _reinline_media(
        messages: list[dict[str, Any]], media: list[taaf.trace.MediaRef], markers: list[tuple[int, int]]
    ) -> list[dict[str, Any]]:
        """Rebuild the as-sent messages from the split form + stored bytes:
        data URLs regenerated from bytes, and the slots that arrived as bare
        markers restored to markers."""
        by_path: dict[str, taaf.trace.MediaRef] = {ref.path: ref for ref in media}
        marker_slots = set(markers)
        rebuilt = json.loads(json.dumps(messages))
        for message_index, message in enumerate(rebuilt):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(cast("list[Any]", content)):
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    continue
                part_dict = cast("dict[str, Any]", part)
                if (message_index, part_index) in marker_slots:
                    part_dict.clear()
                    part_dict["type"] = "image"
                    continue
                image_url = cast("dict[str, Any]", part_dict.get("image_url") or {})
                url = str(image_url.get("url", ""))
                ref = by_path.get(url)
                if ref is not None:
                    assert ref.source_bytes is not None
                    ext = url.rsplit(".", 1)[-1]
                    mime = next((m for m, e in _MIME_EXT.items() if e == ext), "application/octet-stream")
                    encoded = base64.b64encode(ref.source_bytes).decode("ascii")
                    image_url["url"] = f"data:{mime};base64,{encoded}"
        return rebuilt
