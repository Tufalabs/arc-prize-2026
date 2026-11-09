"""The target's pre-final-mixer stream, captured inside the engine core.

The drafter reads the target's hidden state as the model hands it to an MTP
head: the multi-stream ``[4 x 2,560]`` before the final hyper-connection
mixer, which ``Qwen4ExpModel`` copies into ``_mtp_hidden_buffer`` when a
speculative config asks for MTP. This plugin makes it keep that buffer with
no drafter configured, and copies each prompt's rows out after every model
step, so a prefill of a whole chain yields the stream at every position,
computed by the served engine: its kernels, its compile, its graphs.

Class-level patches, all made before the model exists:

- ``Qwen4ExpModel.__init__`` allocates the buffer when the speculative
  config did not. The copy into it is then part of the forward that
  torch.compile traces, exactly as under a drafter.
- ``GPUModelRunner.execute_model`` (the V2 runner the deployment serves
  with) copies each captured request's rows to host memory once the step
  has run, at the positions the step computed, and writes a request's file
  when its whole prompt is in.
- With ``DRAFTER_CAPTURE_DRAFTER`` set and a k=1 MTP drafter configured,
  ``Qwen4ExpMultiTokenPredictor`` also copies its pass-0 single stream (what
  its LM head reads) into a buffer of its own, and ``sample_tokens``, where
  the V2 runner runs the drafter, copies those rows out beside the target's:
  the served drafter's output at every position, for parity with the trainer.

Only requests whose id carries ``REQUEST_TAG`` are captured, so profiling
and warm-up runs pass untouched. A cached prefix computes nothing, so the
engine must run with prefix caching off; a position that is never computed
leaves the file unwritten and says so.

Loaded as a ``vllm.general_plugins`` entry point, which vLLM runs in every
worker process. Inert unless ``DRAFTER_CAPTURE_DIR`` is set, which the
extraction job does before the engine starts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ENV_DIR = "DRAFTER_CAPTURE_DIR"
REQUEST_TAG = "drafter-capture-"


def capture_file(directory: Path, request_id: str) -> Path:
    """Where the capture of a request lands. vLLM may suffix the id it was
    given, so the file carries the engine's id and callers match on prefix."""
    return Path(directory) / (re.sub(r"[^A-Za-z0-9._-]", "_", request_id) + ".safetensors")


#: Also capture a k=1 drafter's pass-0 single stream (its LM head input) at
#: every position, for parity against the trainer's head.
ENV_DRAFTER = "DRAFTER_CAPTURE_DRAFTER"


class _Pending:
    def __init__(self, prompt_ids: list[int], width: int, draft_width: int = 0) -> None:
        import torch

        self.prompt_ids = prompt_ids
        self.hidden = torch.empty((len(prompt_ids), width), dtype=torch.bfloat16, pin_memory=True)
        self.filled = torch.zeros(len(prompt_ids), dtype=torch.bool)
        self.draft = (
            torch.empty((len(prompt_ids), draft_width), dtype=torch.bfloat16, pin_memory=True) if draft_width else None
        )
        self.draft_filled = torch.zeros(len(prompt_ids), dtype=torch.bool)
        self.complete = False


def register() -> None:
    """vLLM general plugin entry point."""
    directory = os.environ.get(ENV_DIR)
    if not directory:
        return
    import torch

    from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    with_drafter = bool(os.environ.get(ENV_DRAFTER))

    original_init = qwen4_exp_model.Qwen4ExpModel.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if self._mtp_hidden_buffer is None and self.hyper_connection_mixer is not None:
            vllm_config = kwargs["vllm_config"]
            # The allocation the model makes itself under an MTP drafter.
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.config.hc_count * self.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )

    qwen4_exp_model.Qwen4ExpModel.__init__ = __init__

    if with_drafter:
        from vllm.models.qwen4_exp.nvidia import mtp as qwen4_exp_mtp

        predictor = qwen4_exp_mtp.Qwen4ExpMultiTokenPredictor
        original_mtp_init = predictor.__init__
        original_mtp_forward = predictor.forward

        def mtp_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_mtp_init(self, *args, **kwargs)
            scheduler = kwargs["vllm_config"].scheduler_config
            self._drafter_sample_buffer = torch.empty(
                scheduler.max_num_batched_tokens + scheduler.max_num_seqs,
                self.config.hidden_size,
                dtype=kwargs["vllm_config"].model_config.dtype,
            )

        def mtp_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_mtp_forward(self, *args, **kwargs)
            if isinstance(result, tuple):
                # Traced into the compiled forward, like the target's copy.
                self._drafter_sample_buffer[: result[0].shape[0]].copy_(result[0])
            return result

        predictor.__init__ = mtp_init
        predictor.forward = mtp_forward

    pending: dict[str, _Pending] = {}
    step_rows: list[tuple[str, int, int, int]] = []  # request, first row, count, first position

    original_execute = GPUModelRunner.execute_model

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(self, scheduler_output, *args, **kwargs)
        step_rows.clear()
        if kwargs.get("dummy_run"):
            return result
        for new in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
            if REQUEST_TAG in new.req_id and new.prompt_token_ids is not None:
                buffer = self.model.get_mtp_target_hidden_states()
                draft_width = self.speculator.model.model.config.hidden_size if with_drafter else 0
                pending[new.req_id] = _Pending(list(new.prompt_token_ids), buffer.shape[1], draft_width)
        state = getattr(self, "execute_model_state", None)
        if state is None or not pending:
            return result
        batch = state.input_batch
        rows = [(i, rid) for i, rid in enumerate(batch.req_ids[: batch.num_reqs]) if rid in pending]
        if not rows:
            return result
        torch.cuda.synchronize()
        buffer = self.model.get_mtp_target_hidden_states()
        for i, rid in rows:
            job = pending[rid]
            first, last = int(batch.query_start_loc_np[i]), int(batch.query_start_loc_np[i + 1])
            start = int(batch.num_computed_tokens_np[i])
            count = min(last - first, len(job.prompt_ids) - start)
            if count <= 0:
                continue
            if bool(job.filled[start : start + count].any()):
                raise RuntimeError(f"{rid}: positions {start}-{start + count} computed twice")
            job.hidden[start : start + count].copy_(buffer[first : first + count])
            job.filled[start : start + count] = True
            step_rows.append((rid, first, count, start))
            if start + count >= len(job.prompt_ids):
                job.complete = True
        if not with_drafter:
            for rid in [rid for rid, job in pending.items() if job.complete]:
                _write(out, rid, pending.pop(rid))
        return result

    GPUModelRunner.execute_model = execute_model

    if with_drafter:
        original_sample = GPUModelRunner.sample_tokens

        def sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
            # The drafter's pass 0 over the step's rows runs in here.
            result = original_sample(self, *args, **kwargs)
            if not step_rows:
                return result
            torch.cuda.synchronize()
            buffer = self.speculator.model.model._drafter_sample_buffer
            for rid, first, count, start in step_rows:
                job = pending[rid]
                assert job.draft is not None
                job.draft[start : start + count].copy_(buffer[first : first + count])
                job.draft_filled[start : start + count] = True
            step_rows.clear()
            for rid in [rid for rid, job in pending.items() if job.complete]:
                _write(out, rid, pending.pop(rid))
            return result

        GPUModelRunner.sample_tokens = sample_tokens


def _write(out: Path, request_id: str, job: _Pending) -> None:
    import torch

    from drafter.files import save_file

    missing = int((~job.filled).sum())
    if job.draft is not None:
        missing += int((~job.draft_filled).sum())
    if missing:
        (out / (capture_file(out, request_id).stem + ".failed")).write_text(
            f"{missing} of {len(job.prompt_ids)} positions never computed (prefix cache on?)\n"
        )
        return
    path = capture_file(out, request_id)
    tmp = path.with_suffix(".tmp")
    tensors = {"hidden": job.hidden, "ids": torch.tensor(job.prompt_ids, dtype=torch.int32)}
    if job.draft is not None:
        tensors["draft_single"] = job.draft
    save_file(tensors, str(tmp), metadata={"request_id": request_id})
    tmp.replace(path)
