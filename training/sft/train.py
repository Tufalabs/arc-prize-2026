"""SFT entrypoint: select TAAF traces and train via TRL SFTTrainer.

    uv run python -m sft.train --config configs/sft.json
    uv run python -m sft.train --config configs/sft.json --check-data   # no model load

Data selection (config `datasets` / `eval_datasets`) filters traces by game, tag,
and a generic score predicate before building the dataset; see sft/selection.py.
`eval_datasets` (optional) is the held-out validation set — TRL logs its eval loss.

Modes (config `training_mode`, default "auto"):
- retemplate : messages + single-turn completion; loss on completion only
               (`completion_only_loss`). Re-tokenized by the target template.
- token_exact: pre-tokenized rows from TAAF `input_token_ids`/`output_token_ids`;
               labels already mask the prompt. Text-only.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

from sft.config import SFTRunConfig
from sft.lora_wrap import assert_lora_wrap
from sft.selection import ResolvedTrace, SourceSelection, resolve_sources
from sft.dataset_cache import load_or_build


def _report_selection(title: str, traces: list[ResolvedTrace], reports: list[SourceSelection]) -> None:
    """Print what each source contributed and why traces were dropped."""
    print(f"== {title}: {len(traces)} trace(s) from {len(reports)} source(s) ==")
    for report in reports:
        dropped = ", ".join(f"{reason}={n}" for reason, n in sorted(report.dropped.items()))
        print(
            f"  {report.path}: kept {len(report.selected)}/{report.total}"
            + (f"  (dropped: {dropped})" if dropped else "")
        )
    by_game = Counter(t.game_id or "<unknown>" for t in traces)
    by_tag = Counter(tag for t in traces for tag in t.tags)
    if by_game:
        print("  by game_id: " + ", ".join(f"{g}={n}" for g, n in sorted(by_game.items())))
    if by_tag:
        print("  by tag: " + ", ".join(f"{tag}={n}" for tag, n in sorted(by_tag.items())))


def _select(cfg: SFTRunConfig):
    """Resolve train (and optional eval) traces from the config specs."""
    train_traces, train_reports = resolve_sources(
        cfg.train_specs(), env_dir=cfg.re_arc_env_dir, datasets_dir=cfg.re_arc_datasets_dir, for_training=True
    )
    _report_selection("train", train_traces, train_reports)
    if not train_traces:
        raise ValueError("training selection resolved to zero finalized traces")
    eval_traces: list[ResolvedTrace] = []
    if cfg.eval_specs():
        eval_traces, eval_reports = resolve_sources(
            cfg.eval_specs(), env_dir=cfg.re_arc_env_dir, datasets_dir=cfg.re_arc_datasets_dir
        )
        _report_selection("eval", eval_traces, eval_reports)
        if not eval_traces:
            raise ValueError("eval selection resolved to zero finalized traces")
        train_paths = {trace.trace_dir.resolve() for trace in train_traces}
        overlap = sorted(
            str(trace.trace_dir) for trace in eval_traces if trace.trace_dir.resolve() in train_paths
        )
        if overlap:
            preview = ", ".join(overlap[:5])
            raise ValueError(
                f"train and eval selections overlap on {len(overlap)} trace(s): {preview}"
            )
    return train_traces, eval_traces


def _completion_text(completion) -> str:
    """Flatten a retemplate completion message's text parts."""
    if not completion:
        return ""
    content = completion[0].get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _summarize(name: str, dataset, mode: str) -> None:
    row = dataset[0]
    print(f"{name}: mode={mode}; {len(dataset)} rows; columns={sorted(row)}")
    if "input_ids" in row:
        trained = sum(1 for x in row["labels"] if x != -100)
        print(f"  row[0]: input_ids={len(row['input_ids'])}, trained_tokens={trained}")
        return
    # Retemplate: surface completion health so a dataset dominated by empty or
    # tool-call-free completions (the pre-R5.37 failure) can't slip into training
    # unnoticed. Reading only the completion column avoids decoding images.
    completions = dataset["completion"]
    texts = [_completion_text(c) for c in completions]
    n = max(len(texts), 1)
    empty = sum(1 for t in texts if not t.strip())
    with_tool = sum(1 for t in texts if "<tool_call>" in t)
    mean_chars = sum(len(t) for t in texts) / n
    print(
        f"  completions: {len(texts)} rows, empty={empty} ({empty / n:.1%}), "
        f"with <tool_call>={with_tool} ({with_tool / n:.1%}), mean_chars={mean_chars:.0f}"
    )
    print(f"  row[0]: images={len(row.get('images', []))}, prompt_msgs={len(row.get('prompt', []))}")
    if empty:
        # The only backstop for empty completions that slip past the adapter
        # guard — a WARNING in a multi-hour SLURM log is not one. Abort (both
        # splits: empty rows corrupt training and blind eval loss alike).
        if os.environ.get("SFT_ALLOW_EMPTY_COMPLETIONS", "").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"  WARNING: {empty} empty completion(s) — allowed by SFT_ALLOW_EMPTY_COMPLETIONS.")
        else:
            raise ValueError(
                f"{name} dataset contains {empty} empty completion(s) — a dropped completion trains "
                "the model to emit nothing (the instant-EOS failure). Repair the dataset, or set "
                "SFT_ALLOW_EMPTY_COMPLETIONS=1 to proceed anyway."
            )


def _timed_checkpoint_callback(interval_minutes: float):
    """Force a checkpoint save every ``interval_minutes`` of wall time.

    Rank-consistent: rank 0's clock decides and the decision is broadcast, so
    every rank forces the save on the same optimizer step (per-rank clocks can
    cross the boundary on different steps, which desyncs distributed saves).
    """
    import torch.distributed as dist
    from transformers import TrainerCallback

    class TimedCheckpointCallback(TrainerCallback):
        def __init__(self) -> None:
            self._interval_seconds = float(interval_minutes) * 60.0
            self._last_save = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            due = (time.monotonic() - self._last_save) >= self._interval_seconds
            if dist.is_available() and dist.is_initialized():
                decision = [due if state.is_world_process_zero else None]
                dist.broadcast_object_list(decision, src=0)
                due = bool(decision[0])
            if due:
                control.should_save = True
                self._last_save = time.monotonic()
            return control

    return TimedCheckpointCallback()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-action SFT from TAAF traces")
    parser.add_argument("--config", default="configs/sft.json")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Select + build the dataset(s) and print a summary, then exit (no model load).",
    )
    args = parser.parse_args()

    cfg = SFTRunConfig.load(args.config)
    train_traces, eval_traces = _select(cfg)

    # SFT_STAGE_DIR (set by slurm/sft_train.sbatch to node-local scratch) is
    # where raw traces are staged on a dataset-cache miss before building.
    stage_dir = os.environ.get("SFT_STAGE_DIR") or None
    train_dataset, mode = load_or_build(
        "train",
        [t.trace_dir for t in train_traces],
        mode=cfg.training_mode,
        cache_dir=cfg.dataset_cache_dir,
        stage_dir=stage_dir,
    )
    _summarize("train", train_dataset, mode)
    eval_dataset = None
    if eval_traces:
        # Force the eval set to the same resolved mode so rows are comparable.
        eval_dataset, _ = load_or_build(
            "eval",
            [t.trace_dir for t in eval_traces],
            mode=mode,
            cache_dir=cfg.dataset_cache_dir,
            stage_dir=stage_dir,
        )
        if cfg.eval_max_rows is not None and len(eval_dataset) > cfg.eval_max_rows:
            # Deterministic sample spread across traces; contiguous rows would
            # all be consecutive steps of the same episode.
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(cfg.eval_max_rows))
            print(f"eval: capped to {cfg.eval_max_rows} row(s) (eval_max_rows)")
        _summarize("eval", eval_dataset, mode)

    if args.check_data:
        return

    # Heavy imports deferred so --check-data stays light.
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = AutoModelForImageTextToText.from_pretrained(cfg.model_name, torch_dtype="auto")

    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        report_to=["tensorboard"],
        logging_dir=cfg.logging_dir,
    )
    callbacks = []
    if eval_dataset is not None:
        # Held-out validation: TRL runs eval every `eval_steps` and logs eval loss.
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = cfg.eval_steps
    if cfg.checkpoint_interval_minutes is not None:
        # Wall-clock checkpointing: park the step scheduler out of reach and
        # let the callback force a save when the interval elapses. Eval-loss
        # cadence above is unaffected.
        sft_kwargs["save_strategy"] = "steps"
        sft_kwargs["save_steps"] = 1_000_000_000
        callbacks.append(_timed_checkpoint_callback(cfg.checkpoint_interval_minutes))
    elif eval_dataset is not None:
        # Checkpoint at every validation, so a wall-kill still leaves the last
        # evaluated adapter (the transformers default of save_steps=500 was
        # unreachable within the job time limit).
        sft_kwargs["save_strategy"] = "steps"
        sft_kwargs["save_steps"] = cfg.eval_steps
    if mode == "retemplate":
        # Loss on the completion (last assistant turn) only.
        sft_kwargs["completion_only_loss"] = True
    # token_exact rows are pre-tokenized with prompt-masked labels; TRL trains
    # them directly (input_ids present) and we leave the labels as-is.

    if cfg.use_lora:
        # LoRA forward uses every trainable parameter, so skip the per-step
        # unused-parameter graph traversal DDP defaults to under PEFT. Full
        # fine-tuning keeps the framework default because conditional models
        # can contain heads that a given batch does not exercise.
        sft_kwargs["ddp_find_unused_parameters"] = False
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    # TRL built the PeftModel from peft_config above; fail fast (and write the
    # census) if the LoRA wrap strayed outside the language model. See sft/lora_wrap.py.
    assert_lora_wrap(trainer.model, cfg)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"saved adapter/model to {cfg.output_dir}")


if __name__ == "__main__":
    main()
