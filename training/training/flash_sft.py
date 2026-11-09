"""Supervised fine-tuning of Qwen3.8-Flash on fixed token rows: completion cross-entropy, LoRA adapters.

    torchrun --nproc-per-node=G -m training.flash_sft --config CONFIG.json --run-dir RUN_DIR

A row is one token-exact model request in the trainer's row format (one JSON object per line):

    input_token_ids, output_token_ids    the request's prompt and its supervised completion
    packed_completions                   [[start, length], ...] earlier completions inside the prompt, also
                                         supervised (a chain of requests packed into its last one); optional
    images                               [{path, sha256}, ...] the prompt's images, in placeholder order
    mm_processor_kwargs                  the image processor's arguments (e.g. {"max_pixels": 262144}), given
                                         when the processor is built
    game_id and any other fields         carried for reporting (validation group_by)

Each rank trains one row per update, so an update holds world-size rows. The loss is the summed token CE of
every supervised token of the update's rows divided by their total token count (per-token weighting). Model
construction, adapters, kernels and the vision tower are training.flash's (build_flash_model,
dense_lora_policy, kernel_mode, FlashMultimodalDecoder); image pixels and M-RoPE positions are rebuilt from the
row's own token ids with ullm.media, as the RL learner does. The FP32 adapter masters are replicated: rank zero's
initial values are broadcast, gradients are summed across ranks, clipped by their global norm and stepped by
AdamW on every rank alike, on the GPU or (optimizer.placement "host") on pinned host copies (HostAdamW). With
expert_lora_sharded the routed-expert adapters are instead split along their rank across the ranks after that
broadcast (training.flash.expert_lora_shard): each rank holds, steps and checkpoints only its slice, every use
gathers the full master and the backward sums the full gradient.

Long rows (~190k tokens on 4 B200): offload_chunk_mib copies offloaded boundaries in slabs (no power-of-two
rounding of pinned host memory), retain_qsa_attention false recomputes QSA attention in backward instead of
keeping it on the GPU, placement "host" moves AdamW's moments off the GPU and cuda_expandable_segments avoids
allocator fragmentation. preflight_longest_rows checks the longest rows fit before update 0 (_preflight).

rows_render_files pins the tokenizer, chat template and image processor the rows were rendered with: a model whose
files differ is refused, since it would silently train on another tokenizer's ids (check_render_files).

Every scientifically meaningful setting is a required config key (load_config); there are no hidden defaults.
The one optional key is recorded_logprobs.ce_positions ("all" or "unaligned"): absent, it keeps the mixed
objective that every earlier recorded-logprob config (frozen by hash) was trained with, CE on every supervised
token plus weight * KL at the recorded positions; "unaligned" makes CE and KL exclusive per token (_row_loss).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROW_MODEL_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
CONFIG_KEYS = {
    "model", "rows", "rows_render_files", "validation", "lora", "expert_lora", "train_routers", "expert_storage",
    "nvfp4_materialization", "kernel_mode", "kernel_overrides", "offload_every", "offload_chunk_mib",
    "reuse_qsa_selection", "retain_qsa_attention", "cuda_expandable_segments", "preflight_longest_rows", "expert_lora_sharded", "expert_lora_inplace_delta", "optimizer", "passes", "batching", "max_length", "head_chunk", "save_every", "seed", "wandb",
}
OPTIMIZER_KEYS = {"lr", "warmup_updates", "betas", "eps", "weight_decay", "max_grad_norm", "placement"}
OPTIMIZER_PLACEMENTS = ("device", "host")
LORA_KEYS = {"r", "alpha", "target_modules", "group_ranks"}
RECORDED_LOGPROB_KEYS = {"weight", "tokenizer_sha256", "require_logprobs"}
# Which supervised tokens receive CE next to the recorded-teacher KL: every token (the
# mixed objective) or only the tokens without a recorded target (CE and KL exclusive).
CE_POSITIONS = ("all", "unaligned")
RENDER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _exact(name, value, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} needs exactly the keys {sorted(keys)}; got {sorted(value) if isinstance(value, dict) else value!r}")


def _positive_int(name, value, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name}={value!r}; requires a {'nonnegative' if zero else 'positive'} integer")


def _positive_float(name, value, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name}={value!r}; requires a finite {'nonnegative' if zero else 'positive'} number")


def load_config(path):
    """The run's configuration with every key validated; nothing is defaulted."""
    config = json.loads(Path(path).read_text())
    _exact("config", config, CONFIG_KEYS | (set(config) & {"recorded_logprobs"}))
    if "recorded_logprobs" in config:
        spec = config["recorded_logprobs"]
        _exact("recorded_logprobs", spec, RECORDED_LOGPROB_KEYS | (set(spec) & {"ce_positions"}))
        _positive_float("recorded_logprobs.weight", spec["weight"])
        if spec.get("ce_positions", "all") not in CE_POSITIONS:
            raise ValueError(f"recorded_logprobs.ce_positions must be one of {CE_POSITIONS}")
        digest = spec["tokenizer_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("recorded_logprobs.tokenizer_sha256 must be a SHA256 digest")
        if type(spec["require_logprobs"]) is not bool:
            raise ValueError("recorded_logprobs.require_logprobs must be boolean")
    for name in ("rows", "validation"):
        block = config[name]
        keys = {"data", "data_sha256"} | ({"every", "group_by"} if name == "validation" else set())
        _exact(name, block, keys)
    _exact("rows_render_files", config["rows_render_files"], set(RENDER_FILES))
    for name, digest in config["rows_render_files"].items():
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError(f"rows_render_files[{name!r}]={digest!r}; requires a sha256 hex digest")
    if config["validation"] is not None:
        _positive_int("validation.every", config["validation"]["every"])
        group_by = config["validation"]["group_by"]
        if not isinstance(group_by, list) or len(set(group_by)) != len(group_by) or not all(
                isinstance(f, str) and f and f != "game_id" for f in group_by):
            raise ValueError("validation.group_by must list distinct row fields other than game_id (may be empty)")
    _exact("lora", config["lora"], LORA_KEYS)
    lora = config["lora"]
    _positive_int("lora.r", lora["r"])
    _positive_float("lora.alpha", lora["alpha"])
    if not isinstance(lora["group_ranks"], dict):
        raise ValueError("lora.group_ranks must be an object {group: [r, alpha]} (may be empty)")
    if config["expert_lora"] is not None:
        _exact("expert_lora", config["expert_lora"], {"rank", "alpha", "dropout"})
    from training.flash.expert_lora_shard import validate_expert_lora_sharded

    validate_expert_lora_sharded(config["expert_lora_sharded"], config["expert_lora"])
    from training.flash.expert_lora import validate_expert_lora_inplace_delta

    validate_expert_lora_inplace_delta(config["expert_lora_inplace_delta"], config["expert_lora"])
    for name in ("train_routers", "reuse_qsa_selection", "retain_qsa_attention", "cuda_expandable_segments"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config["expert_storage"] not in ("bf16", "nvfp4_sharded"):
        raise ValueError("expert_storage must be 'bf16' or 'nvfp4_sharded'")
    if config["nvfp4_materialization"] not in ("eager", "fused"):
        raise ValueError("nvfp4_materialization must be 'eager' or 'fused'")
    if not isinstance(config["kernel_overrides"], dict):
        raise ValueError("kernel_overrides must be an object (may be empty)")
    _positive_int("offload_every", config["offload_every"], zero=True)
    if config["offload_chunk_mib"] is not None:
        from training.flash.offload import validate_offload_chunk

        validate_offload_chunk(config["offload_chunk_mib"])
    opt = config["optimizer"]
    _exact("optimizer", opt, OPTIMIZER_KEYS | ({"decay"} if isinstance(opt, dict) and "decay" in opt else set()))
    _positive_float("optimizer.lr", opt["lr"])
    _positive_int("optimizer.warmup_updates", opt["warmup_updates"], zero=True)
    if "decay" in opt:
        _exact("optimizer.decay", opt["decay"], {"decay_from", "decay_until", "final_fraction"})
        decay = opt["decay"]
        _positive_int("optimizer.decay.decay_from", decay["decay_from"], zero=True)
        _positive_int("optimizer.decay.decay_until", decay["decay_until"])
        if not opt["warmup_updates"] <= decay["decay_from"] < decay["decay_until"]:
            raise ValueError("optimizer.decay needs warmup_updates <= decay_from < decay_until")
        if type(decay["final_fraction"]) is not float or not 0 < decay["final_fraction"] < 1:
            raise ValueError("optimizer.decay.final_fraction must be a float in (0, 1)")
    if (not isinstance(opt["betas"], list) or len(opt["betas"]) != 2
            or not all(type(b) in (int, float) and 0 <= b < 1 for b in opt["betas"])):
        raise ValueError("optimizer.betas must be [beta1, beta2] in [0, 1)")
    _positive_float("optimizer.eps", opt["eps"])
    _positive_float("optimizer.weight_decay", opt["weight_decay"], zero=True)
    _positive_float("optimizer.max_grad_norm", opt["max_grad_norm"])
    if opt["placement"] not in OPTIMIZER_PLACEMENTS:
        raise ValueError(f"optimizer.placement must be one of {OPTIMIZER_PLACEMENTS}")
    _positive_int("passes", config["passes"])
    _exact("batching", config["batching"], {"policy", "pool_size"})
    if config["batching"]["policy"] not in ("length_bucket", "random"):
        raise ValueError("batching.policy must be 'length_bucket' or 'random'")
    _positive_int("batching.pool_size", config["batching"]["pool_size"])
    for name in ("max_length", "head_chunk", "save_every"):
        _positive_int(name, config[name])
    _positive_int("preflight_longest_rows", config["preflight_longest_rows"], zero=True)
    _positive_int("seed", config["seed"], zero=True)
    if config["wandb"] is not None:
        _exact("wandb", config["wandb"], {"entity", "project", "group", "name", "id"})
        if config["wandb"]["id"] is not None and (not isinstance(config["wandb"]["id"], str) or not config["wandb"]["id"]):
            raise ValueError("wandb.id must be a nonempty string, or null to derive it from the run directory")
    # The adapter and kernel blocks are checked by the builders that take them.
    from training.flash.dense_lora import _DenseLoRAConfigurator
    from training.flash.expert_lora import normalize_lora
    from training.flash.kernel_mode import resolve_kernel_stack
    from training.flash.lora_targets import validate_target_modules

    validate_target_modules(lora["target_modules"])
    _DenseLoRAConfigurator(lora["r"], lora["alpha"], lora["target_modules"],
                           group_ranks={k: tuple(v) for k, v in lora["group_ranks"].items()})
    normalize_lora(config["expert_lora"])
    resolve_kernel_stack(config["kernel_mode"], config["kernel_overrides"])
    return config


def check_render_files(model_root, expected):
    """Refuse a model whose tokenizer, chat template or image processor differ from those the rows were rendered with."""
    differ = [name for name in RENDER_FILES if sha256_file(Path(model_root) / name) != expected[name]]
    if differ:
        raise ValueError(f"{model_root} differs from the rows' render checkpoint in {differ}")


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


# Keys that only place memory (which layer boundaries are offloaded to the host, and in what slabs); every computed
# value is identical under any setting, so a checkpoint may be resumed under another one when the run names the
# configuration it was written with (--resume-identity).
PLACEMENT_KEYS = ("offload_every", "offload_chunk_mib")


def placement_free(config):
    return {k: v for k, v in config.items() if k not in PLACEMENT_KEYS}


def check_resume_identity(state, config, identity=None):
    """The checkpoint ``state`` must come from ``config``, or from ``identity``, which differs from ``config`` only in
    PLACEMENT_KEYS."""
    if state["config_sha256"] == config_sha256(config):
        return
    if identity is None or state["config_sha256"] != config_sha256(identity):
        raise ValueError("checkpoint was written by another configuration")
    if placement_free(identity) != placement_free(config):
        changed = sorted(k for k in set(identity) | set(config) if identity.get(k) != config.get(k) and k not in PLACEMENT_KEYS)
        raise ValueError(f"resume identity differs from the configuration beyond memory placement: {changed}")


def parameter_digest(parameters):
    """Byte identity of full named tensors; all ranks gather any rank-sharded adapters."""
    import torch
    from training.flash.expert_lora_shard import full_value

    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        value = full_value(parameter)
        if hasattr(value, "to_local"):
            value = value.to_local()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def completion_spans(row):
    """Every supervised [start, length] of a row: its packed earlier completions, then its own completion."""
    prompt, output = len(row["input_token_ids"]), len(row["output_token_ids"])
    spans = [tuple(span) for span in row.get("packed_completions", [])] + [(prompt, output)]
    end = 0
    for start, length in spans:
        if type(start) is not int or type(length) is not int or start < 1 or length < 1 or start < end:
            raise ValueError(f"row completion spans must be ordered, positive and disjoint: {spans}")
        end = start + length
    if end != prompt + output:
        raise ValueError("the row's own completion must end the sequence")
    return spans


def target_positions(spans):
    """Hidden positions whose next token is supervised: start - 1 .. start + length - 2 of each span."""
    return [i for start, length in spans for i in range(start - 1, start + length - 1)]


class RowIndex:
    """Byte offsets and sizes of a JSONL row file; rows are parsed again only when used."""

    def __init__(self, path, expected_sha256, *, max_length, fields=(), recorded_logprobs=None,
                 vocab_size=None, for_training=False):
        self.path = Path(path)
        if sha256_file(self.path) != expected_sha256:
            raise ValueError(f"{self.path} does not match its SHA256")
        self.offsets, self.lengths, self.tokens, self.fields = [], [], [], []
        self.aligned_positions = 0
        offset = 0
        with self.path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    length = len(row["input_token_ids"]) + len(row["output_token_ids"])
                    if length > max_length:
                        raise ValueError(f"row {len(self.offsets)} has {length} tokens > max_length {max_length}")
                    if recorded_logprobs is not None:
                        from sft.logprob.targets import FIELD, validate_targets

                        events = validate_targets(row, vocab_size, recorded_logprobs["tokenizer_sha256"])
                        coverage = row[FIELD]["coverage"]
                        if recorded_logprobs["require_logprobs"] and (
                            coverage.get("steps_without_logprobs", 0)
                            or coverage.get("steps") != len(completion_spans(row))
                        ):
                            raise ValueError("every supervised completion must have audited original logprobs")
                        self.aligned_positions += len(events)
                    self.offsets.append(offset)
                    self.lengths.append(length)
                    self.tokens.append(sum(n for _, n in completion_spans(row)))
                    self.fields.append({f: str(row.get(f) or "") for f in ("game_id", *fields)})
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"{self.path} holds no rows")
        if recorded_logprobs is not None and not self.aligned_positions:
            raise ValueError("recorded-logprob data has no usable KL positions")
        if for_training:
            from sft.holdouts import validate_arc3_sft_training_games

            validate_arc3_sft_training_games(row["game_id"] for row in self.fields)

    def __len__(self):
        return len(self.offsets)

    def row(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


def update_schedule(lengths, world, passes, seed, policy, pool_size):
    """Rows of every update, one per rank: each pass covers every row once; None pads the pass's last update.

    length_bucket: each pass shuffles the rows, sorts pools of pool_size by length (longest first), cuts them
    into groups of `world` similar-length rows and shuffles the groups. Deterministic in (seed, pass)."""
    if policy == "length_bucket" and pool_size % world:
        raise ValueError(f"batching.pool_size {pool_size} must be a multiple of the world size {world}")
    updates = []
    for pass_index in range(passes):
        rng = random.Random(seed * 1_000_003 + pass_index)
        order = list(range(len(lengths)))
        rng.shuffle(order)
        if policy == "length_bucket":
            groups = []
            for start in range(0, len(order), pool_size):
                pool = sorted(order[start:start + pool_size], key=lambda i: (-lengths[i], i))
                groups += [pool[i:i + world] for i in range(0, len(pool), world)]
            rng.shuffle(groups)
        else:
            groups = [order[i:i + world] for i in range(0, len(order), world)]
        updates += [group + [None] * (world - len(group)) for group in groups]
    return updates


def lr_factor(step, warmup_updates, decay=None):
    """Linear warmup over the first updates, then the full rate (flat).

    decay (optional): from update decay_from the rate falls on a cosine to final_fraction at update decay_until
    (both 0-based), then holds there. It depends only on the update number, so a resumed run stays on the curve."""
    if step < warmup_updates:
        return (step + 1) / warmup_updates
    if decay is None or step <= decay["decay_from"]:
        return 1.0
    final = decay["final_fraction"]
    progress = min(1.0, (step - decay["decay_from"]) / (decay["decay_until"] - decay["decay_from"]))
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))


def summarize(values, labels, teacher_weight=None):
    """Native [log p, tokens, hits, entropy], optionally followed by KL and aligned positions."""

    def metrics(rows):
        logp, tokens, top, entropy = (sum(values[i][k] for i in rows) for k in range(4))
        result = {"nll": -logp / tokens, "top1": top / tokens, "entropy_mean": entropy / tokens,
                  "rows": len(rows), "tokens": int(tokens)}
        if teacher_weight is not None:
            kl, aligned = (sum(values[i][k] for i in rows) for k in (4, 5))
            result.update(ce_loss=result["nll"], teacher_kl=kl / tokens,
                          ce_plus_kl_loss=result["nll"] + teacher_weight * kl / tokens,
                          aligned_positions=int(aligned), kl_coverage=aligned / tokens)
        return result

    result = metrics(range(len(values)))
    for field, per_row in labels.items():
        groups = {}
        for i, value in enumerate(per_row):
            groups.setdefault(value, []).append(i)
        result["by_" + field] = {g: metrics(rows) for g, rows in sorted(groups.items())}
    return result


class RowInputs:
    """Model inputs of one row: token ids, M-RoPE positions and, with images, pixels and grids (ullm.media)."""

    def __init__(self, model_root, mm_processor_kwargs, image_token_id):
        import transformers
        from ullm.media import processing_identity

        # At construction: a Qwen image processor folds max_pixels into its size there and ignores it per call,
        # so images would otherwise be sized differently from the engine (ullm's vllmx does the same).
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=True, **mm_processor_kwargs)
        self.processing = processing_identity(
            self.processor, model=str(model_root), revision=None, mm_processor_kwargs=mm_processor_kwargs,
            model_architecture=ROW_MODEL_ARCHITECTURE)
        self.mm_processor_kwargs = mm_processor_kwargs
        self.image_token_id = image_token_id

    def __call__(self, row, device):
        import torch
        from PIL import Image
        from ullm.media import image_layout, position_inputs, process_images

        if row.get("mm_processor_kwargs", {}) != self.mm_processor_kwargs:
            raise ValueError("row mm_processor_kwargs differ from the run's")
        ids = row["input_token_ids"] + row["output_token_ids"]
        images = []
        for item in row.get("images", []):
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"row image changed: {item['path']}")
            with Image.open(item["path"]) as image:
                images.append(image.convert("RGB"))
        inputs = {"input_ids": torch.tensor([ids], device=device)}
        if images:
            tensors = process_images(self.processor, images, self.processing)
            layout = image_layout(row["input_token_ids"], tensors["image_grid_thw"], self.processing)
            if self.image_token_id in row["output_token_ids"]:
                raise ValueError("image placeholders inside a completion")
            inputs.update(pixel_values=tensors["pixel_values"].to(device),
                          image_grid_thw=tensors["image_grid_thw"].to(device))
        else:
            if self.image_token_id in ids:
                raise ValueError("image placeholders without images")
            layout = []
        positions = position_inputs(len(ids), layout, self.processing)
        inputs.update(position_ids=positions["position_ids"].unsqueeze(1).to(device),
                      mm_token_type_ids=positions["mm_token_type_ids"].unsqueeze(0).to(device),
                      attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device))
        return inputs


def model_reader_factory(root, staged=None):
    """The checkpoint reader for config["model"]: its shards from the launcher's scratch copy when one is staged.

    FLASH_SFT_STAGED_MODEL (set by slurm/flash_sft.sbatch with FLASH_SFT_STAGE_MODEL=1) names a copy made by
    sft.distillation.staging; it must stage exactly config["model"] (validate_stage). The config keeps the
    canonical path, so provenance and resume are unchanged; only the shard reads move off /data."""
    from training.flash.reader import Reader

    staged = staged if staged is not None else os.environ.get("FLASH_SFT_STAGED_MODEL")
    if not staged:
        return Reader
    from sft.distillation.staging import validate_stage

    source, shard_root = Path(root).resolve(), Path(staged)
    validate_stage(source, shard_root)

    def factory(path):
        if Path(path).resolve() != source:
            raise ValueError(f"{path} is not the staged model {source}")
        return Reader(path, shard_root=shard_root)

    return factory


class HostAdamW:
    """AdamW whose FP32 masters and moments live in pinned host memory (optimizer.placement "host").

    The device parameters stay the forward's FP32 adapters and receive the reduced, clipped gradients as before;
    step() copies those gradients to the host masters, runs a fused CPU AdamW there and copies the masters back.
    The update is AdamW's (CPU arithmetic, not bitwise the device kernel's); the device no longer holds the two
    moments (8 bytes per trainable). state_dict() has torch.optim.AdamW's layout."""

    def __init__(self, params, **hyper):
        import torch

        self.params = list(params)
        pin = torch.cuda.is_available()
        self.masters = []
        for p in self.params:
            master = torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
            master.copy_(p.detach())
            master.grad = torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
            self.masters.append(master)
        self.optimizer = torch.optim.AdamW([{"params": self.masters, "name": "adapters"}], fused=True, **hyper)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _synchronize(self):
        import torch

        if any(p.is_cuda for p in self.params):
            torch.cuda.synchronize()

    def step(self):
        import torch

        with torch.no_grad():
            for p, master in zip(self.params, self.masters):
                master.grad.copy_(p.grad, non_blocking=True)
            self._synchronize()
            self.optimizer.step()
            for p, master in zip(self.params, self.masters):
                p.copy_(master, non_blocking=True)
            self._synchronize()

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        """The moments, and the masters again from the device parameters (resume sets those first)."""
        import torch

        self.optimizer.load_state_dict(state)
        with torch.no_grad():
            for p, master in zip(self.params, self.masters):
                master.copy_(p.detach())


class FlashSFT:
    def __init__(self, config, run_dir, resume_identity=None):
        import torch
        import torch.distributed as dist

        self.config, self.run_dir, self.resume_identity = config, Path(run_dir), resume_identity
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(self.device)
        torch.manual_seed(config["seed"])
        root = Path(config["model"])
        model_config = json.loads((root / "config.json").read_text())
        self.recorded_logprobs = config.get("recorded_logprobs")
        if self.recorded_logprobs is not None and sha256_file(root / "tokenizer.json") != self.recorded_logprobs["tokenizer_sha256"]:
            raise ValueError("recorded targets use a different model tokenizer")
        if model_config.get("architectures") != [ROW_MODEL_ARCHITECTURE]:
            raise ValueError(f"model architecture {model_config.get('architectures')} is not {ROW_MODEL_ARCHITECTURE}")
        check_render_files(root, config["rows_render_files"])
        group_by = tuple(config["validation"]["group_by"]) if config["validation"] else ()
        target_options = {"recorded_logprobs": self.recorded_logprobs,
                          "vocab_size": model_config["text_config"]["vocab_size"]}
        self.rows = RowIndex(config["rows"]["data"], config["rows"]["data_sha256"],
                             max_length=config["max_length"], for_training=True, **target_options)
        self.validation = None if config["validation"] is None else RowIndex(
            config["validation"]["data"], config["validation"]["data_sha256"], max_length=config["max_length"],
            fields=group_by, **target_options)
        if self.recorded_logprobs is not None and self.validation is not None:
            training_games = {row["game_id"] for row in self.rows.fields}
            validation_games = {row["game_id"] for row in self.validation.fields}
            if "" in training_games | validation_games or not training_games.isdisjoint(validation_games):
                raise ValueError("recorded-logprob validation must use distinct held-out games")
        mm = self.rows.row(0).get("mm_processor_kwargs", {})
        self.inputs = RowInputs(root, mm, int(model_config["image_token_id"]))
        self.schedule = update_schedule(self.rows.lengths, self.world, config["passes"], config["seed"],
                                        config["batching"]["policy"], config["batching"]["pool_size"])
        decay = config["optimizer"].get("decay")
        if decay is not None and decay["decay_until"] >= len(self.schedule):
            raise ValueError(f"optimizer.decay.decay_until={decay['decay_until']} is past the run's {len(self.schedule)} updates")
        self._build_model(root, model_config)
        self.trainables = {name: p for group in self.bundle.parameter_groups.values() for name, p in group.items()}
        from training.flash.expert_lora_shard import is_rank_shard

        # Optimizer state_dict indices of the rank-sharded trainables (one parameter group, in this order).
        self.sharded = {i for i, p in enumerate(self.trainables.values()) if is_rank_shard(p)}
        opt = config["optimizer"]
        hyper = dict(lr=opt["lr"], betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
        if opt["placement"] == "host":
            # The ranks share the job's CPUs; torchrun starts every worker with one thread.
            torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // int(os.environ.get("LOCAL_WORLD_SIZE", 1))))
            self.optimizer = HostAdamW(list(self.trainables.values()), **hyper)
        else:
            self.optimizer = torch.optim.AdamW(
                [{"params": list(self.trainables.values()), "name": "adapters"}], foreach=False, **hyper)
        self.update = 0
        self.first_nll = None  # the run's update-0 held-out CE, for nll_change (kept in checkpoints)
        self._resume()

    def _build_model(self, root, model_config):
        import torch.distributed as dist

        from training.blackwell_gdn import configure_blackwell_gdn
        from training.flash.dense_lora import dense_lora_policy
        from training.flash.kernel_mode import builder_kwargs, resolve_kernel_stack
        from training.flash.model import build_flash_model
        from training.flash.policies import DistributionPolicy
        from training.flash_model import FlashMultimodalDecoder, load_flash_vision_tower

        config = self.config
        stack = resolve_kernel_stack(config["kernel_mode"], config["kernel_overrides"])
        self.kernel_stack = stack
        lora = config["lora"]
        self.bundle = build_flash_model(
            root, role="student",
            trainability=dense_lora_policy(
                lora["r"], lora["alpha"], lora["target_modules"], train_routers=config["train_routers"],
                group_ranks={k: tuple(v) for k, v in lora["group_ranks"].items()},
                expert_lora=config["expert_lora"]),
            distribution=DistributionPolicy(expert_storage=config["expert_storage"],
                                            nvfp4_materialization=config["nvfp4_materialization"]),
            distributed=True, device=self.device, offload_every=config["offload_every"],
            offload_chunk_mib=config["offload_chunk_mib"], reuse_qsa_selection=config["reuse_qsa_selection"],
            retain_qsa_attention=config["retain_qsa_attention"], reader_factory=model_reader_factory(root),
            **builder_kwargs(stack))
        if config["expert_lora_inplace_delta"]:
            from training.flash.expert_lora import enable_inplace_delta

            # Routed-expert LoRA deltas added into the projections in place: one fewer [rows x top-k, hidden]
            # temporary at the MoE peak; same values and gradients.
            enable_inplace_delta(self.bundle.model)
        if stack["gdn_guard"]:
            configure_blackwell_gdn("fla", device=self.device)
        visual = load_flash_vision_tower(root, device=self.device)
        self.decoder = FlashMultimodalDecoder(visual, self.bundle.model,
                                              image_token_id=int(model_config["image_token_id"]))
        self.head = self.bundle.head
        # Replicated masters start from rank zero's values on every rank.
        for group in self.bundle.parameter_groups.values():
            for parameter in group.values():
                dist.broadcast(parameter.data, src=0)
        if config["expert_lora_sharded"]:
            from training.flash.expert_lora_shard import shard_expert_adapters

            # After the broadcast: the slices are of rank zero's full initial adapters.
            shard_expert_adapters(self.bundle.parameter_groups["expert_lora"], self.rank, self.world)

    # -- one row ---------------------------------------------------------------------------------------------

    def _hidden(self, row):
        import torch

        inputs = self.inputs(row, self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.decoder(**inputs).last_hidden_state
        return hidden, inputs["input_ids"][0]

    def _row_loss(self, row):
        """Summed token CE of the row's supervised tokens (differentiable)."""
        import torch

        from training.flash.head_loss import ChunkedCE

        hidden, ids = self._hidden(row)
        positions = torch.tensor(target_positions(completion_spans(row)), device=self.device)
        states = hidden[0].index_select(0, positions).contiguous()
        if self.recorded_logprobs is None:
            del hidden
            return ChunkedCE.apply(states, self.head.weight, ids.index_select(0, positions + 1),
                                   self.config["head_chunk"])
        from sft.logprob.sparse_loss import completion_teacher_losses
        from sft.logprob.targets import FIELD

        targets = ids.index_select(0, positions + 1)
        if self.recorded_logprobs.get("ce_positions", "all") == "unaligned":
            # CE and KL are exclusive per token: a supervised token with a recorded teacher
            # target (event position = the token's index; its state is one earlier) is
            # trained by the KL alone, every other supervised token by CE alone.
            aligned = {event["position"] - 1 for event in row[FIELD]["events"]}
            keep = torch.tensor([i for i, p in enumerate(positions.tolist()) if p not in aligned],
                                device=self.device, dtype=torch.long)
            states, targets = states.index_select(0, keep).contiguous(), targets.index_select(0, keep)
        if len(targets):
            ce = ChunkedCE.apply(states, self.head.weight, targets, self.config["head_chunk"])
        else:
            ce = hidden[0, 0].sum() * 0
        kl = sum(completion_teacher_losses(hidden, self.head.weight, row, completion_spans(row),
                                          self.config["head_chunk"], self.recorded_logprobs["tokenizer_sha256"]))
        self._loss_components = (ce.detach(), kl.detach(), len(row[FIELD]["events"]))
        return ce + self.recorded_logprobs["weight"] * kl

    def _row_scores(self, row):
        """[summed log p, tokens, top-1 hits, summed entropy] of the row's supervised tokens, without gradients."""
        import torch

        with torch.no_grad():
            hidden, ids = self._hidden(row)
            positions = torch.tensor(target_positions(completion_spans(row)), device=self.device)
            states = hidden[0].index_select(0, positions)
            targets = ids.index_select(0, positions + 1)
            logp = top = entropy = 0.0
            chunk = self.config["head_chunk"]
            for start in range(0, len(targets), chunk):
                logits = torch.nn.functional.linear(states[start:start + chunk], self.head.weight).float()
                logp -= float(torch.nn.functional.cross_entropy(logits, targets[start:start + chunk],
                                                                reduction="sum"))
                top += float((logits.argmax(-1) == targets[start:start + chunk]).sum())
                log_probs = logits.log_softmax(-1)
                entropy -= float((log_probs.exp() * log_probs).sum())
                del log_probs
            extra = []
            if self.recorded_logprobs is not None:
                from sft.logprob.sparse_loss import completion_teacher_losses
                from sft.logprob.targets import FIELD

                kl = sum(completion_teacher_losses(hidden, self.head.weight, row, completion_spans(row),
                                                  chunk, self.recorded_logprobs["tokenizer_sha256"]))
                extra = [float(kl), float(len(row[FIELD]["events"]))]
        return [logp, float(len(targets)), top, entropy, *extra]

    # -- updates -----------------------------------------------------------------------------------------------

    def _reduce_gradients(self, bucket_bytes=256 << 20):
        """Sum the replicated adapters' gradients across ranks (flattened buckets); rank-sharded adapters' gradients
        were summed in the backward."""
        import torch
        import torch.distributed as dist

        params = [p for i, p in enumerate(self.trainables.values()) if i not in self.sharded]
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        bucket, size = [], 0
        for p in params + [None]:
            if p is not None:
                bucket.append(p)
                size += p.grad.numel() * p.grad.element_size()
            if bucket and (p is None or size >= bucket_bytes):
                flat = torch.cat([q.grad.reshape(-1) for q in bucket])
                dist.all_reduce(flat)
                offset = 0
                for q in bucket:
                    q.grad.copy_(flat[offset:offset + q.grad.numel()].view_as(q.grad))
                    offset += q.grad.numel()
                bucket, size = [], 0

    def _train_update(self, step):
        import torch
        import torch.distributed as dist

        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)
        if os.environ.get("FLASH_SFT_PINNED_TELEMETRY") == "1":
            from sft.distillation.host_memory import record_pinned

            record_pinned(self.run_dir, self.rank, step + 1, "before_forward")
        group = self.schedule[step]
        real = [i for i in group if i is not None]
        tokens = sum(self.rows.tokens[i] for i in real)
        mine = group[self.rank]
        # A padding slot trains the update's first row at zero weight, so every rank runs the same collectives.
        row = self.rows.row(real[0] if mine is None else mine)
        weight = 0.0 if mine is None else 1.0 / tokens
        active = mine is not None
        row_tokens = self.rows.tokens[mine] if active else 0
        loss_sum = self._row_loss(row)
        if self.recorded_logprobs is not None:
            finite = torch.isfinite(loss_sum.detach()).to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("nonfinite recorded-logprob training loss")
        (loss_sum * weight).backward()
        self._reduce_gradients()
        if self.sharded:
            from training.flash.expert_lora_shard import clip_grad_norm_
        else:
            clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
        options = {"error_if_nonfinite": True} if self.recorded_logprobs is not None else {}
        norm = clip_grad_norm_(list(self.trainables.values()), self.config["optimizer"]["max_grad_norm"], **options)
        factor = lr_factor(step, self.config["optimizer"]["warmup_updates"], self.config["optimizer"].get("decay"))
        for g in self.optimizer.param_groups:
            g["lr"] = self.config["optimizer"]["lr"] * factor
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device)
        if os.environ.get("FLASH_SFT_PINNED_TELEMETRY") == "1":
            record_pinned(self.run_dir, self.rank, step + 1, "after_update")
        loss = float(loss_sum.detach())
        extra = []
        if self.recorded_logprobs is not None:
            extra = [float(value) * active for value in self._loss_components]
        stats = torch.tensor([loss * active,
                              torch.cuda.max_memory_allocated(self.device) / 2**30,
                              len(row["input_token_ids"]) + len(row["output_token_ids"]),
                              float(active), row_tokens, loss / row_tokens if active else 0.0,
                              torch.cuda.max_memory_reserved(self.device) / 2**30,
                              time.monotonic() - begin, *extra], dtype=torch.float64, device=self.device)
        gathered = [torch.zeros_like(stats) for _ in range(self.world)]
        dist.all_gather(gathered, stats)
        gathered = torch.stack(gathered).cpu()
        per_pass = len(self.schedule) // self.config["passes"]
        active_ranks = gathered[:, 3] > 0
        metrics = {"update": step + 1, "pass": step // per_pass + 1, "loss_per_token": float(gathered[:, 0].sum()) / tokens,
                "tokens": tokens, "rows": len(real), "grad_norm": float(norm), "lr": self.config["optimizer"]["lr"] * factor,
                "lr_factor": factor, "loss_per_example": float(gathered[active_ranks, 5].mean()),
                "peak_allocated_gib": float(gathered[:, 1].max()), "max_sequence": int(gathered[:, 2].max()),
                "seconds": time.monotonic() - begin,
                # One entry per rank; masked padding counts as no sequence.
                "per_rank": {"sequence_length": [int(n) for n in gathered[:, 2] * gathered[:, 3]],
                             "seconds": gathered[:, 7].tolist(), "peak_allocated_gib": gathered[:, 1].tolist(),
                             "peak_reserved_gib": gathered[:, 6].tolist()}}
        if self.recorded_logprobs is not None:
            metrics.update(ce_loss=float(gathered[:, 8].sum()) / tokens,
                           teacher_kl=float(gathered[:, 9].sum()) / tokens,
                           aligned_positions=int(gathered[:, 10].sum()),
                           kl_coverage=float(gathered[:, 10].sum()) / tokens,
                           teacher_logprob_weight=self.recorded_logprobs["weight"])
        return metrics

    def _preflight(self):
        """Every rank runs forward and backward on the preflight_longest_rows longest training rows, and a forward on
        the longest validation row, before update 0: a row that does not fit fails minutes after the model loads,
        not at the update that first draws it. Gradients are discarded and the RNG states restored, so training
        is unchanged."""
        import resource

        import torch
        import torch.distributed as dist

        count = self.config["preflight_longest_rows"]
        if not count:
            return None
        begin = time.monotonic()
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device), random.getstate())
        longest = sorted(range(len(self.rows)), key=lambda i: (-self.rows.lengths[i], i))[:count]
        rows = []
        for index in longest:
            torch.cuda.reset_peak_memory_stats(self.device)
            (self._row_loss(self.rows.row(index)) * 0.0).backward()
            self._reduce_gradients()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            rows.append({"row": index, "tokens": self.rows.lengths[index],
                         "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                         "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30})
        if self.validation is not None:
            index = max(range(len(self.validation)), key=lambda i: (self.validation.lengths[i], -i))
            torch.cuda.reset_peak_memory_stats(self.device)
            self._row_scores(self.validation.row(index))
            torch.cuda.synchronize(self.device)
            rows.append({"validation_row": index, "tokens": self.validation.lengths[index],
                         "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                         "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30})
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], self.device)
        random.setstate(rng[2])
        peaks = torch.tensor([max(r["peak_allocated_gib"] for r in rows), max(r["peak_reserved_gib"] for r in rows),
                              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20],
                             dtype=torch.float64, device=self.device)
        gathered = [torch.zeros_like(peaks) for _ in range(self.world)]
        dist.all_gather(gathered, peaks)
        gathered = torch.stack(gathered).cpu()
        return {"rows": rows, "peak_allocated_gib": gathered[:, 0].tolist(), "peak_reserved_gib": gathered[:, 1].tolist(),
                "host_max_rss_gib": gathered[:, 2].tolist(),
                "device_total_gib": torch.cuda.get_device_properties(self.device).total_memory / 2**30,
                "seconds": time.monotonic() - begin}

    def _validate(self, step):
        import torch
        import torch.distributed as dist

        begin = time.monotonic()
        index = self.validation
        values = torch.zeros(len(index), 4 if self.recorded_logprobs is None else 6, dtype=torch.float64, device=self.device)
        mine = list(range(self.rank, len(index), self.world))
        for i in mine:
            values[i] = torch.tensor(self._row_scores(index.row(i)), dtype=torch.float64, device=self.device)
        # Every rank runs the same number of forwards (FSDP gathers each layer together); extras are discarded.
        shortest = min(range(len(index)), key=index.lengths.__getitem__)
        for _ in range(-(-len(index) // self.world) - len(mine)):
            self._row_scores(index.row(shortest))
        dist.all_reduce(values)
        labels = {f: [row[f] for row in index.fields] for f in index.fields[0]}
        labels["game"] = labels.pop("game_id")
        result = summarize(values.cpu().tolist(), labels,
                           None if self.recorded_logprobs is None else self.recorded_logprobs["weight"])
        if self.first_nll is None:
            self.first_nll = result["nll"]
        result.update(update=step, nll_change=result["nll"] - self.first_nll, seconds=time.monotonic() - begin)
        return result

    # -- checkpoints -------------------------------------------------------------------------------------------

    def _checkpoint_dir(self, step):
        return self.run_dir / "checkpoints" / f"update-{step:06d}"

    def _save(self, step):
        import torch
        import torch.distributed as dist

        from training.flash.expert_lora_shard import full_value, split_optimizer_state

        final = self._checkpoint_dir(step)
        partial = final.with_name(final.name + ".partial")
        # adapters.pt holds the full adapters in either layout (rank-sharded ones gathered: every rank takes part).
        adapters = {}
        for n, p in self.trainables.items():
            value = full_value(p)
            if self.rank == 0:
                adapters[n] = value.cpu()
        optimizer_state = self.optimizer.state_dict()
        if self.rank == 0:
            partial.mkdir(parents=True, exist_ok=True)
            torch.save(adapters, partial / "adapters.pt")
        del adapters
        if self.sharded:
            # Each rank's slices' optimizer state beside the shared state.
            optimizer_state, own = split_optimizer_state(optimizer_state, self.sharded)
        if self.rank == 0:
            torch.save(optimizer_state, partial / "optimizer.pt")
        dist.barrier()
        if self.sharded:
            torch.save(own, partial / f"optimizer-rank{self.rank}.pt")
        torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device),
                    "python": random.getstate()}, partial / f"rng-rank{self.rank}.pt")
        dist.barrier()
        if self.rank == 0:
            (partial / "state.json").write_text(json.dumps(
                {"update": step, "config_sha256": config_sha256(self.config), "world_size": self.world,
                 "first_nll": self.first_nll}) + "\n")
            partial.rename(final)
            latest = self.run_dir / "checkpoints" / "latest.json"
            latest.with_suffix(".tmp").write_text(json.dumps({"update": step, "path": final.name}) + "\n")
            latest.with_suffix(".tmp").replace(latest)
        dist.barrier()

    def _resume(self):
        import torch

        latest = self.run_dir / "checkpoints" / "latest.json"
        if not latest.is_file():
            return
        path = self.run_dir / "checkpoints" / json.loads(latest.read_text())["path"]
        state = json.loads((path / "state.json").read_text())
        if state["world_size"] != self.world:
            raise ValueError(f"{path} was written by another world size")
        try:
            check_resume_identity(state, self.config, getattr(self, "resume_identity", None))
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        from training.flash.expert_lora_shard import is_rank_shard, merge_optimizer_state, rank_slice

        saved = torch.load(path / "adapters.pt", map_location="cpu")
        if set(saved) != set(self.trainables):
            raise ValueError("checkpoint adapters differ from the configured trainables")
        with torch.no_grad():
            for name, parameter in self.trainables.items():
                value = saved[name]
                parameter.copy_(rank_slice(value, parameter, self.rank) if is_rank_shard(parameter) else value)
        del saved
        optimizer_state = torch.load(path / "optimizer.pt", map_location="cpu")
        if self.sharded:
            optimizer_state = merge_optimizer_state(
                optimizer_state, torch.load(path / f"optimizer-rank{self.rank}.pt", map_location="cpu"))
        self.optimizer.load_state_dict(optimizer_state)
        rng = torch.load(path / f"rng-rank{self.rank}.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)
        random.setstate(rng["python"])
        self.update, self.first_nll = state["update"], state["first_nll"]

    # -- the run -----------------------------------------------------------------------------------------------

    def run(self, *, max_updates=None, time_budget_seconds=None, stop_file=None, verify_parameters=False):
        import torch
        import torch.distributed as dist

        if max_updates is not None:
            _positive_int("max_updates", max_updates)
        if time_budget_seconds is not None:
            _positive_float("time_budget_seconds", time_budget_seconds)
        config = self.config
        frozen_routers = {}
        if verify_parameters:
            if config["train_routers"] or self.head.weight.requires_grad or any("lora_" not in n for n in self.trainables):
                raise ValueError("qualification expects only adapters to be optimized and a frozen head/router")
            frozen_routers = {n: p for n, p in [*self.bundle.model.named_parameters(), *self.bundle.model.named_buffers()]
                              if n.endswith("mlp.gate.weight")}
            if not frozen_routers or any(p.requires_grad for p in frozen_routers.values()):
                raise ValueError("qualification requires a census of frozen routers")
            router_digest = parameter_digest(frozen_routers)
            adapter_groups = {}
            for name, parameter in self.trainables.items():
                group = "routed" if ".mlp.experts." in name else "shared" if ".mlp.shared_expert." in name else "attention"
                adapter_groups.setdefault(group, {})[name] = parameter
            initial_adapters = {name: parameter_digest(group) for name, group in adapter_groups.items()}
        target = len(self.schedule)
        end = target if max_updates is None else min(target, self.update + max_updates)
        started = time.monotonic()
        log = _Log(self.run_dir, config, self.rank)
        log.event(event="start", updates=len(self.schedule), rows=len(self.rows), world_size=self.world,
                  resume_update=self.update, kernel_stack=self.kernel_stack,
                  cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  trainables=sum(p.numel() * (self.world if i in self.sharded else 1)
                                 for i, p in enumerate(self.trainables.values())),
                  rank_trainables=sum(p.numel() for p in self.trainables.values()))
        preflight = self._preflight()
        if preflight is not None:
            log.event(event="preflight", **preflight)
        every = config["validation"]["every"] if config["validation"] else None
        if every and self.update == 0:
            log.metrics(0, validation=self._validate(0))
        for step in range(self.update, end):
            metrics = {"train": self._train_update(step)}
            done = step + 1
            stop = done == end
            if time_budget_seconds is not None or stop_file is not None:
                requested = ((time_budget_seconds is not None and time.monotonic() - started >= time_budget_seconds)
                             or (stop_file is not None and Path(stop_file).exists()))
                flag = torch.tensor(int(requested), device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = stop or bool(flag.item())
            if every and (done % every == 0 or stop):
                metrics["validation"] = self._validate(done)
            log.metrics(done, **metrics)
            if done % config["save_every"] == 0 or stop:
                self._save(done)
            self.update = done
            if stop:
                break
        if verify_parameters:
            final_adapters = {name: parameter_digest(group) for name, group in adapter_groups.items()}
            replicas = [None] * self.world
            dist.all_gather_object(replicas, final_adapters)
            if (any(replica != final_adapters for replica in replicas)
                    or any(initial_adapters[name] == digest for name, digest in final_adapters.items())
                    or router_digest != parameter_digest(frozen_routers)):
                raise RuntimeError("qualification: adapters differ, an adapter group did not update, or a frozen router changed")
            if self.rank == 0:
                report = {"status": "passed", "update": self.update, "trainable_replicas_equal": True,
                          "frozen_routers_unchanged": True, "router_tensors": len(frozen_routers),
                          "optimizer_contains_only_adapters": True, "head_frozen": True,
                          "updated_adapter_groups": sorted(final_adapters),
                          "expert_lora_sharded": config["expert_lora_sharded"],
                          "rank_sharded_tensors": len(self.sharded)}
                (self.run_dir / "PARAMETERS_PASSED.json").write_text(json.dumps(report, indent=2) + "\n")
        if self.rank == 0:
            state = {"update": self.update, "total_updates": target, "complete": self.update == target,
                     "config_sha256": config_sha256(config), "world_size": self.world}
            temporary = self.run_dir / "RUN_STATE.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(self.run_dir / "RUN_STATE.json")
        log.event(event="done", updates=self.update, total_updates=target)
        log.close()


def wandb_run_id(wandb_config, run_dir):
    """The config's wandb.id, or with null one derived from the run directory: resubmissions of a run share it and
    a copy of the config in another run directory cannot append to this run's history."""
    return wandb_config["id"] or hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:16]


def wandb_metrics(step, train=None, validation=None):
    """One W&B row under the sft.distillation trainer's keys (wandb_logging.aggregate_step), so runs of both
    trainers share their panels. Its metrics this trainer does not compute (KL probe, expert, attention and GPU
    telemetry, per-group gradient norms, FLOP estimates) are absent."""
    flat = {"train/step": step}
    if train is not None:
        seconds = max(train["per_rank"]["seconds"])
        flat.update({
            "train/loss_per_completion_token": train["loss_per_token"],
            # Per-token weighting: the objective is the per-token loss (the old trainer logs both).
            "train/objective_loss": train["loss_per_token"],
            "train/loss_per_example": train["loss_per_example"],
            "train/global_examples": train["rows"],
            "train/global_completion_tokens": train["tokens"],
            # The old trainer logs the global clip norm of all trainables under both names.
            "train/router_grad_norm": train["grad_norm"],
            "train/trainable_grad_norm": train["grad_norm"],
            "train/lr_factor": train["lr_factor"],
            "train/step_seconds": seconds,
            "train/completion_tokens_per_second": train["tokens"] / seconds,
            "train/input_tokens_per_second": sum(train["per_rank"]["sequence_length"]) / seconds,
        })
        for key in ("ce_loss", "teacher_kl", "aligned_positions", "kl_coverage", "teacher_logprob_weight"):
            if key in train:
                flat["train/" + key] = train[key]
        for key, values in train["per_rank"].items():
            for rank, value in enumerate(values):
                flat[f"train/rank{rank}/{key}"] = value
    if validation is not None:
        for key, value in validation.items():
            if isinstance(value, dict):
                for group, group_values in value.items():
                    for k in ("nll", "top1", "ce_loss", "teacher_kl", "ce_plus_kl_loss", "kl_coverage"):
                        if k in group_values:
                            flat[f"train/sft_validation/{key.removeprefix('by_')}/{group}/{k}"] = group_values[k]
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat["train/sft_validation/" + ("step" if key == "update" else key)] = value
    return flat


class _Log:
    """Rank zero's metrics.jsonl and W&B run (one id across resubmissions)."""

    def __init__(self, run_dir, config, rank):
        self.rank, self.wandb = rank, None
        if rank:
            return
        self.path = Path(run_dir) / "metrics.jsonl"
        if config["wandb"] is not None:
            import wandb

            w = config["wandb"]
            self.wandb = wandb.init(entity=w["entity"], project=w["project"], group=w["group"], name=w["name"],
                                    id=wandb_run_id(w, run_dir), resume="allow", config=config)
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("train/*", step_metric="train/step")

    def event(self, **fields):
        if self.rank:
            return
        print(json.dumps(fields), flush=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(fields) + "\n")

    def metrics(self, step, **blocks):
        if self.rank:
            return
        self.event(update=step, **blocks)
        if self.wandb is not None:
            self.wandb.log(wandb_metrics(step, **blocks), step=step)

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()


def configure_cuda_allocator(expandable_segments, environ=os.environ):
    """cuda_expandable_segments: the caching allocator grows segments in place instead of keeping fixed blocks, so
    memory freed by one long sequence's tensors serves larger later requests (no fragmentation OOM). Set before CUDA
    initializes; a conflicting PYTORCH_CUDA_ALLOC_CONF from the environment is refused."""
    wanted = "expandable_segments:" + ("True" if expandable_segments else "False")
    current = environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if current is None:
        environ["PYTORCH_CUDA_ALLOC_CONF"] = wanted
    elif "expandable_segments" in current and wanted not in current.split(","):
        raise ValueError(f"PYTORCH_CUDA_ALLOC_CONF={current!r} contradicts cuda_expandable_segments={expandable_segments}")
    elif "expandable_segments" not in current:
        environ["PYTORCH_CUDA_ALLOC_CONF"] = current + "," + wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-updates", type=int, help="Bound this segment, preserving the full configured schedule")
    parser.add_argument("--time-budget-seconds", type=float, help="Checkpoint and stop at an update boundary after this budget")
    parser.add_argument("--stop-file", help="Checkpoint and stop at an update boundary when this file exists")
    parser.add_argument("--verify-parameters", action="store_true", help="Qualify frozen routers and replicated adapters")
    parser.add_argument("--resume-identity", help="The configuration the resumed checkpoint was written with, when this "
                        "run changes only its memory placement (offload_every, offload_chunk_mib)")
    args = parser.parse_args(argv)
    # Before anything can initialize CUDA (load_config imports the Flash modules); load_config validates the key.
    expandable = json.loads(Path(args.config).read_text()).get("cuda_expandable_segments")
    if type(expandable) is bool:
        configure_cuda_allocator(expandable)
    config = load_config(args.config)
    identity = load_config(args.resume_identity) if args.resume_identity else None
    import torch.distributed as dist

    run_dir = Path(args.run_dir)
    dist.init_process_group("nccl")
    try:
        if dist.get_rank() == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = run_dir / "config.json"
            if snapshot.exists() and placement_free(json.loads(snapshot.read_text())) != placement_free(config):
                # The run directory keeps one configuration; only its memory placement may change. (The snapshot may
                # hold a placement that failed to fit, which is why the checkpoint, not the snapshot, is the identity.)
                raise ValueError(f"{snapshot} holds another configuration; a run directory has one")
            snapshot.write_text(json.dumps(config, indent=2) + "\n")
        dist.barrier()
        FlashSFT(config, run_dir, identity).run(max_updates=args.max_updates,
                                    time_budget_seconds=args.time_budget_seconds, stop_file=args.stop_file,
                                    verify_parameters=args.verify_parameters)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
