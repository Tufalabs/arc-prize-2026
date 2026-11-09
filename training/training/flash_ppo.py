"""Flash dense-LoRA policy training on torchrun/FSDP ranks."""

from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from training.dataset import collate_samples
from training.expert_load import (
    EVAL_ROLES, TRAINING_ROLES, install_expert_load_recorder, router_token_weights,
    summarize_counts, take_eval_handoff,
)
from training.flash.dense_lora import (
    adapters_disabled, dense_lora_policy, from_peft_state_dict, to_peft_state_dict,
)
from training.flash.lora_targets import peft_adapter_config
from training.flash_config import (
    DIAGNOSTIC_SETTINGS, RETIRED_SETTINGS, drop_retired_settings, warn_retired_setting,
)
from training.flash.gradient_sync import ReplicatedGradientReducer
from training.flash_policy_weights import (
    dither_seed, hf_weight_name, merge_lora_weight, parse_weight_name, policy_checksum, policy_manifest,
    manifest_sha256, write_policy_manifest,
)
from training.flash_routers import (
    ROUTER_FILE, hf_router_name, load_router_file, read_router_manifest, write_router_publication,
)
from training.full_weight_distributed import FullWeightDistributedRuntime
from training.full_weight_ppo import DistributedCycleMixin
from training.ppo import PPOTrainer, _pack_micro_batches, _sample_token_length

log = logging.getLogger(__name__)


# Diagnostics (expert load metrics), the parity gate's thresholds and the
# rounding of the servers' merged weights never change what the optimizer
# sees, so a checkpoint resumes under new defaults for them; retired settings
# in older states are ignored (a state saved with NVFP4 expert storage resumes
# with the BF16 experts, which compute bitwise the same).
_RECIPE_DIAGNOSTIC_KEYS = frozenset({
    *DIAGNOSTIC_SETTINGS, *RETIRED_SETTINGS, "parity_thresholds", "policy_weights_rounding",
})
# Placement, not recipe: which GPUs the ranks run on (the heterogeneous job
# renumbers them from 0, the shared layout uses 4-7), how many layers offload
# their checkpoint inputs to the host, and where the checkpoint is read from.
# None changes what a rank computes, and the world size is checked on its own,
# so a run moves between job shapes and storage (training/FLASH_RL.md, "Resume").
_RECIPE_PLACEMENT_KEYS = frozenset({"device_ids", "offload_every", "checkpoint_root"})
_ABSENT = object()


def _recipe_identity(recipe):
    if not isinstance(recipe, dict):
        return recipe
    # Recipes saved before train_routers existed trained the LoRA alone: a
    # missing field reads as false, so those states resume as LoRA-only runs
    # and a router-training config still refuses them. Recipes saved before
    # kernel_mode existed record the fast stack's resolved fields, so the
    # missing label reads as "fast"; a reference-mode config differs in those
    # fields and is refused.
    return {"train_routers": False, "kernel_mode": "fast"} | {
        key: value for key, value in recipe.items()
        if key not in _RECIPE_DIAGNOSTIC_KEYS and key not in _RECIPE_PLACEMENT_KEYS
    }


def _recipe_differences(saved, configured):
    """Describe each recipe identity field where a checkpoint and the config differ ("" if none)."""
    saved, configured = _recipe_identity(saved), _recipe_identity(configured)
    if saved == configured:
        return ""
    if not isinstance(saved, dict):
        return f"the checkpoint records no Flash recipe ({saved!r})"

    def shown(value):
        return "absent" if value is _ABSENT else repr(value)

    return "; ".join(
        f"{key}: checkpoint {shown(saved.get(key, _ABSENT))}, configured {shown(configured.get(key, _ABSENT))}"
        for key in sorted(saved.keys() | configured.keys())
        if saved.get(key, _ABSENT) != configured.get(key, _ABSENT)
    )


class FlashPPOTrainer(DistributedCycleMixin, PPOTrainer):
    def _load_policy_model(self, *, resume_adapter_dir=None):
        from training.flash.model import build_flash_model

        if self.config.worker_device_map_gpus:
            raise ValueError("Flash requires torchrun training without worker device maps")
        if self.config.lora.dropout != 0:
            raise ValueError("Flash dense LoRA requires zero dropout")
        settings = self.config.flash
        log.info("Flash kernel stack: kernel_mode=%s %s", settings.kernel_mode,
                 json.dumps(settings.kernel_stack(), sort_keys=True))
        self.bundle = build_flash_model(
            settings.checkpoint_root, role="student",
            trainability=dense_lora_policy(
                self.config.lora.r, self.config.lora.alpha, self.config.lora.target_modules,
                train_routers=settings.train_routers,
            ),
            distributed=dist.is_initialized(), device=self.device,
            **settings.model_kwargs(),
        )
        self._multimodal_decoder = None
        if self.config.multimodal:
            from training.flash_model import (
                FlashMultimodalDecoder, load_flash_vision_tower,
            )

            raw = json.loads((Path(settings.checkpoint_root) / "config.json").read_text())
            visual = load_flash_vision_tower(settings.checkpoint_root, device=self.device)
            self._multimodal_decoder = FlashMultimodalDecoder(
                visual, self.bundle.model, image_token_id=raw["image_token_id"],
            )
        if settings.train_routers:
            self._remember_checkpoint_routers()
        if resume_adapter_dir is not None:
            self._load_adapter(resume_adapter_dir)
        self._expert_load = None
        if settings.load_balance_metrics:
            from training.flash.adapters import FP32TopKRouter

            self._expert_load = install_expert_load_recorder(
                self.bundle.model, router_type=FP32TopKRouter, device=self.device,
            )
            if self._expert_load is None:
                log.warning("expert load metrics off: the policy has no FP32TopKRouter routers to observe")
        if dist.is_initialized():
            for parameter in self._trainable_parameters():
                dist.broadcast(parameter.data, src=0)
        self._gradient_reducer = (
            ReplicatedGradientReducer(self._gradient_buckets())
            if dist.is_initialized() else None
        )
        return self.bundle.model

    def _place_policy_model(self):
        pass  # build_flash_model owns placement and FSDP wrapping.

    def _base_policy_model(self):
        return self.bundle.model

    def _decoder_and_lm_head(self):
        decoder = getattr(self, "_multimodal_decoder", None)
        return self.bundle.model if decoder is None else decoder, self.bundle.head

    def _trainable_parameters(self):
        return [p for group in self.bundle.parameter_groups.values() for p in group.values()]

    def _gradient_buckets(self):
        # Last decoder layer completes backward first. Each bucket can start
        # its all-reduce while earlier layers are still computing gradients.
        # A trainable router reduces with its layer's adapters.
        layers = {}
        for group in self.bundle.parameter_groups.values():
            for name, parameter in group.items():
                layer = int(name.split(".")[1])
                layers.setdefault(layer, []).append(parameter)
        return [layers[index] for index in sorted(layers, reverse=True)]

    def _build_optimizer(self):
        groups = [
            {"params": list(params.values()), "name": name,
             "lr": self.config.flash.group_learning_rates.get(name, self.config.learning_rate),
             # Decoupled decay pulls toward zero, which for a router is not
             # toward the base model but toward uniform routing.
             **({"weight_decay": 0.0} if name == "routers" else {})}
            for name, params in self.bundle.parameter_groups.items() if params
        ]
        if not groups:
            raise ValueError("Flash has no trainable parameters")
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def _set_optimizer_learning_rate(self, step_index):
        rate = self._learning_rate_for_step(step_index)
        scale = rate / self.config.learning_rate
        for group in self.optimizer.param_groups:
            base = self.config.flash.group_learning_rates.get(group["name"], self.config.learning_rate)
            group["lr"] = base * scale
        return self._current_learning_rate()

    def _token_scores(self, batch, *, compute_entropies):
        if "pixel_values_videos" in batch:
            raise ValueError("Flash RL supports images, not video inputs")
        if "pixel_values" in batch and getattr(self, "_multimodal_decoder", None) is None:
            raise ValueError("Image batches require training.multimodal=true")
        from training.flash.decision_replay import DecisionTape

        # One tape per graph freezes discrete decisions during checkpoint replay,
        # never across policies or the disabled-adapter reference forward.
        tape = DecisionTape().record() if self.config.flash.decision_replay else nullcontext()
        try:
            # Parity scores stay rank-local: the gate pools every rank's error
            # statistics (DistributedCycleMixin._pool_parity_evidence).
            with tape:
                return super()._token_scores(batch, compute_entropies=compute_entropies)
        finally:
            if not torch.is_grad_enabled():
                self._reshard_policy()

    def _parity_thresholds(self):
        # Above the Flash engine's own floor (training/FLASH_RL.md, "Parity thresholds").
        return self.config.flash.parity_thresholds

    def _reference_token_logprobs(self, batch):
        with torch.no_grad(), adapters_disabled(self.bundle.model):
            return self._token_logprobs(batch).detach()

    def _training_world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    def _set_fsdp_gradient_sync(self, *, is_final_slot):
        self._final_microbatch = is_final_slot

    def _micro_batch_forward_backward(self, chunk, *args, **kwargs):
        reducer = self._gradient_reducer
        context = reducer.no_sync() if reducer is not None and not self._final_microbatch else nullcontext()
        # FSDP padding slots run at zero backward weight: their tokens are not
        # trained, so their routing is not counted.
        if getattr(self, "_expert_load", None) is not None and kwargs.get("backward_scale", 1.0) != 0.0:
            self._pending_expert_load = self._expert_load_weights(chunk, TRAINING_ROLES)
        try:
            with context:
                return super()._micro_batch_forward_backward(chunk, *args, **kwargs)
        finally:
            self._pending_expert_load = None

    def _token_logprobs_and_entropies(self, batch):
        # The micro-batch's policy forward (its first scoring call) counts the
        # routing of its tokens. The capture closes before backward, so
        # checkpoint recomputation never counts twice.
        weights = getattr(self, "_pending_expert_load", None)
        if weights is None:
            return super()._token_logprobs_and_entropies(batch)
        self._pending_expert_load = None
        with self._expert_load.capture(TRAINING_ROLES, weights.to(self.device)):
            return super()._token_logprobs_and_entropies(batch)

    def _expert_load_weights(self, chunk, roles):
        # The flex-QSA model packs a padded batch into one stream before its
        # first layer; the SDPA model routes it row by row.
        alignment = None
        if getattr(self.bundle.model, "_flash_flex_qsa", False):
            from training.flash.flex_qsa import QUERY_TILE as alignment
        return router_token_weights(chunk, roles, packed_alignment=alignment)

    def train_window(self, rollouts, **kwargs):
        recorder = getattr(self, "_expert_load", None)
        if recorder is not None:
            recorder.reset()  # a failed earlier window must not leak counts
        self._window_expert_load = {}
        result = super().train_window(rollouts, **kwargs)
        result.expert_load = self._window_expert_load
        drift = self._router_drift_metrics()
        if drift:
            result.policy_update_metrics.update(drift)
            if not dist.is_initialized() or dist.get_rank() == 0:
                log.info("routers: %s", " ".join(f"{key}={value:.4g}" for key, value in drift.items()))
        return result

    def _remember_checkpoint_routers(self):
        """Keep the checkpoint's routers, the reference of their drift."""
        self._checkpoint_routers = {
            name: p.detach().to(torch.bfloat16, copy=True) for name, p in self._router_parameters().items()
        }

    def _router_drift_metrics(self):
        """The trainable routers' relative drift from the checkpoint. Expert load
        balance is the expert-load recorder's (training/expert_load.py); this
        reads the replicated masters only, so it needs no collective."""
        checkpoint = getattr(self, "_checkpoint_routers", None)
        if not checkpoint:
            return {}
        with torch.no_grad():
            routers = self._router_parameters()
            moved = sum(float((p.float() - checkpoint[name].float()).norm()) ** 2 for name, p in routers.items())
            scale = sum(float(checkpoint[name].float().norm()) ** 2 for name in routers)
        return {"router_relative_drift": math.sqrt(moved / scale) if scale else 0.0}

    def _close_expert_load_window(self, output_dir):
        """Run the eval pass, pool every rank's counts (one all_reduce) and
        summarize them. Collective: every rank calls it at the window's end."""
        summary = {}
        episodes = self.config.flash.load_balance_eval_episodes
        if episodes > 0:
            payload = [None]
            if not dist.is_initialized() or dist.get_rank() == 0:
                # output_dir is <run root>/checkpoints; official eval hands its
                # episodes to <run root>/expert_load_eval.
                handoff = take_eval_handoff(Path(output_dir).parent)
                if handoff is not None:
                    payload[0] = handoff["samples"]
            if dist.is_initialized():
                dist.broadcast_object_list(payload, src=0)
            samples = self._eval_expert_load_samples(payload[0] or [], episodes)
            if samples:
                started = time.perf_counter()
                self._measure_eval_expert_load(samples)
                summary["eval_seconds"] = time.perf_counter() - started
                summary["eval_episodes"] = float(len({s.trajectory_id for s in samples}))
        counts = self._expert_load.pooled_counts()
        return {**summarize_counts(counts, top_k=self._expert_load.top_k),
                **({"eval_pass": summary} if summary else {})}

    def _eval_expert_load_samples(self, samples, episodes):
        # The handoff holds at most `episodes` episodes; enforce the bound and
        # the training sequence budget here too.
        limit = max(1, int(self.config.max_sequence_tokens))
        kept, trajectories = [], []
        for sample in samples:
            if len(sample.input_ids) > limit or not any(sample.completion_mask):
                continue
            if sample.trajectory_id not in trajectories:
                if len(trajectories) == episodes:
                    continue
                trajectories.append(sample.trajectory_id)
            kept.append(sample)
        return kept

    def _measure_eval_expert_load(self, samples):
        """No-grad forwards over eval episodes that count their routing.

        Each rank scores its round-robin share of the micro-batches and pads
        to the same number of forwards (FSDP lockstep), as the probe entropy
        does. Micro-batches are cut without shuffling: this pass must not
        consume the run's random numbers."""
        ordered = sorted(samples, key=_sample_token_length, reverse=True)
        chunks = _pack_micro_batches(
            ordered, batch_size=max(1, self.config.micro_batch_size),
            token_budget=max(0, int(getattr(self.config, "max_micro_batch_tokens", 0))),
        )
        share, padding = self._diagnostic_shard(chunks)
        self._set_policy_mode_for_logprob_computation()
        processor = getattr(self, "processor", None)
        # Padding forwards keep FSDP in step; their routing is not counted.
        forwards = [(chunk, True) for chunk in share] + [(chunk, False) for chunk in padding]
        with torch.no_grad():
            for chunk, counted in forwards:
                batch = collate_samples(chunk, pad_token_id=self.tokenizer.pad_token_id, processor=processor)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                if not counted:
                    self._token_logprobs(batch)
                    continue
                weights = self._expert_load_weights(chunk, EVAL_ROLES).to(self.device)
                with self._expert_load.capture(EVAL_ROLES, weights):
                    self._token_logprobs(batch)

    def _synchronize_replicated_gradients(self):
        if self._gradient_reducer is not None:
            self._gradient_reducer.finish()

    def _global_grad_norm_and_clip(self, max_norm):
        # Adapter masters and their averaged gradients are replicated.
        return float(torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm, error_if_nonfinite=True))

    def _peft_config(self):
        return peft_adapter_config(
            base_model_name_or_path=self.config.model_name, r=self.config.lora.r,
            alpha=self.config.lora.alpha, target_modules=self.config.lora.target_modules,
        )

    def _adapter_parameters(self):
        """The LoRA masters PEFT exports; routers are checkpointed separately."""
        return {name: p for group_name, group in self.bundle.parameter_groups.items() if group_name != "routers"
                for name, p in group.items()}

    def _router_parameters(self):
        """Trainable FP32 router masters by learner name (layers.<i>.mlp.gate.weight)."""
        return dict(self.bundle.parameter_groups.get("routers", {})) if self.config.flash.train_routers else {}

    def _published_router_tensors(self):
        # The BF16 cast of each master is exactly what the learner routes with.
        return {hf_router_name(int(name.split(".")[1])): parameter.detach().to(torch.bfloat16).cpu()
                for name, parameter in self._router_parameters().items()}

    def _save_latest_adapter(self, output_dir, iteration_index):
        self._publishing_iteration = iteration_index
        try:
            return super()._save_latest_adapter(output_dir, iteration_index)
        finally:
            self._publishing_iteration = None

    def _write_policy_adapter(self, path):
        from safetensors.torch import save_file

        save_file(to_peft_state_dict(self._adapter_parameters(), self.config.lora),
                  path / "adapter_model.safetensors", metadata={"format": "pt"})
        (path / "adapter_config.json").write_text(json.dumps(self._peft_config(), indent=2) + "\n")
        if self.config.flash.train_routers:
            iteration = getattr(self, "_publishing_iteration", None)
            if iteration is None:
                raise RuntimeError("Flash routers are published only through _save_latest_adapter")
            started_at = time.perf_counter()
            manifest = write_router_publication(path, self._published_router_tensors(), iteration=iteration)
            log.info("published %d routers (%.1f MB, sha256 %s) for iteration %s in %.2fs",
                     manifest["tensors"], (path / ROUTER_FILE).stat().st_size / 1e6, manifest["sha256"],
                     manifest["iteration"], time.perf_counter() - started_at)

    def _dense_lora_targets(self):
        """(HF name, LoRALinear) of every adapted dense target, in layer and name order."""
        from training.flash.dense_lora import LoRALinear

        owners = {id(module.lora_A): module for module in self.bundle.model.modules()
                  if isinstance(module, LoRALinear)}
        targets = []
        for name, parameter in self._adapter_parameters().items():
            if name.endswith(".lora_A"):
                # layers.<i>.<parent>.<leaf>.lora_A
                layer, _, module = name.removeprefix("layers.").removesuffix(".lora_A").partition(".")
                targets.append((hf_weight_name(int(layer), module), owners[id(parameter)]))
        return sorted(targets, key=lambda item: parse_weight_name(item[0]))

    def _materialize_policy_weights(self, iteration_index):
        """Collective: the merged BF16 policy weights, held on rank zero's GPU until they are sent.

        Every rank gathers each target's FSDP-sharded base weight in the same
        order; rank zero merges it with the replicated FP32 masters
        (``W + alpha/r * B @ A`` in FP32, one BF16 rounding of
        ``training.flash.policy_weights_rounding``) and takes its checksum on
        the device. Trained routers add their BF16 casts, exactly what the
        learner routes with. Rank zero returns the publication's manifest, its
        identity and the tensors, which stay on its GPU for the broadcast
        (training/flash_backend.py); None on the other ranks.
        """
        from torch.distributed.tensor import DTensor

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        rounding = self.config.flash.policy_weights_rounding
        started_at = time.perf_counter()
        self._reshard_policy()
        tensors, sums = {}, []
        with torch.no_grad():
            for name, module in self._dense_lora_targets():
                weight = module.base.weight.detach()
                full = weight.full_tensor() if isinstance(weight, DTensor) else weight
                if rank_zero:
                    merged = merge_lora_weight(full, module.lora_A.detach(), module.lora_B.detach(), module.scale,
                                               seed=dither_seed(name), rounding=rounding)
                    sums.append(policy_checksum(merged))
                    tensors[name] = merged.contiguous()
                    del merged
                del full
            if rank_zero:
                for name, parameter in self._router_parameters().items():
                    cast = parameter.detach().to(torch.bfloat16).contiguous()
                    sums.append(policy_checksum(cast))
                    tensors[hf_weight_name(int(name.split(".")[1]), "mlp.gate")] = cast
        if not rank_zero:
            return None
        manifest = policy_manifest(
            iteration=iteration_index, lora_targets=self.config.lora.target_modules,
            routers=self.config.flash.train_routers, rounding=rounding,
            shapes={name: value.shape for name, value in tensors.items()},
            checksums=dict(zip(tensors, torch.cat(sums).tolist(), strict=True)),
        )
        return {"manifest": manifest, "sha256": manifest_sha256(manifest), "tensors": tensors,
                "seconds": time.perf_counter() - started_at}

    def _save_runtime_adapter(self, adapter_dir, *, publish_latest=True):
        """Flash has no runtime export: record the held publication's manifest beside its adapter.

        The tensors come from the collective _materialize_policy_weights that
        precedes every publication (_finalize_window_artifacts, or
        publish_policy_weights at a resume); rank zero writes
        ``<adapter_dir>/policy_weights.json`` (names, shapes, checksums,
        identity, iteration, rounding) and keeps the tensors on its GPU until
        the backend broadcasts them. The adapter directory is the publication's
        durable record.
        """
        held = getattr(self, "_held_policy_weights", None)
        if held is None:
            raise RuntimeError("Flash publishes merged policy weights only after their collective materialization")
        write_policy_manifest(Path(adapter_dir), held["manifest"])
        manifest = held["manifest"]
        log.info(
            "merged policy weights for iteration %s: %d tensors (%s, routers=%s, rounding=%s), %.2f GiB, "
            "identity %s; gather+merge %.2fs", manifest["iteration"], len(manifest["names"]),
            ",".join(manifest["lora_targets"]), manifest["routers"], manifest["rounding"],
            sum(value.numel() for value in held["tensors"].values()) * 2 / 2**30, held["sha256"], held["seconds"],
        )
        return Path(adapter_dir)

    def take_policy_weights(self):
        """Rank zero: the held publication (manifest, identity, tensors in broadcast order); releases it."""
        held, self._held_policy_weights = getattr(self, "_held_policy_weights", None), None
        if held is None:
            raise RuntimeError("no merged policy weights are held for publication")
        manifest = held["manifest"]
        return manifest, held["sha256"], [(name, held["tensors"][name]) for name in manifest["names"]]

    def release_policy_weights(self):
        """Rank zero: drop a held publication that will not be sent (final iteration, deadline)."""
        self._held_policy_weights = None

    def publish_policy_weights(self, adapter_dir, iteration_index):
        """Collective: materialize the current policy's merged weights for ``adapter_dir``.

        A resume calls it (through the distributed runtime) for the restored
        adapter: no publication outlives its job, so every segment publishes
        its restored policy. Rank zero records the manifest beside the adapter
        and holds the tensors for the broadcast. Returns the adapter directory
        on every rank.
        """
        materialized = self._materialize_policy_weights(iteration_index)
        if materialized is not None:
            self._held_policy_weights = materialized
            self._save_runtime_adapter(Path(adapter_dir))
        return Path(adapter_dir)

    def _load_adapter(self, path):
        from safetensors.torch import load_file

        path = Path(path)
        saved = json.loads((path / "adapter_config.json").read_text())
        for key, expected in self._peft_config().items():
            if saved.get(key) != expected:
                raise ValueError(f"Flash adapter config mismatch: {key}")
        from_peft_state_dict(load_file(path / "adapter_model.safetensors"),
                             self._adapter_parameters(), self.config.lora)
        manifest = read_router_manifest(path)
        if not self.config.flash.train_routers:
            if manifest is not None:
                raise ValueError(f"{path} carries trained routers but training.flash.train_routers is false")
            return
        if manifest is None:
            raise ValueError(f"{path} lacks {ROUTER_FILE} although training.flash.train_routers is true")
        # The BF16 published values; a resume then restores the exact FP32
        # masters from the trainer state.
        self._copy_router_masters(self._routers_from_file(path, manifest), source=path / ROUTER_FILE)

    def _routers_from_file(self, path, manifest):
        tensors = load_router_file(path / ROUTER_FILE, manifest["sha256"])
        names = {hf_router_name(int(name.split(".")[1])): name for name in self._router_parameters()}
        if set(tensors) != set(names):
            raise ValueError(f"{path / ROUTER_FILE} has {len(tensors)} routers; the learner trains {len(names)}")
        return {names[key]: value for key, value in tensors.items()}

    def _copy_router_masters(self, values, *, source):
        parameters = self._router_parameters()
        if set(values) != set(parameters):
            raise ValueError(f"{source}: router masters do not match the trainable routers")
        for name, parameter in parameters.items():
            value = values[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"{source}: invalid router master {name}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))

    def _training_state_payload(self, **kwargs):
        payload = {**super()._training_state_payload(**kwargs),
                   "flash_recipe": asdict(self.config.flash),
                   "flash_rank_rng": self._checkpoint_rank_rng}
        held = getattr(self, "_held_policy_weights", None)
        if held is not None:
            # The identity of this policy's publication (its manifest's sha256).
            payload["flash_policy_sha256"] = held["sha256"]
        if self.config.flash.train_routers:
            # The adapter directory holds their BF16 casts; resume needs FP32.
            payload["flash_router_masters"] = {
                name: parameter.detach().to("cpu", copy=True) for name, parameter in self._router_parameters().items()
            }
        return payload

    def _load_training_state_payload(self, state, *, state_path):
        # Older states carry retired settings in their recipe (and, from the
        # amplified-adapter probe, its scale); all are ignored.
        recipe = state.get("flash_recipe")
        if isinstance(recipe, dict):
            drop_retired_settings(recipe, state_path)
        if "adapter_probe_scale" in state:
            warn_retired_setting("adapter_probe", state_path)
        differences = _recipe_differences(state.get("flash_recipe"), asdict(self.config.flash))
        if differences:
            raise ValueError(f"Flash checkpoint recipe differs from the configured recipe: {differences}")
        saved = state.get("training_config", {})
        for key in ("model_name", "model_revision", "lora"):
            expected = asdict(self.config.lora) if key == "lora" else getattr(self.config, key)
            if saved.get(key) != expected:
                raise ValueError(f"Flash checkpoint model/adapter identity differs: {key}")
        rank_rng = state.get("flash_rank_rng", [])
        if len(rank_rng) != self._training_world_size():
            raise ValueError("Exact Flash resume requires the saved world size")
        super()._load_training_state_payload(state, state_path=state_path)
        if self.config.flash.train_routers:
            masters = state.get("flash_router_masters")
            if not isinstance(masters, dict) or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in masters.values()
            ):
                raise ValueError("Flash checkpoint lacks FP32 router masters although train_routers is true")
            self._copy_router_masters(masters, source=state_path)
        FullWeightDistributedRuntime._restore_rng_state(rank_rng[dist.get_rank() if dist.is_initialized() else 0])

    def _finalize_window_artifacts(self, output_dir, iteration_index, *, check_health=None):
        # Before the RNG capture, so a resume continues from the same state
        # even if the expert load eval pass ever consumed random numbers.
        if getattr(self, "_expert_load", None) is not None and iteration_index >= 0:
            self._window_expert_load = self._close_expert_load_window(output_dir)
        rng = FullWeightDistributedRuntime._capture_rng_state()
        self._checkpoint_rank_rng = [rng]
        if dist.is_initialized():
            self._checkpoint_rank_rng = [None] * dist.get_world_size()
            dist.all_gather_object(self._checkpoint_rank_rng, rng)
        # Health callbacks use the command runtime, so all ranks must call them.
        if check_health is not None:
            check_health()
        materialized = self._materialize_policy_weights(iteration_index)
        artifacts = [None]
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Held until the backend sends it at the rollout boundary.
            self._held_policy_weights = materialized
            try:
                artifacts[0] = super()._finalize_window_artifacts(output_dir, iteration_index)
            except BaseException:
                self._held_policy_weights = None
                raise
        if dist.is_initialized():
            dist.broadcast_object_list(artifacts, src=0)
        return artifacts[0]

    def shutdown_workers(self):
        if self._gradient_reducer is not None:
            self._gradient_reducer.close()
        super().shutdown_workers()
