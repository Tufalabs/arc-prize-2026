"""Flash RL hooks on the random two-layer GDN/QSA checkpoint.

The distributed check also runs from test_learner_gradient_sync under torchrun.
"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist

from training.config import TrainingConfig
from training.flash.dense_lora import ALL_TARGETS
from training.flash.gradient_sync import ReplicatedGradientReducer
from training.flash_ppo import FlashPPOTrainer


def make_trainer(bundle, **flash):
    trainer = FlashPPOTrainer.__new__(FlashPPOTrainer)
    trainer.config = TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.8-Flash-Next", "output_dir": "runs", "device": "cuda",
        "weight_mode": "flash", "flash": {"checkpoint_root": "/fixture", **flash},
        "lora": {"r": 8, "alpha": 16, "dropout": 0, "target_modules": list(ALL_TARGETS)},
    })
    trainer.bundle = bundle
    trainer.model = bundle.model
    trainer.device = torch.device("cuda", torch.cuda.current_device())
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.optimizer = trainer._build_optimizer()
    trainer._optimizer_step_count = 0
    trainer._gradient_reducer = None
    return trainer


def batch():
    ids = (torch.arange(64, device="cuda").reshape(2, 32) % 254) + 1
    mask = torch.ones_like(ids)
    mask[1, 19:] = 0
    ids[1, 19:] = 0
    labels = ids.clone()
    labels[:, :4] = -100
    labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@unittest.skipUnless(torch.cuda.is_available(), "Flash requires CUDA/FLA")
class FlashPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.distillation.flash_test_model import build_test_model

        cls.trainer = make_trainer(build_test_model(fast=True))

    def test_scores_padding_reference_and_optimizer_census(self):
        from tests.distillation.flash_test_model import build_test_model, snapshot_kernels
        from training.flash.policies import TrainabilityPolicy
        from training.flash.input_layout import pack_padded_batch
        from training.flash.flex_qsa import QUERY_TILE

        trainer = self.trainer
        data = batch()
        mask = data["labels"][:, 1:] != -100
        with snapshot_kernels(fixed_config=True), torch.no_grad(), trainer._autocast_context():
            actual, entropy = trainer._token_scores(data, compute_entropies=True)
            hidden = trainer.model(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).last_hidden_state
            logits = trainer.bundle.head(hidden[:, :-1][mask]).float()
            expected = logits.log_softmax(-1).gather(-1, data["input_ids"][:, 1:][mask].unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(actual[mask], expected, rtol=0, atol=1e-6)
            self.assertTrue(torch.isfinite(entropy).all())
            layout = pack_padded_batch(data["input_ids"], data["attention_mask"], None,
                                       pad_token_id=0, eos_token_id=0, alignment=QUERY_TILE)
            packed = trainer.model(input_ids=layout.input_ids, packed_lengths=layout.packed_lengths).last_hidden_state
            torch.testing.assert_close(hidden, layout.unpack(packed), rtol=0, atol=0)
            reference = trainer._reference_token_logprobs(data)
            frozen = build_test_model(fast=True, lora=False, trainability=TrainabilityPolicy.frozen_reference())
            frozen_hidden = frozen.model(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).last_hidden_state
            frozen_logits = frozen.head(frozen_hidden[:, :-1][mask]).float()
            frozen_logprobs = frozen_logits.gather(-1, data["input_ids"][:, 1:][mask].unsqueeze(-1)).squeeze(-1) - frozen_logits.logsumexp(-1)
            torch.testing.assert_close(reference[mask], frozen_logprobs, rtol=0, atol=0)
        trainables = {id(p) for p in trainer.model.parameters() if p.requires_grad}
        optimized = [p for group in trainer.optimizer.param_groups for p in group["params"]]
        self.assertEqual(trainables, {id(p) for p in optimized})
        self.assertEqual(len(trainables), len(optimized))

    def test_training_cycle_reports_expert_load(self):
        from tests.distillation.flash_test_model import build_test_model, snapshot_kernels
        from training.dataset import TokenizedSample
        from training.expert_load import EVAL_HANDOFF_DIR, install_expert_load_recorder
        from training.flash.adapters import FP32TopKRouter
        from training.memory_profile import NullMemoryLogger
        from training.metrics import EXPERT_LOAD_SCALARS, expert_load_scalars
        from training.ppo import PPOTrainer

        trainer = make_trainer(build_test_model(fast=True))
        recorder = install_expert_load_recorder(trainer.bundle.model, router_type=FP32TopKRouter,
                                                device=trainer.device)
        trainer._expert_load = recorder
        self.assertEqual((tuple(recorder.counts.shape), recorder.top_k), ((3, 2, 12), 10))
        cycle, tokens, decisions = [], 0, 0
        with snapshot_kernels(fixed_config=True), torch.no_grad():
            for micro, length in enumerate((32, 19, 150)):
                ids = [(index * 7 + micro) % 254 + 1 for index in range(length)]
                completion = [0] * 4 + [1] * (length - 4)
                data = {"input_ids": torch.tensor([ids], device="cuda"),
                        "attention_mask": torch.ones(1, length, dtype=torch.long, device="cuda"),
                        "labels": torch.tensor([[-100] * 4 + ids[4:]], device="cuda")}
                old = trainer._token_logprobs(data)[0].tolist()
                cycle.append([TokenizedSample(
                    trajectory_id=str(micro), step_index=0, input_ids=ids,
                    labels=[-100] * 4 + ids[4:], attention_mask=[1] * length,
                    completion_mask=completion, prompt_token_count=4,
                    reward=1, return_to_go=1, advantage=0.25 + micro, old_token_logprobs=old,
                )])
                tokens += length
                decisions += length - 4
        # Scoring the old logprobs was not a training forward: nothing counted.
        self.assertEqual(float(recorder.counts.sum()), 0.0)
        samples = [chunk[0] for chunk in cycle]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The eval pass, before any update, over the same episodes.
            (root / EVAL_HANDOFF_DIR).mkdir()
            torch.save({"version": 1, "cycle_index": 0, "policy_iteration": 0, "episodes": 3,
                        "samples": samples}, root / EVAL_HANDOFF_DIR / "cycle_00000.pt")
            with snapshot_kernels(fixed_config=True):
                before = trainer._close_expert_load_window(root / "checkpoints")
            self.assertEqual(set(before), {"eval", "eval_pass"})
            self.assertEqual(before["eval"]["tokens"], decisions)
            # Independently, from the same forwards: the routers' raw indices
            # in packed order. Each sample is its own micro-batch (longest
            # first), a lone document at the stream's start, so position i
            # (deciding token i + 1) is a sampled decision for 3 <= i < len - 1.
            # Same forwards, because two forwards of one sample need not route
            # identically (a few tokens differed on a B200, job 2919089).
            routers = [m for m in trainer.bundle.model.modules() if isinstance(m, FP32TopKRouter)]
            raw = []

            def tee(layer, indices, values):
                raw.append((layer, indices.clone()))
                recorder.record(layer, indices, values)

            for router in routers:
                router.route_recorder = SimpleNamespace(record=tee)
            try:
                with snapshot_kernels(fixed_config=True):
                    trainer._measure_eval_expert_load(samples)
            finally:
                for router in routers:
                    router.route_recorder = recorder
            eval_counts = recorder.counts[2].clone()
            recorder.reset()
            ordered = sorted(samples, key=lambda sample: len(sample.input_ids), reverse=True)
            self.assertEqual(len(raw), 2 * len(ordered))
            expected = torch.zeros_like(eval_counts)
            for position, (layer, indices) in enumerate(raw):
                length = len(ordered[position // 2].input_ids)
                expected[layer] += torch.bincount(indices[3:length - 1].flatten(), minlength=12).double()
            torch.testing.assert_close(eval_counts, expected, rtol=0, atol=0)
            names = ("mean_ratio_sum", "clip_fraction_sum", "entropy_sum", "entropy_loss_sum",
                     "previous_policy_kl_sum", "reference_kl_sum", "ratio_updates", "final_loss",
                     "grad_norm_sum", "max_grad_norm", "grad_norm_updates", "global_step")
            with snapshot_kernels(fixed_config=True):
                PPOTrainer._execute_train_cycle(
                    trainer, cycle, world_size=1, accumulators=dict.fromkeys(names, 0), cycle_index=1,
                    epoch_label="test", epoch_number=1, iteration_index=0, loss_denominator=float(decisions),
                    mem=NullMemoryLogger(), expected_optimizer_steps_total=1)
            # Every layer counts each token once (top-10), although backward
            # recomputed every checkpointed layer's routers.
            per_layer = recorder.counts.sum(-1)
            torch.testing.assert_close(per_layer[0], torch.full((2,), 10.0 * tokens, dtype=torch.float64,
                                                                device="cuda"), rtol=0, atol=0)
            torch.testing.assert_close(per_layer[1], torch.full((2,), 10.0 * decisions, dtype=torch.float64,
                                                                device="cuda"), rtol=0, atol=0)
            # The training forward routed the sampled positions as the no-grad
            # eval pass did with the same weights (packed layout and roles agree).
            difference = float((recorder.counts[1] - eval_counts).abs().sum())
            self.assertLessEqual(difference, 0.01 * float(eval_counts.sum()), difference)
            with snapshot_kernels(fixed_config=True):
                summary = trainer._close_expert_load_window(root / "checkpoints")
        scalars = expert_load_scalars(summary)
        for prefix in ("train", "rollout"):
            self.assertLessEqual({f"{prefix}/{name}" for name in EXPERT_LOAD_SCALARS}, set(scalars))
            self.assertGreaterEqual(scalars[f"{prefix}/expert_load_max_share"], 1.0)
            self.assertGreater(scalars[f"{prefix}/expert_load_entropy"], 0.0)
            self.assertLessEqual(scalars[f"{prefix}/expert_load_entropy"], 1.0 + 1e-9)
        self.assertEqual((summary["train"]["tokens"], summary["rollout"]["tokens"]), (tokens, decisions))
        self.assertFalse([name for name in scalars if name.startswith("eval/")])

    def test_save_load_save_bitwise(self):
        trainer = self.trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = trainer._finalize_window_artifacts(root / "first", 0)
            expected = (first.adapter_dir / "adapter_model.safetensors").read_bytes()
            for p in trainer._trainable_parameters():
                p.data.zero_()
            trainer._load_adapter(first.adapter_dir)
            trainer.load_training_state(first.checkpoint_path)
            second = trainer._save_latest_adapter(root / "second", 0)
            self.assertEqual(expected, (second / "adapter_model.safetensors").read_bytes())


def cycle_samples(reference):
    """Three one-sample chunks scored by ``reference`` as the sampler's logprobs."""
    from tests.distillation.flash_test_model import snapshot_kernels
    from training.dataset import TokenizedSample

    cycle = []
    with snapshot_kernels(fixed_config=True), torch.no_grad():
        for micro in range(3):
            data = {key: value[:1].clone() for key, value in batch().items()}
            data["input_ids"] = (data["input_ids"] + micro) % 254 + 1
            ids = data["input_ids"][0].tolist()
            old = reference._token_logprobs(data)[0].tolist()
            cycle.append([TokenizedSample(
                trajectory_id=str(micro), step_index=0, input_ids=ids,
                labels=[-100] * 4 + ids[4:], attention_mask=[1] * 32,
                completion_mask=[0] * 4 + [1] * 28, prompt_token_count=4,
                reward=1, return_to_go=1, advantage=0.25 + micro,
                old_token_logprobs=old,
            )])
    return cycle


CYCLE_ACCUMULATORS = ("mean_ratio_sum", "clip_fraction_sum", "entropy_sum", "entropy_loss_sum",
                      "previous_policy_kl_sum", "reference_kl_sum", "ratio_updates", "final_loss",
                      "grad_norm_sum", "max_grad_norm", "grad_norm_updates", "global_step")


def cycle_options():
    from training.memory_profile import NullMemoryLogger

    return dict(cycle_index=1, epoch_label="test", epoch_number=1, iteration_index=0,
                loss_denominator=84.0, mem=NullMemoryLogger(), expected_optimizer_steps_total=1)


@unittest.skipUnless(torch.cuda.is_available(), "Flash requires CUDA/FLA")
class FlashRouterTests(unittest.TestCase):
    def test_router_cycle_publication_and_bitwise_reload(self):
        """Routers train for one PPO cycle beside the LoRA, publish as the BF16
        file the learner routes with, and reload exactly from the trainer state."""
        import hashlib

        from safetensors.torch import load_file
        from tests.distillation.flash_test_model import build_test_model, random_tensor, snapshot_kernels
        from training.expert_load import install_expert_load_recorder
        from training.flash.adapters import FP32TopKRouter
        from training.flash_routers import hf_router_name

        trainer = make_trainer(build_test_model(fast=True, train_routers=True), train_routers=True)
        trainer._remember_checkpoint_routers()
        # The expert-load recorder owns the trainable routers' route_recorder
        # hook, as _load_policy_model installs it.
        recorder = install_expert_load_recorder(trainer.bundle.model, router_type=FP32TopKRouter,
                                                device=trainer.device)
        trainer._expert_load = recorder
        routers = trainer._router_parameters()
        self.assertEqual(sorted(routers), ["layers.0.mlp.gate.weight", "layers.1.mlp.gate.weight"])
        groups = {group["name"]: group for group in trainer.optimizer.param_groups}
        self.assertEqual({id(p) for p in groups["routers"]["params"]}, {id(p) for p in routers.values()})
        self.assertEqual(groups["routers"]["weight_decay"], 0.0)
        checkpoint = {hf_router_name(i): random_tensor(hf_router_name(i), routers[f"layers.{i}.mlp.gate.weight"].shape)
                      for i in range(2)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = trainer._finalize_window_artifacts(root / "initial", -1)
            published = load_file(initial.adapter_dir / "routers.safetensors")
            for name, value in checkpoint.items():
                self.assertTrue(torch.equal(published[name], value), name)
            before = {name: p.detach().clone() for name, p in routers.items()}
            norms = {}
            hooks = [p.register_post_accumulate_grad_hook(
                lambda p, name=name: norms.__setitem__(name, float(p.grad.norm()))) for name, p in routers.items()]
            try:
                from training.ppo import PPOTrainer

                with snapshot_kernels(fixed_config=True):
                    PPOTrainer._execute_train_cycle(trainer, cycle_samples(trainer), world_size=1,
                                                    accumulators=dict.fromkeys(CYCLE_ACCUMULATORS, 0),
                                                    **cycle_options())
            finally:
                for hook in hooks:
                    hook.remove()
            print({"router_grad_norms": norms}, flush=True)
            self.assertEqual(sorted(norms), sorted(routers))
            self.assertTrue(all(value > 0 and value == value for value in norms.values()), norms)
            self.assertTrue(all(not torch.equal(before[name], p) for name, p in routers.items()))
            # Drift reads the masters only; load balance stays the recorder's,
            # which counted the trainable routers' training forwards.
            drift = trainer._router_drift_metrics()
            print({"router_drift_metrics": drift}, flush=True)
            self.assertEqual(set(drift), {"router_relative_drift"})
            self.assertGreater(drift["router_relative_drift"], 0)
            counted = recorder.counts.sum(-1)
            torch.testing.assert_close(counted[0], torch.full((2,), 10.0 * 96, dtype=torch.float64,
                                                              device="cuda"), rtol=0, atol=0)
            torch.testing.assert_close(counted[1], torch.full((2,), 10.0 * 84, dtype=torch.float64,
                                                              device="cuda"), rtol=0, atol=0)
            trainer._window_expert_load = {}
            artifacts = trainer._finalize_window_artifacts(root / "run", 0)
            load = trainer._window_expert_load
            self.assertEqual((load["train"]["tokens"], load["rollout"]["tokens"]), (96, 84))
            self.assertEqual(float(recorder.counts.sum()), 0.0)
            data = (artifacts.adapter_dir / "routers.safetensors").read_bytes()
            manifest = json.loads((artifacts.adapter_dir / "routers.json").read_text())
            self.assertEqual(manifest, {"file": "routers.safetensors", "sha256": hashlib.sha256(data).hexdigest(),
                                        "iteration": 0, "tensors": 2})
            # The servers receive merged weights: every LoRA target folded into its
            # base in FP32 and rounded once, and the routers' BF16 casts, held on
            # rank zero's GPU for the broadcast with only the manifest written.
            from training.flash_policy_weights import (
                dither_seed, merge_lora_weight, read_policy_manifest, tensor_checksums,
            )

            policy, digest = read_policy_manifest(artifacts.publication_path)
            held, held_digest, pairs = trainer.take_policy_weights()
            merged = dict(pairs)
            self.assertEqual((held, held_digest), (policy, digest))
            self.assertEqual((policy["iteration"], policy["routers"]), (0, True))
            targets = trainer._dense_lora_targets()
            self.assertEqual(len(merged), len(targets) + len(routers))
            for name, module in targets:
                self.assertTrue(merged[name].is_cuda, name)
                self.assertTrue(torch.equal(merged[name], merge_lora_weight(
                    module.base.weight.detach(), module.lora_A.detach(), module.lora_B.detach(), module.scale,
                    seed=dither_seed(name),
                )), name)
            for name, p in routers.items():
                self.assertTrue(torch.equal(merged[hf_router_name(int(name.split(".")[1]))], p.detach().bfloat16()))
            self.assertEqual(policy["checksums"], tensor_checksums(merged.values()))
            self.assertFalse(any("gate.weight" in key and "mlp.gate" in key
                                 for key in load_file(artifacts.adapter_dir / "adapter_model.safetensors")))
            published = load_file(artifacts.adapter_dir / "routers.safetensors")
            for name, p in routers.items():
                self.assertTrue(torch.equal(published[hf_router_name(int(name.split(".")[1]))],
                                            p.detach().bfloat16().cpu()))
            # The learner routes with the BF16 cast: the published file scores identically.
            data_batch = batch()
            with snapshot_kernels(fixed_config=True), torch.no_grad(), trainer._autocast_context():
                masters = trainer._token_logprobs(data_batch)
            exact = {name: p.detach().clone() for name, p in routers.items()}
            trainer._load_adapter(artifacts.adapter_dir)
            with snapshot_kernels(fixed_config=True), torch.no_grad(), trainer._autocast_context():
                rounded = trainer._token_logprobs(data_batch)
            self.assertTrue(torch.equal(masters, rounded))
            trainer.load_training_state(artifacts.checkpoint_path)
            for name, p in routers.items():
                self.assertTrue(torch.equal(p, exact[name]), name)
            again = trainer._save_latest_adapter(root / "again", 0)
            self.assertEqual((again / "routers.safetensors").read_bytes(), data)
            self.assertEqual((again / "adapter_model.safetensors").read_bytes(),
                             (artifacts.adapter_dir / "adapter_model.safetensors").read_bytes())


@unittest.skipUnless(torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 2,
                     "Two CUDA ranks required")
class FlashDistributedTests(unittest.TestCase):
    def test_two_rank_accumulation_matches_single_rank(self):
        run_two_rank_flash_step()

    def test_two_rank_router_accumulation_matches_single_rank(self):
        run_two_rank_flash_step(train_routers=True)


def run_two_rank_flash_step(*, train_routers=False):
    """Also run inside the reducer suite's existing process group."""
    from tests.distillation.flash_test_model import build_test_model, snapshot_kernels

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    reducer = None
    try:
        reference = make_trainer(build_test_model(fast=True, train_routers=train_routers),
                                 train_routers=train_routers)
        # Trainable routers compose with replicate_frozen_routers: they
        # replicate as trainables and reduce with their layer's bucket.
        distributed = make_trainer(build_test_model(fast=True, distributed=True, train_routers=train_routers,
                                                     replicate_frozen_routers=True),
                                   train_routers=train_routers)
        buckets = distributed._gradient_buckets()
        expected = {name: p for group in distributed.bundle.parameter_groups.values() for name, p in group.items()}
        assert any("mlp.gate" in name for name in expected) == train_routers
        assert len(buckets) == 2
        for layer, bucket in zip((1, 0), buckets, strict=True):
            assert {id(p) for p in bucket} == {
                id(p) for name, p in expected.items() if name.startswith(f"layers.{layer}.")
            }
        reducer = ReplicatedGradientReducer(buckets)
        distributed._gradient_reducer = reducer
        from training.ppo import PPOTrainer

        # Three chunks force two slots on each rank, including a zero-loss
        # padding chunk. This exercises the actual PPO cycle and no_sync hook.
        cycle = cycle_samples(reference)
        with snapshot_kernels(fixed_config=True):
            PPOTrainer._execute_train_cycle(reference, cycle, world_size=1,
                                             accumulators=dict.fromkeys(CYCLE_ACCUMULATORS, 0), **cycle_options())
            distributed._execute_train_cycle(cycle, world_size=2,
                                              accumulators=dict.fromkeys(CYCLE_ACCUMULATORS, 0), **cycle_options())
        for a, b in zip(distributed._trainable_parameters(), reference._trainable_parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            # Adam's first update can hide a uniform gradient-scale error;
            # compare moments as well as the resulting parameter values.
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(distributed.optimizer.state[a][key],
                                           reference.optimizer.state[b][key], rtol=2e-5, atol=1e-8)
        assert distributed.optimizer_step_count == reference.optimizer_step_count
        # Parity scores stay rank-local (each rank trains on its own); the
        # gate pools the ranks' error statistics so every rank takes the same
        # branch. Deliberate input differences stand in for rank-local
        # numerical differences.
        data = batch()
        data["input_ids"] = (data["input_ids"] + rank) % 254 + 1
        with torch.no_grad(), snapshot_kernels(fixed_config=True):
            parity_scores = distributed._token_logprobs(data)
        gathered = [torch.empty_like(parity_scores) for _ in range(2)]
        dist.all_gather(gathered, parity_scores)
        assert not torch.equal(gathered[0], gathered[1])
        evidence = {"nonfinite_count": 0, "token_count": 128, "mean_abs": 0.1 * (rank + 1), "p95_abs": 0.3,
                    "max_abs": 1.0 + rank, "outlier_count": rank, "inlier_mean_abs": 0.1, "signed_mean": -rank,
                    "biased_token_count": 0, "biased_mean_abs": 0.0}
        pooled = [None, None]
        dist.all_gather_object(pooled, distributed._pool_parity_evidence(evidence))
        assert pooled[0] == pooled[1]
        assert (pooled[0]["mean_abs"], pooled[0]["max_abs"], pooled[0]["outlier_count"]) == (0.2, 2.0, 1)
    finally:
        if reducer is not None:
            reducer.close()
        if owns_group:
            dist.destroy_process_group()
