# Shared Qwen3.8-Flash model stack

This package contains trainer-independent Qwen3.8-Flash checkpoint loading and
execution code. Distillation and RL packages own their objectives, data,
launchers, telemetry, and qualified recipe assertions.

Construction has two explicit policies:

- `TrainabilityPolicy` selects frozen, router-only, expert-LoRA, or caller-configured layer
  parameters. The custom callback runs before activation-checkpoint and FSDP
  wrapping, which is the extension point for LoRA/adapters.
- `DistributionPolicy` holds the FSDP wrapping choices. A distributed build
  shards every frozen weight with FSDP; the routed experts are BF16 parameters,
  expanded once at load from the checkpoint's NVFP4 (`reader.py`). There is no
  other expert storage or layout (training/FLASH_RL.md, "Expert storage").
  Process-count restrictions belong to trainer recipes, not this model package.

`sft.distillation.flash_model.build_model` preserves the historical API and
validates the exact configured router/adapter trainable set. Callers
import reusable QSA, FlexAttention, expert, pointwise, packing, and checkpoint
helpers directly from this package.

## Routed expert LoRA

`TrainabilityPolicy(mode="routers", expert_lora={"rank": 8, "alpha": 16,
"dropout": 0})` trains routers and independent gate/up/down adapters for every
routed expert. Use `mode="experts"` to freeze routers. The default remains
router-only. Custom callbacks can still implement other adapter layouts;
`TrainabilityPolicy(mode="custom", ..., train_routers=True)` (as
`dense_lora_policy(..., train_routers=True)` builds it) adds the FP32 routers
to them as the separate `routers` group.

`expert_lora.py` installs FP32 replicated A/B parameters after loading each layer
and before checkpoint/FSDP wrapping. `pointwise_optimizations.py` adds their
low-rank corrections using the existing grouped GEMMs over the frozen BF16
experts. Base expert tensors remain frozen. A starts random
and B starts at zero; initialization preserves the caller's RNG state. Execution
casts A/B to the activation dtype and pads the inner rank to a multiple of 16
without changing the actual trainable rank or alpha/r scale. Only zero dropout
and the optimized expert implementations are supported.

The caller owns gradient synchronization and optimization of the returned
`expert_lora` parameter group, just as it does for replicated routers.

## Attention backends

The builder's default remains `attention_backend="sdpa"`. Selecting
`attention_backend="flex_qsa"` enables the shared sparse-attention adapter:

| `qsa_kernel` | Status and constraints |
| --- | --- |
| `"flex"` | Qualified FlexAttention path, including aligned packed documents. Explicitly select it for historical manifests that omit the kernel field. |
| `"physical_sparse"` | Default for `flex_qsa`; experimental full-model status. Dedicated sparse forward/backward, numerically qualified against Flex at operator level. Requires one unpacked CUDA BF16 document, Q/KV heads 24/2, head dimension 256, four-token blocks and top-512 selection. |
| `"indexed_backward"` | Experimental; Flex forward with the dedicated backward and the same geometry restrictions as `physical_sparse`. |
| `"group_sparse"` | Experimental; operator-qualified, **not full-model qualified**. Packs eight queries and twelve query heads per KV head into tensor-core tiles. Supports packed documents with the same head/block geometry. Deterministic reductions differ from Flex's reduction order. |

Every backend preserves the causal prefix, incomplete tail and document isolation.
The public Flex adapter packs padded batches into aligned documents. Standalone
`group_qsa.attention` also accepts unaligned document lengths; its block IDs use
per-document `ceil(length / 4)` offsets. Group attention uses linear-size metadata
and bounded partial buffers (up to approximately 16.2 GiB for backward); this is
not a claim of linear execution time or a training-trajectory qualification.

`qsa_kernel` is recorded in distillation checkpoint manifests; switching kernels
on resume is rejected. SDPA normalizes the unused kernel field to `"flex"`.
`reuse_qsa_selection=False` remains the builder default. With selection reuse,
Flex retains attention state by default; `retain_qsa_attention=False` opts out.
These existing policies and their manifest normalization are unchanged.

## Optional learner execution

Pass these keyword arguments to `build_flash_model`. All options below are
**experimental**; passing operator tests does not qualify a full training run.
They remain opt-in and do not change the qualified distillation defaults
(`expert_optimization="weighted_permutation"`, `fuse_norm_tail=True`). Internal
adapters receive a frozen `FlashOptions` value during construction; it adds no
manifest fields.

`kernel_mode.py` names the two stacks the trainers use: `reference`, these
builder defaults, and `fast`, the RL learner's selection of the options below
(plus the Blackwell GDN guard).
`resolve_kernel_stack(mode, overrides)` resolves a preset with per-field
overrides; `builder_kwargs` turns the result into these keyword arguments. The
distillation config and `FlashSettings` both resolve through it
([training/FLASH_RL.md](../FLASH_RL.md#kernel-mode)).

| Option (default) | Enable and constraints |
| --- | --- |
| `gdn_convolution="native"` | `"fla"` uses FLA's Triton causal convolution; packed inputs require `cu_seqlens`. Cache-free prefill only. |
| `ple_convolution="native"` | `"compiled"` uses a document-masked four-tap dilated convolution with FP32 accumulation and the existing BF16 rounding before SiLU. |
| `expert_optimization="weighted_permutation"` | `"fused_permutation"` adds deterministic Triton gather and weighted reduction adjoints. Requires ordinary local top-k routing and bias-free, non-transposed experts. |
| `selector_policy="batched_fp32_top513_v1"` | `"compiled_fp32_top513_v1"` retains FP32 scoring with ordered head reduction; `"bf16_dot_fp32_top513_v1"` uses BF16 dot inputs and FP32 scores. Both preserve top-(k+1), prefix, tail and packing rules; BF16 scoring can change selections. |
| `hyper_connection="none"` | `"triton"` fuses norm, activation, stream mixing and residual injection. `"triton_inject_norm"` also combines adjacent injection/norm. `"triton_manual"` combines norm-input gradient branches. All require frozen HC norm weights; manual mode also requires frozen HC projection weights. |
| `fuse_gdn_norm=False` | `True` fuses GDN gated RMSNorm; frozen norm weights and SiLU or sigmoid gates required. |
| `fuse_mlp_gates=False` | `True` fuses shared/routed expert SiLU multiplication; SiLU/Swish only. |
| `fuse_gdn_qkv=False` | `True` fuses packing and repeated-head gradient merging; equal K/V head dimensions, repeated heads, and `gdn_grouped_heads=False` required. |
| `fuse_qsa_norm=False` | `True` fuses frozen QSA projection RMSNorms, including strided query slices. |
| `pointwise_rounding="hf"` | `"vllm"` uses serving-style FP32 HC intermediates and GDN gated normalization. Does not change MLP arithmetic or recurrent decay/update gates. HF mode retains eager BF16 intermediate rounding. Serving mode requires `fuse_gdn_norm=True` and a Triton HC mode for all HC norm, activation, mixing and injection operations. |
| `gdn_grouped_heads=False` | `True` enables FLA grouped-value heads. **Rejected until requalified** by `tests/distillation/test_gdn_grouped_heads.py`; intermittent accuracy failures remain unresolved. |
| `compile_layers=False` | `True` compiles tensor glue with graph breaks around stateful attention, GDN kernels, grouped GEMMs, convolutions and host lookup. `compile_precision="reference"` (default) also keeps projections, norms, router arithmetic and decay gates eager; `"fused"` lets Inductor compile them. |
| `checkpoint_policy="full"` | `"small_gemm"` retains GEMM outputs of width at most 128 in GDN layers. QSA keeps its existing replay contexts. CPU offload is supported. |
| `offload_chunk_mib=None` | A power of two (MiB) copies each offloaded boundary to pinned host memory in slabs of that size instead of one whole tensor. The pinned host allocator rounds every allocation up to a power of two, so whole-tensor offload of long sequences can hold almost twice the boundary size in host RAM. Values are restored bit for bit. |
| `replicate_frozen_routers=False` | `True` excludes frozen FP32 routers from layer/root FSDP sharding, allowing homogeneous BF16 gathers. Requires FSDP (`distributed=True`); trainable routers are replicated as trainables either way, so the flag only affects frozen ones; replication costs additional per-rank memory. |

`DistributionPolicy` accepts `reshard_after_forward=True` (boolean or positive
shard count), `forward_prefetch=0` and `backward_prefetch=0` (layer counts).
Nondefault placement/prefetch settings are experimental and can increase memory.
`gradient_sync.ReplicatedGradientReducer` optionally overlaps replicated adapter
all-reduces with backward. Pass buckets in identical order on every rank, ensure
every adapter receives a gradient, and call `finish()` before clipping/stepping.
For gradient accumulation, wrap all but the final microbatch backward in
`reducer.no_sync()`; the final backward submits accumulated buckets. Contributions
after bucket readiness/submission raise. FSDP owns synchronization of sharded parameters.

### Dense LoRA and the frozen head

`dense_lora.dense_lora_policy(r, alpha, target_modules)` returns an
experimental `TrainabilityPolicy(mode="custom")` whose `configure_layer` wraps
targets and whose `validate_trainables` checks the model-wide result.
Supported targets are QSA `q/k/v/o_proj`, GDN
`in_proj_qkv/in_proj_z/out_proj`, and shared-expert `gate/up/down_proj`.
Unknown or empty lists fail, as do requested targets that match nowhere across
the constructed model; per-layer skips remain valid. Routed experts are untouched. Adapters have FP32
masters, zero dropout and zero-initialized B. The caller synchronizes their
replicated gradients. Set `fuse_epilogue=True` on the policy to enable the
experimental BF16 scale/add fusion; the default is `False`.

`completion_loss(..., cache_logits=True)` experimentally retains BF16 frozen-head
logits until loss backward, avoiding the second head GEMM. It requires frozen
head weights and nonempty targets, trades memory for recomputation, and releases
logits before decoder backward. Default `False` preserves the original loss path.

### Decision replay and numerical tests

`DecisionTape` is an opt-in diagnostic that freezes QSA block IDs and expert
top-k indices by layer. Routing weights are recomputed from live probabilities,
so router gradients remain live. Use a fresh tape per input/rank and identical
shapes and packed boundaries:

```python
from training.flash.decision_replay import DecisionTape

tape = DecisionTape()
with tape.record():
    reference_forward_and_loss().backward()
with tape.replay():
    candidate_forward_and_loss().backward()
```

SDPA indexers reject active tapes before selection; use `flex_qsa`. Tape contexts
reject inference mode at entry.

Flash checkpoint wrappers capture the tape per graph, including retention,
recomputation on autograd worker threads, and repeated backward. Third-party
non-reentrant checkpoint calls should use `decision_replay.checkpoint_contexts`
(or compose it with their own contexts). Tapes are runtime data, not model state
or a checkpoint-manifest setting; the default retains no decisions.

Run `python -m unittest discover -s tests/distillation` in the pinned Flash GPU
runtime. Custom backward tests compare outputs and gradients against unfused
FP32 autograd: RMS and maximum error must be at most 1.5 times eager BF16's error
against that same reference. Tests include production widths, packed documents
and bitwise repeatability for deterministic kernels. A two-layer GDN/QSA test
adds nonzero dense LoRA and frozen decisions. B200 builder snapshots separately
check byte-for-byte preservation across the configuration cleanup with test-only
deterministic algorithms, SDPA math, one FLA recurrence configuration and native
CUDA convolutions in a fresh subprocess. Native
unpacked input gradients do not generally repeat bitwise. These checks do not
establish full-depth training-trajectory or serving parity.

### Fused operator dtype contracts

These contracts describe standalone operators, not a model-wide dtype setting.
The builder still loads BF16 execution weights; `flex_qsa` model execution still
requires BF16. Kernels accept strided inputs by copying. Triton-only entry points
reject CPU tensors before launching. FP8 activations, float64, integers and
other floating formats are rejected.

| Operator | CUDA activation dtypes | Mixed inputs/parameters | CPU |
| --- | --- | --- | --- |
| `ChunkedCE`, `CachedHeadCE` | BF16, FP16, FP32 | Mixed head/hidden dtypes require autocast; backward reuses forward GEMM dtype | Supported, including BF16 autocast |
| `norm_gate` | BF16, FP16 | BF16/FP16/FP32 norm weights and gates; HF promotes the affine product, serving retains FP32 intermediates | Rejected; HF module falls back to its reference |
| `swiglu` | BF16, FP16 | Gate/up must share a dtype | Reference fallback, including FP32 |
| `pack_qkv`, `scaled_add` | BF16, FP16, FP32 | Scale/add inputs must share a dtype | Rejected |
| `projection_norm` | BF16, FP16 | BF16/FP16/FP32 frozen norm weights | Rejected; module falls back to its reference |
| HC norm and inject/norm | BF16, FP16 | BF16/FP16/FP32 frozen norm weights; injection inputs must share a dtype | Rejected |
| HC activation, mix and injection | BF16, FP16 | Mix/injection inputs must share a dtype | Rejected |
| HC `triton` module | BF16, FP16 | Mixed projection dtypes require autocast in the activation dtype; frozen norm | HF module fallback; precomputed norm and serving mode rejected |
| HC `triton_manual` | BF16, FP16 | Mixed projection dtypes require autocast in the activation dtype; frozen projections and norm | HF module fallback; explicit injection and serving mode rejected |
| Group QSA | BF16, FP16 | Q/K/V must share a dtype | Rejected; metadata layout/reference utilities support CPU |
| MoE gather | BF16, FP16 | Integer permutations on the activation device | Reference fallback, including FP32 |
| MoE combine | BF16, FP16; FP32 reference fallback | Mixed routing weights use the reference reduction and promotion | Reference fallback |
| FP32 selector scoring | BF16, FP16, FP32 | Query/key dtypes may differ; both cast to FP32 | Reference fallback |
| `bf16_scores` tensor-core selector | BF16, FP16 | Query/key must share a dtype | Rejected |
| Dense LoRA fused epilogue | BF16, FP16, FP32 | FP32 adapter masters; GEMMs follow eager/autocast, epilogue inputs must share a dtype | Reference fallback |

FP16 group-QSA dK/dV waves use FP32 partials to avoid additional half-precision
rounding. BF16 retains the existing scaled-FP16 partials and launch choices: each partial
row is scaled by a power of two so its maximum lies in [4096, 8192), so only
elements below 2^-26 of their row maximum lose relative precision. An in-kernel
check for that case doubled backward time, so it is documented here instead.
`tests/distillation/test_head_logit_cache.py` checks eager loss adjoints and ignored
labels. The pointwise, HC, group-QSA, permutation, selector, dense-LoRA and
serving-rounding tests cover dtype support/rejection and mixed norm parameters;
FP16 output/gradient RMS and maximum errors use the same 1.5× eager error gate.

Hand-written Triton and cached-logit backwards support first derivatives only and
use `once_differentiable` to reject double backward. Pure PyTorch adjoints
(`NormTail`, `WeightedPermutation`, `scaled_add`)
retain differentiable backwards. `small_gemm` checkpointing rejects selection
reuse in the standalone helper and repeated backward on a retained graph; use
`full` for that pattern. Packed GDN callers should pass Python lengths alone;
the stack derives cumulative boundaries once and only accepts both representations
when they carry its unchanged provenance. Public MoE gather/combine derive their
inverse from a single permutation; the expert path shares that derived inverse.
