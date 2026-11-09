"""Flash serving's MTP drafting: ullm's Flash deployment's settings, copied.

Shared by the experiment config (inference.speculative_decoding), the job
shape that hands the servers their ``--speculative-config`` and the runtime
attestation (training/vllm_runtime.py), without importing either.
"""

# ullm's Flash-Next drafting, copied (platypus submodule,
# ``backends/deployment.py``: ``FLASHNEXT_E307_DRAFTER`` and the engine's
# ``speculative_config``). The ARC-2 Duck's Flash column keeps these settings
# but serves REAP-307 with its own drafter; RL trains e307 and drafts as ullm
# does. The drafter is e307's MTP head fine-tuned on ARC play, its experts
# NVFP4, its LM head cut to 16,384 draft ids; it is valid for the e307 target
# and is never trained with the policy. Drafts are sampled and verified
# against their probabilities; GDN RecoverSSM verifies from the step's
# starting state.
FLASH_E307_DRAFTER = (
    "/data/shared/research/checkpoints/"
    "Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform-drafter-v1-nvfp4-dv16k"
)
FLASH_DRAFT_TOKENS = 4
HARNESS_SPECULATIVE = {
    "method": "mtp",
    "quantization": "modelopt_fp4",
    "draft_sample_method": "probabilistic",
    "gdn_recoverssm": True,
    "disable_eagle_block_drop": True,
}
# The fork's exact-verification defaults, which the harness leaves in place:
# standard rejection sampling with no relaxed thresholds, so a served token is
# a draw from the target's own distribution. Synthetic or relaxed acceptance
# is lossy at temperature 1 and is refused.
EXACT_VERIFICATION = {
    "rejection_sample_method": "standard",
    "accept_threshold_single": 1.0,
    "accept_threshold_acc": 1.0,
    "draft_top_k_top_p": False,
}
SPECULATIVE_FIELDS = (
    "model", "num_speculative_tokens", *HARNESS_SPECULATIVE, *EXACT_VERIFICATION
)


def speculative_engine_config(drafter: str, num_speculative_tokens: int) -> dict:
    """The fork's ``--speculative-config``: ullm's settings, this drafter."""
    return {
        **HARNESS_SPECULATIVE,
        "num_speculative_tokens": num_speculative_tokens,
        "model": drafter,
    }


def speculative_identity(drafter: str, num_speculative_tokens: int) -> dict:
    """What ``/runtime`` attests for an engine started with that config."""
    return {
        **speculative_engine_config(drafter, num_speculative_tokens),
        **EXACT_VERIFICATION,
    }
