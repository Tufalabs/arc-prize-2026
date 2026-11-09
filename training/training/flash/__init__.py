"""Trainer-independent Qwen3.8-Flash model construction and kernels.

The distillation and RL trainers own objectives, launch policy, and recipe
validation.  This package owns only model loading, execution backends,
trainability seams, and distribution choices.

Import from submodules (``training.flash.model``, ``.policies`` ...). The root
re-exports nothing, so configuration helpers (``policies``, ``options``,
``qsa_policy``, ``offload``, ``head_loss``) import without Qwen4-Exp, Triton or
FLA, e.g. in a launcher environment pinned to Transformers 5.14.1.
"""
