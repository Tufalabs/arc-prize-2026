"""Token-native context is a declared serving mode: every vLLM preset turns
it on, the class default leaves it off (U1).

The vLLM topologies answer ``token_count`` exactly — the id-ledger length on
continued calls — so the presets that build them declare the mode; a cloud
or stub backend must never be asked to count (its recorded streams are
pinned), which is exactly what the off default guarantees. The measurement
behavior itself is unit-tested in ``test_context_accounting.py``; what this
file pins is the DECLARATION surface: where each deployment column stands.
"""

from __future__ import annotations

import pytest
import ullm.backends.deployment as deployment
import ullm.backends.oai
import ullm.backends.vllmx
import ullm.factory

from seal.agent import SealAgent

presets = pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")


def test_the_mode_is_off_unless_a_preset_declares_it() -> None:
    assert SealAgent(reply_cap_tokens=None, compaction_strategy="prune").token_native_context is False


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_every_vllm_preset_declares_token_native_context() -> None:
    """Both qwen_vllm topologies, the training server, and Kaggle serve
    vLLM; each declares the mode so the context decision runs on exact
    weights rather than the bill+bound arithmetic."""
    assert SealAgent.build().token_native_context is True
    assert SealAgent.build(engine="qwen3.8-27b", serving="server").token_native_context is True
    assert (
        SealAgent.build(
            engine="qwen3.8-27b",
            environment="slurm-b200",
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.oai.OpenAIServer(
                    base_url="http://x/v1", model="m", **deployment.client_kwargs("qwen3.8-27b")
                )
            ),
        ).token_native_context
        is True
    )
    assert SealAgent.build(environment="kaggle-rtx").token_native_context is True


if __name__ == "__main__":
    pytest.main([__file__])
