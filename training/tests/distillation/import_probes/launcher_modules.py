"""Login-node processes: the distillation launcher, the exporter and the RL
config/frozen launcher may run on Transformers 5.14.1 without Qwen4-Exp,
Triton or FLA."""

import json
import tempfile
from pathlib import Path

from tests.distillation.import_probes import assert_never_imported, block_modules

RL_CONFIG = {
    "parallel_rollouts": 2,
    "inference": {"base_url": "http://localhost:1234", "model_id": "Qwen/Qwen3.8-Flash-Next",
                  "games": ["pathfinder-0001"], "platypus_venv": "/tmp/platypus",
                  "agent_max_completion_tokens": 128},
    "training": {"model_name": "Qwen/Qwen3.8-Flash-Next", "weight_mode": "flash",
                 "output_dir": "runs", "device": "cuda",
                 "flash": {"checkpoint_root": "/checkpoint"},
                 "lora": {"dropout": 0, "target_modules": ["q_proj"]}},
}


def main():
    block_modules("fla", "triton", prefixes=("transformers.models.qwen4_exp",))
    import sys

    import torch

    import sft.distillation.export  # noqa: F401
    import sft.distillation.launch  # noqa: F401
    import sft.distillation.train  # noqa: F401
    import training.launch_frozen  # noqa: F401
    from training.config import ExperimentConfig
    from training.flash.expert_lora import normalize_lora  # noqa: F401
    from training.flash.head_loss import CachedHeadCE, ChunkedCE, completion_loss  # noqa: F401
    from training.flash.offload import resolve_qsa_retention  # noqa: F401
    from training.flash.options import FlashOptions
    from training.flash.policies import DistributionPolicy, TrainabilityPolicy  # noqa: F401
    from training.flash.qsa_policy import validate_qsa_kernel
    from training.flash.reader import Reader, dequant_nvfp4  # noqa: F401

    # The complete ExperimentConfig validation path runs before an interpreter
    # is chosen, so it must not pull the kernel modules in.
    payload = json.loads(json.dumps(RL_CONFIG))
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "config.json"
        path.write_text(json.dumps(payload))
        assert ExperimentConfig.load(path).training.weight_mode == "flash"
        payload["training"]["lora"]["target_modules"] = ["experts.gate_proj"]
        path.write_text(json.dumps(payload))
        try:
            ExperimentConfig.load(path)
        except ValueError as error:
            assert "target_modules" in str(error)
        else:
            raise AssertionError("Invalid Flash targets were accepted")
    assert "training.flash.dense_lora" not in sys.modules

    FlashOptions(selector_policy="compiled_fp32_top513_v1")
    validate_qsa_kernel("group_sparse", "flex_qsa")
    expanded = dequant_nvfp4(torch.full((1, 8), 0x21, dtype=torch.uint8), torch.ones(1, 1), torch.tensor(2.0))
    assert expanded.tolist() == [[1.0, 2.0] * 8]
    hidden = torch.randn(4, 4, requires_grad=True)
    for head in (ChunkedCE, CachedHeadCE):
        head.apply(hidden, torch.randn(7, 4), torch.tensor([1, 2, -100, 4]), 3).backward()
    assert_never_imported("fla", "triton")


if __name__ == "__main__":
    main()
