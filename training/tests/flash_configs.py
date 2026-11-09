"""Flash configs for tests: the shipped recipe and an ARC-3 variant of it.

The repository ships one Flash recipe, an ARC-2 run. The launchers still
support ARC-3, so the ARC-3 tests derive a config from that recipe with the
shipped ARC-3 multimodal smoke's rollout settings.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/flash_8gpu_arc2_grpo_tokmean.json"
ARC3_ROLLOUTS = ROOT / "configs/dppo_2gpu_arc3_multimodal_smoke.json"


def write_arc3_flash_config(directory: Path, **flash) -> Path:
    """The Flash recipe with ARC-3 rollouts (32768-token window, current_grid
    images) served by the recipe's model on its four server GPUs; ``flash``
    overrides fields of its training.flash section (e.g. kernel_mode)."""
    payload = json.loads(RECIPE.read_text())
    payload["training"]["flash"].update(flash)
    # ARC-2's penalty cap: ARC-3 charges no reply penalties and refuses it.
    payload["training"].pop("max_episode_penalty", None)
    inference = json.loads(ARC3_ROLLOUTS.read_text())["inference"]
    inference.update(model_id=payload["inference"]["model_id"],
                     server_gpus=payload["inference"]["server_gpus"])
    payload["inference"] = inference
    path = Path(directory) / "flash_arc3.json"
    path.write_text(json.dumps(payload))
    return path
