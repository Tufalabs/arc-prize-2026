"""Derive the three experiment recipes from PR 121's frozen mixed recipe and the targets job's DATA_READY.json.

Each experiment keeps every baseline setting; only the data files (PR 121's rows with this
branch's targets), the W&B identity and the recorded-logprob loss differ:

  reap307-kimi-mixed-toolcalls        CE on every token + 0.3 KL (the baseline objective, wider KL scope)
  reap307-kimi-mixed-toolcalls-xcl03  CE only where no recorded target exists, 0.3 KL elsewhere
  reap307-kimi-mixed-toolcalls-xcl10  CE only where no recorded target exists, 1.0 KL elsewhere
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs/flash_sft/reap307-kimi-mixed-cosine.json"
EXPERIMENTS = {
    "reap307-kimi-mixed-toolcalls": {"weight": 0.3, "ce_positions": "all"},
    "reap307-kimi-mixed-toolcalls-xcl03": {"weight": 0.3, "ce_positions": "unaligned"},
    "reap307-kimi-mixed-toolcalls-xcl10": {"weight": 1.0, "ce_positions": "unaligned"},
}


def recipes(ready):
    baseline = json.loads(BASELINE.read_text())
    for name, loss in EXPERIMENTS.items():
        config = json.loads(json.dumps(baseline))
        config["rows"] = {"data": ready["rows"]["data"], "data_sha256": ready["rows"]["data_sha256"]}
        config["validation"].update(data=ready["validation-probe"]["data"],
                                    data_sha256=ready["validation-probe"]["data_sha256"])
        config["wandb"].update(name=name, id=name.replace("-", "") + "20261030")
        config["recorded_logprobs"].update(loss)
        yield name, config


def main(ready_path):
    ready = json.loads(Path(ready_path).read_text())
    for name, config in recipes(ready):
        path = ROOT / "configs/flash_sft" / (name + ".json")
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main(sys.argv[1])
