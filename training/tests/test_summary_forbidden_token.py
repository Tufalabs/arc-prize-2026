"""The token a summary must not write, ``<tool_call>``: one id wherever training
names it (the recipes, the ARC-3 trainer's check, the launcher's default), the
Duck's ``QWEN_TOOL_CALL_TOKEN_ID`` in both harnesses, and ``<tool_call>`` under
the recipe checkpoint's tokenizer, which the native server also checks at
startup (training/native_vllm_server.py, verify_summary_forbidden_tokens)."""

import ast
import json
from pathlib import Path
import re

import pytest

from flash_configs import RECIPE, ROOT
from training.config import ExperimentConfig

TOOL_CALL = "<tool_call>"
DUCK_PRESETS = (
    ROOT / "arc2-platypus/packages/duck/src/duck/presets.py",
    ROOT / "platypus/packages/duck/src/duck/presets.py",
)


def _recipe_ban():
    return tuple(json.loads(RECIPE.read_text())["inference"]["agent_summary_forbidden_token_ids"])


def _launcher_default():
    launcher = (ROOT / "scripts/launch_platypus_server.sh").read_text()
    (default,) = re.findall(r'"\$\{SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS-([0-9,]*)\}"', launcher)
    return tuple(int(token) for token in default.split(","))


def _arc3_trainer_ban():
    """The literal ARC-3 summaries are required to ban (harness_support/arc3/trainer.py)."""
    tree = ast.parse((ROOT / "harness_support/arc3/trainer.py").read_text())
    bans = [
        ast.literal_eval(node.comparators[0]) for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Tuple)
        and ast.unparse(node.left).endswith("summary_forbidden_token_ids")
    ]
    assert len(bans) == 1
    return bans[0]


def test_training_names_one_tool_call_id():
    assert _recipe_ban() == _arc3_trainer_ban() == _launcher_default() == (248058,)
    for config in sorted((ROOT / "configs").glob("*.json")):
        inference = json.loads(config.read_text()).get("inference", {})
        assert inference.get("agent_summary_forbidden_token_ids") in (None, [], [248058]), config.name


def test_the_id_is_the_ducks_tool_call_token():
    if not all(path.is_file() for path in DUCK_PRESETS):
        pytest.skip("the harness submodules are not checked out")
    for path in DUCK_PRESETS:
        constants = [
            ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            and [ast.unparse(target) for target in node.targets] == ["QWEN_TOOL_CALL_TOKEN_ID"]
        ]
        assert (*constants,) == _recipe_ban(), path


def test_the_id_decodes_to_tool_call_under_the_recipe_checkpoint():
    checkpoint = Path(ExperimentConfig.load(RECIPE).training.flash.checkpoint_root)
    if not (checkpoint / "tokenizer.json").is_file():
        pytest.skip(f"the recipe's checkpoint is not on this machine ({checkpoint})")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    (token,) = _recipe_ban()
    # As the native server checks it at startup, and as the gateway's bias names it.
    assert tokenizer.decode([token], skip_special_tokens=False) == TOOL_CALL
    assert tokenizer.convert_tokens_to_ids(TOOL_CALL) == token
