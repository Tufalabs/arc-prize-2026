"""An evaluation must serve the target the drafter was trained against."""

from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from drafter import head
from drafter.extract import build_factory as extraction_factory
from drafter.online import build_factory as online_factory


@pytest.mark.parametrize('quantization,backend', [
    (None, 'triton'),
    ({'quant_algo': 'NVFP4', 'ignore': ['mtp.*', 'model.mtp.*']}, 'triton'),
    ({'quant_algo': 'NVFP4', 'ignore': ['mtp.fc_embedding', 'mtp.fc_hidden']}, 'marlin'),
])
def test_extraction_and_online_use_selected_target(tmp_path, monkeypatch, quantization, backend):
    """A pruned target cannot silently fall back to the harness default model."""
    agents = []

    def build(**kwargs):
        """Stand in for either harness's DuckAgent factory."""
        factory = SimpleNamespace(engine_args=SimpleNamespace(
            model='harness-default', quantization='modelopt',
            speculative_config={'disable_eagle_block_drop': True, 'gdn_recoverssm': True, 'moe_backend': 'harness-default'},
        ))
        agent = SimpleNamespace(
            run_config=SimpleNamespace(max_concurrent=100),
            llm_config=SimpleNamespace(backend_factory=factory),
            prepare_for_backend=lambda: None,
        )
        agents.append(agent)
        return agent

    module = ModuleType('duck.agent')
    module.DuckAgent = SimpleNamespace(build=build)
    monkeypatch.setitem(sys.modules, 'duck.agent', module)
    monkeypatch.setattr(head, 'TARGET', Path('/checkpoints/pruned-for-arc2'))
    drafter = tmp_path / 'trained-drafter'
    drafter.mkdir()
    (drafter / 'config.json').write_text(json.dumps({'quantization_config': quantization}))
    _, extract = extraction_factory()
    online = online_factory(drafter, 4, 2)
    assert extract.engine_args.model == online.engine_args.model == '/checkpoints/pruned-for-arc2'
    assert extract.engine_args.speculative_config is None
    assert online.engine_args.speculative_config['model'] == str(drafter)
    assert online.engine_args.speculative_config['num_speculative_tokens'] == 4
    assert online.engine_args.speculative_config['disable_eagle_block_drop']
    assert online.engine_args.speculative_config['gdn_recoverssm']
    assert online.engine_args.speculative_config['moe_backend'] == backend
    _, parity = extraction_factory(drafter)
    assert parity.engine_args.speculative_config['moe_backend'] == backend
    assert parity.engine_args.speculative_config['disable_eagle_block_drop']
    assert parity.engine_args.speculative_config['num_speculative_tokens'] == 1
    assert agents[0].run_config.max_concurrent == 1
    assert agents[1].run_config.max_concurrent == 2
    assert online_factory(drafter, 4, 2, drafter_moe_backend='triton').engine_args.speculative_config['moe_backend'] == 'triton'
    _, parity = extraction_factory(drafter, drafter_moe_backend='marlin')
    assert parity.engine_args.speculative_config['moe_backend'] == 'marlin'


def test_unknown_quantization_requires_backend(tmp_path):
    """Unknown quantization must not inherit an incompatible target kernel."""
    from drafter.serving import moe_backend

    (tmp_path / 'config.json').write_text(json.dumps({'quantization_config': {'quant_algo': 'unknown'}}))
    with pytest.raises(ValueError, match='--moe-backend'):
        moe_backend(tmp_path)
