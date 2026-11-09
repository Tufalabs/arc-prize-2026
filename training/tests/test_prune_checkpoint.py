"""The training-owned checkpoint writer preserves expert/router alignment."""

import json
import os
from pathlib import Path
import runpy
import sys

import torch
from safetensors.torch import load_file, save_file


def test_physical_prune_preserves_drafter_and_slices_router(tmp_path, monkeypatch):
    """Retained experts keep their router rows; the MTP block stays untouched."""
    source = tmp_path / 'original'
    source.mkdir()
    prefix = 'model.language_model.layers.0.mlp.'
    def expert(i):
        """Name one original or renumbered routed expert tensor."""
        return f'{prefix}experts.{i}.gate_proj.weight'
    mtp = 'mtp.layers.0.mlp.experts.2.gate_proj.weight'
    weights = {expert(i): torch.full((2, 2), float(i)) for i in range(3)}
    weights[prefix + 'gate.weight'] = torch.arange(6).reshape(3, 2).float()
    weights[mtp] = torch.full((2, 2), 99.0)
    weights[prefix + 'shared_expert.weight'] = torch.ones(2, 2)
    save_file(weights, str(source / 'model.safetensors'))
    (source / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {key: 'model.safetensors' for key in weights},
    }))
    (source / 'config.json').write_text(json.dumps({'text_config': {'num_experts': 3}}))
    (source / 'tokenizer.json').write_text('{}')
    (source / 'hf_quant_config.json').write_text('{}')
    for name in ('conversion.done', 'metrics.json', 'report.txt', 'pruning.json'):
        (source / name).write_text('stale source metadata')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'num_experts': 3, 'kept': {'0': [0, 2]}}))
    output = tmp_path / 'pruned'
    script = Path(__file__).resolve().parents[1] / 'scripts/prune_flashnext_experts.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--src', str(source), '--plan', str(plan), '--out', str(output)])
    previous_umask = os.umask(0o002)
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        os.umask(previous_umask)
    assert next(output.glob('model-*.safetensors')).stat().st_mode & 0o777 == 0o664
    assert (output / 'tokenizer.json').read_text() == '{}'
    assert (output / 'hf_quant_config.json').read_text() == '{}'
    assert not (output / 'conversion.done').exists()
    assert not (output / 'metrics.json').exists()
    assert not (output / 'report.txt').exists()
    assert json.loads((output / 'pruning.json').read_text())['kept'] == {'0': [0, 2]}
    result = load_file(str(next(output.glob('model-*.safetensors'))))
    assert torch.equal(result[expert(1)], weights[expert(2)])
    assert expert(2) not in result
    assert torch.equal(result[prefix + 'gate.weight'], weights[prefix + 'gate.weight'][[0, 2]])
    assert torch.equal(result[mtp], weights[mtp])
    assert torch.equal(result[prefix + 'shared_expert.weight'], weights[prefix + 'shared_expert.weight'])
    config = json.loads((output / 'config.json').read_text())['text_config']
    assert config['num_experts'] == 2
    assert config['mtp_num_experts'] == 3
    assert json.loads((source / 'config.json').read_text())['text_config']['num_experts'] == 3
