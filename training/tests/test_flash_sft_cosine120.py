"""PR #122 native schedule with PR #120 learning-rate replay, mixed-data coverage, and resumed-update checks."""
import math

import pytest

from training.flash_sft import lr_factor


def test_matches_recorded_pr120_schedule_then_continues_at_floor():
    """Check the exact reference update coordinates and the second-pass continuation."""
    for step in range(396):
        if step < 5:
            reference = (step + 1) / 5
        elif step <= 5:
            reference = 1.0
        else:
            reference = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step - 5) / 192)))
        assert lr_factor(step, 5, decay={'decay_from':5,'decay_until':197,'final_fraction':0.1}) == pytest.approx(reference)
    assert lr_factor(197, 5, decay={'decay_from':5,'decay_until':197,'final_fraction':0.1}) == pytest.approx(0.1)
    assert lr_factor(198, 5, decay={'decay_from':5,'decay_until':197,'final_fraction':0.1}) == pytest.approx(0.1)
    assert lr_factor(359, 5, decay={'decay_from':5,'decay_until':197,'final_fraction':0.1}) == pytest.approx(0.1)


def test_longer_corpus_decays_at_its_own_first_pass_boundary():
    """Additional data extends the first cosine pass rather than reaching the floor early."""
    assert lr_factor(197, 5, decay={'decay_from':5,'decay_until':499,'final_fraction':0.1}) > 0.1
    assert lr_factor(499, 5, decay={'decay_from':5,'decay_until':499,'final_fraction':0.1}) == pytest.approx(0.1)
    uninterrupted = [lr_factor(s, 5, decay={'decay_from':5,'decay_until':499,'final_fraction':0.1}) for s in range(1000)]
    resumed = [lr_factor(s, 5, decay={'decay_from':5,'decay_until':499,'final_fraction':0.1}) for s in range(360, 1000)]
    assert resumed == uninterrupted[360:]
