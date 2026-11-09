"""Three windowed completion-routing summaries; candidate stability uses the probe."""

import math

from sft.distillation.pruning_metrics import bottom


class ExpertUsageWindow:
    def __init__(self, probe_config):
        self.probe_config = probe_config
        self.counts = self.masses = None
        self.steps = 0

    def add(self, usage):
        scope = usage["completion_predictions"]
        counts, masses = scope["selection_counts"], scope["routing_mass"]
        if not counts or len(counts) != len(masses):
            raise ValueError("invalid expert layer census")
        for c, m in zip(counts, masses, strict=True):
            if (
                not c
                or len(c) != len(m)
                or any(not math.isfinite(v) or v < 0 for v in [*c, *m])
            ):
                raise ValueError("invalid expert usage")
        if self.counts is None:
            self.counts = [[0] * len(c) for c in counts]
            self.masses = [[0.0] * len(m) for m in masses]
        if [len(c) for c in counts] != [len(c) for c in self.counts]:
            raise ValueError("expert census changed within logging window")
        for i, (c, m) in enumerate(zip(counts, masses, strict=True)):
            self.counts[i] = [a + b for a, b in zip(self.counts[i], c, strict=True)]
            self.masses[i] = [a + b for a, b in zip(self.masses[i], m, strict=True)]
        self.steps += 1

    def flush(self):
        if not self.steps:
            return {}
        effective, candidate_mass = [], []
        for counts, masses in zip(self.counts, self.masses, strict=True):
            total, total_mass = sum(counts), sum(masses)
            if total <= 0 or total_mass <= 0:
                # No observations are not evidence of routing collapse.
                self.counts = self.masses = None
                self.steps = 0
                return {}
            k = self.probe_config.get("bottom_k") or max(
                1,
                math.ceil(self.probe_config.get("bottom_fraction", 0.1) * len(counts)),
            )
            candidates = bottom(counts, k)
            entropy = -sum((c / total) * math.log(c / total) for c in counts if c)
            effective.append(min(1.0, math.exp(entropy) / len(counts)))
            candidate_mass.append(sum(masses[i] for i in candidates) / total_mass)
        result = {
            "experts/effective_expert_fraction": sum(effective) / len(effective),
            "experts/bottom_candidate_routing_mass": sum(candidate_mass)
            / len(candidate_mass),
            "experts/worst_layer_effective_expert_fraction": min(effective),
        }
        self.counts = self.masses = None
        self.steps = 0
        return result
