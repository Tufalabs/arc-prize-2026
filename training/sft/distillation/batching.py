"""Deterministic, resumable whole-sequence batching; no dropped rows; optional zero-weight dummy slots."""

import torch


def normalize_batching(config=None, world=4):
    if config is not None and not isinstance(config, dict):
        raise ValueError("Batching configuration must be an object")
    config = {} if config is None else dict(config)
    policy = config.get("policy", "length_bucket")
    if policy not in ("random", "length_bucket", "length_within_update"):
        raise ValueError("Unknown batching policy")
    allowed = {"policy", "version", "tail_policy"} | (
        {"pool_size"} if policy == "length_bucket" else set()
    )
    if (
        set(config) - allowed
        or type(config.get("version", 1)) is not int
        or config.get("version", 1) != 1
    ):
        raise ValueError("Invalid batching configuration/version")
    tail = config.get("tail_policy", "error")
    if tail not in ("error", "masked"):
        raise ValueError("Unknown tail policy")
    if tail == "masked" and policy == "length_within_update":
        raise ValueError("masked tails require random or length_bucket batching")
    result = {"policy": policy, "version": 1}
    if tail == "masked":
        result["tail_policy"] = tail
    if policy == "length_bucket":
        pool = config.get("pool_size", 128)
        if type(pool) is not int or pool < world or pool % world:
            raise ValueError("pool_size must be a positive multiple of world size")
        result["pool_size"] = pool
    return result


class BatchPlanner:
    def __init__(self, lengths, world, seed, config=None):
        if type(world) is not int or world < 1 or type(seed) is not int:
            raise ValueError("Invalid world size/seed")
        self.lengths = tuple(lengths)
        self.config = normalize_batching(config, world)
        if len(lengths) < world or (
            len(lengths) % world and self.config.get("tail_policy") != "masked"
        ):
            raise ValueError("Dataset size must be a nonzero multiple of world size")
        if any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError("Sequence lengths must be positive integers")
        self.world, self.seed = world, seed
        self.batches = (len(lengths) + world - 1) // world
        self.cached_epoch = None
        self.cached_groups = None

    def _pad(self, group):
        return group + [-1] * (self.world - len(group))

    def _regroup(self, indices, generator):
        ordered = sorted(indices, key=self.lengths.__getitem__)
        groups = [
            self._pad(ordered[i : i + self.world])
            for i in range(0, len(ordered), self.world)
        ]
        # Shuffle cohort order, then randomly assign examples within each cohort.
        groups = [
            groups[i] for i in torch.randperm(len(groups), generator=generator).tolist()
        ]
        return [
            [group[i] for i in torch.randperm(self.world, generator=generator).tolist()]
            for group in groups
        ]

    def _epoch(self, epoch):
        if epoch == self.cached_epoch:
            return self.cached_groups
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        if self.config["policy"] == "length_bucket":
            pool = self.config["pool_size"]
            groups = []
            for start in range(0, len(order), pool):
                groups.extend(self._regroup(order[start : start + pool], generator))
        else:
            groups = [
                self._pad(order[i : i + self.world])
                for i in range(0, len(order), self.world)
            ]
        self.cached_epoch, self.cached_groups = epoch, groups
        return groups

    def update(self, cursor, microbatches=1):
        if (
            type(cursor) is not int
            or cursor < 0
            or type(microbatches) is not int
            or microbatches < 1
        ):
            raise ValueError("Invalid data cursor/microbatch count")
        groups = []
        for offset in range(microbatches):
            epoch, batch = divmod(cursor + offset, self.batches)
            groups.append(list(self._epoch(epoch)[batch]))
        if self.config["policy"] == "length_within_update" and microbatches > 1:
            generator = torch.Generator().manual_seed(
                (self.seed + 104729 * (cursor + 1)) % (2**63 - 1)
            )
            groups = self._regroup([i for group in groups for i in group], generator)
        return groups
