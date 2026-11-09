"""Exercise real tensor state rollback with an interrupted four-rank commit."""

import tempfile
import unittest
from pathlib import Path

import torch
from sft.distillation.cache_transaction import (
    atomic_torch_save,
    commit,
    committed,
    initialize,
)


class TensorResumeTests(unittest.TestCase):
    def test_restart_preserves_counts_without_duplicate_accumulation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            identity = {"world_size": 4, "rows": 7, "teacher": "frozen"}
            initialize(root, identity)
            initial = []
            for rank in range(4):
                state = {
                    "counts": torch.full((2, 3, 4), rank + 1, dtype=torch.int64),
                    "entries": {str(rank): {"hash": str(rank)}},
                }
                initial.append(state)
                atomic_torch_save(state, root / f"state-b1-rank{rank}.pt")
            commit(root, identity, 1)
            # Only two workers publish batch2 before interruption. No global commit.
            for rank in range(2):
                atomic_torch_save(
                    {"counts": torch.zeros(1)}, root / f"state-b2-rank{rank}.pt"
                )
            progress = committed(root, identity)
            self.assertEqual(progress["batches"], 1)
            for rank in range(4):
                state = torch.load(
                    root / progress["states"][rank]["name"], weights_only=True
                )
                self.assertTrue(torch.equal(state["counts"], initial[rank]["counts"]))
                if rank + 4 < 7:
                    state["counts"] += 10
                    state["entries"][str(rank + 4)] = {"hash": str(rank + 4)}
                atomic_torch_save(state, root / f"state-b2-rank{rank}.pt")
            commit(root, identity, 2)
            progress = committed(root, identity)
            keys = set()
            for rank, record in enumerate(progress["states"]):
                state = torch.load(root / record["name"], weights_only=True)
                self.assertTrue(
                    torch.equal(
                        state["counts"],
                        initial[rank]["counts"] + (10 if rank < 3 else 0),
                    )
                )
                keys.update(state["entries"])
            self.assertEqual(keys, set(map(str, range(7))))


if __name__ == "__main__":
    unittest.main()
