import tempfile
import unittest
from pathlib import Path

from sft.distillation.cache_transaction import commit, committed, initialize


class CacheTransactionTests(unittest.TestCase):
    def test_interrupted_batch_keeps_prior_commit(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            identity = {"world_size": 4, "data": "fixture"}
            initialize(r, identity)
            self.assertIsNone(committed(r, identity))
            for rank in range(4):
                (r / f"state-b1-rank{rank}.pt").write_bytes(bytes([rank]))
            commit(r, identity, 1)
            (r / "state-b2-rank0.pt").write_bytes(b"new")
            with self.assertRaises(FileNotFoundError):
                commit(r, identity, 2)
            self.assertEqual(committed(r, identity)["batches"], 1)
            for rank in range(1, 4):
                (r / f"state-b2-rank{rank}.pt").write_bytes(bytes([rank + 10]))
            commit(r, identity, 2)
            self.assertEqual(committed(r, identity)["batches"], 2)
            with self.assertRaises(ValueError):
                commit(r, identity, 2)
            (r / "state-b2-rank3.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                committed(r, identity)

    def test_changed_identity_and_unknown_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            initialize(r, {"world_size": 4, "data": "a"})
            with self.assertRaisesRegex(ValueError, "identity"):
                initialize(r, {"world_size": 4, "data": "b"})
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "head.safetensors").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Unidentified"):
                initialize(r, {"world_size": 4})


if __name__ == "__main__":
    unittest.main()
