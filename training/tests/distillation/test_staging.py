import json
import tempfile
import unittest
from pathlib import Path

from sft.distillation.staging import stage, validate_stage


class StagingTests(unittest.TestCase):
    def test_payload_and_provenance_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            dest = root / "local"
            (source / "weights.safetensors").write_bytes(b"fixture-payload")
            index = source / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
            stage(source, dest, workers=2)
            validate_stage(source, dest, full=True)
            (dest / "weights.safetensors").write_bytes(b"Fixture-payload")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                validate_stage(source, dest, full=True)
            index.write_text(
                json.dumps({"weight_map": {"different": "weights.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_stage(source, dest)
            with self.assertRaises(FileExistsError):
                stage(source, dest)


if __name__ == "__main__":
    unittest.main()
