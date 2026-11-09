import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.distillation.common import teacher_identity, validate_roles, validate_student


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.teacher = Path(self.tmp.name) / "teacher"
        self.student = Path(self.tmp.name) / "student"
        for root, experts in [(self.teacher, 307), (self.student, 256)]:
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "num_experts": experts,
                            "num_experts_per_tok": 10,
                            "num_hidden_layers": 2,
                            "hidden_size": 8,
                        }
                    }
                )
            )
            (root / "tokenizer.json").write_text("same tokenizer")
        (self.teacher / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        (self.teacher / "pruning.json").write_text('{"source": "original ancestor"}')
        self.provenance = {
            "teacher_identity": teacher_identity(self.teacher),
            "kept_teacher_expert_ids": {str(i): list(range(256)) for i in range(2)},
        }
        self.write_provenance()

    def write_provenance(self):
        (self.student / "distillation-parent.json").write_text(
            json.dumps(self.provenance)
        )

    def test_accept_future_256_derived_from_307(self):
        result = validate_student(self.teacher, self.student)
        self.assertEqual(result["experts"], 256)

    def test_unbound_student_stops_before_scheduler(self):
        with (
            patch("sft.distillation.common.TEACHER", self.teacher),
            self.assertRaisesRegex(ValueError, "not configured"),
        ):
            validate_roles({"teacher": str(self.teacher), "student": None})

    def test_old_256_sibling_without_parent_contract_rejected(self):
        (self.student / "distillation-parent.json").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_student(self.teacher, self.student)

    def test_wrong_parent_and_bad_expert_ids_rejected(self):
        self.provenance["teacher_identity"]["experts"] = 512
        self.write_provenance()
        with self.assertRaisesRegex(ValueError, "not derived"):
            validate_student(self.teacher, self.student)
        self.provenance["teacher_identity"] = teacher_identity(self.teacher)
        self.provenance["kept_teacher_expert_ids"]["0"][0] = 400
        self.write_provenance()
        with self.assertRaisesRegex(ValueError, "retained-expert"):
            validate_student(self.teacher, self.student)

    def test_cannot_train_the_teacher_as_student(self):
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            validate_student(self.teacher, self.teacher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
