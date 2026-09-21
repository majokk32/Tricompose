from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tricompose_maira2_report.model import validate_model_snapshot


class Maira2ModelTests(unittest.TestCase):
    def test_snapshot_validation_checks_architecture_and_weight_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"architectures": ["Maira2ForConditionalGeneration"]}),
                encoding="utf-8",
            )
            (root / "configuration_maira2.py").write_text("x", encoding="utf-8")
            (root / "modeling_maira2.py").write_text("x", encoding="utf-8")
            (root / "processing_maira2.py").write_text("x", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model.safetensors"}}),
                encoding="utf-8",
            )
            (root / "model.safetensors").write_bytes(b"abc")
            sizes = {
                path.name: path.stat().st_size
                for path in root.iterdir()
                if path.is_file()
            }
            with patch(
                "tricompose_maira2_report.model.EXPECTED_FILE_SIZES", sizes
            ):
                audit = validate_model_snapshot(root)
            self.assertEqual(audit["weight_bytes"], 3)


if __name__ == "__main__":
    unittest.main()

