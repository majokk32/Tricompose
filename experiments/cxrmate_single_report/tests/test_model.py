from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tricompose_cxrmate_single_report.model import validate_model_snapshot


class CXRMateSingleModelTests(unittest.TestCase):
    def test_snapshot_validation_checks_auto_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["SingleCXREncoderDecoderModel"],
                        "auto_map": {
                            "AutoModel": "modelling_single.SingleCXREncoderDecoderModel"
                        },
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "generation_config.json",
                "modelling_single.py",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (root / name).write_text("x", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"abc")
            sizes = {
                path.name: path.stat().st_size
                for path in root.iterdir()
                if path.is_file()
            }
            with patch(
                "tricompose_cxrmate_single_report.model.EXPECTED_FILE_SIZES",
                sizes,
            ):
                audit = validate_model_snapshot(root)
            self.assertEqual(audit["weight_bytes"], 3)


if __name__ == "__main__":
    unittest.main()

