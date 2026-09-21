"""Synthetic-only Liquid deployment-layout test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tricompose_liquid.runtime import (
    EXPECTED_VQ_CHECKPOINT_SIZE,
    EXPECTED_VQ_CONFIG_SIZE,
    EXPECTED_WEIGHT_SIZES,
    REQUIRED_MODEL_FILES,
    validate_liquid_deployment,
)


class DeploymentValidationTest(unittest.TestCase):
    def test_sparse_layout_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "model"
            root.mkdir()
            for name in REQUIRED_MODEL_FILES:
                (root / name).write_text("{}\n", encoding="utf-8")
            for name, size in EXPECTED_WEIGHT_SIZES.items():
                with (root / name).open("wb") as handle:
                    handle.truncate(size)
            vq_config = Path(temp_dir) / "vqgan.yaml"
            vq_checkpoint = Path(temp_dir) / "vqgan.ckpt"
            with vq_config.open("wb") as handle:
                handle.truncate(EXPECTED_VQ_CONFIG_SIZE)
            with vq_checkpoint.open("wb") as handle:
                handle.truncate(EXPECTED_VQ_CHECKPOINT_SIZE)
            audit = validate_liquid_deployment(root, vq_config, vq_checkpoint)
            self.assertEqual(audit["weight_shards"], 4)


if __name__ == "__main__":
    unittest.main()
