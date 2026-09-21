"""Synthetic-only UniDisc deployment-layout test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tricompose_unidisc.runtime import (
    CHECKPOINT_RELATIVE,
    EXPECTED_SIZES,
    OFFLINE_EVAL_TOKENIZER,
    REQUIRED_SOURCE_FILES,
    build_unidisc_overrides,
    validate_unidisc_deployment,
)


class DeploymentValidationTest(unittest.TestCase):
    def test_sparse_layout_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in REQUIRED_SOURCE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic\n", encoding="utf-8")
            for relative, size in EXPECTED_SIZES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    handle.truncate(size)
            audit = validate_unidisc_deployment(root)
            self.assertTrue(audit["checkpoint_layout_verified"])

    def test_offline_overrides_replace_unused_gpt2_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            overrides = build_unidisc_overrides(root)
            self.assertIn(
                f"trainer.load_from_state_dict={root / CHECKPOINT_RELATIVE}",
                overrides,
            )
            self.assertIn(
                f"eval.gen_ppl_eval_model_name_or_path={OFFLINE_EVAL_TOKENIZER}",
                overrides,
            )
            self.assertIn("eval.compute_generative_perplexity=false", overrides)


if __name__ == "__main__":
    unittest.main()
