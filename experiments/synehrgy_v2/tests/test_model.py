from __future__ import annotations

import unittest

from tricompose_synehrgy_v2.model import (
    MODEL_SPECS,
    _model_context_capacity,
    get_model_spec,
)


class ModelSpecTests(unittest.TestCase):
    def test_qwen2_variant_is_pinned_and_complete(self) -> None:
        spec = get_model_spec("qwen2-40bins")
        self.assertEqual(spec.architecture, "Qwen2ForCausalLM")
        self.assertEqual(spec.model_type, "qwen2")
        self.assertEqual(spec.context_length, 2048)
        self.assertEqual(len(spec.revision), 40)
        self.assertTrue(
            set(spec.expected_sha256).issubset(spec.expected_file_sizes)
        )
        self.assertIn(
            f"{spec.checkpoint}/model.safetensors",
            spec.expected_sha256,
        )

    def test_legacy_and_qwen2_variants_are_both_available(self) -> None:
        self.assertEqual(set(MODEL_SPECS), {"gpt2-10bins", "qwen2-40bins"})

    def test_context_capacity_supports_both_architectures(self) -> None:
        self.assertEqual(_model_context_capacity({"n_positions": 2048}), 2048)
        self.assertEqual(
            _model_context_capacity({"max_position_embeddings": 4096}),
            4096,
        )

    def test_unknown_variant_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            get_model_spec("unknown")


if __name__ == "__main__":
    unittest.main()
