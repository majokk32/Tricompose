"""Authored fixture metadata only; no actual model/image/data/network calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_text_conditioning_probe as probe


def fixture():
    generation, predictions = [], []
    for i, slot in enumerate(probe.slots()):
        row = {**slot, "status": "completed", "image_sha256": f"{i+1:064x}",
            "input_ids_sha256": "a"*64 if slot["arm"] == "present" else "b"*64}
        generation.append(row)
        predictions.append({"slot_id": slot["slot_id"], "image_sha256": row["image_sha256"],
            "scores": {"pneumonia": .7 if slot["arm"] == "present" else .2,
                "consolidation": .6 if slot["arm"] == "present" else .1}})
    return generation, predictions


class FakeModule:
    def __init__(self):
        self.training = True; self.parameter = SimpleNamespace(requires_grad=True)
    def eval(self): self.training = False; return self
    def requires_grad_(self, value): self.parameter.requires_grad = value; return self
    def parameters(self): return [self.parameter]


class TextConditioningProbeTests(unittest.TestCase):
    def test_exact_eight_slots_and_within_model_paired_seeds(self):
        slots = probe.slots()
        self.assertEqual(len(slots), 8)
        self.assertEqual(len({s["slot_id"] for s in slots}), 8)
        for model in probe.MODELS:
            for seed in probe.SEEDS:
                self.assertEqual({s["arm"] for s in slots if s["model"] == model and s["seed"] == seed}, set(probe.ARMS))

    def test_fixed_main_prompts_not_diffusion_negative_prompt(self):
        self.assertFalse(probe.CONFIG["prompt_changes_diffusion_negative_prompt"])
        self.assertNotEqual(probe.PROMPTS["present"], probe.PROMPTS["absent"])
        self.assertTrue(all(probe.PROMPTS[a].startswith("Chest radiograph.") for a in probe.ARMS))
        self.assertFalse(probe.CONFIG["report_generation"])

    def test_finite_four_pair_readout_without_clinical_claim(self):
        data = fixture(); original = deepcopy(data)
        paired = probe.paired_readout(*data)
        self.assertEqual(len(paired), 4)
        self.assertTrue(all(p["image_bytes_differ"] and p["tokenizer_ids_differ"] for p in paired))
        self.assertTrue(all(p["clinical_accuracy"] is None and p["clinical_acceptance"] is False for p in paired))
        self.assertAlmostEqual(paired[0]["pneumonia_score_delta_present_minus_absent"], .5)
        self.assertEqual(data, original)

    def test_identical_image_or_ids_retained_not_hidden(self):
        g, p = fixture()
        g[1]["image_sha256"] = g[0]["image_sha256"]
        g[1]["input_ids_sha256"] = g[0]["input_ids_sha256"]
        p[1]["image_sha256"] = g[1]["image_sha256"]
        row = probe.paired_readout(g, p)[0]
        self.assertFalse(row["image_bytes_differ"])
        self.assertFalse(row["tokenizer_ids_differ"])

    def test_failed_and_missing_slot_stays_in_pair_denominator(self):
        g, p = fixture()
        g[0] = {**probe.slots()[0], "status": "failed_charged"}
        result = probe.paired_readout(g, p[1:])
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[0]["tokenizer_ids_differ"])
        self.assertIsNone(result[0]["pneumonia_score_delta_present_minus_absent"])

    def test_no_classifier_results_retains_four_null_pairs(self):
        rows = probe.paired_readout(fixture()[0], [])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["pneumonia_score_delta_present_minus_absent"] is None for r in rows))

    def test_unavailable_finding_not_zero(self):
        g, p = fixture(); p[0]["scores"]["pneumonia"] = None
        self.assertIsNone(probe.paired_readout(g,p)[0]["present_pneumonia_score"])

    def test_dropped_or_duplicate_generation_rejected(self):
        g, p = fixture()
        for bad in (g[:-1], [g[0]]*8):
            with self.assertRaises(ValueError): probe.paired_readout(bad, p)

    def test_cross_image_classifier_hash_rejected(self):
        g, p = fixture(); p[0]["image_sha256"] = "c"*64
        with self.assertRaises(ValueError): probe.paired_readout(g,p)

    def test_wrong_seed_or_arm_metadata_rejected(self):
        for field, value in (("seed", 9), ("arm", "absent"), ("model", "other"), ("prompt_sha256", "c"*64)):
            g,p = fixture(); g[0][field] = value
            with self.assertRaises(ValueError): probe.paired_readout(g,p)

    def test_classifier_on_failed_image_rejected(self):
        g, p = fixture(); g[0]["status"] = "failed_charged"
        with self.assertRaises(ValueError): probe.paired_readout(g,p)

    def test_duplicate_prediction_rejected(self):
        g, p = fixture()
        with self.assertRaises(ValueError): probe.paired_readout(g,p+[p[0]])

    def test_boolean_nonfinite_or_out_of_range_score_rejected(self):
        for value in (True, float("nan"), float("inf"), -1, 2):
            g,p = fixture(); p[0]["scores"]["pneumonia"] = value
            with self.assertRaises(ValueError): probe.paired_readout(g,p)

    def test_parameter_and_eval_flags_explicitly_frozen(self):
        module = FakeModule()
        runtime = SimpleNamespace(_torch=SimpleNamespace(nn=SimpleNamespace(Module=FakeModule)),
            pipe=SimpleNamespace(components={"model": module, "tokenizer": object()}))
        self.assertEqual(probe.freeze_components(runtime), 1)
        self.assertFalse(module.training)
        self.assertFalse(module.parameter.requires_grad)

    def test_cpu_guard_before_reads(self):
        with patch.object(probe,"cpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(probe,"MetadataReader") as reader:
            with self.assertRaises(RuntimeError): probe.prepare(SimpleNamespace())
            reader.assert_not_called()

    def test_gpu_guard_before_plan_or_model_loading(self):
        with patch.object(probe,"gpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(probe,"load_plan") as load:
            with self.assertRaises(RuntimeError): probe.run(SimpleNamespace())
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
