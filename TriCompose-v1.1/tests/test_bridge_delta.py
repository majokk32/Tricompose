import copy
import unittest

from tricompose_v11.bridge_delta import compare_inventories
from tricompose_v11.prompts import ACTIVE_PROMPT_MODELS_V11


class DeltaTests(unittest.TestCase):
    def fixture(self):
        return {"case_000": {"ehr": "a", "facts": "b", "intent": "c",
                             "prompts": dict.fromkeys(ACTIVE_PROMPT_MODELS_V11, "d")}}

    def test_metadata_change_does_not_trigger_gpu_regeneration(self):
        old = self.fixture()
        new = copy.deepcopy(old)
        new["case_000"]["facts"] = "new-facts"
        result = compare_inventories(old, new)
        self.assertEqual(result["counts"]["changed_model_prompts"], 0)
        self.assertFalse(result["old_candidates_retagged"])
        self.assertEqual(result["records"][0]["action"], "reuse_requires_explicit_lineage_certificate")

    def test_only_changed_model_is_scheduled(self):
        old = self.fixture()
        new = copy.deepcopy(old)
        new["case_000"]["prompts"]["roentgen_v2"] = "changed"
        result = compare_inventories(old, new)
        self.assertEqual(result["counts"]["changed_model_prompts"], 1)
        self.assertEqual(result["changed_cases"], ["case_000"])
        self.assertEqual(result, compare_inventories(old, new))

    def test_changed_ehr_or_case_set_fails_closed(self):
        old = self.fixture()
        new = copy.deepcopy(old)
        new["case_000"]["ehr"] = "changed"
        with self.assertRaises(ValueError):
            compare_inventories(old, new)
        with self.assertRaises(ValueError):
            compare_inventories(old, {})
