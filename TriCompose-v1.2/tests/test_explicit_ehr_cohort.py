from __future__ import annotations

import importlib.util
import json
import stat
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "select_explicit_ehr_cohort.py"
SPEC = importlib.util.spec_from_file_location("select_explicit_ehr_cohort", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

GENERATOR_PATH = MODULE_PATH.with_name("generate_unconditional_ehr_pool.py")
GENERATOR_SPEC = importlib.util.spec_from_file_location("generate_unconditional_ehr_pool", GENERATOR_PATH)
assert GENERATOR_SPEC and GENERATOR_SPEC.loader
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(generator)


def _case(case_id: str, diagnosis: str, *, valid: bool = True) -> dict:
    return {
        "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
        "case_id": case_id,
        "structure": {
            "top_level_tokens": [],
            "visits": [{
                "covariates": [], "problems": [diagnosis], "labs": [],
                "charts": [], "other_tokens": [],
            }],
        },
        "validation": {"strict_valid": valid},
    }


class ExplicitEHRSelectionTests(unittest.TestCase):
    def test_new_pool_writer_uses_group_protected_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cases"
            generator._mkdir(directory)
            record = directory / "case_000.json"
            generator._write_json(record, {"synthetic": True})
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o2770)
            self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o660)
            with self.assertRaises(FileExistsError):
                generator._write_json(record, {"synthetic": False})

    def test_only_latest_diagnosis_explicit_positive_is_eligible(self) -> None:
        def facts(problems: list[list[str]]) -> dict:
            case = _case("case_000", "I10_Hypertension")
            case["structure"]["visits"] = [
                {"covariates": [], "problems": visit, "labs": [], "charts": [], "other_tokens": []}
                for visit in problems
            ]
            from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
            from tricompose_v11.facts import extract_v11_facts
            return extract_v11_facts(canonicalize_synehrgy_case(case, source_model_id="synehrgy_qwen2_40bins"))

        self.assertEqual(module.eligible_positive_facts(facts([["J90_Pleural effusion"]])), ("pleural_effusion",))
        self.assertEqual(module.eligible_positive_facts(facts([["I50_Heart failure"]])), ())
        self.assertEqual(module.eligible_positive_facts(facts([["J90_Pleural effusion"], ["I10_Hypertension"]])), ())
        self.assertEqual(module.eligible_positive_facts(facts([["Z_No pleural effusion"]])), ())
        self.assertEqual(module.eligible_positive_facts(facts([["Z_Possible pleural effusion"]])), ())

    def test_fixed_order_no_quality_ranking(self) -> None:
        rows = [
            {"case_id": "case_000", "eligible": False},
            {"case_id": "case_001", "eligible": True},
            {"case_id": "case_002", "eligible": True},
            {"case_id": "case_003", "eligible": True},
        ]
        self.assertEqual([row["case_id"] for row in module.first_eligible(rows, 2)], ["case_001", "case_002"])

    def test_run_is_immutable_and_keeps_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protected = Path(temporary) / "protected"
            source = protected / "synehrgy_v2" / "runs" / "new_pool"
            cases = source / "cases"
            cases.mkdir(parents=True)
            entries = []
            diagnoses = ["I50_Heart failure", "J90_Pleural effusion", "J81_Pulmonary edema", "I10_Hypertension"]
            for index, diagnosis in enumerate(diagnoses):
                case_id = f"case_{index:03d}"
                path = cases / f"{case_id}.json"
                path.write_text(json.dumps(_case(case_id, diagnosis)), encoding="utf-8")
                entries.append({
                    "case_id": case_id,
                    "relative_path": f"cases/{case_id}.json",
                    "sha256": module._hash_file(path),
                })
            (source / "run.json").write_text(json.dumps({
                "schema": module.SOURCE_SCHEMA,
                "run_id": "new_pool",
                "generator": {"variant": "qwen2-40bins"},
                "privacy": {"real_patient_input_used": False},
                "generation": {"input": "bos_token_only", "count": 4, "base_seed": 5200},
                "cases": entries,
            }), encoding="utf-8")
            old_protected, old_parent = module.PROTECTED, module.OUTPUT_PARENT
            module.PROTECTED = protected
            module.OUTPUT_PARENT = protected / "tricompose_v1_2" / "ehr_cohorts"
            try:
                summary = module.screen_pool(
                    source_run=source, expected_source_run_id="new_pool",
                    output_run_id="selected_001", expected_count=4,
                )
                self.assertEqual(summary["status"], "ready_for_two_case_staging")
                result = json.loads((module.OUTPUT_PARENT / "selected_001" / "screen_manifest.json").read_text())
                self.assertEqual(result["source_count"], 4)
                self.assertEqual(result["eligible_count"], 2)
                self.assertEqual(result["selected_case_ids"], ["case_001", "case_002"])
                self.assertFalse(result["generation_is_prompt_conditioned"])
                with self.assertRaises(FileExistsError):
                    module.screen_pool(source_run=source, expected_source_run_id="new_pool", output_run_id="selected_001", expected_count=4)
            finally:
                module.PROTECTED, module.OUTPUT_PARENT = old_protected, old_parent


if __name__ == "__main__":
    unittest.main()
