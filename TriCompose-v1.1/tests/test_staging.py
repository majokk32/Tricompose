from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import tricompose_v11.staging as staging
from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v1.facts import extract_ehr_facts
from tricompose_v1.prompts import ACTIVE_PROMPT_MODELS, render_prompt


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(case_id: str, diagnosis: str) -> dict:
    case = canonicalize_synehrgy_case(
        {
            "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
            "case_id": case_id,
            "structure": {
                "top_level_tokens": [],
                "visits": [
                    {
                        "covariates": [],
                        "problems": [diagnosis],
                        "labs": [],
                        "charts": [],
                        "other_tokens": [],
                    }
                ],
            },
            "validation": {"strict_valid": True},
        },
        source_model_id="synehrgy_gpt2_10bins",
    )
    return case


class V11StagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="tricompose_v11_")
        self.workspace = Path(self.temp.name)
        self.protected = self.workspace / "artifacts" / "protected"
        self.source = self.workspace / "source" / "artifacts" / "protected" / "v1"
        self.protected.mkdir(parents=True)
        self.source.mkdir(parents=True)
        (self.source / "cases").mkdir()
        rows = []
        for case_id, diagnosis in (
            ("case_000", "J44_Chronic obstructive pulmonary disease"),
            ("case_001", "J45_Asthma"),
        ):
            case_dir = self.source / "cases" / case_id
            (case_dir / "cxr_prompts").mkdir(parents=True)
            ehr_path = case_dir / "synthetic_ehr.json"
            ehr_path.write_text(json.dumps(_canonical(case_id, diagnosis), sort_keys=True))
            facts = extract_ehr_facts(json.loads(ehr_path.read_text()))
            models = {}
            for model_id in ACTIVE_PROMPT_MODELS:
                rendered = render_prompt(facts, model_id)
                models[model_id] = {"prompt_sha256": rendered["prompt_sha256"]}
            (case_dir / "cxr_prompts" / "prompt_manifest.json").write_text(
                json.dumps({"models": models})
            )
            rows.append(
                {
                    "case_id": case_id,
                    "status": "staged",
                    "relative_path": f"cases/{case_id}",
                    "hashes": {"synthetic_ehr_sha256": _sha(ehr_path)},
                }
            )
        (self.source / "cohort.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (self.source / "run_manifest.json").write_text(
            json.dumps({"schema_version": "tricompose.staging.run.v1"})
        )
        self.old_workspace = staging.WORKSPACE
        self.old_protected = staging.MAIN_PROTECTED_ROOT
        staging.WORKSPACE = self.workspace
        staging.MAIN_PROTECTED_ROOT = self.protected

    def tearDown(self) -> None:
        staging.WORKSPACE = self.old_workspace
        staging.MAIN_PROTECTED_ROOT = self.old_protected
        self.temp.cleanup()

    def test_stage_is_non_overwriting_and_does_not_modify_source_ehr(self) -> None:
        before = {
            case_id: _sha(self.source / "cases" / case_id / "synthetic_ehr.json")
            for case_id in ("case_000", "case_001")
        }
        result = staging.stage_existing_v1_ehrs(
            source_v1_run=self.source,
            output_root=self.protected / "tricompose_v1_1" / "staging",
            run_id="bridge_test_001",
        )
        self.assertTrue(result["overall_valid"])
        self.assertEqual(result["v1_unique_prompt_count_by_model"]["roentgen_v2"], 1)
        self.assertEqual(result["v1_1_unique_prompt_count_by_model"]["roentgen_v2"], 2)
        run = Path(result["run_directory"])
        for case_id in before:
            source_ehr = self.source / "cases" / case_id / "synthetic_ehr.json"
            copied_ehr = run / "cases" / case_id / "synthetic_ehr.json"
            self.assertEqual(_sha(source_ehr), before[case_id])
            self.assertEqual(_sha(copied_ehr), before[case_id])
            self.assertEqual(stat.S_IMODE(copied_ehr.stat().st_mode), 0o660)
            self.assertEqual(stat.S_IMODE(copied_ehr.parent.stat().st_mode), 0o2770)
        with self.assertRaises(FileExistsError):
            staging.stage_existing_v1_ehrs(
                source_v1_run=self.source,
                output_root=self.protected / "tricompose_v1_1" / "staging",
                run_id="bridge_test_001",
            )


if __name__ == "__main__":
    unittest.main()
