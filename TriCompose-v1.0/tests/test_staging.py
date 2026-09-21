from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from tricompose.privacy import PROTECTED_ROOT, sha256_file, write_private_json, write_private_text
from tricompose_v1.staging import stage_tricompose_v1


PROMPT_MODELS = (
    "roentgen_v2",
    "chexgenbench_sana",
    "chexgenbench_pixart",
)


def _source_case(case_id: str, *, valid: bool) -> dict:
    return {
        "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
        "case_id": case_id,
        "structure": {
            "top_level_tokens": [],
            "visits": [
                {
                    "covariates": [],
                    "problems": ["I517_Cardiomegaly"],
                    "labs": [],
                    "charts": ["Presence of cardiac pacemaker"],
                    "other_tokens": [],
                }
            ],
        },
        "validation": {"strict_valid": valid},
    }


class StagingTests(unittest.TestCase):
    def setUp(self) -> None:
        PROTECTED_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(PROTECTED_ROOT, 0o700)
        self.temp = tempfile.TemporaryDirectory(
            prefix="tricompose_staging_test_", dir=PROTECTED_ROOT
        )
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.source = self.root / "source"
        cases = self.source / "cases"
        cases.mkdir(parents=True, mode=0o700)
        os.chmod(self.source, 0o700)
        os.chmod(cases, 0o700)
        entries = []
        for case_id, valid in (("case_000", True), ("case_001", False)):
            path = cases / f"{case_id}.json"
            write_private_json(path, _source_case(case_id, valid=valid))
            entries.append(
                {
                    "case_id": case_id,
                    "relative_path": f"cases/{case_id}.json",
                    "sha256": sha256_file(path),
                    "strict_valid": valid,
                }
            )
        write_private_json(
            self.source / "run.json",
            {
                "schema": "tricompose.synehrgy_v2.run.v1",
                "generator": {"variant": "qwen2-40bins"},
                "cases": entries,
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _case_file(self, name: str, text: str) -> Path:
        path = self.root / name
        write_private_text(path, text)
        return path

    def test_staging_is_deterministic_private_and_non_overwriting(self) -> None:
        case_file = self._case_file("one_case.txt", "case_000\n")
        first = stage_tricompose_v1(
            source_root=self.source,
            case_ids_file=case_file,
            output_root=self.root / "output",
            run_id="staging_a",
            prompt_models=PROMPT_MODELS,
            validate=True,
        )
        second = stage_tricompose_v1(
            source_root=self.source,
            case_ids_file=case_file,
            output_root=self.root / "output",
            run_id="staging_b",
            prompt_models=PROMPT_MODELS,
            validate=True,
        )
        self.assertTrue(first["overall_valid"])
        self.assertTrue(second["overall_valid"])
        first_case = Path(first["run_directory"]) / "cases" / "case_000"
        second_case = Path(second["run_directory"]) / "cases" / "case_000"
        relative_files = (
            "source_manifest.json",
            "synthetic_ehr.json",
            "ehr_facts.json",
            "cxr_prompts/prompt_manifest.json",
            "cxr_prompts/roentgen_v2.txt",
            "cxr_prompts/chexgenbench_sana.txt",
            "cxr_prompts/chexgenbench_pixart.txt",
        )
        for relative in relative_files:
            self.assertEqual(
                sha256_file(first_case / relative),
                sha256_file(second_case / relative),
            )
            self.assertEqual(
                stat.S_IMODE((first_case / relative).stat().st_mode), 0o600
            )
        for directory in (first_case, first_case / "cxr_prompts"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        with self.assertRaises(FileExistsError):
            stage_tricompose_v1(
                source_root=self.source,
                case_ids_file=case_file,
                output_root=self.root / "output",
                run_id="staging_a",
                prompt_models=PROMPT_MODELS,
                validate=True,
            )

    def test_invalid_case_is_rejected_without_partial_case_directory(self) -> None:
        case_file = self._case_file("two_cases.txt", "case_000\ncase_001\n")
        result = stage_tricompose_v1(
            source_root=self.source,
            case_ids_file=case_file,
            output_root=self.root / "output",
            run_id="staging_rejection",
            prompt_models=PROMPT_MODELS,
            validate=True,
        )
        self.assertEqual(result["successful_case_count"], 1)
        self.assertEqual(result["rejected_case_count"], 1)
        run_dir = Path(result["run_directory"])
        self.assertFalse((run_dir / "cases" / "case_001").exists())
        report = json.loads((run_dir / "validation_report.json").read_text())
        self.assertEqual(
            report["rejection_reasons"], {"source_case_not_strict_valid": 1}
        )
        self.assertFalse(any(path.name.startswith(".case_") for path in (run_dir / "cases").iterdir()))


if __name__ == "__main__":
    unittest.main()
