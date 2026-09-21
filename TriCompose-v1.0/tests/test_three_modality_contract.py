from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from tricompose.privacy import (
    PROTECTED_ROOT,
    enforce_private_file_mode,
    sha256_file,
    write_private_json,
    write_private_text,
)
from tricompose_v1.contracts import (
    ACTIVE_REPORT_MODELS,
    build_cxr_candidate,
    build_cxr_request,
    build_report_candidate,
    build_report_request,
    export_selected_triple,
)
from tricompose_v1.candidate_bank import finalize_candidate_bank
from tricompose_v1.prompts import ACTIVE_PROMPT_MODELS
from tricompose_v1.execution import (
    CXR_CANDIDATE_RUN_SCHEMA,
    REPORT_CANDIDATE_RUN_SCHEMA,
    prepare_cxr_request_run,
    run_cxr_request_run,
    run_report_candidate_run,
)
from tricompose_v1.staging import stage_tricompose_v1


class ThreeModalityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        PROTECTED_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(PROTECTED_ROOT, 0o700)
        self.temp = tempfile.TemporaryDirectory(
            prefix="tricompose_contract_test_", dir=PROTECTED_ROOT
        )
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        source = self.root / "source"
        cases = source / "cases"
        cases.mkdir(parents=True, mode=0o700)
        os.chmod(source, 0o700)
        os.chmod(cases, 0o700)
        case_path = cases / "case_000.json"
        write_private_json(
            case_path,
            {
                "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
                "case_id": "case_000",
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
                "validation": {"strict_valid": True},
            },
        )
        write_private_json(
            source / "run.json",
            {
                "schema": "tricompose.synehrgy_v2.run.v1",
                "generator": {"variant": "qwen2-40bins"},
                "cases": [
                    {
                        "case_id": "case_000",
                        "relative_path": "cases/case_000.json",
                        "sha256": sha256_file(case_path),
                        "strict_valid": True,
                    }
                ],
            },
        )
        case_list = self.root / "cases.txt"
        write_private_text(case_list, "case_000\n")
        staged = stage_tricompose_v1(
            source_root=source,
            case_ids_file=case_list,
            output_root=self.root / "staging",
            run_id="staging_contract",
            prompt_models=ACTIVE_PROMPT_MODELS,
            validate=True,
        )
        self.staging_run = Path(staged["run_directory"])
        self.image_path = self.root / "generated_cxr.png"
        self.image_path.write_bytes(b"synthetic-png-fixture")
        enforce_private_file_mode(self.image_path)
        self.report_path = self.root / "generated_report.txt"
        write_private_text(self.report_path, "Synthetic report fixture.")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _candidates(self):
        request = build_cxr_request(
            staging_run=self.staging_run,
            case_id="case_000",
            model_id="chexgenbench_sana",
            seed=0,
        )
        cxr = build_cxr_candidate(
            request=request,
            image_path=self.image_path,
            image_dimensions=(1024, 1024),
            model_revision="fixture-revision",
            prompt_token_count=8,
            runtime_seconds=1.0,
            peak_vram_gib=1.0,
        )
        report_request = build_report_request(
            cxr_candidate=cxr,
            report_model_id="maira2",
        )
        report = build_report_candidate(
            request=report_request,
            report_path=self.report_path,
            model_revision="fixture-revision",
            runtime_seconds=1.0,
        )
        return request, cxr, report_request, report

    def test_all_three_cxr_models_share_ehr_and_fact_inputs(self) -> None:
        requests = [
            build_cxr_request(
                staging_run=self.staging_run,
                case_id="case_000",
                model_id=model_id,
                seed=0,
            )
            for model_id in ACTIVE_PROMPT_MODELS
        ]
        self.assertEqual(
            len({row["inputs"]["synthetic_ehr"]["sha256"] for row in requests}),
            1,
        )
        self.assertEqual(
            len({row["inputs"]["ehr_facts"]["sha256"] for row in requests}),
            1,
        )
        self.assertTrue(
            all(
                row["inputs"]["final_prompt"]["is_final_model_input"]
                and row["inputs"]["final_prompt"]["adapter_must_not_add_prefix"]
                for row in requests
            )
        )

    def test_all_report_models_share_the_same_cxr_and_lineage_contract(self) -> None:
        _, cxr, _, _ = self._candidates()
        requests = [
            build_report_request(cxr_candidate=cxr, report_model_id=model_id)
            for model_id in ACTIVE_REPORT_MODELS
        ]
        self.assertEqual(
            len({row["inputs"]["synthetic_cxr"]["sha256"] for row in requests}),
            1,
        )
        self.assertTrue(
            all(row["parent_cxr_candidate_id"] == cxr["candidate_id"] for row in requests)
        )
        self.assertTrue(
            all(row["inputs"]["ehr_content_supplied_to_model"] is False for row in requests)
        )

    def test_export_happens_once_only_after_complete_lineage_selection(self) -> None:
        _, cxr, _, report = self._candidates()
        selection = {
            "action": "stop_and_select",
            "selected": {
                "ehr_candidate_id": "ehr_case_000",
                "cxr_candidate_id": cxr["candidate_id"],
                "report_candidate_id": report["candidate_id"],
            },
        }
        result = export_selected_triple(
            staging_run=self.staging_run,
            cxr_candidate=cxr,
            report_candidate=report,
            selection=selection,
            output_root=self.root / "exports",
            export_id="triple_001",
        )
        export_dir = Path(result["export_directory"])
        self.assertTrue(result["complete_three_modality_export"])
        self.assertTrue((export_dir / "synthetic_ehr.json").is_file())
        self.assertTrue((export_dir / "synthetic_cxr.png").is_file())
        self.assertTrue((export_dir / "synthetic_report.txt").is_file())
        manifest = json.loads((export_dir / "triple_manifest.json").read_text())
        self.assertTrue(manifest["complete_three_modality_export"])
        with self.assertRaises(FileExistsError):
            export_selected_triple(
                staging_run=self.staging_run,
                cxr_candidate=cxr,
                report_candidate=report,
                selection=selection,
                output_root=self.root / "exports",
                export_id="triple_001",
            )

    def test_partial_or_cross_lineage_export_is_refused(self) -> None:
        _, cxr, _, report = self._candidates()
        incomplete = {"action": "verify_more"}
        with self.assertRaises(ValueError):
            export_selected_triple(
                staging_run=self.staging_run,
                cxr_candidate=cxr,
                report_candidate=report,
                selection=incomplete,
                output_root=self.root / "exports",
                export_id="partial_001",
            )
        wrong_report = copy.deepcopy(report)
        wrong_report["parent_ids"][1] = "cxr_case_000_roentgen_v2_s000999"
        wrong_report["candidate_id"] = (
            "report_cxr_case_000_roentgen_v2_s000999_maira2"
        )
        selection = {
            "action": "stop_and_select",
            "selected": {
                "ehr_candidate_id": "ehr_case_000",
                "cxr_candidate_id": cxr["candidate_id"],
                "report_candidate_id": wrong_report["candidate_id"],
            },
        }
        with self.assertRaises(ValueError):
            export_selected_triple(
                staging_run=self.staging_run,
                cxr_candidate=cxr,
                report_candidate=wrong_report,
                selection=selection,
                output_root=self.root / "exports",
                export_id="wrong_lineage_001",
            )

    def test_common_execution_adapters_emit_aligned_candidate_runs(self) -> None:
        case_list = self.root / "execution_cases.txt"
        write_private_text(case_list, "case_000\n")
        prepared = prepare_cxr_request_run(
            staging_run=self.staging_run,
            case_ids_file=case_list,
            output_root=self.root / "request_runs",
            run_id="requests_001",
            model_ids=ACTIVE_PROMPT_MODELS,
            seeds=(0, 1),
            require_conditioned=True,
        )
        self.assertEqual(prepared["request_count"], 6)

        class FakeImage:
            size = (1024, 1024)

            def convert(self, mode):
                if mode != "RGB":
                    raise ValueError("unexpected fixture mode")
                return self

            def save(self, path, format):
                if format != "PNG":
                    raise ValueError("unexpected fixture format")
                Path(path).write_bytes(b"synthetic-png-fixture")

        class FakeCXRRuntime:
            audit = {"fixture": "cxr"}

            def generate_batch(self, prompts, seeds):
                if any(not prompt.strip() for prompt in prompts):
                    raise ValueError("empty fixture prompt")
                return SimpleNamespace(
                    images=[FakeImage() for _ in prompts],
                    prompt_token_counts=[8 for _ in prompts],
                    elapsed_seconds=float(len(prompts)),
                    peak_vram_gib=1.0,
                )

        cxr_results = []
        for model_id in ACTIVE_PROMPT_MODELS:
            result = run_cxr_request_run(
                request_run=prepared["run_directory"],
                output_root=self.root / "cxr_runs",
                output_run_id=f"{model_id}_001",
                model_id=model_id,
                model_revision="fixture-revision",
                model_audit={"fixture": "cxr"},
                runtime_factory=FakeCXRRuntime,
                batch_size=1 if model_id == "chexgenbench_pixart" else 2,
            )
            self.assertEqual(result["candidate_count"], 2)
            cxr_manifest = json.loads(
                (Path(result["run_directory"]) / "manifest.json").read_text()
            )
            self.assertEqual(
                cxr_manifest["schema_version"], CXR_CANDIDATE_RUN_SCHEMA
            )
            cxr_results.append(result)

        class FakeReportRuntime:
            audit = {"fixture": "report"}
            peak_vram_gib = 1.0

            def generate(self, image_path):
                if not Path(image_path).is_file():
                    raise FileNotFoundError("fixture CXR missing")
                return SimpleNamespace(
                    canonical_text="Synthetic report fixture.",
                    elapsed_seconds=1.0,
                )

        report_results = []
        cxr_run_paths = [row["run_directory"] for row in cxr_results]
        for model_id in ACTIVE_REPORT_MODELS:
            result = run_report_candidate_run(
                cxr_run=cxr_run_paths,
                output_root=self.root / "report_runs",
                output_run_id=f"{model_id}_001",
                model_id=model_id,
                model_revision="fixture-revision",
                model_audit={"fixture": "report"},
                runtime_factory=FakeReportRuntime,
            )
            self.assertEqual(result["candidate_count"], 6)
            report_manifest = json.loads(
                (Path(result["run_directory"]) / "manifest.json").read_text()
            )
            self.assertEqual(
                report_manifest["schema_version"], REPORT_CANDIDATE_RUN_SCHEMA
            )
            self.assertEqual(len(report_manifest["source_cxr_runs"]), 3)
            report_results.append(result)

        bank = finalize_candidate_bank(
            staging_run=self.staging_run,
            cxr_runs=cxr_run_paths,
            report_runs=[row["run_directory"] for row in report_results],
            output_root=self.root / "candidate_banks",
            run_id="bank_001",
        )
        self.assertEqual(bank["ehr_candidate_count"], 1)
        self.assertEqual(bank["cxr_candidate_count"], 6)
        self.assertEqual(bank["report_candidate_count"], 24)


if __name__ == "__main__":
    unittest.main()
