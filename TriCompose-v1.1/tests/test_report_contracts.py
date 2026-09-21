from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tricompose_v11.cxr_contracts as cxr_contracts
import tricompose_v11.report_contracts as contracts


class V11ReportContractTests(unittest.TestCase):
    def test_eight_cxrs_produce_four_hash_bound_requests_each(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tricompose_v11_report_") as temp:
            root = Path(temp)
            cxr_root = root / "cxr"
            output_root = root / "report_requests"
            cxr_root.mkdir()
            candidate_runs = []
            for model_index, cxr_model in enumerate(("sana", "pixart")):
                run = cxr_root / cxr_model
                (run / "candidates").mkdir(parents=True)
                rows = []
                for offset in range(4):
                    index = model_index * 4 + offset
                    candidate_id = f"cxr_case_{index:03d}_{cxr_model}_s000000"
                    candidate_dir = run / "candidates" / candidate_id
                    candidate_dir.mkdir()
                    image_path = candidate_dir / "synthetic_cxr.png"
                    image_path.write_bytes(b"synthetic-png-" + bytes([index]))
                    candidate = {
                        "schema_version": "tricompose-cxr-candidate-v1.1",
                        "candidate_id": candidate_id,
                        "case_id": f"case_{index:03d}",
                        "model_id": cxr_model,
                        "frozen_model": True,
                        "adapter_added_prefix": False,
                        "seed": 0,
                        "ehr_sha256": "e" * 64,
                        "ehr_facts_sha256": "f" * 64,
                        "clinical_intent_sha256": "c" * 64,
                        "artifact": {
                            "path": str(image_path),
                            "sha256": contracts.sha256_file(image_path),
                            "mime_type": "image/png",
                        },
                    }
                    candidate_path = candidate_dir / "candidate.json"
                    cxr_contracts.write_private_json(candidate_path, candidate)
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "path": f"candidates/{candidate_id}/candidate.json",
                            "sha256": contracts.sha256_file(candidate_path),
                        }
                    )
                cxr_contracts.write_private_json(
                    run / "manifest.json",
                    {
                        "schema_version": "tricompose-cxr-candidate-run-v1.1",
                        "candidates": rows,
                    },
                )
                candidate_runs.append(run)

            with (
                mock.patch.object(contracts, "MAIN_PROTECTED_ROOT", root),
                mock.patch.object(cxr_contracts, "MAIN_PROTECTED_ROOT", root),
            ):
                result = contracts.prepare_report_request_run(
                    cxr_runs=candidate_runs,
                    output_root=output_root,
                    run_id="request_run_001",
                )
                self.assertEqual(result["cxr_candidate_count"], 8)
                self.assertEqual(result["request_count"], 32)
                manifest = cxr_contracts.read_json(
                    output_root / "request_run_001" / "manifest.json"
                )
                self.assertEqual(manifest["request_count"], 32)
                self.assertEqual(
                    {row["model_id"] for row in manifest["requests"]},
                    set(contracts.ACTIVE_REPORT_MODELS_V11),
                )
                for model_id in contracts.ACTIVE_REPORT_MODELS_V11:
                    requests = contracts.load_model_requests(
                        output_root / "request_run_001", model_id
                    )
                    self.assertEqual(len(requests), 8)
                    for request in requests:
                        self.assertEqual(
                            request["model_input_signature"],
                            "single_current_synthetic_cxr",
                        )
                        self.assertFalse(
                            request["inputs"][
                                "structured_ehr_content_supplied_to_model"
                            ]
                        )
                        self.assertFalse(
                            request["inputs"][
                                "source_report_or_real_target_supplied"
                            ]
                        )


if __name__ == "__main__":
    unittest.main()
