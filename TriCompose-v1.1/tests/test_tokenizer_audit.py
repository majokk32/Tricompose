"""Small synthetic fixtures only; no tokenizers, checkpoints or patient data."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tricompose_v11.cxr_contracts as io
import tricompose_v11.report_contracts as reports
import tricompose_v11.tokenizer_audit as audit
from tricompose_v11.tokenizer_trace import text_sha256, validate_tokenizer_trace


class TokenizerAuditTests(unittest.TestCase):
    def test_reconstruction_binding_immutability_and_tampering(self):
        with tempfile.TemporaryDirectory(prefix="tokenizer_audit_") as temp:
            root = Path(temp)
            source = root / "cxr_run"
            candidate_dir = source / "candidates" / "cxr_case_000"
            candidate_dir.mkdir(parents=True)
            prompt = io.write_private_text(root / "prompt.txt", "Synthetic Test Prompt")
            image = candidate_dir / "synthetic_cxr.png"
            image.write_bytes(b"synthetic-test-bytes")
            request = {"inputs": {"final_prompt": {"path": str(prompt), "sha256": io.sha256_file(prompt)}}}
            io.write_private_json(candidate_dir / "request.json", request)
            candidate = {
                "schema_version": "tricompose-cxr-candidate-v1.1",
                "candidate_id": "cxr_case_000", "case_id": "case_000",
                "model_id": "roentgen_v2", "frozen_model": True,
                "adapter_added_prefix": False, "seed": 0,
                "ehr_sha256": "e" * 64, "ehr_facts_sha256": "f" * 64,
                "clinical_intent_sha256": "c" * 64,
                "input_request_sha256": io.canonical_json_sha256(request),
                "prompt_sha256": io.sha256_file(prompt),
                "artifact": {"path": str(image), "sha256": io.sha256_file(image), "mime_type": "image/png"},
            }
            candidate_file = io.write_private_json(candidate_dir / "candidate.json", candidate)
            io.write_private_json(source / "manifest.json", {
                "schema_version": "tricompose-cxr-candidate-run-v1.1",
                "model_id": "roentgen_v2", "frozen_model": True, "candidate_count": 1,
                "candidates": [{"candidate_id": "cxr_case_000", "path": "candidates/cxr_case_000/candidate.json",
                                "sha256": io.sha256_file(candidate_file)}],
            })
            pipeline = io.write_private_text(root / "pipeline.py", "# synthetic source fixture\n")
            before = {p: io.sha256_file(p) for p in source.rglob("*") if p.is_file()}
            with (
                mock.patch.object(audit, "WORKSPACE", root),
                mock.patch.object(audit, "MAIN_PROTECTED_ROOT", root),
                mock.patch.object(reports, "MAIN_PROTECTED_ROOT", root),
                mock.patch.object(io, "MAIN_PROTECTED_ROOT", root),
                mock.patch.dict(audit.PIPELINE_FILES, {"roentgen_v2": pipeline.name}),
                mock.patch.dict(audit.ADAPTER_FILES, {"roentgen_v2": pipeline.name}),
            ):
                args = dict(cxr_runs=[source], output_root=root / "audits", run_id="audit_001")
                result = audit.build_tokenizer_audit(**args)
                self.assertFalse(result["runtime_observed"])
                with self.assertRaises(FileExistsError):
                    audit.build_tokenizer_audit(**args)
                self.assertEqual(before, {p: io.sha256_file(p) for p in before})
                refs = audit.audit_references(result["run_directory"], [candidate])
                report = reports.build_report_request(
                    cxr_candidate=candidate, report_model_id="maira2",
                    cxr_input_audit=refs[candidate["candidate_id"]],
                )
                reports.validate_report_request(report)
                changed = copy.deepcopy(report)
                changed["inputs"]["cxr_candidate_sha256"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "different candidate"):
                    reports.validate_report_request(changed)
                changed = copy.deepcopy(candidate)
                changed["seed"] = 123
                with self.assertRaisesRegex(ValueError, "different CXR candidate"):
                    audit.audit_references(result["run_directory"], [changed])
                Path(refs[candidate["candidate_id"]]["record_path"]).write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "audit hash mismatch"):
                    reports.validate_report_request(report)

    def test_observed_artifact_binding_and_legacy_is_not_relabelled(self):
        with (
            tempfile.TemporaryDirectory(prefix="tokenizer_trace_binding_") as temp,
            mock.patch.object(io, "MAIN_PROTECTED_ROOT", Path(temp)),
        ):
            root = Path(temp)
            text = io.write_private_text(root / "tokenizer_input.txt", "synthetic prompt")
            trace = io.write_private_json(root / "trace.json", {
                "runtime_observed": True, "official_generation_arguments_changed": False,
                "supplied_prompt_sha256": "a" * 64,
                "positive_tokenizer_text": "synthetic prompt",
                "positive_tokenizer_text_sha256": text_sha256("synthetic prompt"),
                "positive_input_ids_sha256": "i" * 64,
            })
            candidate = {"prompt_sha256": "a" * 64, "tokenizer_input": {
                "runtime_observed": True, "path": str(text), "sha256": io.sha256_file(text),
                "trace_path": str(trace), "trace_sha256": io.sha256_file(trace),
                "input_ids_sha256": "i" * 64,
            }}
            validate_tokenizer_trace(candidate, root)
            legacy = {"prompt_sha256": "a" * 64}
            validate_tokenizer_trace(legacy, root)
            self.assertNotIn("tokenizer_input", legacy)
            candidate["prompt_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "another supplied prompt"):
                validate_tokenizer_trace(candidate, root)
            candidate["prompt_sha256"] = "a" * 64
            text.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_tokenizer_trace(candidate, root)


if __name__ == "__main__":
    unittest.main()
