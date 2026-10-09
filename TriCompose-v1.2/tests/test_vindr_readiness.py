"""Invented schema/network/authorization fixtures only; no dataset/model calls."""
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

path = Path(__file__).resolve().parents[1] / "real_validation/vindr_readiness.py"
spec = importlib.util.spec_from_file_location("vindr_readiness", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def invented_header(extra=()):
    return (",".join(("image_id", "Atelectasis", "Cardiomegaly", "Consolidation",
                      "Edema", "Lung opacity", "Pleural effusion", "Pneumonia",
                      "Pneumothorax", *extra)) + "\n").encode("utf-8")


def invented_attestation():
    return {"schema_version": module.ATTESTATION_SCHEMA, "dataset": "vindr-cxr",
            "dataset_version": "1.0.0", "purpose": "header_only_readiness",
            "local_dataset_root": "/invented_fixture_not_a_real_dataset",
            **dict.fromkeys(module.REQUIRED_CONFIRMATIONS, True)}


class VinDrReadinessTests(unittest.TestCase):
    def test_candidate_header_is_not_an_executed_benchmark(self):
        result = module.header_inventory(invented_header())
        self.assertEqual(result["status"], "named_header_candidate_requires_semantics_review")
        self.assertFalse(result["official_schema_byte_authenticity_verified"])
        self.assertFalse(result["annotation_rows_read"])
        self.assertFalse(result["reference_states_inferred"])

    def test_header_hash_and_mapping_are_deterministic(self):
        self.assertEqual(module.header_inventory(invented_header()),
                         module.header_inventory(invented_header()))

    def test_unknown_column_text_never_appears_in_inventory(self):
        result = module.header_inventory(invented_header(("invented_sensitive_header",)))
        self.assertNotIn("invented_sensitive_header", json.dumps(result))
        self.assertEqual(len(result["unrecognized_columns"]), 1)

    def test_missing_pneumonia_is_unknown_not_negative(self):
        line = invented_header().replace(b",Pneumonia", b"")
        result = module.header_inventory(line)
        self.assertIn("pneumonia", result["missing_required_names"])
        self.assertEqual(result["missing_reference_policy"], "unknown_not_negative")
        self.assertFalse(result["reference_states_inferred"])
        self.assertEqual(result["status"], "blocked_missing_named_reference_heads")

    def test_no_synonyms_or_vector_positions_are_inferred(self):
        line = invented_header().replace(b"Pneumonia", b"pneumonia_suspected")
        self.assertIn("pneumonia", module.header_inventory(line)["missing_required_names"])

    def test_duplicate_exact_and_aliased_headers_fail_closed(self):
        for extra in ("Pneumonia", "Lung Opacity", "Pleural Effusion"):
            with self.subTest(extra=extra):
                result = module.header_inventory(invented_header((extra,)))
                self.assertEqual(result["status"], "blocked_duplicate_or_aliased_columns")

    def test_bbox_source_is_not_a_global_diagnostic_reference(self):
        line = b"image_id,class_name,x_min,y_min,x_max,y_max\n"
        self.assertEqual(module.header_inventory(line)["status"],
                         "blocked_bounding_box_source_not_global_labels")

    def test_label_vector_requires_documented_order(self):
        self.assertEqual(module.header_inventory(b"image_id,labels\n")["status"],
                         "blocked_vector_schema_requires_documented_order")

    def test_train_reader_header_not_used_as_test_consensus(self):
        self.assertEqual(module.header_inventory(invented_header(("rad_ID",)))["status"],
                         "blocked_train_reader_schema_not_test_consensus")

    def test_malformed_or_overlong_header_never_accepted(self):
        for line in (b"", b"image_id,Pneumonia", b'"unclosed\n', b"image_id,,Pneumonia\n",
                     b"\xff\n", b"x" * module.MAX_HEADER_BYTES + b"\n"):
            with self.subTest(length=len(line)):
                with self.assertRaises(ValueError):
                    module.header_inventory(line)

    def test_default_run_does_not_touch_paths_network_or_labels(self):
        with patch.object(module, "inspect_authorized_header") as inspect, \
                patch.object(module, "probe_unauthenticated_head") as probe:
            result = module.build_readiness()
        inspect.assert_not_called()
        probe.assert_not_called()
        self.assertFalse(result["benchmark_executed"])
        self.assertFalse(result["can_prepare_gpu_job"])
        self.assertEqual(result["patient_grouping"], "unverified_image_id_is_not_patient_id")

    def test_explicit_header_flag_without_receipt_fails(self):
        with self.assertRaisesRegex(ValueError, "access_attestation_required"):
            module.build_readiness(allow_authorized_header=True)

    def test_slurm_or_explicit_flag_missing_refuses_before_path_operations(self):
        for environment, allow in (({}, True), ({"SLURM_JOB_ID": "fixture"}, False)):
            with patch.dict(module.os.environ, environment, clear=True), \
                    patch.object(module.Path, "resolve") as resolve:
                with self.assertRaisesRegex(RuntimeError, "explicit_header_approval_and_slurm_required"):
                    module.inspect_authorized_header("invented", allow_authorized_header=allow)
                resolve.assert_not_called()

    def test_attestation_is_exact_does_not_accept_credentials_or_missing_confirmations(self):
        for mutate in (lambda x: x.update(token="invented_not_a_token"),
                       lambda x: x.update(dataset="mimic"),
                       lambda x: x.update(purpose="inference"),
                       lambda x: x.update(dataset_specific_dua_confirmed=1),
                       lambda x: x.update(all_project_group_readers_authorized_confirmed=False),
                       lambda x: x.pop("required_training_confirmed")):
            value = invented_attestation()
            mutate(value)
            with self.assertRaisesRegex(ValueError, "invalid_access_attestation"):
                module.validate_attestation(value)

    def test_attestation_validation_has_no_network_effects(self):
        self.assertEqual(module.validate_attestation(invented_attestation()),
                         "/invented_fixture_not_a_real_dataset")

    def test_header_candidate_does_not_authorize_gpu_or_confirm_label_semantics(self):
        checked = {"header": module.header_inventory(invented_header())}
        with patch.object(module, "inspect_authorized_header", return_value=checked):
            result = module.build_readiness(access_attestation="fixture", allow_authorized_header=True)
        self.assertEqual(result["status"], "pending_label_semantics_and_dicom_preflight")
        self.assertFalse(result["can_prepare_gpu_job"])
        self.assertFalse(result["primary_metric_eligible"])
        self.assertEqual(result["label_encoding"], "not_verified_no_label_rows_read")

    def test_authorized_fixture_reads_exactly_one_physical_header_line(self):
        with tempfile.TemporaryDirectory(prefix="vindr_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            root = Path(folder)
            module.os.chmod(root, 0o2770)
            annotations = root / "annotations"
            annotations.mkdir(mode=0o2770)
            module.os.chmod(annotations, 0o2770)
            header = invented_header()
            module.write_private_text(annotations / "image_labels_test.csv",
                                      header.decode() + "invented_row_must_not_be_read\n")
            receipt = invented_attestation()
            receipt["local_dataset_root"] = str(root)
            module.write_private_json(root / "access.json", receipt)
            fdopen = module.os.fdopen
            positions = []

            class TrackedFile:
                def __init__(self, handle):
                    self.handle = handle

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.handle.close()

                def fileno(self):
                    return self.handle.fileno()

                def readline(self, limit):
                    result = self.handle.readline(limit)
                    positions.append(self.handle.tell())
                    return result

            def tracked_open(fd, *args, **kwargs):
                self.assertEqual(kwargs.get("buffering"), 0)
                return TrackedFile(fdopen(fd, *args, **kwargs))

            with patch.dict(module.os.environ, {"SLURM_JOB_ID": "invented_fixture"}), \
                    patch.object(module.os, "fdopen", side_effect=tracked_open):
                checked = module.inspect_authorized_header(root / "access.json", allow_authorized_header=True)
            self.assertEqual(positions, [len(header)])
            self.assertEqual(checked["header"]["header_sha256"], module.digest_bytes(header))
            self.assertNotIn("invented_row_must_not_be_read", json.dumps(checked))
            self.assertNotIn(str(root), json.dumps(checked))

    def test_invalid_authorization_never_opens_annotation(self):
        with tempfile.TemporaryDirectory(prefix="vindr_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            root = Path(folder)
            module.os.chmod(root, 0o2770)
            receipt = invented_attestation()
            receipt["dataset_specific_dua_confirmed"] = False
            module.write_private_json(root / "access.json", receipt)
            with patch.dict(module.os.environ, {"SLURM_JOB_ID": "invented_fixture"}), \
                    patch.object(module.os, "open") as source_open:
                with self.assertRaisesRegex(ValueError, "invalid_access_attestation"):
                    module.inspect_authorized_header(root / "access.json", allow_authorized_header=True)
            source_open.assert_not_called()

    def test_public_attestation_refused_before_receipt_read(self):
        with tempfile.TemporaryDirectory(prefix="vindr_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            root = Path(folder)
            module.os.chmod(root, 0o2770)
            receipt = root / "access.json"
            module.write_private_json(receipt, invented_attestation())
            module.os.chmod(receipt, 0o664)
            with patch.dict(module.os.environ, {"SLURM_JOB_ID": "invented_fixture"}), \
                    patch.object(module.Path, "read_text") as read:
                with self.assertRaisesRegex(ValueError, "private_attestation_boundary_required"):
                    module.inspect_authorized_header(receipt, allow_authorized_header=True)
            read.assert_not_called()

    def test_head_uses_only_fixed_url_no_body_or_authentication(self):
        response = Mock(status=200, headers={"Content-Type": "text/csv"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(module, "build_opener", return_value=opener) as build:
            result = module.probe_unauthenticated_head()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, module.PROBE_URL)
        self.assertEqual(request.get_method(), "HEAD")
        self.assertNotIn("Authorization", dict(request.header_items()))
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 15)
        self.assertEqual(build.call_args.args[0].proxies, {})
        response.read.assert_not_called()
        self.assertFalse(result["response_body_read"])
        self.assertEqual(result["status"], "head_responded_not_data_or_dua_verified")
        datetime.fromisoformat(result["checked_at_utc"]).astimezone(timezone.utc)

    def test_403_does_not_read_body_or_attempt_authentication(self):
        body = io.BytesIO(b"invented response must not be read")
        error = HTTPError(module.PROBE_URL, 403, "forbidden", {}, body)
        with patch.object(body, "read", wraps=body.read) as read, \
                patch.object(module, "build_opener", return_value=Mock(open=Mock(side_effect=error))):
            result = module.probe_unauthenticated_head()
        read.assert_not_called()
        self.assertEqual(result["status"], "unauthenticated_access_denied")
        self.assertEqual(result["http_status"], 403)

    def test_redirect_html_missing_url_and_network_failure_not_ready(self):
        self.assertEqual(module.NoRedirect().redirect_request(None, None, 302, "", {}, ""), None)
        for code, content, expected in (
                (302, "text/csv", "redirect_not_followed"),
                (200, "text/html", "html_response_not_dataset_access"),
                (404, "", "resource_unavailable_at_probed_url")):
            self.assertEqual(module._head_result(code, content)["status"], expected)
        with patch.object(module, "build_opener", return_value=Mock(open=Mock(side_effect=URLError("fixture")))):
            self.assertEqual(module.probe_unauthenticated_head()["status"],
                             "network_unavailable_not_access_verified")

    def test_private_receipt_integrity_modes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="vindr_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            result = module.build_readiness()
            target = module.write_run(result, output_root=folder, run_id="fixture_001")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o2770)
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "fixture_001")
            for name, digest in manifest["artifacts"].items():
                self.assertEqual(module.sha256_file(target / name), digest)
                self.assertEqual(stat.S_IMODE((target / name).stat().st_mode), 0o660)
            with self.assertRaises(FileExistsError):
                module.write_run(result, output_root=folder, run_id="fixture_001")

    def test_existing_cli_run_fails_before_probe_or_real_access_and_masks_errors(self):
        with patch.object(module, "new_atomic_run", side_effect=FileExistsError("invented_sensitive_value")), \
                patch.object(module, "build_readiness") as build, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(module.main(["--run-id", "fixture_001", "--probe-public-access"]), 2)
        build.assert_not_called()
        self.assertNotIn("invented_sensitive_value", output.getvalue())

    def test_cli_default_persists_correct_run_id_and_no_attestation(self):
        with tempfile.TemporaryDirectory(prefix="vindr_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = module.main(["--run-id", "fixture_002", "--output-root", folder])
            self.assertEqual(code, 0)
            target = Path(folder) / "fixture_002"
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "fixture_002")
            self.assertFalse(json.loads(output.getvalue())["benchmark_executed"])


if __name__ == "__main__":
    unittest.main()
