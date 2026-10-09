"""Invented archives/headers and mocked network; no clinical fixtures."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

path = Path(__file__).resolve().parents[1] / "real_validation/acquire_rsua.py"
spec = importlib.util.spec_from_file_location("acquire_rsua", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_row():
    return {"filename": "Invented Validated.zip", "size": 3,
            "content_details": {"size": 3, "sha256_hash": hashlib.sha256(b"abc").hexdigest(),
                                "download_url": "https://data.mendeley.com/public-files/datasets/2jg8vfdmpm/files/fixture/file_downloaded"}}


def fixture_npy_header(*, shape=(2, 256, 256), dtype="|u1", version=1):
    raw = (repr({"descr": dtype, "fortran_order": False, "shape": shape}) + "\n").encode()
    width = "<H" if version == 1 else "<I"
    return b"\x93NUMPY" + bytes((version, 0)) + struct.pack(width, len(raw)) + raw


class RSUAAcquisitionTests(unittest.TestCase):
    def test_download_guard_precedes_network_or_output_creation(self):
        with patch.object(module, "new_atomic_run") as create, patch.object(module, "public_opener") as network:
            with self.assertRaisesRegex(RuntimeError, "explicit_public_download_approval_required"):
                module.run(output_root="invented", run_id="fixture", allow_public_download=False)
        create.assert_not_called()
        network.assert_not_called()

    def test_only_validated_unique_bounded_archive_is_selected(self):
        self.assertEqual(module.select_validated_archive([fixture_row()])["size_bytes"], 3)
        for rows in ([], [fixture_row(), fixture_row()], [{**fixture_row(), "filename": "Annotated.zip"}]):
            with self.assertRaises(ValueError):
                module.select_validated_archive(rows)

    def test_archive_contract_requires_published_hash_size_and_official_url(self):
        for key, value in (("size", module.MAX_DOWNLOAD_BYTES + 1), ("sha256_hash", "invalid"),
                           ("download_url", "https://untrusted.example/anything"),
                           ("download_url", "https://data.mendeley.com/other/file_downloaded")):
            row = fixture_row()
            row["content_details"][key] = value
            with self.assertRaises(ValueError):
                module.select_validated_archive([row])

    def test_https_public_hosts_only_and_proxy_not_used(self):
        for url in ("http://data.mendeley.com/a", "https://data.mendeley.com.evil.example/a",
                    "https://user:password@data.mendeley.com/a", "file:///invented",
                    "https://data.mendeley.com:444/a"):
            self.assertFalse(module.allowed_url(url))
        with patch.object(module, "build_opener") as build:
            module.public_opener()
        self.assertEqual(build.call_args.args[0].proxies, {})

    def test_redirect_outside_allowlist_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "redirect_boundary_rejected"):
            module.PublicRedirect().redirect_request(None, None, 302, "", {}, "https://untrusted.example/file")

    def test_npy_inspection_does_not_read_array_values_or_use_pickle(self):
        for version in (1, 2, 3):
            header = fixture_npy_header(version=version)
            source = io.BytesIO(header + b"invented_pixel_payload_must_not_be_read")
            result = module.read_npy_header(source)
            self.assertEqual(source.tell(), len(header))
            self.assertEqual(result["shape"], [2, 256, 256])
            self.assertFalse(result["array_values_read"])
            self.assertTrue(result["numeric_plain_dtype"])

    def test_object_dtype_marked_unsafe_never_deserialized(self):
        result = module.read_npy_header(io.BytesIO(fixture_npy_header(dtype="|O")))
        self.assertTrue(result["unsafe_or_non_numeric_dtype"])
        self.assertIsNone(result["dtype"])

    def test_bad_npy_headers_refused(self):
        for raw in (b"bad", fixture_npy_header(shape=(True,)), fixture_npy_header(shape=(0, 2)),
                    b"\x93NUMPY\x01\x00" + struct.pack("<H", 20000), fixture_npy_header()[:-1]):
            with self.assertRaises(ValueError):
                module.read_npy_header(io.BytesIO(raw))

    def test_archive_member_names_not_exposed_and_no_extraction(self):
        with tempfile.TemporaryDirectory(prefix="rsua_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            archive_path = Path(folder) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("invented_sensitive_name.npy", fixture_npy_header() + b"fixture")
            result = module.archive_inventory(archive_path)
            self.assertNotIn("invented_sensitive_name", json.dumps(result))
            self.assertFalse(result["archive_extracted"])
            self.assertFalse(result["clinical_class_mapping_verified"])
            self.assertFalse(result["primary_metric_eligible"])

    def test_unsafe_archive_member_paths_rejected(self):
        for name in ("../outside.npy", "/absolute.npy", "folder\\file.npy"):
            with tempfile.TemporaryDirectory(prefix="rsua_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
                archive_path = Path(folder) / "fixture.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(name, fixture_npy_header())
                with self.assertRaisesRegex(ValueError, "archive_member_boundary_rejected"):
                    module.archive_inventory(archive_path)

    def test_download_hash_and_size_checked_and_file_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="rsua_fixture_", dir=module.PROTECTED / "tricompose_v1_2") as folder:
            contract = module.select_validated_archive([fixture_row()])
            source = io.BytesIO(b"abc")
            response = Mock(status=200, headers={"Content-Length": "3"}, read=source.read)
            response.geturl.return_value = contract["download_url"]
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            opener = Mock(open=Mock(return_value=response))
            target = Path(folder) / "fixture.zip"
            result = module.download_verified(opener, contract, target)
            self.assertTrue(result["published_checksum_verified"])
            self.assertEqual(result["size_bytes"], 3)
            with self.assertRaises(FileExistsError):
                module.download_verified(opener, contract, target)

    def test_cli_masks_native_signed_url_or_member_name_errors(self):
        with patch.object(module, "run", side_effect=ValueError("invented_sensitive_value")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(module.main(["--run-id", "fixture", "--allow-public-download"]), 2)
            self.assertNotIn("invented_sensitive_value", output.getvalue())


if __name__ == "__main__":
    unittest.main()
