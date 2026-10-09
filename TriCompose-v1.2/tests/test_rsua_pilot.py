"""Invented BMP metadata and score vectors only; never actual pixels/models."""
import importlib.util
import io
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

path = Path(__file__).resolve().parents[1] / "real_validation/rsua_pilot.py"
spec = importlib.util.spec_from_file_location("rsua_pilot", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def invented_bmp():
    header = bytearray(54)
    header[:2] = b"BM"
    struct.pack_into("<I", header, 2, 60)
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<IiiHHI", header, 14, 40, 256, 256, 1, 8, 0)
    return bytes(header) + b"FIXTURE"


def invented_archive(reverse=False):
    stream = io.BytesIO()
    rows = []
    for label, count in (("Non_Covid", 32), ("Non_Covid_Pneumonia", 53), ("Covid", 207)):
        for role in ("Image", "Mask"):
            for index in range(count):
                # Fix the fixture byte size in the header. No real image exists.
                payload = invented_bmp()[:-1]
                rows.append((f"Validated/{label}/{role}_{label}/{role.casefold()}_{index:04d}.bmp", payload))
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in reversed(rows) if reverse else rows:
            archive.writestr(name, payload)
    stream.seek(0)
    return zipfile.ZipFile(stream)


class RSUAPilotTests(unittest.TestCase):
    def test_role_markers_are_in_parent_and_basename_not_root_title(self):
        image = zipfile.ZipInfo("Validated/Non_Covid_Pneumonia/Image_Pneumonia/image_fixture.bmp")
        mask = zipfile.ZipInfo("Validated/Non_Covid_Pneumonia/Mask_Pneumonia/mask_fixture.bmp")
        self.assertEqual(module.metadata_role(image), ("pneumonia", "image"))
        self.assertEqual(module.metadata_role(mask), ("pneumonia", "mask"))

    def test_inconsistent_role_basename_refused(self):
        wrong = zipfile.ZipInfo("Validated/Non_Covid/Image_Non_Covid/mask_fixture.bmp")
        with self.assertRaisesRegex(ValueError, "role_marker_unrecognized"):
            module.metadata_role(wrong)

    def test_numpy_duplicate_format_not_selected(self):
        duplicate = zipfile.ZipInfo("Validated/Non_Covid/Image_Non_Covid/image_fixture.npy")
        self.assertIsNone(module.metadata_role(duplicate))

    def test_metadata_selection_has_25_per_class_excludes_masks_and_covid(self):
        with invented_archive() as archive:
            rows, counts = module.select_metadata_cohort(archive)
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(row["reference_state"] == "positive" for row in rows), 25)
        self.assertEqual(sum(row["reference_state"] == "negative" for row in rows), 25)
        self.assertEqual(counts["covid_image"], 207)
        self.assertEqual(counts["pneumonia_mask"], 53)
        self.assertTrue(all(set(row) == {"case_id", "member_index", "member_name_sha256", "size_bytes",
                                        "published_class", "reference_state", "bmp_header"} for row in rows))

    def test_selection_name_hashes_are_stable_when_archive_order_changes(self):
        with invented_archive() as first, invented_archive(reverse=True) as second:
            a, _ = module.select_metadata_cohort(first)
            b, _ = module.select_metadata_cohort(second)
        self.assertEqual([row["member_name_sha256"] for row in a], [row["member_name_sha256"] for row in b])

    def test_class_counts_must_match_published_inventory(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("Validated/Non_Covid/Image_Non_Covid/image_fixture.bmp", invented_bmp()[:-1])
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            with self.assertRaisesRegex(ValueError, "class_counts_mismatch"):
                module.select_metadata_cohort(archive)

    def test_pixels_approval_guard_precedes_torch_factory_and_filesystem(self):
        for environment, approved in (({}, True), ({"SLURM_JOB_ID": "fixture"}, False)):
            with patch.dict(module.os.environ, environment, clear=True), \
                    patch.object(module.Path, "resolve") as resolve, \
                    patch.object(module, "FrozenXRVRuntime") as runtime:
                with self.assertRaisesRegex(RuntimeError, "approved_rsua_pixel_slurm_required"):
                    module.evaluate(SimpleNamespace(allow_rsua_pixels=approved))
            resolve.assert_not_called()
            runtime.assert_not_called()

    def test_header_does_not_decode_pixels(self):
        payload = invented_bmp()[:-1]
        result = module.bmp_header(payload[:54], expected_size=len(payload))
        self.assertEqual((result["width"], result["height"], result["bits_per_pixel"]), (256, 256, 8))
        with self.assertRaisesRegex(ValueError, "bmp_protocol_mismatch"):
            module.bmp_header(payload[:54], expected_size=999)

    def test_perfect_inverse_and_tied_classifier_metrics(self):
        rows = [{"reference_state": "positive", "pneumonia_score": 0.9},
                {"reference_state": "negative", "pneumonia_score": 0.1}]
        self.assertEqual(module.summarize_binary(rows, 0.5)["auroc"], 1)
        inverse = [{**row, "pneumonia_score": 1 - row["pneumonia_score"]} for row in rows]
        self.assertEqual(module.summarize_binary(inverse, 0.5)["auroc"], 0)
        tied = [{**row, "pneumonia_score": 0.5} for row in rows]
        result = module.summarize_binary(tied, 0.5)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["average_precision"], 0.5)
        self.assertEqual((result["tp"], result["fp"]), (1, 1))

    def test_unknown_reference_and_nonfinite_scores_refused_not_mapped_negative(self):
        for row in ({"reference_state": "unknown", "pneumonia_score": 0.9},
                    {"reference_state": "positive", "pneumonia_score": float("nan")}):
            with self.assertRaisesRegex(ValueError, "invalid_binary_diagnostic_inputs"):
                module.summarize_binary([row], 0.5)

    def test_single_class_auc_unavailable_and_no_threshold_fitting(self):
        rows = [{"reference_state": "positive", "pneumonia_score": 0.6}]
        result = module.summarize_binary(rows, 0.5)
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(result["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
