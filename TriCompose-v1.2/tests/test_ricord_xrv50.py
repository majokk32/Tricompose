"""Invented arrays and invented JPEG bitstream only; no real images/models."""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'real_validation/ricord_xrv50.py'
spec = importlib.util.spec_from_file_location('ricord_xrv50_test_module', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
display = module.display


def dataset():
    _, pd = display.dependencies()
    ds = pd.dataset.Dataset(); ds.file_meta = pd.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    ds.Rows = 2; ds.Columns = 2; ds.Modality = 'CR'; ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.RescaleSlope = 1; ds.RescaleIntercept = 0
    ds.WindowCenter = 127.5; ds.WindowWidth = 255; ds.VOILUTFunction = 'LINEAR_EXACT'
    return ds


class RicordDisplayTests(unittest.TestCase):
    def test_metadata_range_not_fitted_image_minmax(self):
        np, _ = display.dependencies(); ds = dataset()
        raw = np.array([[64, 128], [160, 192]], dtype=np.uint8)
        output, trace = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, raw)
        self.assertFalse(trace['observed_minmax_used_for_scaling'])
        self.assertEqual(int(output.min()), 64)
        self.assertEqual(int(output.max()), 192)

    def test_monochrome1_shape_inverts_exactly_once(self):
        np, _ = display.dependencies(); ds = dataset()
        ds.PhotometricInterpretation = 'MONOCHROME1'; ds.PresentationLUTShape = 'INVERSE'
        raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        output, trace = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, 255 - raw)
        self.assertTrue(trace['invert_once']); self.assertEqual(trace['polarity_source'], 'presentation_shape')

    def test_missing_shape_photometric_fallback_same_result(self):
        np, _ = display.dependencies(); ds = dataset(); ds.PhotometricInterpretation = 'MONOCHROME1'
        raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        output, trace = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, 255 - raw)
        self.assertEqual(trace['polarity_source'], 'photometric')

    def test_contradictory_shape_rejected_not_inferred_from_scores(self):
        ds = dataset(); ds.PhotometricInterpretation = 'MONOCHROME1'; ds.PresentationLUTShape = 'IDENTITY'
        with self.assertRaisesRegex(ValueError, 'shape_conflict'): display.presentation_contract(ds)

    def test_modality_rescale_precedes_window(self):
        np, _ = display.dependencies(); ds = dataset()
        ds.RescaleSlope = 2; ds.RescaleIntercept = -100
        ds.WindowCenter = 155; ds.WindowWidth = 510
        raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        output, _ = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, raw)

    def test_signed_storage_window_and_range(self):
        np, _ = display.dependencies(); ds = dataset(); ds.PixelRepresentation = 1
        ds.WindowCenter = -0.5; ds.WindowWidth = 255
        raw = np.array([[-128, -64], [0, 127]], dtype=np.int16)
        output, _ = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, np.array([[0, 64], [128, 255]], dtype=np.uint8))

    def test_voi_lut_preferred_and_nonidentity_curve_preserved(self):
        np, pd = display.dependencies(); ds = dataset()
        item = pd.dataset.Dataset(); item.LUTDescriptor = [4, 0, 8]; item.LUTData = [0, 32, 192, 255]
        ds.VOILUTSequence = pd.sequence.Sequence([item])
        raw = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        output, trace = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, np.array([[0, 32], [192, 255]], dtype=np.uint8))
        self.assertEqual(trace['voi_transform'], 'first_voi_lut')

    def test_first_window_fixed_not_best_scoring_view(self):
        np, _ = display.dependencies(); ds = dataset()
        ds.WindowCenter = [127.5, 64]; ds.WindowWidth = [255, 32]
        raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        output, _ = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, raw)

    def test_padding_stays_black_after_monochrome1_inversion(self):
        np, _ = display.dependencies(); ds = dataset(); ds.PhotometricInterpretation = 'MONOCHROME1'
        ds.PixelPaddingValue = 0
        raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        output, trace = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, np.array([[0, 191], [127, 0]], dtype=np.uint8))
        self.assertEqual(trace['padding_pixels'], 1)

    def test_missing_voi_no_invented_minmax(self):
        ds = dataset(); del ds.WindowWidth
        with self.assertRaisesRegex(ValueError, 'missing_voi_no_minmax_fallback'): display.presentation_contract(ds)

    def test_bad_width_multiplicity_slope_syntax_rejected(self):
        for field, value, reason in [('WindowWidth', 0, 'window_function'),
                ('WindowWidth', [255, 100], 'multiplicity'), ('RescaleSlope', -1, 'nonpositive'),
                ('NumberOfFrames', 2, 'single_frame')]:
            ds = dataset(); setattr(ds, field, value)
            with self.assertRaisesRegex(ValueError, reason): display.presentation_contract(ds)
        ds = dataset(); ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.50'
        with self.assertRaisesRegex(ValueError, 'unapproved_transfer'): display.presentation_contract(ds)

    def test_shape_float_storage_range_constant_rejected(self):
        np, _ = display.dependencies(); ds = dataset()
        for raw, reason in [(np.ones((3, 2), dtype=np.uint8), 'shape'),
                (np.ones((2, 2), dtype=np.float32), 'type'),
                (np.array([[0, 64], [128, 256]], dtype=np.uint16), 'storage_range'),
                (np.ones((2, 2), dtype=np.uint8), 'constant_display')]:
            with self.assertRaisesRegex(ValueError, reason): display.render_array(raw, ds)

    def test_nonintegral_rescaled_lut_indices_not_silently_rounded(self):
        np, pd = display.dependencies(); ds = dataset(); ds.RescaleSlope = 0.5
        item = pd.dataset.Dataset(); item.LUTDescriptor = [4, 0, 8]; item.LUTData = [0, 32, 192, 255]
        ds.VOILUTSequence = pd.sequence.Sequence([item])
        with self.assertRaisesRegex(ValueError, 'nonintegral_voi_lut_input'):
            display.render_array(np.array([[0, 1], [2, 3]], dtype=np.uint8), ds)

    def test_sigmoid_official_function_finite(self):
        np, _ = display.dependencies(); ds = dataset(); ds.VOILUTFunction = 'SIGMOID'
        output, _ = display.render_array(np.array([[0, 64], [128, 255]], dtype=np.uint8), ds)
        self.assertEqual(output.dtype, np.uint8); self.assertTrue(np.isfinite(output).all())
        self.assertGreater(int(output.max()), int(output.min()))

    def test_invented_jpeg_lossless_2x2_decoder_roundtrip(self):
        np, pd = display.dependencies(); ds = dataset()
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.70'
        # Hand-built SOF3/SV1 stream: known four pixels, no patient/test dataset.
        jpeg = bytes.fromhex('ffd8 ffc4 0016 00 00030000000000000000000000000000 000102 '
            'ffc3 000b 08 0002 0002 01 01 11 00 ffda 0008 01 01 00 01 00 00 1d3f ffd9')
        from pydicom.encaps import encapsulate
        ds.PixelData = encapsulate([jpeg])
        raw = pd.pixels.pixel_array(ds, decoding_plugin='pylibjpeg')
        np.testing.assert_array_equal(raw, np.array([[128, 129], [130, 131]], dtype=np.uint8))
        output, _ = display.render_array(raw, ds)
        np.testing.assert_array_equal(output, raw)


class RicordMetricGuardTests(unittest.TestCase):
    def test_perfect_and_tied_metrics(self):
        perfect = module.binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        self.assertEqual(perfect['auroc'], 1); self.assertEqual(perfect['average_precision'], 1)
        self.assertEqual(perfect['balanced_accuracy'], 1)
        tied = module.binary_metrics([0, 0, 1, 1], [0.5] * 4, 0.5)
        self.assertEqual(tied['auroc'], 0.5); self.assertEqual(tied['average_precision'], 0.5)

    def test_metrics_invalid_unknown_and_nonfinite_rejected(self):
        for labels, scores in [([0, 2], [0.1, 0.9]), ([0, 1], [0.1, float('nan')]),
                ([0, 1], [0.1]), ([False, True], [0.1, 0.9])]:
            with self.assertRaises(ValueError): module.binary_metrics(labels, scores, 0.5)

    def test_wilson_and_missing_class(self):
        self.assertIsNone(module.wilson(0, 0))
        low, high = module.wilson(25, 25)
        self.assertAlmostEqual(low, 0.86680775, places=7); self.assertEqual(high, 1.0)
        result = module.binary_metrics([1, 1], [0.8, 0.9], 0.5)
        self.assertIsNone(result['specificity']); self.assertIsNone(result['balanced_accuracy'])

    def test_patient_bootstrap_deterministic(self):
        args = ([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9])
        result = module.bootstrap(*args, count=20, seed=11)
        self.assertEqual(result, module.bootstrap(*args, count=20, seed=11))
        self.assertEqual(result['unit'], 'patient_one_image_class_stratified')

    def test_guard_blocks_before_inputs_and_model_imports(self):
        args = argparse.Namespace(allow_ricord_pixels_and_xrv=True)
        with patch.dict(os.environ, {}, clear=True), patch.object(module, 'input_contract') as inputs:
            with self.assertRaisesRegex(RuntimeError, 'approval_required'): module.evaluate(args)
        inputs.assert_not_called()
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'invented', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
            with self.assertRaises(RuntimeError): module.guard(False)
            module.guard(True)

    def test_cli_masks_native_error_details(self):
        with patch.object(module, 'prepare', side_effect=ValueError('invented_sensitive_error')), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(module.main(['prepare', '--run-id', 'fixture']), 2)
        self.assertNotIn('invented_sensitive', output.getvalue())


if __name__ == '__main__':
    unittest.main()
