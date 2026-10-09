"""Invented metadata/DICOM fixtures only; no live requests or clinical outputs."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'real_validation/acquire_ricord_images.py'
spec = importlib.util.spec_from_file_location('ricord_image_acquisition', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from contracts import write_private_text


def fixture():
    names = ['Typical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'Negative for Pneumonia']
    studies = [{'StudyInstanceUID': '1.2.3.' + str(i + 1)} for i in range(4)]
    groups, annotations = [], []
    for reader in range(3):
        groups.append({'name': 'invented_reader', 'labels': [
            {'id': f'label_{reader}_{i}', 'name': name, 'scope': 'STUDY'} for i, name in enumerate(names)]})
        annotations.extend({'StudyInstanceUID': study['StudyInstanceUID'],
                            'labelId': f'label_{reader}_{i}', 'data': None} for i, study in enumerate(studies))
    payload = {'labelGroups': groups, 'datasets': [{'studies': studies, 'annotations': annotations}]}
    rows = [{'StudyInstanceUID': s['StudyInstanceUID'], 'SeriesInstanceUID': '1.2.4.' + str(i + 1),
             'PatientID': 'invented_sensitive_patient_' + str(i), 'Collection': module.COLLECTION,
             'Modality': 'DX', 'ImageCount': 1, 'FileSize': 1000,
             'LicenseURI': 'https://creativecommons.org/licenses/by-nc/4.0/'} for i, s in enumerate(studies)]
    return payload, rows


def select(payload, rows, quota=2, removed=None):
    studies, labels, _ = module.derived_labels(payload, removed or set())
    return module.select_cases(studies, labels, rows, removed or set(), quota=quota)


class FakeResponse(io.BytesIO):
    def __init__(self, data, url, length=None):
        super().__init__(data)
        self.status, self.url = 200, url
        self.headers = {} if length is None else {'Content-Length': str(length)}

    def geturl(self):
        return self.url


class RicordImageTests(unittest.TestCase):
    def test_deterministic_balanced_selection_private_keys_not_emitted(self):
        payload, rows = fixture()
        first = select(payload, rows)
        self.assertEqual(first, select(payload, rows))
        self.assertEqual(first[1]['selected_class_counts'], {'positive': 2, 'negative': 2})
        serialized = json.dumps(first)
        self.assertNotIn('invented_sensitive', serialized)
        self.assertNotIn('1.2.3.', serialized)
        self.assertNotIn('1.2.4.', serialized)
        self.assertEqual([r['annotation_study_index'] for r in first[0]], [0, 1, 2, 3])

    def test_patient_duplicate_does_not_fill_balanced_quota(self):
        payload, rows = fixture()
        rows[3]['PatientID'] = rows[0]['PatientID']
        with self.assertRaisesRegex(ValueError, 'patient_unique_balanced_quota_not_met'):
            select(payload, rows)

    def test_multi_image_study_not_silently_assigned_to_single_series(self):
        payload, rows = fixture()
        extra = copy.deepcopy(rows[0]); extra['SeriesInstanceUID'] = '1.2.99.1'
        rows.append(extra)
        with self.assertRaisesRegex(ValueError, 'patient_unique_balanced_quota_not_met'):
            select(payload, rows)

    def test_withdrawals_and_atypical_never_fill_quotas(self):
        payload, rows = fixture()
        with self.assertRaisesRegex(ValueError, 'patient_unique_balanced_quota_not_met'):
            select(payload, rows, removed={'1.2.3.1'})
        for group in payload['labelGroups']:
            group['labels'][0]['name'] = 'Atypical Appearance'
        with self.assertRaisesRegex(ValueError, 'patient_unique_balanced_quota_not_met'):
            select(payload, rows)

    def test_disagreement_not_majority_or_negative(self):
        payload, rows = fixture()
        payload['datasets'][0]['annotations'][0]['labelId'] = 'label_0_2'
        _, labels, _ = module.derived_labels(payload, set())
        self.assertNotIn('1.2.3.1', labels)

    def test_no_missing_patient_or_unsupported_modality(self):
        for field, value in [('PatientID', ''), ('Modality', 'CT')]:
            payload, rows = fixture(); rows[0][field] = value
            with self.assertRaisesRegex(ValueError, 'patient_unique_balanced_quota_not_met'):
                select(payload, rows)

    def test_mixed_collection_bad_license_duplicate_series_rejected(self):
        for field, value, reason in [('Collection', 'invented', 'mixed_collection'),
                ('LicenseURI', 'https://invented.org/licenses', 'license_contract'),
                ('SeriesInstanceUID', '1.2.4.2', 'duplicate_series')]:
            payload, rows = fixture(); rows[0][field] = value
            with self.assertRaisesRegex(ValueError, reason):
                select(payload, rows)

    def test_invalid_counts_and_quota_rejected(self):
        for value in (True, -1, 0, '1000', None):
            payload, rows = fixture(); rows[0]['FileSize'] = value
            with self.assertRaisesRegex(ValueError, 'positive_integer_required'):
                select(payload, rows)
        payload, rows = fixture()
        with self.assertRaisesRegex(ValueError, 'quota_rejected'):
            select(payload, rows, quota=26)

    def test_endpoint_allowlist_and_source_format(self):
        for endpoint, params in [('getPatient', {}), ('getSeries', {'Collection': 'other'}),
                ('getSingleImage', {'SeriesInstanceUID': '1.2.3', 'SOPInstanceUID': 'secret\nvalue'})]:
            with self.assertRaises(ValueError): module.PublicClient.url(endpoint, params)

    def test_redirect_denied(self):
        with self.assertRaisesRegex(ValueError, 'redirect_not_authorized'):
            module.annotation.NoRedirect().redirect_request(None, None, 302, None, {}, 'https://elsewhere')

    def test_body_budget_charges_failed_partial_response(self):
        client = module.PublicClient(initial_bytes=module.MAX_BYTES - 4)
        params = {'Collection': module.COLLECTION}
        response = FakeResponse(b'123456789', client.url('getSeries', params))
        with patch.object(client.opener, 'open', return_value=response):
            with self.assertRaises(module.BoundError): client.get('getSeries', params, 20)
        self.assertEqual(client.received, module.MAX_BYTES)

    def test_preflight_oversized_content_not_consumed(self):
        client = module.PublicClient()
        params = {'Collection': module.COLLECTION}
        response = FakeResponse(b'123456789', client.url('getSeries', params), length=9)
        with patch.object(client.opener, 'open', return_value=response):
            with self.assertRaises(module.BoundError): client.get('getSeries', params, 8)
        self.assertEqual(client.received, 0)

    def test_exact_image_file_no_overwrite_and_limit(self):
        client = module.PublicClient()
        params = {'SeriesInstanceUID': '1.2.4', 'SOPInstanceUID': '1.2.5'}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'image.dcm'
            response = FakeResponse(b'1234', client.url('getSingleImage', params), length=4)
            with patch.object(client.opener, 'open', return_value=response):
                self.assertEqual(client.get('getSingleImage', params, 4, target, 4), 4)
            self.assertEqual(target.stat().st_mode & 0o777, 0o660)
            response = FakeResponse(b'1234', client.url('getSingleImage', params), length=4)
            with patch.object(client.opener, 'open', return_value=response):
                with self.assertRaises(FileExistsError): client.get('getSingleImage', params, 4, target, 4)
            self.assertEqual(target.read_bytes(), b'1234')
            client.image_requests = 50
            with self.assertRaises(module.BoundError): client.get('getSingleImage', params, 4, target, 4)

    def test_dependency_pin_and_header_binding_without_pixels(self):
        pd = module.load_pydicom()
        payload, rows = fixture()
        ds = pd.dataset.Dataset(); ds.file_meta = pd.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
        for key in ('StudyInstanceUID', 'SeriesInstanceUID', 'PatientID', 'Modality'):
            setattr(ds, key, rows[0][key])
        ds.SOPInstanceUID = '1.2.5.1'; ds.Rows = 256; ds.Columns = 512
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME1'
        ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        header = module.header_contract(ds, rows[0], '1.2.5.1')
        self.assertTrue(header['membership_verified']); self.assertFalse(header['pixels_decoded'])
        self.assertEqual(header['view'], 'unknown')
        self.assertNotIn('invented_sensitive', json.dumps(header))
        ds.PatientID = 'invented_wrong_patient'
        with self.assertRaisesRegex(ValueError, 'membership_mismatch'):
            module.header_contract(ds, rows[0], '1.2.5.1')

    def test_lateral_multi_frame_rejected(self):
        pd = module.load_pydicom()
        _, rows = fixture()
        ds = pd.dataset.Dataset(); ds.file_meta = pd.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
        for key in ('StudyInstanceUID', 'SeriesInstanceUID', 'PatientID', 'Modality'):
            setattr(ds, key, rows[0][key])
        ds.SOPInstanceUID = '1.2.5.1'; ds.Rows = 256; ds.Columns = 256
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        ds.NumberOfFrames = 2
        with self.assertRaisesRegex(ValueError, 'single_frame'):
            module.header_contract(ds, rows[0], '1.2.5.1')
        ds.NumberOfFrames = 1; ds.ViewPosition = 'LL'
        with self.assertRaisesRegex(ValueError, 'unsupported_view'):
            module.header_contract(ds, rows[0], '1.2.5.1')

    def test_approval_required_before_network_or_writes(self):
        with patch.object(module, 'verified_manifest') as verifier:
            with self.assertRaisesRegex(ValueError, 'explicit_small_acquisition_approval_required'):
                module.prepare(acquisition_root='invented', audit_root='invented', output_root='invented',
                               run_id='invented', approved=False)
        verifier.assert_not_called()

    def test_cli_masks_native_sensitive_error(self):
        with patch.object(module, 'prepare', side_effect=ValueError('invented_sensitive_error')), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(module.main(['prepare', '--run-id', 'fixture']), 2)
        self.assertNotIn('invented_sensitive', output.getvalue())

    def test_prepare_cli_success_return(self):
        target = module.WORKSPACE / 'artifacts/protected/invented_run'
        with patch.object(module, 'prepare', return_value=(target, {'counts': {}, 'expected_dicom_bytes': 1})), \
                patch.object(module, 'sha256_file', return_value='a' * 64), \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(module.main(['prepare', '--run-id', 'fixture']), 0)

    def test_failed_case_not_replaced_and_original_denominator_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan_root = root / 'plan'; plan_root.mkdir()
            staging = root / 'staging'; staging.mkdir(); target = root / 'completed'
            _, rows = fixture()
            cases = [{'case_id': f'case_{i:03d}', 'series_inventory_index': 0,
                      'sop_inventory_index': 0, 'expected_dicom_bytes': 4,
                      'lung_opacity_reference': 'positive' if i < 25 else 'negative'} for i in range(50)]
            module.write_private_json(plan_root / 'plan.json', {
                'source_acquisition_root': '.tmp/invented_source',
                'source_reader_audit_root': '.tmp/invented_reader_audit',
                'source_acquisition_manifest_sha256': 'a' * 64,
                'source_reader_audit_manifest_sha256': 'a' * 64,
                'metadata_response_bytes': 0, 'reference_kind': 'invented_unanimity', 'cases': cases,
                'limits': {'max_images': 50, 'max_combined_response_bytes': module.MAX_BYTES,
                           'max_object_bytes': module.MAX_OBJECT}})
            module.write_private_json(plan_root / 'raw_series_inventory.json', {'rows': rows})
            module.write_private_json(plan_root / 'raw_sop_inventory.json', {
                'responses': [[{'SOPInstanceUID': '1.2.5.1'}]]})

            class FakeClient:
                def __init__(self, initial_bytes=0): self.received = initial_bytes; self.calls = 0
                def get(self, endpoint, params, limit, destination=None, expected=None):
                    self.calls += 1; self.received += 4
                    write_private_text(destination, '1234')
                    if self.calls == 1: raise ValueError('invented_sensitive_failure')
                    return 4

            client = FakeClient()
            def verify(path, schema):
                return (plan_root if schema == module.PLAN_SCHEMA else root / 'invented_source'), {}
            with patch.object(module, 'verified_manifest', side_effect=verify), \
                    patch.object(module, 'sha256_file', return_value='a' * 64), \
                    patch.object(module, 'new_atomic_run', return_value=(staging, target)), \
                    patch.object(module, 'PublicClient', return_value=client), \
                    patch.object(module, 'load_pydicom', return_value=SimpleNamespace(__version__='3.0.2')), \
                    patch.object(module, 'check_header', return_value={'membership_verified': True}), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                _, report = module.acquire(plan_root=plan_root, output_root=root, run_id='invented', approved=True)
            self.assertEqual(client.calls, 50)
            self.assertEqual(report['planned_images'], 50)
            self.assertEqual(report['acquired_images'], 49)
            self.assertEqual(report['failed_images'], 1)
            self.assertEqual(report['cases'][0]['case_id'], 'case_000')
            self.assertEqual(report['cases'][0]['partial_bytes'], 4)
            self.assertFalse(report['image_label_binding_verified'])
            self.assertNotIn('invented_sensitive', output.getvalue())
            self.assertNotIn('invented_sensitive', (target / 'acquisition.json').read_text())


if __name__ == '__main__':
    unittest.main()
