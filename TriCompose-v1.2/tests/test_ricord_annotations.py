"""Invented metadata only. No real annotation, pixel, model or network call."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'real_validation/acquire_ricord_annotations.py'
spec = importlib.util.spec_from_file_location('acquire_ricord_annotations', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {'labelGroups': [{'labels': [
        {'id': 'label_a', 'name': 'Typical Appearance', 'scope': 'STUDY'},
        {'id': 'label_b', 'name': 'Negative for Pneumonia', 'scope': 'STUDY'},
        {'id': 'label_c', 'name': 'Mild', 'scope': 'STUDY'},
        {'id': 'label_d', 'name': 'invented_sensitive_free_text', 'scope': 'invented_sensitive_scope'},
    ]}], 'datasets': [{'studies': [
        {'StudyInstanceUID': 'invented_sensitive_study_a', 'series': [{'images': [{}, {}]}],
         'annotations': [{'labelId': 'label_a', 'StudyInstanceUID': 'invented_sensitive_study_a'},
                         {'labelId': 'label_c', 'StudyInstanceUID': 'invented_sensitive_study_a'}]},
        {'StudyInstanceUID': 'invented_sensitive_study_b', 'series': [{'images': [{}]}],
         'annotations': [{'labelId': 'label_b', 'StudyInstanceUID': 'invented_sensitive_study_b'}]},
        {'StudyInstanceUID': 'invented_sensitive_study_c', 'annotations': []},
    ]}]}


class RicordAnnotationTests(unittest.TestCase):
    def test_approval_required_before_network_or_writes(self):
        with patch.object(module, 'new_atomic_run') as create, patch.object(module, 'public_opener') as net:
            with self.assertRaises(RuntimeError):
                module.run(output_root='invented', run_id='fixture', allow_annotation_download=False)
        create.assert_not_called()
        net.assert_not_called()

    def test_schema_and_aggregate_output_contains_no_patient_values(self):
        result = module.inventory(fixture(), set())
        self.assertNotIn('invented_sensitive', json.dumps(result))
        self.assertEqual(result['active_study_count'], 3)
        self.assertEqual(result['active_image_metadata_count'], 3)
        self.assertEqual(result['study_classification_counts'],
                         {'negative_for_pneumonia': 1, 'typical_appearance': 1, 'unknown': 1})
        self.assertFalse(result['primary_metric_eligible'])
        self.assertFalse(result['image_label_binding_verified'])

    def test_missing_labels_never_become_negative(self):
        payload = fixture()
        payload['datasets'][0]['studies'][0]['annotations'] = []
        result = module.inventory(payload, set())
        self.assertEqual(result['study_classification_counts']['unknown'], 2)
        self.assertEqual(result['study_classification_counts']['negative_for_pneumonia'], 1)

    def test_conflicting_study_classification_not_majority_vote(self):
        payload = fixture()
        rows = payload['datasets'][0]['studies'][0]['annotations']
        rows.extend([{'labelId': 'label_b', 'StudyInstanceUID': 'invented_sensitive_study_a'}] * 3)
        result = module.inventory(payload, set())
        self.assertEqual(result['study_classification_counts']['conflicting'], 1)
        self.assertNotIn('typical_appearance', result['study_classification_counts'])

    def test_withdrawn_study_excluded_from_counts(self):
        result = module.inventory(fixture(), {'invented_sensitive_study_a'})
        self.assertEqual(result['withdrawn_studies_excluded'], 1)
        self.assertEqual(result['active_study_count'], 2)
        self.assertEqual(result['active_image_metadata_count'], 1)
        self.assertNotIn('typical_appearance', result['recognized_annotation_counts'])

    def test_unknown_labels_are_not_forced_into_known_classes(self):
        payload = fixture()
        payload['datasets'][0]['studies'][2]['annotations'].append(
            {'labelId': 'label_d', 'StudyInstanceUID': 'invented_sensitive_study_c'})
        result = module.inventory(payload, set())
        self.assertEqual(result['unrecognized_label_definition_count'], 1)
        self.assertEqual(result['unrecognized_label_annotation_count'], 1)
        self.assertEqual(result['study_classification_counts']['unknown'], 1)

    def test_duplicates_rejected_without_echoing_keys(self):
        with self.assertRaisesRegex(ValueError, '^duplicate_json_key$'):
            json.loads('{"invented_sensitive_key": 1, "invented_sensitive_key": 2}',
                       object_pairs_hook=module.reject_duplicate_keys)
        payload = fixture()
        payload['datasets'][0]['studies'].append(payload['datasets'][0]['studies'][0])
        with self.assertRaisesRegex(ValueError, '^duplicate_study_key$'):
            module.inventory(payload, set())

    def test_no_proxy_or_redirect(self):
        with patch.object(module, 'build_opener') as build:
            module.public_opener()
        self.assertEqual(build.call_args.args[0].proxies, {})
        with self.assertRaisesRegex(ValueError, 'redirect_not_authorized'):
            module.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://untrusted.example')

    def test_no_arbitrary_download_and_bounded_bytes(self):
        with self.assertRaisesRegex(ValueError, 'nonallowlisted_url'):
            module.public_get(Mock(), 'https://untrusted.example/', 3)
        data = io.BytesIO(b'abcd')
        response = Mock(status=200, headers={}, read=data.read)
        response.geturl.return_value = module.ANNOTATIONS
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with self.assertRaisesRegex(ValueError, 'public_download_bound_exceeded'):
            module.public_get(Mock(open=Mock(return_value=response)), module.ANNOTATIONS, 3)

    def test_catalog_requires_official_link_license_and_withdrawal_notice(self):
        html = ('MIDRC-RICORD-1C <a href="' + module.ANNOTATIONS + '">Download</a>'
                '<a href="https://creativecommons.org/licenses/by-nc/4.0/">License</a>'
                'subsequently removed 1.2.826.0.1.3680043.10.474.111 '
                '1.2.826.0.1.3680043.10.474.222')
        self.assertEqual(len(module.catalog_contract(html.encode())), 2)
        with self.assertRaises(ValueError):
            module.catalog_contract(html.replace('by-nc/4.0', 'by/4.0').encode())

    def test_unknown_schema_field_names_never_echoed(self):
        payload = fixture()
        payload['invented_sensitive_dictionary_key'] = {'invented_sensitive_key': 'body'}
        self.assertNotIn('invented_sensitive', json.dumps(module.inventory(payload, set())))

    def test_cli_hides_native_exception_content(self):
        with patch.object(module, 'run', side_effect=ValueError('invented_sensitive_native_error')), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(module.main(['--run-id', 'fixture', '--allow-annotation-download']), 2)
        self.assertNotIn('invented_sensitive', output.getvalue())


if __name__ == '__main__':
    unittest.main()
