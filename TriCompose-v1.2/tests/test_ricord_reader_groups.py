"""Invented study keys and labels only; no downloaded data or network calls."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'real_validation/audit_ricord_reader_groups.py'
spec = importlib.util.spec_from_file_location('audit_ricord_reader_groups', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    groups, rows = [], []
    names = ['Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance', 'Negative for Pneumonia']
    studies = [{'StudyInstanceUID': 'invented_sensitive_' + str(i)} for i in range(4)]
    for index in range(3):
        groups.append({'name': 'invented_reader_metadata', 'labels': [
            {'id': 'label_%s_%s' % (index, j), 'name': name, 'scope': 'STUDY'}
            for j, name in enumerate(names)]})
        rows.extend({'labelId': 'label_%s_%s' % (index, j),
                     'StudyInstanceUID': study['StudyInstanceUID'], 'data': None}
                    for j, study in enumerate(studies))
    return {'labelGroups': groups, 'datasets': [{'studies': studies, 'annotations': rows}]}


class RicordReaderGroupTests(unittest.TestCase):
    def test_derived_counts_and_no_sensitive_values_emitted(self):
        result = module.audit_groups(fixture(), set())
        self.assertNotIn('invented_sensitive', json.dumps(result))
        self.assertNotIn('invented_reader_metadata', json.dumps(result))
        self.assertEqual(result['derived_opacity_positive_studies'], 2)
        self.assertEqual(result['derived_opacity_negative_studies'], 1)
        self.assertEqual(result['derived_opacity_unknown_or_excluded_studies'], 1)
        self.assertFalse(result['official_majority_adjudication_reproduced'])
        self.assertFalse(result['primary_metric_eligible'])

    def test_disagreement_does_not_vote_or_become_negative(self):
        payload = fixture()
        payload['datasets'][0]['annotations'][8]['labelId'] = 'label_2_1'
        result = module.audit_groups(payload, set())
        self.assertEqual(result['three_group_classification_counts']['between_group_disagreement'], 1)
        self.assertEqual(result['derived_opacity_negative_studies'], 1)
        self.assertEqual(result['derived_opacity_positive_studies'], 1)

    def test_withdrawn_studies_excluded(self):
        result = module.audit_groups(fixture(), {'invented_sensitive_0'})
        self.assertEqual(result['active_study_count'], 3)
        self.assertEqual(result['derived_opacity_positive_studies'], 1)
        self.assertEqual(result['withdrawn_studies_excluded'], 1)

    def test_no_comprehensive_reader_imputed_from_sparse_groups(self):
        payload = fixture()
        payload['datasets'][0]['annotations'].pop()
        with self.assertRaisesRegex(ValueError, 'three_comprehensive_classification_groups_required'):
            module.audit_groups(payload, set())

    def test_adjudication_not_added_as_fourth_independent_vote(self):
        payload = fixture()
        payload['labelGroups'].append({'name': 'Adjudication', 'labels': [
            {'id': 'judge_label', 'name': 'Negative for Pneumonia', 'scope': 'STUDY'}]})
        payload['datasets'][0]['annotations'].append({'labelId': 'judge_label',
            'StudyInstanceUID': 'invented_sensitive_0', 'data': None})
        result = module.audit_groups(payload, set())
        self.assertEqual(result['comprehensive_group_indices'], [0, 1, 2])
        self.assertEqual(result['derived_opacity_positive_studies'], 2)

    def test_non_null_data_not_interpreted_as_positive_tag(self):
        payload = fixture()
        payload['datasets'][0]['annotations'][0]['data'] = {'negative': True}
        result = module.audit_groups(payload, set())
        self.assertEqual(result['three_group_classification_counts']['missing_group_classification'], 1)
        self.assertEqual(result['groups'][0]['invalid_classification_annotation_count'], 1)
        self.assertEqual(result['derived_opacity_positive_studies'], 1)
        self.assertEqual(result['derived_opacity_negative_studies'], 1)

    def test_uninterpretable_reads_preserve_reader_availability_not_diagnosis(self):
        payload = fixture()
        payload['labelGroups'][0]['labels'].append(
            {'id': 'unknown_read', 'name': 'invented_uninterpretable', 'scope': 'STUDY'})
        payload['datasets'][0]['annotations'][0]['labelId'] = 'unknown_read'
        result = module.audit_groups(payload, set())
        self.assertEqual(result['comprehensive_group_indices'], [0, 1, 2])
        self.assertEqual(result['groups'][0]['annotated_study_count'], 4)
        self.assertEqual(result['groups'][0]['singleton_classified_studies'], 3)
        self.assertEqual(result['three_group_classification_counts']['missing_group_classification'], 1)

    def test_unknown_annotation_label_rejected(self):
        payload = fixture()
        payload['datasets'][0]['annotations'][0]['labelId'] = 'invented_sensitive_unknown'
        with self.assertRaisesRegex(ValueError, '^undefined_label_reference$'):
            module.audit_groups(payload, set())

    def test_orphan_reference_is_counted_not_silently_joined(self):
        payload = fixture()
        payload['datasets'][0]['annotations'].append({'labelId': 'label_0_0',
            'StudyInstanceUID': 'invented_sensitive_orphan', 'data': None})
        result = module.audit_groups(payload, set())
        self.assertEqual(result['groups'][0]['orphan_annotation_count'], 1)
        self.assertEqual(result['active_study_count'], 4)

    def test_duplicate_study_keys_rejected(self):
        payload = fixture()
        payload['datasets'][0]['studies'].append(payload['datasets'][0]['studies'][0])
        with self.assertRaisesRegex(ValueError, '^duplicate_study_key$'):
            module.audit_groups(payload, set())

    def test_cli_masks_native_error_content(self):
        with patch.object(module, 'run', side_effect=ValueError('invented_sensitive_error')), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(module.main(['--acquisition-root', 'fixture', '--run-id', 'fixture']), 2)
        self.assertNotIn('invented_sensitive', output.getvalue())


if __name__ == '__main__':
    unittest.main()
