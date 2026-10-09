"""Authored metadata keys and poison fields; no report/numeric access in cohort."""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'real_validation/radevalx_published_adapter_v2.py'
spec=importlib.util.spec_from_file_location('test_radevalx_published_adapter_v2',PATH)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class MetadataRow(dict):
    def __getitem__(self,key):
        if key!='report_id':
            raise AssertionError('non_metadata_eligibility_access')
        return super().__getitem__(key)


def fixture():
    scores=[MetadataRow({key:('invented_'+str(i) if key=='report_id' else 'unused')
        for key in adapter.METRIC_FIELDS}) for i in range(590)]
    refs=[MetadataRow({key:('invented_'+str(i+200) if key=='report_id' else 'unused')
        for key in adapter.ANNOTATION_FIELDS}) for i in range(100)]
    return scores,refs


class CohortMetadataTests(unittest.TestCase):
    def test_all_100_annotations_and_490_unannotated_preserved_in_counts(self):
        scores,refs=fixture()
        selected,inventory=adapter.cohort(scores,refs,refs)
        self.assertEqual(len(selected),100)
        self.assertEqual(inventory['published_rows_without_expert_annotation'],490)
        self.assertEqual(selected[0]['report_id'],'invented_200')
        self.assertFalse(inventory['numeric_score_error_or_report_fields_used_for_eligibility'])

    def test_reference_order_does_not_select_or_reorder_score_examples(self):
        scores,refs=fixture()
        self.assertEqual(adapter.cohort(scores,refs,refs),adapter.cohort(scores,refs[::-1],refs[::-1]))

    def test_duplicate_reference_key_refused(self):
        scores,refs=fixture()
        refs[1]['report_id']=refs[0]['report_id']
        with self.assertRaises(ValueError):
            adapter.cohort(scores,refs,refs)

    def test_missing_score_key_refused_not_case_dropped(self):
        scores,refs=fixture()
        scores[200]['report_id']='invented_unmatched'
        with self.assertRaises(ValueError):
            adapter.cohort(scores,refs,refs)

    def test_reference_significance_key_sets_must_match(self):
        scores,refs=fixture()
        other=[MetadataRow(row) for row in refs]
        other[0]['report_id']='invented_100'
        with self.assertRaises(ValueError):
            adapter.cohort(scores,refs,other)

    def test_unknown_source_inventory_is_not_silently_relaxed(self):
        scores,refs=fixture()
        with self.assertRaises(ValueError):
            adapter.cohort(scores[:-1],refs,refs)


if __name__=='__main__':
    unittest.main()
