"""Invented metadata only; no real/synthetic report text or model execution."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reportcheck_progress_test',
    ROOT/'tools/diagnose_report_verifier_progress.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fact(finding='cardiomegaly', left='positive', right='positive', candidate='invented_000'):
    row = dict.fromkeys(m.FACT_FIELDS)
    row.update(finding=finding, raw_chexbert_state=left, qwen_assertion_state=right,
        qwen_contract_status='complete' if finding in m.HEADS else 'outside_scope_inventory',
        triple_candidate_id=candidate, report_candidate_id='invented_report_000', report_sha256='a'*64,
        independent_clinical_validation=False, regeneration_authorized=False, selection_changed=False)
    return row


def request(kind='verify_report_assertion', consumers=None, finding='cardiomegaly'):
    row = dict.fromkeys(m.REQUEST_FIELDS)
    row.update(request_id='invented_request_000', request_kind=kind, finding=finding,
        consumer_candidate_ids=consumers or ['invented_000'], dependency_hashes={'report_sha256': 'a'*64},
        execution_status='not_executed', clinical_truth_established=False, model_execution_allowed=False)
    return row


class ReportVerifierProgressTests(unittest.TestCase):
    def test_all_sixteen_state_pairs_have_semantic_categories(self):
        values = {(a, b): m.state_transition(a, b) for a in m.STATES for b in m.STATES}
        self.assertEqual(sum(v == 'hard_polarity_flip' for v in values.values()), 2)
        self.assertEqual(values['unknown', 'negative'], 'unmentioned_to_determinate')
        self.assertEqual(values['uncertain', 'negative'], 'uncertain_to_determinate')
        self.assertEqual(values['negative', 'uncertain'], 'determinate_to_uncertain')
        self.assertEqual(values['unknown', 'unknown'], 'same_unmentioned_state')
        self.assertEqual(values['positive', 'unknown'], 'annotated_to_unmentioned')

    def test_invalid_state_is_not_normalized_to_unknown(self):
        with self.assertRaises(ValueError):
            m.state_transition('missing', 'negative')

    def test_manual_table_is_aggregate_not_per_report_rows(self):
        matrix = {a: dict.fromkeys(m.STATES, 0) for a in m.STATES}
        matrix['positive']['negative'] = 1
        matrix['unknown']['unknown'] = 2
        summary = {'paired_comparison': {name: {finding: {'confusion_matrix': copy.deepcopy(matrix),
            'checks': 3} for finding in m.HEADS} for name in ('chexbert', 'qwen_v2')}}
        rows = m.manual_transitions(summary)
        self.assertEqual(len(rows), 128)
        self.assertEqual(sum(r['count'] for r in rows), 24)
        self.assertFalse(any('item_id' in r or 'report_path' in r for r in rows))

    def test_invalid_manual_counts_and_matrix_shape_rejected(self):
        matrix = {a: dict.fromkeys(m.STATES, 0) for a in m.STATES}
        summary = {'paired_comparison': {name: {finding: {'confusion_matrix': copy.deepcopy(matrix),
            'checks': 0} for finding in m.HEADS} for name in ('chexbert', 'qwen_v2')}}
        for value in (-1, True, 0.5):
            bad = copy.deepcopy(summary)
            bad['paired_comparison']['qwen_v2']['cardiomegaly']['confusion_matrix']['positive']['positive'] = value
            with self.assertRaises(ValueError):
                m.manual_transitions(bad)
        bad = copy.deepcopy(summary)
        bad['paired_comparison']['qwen_v2']['cardiomegaly']['confusion_matrix'].pop('unknown')
        with self.assertRaises(ValueError):
            m.manual_transitions(bad)

    def test_uncertain_pair_never_counted_as_explicit_agreement(self):
        self.assertEqual(m.fact_status(fact(left='uncertain', right='uncertain')), 'uncertainty_not_contradiction')

    def test_csv_serializes_nested_progress_without_model_execution(self):
        self.assertIn('not_executed', m.csv_text([request()], ('request_id', 'execution_status')))

    def test_synthetic_opposition_is_not_clinical_fault(self):
        self.assertEqual(m.fact_status(fact(right='negative')), 'explicit_polarity_disagreement_unqualified')
        self.assertEqual(m.fact_status(fact(left='unknown', right='negative')), 'single_extractor_assertion_unqualified')
        self.assertEqual(m.fact_status(fact(left='unknown', right='unknown')), 'both_unmentioned')
        self.assertEqual(m.fact_status(fact(right='uncertain')), 'uncertainty_not_contradiction')

    def test_contract_failure_is_not_successful_unknown(self):
        row = fact(left='unknown', right='unknown')
        row['qwen_contract_status'] = 'failed_unavailable'
        self.assertEqual(m.fact_status(row), 'verifier_unavailable')

    def test_outside_scope_is_retained_not_filled(self):
        self.assertEqual(m.fact_status(fact(finding='edema')), 'outside_verifier_scope')

    def test_complete_inventory_and_every_original_cell_preserved(self):
        rows = [fact(finding=f) for f in m.CHEXPERT_FINDINGS]
        before = copy.deepcopy(rows)
        annotated, candidates, lookup = m.annotate_facts(rows)
        self.assertEqual(rows, before)
        for old, new in zip(rows, annotated):
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertFalse(new['reportcheck_clinical_truth_established'])
        self.assertEqual(len(lookup), 14)
        self.assertEqual(candidates[0]['outside_verifier_scope'], 10)
        self.assertIsNone(candidates[0]['clinical_selection_score'])

    def test_duplicate_missing_head_and_lineage_mismatch_rejected(self):
        rows = [fact(finding=f) for f in m.CHEXPERT_FINDINGS]
        for bad in (rows[:-1], rows+rows[:1]):
            with self.assertRaises(ValueError):
                m.annotate_facts(bad)
        rows[1]['report_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            m.annotate_facts(rows)

    def test_text_fields_and_claimed_truth_rejected(self):
        for key, value in (('quote', 'invented text'), ('independent_clinical_validation', True)):
            rows = [fact(finding=f) for f in m.CHEXPERT_FINDINGS]
            rows[0][key] = value
            with self.assertRaises(ValueError):
                m.annotate_facts(rows)

    def test_review_appends_progress_without_rewriting_execution_history(self):
        original = request()
        review = {**fact(), 'reportcheck_status': m.fact_status(fact())}
        output = m.request_progress([original], {('invented_000', 'cardiomegaly'): review})[0]
        self.assertEqual({k: output[k] for k in original}, original)
        self.assertEqual(output['reportcheck_progress'], 'technical_review_complete_clinically_unqualified')
        self.assertFalse(output['reportcheck_clinically_resolved'])
        self.assertEqual(output['reportcheck_new_model_calls'], 0)

    def test_same_hash_outside_fixed_candidate_ids_not_covered(self):
        output = m.request_progress([request(consumers=['invented_other'])],
            {('invented_000', 'cardiomegaly'): fact()})[0]
        self.assertEqual(output['reportcheck_progress'], 'not_covered_by_this_report_check')

    def test_image_relation_not_fulfilled_by_report_only_check(self):
        output = m.request_progress([request(kind='verify_image_report_relation')],
            {('invented_000', 'cardiomegaly'): fact()})[0]
        self.assertEqual(output['reportcheck_matched_consumer_ids'], [])

    def test_exact_dependency_hash_required(self):
        row = request()
        row['dependency_hashes']['report_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'exact_report_dependency_hash_required'):
            m.request_progress([row], {('invented_000', 'cardiomegaly'): fact()})

    def test_outside_scope_and_failed_review_requests_remain_unresolved(self):
        for finding, status in (('edema', 'outside_verifier_scope'), ('cardiomegaly', 'verifier_unavailable')):
            row = fact(finding=finding)
            row['reportcheck_status'] = status
            output = m.request_progress([request(finding=finding)], {('invented_000', finding): row})[0]
            self.assertEqual(output['reportcheck_progress'], status)
            self.assertFalse(output['reportcheck_clinically_resolved'])

    def test_repeated_requests_and_nonpreview_requests_rejected(self):
        for rows in ([request(), request()], [{**request(), 'execution_status': 'complete'}]):
            with self.assertRaises(ValueError):
                m.request_progress(rows, {})

    def test_existing_run_guard_before_consumed_inputs_opened(self):
        with patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'load_fixed_inputs') as inputs:
            with self.assertRaises(FileExistsError):
                m.execute('/invented', 'invented_001')
            inputs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
