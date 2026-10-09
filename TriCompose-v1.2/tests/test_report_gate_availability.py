"""Invented metadata only; no patient inputs, reports, pixels or models."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from test_reliability_preview import sidecar

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_gate_availability_fixture',
    ROOT/'tools/attach_report_gate_availability.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def gates_for(row, facts, *, decision='scope_commit', state='negative'):
    old = {(f['triple_candidate_id'], f['finding']): f for f in facts}
    result = []
    for finding in m.CHEXPERT_FINDINGS:
        supported = finding in m.HEADS
        gate = dict.fromkeys(m.GATE_FIELDS)
        gate.update(triple_candidate_id=row['triple_candidate_id'], finding=finding,
            report_candidate_id=row['report_candidate_id'], report_sha256=row['report_sha256'],
            raw_chexbert_state=old[row['triple_candidate_id'], finding]['states']['chexbert'],
            qwen_assertion_state=state if supported else 'unknown',
            qwen_contract_status='complete' if supported else 'outside_scope_inventory',
            independent_clinical_validation=False, regeneration_authorized=False, selection_changed=False,
            scopegate_independent_clinical_validation=False, scopegate_regeneration_authorized=False,
            scopegate_decision=decision if supported else 'outside_verifier_scope',
            scopegate_retained_state=state if supported and decision == 'scope_commit' else None,
            scopegate_scope_verified=supported and decision == 'scope_commit', scopegate_evidence=[],
            scopegate_version='invented_version', scopegate_reason='invented_reason')
        result.append(gate)
    return result


def requests_for(row, kind='verify_report_assertion', finding='cardiomegaly', consumers=None):
    return {'request_id': 'invented_request', 'request_kind': kind, 'finding': finding,
        'case_id': row['case_id'], 'consumer_candidate_ids': consumers or [row['triple_candidate_id']],
        'dependency_hashes': {k: row[k] for k in m.REQUEST_DEPENDENCIES[kind]},
        'execution_status': 'not_executed', 'clinical_truth_established': False,
        'model_execution_allowed': False, 'reportcheck_clinically_resolved': False,
        'reportcheck_new_model_calls': 0, 'reportcheck_progress': 'invented_technical_progress',
        'estimated_model_calls': None, 'estimated_gpu_seconds': None}


class ReportGateAvailabilityTests(unittest.TestCase):
    def inputs(self, **kwargs):
        rows, facts, _ = sidecar(**kwargs)
        return rows, facts, gates_for(rows[0], facts)

    def test_all_original_cells_order_and_inputs_preserved(self):
        rows, facts, gates = self.inputs()
        before = copy.deepcopy((rows, facts, gates))
        lookup, grouped = m.index_gate(rows, facts, gates)
        attached, annotated = m.candidate_availability(rows, facts, lookup, grouped)
        self.assertEqual((rows, facts, gates), before)
        for original, output in ((rows, attached), (facts, annotated)):
            for old, new in zip(original, output):
                self.assertEqual({k: new[k] for k in old}, old)

    def test_unchecked_candidate_counts_null_not_zero(self):
        rows, facts, gates = self.inputs()
        lookup, grouped = m.index_gate(rows, facts, gates)
        output, annotated = m.candidate_availability(rows, facts, lookup, grouped)
        self.assertEqual(output[1]['reportgate_status'], 'not_checked')
        self.assertIsNone(output[1]['reportgate_scope_commit_count'])
        self.assertEqual(output[1]['reportgate_inventory_rows_unchecked'], 14)
        self.assertTrue(all(f['reportgate_retained_state'] is None
            for f in annotated if f['triple_candidate_id'] == rows[1]['triple_candidate_id']))

    def test_same_report_hash_does_not_expand_checked_candidate_scope(self):
        rows, facts, gates = self.inputs(experts=('positive', 'positive'))
        rows[1]['report_sha256'] = rows[0]['report_sha256']
        for fact in facts:
            if fact['triple_candidate_id'] == rows[1]['triple_candidate_id']:
                fact['artifact_hashes']['report_sha256'] = rows[0]['report_sha256']
        lookup, grouped = m.index_gate(rows, facts, gates)
        output, _ = m.candidate_availability(rows, facts, lookup, grouped)
        self.assertEqual(output[1]['reportgate_status'], 'not_checked')

    def test_explicit_counts_and_outside_inventory(self):
        rows, facts, gates = self.inputs(ehr='positive')
        lookup, grouped = m.index_gate(rows, facts, gates)
        output, _ = m.candidate_availability(rows, facts, lookup, grouped)
        self.assertEqual(output[0]['reportgate_supported_head_checks'], 4)
        self.assertEqual(output[0]['reportgate_scope_commit_count'], 4)
        self.assertEqual(output[0]['reportgate_outside_verifier_scope_count'], 10)
        self.assertEqual(output[0]['reportgate_direct_ehr_reference_facts'], 1)
        self.assertFalse(output[0]['reportgate_independent_clinical_validation'])
        self.assertFalse(output[0]['reportgate_regeneration_authorized'])

    def test_unknown_and_abstention_not_corrected_predictions(self):
        for decision, state in (('no_model_assertion', 'unknown'), ('abstain', 'positive')):
            rows, facts, _ = self.inputs()
            lookup, grouped = m.index_gate(rows, facts, gates_for(rows[0], facts, decision=decision, state=state))
            _, annotated = m.candidate_availability(rows, facts, lookup, grouped)
            self.assertTrue(all(f['reportgate_retained_state'] is None for f in annotated))

    def test_uncertain_commit_not_promoted_to_positive(self):
        rows, facts, _ = self.inputs()
        lookup, _ = m.index_gate(rows, facts, gates_for(rows[0], facts, state='uncertain'))
        self.assertEqual(lookup[rows[0]['triple_candidate_id'], 'cardiomegaly']['scopegate_retained_state'], 'uncertain')

    def test_failed_verifier_distinct_from_successful_unknown(self):
        rows, facts, _ = self.inputs()
        gates = gates_for(rows[0], facts, decision='verifier_unavailable', state=None)
        for gate in gates:
            if gate['finding'] in m.HEADS:
                gate['qwen_contract_status'] = 'failed_unavailable'
        lookup, grouped = m.index_gate(rows, facts, gates)
        output, _ = m.candidate_availability(rows, facts, lookup, grouped)
        self.assertEqual(output[0]['reportgate_verifier_unavailable_count'], 4)

    def test_missing_duplicate_and_foreign_gate_refused(self):
        rows, facts, gates = self.inputs()
        foreign = copy.deepcopy(gates)
        foreign[0]['triple_candidate_id'] = 'invented_other'
        for bad in (gates[:-1], gates+gates[:1], foreign):
            with self.assertRaises(ValueError):
                m.index_gate(rows, facts, bad)

    def test_report_identity_hash_and_raw_state_must_match(self):
        rows, facts, gates = self.inputs()
        for key, value in (('report_sha256', 'b'*64), ('report_candidate_id', 'invented_other'),
                ('raw_chexbert_state', 'positive')):
            bad = copy.deepcopy(gates)
            bad[0][key] = value
            with self.assertRaisesRegex(ValueError, 'exact_gate_report_and_raw_state_required'):
                m.index_gate(rows, facts, bad)

    def test_no_flip_or_unknown_commit(self):
        rows, facts, gates = self.inputs()
        index = next(i for i, g in enumerate(gates) if g['finding'] in m.HEADS)
        for state in ('positive', 'unknown', None):
            bad = copy.deepcopy(gates)
            bad[index]['scopegate_retained_state'] = state
            with self.assertRaises(ValueError):
                m.index_gate(rows, facts, bad)

    def test_claimed_truth_or_text_field_refused(self):
        rows, facts, gates = self.inputs()
        for key, value in (('scopegate_independent_clinical_validation', True),
                ('regeneration_authorized', True), ('clinical_selection_score', 1),
                ('quote', 'invented text')):
            bad = copy.deepcopy(gates)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                m.index_gate(rows, facts, bad)

    def test_evidence_quote_not_exported(self):
        rows, facts, gates = self.inputs()
        gates[0]['scopegate_evidence'] = [{'quote': 'invented text'}]
        with self.assertRaisesRegex(ValueError, 'quote_free_evidence_fields_required'):
            m.index_gate(rows, facts, gates)

    def test_requests_keep_history_and_unresolved_status(self):
        rows, facts, gates = self.inputs()
        lookup, _ = m.index_gate(rows, facts, gates)
        request = requests_for(rows[0])
        before = copy.deepcopy(request)
        output = m.request_availability([request], lookup, {r['triple_candidate_id']: r for r in rows})[0]
        self.assertEqual(request, before)
        self.assertEqual({k: output[k] for k in request}, request)
        self.assertEqual(output['reportgate_status'], 'scope_commit')
        self.assertFalse(output['reportgate_clinically_resolved'])
        self.assertEqual(output['reportgate_new_model_calls'], 0)

    def test_report_only_gate_cannot_cover_image_edges(self):
        rows, facts, gates = self.inputs()
        lookup, _ = m.index_gate(rows, facts, gates)
        for kind in m.REQUEST_DEPENDENCIES:
            if kind != 'verify_report_assertion':
                output = m.request_availability([requests_for(rows[0], kind=kind)], lookup,
                    {r['triple_candidate_id']: r for r in rows})[0]
                self.assertEqual(output['reportgate_status'], 'not_checked')
                self.assertEqual(output['reportgate_matched_consumer_ids'], [])

    def test_unchecked_consumer_not_covered_by_matching_hash(self):
        rows, facts, gates = self.inputs(experts=('positive', 'positive'))
        rows[1]['report_sha256'] = rows[0]['report_sha256']
        lookup, _ = m.index_gate(rows, facts, gates)
        request = requests_for(rows[0], consumers=[r['triple_candidate_id'] for r in rows])
        output = m.request_availability([request], lookup, {r['triple_candidate_id']: r for r in rows})[0]
        self.assertEqual(output['reportgate_status'], 'partial_consumer_check')
        self.assertEqual(output['reportgate_unchecked_consumer_count'], 1)

    def test_mixed_checked_consumers_not_all_committed(self):
        rows, facts, gates = self.inputs()
        rows[1]['report_sha256'] = rows[0]['report_sha256']
        for fact in facts:
            if fact['triple_candidate_id'] == rows[1]['triple_candidate_id']:
                fact['artifact_hashes']['report_sha256'] = rows[0]['report_sha256']
        gates += gates_for(rows[1], facts, decision='abstain', state='positive')
        lookup, _ = m.index_gate(rows, facts, gates)
        output = m.request_availability([requests_for(rows[0], consumers=[r['triple_candidate_id'] for r in rows])],
            lookup, {r['triple_candidate_id']: r for r in rows})[0]
        self.assertEqual(output['reportgate_status'], 'mixed_consumer_availability')

    def test_request_hash_case_foreign_consumer_and_duplicates_refused(self):
        rows, facts, gates = self.inputs()
        lookup, _ = m.index_gate(rows, facts, gates)
        candidates = {r['triple_candidate_id']: r for r in rows}
        original = requests_for(rows[0])
        bad = []
        for key, value in (('case_id', 'invented_other'), ('consumer_candidate_ids', ['invented_other']),
                ('execution_status', 'complete'), ('clinical_truth_established', True)):
            bad.append([{**original, key: value}])
        bad.append([{**original, 'dependency_hashes': {'report_sha256': 'a'*64}}])
        bad.append([original, original])
        for requests in bad:
            with self.assertRaises(ValueError):
                m.request_availability(requests, lookup, candidates)

    def test_outside_scope_request_stays_outside_not_negative(self):
        rows, facts, gates = self.inputs()
        lookup, _ = m.index_gate(rows, facts, gates)
        output = m.request_availability([requests_for(rows[0], finding='edema')], lookup,
            {r['triple_candidate_id']: r for r in rows})[0]
        self.assertEqual(output['reportgate_status'], 'outside_verifier_scope')
        self.assertFalse(output['reportgate_clinically_resolved'])

    def test_existing_run_refused_before_metadata_load(self):
        with patch.object(m, 'require_cpu_slurm'), \
                patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'load_fixed_inputs') as load:
            with self.assertRaises(FileExistsError):
                m.execute('/invented', 'invented')
            load.assert_not_called()

    def test_actual_slurm_guard_not_env_alone(self):
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': '42'}), \
                patch.object(m.Path, 'read_text', return_value='no allocation'):
            with self.assertRaises(RuntimeError):
                m.require_cpu_slurm()

    def test_duplicate_attachment_refused(self):
        rows, facts, gates = self.inputs()
        lookup, grouped = m.index_gate(rows, facts, gates)
        rows[0]['reportgate_status'] = 'forged'
        with self.assertRaises(ValueError):
            m.candidate_availability(rows, facts, lookup, grouped)


if __name__ == '__main__':
    unittest.main()
