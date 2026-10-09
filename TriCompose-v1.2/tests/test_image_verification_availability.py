"""Invented metadata only: no patient inputs, report bodies, pixels or models."""
import copy
import csv
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from test_reliability_preview import sidecar
from test_report_gate_availability import gates_for, requests_for

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('image_availability_fixture',
    ROOT/'tools/attach_image_verification_availability.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(state='negative', contract='complete', decision='scope_commit', retained='negative'):
    rows, facts, groups = sidecar()
    gates = gates_for(rows[0], facts, decision=decision, state=retained)
    lookup, grouped = m.reportgate.index_gate(rows, facts, gates)
    rows, facts = m.reportgate.candidate_availability(rows, facts, lookup, grouped)
    record = {'cxr_candidate_id': rows[0]['cxr_candidate_id'], 'cxr_sha256': rows[0]['cxr_sha256'],
        'response_sha256': 'd'*64, 'contract_status': contract,
        'states': dict.fromkeys(m.HEADS, state) if contract == 'complete' else None,
        'failure_reason': None if contract == 'complete' else 'invalid_eight_state_json',
        'independent_clinical_validation': False, 'token_limit_reached': False,
        'input_tokens': 512, 'output_tokens': 90, 'elapsed_seconds': 1.0}
    prediction = {'schema_version': m.imageworker.SCHEMA, 'frozen': True, 'image_only': True,
        'image_prompt_sha256': 'e'*64, 'model_received_ehr_reports_scores_or_candidate_ids': False,
        'records': [record]}
    fact_index = {(f['triple_candidate_id'], f['finding']): f for f in facts}
    comparisons = []
    for gate in gates:
        fact = fact_index[gate['triple_candidate_id'], gate['finding']]
        qwen, xrv_relation, report_relation = m.image_relations(record, fact, gate)
        comparisons.append({**gate, 'imagecheck_cxr_candidate_id': record['cxr_candidate_id'],
            'imagecheck_cxr_sha256': record['cxr_sha256'], 'imagecheck_contract_status': contract,
            'imagecheck_raw_xrv_state': fact['states']['xrv'], 'imagecheck_qwen_state': qwen,
            'imagecheck_xrv_qwen_relation': xrv_relation,
            'imagecheck_retained_report_relation': report_relation,
            'imagecheck_independent_clinical_validation': False, 'imagecheck_primary_metric_eligible': False,
            'imagecheck_confirmed_faulty_modality': None, 'imagecheck_regeneration_authorized': False})
    return rows, facts, groups, prediction, comparisons


def annotate(parts):
    rows, facts, _, prediction, comparisons = parts
    records, lookup, _ = m.index_evidence(rows, facts, prediction, comparisons)
    output, annotated, unique = m.annotate_pool(rows, facts, records, lookup)
    return output, annotated, unique, records, lookup


def request(parts, kind='verify_image_finding', finding='pneumonia', row_index=0, consumers=None):
    rows, facts, _, _, comps = parts
    gates = [{k: c[k] for k in m.reportgate.GATE_FIELDS} for c in comps]
    lookup, _ = m.reportgate.index_gate(rows, facts, gates)
    raw = requests_for(rows[row_index], kind, finding, consumers)
    return m.reportgate.request_availability([raw], lookup,
        {r['triple_candidate_id']: r for r in rows})


def annotate_request(parts, requests):
    rows, facts, unique, records, lookup = annotate(parts)
    return m.annotate_requests(requests, {r['triple_candidate_id']: r for r in rows}, records, lookup, facts)


class ImageAvailabilityTests(unittest.TestCase):
    def test_original_inputs_cells_order_scores_states_and_histories_unchanged(self):
        parts = fixture()
        original = copy.deepcopy(parts)
        rows, facts, _, _, _ = annotate(parts)
        self.assertEqual(parts, original)
        m.preserve(parts[0], rows)
        m.preserve(parts[1], facts)
        old = request(parts)
        output = annotate_request(parts, old)
        m.preserve(old, output)
        self.assertEqual(output[0]['execution_status'], 'not_executed')
        self.assertFalse(output[0]['imageverify_clinically_resolved'])

    def test_shared_image_reused_without_expanding_exact_report_check(self):
        rows, facts, unique, _, _ = annotate(fixture())
        self.assertEqual(len(unique), 14)
        self.assertTrue(all(r['imageverify_status'] == 'checked_clinically_unqualified' for r in rows))
        unscoped = [f for f in facts if f['triple_candidate_id'] == rows[1]['triple_candidate_id']]
        self.assertTrue(all(not f['imageverify_exact_report_comparison_available'] for f in unscoped))
        self.assertTrue(all(f['imageverify_retained_report_relation'] == 'no_exact_report_check'
            for f in unscoped if f['finding'] in m.HEADS))

    def test_hash_alone_never_expands_image_scope(self):
        parts = fixture()
        parts[0][1]['cxr_candidate_id'] = 'invented_other_image_same_hash'
        for fact in parts[1]:
            if fact['triple_candidate_id'] == parts[0][1]['triple_candidate_id']:
                fact['cxr_candidate_id'] = parts[0][1]['cxr_candidate_id']
        rows, facts, _, _, _ = annotate(parts)
        self.assertEqual(rows[1]['imageverify_status'], 'not_checked')
        self.assertIsNone(rows[1]['imageverify_explicit_opposition_unqualified_count'])
        self.assertTrue(all(f['imageverify_qwen_state'] is None for f in facts
            if f['triple_candidate_id'] == rows[1]['triple_candidate_id']))

    def test_missing_duplicate_foreign_or_changed_image_result_refused(self):
        for mode in ('missing', 'duplicate', 'foreign', 'hash'):
            parts = fixture()
            records = parts[3]['records']
            if mode == 'missing': records.clear()
            elif mode == 'duplicate': records.append(copy.deepcopy(records[0]))
            elif mode == 'foreign': records[0]['cxr_candidate_id'] = 'other'
            else: records[0]['cxr_sha256'] = 'f'*64
            with self.assertRaises(ValueError):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_changed_fixed_ehr_anchor_refused(self):
        parts = fixture()
        parts[0][1]['ehr_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'fixed_shared_image_anchor_required'):
            m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_exact_report_id_hash_and_original_raw_state_required(self):
        for key, value in (('report_candidate_id', 'wrong'), ('report_sha256', 'f'*64),
                ('raw_chexbert_state', 'negative'), ('imagecheck_raw_xrv_state', 'negative')):
            parts = fixture()
            parts[4][0][key] = value
            with self.assertRaises(ValueError):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_missing_duplicate_or_foreign_comparison_refused(self):
        for mode in ('missing', 'duplicate', 'foreign'):
            parts = fixture()
            if mode == 'missing': parts[4].pop()
            elif mode == 'duplicate': parts[4][-1] = copy.deepcopy(parts[4][0])
            else: parts[4][0]['triple_candidate_id'] = 'wrong'
            with self.assertRaises(ValueError):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_unknown_never_negative_or_explicit_agreement(self):
        _, facts, unique, _, _ = annotate(fixture(state='unknown'))
        self.assertTrue(all(f['imageverify_qwen_state'] == 'unknown' for f in facts if f['finding'] in m.HEADS))
        self.assertFalse(any(r['xrv_qwen_relation'] == 'explicit_agreement_unqualified' for r in unique))
        self.assertTrue(any(r['xrv_qwen_relation'] == 'single_source_assertion_unqualified' for r in unique))

    def test_uncertainty_never_hard_opposition(self):
        _, _, unique, _, _ = annotate(fixture(state='uncertain'))
        self.assertTrue(all(r['xrv_qwen_relation'] == 'uncertainty_not_comparable'
            for r in unique if r['finding'] in m.HEADS))

    def test_unavailable_response_is_null_not_successful_unknown(self):
        rows, facts, unique, _, _ = annotate(fixture(contract='failed_unavailable'))
        self.assertTrue(all(r['imageverify_status'] == 'image_verifier_unavailable' for r in rows))
        self.assertTrue(all(f['imageverify_qwen_state'] is None for f in facts))
        self.assertEqual(sum(r['xrv_qwen_relation'] == 'image_verifier_unavailable' for r in unique), 8)

    def test_unretained_assertion_cannot_be_matched_to_image(self):
        _, facts, _, _, _ = annotate(fixture(decision='abstain'))
        self.assertTrue(all(f['imageverify_retained_report_relation'] == 'report_assertion_not_retained'
            for f in facts if f['imageverify_exact_report_comparison_available'] and f['finding'] in m.HEADS))

    def test_unsupported_heads_remain_outside_scope(self):
        _, facts, unique, _, _ = annotate(fixture())
        self.assertEqual(sum(r['xrv_qwen_relation'] == 'outside_image_verifier_scope' for r in unique), 6)
        self.assertTrue(all(f['imageverify_qwen_state'] is None and
            f['imageverify_status'] == 'outside_image_verifier_scope'
            for f in facts if f['finding'] not in m.HEADS))

    def test_corrupt_recomputed_relation_or_state_refused(self):
        for key in ('imagecheck_qwen_state', 'imagecheck_xrv_qwen_relation', 'imagecheck_retained_report_relation'):
            parts = fixture()
            parts[4][0][key] = 'fabricated'
            with self.assertRaisesRegex(ValueError, 'unchanged_four_state_relations_required'):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_promoted_truth_or_embedded_body_refused(self):
        for key, value in (('imagecheck_regeneration_authorized', True),
                ('imagecheck_primary_metric_eligible', True), ('imagecheck_confirmed_faulty_modality', 'cxr'),
                ('quote', 'invented extra text')):
            parts = fixture()
            parts[4][0][key] = value
            with self.assertRaises(ValueError):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_prediction_requires_frozen_blind_image_only_interface(self):
        for key, value in (('frozen', False), ('image_only', False),
                ('model_received_ehr_reports_scores_or_candidate_ids', True), ('raw_response', 'invented')):
            parts = fixture()
            parts[3][key] = value
            with self.assertRaisesRegex(ValueError, 'frozen_image_only_prediction_required'):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_complete_response_rejects_missing_extra_invalid_keys_and_token_cap(self):
        for mode in ('missing', 'extra', 'state', 'cap'):
            parts = fixture()
            rec = parts[3]['records'][0]
            if mode == 'missing': rec['states'].pop('edema')
            elif mode == 'extra': rec['states']['device'] = 'unknown'
            elif mode == 'state': rec['states']['edema'] = 'absent'
            else: rec['token_limit_reached'] = True
            with self.assertRaisesRegex(ValueError, 'complete_eight_state_image_contract_required'):
                m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_request_image_finding_reuses_exact_image_for_shared_reports(self):
        parts = fixture()
        ids = [r['triple_candidate_id'] for r in parts[0]]
        output = annotate_request(parts, request(parts, consumers=ids))[0]
        self.assertEqual(output['imageverify_matched_consumer_ids'], ids)
        self.assertEqual(output['imageverify_unchecked_consumer_count'], 0)
        self.assertEqual(output['imageverify_new_model_calls'], 0)

    def test_image_report_request_requires_exact_report_scope_and_reports_partial_coverage(self):
        parts = fixture()
        parts[0][1]['report_sha256'] = parts[0][0]['report_sha256']
        for f in parts[1]:
            if f['triple_candidate_id'] == parts[0][1]['triple_candidate_id']:
                f['artifact_hashes']['report_sha256'] = parts[0][0]['report_sha256']
        ids = [r['triple_candidate_id'] for r in parts[0]]
        output = annotate_request(parts, request(parts, kind='verify_image_report_relation',
            finding='cardiomegaly', consumers=ids))[0]
        self.assertEqual(output['imageverify_status'], 'partial_consumer_check')
        self.assertEqual(output['imageverify_matched_consumer_ids'], ids[:1])
        self.assertEqual(output['imageverify_unchecked_consumer_count'], 1)

    def test_image_evidence_never_fulfills_report_extraction_or_ehr_requests(self):
        parts = fixture()
        for kind in ('verify_report_assertion', 'assess_ehr_radiographic_observability',
                'verify_conditioning_and_ehr_fact_scope'):
            output = annotate_request(parts, request(parts, kind=kind))[0]
            self.assertEqual(output['imageverify_status'], 'not_applicable_request_kind')
            self.assertEqual(output['imageverify_not_applicable_consumer_count'], 1)
            self.assertEqual(output['imageverify_matched_consumer_ids'], [])
            self.assertFalse(output['imageverify_clinically_resolved'])

    def test_request_wrong_dependencies_eligible_truth_and_duplicate_refused(self):
        parts = fixture()
        for mode in ('dependencies', 'truth', 'duplicate'):
            req = request(parts)
            if mode == 'dependencies': req[0]['dependency_hashes']['cxr_sha256'] = 'f'*64
            elif mode == 'truth': req[0]['clinical_truth_established'] = True
            else: req.append(copy.deepcopy(req[0]))
            with self.assertRaises(ValueError):
                annotate_request(parts, req)

    def test_gate_metadata_in_source_fact_cannot_be_changed(self):
        parts = fixture()
        parts[1][0]['reportgate_retained_state'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'original_reportgate_metadata_required'):
            m.index_evidence(parts[0], parts[1], parts[3], parts[4])

    def test_repeat_attach_rejected(self):
        parts = fixture()
        rows, facts, _, records, lookup = annotate(parts)
        with self.assertRaises(ValueError):
            m.annotate_pool(rows, facts, records, lookup)

    def test_deterministic_counts_and_no_model_truth_or_cost_saving_claims(self):
        parts = fixture()
        output = annotate(parts)
        self.assertEqual(output, annotate(parts))
        rows, facts, unique, records, _ = output
        req = annotate_request(parts, request(parts))
        summary = m.summarize(rows, facts, req, unique, records)
        self.assertEqual(summary['unique_supported_image_finding_checks'], 8)
        self.assertEqual(summary['checked_candidate_slots'], 2)
        self.assertEqual(summary['cached_image_run_model_calls'], 1)
        self.assertEqual(summary['new_model_calls'], 0)
        self.assertEqual(summary['clinically_resolved_requests'], 0)
        self.assertIsNone(summary['independent_clinical_accuracy'])
        self.assertFalse(summary['primary_metric_eligible'])
        self.assertFalse(summary['regeneration_authorized'])

    def test_csv_roundtrip_preserves_all_original_cells_and_columns(self):
        parts = fixture()
        rows, _, _, _, _ = annotate(parts)
        csv_rows = list(csv.DictReader(io.StringIO(m.reportgate.csv_text(rows))))
        for original, output in zip(parts[0], csv_rows):
            for key, value in original.items():
                expected = '' if value is None else str(value)
                self.assertEqual(output[key], expected)

    def test_preservation_rejects_changed_order_cell_or_row_count(self):
        original = [{'score': '0.3'}, {'score': '0.6'}]
        for new in (original[::-1], original[:1], [{'score': '0.9'}, original[1]]):
            with self.assertRaises(ValueError):
                m.preserve(original, new)

    def test_no_new_run_when_output_exists_or_not_in_actual_slurm(self):
        with patch.object(m.reportgate, 'require_cpu_slurm', side_effect=RuntimeError), \
                patch.object(m, 'new_atomic_run') as create:
            with self.assertRaises(RuntimeError):
                m.execute(Path('/invented'), 'invented')
            create.assert_not_called()
        with patch.object(m.reportgate, 'require_cpu_slurm'), \
                patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'load_fixed_inputs') as load:
            with self.assertRaises(FileExistsError):
                m.execute(Path('/invented'), 'invented')
            load.assert_not_called()

    def test_failed_load_discards_only_created_temp(self):
        with patch.object(m.reportgate, 'require_cpu_slurm'), \
                patch.object(m, 'new_atomic_run', return_value=(Path('/invented/temp'), Path('/invented/target'))), \
                patch.object(m, 'load_fixed_inputs', side_effect=ValueError), \
                patch.object(m, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):
                m.execute(Path('/invented'), 'invented')
            discard.assert_called_once_with(Path('/invented/temp'))


if __name__ == '__main__':
    unittest.main()
