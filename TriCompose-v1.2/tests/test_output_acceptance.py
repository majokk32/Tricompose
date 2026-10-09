"""Invented fact signatures/charged traces; no clinical/model artifacts."""
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('acceptance_fixture', ROOT / 'tools/apply_output_acceptance.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
from tricompose_v12.invariant_verification import PROVENANCE


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidate(image='sana', report='maira2', ehr='positive', xrv='positive', chex='negative', quality=.8):
    cid = image + '_' + report
    values = {key: {f: 'unknown' for f in w.guarded.legacy.FINDINGS} for key in ('ehr', 'xrv', 'chexbert')}
    for key, state in (('ehr', ehr), ('xrv', xrv), ('chexbert', chex)):
        values[key]['edema'] = state
    lineage = {'ehr_sha256': h('invented_ehr'), 'ehr_facts_sha256': h('invented_facts'),
        'cxr_candidate_id': image, 'cxr_sha256': h(image), 'cxr_model_id': image, 'cxr_seed': 0,
        'report_candidate_id': cid, 'report_sha256': h(cid), 'report_model_id': report}
    hashes = {k: lineage[k] for k in ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')}
    facts = []
    for f in w.guarded.legacy.FINDINGS:
        facts.append({'finding': f, 'case_id': 'invented_case', 'triple_candidate_id': cid,
            'cxr_candidate_id': image, 'report_candidate_id': cid, 'artifact_hashes': dict(hashes),
            'evidence_id': h(cid + f), 'states': {k: v[f] for k, v in values.items()},
            'source_categories': ['diagnoses'] if values['ehr'][f] != 'unknown' else [],
            'weak_context_promoted': False, 'clinical_truth_verified': False, 'provenance_resolution': PROVENANCE})
    return reseal({'score_record': {'schema_version': 'tricompose-edge-specific-selection-v1.1',
        'case_id': 'invented_case', 'triple_candidate_id': cid, 'lineage': lineage,
        'scoring': {'selection': {'hard_gate_failure_count': 0},
            'modality_quality': {'cxr_basic_validity_pass': True, 'report_structure_quality_score_0_1': quality},
            'cost': {'known_runtime_seconds': 1.}}}, 'facts': facts})


def finding(c, name, **states):
    f = next(f for f in c['facts'] if f['finding'] == name)
    f['states'].update(states)
    f['source_categories'] = ['diagnoses'] if f['states']['ehr'] != 'unknown' else []


def reseal(c):
    vals = {k: {f['finding']: f['states'][k] for f in c['facts']} for k in ('ehr', 'xrv', 'chexbert')}
    edges = w.guarded.legacy.metric_counts(vals['ehr'], vals['xrv'], vals['chexbert'])
    for edge in edges.values():
        edge['support_recall'] = edge['support_count'] / edge['known_reference_fact_count'] if edge['known_reference_fact_count'] else None
    s = c['score_record']['scoring']; s['edge_metrics'] = edges
    s['clinical_totals'] = {'total_hard_contradiction_count': sum(e['contradiction_count'] for e in edges.values()),
        'total_direct_support_count': sum(e['support_count'] for e in edges.values()),
        'ehr_direct_support_count': edges['ehr_cxr']['support_count'] + edges['ehr_report']['support_count'],
        'total_known_reference_fact_count': sum(e['known_reference_fact_count'] for e in edges.values())}
    for f in c['facts']:
        states = f['states']
        f['relations'] = {'raw': {e: w.guarded.legacy.relation(states[a], states[b]) for e, a, b in
            (('ehr_cxr', 'ehr', 'xrv'), ('ehr_report', 'ehr', 'chexbert'), ('cxr_report', 'xrv', 'chexbert'))}}
    return c


def report_pair():
    return candidate(), candidate(report='cxrmate_single', chex='positive')


def image_pair():
    return candidate(xrv='negative'), candidate('roentgen', 'cxrmate_single', xrv='positive', chex='positive')


class OutputAcceptanceTests(unittest.TestCase):
    def test_unchanged_is_unverified_not_a_repair(self):
        a = candidate(); r = w.assess_output(a, a, [a])
        self.assertEqual(r['status'], 'unchanged_unverified')
        self.assertFalse(r['proposal_passes_proxy_preservation'])
        self.assertFalse(r['clinical_acceptance'])

    def test_strict_same_image_report_improvement_can_be_retained(self):
        a, b = report_pair(); r = w.assess_output(a, b, [a, b])
        self.assertEqual(r['selected_candidate_id'], b['score_record']['triple_candidate_id'])
        self.assertEqual(r['status'], 'proxy_preserving_report_change_unverified')
        self.assertFalse(r['clinical_repair_success'])

    def test_unknown_ehr_does_not_prevent_image_report_only_proxy_check(self):
        a = candidate(ehr='unknown'); b = candidate(report='cxrmate_single', ehr='unknown', chex='positive')
        r = w.assess_output(a, b, [a, b])
        self.assertTrue(r['proposal_passes_proxy_preservation'])
        self.assertFalse(r['clinical_acceptance'])

    def test_uncertain_ehr_cannot_supply_missing_image_constraint(self):
        a = candidate(ehr='uncertain', xrv='unknown'); b = candidate(report='cxrmate_single', ehr='uncertain', xrv='unknown', chex='positive')
        r = w.assess_output(a, b, [a, b])
        self.assertFalse(r['proposal_passes_proxy_preservation'])

    def test_unknown_or_uncertain_cannot_silence_existing_opposition(self):
        for state in ('unknown', 'uncertain'):
            a, b = report_pair(); finding(b, 'edema', chexbert=state); reseal(b)
            r = w.assess_output(a, b, [a, b])
            self.assertTrue(r['proposal_vetoed'])
            self.assertEqual(r['selected_candidate_id'], a['score_record']['triple_candidate_id'])

    def test_equal_positive_count_with_different_ids_loses_old_support(self):
        a, b = report_pair()
        for c in (a, b): finding(c, 'pneumonia', xrv='positive')
        finding(a, 'edema', chexbert='positive'); finding(a, 'pneumonia', chexbert='unknown')
        finding(b, 'edema', chexbert='unknown'); finding(b, 'pneumonia', chexbert='positive')
        for c in (a, b): reseal(c)
        self.assertTrue(w.assess_output(a, b, [a, b])['proposal_vetoed'])

    def test_new_different_opposition_cannot_be_paid_for_by_old_reduction(self):
        a, b = report_pair()
        for c in (a, b): finding(c, 'pneumonia', xrv='positive', chexbert='positive')
        finding(b, 'pneumonia', chexbert='negative')
        for c in (a, b): reseal(c)
        self.assertTrue(w.assess_output(a, b, [a, b])['proposal_vetoed'])

    def test_lower_or_missing_structure_vetoes_proposal(self):
        for value in (.7, None):
            a, b = report_pair(); b['score_record']['scoring']['modality_quality']['report_structure_quality_score_0_1'] = value
            self.assertTrue(w.assess_output(a, b, [a, b])['proposal_vetoed'])

    def test_better_structure_alone_is_not_fact_gain(self):
        a, b = report_pair(); finding(a, 'edema', chexbert='positive'); reseal(a)
        b['score_record']['scoring']['modality_quality']['report_structure_quality_score_0_1'] = .9
        self.assertFalse(w.assess_output(a, b, [a, b])['proposal_passes_proxy_preservation'])

    def test_cross_image_gain_needs_preservation_and_branch_basis(self):
        a, b = image_pair(); r = w.assess_output(a, b, [a, b])
        self.assertEqual(r['status'], 'proxy_preserving_image_change_unverified')
        self.assertFalse(r['regeneration_authorized'])

    def test_missing_image_evidence_gain_does_not_authorize_image_change(self):
        a = candidate(xrv='unknown', chex='positive'); b = candidate('roentgen', chex='positive')
        r = w.assess_output(a, b, [a, b])
        self.assertTrue(r['proposal_vetoed'])
        self.assertIn('no_direct_observed_image_branch_basis', r['rejection_reason_codes'])

    def test_cross_image_without_direct_ehr_anchor_vetoed(self):
        a = candidate(ehr='unknown'); b = candidate('roentgen', ehr='unknown', chex='positive')
        self.assertTrue(w.assess_output(a, b, [a, b])['proposal_vetoed'])

    def test_unobserved_report_winner_cannot_enter_image_reference(self):
        a, b = image_pair()
        finding(a, 'pneumonia', xrv='positive', chexbert='unknown')
        finding(b, 'pneumonia', xrv='negative', chexbert='negative')
        for c in (a, b): reseal(c)
        r = w.assess_output(a, b, [a, b])
        self.assertTrue(r['proposal_passes_proxy_preservation'])
        self.assertEqual(r['observed_initial_report_reference_inventory'], [a['score_record']['triple_candidate_id']])

    def test_observed_stronger_report_reference_must_be_preserved_too(self):
        a, b = image_pair(); ref = candidate(report='llavarad', xrv='negative')
        for c in (a, ref): finding(c, 'pneumonia', xrv='positive', chexbert='unknown')
        finding(ref, 'pneumonia', chexbert='positive')
        finding(b, 'pneumonia', xrv='negative', chexbert='negative')
        for c in (a, ref, b): reseal(c)
        r = w.assess_output(a, b, [a, ref, b])
        self.assertEqual(r['observed_initial_report_reference_candidate_id'], ref['score_record']['triple_candidate_id'])
        self.assertTrue(r['proposal_vetoed'])

    def test_proposal_must_be_observed_and_identical_to_observation(self):
        a, b = report_pair()
        with self.assertRaises(ValueError): w.assess_output(a, b, [a])
        changed = copy.deepcopy(b); finding(changed, 'edema', chexbert='unknown'); reseal(changed)
        with self.assertRaises(ValueError): w.assess_output(a, changed, [a, b])

    def test_baseline_must_be_first_observed_not_a_hidden_later_reference(self):
        a, b = report_pair()
        with self.assertRaises(ValueError): w.assess_output(b, a, [a, b])

    def test_duplicate_observations_refused(self):
        a = candidate()
        with self.assertRaises(ValueError): w.assess_output(a, a, [a, a])

    def test_changed_fixed_ehr_provenance_or_states_refused(self):
        for mode in ('state', 'source'):
            a, b = report_pair()
            if mode == 'state': finding(b, 'edema', ehr='negative'); reseal(b)
            else: b['facts'][3]['source_categories'] = ['other']
            with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_same_image_cannot_change_classifier_between_reports(self):
        a, b = report_pair(); finding(b, 'edema', xrv='negative'); reseal(b)
        with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_shared_image_bytes_require_shared_classifier_even_with_new_id(self):
        a, b = image_pair(); digest = a['score_record']['lineage']['cxr_sha256']
        b['score_record']['lineage']['cxr_sha256'] = digest
        for f in b['facts']: f['artifact_hashes']['cxr_sha256'] = digest
        with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_shared_report_bytes_cannot_have_different_frozen_labels(self):
        a, b = report_pair(); digest = a['score_record']['lineage']['report_sha256']
        b['score_record']['lineage']['report_sha256'] = digest
        for f in b['facts']: f['artifact_hashes']['report_sha256'] = digest
        with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_weak_context_cannot_be_promoted_to_direct_fact(self):
        a, b = report_pair(); b['facts'][3]['weak_context_promoted'] = True
        with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_invalid_initial_fallback_returns_null_not_accepted_invalid_output(self):
        a, b = report_pair(); a['score_record']['scoring']['selection']['hard_gate_failure_count'] = 1
        r = w.assess_output(a, b, [a, b])
        self.assertIsNone(r['selected_candidate_id'])
        self.assertEqual(r['status'], 'unresolved_no_metadata_eligible_output')
        self.assertFalse(r['clinical_acceptance'])

    def test_invalid_unchanged_image_is_not_falsely_accepted(self):
        a = candidate(); a['score_record']['scoring']['modality_quality']['cxr_basic_validity_pass'] = False
        self.assertIsNone(w.assess_output(a, a, [a])['selected_candidate_id'])

    def test_missing_parent_proposal_is_separate_unresolved_fallback(self):
        a = candidate(); r = w.assess_output(a, None, [a])
        self.assertEqual(r['status'], 'unresolved_missing_proposal_fixed_retained')
        self.assertEqual(r['selected_candidate_id'], a['score_record']['triple_candidate_id'])

    def test_bad_metadata_type_or_nan_contract_error_not_clinical_failure(self):
        for value in (float('nan'), True, '0.8'):
            a, b = report_pair(); b['score_record']['scoring']['modality_quality']['report_structure_quality_score_0_1'] = value
            with self.assertRaises(ValueError): w.assess_output(a, b, [a, b])

    def test_secondary_scores_winner_flags_and_ranks_are_stripped(self):
        a, b = report_pair(); expected = w.assess_output(a, b, [a, b])
        for c in (a, b):
            c['score_record']['secondary_scores'] = {'biovil': .999}
            c['score_record']['old_bank_rank'] = 1
            c['score_record']['scoring']['selection']['selected'] = True
        view = w.controller_view(b)
        self.assertNotIn('secondary_scores', view['score_record'])
        self.assertNotIn('old_bank_rank', view['score_record'])
        self.assertEqual(view['score_record']['scoring']['selection'], {'hard_gate_failure_count': 0})
        self.assertEqual(expected, w.assess_output(a, b, [a, b]))

    def test_gate_deterministic_and_input_immutable(self):
        a, b = image_pair(); before = copy.deepcopy((a, b))
        self.assertEqual(w.assess_output(a, b, [a, b]), w.assess_output(a, b, [a, b]))
        self.assertEqual((a, b), before)

    def test_gate_has_no_endpoint_or_full_bank_argument(self):
        self.assertEqual(list(inspect.signature(w.assess_output).parameters), ['baseline', 'proposed', 'observed_candidates'])

    def test_veto_keeps_parent_acquisition_charges_and_hash_binding(self):
        a, b = report_pair(); finding(b, 'edema', chexbert='unknown'); reseal(b)
        ids = [c['score_record']['triple_candidate_id'] for c in (a, b)]
        candidates = {c['score_record']['triple_candidate_id']: {'case_id': 'invented_case',
            'ehr_sha256': c['score_record']['lineage']['ehr_sha256'], 'cxr_candidate_id': 'sana',
            'report_model_id': c['score_record']['lineage']['report_model_id']} for c in (a, b)}
        grid = {('sana', 0, 'maira2'): a, ('sana', 0, 'cxrmate_single'): b}
        parent = {'case_id': 'invented_case', 'method': 'report_only_static', 'selected_candidate_id': ids[1],
            'selected_ehr_sha256': a['score_record']['lineage']['ehr_sha256'], 'model_call_budget': 8,
            'input_ehr_assessment_scope': 'explicit_fact_proxy', 'observed_candidates': 2, 'observed_images': 1,
            'simulated_calls': {'cxr_generator': 1, 'xrv': 1, 'report_generator': 2, 'chexbert': 2},
            'simulated_model_calls': 6, 'terminal_reason': 'fixture_budget', 'actual_regeneration_executed': False,
            'clinical_fault_confirmed': False, 'clinical_acceptance': False, 'action_trace': [
                {'step': i, 'observed_candidate_id': ids[i], 'request_slot': ['sana', 0, model],
                    'charged_model_calls': 4 if i == 0 else 2, 'cumulative_model_calls': 4 if i == 0 else 6}
                for i, model in enumerate(('maira2', 'cxrmate_single'))]}
        before = copy.deepcopy(parent); r = w.apply_trial(parent, grid, candidates)
        self.assertEqual(r['simulated_model_calls'], 6)
        self.assertEqual(r['simulated_calls'], parent['simulated_calls'])
        self.assertEqual(r['action_trace'], parent['action_trace'])
        self.assertEqual(r['selected_candidate_id'], ids[0])
        self.assertEqual(r['parent_trial_sha256'], w.guarded.baseline.digest(parent))
        self.assertEqual(parent, before)

    def test_policy_does_not_authorize_clinical_acceptance_or_live_inference(self):
        p = w.policy_description()
        self.assertFalse(p['clinical_acceptance'])
        self.assertFalse(p['regeneration_authorized'])
        self.assertFalse(p['installed_into_live_gpu_controller'])
        self.assertTrue(p['already_incurred_parent_costs_retained'])

    def test_slurm_guard_precedes_cache_loading(self):
        with patch.dict(w.guarded.baseline.os.environ, {}, clear=True), patch.object(w, 'load') as load:
            with self.assertRaises(RuntimeError): w.execute(Path('/invented'), 'opaque')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
