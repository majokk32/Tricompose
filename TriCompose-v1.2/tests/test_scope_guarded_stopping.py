"""Invented fourteen-state facts and traces only; no model/patient artifacts."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scope_stop_fixture', ROOT / 'tools/scope_guarded_stopping.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

POLICY = {'image_order': [['chexgenbench_sana', 0], ['chexgenbench_pixart', 0], ['roentgen_v2', 0]],
    'report_order': ['maira2', 'cxrmate_single', 'llavarad', 'chexagent2']}


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def candidate(image, report, ehr='unknown', xrv='positive', chex='negative', valid=True):
    cid = image + '_' + report
    vectors = {name: {f: 'unknown' for f in w.legacy.FINDINGS} for name in ('ehr', 'xrv', 'chexbert')}
    for name, state in (('ehr', ehr), ('xrv', xrv), ('chexbert', chex)):
        vectors[name]['pneumonia'] = state
    edges = w.legacy.metric_counts(vectors['ehr'], vectors['xrv'], vectors['chexbert'])
    for edge in edges.values():
        edge['support_recall'] = edge['support_count'] / edge['known_reference_fact_count'] if edge['known_reference_fact_count'] else None
    totals = {'total_hard_contradiction_count': sum(v['contradiction_count'] for v in edges.values()),
        'total_direct_support_count': sum(v['support_count'] for v in edges.values()),
        'total_known_reference_fact_count': sum(v['known_reference_fact_count'] for v in edges.values()),
        'ehr_direct_support_count': edges['ehr_cxr']['support_count'] + edges['ehr_report']['support_count']}
    row = {'case_id': 'invented_case', 'triple_candidate_id': cid,
        'lineage': {'ehr_sha256': 'a' * 64, 'ehr_facts_sha256': 'b' * 64,
            'cxr_sha256': hashed(image), 'report_sha256': hashed(cid),
            'cxr_candidate_id': image, 'report_candidate_id': cid, 'cxr_model_id': image,
            'cxr_seed': 0, 'report_model_id': report},
        'scoring': {'selection': {'hard_gate_failure_count': int(not valid)}, 'clinical_totals': totals,
            'edge_metrics': edges, 'modality_quality': {'cxr_basic_validity_pass': valid,
                'report_structure_quality_score_0_1': .7}, 'cost': {'known_runtime_seconds': 1.}}}
    facts = []
    for finding in w.legacy.FINDINGS:
        states = {name: vector[finding] for name, vector in vectors.items()}
        facts.append({'finding': finding, 'evidence_id': hashed(cid + finding), 'states': states,
            'relations': {'raw': {'ehr_cxr': w.legacy.relation(states['ehr'], states['xrv']),
                'ehr_report': w.legacy.relation(states['ehr'], states['chexbert']),
                'cxr_report': w.legacy.relation(states['xrv'], states['chexbert'])}},
            'source_categories': ['diagnoses'] if states['ehr'] != 'unknown' else [],
            'weak_context_promoted': False})
    return {'score_record': row, 'facts': facts}


def bank(ehr='unknown', image_states=('positive', 'positive', 'positive'), report_state='negative', invalid_first=False):
    return {(model, 0, report): candidate(model, report, ehr, image_states[i], report_state,
        valid=not (invalid_first and i == 0))
        for i, (model, _) in enumerate(POLICY['image_order']) for report in POLICY['report_order']}


class ScopeGuardedStoppingTests(unittest.TestCase):
    def test_unknown_ehr_does_not_allow_report_votes_to_condemn_image(self):
        b = bank()
        result = w.replay(b, POLICY, 30)
        self.assertEqual(result['observed_images'], 1)
        self.assertEqual(result['observed_candidates'], 4)
        self.assertEqual(result['simulated_model_calls'], 10)
        self.assertEqual(result['terminal_reason'], 'stop_unresolved_no_direct_ehr_image_reference')
        self.assertTrue(result['decision_trace'][-1]['vetoed_image_branch'])
        self.assertFalse(result['clinical_acceptance'])

    def test_direct_ehr_image_opposition_allows_only_proxy_exploration(self):
        b = bank(ehr='positive', image_states=('negative', 'positive', 'positive'))
        result = w.replay(b, POLICY, 30)
        self.assertEqual(result['observed_images'], 2)
        self.assertEqual(result['simulated_model_calls'], 14)
        self.assertEqual(result['terminal_reason'], 'stop_unresolved_no_direct_ehr_image_opposition')
        allowed = [d['basis'] for d in result['decision_trace'] if d['decision'] == 'allow_cached_image_branch_unqualified']
        self.assertEqual(len(allowed), 1)
        self.assertEqual(allowed[0]['known_direct_ehr_facts'], 1)
        self.assertEqual(len(allowed[0]['explicit_proxy_opposition_evidence_ids']), 1)
        self.assertFalse(allowed[0]['confirmed_image_fault'])
        self.assertFalse(allowed[0]['clinical_regeneration_authorized'])

    def test_uncertain_ehr_not_an_explicit_opposition(self):
        b = bank(ehr='uncertain')
        result = w.replay(b, POLICY, 30)
        self.assertEqual(result['observed_images'], 1)
        self.assertEqual(result['terminal_reason'], 'stop_unresolved_no_direct_ehr_image_reference')

    def test_missing_image_evidence_can_try_reports_but_not_new_image(self):
        b = bank(ehr='positive', image_states=('unknown', 'negative', 'negative'), report_state='unknown')
        result = w.replay(b, POLICY, 30)
        self.assertEqual(result['observed_images'], 1)
        self.assertEqual(result['observed_candidates'], 4)
        self.assertEqual(result['terminal_reason'], 'stop_unresolved_no_comparable_ehr_image_evidence')

    def test_supported_ehr_image_report_conflict_only_changes_report(self):
        result = w.replay(bank(ehr='positive'), POLICY, 30)
        self.assertEqual(result['observed_images'], 1)
        self.assertTrue(all(a['action'] != 'regenerate_cxr' for a in result['action_trace']))
        self.assertEqual(result['terminal_reason'], 'stop_unresolved_no_direct_ehr_image_opposition')

    def test_basic_invalid_image_allows_branch_without_ehr_anchor(self):
        b = bank(report_state='positive', invalid_first=True)
        result = w.replay(b, POLICY, 30)
        self.assertEqual(result['observed_images'], 2)
        self.assertEqual(result['simulated_model_calls'], 8)
        self.assertEqual(result['terminal_reason'], 'stop_proxy_satisfied_not_clinical_acceptance')
        self.assertEqual(result['decision_trace'][0]['basis']['reason'], 'basic_invalid_image_metadata')
        self.assertFalse(result['clinical_acceptance'])

    def test_proxy_stop_with_missing_ehr_is_not_complete_triple_success(self):
        result = w.replay(bank(report_state='positive'), POLICY, 30)
        self.assertEqual(result['simulated_model_calls'], 4)
        self.assertEqual(result['input_ehr_assessment_scope'], 'no_direct_comparable_ehr_facts')
        self.assertTrue(result['selected_proxy_stop_conditions_met'])
        self.assertFalse(result['clinical_fault_confirmed'])
        self.assertFalse(result['actual_regeneration_executed'])
        self.assertFalse(result['clinical_acceptance'])

    def test_fully_agreeing_proxy_stops_but_still_no_clinical_acceptance(self):
        result = w.replay(bank(ehr='positive', report_state='positive'), POLICY, 30)
        self.assertEqual(result['simulated_model_calls'], 4)
        self.assertEqual(result['terminal_reason'], 'stop_proxy_satisfied_not_clinical_acceptance')
        self.assertFalse(result['clinical_acceptance'])

    def test_budget_checked_before_exposing_or_charging_new_candidate(self):
        with patch.object(w, 'controller_view', wraps=w.controller_view) as observe:
            result = w.replay(bank(), POLICY, 4)
        self.assertEqual(observe.call_count, 1)
        self.assertEqual(result['simulated_model_calls'], 4)
        self.assertEqual(result['observed_candidates'], 1)
        self.assertEqual(result['terminal_reason'], 'model_call_budget_exhausted')

    def test_too_small_budget_keeps_null_selection_and_no_partial_charge(self):
        with patch.object(w, 'controller_view', side_effect=AssertionError('no observation at this cap')):
            result = w.replay(bank(), POLICY, 3)
        self.assertEqual(result['simulated_model_calls'], 0)
        self.assertIsNone(result['selected_candidate_id'])
        self.assertEqual(result['action_trace'], [])

    def test_same_image_reuse_charged_once(self):
        result = w.replay(bank(), POLICY, 30)
        self.assertEqual([a['charged_model_calls'] for a in result['action_trace']], [4, 2, 2, 2])
        self.assertEqual(result['simulated_calls'], {'cxr_generator': 1, 'xrv': 1, 'report_generator': 4, 'chexbert': 4})

    def test_known_fact_without_provenance_refused(self):
        c = candidate('sana', 'maira', ehr='positive', xrv='negative')
        for f in c['facts']:
            f['source_categories'] = []
        with self.assertRaises(ValueError):
            w.image_branch_basis(c)

    def test_weak_context_cannot_be_promoted(self):
        c = candidate('sana', 'maira', ehr='positive', xrv='negative')
        c['facts'][0]['weak_context_promoted'] = True
        with self.assertRaises(ValueError):
            w.image_branch_basis(c)

    def test_forged_opposition_from_unknown_or_support_refused(self):
        for ehr, xrv in (('unknown', 'negative'), ('positive', 'positive'), ('uncertain', 'negative')):
            c = candidate('sana', 'maira', ehr=ehr, xrv=xrv)
            next(f for f in c['facts'] if f['finding'] == 'pneumonia')['relations']['raw']['ehr_cxr'] = 'opposition'
            with self.assertRaises(ValueError):
                w.image_branch_basis(c)

    def test_changed_ehr_state_hash_or_provenance_refused(self):
        for mode in ('state', 'hash', 'source'):
            b = bank(ehr='positive')
            target = b['chexgenbench_sana', 0, 'cxrmate_single']
            if mode == 'hash':
                target['score_record']['lineage']['ehr_sha256'] = 'c' * 64
            elif mode == 'state':
                next(f for f in target['facts'] if f['finding'] == 'pneumonia')['states']['ehr'] = 'negative'
            else:
                next(f for f in target['facts'] if f['finding'] == 'pneumonia')['source_categories'] = ['other']
            with self.assertRaises(ValueError):
                w.replay(b, POLICY, 30)

    def test_report_switch_cannot_change_image_hash_or_classifier(self):
        for mode in ('hash', 'state'):
            b = bank()
            target = b['chexgenbench_sana', 0, 'cxrmate_single']
            if mode == 'hash':
                target['score_record']['lineage']['cxr_sha256'] = 'd' * 64
            else:
                next(f for f in target['facts'] if f['finding'] == 'pneumonia')['states']['xrv'] = 'negative'
            with self.assertRaises(ValueError):
                w.replay(b, POLICY, 30)

    def test_secondary_scores_and_old_winner_metadata_not_exposed(self):
        c = candidate('sana', 'maira')
        c['score_record']['secondary_scores'] = {'biovil': .999}
        c['score_record']['old_bank_rank'] = 1
        c['score_record']['scoring']['selection']['selected'] = True
        view = w.controller_view(c)
        self.assertNotIn('secondary_scores', view['score_record'])
        self.assertNotIn('old_bank_rank', view['score_record'])
        self.assertEqual(view['score_record']['scoring']['selection'], {'hard_gate_failure_count': 0})

    def test_alternate_scores_and_unseen_candidates_do_not_change_route(self):
        a = bank()
        b = copy.deepcopy(a)
        for slot, c in b.items():
            c['score_record']['secondary_scores'] = {'biovil': -1 if slot[0] == 'chexgenbench_sana' else 1}
            if slot[0] != 'chexgenbench_sana':
                c['score_record']['scoring']['clinical_totals']['total_hard_contradiction_count'] = 0
        self.assertEqual(w.replay(a, POLICY, 30), w.replay(b, POLICY, 30))

    def test_returned_candidate_is_best_eligible_observed_not_global_winner(self):
        b = bank()
        b['roentgen_v2', 0, 'maira2']['score_record']['scoring']['clinical_totals']['total_hard_contradiction_count'] = 0
        result = w.replay(b, POLICY, 30)
        observed = {a['observed_candidate_id'] for a in result['action_trace']}
        self.assertIn(result['selected_candidate_id'], observed)
        self.assertTrue(result['selected_candidate_id'].startswith('chexgenbench_sana'))

    def test_invalid_budget_type_refused(self):
        for value in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                w.replay(bank(), POLICY, value)

    def test_replay_deterministic_and_inputs_unchanged(self):
        b = bank(ehr='positive', image_states=('negative', 'positive', 'positive'))
        before = copy.deepcopy(b)
        self.assertEqual(w.replay(b, POLICY, 30), w.replay(b, POLICY, 30))
        self.assertEqual(b, before)

    def test_policy_description_not_new_threshold_or_execution_permission(self):
        p = w.policy_description()
        self.assertTrue(p['no_new_numerical_threshold_or_weight'])
        self.assertFalse(p['alternate_endpoints_used_for_routing'])
        self.assertFalse(p['regeneration_authorized'])
        self.assertFalse(p['clinical_acceptance'])
        self.assertFalse(p['replace_or_enrich_ehr'])

    def test_all_original_caps_run_without_actual_inference(self):
        for cap in w.baseline.CAPS:
            result = w.replay(bank(), POLICY, cap)
            self.assertLessEqual(result['simulated_model_calls'], cap)
            self.assertFalse(result['actual_regeneration_executed'])

    def test_slurm_guard_precedes_load(self):
        with patch.dict(w.baseline.os.environ, {}, clear=True), patch.object(w, 'load') as load:
            with self.assertRaises(RuntimeError):
                w.execute(Path('/invented'), 'new')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
