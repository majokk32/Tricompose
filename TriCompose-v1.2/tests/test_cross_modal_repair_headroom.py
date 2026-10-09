"""Invented four-state signatures; never patient/model/image artifacts."""
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cross_headroom_fixture', ROOT / 'tools/audit_cross_modal_repair_headroom.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def signature(image='sana', report='maira', ehr='positive', xrv='negative', chex='negative', quality=.8):
    values = {key: {f: 'unknown' for f in w.guarded.legacy.FINDINGS} for key in ('ehr', 'xrv', 'chexbert')}
    for key, state in (('ehr', ehr), ('xrv', xrv), ('chexbert', chex)):
        values[key]['edema'] = state
    return reseal({'case_id': 'invented_case', 'triple_candidate_id': image + '_' + report,
        'lineage': {'ehr_sha256': h('invented_ehr'), 'ehr_facts_sha256': h('invented_facts'),
            'cxr_candidate_id': image, 'cxr_sha256': h(image), 'cxr_model_id': image, 'cxr_seed': 0,
            'report_candidate_id': image + '_' + report, 'report_sha256': h(image + report), 'report_model_id': report},
        'ehr_reference_states': values['ehr'], 'image_reference_states': values['xrv'],
        'report_reference_states': values['chexbert'],
        'ehr_source_categories': {f: ['diagnoses'] if s != 'unknown' else [] for f, s in values['ehr'].items()},
        'source_gate_failure_count': 0, 'source_image_validity': True,
        'report_structure_quality': quality, 'clinical_accuracy': None, 'clinical_acceptance': False})


def reseal(s):
    edges = w.raw_edges({'ehr': s['ehr_reference_states'], 'xrv': s['image_reference_states'], 'chexbert': s['report_reference_states']})
    s['raw_edges'] = edges
    for key, edge, field in (('image_positive_support', 'cxr_report', 'positive_support'),
            ('ehr_direct_support', 'ehr_report', 'support'), ('image_comparable', 'cxr_report', 'comparable'),
            ('ehr_comparable', 'ehr_report', 'comparable'), ('image_opposition', 'cxr_report', 'opposition'),
            ('ehr_opposition', 'ehr_report', 'opposition')):
        s[key] = edges[edge][field]
    s['direct_ehr_known_facts'] = len(edges['ehr_cxr']['known'])
    s['image_known_facts'] = len(edges['cxr_report']['known'])
    s['report_positive_unknown_image'] = [f for f, state in s['report_reference_states'].items()
        if state == 'positive' and s['image_reference_states'][f] not in w.EXPLICIT]
    opp = edges['ehr_cxr']['opposition']
    s['image_branch_basis'] = {'cached_image_branch_allowed': bool(opp) or not s['source_image_validity'],
        'reason': 'direct_ehr_xrv_opposition_proxy' if opp else 'no_direct_ehr_image_reference' if not edges['ehr_cxr']['known']
            else 'no_comparable_ehr_image_evidence' if not edges['ehr_cxr']['comparable'] else 'no_direct_ehr_image_opposition',
        'known_direct_ehr_facts': len(edges['ehr_cxr']['known']), 'comparable_direct_ehr_image_facts': len(edges['ehr_cxr']['comparable']),
        'explicit_proxy_opposition_evidence_ids': [h(f) for f in opp], 'report_votes_used': False,
        'confirmed_image_fault': False, 'clinical_regeneration_authorized': False}
    return s


def corrected():
    return signature(), signature('roentgen', 'cxrmate', xrv='positive', chex='positive')


class CrossModalRepairHeadroomTests(unittest.TestCase):
    def test_explicit_ehr_image_conflict_can_have_preserving_proxy_gain(self):
        a, b = corrected(); r = w.compare_image(a, b)
        self.assertTrue(r['cross_image_preserving_proxy_gain'])
        self.assertEqual(r['removed_opposition_fact_ids']['ehr_cxr'], ['edema'])
        self.assertFalse(r['clinical_repair_success'])

    def test_unknown_and_uncertain_ehr_cannot_become_image_repair_truth(self):
        for state in ('unknown', 'uncertain'):
            a = signature(ehr=state); b = signature('roentgen', ehr=state, xrv='positive', chex='positive')
            r = w.compare_image(a, b)
            self.assertFalse(r['cross_image_preserving_proxy_gain'])
            self.assertEqual(r['gained_support_fact_ids']['ehr_cxr'], [])

    def test_silencing_opposition_to_unknown_or_uncertain_is_not_repair(self):
        for state in ('unknown', 'uncertain'):
            a, b = corrected(); b['image_reference_states']['edema'] = state; reseal(b)
            r = w.compare_image(a, b)
            self.assertFalse(r['cross_image_preserving_proxy_gain'])
            self.assertEqual(r['silenced_opposition_fact_ids']['ehr_cxr'], ['edema'])

    def test_better_image_report_agreement_alone_is_not_anchored_gain(self):
        a = signature(xrv='positive', chex='negative')
        b = signature('roentgen', xrv='positive', chex='positive')
        self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_new_conflict_cannot_be_paid_for_by_lower_total_count(self):
        a, b = corrected()
        for s in (a, b):
            s['ehr_reference_states']['pneumonia'] = 'positive'
            s['ehr_source_categories']['pneumonia'] = ['diagnoses']
            s['image_reference_states']['pneumonia'] = 'positive'
            s['report_reference_states']['pneumonia'] = 'positive'
        b['report_reference_states']['pneumonia'] = 'negative'
        for s in (a, b): reseal(s)
        r = w.compare_image(a, b)
        self.assertFalse(r['cross_image_preserving_proxy_gain'])
        self.assertEqual(r['new_opposition_fact_ids']['ehr_report'], ['pneumonia'])

    def test_positive_image_report_support_must_survive_image_change(self):
        a, b = corrected()
        a['image_reference_states']['pneumonia'] = a['report_reference_states']['pneumonia'] = 'positive'
        b['image_reference_states']['pneumonia'] = b['report_reference_states']['pneumonia'] = 'negative'
        for s in (a, b): reseal(s)
        r = w.compare_image(a, b)
        self.assertFalse(r['cross_image_preserving_proxy_gain'])
        self.assertEqual(r['lost_fact_ids']['cxr_report']['positive_support'], ['pneumonia'])

    def test_equal_support_count_with_different_ids_is_not_preservation(self):
        a, b = corrected()
        a['image_reference_states']['pneumonia'] = a['report_reference_states']['pneumonia'] = 'positive'
        b['image_reference_states']['atelectasis'] = b['report_reference_states']['atelectasis'] = 'positive'
        for s in (a, b): reseal(s)
        self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_alternative_must_pass_against_report_only_reference_too(self):
        a, b = corrected(); static = copy.deepcopy(a)
        static['triple_candidate_id'] = 'sana_static'
        static['report_reference_states']['pneumonia'] = 'positive'
        for s in (a, static, b): s['image_reference_states']['pneumonia'] = 'positive'; reseal(s)
        self.assertTrue(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])
        r = w.image_pair(a, static, b)
        self.assertFalse(r['proxy_preserving_opportunity'])
        self.assertFalse(r['opportunity_with_current_scope_basis'])

    def test_missing_image_comparison_gain_is_not_current_policy_authorized(self):
        a = signature(xrv='unknown', chex='positive')
        b = signature('roentgen', xrv='positive', chex='positive')
        r = w.image_pair(a, a, b)
        self.assertTrue(r['proxy_preserving_opportunity'])
        self.assertFalse(r['scope_allows_cached_image_branch'])
        self.assertFalse(r['opportunity_with_current_scope_basis'])

    def test_fixed_ehr_state_hash_and_provenance_cannot_drift(self):
        for field in ('hash', 'state', 'provenance'):
            a, b = corrected()
            if field == 'hash': b['lineage']['ehr_sha256'] = h('different')
            elif field == 'state': b['ehr_reference_states']['edema'] = 'negative'; reseal(b)
            else: b['ehr_source_categories']['edema'] = ['other']
            with self.assertRaises(ValueError): w.compare_image(a, b)

    def test_report_only_reference_cannot_change_image(self):
        a, b = corrected()
        with self.assertRaises(ValueError): w.image_pair(a, b, b)

    def test_duplicate_image_hash_or_identity_not_cross_image_gain(self):
        for key in ('cxr_candidate_id', 'cxr_sha256'):
            a, b = corrected(); b['lineage'][key] = a['lineage'][key]
            self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_lower_or_missing_structure_blocks_gain(self):
        for quality in (.7, None):
            a, b = corrected(); b['report_structure_quality'] = quality
            self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_artifact_or_image_flag_failure_blocks_gain(self):
        for field, value in (('source_gate_failure_count', 1), ('source_image_validity', False)):
            a, b = corrected(); b[field] = value; reseal(b)
            self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_nan_boolean_and_wrong_type_quality_refused(self):
        for value in (float('nan'), float('inf'), True, '0.9', -1, 2):
            a, b = corrected(); b['report_structure_quality'] = value
            with self.assertRaises(ValueError): w.compare_image(a, b)

    def test_forged_raw_counters_and_branch_permission_refused(self):
        a, b = corrected(); b['raw_edges']['ehr_cxr']['support'] = []
        with self.assertRaises(ValueError): w.compare_image(a, b)
        a, b = corrected(); a['image_branch_basis']['cached_image_branch_allowed'] = False
        with self.assertRaises(ValueError): w.image_pair(a, a, b)

    def test_global_no_finding_does_not_invent_negative_states(self):
        a, b = corrected()
        a['report_reference_states']['no_finding'] = 'positive'; a['report_reference_states']['edema'] = 'unknown'; reseal(a)
        self.assertEqual(a['raw_edges']['ehr_report']['opposition'], [])
        self.assertTrue(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_only_changed_structure_without_finding_gain_is_not_opportunity(self):
        a = signature(xrv='positive', chex='positive')
        b = signature('roentgen', xrv='positive', chex='positive', quality=.9)
        self.assertFalse(w.compare_image(a, b)['cross_image_preserving_proxy_gain'])

    def test_no_endpoint_argument_in_predicate_or_plan(self):
        self.assertEqual(list(inspect.signature(w.compare_image).parameters), ['base', 'alternative'])
        self.assertEqual(list(inspect.signature(w.prepare).parameters), ['bank', 'policy', 'reportonly_ids', 'guarded_ids'])

    def test_endpoint_metadata_never_changes_proxy_comparison(self):
        a, b = corrected(); expected = w.image_pair(a, a, b)
        a['biovil_raw_cosine'] = 1; b['biovil_raw_cosine'] = -1
        self.assertEqual(w.image_pair(a, a, b), expected)

    def test_deterministic_input_preservation(self):
        a, b = corrected(); before = copy.deepcopy((a, b))
        self.assertEqual(w.image_pair(a, a, b), w.image_pair(a, a, b))
        self.assertEqual((a, b), before)

    def test_prepare_keeps_all_eleven_alternatives_and_no_new_winner(self):
        policy = {'image_order': [['sana', 0], ['pixart', 0], ['roentgen', 0]],
            'report_order': ['maira', 'cxrmate', 'llava', 'chexagent']}
        evidence, grid = {}, {}
        for image, _ in policy['image_order']:
            for report in policy['report_order']:
                s = signature(image, report, xrv='negative' if image == 'sana' else 'positive',
                    chex='negative' if image == 'sana' else 'positive')
                evidence[s['triple_candidate_id']] = s
                grid[image, 0, report] = {'score_record': {'triple_candidate_id': s['triple_candidate_id']}}
        bank = {'invented_case': grid}; fixed = {'invented_case': 'sana_maira'}
        current = {'invented_case': 'roentgen_cxrmate'}
        before = copy.deepcopy((bank, evidence, fixed, current))
        with patch.object(w, 'signature', side_effect=lambda c: evidence[c['score_record']['triple_candidate_id']]):
            plan = w.prepare(bank, policy, fixed, current)
            self.assertEqual(plan, w.prepare(bank, policy, fixed, current))
            self.assertEqual(len(plan['directed_pairs']), 11)
            self.assertEqual(sum(p['pair_kind'] == 'same_image_report' for p in plan['directed_pairs']), 3)
            self.assertEqual(sum(p['pair_kind'] == 'cross_image' for p in plan['directed_pairs']), 8)
            self.assertEqual(plan['cases'][0]['finite_cache_category'], 'cross_image_only')
            self.assertIsNone(plan['cases'][0]['new_selected_candidate_id'])
            self.assertEqual(plan['guarded_selection_audit'][0]['change'], 'cross_image')
            self.assertFalse(plan['clinical_acceptance'])
            with self.assertRaises(ValueError): w.prepare(bank, policy, {}, current)
            with self.assertRaises(ValueError): w.prepare(bank, policy, fixed, {'invented_case': 'other_case'})
        self.assertEqual((bank, evidence, fixed, current), before)

    def test_conditional_endpoints_average_within_ehr_not_by_pair_count(self):
        plan = {'cases': [{'case_id': 'a'}, {'case_id': 'b'}], 'directed_pairs': [
            {'case_id': case, 'pair_kind': 'same_image_report', 'baseline_candidate_id': 'base',
                'alternative_candidate_id': alt, 'proxy_preserving_opportunity': True}
            for case, alt in (('a', 'zero'), ('a', 'one'), ('b', 'nine'))]}
        candidates = {'base': 0., 'zero': 0., 'one': 1., 'nine': .9}
        before = copy.deepcopy(plan)
        with patch.object(w.guarded.baseline, 'candidate_readout', side_effect=lambda v: {'biovil_raw_cosine': v}):
            _, rows = w.attach(plan, candidates)
        self.assertAlmostEqual(rows[0]['conditional_case_balanced_all_alternative_mean_delta'], .7)
        self.assertEqual(rows[0]['opportunity_ehr_cases'], 2)
        self.assertFalse(rows[0]['selected_output_effect'])
        self.assertEqual(plan, before)

    def test_missing_required_endpoint_is_na_not_zero_or_available_only_mean(self):
        plan = {'cases': [{'case_id': 'a'}, {'case_id': 'b'}], 'directed_pairs': [
            {'case_id': case, 'pair_kind': 'cross_image', 'baseline_candidate_id': 'base',
                'alternative_candidate_id': alt, 'proxy_preserving_opportunity': True}
            for case, alt in (('a', 'available'), ('b', 'missing'))]}
        candidates = {'base': 0., 'available': .5, 'missing': None}
        with patch.object(w.guarded.baseline, 'candidate_readout', side_effect=lambda v: {'biovil_raw_cosine': v}):
            measured, rows = w.attach(plan, candidates)
        self.assertIsNone(measured[1]['biovil_delta_alternative_minus_baseline'])
        self.assertIsNone(rows[1]['conditional_case_balanced_all_alternative_mean_delta'])
        self.assertEqual(rows[1]['complete_endpoint_opportunity_ehr_cases'], 1)

    def test_bad_four_state_vector_and_unsupported_ehr_state_refused(self):
        a, b = corrected(); b['ehr_reference_states']['edema'] = 'invented_yes'
        with self.assertRaises(ValueError): w.compare_image(a, b)
        a, b = corrected(); b['ehr_source_categories']['edema'] = []
        with self.assertRaises(ValueError): w.compare_image(a, b)

    def test_no_opportunity_is_na_not_zero_benefit(self):
        plan = {'cases': [{'case_id': 'invented_case'}], 'directed_pairs': []}
        pairs, rows = w.attach(plan, {})
        self.assertEqual(pairs, [])
        self.assertTrue(all(r['conditional_case_balanced_all_alternative_mean_delta'] is None for r in rows))
        self.assertTrue(all(r['fixed_ehr_denominator'] == 1 for r in rows))

    def test_slurm_guard_precedes_cache_loading(self):
        with patch.dict(w.guarded.baseline.os.environ, {}, clear=True), patch.object(w.guarded, 'load') as load:
            with self.assertRaises(RuntimeError): w.execute(Path('/invented'), 'opaque')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
