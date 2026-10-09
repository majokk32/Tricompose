"""Invented numeric states/hashes only; no pixels, bodies, weights or inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import benchmark_finding_matched_stress_v1 as m


def pair(positive=.8, negative=.1):
    return {family: {'positive_cosine': positive, 'negative_cosine': negative}
            for family in m.scoring.FAMILIES}


def packet(item='item_0000'):
    return {'item_id': item, 'findings': {finding: {
        'ehr': 'unknown', 'report': 'unknown',
        'cached_xrv': 'unknown' if finding == 'support_devices' else 'positive',
        'score_pairs': pair()} for finding in m.FINDINGS}}


def fixture_bank():
    bank, records = {}, []
    for number in (0, 1):
        case = f'fixture_case_{number}'
        image = f'fixture_image_{number}'
        report = f'fixture_report_{number}'
        lin = {'ehr_sha256': str(number+1)*64, 'ehr_facts_sha256': str(number+2)*64,
            'cxr_sha256': str(number+3)*64, 'report_sha256': str(number+4)*64,
            'cxr_candidate_id': image, 'report_candidate_id': report,
            'cxr_model_id': 'fixture_generator', 'report_model_id': 'fixture_report_expert', 'cxr_seed': 0}
        states = {name: {f: 'unknown' for f in m.b.policy_module.FINDINGS} for name in ('ehr', 'xrv', 'chexbert')}
        states['ehr']['pleural_effusion'] = 'positive' if number == 0 else 'negative'
        states['xrv']['pleural_effusion'] = 'positive' if number == 0 else 'negative'
        states['chexbert']['pleural_effusion'] = 'positive' if number == 0 else 'negative'
        bank[case] = {number: {'view': {'case_id': case, 'lineage': lin, 'states': states}}}
        record = {'case_id': case, **{k: lin[k] for k in ('cxr_candidate_id', 'cxr_model_id',
            'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256')},
            'cxr_path': str(m.PROTECTED_ROOT / f'invented/{image}.png'),
            'status': 'scored', 'score_pairs': {f: pair(.8 if number == 0 else .1, .1 if number == 0 else .8) for f in m.FINDINGS}}
        records.append(record)
    return bank, records


class FindingStressTests(unittest.TestCase):
    def test_protocol_has_all_policies_not_posthoc_best_pick(self):
        value = m.protocol()
        self.assertEqual(value['image_policies'], list(m.POLICIES))
        self.assertEqual(value['findings'], list(m.FINDINGS))
        self.assertFalse(value['key_target_finding_is_predictor_input'])
        self.assertFalse(value['regeneration_authorized'])

    def test_unknown_ehr_never_localizes_even_with_pair_conflict(self):
        p = packet(); p['findings']['edema']['report'] = 'negative'
        for policy in m.POLICIES:
            result = m.predict(p, policy)
            self.assertTrue(result['proxy_mismatch_detected'])
            self.assertIsNone(result['ehr_asymmetry_only'])
            self.assertEqual(result['unsafe_report_blame'], 'report')
            self.assertIsNone(result['clinical_fault_confirmed'])

    def test_ehr_image_support_report_opposition_is_tentative_report_hint(self):
        p = packet(); p['findings']['edema'].update(ehr='positive', report='negative')
        result = m.predict(p, 'agree_mean')
        self.assertEqual(result['ehr_asymmetry_only'], 'report')
        self.assertFalse(result['regeneration_authorized'])

    def test_ehr_report_support_image_opposition_is_tentative_image_hint(self):
        p = packet(); p['findings']['pleural_effusion'].update(ehr='negative', report='negative')
        self.assertEqual(m.predict(p, 'agree_mean')['ehr_asymmetry_only'], 'cxr')

    def test_incompatible_finding_hints_abstain(self):
        p = packet()
        p['findings']['edema'].update(ehr='positive', report='negative')
        p['findings']['pleural_effusion'].update(ehr='negative', report='negative')
        result = m.predict(p, 'agree_mean')
        self.assertIsNone(result['ehr_asymmetry_only'])
        self.assertEqual(result['ehr_asymmetry_status'], 'conflicting_target_hints_abstain')

    def test_uncertain_ehr_not_known(self):
        p = packet(); p['findings']['edema'].update(ehr='uncertain', report='negative')
        self.assertIsNone(m.predict(p, 'agree_mean')['ehr_asymmetry_only'])

    def test_empty_report_comparison_not_success(self):
        result = m.predict(packet(), 'agree_mean')
        self.assertEqual(result['pair_status'], 'no_comparable_findings')
        self.assertFalse(result['proxy_mismatch_detected'])
        self.assertEqual(result['comparable_findings'], [])

    def test_supported_subset_does_not_imply_full_cohort_clean(self):
        p = packet(); p['findings']['edema']['report'] = 'positive'
        result = m.predict(p, 'agree_mean')
        self.assertEqual(result['pair_status'], 'no_opposition_on_available_subset')
        self.assertEqual(result['comparable_findings'], ['edema'])
        self.assertIsNone(result['clinical_localization_accuracy'])

    def test_device_head_missing_not_backfilled(self):
        p = packet(); p['findings']['support_devices'].update(ehr='positive', report='negative')
        self.assertEqual(m.predict(p, 'biovil_mean')['ehr_asymmetry_only'], 'report')
        self.assertIsNone(m.predict(p, 'agree_mean')['ehr_asymmetry_only'])
        self.assertFalse(m.predict(p, 'cached_xrv')['proxy_mismatch_detected'])

    def test_manufactured_device_classifier_refused(self):
        p = packet(); p['findings']['support_devices']['cached_xrv'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'device_head'):
            m.predict(p, 'cached_xrv')

    def test_image_readers_disagree_abstain_not_negate(self):
        p = packet(); p['findings']['edema'].update(cached_xrv='negative', report='negative')
        self.assertTrue(m.predict(p, 'biovil_mean')['proxy_mismatch_detected'])
        result = m.predict(p, 'agree_mean')
        fact = next(r for r in result['facts'] if r['finding'] == 'edema')
        self.assertEqual(fact['image_state'], 'unknown')
        self.assertEqual(fact['image_report_relation'], 'not_comparable')

    def test_template_variation_does_not_count_as_independent_votes(self):
        p = packet(); p['findings']['edema'].update(report='negative')
        p['findings']['edema']['score_pairs']['shows_no'] = {'positive_cosine': .1, 'negative_cosine': .8}
        self.assertTrue(m.predict(p, 'agree_mean')['proxy_mismatch_detected'])
        self.assertFalse(m.predict(p, 'agree_all_templates')['proxy_mismatch_detected'])

    def test_failed_scores_held_as_unknown(self):
        p = packet()
        for f in p['findings'].values():
            f.update(score_pairs=None, report='negative')
        self.assertEqual(m.predict(p, 'biovil_mean')['pair_status'], 'no_comparable_findings')
        self.assertTrue(m.predict(p, 'cached_xrv')['proxy_mismatch_detected'])

    def test_targeted_finding_forbidden_in_predictor(self):
        p = packet(); p['targeted_finding'] = 'edema'
        with self.assertRaisesRegex(ValueError, 'complete_finding_packet'):
            m.predict(p, 'agree_mean')

    def test_donor_case_model_hash_and_key_fields_forbidden(self):
        for field in ('donor_case_id', 'case_id', 'model', 'cxr_sha256', 'intervention_type', 'benchmark_split'):
            p = packet(); p[field] = 'forbidden'
            with self.assertRaises(ValueError):
                m.predict(p, 'agree_mean')

    def test_extra_fact_field_forbidden(self):
        p = packet(); p['findings']['edema']['target_label'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'four_states_only'):
            m.predict(p, 'agree_mean')

    def test_no_disease_can_be_silently_dropped(self):
        p = packet(); p['findings'].pop('pneumonia')
        with self.assertRaisesRegex(ValueError, 'complete_finding_packet'):
            m.predict(p, 'agree_mean')

    def test_invalid_policy_refused(self):
        with self.assertRaisesRegex(ValueError, 'predeclared_image_policy'):
            m.predict(packet(), 'choose_best_after_results')

    def test_item_identity_not_a_prediction_feature(self):
        a = m.predict(packet('item_0001'), 'agree_mean')
        z = m.predict(packet('item_0999'), 'agree_mean')
        a.pop('item_id'); z.pop('item_id')
        self.assertEqual(a, z)

    def test_prediction_preserves_packet(self):
        p = packet(); old = deepcopy(p)
        m.predict(p, 'agree_mean')
        self.assertEqual(p, old)

    def test_bind_uses_recipient_ehr_with_displayed_donor_image(self):
        bank, records = fixture_bank()
        resolver = [{'item_id': 'item_0000', 'case_id': 'fixture_case_0', 'benchmark_split': 'development',
            'ehr_sha256': '1'*64, 'displayed_cxr_candidate_id': 'fixture_image_1',
            'displayed_cxr_sha256': '4'*64, 'displayed_report_candidate_id': 'fixture_report_0',
            'displayed_report_sha256': '4'*64}]
        with patch.object(m.b.policy_module, 'snapshot', side_effect=lambda c: c['view']):
            images = m.numeric_images(records, bank)
            result = m.packets(resolver, bank, images)
        fact = result[0]['findings']['pleural_effusion']
        self.assertEqual((fact['ehr'], fact['cached_xrv'], fact['report']), ('positive', 'negative', 'positive'))
        self.assertEqual(m.predict(result[0], 'agree_mean')['ehr_asymmetry_only'], 'cxr')
        self.assertNotIn('cxr_path', result[0])

    def test_numeric_cache_binding_does_not_read_referenced_png(self):
        bank, records = fixture_bank()
        with patch.object(m.b.policy_module, 'snapshot', side_effect=lambda c: c['view']):
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('no pixel read')):
                images = m.numeric_images(records, bank)
        self.assertEqual(len(images), 2)

    def test_changed_cached_hash_refused(self):
        bank, records = fixture_bank(); records[0]['cxr_sha256'] = '9'*64
        with patch.object(m.b.policy_module, 'snapshot', side_effect=lambda c: c['view']):
            with self.assertRaisesRegex(ValueError, 'exact_unique_numeric'):
                m.numeric_images(records, bank)

    def test_all_image_failures_need_explicit_record(self):
        bank, records = fixture_bank(); records.pop()
        with patch.object(m.b.policy_module, 'snapshot', side_effect=lambda c: c['view']):
            with self.assertRaisesRegex(ValueError, 'every_original_image'):
                m.numeric_images(records, bank)

    def test_cpu_guard_precedes_source_reads(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('cpu allocation required')):
            with patch.object(m, 'protocol', side_effect=AssertionError('no read')):
                with self.assertRaises(RuntimeError):
                    m.run(SimpleNamespace())

    def test_evaluation_key_does_not_change_already_sealed_prediction(self):
        resolver = [{'item_id': 'item_0000', 'benchmark_split': 'development'}]
        p = packet(); p['findings']['edema'].update(ehr='positive', report='negative')
        outcomes = [m.predict(p, policy) for policy in m.POLICIES]
        before = deepcopy(outcomes)
        key = [{'item_id': 'item_0000', 'intervention_type': 'report_swap',
                'intervention_target': 'report', 'targeted_finding': 'edema'}]
        table = m.summarize(outcomes, resolver, key)
        self.assertEqual(outcomes, before)
        self.assertTrue(all(r['evaluation_only_target_finding_items'] == 1 for r in table))
        key[0]['targeted_finding'] = 'pneumonia'
        m.summarize(outcomes, resolver, key)
        self.assertEqual(outcomes, before)

    def test_controls_have_no_intervened_target_recovery_metric(self):
        resolver = [{'item_id': 'item_0000', 'benchmark_split': 'development'}]
        key = [{'item_id': 'item_0000', 'intervention_type': 'no_corruption',
                'intervention_target': 'none', 'targeted_finding': None}]
        outcomes = [m.predict(packet(), policy) for policy in m.POLICIES]
        table = m.summarize(outcomes, resolver, key)
        self.assertTrue(all(r['target_recovery_over_all_available_intervened'] is None for r in table))
        self.assertTrue(all(r['evaluation_only_target_finding_items'] == 0 for r in table))

    def test_abstention_remains_in_all_intervened_denominator(self):
        resolver = [{'item_id': 'item_0000', 'benchmark_split': 'development'}]
        key = [{'item_id': 'item_0000', 'intervention_type': 'cxr_swap',
                'intervention_target': 'cxr', 'targeted_finding': 'edema'}]
        outcomes = [m.predict(packet(), policy) for policy in m.POLICIES]
        table = m.summarize(outcomes, resolver, key)
        self.assertTrue(all(r['available_items'] == 1 and r['target_recovery_over_all_available_intervened'] == 0 for r in table))

    def test_missing_policy_output_cannot_shrink_denominator(self):
        resolver = [{'item_id': 'item_0000', 'benchmark_split': 'development'}]
        key = [{'item_id': 'item_0000', 'intervention_type': 'cxr_swap',
                'intervention_target': 'cxr', 'targeted_finding': 'edema'}]
        outcomes = [m.predict(packet(), policy) for policy in m.POLICIES[:-1]]
        with self.assertRaisesRegex(ValueError, 'complete_blind'):
            m.summarize(outcomes, resolver, key)

    def test_loading_numeric_scores_does_not_reopen_manifest_body_pins(self):
        run = m.PROTECTED_ROOT / 'invented_scores'
        manifest = {'schema_version': m.scoring.SCHEMA + '-manifest', 'clinical_qualified': False,
            'original_selection_changed': False, 'regeneration_authorized': False,
            'new_training_calls': 0, 'new_generation_calls': 0,
            'sources': {'/forbidden/body.png': '1'*64}, 'artifacts': {'image_scores.json': '2'*64}}
        data = {'schema_version': m.scoring.SCHEMA + '-image-scores', 'records': []}
        with patch.object(m, 'require_inside', return_value=run):
            with patch.object(m.b, 'checked', side_effect=[json.dumps(manifest), json.dumps(data)]) as checked:
                records, _ = m.load_scores(run, '3'*64, {})
        self.assertEqual(records, [])
        self.assertEqual(checked.call_count, 2)
        self.assertEqual([c.args[0].name for c in checked.call_args_list], ['manifest.json', 'image_scores.json'])


if __name__ == '__main__':
    unittest.main()
