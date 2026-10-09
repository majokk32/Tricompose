"""Wholly invented state fixtures; no source patients, inference or IO."""
from copy import deepcopy
import unittest

from tricompose_v12 import probe_repair_v1 as p


def observation(*, model='sana', report='maira', number=0, ehr='positive', image='positive', text='negative'):
    states = {key: {f: 'unknown' for f in p.FINDINGS} for key in ('ehr', 'xrv', 'chexbert')}
    states['ehr']['edema'], states['xrv']['edema'], states['chexbert']['edema'] = ehr, image, text
    return {'schema_version': p.VERSION + '-observation', 'case_id': 'case_000',
        'candidate_id': f'triple_{number}', 'lineage': {
            'ehr_sha256': '1'*64, 'ehr_facts_sha256': '2'*64,
            'cxr_sha256': ('3' if model == 'sana' else '4')*64,
            'report_sha256': f'{number+5:064x}', 'cxr_candidate_id': 'image_'+model,
            'report_candidate_id': f'report_{number}', 'cxr_model_id': model,
            'cxr_seed': 0, 'report_model_id': report}, 'states': states,
        'ehr_sources': {f: ['diagnosis'] if f == 'edema' and ehr != 'unknown' else [] for f in p.FINDINGS},
        'quality': {'cxr_basic_validity_pass': True, 'report_structure_quality_score_0_1': .8},
        'artifact_gate_failures': 0, 'clinical_qualified': False}


def controller(value=None, budget=12, **kwargs):
    return p.ProbeController(value or observation(), [('sana', 0), ('roentgen', 0)],
        ['maira', 'cxrmate', 'llava'], budget, **kwargs)


class ProbeCreditTests(unittest.TestCase):
    def test_exact_positive_conflict_repair_preserves_comparisons(self):
        a, b = observation(), observation(number=1, report='cxrmate', text='positive')
        credit = p.action_credit(a, b, 'report_probe')
        self.assertTrue(credit['replacement_allowed_under_proxy_contract'])
        self.assertEqual(credit['resolved_proxy_conflicts']['cxr_report'], ['edema'])
        self.assertEqual(credit['resolved_proxy_conflicts']['ehr_report'], ['edema'])
        self.assertFalse(credit['clinical_repair_success'])

    def test_unknown_and_uncertain_cannot_silence_conflict(self):
        for state in ('unknown', 'uncertain'):
            credit = p.action_credit(observation(), observation(number=1, text=state), 'report_probe')
            self.assertFalse(credit['replacement_allowed_under_proxy_contract'])
            self.assertEqual(credit['resolved_proxy_conflicts']['cxr_report'], [])
            self.assertEqual(credit['silenced_conflict_fact_ids']['cxr_report'], ['edema'])

    def test_other_finding_cannot_pay_for_lost_supported_fact(self):
        a, b = observation(), observation(number=1, text='positive')
        for v in (a, b): v['states']['xrv']['cardiomegaly'] = 'positive'
        a['states']['chexbert']['cardiomegaly'] = 'positive'
        credit = p.action_credit(a, b, 'report_probe')
        self.assertFalse(credit['replacement_allowed_under_proxy_contract'])
        self.assertEqual(credit['lost_protected_fact_ids']['cxr_report'], ['cardiomegaly'])

    def test_new_opposition_blocks_replacement(self):
        a, b = observation(), observation(number=1, text='positive')
        for v in (a, b): v['states']['xrv']['pneumothorax'] = 'negative'
        b['states']['chexbert']['pneumothorax'] = 'positive'
        credit = p.action_credit(a, b, 'report_probe')
        self.assertIn('new_explicit_proxy_opposition', credit['reasons'])

    def test_missing_negative_comparison_is_not_improvement(self):
        a, b = observation(), observation(number=1, text='positive')
        for v in (a, b): v['states']['xrv']['pneumothorax'] = 'negative'
        a['states']['chexbert']['pneumothorax'] = 'negative'
        credit = p.action_credit(a, b, 'report_probe')
        self.assertEqual(credit['lost_comparable_fact_ids']['cxr_report'], ['pneumothorax'])
        self.assertFalse(credit['replacement_allowed_under_proxy_contract'])

    def test_unknown_ehr_and_missing_device_head_do_not_create_negative(self):
        a, b = observation(ehr='unknown'), observation(number=1, ehr='unknown', text='positive')
        b['states']['chexbert']['support_devices'] = 'positive'
        credit = p.action_credit(a, b, 'report_probe')
        self.assertTrue(credit['replacement_allowed_under_proxy_contract'])
        self.assertEqual(credit['new_proxy_conflicts']['ehr_report'], [])
        self.assertEqual(credit['new_proxy_conflicts']['cxr_report'], [])

    def test_report_probe_cannot_change_image_or_classifier(self):
        a = observation()
        for field in ('cxr_sha256', 'cxr_candidate_id', 'cxr_model_id', 'cxr_seed'):
            b = observation(number=1)
            b['lineage'][field] = 1 if field == 'cxr_seed' else '9'*64
            with self.assertRaises(ValueError): p.action_credit(a, b, 'report_probe')
        b = observation(number=1, image='negative')
        with self.assertRaises(ValueError): p.action_credit(a, b, 'report_probe')

    def test_image_probe_holds_report_expert_fixed(self):
        with self.assertRaises(ValueError):
            p.action_credit(observation(), observation(model='roentgen', report='cxrmate', number=1), 'image_probe')

    def test_image_probe_can_resolve_direct_ehr_opposition(self):
        credit = p.action_credit(observation(image='negative', text='positive'),
            observation(model='roentgen', number=1, text='positive'), 'image_probe')
        self.assertTrue(credit['replacement_allowed_under_proxy_contract'])
        self.assertEqual(credit['resolved_proxy_conflicts']['ehr_cxr'], ['edema'])

    def test_fixed_ehr_hash_states_and_provenance_enforced(self):
        for field in ('ehr_sha256', 'ehr_facts_sha256'):
            b = observation(number=1); b['lineage'][field] = '9'*64
            with self.assertRaises(ValueError): p.action_credit(observation(), b, 'report_probe')
        for field in ('state', 'sources', 'case'):
            b = observation(number=1)
            if field == 'state': b['states']['ehr']['edema'] = 'negative'
            elif field == 'sources': b['ehr_sources']['edema'] = ['other_source']
            else: b['case_id'] = 'case_001'
            with self.assertRaises(ValueError): p.action_credit(observation(), b, 'report_probe')

    def test_duplicate_bytes_are_not_independent_probe_success(self):
        a, b = observation(), observation(number=1, text='positive')
        b['lineage']['report_sha256'] = a['lineage']['report_sha256']
        credit = p.action_credit(a, b, 'report_probe')
        self.assertTrue(credit['duplicate_bytes'])
        self.assertFalse(credit['replacement_allowed_under_proxy_contract'])

    def test_missing_or_worse_quality_blocks_replacement(self):
        for q in (None, .7):
            b = observation(number=1, text='positive')
            b['quality']['report_structure_quality_score_0_1'] = q
            self.assertFalse(p.action_credit(observation(), b, 'report_probe')['replacement_allowed_under_proxy_contract'])

    def test_invalid_new_artifact_blocks_replacement(self):
        for field in ('gate', 'image'):
            b = observation(number=1, text='positive')
            if field == 'gate': b['artifact_gate_failures'] = 1
            else: b['quality']['cxr_basic_validity_pass'] = False
            self.assertFalse(p.action_credit(observation(), b, 'report_probe')['replacement_allowed_under_proxy_contract'])

    def test_no_finding_does_not_expand_unknown_states(self):
        a = observation(ehr='unknown', text='unknown')
        a['states']['chexbert']['no_finding'] = 'positive'
        self.assertEqual(p.edges(a)['cxr_report']['opposition'], set())
        self.assertEqual(p.edges(a)['ehr_report']['known'], set())

    def test_invalid_quality_types_rejected(self):
        for q in (True, '0.8', float('nan'), float('inf'), -.1, 1.1):
            a = observation(); a['quality']['report_structure_quality_score_0_1'] = q
            with self.assertRaises(ValueError): p.validate_snapshot(a)

    def test_extra_payload_or_secondary_score_cannot_enter_controller(self):
        for key in ('report_text', 'patient_id', 'biovil_score', 'clinical_score'):
            a = observation(); a[key] = 'forbidden_authored_fixture'
            with self.assertRaises(ValueError): controller(a)

    def test_cannot_promote_clinical_qualification(self):
        a = observation(); a['clinical_qualified'] = True
        with self.assertRaises(ValueError): controller(a)


class ProbeControllerTests(unittest.TestCase):
    def test_first_probe_is_report_and_strict_gain_commits(self):
        c = controller(); request = c.propose()
        self.assertEqual(request['action'], 'report_probe')
        c.complete(request, observation(number=1, report='cxrmate', text='positive'))
        self.assertEqual(c.propose()['action'], 'stop')
        self.assertEqual(c.result()['selected_candidate_id'], 'triple_1')
        self.assertEqual(c.result()['simulated_calls'], 6)

    def test_failed_report_gain_changes_next_branch_without_truth_attribution(self):
        c = controller(observation(image='negative', text='positive'))
        req = c.propose()
        c.complete(req, observation(number=1, report='cxrmate', image='negative', text='negative'))
        self.assertEqual(c.current['candidate_id'], 'triple_0')
        req = c.propose()
        self.assertEqual(req['action'], 'image_probe')
        self.assertEqual(req['reason'], 'observed_report_probe_failed_try_image')
        c.complete(req, observation(model='roentgen', number=2, text='positive'))
        result = c.result()
        self.assertEqual(result['selected_candidate_id'], 'triple_2')
        self.assertEqual(result['simulated_calls'], 10)
        self.assertIsNone(result['clinical_fault_location'])

    def test_no_feedback_ablation_keeps_report_order(self):
        c = controller(observation(image='negative', text='positive'), feedback_enabled=False)
        req = c.propose()
        c.complete(req, observation(number=1, report='cxrmate', image='negative', text='negative'))
        self.assertEqual(c.propose()['action'], 'report_probe')

    def test_unknown_ehr_never_authorizes_image_branch_from_report_votes(self):
        c = controller(observation(ehr='unknown'))
        for report, number in [('cxrmate', 1), ('llava', 2)]:
            req = c.propose()
            self.assertEqual(req['action'], 'report_probe')
            c.complete(req, observation(number=number, report=report, ehr='unknown'))
        self.assertEqual(c.propose()['action'], 'stop')
        self.assertEqual(c.result()['simulated_calls'], 8)

    def test_failed_execution_is_reserved_and_not_free(self):
        c = controller(budget=6); req = c.propose()
        c.complete(req, failure_type='worker_failed')
        self.assertEqual(c.result()['simulated_calls'], 6)
        self.assertEqual(c.propose()['action'], 'stop')
        self.assertFalse(c.result()['history'][0]['accepted'])

    def test_budget_four_keeps_initial_candidate(self):
        c = controller(budget=4)
        self.assertEqual(c.propose()['action'], 'stop')
        self.assertEqual(c.result()['selected_candidate_id'], 'triple_0')

    def test_request_is_reserved_before_completion(self):
        c = controller(); req = c.propose()
        self.assertEqual(c.spent, 6)
        with self.assertRaises(RuntimeError): c.propose()
        with self.assertRaises(RuntimeError): c.result()
        self.assertEqual(req['parent_observation_sha256'], p.digest(c.current))

    def test_tampered_request_or_foreign_observation_rejected(self):
        c = controller(); req = c.propose(); bad = deepcopy(req); bad['slot'][2] = 'llava'
        with self.assertRaises(ValueError): c.complete(bad, observation(number=1, report='llava'))
        with self.assertRaises(ValueError): c.complete(req, observation(number=1, report='llava'))
        self.assertIsNotNone(c.pending)

    def test_input_and_result_are_not_mutable_aliases(self):
        a = observation(); c = controller(a); a['states']['ehr']['edema'] = 'negative'
        self.assertEqual(c.current['states']['ehr']['edema'], 'positive')
        result = c.result(); result['selected_observation']['states']['ehr']['edema'] = 'unknown'
        self.assertEqual(c.current['states']['ehr']['edema'], 'positive')

    def test_initial_registration_budget_and_failure_schema(self):
        for budget in (True, 3, -1, 4.5):
            with self.assertRaises(ValueError): controller(budget=budget)
        with self.assertRaises(ValueError): controller(observation(report='foreign'))
        c = controller(); req = c.propose()
        with self.assertRaises(ValueError): c.complete(req)

    def test_invalid_image_can_request_image_probe_without_disease_truth(self):
        a = observation(ehr='unknown', image='unknown', text='unknown')
        a['quality']['cxr_basic_validity_pass'] = False
        c = controller(a)
        self.assertEqual(c.propose()['action'], 'image_probe')

    def test_deterministic_action_history_and_selection(self):
        results = []
        for _ in range(2):
            c = controller(); req = c.propose()
            c.complete(req, observation(number=1, report='cxrmate', text='positive'))
            results.append(c.result())
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
