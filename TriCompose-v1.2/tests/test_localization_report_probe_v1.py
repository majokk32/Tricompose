"""Invented text-free facts only; a second report never becomes clinical gold."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_localization_report_probe_v1 as m
from test_probe_repair_v1 import observation
from test_localization_observability_v1 import packet, resolver


def probe(packet_value=None, state='positive', distinct=True):
    value = deepcopy((packet_value or packet())['states']['chexbert'])
    value['edema'] = state
    return {'report_states': value, 'distinct_report_bytes': distinct, 'same_image_binding_verified': True}


class ReportProbeTests(unittest.TestCase):
    def test_requests_depend_on_explicit_ehr_not_target_key(self):
        for state, anchored in (('positive', True), ('negative', True), ('uncertain', False), ('unknown', False)):
            value = packet(ehr=state)
            self.assertEqual(m.wants_probe(value, 'anchored_report_probe'), anchored)
            self.assertTrue(m.wants_probe(value, 'uniform_report_probe'))
            self.assertFalse(m.wants_probe(value, 'no_report_probe'))

    def test_unknown_ehr_cannot_gain_target_from_report_consensus(self):
        value = packet(ehr='unknown')
        result = m.verify(value, 'uniform_report_probe', probe(value))
        self.assertIsNone(result['corroborated_proxy_target'])
        self.assertEqual(result['probe_status'], 'observed_without_unique_initial_target')

    def test_report_target_corroborated_only_by_ehr_matching_probe(self):
        result = m.verify(packet(), 'uniform_report_probe', probe())
        self.assertEqual(result['corroborated_proxy_target'], 'report')
        self.assertEqual(result['corroborated_fact_ids'], ['edema'])
        self.assertIsNone(result['confirmed_faulty_modality'])
        self.assertFalse(result['clinical_qualified']); self.assertFalse(result['regeneration_authorized'])

    def test_cxr_target_is_only_dependent_proxy(self):
        value = packet(image='negative', text='positive')
        result = m.verify(value, 'anchored_report_probe', probe(value))
        self.assertEqual(result['corroborated_proxy_target'], 'cxr')
        self.assertFalse(result['reports_are_independent_clinical_evidence'])

    def test_probe_challenges_report_target(self):
        result = m.verify(packet(), 'uniform_report_probe', probe(state='negative'))
        self.assertEqual(result['probe_status'], 'challenged_or_mixed')
        self.assertEqual(result['challenged_fact_ids'], ['edema'])
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_probe_challenges_cxr_target(self):
        value = packet(image='negative', text='positive')
        result = m.verify(value, 'uniform_report_probe', probe(value, 'negative'))
        self.assertEqual(result['probe_status'], 'challenged_or_mixed')

    def test_unknown_or_uncertain_probe_never_resolves_target(self):
        for state in ('unknown', 'uncertain'):
            result = m.verify(packet(), 'uniform_report_probe', probe(state=state))
            self.assertEqual(result['probe_status'], 'unconfirmed_missing_explicit_evidence')
            self.assertIsNone(result['corroborated_proxy_target'])

    def test_all_facts_required_not_equal_count_different_finding(self):
        value = packet()
        for name in ('ehr', 'xrv'): value['states'][name]['pleural_effusion'] = 'positive'
        value['states']['chexbert']['pleural_effusion'] = 'negative'
        result = m.verify(value, 'uniform_report_probe', probe(value))
        self.assertEqual(result['original_target_fact_ids'], ['edema', 'pleural_effusion'])
        self.assertEqual(result['probe_status'], 'challenged_or_mixed')
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_probe_on_different_finding_cannot_pay_for_unknown(self):
        value = probe(state='unknown'); value['report_states']['pneumothorax'] = 'positive'
        result = m.verify(packet(), 'uniform_report_probe', value)
        self.assertEqual(result['uninformative_fact_ids'], ['edema'])
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_duplicate_bytes_do_not_count_as_verification(self):
        result = m.verify(packet(), 'uniform_report_probe', probe(distinct=False))
        self.assertEqual(result['probe_status'], 'duplicate_report_bytes')
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_unavailable_is_still_charged_not_unknown_success(self):
        result = m.verify(packet(), 'uniform_report_probe')
        self.assertEqual(result['probe_status'], 'unavailable')
        self.assertEqual(result['simulated_additional_report_charges'], 2)
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_unrequested_evidence_is_forbidden(self):
        with self.assertRaises(ValueError): m.verify(packet(), 'no_report_probe', probe())
        with self.assertRaises(ValueError): m.verify(packet(ehr='unknown'), 'anchored_report_probe', probe())

    def test_no_probe_preserves_initial_signal_and_zero_charge(self):
        result = m.verify(packet(), 'no_report_probe')
        self.assertEqual(result['initial_tentative_proxy_target'], 'report')
        self.assertEqual(result['simulated_additional_report_charges'], 0)
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_probe_cannot_create_original_target_for_unmentioned_report(self):
        value = packet(text='unknown')
        result = m.verify(value, 'uniform_report_probe', probe(value, 'negative'))
        self.assertIsNone(result['initial_tentative_proxy_target'])
        self.assertIsNone(result['corroborated_proxy_target'])

    def test_conflicting_original_loci_remain_unresolved(self):
        value = packet()
        for name in ('ehr', 'chexbert'): value['states'][name]['pneumothorax'] = 'positive'
        value['states']['xrv']['pneumothorax'] = 'negative'
        self.assertIsNone(m.verify(value, 'uniform_report_probe', probe(value))['corroborated_proxy_target'])

    def test_probe_packet_forbids_payload_hash_key_and_invalid_parent(self):
        for field in ('report_text', 'intervention_target', 'hash'):
            value = probe(); value[field] = 'invented'
            with self.assertRaises(ValueError): m.verify(packet(), 'uniform_report_probe', value)
        for bad in (False, None, 1):
            value = probe(); value['same_image_binding_verified'] = bad
            with self.assertRaises(ValueError): m.verify(packet(), 'uniform_report_probe', value)

    def test_invalid_four_state_vector_fails_closed(self):
        value = probe(); value['report_states']['edema'] = 'absent'
        with self.assertRaises(ValueError): m.verify(packet(), 'uniform_report_probe', value)

    def test_predictor_does_not_mutate_fixed_inputs(self):
        value, extra = packet(), probe(); original, before = deepcopy(value), deepcopy(extra)
        m.verify(value, 'uniform_report_probe', extra)
        self.assertEqual(value, original); self.assertEqual(extra, before)

    def test_maira_parent_is_displayed_donor_image_not_original_case(self):
        base = observation(report='chexagent2')
        donor = observation(model='other', report='maira2', number=1, ehr='negative', image='negative', text='negative')
        row = resolver(base); row['displayed_cxr_candidate_id'] = donor['lineage']['cxr_candidate_id']; row['displayed_cxr_sha256'] = donor['lineage']['cxr_sha256']
        value = packet(image='negative')
        with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
            store = m.probe_store({'case_000': {0: base}, 'case_001': {0: donor}}, [row], [value])
        self.assertEqual(store['item_0000']['report_states']['edema'], 'negative')
        self.assertEqual(value['states']['ehr']['edema'], 'positive')

    def test_missing_or_wrong_image_probe_rejected(self):
        view = observation(report='maira2')
        for kind in ('missing', 'hash', 'states'):
            row = resolver(view)
            if kind == 'missing': row['displayed_cxr_candidate_id'] = 'invented_missing'
            elif kind == 'hash': row['displayed_cxr_sha256'] = '9'*64
            value = packet(image='negative') if kind == 'states' else packet()
            with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
                with self.assertRaises(ValueError): m.probe_store({'case_000': {0: view}}, [row], [value])

    def test_initial_report_expert_or_hash_cannot_change(self):
        base = observation(report='chexagent2')
        extra = observation(report='maira2', number=1)
        for kind in ('expert', 'hash'):
            initial = deepcopy(base); row = resolver(base)
            if kind == 'expert': initial['lineage']['report_model_id'] = 'maira2'
            else: row['displayed_report_sha256'] = '9'*64
            with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
                with self.assertRaises(ValueError): m.probe_store({'case_000': {0: initial, 1: extra}}, [row], [packet()])

    def test_protocol_mutations_and_undeclared_methods_rejected(self):
        protocol = json.loads(m.PROTOCOL.read_text()); m.validate_protocol(protocol)
        for key in ('clinical_qualified', 'can_create_a_new_target_from_probe', 'reports_are_independent_clinical_evidence', 'training_allowed'):
            bad = deepcopy(protocol); bad[key] = True
            with self.assertRaises(ValueError): m.validate_protocol(bad)
        with self.assertRaises(ValueError): m.wants_probe(packet(), 'choose_best_expert')

    def test_guard_precedes_source_read_and_write(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('fixture')), patch.object(m.b, 'load_primary') as read, patch.object(m, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): m.run(None)
            read.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()
