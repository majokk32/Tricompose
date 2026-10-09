"""Authored evidence only: privacy, ambiguity arithmetic and no clinical claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import diagnose_localization_observability_v1 as m
from test_probe_repair_v1 import observation


def packet(**kwargs):
    return {'item_id': 'item_0000', 'states': observation(**kwargs)['states']}


def resolver(view, item='item_0000'):
    line = view['lineage']
    return {'item_id': item, 'case_id': view['case_id'], 'benchmark_split': 'development',
        'ehr_sha256': line['ehr_sha256'], 'displayed_cxr_candidate_id': line['cxr_candidate_id'],
        'displayed_cxr_sha256': line['cxr_sha256'], 'displayed_report_candidate_id': line['report_candidate_id'],
        'displayed_report_sha256': line['report_sha256']}


class LocalizationObservabilityTests(unittest.TestCase):
    def test_named_states_and_metadata_only_packet(self):
        m.validate_packet(packet())
        for field in ('case_id', 'report_text', 'intervention_target', 'biovil', 'source_hash'):
            value = packet(); value[field] = 'invented_forbidden'
            with self.assertRaises(ValueError): m.validate_packet(value)

    def test_incomplete_or_invalid_states_fail_closed(self):
        for mode in ('missing', 'extra', 'invalid', 'type'):
            value = packet()
            if mode == 'missing': del value['states']['ehr']['edema']
            elif mode == 'extra': value['states']['ehr']['invented'] = 'unknown'
            elif mode == 'invalid': value['states']['ehr']['edema'] = False
            else: value['states']['ehr'] = False
            with self.assertRaises(ValueError): m.validate_packet(value)

    def test_report_signal_is_not_confirmed_clinical_fault(self):
        result = m.predict(packet())
        self.assertEqual(result['tentative_proxy_target'], 'report')
        self.assertIsNone(result['confirmed_faulty_modality'])
        self.assertFalse(result['clinical_qualified']); self.assertFalse(result['regeneration_authorized'])

    def test_image_signal_is_tentative_only(self):
        result = m.predict(packet(image='negative', text='positive'))
        self.assertEqual(result['tentative_proxy_target'], 'cxr')
        self.assertIsNone(result['confirmed_faulty_modality'])

    def test_no_direct_ehr_forces_abstention(self):
        value = packet(ehr='unknown')
        result = m.predict(value)
        self.assertIsNone(result['tentative_proxy_target'])
        self.assertEqual(result['known_direct_fact_count'], 0)
        self.assertEqual(m.edge_counts(value)['ehr_report']['known'], 0)

    def test_uncertain_is_not_negative(self):
        result = m.predict(packet(image='uncertain'))
        self.assertIsNone(result['tentative_proxy_target'])
        self.assertEqual(result['three_way_comparable_fact_count'], 0)

    def test_both_opposed_not_single_fault(self):
        result = m.predict(packet(image='negative', text='negative'))
        self.assertEqual(result['pattern'], 'abstain_both_modalities_oppose_ehr')
        self.assertIsNone(result['tentative_proxy_target'])

    def test_conflicting_locus_signals_abstain(self):
        value = packet()
        for k in ('ehr', 'chexbert'): value['states'][k]['cardiomegaly'] = 'positive'
        value['states']['xrv']['cardiomegaly'] = 'negative'
        result = m.predict(value)
        self.assertEqual(result['pattern'], 'conflicting_locus_signals')
        self.assertIsNone(result['tentative_proxy_target'])

    def test_item_identity_never_changes_signature(self):
        a = packet(); z = deepcopy(a); z['item_id'] = 'item_0123'
        for feature in m.FEATURES: self.assertEqual(m.signature(a, feature), m.signature(z, feature))

    def test_counts_can_collide_while_named_facts_differ(self):
        a = packet(); z = deepcopy(a)
        for values in z['states'].values(): values['pleural_effusion'], values['edema'] = values['edema'], 'unknown'
        self.assertEqual(m.signature(a, 'edge_counts_only'), m.signature(z, 'edge_counts_only'))
        self.assertNotEqual(m.signature(a, 'named_finding_states'), m.signature(z, 'named_finding_states'))

    def test_unknown_and_explicit_negative_different_signature(self):
        a = packet(ehr='unknown'); z = packet(ehr='negative')
        for feature in m.FEATURES: self.assertNotEqual(m.signature(a, feature), m.signature(z, feature))

    def test_single_observation_three_targets_is_ambiguous(self):
        rows = [{'signature': 'same', 'target': target} for target in m.TARGETS]
        result, cells = m.collision_summary(rows, 'named_finding_states', 'overall')
        self.assertAlmostEqual(result['empirical_target_reconstruction_upper_bound'], 1/3)
        self.assertAlmostEqual(result['empirical_balanced_target_reconstruction_upper_bound'], 1/3)
        self.assertEqual(result['unavoidable_empirical_mechanical_target_errors'], 2)
        self.assertEqual(result['items_in_mixed_target_signatures'], 3)
        self.assertEqual(len(cells), 1)
        self.assertIsNone(result['clinical_localization_accuracy'])

    def test_balanced_bound_not_majority_accuracy(self):
        rows = [{'signature': 'a', 'target': target} for target in ('none', 'none', 'report')]
        rows += [{'signature': 'b', 'target': 'cxr'} for _ in range(4)]
        result, _ = m.collision_summary(rows, 'edge_counts_only', 'overall')
        self.assertAlmostEqual(result['empirical_target_reconstruction_upper_bound'], 6/7)
        self.assertAlmostEqual(result['empirical_balanced_target_reconstruction_upper_bound'], 2/3)

    def test_missing_class_or_empty_is_na_not_zero(self):
        result, _ = m.collision_summary([{'signature': 'a', 'target': 'none'}], 'edge_counts_only', 'overall')
        self.assertIsNone(result['empirical_balanced_target_reconstruction_upper_bound'])
        empty, _ = m.collision_summary([], 'edge_counts_only', 'overall')
        self.assertIsNone(empty['empirical_target_reconstruction_upper_bound'])

    def test_unique_mechanical_signatures_not_clinical_accuracy(self):
        result, _ = m.collision_summary([{'signature': target, 'target': target} for target in m.TARGETS], 'named_finding_states', 'overall')
        self.assertEqual(result['empirical_target_reconstruction_upper_bound'], 1)
        self.assertIsNone(result['clinical_localization_accuracy'])
        self.assertFalse(result['new_policy_trained']); self.assertFalse(result['bound_applies_to_arbitrary_new_cases'])

    def test_bad_target_or_projection_rejected(self):
        with self.assertRaises(ValueError): m.collision_summary([{'signature': 'a', 'target': 'clinical_gold'}], 'named_finding_states', 'overall')
        with self.assertRaises(ValueError): m.signature(packet(), 'artifact_hashes')

    def test_hash_bound_source_packet(self):
        a = observation(); bank = {'case_000': {('sana', 0, 'maira'): a}}
        with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
            result = m.bind_packets(bank, [resolver(a)])
        self.assertEqual(result, [packet()])

    def test_changed_artifact_hash_or_duplicate_item_rejected(self):
        a = observation(); bank = {'case_000': {('sana', 0, 'maira'): a}}
        for mode in ('hash', 'duplicate'):
            row = resolver(a); rows = [row]
            if mode == 'hash': row['displayed_cxr_sha256'] = '9' * 64
            else: rows.append(row)
            with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
                with self.assertRaises(ValueError): m.bind_packets(bank, rows)

    def test_cross_case_donor_cannot_replace_ehr(self):
        a = observation(); z = observation(model='roentgen', number=1, ehr='negative', image='negative')
        z['case_id'] = 'case_001'; z['lineage']['ehr_sha256'] = '9' * 64
        row = resolver(a); row['displayed_cxr_candidate_id'] = z['lineage']['cxr_candidate_id']; row['displayed_cxr_sha256'] = z['lineage']['cxr_sha256']
        with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
            result = m.bind_packets({'case_000': {0: a}, 'case_001': {0: z}}, [row])[0]
        self.assertEqual(result['states']['ehr']['edema'], 'positive')
        self.assertEqual(result['states']['xrv']['edema'], 'negative')

    def test_claim_or_training_protocol_changes_fail_closed(self):
        original = json.loads(m.PROTOCOL.read_text()); m.validate_protocol(original)
        for key in ('clinical_qualified', 'training_allowed', 'regeneration_authorized',
                'mechanical_target_is_clinical_fault', 'construction_labels_are_independent_evaluation',
                'secondary_endpoint_used_for_prediction', 'historical_final_role_is_untouched_test'):
            value = deepcopy(original); value[key] = True
            with self.assertRaises(ValueError): m.validate_protocol(value)

    def test_cpu_guard_precedes_all_inputs_and_writes(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('authored')), \
                patch.object(m.b, 'load_primary') as read, patch.object(m, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): m.run(None)
            read.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()
