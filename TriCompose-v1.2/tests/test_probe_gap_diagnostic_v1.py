"""Invented four-state fixtures only; explain rejections without changing them."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import diagnose_probe_gap_v1 as gap
from test_probe_repair_v1 import observation


def fixture(*, negative_only=False, unrequested=False):
    initial = observation(ehr='unknown', text='unknown')
    after = observation(ehr='unknown', report='cxrmate', number=1, text='positive')
    if negative_only:
        for v in (initial, after):
            v['states']['xrv']['edema'] = 'negative'
            v['states']['xrv']['lung_opacity'] = 'positive'
        after['states']['chexbert']['edema'] = 'negative'
    policy = gap.p.ProbeController(initial, [('sana', 0), ('roentgen', 0)], ['maira', 'cxrmate'], 6)
    if not unrequested:
        request = policy.propose(); policy.complete(request, after)
    probe = policy.result(); probe['method'] = gap.benchmark.NEW_METHODS[0]
    static = {'case_id': 'case_000', 'model_call_budget': 6, 'method': 'static_rerank',
        'selected_candidate_id': after['candidate_id'], 'simulated_model_calls': 6,
        'actual_regeneration_executed': False}
    views = {v['candidate_id']: v for v in (initial, after)}
    return probe, static, views, initial['candidate_id']


class ProbeGapTests(unittest.TestCase):
    def test_accepted_choice_is_same_final(self):
        args = fixture(); old = deepcopy(args); row = gap.diagnose(*args)
        self.assertEqual(row['static_choice_status'], 'same_final'); self.assertEqual(args, old)
        self.assertFalse(row['static_total_support_is_clinical_superiority'])

    def test_negative_only_gain_is_observed_but_rejected(self):
        row = gap.diagnose(*fixture(negative_only=True))
        self.assertEqual(row['static_choice_status'], 'observed_rejected')
        self.assertIn('no_strict_target_action_gain', row['static_rejection_reasons'])
        self.assertEqual(row['static_minus_probe_cxr_report_positive_support'], 0)
        self.assertEqual(row['static_minus_probe_cxr_report_negative_support'], 1)
        self.assertTrue(row['static_has_higher_raw_cxr_report_support'])

    def test_unrequested_candidate_is_not_relabelled_rejected(self):
        row = gap.diagnose(*fixture(unrequested=True))
        self.assertEqual(row['static_choice_status'], 'not_requested')
        self.assertEqual(row['static_rejection_reasons'], '')
        self.assertFalse(row['diagnostic_unseen_choices_are_routing_inputs'])

    def test_initial_superseded_is_not_a_rejected_probe(self):
        args = fixture(); args[1]['selected_candidate_id'] = args[3]
        row = gap.diagnose(*args)
        self.assertEqual(row['static_choice_status'], 'initial_superseded')
        self.assertEqual(row['static_rejection_reasons'], '')

    def test_tampered_credit_and_committed_id_fail(self):
        for kind in ('credit', 'commit', 'charge'):
            args = fixture(); h = args[0]['history'][0]
            if kind == 'credit': h['credit']['target_gain_observed'] = False
            elif kind == 'commit': h['committed_candidate_id'] = 'another_candidate'
            else: h['reserved_simulated_calls'] = 0
            with self.assertRaises(ValueError): gap.diagnose(*args)

    def test_cross_case_cross_ehr_budget_or_secondary_leak_fails(self):
        for kind in ('case', 'ehr', 'cap', 'secondary'):
            args = fixture()
            if kind == 'case': args[1]['case_id'] = 'case_001'
            elif kind == 'ehr': args[2]['triple_1']['lineage']['ehr_sha256'] = '9'*64
            elif kind == 'cap': args[1]['model_call_budget'] = 8
            else: args[0]['uses_secondary_endpoint'] = True
            with self.assertRaises(ValueError): gap.diagnose(*args)

    def test_missing_report_finding_does_not_become_negative_support(self):
        args = fixture(unrequested=True)
        args[2]['triple_1']['states']['chexbert']['edema'] = 'unknown'
        row = gap.diagnose(*args)
        self.assertEqual(row['static_minus_probe_cxr_report_support'], 0)

    def test_failed_probe_is_charged_but_not_observed(self):
        args = fixture(); initial = args[2][args[3]]
        c = gap.p.ProbeController(initial, [('sana', 0)], ['maira', 'cxrmate'], 6)
        request = c.propose(); c.complete(request, failure_type='worker_failed')
        args = list(args); args[0] = c.result(); args[0]['method'] = gap.benchmark.NEW_METHODS[0]
        row = gap.diagnose(*args)
        self.assertEqual(row['probe_simulated_calls'], 6)
        self.assertEqual(row['static_choice_status'], 'not_requested')

    def test_duplicate_attempt_cannot_fake_coverage(self):
        args = fixture(); args[0]['history'].append(deepcopy(args[0]['history'][0]))
        with self.assertRaises(ValueError): gap.diagnose(*args)

    def test_guard_before_cached_inputs_and_directory_creation(self):
        with patch.object(gap.benchmark, 'cpu_guard', side_effect=RuntimeError('invented_guard')), \
                patch.object(gap.benchmark, 'load_primary') as read, patch.object(gap, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): gap.run(None)
            read.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()
