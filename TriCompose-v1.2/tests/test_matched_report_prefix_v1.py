"""Invented inputs verify fixed scheduling and the unchanged action-credit rule."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from tricompose_v12 import matched_report_prefix_v1 as module
from test_probe_repair_v1 import observation
import benchmark_matched_report_prefix_v1 as benchmark


def runner(initial=None, cap=10):
    return module.FixedReportPrefix(initial or observation(), ['maira', 'cxrmate', 'llava'], cap)


class MatchedReportPrefixTests(unittest.TestCase):
    def test_no_need_stopping_signal_does_not_choose_the_fixed_schedule(self):
        initial = observation(ehr='unknown', text='positive')
        self.assertFalse(module.p.needs(initial)['report'])
        r = runner(initial); req = r.propose()
        self.assertEqual(req['slot'], ['sana', 0, 'cxrmate'])
        self.assertEqual(r.spent, 6)

    def test_gain_and_veto_are_identical_to_original_credit(self):
        initial = observation(); after = observation(report='cxrmate', number=1, text='positive')
        r = runner(initial); req = r.propose(); receipt = r.complete(req, after)
        self.assertEqual(receipt['credit'], module.p.action_credit(initial, after, 'report_probe'))
        self.assertTrue(receipt['accepted']); self.assertEqual(r.current['candidate_id'], after['candidate_id'])
        self.assertEqual(r.propose()['slot'][2], 'llava')

    def test_negative_only_gain_stays_rejected(self):
        initial = observation(ehr='unknown', image='negative', text='unknown')
        after = observation(ehr='unknown', image='negative', report='cxrmate', number=1, text='negative')
        r = runner(initial); req = r.propose(); receipt = r.complete(req, after)
        self.assertFalse(receipt['accepted'])
        self.assertIn('no_strict_target_action_gain', receipt['credit']['reasons'])

    def test_unknown_does_not_silence_a_conflict(self):
        r = runner(); req = r.propose()
        receipt = r.complete(req, observation(report='cxrmate', number=1, text='unknown'))
        self.assertFalse(receipt['accepted'])
        self.assertIn('conflict_silenced_not_resolved', receipt['credit']['reasons'])

    def test_quality_regression_remains_a_veto(self):
        r = runner(); req = r.propose(); after = observation(report='cxrmate', number=1, text='positive')
        after['quality']['report_structure_quality_score_0_1'] = .5
        self.assertIn('structure_proxy_regressed', r.complete(req, after)['credit']['reasons'])

    def test_budget_four_no_new_report_and_failures_are_not_free(self):
        self.assertEqual(runner(cap=4).propose()['action'], 'stop')
        r = runner(cap=6); req = r.propose(); r.complete(req, failure_type='worker_failed')
        self.assertEqual(r.result()['simulated_calls'], 6); self.assertEqual(r.propose()['action'], 'stop')

    def test_cannot_switch_images_or_ehrs(self):
        for kind in ('image', 'ehr'):
            r = runner(); req = r.propose(); after = observation(report='cxrmate', number=1, text='positive')
            if kind == 'image': after['lineage']['cxr_sha256'] = '9'*64
            else: after['lineage']['ehr_sha256'] = '9'*64
            with self.assertRaises(ValueError): r.complete(req, after)

    def test_unrequested_and_secondary_payloads_rejected(self):
        r = runner(); req = r.propose()
        with self.assertRaises(ValueError): r.complete(req, observation(report='llava', number=1))
        initial = observation(); initial['biovil'] = .9
        with self.assertRaises(ValueError): runner(initial)

    def test_pending_observation_cannot_be_skipped(self):
        r = runner(); r.propose()
        with self.assertRaises(RuntimeError): r.propose()
        with self.assertRaises(RuntimeError): r.result()

    def test_result_is_not_clinical_or_actual_regeneration(self):
        result = runner(cap=4).result()
        self.assertEqual(result['same_commit_guard_version'], module.p.VERSION)
        self.assertFalse(result['actual_regeneration_executed'])
        self.assertFalse(result['image_regeneration_allowed'])

    def test_guard_precedes_cache_access_and_output_creation(self):
        with patch.object(benchmark.b, 'cpu_guard', side_effect=RuntimeError('invented_guard')), \
                patch.object(benchmark.b, 'load_primary') as read, patch.object(benchmark, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): benchmark.run(None)
            read.assert_not_called(); write.assert_not_called()

    def test_endpoint_na_is_not_improvement_or_zero(self):
        view = observation()
        row = {key: view['lineage'][key] for key in ('ehr_sha256', 'ehr_facts_sha256',
            'cxr_sha256', 'report_sha256', 'cxr_candidate_id', 'report_candidate_id')}
        row.update(triple_candidate_id=view['candidate_id'], case_id=view['case_id'],
            calibrated=False, biovil_raw_cosine=None, status='not_available', reason='authored_missing')
        result = benchmark.endpoint_values([row], {view['candidate_id']: view})
        self.assertIsNone(result[view['candidate_id']])

    def test_endpoint_identity_calibration_and_numeric_fail_closed(self):
        view = observation()
        original = {key: view['lineage'][key] for key in ('ehr_sha256', 'ehr_facts_sha256',
            'cxr_sha256', 'report_sha256', 'cxr_candidate_id', 'report_candidate_id')}
        original.update(triple_candidate_id=view['candidate_id'], case_id=view['case_id'],
            calibrated=False, biovil_raw_cosine=.3, status='computed_secondary_uncalibrated', reason=None)
        for field, value in (('calibrated', True), ('case_id', 'case_001'),
                ('report_sha256', '9'*64), ('biovil_raw_cosine', float('nan')),
                ('biovil_raw_cosine', True)):
            row = dict(original); row[field] = value
            with self.assertRaises(ValueError): benchmark.endpoint_values([row], {view['candidate_id']: view})


if __name__ == '__main__': unittest.main()
