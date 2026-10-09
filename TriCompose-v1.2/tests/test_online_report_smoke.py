"""Invented chronological workers only. No real generation, weights or bodies."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import online_report_smoke as online
from test_fresh_output_acceptance import fixture, reseal, book_for
from tricompose_v12.execution_ledger import CallRequest, CallResult, restore_ledger
from contracts import private_directory, write_private_json


def examples(index=0, *, stop=False, image_opposition=False):
    base, ctx = fixture(known=not stop, report_state='positive' if stop else 'unknown',
        image_state='negative' if image_opposition else 'positive',
        image_id=f'fixture_online_image_{index}', report_id=f'fixture_online_base_{index}', seed=2)
    alt, _ = fixture(known=not stop, report_state='positive', image_state='negative' if image_opposition else 'positive',
        image_id=f'fixture_online_image_{index}', report_id=f'fixture_online_alt_{index}', model='chexagent2', seed=2)
    case_id = f'fixture_online_case_{index}'
    ctx['anchor']['case_id'] = case_id
    anchor = online.gate.anchor_from_record(ctx['anchor'])
    for r in (base, alt):
        r['case_id'] = r['receipt']['case_id'] = r['structure']['case_id'] = case_id
        r['receipt']['ehr_anchor_sha256'] = anchor.sha256
        reseal(r)
    return base, alt, ctx


def full_book(base, alt, ctx, prefix):
    anchor = online.gate.anchor_from_record(ctx['anchor'])
    ledger = restore_ledger(prefix['events'], case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=prefix['call_budget'], max_retries=prefix['max_retries'],
        execution_mode=prefix['execution_mode'], sink=lambda e: None)
    req = CallRequest('fixture_second_report', anchor.case_id, anchor.sha256, 'report_generator',
        alt['report_model_id'], online._digest('fixture_report_audit'), 2,
        'fixture_op_0_xrv', base['cxr_sha256'])
    token = ledger.reserve(req); ledger.complete(token, CallResult(alt['report_sha256']), elapsed_seconds=.1)
    req = CallRequest('fixture_second_labels', anchor.case_id, anchor.sha256, 'chexbert', 'chexbert',
        online._digest('fixture_label_audit'), 2, req.operation_id, base['cxr_sha256'], alt['report_sha256'])
    token = ledger.reserve(req)
    ledger.complete(token, CallResult(alt['receipt']['chexbert_labels_sha256'], alt['receipt']['receipt_id']), elapsed_seconds=.1)
    return ledger.snapshot()


class OnlineReportRouteTests(unittest.TestCase):
    def test_missing_positive_or_direct_fact_requests_second_report(self):
        base, _, ctx = examples()
        result = online.next_report_action(base, ctx)
        self.assertEqual(result['action'], 'switch_report_model')
        self.assertIn('missing_image_positive_report_comparison', result['reasons'])
        self.assertIn('missing_direct_ehr_report_comparison', result['reasons'])
        self.assertIsNone(result['clinical_fault_location'])

    def test_direct_image_opposition_stops_unresolved_not_more_reports(self):
        base, _, ctx = examples(image_opposition=True)
        result = online.next_report_action(base, ctx)
        self.assertEqual(result['action'], 'stop_unresolved_image_proxy')
        self.assertFalse(result['clinical_acceptance'])

    def test_report_image_opposition_switches_without_localization_claim(self):
        base, _, ctx = examples(stop=True)
        base['receipt']['fact_states'][3]['chexbert'] = 'negative'; reseal(base)
        result = online.next_report_action(base, ctx)
        self.assertEqual(result['action'], 'switch_report_model')
        self.assertIn('image_report_proxy_opposition', result['reasons'])

    def test_risk_and_repetition_triggers_are_explicit(self):
        for key, value in (('generic_report', True), ('unsupported_temporal_comparison_language', True),
                           ('repeated_sentence_count', 1), ('repeated_4gram_ratio', .1)):
            base, _, ctx = examples(stop=True); base['structure'][key] = value
            with self.subTest(key=key):
                self.assertEqual(online.next_report_action(base, ctx)['action'], 'switch_report_model')

    def test_section_contract_failure_requests_report(self):
        base, _, ctx = examples(stop=True)
        base['structure'].update(findings_complete=False, section_contract_pass=False)
        self.assertIn('baseline_section_contract_failed', online.next_report_action(base, ctx)['reasons'])

    def test_missing_negative_is_not_required_report_content(self):
        base, _, ctx = examples(stop=True)
        base['receipt']['fact_states'][0]['xrv'] = 'negative'; reseal(base)
        self.assertEqual(online.next_report_action(base, ctx)['action'], 'stop_unverified')

    def test_underconditioned_ehr_stays_and_no_unknown_becomes_negative(self):
        base, _, ctx = examples(stop=True)
        result = online.next_report_action(base, ctx)
        self.assertEqual(result['action'], 'stop_unverified')
        self.assertIsNone(base['raw_edge_readouts']['ehr_cxr']['support_over_known'])

    def test_endpoint_and_old_winner_do_not_route(self):
        base, _, ctx = examples(stop=True); before = online.next_report_action(base, ctx)
        base.update(biovil_raw_cosine=-1, old_winner=False, ranking_score=-100)
        self.assertEqual(before, online.next_report_action(base, ctx))

    def test_route_does_not_modify_fixed_inputs(self):
        base, _, ctx = examples(); before = copy.deepcopy((base, ctx))
        online.next_report_action(base, ctx)
        self.assertEqual((base, ctx), before)

    def test_seed_is_preregistered_not_chosen_by_endpoint(self):
        self.assertEqual(online.POLICY['cxr_seed'], 2)
        self.assertEqual(online.POLICY['maximum_image_regenerations'], 0)
        self.assertEqual(online.POLICY['call_budget_per_case'], 6)
        self.assertFalse(online.POLICY['actual_gpu_savings_claim_allowed'])

    def test_gpu_guard_precedes_load_or_cuda_import(self):
        with patch('online_report_smoke.require_gpu_slurm', side_effect=RuntimeError('fixture_gpu_required')), \
             patch('online_report_smoke.load') as load:
            with self.assertRaises(RuntimeError): online.run(argparse.Namespace())
            load.assert_not_called()

    def test_cpu_guard_precedes_checkpoint_preflight(self):
        with patch('online_report_smoke.gate.cpu_guard', side_effect=RuntimeError('fixture_cpu_required')), \
             patch('online_report_smoke.load_parent') as load:
            with self.assertRaises(RuntimeError): online.prepare(argparse.Namespace())
            load.assert_not_called()

    def test_vendor_bootstrap_is_present_and_no_model_training(self):
        text = Path(online.__file__).read_text()
        self.assertIn("sys.path.insert(0, str(VENDOR))", text)
        self.assertNotIn('model.train(', text)
        self.assertNotIn('torchrun', text)


class OnlineReportChronologyTests(unittest.TestCase):
    def run_fixture(self, *, stop=False, image_opposition=False, fail_initial=False, fail_second=False,
                    better_static_negative=False, endpoint_na=False):
        rows = {}; contexts = {}; cases = []
        for i in range(2):
            base, alt, ctx = examples(i, stop=stop, image_opposition=image_opposition)
            if better_static_negative:
                for r in (base, alt): r['receipt']['fact_states'][0]['xrv'] = 'negative'
                alt['receipt']['fact_states'][0]['chexbert'] = 'negative'
                for r in (base, alt): reseal(r)
            case_id = base['case_id']; rows[case_id] = (base, alt); contexts[case_id] = ctx
            cases.append({'case_id': case_id, 'anchor': ctx['anchor']})
        plan = {'cases': cases, 'workers': {'xrv': {'thresholds': {n: {'enabled': n in contexts[cases[0]['case_id']]['enabled_xrv_findings']} for n in online.gate.FINDINGS},
            'thresholds_sha256': contexts[cases[0]['case_id']]['thresholds_sha256'],
            'checkpoint_sha256': contexts[cases[0]['case_id']]['xrv_checkpoint_sha256']},
            'chexbert': {'checkpoint_sha256': contexts[cases[0]['case_id']]['chexbert_checkpoint_sha256']}},
            'minimum_gpu_vram_gib': 24, 'source_pins': {}, 'artifact_pins': {}}
        calls = []
        def prefix(case, plan, root):
            base, _ = rows[case['case_id']]; cr = root / 'cases' / case['case_id']; private_directory(cr)
            book = book_for([base], contexts[case['case_id']])
            write_private_json(cr / 'fixture_prefix_ledger.json', book)
            if fail_initial:
                return [], book, []
            return [{'case_id': case['case_id'], '_fixture_row': base}], book, []
        def second(case, plan, root, triple, prefix_book):
            base, alt = rows[case['case_id']]
            seal = root / 'cases' / case['case_id'] / 'online_selection.json'
            calls.append({'case_id': case['case_id'], 'online_sealed_before_second': seal.exists()})
            if fail_second:
                anchor = online.gate.anchor_from_record(contexts[case['case_id']]['anchor'])
                ledger = restore_ledger(prefix_book['events'], case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
                    call_budget=prefix_book['call_budget'], max_retries=prefix_book['max_retries'],
                    execution_mode=prefix_book['execution_mode'], sink=lambda e: None)
                req = CallRequest('fixture_failed_second', anchor.case_id, anchor.sha256, 'report_generator',
                    'chexagent2', online._digest('fixture_audit'), 2, 'fixture_op_0_xrv', base['cxr_sha256'])
                token = ledger.reserve(req); ledger.fail(token, error_code='runtime_exception', retryable=False, elapsed_seconds=.1)
                return [], ledger.snapshot()
            return [{'case_id': case['case_id'], '_fixture_row': alt}], full_book(base, alt, contexts[case['case_id']], prefix_book)
        def endpoint(root, plan, rs, triples, sealed, journal):
            self.assertEqual(online.sha256_file(root / 'selection.json'), sealed)
            return {'records': [{k: r[k] for k in PAIR_FIELDS} | {'biovil_raw_cosine': None if endpoint_na else
                (.9 if r['report_model_id'] == 'cxrmate_single' else .1)} for r in rs]}
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda n: SimpleNamespace(total_memory=32 * 1024**3)))
        with TemporaryDirectory(prefix='online_fixture_', dir=online.gate.BASE) as temp:
            output = Path(temp); os.chmod(output, 0o2770)
            args = argparse.Namespace(output_root=output, run_id='fixture_run', plan_manifest_sha256=online._digest('fixture_plan'))
            with patch('online_report_smoke.require_gpu_slurm'), patch('online_report_smoke.load', return_value=plan), \
                 patch.dict(sys.modules, {'torch': torch}), patch('online_report_smoke.one_case', side_effect=prefix), \
                 patch('online_report_smoke.additional_report', side_effect=second), \
                 patch('online_report_smoke.candidate_row', side_effect=lambda t: t['_fixture_row']), \
                 patch('online_report_smoke.secondary', side_effect=endpoint), patch('online_report_smoke.check_pins'):
                root, summary = online.run(args)
            selection = json.loads((root / 'selection.json').read_text())
            methods = json.loads((root / 'method_comparison.json').read_text())['records']
            journal = [json.loads(line) for line in (root / 'controller.journal.jsonl').read_text().splitlines()]
            return summary, selection, methods, journal, calls

    def test_live_escalation_costs_and_gate_are_measured_prefixes(self):
        summary, selection, methods, _, calls = self.run_fixture()
        self.assertEqual(summary['actual_collection_charged_attempts'], 12)
        self.assertEqual(summary['method_charged_attempts'], {'fixed': 8, 'online_report_escalation': 12, 'always_second_static': 12})
        self.assertEqual(summary['online_second_expert_cases'], 2)
        self.assertTrue(all(not c['online_sealed_before_second'] for c in calls))
        for case in selection['cases']:
            self.assertEqual(case['selections']['online_report_escalation'], case['selections']['always_second_static'])
        self.assertEqual(len(methods), 6)

    def test_early_stop_is_sealed_before_paid_shadow_control(self):
        summary, selection, _, journal, calls = self.run_fixture(stop=True)
        self.assertTrue(all(c['online_sealed_before_second'] for c in calls))
        self.assertEqual(summary['method_charged_attempts']['online_report_escalation'], 8)
        self.assertEqual(summary['actual_collection_charged_attempts'], 12)
        self.assertFalse(summary['actual_gpu_savings_demonstrated'])
        for case in selection['cases']:
            self.assertEqual(case['selections']['online_report_escalation'], case['selections']['fixed'])
            events = [e['stage'] for e in journal if e.get('case_id') == case['case_id']]
            self.assertLess(events.index('online_selection_sealed'), events.index('shadow_static_only'))

    def test_static_can_improve_while_online_stops_no_hidden_control_access(self):
        summary, selection, _, _, _ = self.run_fixture(stop=True, better_static_negative=True)
        for case in selection['cases']:
            self.assertNotEqual(case['selections']['online_report_escalation'], case['selections']['always_second_static'])
            self.assertEqual(case['selections']['online_report_escalation'], case['selections']['fixed'])
        self.assertEqual(summary['online_second_expert_cases'], 0)

    def test_image_proxy_conflict_remains_unresolved_and_no_new_image_retry(self):
        summary, selection, _, _, calls = self.run_fixture(image_opposition=True)
        self.assertEqual(summary['online_second_expert_cases'], 0)
        self.assertTrue(all(c['online_sealed_before_second'] for c in calls))
        self.assertTrue(all(c['statuses']['online_report_escalation'] == 'stop_unresolved_image_proxy' for c in selection['cases']))

    def test_failed_second_model_keeps_charge_and_original_output(self):
        summary, selection, _, _, _ = self.run_fixture(fail_second=True)
        self.assertEqual(summary['actual_collection_charged_attempts'], 10)
        self.assertEqual(summary['method_charged_attempts']['online_report_escalation'], 10)
        for case in selection['cases']:
            self.assertEqual(case['selections']['online_report_escalation'], case['selections']['fixed'])
            self.assertEqual(case['static_decision']['cost']['failed_attempts'], 1)

    def test_missing_initial_candidate_is_retained_as_null_not_dropped(self):
        summary, selection, methods, _, calls = self.run_fixture(fail_initial=True)
        self.assertEqual(len(selection['cases']), 2); self.assertEqual(len(methods), 6)
        self.assertEqual(calls, [])
        self.assertTrue(all(r['selected_candidate_id'] is None for r in methods))
        self.assertFalse(summary['clinical_acceptance'])

    def test_biovil_is_measured_after_choices_not_used_to_pick_winner(self):
        _, selection, methods, journal, _ = self.run_fixture()
        online_rows = [r for r in methods if r['method'] == 'online_report_escalation']
        fixed_rows = [r for r in methods if r['method'] == 'fixed']
        self.assertTrue(all(r['biovil_raw_cosine'] == .1 for r in online_rows))
        self.assertTrue(all(r['biovil_raw_cosine'] == .9 for r in fixed_rows))
        self.assertEqual(journal[-1]['stage'], 'all_selections_sealed')
        self.assertFalse(selection['secondary_used'])

    def test_endpoint_missingness_stays_na(self):
        _, _, methods, _, _ = self.run_fixture(endpoint_na=True)
        self.assertTrue(all(r['biovil_raw_cosine'] is None for r in methods))

    def test_approved_plan_must_have_exact_policy(self):
        with self.assertRaises(ValueError): online.validate_plan({'schema_version': online.SCHEMA, 'policy': {}})

    def test_missing_and_available_edges_use_same_csv_columns(self):
        import csv, io
        base, _, _ = examples()
        a = {'case_id': 'fixture_available', 'method': 'fixed', 'raw_edge_readouts': base['raw_edge_readouts']}
        b = {'case_id': 'fixture_missing', 'method': 'fixed', 'raw_edge_readouts': None}
        parsed = list(csv.DictReader(io.StringIO(online.method_csv([a, b]))))
        self.assertEqual(set(parsed[0]), set(parsed[1]))
        self.assertEqual(parsed[1]['ehr_cxr_support_over_known'], 'NA')


class AdditionalFrozenReportWiringTests(unittest.TestCase):
    def test_actual_continuation_reuses_xrv_and_binds_two_new_operations(self):
        from test_live_receipts import data, H
        from tricompose_v12.execution_ledger import BoundedCallLedger
        anchor, image, il, report, tl = data()
        for n, entry in il['thresholds'].items(): entry['enabled'] = n in online.gate.FINDINGS[:8]
        workers = {'xrv': {'thresholds_sha256': H['thresholds'], 'checkpoint_sha256': H['xrv']},
            'chexbert': {'model_id': 'chexbert', 'checkpoint_sha256': H['chexbert']},
            'chexagent2': {'model_id': 'chexagent2'}}
        with TemporaryDirectory(prefix='continuation_fixture_', dir=online.gate.BASE) as temp:
            root = Path(temp); os.chmod(root, 0o2770)
            private_directory(root / 'cases'); cr = root / 'cases' / anchor.case_id
            private_directory(cr); private_directory(cr / 'operations')
            xr = cr / 'operations/xrv_0_a1'; private_directory(xr); private_directory(xr / 'scored')
            ip = write_private_json(xr / 'scored/cxr_finding_labels.json', il)
            image.update(model_id='roentgen_v2', seed=2)
            image['artifact']['path'] = str(root / 'invented_image_not_opened')
            report['artifact']['path'] = str(root / 'invented_report_not_opened')
            report['model_id'] = 'chexagent2'
            partial = online.image_receipt(anchor, image, il, label_sha256=online.sha256_file(ip),
                thresholds_sha256=H['thresholds'], checkpoint_sha256=H['xrv'])
            captured = []
            ledger = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
                call_budget=6, max_retries=0, execution_mode='invented_fixture_no_models', sink=captured.append)
            parent = None
            for kind, op, artifact, receipt in (
                ('cxr_generator', 'cxr_0', H['image'], None),
                ('xrv', 'xrv_0', online.sha256_file(ip), partial['receipt_id']),
                ('report_generator', 'report_0_0', H['report'], None),
                ('chexbert', 'chexbert_0_0', H['report_labels'], online._digest('invented_initial_receipt'))):
                model = 'roentgen_v2' if kind == 'cxr_generator' else 'cxrmate_single' if kind == 'report_generator' else kind
                req = CallRequest(op, anchor.case_id, anchor.sha256, kind, model, online._digest(['fixture_audit', kind]),
                    2, parent, H['image'] if parent is not None else None, H['report'] if kind == 'chexbert' else None)
                token = ledger.reserve(req); ledger.complete(token, CallResult(artifact, receipt), elapsed_seconds=.1)
                parent = op
            prefix = ledger.snapshot(); before = copy.deepcopy(prefix)
            calls = []
            def mocked_attempt(book, req, spec, case_root, policy, validator, **inputs):
                calls.append(req.kind)
                token = book.reserve(req)
                op_root = case_root / 'operations' / (req.operation_id + '_a1'); private_directory(op_root)
                if req.kind == 'chexbert':
                    private_directory(op_root / 'scored')
                    write_private_json(op_root / 'scored/report_finding_labels.json', tl)
                result = validator({'output_root': str(op_root), 'worker_audit_sha256': req.frozen_model_audit_sha256}, req)
                book.complete(token, result, elapsed_seconds=.1)
                return result
            request_pack = root / 'fixture_request'; private_directory(request_pack)
            baseline = {'cxr_run': str(root / 'fixture_cxr_run'), 'cxr_sha256': H['image']}
            case = {'anchor': anchor.record()}
            with patch('online_report_smoke.load_cxr_candidates', return_value={image['candidate_id']: image}), \
                 patch('online_report_smoke.prepare_report_request_run', return_value={'run_directory': str(request_pack)}), \
                 patch('online_report_smoke.single_generated', return_value=(root / 'fixture_report_run', report)), \
                 patch('online_report_smoke.validate_report_binding') as binding, \
                 patch('online_report_smoke.attempt', side_effect=mocked_attempt):
                triples, full = online.additional_report(case, {'workers': workers}, root, baseline, prefix)
            self.assertEqual(calls, ['report_generator', 'chexbert'])
            self.assertEqual(prefix, before)
            self.assertEqual(full['charged_model_attempts'], 6)
            self.assertEqual(full['events'][:len(prefix['events'])], prefix['events'])
            self.assertEqual(len(triples), 1); binding.assert_called_once()
            saved = json.loads(Path(triples[0]['receipt_path']).read_text())
            self.assertEqual(saved['partial_receipt_id'], partial['receipt_id'])
            self.assertEqual(saved['xrv_labels_sha256'], online.sha256_file(ip))
            continuation = [json.loads(line) for line in (cr / 'continuation.journal.jsonl').read_text().splitlines()]
            self.assertEqual(full['events'], prefix['events'] + continuation)


PAIR_FIELDS = ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id',
    'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')


if __name__ == '__main__':
    unittest.main()
