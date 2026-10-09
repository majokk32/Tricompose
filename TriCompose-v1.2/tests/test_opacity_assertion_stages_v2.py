"""Wholly invented CPU contract tests, never model/clinical validation."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'interfaces'), str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
import opacity_assertion_stages_v2 as m
import opacity_assertion_controls_v2 as fixtures
import benchmark_opacity_assertion_stages_v2 as worker


def item(text='Possible lung opacity.'):
    return {'item_id': 'authored_test', 'text': text, 'report_sha256': m.digest(text), 'segments': m.segments(text)}


def returned(value, cap=False):
    return {'response': json.dumps(value), 'input_tokens': 30,
            'output_tokens': 128 if cap else 10, 'token_limit_reached': cap}


class SegmentContractTests(unittest.TestCase):
    def test_mechanical_boundaries_preserve_offsets(self):
        text = '  Lung opacity.  No effusion.\nNormal heart. '
        inv = m.segments(text)
        self.assertEqual([text[r['char_start']:r['char_end']] for r in inv],
                         ['Lung opacity.', 'No effusion.', 'Normal heart.'])

    def test_decimal_not_split(self):
        text = 'Opacity measures 1.5 cm. No effusion.'
        self.assertEqual(len(m.segments(text)), 2)

    def test_unicode_positions_not_byte_offsets(self):
        text = '合成。\nLung opacity.'
        self.assertEqual(m.segments(text)[1]['char_start'], len('合成。\n'))

    def test_repeated_sentence_has_distinct_ids(self):
        inv = m.segments('Lung opacity. Lung opacity.')
        self.assertEqual([r['segment_id'] for r in inv], [0, 1])
        self.assertEqual(inv[0]['quote_sha256'], inv[1]['quote_sha256'])
        self.assertNotEqual(inv[0]['char_start'], inv[1]['char_start'])

    def test_empty_and_nontext_refused(self):
        for text in ('', ' ', None):
            with self.subTest(text=text), self.assertRaises(ValueError): m.segments(text)

    def test_no_silent_source_truncation(self):
        with self.assertRaises(ValueError): m.segments('x' * 8193)

    def test_segment_cap_refused_not_shortened(self):
        with self.assertRaises(ValueError): m.segments('Sentence. ' * 65)

    def test_exact_inventory_hash_required(self):
        text = 'Lung opacity.'; inv = m.segments(text); inv[0]['quote_sha256'] = 'a' * 64
        with self.assertRaises(ValueError): m.locator_messages(text, inv)

    def test_inventory_deterministic_and_metadata_only(self):
        text = 'Lung opacity. No effusion.'
        self.assertEqual(m.segments(text), m.segments(text))
        self.assertTrue(all(set(r) == {'segment_id', 'char_start', 'char_end', 'quote_sha256', 'offset_unit'} for r in m.segments(text)))

    def test_locator_accepts_empty_without_negative(self):
        self.assertEqual(m.decode_locator('{"segment_ids":[]}', m.segments('Normal heart.')), [])
        self.assertEqual(m.reduce_states([]), 'unknown')

    def test_locator_bounded_ids(self):
        inv = m.segments('One. Two.')
        for selected in ([2], [-1], [True], [0.0], ['0'], [1, 0], [0, 0], None):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                m.decode_locator(json.dumps({'segment_ids': selected}), inv)

    def test_locator_exact_key_inventory(self):
        for response in ('[]', '{}', '{"segment_ids":[],"state":"negative"}'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                m.decode_locator(response, m.segments('Normal heart.'))

    def test_duplicate_json_keys_refused(self):
        with self.assertRaises(ValueError):
            m.decode_locator('{"segment_ids":[],"segment_ids":[0]}', m.segments('Lung opacity.'))

    def test_nonjson_no_substring_rescue(self):
        with self.assertRaises(ValueError):
            m.decode_locator('Here is {"segment_ids":[0]}', m.segments('Lung opacity.'))

    def test_exact_json_fence_supported(self):
        self.assertEqual(m.decode_locator('```json\n{"segment_ids":[0]}\n```', m.segments('Lung opacity.')), [0])

    def test_token_cap_unavailable_even_valid_json(self):
        with self.assertRaisesRegex(ValueError, 'token_limit_reached'):
            m.decode_locator('{"segment_ids":[]}', m.segments('Normal heart.'), token_limit_reached=True)

    def test_polarity_all_selected_ids_exhaustive(self):
        inv = m.segments('One. Two.')
        self.assertEqual(m.decode_polarity('{"assertions":[{"segment_id":1,"state":"uncertain"}]}', inv, [1]),
                         [{'segment_id': 1, 'state': 'uncertain'}])
        for rows in ([], [{'segment_id': 0, 'state': 'positive'}],
                     [{'segment_id': 1, 'state': 'positive'}] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                m.decode_polarity(json.dumps({'assertions': rows}), inv, [1])

    def test_polarity_rejects_invalid_state_or_extra_fields(self):
        inv = m.segments('Lung opacity.')
        for row in ({'segment_id': 0, 'state': 'absent'}, {'segment_id': 0, 'state': True},
                    {'segment_id': False, 'state': 'positive'},
                    {'segment_id': 0, 'state': 'positive', 'confidence': 1}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                m.decode_polarity(json.dumps({'assertions': [row]}), inv, [0])

    def test_polarity_duplicate_nested_keys_refused(self):
        with self.assertRaises(ValueError):
            m.decode_polarity('{"assertions":[{"segment_id":0,"state":"positive","state":"negative"}]}', m.segments('Lung opacity.'), [0])

    def test_opposed_segments_reduce_to_uncertain(self):
        self.assertEqual(m.reduce_states([{'state': 'positive'}, {'state': 'negative'}]), 'uncertain')

    def test_any_uncertain_retained(self):
        self.assertEqual(m.reduce_states([{'state': 'positive'}, {'state': 'uncertain'}]), 'uncertain')

    def test_unknown_never_negative(self):
        self.assertEqual(m.reduce_states([{'state': 'unknown'}]), 'unknown')
        self.assertEqual(m.reduce_states([{'state': 'unknown'}, {'state': 'positive'}]), 'positive')

    def test_source_references_have_no_quote_body(self):
        inv = m.segments('Lung opacity.')
        self.assertEqual(m.evidence(inv, [0]), inv)
        self.assertNotIn('quote', inv[0])

    def test_model_messages_have_only_source_and_instructions(self):
        inv = m.segments('Lung opacity.')
        for messages in (m.locator_messages('Lung opacity.', inv), m.polarity_messages('Lung opacity.', inv, [0])):
            self.assertEqual(len(messages), 1)
            self.assertEqual([r['type'] for r in messages[0]['content']], ['text'])
            for marker in ('expected_state', 'legacy_control_id', 'case_id', 'report_sha256', 'report_model_id'):
                self.assertNotIn(marker, messages[0]['content'][0]['text'])

    def test_untrusted_markers_are_not_formatted(self):
        text = 'Invented {segments} {selected} {report} text.'
        message = m.polarity_messages(text, m.segments(text), [0])[0]['content'][0]['text']
        self.assertIn(text, message)

    def test_empty_polarity_request_refused(self):
        with self.assertRaises(ValueError): m.polarity_messages('Normal heart.', m.segments('Normal heart.'), [])


class WorkerContractTests(unittest.TestCase):
    def test_48_unique_cases_12_families(self):
        rows = fixtures.cases()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r['text'] for r in rows}), 48)
        self.assertEqual(len({r['family'] for r in rows}), 12)
        for row in rows:
            m.validate_ids(row['relevant_segment_ids'], m.segments(row['text']))

    def test_all_six_known_controls_preserved_exactly(self):
        marked = {r['legacy_control_id']: r for r in fixtures.cases() if r['legacy_control_id']}
        for control in worker.legacy.controls():
            self.assertEqual(marked[control['control_id']]['text'], control['text'])
            self.assertEqual(marked[control['control_id']]['expected_state'], control['expected_state'])
        self.assertEqual(len(marked), 6)

    def test_fixed_call_cap(self):
        p = worker.POLICY
        self.assertEqual(p['max_model_calls'], p['legacy_primary_calls'] + p['locator_primary_calls'] + p['polarity_primary_max_calls'] + p['staged_replay_max_calls'])
        self.assertEqual(p['max_model_calls'], 148)

    def test_complete_empty_locator_skips_polarity_unknown(self):
        calls = []
        def call(messages, name, limit):
            calls.append(name); return returned({'segment_ids': []})
        out = worker.run_staged(item('Normal heart.'), call, 'test')
        self.assertEqual(out['state'], 'unknown')
        self.assertEqual(out['status'], 'complete')
        self.assertEqual(calls, ['test_locator'])
        self.assertEqual(out['stage_model_calls'], {'locator': 1, 'polarity': 0})

    def test_stage_a_failure_has_null_state_not_unknown(self):
        out = worker.run_staged(item(), lambda *args: returned({'segment_ids': [7]}), 'test')
        self.assertIsNone(out['state']); self.assertEqual(out['status'], 'failed_unavailable')
        self.assertIsNone(out['selected_segment_ids'])
        self.assertEqual(out['stage_model_calls']['polarity'], 0)

    def test_stage_b_failure_preserves_locator_not_successful_unknown(self):
        def call(messages, name, limit):
            return returned({'segment_ids': [0]}) if name.endswith('locator') else returned({'assertions': []})
        out = worker.run_staged(item(), call, 'test')
        self.assertIsNone(out['state'])
        self.assertEqual(out['selected_segment_ids'], [0])
        self.assertEqual(out['stage_model_calls'], {'locator': 1, 'polarity': 1})

    def test_model_exception_body_not_exported(self):
        def call(*args): raise RuntimeError('SECRET_SYNTHETIC_BODY_SENTINEL')
        out = worker.run_staged(item(), call, 'test')
        self.assertEqual(out['failure_reason'], 'RuntimeError')
        self.assertNotIn('SENTINEL', json.dumps(out))

    def test_stage_token_limit_not_retried(self):
        calls = []
        def call(messages, name, limit):
            calls.append(name); return returned({'segment_ids': [0]}, cap=True)
        out = worker.run_staged(item(), call, 'test')
        self.assertEqual(out['failure_reason'], 'token_limit_reached')
        self.assertEqual(len(calls), 1)

    def test_fixture_oracle_tests_contract_not_llm_accuracy(self):
        # Inject authored answers deliberately: verifies only plumbing/reducer.
        for case in fixtures.cases():
            def call(messages, name, limit):
                if name.endswith('locator'):
                    return returned({'segment_ids': case['relevant_segment_ids']})
                return returned({'assertions': [{'segment_id': i, 'state': case['expected_state']}
                    for i in case['relevant_segment_ids']]})
            out = worker.run_staged(item(case['text']), call, 'test')
            self.assertEqual(out['state'], case['expected_state'])
            self.assertFalse(out['semantic_correctness_verified'])

    def test_legacy_fail_has_null_state(self):
        out = worker.baseline_outcome('junk', 'Invented text.', {'token_limit_reached': False})
        self.assertIsNone(out['state'])

    def test_failure_denominator_and_error_metrics(self):
        rows = [{'expected_state': 'positive', 'status': 'failed_unavailable', 'state': None},
                {'expected_state': 'uncertain', 'status': 'complete', 'state': 'negative'},
                {'expected_state': 'negative', 'status': 'complete', 'state': 'positive'}]
        out = worker.metrics(rows)
        self.assertEqual(out['accuracy_all_attempted'], 0)
        self.assertEqual(out['unavailable'], 1)
        self.assertEqual(out['hard_positive_negative_flips'], 1)
        self.assertEqual(out['determinate_on_uncertain_unknown'], 1)

    def test_authored_gate_cannot_claim_clinical_validity(self):
        self.assertFalse(worker.POLICY['clinical_accuracy_verified'])
        self.assertFalse(worker.POLICY['primary_metric_eligible'])
        self.assertFalse(worker.POLICY['regeneration_authorized'])

    def test_source_input_not_mutated(self):
        source = item(); before = copy.deepcopy(source)
        worker.run_staged(source, lambda *args: returned({'segment_ids': []}), 'test')
        self.assertEqual(source, before)

    def test_login_guard_before_plan_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(worker, 'require_inside') as resolver:
            with self.assertRaises(RuntimeError):
                worker.evaluate(SimpleNamespace(allow_authored_text_benchmark=True))
            resolver.assert_not_called()

    def test_gpu_approval_guard_before_plan_access(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), patch.object(worker, 'require_inside') as resolver:
            with self.assertRaises(RuntimeError):
                worker.evaluate(SimpleNamespace(allow_authored_text_benchmark=False))
            resolver.assert_not_called()

    def test_existing_run_refused_before_model_access(self):
        with patch.object(worker.legacy.OP, 'guard'), patch.object(worker, 'require_inside', return_value=Path(__file__)), patch.object(worker.legacy, 'runtime_metadata') as runtime:
            with self.assertRaises(FileExistsError):
                worker.evaluate(SimpleNamespace(allow_authored_text_benchmark=True, output_root='fixture', run_id='fixture'))
            runtime.assert_not_called()

    def test_prepare_guard_before_fixture_or_runtime_use(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(worker, 'pinned_sources') as pins:
            with self.assertRaises(RuntimeError): worker.prepare(SimpleNamespace())
            pins.assert_not_called()

    def test_failed_prediction_cannot_pass_gate_as_unknown(self):
        refs, predictions = [], []
        for case in fixtures.cases():
            ref = {k: case[k] for k in ('item_id', 'family', 'expected_state', 'relevant_segment_ids', 'legacy_control_id')}
            ref['report_sha256'] = m.digest(case['text']); refs.append(ref)
            predictions.append({'item_id': ref['item_id'], 'report_sha256': ref['report_sha256'],
                'legacy': worker.unavailable('fixture_failure'), 'staged': worker.unavailable('fixture_failure')})
        out, checks = worker.score(refs, predictions, [])
        self.assertEqual(out['paired_authored_metrics']['staged']['unavailable'], 48)
        self.assertFalse(out['language_gate_passed'])
        self.assertTrue(out['language_gate_is_not_clinical_qualification'])
        self.assertEqual(len(checks), 96)

    def test_incomplete_authored_inventory_refused(self):
        with self.assertRaises(ValueError): worker.score([], [], [])


if __name__ == '__main__':
    unittest.main()
