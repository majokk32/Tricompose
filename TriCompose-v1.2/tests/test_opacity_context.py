"""Invented CPU metadata tests; no installed parser execution or clinical claim."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'benchmarks')]
from tricompose_v12 import opacity_context as m
from tricompose_v12 import report_assertions as old
import opacity_context_controls_v1 as fixtures
import benchmark_opacity_context as worker


def flags(**changes):
    values = dict.fromkeys(old.CONTEXT_FLAGS, False)
    values.update(changes)
    return values


def proposal(state='positive', count=1):
    return {'status': 'complete', 'state': state, 'failure_reason': None, 'literal_target_mentions': count}


def failure():
    return {'status': 'failed_unavailable', 'state': None, 'failure_reason': 'ValueError', 'literal_target_mentions': None}


class TargetTests(unittest.TestCase):
    def test_only_two_noun_forms_case_insensitive(self):
        text = 'Opacity and OPACITIES are mentioned.'
        self.assertEqual([text[r['char_start']:r['char_end']] for r in m.targets(text)], ['Opacity', 'OPACITIES'])

    def test_other_disease_is_not_opacity(self):
        self.assertEqual(m.targets('Pneumonia, edema and atelectasis.'), [])

    def test_generic_summary_does_not_expand_negatives(self):
        self.assertEqual(m.targets('No acute cardiopulmonary disease.'), [])
        self.assertEqual(m.aggregate([]), 'unknown')

    def test_plural_only_not_substrings(self):
        self.assertEqual(m.targets('opacification opacitylike microopacities'), [])

    def test_nonpulmonary_noun_limitation_is_exposed(self):
        self.assertEqual(len(m.targets('A corneal opacity is present.')), 1)

    def test_unicode_offsets_are_codepoints(self):
        text = '合成文字。 Opacity.'
        start = text.index('Opacity')
        self.assertEqual(m.targets(text)[0]['char_start'], start)
        self.assertNotEqual(start, len(text[:start].encode()))

    def test_repeated_nouns_retain_distinct_source_offsets(self):
        rows = m.targets('Opacity. Opacity.')
        self.assertEqual(rows[0]['quote_sha256'], rows[1]['quote_sha256'])
        self.assertNotEqual(rows[0]['char_start'], rows[1]['char_start'])

    def test_target_evidence_contains_no_text_or_quote(self):
        self.assertEqual(set(m.targets('Lung opacity.')[0]),
            {'char_start', 'char_end', 'quote_sha256', 'offset_unit'})

    def test_bounded_nonempty_text_no_truncation(self):
        for text in ('', ' ', None, 'x'*8193):
            with self.subTest(text_type=type(text).__name__), self.assertRaises(ValueError): m.targets(text)

    def test_invalid_span_or_boolean_offset_refused(self):
        for start, end in ((True, 2), (0, 0), (-1, 2), (0, 999)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError): m.span('opacity', start, end)


class FlagAndEvidenceTests(unittest.TestCase):
    def test_old_mapping_is_used_without_new_semantic_rules(self):
        self.assertIs(m.aggregate_context_mentions, old.aggregate_context_mentions)
        self.assertIs(m.context_state, old.context_state)

    def test_uncertainty_not_determinate(self):
        self.assertEqual(m.aggregate([{'flags': flags(is_uncertain=True)}]), 'uncertain')

    def test_noncurrent_flags_are_unknown(self):
        for key in ('is_historical', 'is_hypothetical', 'is_family'):
            with self.subTest(key=key): self.assertEqual(m.aggregate([{'flags': flags(**{key: True})}]), 'unknown')

    def test_opposed_current_states_reduce_to_uncertain(self):
        self.assertEqual(m.aggregate([{'flags': flags()}, {'flags': flags(is_negated=True)}]), 'uncertain')

    def test_history_does_not_erase_current_positive(self):
        self.assertEqual(m.aggregate([{'flags': flags(is_historical=True)}, {'flags': flags()}]), 'positive')

    def test_no_mentions_not_negative(self):
        self.assertEqual(m.aggregate([]), 'unknown')

    def test_serialized_unmodified_mention_is_unverified(self):
        text = 'Lung opacity.'
        target = m.targets(text)[0]
        entity = SimpleNamespace(start_char=target['char_start'], end_char=target['char_end'],
            label_='lung_opacity', _=SimpleNamespace(**flags(), modifiers=[]))
        r = m.serialize(SimpleNamespace(text=text, ents=[entity]), text)
        self.assertEqual(r['state'], 'positive')
        self.assertTrue(r['mentions'][0]['unmodified_positive_is_semantically_unverified'])
        self.assertFalse(r['semantic_scope_verified'])
        self.assertFalse(r['hard_action_eligible'])

    def test_changed_source_or_missing_entity_refused(self):
        with self.assertRaises(ValueError): m.serialize(SimpleNamespace(text='changed', ents=[]), 'Lung opacity.')
        with self.assertRaises(ValueError): m.serialize(SimpleNamespace(text='Lung opacity.', ents=[]), 'Lung opacity.')


class VetoTests(unittest.TestCase):
    def test_agreement_is_soft_only_not_clinical_vote(self):
        r = m.veto(proposal(), proposal())
        self.assertEqual(r['soft_retained_state'], 'positive')
        self.assertFalse(r['hard_action_eligible'])
        self.assertFalse(r['parsers_are_independent_clinical_votes'])

    def test_disagreement_does_not_flip_proposal(self):
        r = m.veto(proposal(), proposal('negative'))
        self.assertEqual(r['raw_state'], 'positive')
        self.assertIsNone(r['soft_retained_state'])
        self.assertEqual(r['decision'], 'context_state_disagreement')

    def test_unknown_uncertain_not_signed_comparisons(self):
        for state in ('unknown', 'uncertain'):
            with self.subTest(state=state): self.assertFalse(m.veto(proposal(state), proposal(state))['soft_comparable'])

    def test_context_missing_target_is_not_agreement(self):
        r = m.veto(proposal(), proposal('positive', count=0))
        self.assertEqual(r['decision'], 'context_no_literal_target')

    def test_failure_stays_unavailable_null(self):
        r = m.veto(proposal(), failure())
        self.assertIsNone(r['context_state'])
        self.assertEqual(r['decision'], 'context_unavailable')
        r = m.veto(failure(), proposal())
        self.assertIsNone(r['raw_state'])

    def test_placeholder_unknown_on_failure_refused(self):
        bad = failure(); bad['state'] = 'unknown'
        with self.assertRaises(ValueError): m.veto(bad, proposal())

    def test_no_case_dropped_or_winner_changed_any_state(self):
        for a in m.STATES:
            for b in m.STATES:
                r = m.veto(proposal(a), proposal(b))
                for key in ('candidate_dropped', 'state_corrected', 'selection_changed', 'regeneration_authorized'):
                    self.assertIs(r[key], False)

    def test_same_wrong_agreement_is_not_corrected(self):
        r = m.veto(proposal('negative'), proposal('negative'))
        self.assertEqual(r['soft_retained_state'], 'negative')
        self.assertFalse(r['independent_clinical_validation'])


class MetricsAndWorkerTests(unittest.TestCase):
    def test_all_attempts_including_failed_reference_support(self):
        r = m.metrics([{**proposal(), 'expected_state': 'positive'},
                       {**failure(), 'expected_state': 'negative'}])
        self.assertEqual(r['rows'], 2)
        self.assertEqual(r['complete'], 1)
        self.assertEqual(r['unavailable'], 1)
        self.assertEqual(r['macro_f1_present_classes'], 0.5)

    def test_uncertain_unknown_promotions_are_counted(self):
        r = m.metrics([{**proposal(), 'expected_state': 'uncertain'},
                       {**proposal('negative'), 'expected_state': 'unknown'}])
        self.assertEqual(r['determinate_on_uncertain_unknown'], 2)
        self.assertEqual(r['hard_positive_negative_flips'], 0)

    def test_hard_flip_count_separate_from_missing(self):
        r = m.metrics([{**proposal('negative'), 'expected_state': 'positive'},
                       {**failure(), 'expected_state': 'positive'}])
        self.assertEqual(r['hard_positive_negative_flips'], 1)

    def test_64_fixed_unique_authored_cases_in_16_groups(self):
        rows = fixtures.cases()
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({r['text'] for r in rows}), 64)
        self.assertEqual(len({r['family'] for r in rows}), 16)
        self.assertTrue(all(r['expected_state'] in m.STATES for r in rows))

    def test_fixture_selection_not_driven_by_target_detection(self):
        inputs, references = worker.fresh_inputs()
        self.assertEqual(len(inputs), 64)
        self.assertEqual(len(references), 64)
        self.assertTrue(any(not m.targets(r['text']) for r in inputs))

    def test_prepared_model_inputs_have_no_answers_or_families(self):
        inputs, _ = worker.fresh_inputs()
        self.assertTrue(all(set(r) == {'item_id', 'text', 'report_sha256'} for r in inputs))

    def test_input_hash_and_inventory_checked(self):
        inputs, _ = worker.fresh_inputs()
        worker.validate_inputs(inputs, 64)
        changed = copy.deepcopy(inputs); changed[0]['report_sha256'] = 'a'*64
        with self.assertRaises(ValueError): worker.validate_inputs(changed, 64)
        with self.assertRaises(ValueError): worker.validate_inputs(inputs[:-1], 64)

    def test_failed_parse_keeps_null_and_sanitizes_exception(self):
        nlp = SimpleNamespace(make_doc=lambda text: (_ for _ in ()).throw(RuntimeError('private source')))
        r = worker.parse_text(nlp, 'Wholly invented opacity.')
        self.assertIsNone(r['state'])
        self.assertEqual(r['failure_reason'], 'RuntimeError')
        self.assertNotIn('private source', str(r))

    def test_actual_slurm_guard_precedes_environment_or_inputs(self):
        with patch.object(worker.guard, 'guard', side_effect=RuntimeError('slurm_required')), \
                patch.object(worker, 'initialize') as init:
            with self.assertRaises(RuntimeError): worker.prepare(SimpleNamespace())
            init.assert_not_called()

    def test_evaluate_requires_exact_manifest_hash_before_parser_load(self):
        with patch.object(worker.guard, 'guard'), patch.object(worker, 'initialize') as init:
            for value in (None, '', '123', 'z'*64):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    worker.evaluate(SimpleNamespace(plan_manifest_sha256=value))
            init.assert_not_called()

    def test_official_scope_assets_are_pinned_without_new_rules(self):
        self.assertFalse(worker.POLICY['official_rules_modified'])
        self.assertEqual(worker.POLICY['context_rule_count'], 102)
        self.assertFalse(worker.POLICY['trained_model_components'])
        self.assertEqual(worker.POLICY['target_pattern'], m.TARGET_PATTERN)
        self.assertTrue(all(len(value[1]) == 64 for value in worker.ASSETS.values()))


if __name__ == '__main__':
    unittest.main()
