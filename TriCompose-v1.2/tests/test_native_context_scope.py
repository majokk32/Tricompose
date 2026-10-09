"""Wholly invented metadata tests; no installed language parser or patient data."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from tricompose_v12 import chexpert_negbio_contract as contract
from tricompose_v12 import native_context_scope as scope
from tricompose_v12.report_assertions import CONTEXT_FLAGS, context_state
from tricompose_v12.assertion_abstention import authored_readout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_scope88_fixture', ROOT/'benchmarks/native_context_scope_controls_v1.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def native(state='positive', finding='lung_opacity', text='opacity'):
    values = [math.nan]*14
    values[contract.FINDINGS.index(finding)] = {'positive': 1, 'negative': 0, 'uncertain': -1, 'unknown': math.nan}[state]
    values[0] = 1
    cleaned = scope.cleaned_full_report(text)
    mentions = [] if state == 'unknown' else [{'annotation_id': '0', 'finding': finding,
        'span': contract.span(cleaned, 0, len(cleaned)), 'native_mention_state': state}]
    views = {name: contract.conflict_view([]) for name in contract.FINDINGS}
    views['no_finding'] = None
    views[finding] = contract.conflict_view([state] if state != 'unknown' else [])
    return {'status': 'complete', 'failure_reason': None, 'native_labels': contract.native_vector(values),
            'mentions': mentions, 'mention_conflict_view': views,
            'original_text_sha256': contract.digest(text), 'cleaned_text_sha256': contract.digest(cleaned)}


def context(raw, **changes):
    rows = []
    for row in raw['mentions']:
        flags = dict.fromkeys(CONTEXT_FLAGS, False)
        if row['native_mention_state'] == 'negative':
            flags['is_negated'] = True
        if row['native_mention_state'] == 'uncertain':
            flags['is_uncertain'] = True
        flags.update(changes)
        rows.append({key: row[key] for key in ('annotation_id', 'finding', 'span')})
        rows[-1].update(flags=flags, context_state=context_state(flags))
    return {'status': 'complete', 'failure_reason': None, 'mentions': rows,
            'original_text_sha256': raw['original_text_sha256'], 'cleaned_text_sha256': raw['cleaned_text_sha256']}


class NativeContextScopeTests(unittest.TestCase):
    def test_same_signed_state_can_only_be_soft(self):
        for state in ('positive', 'negative'):
            raw = native(state)
            row = scope.gate(raw, context(raw))['findings']['lung_opacity']
            self.assertEqual(row['soft_retained_state'], state)
            self.assertTrue(row['soft_comparable'])
            self.assertFalse(row['semantic_scope_verified'])
            self.assertFalse(row['hard_action_eligible'])

    def test_no_finding_is_never_expanded_or_comparable(self):
        raw = native()
        result = scope.gate(raw, context(raw))
        self.assertEqual(result['findings']['no_finding']['decision'], 'aggregate_only_no_finding')
        self.assertFalse(result['findings']['no_finding']['soft_comparable'])
        self.assertIsNone(result['findings']['pneumonia']['soft_retained_state'])
        self.assertEqual(result['findings']['pneumonia']['raw_state'], 'unknown')

    def test_all_native_categories_are_losslessly_represented(self):
        raw = native()
        self.assertEqual(set(scope.gate(raw, context(raw))['findings']), set(contract.FINDINGS))

    def test_unknown_and_uncertain_are_not_signed(self):
        for state in ('unknown', 'uncertain'):
            raw = native(state)
            row = scope.gate(raw, context(raw))['findings']['lung_opacity']
            self.assertEqual(row['raw_state'], state)
            self.assertIsNone(row['soft_retained_state'])

    def test_history_hypothesis_and_family_each_veto(self):
        for key in ('is_historical', 'is_hypothetical', 'is_family'):
            raw = native()
            row = scope.gate(raw, context(raw, **{key: True}))['findings']['lung_opacity']
            self.assertEqual(row['decision'], 'noncurrent_context_flagged')
            self.assertEqual(row['raw_state'], 'positive')
            self.assertIsNone(row['soft_retained_state'])

    def test_uncertainty_flag_vetoes_without_correction(self):
        raw = native()
        row = scope.gate(raw, context(raw, is_uncertain=True))['findings']['lung_opacity']
        self.assertEqual(row['decision'], 'context_uncertainty_flagged')
        self.assertFalse(row['state_corrected'])

    def test_polarity_disagreement_does_not_flip(self):
        raw = native()
        row = scope.gate(raw, context(raw, is_negated=True))['findings']['lung_opacity']
        self.assertEqual(row['decision'], 'native_context_polarity_disagreement')
        self.assertEqual(row['raw_state'], 'positive')
        self.assertIsNone(row['soft_retained_state'])

    def test_native_conflict_is_not_resolved_by_context_agreement(self):
        raw = native()
        raw['mention_conflict_view']['lung_opacity'] = contract.conflict_view(['positive', 'negative'])
        row = scope.gate(raw, context(raw))['findings']['lung_opacity']
        self.assertEqual(row['decision'], 'native_mention_conflict')

    def test_one_noncurrent_mention_vetoes_mixed_scope_conservatively(self):
        raw = native(text='opacity opacity')
        raw['mentions'][0]['span'] = contract.span('opacity opacity', 0, 7)
        other = copy.deepcopy(raw['mentions'][0])
        other['annotation_id'] = '1'
        other['span'] = contract.span('opacity opacity', 8, 7)
        raw['mentions'].append(other)
        ctx = context(raw)
        ctx['mentions'][1]['flags']['is_historical'] = True
        ctx['mentions'][1]['context_state'] = 'unknown'
        self.assertEqual(scope.gate(raw, ctx)['findings']['lung_opacity']['decision'], 'noncurrent_context_flagged')

    def test_missing_one_mention_cannot_inflate_coverage(self):
        raw = native()
        ctx = context(raw)
        ctx['mentions'] = []
        with self.assertRaises(ValueError):
            scope.gate(raw, ctx)

    def test_duplicate_id_rejected_by_source_check(self):
        raw = native()
        raw['mentions'].append(copy.deepcopy(raw['mentions'][0]))
        with self.assertRaises(ValueError):
            scope.check_native('opacity', raw)

    def test_context_source_hash_mismatch_is_refused(self):
        raw = native()
        for key in ('original_text_sha256', 'cleaned_text_sha256'):
            ctx = context(raw)
            ctx[key] = '0'*64
            with self.assertRaises(ValueError):
                scope.gate(raw, ctx)

    def test_changed_flag_state_mapping_is_refused(self):
        raw = native()
        ctx = context(raw)
        ctx['mentions'][0]['context_state'] = 'negative'
        with self.assertRaises(ValueError):
            scope.gate(raw, ctx)

    def test_complete_context_for_failed_native_is_refused(self):
        raw = native()
        with self.assertRaises(ValueError):
            scope.gate(contract.unavailable('detector_error'), context(raw))

    def test_native_failure_stays_failed_null(self):
        result = scope.gate(contract.unavailable('detector_error'), scope.context_unavailable('native_unavailable'))
        row = result['findings']['lung_opacity']
        self.assertEqual(row['raw_status'], 'failed_unavailable')
        self.assertIsNone(row['raw_state'])
        self.assertEqual(row['decision'], 'native_unavailable')

    def test_context_failure_does_not_become_native_unknown(self):
        raw = native()
        row = scope.gate(raw, scope.context_unavailable('context_parser_failed'))['findings']['lung_opacity']
        self.assertEqual(row['raw_state'], 'positive')
        self.assertEqual(row['decision'], 'context_unavailable')

    def test_source_normalization_is_explicit_not_semantic(self):
        self.assertEqual(scope.cleaned_full_report('  OPACITY..\nAND/OR lung/airspace,opacity  '),
                         'opacity. or lung or airspace, opacity')
        self.assertEqual(scope.cleaned_full_report('é opacity'), ' opacity')

    def test_invalid_or_unbounded_source_is_not_truncated(self):
        for text in ('', ' ', None, 'x'*8193):
            with self.assertRaises(ValueError):
                scope.cleaned_full_report(text)

    def test_exact_cleaned_hash_must_match_native(self):
        raw = native()
        raw['cleaned_text_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            scope.check_native('opacity', raw)

    def test_anatomy_and_qualification_never_claimed_verified(self):
        raw = native()
        row = scope.gate(raw, context(raw))['findings']['lung_opacity']
        self.assertFalse(row['anatomy_verified'])
        self.assertFalse(row['qualified_absence_verified'])

    def test_same_wrong_agreement_is_not_clinical_gold(self):
        raw = native(text='invented opacity')
        row = scope.gate(raw, context(raw))['findings']['lung_opacity']
        self.assertTrue(row['soft_comparable'])
        self.assertFalse(row['independent_clinical_validation'])
        self.assertFalse(row['same_report_parsers_are_independent_votes'])

    def test_no_state_or_candidate_changes_in_any_native_state(self):
        for state in contract.STATES:
            raw = native(state)
            previous = copy.deepcopy(raw)
            result = scope.gate(raw, context(raw))
            self.assertEqual(raw, previous)
            for row in result['findings'].values():
                for flag in ('state_corrected', 'candidate_dropped', 'selection_changed', 'regeneration_authorized'):
                    self.assertFalse(row[flag])

    def test_invalid_context_failure_reason_not_source_bearing(self):
        with self.assertRaises(ValueError):
            scope.context_unavailable('arbitrary private exception body')

    def test_serializer_preserves_native_spans_and_flags_without_text(self):
        raw = native()
        flags = dict.fromkeys(CONTEXT_FLAGS, False)
        entity = NS(start_char=0, end_char=7, label_='lung_opacity', _=NS(**flags, modifiers=[]))
        doc = NS(text='opacity', spans={scope.GROUP: [entity]})
        result = scope.serialize_context(doc, 'opacity', raw)
        self.assertEqual(result['mentions'][0]['span'], raw['mentions'][0]['span'])
        self.assertEqual(result['mentions'][0]['temporal_scope'], 'not_flagged')
        self.assertEqual(result['mentions'][0]['anatomy_scope'], 'unverified')
        self.assertNotIn('"text"', json.dumps(result))
        self.assertNotIn('"quote"', json.dumps(result))

    def test_serializer_missing_or_extra_target_is_refused(self):
        raw = native()
        doc = NS(text='opacity', spans={scope.GROUP: []})
        with self.assertRaises(ValueError):
            scope.serialize_context(doc, 'opacity', raw)

    def test_duplicate_processing_span_is_refused(self):
        raw = native()
        flags = dict.fromkeys(CONTEXT_FLAGS, False)
        entity = NS(start_char=0, end_char=7, label_='lung_opacity', _=NS(**flags, modifiers=[]))
        doc = NS(text='opacity', spans={scope.GROUP: [entity, entity]})
        with self.assertRaises(ValueError):
            scope.serialize_context(doc, 'opacity', raw)

    def test_same_target_native_annotation_ids_remain_linked(self):
        raw = native()
        second = copy.deepcopy(raw['mentions'][0])
        second['annotation_id'] = '1'
        raw['mentions'].append(second)
        flags = dict.fromkeys(CONTEXT_FLAGS, False)
        entity = NS(start_char=0, end_char=7, label_='lung_opacity', _=NS(**flags, modifiers=[]))
        result = scope.serialize_context(NS(text='opacity', spans={scope.GROUP: [entity]}), 'opacity', raw)
        self.assertEqual([row['annotation_id'] for row in result['mentions']], ['0', '1'])

    def test_fixture_88_distinct_rows_and_eight_main_findings(self):
        rows = fixtures.cases()
        self.assertEqual(len(rows), 88)
        self.assertEqual(len({row['text'] for row in rows}), 88)
        self.assertEqual(len({row['finding'] for row in rows}), 8)
        self.assertEqual(len({row['family'] for row in rows}), 11)
        self.assertTrue(all(row['expected_state'] in contract.STATES for row in rows))

    def test_factorial_families_have_eight_each_including_anatomy_limit(self):
        from collections import Counter
        self.assertEqual(set(Counter(row['family'] for row in fixtures.cases()).values()), {8})

    def test_new_inputs_not_exact_duplicates_of_old112_controls(self):
        def rows(name):
            spec = importlib.util.spec_from_file_location('_old_fixture_'+name, ROOT/'benchmarks'/name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.cases()
        old = {row['text'] for name in ('opacity_assertion_controls_v2.py', 'opacity_context_controls_v1.py') for row in rows(name)}
        self.assertFalse(old & {row['text'] for row in fixtures.cases()})

    def test_abstention_not_rewarded_as_correct_unknown(self):
        raw = native()
        row = scope.gate(raw, context(raw, is_historical=True))['findings']['lung_opacity']
        result = authored_readout([{**row, 'expected_state': 'unknown'}])
        self.assertEqual(result['raw_four_state_matches'], 0)
        self.assertEqual(result['incorrect_determinate_withheld'], 1)
        self.assertEqual(result['soft_correct'], 0)
        self.assertIsNone(result['overall_gated_accuracy'])

    def test_correct_withheld_is_counted_as_coverage_loss(self):
        raw = native()
        row = scope.gate(raw, context(raw, is_historical=True))['findings']['lung_opacity']
        result = authored_readout([{**row, 'expected_state': 'positive'}])
        self.assertEqual(result['correct_determinate_withheld'], 1)
        self.assertEqual(result['soft_coverage_all_rows'], 0)


if __name__ == '__main__':
    unittest.main()
