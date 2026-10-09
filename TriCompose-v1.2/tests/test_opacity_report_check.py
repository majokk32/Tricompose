"""Invented report/quote/metadata fixtures. No model inference or patient data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('opacity_report_fixture', ROOT / 'tools/check_opacity_report_assertions.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def response(**kwargs):
    arrays = dict.fromkeys(m.POLARITIES, None)
    arrays.update({p: [] for p in m.POLARITIES})
    arrays.update(kwargs)
    return json.dumps({'lung_opacity': arrays})


def outcomes_fixture():
    outcomes = [
        {'report_sha256': 'a'*64, **m.decode_evidence(response(negative=['No lung opacity.']), 'No lung opacity.')},
        {'report_sha256': 'b'*64, **m.decode_evidence(response(), 'Heart size is normal.')},
        {'report_sha256': 'c'*64, **m.decode_evidence('not-json', 'Invented report.')},
    ]
    contexts = [{'report_sha256': d*64, 'report_opacity_state': 'positive', 'triple_candidate_id': f'context_{i}',
                 'ehr_sha256': 'f'*64, 'joint_proxy_pattern': 'report_proposal_opposes_two_image_sources'}
                for i,d in enumerate(('a','a','b','c'))]
    return contexts, outcomes


class OpacityReportCheckTests(unittest.TestCase):
    def test_positive_and_negative_quotes_with_source_offsets(self):
        for polarity, quote in [('positive', 'Lung opacity is present.'), ('negative', 'No lung opacity.')]:
            source = 'Synthetic: ' + quote
            out = m.parse_evidence(response(**{polarity:[quote]}), source)
            self.assertEqual(out['state'], polarity)
            span = out['evidence'][polarity][0]
            self.assertEqual(source[span['char_start']:span['char_end']], quote)
            self.assertEqual(span['quote_sha256'], m.digest_text(quote))
            self.assertFalse(out['semantic_correctness_independently_verified'])

    def test_empty_completed_response_unknown_not_failed_or_negative(self):
        out = m.decode_evidence(response(), 'The heart size is normal.')
        self.assertEqual((out['contract_status'], out['state']), ('complete','unknown'))

    def test_uncertain_quote_never_negative(self):
        out = m.parse_evidence(response(uncertain=['Possible lung opacity.']), 'Possible lung opacity.')
        self.assertEqual(out['state'], 'uncertain')

    def test_conflicting_quotes_retained_without_voting(self):
        positive,negative = 'Lung opacity is present.', 'No lung opacity is present.'
        out = m.parse_evidence(response(positive=[positive], negative=[negative]), positive+' '+negative)
        self.assertTrue(out['opposed_quoted_assertions'])
        self.assertEqual(out['state'], 'uncertain')

    def test_qualified_absence_quoted_uncertain(self):
        out = m.parse_evidence(response(uncertain=['No large lung opacity.']), 'No large lung opacity.')
        self.assertEqual(out['state'], 'uncertain')

    def test_quote_traceability_is_not_semantic_adjudication(self):
        out = m.parse_evidence(response(positive=['No lung opacity.']), 'No lung opacity.')
        self.assertEqual(out['state'], 'positive')
        self.assertFalse(out['semantic_correctness_independently_verified'])

    def test_quotes_must_be_exact_not_paraphrase_or_hallucination(self):
        out = m.decode_evidence(response(positive=['Lung opacity is present.']), 'No lung opacity.')
        self.assertEqual(out['contract_status'], 'failed_unavailable')
        self.assertEqual(out['contract_failure_reason'], 'quote_not_in_source')

    def test_ambiguous_repeated_quote_not_guessed(self):
        quote='No lung opacity.'
        with self.assertRaisesRegex(m.EvidenceContractError,'quote_location_ambiguous'):
            m.parse_evidence(response(negative=[quote]),quote+' '+quote)

    def test_duplicate_quote_across_polarities_refused(self):
        quote='No lung opacity.'
        with self.assertRaisesRegex(m.EvidenceContractError,'duplicate_or_conflicting_quote'):
            m.parse_evidence(response(positive=[quote],negative=[quote]),quote)

    def test_duplicate_json_keys_refused(self):
        out=m.decode_evidence('{"lung_opacity":{},"lung_opacity":{}}','Invented report.')
        self.assertEqual(out['contract_failure_reason'],'invalid_json_or_duplicate_key')

    def test_missing_extra_and_wrong_finding_inventory_refused(self):
        for value in ('{}','{"pneumonia":{}}','{"lung_opacity":{},"score":1}','[]'):
            out=m.decode_evidence(value,'Invented report.')
            self.assertEqual(out['contract_failure_reason'],'finding_inventory_mismatch')

    def test_extra_or_missing_polarity_refused(self):
        for value in ({'positive':[]},{'positive':[],'negative':[],'uncertain':[],'score':1}):
            out=m.decode_evidence(json.dumps({'lung_opacity':value}),'Invented report.')
            self.assertEqual(out['contract_failure_reason'],'polarity_inventory_mismatch')

    def test_quote_list_type_count_length_and_whitespace_bounds(self):
        for value in ('No lung opacity.', ['No'], ['x'*257], ['Invented quote.']*3, [' No lung opacity.']):
            self.assertEqual(m.decode_evidence(response(negative=value),'No lung opacity.')['contract_status'], 'failed_unavailable')

    def test_unicode_offsets_and_json_fences(self):
        quote='No lung opacity.'; source='合成病例：'+quote
        out=m.parse_evidence('```json\n'+response(negative=[quote])+'\n```',source)
        self.assertEqual(out['evidence']['negative'][0]['char_start'],len('合成病例：'))

    def test_token_limit_makes_response_unavailable_even_valid_json(self):
        out=m.decode_evidence(response(), 'Invented report.', token_limit_reached=True)
        self.assertEqual(out['contract_failure_reason'],'token_limit_reached')

    def test_non_text_response_fails_closed(self):
        out=m.decode_evidence(None,'Invented report.')
        self.assertEqual(out['contract_failure_reason'],'invalid_response_type')

    def test_request_only_text_no_prediction_id_score_image_or_ehr(self):
        messages=m.request_messages('Invented report.')
        self.assertEqual(len(messages),1)
        self.assertEqual([r['type'] for r in messages[0]['content']],['text'])
        for token in ('report_sha256','cached_state','expected_state','candidate_id','model_id'):
            self.assertNotIn(token,messages[0]['content'][0]['text'])

    def test_untrusted_braces_are_not_format_instructions(self):
        text='Invented {report} {score} text.'
        self.assertIn(text,m.request_messages(text)[0]['content'][0]['text'])

    def test_empty_or_overlength_report_not_truncated(self):
        for text in ('',' ', 'x'*8193,None):
            with self.assertRaises(ValueError): m.request_messages(text)

    def test_authored_controls_fixed_all_states_and_no_expected_state_in_input(self):
        controls=m.controls()
        self.assertEqual(len(controls),6)
        self.assertEqual({r['expected_state'] for r in controls}, {'positive','negative','uncertain','unknown'})
        for row in controls:
            self.assertNotIn('expected_state',m.request_messages(row['text'])[0]['content'][0]['text'])

    def test_failure_unknown_different_from_complete_unknown(self):
        contexts,outcomes=outcomes_fixture(); rows=m.candidate_readout(contexts,outcomes)
        self.assertEqual([r['text_check_cached_proposal_relation'] for r in rows],
            ['proxy_opposition','proxy_opposition','not_comparable','response_unavailable'])

    def test_every_shared_context_preserved_and_original_metadata_unchanged(self):
        args=outcomes_fixture(); before=copy.deepcopy(args)
        rows=m.candidate_readout(*args)
        self.assertEqual(args,before)
        self.assertEqual(len(rows),4)
        for old,new in zip(args[0],rows):
            self.assertEqual({k:new[k] for k in old},old)
            self.assertIsNone(new['text_check_semantic_accuracy'])
            self.assertFalse(new['text_check_selector_used'])
            self.assertFalse(new['text_check_regeneration_authorized'])

    def test_missing_duplicate_and_foreign_text_outcomes_refused(self):
        contexts,outcomes=outcomes_fixture()
        for bad in (outcomes[:2], outcomes+outcomes[:1],outcomes+[{'report_sha256':'e'*64}]):
            with self.assertRaisesRegex(ValueError,'all_unique_text_outcomes'):
                m.candidate_readout(contexts,bad)

    def test_failed_state_cannot_smuggle_determinate_proposal(self):
        contexts,outcomes=outcomes_fixture(); outcomes[-1]['state']='positive'
        with self.assertRaisesRegex(ValueError,'explicit_four_state'):
            m.candidate_readout(contexts,outcomes)

    def test_cached_unknown_not_negative_or_agreement(self):
        complete=m.decode_evidence(response(positive=['Lung opacity.']), 'Lung opacity.')
        self.assertEqual(m.proposal_relation('unknown',complete),'not_comparable')
        self.assertEqual(m.proposal_relation('uncertain',complete),'not_comparable')

    def test_login_guard_before_paths_text_or_runtime(self):
        with patch.dict(os.environ,{},clear=True),patch.object(m,'require_inside') as resolver,patch.object(m,'runtime_metadata') as runtime:
            with self.assertRaises(RuntimeError):
                m.evaluate(SimpleNamespace(allow_synthetic_text_check=True))
            resolver.assert_not_called();runtime.assert_not_called()

    def test_gpu_approval_guard_before_paths(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'},clear=True),patch.object(m,'require_inside') as resolver:
            with self.assertRaises(RuntimeError):
                m.evaluate(SimpleNamespace(allow_synthetic_text_check=False))
            resolver.assert_not_called()

    def test_existing_run_refused_before_model_access(self):
        with patch.object(m.OP,'guard'),patch.object(m,'require_inside',return_value=Path(__file__)),patch.object(m,'runtime_metadata') as runtime:
            with self.assertRaises(FileExistsError):
                m.evaluate(SimpleNamespace(allow_synthetic_text_check=True,output_root='fixture',run_id='fixture'))
            runtime.assert_not_called()

    def test_offline_required_before_plan_or_text(self):
        with patch.object(m.OP,'guard'),patch.object(m,'require_inside',return_value=Path('/nonexistent_fixture')), \
                patch.dict(os.environ,{},clear=True),patch.object(m.OP,'metadata') as loader:
            with self.assertRaisesRegex(RuntimeError,'offline_frozen_runtime_required'):
                m.evaluate(SimpleNamespace(allow_synthetic_text_check=True,output_root='fixture',run_id='fixture'))
            loader.assert_not_called()

    def test_prepare_guard_before_any_source_metadata(self):
        with patch.dict(os.environ,{},clear=True),patch.object(m.OP,'metadata') as loader:
            with self.assertRaises(RuntimeError): m.prepare(SimpleNamespace())
            loader.assert_not_called()


if __name__=='__main__':
    unittest.main()
