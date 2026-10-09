"""Invented metadata only: no clinical body, source image, weight or model."""
import argparse
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import diagnose_online_conflicts as diagnostic
from tricompose_v12.invariant_verification import EHRAnchor, _digest
from tricompose_v12.legacy_replay_adapter import FINDINGS


def fixture():
    anchor = EHRAnchor('case_999', _digest('invented_ehr'), _digest('invented_facts'),
        tuple((n, 'positive' if n == 'pneumonia' else 'unknown', ('diagnosis',) if n == 'pneumonia' else ()) for n in FINDINGS))
    final = {'sha256': _digest('invented_prompt'), 'clinical_intent_sha256': _digest('invented_intent'),
        'included_direct_fact_ids': ['pneumonia'], 'renderer_version': 'invented_renderer',
        'available_context_ids': [], 'included_context_ids': [], 'omitted_context_ids': []}
    request = {'case_id': anchor.case_id, 'model_id': 'roentgen_v2', 'seed': 2, 'request_id': 'invented_s2',
        'inputs': {'final_prompt': final}}
    case = {'opaque_source_index': 5, 'anchor': anchor.record(), 'requests': [{'request': request}]}
    original = {**copy.deepcopy(request), 'seed': 0, 'request_id': 'invented_s0'}
    candidate = {'case_id': anchor.case_id, 'model_id': 'roentgen_v2', 'seed': 2, 'frozen_model': True,
        'adapter_added_prefix': False, 'input_request_sha256': diagnostic.canonical_json_sha256(request),
        'ehr_sha256': anchor.ehr_sha256, 'ehr_facts_sha256': anchor.ehr_facts_sha256,
        'clinical_intent_sha256': final['clinical_intent_sha256'], 'prompt_sha256': final['sha256'],
        'tokenizer_input': {'runtime_observed': True, 'sha256': final['sha256'], 'pipeline_changed_text': False},
        'cost': {'prompt_token_count': 16}}
    pm = {'case_id': anchor.case_id, 'ehr_facts_sha256': anchor.ehr_facts_sha256,
        'one_shared_clinical_intent': True, 'adapter_must_not_add_prefix': True,
        'models': {'roentgen_v2': {**copy.deepcopy(final), 'prompt_sha256': final['sha256'],
            'derived_rule_ids': ['pneumonia_to_airspace_opacity_prior_v1']}}}
    return case, original, candidate, pm


class OnlineConflictDiagnosticTests(unittest.TestCase):
    def test_exact_transport_is_not_semantic_or_clinical_validation(self):
        result = diagnostic.transfer(*fixture())
        self.assertTrue(result['tokenizer_bytes_equal_supplied_prompt'])
        self.assertFalse(result['extraction_semantics_independently_checked'])
        self.assertFalse(result['clinical_image_truth_available'])
        self.assertTrue(result['tokenizer_boundary_not_encoder_hook'])

    def test_no_input_mutation(self):
        args = fixture(); before = copy.deepcopy(args)
        diagnostic.transfer(*args)
        self.assertEqual(args, before)

    def test_prompt_rewriting_is_not_a_seed_only_change(self):
        args = fixture(); args[1]['inputs']['final_prompt']['sha256'] = _digest('other_invented_text')
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_changed_generator_or_artifact_bindings_are_rejected(self):
        for key, value in (('seed', 1), ('model_id', 'chexgenbench_sana'), ('frozen_model', False),
                           ('adapter_added_prefix', True), ('ehr_sha256', _digest('other')),
                           ('input_request_sha256', _digest('other')), ('clinical_intent_sha256', _digest('other'))):
            with self.subTest(key=key):
                args = fixture(); args[2][key] = value
                with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_declared_intent_must_match_frozen_request(self):
        args = fixture(); args[3]['models']['roentgen_v2']['included_direct_fact_ids'] = []
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_derived_prior_cannot_be_silently_unmarked(self):
        args = fixture(); args[3]['models']['roentgen_v2']['derived_rule_ids'] = []
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_unknown_intent_cannot_become_the_old_positive_case(self):
        args = fixture()
        for fact in args[0]['anchor']['findings']:
            if fact['finding'] == 'pneumonia': fact.update(state='unknown', source_categories=[])
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_shared_manifest_flags_required(self):
        args = fixture(); args[3]['one_shared_clinical_intent'] = False
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_observed_tokenizer_change_is_diagnostic_not_an_asserted_repair(self):
        args = fixture(); args[2]['tokenizer_input'].update(sha256=_digest('invented_modified_tokens'), pipeline_changed_text=True)
        self.assertFalse(diagnostic.transfer(*args)['tokenizer_bytes_equal_supplied_prompt'])

    def test_inconsistent_tokenizer_change_flag_rejected(self):
        args = fixture(); args[2]['tokenizer_input']['pipeline_changed_text'] = True
        with self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_unobserved_or_overlength_input_cannot_be_certified(self):
        args = fixture(); args[2]['tokenizer_input']['runtime_observed'] = False
        with self.assertRaises(ValueError): diagnostic.transfer(*args)
        for count in (0, 78, True):
            args = fixture(); args[2]['cost']['prompt_token_count'] = count
            with self.subTest(count=count), self.assertRaises(ValueError): diagnostic.transfer(*args)

    def test_metadata_reader_refuses_body_bearing_files_before_read(self):
        for name in diagnostic.BODY_FILES:
            with patch('diagnose_online_conflicts.require_inside', return_value=Path('/invented') / name), \
                 patch('diagnose_online_conflicts.read_json') as read:
                with self.assertRaises(ValueError): diagnostic.metadata('/invented/' + name, {})
                read.assert_not_called()

    def test_cpu_guard_precedes_any_source_bundle_read(self):
        with patch('diagnose_online_conflicts.cpu_guard', side_effect=RuntimeError('invented_guard')), \
             patch('diagnose_online_conflicts.bundle') as bundle:
            with self.assertRaises(RuntimeError): diagnostic.run(argparse.Namespace())
            bundle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
