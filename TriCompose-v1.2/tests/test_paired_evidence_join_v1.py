"""Invented, text-free fixtures: transfer checks never rewrite clinical scores."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import join_paired_evidence_v1 as join


def fixture():
    scores = [{'case_id': 'case_000', 'triple_candidate_id': 'invented_triple',
        'cxr_model_id': 'roentgen_v2', 'seed': '3', 'report_model_id': 'invented',
        'biovil_raw_cosine': '', 'ehr_cxr_supported_facts': '0', 'clinical_accuracy': ''}]
    lineage = [dict(scores[0], seed=3, cxr_candidate_id='invented_image',
        ehr_sha256='a'*64, ehr_facts_sha256='b'*64, cxr_sha256='c'*64, report_sha256='d'*64)]
    trace = {k: None for k in join.TRACE_FIELDS}
    trace.update(case_id='case_000', cxr_candidate_id='invented_image', seed=3,
        length_and_hash_checks_pass=True, matched_phrase_count=1, included_phrase_count=1,
        legacy_context_id_count=1, radiographic_id_count=1, old_combined_guard_would_pass=False,
        clinical_truth_available=False, scores_or_source_inputs_modified=False,
        clinical_context_promoted_to_image_finding=False, metadata_projection_is_not_runtime_request=True)
    return scores, lineage, [trace]


class EvidenceJoinTests(unittest.TestCase):
    def test_all_score_cells_and_na_unchanged(self):
        args = fixture(); original = deepcopy(args)
        result = join.join_candidates(*args)[0]
        self.assertEqual({k: result[k] for k in args[0][0]}, args[0][0])
        self.assertEqual(result['biovil_raw_cosine'], '')
        self.assertEqual(args, original)
        self.assertFalse(result['conditioning_is_clinical_truth'])
        self.assertEqual(result['source_gpu_job_state'], 'FAILED')

    def test_legacy_context_does_not_add_support(self):
        r = join.join_candidates(*fixture())[0]
        self.assertEqual(r['ehr_cxr_supported_facts'], '0')
        self.assertEqual(r['conditioning_legacy_context_id_count'], 1)
        self.assertFalse(r['conditioning_old_combined_guard_would_pass'])
        self.assertFalse(r['selection_or_primary_acceptance_changed'])

    def test_missing_or_duplicate_trace_fails(self):
        for traces in ([], fixture()[2] * 2):
            args = fixture(); args = (args[0], args[1], traces)
            with self.assertRaises(ValueError): join.join_candidates(*args)

    def test_duplicate_scores_or_lineage_fail(self):
        for i in (0, 1):
            args = list(fixture()); args[i] *= 2
            with self.assertRaises(ValueError): join.join_candidates(*args)

    def test_wrong_case_seed_model_or_id_fails(self):
        for key, value in (('case_id', 'case_001'), ('seed', 4),
                ('cxr_model_id', 'another_model'), ('triple_candidate_id', 'another_triple')):
            args = fixture(); args[1][0][key] = value
            with self.assertRaises(ValueError): join.join_candidates(*args)

    def test_private_body_keys_never_copied(self):
        args = fixture(); args[2][0]['positive_tokenizer_text'] = 'invented forbidden body'
        r = join.join_candidates(*args)[0]
        self.assertNotIn('positive_tokenizer_text', str(r))
        self.assertNotIn('invented forbidden body', str(r))

    def test_context_promotion_or_clinical_truth_fails(self):
        for field in ('clinical_truth_available', 'scores_or_source_inputs_modified',
                'clinical_context_promoted_to_image_finding'):
            args = fixture(); args[2][0][field] = True
            with self.assertRaises(ValueError): join.join_candidates(*args)

    def test_existing_columns_cannot_be_overwritten(self):
        args = fixture(); args[0][0]['conditioning_length_and_hash_checks_pass'] = 'old'
        with self.assertRaises(ValueError): join.join_candidates(*args)

    def test_nullable_trace_stays_unavailable(self):
        args = fixture(); args[2][0]['attention_token_count'] = None
        args[2][0]['length_and_hash_checks_pass'] = False
        r = join.join_candidates(*args)[0]
        self.assertIsNone(r['conditioning_attention_token_count'])
        self.assertFalse(r['conditioning_length_and_hash_checks_pass'])

    def test_guard_before_any_read_or_write(self):
        with patch.object(join, 'cpu_guard', side_effect=RuntimeError('invented_guard')), \
                patch.object(join.cache.secondary, 'load_metadata') as load, \
                patch.object(join, 'new_atomic_run') as write:
            with self.assertRaises(RuntimeError): join.run(None)
            load.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()
