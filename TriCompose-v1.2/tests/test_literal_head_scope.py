import unittest

from audit_literal_head_scope import match_heads
from tricompose_v12.radgraph_literal_evidence import digest


class LiteralHeadScopeTests(unittest.TestCase):
    def fixture(self, token='edema'):
        graphs = {g: {'atoms': {'atom_fixture': {'concept_sha256': digest(['literal_observation', token])}}}
                  for g in ('left', 'right')}
        pairs = [{'status': 'complete', 'pair_id': 'pair_0000', 'case_id': 'case_000',
            'cxr_candidate_id': 'cxr_000', 'left_graph_id': 'left', 'right_graph_id': 'right',
            'details': [{'atom_id': 'atom_fixture', 'relation': 'explicit_polarity_opposition_proposal'}]}]
        return graphs, pairs

    def test_exact_existing_head(self):
        graphs, pairs = self.fixture()
        self.assertEqual(match_heads(('edema',), graphs, pairs)[0]['exact_existing_head'], 'edema')

    def test_underscore_display_format_not_semantic_alias(self):
        graphs, pairs = self.fixture('pleural effusion')
        self.assertEqual(match_heads(('pleural_effusion',), graphs, pairs)[0]['exact_existing_head'], 'pleural_effusion')

    def test_no_synonym_or_partial_concept_guessing(self):
        graphs, pairs = self.fixture('effusion')
        self.assertIsNone(match_heads(('pleural_effusion',), graphs, pairs)[0]['exact_existing_head'])

    def test_unmatched_record_preserved_not_failed_clinical_case(self):
        graphs, pairs = self.fixture('fixture_other')
        result = match_heads(('edema',), graphs, pairs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'no_exact_literal_head_match')
        self.assertFalse(result[0]['clinical_resolved'])

    def test_agreement_not_added_to_opposition_inventory(self):
        graphs, pairs = self.fixture()
        pairs[0]['details'][0]['relation'] = 'native_state_agreement'
        self.assertEqual(match_heads(('edema',), graphs, pairs), [])

    def test_invalid_head_and_mismatched_concept_refused(self):
        graphs, pairs = self.fixture()
        with self.assertRaises(ValueError):
            match_heads(('edema', 'edema'), graphs, pairs)
        graphs['right']['atoms']['atom_fixture']['concept_sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            match_heads(('edema',), graphs, pairs)

    def test_no_scope_or_execution_permission_inferred(self):
        graphs, pairs = self.fixture()
        result = match_heads(('edema',), graphs, pairs)[0]
        for key in ('current_patient_scope_verified', 'clinical_resolved', 'model_request_submitted'):
            self.assertFalse(result[key])


if __name__ == '__main__':
    unittest.main()
