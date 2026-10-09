from copy import deepcopy
import json
import unittest

from tricompose_v12.radgraph_literal_evidence import candidate_overlay, compare, extract
from tricompose_v12.radgraph_bank import inventory
from test_radgraph_bank import fixture

POS = 'Observation::definitely present'
NEG = 'Observation::definitely absent'
UNC = 'Observation::uncertain'
ANAT = 'Anatomy::definitely present'


def graph(specs):
    tokens, entities = [], {}
    for i, (text, label, relations) in enumerate(specs, 1):
        start = len(tokens)
        tokens.extend(text.split())
        entities[str(i)] = {'tokens': text, 'label': label, 'start_ix': start,
                            'end_ix': len(tokens)-1, 'relations': relations}
    return {'text': ' '.join(tokens), 'entities': entities}


def finding(state=POS, text='fixture_finding'):
    return graph([(text, state, [])])


class LiteralEvidenceTests(unittest.TestCase):
    def pair(self, a, b):
        return compare(extract(a), extract(b))

    def test_explicit_polarity_opposition_is_proposal_only(self):
        result = self.pair(finding(), finding(NEG))
        self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 1)
        self.assertFalse(result['clinical_qualified'])
        self.assertIsNone(result['clinical_score'])

    def test_positive_and_negative_agreements_both_preserved(self):
        for state in (POS, NEG):
            self.assertEqual(self.pair(finding(state), finding(state))['counts']['native_state_agreement'], 1)

    def test_unmentioned_not_negative_or_contradiction(self):
        result = self.pair(finding(), graph([]))
        self.assertEqual(result['counts']['unmentioned_in_right'], 1)
        self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 0)
        self.assertIsNone(result['details'][0]['right_native_states'])

    def test_uncertain_not_negative(self):
        for state in (POS, NEG, UNC):
            result = self.pair(finding(UNC), finding(state))
            self.assertEqual(result['counts']['uncertain_or_mixed_state'], 1)
            self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 0)

    def test_conflicting_mentions_not_resolved_by_vote(self):
        result = self.pair(graph([('fixture_finding', POS, []), ('fixture_finding', NEG, [])]), finding())
        self.assertEqual(result['counts']['uncertain_or_mixed_state'], 1)

    def test_measurement_and_anatomy_not_disease_states(self):
        source = graph([('fixture_measure', 'Observation::measurement::definitely present', []),
                        ('fixture_anatomy', 'Anatomy::definitely absent', [])])
        result = extract(source)
        self.assertEqual(result['atoms'], {})
        self.assertEqual(result['excluded_measurement_entities'], 1)

    def test_modifier_not_independent_finding(self):
        source = graph([('fixture_detail', POS, [['modify', '2']]), ('fixture_finding', POS, [])])
        result = extract(source)
        self.assertEqual(len(result['atoms']), 1)
        self.assertEqual(result['observation_modifiers_not_counted_as_findings'], 1)

    def test_modifier_difference_not_hard_contradiction(self):
        def example(modifier):
            return graph([(modifier, POS, [['modify', '2']]), ('fixture_finding', POS, [])])
        result = self.pair(example('fixture_small'), example('fixture_large'))
        self.assertEqual(result['modifier_token_set_difference_atoms'], 1)
        self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 0)

    def test_different_anatomy_not_exclusive_claim(self):
        def example(location, state):
            return graph([('fixture_finding', state, [['located_at', '2']]), (location, ANAT, [])])
        result = self.pair(example('fixture_left', POS), example('fixture_right', NEG))
        self.assertEqual(result['different_anatomy_context_concepts'], 1)
        self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 0)
        self.assertEqual(result['counts']['unmentioned_in_left'], 1)
        self.assertEqual(result['counts']['unmentioned_in_right'], 1)

    def test_anatomy_modifier_changes_context_key(self):
        def example(side):
            return graph([('fixture_finding', POS, [['located_at', '2']]),
                          ('fixture_lung', ANAT, []), (side, ANAT, [['modify', '2']])])
        result = self.pair(example('fixture_left'), example('fixture_right'))
        self.assertEqual(result['different_anatomy_context_concepts'], 1)

    def test_uncertain_anatomy_prevents_explicit_opposition(self):
        def example(state):
            return graph([('fixture_finding', state, [['located_at', '2']]),
                          ('fixture_lung', 'Anatomy::uncertain', [])])
        self.assertEqual(self.pair(example(POS), example(NEG))['counts']['anatomy_context_not_definite'], 1)

    def test_casefold_only_no_synonym_guessing(self):
        self.assertEqual(self.pair(finding(text='FIXTURE'), finding(text='fixture'))['counts']['native_state_agreement'], 1)
        result = self.pair(finding(text='fixture_a'), finding(text='fixture_b'))
        self.assertEqual(result['counts']['native_state_agreement'], 0)

    def test_no_clinical_text_export_and_exact_entity_provenance(self):
        result = extract(finding(text='INVENTED_SENSITIVE_TOKEN'))
        self.assertNotIn('INVENTED_SENSITIVE_TOKEN', json.dumps(result))
        occurrence = next(iter(result['atoms'].values()))['occurrences'][0]
        self.assertEqual((occurrence['entity_id'], occurrence['word_start'], occurrence['word_end']), ('1', 0, 0))

    def test_empty_observation_inventory_not_perfect_compatibility(self):
        result = self.pair(graph([]), graph([]))
        self.assertEqual(result['union_literal_atoms'], 0)
        self.assertIsNone(result['clinical_score'])

    def test_duplicate_entities_aggregate_states_not_votes(self):
        result = self.pair(graph([('fixture_finding', POS, []), ('fixture_finding', POS, [])]), finding(NEG))
        self.assertEqual(result['union_literal_atoms'], 1)
        self.assertEqual(result['counts']['explicit_polarity_opposition_proposal'], 1)

    def test_cycles_terminate(self):
        source = graph([('fixture_finding', POS, [['located_at', '2']]),
                        ('fixture_a', ANAT, [['modify', '3']]), ('fixture_b', ANAT, [['modify', '2']])])
        self.assertEqual(len(extract(source)['atoms']), 1)

    def test_tampered_offsets_and_unknown_labels_refused(self):
        for key, value in (('start_ix', 99), ('label', 'invented::positive')):
            source = finding()
            source['entities']['1'][key] = value
            with self.assertRaises(ValueError):
                extract(source)

    def test_input_not_mutated_and_deterministic(self):
        source = finding()
        original = deepcopy(source)
        self.assertEqual(extract(source), extract(source))
        self.assertEqual(source, original)

    def test_scope_and_regeneration_unqualified(self):
        result = self.pair(finding(), finding())
        for key in ('clinical_qualified', 'current_patient_temporal_scope_verified',
                    'independent_truth_votes', 'selection_changed', 'regeneration_authorized'):
            self.assertFalse(result[key])

    def make_bank(self):
        plan = inventory(fixture(), expected_cases=1)
        extracted = {g['graph_id']: extract(finding()) for g in plan['graphs']}
        pairs = [{**p, **compare(extracted[p['left_graph_id']], extracted[p['right_graph_id']])}
                 for p in plan['pairs']]
        return plan, extracted, pairs

    def test_candidate_peer_binding_and_no_new_clinical_score(self):
        plan, extracted, pairs = self.make_bank()
        result = candidate_overlay(plan, extracted, pairs)
        self.assertEqual(len(result), 12)
        self.assertTrue(all(r['rg_literal_state_agreement_occurrences'] == 3 for r in result))
        self.assertTrue(all(r['rg_literal_clinical_score'] is None for r in result))

    def test_failed_graph_not_zero_error_or_unknown(self):
        plan, extracted, pairs = self.make_bank()
        graph_id = plan['graphs'][0]['graph_id']
        extracted[graph_id] = None
        for row in pairs:
            if graph_id in (row['left_graph_id'], row['right_graph_id']):
                row.update(status='unavailable_graph', counts=None)
        result = candidate_overlay(plan, extracted, pairs)
        failed = next(r for r in result if r['rg_literal_status'] == 'unavailable_graph')
        self.assertEqual(failed['rg_literal_available_peers'], 0)
        self.assertIsNone(failed['rg_literal_opposition_proposal_occurrences'])
        self.assertTrue(any(r['rg_literal_available_peers'] == 2 for r in result))

    def test_pair_identity_tampering_refused(self):
        plan, extracted, pairs = self.make_bank()
        pairs[0]['cxr_candidate_id'] = 'invented_wrong_image'
        with self.assertRaises(ValueError):
            candidate_overlay(plan, extracted, pairs)

    def test_forged_graph_hash_or_clinical_promotion_refused(self):
        for name, value in (('left_native_graph_sha256', 'a'*64), ('clinical_qualified', True)):
            plan, extracted, pairs = self.make_bank()
            pairs[0][name] = value
            with self.assertRaises(ValueError):
                candidate_overlay(plan, extracted, pairs)

    def test_directed_unmentioned_counts_bound_to_correct_candidate(self):
        plan, extracted, _ = self.make_bank()
        target = plan['graphs'][0]['graph_id']
        extracted = {key: extract(finding() if key == target else graph([])) for key in extracted}
        pairs = [{**p, **compare(extracted[p['left_graph_id']], extracted[p['right_graph_id']])}
                 for p in plan['pairs']]
        result = candidate_overlay(plan, extracted, pairs)
        row = next(r for r in result if r['rg_literal_atom_count'] == 1)
        self.assertEqual(row['rg_literal_own_atoms_unmentioned_by_peer_occurrences'], 3)
        self.assertEqual(row['rg_literal_peer_atoms_unmentioned_by_self_occurrences'], 0)
        self.assertEqual(sum(r['rg_literal_peer_atoms_unmentioned_by_self_occurrences'] for r in result), 3)


if __name__ == '__main__':
    unittest.main()
