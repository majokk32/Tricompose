"""Authored typed fixtures only; no parser accuracy or clinical gold claims."""
from copy import deepcopy
import json
import unittest

from tricompose_v12.fact_comparison_contract import (
    ATTRIBUTES, POLICY, compare_facts, compare_native_reports, digest, from_native_report, make_fact, validate_fact,
)
from tricompose_v12.radgraph_literal_evidence import extract
from test_radgraph_literal_evidence import graph, POS, NEG, UNC, ANAT


def fixture(*, polarity='positive', modality='report', temporality='current', experiencer='patient',
            laterality='unknown', severity='unknown', location=None, concept='fixture_concept',
            source='fixture_source_a', strength='explicit_proposal', dependencies=(), case='case_000'):
    source_sha = digest(source)
    span = {'source_artifact_sha256': source_sha, 'source_representation_sha256': digest('authored_tokens'),
            'offset_unit': 'authored_fixture_token', 'start': 0, 'end_exclusive': 1}
    key = 'evidence_' + digest(span)
    values = dict(polarity=polarity, temporality=temporality, experiencer=experiencer,
                  laterality=laterality, severity=severity, location=location)
    attrs = {name: {'value': value, 'evidence_ids': [] if value in (None, 'unknown') else [key]}
             for name, value in values.items()}
    return make_fact(case_id=case, modality=modality, source_sha256=source_sha,
        concept_sha256=digest(concept), evidence={key: span}, attributes=attrs,
        evidence_strength=strength, dependencies=dependencies, extractor_kind='authored_fixture')


class ComparisonTests(unittest.TestCase):
    def relation(self, left, right, attr='polarity'):
        return compare_facts(left, right)['comparisons'][attr]['relation']

    def test_deterministic_and_no_mutation(self):
        a, b = fixture(), fixture(source='fixture_b')
        original = deepcopy((a, b))
        self.assertEqual(compare_facts(a, b), compare_facts(a, b))
        self.assertEqual(original, (a, b))

    def test_positive_agreement_is_proposal(self):
        self.assertEqual(self.relation(fixture(), fixture()), 'attribute_agreement_proposal')

    def test_negative_agreement_is_not_truth(self):
        self.assertEqual(self.relation(fixture(polarity='negative'), fixture(polarity='negative')),
                         'attribute_agreement_proposal')

    def test_polarity_opposition_is_not_confirmed(self):
        r = compare_facts(fixture(), fixture(polarity='negative'))
        self.assertEqual(r['comparisons']['polarity']['relation'], 'polarity_opposition_proposal')
        self.assertFalse(r['clinical_contradiction_verified'])
        self.assertIsNone(r['clinical_score'])
        self.assertIsNone(r['confirmed_faulty_modality'])

    def test_unknown_not_negative(self):
        self.assertEqual(self.relation(fixture(), fixture(polarity='unknown')), 'unknown_not_comparable')

    def test_uncertain_not_definite(self):
        self.assertEqual(self.relation(fixture(), fixture(polarity='uncertain')), 'uncertain_not_determinate')

    def test_weak_context_cannot_make_opposition(self):
        a = fixture(modality='ehr', strength='weak_context')
        r = compare_facts(a, fixture(polarity='negative'))
        self.assertEqual(r['status'], 'weak_or_missing_context_not_hard_constraint')
        self.assertEqual(r['comparisons']['polarity']['relation'], 'not_comparable')

    def test_unavailable_strength_not_hard(self):
        self.assertEqual(self.relation(fixture(strength='unknown'), fixture()), 'not_comparable')

    def test_different_exact_concepts_not_equivalent(self):
        r = compare_facts(fixture(concept='fixture_a'), fixture(concept='fixture_b'))
        self.assertEqual(r['status'], 'not_comparable_no_exact_concept')
        self.assertTrue(all(v['relation'] == 'not_comparable' for v in r['comparisons'].values()))

    def test_prior_vs_current_not_opposition(self):
        r = compare_facts(fixture(), fixture(polarity='negative', temporality='prior'))
        self.assertEqual(r['status'], 'not_comparable_different_scope')
        self.assertEqual(r['comparisons']['polarity']['relation'], 'not_comparable')

    def test_both_prior_do_not_become_current_support(self):
        r = compare_facts(fixture(temporality='prior'), fixture(temporality='prior'))
        self.assertEqual(r['status'], 'not_comparable_noncurrent_or_nonpatient')

    def test_hypothetical_not_current(self):
        self.assertEqual(self.relation(fixture(temporality='hypothetical'), fixture()), 'not_comparable')

    def test_family_not_patient(self):
        self.assertEqual(self.relation(fixture(experiencer='family'), fixture()), 'not_comparable')

    def test_both_family_not_patient_support(self):
        self.assertEqual(self.relation(fixture(experiencer='family'), fixture(experiencer='family')), 'not_comparable')

    def test_unknown_scope_remains_explicitly_unverified(self):
        r = compare_facts(fixture(temporality='unknown'), fixture(polarity='negative', temporality='unknown'))
        self.assertEqual(r['status'], 'scope_unavailable_proposal_only')
        self.assertFalse(r['scope_proposals_available'])
        self.assertFalse(r['policy']['source_scope_clinically_verified'])

    def test_laterality_difference_not_exclusive(self):
        a, b = fixture(laterality='left'), fixture(laterality='right')
        self.assertEqual(self.relation(a, b, 'laterality'), 'detail_difference_not_exclusive')
        self.assertEqual(self.relation(a, b), 'anatomical_context_not_comparable')

    def test_left_positive_right_negative_not_same_fact_opposition(self):
        a, b = fixture(laterality='left'), fixture(laterality='right', polarity='negative')
        self.assertEqual(self.relation(a, b), 'anatomical_context_not_comparable')

    def test_bilateral_and_left_not_declared_exclusive(self):
        self.assertEqual(self.relation(fixture(laterality='bilateral'), fixture(laterality='left'), 'laterality'),
                         'detail_difference_not_exclusive')

    def test_location_diff_not_hard_contradiction(self):
        a, b = fixture(location=digest('location_a')), fixture(location=digest('location_b'))
        self.assertEqual(self.relation(a, b, 'location'), 'detail_difference_not_exclusive')

    def test_severity_difference_only(self):
        self.assertEqual(self.relation(fixture(severity='mild'), fixture(severity='severe'), 'severity'),
                         'attribute_difference_proposal')

    def test_missing_detail_not_agreement(self):
        self.assertEqual(self.relation(fixture(), fixture(), 'severity'), 'unknown_not_comparable')

    def test_same_image_dependency_preserved_not_independent_votes(self):
        common = digest('authored_shared_image')
        r = compare_facts(fixture(source='fixture_a', dependencies=(common,)),
                          fixture(source='fixture_b', dependencies=(common,)))
        self.assertEqual(r['shared_dependency_artifact_sha256'], [common])
        self.assertFalse(r['policy']['independent_truth_votes'])

    def test_all_three_edges_have_same_contract(self):
        for left, right in (('ehr', 'cxr'), ('ehr', 'report'), ('cxr', 'report')):
            r = compare_facts(fixture(modality=left), fixture(modality=right))
            self.assertEqual(r['edge'], left + '_' + right)
            self.assertTrue(all(v is False for v in r['policy'].values()))

    def test_cross_case_refused(self):
        with self.assertRaises(ValueError):
            compare_facts(fixture(case='case_000'), fixture(case='case_001'))


class ProvenanceTests(unittest.TestCase):
    def test_tampered_value_refused(self):
        f = fixture()
        f['attributes']['polarity']['value'] = 'negative'
        with self.assertRaises(ValueError):
            validate_fact(f)

    def test_known_detail_needs_evidence(self):
        f = fixture()
        f['attributes']['severity']['value'] = 'severe'
        payload = {k: v for k, v in f.items() if k != 'fact_id'}
        f['fact_id'] = 'fact_' + digest(payload)
        with self.assertRaises(ValueError):
            validate_fact(f)

    def test_wrong_source_span_refused(self):
        f = fixture()
        span = next(iter(f['evidence'].values()))
        span['source_artifact_sha256'] = digest('other_source')
        with self.assertRaises(ValueError):
            validate_fact(f)

    def test_invalid_span_refused(self):
        f = fixture()
        next(iter(f['evidence'].values()))['start'] = True
        with self.assertRaises(ValueError):
            validate_fact(f)

    def test_extra_text_or_alias_field_refused(self):
        for key in ('report_text', 'alias', 'cosine'):
            f = fixture()
            f[key] = 'invented fixture'
            with self.assertRaises(ValueError):
                validate_fact(f)

    def test_clinical_promotion_refused(self):
        f = fixture()
        f['policy']['clinical_qualified'] = True
        with self.assertRaises(ValueError):
            validate_fact(f)

    def test_attribute_inventory_fixed(self):
        self.assertEqual(set(fixture()['attributes']), set(ATTRIBUTES))


class NativeAdapterTests(unittest.TestCase):
    def adapter(self, source):
        return from_native_report(source, case_id='case_000', report_sha256=digest('authored_input_bytes'),
                                  expected_native_graph_sha256=extract(source)['native_graph_sha256'])

    def test_native_state_provenance_and_no_new_scope(self):
        for state, value in ((POS, 'positive'), (NEG, 'negative'), (UNC, 'uncertain')):
            source = graph([('authored_finding', state, [])])
            result = self.adapter(source)
            f = result['facts'][0]
            self.assertEqual(f['attributes']['polarity']['value'], value)
            self.assertEqual(f['attributes']['temporality']['value'], 'unknown')
            self.assertEqual(f['attributes']['experiencer']['value'], 'unknown')
            self.assertFalse(result['scope_or_detail_extraction_performed'])

    def test_modifiers_not_invented_as_severity(self):
        source = graph([('severe', POS, [['modify', '2']]), ('authored_finding', POS, [])])
        result = self.adapter(source)
        self.assertEqual(len(result['facts']), 1)
        self.assertEqual(result['facts'][0]['attributes']['severity']['value'], 'unknown')

    def test_native_anatomy_retained_in_sidecar_not_guessed_location(self):
        source = graph([('authored_finding', POS, [['located_at', '2']]), ('left', ANAT, [])])
        result = self.adapter(source)
        self.assertIsNone(result['facts'][0]['attributes']['location']['value'])
        self.assertEqual(result['facts'][0]['attributes']['laterality']['value'], 'unknown')
        self.assertTrue(next(iter(result['native_context_sidecar']['atoms'].values()))['has_explicit_anatomy'])

    def test_empty_graph_not_perfect_agreement(self):
        self.assertEqual(self.adapter(graph([]))['facts'], [])

    def test_native_identity_mismatch_refused(self):
        with self.assertRaises(ValueError):
            from_native_report(graph([('authored_finding', POS, [])]), case_id='case_000',
                report_sha256=digest('authored_input'), expected_native_graph_sha256='0' * 64)

    def test_no_raw_text_export(self):
        result = self.adapter(graph([('AUTHORED_SECRET_FIXTURE', POS, [])]))
        self.assertNotIn('AUTHORED_SECRET_FIXTURE', json.dumps(result))

    def test_word_offsets_not_raw_char_offsets(self):
        result = self.adapter(graph([('authored two words', POS, [])]))
        span = next(iter(result['facts'][0]['evidence'].values()))
        self.assertEqual((span['start'], span['end_exclusive']), (0, 3))
        self.assertEqual(span['offset_unit'], 'native_graph_word')
        self.assertFalse(result['raw_report_byte_identity_verified_here'])

    def test_native_compare_preserves_anatomy_partition(self):
        a = self.adapter(graph([('authored_finding', POS, [['located_at', '2']]), ('left', ANAT, [])]))
        b = self.adapter(graph([('authored_finding', NEG, [['located_at', '2']]), ('right', ANAT, [])]))
        result = compare_native_reports(a, b)
        self.assertEqual(result['presence']['counts']['explicit_polarity_opposition_proposal'], 0)
        self.assertEqual(result['anatomy']['different_context_concepts'], 1)
        self.assertIsNone(result['severity']['comparison'])
        self.assertIsNone(result['temporality']['comparison'])

    def test_native_compare_polarity_is_still_unqualified(self):
        a = self.adapter(graph([('authored_finding', POS, [])]))
        b = self.adapter(graph([('authored_finding', NEG, [])]))
        result = compare_native_reports(a, b)
        self.assertEqual(result['presence']['counts']['explicit_polarity_opposition_proposal'], 1)
        self.assertIsNone(result['clinical_score'])
        self.assertFalse(result['presence']['current_scope_verified'])

    def test_native_missing_mention_not_negative(self):
        a = self.adapter(graph([('authored_finding', POS, [])]))
        result = compare_native_reports(a, self.adapter(graph([])))
        self.assertEqual(result['presence']['counts']['unmentioned_in_right'], 1)
        self.assertEqual(result['presence']['counts']['explicit_polarity_opposition_proposal'], 0)

    def test_native_links_preserve_occurrence_identity(self):
        result = self.adapter(graph([('authored_finding', POS, []), ('authored_finding', NEG, [])]))
        self.assertEqual(len(result['facts']), 2)
        self.assertEqual([v['native_entity_id'] for v in result['native_fact_links']], ['1', '2'])
        self.assertEqual(len({v['fact_id'] for v in result['facts']}), 2)

    def test_adapter_tampering_refused(self):
        a = self.adapter(graph([('authored_finding', POS, [])]))
        a['native_fact_links'][0]['native_entity_id'] = '99'
        with self.assertRaises(ValueError):
            compare_native_reports(a, self.adapter(graph([])))


if __name__ == '__main__':
    unittest.main()
