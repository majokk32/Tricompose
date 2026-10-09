"""Source-bound fact proposals and attribute-wise comparison, not truth.

Native adapter reuses the frozen literal extractor. No new NLP rules, aliases,
model calls, thresholds, weights or fault assignment. Scope remains unknown
unless upstream supplies an explicit source-bound proposal; never clinical gold.
"""
from copy import deepcopy
import hashlib
import json
import re

from .radgraph_literal_evidence import extract, compare as compare_native_literal

VERSION = 'tricompose-fact-comparison-contract-v1'
ATTRIBUTES = {
    'polarity': ('positive', 'negative', 'uncertain', 'unknown'),
    'temporality': ('current', 'prior', 'hypothetical', 'unknown'),
    'experiencer': ('patient', 'family', 'unknown'),
    'laterality': ('left', 'right', 'bilateral', 'midline', 'unknown'),
    'severity': ('mild', 'moderate', 'severe', 'unknown'),
    'location': None,
}
HASH = re.compile(r'[a-f0-9]{64}\Z')
POLICY = {'clinical_qualified': False, 'semantic_equivalence_verified': False,
    'source_scope_clinically_verified': False, 'image_truth_verified': False,
    'ehr_truth_verified': False, 'independent_truth_votes': False,
    'missing_is_negative': False, 'weak_context_is_hard_constraint': False,
    'selection_changed': False, 'regeneration_authorized': False, 'new_training': False}


def require(condition):
    if not condition:
        raise ValueError('source_bound_fact_proposal_contract_required')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and bool(HASH.fullmatch(value))


def validate_fact(fact):
    fields = {'schema_version', 'case_id', 'modality', 'source_artifact_sha256',
        'dependency_artifact_sha256', 'concept_sha256', 'evidence_strength', 'extractor_kind',
        'attributes', 'evidence', 'policy', 'fact_id'}
    require(isinstance(fact, dict) and set(fact) == fields and fact['schema_version'] == VERSION
        and fact['policy'] == POLICY and isinstance(fact['case_id'], str)
        and re.fullmatch(r'case_[0-9]{3,6}', fact['case_id'])
        and fact['modality'] in ('ehr', 'cxr', 'report')
        and valid_hash(fact['source_artifact_sha256']) and valid_hash(fact['concept_sha256'])
        and fact['evidence_strength'] in ('explicit_proposal', 'weak_context', 'unknown')
        and fact['extractor_kind'] in ('native_radgraph_literal_unqualified',
                                      'upstream_explicit_proposal', 'authored_fixture'))
    dependencies = fact['dependency_artifact_sha256']
    require(isinstance(dependencies, list) and 1 <= len(dependencies) <= 16
        and all(valid_hash(v) for v in dependencies) and dependencies == sorted(set(dependencies))
        and fact['source_artifact_sha256'] in dependencies)
    evidence = fact['evidence']
    require(isinstance(evidence, dict) and 1 <= len(evidence) <= 128)
    for key, span in evidence.items():
        require(isinstance(span, dict) and set(span) == {'source_artifact_sha256',
            'source_representation_sha256', 'offset_unit', 'start', 'end_exclusive'}
            and span['source_artifact_sha256'] == fact['source_artifact_sha256']
            and valid_hash(span['source_representation_sha256'])
            and span['offset_unit'] in ('native_graph_word', 'upstream_token', 'authored_fixture_token')
            and type(span['start']) is int and type(span['end_exclusive']) is int
            and 0 <= span['start'] < span['end_exclusive'] <= 100000
            and key == 'evidence_' + digest(span))
    attributes = fact['attributes']
    require(isinstance(attributes, dict) and set(attributes) == set(ATTRIBUTES))
    for name, allowed in ATTRIBUTES.items():
        item = attributes[name]
        require(isinstance(item, dict) and set(item) == {'value', 'evidence_ids'}
            and isinstance(item['evidence_ids'], list)
            and item['evidence_ids'] == sorted(set(item['evidence_ids']))
            and all(key in evidence for key in item['evidence_ids']))
        value = item['value']
        require(value is None or valid_hash(value) if name == 'location' else value in allowed)
        known = value is not None if name == 'location' else value != 'unknown'
        require(not known or bool(item['evidence_ids']))
    payload = {k: v for k, v in fact.items() if k != 'fact_id'}
    require(fact['fact_id'] == 'fact_' + digest(payload))
    return fact


def make_fact(*, case_id, modality, source_sha256, concept_sha256, evidence,
              attributes=None, dependencies=(), evidence_strength='explicit_proposal',
              extractor_kind='upstream_explicit_proposal'):
    values = {name: {'value': None if name == 'location' else 'unknown', 'evidence_ids': []}
              for name in ATTRIBUTES}
    if attributes is not None:
        require(isinstance(attributes, dict) and set(attributes) <= set(ATTRIBUTES))
        values.update(deepcopy(attributes))
    payload = {'schema_version': VERSION, 'case_id': case_id, 'modality': modality,
        'source_artifact_sha256': source_sha256,
        'dependency_artifact_sha256': sorted(set((source_sha256, *dependencies))),
        'concept_sha256': concept_sha256, 'evidence_strength': evidence_strength,
        'extractor_kind': extractor_kind, 'attributes': values, 'evidence': deepcopy(evidence),
        'policy': dict(POLICY)}
    fact = {**payload, 'fact_id': 'fact_' + digest(payload)}
    validate_fact(fact)
    return fact


def from_native_report(graph, *, case_id, report_sha256, expected_native_graph_sha256, dependencies=()):
    """Caller supplies raw-report lineage; native graph identity checked here.

Does not equate native graph text with raw input bytes. Native word offsets,
not Unicode source offsets. No interpretation of laterality/severity/scope.
"""
    native = extract(graph)
    require(native['native_graph_sha256'] == expected_native_graph_sha256)
    facts, links = [], []
    for atom in native['atoms'].values():
        for occurrence in atom['occurrences']:
            span = {'source_artifact_sha256': report_sha256,
                'source_representation_sha256': native['native_graph_sha256'],
                'offset_unit': 'native_graph_word', 'start': occurrence['word_start'],
                'end_exclusive': occurrence['word_end'] + 1}
            evidence_id = 'evidence_' + digest(span)
            attributes = {'polarity': {'value': occurrence['native_state'], 'evidence_ids': [evidence_id]}}
            # Anatomy's hash is only an exact native context identity, not a location ontology.
            # Keep it unknown here: source anatomy nodes are not in the occurrence span.
            fact = make_fact(case_id=case_id, modality='report', source_sha256=report_sha256,
                concept_sha256=atom['concept_sha256'], evidence={evidence_id: span},
                attributes=attributes, dependencies=dependencies,
                extractor_kind='native_radgraph_literal_unqualified')
            facts.append(fact)
            links.append({'fact_id': fact['fact_id'], 'native_atom_id': atom['atom_id'],
                'native_entity_id': occurrence['entity_id'], 'anatomy_context_sha256': atom['anatomy_context_sha256'],
                'native_modifier_set_sha256': occurrence['modifier_set_sha256']})
    result = {'schema_version': VERSION + '-native-adapter', 'case_id': case_id,
        'source_artifact_sha256': report_sha256, 'native_graph_sha256': native['native_graph_sha256'],
        'facts': facts, 'native_fact_links': links, 'native_context_sidecar': native,
        'raw_report_byte_identity_verified_here': False,
        'scope_or_detail_extraction_performed': False, 'policy': dict(POLICY)}
    return {**result, 'adapter_id': 'adapter_' + digest(result)}


def compare_facts(left, right):
    validate_fact(left)
    validate_fact(right)
    require(left['case_id'] == right['case_id'])
    same_concept = left['concept_sha256'] == right['concept_sha256']
    a, b = left['attributes'], right['attributes']
    scope_conflict = any(a[k]['value'] != 'unknown' and b[k]['value'] != 'unknown'
                         and a[k]['value'] != b[k]['value'] for k in ('temporality', 'experiencer'))
    scope_available = all(a[k]['value'] != 'unknown' and b[k]['value'] != 'unknown'
                          for k in ('temporality', 'experiencer'))
    noncurrent = any(side['temporality']['value'] in ('prior', 'hypothetical') or
                     side['experiencer']['value'] == 'family' for side in (a, b))
    anatomy_difference = any(a[k]['value'] != b[k]['value'] and
        all(v not in (None, 'unknown') for v in (a[k]['value'], b[k]['value']))
        for k in ('location', 'laterality'))
    explicit = left['evidence_strength'] == right['evidence_strength'] == 'explicit_proposal'
    status = ('not_comparable_no_exact_concept' if not same_concept else
              'not_comparable_different_scope' if scope_conflict else
              'not_comparable_noncurrent_or_nonpatient' if noncurrent else
              'weak_or_missing_context_not_hard_constraint' if not explicit else
              'matching_scope_proposal' if scope_available else 'scope_unavailable_proposal_only')
    comparisons = {}
    for name in ATTRIBUTES:
        x, y = a[name]['value'], b[name]['value']
        unknown = x is None or y is None if name == 'location' else 'unknown' in (x, y)
        if not same_concept or scope_conflict or noncurrent or not explicit:
            relation = 'not_comparable'
        elif unknown:
            relation = 'unknown_not_comparable'
        elif name == 'polarity' and 'uncertain' in (x, y):
            relation = 'uncertain_not_determinate'
        elif name == 'polarity' and anatomy_difference:
            relation = 'anatomical_context_not_comparable'
        elif x == y:
            relation = 'attribute_agreement_proposal'
        elif name == 'polarity':
            relation = 'polarity_opposition_proposal'
        elif name in ('laterality', 'location'):
            relation = 'detail_difference_not_exclusive'
        else:
            relation = 'attribute_difference_proposal'
        comparisons[name] = {'relation': relation, 'left_value': x, 'right_value': y,
            'left_evidence_ids': list(a[name]['evidence_ids']),
            'right_evidence_ids': list(b[name]['evidence_ids'])}
    result = {'schema_version': VERSION + '-comparison', 'case_id': left['case_id'],
        'left_fact_id': left['fact_id'], 'right_fact_id': right['fact_id'],
        'edge': left['modality'] + '_' + right['modality'], 'status': status,
        'same_exact_concept': same_concept, 'scope_proposals_available': scope_available,
        'comparisons': comparisons,
        'shared_dependency_artifact_sha256': sorted(set(left['dependency_artifact_sha256']) &
                                                   set(right['dependency_artifact_sha256'])),
        'left_evidence_strength': left['evidence_strength'], 'right_evidence_strength': right['evidence_strength'],
        'clinical_score': None, 'confirmed_faulty_modality': None, 'clinical_contradiction_verified': False,
        'policy': dict(POLICY)}
    return {**result, 'comparison_id': 'comparison_' + digest(result)}


def compare_native_reports(left, right):
    """Existing anatomy-aware native comparisons plus honest missing dimensions.

No reinterpretation of native modifiers as severity, no generic positive
vote over different anatomy, and no conversion of missing mentions to negative.
"""
    for adapter in (left, right):
        require(adapter.get('schema_version') == VERSION + '-native-adapter' and adapter['policy'] == POLICY
            and adapter['adapter_id'] == 'adapter_' + digest({k: v for k, v in adapter.items() if k != 'adapter_id'}))
        require(len(adapter['facts']) == len(adapter['native_fact_links']))
        require(adapter['native_context_sidecar']['native_graph_sha256'] == adapter['native_graph_sha256'])
        for fact, link in zip(adapter['facts'], adapter['native_fact_links']):
            validate_fact(fact)
            require(fact['fact_id'] == link['fact_id'] and fact['case_id'] == adapter['case_id']
                and fact['source_artifact_sha256'] == adapter['source_artifact_sha256'])
            atom = adapter['native_context_sidecar']['atoms'][link['native_atom_id']]
            require(atom['concept_sha256'] == fact['concept_sha256'] and
                    atom['anatomy_context_sha256'] == link['anatomy_context_sha256'])
            occurrences = [v for v in atom['occurrences'] if v['entity_id'] == link['native_entity_id']]
            require(len(occurrences) == 1)
            occurrence = occurrences[0]
            require(occurrence['modifier_set_sha256'] == link['native_modifier_set_sha256'] and
                    fact['attributes']['polarity']['value'] == occurrence['native_state'])
            require(len(fact['evidence']) == 1)
            span = next(iter(fact['evidence'].values()))
            require(span['source_representation_sha256'] == adapter['native_graph_sha256'] and
                    span['offset_unit'] == 'native_graph_word' and span['start'] == occurrence['word_start'] and
                    span['end_exclusive'] == occurrence['word_end'] + 1)
    require(left['case_id'] == right['case_id'])
    native = compare_native_literal(left['native_context_sidecar'], right['native_context_sidecar'])
    result = {'schema_version': VERSION + '-native-report-comparison', 'case_id': left['case_id'],
        'left_adapter_id': left['adapter_id'], 'right_adapter_id': right['adapter_id'],
        'left_source_artifact_sha256': left['source_artifact_sha256'],
        'right_source_artifact_sha256': right['source_artifact_sha256'],
        'presence': {'status': 'native_proposals_only', 'counts': native['counts'],
                     'details': native['details'], 'current_scope_verified': False},
        'anatomy': {'status': 'context_differences_not_exclusive',
                   'different_context_concepts': native['different_anatomy_context_concepts']},
        'severity': {'status': 'unavailable_not_extracted', 'comparison': None},
        'temporality': {'status': 'unavailable_not_extracted', 'comparison': None},
        'experiencer': {'status': 'unavailable_not_extracted', 'comparison': None},
        'modifier_difference_not_severity': native['modifier_token_set_difference_atoms'],
        'clinical_score': None, 'confirmed_faulty_modality': None, 'policy': dict(POLICY)}
    return {**result, 'comparison_id': 'comparison_' + digest(result)}
