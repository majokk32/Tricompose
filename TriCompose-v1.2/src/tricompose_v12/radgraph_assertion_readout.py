"""Fixed four-head native assertion readout for an authored language diagnostic.

No clinical scorer, learned mapping, synonyms, scope correction, thresholds,
model calls or IO. Native observations covering literal target heads supply
states; modifiers, measurements and generic normality do not become findings.
"""
import hashlib
import re

from .entity_gold_character_contract import token_bounds
from .entity_gold_contract import STATES
from .radgraph_reference_contract import require
from .radgraph_reference_contract_v2 import graph_metadata

VERSION = 'radgraph-literal-four-head-assertion-readout-v1'
PATTERNS = {
    'cardiomegaly': r'\b(?P<head>cardiomegaly)\b',
    'consolidation': r'\b(?P<head>consolidation)\b',
    'pleural_effusion': r'\bpleural\s+(?P<head>effusion)\b',
    'pneumothorax': r'\b(?P<head>pneumothorax)\b',
}


def pool(states):
    values = set(states)
    require(values <= {'positive', 'negative', 'uncertain'}, 'native_observation_states_required')
    if not values:
        return 'unknown'
    if 'uncertain' in values or len(values) > 1:
        return 'uncertain'
    return next(iter(values))


def readout(source_text, graph):
    require(isinstance(source_text, str) and 0 < len(source_text) <= 100000,
            'bounded_literal_source_required')
    metadata = graph_metadata(graph)
    _, native_bounds = token_bounds(source_text.split(), graph['text'].split())
    canonical = ' '.join(source_text.split())
    evidence = {name: [] for name in PATTERNS}
    for finding, pattern in PATTERNS.items():
        for mention in re.finditer(pattern, canonical, flags=re.IGNORECASE):
            start, end = mention.span('head')
            for entity_id, entity in graph['entities'].items():
                if entity['label'] not in STATES:
                    continue
                left = native_bounds[entity['start_ix']][0]
                right = native_bounds[entity['end_ix']][1]
                if left <= start and end <= right:
                    evidence[finding].append({'entity_id': entity_id,
                        'char_start': left, 'char_end_exclusive': right,
                        'head_start': start, 'head_end_exclusive': end,
                        'native_state': STATES[entity['label']]})
        evidence[finding].sort(key=lambda row: (row['head_start'], int(row['entity_id'])))
    return {'schema_version': VERSION, 'status': 'complete',
        'finding_states': {name: pool(e['native_state'] for e in entries)
                           for name, entries in evidence.items()},
        'source_span_references': evidence,
        'source_representation': 'original_words_joined_with_single_spaces',
        'source_artifact_sha256': hashlib.sha256(source_text.encode()).hexdigest(),
        'source_representation_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
        'native_graph_text_sha256': metadata['tokenized_text_sha256'],
        'generic_normality_expanded_to_negatives': False,
        'scope_semantics_corrected': False, 'clinical_qualified': False,
        'clinical_score': None, 'selection_changed': False, 'regeneration_authorized': False}


def assertion_reliability(evaluation):
    """Denominator-aware view of already frozen manual entity-span diagnostics.

No report strings or fitted thresholds. Missing observation spans are a separate
unavailable column, never inferred negative/unknown clinical truth.
"""
    result = {}
    for domain, summary in evaluation.items():
        observed = summary['polarity']['confusion_on_matched_observation_spans']
        matrix, per_state = {}, {}
        for label, state in STATES.items():
            metric = summary['entity_by_label'][label]
            require(metric is not None, 'available_manual_entity_metrics_required')
            support = metric['tp'] + metric['fn']
            predicted = metric['tp'] + metric['fp']
            matched = sum(observed[state].values())
            require(0 <= matched <= support and metric['tp'] == observed[state][state],
                    'matched_confusion_and_span_counts_required')
            matrix[state] = {**observed[state], 'unmatched': support - matched}
            per_state[state] = {'gold_spans': support, 'predicted_spans': predicted,
                'correct_spans': metric['tp'], 'unmatched_gold_spans': support - matched,
                'end_to_end_recall': metric['tp'] / support if support else None,
                'end_to_end_precision': metric['tp'] / predicted if predicted else None,
                'exact_span_state_f1': metric['f1'],
                'conditional_matched_accuracy': observed[state][state] / matched if matched else None}
        result[domain] = {'attempted_reports': summary['attempted_reports'],
            'eligible_reports': summary['eligible_reports'], 'per_state': per_state,
            'confusion_with_unmatched': matrix,
            'positive_negative_flips_on_matched_spans': observed['positive']['negative'] + observed['negative']['positive'],
            'determinate_on_gold_uncertain_matched_spans': observed['uncertain']['positive'] + observed['uncertain']['negative'],
            'clinical_qualified': False, 'thresholds_fitted': False,
            'scope_or_image_or_ehr_accuracy': None}
    return result
