"""Prepared directional NLI contracts; no dataset IO, model or API calls.

Real RadNLI sentences may enter these helpers only inside a separately approved
protected worker. Tests use wholly invented strings. Hash-only records never
contain source pair IDs or sentence text. An NLI result does not identify a
faulty modality or prove image/EHR truth. Neutral is not a clinical negative.
"""
from collections import Counter, defaultdict
import hashlib
import json
import re

VERSION = 'tricompose-radnli-directional-diagnostic-v1'
SOURCE = 'https://physionet.org/content/radnli-report-inference/1.0.0/'
LABELS = ('entailment', 'neutral', 'contradiction')
HASH = re.compile(r'[a-f0-9]{64}\Z')


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def request_inventory(documents, *, split):
    """Ignore gold_label/pair_id; return input text separately, in memory only.

    Preserve exact characters, case, punctuation and whitespace. Never replace
    missing text with a dummy pair. All source rows remain attempted slots.
    """
    require(split in ('dev', 'test') and isinstance(documents, list)
            and 0 < len(documents) <= 1024, 'bounded_fixed_nli_source_split_required')
    inputs, rows = {}, []
    for i, doc in enumerate(documents):
        key = f'pair_{i:04d}'
        row = {'pair_id': key, 'source_index': i, 'source_split': split,
            'source_pair_sha256': None, 'premise_sha256': None, 'hypothesis_sha256': None,
            'unordered_pair_sha256': None, 'status': 'failed_unavailable', 'failure_type': None}
        try:
            require(isinstance(doc, dict), 'nli_document_required')
            pair = (doc.get('sentence1'), doc.get('sentence2'))
            require(all(isinstance(s, str) and 0 < len(s) <= 16000 and s.strip() for s in pair),
                    'two_nonempty_bounded_exact_sentences_required')
            premise, hypothesis = (hashlib.sha256(s.encode('utf-8')).hexdigest() for s in pair)
            row.update(source_pair_sha256=digest(pair), premise_sha256=premise,
                hypothesis_sha256=hypothesis, unordered_pair_sha256=digest(sorted((premise, hypothesis))),
                status='ready_input')
            inputs[key] = pair
        except Exception as error:
            row['failure_type'] = type(error).__name__
        rows.append(row)
    return inputs, rows


def reference_inventory(documents, *, split):
    """Future worker calls AFTER sealing model predictions, not for prompts.

    Invalid source schemas/labels remain unavailable, not neutral. This helper
    verifies schema, not licensing, human provenance or model training overlap.
    """
    inputs, requests = request_inventory(documents, split=split)
    del inputs
    result, original_ids = [], set()
    for doc, request in zip(documents, requests):
        row = {**request, 'status': 'failed_unavailable', 'gold_label': None}
        try:
            require(request['status'] == 'ready_input' and set(doc) ==
                {'pair_id', 'sentence1', 'sentence2', 'gold_label'}, 'published_nli_document_schema_required')
            original = doc['pair_id']
            require(type(original) in (str, int) and (not isinstance(original, str) or original)
                    and (type(original), original) not in original_ids, 'unique_internal_source_pair_id_required')
            original_ids.add((type(original), original))
            require(doc['gold_label'] in LABELS, 'published_nli_label_required')
            row.update(status='complete', gold_label=doc['gold_label'], failure_type=None)
        except Exception as error:
            row['failure_type'] = type(error).__name__
        result.append(row)
    return result


def constant_neutral_predictions(requests):
    """Predeclared no-model baseline; receives request hashes, never gold."""
    require(isinstance(requests, list) and requests, 'nonempty_nli_request_inventory_required')
    result = []
    for request in requests:
        ready = request['status'] == 'ready_input'
        require(ready or request['status'] == 'failed_unavailable', 'explicit_input_availability_required')
        result.append({'pair_id': request['pair_id'], 'source_split': request['source_split'],
            'source_pair_sha256': request['source_pair_sha256'],
            'status': 'complete' if ready else 'failed_unavailable',
            'prediction_label': 'neutral' if ready else None,
            'failure_type': None if ready else request['failure_type']})
    return result


def evaluate(references, predictions):
    require(isinstance(references, list) and isinstance(predictions, list)
            and 0 < len(references) == len(predictions) <= 1024, 'all_nli_attempted_slots_required')
    require(len({p['pair_id'] for p in predictions}) == len(predictions), 'unique_nli_prediction_slots_required')
    index = {p['pair_id']: p for p in predictions}
    split = references[0]['source_split']
    matrix = {gold: dict.fromkeys((*LABELS, 'prediction_unavailable'), 0)
              for gold in (*LABELS, 'reference_unavailable')}
    grouped, ordered_gold = defaultdict(list), defaultdict(set)
    for i, ref in enumerate(references):
        key = f'pair_{i:04d}'
        require(ref['pair_id'] == key and ref['source_index'] == i and key in index
                and split in ('dev', 'test') and ref['source_split'] == split,
                'same_complete_release_order_and_split_required')
        pred = index[key]
        require(pred['source_split'] == split and pred['source_pair_sha256'] == ref['source_pair_sha256'],
                'same_directional_source_input_required')
        h = ref['source_pair_sha256']
        require(h is None or isinstance(h, str) and HASH.fullmatch(h), 'source_pair_hash_required')
        if ref['status'] == 'complete':
            require(ref['gold_label'] in LABELS and h is not None and ref['failure_type'] is None,
                    'complete_expert_nli_reference_required')
            truth = ref['gold_label']
            ordered_gold[h].add(truth)
        else:
            require(ref['status'] == 'failed_unavailable' and ref['gold_label'] is None
                    and ref['failure_type'] is not None, 'failed_nli_reference_is_not_neutral')
            truth = 'reference_unavailable'
        if pred['status'] == 'complete':
            require(pred['prediction_label'] in LABELS and h is not None and pred['failure_type'] is None,
                    'complete_named_nli_class_prediction_required')
            decision = pred['prediction_label']
        else:
            require(pred['status'] == 'failed_unavailable' and pred['prediction_label'] is None
                    and pred['failure_type'] is not None, 'failed_nli_prediction_is_not_neutral')
            decision = 'prediction_unavailable'
        matrix[truth][decision] += 1
        if h is not None:
            require(all(isinstance(ref[field], str) and HASH.fullmatch(ref[field])
                        for field in ('premise_sha256', 'hypothesis_sha256', 'unordered_pair_sha256'))
                    and ref['unordered_pair_sha256'] == digest(sorted((ref['premise_sha256'], ref['hypothesis_sha256']))),
                    'unordered_pair_hash_must_preserve_directional_inputs')
            grouped[ref['unordered_pair_sha256']].append((h, truth, decision))
    per_label = {}
    for label in LABELS:
        support = sum(matrix[label].values())
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = support - tp  # Includes missing predictions, never silently dropped.
        per_label[label] = {'reference_support': support, 'tp': tp, 'fp': fp, 'fn': fn,
            'prediction_unavailable': matrix[label]['prediction_unavailable'],
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / support if support else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if support else None}
    scored = sum(sum(matrix[label].values()) for label in LABELS)
    available = scored - sum(matrix[label]['prediction_unavailable'] for label in LABELS)
    correct = sum(matrix[label][label] for label in LABELS)
    f1s = [r['f1'] for r in per_label.values() if r['f1'] is not None]
    bidirectional = [rows for rows in grouped.values() if len({r[0] for r in rows}) > 1]
    annotated_groups = [rows for rows in bidirectional if all(r[1] in LABELS for r in rows)]
    return {'schema_version': VERSION, 'source_split': split, 'attempted_pairs': len(references),
        'reference_available': scored, 'reference_unavailable': len(references) - scored,
        'jointly_available': available, 'correct_predictions': correct,
        'failure_aware_accuracy': correct / scored if scored else None,
        'completed_only_accuracy': correct / available if available else None,
        'per_label': per_label, 'confusion_matrix_all_attempted': matrix,
        'macro_f1_reference_supported_classes': sum(f1s) / len(f1s) if f1s else None,
        'macro_f1_included_labels': [label for label in LABELS if per_label[label]['reference_support']],
        'prediction_status_counts': dict(Counter(p['status'] for p in predictions)),
        'distinct_unordered_sentence_pairs': len(grouped),
        'bidirectional_sentence_pair_groups': len(bidirectional),
        'fully_annotated_bidirectional_groups': len(annotated_groups),
        'bidirectional_groups_all_directions_correct': sum(all(a == b for _, a, b in rows) for rows in annotated_groups),
        'ordered_input_groups_with_conflicting_gold_labels': sum(len(labels) > 1 for labels in ordered_gold.values()),
        'neutral_is_clinical_negative': False, 'entailment_is_symmetric': False,
        'sentence_pairs_are_independent_patients': False, 'scope_subgroup_annotations_available': False,
        'image_truth_verified': False, 'ehr_truth_verified': False,
        'checkpoint_training_overlap_verified': False, 'clinical_qualified': False,
        'primary_clinical_metric_eligible': False, 'selection_changed': False,
        'regeneration_authorized': False, 'new_training': False}


def split_overlap(dev_references, test_references):
    """Exact-hash inventory only; not patient independence or training exclusion."""
    for split, records in (('dev', dev_references), ('test', test_references)):
        require(isinstance(records, list) and records and all(r['source_split'] == split for r in records),
                'both_fixed_split_inventories_required')
    def hashes(records, fields):
        values = {r[field] for r in records for field in fields if r[field] is not None}
        require(all(isinstance(value, str) and HASH.fullmatch(value) for value in values), 'opaque_nli_hashes_required')
        return values
    ordered = len(hashes(dev_references, ('source_pair_sha256',)) & hashes(test_references, ('source_pair_sha256',)))
    unordered = len(hashes(dev_references, ('unordered_pair_sha256',)) & hashes(test_references, ('unordered_pair_sha256',)))
    sentences = len(hashes(dev_references, ('premise_sha256', 'hypothesis_sha256')) &
                    hashes(test_references, ('premise_sha256', 'hypothesis_sha256')))
    return {'ordered_pair_hash_overlap': ordered, 'unordered_pair_hash_overlap': unordered,
        'sentence_hash_overlap': sentences,
        'both_splits_all_text_hashes_available': all(r['source_pair_sha256'] is not None for r in dev_references + test_references),
        'patient_disjointness_verified': False, 'checkpoint_training_exclusion_verified': False,
        'test_examples_automatically_dropped': False, 'clinical_qualified': False}
