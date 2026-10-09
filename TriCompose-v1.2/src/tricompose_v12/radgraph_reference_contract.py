"""Lightweight contract for the official, frozen RadGraph-XL reference metric.

No inference, downloads, file access, or replacement clinical scoring formula.
Model calls belong in approved Slurm workers. Graphs can contain sensitive text;
validation errors and the returned score table deliberately contain none of it.
"""
import hashlib
import math
from numbers import Real
import re


VERSION = "tricompose-radgraph-reference-v1"
CODE_REVISION = "87f11a1ff4d2046a838be5f0243857d93780ddec"
WEIGHT_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"
MODEL_TYPE = "radgraph-xl"
WEIGHT_FILENAME = "radgraph-xl.tar.gz"
WEIGHT_BYTES = 416179571
WEIGHT_SHA256 = "fadb5a3454e8996714b609e3105a07e447fa61aa88fffe966b650959475117b6"
METRICS = ("radgraph_entity_f1", "radgraph_relation_presence_f1",
           "radgraph_full_relation_f1")
LABELS = {
    "Anatomy::definitely present": "anatomy",
    "Observation::definitely present": "positive",
    "Observation::definitely absent": "negative",
    "Observation::uncertain": "uncertain",
}
POLICY = {
    "reference_free": False,
    "new_training": False,
    "clinical_qualified": False,
    "image_factuality_verified": False,
    "ehr_consistency_verified": False,
    "selection_changed": False,
    "regeneration_authorized": False,
    "missing_or_empty_is_zero": False,
    "absence_of_entity_means_negative": False,
    "graph_encodes_current_patient_scope": False,
}


def require(condition, code):
    if not condition:
        # Never interpolate a report, identifier, entity token, or exception.
        raise ValueError(code)


def finite_score(value):
    return not isinstance(value, bool) and isinstance(value, Real) and \
        math.isfinite(float(value)) and 0 <= float(value) <= 1


def graph_metadata(graph):
    """Validate native XL output; return hashes/counts, not report/entity text.

    These are model predictions, not medical truth or current-patient facts.
    Native word-token offsets are not original-report character offsets.
    Unknown ontology labels fail closed; absent entities are not negative.
    """
    require(isinstance(graph, dict) and isinstance(graph.get("text"), str) and
            isinstance(graph.get("entities"), dict), "native_graph_required")
    require(len(graph["text"]) <= 100000 and len(graph["entities"]) <= 4096,
            "bounded_graph_required")
    tokens = graph["text"].split()
    entities = graph["entities"]
    counts = {state: 0 for state in ("anatomy", "positive", "negative", "uncertain")}
    relations = 0
    for key, entity in entities.items():
        require(isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key) and
                isinstance(entity, dict), "native_entity_required")
        label = entity.get("label")
        require(isinstance(label, str) and label in LABELS, "unsupported_xl_label")
        start, end = entity.get("start_ix"), entity.get("end_ix")
        require(type(start) is int and type(end) is int and
                0 <= start <= end < len(tokens), "valid_word_offsets_required")
        require(entity.get("tokens") == " ".join(tokens[start:end + 1]),
                "entity_token_span_mismatch")
        outgoing = entity.get("relations")
        require(isinstance(outgoing, list) and len(outgoing) <= 4096,
                "bounded_native_relations_required")
        for relation in outgoing:
            require(isinstance(relation, (tuple, list)) and len(relation) == 2 and
                    isinstance(relation[0], str) and 0 < len(relation[0]) <= 128 and
                    isinstance(relation[1], str) and relation[1] in entities,
                    "valid_relation_destination_required")
        counts[LABELS[label]] += 1
        relations += len(outgoing)
    return {"tokenized_text_sha256": hashlib.sha256(graph["text"].encode()).hexdigest(),
            "entity_count": len(entities), "relation_count": relations,
            "native_state_counts": counts, "scope_verified": False}


def score_table(pair_ids, nonempty_inputs, official_result):
    """Normalize F1RadGraph(reward_level='all') without inventing another F1.

    ``nonempty_inputs`` is fixed from BOTH input texts before model invocation.
    Upstream empty-pair zeros remain checked, but are ineligible/null here.
    Any model failure must be handled by the worker as a failed receipt, not by
    passing fabricated graphs/results to this function.
    """
    require(isinstance(pair_ids, list) and 1 <= len(pair_ids) <= 1024 and
            all(isinstance(p, str) and re.fullmatch(r"pair_[0-9]{4}", p)
                for p in pair_ids) and len(set(pair_ids)) == len(pair_ids),
            "bounded_unique_opaque_pair_ids_required")
    require(isinstance(nonempty_inputs, list) and len(nonempty_inputs) == len(pair_ids)
            and all(type(v) is bool for v in nonempty_inputs),
            "pre_call_input_availability_required")
    require(isinstance(official_result, (tuple, list)) and len(official_result) == 4,
            "official_all_result_required")
    means, components, hypotheses, references = official_result
    require(isinstance(means, (tuple, list)) and len(means) == 3 and
            isinstance(components, (tuple, list)) and len(components) == 3,
            "three_native_reward_components_required")
    for mean, scores in zip(means, components):
        require(isinstance(scores, (tuple, list)) and len(scores) == len(pair_ids) and
                all(finite_score(v) for v in scores) and finite_score(mean),
                "finite_complete_reward_inventory_required")
        require(abs(float(mean) - sum(map(float, scores)) / len(scores)) < 1e-7,
                "native_mean_inventory_mismatch")
    available = sum(nonempty_inputs)
    require(isinstance(hypotheses, list) and isinstance(references, list) and
            len(hypotheses) == len(references) == available,
            "complete_nonempty_annotation_inventory_required")
    rows, graph_index = [], 0
    for index, (pair_id, nonempty) in enumerate(zip(pair_ids, nonempty_inputs)):
        scores = [float(component[index]) for component in components]
        if nonempty:
            hypothesis = graph_metadata(hypotheses[graph_index])
            reference = graph_metadata(references[graph_index])
            graph_index += 1
            status = "complete"
        else:
            require(all(v == 0 for v in scores), "upstream_empty_pair_zero_required")
            hypothesis = reference = None
            status = "empty_input_not_eligible"
        rows.append({"item_id": pair_id, "status": status,
                     "scores": {name: value if nonempty else None
                                for name, value in zip(METRICS, scores)},
                     "hypothesis_graph": hypothesis, "reference_graph": reference})
    return {"schema_version": VERSION, "model_type": MODEL_TYPE,
            "metric_provenance": "official_frozen_reference_comparison",
            "policy": dict(POLICY), "all_attempted_pairs": len(rows),
            "eligible_pairs": available, "records": rows}
