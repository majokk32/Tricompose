"""Metadata-defined annotation cohort, not score/finding-selected examples.

The release has 590 published-score rows but 100 expert annotations. V1
incorrectly bounded the entire score release at 100 and failed closed. Preserve
V1; evaluate all 100 annotated keys, documenting the 490 unannotated rows.
"""
import importlib.util
from pathlib import Path

PATH=Path(__file__).with_name('radevalx_published_adapter.py')
spec=importlib.util.spec_from_file_location('immutable_radevalx_adapter_v1',PATH)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ANNOTATION_FIELDS=base.ANNOTATION_FIELDS
METRIC_FIELDS=base.METRIC_FIELDS
METRICS=base.METRICS


def key_set(rows, fields, expected):
    if len(rows)!=expected or any(set(row)!=set(fields) for row in rows):
        raise ValueError('exact_released_metadata_inventory_required')
    result={base.key(row) for row in rows}
    if len(result)!=len(rows):
        raise ValueError('duplicate_released_key_refused')
    return result


def cohort(score_rows, significant_rows, insignificant_rows):
    score_keys=key_set(score_rows,METRIC_FIELDS,590)
    sig_keys=key_set(significant_rows,ANNOTATION_FIELDS,100)
    insig_keys=key_set(insignificant_rows,ANNOTATION_FIELDS,100)
    if sig_keys!=insig_keys or not sig_keys<=score_keys:
        raise ValueError('same_all_annotated_keys_and_scores_required')
    selected=[row for row in score_rows if base.key(row) in sig_keys]
    return selected,{'published_score_rows':590,'significant_annotation_rows':100,
        'insignificant_annotation_rows':100,'all_annotated_pairs_included':100,
        'published_rows_without_expert_annotation':490,'annotated_rows_without_scores':0,
        'selection_by_annotation_key_membership_only':True,'selection_by_score_finding_or_error':False,
        'numeric_score_error_or_report_fields_used_for_eligibility':False}


predictions=base.predictions
references=base.references
