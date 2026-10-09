"""Quote-free comparison of frozen report extractors, never clinical truth."""
from collections import Counter

from .scorer_reliability import FINDINGS as ALL_FINDINGS

SCHEMA = 'tricompose-legacy-report-semantics-v1'
HEADS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
STATES = {'positive', 'negative', 'uncertain', 'unknown'}


def compare_extractions(rows, scoped, records):
    """Preserve every old state and denominator; model disagreement is not error.

    Deduplicate inference by text hash, but retain candidate-level and distinct
    text-level counts separately. No image/EHR edge is rescored here.
    """
    ids = [r['triple_candidate_id'] for r in rows]
    if not rows or len(rows) > 24 or len(ids) != len(set(ids)):
        raise ValueError('small_distinct_candidate_inventory_required')
    by_id = {r['triple_candidate_id']: r for r in rows}
    by_hash = {r['report_sha256']: r for r in records}
    if (len(by_hash) != len(records)
            or set(by_hash) != {r['report_sha256'] for r in rows}):
        raise ValueError('complete_distinct_report_receipts_required')
    for record in records:
        if (record['contract_status'] not in {'complete', 'failed_unavailable'}
                or set(record['findings']) != set(HEADS)):
            raise ValueError('frozen_four_head_receipt_required')
        for finding in record['findings'].values():
            if finding['state'] not in STATES:
                raise ValueError('unknown_safe_states_required')
            if finding['semantic_correctness_independently_verified'] is not False:
                raise ValueError('automatic_extraction_is_not_clinical_validation')
            if record['contract_status'] != 'complete' and finding['state'] != 'unknown':
                raise ValueError('failed_response_must_remain_unknown')
    seen, output = set(), []
    for fact in scoped:
        cid, finding = fact['triple_candidate_id'], fact['finding']
        key = (cid, finding)
        if cid not in by_id or finding not in ALL_FINDINGS or key in seen:
            raise ValueError('complete_unique_fourteen_head_inventory_required')
        seen.add(key)
        row = by_id[cid]
        if (fact['artifact_hashes']['report_sha256'] != row['report_sha256']
                or fact['report_candidate_id'] != row['report_candidate_id']):
            raise ValueError('source_report_lineage_mismatch')
        raw, literal = fact['states']['chexbert'], fact['scope_report_state']
        if raw not in STATES or literal not in STATES:
            raise ValueError('unknown_safe_states_required')
        record = by_hash[row['report_sha256']]
        covered = finding in HEADS
        state = record['findings'][finding]['state'] if covered else 'unknown'
        available = covered and record['contract_status'] == 'complete'
        if not available or raw not in {'positive', 'negative'} or state not in {'positive', 'negative'}:
            comparison = 'not_comparable'
        else:
            comparison = 'same_explicit_state' if raw == state else 'opposed_explicit_state'
        output.append({'triple_candidate_id': cid, 'report_candidate_id': row['report_candidate_id'],
            'report_sha256': row['report_sha256'], 'finding': finding,
            'raw_chexbert_state': raw, 'literal_scope_state': literal,
            'literal_scope_decision': fact['scope_decision'], 'qwen_assertion_state': state,
            'qwen_contract_status': record['contract_status'] if covered else 'outside_scope_inventory',
            'qwen_contract_failure_reason': record['contract_failure_reason'] if covered else None,
            'chexbert_qwen_comparison': comparison, 'independent_clinical_validation': False,
            'clinical_selection_score': None, 'confirmed_faulty_modality': None,
            'selection_changed': False, 'regeneration_authorized': False})
    if seen != {(cid, finding) for cid in ids for finding in ALL_FINDINGS}:
        raise ValueError('complete_unique_fourteen_head_inventory_required')
    counts = lambda values: dict(sorted(Counter(values).items()))
    summary = {'schema_version': SCHEMA, 'candidate_slots': len(rows), 'fact_rows': len(output),
        'distinct_report_texts': len(records), 'supported_head_checks_candidate_slots': len(rows)*len(HEADS),
        'outside_scope_candidate_rows': len(rows)*(len(ALL_FINDINGS)-len(HEADS)),
        'same_four_heads': {name: {
            'raw_chexbert_states_candidate_slots': counts(r['raw_chexbert_state'] for r in output if r['finding'] == name),
            'literal_scope_states_candidate_slots': counts(r['literal_scope_state'] for r in output if r['finding'] == name),
            'qwen_states_candidate_slots': counts(r['qwen_assertion_state'] for r in output if r['finding'] == name),
            'chexbert_qwen_comparison_candidate_slots': counts(r['chexbert_qwen_comparison'] for r in output if r['finding'] == name),
            'qwen_states_distinct_texts': counts(r['findings'][name]['state'] for r in records)} for name in HEADS},
        'complete_distinct_responses': sum(r['contract_status'] == 'complete' for r in records),
        'failed_distinct_responses': sum(r['contract_status'] != 'complete' for r in records),
        'clinical_accuracy': None, 'primary_metric_eligible': False, 'selection_changed': False,
        'regeneration_authorized': False, 'clinical_faults_confirmed': None,
        'interpretation': 'Secondary report assertion extraction only. Exact quotes establish traceability, not semantic truth. Unknown/failed/outside-scope are not negative or agreement. Extractor disagreement does not identify which modality is wrong.'}
    return output, summary
