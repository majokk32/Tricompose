#!/usr/bin/env python3
"""Pure fresh-receipt output veto plus an archived metadata-only contract smoke.

Not an inference controller, clinical acceptance, or authorization to regenerate.
Callers must authenticate source artifacts upstream; self-digests are not trust.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ('TriCompose-v1.2/src', 'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(ROOT.parent / relative))
sys.path.insert(0, str(ROOT / 'tools'))
from tricompose_v12 import bounded_regeneration as image_gate
from tricompose_v12 import report_expert_control as report_gate
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.invariant_verification import _digest, _HASH, _ID, EXPLICIT
from tricompose_v12.legacy_replay_adapter import FINDINGS
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import PROFILE
from score_free_random_control import cpu_guard
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-fresh-observed-output-veto-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
ROW_FIELDS = ('case_id', 'triple_candidate_id', 'ehr_sha256', 'ehr_facts_sha256',
    'cxr_candidate_id', 'cxr_sha256', 'report_candidate_id', 'report_sha256',
    'cxr_model_id', 'report_model_id', 'seed', 'receipt', 'structure', 'raw_edge_readouts')
STRUCTURE_FIELDS = ('report_candidate_id', 'report_sha256', 'image_sha256', 'case_id',
    'report_model_id', 'source_cxr_model_id', 'parent_cxr_candidate_id',
    'findings_complete', 'impression_complete', 'section_contract_pass',
    'impression_required_by_model_contract', 'empty', 'generic_report',
    'unsupported_temporal_comparison_language', 'repeated_sentence_count',
    'repeated_4gram_ratio', 'normalized_report_sha256')
SOURCES = {
    'plan': (BASE / 'bounded_regeneration_plans/retry2_12625457_001',
        '5a4e3bfbfd7e87879e14ea545932e3fef6331f3440bb4ef2973e52355fec8542'),
    'run': (BASE / 'bounded_regeneration_runs/retry2_12632531',
        'eb8099458d895da46281f1955c8e73f19082b61b0759bb8064bcb79ff922c32d'),
    'audit': (BASE / 'bounded_regeneration_audits/retry2_12632531',
        '58def50df7b72bc1d2cd6fc99bd4dd411f10241cc3633625d2c1e98e846a53f5'),
}


def context(anchor_record, xrv_spec, chexbert_spec):
    """Explicit fixed-EHR provenance and scorer mask; no legacy-label conversion."""
    anchor = anchor_from_record(anchor_record)
    thresholds = xrv_spec['thresholds']
    if set(thresholds) != set(FINDINGS) or any(type(t['enabled']) is not bool for t in thresholds.values()):
        raise ValueError('complete_explicit_head_mask_required')
    enabled = [n for n in FINDINGS if thresholds[n]['enabled']]
    if len(enabled) != 8:
        raise ValueError('exact_frozen_eight_enabled_head_profile_required')
    hashes = {'thresholds_sha256': xrv_spec['thresholds_sha256'],
        'xrv_checkpoint_sha256': xrv_spec['checkpoint_sha256'],
        'chexbert_checkpoint_sha256': chexbert_spec['checkpoint_sha256']}
    if any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in hashes.values()):
        raise ValueError('valid_scorer_provenance_hashes_required')
    return {'profile': PROFILE, 'anchor': anchor.record(), 'enabled_xrv_findings': enabled, **hashes}


def controller_view(row):
    """Do not admit old winners, alternate endpoints or unneeded source fields."""
    result = copy.deepcopy({k: row[k] for k in ROW_FIELDS})
    result['structure'] = {k: result['structure'][k] for k in STRUCTURE_FIELDS}
    return result


def validate_observation(row, ctx):
    anchor = anchor_from_record(ctx['anchor'])
    if (set(ctx) != {'profile', 'anchor', 'enabled_xrv_findings', 'thresholds_sha256',
            'xrv_checkpoint_sha256', 'chexbert_checkpoint_sha256'}
            or ctx['profile'] != PROFILE or len(ctx['enabled_xrv_findings']) != 8
            or len(set(ctx['enabled_xrv_findings'])) != 8
            or ctx['enabled_xrv_findings'] != [n for n in FINDINGS if n in ctx['enabled_xrv_findings']]):
        raise ValueError('frozen_fresh_profile_required')
    for key in ('case_id', 'triple_candidate_id', 'cxr_candidate_id', 'report_candidate_id',
                'cxr_model_id', 'report_model_id'):
        if not isinstance(row[key], str) or not _ID.fullmatch(row[key]):
            raise ValueError('opaque_metadata_identity_required')
    for key in ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256'):
        if not isinstance(row[key], str) or not _HASH.fullmatch(row[key]):
            raise ValueError('artifact_sha256_required')
    if (row['case_id'] != anchor.case_id or row['ehr_sha256'] != anchor.ehr_sha256
            or row['ehr_facts_sha256'] != anchor.ehr_facts_sha256
            or type(row['seed']) is not int or row['seed'] < 0
            or row['cxr_model_id'] not in ('roentgen_v2', 'chexgenbench_sana', 'chexgenbench_pixart')):
        raise ValueError('fixed_ehr_and_frozen_cold_start_model_required')
    facts = image_gate.validate_row(row)
    receipt = row['receipt']
    if (receipt['ehr_anchor_sha256'] != anchor.sha256
            or any(receipt[k] != ctx[k] for k in ('profile', 'thresholds_sha256',
                'xrv_checkpoint_sha256', 'chexbert_checkpoint_sha256'))
            or [(f['finding'], f['ehr']) for f in facts] != [(n, s) for n, s, _ in anchor.findings]
            or any(f['xrv'] != 'unknown' for f in facts if f['finding'] not in ctx['enabled_xrv_findings'])):
        raise ValueError('anchor_head_mask_or_scorer_provenance_changed')
    for key in ('ehr_anchor_sha256', 'partial_receipt_id', 'xrv_labels_sha256',
                'chexbert_labels_sha256', 'thresholds_sha256', 'xrv_checkpoint_sha256',
                'chexbert_checkpoint_sha256', 'receipt_id'):
        if not isinstance(receipt[key], str) or not _HASH.fullmatch(receipt[key]):
            raise ValueError('receipt_provenance_sha256_required')
    if (any(receipt[k] is not False for k in ('clinical_acceptance', 'clinical_repair_success',
            'primary_clinical_metric_eligible', 'model_execution_allowed_by_receipt'))
            or receipt['clinical_accuracy'] is not None
            or receipt['report_lifecycle_status'] != 'generated_and_labelled'
            or receipt['report_assertion_scope'] != 'raw_chexbert_unverified_not_guarded_assertions'):
        raise ValueError('unvalidated_completed_diagnostic_receipt_required')
    known = receipt['known_ehr_facts']
    joint = [f for f in facts if all(f[k] in EXPLICIT for k in ('ehr', 'xrv', 'chexbert'))]
    supported = sum(f['ehr'] == f['xrv'] == f['chexbert'] for f in joint)
    status = ('unverified_no_direct_ehr_constraints' if not known else
        'explicit_proxy_opposition_unvalidated' if any(e['proxy_opposition_facts'] for e in receipt['raw_edge_readouts'].values()) else
        'unverified_missing_direct_ehr_comparison' if len(joint) < known else
        'direct_ehr_label_agreement_unvalidated')
    count_fields = ('inventory_findings', 'known_reference_facts', 'comparable_facts',
        'supported_facts', 'supported_positive', 'supported_negative', 'proxy_opposition_facts', 'missing_comparisons')
    if (type(known) is not int or type(receipt['all_three_supported_facts']) is not int
            or any(type(e[k]) is not int for e in receipt['raw_edge_readouts'].values() for k in count_fields)
            or receipt['verification_status'] != status
            or receipt['all_three_supported_facts'] != supported
            or receipt['all_three_support_over_known'] != (supported / known if known else None)):
        raise ValueError('unknown_safe_joint_arithmetic_required')
    s = row['structure']
    if (type(s['impression_required_by_model_contract']) is not bool
            or s['source_cxr_model_id'] != row['cxr_model_id']
            or s['parent_cxr_candidate_id'] != row['cxr_candidate_id']
            or not isinstance(s['normalized_report_sha256'], str)
            or not _HASH.fullmatch(s['normalized_report_sha256'])):
        raise ValueError('structure_parent_lineage_required')
    return facts


def validate_ledger(book, anchor):
    """Replay the complete durable ledger, including failures and in-flight cost."""
    if book is None:
        return None
    if book['case_id'] != anchor.case_id or book['ehr_anchor_sha256'] != anchor.sha256:
        raise ValueError('execution_cost_anchor_changed')
    replay = restore_ledger(book['events'], case_id=anchor.case_id,
        ehr_anchor_sha256=anchor.sha256, call_budget=book['call_budget'],
        max_retries=book['max_retries'], execution_mode=book['execution_mode'], sink=lambda e: None)
    if replay.snapshot() != book:
        raise ValueError('execution_cost_snapshot_must_replay_exactly')
    return book


def completed_operation(row, book):
    """Bind one observation to an actually completed four-phase ledger chain."""
    if book is None:
        raise ValueError('completed_observation_requires_execution_ledger')
    requests, results, positions = {}, {}, {}
    for position, event in enumerate(book['events']):
        if event['event'] == 'attempt_reserved':
            requests[event['request']['operation_id']] = event['request']
        elif event['event'] == 'attempt_completed':
            results[event['operation_id']] = event['result']
            positions[event['operation_id']] = position
    r = row['receipt']
    matched = [oid for oid in results if requests[oid]['kind'] == 'chexbert'
        and results[oid]['verification_receipt_id'] == r['receipt_id']]
    if len(matched) != 1:
        raise ValueError('one_completed_receipt_operation_required')
    cb = matched[0]; report = requests[cb]['parent_operation_id']
    xrv = requests[report]['parent_operation_id']; image = requests[xrv]['parent_operation_id']
    if (results[cb]['output_artifact_sha256'] != r['chexbert_labels_sha256']
            or requests[cb]['model_id'] != 'chexbert' or requests[xrv]['model_id'] != 'xrv'
            or requests[cb]['input_report_sha256'] != row['report_sha256']
            or results[report]['output_artifact_sha256'] != row['report_sha256']
            or requests[report]['model_id'] != row['report_model_id']
            or results[xrv]['output_artifact_sha256'] != r['xrv_labels_sha256']
            or results[xrv]['verification_receipt_id'] != r['partial_receipt_id']
            or results[image]['output_artifact_sha256'] != row['cxr_sha256']
            or requests[image]['model_id'] != row['cxr_model_id']
            or requests[image]['seed'] != row['seed']
            or any(requests[oid]['input_image_sha256'] != row['cxr_sha256'] for oid in (xrv, report, cb))):
        raise ValueError('completed_receipt_and_cost_chain_lineage_differ')
    return positions[cb]


def assess_fresh_output(baseline_id, proposed_id, observations, ctx, *, ledger_snapshot=None):
    """Final output veto over explicitly available observations, not a router.

    `pinned_cached_reference` is an upstream-authenticated sunk-cost reference,
    not a new call. `completed_ledger_receipt` must match a completed operation.
    Veto fallback is the initial nonempty, section-valid fixed reference only.
    The observed report-only reference is used for comparison, not fallback.
    """
    if not observations:
        raise ValueError('nonempty_observed_inventory_required')
    anchor = anchor_from_record(ctx['anchor'])
    book = validate_ledger(ledger_snapshot, anchor)
    rows, origins, images, image_hashes, reports, report_hashes = {}, {}, {}, {}, {}, {}
    last_completion, seen_live = -1, False
    for item in observations:
        if set(item) != {'row', 'origin'}:
            raise ValueError('explicit_observation_origin_required')
        row = controller_view(item['row']); cid = row['triple_candidate_id']
        if cid in rows:
            raise ValueError('duplicate_observed_candidate')
        facts = validate_observation(row, ctx)
        origin = item['origin']
        if origin.get('kind') == 'pinned_cached_reference':
            if (set(origin) != {'kind', 'source_manifest_sha256'} or seen_live
                    or not isinstance(origin['source_manifest_sha256'], str)
                    or not _HASH.fullmatch(origin['source_manifest_sha256'])):
                raise ValueError('cached_reference_must_precede_new_acquisitions')
        elif origin == {'kind': 'completed_ledger_receipt'}:
            completion = completed_operation(row, book)
            if completion <= last_completion:
                raise ValueError('completed_observation_order_required')
            last_completion, seen_live = completion, True
        else:
            raise ValueError('unsupported_observation_origin')
        image_value = (row['cxr_sha256'], [(f['finding'], f['xrv']) for f in facts],
            row['cxr_model_id'], row['seed'], row['receipt']['xrv_labels_sha256'], row['receipt']['partial_receipt_id'])
        if images.setdefault(row['cxr_candidate_id'], image_value) != image_value:
            raise ValueError('same_image_must_keep_classifier_lineage_and_states')
        if image_hashes.setdefault(row['cxr_sha256'], image_value[1]) != image_value[1]:
            raise ValueError('shared_image_hash_must_keep_classifier_states')
        report_value = (row['report_sha256'], [(f['finding'], f['chexbert']) for f in facts],
            row['cxr_candidate_id'], row['report_model_id'], row['receipt']['chexbert_labels_sha256'])
        if reports.setdefault(row['report_candidate_id'], report_value) != report_value:
            raise ValueError('same_report_id_must_keep_labels_and_parent')
        if report_hashes.setdefault(row['report_sha256'], report_value[1]) != report_value[1]:
            raise ValueError('shared_report_hash_must_keep_frozen_states')
        rows[cid], origins[cid] = row, copy.deepcopy(origin)
    if baseline_id != next(iter(rows)) or proposed_id is not None and proposed_id not in rows:
        raise ValueError('initial_baseline_and_already_observed_proposal_required')
    base = rows[baseline_id]
    eligible = report_gate.structure_ok(base)
    reference = base
    same_image = [r for r in rows.values() if r['triple_candidate_id'] != baseline_id
        and r['cxr_candidate_id'] == base['cxr_candidate_id']]
    # Predeclared expert priority, then observation order; no full-bank access.
    for candidate in sorted(same_image, key=lambda r: report_gate.MODELS.index(r['report_model_id'])):
        if report_gate.compare(base, candidate)['exploratory_gate_pass']:
            reference = candidate
            break
    comparisons, reasons, passed = [], [], False
    if proposed_id == baseline_id:
        status = 'unchanged_unverified'
    elif proposed_id is None:
        status = 'unresolved_missing_proposal_fixed_retained'
        reasons = ['proposal_unavailable']
    else:
        proposal = rows[proposed_id]
        if proposal['cxr_candidate_id'] == base['cxr_candidate_id']:
            comparisons = [report_gate.compare(base, proposal)]
            passed = comparisons[0]['exploratory_gate_pass']
            status = 'proxy_preserving_report_change_unverified'
        elif image_gate.route(base)['action'] == 'regenerate_cxr':
            comparisons = [image_gate.compare(base, proposal)]
            if reference['triple_candidate_id'] != baseline_id:
                comparisons.append(image_gate.compare(reference, proposal))
            passed = all(c['exploratory_gate_pass'] for c in comparisons) and eligible
            status = 'proxy_preserving_image_change_unverified'
        else:
            reasons.append('no_direct_ehr_image_branch_basis')
            status = 'unresolved_proposal_veto_fixed_retained'
        reasons.extend(reason for c in comparisons for reason in c['reasons'])
        if not passed:
            status = 'unresolved_proposal_veto_fixed_retained'
    selected = proposed_id if passed else baseline_id if eligible else None
    if selected is None:
        status = 'unresolved_no_section_eligible_output'
    cost = {'ledger_snapshot_sha256': _digest(book) if book is not None else None,
        'charged_model_attempts': book['charged_model_attempts'] if book is not None else 0,
        'failed_attempts': book['failed_attempts'] if book is not None else 0,
        'pending_attempts': book['pending_attempts'] if book is not None else 0,
        'charged_attempts_by_kind': copy.deepcopy(book['charged_attempts_by_kind']) if book is not None else {},
        'cached_reference_count': sum(o['kind'] == 'pinned_cached_reference' for o in origins.values()),
        'historical_generation_scoring_cost': 'shared_sunk_not_measured_not_zero',
        'veto_refunds': 0, 'measured_gpu_seconds': None}
    decision = {'schema_version': SCHEMA, 'profile': PROFILE, 'case_id': anchor.case_id,
        'scorer_context_sha256': _digest(ctx), 'enabled_xrv_findings': list(ctx['enabled_xrv_findings']),
        'ehr_anchor_sha256': anchor.sha256, 'baseline_candidate_id': baseline_id,
        'proposed_candidate_id': proposed_id, 'selected_candidate_id': selected,
        'observed_candidate_ids': list(rows), 'observed_reference_candidate_id': reference['triple_candidate_id'],
        'observation_metadata_sha256': _digest([{'row': r, 'origin': origins[c]} for c, r in rows.items()]),
        'status': status, 'proposal_passes_proxy_preservation': bool(passed),
        'comparisons': comparisons, 'rejection_reason_codes': sorted(set(reasons)), 'cost': cost,
        'selected_raw_edge_readouts': copy.deepcopy(rows[selected]['raw_edge_readouts']) if selected is not None else None,
        'clinical_acceptance': False, 'clinical_repair_success': False, 'confirmed_faulty_modality': None,
        'model_execution_allowed': False, 'secondary_endpoint_used': False,
        'source_artifact_authentication': 'required_upstream_not_proven_by_self_digest',
        'installed_in_live_controller': False}
    return {**decision, 'decision_sha256': _digest(decision)}


def _checked_json(path, expected, sources, label):
    path = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if path.stat().st_size > 16 * 1024 ** 2 or sha256_file(path) != expected:
        raise ValueError('bounded_pinned_metadata_required')
    value = json.loads(path.read_text())
    if sha256_file(path) != expected:
        raise ValueError('metadata_changed_during_read')
    sources[label] = path
    return value


def load_archived_smoke():
    """Only sealed plan/state/cost/audit metadata; never read model/body files."""
    sources, manifests = {}, {}
    for label, (root, expected) in SOURCES.items():
        manifests[label] = _checked_json(root / 'manifest.json', expected, sources, label + '_manifest')
    plan = _checked_json(SOURCES['plan'][0] / 'plan.json', manifests['plan']['plan_sha256'], sources, 'plan')
    run = manifests['run']
    if run['plan_manifest_sha256'] != SOURCES['plan'][1] or run['clinical_acceptance'] is not False:
        raise ValueError('fixed_nonclinical_source_required')
    audit = _checked_json(SOURCES['audit'][0] / 'audit.json', manifests['audit']['audit_sha256'], sources, 'audit')
    if audit['source_manifest_sha256'] != SOURCES['run'][1] or audit['status'] != 'passed_metadata_not_clinical':
        raise ValueError('pinned_prior_metadata_audit_required')
    data = {}
    for name in ('score_rows.json', 'execution_summary.json'):
        data[name] = _checked_json(SOURCES['run'][0] / name, run['artifacts'][name]['sha256'], sources, name)
    for label, expected in plan['source_pins'].items():
        # Source code is small; no checkpoint/patient artifact pins are loaded.
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise ValueError('typed_frozen_source_pin_required')
        p = require_inside(label, ROOT.parent, must_exist=True)
        if p.suffix == '.py' and p.stat().st_size < 1024 ** 2:
            if sha256_file(p) != expected:
                raise ValueError('sealed_source_helper_changed')
            sources['sealed_' + _digest(label)[:16]] = p
    sources.update(worker=Path(__file__), tests=ROOT / 'tests/test_fresh_output_acceptance.py',
        fixture_tests=ROOT / 'tests/test_live_receipts.py',
        protocol=ROOT.parent / 'docs/fresh_output_acceptance_protocol.md',
        cpu_guard_source=Path(sys.modules['score_free_random_control'].__file__),
        atomic_contracts=Path(sys.modules['contracts'].__file__))
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith('tricompose_v12') and getattr(module, '__file__', None):
            sources[name] = Path(module.__file__)
    return plan, data, sources


def run_smoke(output_root, run_id):
    cpu_guard()
    started = time.monotonic()
    plan, data, sources = load_archived_smoke()
    before = {k: sha256_file(p) for k, p in sources.items()}
    rows = {r['triple_candidate_id']: r for r in data['score_rows.json']['records']}
    books = {b['case_id']: b for b in data['execution_summary.json']['case_ledgers']}
    if len(plan['cases']) != 2 or len(books) != 2 or len(rows) != len(data['score_rows.json']['records']):
        raise ValueError('exact_archived_two_case_inventory_required')
    decisions = []
    for case in plan['cases']:
        ctx = context(case['anchor'], plan['workers']['xrv'], plan['workers']['chexbert'])
        base, static = case['baseline'], case['static']
        cached = {'kind': 'pinned_cached_reference', 'source_manifest_sha256': SOURCES['plan'][1]}
        observed = [{'row': base, 'origin': cached}]
        if static['triple_candidate_id'] != base['triple_candidate_id']:
            observed.append({'row': static, 'origin': cached})
        decisions.append({'check': 'observed_same_image_reference', 'decision': assess_fresh_output(
            base['triple_candidate_id'], static['triple_candidate_id'], observed, ctx,
            ledger_snapshot=books[case['case_id']])})
        new = [r for cid, r in rows.items() if r['case_id'] == case['case_id']
            and cid not in {base['triple_candidate_id'], static['triple_candidate_id']}]
        if len(new) != 1 or any(rows.get(r['triple_candidate_id']) != r for r in (base, static)):
            raise ValueError('one_archived_retry_and_unchanged_fixed_references_required')
        observed.append({'row': new[0], 'origin': {'kind': 'completed_ledger_receipt'}})
        decisions.append({'check': 'completed_new_image_receipt', 'decision': assess_fresh_output(
            base['triple_candidate_id'], new[0]['triple_candidate_id'], observed, ctx,
            ledger_snapshot=books[case['case_id']])})
    summary = {'schema_version': SCHEMA, 'cohort_role': 'already_inspected_development_archived_contract_smoke',
        'fixed_ehr_cases': 2, 'decision_checks': len(decisions),
        'statuses': dict(Counter(r['decision']['status'] for r in decisions)),
        'completed_image_proposals_passing': sum(r['decision']['proposal_passes_proxy_preservation']
            for r in decisions if r['check'] == 'completed_new_image_receipt'),
        'original_charged_attempts_retained_once_per_case': sum(b['charged_model_attempts'] for b in books.values()),
        'new_model_calls': 0, 'new_gpu_tasks': 0, 'bodies_or_pixels_opened': False,
        'clinical_acceptance': False, 'clinical_repair_success': False,
        'installed_in_live_controller': False, 'actual_new_prospective_experiment': False,
        'cpu_seconds_before_serialization': round(time.monotonic() - started, 6)}
    temp, target = new_atomic_run(output_root, run_id)
    try:
        write_private_json(temp / 'decisions.json', {'records': decisions})
        write_private_json(temp / 'summary.json', summary)
        report = ('# Fresh receipt adapter / 新回执接口\n\n'
            'Archived DEVELOPMENT metadata-only contract smoke, not new inference or clinical repair.\n\n'
            f"2 fixed EHRs; {len(decisions)} checks; "
            f"{summary['completed_image_proposals_passing']}/2 image proposals pass the proxy veto.\n\n"
            '固定 EHR、八头 mask、三条边、观察边界和完整成本账本均核验；没有重新推理、正文或像素读取。\n\n'
            '成本按病例保留，不能将两个决策检查重复计成两次真实调用。旧的 8 次尝试不退款；历史成本不为零。\n\n'
            '本接口只验收已完成候选，不选择下一动作，不授权 GPU，不代表临床正确；尚未安装到在线控制器。\n')
        write_private_text(temp / 'RESULTS_CN_EN.md', report)
        if any(sha256_file(p) != before[k] for k, p in sources.items()):
            raise ValueError('sources_changed_during_contract_smoke')
        write_private_json(temp / 'manifest.json', {'schema_version': SCHEMA,
            'status': 'completed_archived_metadata_contract_smoke_not_prospective',
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temp.iterdir()},
            'new_model_calls': 0, 'clinical_acceptance': False})
        commit_atomic_run(temp, target)
    except BaseException:
        discard_atomic_run(temp)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        root, summary = run_smoke(args.output_root, args.run_id)
    except Exception as exc:
        print(json.dumps({'status': 'contract_smoke_failed', 'error_type': type(exc).__name__}))
        return 1
    print(json.dumps({'status': 'metadata_contract_smoke_complete', 'new_model_calls': 0,
        'checks': summary['decision_checks'], 'manifest_sha256': sha256_file(root / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
