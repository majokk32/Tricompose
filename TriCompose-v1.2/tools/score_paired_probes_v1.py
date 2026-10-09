#!/usr/bin/env python3
"""Frozen secondary endpoint for the already sealed paired development bank.

Inspect is cache-only CPU Slurm. Run requires separately approved GPU Slurm.
No generation, decision changes, training, gold labels or payloads in tables.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import audit_paired_probes_v1 as audit
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, read_json,
    private_directory, write_private_json, write_private_text, load_cxr_candidates)
from tricompose_v12.live_workers import check_pins, require_gpu_slurm, run_private_process
from tricompose_v12.full_pool_report_control import validate_endpoint
from tricompose_v11.prompts import ROENTGEN_FINDING_PHRASES
from tricompose_v11.tokenizer_trace import text_sha256
from tricompose_v12.live_execution import validate_cxr_binding
from run_fixed_image_reports import normalize_owned_modes
from score_automatic_replay_biovil import PAIR_FIELDS, MODEL_HASHES, REQUEST_SCHEMA, SCORE_SCHEMA

VERSION = 'tricompose-paired-probe-secondary-v1'
REQUEST_MANIFEST_SHA = '8fd402fd7071e556948163fb8c333ac4b99a0303c0fd6cde565f04775f2b9033'
REQUEST_SHA = 'fdc3c72bb0d5f54159d5fd9edcb1ec7b7fac89ad5a1d27695c043d83e197b669'
MAX_COUNTS = {'requested_pairs': 8, 'image_encoder_calls': 4, 'text_encoder_calls': 8}
TEXT_POLICY = 'full_report_no_silent_truncation_overlength_is_na'


def load_metadata(args):
    """Only bounded authenticated metadata; callers must enforce allocation."""
    sources = {}
    def document(root, name, pin, limit=4*1024**2):
        return json.loads(checked(root/name, pin, limit, sources))
    plan_root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    pm = document(plan_root, 'manifest.json', args.plan_manifest_sha256)
    plan = document(plan_root, 'plan.json', pm['plan_sha256'])
    audit.paired.validate_plan(plan)
    if any(not Path(k).resolve().is_relative_to(audit.paired.WORKSPACE)
           or Path(k).suffix not in ('.py', '.md') for k in plan['source_pins']):
        raise ValueError('workspace_code_only_required')
    check_pins(plan['source_pins'])
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    manifest = document(root, 'manifest.json', args.source_manifest_sha256)
    if (manifest['schema_version'] != audit.paired.VERSION
            or manifest['status'] != 'paired_collection_complete_unvalidated'
            or manifest['secondary_endpoint_executed'] is not False
            or manifest['online_adaptive_repair_executed'] is not False
            or manifest['old_ehr_prompts_and_winners_changed'] is not False
            or manifest['charged_model_attempts'] != 24 or manifest['failed_attempts'] != 0):
        raise ValueError('exact_completed_unmodified_paired_collection_required')
    docs = {n: document(root, n, manifest['artifacts'][n]['sha256']) for n in
        ('score_rows.json', 'completed_triplets.json', 'choices_and_action_credits.json', 'execution_summary.json')}
    rows, triples = docs['score_rows.json']['records'], docs['completed_triplets.json']['records']
    inventory = audit.matrix(rows, triples, plan)
    if inventory['completed_slots'] != 8 or inventory['missing_slots'] != 0:
        raise ValueError('all_eight_sealed_pairs_required')
    seal = sources[str(root/'choices_and_action_credits.json')]
    if seal != manifest['selection_sha256_before_secondary']:
        raise ValueError('original_choices_must_remain_sealed')
    ar = require_inside(args.audit_run, PROTECTED_ROOT, must_exist=True)
    am = document(ar, 'manifest.json', args.audit_manifest_sha256)
    if (am['schema_version'] != audit.VERSION or am['source_manifest_sha256'] != args.source_manifest_sha256
            or am['clinical_qualified'] is not False or am['new_model_calls'] != 0):
        raise ValueError('matching_metadata_audit_required')
    summary = document(ar, 'summary.json', am['artifacts']['summary.json']['sha256'])
    if summary['status'] != 'metadata_audit_passed_not_clinical' or summary['completed_slots'] != 8:
        raise ValueError('completed_metadata_audit_required')
    request_root = root/'endpoint_request'
    rm = document(request_root, 'manifest.json', REQUEST_MANIFEST_SHA)
    request = document(request_root, 'request.json', REQUEST_SHA)
    if (rm['schema_version'] != REQUEST_SCHEMA or rm['artifacts']['request.json']['sha256'] != REQUEST_SHA
            or request['schema_version'] != REQUEST_SCHEMA or request['modality_source'] != 'fully_synthetic'
            or request['selection_used_biovil'] is not False or request['clinical_truth_available'] is not False
            or request['routing_or_calibration_update_allowed'] is not False
            or request['text_policy'] != TEXT_POLICY or request['selection_sha256_sealed_before_endpoint'] != seal
            or request['pairs'] != [{k:r[k] for k in PAIR_FIELDS} for r in rows]):
        raise ValueError('unchanged_secondary_only_request_required')
    return dict(plan=plan, rows=rows, triples=triples, sources=sources, source_root=root,
        request_root=request_root, seal=seal, inventory=inventory,
        readouts=docs['choices_and_action_credits.json']['records'],
        books=docs['execution_summary.json']['case_ledgers'])


def conditioning_row(candidate, request, anchor, trace):
    """Lexical/length transfer only, NOT radiographic truth or encoder-hook proof.

    Runtime trace body is permitted only inside the approved GPU job. Tests
    supply invented text. No supplied/observed text is returned or logged.
    """
    prompt = request['inputs']['final_prompt']
    text = trace['positive_tokenizer_text']
    count, attention = candidate['cost']['prompt_token_count'], trace['attention_token_count']
    if (candidate['model_id'] != 'roentgen_v2' or candidate['prompt_sha256'] != prompt['sha256']
            or trace['runtime_observed'] is not True or trace['supplied_prompt_sha256'] != prompt['sha256']
            or trace['pipeline_changed_text'] is not False
            or trace['official_generation_arguments_changed'] is not False
            or trace['positive_tokenizer_text_sha256'] != prompt['sha256']
            or text_sha256(text) != prompt['sha256']
            or type(count) is not int or type(attention) is not int or not 1 <= count <= 77
            or trace['padded_token_count'] != 77 or count != attention):
        raise ValueError('unchanged_untruncated_roentgen_runtime_boundary_required')
    included = prompt['included_direct_fact_ids']
    if len(included) != len(set(included)) or any(f not in ROENTGEN_FINDING_PHRASES for f in included):
        raise ValueError('recognized_unique_rendered_fact_ids_required')
    matched = {f for f in included if ROENTGEN_FINDING_PHRASES[f] in text.lower()}
    # State inventories differ: no unsupported negative/unknown or device
    # head is promoted into a radiographic requirement.
    aliases = {'edema': ['pulmonary_edema'], 'support_devices': [
        'endotracheal_tube', 'central_venous_catheter', 'enteric_tube', 'cardiac_pacemaker']}
    known = [f['finding'] for f in anchor['findings'] if f['state'] == 'positive']
    mapped = {f: aliases.get(f, [f]) for f in known}
    observable_mapping = {f: ids for f, ids in mapped.items() if any(i in ROENTGEN_FINDING_PHRASES for i in ids)}
    return {'case_id': candidate['case_id'], 'cxr_candidate_id': candidate['candidate_id'],
        'seed': candidate['seed'], 'prompt_sha256': prompt['sha256'], 'prompt_token_count': count,
        'attention_token_count': attention, 'runtime_text_unchanged': True,
        'text_encoder_hook_observed': False, 'lexical_checks_are_clinical_truth': False,
        'included_direct_fact_count': len(included), 'matched_phrase_count': len(matched),
        'missing_phrase_count': len(included)-len(matched), 'known_positive_anchor_count': len(known),
        'mappable_positive_anchor_count': len(observable_mapping),
        'positive_anchor_phrase_match_count': sum(any(i in matched for i in ids) for ids in observable_mapping.values())}


def check_endpoint(rows, endpoint):
    if (endpoint.get('schema_version') != SCORE_SCHEMA or endpoint.get('status') != 'completed_secondary_biovil'
            or endpoint.get('request_sha256') != REQUEST_SHA or endpoint.get('original_selection_changed') is not False
            or endpoint.get('primary_clinical_metric') is not False
            or endpoint.get('producer', {}).get('frozen') is not True
            or endpoint['producer'].get('model_id') != 'biovil_t'
            or endpoint['producer'].get('checkpoint_sha256') != MODEL_HASHES
            or endpoint['producer'].get('text_policy') != TEXT_POLICY):
        raise ValueError('frozen_secondary_endpoint_provenance_required')
    scores = validate_endpoint(rows, endpoint)
    available = [r for r in endpoint['records'] if r['biovil_raw_cosine'] is not None]
    expected = {'requested_pairs': len(rows),
        'image_encoder_calls': len({r['cxr_candidate_id'] for r in available}),
        'text_encoder_calls': len({r['report_candidate_id'] for r in available}),
        'unavailable_reports': len({r['report_candidate_id'] for r in endpoint['records'] if r['biovil_raw_cosine'] is None})}
    if (endpoint.get('counts') != expected
            or any(type(endpoint['counts'][k]) is not int for k in expected)
            or any(expected[k] > cap for k,cap in MAX_COUNTS.items())):
        raise ValueError('bounded_complete_encoder_accounting_required')
    return scores


def endpoint_tables(data, endpoint):
    scores = check_endpoint(data['rows'], endpoint)
    candidate = audit.candidate_table(data['rows'])
    methods = audit.method_table(data['readouts'], data['rows'], data['books'])
    for table in (candidate, methods):
        for row in table:
            cid = row.get('selected_candidate_id', row.get('triple_candidate_id'))
            score = scores.get(cid)
            row.update(biovil_raw_cosine=score['biovil_raw_cosine'] if score else None,
                secondary_status=score['status'] if score else 'not_available_no_selected_pair',
                secondary_reason=score['reason'] if score else 'no_selected_pair',
                clinical_accuracy=None)
    actions = []
    for readout in data['readouts']:
        case = next(c for c in data['plan']['cases'] if c['case_id'] == readout['case_id'])
        ctx = audit.paired.fresh.context(case['anchor'], data['plan']['workers']['xrv'], data['plan']['workers']['chexbert'])
        lookup = {audit.paired.controller.digest(audit.paired.fresh_snapshot(r, ctx)):r['triple_candidate_id']
                  for r in data['rows'] if r['case_id'] == case['case_id']}
        for credit in readout['readout']['action_credits']:
            before = lookup[credit['before_observation_sha256']]
            after = lookup[credit['after_observation_sha256']]
            a,b = (scores[cid]['biovil_raw_cosine'] for cid in (before,after))
            actions.append({'case_id':case['case_id'], 'action':credit['action'],
                'before_candidate_id':before, 'after_candidate_id':after,
                'primary_target_gain':credit['target_gain_observed'],
                'primary_replacement_allowed':credit['replacement_allowed_under_proxy_contract'],
                'biovil_before':a, 'biovil_after':b, 'biovil_delta':b-a if a is not None and b is not None else None,
                'secondary_used_to_change_acceptance':False, 'clinical_repair_success':None})
    return candidate, methods, actions


def run(args):
    require_gpu_slurm()  # Before metadata, source bodies, model imports or writes.
    data = load_metadata(args)
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    if not audit.paired.RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError('new_opaque_run_id_required')
    private_directory(output, exist_ok=True)
    root = output/args.run_id
    private_directory(root)  # Stable exclusive directory; failed receipts retained.
    beginning = time.monotonic()
    code_pins = {str(p):sha256_file(p) for p in (Path(__file__), ROOT/'tests/test_paired_probe_secondary_v1.py')}
    write_private_json(root/'start_manifest.json', {'schema_version':VERSION, 'status':'in_progress',
        'source_manifest_sha256':args.source_manifest_sha256, 'selection_sha256_before_secondary':data['seal'],
        'source_pins':code_pins, 'generation_calls':0, 'training_allowed':False})
    reserved = False
    try:
        check_pins(data['plan']['biovil_asset_pins'])  # Weights only in approved GPU allocation.
        runtime = root/'endpoint_runtime'; private_directory(runtime)
        argv = [data['plan']['biovil_python'], str(ROOT/'tools/online_report_smoke.py'), 'biovil-worker',
            '--request-run', str(data['request_root']), '--model-path', data['plan']['biovil_model'],
            '--output-file', str(root/'native_endpoint.json')]
        for name in ('cxr_run', 'report_run'):
            for p in sorted({t[name] for t in data['triples']}): argv += ['--'+name.replace('_','-'), p]
        write_private_json(root/'endpoint_reservation.json', {'status':'reserved_before_spawn',
            'maximum_image_encoder_calls':4, 'maximum_text_encoder_calls':8, 'retries':0,
            'worker_process_timeout_seconds':300, 'generation_calls':0})
        reserved = True
        run_private_process(argv, runtime, 300)
        endpoint = read_json(root/'native_endpoint.json')
        endpoint['historical_pool_is_untouched_test'] = False
        candidate, methods, actions = endpoint_tables(data, endpoint)
        # Existing frozen loader authenticates request, image and trace binding.
        cxrs = load_cxr_candidates(sorted({t['cxr_run'] for t in data['triples']}))
        if len(cxrs) != 4: raise ValueError('four_authenticated_source_images_required')
        conditioning = []
        for image in cxrs.values():
            case = next(c for c in data['plan']['cases'] if c['case_id'] == image['case_id'])
            request_row = next(r for r in case['requests'] if r['request']['seed'] == image['seed'])
            validate_cxr_binding(image, request_row, audit.paired.anchor_from_record(case['anchor']))
            request = request_row['request']
            trace = read_json(image['tokenizer_input']['trace_path'])
            conditioning.append(conditioning_row(image, request, case['anchor'], trace))
        write_private_json(root/'scores.json', endpoint)
        for filename, table in (('candidate_scores.csv', candidate), ('method_comparison.csv', methods),
                ('paired_action_endpoints.csv', actions), ('conditioning_transfer.csv', conditioning)):
            write_private_text(root/filename, csv_text(table))
        summary = {'schema_version':VERSION, 'status':'paired_secondary_completed_not_clinical',
            'fixed_ehr_cases':2, 'candidate_pairs':8, 'encoder_counts':endpoint['counts'],
            'unique_image_hashes':data['inventory']['unique_image_sha256'],
            'unique_report_hashes':data['inventory']['unique_report_sha256'],
            'source_generation_verification_attempts':24, 'new_generation_calls':0,
            'source_selection_changed':False, 'selection_was_online':False,
            'endpoint_used_to_tune_policy':False, 'clinical_repair_success':None,
            'conditioning_phrase_missing_count':sum(r['missing_phrase_count'] for r in conditioning),
            'tokenizer_input_not_text_encoder_hook':True, 'runtime_seconds':time.monotonic()-beginning}
        write_private_json(root/'summary.json', summary)
        if any(sha256_file(p) != pin for p,pin in data['sources'].items()):
            raise ValueError('sealed_metadata_changed_during_measurement')
        check_pins(data['plan']['source_pins']); check_pins(code_pins)
        files = ('scores.json','candidate_scores.csv','method_comparison.csv','paired_action_endpoints.csv',
            'conditioning_transfer.csv','summary.json','start_manifest.json','endpoint_reservation.json','native_endpoint.json')
        write_private_json(root/'manifest.json', {'schema_version':VERSION, 'status':summary['status'],
            'sources':data['sources'], 'code_pins':code_pins, 'artifacts':{n:{'sha256':sha256_file(root/n)} for n in files},
            'selection_sha256_before_secondary':data['seal'], 'original_selection_changed':False,
            'clinical_qualified':False})
    except BaseException as exc:
        write_private_json(root/'failure_manifest.json', {'status':'failed_retained_no_automatic_resume',
            'error_type':type(exc).__name__, 'endpoint_reserved_before_spawn':reserved,
            'reserved_maximum_image_encodings':4 if reserved else 0,
            'reserved_maximum_text_encodings':8 if reserved else 0, 'new_generation_calls':0})
        raise
    finally:
        normalize_owned_modes(root)
    return root, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('inspect','run'))
    for n in ('plan-run','plan-manifest-sha256','source-run','source-manifest-sha256','audit-run','audit-manifest-sha256'):
        p.add_argument('--'+n, required=True)
    p.add_argument('--output-root', default=str(audit.paired.BASE/'paired_probe_endpoints'))
    p.add_argument('--run-id')
    args = p.parse_args()
    try:
        if args.mode == 'inspect':
            cpu_guard(); data = load_metadata(args)
            print(json.dumps({'status':'secondary_inputs_ready_gpu_not_submitted',
                'pairs':len(data['rows']), 'fixed_ehr_cases':len(data['plan']['cases']),
                'selection_sha256_before_secondary':data['seal'], 'maximum_encoder_counts':MAX_COUNTS}))
        else:
            root, summary = run(args)
            print(json.dumps({'status':summary['status'], 'manifest_sha256':sha256_file(root/'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__})); return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
