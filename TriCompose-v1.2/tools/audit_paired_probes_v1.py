#!/usr/bin/env python3
"""Existing CPU Slurm: authenticate derived paired metadata, never bodies.

This audit does not reopen report text, pixels or checkpoints. It recomputes
the sealed developer replay and summarizes all declared cases, including
incomplete ones. A successful metadata audit is not clinical verification.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import collect_paired_probes_v1 as paired
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.live_workers import check_pins

VERSION = 'tricompose-paired-probe-metadata-audit-v1'


def matrix(rows, triples, plan):
    cases = {c['case_id']:c for c in plan['cases']}
    expected = {(case,*slot,model) for case in cases for slot in paired.POLICY['image_order']
        for model in paired.POLICY['report_models']}
    found, ids, hashes = set(), set(), {}
    if len(rows)!=len(triples) or len(rows)>8:
        raise ValueError('bounded_equal_derived_and_generated_inventory_required')
    for row in rows:
        slot=(row['case_id'],row['cxr_model_id'],row['seed'],row['report_model_id'])
        if slot not in expected or slot in found or row['triple_candidate_id'] in ids:
            raise ValueError('unique_predeclared_paired_slot_required')
        found.add(slot);ids.add(row['triple_candidate_id'])
        case=cases[row['case_id']]
        ctx=paired.fresh.context(case['anchor'],plan['workers']['xrv'],plan['workers']['chexbert'])
        paired.fresh_snapshot(row,ctx)
        image_key=slot[:3]
        signature=(row['cxr_candidate_id'],row['cxr_sha256'],
            tuple((f['finding'],f['xrv']) for f in row['receipt']['fact_states']))
        if hashes.setdefault(image_key,signature)!=signature:
            raise ValueError('shared_image_and_classifier_must_be_identical')
        bound=[t for t in triples if all(t[k]==row[k] for k in
            ('case_id','cxr_model_id','seed','report_model_id','cxr_sha256','report_sha256',
                'ehr_sha256','ehr_facts_sha256'))]
        if len(bound)!=1:raise ValueError('one_authenticated_generated_triple_per_score_row_required')
    return {'planned_slots':len(expected),'completed_slots':len(found),'missing_slots':len(expected-found),
        'unique_image_sha256':len({r['cxr_sha256'] for r in rows}),
        'unique_report_sha256':len({r['report_sha256'] for r in rows}),
        'clinical_repair_success':None,'all_declared_ehrs_retained':True}


def candidate_table(rows):
    output=[]
    fields=('known_reference_facts','comparable_facts','supported_facts','supported_positive',
        'supported_negative','proxy_opposition_facts','missing_comparisons','coverage_over_known','support_over_known')
    for row in rows:
        r={k:row[k] for k in ('case_id','triple_candidate_id','cxr_model_id','seed','report_model_id')}
        r.update({k:row['structure'][k] for k in ('section_contract_pass','empty','generic_report',
            'unsupported_temporal_comparison_language','repeated_sentence_count','repeated_4gram_ratio')})
        for edge in ('ehr_cxr','ehr_report','cxr_report'):
            r.update({edge+'_'+k:row['raw_edge_readouts'][edge][k] for k in fields})
        r.update(clinical_accuracy=None,biovil_raw_cosine=None,secondary_status='not_executed')
        output.append(r)
    return output


def method_table(readouts,rows,books):
    index={r['triple_candidate_id']:r for r in rows};output=[]
    methods=('fixed','observed_probe_replay','without_feedback_replay')
    fields=('known_reference_facts','comparable_facts','supported_facts',
        'proxy_opposition_facts','missing_comparisons','coverage_over_known','support_over_known')
    for case in readouts:
        choices={c['method']:c for c in case['readout']['choices']}
        for method in methods:
            choice=choices.get(method);cid=choice['selected_candidate_id'] if choice else None
            if cid is not None and index[cid]['case_id']!=case['case_id']:
                raise ValueError('method_selection_must_keep_fixed_case')
            row=index.get(cid)
            result={'case_id':case['case_id'],'method':method,'selected_candidate_id':cid,
                'selected_available':row is not None,
                'simulated_replay_calls':choice['simulated_calls'] if choice else None,
                'actual_shared_collection_attempts':sum(b['charged_model_attempts'] for b in books if b['case_id']==case['case_id']),
                'selection_was_online':False,'clinical_accuracy':None,'biovil_raw_cosine':None,
                'secondary_status':'not_executed'}
            for edge in ('ehr_cxr','ehr_report','cxr_report'):
                for name in fields:
                    result[edge+'_'+name]=row['raw_edge_readouts'][edge][name] if row is not None else None
            output.append(result)
    return output


def run(args):
    cpu_guard()
    sources={}
    plan_root=require_inside(args.plan_run,PROTECTED_ROOT,must_exist=True)
    receipt=json.loads(checked(plan_root/'manifest.json',args.plan_manifest_sha256,1024**2,sources))
    plan=json.loads(checked(plan_root/'plan.json',receipt['plan_sha256'],4*1024**2,sources))
    paired.validate_plan(plan)
    code_pins={str(p.resolve()):sha256_file(p) for p in (Path(__file__),
        ROOT/'tests/test_paired_probe_audit_v1.py',ROOT/'tools/benchmark_probe_repair_v1.py')}
    # Workspace code only, never inherited checkpoint/asset pins in this CPU job.
    if any(not Path(k).resolve().is_relative_to(paired.WORKSPACE) or Path(k).suffix not in ('.py','.md')
           for k in plan['source_pins']):raise ValueError('workspace_source_code_only_required')
    check_pins(plan['source_pins'])
    root=require_inside(args.source_run,PROTECTED_ROOT,must_exist=True)
    manifest=json.loads(checked(root/'manifest.json',args.source_manifest_sha256,1024**2,sources))
    if (manifest['schema_version']!=paired.VERSION or manifest['status']!='paired_collection_complete_unvalidated'
            or manifest['old_ehr_prompts_and_winners_changed'] is not False
            or manifest['secondary_endpoint_executed'] is not False
            or manifest['online_adaptive_repair_executed'] is not False):
        raise ValueError('exact_unvalidated_paired_collection_required')
    documents={}
    for name,pin in manifest['artifacts'].items():
        text=checked(root/name,pin['sha256'],16*1024**2,sources)
        if name.endswith('.json'):documents[name]=json.loads(text)
    rows=documents['score_rows.json']['records'];triples=documents['completed_triplets.json']['records']
    inventory=matrix(rows,triples,plan)
    actual_readouts=[]
    for case in plan['cases']:
        ctx=paired.fresh.context(case['anchor'],plan['workers']['xrv'],plan['workers']['chexbert'])
        actual_readouts.append({'case_id':case['case_id'],
            'readout':paired.paired_readout([r for r in rows if r['case_id']==case['case_id']],ctx)})
    if actual_readouts!=documents['choices_and_action_credits.json']['records'] \
            or sha256_file(root/'choices_and_action_credits.json')!=manifest['selection_sha256_before_secondary']:
        raise ValueError('sealed_actions_and_choices_must_replay_exactly')
    books=documents['execution_summary.json']['case_ledgers']
    if len(books)!=4 or Counter(b['case_id'] for b in books)!=Counter({c['case_id']:2 for c in plan['cases']}):
        raise ValueError('two_paid_image_branch_ledgers_per_fixed_ehr_required')
    for book in books:
        if book['call_budget']!=6 or book['max_retries']!=0 or book['execution_mode']!='approved_slurm_backend':
            raise ValueError('bounded_real_worker_branch_required')
        replay=restore_ledger(book['events'],case_id=book['case_id'],ehr_anchor_sha256=book['ehr_anchor_sha256'],
            call_budget=6,max_retries=0,execution_mode=book['execution_mode'],sink=lambda e:None)
        if replay.snapshot()!=book:raise ValueError('durable_attempt_accounting_required')
    for case_index,case in enumerate(plan['cases']):
        for branch in range(2):
            book=books[case_index*2+branch]
            base=root/f'image_branch_{branch}'/'cases'/case['case_id']
            journal=base/'execution.journal.jsonl';snapshot=base/'ledger_snapshot.json'
            events=[json.loads(line) for line in checked(journal,sha256_file(journal),16*1024**2,sources).splitlines()]
            if (events!=book['events'] or json.loads(checked(snapshot,sha256_file(snapshot),16*1024**2,sources))!=book
                    or book['case_id']!=case['case_id']
                    or book['ehr_anchor_sha256']!=case['ehr_anchor_sha256']):
                raise ValueError('exact_branch_journal_and_immutable_anchor_required')
            anchor=paired.anchor_from_record(case['anchor'])
            paired.fresh.validate_ledger(book,anchor)
            for row in rows:
                if row['case_id']==case['case_id'] and row['seed']==paired.POLICY['image_order'][branch][1]:
                    paired.fresh.completed_operation(row,book)
    calls=sum(b['charged_model_attempts'] for b in books)
    if calls!=manifest['charged_model_attempts'] or calls>24:raise ValueError('all_attempts_including_failures_counted')
    summary={'schema_version':VERSION,'status':'metadata_audit_passed_not_clinical',**inventory,
        'fixed_ehr_cases':2,'charged_model_attempts':calls,'new_model_calls_in_audit':0,
        'failed_attempts':sum(b['failed_attempts'] for b in books),
        'policy_replay_exact':True,'original_winners_changed':False,
        'online_adaptive_repair_executed':False,'source_bodies_pixels_or_checkpoints_read':False,
        'secondary_endpoint_executed':False}
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        write_private_json(temporary/'summary.json',summary)
        table=candidate_table(rows)
        if table:write_private_text(temporary/'candidate_scores.csv',csv_text(table))
        write_private_text(temporary/'method_comparison.csv',csv_text(method_table(actual_readouts,rows,books)))
        write_private_json(temporary/'case_readouts.json',{'records':actual_readouts})
        if any(sha256_file(p)!=pin for p,pin in sources.items()):raise ValueError('source_changed_during_audit')
        check_pins(code_pins)
        artifacts={f.name:{'sha256':sha256_file(f)} for f in temporary.iterdir() if f.is_file()}
        write_private_json(temporary/'manifest.json',{'schema_version':VERSION,'sources':sources,'artifacts':artifacts,
            'code_pins':code_pins,'source_manifest_sha256':args.source_manifest_sha256,
            'clinical_qualified':False,'new_model_calls':0})
        commit_atomic_run(temporary,target)
    except BaseException:discard_atomic_run(temporary);raise
    return target,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan-run','plan-manifest-sha256','source-run','source-manifest-sha256','run-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--output-root',type=Path,default=paired.BASE/'paired_probe_audits')
    args=parser.parse_args()
    try:
        target,summary=run(args)
        print(json.dumps({'status':summary['status'],'manifest_sha256':sha256_file(target/'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
