#!/usr/bin/env python3
"""Prepared only: approved GPU Slurm, real report-label diagnostic, no training.

Only derived opaque predictions/aggregate metrics/hashes are exported privately.
Never exports report text, source IDs/paths, EHR fields, images or reference rows.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE=Path('/project2/ruishanl_1185/inference_3mod')
sys.path.insert(0,str(WORKSPACE/'TriCompose-v1.0/eval/report_v1_1'))
sys.path.insert(0,str(WORKSPACE/'TriCompose-v1.2/src'))
sys.path.insert(0,str(WORKSPACE/'TriCompose-v1.2/benchmarks'))
from contracts import (commit_atomic_run,discard_atomic_run,new_atomic_run,
    read_json,sha256_file,write_private_json,write_private_text)
from audit_official_report_gold import source_file,safe_error_code
from official_report_benchmark import (SCHEMA,HEADS,GATE_FINDINGS,collect_sources,
    decode,input_normalization,select_impression,summarize,markdown)

CHECKPOINT_SHA='6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1'
ADAPTER_SHA='5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6'
GUARD_SHA='7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18'
GATE_SHA='00459f4c8c9faf009d60b56529991a690a53ac2b782e9ea1c21604c2789779cd'
AUDIT_MANIFEST_SHA='bf0b5064c5a477bc673196080e2c16b2dc7dc15672f6cf589161761ddfe77fc4'
SAFE_FAILURE_CODES=frozenset({
    'approved_real_report_slurm_required','slurm_cgroup_required','batch_size_out_of_bounds','cuda_required',
    'metadata_audit_binding_changed','metadata_audit_summary_changed','metadata_audit_incomplete',
    'metadata_sources_changed','metadata_sources_changed_during_run','source_report_changed',
    'frozen_model_or_guard_changed','frozen_assets_changed_during_run','model_not_frozen',
    'prediction_batch_length_changed','prediction_width_mismatch','prediction_class_mismatch',
    'annotation_schema_mismatch','malformed_annotation_row','duplicate_annotation_study',
    'annotation_row_bound_exceeded','empty_annotation_source','linkage_schema_mismatch',
    'linkage_row_bound_exceeded','malformed_linkage_row','unsupported_source_split',
    'patient_split_overlap','inconsistent_study_linkage','eligible_report_bound_exceeded',
    'invalid_four_state_pair','duplicate_prediction_item','prediction_inventory_mismatch',
    'invalid_completed_prediction','invalid_prediction_state','invalid_scope_decision',
    'scope_cannot_flip_or_promote_prediction','noncommit_must_remain_unknown','unavailable_item_has_prediction'})


def failure_code(exc):
    code=str(exc)
    return code if code in SAFE_FAILURE_CODES else safe_error_code(exc)


def require_approved_slurm(args):
    job=os.environ.get('SLURM_JOB_ID','')
    if not args.allow_real_report_benchmark or not job.isdigit():
        raise RuntimeError('approved_real_report_slurm_required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('slurm_cgroup_required')


def read_report(path_text,base):
    """Only the authorized model job calls this. No exception text is returned."""
    try:
        candidate=Path(path_text)
        if not candidate.is_absolute():candidate=base/candidate
        path=source_file(candidate)
        if path.stat().st_size>256*1024:return None,None,'report_size_bound_exceeded'
        before=sha256_file(path)
        text=path.read_text(encoding='utf-8',errors='strict')
        if sha256_file(path)!=before:raise RuntimeError('source_report_changed')
        return text,before,None
    except FileNotFoundError:return None,None,'report_file_missing'
    except PermissionError:return None,None,'report_file_unreadable'
    except UnicodeError:return None,None,'report_encoding_unsupported'
    except ValueError:return None,None,'report_path_boundary_rejected'


def run(args):
    require_approved_slurm(args)
    if not 1<=args.batch_size<=16:raise ValueError('batch_size_out_of_bounds')
    with open(os.devnull,'w') as devnull,contextlib.redirect_stdout(devnull),contextlib.redirect_stderr(devnull):
        import torch
    if not torch.cuda.is_available():raise RuntimeError('cuda_required')
    audit_root=Path(args.metadata_audit_run)
    if sha256_file(audit_root/'manifest.json')!=AUDIT_MANIFEST_SHA:
        raise ValueError('metadata_audit_binding_changed')
    audit_manifest=read_json(audit_root/'manifest.json')
    audited=read_json(audit_root/'summary.json')
    if sha256_file(audit_root/'summary.json')!=audit_manifest['artifacts']['summary.json']['sha256']:
        raise ValueError('metadata_audit_summary_changed')
    if audited['status']!='metadata_coverage_audit_only_not_model_validation':
        raise ValueError('metadata_audit_incomplete')
    gold,linkage=source_file(args.gold_labels),source_file(args.linkage_manifest)
    sources={'gold_annotation_csv':gold,'matched_linkage_manifest':linkage}
    fingerprints={name:sha256_file(path) for name,path in sources.items()}
    if fingerprints!=audited['source_sha256']:raise ValueError('metadata_sources_changed')
    previous_limit=csv.field_size_limit()
    try:
        csv.field_size_limit(16*1024*1024)
        with gold.open(encoding='utf-8-sig',newline='') as first,linkage.open(encoding='utf-8-sig',newline='') as second:
            items,conditions=collect_sources(csv.DictReader(first),csv.DictReader(second))
    finally:csv.field_size_limit(previous_limit)
    texts={};records=[]
    for item in items:
        record={'item_id':item['item_id'],'status':item['status'],'finding_states':None,
            'source_report_sha256':None,'selected_input_sha256':None,'token_count':None}
        if item['status']=='ready_metadata':
            report,report_sha,error=read_report(item['report_path'],linkage.parent)
            if error:record['status']=error
            else:
                selected,reason=select_impression(report)
                record['source_report_sha256']=report_sha
                if selected is None:record['status']=reason
                else:
                    texts[item['item_id']]=selected
                    record['status']='ready_text'
                    record['selected_input_sha256']=hashlib.sha256(selected.encode('utf-8')).hexdigest()
        records.append(record)
    by_id={record['item_id']:record for record in records}
    torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
    checkpoint,bert=Path(args.checkpoint).resolve(strict=True),Path(args.bert_path).resolve(strict=True)
    assets={checkpoint:CHECKPOINT_SHA,WORKSPACE/'cxrmate/tools/chexbert.py':ADAPTER_SHA,
        bert/'config.json':'7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
        bert/'vocab.txt':'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        bert/'tokenizer_config.json':'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
        WORKSPACE/'TriCompose-v1.2/benchmarks/repair_cached_report_evidence.py':GUARD_SHA,
        WORKSPACE/'TriCompose-v1.2/src/tricompose_v12/report_assertions.py':GATE_SHA}
    if any(sha256_file(path)!=expected for path,expected in assets.items()):
        raise ValueError('frozen_model_or_guard_changed')
    asset_stats={path:(path.stat().st_size,path.stat().st_mtime_ns) for path in assets}
    batches=examples=0;replay_changes=[]
    with open(os.devnull,'w') as devnull,contextlib.redirect_stdout(devnull),contextlib.redirect_stderr(devnull):
        from tools.chexbert import CheXbert
        from tricompose_v12.report_assertions import gate_assertion
        from repair_cached_report_evidence import scope_check
        model=CheXbert(ckpt_dir=str(checkpoint.parent),bert_path=str(bert),checkpoint_path=checkpoint.name,
            device=torch.device('cuda:0')).to('cuda:0')
        model.eval().requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError('model_not_frozen')
        ready=[]
        for item_id,text in texts.items():
            token_ids=model.tokenizer(input_normalization(text),truncation=False)['input_ids']
            by_id[item_id]['token_count']=len(token_ids)
            if len(token_ids)>model.bert.config.max_position_embeddings:
                by_id[item_id]['status']='token_budget_exceeded_no_truncation'
            else:ready.append(item_id)
        for start in range(0,len(ready),args.batch_size):
            ids=ready[start:start+args.batch_size]
            with torch.inference_mode():
                values=model([texts[item_id] for item_id in ids]).detach().cpu().tolist()
            if len(values)!=len(ids):raise ValueError('prediction_batch_length_changed')
            batches+=1;examples+=len(ids)
            for item_id,raw in zip(ids,values,strict=True):
                states=decode(raw);record=by_id[item_id]
                record.update(status='complete',finding_states=states)
                decisions={}
                for finding in GATE_FINDINGS:
                    decision=gate_assertion(texts[item_id],finding,states[finding],scope_check)
                    # Do not persist evidence quotes, spans or source report text.
                    decisions[finding]={key:decision[key] for key in ('decision','state','reason','scope_verified')}
                record['scope_decisions']=decisions
        # Predeclared first-eight replay, not gold-selected samples.
        for item_id in ready[:8]:
            with torch.inference_mode():
                replay=decode(model([texts[item_id]]).detach().cpu().tolist()[0])
            batches+=1;examples+=1
            replay_changes.append({'item_id':item_id,'changed_heads':
                [head for head in HEADS if replay[head]!=by_id[item_id]['finding_states'][head]]})
    torch.cuda.synchronize()
    if any((path.stat().st_size,path.stat().st_mtime_ns)!=asset_stats[path] for path in assets):
        raise ValueError('frozen_assets_changed_during_run')
    if {name:sha256_file(path) for name,path in sources.items()}!=fingerprints:
        raise ValueError('metadata_sources_changed_during_run')
    result=summarize(items,records,conditions)
    result.update(source_sha256=fingerprints,source_audit_manifest_sha256=AUDIT_MANIFEST_SHA,
        producer={'model_id':'chexbert','frozen':True,'checkpoint_sha256':CHECKPOINT_SHA,'adapter_sha256':ADAPTER_SHA,
            'bert_config_sha256':assets[bert/'config.json'],'tokenizer_vocab_sha256':assets[bert/'vocab.txt'],
            'tokenizer_config_sha256':assets[bert/'tokenizer_config.json'],
            'model_head_order':list(HEADS),'class_mapping':{'0':'unknown','1':'positive','2':'negative','3':'uncertain'},
            'no_finding_head_classes':2,'scope_library_sha256':GATE_SHA,'scope_guard_sha256':GUARD_SHA},
        execution={'gpu_name':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__),
            'seed':0,'batch_size':args.batch_size,'forward_batches':batches,'encoder_examples_including_replay':examples},
        replay={'policy':'first_eight_completed_inputs_batch_one','items':replay_changes,
            'changed_items':sum(bool(row['changed_heads']) for row in replay_changes)},
        truncation_used=False,raw_ehr_fields_inspected=False,image_pixels_read=False,
        raw_report_text_written=False,patient_keys_written=False,source_paths_written=False,
        peak_allocated_vram_including_load_gib=round(torch.cuda.max_memory_allocated()/1024**3,3))
    return result,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gold-labels','linkage-manifest','metadata-audit-run','checkpoint','bert-path','output-root','run-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--allow-real-report-benchmark',action='store_true')
    args=parser.parse_args();temporary=None;started=time.monotonic();os.umask(0o007)
    try:
        require_approved_slurm(args)
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        summary,records=run(args);summary['elapsed_seconds']=round(time.monotonic()-started,3)
        files=[write_private_json(temporary/'summary.json',summary),
            write_private_text(temporary/'summary.md',markdown(summary)),
            write_private_json(temporary/'predictions.json',{'schema_version':SCHEMA,'records':records})]
        write_private_json(temporary/'manifest.json',{'schema_version':SCHEMA,'run_id':args.run_id,
            'program_sha256':sha256_file(__file__),'helper_sha256':sha256_file(Path(__file__).with_name('official_report_benchmark.py')),
            'source_sha256':summary['source_sha256'],'source_audit_manifest_sha256':AUDIT_MANIFEST_SHA,
            'patient_keys_or_source_report_text_written':False,'primary_metric_eligible':False,
            'selection_changed':False,'regeneration_authorized':False,
            'artifacts':{path.name:{'sha256':sha256_file(path)} for path in files}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:discard_atomic_run(temporary)
        print(json.dumps({'status':'failed_real_report_benchmark','error_type':type(exc).__name__,
            'error_code':failure_code(exc)}))
        return 1
    print(json.dumps({'status':'completed_real_report_label_diagnostic',
        'elapsed_seconds':summary['elapsed_seconds'],
        'peak_allocated_vram_gib':summary['peak_allocated_vram_including_load_gib'],
        'manifest_sha256':sha256_file(target/'manifest.json')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
