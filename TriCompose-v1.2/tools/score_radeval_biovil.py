"""Prepare metadata, then explicitly approved real-CXR CPU scoring only.

No source clinical text/path is displayed, and no model inference occurs in
prepare(). Runtime requires the separate pixel approval flag and real Slurm.
"""
import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
from unittest.mock import patch

import smoke_radgraph_xl as installed
from tricompose_v12.radeval_image_benchmark import join, evaluate

WORKSPACE = installed.WORKSPACE
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
LINK = BASE / 'radeval_image_linkage_runs/exact_index_12714150_001'
LINK_SHA = '251640846d5c89106e04cf67584f2842f175498b0a36a89e80f3c880b93d6d72'
EXPERT = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
EXPERT_SHA = '21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
MODEL = WORKSPACE / 'CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5'
VENDOR = WORKSPACE / 'runtime/vendor/hi_ml_multimodal_0_2_2/health_multimodal'
WEIGHTS = {'biovil_t_image_model_proj_size_128.pt': 'b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77',
           'pytorch_model.bin': '6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c'}
VENDOR_SHA = 'dceafc8b318bc478ee67435d9206cd98d7d8abff801fa0ce8c2685ae840a8992'
PLANS = BASE / 'radeval_image_benchmark_plans'
RUNS = BASE / 'radeval_image_benchmark_runs'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def allocation_guard():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'approved_existing_cpu_slurm_required')


def private_dir(path, exist_ok=False):
    path.mkdir(mode=0o2770, exist_ok=exist_ok)
    path.chmod(0o2770)
    require(path.stat().st_gid in (96293,65534), 'project_group_mount_required')


def verify_receipt(root, digest):
    require(installed.sha256(root / 'manifest.json') == digest, 'sealed_source_receipt_required')
    receipt = json.loads((root / 'manifest.json').read_text())
    for item in receipt.get('outputs', receipt.get('artifacts', [])):
        path = root / item['path']
        require(path.resolve().is_relative_to(root) and installed.sha256(path) == item['sha256'], 'sealed_source_artifact_required')
    return receipt


def metadata():
    # Model file hashes are safe metadata; this does not instantiate a model.
    for name, expected in WEIGHTS.items():
        require(installed.sha256(MODEL / name) == expected, 'frozen_biovil_checkpoint_required')
    vendor = {str(p.relative_to(VENDOR)): installed.sha256(p) for p in sorted(VENDOR.rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(vendor,sort_keys=True).encode()).hexdigest()
    require(digest == VENDOR_SHA, 'frozen_official_vendor_required')
    model_metadata = {p.name: installed.sha256(p) for p in sorted(MODEL.iterdir())
                      if p.is_file() and p.suffix in ('.json','.py','.txt')}
    return {'weight_hashes': WEIGHTS, 'vendor_tree_sha256': digest, 'model_metadata_hashes': model_metadata}


def preflight_inputs():
    verify_receipt(LINK,LINK_SHA)
    verify_receipt(EXPERT,EXPERT_SHA)
    plan = json.loads((EXPERT / 'frozen_plan.json').read_text())['inventory']
    resolver = json.loads((LINK / 'internal_resolver.json').read_text())
    available = set(resolver)
    require(len(available) == 43, 'sealed_linked_image_inventory_required')
    for value in resolver.values():
        path = Path(value['local_path'])
        require(path.resolve(strict=True).is_relative_to(WORKSPACE.parent.resolve()) and path.is_file(),
                'read_only_project_image_required')
        stat = path.stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ino) == (value['bytes'],value['mtime_ns'],value['inode']),
                'source_image_metadata_changed')
    pairs = [p for p in plan['records'] if p['source_id'] in available]
    require(len(pairs) == 132 and len({p['hypothesis_sha256'] for p in pairs}) == 130,
            'fixed_available_pair_and_text_inventory_required')
    return plan,resolver,pairs


def prepare(run_id):
    allocation_guard()
    require(re.fullmatch(r'biovil_cpu_12714150_[0-9]{3}',run_id), 'opaque_run_id_required')
    expert,resolver,pairs = preflight_inputs()
    runtime = metadata()
    paths = [Path(__file__).resolve(), Path(installed.__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radeval_image_benchmark.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/60_radeval_biovil_existing_cpu.sh',
        WORKSPACE / 'docs/radeval_image_benchmark_protocol.md',
        LINK / 'manifest.json', LINK / 'internal_resolver.json',
        EXPERT / 'manifest.json', EXPERT / 'frozen_plan.json', EXPERT / 'scores.json',
        SOURCE / 'manifest.json', SOURCE / 'reader_study_final_with_annotationsv3.csv',
        BASE / 'radgraph_bank_runs/native_pool960_12714150_001/candidate_score_table.csv',
        BASE / 'deliverables/first_version_12714150_001/candidate_index.csv']
    pins = [{'path': str(p.relative_to(WORKSPACE)), 'sha256': installed.sha256(p)} for p in paths]
    os.umask(0o007)
    private_dir(PLANS,True)
    root = PLANS / run_id
    private_dir(root)
    plan = {'schema_version': 'radeval-biovil-cpu-preparation-v1', 'runtime': runtime, 'pins': pins,
        'all_attempted_pairs': len(expert['records']), 'planned_image_forwards': len(resolver),
        'available_report_pairs': len(pairs), 'available_unique_candidate_texts': 130,
        'available_anchor_units': len({(p['source_id'],p['section_id'],p['reference_sha256']) for p in pairs}),
        'available_patient_groups': len({p['source_group_id'] for p in pairs}),
        'image_availability': [{'item_id': p['item_id'], 'source_id': p['source_id'],
            'image_available': p['source_id'] in resolver} for p in expert['records']],
        'primary_outcome': 'clinically_significant_total', 'bootstrap_resamples': 1000, 'seed': 0,
        'cohort_selected_using_scores_or_errors': False, 'reference_report_fed_to_biovil': False,
        'pixel_input_execution_approved_by_this_receipt': False, 'models_executed': 0,
        'image_bytes_read_or_decoded': False, 'device': 'cpu', 'thread_count': 2,
        'existing_allocation': {'job_id': 12714150, 'cpus': 4, 'memory_gib': 32, 'gpu_count': 0},
        'process_timeout_seconds': 600, 'selection_changed': False}
    installed.write_json(root / 'plan.json',plan)
    installed.write_json(root / 'manifest.json', {'schema_version': 'radeval-biovil-cpu-preparation-receipt-v1',
        'plan_sha256': installed.sha256(root / 'plan.json'), 'pins': pins,
        'source_input_execution_authorized': False, 'model_calls': 0})
    return {'status': 'metadata_plan_prepared_no_pixel_or_model_calls',
            'manifest_sha256': installed.sha256(root / 'manifest.json')}


def execute(run_id, plan_root, allow_mimic_images, public_stdout):
    allocation_guard()
    require(allow_mimic_images is True, 'separate_mimic_image_execution_approval_required')
    require(re.fullmatch(r'biovil_cpu_12714150_[0-9]{3}',run_id), 'opaque_run_id_required')
    require(plan_root.resolve().is_relative_to(PLANS), 'protected_prepared_plan_required')
    receipt = json.loads((plan_root / 'manifest.json').read_text())
    require(installed.sha256(plan_root / 'plan.json') == receipt['plan_sha256'], 'frozen_preparation_required')
    plan = json.loads((plan_root / 'plan.json').read_text())
    require(all(installed.sha256(WORKSPACE / p['path']) == p['sha256'] for p in plan['pins']), 'prepared_pins_changed')
    require(metadata() == plan['runtime'], 'prepared_runtime_changed')
    expert,resolver,pairs = preflight_inputs()
    os.umask(0o007)
    private_dir(RUNS,True)
    root = RUNS / run_id
    private_dir(root)
    started = time.monotonic()
    image_vectors,text_vectors,image_receipts,text_receipts = {},{},{},{}
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1',
        'TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'',
        'HF_HOME':str(WORKSPACE / '.cache/radeval_biovil_cpu/huggingface'),
        'XDG_CACHE_HOME':str(WORKSPACE / '.cache/radeval_biovil_cpu/xdg'),
        'TMPDIR':str(WORKSPACE / '.tmp/radeval_biovil_cpu'),
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    with (SOURCE / source_manifest['file']).open(newline='',encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    row_index = {c['item_id']: c['source_row_index'] for c in expert['annotation_cells']}
    texts = {}
    for pair in pairs:
        text = rows[row_index[pair['item_id']]][f'prediction{pair["candidate_slot"]}']
        digest = hashlib.sha256(text.encode()).hexdigest()
        require(digest == pair['hypothesis_sha256'], 'candidate_text_hash_mismatch')
        texts[digest] = text
    with (root / 'worker.log').open('x') as log:
        (root / 'worker.log').chmod(0o660)
        with (contextlib.redirect_stdout(log), contextlib.redirect_stderr(log),
              patch.object(socket.socket,'connect',side_effect=RuntimeError('network_disabled')),
              patch.object(socket.socket,'connect_ex',side_effect=RuntimeError('network_disabled'))):
            import torch
            from score_report_cxr_biovil import _load_runtime
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            require(not torch.cuda.is_available(), 'cpu_only_visibility_required')
            with torch.inference_mode():
                image_engine,text_engine = _load_runtime(MODEL,torch.device('cpu'))
                require(not any(p.requires_grad for engine in (image_engine,text_engine)
                                for p in engine.model.parameters()), 'frozen_parameters_required')
                for source_id,entry in resolver.items():
                    path = Path(entry['local_path'])
                    item = {'source_id':source_id,'status':'failed_unavailable','failure_type':None,'image_sha256':None}
                    try:
                        stat = path.stat()
                        require((stat.st_size,stat.st_mtime_ns,stat.st_ino) ==
                                (entry['bytes'],entry['mtime_ns'],entry['inode']), 'source_stat_changed')
                        digest = installed.sha256(path)
                        embedding = image_engine.get_projected_global_embedding(path)
                        require(tuple(embedding.shape) == (128,) and torch.isfinite(embedding).all().item()
                                and abs(embedding.norm().item()-1) < 1e-3, 'finite_normalized_image_embedding_required')
                        require(installed.sha256(path) == digest, 'image_changed_during_computation')
                        image_vectors[source_id] = embedding.detach().float().cpu().tolist()
                        item.update(status='complete',image_sha256=digest)
                    except Exception as error:
                        item['failure_type'] = type(error).__name__
                    image_receipts[source_id] = item
                for index,(digest,text) in enumerate(texts.items()):
                    item = {'hypothesis_sha256':digest,'status':'failed_unavailable','failure_type':None,
                            'token_count':None,'silent_truncation':False}
                    try:
                        tokenized = text_engine.tokenize_input_prompts([text],verbose=False)
                        item['token_count'] = int(tokenized.input_ids.shape[1])
                        embedding = text_engine.get_embeddings_from_prompt([text],normalize=True,verbose=False)[0]
                        require(tuple(embedding.shape) == (128,) and torch.isfinite(embedding).all().item()
                                and abs(embedding.norm().item()-1) < 1e-3, 'finite_normalized_text_embedding_required')
                        text_vectors[digest] = embedding.detach().float().cpu().tolist()
                        item['status'] = 'complete'
                    except Exception as error:
                        item['failure_type'] = type(error).__name__
                    text_receipts[digest] = item
                    if index % 50 == 0:
                        print(json.dumps({'status':'protected_image_metric_progress',
                            'runtime_seconds':round(time.monotonic()-started,3)}),file=public_stdout,flush=True)
    scores = []
    for pair in expert['records']:
        status,value = 'unavailable_source_image',None
        if pair['source_id'] in resolver:
            status = 'failed_image_or_text'
            image = image_vectors.get(pair['source_id'])
            text = text_vectors.get(pair['hypothesis_sha256'])
            if image is not None and text is not None:
                value = sum(a*b for a,b in zip(image,text))
                require(math.isfinite(value) and abs(value) <= 1.00001,'finite_cosine_required')
                status = 'complete'
        scores.append({'item_id':pair['item_id'],'status':status,'value':value})
    previous = json.loads((EXPERT / 'scores.json').read_text())
    records = join(expert,scores,previous)
    evaluation = evaluate(records,resamples=1000,seed=0)
    installed.write_json(root / 'image_scores.json',scores)
    installed.write_json(root / 'paired_score_table.json',records)
    installed.write_json(root / 'evaluation.json',evaluation)
    installed.write_json(root / 'embedding_receipts.json',{'images':image_receipts,'texts':text_receipts})
    installed.write_json(root / 'private_embeddings.json',{'images':image_vectors,'texts':text_vectors})
    summary = {'status':'complete','schema_version':'radeval-biovil-cpu-run-v1',
        'all_attempted_pairs':len(scores),'available_source_images':len(resolver),
        'image_status_counts':dict(Counter(r['status'] for r in image_receipts.values())),
        'text_status_counts':dict(Counter(r['status'] for r in text_receipts.values())),
        'pair_status_counts':dict(Counter(r['status'] for r in scores)),
        'runtime_seconds':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'device':'cpu','job_id':12714150,'new_slurm_submissions':0,
        'frozen_parameters':True,'network_disabled':True,'reference_report_fed_to_biovil':False,
        'clinical_qualified':False,'selection_changed':False,'real_images_rendered_or_copied':False,
        'pixel_computation_approved_by_explicit_flag':True,'image_encoder_forwards_attempted':len(resolver),
        'unique_candidate_text_forwards_attempted':len(texts)}
    installed.write_json(root / 'summary.json',summary)
    require(all(installed.sha256(WORKSPACE / p['path']) == p['sha256'] for p in plan['pins']), 'sealed_source_changed')
    installed.write_json(root / 'manifest.json',{'schema_version':'radeval-biovil-cpu-run-receipt-v1',
        'prepared_plan_sha256':installed.sha256(plan_root / 'plan.json'),'pins':plan['pins'],
        'artifacts':[{'path':p.name,'sha256':installed.sha256(p)} for p in sorted(root.glob('*.json'))],
        'clinical_qualified':False,'selection_changed':False})
    return {'status':'protected_image_benchmark_complete','runtime_seconds':round(summary['runtime_seconds'],3),
        'peak_rss_gib':round(summary['peak_rss_gib'],3),'manifest_sha256':installed.sha256(root / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--plan-root',type=Path)
    parser.add_argument('--allow-mimic-images',action='store_true')
    try:
        args = parser.parse_args()
        if args.mode == 'prepare':
            result = prepare(args.run_id)
        else:
            require(args.plan_root is not None,'sealed_plan_required')
            result = execute(args.run_id,args.plan_root,args.allow_mimic_images,sys.stdout)
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__}))
        raise SystemExit(1)
