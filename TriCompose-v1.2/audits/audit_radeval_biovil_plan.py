"""Independent pre-run metadata audit. Never hashes/opens image bytes."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
PLAN = BASE / 'radeval_image_benchmark_plans/biovil_cpu_12714150_001'
LINK = BASE / 'radeval_image_linkage_runs/exact_index_12714150_001'
EXPERT = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
OUT = BASE / 'radeval_image_benchmark_plan_audits/biovil_cpu_12714150_001'


def require(condition):
    if not condition:
        raise ValueError('biovil_metadata_preparation_audit_failed')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    receipt = json.loads((PLAN / 'manifest.json').read_text())
    require(sha256(PLAN / 'plan.json') == receipt['plan_sha256'])
    plan = json.loads((PLAN / 'plan.json').read_text())
    hashes = 1
    for pin in plan['pins']:
        require(sha256(ROOT / pin['path']) == pin['sha256'])
        hashes += 1
    pairs = json.loads((EXPERT / 'frozen_plan.json').read_text())['inventory']['records']
    resolver = json.loads((LINK / 'internal_resolver.json').read_text())
    require(len(resolver) == 43 and len(pairs) == 624)
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    with (SOURCE / source_manifest['file']).open(newline='',encoding='utf-8-sig') as stream:
        keys = list(dict.fromkeys(row['images_path'].strip() for row in csv.DictReader(stream)))
    metadata_checks = 0
    for source_id,entry in resolver.items():
        source_key = keys[int(source_id.split('_')[1])]
        local = Path(entry['local_path'])
        require(local.resolve(strict=True).is_relative_to(ROOT.parent.resolve()) and local.is_file())
        # Reconstruct exact complete pXX/p########/s########/DICOM.jpg suffix,
        # independently of the producer's image-path parser.
        parts = Path(source_key).parts
        suffix = '/'.join(parts[-4:])
        require(str(local).endswith('/'+suffix))
        stat = local.stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ino) ==
                (entry['bytes'],entry['mtime_ns'],entry['inode']))
        metadata_checks += 3
    expected = [{'item_id':p['item_id'],'source_id':p['source_id'],
        'image_available':p['source_id'] in resolver} for p in pairs]
    require(plan['image_availability'] == expected)
    available = [p for p in pairs if p['source_id'] in resolver]
    require(len(available) == plan['available_report_pairs'] == 132)
    require(len({p['hypothesis_sha256'] for p in available}) == plan['available_unique_candidate_texts'] == 130)
    require(len({p['source_group_id'] for p in available}) == plan['available_patient_groups'] == 34)
    require(len({(p['source_id'],p['section_id'],p['reference_sha256']) for p in available}) ==
            plan['available_anchor_units'] == 44)
    require(plan['all_attempted_pairs'] == 624 and plan['planned_image_forwards'] == 43)
    require(plan['cohort_selected_using_scores_or_errors'] is False and
            plan['reference_report_fed_to_biovil'] is False and
            plan['image_bytes_read_or_decoded'] is False and plan['models_executed'] == 0 and
            plan['pixel_input_execution_approved_by_this_receipt'] is False)
    require(plan['existing_allocation'] == {'job_id':12714150,'cpus':4,'memory_gib':32,'gpu_count':0})
    exported = (PLAN / 'plan.json').read_text() + (PLAN / 'manifest.json').read_text()
    require(not any(key in exported for key in keys))
    permissions = 0
    for path in (PLAN,*PLAN.iterdir()):
        require(path.stat().st_gid in (96293,65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660))
        permissions += 1
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770,exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status':'passed','plan_manifest_sha256':sha256(PLAN / 'manifest.json'),
        'source_hash_checks':hashes,'independent_source_stat_identity_checks':metadata_checks,
        'all_attempted_pair_availability_rows_verified':624,'linked_source_images':43,
        'linked_candidate_reports':132,'linked_distinct_texts':130,'linked_anchor_units':44,
        'linked_patient_groups':34,'protected_permissions_checked':permissions,
        'image_bytes_read':False,'model_calls':0,'new_job_submissions':0,
        'source_input_execution_authorized':False,'clinical_qualified':False,'selection_changed':False}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(payload,stream,sort_keys=True,indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status':'biovil_metadata_preparation_audit_passed',
                      'audit_sha256':sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__}))
        raise SystemExit(1)
