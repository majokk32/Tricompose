"""Independent approved-CPU receipt/vector/stat replay; no image rendering.

Image bytes are read only for hashes inside the same approved allocation.
No new encoder calls, report inspection, fitting, selection or external API.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
PLAN = BASE / 'radeval_image_benchmark_plans/biovil_cpu_12714150_001'
RUN = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
LINK = BASE / 'radeval_image_linkage_runs/exact_index_12714150_001'
EXPERT = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
OUT = BASE / 'radeval_image_benchmark_audits/biovil_cpu_12714150_001'
METRICS = ('biovil_t_raw_cosine','radgraph_entity_f1',
           'radgraph_relation_presence_f1','radgraph_full_relation_f1')
TARGETS = ('clinically_significant_total','all_errors_total')


def require(condition):
    if not condition:
        raise ValueError('biovil_expert_benchmark_audit_failed')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def rank(values):
    frequencies = Counter(values)
    ranks, cumulative = {}, 0
    for value,count in sorted(frequencies.items()):
        ranks[value] = cumulative + (count+1)/2
        cumulative += count
    return [ranks[v] for v in values]


def spearman(left,right):
    if len(left) < 3:
        return None
    a,b = rank(left),rank(right)
    am,bm = sum(a)/len(a),sum(b)/len(b)
    a,b = [v-am for v in a],[v-bm for v in b]
    denominator = math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
    return sum(x*y for x,y in zip(a,b))/denominator if denominator else None


def tau(left,right):
    concordant = discordant = a_ties = b_ties = 0
    for i in range(len(left)):
        for j in range(i+1,len(left)):
            a,b = left[i]-left[j],right[i]-right[j]
            if a == b == 0:
                continue
            if a == 0:
                a_ties += 1
            elif b == 0:
                b_ties += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    signed = concordant+discordant
    d = math.sqrt((signed+a_ties)*(signed+b_ties))
    return (concordant-discordant)/d if d else None


def percentile(values,q):
    values = sorted(values)
    index = (len(values)-1)*q
    low,high = math.floor(index),math.ceil(index)
    return values[low]+(values[high]-values[low])*(index-low)


def close(left,right):
    return left is right if left is None or right is None else abs(left-right) < 1e-12


def bootstrap_correlation(values):
    groups = defaultdict(list)
    for group,a,b in values:
        groups[group].append((a,b))
    keys = sorted(groups)
    rng,boot = random.Random(0),[]
    for _ in range(1000):
        sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
        value = spearman([v[0] for v in sample],[v[1] for v in sample])
        if value is not None:
            boot.append(value)
    return len(boot),[percentile(boot,.025),percentile(boot,.975)]


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    require(sha256(PLAN / 'manifest.json') == '6f41133e3066724ecfdbdd8d0fb425c193559c82ea7355e5380b584b6549df96')
    prepared = json.loads((PLAN / 'plan.json').read_text())
    manifest = json.loads((RUN / 'manifest.json').read_text())
    hash_checks = 0
    require(manifest['prepared_plan_sha256'] == sha256(PLAN / 'plan.json'))
    for pin in manifest['pins']:
        require(sha256(ROOT / pin['path']) == pin['sha256'])
        hash_checks += 1
    for artifact in manifest['artifacts']:
        source = RUN / artifact['path']
        require(source.resolve().is_relative_to(RUN) and sha256(source) == artifact['sha256'])
        hash_checks += 1
    expert = json.loads((EXPERT / 'frozen_plan.json').read_text())['inventory']
    previous = json.loads((EXPERT / 'scores.json').read_text())
    resolver = json.loads((LINK / 'internal_resolver.json').read_text())
    scores = json.loads((RUN / 'image_scores.json').read_text())
    table = json.loads((RUN / 'paired_score_table.json').read_text())
    vectors = json.loads((RUN / 'private_embeddings.json').read_text())
    receipts = json.loads((RUN / 'embedding_receipts.json').read_text())
    result = json.loads((RUN / 'evaluation.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    require(len(scores) == len(table) == len(expert['records']) == 624)
    require(set(receipts['images']) == set(resolver) and len(resolver) == 43)
    image_hash_checks = vector_checks = 0
    for source_id,entry in resolver.items():
        receipt = receipts['images'][source_id]
        if receipt['status'] == 'complete':
            path = Path(entry['local_path'])
            require(path.resolve(strict=True).is_relative_to(ROOT.parent.resolve()))
            require(sha256(path) == receipt['image_sha256'])
            image_hash_checks += 1
        else:
            require(source_id not in vectors['images'])
    for kind in ('images','texts'):
        require(set(vectors[kind]) == {key for key,receipt in receipts[kind].items()
                                    if receipt['status'] == 'complete'})
        for vector in vectors[kind].values():
            require(isinstance(vector,list) and len(vector) == 128 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in vector))
            require(abs(math.sqrt(sum(v*v for v in vector))-1) < 1e-3)
            vector_checks += 1
    require(all(r['silent_truncation'] is False for r in receipts['texts'].values()))
    pair_checks = 0
    anchors = {}
    for source_pair,score,row,old in zip(expert['records'],scores,table,previous):
        require(source_pair['item_id'] == score['item_id'] == row['item_id'] == old['item_id'])
        for name in ('source_id','source_group_id','section_id','section_name','candidate_slot',
                     'reference_sha256','hypothesis_sha256'):
            require(row[name] == source_pair[name])
        image = vectors['images'].get(source_pair['source_id'])
        text = vectors['texts'].get(source_pair['hypothesis_sha256'])
        expected_status = ('unavailable_source_image' if source_pair['source_id'] not in resolver else
                           'complete' if image is not None and text is not None else 'failed_image_or_text')
        require(score['status'] == row['image_score_status'] == expected_status)
        available = expected_status == 'complete' and old['status'] == 'complete'
        require(row['paired_cohort_available'] == available)
        if available:
            cosine = sum(a*b for a,b in zip(image,text))
            require(close(score['value'],cosine) and close(row['scores'][METRICS[0]],cosine))
            for name in METRICS[1:]:
                require(row['scores'][name] == old['scores'][name])
        else:
            require(score['value'] is None and all(v is None for v in row['scores'].values()))
        errors = source_pair['errors']
        totals = {name:sum(values) if None not in values else None for name,values in errors.items()}
        require(row['expert_outcomes'][TARGETS[0]] == totals['clinically_significant'])
        all_errors = sum(totals.values()) if None not in totals.values() else None
        require(row['expert_outcomes'][TARGETS[1]] == all_errors)
        key = (row['source_id'],row['section_id'],row['reference_sha256'])
        anchors.setdefault(key,[]).append(row)
        pair_checks += 1
    statistical_checks = 0
    for metric in METRICS:
        for target in TARGETS:
            values = [(r['source_group_id'],r['scores'][metric],-r['expert_outcomes'][target]) for r in table
                      if r['scores'][metric] is not None and r['expert_outcomes'][target] is not None]
            stored = result['correlations'][metric][target]
            a,b = [v[1] for v in values],[v[2] for v in values]
            require(stored['paired_rows'] == len(values))
            require(close(stored['spearman'],spearman(a,b)) and close(stored['kendall_tau_b'],tau(a,b)))
            usable,interval = bootstrap_correlation(values)
            require(stored['cluster_bootstrap']['usable_resamples'] == usable)
            require(all(close(a,b) for a,b in zip(interval,stored['cluster_bootstrap']['spearman_interval_95'])))
            statistical_checks += 1
    choice_lookup = {f'anchor_{index:04d}':sorted(rows,key=lambda r:r['candidate_slot'])
                     for index,rows in enumerate(anchors.values())}
    choice_checks = 0
    for choice in result['analytical_choices']:
        rows = choice_lookup[choice['anchor_id']]
        errors = [r['expert_outcomes'][choice['target']] for r in rows]
        values = [r['scores'][choice['metric']] for r in rows]
        require(None not in errors and None not in values)
        winners = [i for i,v in enumerate(values) if v == max(values)]
        selected = sum(errors[i] for i in winners)/len(winners)
        require(choice['selected_expected_errors'] == selected and choice['random_expected_errors'] == sum(errors)/3
                and choice['oracle_errors'] == min(errors) and choice['metric_minus_random_errors'] == selected-sum(errors)/3)
        strict = correct = ties = 0
        for i,j in ((0,1),(0,2),(1,2)):
            if errors[i] == errors[j]:
                continue
            strict += 1
            if values[i] == values[j]:
                ties += 1
                correct += .5
            else:
                correct += int((values[i] > values[j]) == (errors[i] < errors[j]))
        require(choice['strict_pairs'] == strict and choice['expected_correct_pairs'] == correct
                and choice['tied_score_pairs'] == ties and choice['top_tie_size'] == len(winners))
        choice_checks += 1
    for stored in result['selection_diagnostic']:
        choices = [c for c in result['analytical_choices'] if c['metric'] == stored['metric'] and c['target'] == stored['target']]
        expected_complete = sum(all(r['scores'][stored['metric']] is not None and
                                    r['expert_outcomes'][stored['target']] is not None for r in rows)
                                for rows in anchors.values())
        require(len(choices) == stored['complete_anchors'] == expected_complete)
        for key,value in stored['means'].items():
            require(value == sum(c[key] for c in choices)/len(choices))
        groups = defaultdict(list)
        for choice in choices:
            groups[choice['source_group_id']].append(choice['metric_minus_random_errors'])
        keys = sorted(groups)
        rng,boot = random.Random(0),[]
        for _ in range(1000):
            sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
            boot.append(sum(sample)/len(sample))
        require(stored['error_delta_cluster_ci']['interval95'] == [percentile(boot,.025),percentile(boot,.975)])
        strict = sum(c['strict_pairs'] for c in choices)
        require(close(stored['pairwise_accuracy'],sum(c['expected_correct_pairs'] for c in choices)/strict))
        statistical_checks += 1
    permissions = 0
    for path in (RUN,*RUN.iterdir()):
        require(path.stat().st_gid in (96293,65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660))
        permissions += 1
    require(summary['frozen_parameters'] is True and summary['network_disabled'] is True and
            summary['device'] == 'cpu' and summary['reference_report_fed_to_biovil'] is False)
    require(summary['selection_changed'] is False and summary['clinical_qualified'] is False)
    require(result['common_cohort_for_all_four_metrics'] is True and
            result['independent_image_radiologist_adjudication'] is False)
    require(summary['pair_status_counts'] == dict(Counter(s['status'] for s in scores)))
    require(sum(r['paired_cohort_available'] for r in table) == result['common_score_available_pairs'])
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770,exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status':'passed','run_manifest_sha256':sha256(RUN / 'manifest.json'),
        'source_code_output_hash_checks':hash_checks,'approved_source_image_hashes_verified':image_hash_checks,
        'finite_normalized_embedding_vectors_verified':vector_checks,'all_attempted_pairs_replayed':pair_checks,
        'analytical_choice_records_replayed':choice_checks,'correlation_selection_bootstrap_replays':statistical_checks,
        'protected_permissions_checked':permissions,'new_model_calls':0,'real_images_rendered':False,
        'reference_report_fed_to_biovil':False,'clinical_qualified':False,'selection_changed':False}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(payload,stream,sort_keys=True,indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status':'image_expert_benchmark_audit_passed','audit_sha256':sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__}))
        raise SystemExit(1)
