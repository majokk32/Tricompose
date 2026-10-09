#!/usr/bin/env python3
"""Build a protected, metadata-only research-demo handoff from frozen caches.

Never load EHR/report contents, image pixels, model weights or credentials.
Synthetic artifacts are authenticated as bytes only; no generation or scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
import score_free_random_control as cached
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

SCHEMA = 'tricompose-first-version-protected-delivery-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001'
CONTROL = BASE / 'automatic_replays/score_free_random_12654973_001'
BANK_PIN = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
CONTROL_PIN = '771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124'
SELECTION_PIN = 'bd9e4a6f6a261f06305800289c3bd5f81c85f9a48fa9ae7e8c23b4c723b68bbe'
CXR_MODELS = {'roentgen_v2', 'chexgenbench_sana', 'chexgenbench_pixart'}
REPORT_MODELS = {'maira2', 'cxrmate_single', 'llavarad', 'chexagent2'}
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,255}\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, code):
    if not condition:
        raise ValueError(code)


def bounded(path, digest, sources, label, limit=32 * 1024 ** 2):
    original = Path(path)
    require(not original.is_symlink(), 'symlink_source_refused')
    p = require_inside(original, PROTECTED_ROOT, must_exist=True)
    require(p.is_file() and p.stat().st_size <= limit
        and isinstance(digest, str) and HASH.fullmatch(digest)
        and sha256_file(p) == digest, 'bounded_source_hash_required')
    sources[label] = (p, digest)
    return p


def manifest(root, digest, sources, label):
    p = bounded(Path(root) / 'manifest.json', digest, sources, label, 1024 ** 2)
    return json.loads(p.read_text())


def artifact(root, m, name, sources, label):
    pin = m['artifacts'][name]
    digest = pin['sha256'] if isinstance(pin, dict) else pin
    return bounded(Path(root) / name, digest, sources, label)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_records_equal(actual, expected, code):
    require(actual == list(csv.DictReader(io.StringIO(cached.csv_text(expected)))), code)


def project_index(scores, candidates, images, reports, inputs, expected_cases=80):
    """Allowlisted metadata projection; never copy source records wholesale."""
    require(len(scores) == len(candidates) == expected_cases * 12, 'complete_score_grid_required')
    result, seen = [], set()
    anchors, image_identity, report_identity = {}, {}, {}
    grid = defaultdict(set)
    for score in sorted(scores, key=lambda r: r['triple_candidate_id']):
        case, cid, lineage = score['case_id'], score['triple_candidate_id'], score['lineage']
        require(ID.fullmatch(case) and ID.fullmatch(cid) and cid not in seen,
            'opaque_unique_case_candidate_ids_required')
        seen.add(cid)
        row = candidates[cid]
        require(row['case_id'] == case and all(row[k] == lineage[k] for k in
            ('cxr_candidate_id', 'report_candidate_id', 'ehr_sha256', 'ehr_facts_sha256',
             'cxr_sha256', 'report_sha256', 'report_model_id')), 'score_endpoint_lineage_differs')
        model, seed, expert = lineage['cxr_model_id'], lineage['cxr_seed'], lineage['report_model_id']
        require(model in CXR_MODELS and expert in REPORT_MODELS and type(seed) is int and seed == 0,
            'historical_three_by_four_single_seed_grid_required')
        slot = (model, seed, expert)
        require(slot not in grid[case], 'duplicate_model_slot_refused')
        grid[case].add(slot)
        hashes = tuple(lineage[k] for k in ('ehr_sha256', 'ehr_facts_sha256'))
        require(all(isinstance(h, str) and HASH.fullmatch(h) for h in hashes)
            and anchors.setdefault(case, hashes) == hashes, 'fixed_ehr_hash_required')
        image = images[lineage['cxr_candidate_id']]
        report = reports[lineage['report_candidate_id']]
        inp = inputs[case, model, seed]
        require(image['case_id'] == report['case_id'] == case
            and image['model_id'] == model and image['seed'] == seed
            and report['model_id'] == expert
            and report['parent_cxr_candidate_id'] == image['candidate_id']
            and image['artifact']['sha256'] == lineage['cxr_sha256']
            and report['artifact']['sha256'] == lineage['report_sha256']
            and inp['synthetic_ehr']['sha256'] == hashes[0]
            and inp['ehr_facts']['sha256'] == hashes[1]
            and image['prompt_sha256'] == inp['final_prompt']['sha256'],
            'artifact_request_parent_lineage_differs')
        require(image['frozen_model'] is report['frozen_model'] is True
            and report['structured_ehr_content_supplied_to_model'] is False
            and report['source_report_or_real_target_supplied'] is False,
            'frozen_synthetic_cxr_only_report_scope_required')
        identity = (case, model, seed, lineage['cxr_sha256'])
        require(image_identity.setdefault(image['candidate_id'], identity) == identity,
            'shared_image_identity_changed')
        require(report['candidate_id'] not in report_identity, 'report_slot_reused')
        report_identity[report['candidate_id']] = (case, expert, lineage['report_sha256'])
        selected = score['scoring']['selection']['selected']
        require(type(selected) is bool, 'original_selection_boolean_required')
        out = {'case_id': case, 'triple_candidate_id': cid,
            'cxr_candidate_id': image['candidate_id'], 'cxr_model_id': model, 'cxr_seed': seed,
            'report_candidate_id': report['candidate_id'], 'report_model_id': expert,
            'ehr_path': inp['synthetic_ehr']['path'], 'ehr_sha256': hashes[0],
            'ehr_facts_path': inp['ehr_facts']['path'], 'ehr_facts_sha256': hashes[1],
            'prompt_path': inp['final_prompt']['path'], 'prompt_sha256': image['prompt_sha256'],
            'cxr_path': image['artifact']['path'], 'cxr_sha256': lineage['cxr_sha256'],
            'report_path': report['artifact']['path'], 'report_sha256': lineage['report_sha256'],
            'historical_static_selected': selected,
            'report_structure_quality_proxy': score['scoring']['modality_quality']['report_structure_quality_score_0_1'],
            'historical_selected_path_runtime_seconds': score['scoring']['cost']['known_runtime_seconds']}
        out.update(cached.candidate_readout(row))
        result.append(out)
    expected = {(m, 0, r) for m in CXR_MODELS for r in REPORT_MODELS}
    require(len(grid) == expected_cases and set(seen) == set(candidates)
        and all(v == expected for v in grid.values())
        and len(image_identity) == expected_cases * 3
        and len(report_identity) == expected_cases * 12
        and set(images) == set(image_identity) and set(reports) == set(report_identity),
        'complete_fixed_cohort_inventory_required')
    require(all(sum(r['historical_static_selected'] for r in result if r['case_id'] == case) == 1
        for case in grid), 'one_historical_static_choice_per_case_required')
    return result


def inventory(index):
    by_case = defaultdict(list)
    for row in index:
        by_case[row['case_id']].append(row)
    return {'fixed_ehr_cases': len(by_case), 'cxr_candidate_slots': len({r['cxr_candidate_id'] for r in index}),
        'report_candidate_slots': len(index),
        'distinct_ehr_hashes': len({r['ehr_sha256'] for r in index}),
        'distinct_image_hashes': len({r['cxr_sha256'] for r in index}),
        'distinct_report_hashes': len({r['report_sha256'] for r in index}),
        'historical_selected_triples': sum(r['historical_static_selected'] for r in index),
        'cases_with_direct_cached_ehr_facts': sum(any(r['ehr_cxr_known_reference_facts'] > 0 for r in rows) for rows in by_case.values()),
        'cases_without_direct_cached_ehr_facts': sum(all(r['ehr_cxr_known_reference_facts'] == 0 for r in rows) for rows in by_case.values()),
        'cxr_model_slots': dict(sorted(Counter(r['cxr_model_id'] for r in
            {r['cxr_candidate_id']: r for r in index}.values()).items())),
        'report_model_slots': dict(sorted(Counter(r['report_model_id'] for r in index).items())),
        'cohort_role': 'already_inspected_development_bank',
        'source_conditioning': 'historical_august_bridge_not_september_repair',
        'native_structured_ehr_to_cxr': False, 'genuine_ehr_plus_cxr_report_path': False,
        'clinical_best_triple_established': False}


def case_index(index, old):
    results = []
    source = {(r['case_id'], r['method']): r for r in old
        if r['model_call_budget'] == 30 and r['method'] in ('fixed', 'static_rerank')}
    by_case = defaultdict(list)
    for row in index:
        by_case[row['case_id']].append(row)
    for case, rows in sorted(by_case.items()):
        selected = next(r for r in rows if r['historical_static_selected'])
        identities = {r['triple_candidate_id']: r for r in rows}
        for method in ('fixed', 'static_rerank'):
            chosen = source[case, method]['selected_candidate_id']
            require(chosen is None or chosen in identities,
                'same_case_observed_baseline_choice_required')
        out = {'case_id': case, 'ehr_path': selected['ehr_path'], 'ehr_sha256': selected['ehr_sha256'],
            'candidate_slots': len(rows), 'historical_selected_candidate_id': selected['triple_candidate_id'],
            'historical_selected_cxr_path': selected['cxr_path'],
            'historical_selected_report_path': selected['report_path'],
            'fixed_candidate_id': source[case, 'fixed']['selected_candidate_id'],
            'cached_static_candidate_id': source[case, 'static_rerank']['selected_candidate_id'],
            'direct_ehr_edge_available': any(r['ehr_cxr_known_reference_facts'] > 0 for r in rows)}
        for method in ('fixed', 'static_rerank'):
            chosen = identities.get(source[case, method]['selected_candidate_id'])
            for role in ('cxr', 'report'):
                out[method + '_' + role + '_path'] = chosen[role + '_path'] if chosen else None
        results.append(out)
    return results


def edge_table(comparisons):
    fields = (*cached.COUNT_FIELDS, 'coverage_over_known', 'support_over_known', 'opposition_over_known')
    rows = []
    for comparison in comparisons:
        for edge in cached.EDGES:
            row = {k: comparison[k] for k in ('ehr_scope', 'method', 'model_call_budget', 'fixed_ehr_cases')}
            row['edge'] = edge
            for field in fields:
                for suffix in ('mean', 'available_ehr_cases'):
                    row[field + '_' + suffix] = comparison[edge + '_' + field + '_' + suffix]
            rows.append(row)
    return rows


def availability(index):
    rows = []
    for edge in cached.EDGES:
        rows.append({'edge': edge, 'fixed_ehr_cases': len({r['case_id'] for r in index}),
            'candidate_slots': len(index),
            'cases_with_known_reference_facts': len({r['case_id'] for r in index if r[edge + '_known_reference_facts'] > 0}),
            'candidate_slots_with_known_reference_facts': sum(r[edge + '_known_reference_facts'] > 0 for r in index),
            'candidate_slots_with_any_comparison': sum(r[edge + '_comparable_facts'] > 0 for r in index),
            'reference_observations_repeated_across_candidates': sum(r[edge + '_known_reference_facts'] for r in index),
            'comparable_observations_repeated_across_candidates': sum(r[edge + '_comparable_facts'] for r in index),
            'clinical_qualified': False})
    return rows


def show(value):
    return 'NA' if value is None else f'{value:.4f}'


def render(info, comparisons):
    lines = ['# TriCompose first-version demo / 首版汇报入口', '',
        'This is a frozen-model engineering baseline and exploratory DEVELOPMENT comparison, not validated clinical self-correction.',
        '这是可以复现的工程首版；已有生成与筛选结果，不宣称找到临床最优或证明自动纠错有效。', '',
        '## 1. Pipeline / 流水线', '', '```text',
        '80 fixed synthetic EHRs → grounded radiology text',
        '  → RoentGen-v2 / Sana / PixArt (3 CXR slots per EHR)',
        '  → MAIRA-2 / CXRMate-single / LLaVA-Rad / CheXagent-2 (4 reports per CXR)',
        '  → cached edge evidence + fixed / score-free random / static comparison', '```', '',
        'These CXR models accept text/report-style conditioning, not native structured EHR.',
        'CXRMate-single is CXR-only, not CXRMate-ED. All models stay frozen; no training or API patient-data upload.', '',
        '## 2. What exists / 实际输出', '',
        f"- Fixed EHR cases: {info['fixed_ehr_cases']}.",
        f"- CXR slots / distinct image hashes: {info['cxr_candidate_slots']} / {info['distinct_image_hashes']}.",
        f"- Report slots / distinct report hashes: {info['report_candidate_slots']} / {info['distinct_report_hashes']}.",
        f"- Historical selected triples: {info['historical_selected_triples']} (proxy-selected, not clinically best).",
        f"- Direct comparable cached EHR constraints: {info['cases_with_direct_cached_ehr_facts']}/{info['fixed_ehr_cases']}; the other {info['cases_without_direct_cached_ehr_facts']} remain in every comparison.",
        '- This 80-case bank belongs to the historical August bridge, not the newer September repair or a new October generation run.', '',
        '## 3. Where to find outputs / 去哪里看', '',
        '- `case_index.csv`: one row per EHR; paths to historical selected synthetic EHR, image and report; fixed/cached-static IDs.',
        '- `candidate_index.csv`: all 960 exact candidate paths, hashes and unchanged raw edge readouts. No report/EHR body is embedded.',
        '- `baseline_comparison.csv`: all 75 method/budget/subgroup rows, copied only after exact cache replay.',
        '- `baseline_by_edge.csv`: the same results in a readable 225-row long table, including per-edge positive/negative support and available-EHR denominators.',
        '- `evidence_availability.csv`: comparable-evidence availability, separating cases from repeated candidate slots.',
        '- `baseline_case_means.csv`: 2,000 EHR-level means; random replicates averaged within EHR, not new patients.',
        '- `baseline_paired_comparison.csv`: all 60 paired contrasts, not significance tests.',
        '- `metric_dictionary.md`: definitions, missingness and cost conventions.',
        '- `manifest.json`: source pins and artifact hashes for teammate verification.', '',
        'The indexed source artifacts stay on CARC under artifacts/protected. This download bundle contains only metadata and aggregates, NOT image/report/EHR payloads, real targets, checkpoints or credentials.', '',
        '## 4. Full-budget comparison / 满预算对比', '',
        'Cap 30 is shown for orientation, not chosen as the best budget. All five caps are shown below and retained in CSV.',
        'Budget includes simulated CXR, XRV, report and CheXbert calls; EHR is shared sunk cost. Mean expenditure differs despite the same cap.', '',
        '| Method | Mean simulated calls | BioViL-T mean | CXR–Report support / known | Opposition / known | Coverage / known |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in comparisons:
        if row['ehr_scope'] == 'all' and row['model_call_budget'] == 30:
            lines.append(f"| {row['method']} | {show(row['mean_simulated_calls_mean'])} | {show(row['biovil_raw_cosine_mean'])} | {show(row['cxr_report_support_over_known_mean'])} | {show(row['cxr_report_opposition_over_known_mean'])} | {show(row['cxr_report_coverage_over_known_mean'])} |")
    lines += ['', '## 5. All budgets / 全部预算', '',
        '| Method | Cap | Mean simulated calls | BioViL-T | EHR denominator | CXR–Report support / known | Opposition / known | Coverage / known |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in comparisons:
        if row['ehr_scope'] == 'all':
            lines.append(f"| {row['method']} | {row['model_call_budget']} | {show(row['mean_simulated_calls_mean'])} | {show(row['biovil_raw_cosine_mean'])} | {row['biovil_raw_cosine_available_ehr_cases']}/{row['fixed_ehr_cases']} | {show(row['cxr_report_support_over_known_mean'])} | {show(row['cxr_report_opposition_over_known_mean'])} | {show(row['cxr_report_coverage_over_known_mean'])} |")
    lines += ['', '## 6. Interpretation / 老师汇报时怎么讲', '',
        'The bank and baseline comparison run end to end. Static selection can outperform score-free random choice on some proxies, but this is not independent clinical validation.',
        '现有证据支持“多候选筛选值得研究”，还不支持“错误定位与定向修复已经成功”。',
        'Fixed Sana + MAIRA-2 is competitive on the secondary global embedding readout. Lower budgets do not uniformly favor scored selection; do not hide them.',
        'Zero proxy opposition with limited coverage is NOT 100% consistency. EHR edges without direct comparable facts stay NA, not negative or perfect.',
        'Historical `random` means random acquisition followed by scored final selection; `random_acquisition_score_free_final` is the genuine score-free final-choice control.',
        'All four reports depend on the same image: their agreement is correlated evidence, not independent proof of a wrong image.',
        'The two-case actual regeneration diagnostic accepted 0/2 replacements. Published RadEvalX metric alignment remains unavailable until blank-count semantics are verified.',
        'Tests and hashes validate software/lineage, not clinical accuracy. XRV and CheXbert readouts are unqualified clinical proxies; BioViL is secondary, not a calibrated probability.', '',
        '## 7. Handoff / 复现与下一步', '',
        'See `REPRODUCE.txt` for metadata rebuild/audit commands. The builder verifies cached selections, all seed settings, numeric comparisons and byte hashes without inspecting synthetic payload contents.',
        'Do not move these private files to Git or an external clinical API. Download/view only within the authorized project collaboration.',
        'Next research gate: qualify independent finding/scope evidence, then test equal-budget targeted repair with fixed EHRs. New GPU jobs require complete-script review and explicit approval.', '']
    return '\n'.join(lines)


DICTIONARY = '''# Metric and cost dictionary / 指标与成本口径

- Known reference facts: cached explicit positive/negative states, not independent clinical truth.
- Comparable facts: both modalities have explicit states. Unknown/uncertain do not become negative.
- Support: comparable states agree. Positive and negative support remain separate.
- Proxy opposition: comparable states differ. This is not an adjudicated factual error.
- Coverage = comparable / known; support recall = support / known; opposition = opposition / known.
- If known = 0, every ratio is NA, not zero or perfect agreement.
- Candidate index contains raw explicit-state edges. Historical EHR-report global No-Finding adjustments are not reapplied to these raw columns; historical selections remain unchanged.
- BioViL-T: frozen global image-text cosine, secondary retrieval evidence, not probability or finding-level factuality.
- Report structure: cached formatting/repetition proxy, not correctness.
- Simulated calls: acquisition trace accounting, not freshly executed models; not measured GPU time or savings.
- Selected-path runtime: historical path metadata, not exhaustive bank cost. Missing costs stay NA.
- Three images + twelve reports per EHR cost fifteen generator calls before evaluation. The full cached accounting is thirty including XRV and CheXbert. EHR generation, shared initialization, secondary evaluation and failures are not estimated as free.
- Random final choice retains old scorer charges as a selector-only ablation. It is not a cost-optimal random pipeline.
- FID/KID and EHR distribution fidelity are cohort metrics, not candidate-ranking columns.
- No real reference report exists for this fully synthetic cohort: BLEU/ROUGE/METEOR/BERTScore/RadGraph-reference F1 cannot be claimed as its report quality scores.
- Source bank has already been inspected: DEVELOPMENT only; no held-out clinical significance claim or synthetic-patient count inflation from seed replicates.
'''


def execute(output_root, run_id, archive=False):
    cached.cpu_guard()
    started = time.monotonic()
    inputs_cache, cache_sources = cached.load()
    old, candidates = cached.prepare(inputs_cache)
    sources = {label: (path, sha256_file(path)) for label, path in cache_sources.items()}
    bm = manifest(BANK, BANK_PIN, sources, 'bank_manifest')
    cm = manifest(CONTROL, CONTROL_PIN, sources, 'random_control_manifest')
    controls = read_jsonl(artifact(CONTROL, cm, 'control_outcomes.jsonl', sources, 'score_free_controls'))
    require(controls == cached.make_controls(old, candidates), 'all_score_free_choices_exact_replay_required')
    means = cached.case_means([*old, *controls], candidates)
    comparisons, paired = cached.comparisons(means)
    for name, expected in (('case_means.csv', means), ('method_comparison.csv', comparisons),
                           ('paired_case_comparison.csv', paired)):
        check_records_equal(read_csv(artifact(CONTROL, cm, name, sources, 'control_' + name)),
            expected, 'cached_comparison_exact_replay_required')
    selection_root = Path(bm['source_paths']['historical_selection_manifest']).parent
    sm = manifest(selection_root, SELECTION_PIN, sources, 'selection_manifest')
    scores = read_jsonl(artifact(selection_root, sm, 'candidate_score_table.jsonl', sources, 'historical_scores'))
    require(sm['counts'] == {'candidate_rows': 960, 'cases': 80, 'cxr_candidates': 240,
        'fixed_paths': 12, 'report_candidates': 960, 'selected_triples': 80}, 'historical_inventory_required')
    images, reports, requests = {}, {}, {}
    request_roots = {}
    for kind, target, schema in (('cxr', images, 'tricompose-cxr-candidate-run-v1.1'),
                                ('report', reports, 'tricompose-report-candidate-run-v1.1')):
        labels = sorted(k for k in bm['source_paths'] if k.startswith(kind + '_manifest_'))
        require(len(labels) == (3 if kind == 'cxr' else 8), 'all_generation_manifests_required')
        for label in labels:
            p = Path(bm['source_paths'][label])
            m = manifest(p.parent, bm['source_sha256'][label], sources, label)
            require(m['schema_version'] == schema and m['frozen_model'] is True
                and m['candidate_count'] == len(m['candidates']), 'frozen_candidate_manifest_required')
            for entry in m['candidates']:
                q = bounded(p.parent / entry['path'], entry['sha256'], sources,
                    kind + '_metadata_' + entry['candidate_id'], 64 * 1024)
                value = json.loads(q.read_text())
                require(value['candidate_id'] == entry['candidate_id']
                    and value['case_id'] == entry['case_id'] and value['model_id'] == m['model_id']
                    and value['candidate_id'] not in target, 'unique_manifest_candidate_binding_required')
                target[value['candidate_id']] = value
            if kind == 'cxr':
                rp = Path(m['source_request_run'])
                request_roots[str(rp)] = m['source_request_run_manifest_sha256']
    for n, (rp, pin) in enumerate(sorted(request_roots.items())):
        rm = manifest(Path(rp), pin, sources, 'request_manifest_' + str(n))
        require(rm['schema_version'] == 'tricompose-cxr-request-run-v1.1', 'cxr_request_schema_required')
        for entry in rm['requests']:
            p = bounded(Path(rp) / entry['path'], entry['sha256'], sources,
                'request_' + entry['request_id'], 64 * 1024)
            value = json.loads(p.read_text())
            key = (value['case_id'], value['model_id'], value['seed'])
            require(key not in requests and value['frozen_model_required'] is True,
                'unique_frozen_request_required')
            requests[key] = value['inputs']
    index = project_index(scores, candidates, images, reports, requests)
    # Authenticate artifact bytes, but never parse EHR/facts/prompts/reports or images.
    payloads = {}
    for row in index:
        for role in ('ehr', 'ehr_facts', 'prompt', 'cxr', 'report'):
            path, digest = row[role + '_path'], row[role + '_sha256']
            require(payloads.setdefault(path, digest) == digest, 'artifact_path_hash_conflict')
    for n, (path, digest) in enumerate(sorted(payloads.items())):
        bounded(path, digest, sources, 'synthetic_artifact_bytes_' + str(n), 16 * 1024 ** 2)
    info = inventory(index)
    cases = case_index(index, old)
    source_code = (Path(__file__), ROOT / 'tests/test_first_version_delivery.py',
        ROOT.parent / 'docs/first_version_delivery_protocol.md')
    for p in source_code:
        sources['delivery_code_' + p.name] = (p, sha256_file(p))
    output_root = require_inside(output_root, PROTECTED_ROOT, must_exist=False)
    archive_path = output_root / (run_id + '.tar.gz')
    require(not archive or not archive_path.exists(), 'archive_overwrite_refused')
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        write_private_text(temporary / 'START_HERE_CN_EN.md', render(info, comparisons))
        write_private_text(temporary / 'candidate_index.csv', cached.csv_text(index))
        write_private_text(temporary / 'case_index.csv', cached.csv_text(cases))
        write_private_text(temporary / 'baseline_comparison.csv', cached.csv_text(comparisons))
        write_private_text(temporary / 'baseline_by_edge.csv', cached.csv_text(edge_table(comparisons)))
        write_private_text(temporary / 'evidence_availability.csv', cached.csv_text(availability(index)))
        write_private_text(temporary / 'baseline_case_means.csv', cached.csv_text(means))
        write_private_text(temporary / 'baseline_paired_comparison.csv', cached.csv_text(paired))
        write_private_json(temporary / 'generation_inventory.json', info)
        write_private_text(temporary / 'metric_dictionary.md', DICTIONARY)
        write_private_text(temporary / 'REPRODUCE.txt',
            'Metadata-only rebuild; run inside an existing actual CPU Slurm allocation.\n'
            'A new allocation/submission still requires full resource/script review and approval.\n\n'
            f'cd {WORKSPACE}\n'
            'PYTHONDONTWRITEBYTECODE=1 runtime/venvs/report-context-v12-12576792/bin/python '
            'TriCompose-v1.2/tools/build_first_version_delivery.py '
            '--run-id first_version_<NEW_OPAQUE_ID> --archive\n\n'
            'Read START_HERE_CN_EN.md, then case_index.csv and baseline_comparison.csv.\n'
            'candidate_index.csv paths point to original protected synthetic artifacts on CARC.\n'
            'This archive contains no generated payloads or source patient records.\n')
        summary = {'schema_version': SCHEMA, 'status': 'completed_metadata_first_version_delivery',
            'inventory': info, 'comparison_rows': len(comparisons), 'case_mean_rows': len(means),
            'paired_comparison_rows': len(paired), 'score_free_choices_replayed': len(controls),
            'synthetic_artifact_byte_hashes_checked': len(payloads),
            'clinical_qualified': False, 'original_selection_changed': False,
            'new_model_calls': 0, 'new_gpu_calls': 0, 'new_slurm_submissions': 0,
            'payload_semantics_inspected': False, 'payloads_copied_to_delivery': False,
            'mimic_source_or_real_target_read': False, 'regeneration_authorized': False,
            'cost_semantics': 'cached_simulated_calls_not_measured_gpu_savings',
            'elapsed_seconds_before_commit': round(time.monotonic() - started, 6)}
        write_private_json(temporary / 'summary.json', summary)
        require(all(sha256_file(p) == h for p, h in sources.values()), 'immutable_sources_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA,
            'run_id': run_id, 'source_paths': {k: str(p) for k, (p, _) in sources.items()},
            'source_sha256': {k: h for k, (_, h) in sources.items()},
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir()},
            'original_selection_changed': False, 'clinical_qualified': False,
            'new_model_calls': 0, 'regeneration_authorized': False})
        require(not target.exists(), 'delivery_overwrite_refused')
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    if archive:
        committed = json.loads((target / 'manifest.json').read_text())
        names = sorted([*committed['artifacts'], 'manifest.json'])
        require({p.name for p in target.iterdir()} == set(names), 'archive_inventory_changed')
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(archive_path, flags, 0o660)
        with os.fdopen(fd, 'wb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz') as package:
                for name in names:
                    p = target / name
                    require(p.is_file() and not p.is_symlink(), 'metadata_archive_files_only')
                    if name != 'manifest.json':
                        require(sha256_file(p) == committed['artifacts'][name]['sha256'], 'archive_artifact_changed')
                    metadata = package.gettarinfo(str(p), arcname=run_id + '/' + p.name)
                    metadata.uid = metadata.gid = metadata.mtime = 0
                    metadata.uname = metadata.gname = ''
                    with p.open('rb') as contents:
                        package.addfile(metadata, contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(archive_path, 0o660)
    return target, summary, archive_path if archive else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=BASE / 'deliverables')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    try:
        target, summary, archive_path = execute(args.output_root, args.run_id, args.archive)
    except Exception as error:
        print(json.dumps({'status': 'first_version_delivery_failed', 'error_type': type(error).__name__,
            'safe_reason': str(error) if isinstance(error, ValueError)
                and re.fullmatch('[a-z0-9_]+', str(error)) else 'details_suppressed', 'new_model_calls': 0}))
        return 1
    print(json.dumps({'status': summary['status'], 'elapsed_seconds': summary['elapsed_seconds_before_commit'],
        'manifest_sha256': sha256_file(target / 'manifest.json'),
        'archive_sha256': sha256_file(archive_path) if archive_path else None,
        'new_model_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
