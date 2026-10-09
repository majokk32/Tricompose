#!/usr/bin/env python3
"""Independent stdlib audit of metadata-only first-version handoff.

No builder/scorer import, model load, synthetic payload parsing or raw input.
"""
import csv
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import tarfile

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ROOT = WORKSPACE / 'artifacts/protected/tricompose_v1_2/deliverables/first_version_12714150_001'
PIN = 'a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc'
ARCHIVE_PIN = 'e5538b3ff6537396c0e1fd2ed3ef41fc9df04996e0fdfdab95d5b0ebf9307413'
EDGES = ('ehr_cxr', 'ehr_report', 'cxr_report')
COUNTS = ('known_reference_facts', 'comparable_facts', 'supported_facts',
          'supported_positive', 'supported_negative', 'proxy_opposition_facts')
MODES = {'fixed', 'random', 'random_acquisition_score_free_final', 'static_rerank', 'targeted_heuristic'}
CAPS = {4, 8, 12, 20, 30}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value == '':
        return None
    result = float(value)
    require(math.isfinite(result), 'finite_or_explicitly_missing_number_required')
    return result


def near(actual, expected):
    if expected is None:
        return actual == ''
    value = number(actual)
    return value is not None and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def main():
    require(sha(ROOT / 'manifest.json') == PIN, 'delivery_manifest_pin_changed')
    m = json.loads((ROOT / 'manifest.json').read_text())
    files = set(m['artifacts']) | {'manifest.json'}
    require(files == {p.name for p in ROOT.iterdir()}, 'delivery_inventory_changed')
    for p in (ROOT, *(ROOT / name for name in files), ROOT.with_suffix('.tar.gz')):
        info = p.stat()
        require(not p.is_symlink() and info.st_gid in (96293, 65534)
            and stat.S_IMODE(info.st_mode) == (0o2770 if p.is_dir() else 0o660),
            'project_boundary_modes_changed')
    for name, entry in m['artifacts'].items():
        require(sha(ROOT / name) == entry['sha256'], 'delivery_artifact_changed')
    require(set(m['source_paths']) == set(m['source_sha256']), 'source_inventory_differs')
    for label, raw_path in m['source_paths'].items():
        p = Path(raw_path)
        require(not p.is_symlink() and p.resolve().is_relative_to(WSPACE)
            and sha(p) == m['source_sha256'][label], 'source_pin_or_boundary_changed')

    index = csv_rows('candidate_index.csv')
    cases = csv_rows('case_index.csv')
    summary = json.loads((ROOT / 'summary.json').read_text())
    info = json.loads((ROOT / 'generation_inventory.json').read_text())
    groups, choices, ids = defaultdict(list), {}, set()
    for row in index:
        cid, case = row['triple_candidate_id'], row['case_id']
        require(cid not in ids and re.fullmatch(r'[A-Za-z0-9_-]+', case), 'opaque_unique_inventory_required')
        ids.add(cid); groups[case].append(row)
        for role in ('ehr', 'ehr_facts', 'prompt', 'cxr', 'report'):
            p = Path(row[role + '_path'])
            require(p.resolve().is_relative_to(WSPACE / 'artifacts/protected')
                and sha(p) == row[role + '_sha256'], 'indexed_synthetic_byte_binding_changed')
        for edge in EDGES:
            c = {field: int(row[edge + '_' + field]) for field in COUNTS}
            require(0 <= c['comparable_facts'] <= c['known_reference_facts'] <= 14
                and c['supported_facts'] + c['proxy_opposition_facts'] == c['comparable_facts']
                and c['supported_positive'] + c['supported_negative'] == c['supported_facts'],
                'raw_unknown_safe_counter_arithmetic_changed')
            for field, numerator in (('coverage_over_known', 'comparable_facts'),
                    ('support_over_known', 'supported_facts'), ('opposition_over_known', 'proxy_opposition_facts')):
                expected = c[numerator] / c['known_reference_facts'] if c['known_reference_facts'] else None
                require(near(row[edge + '_' + field], expected), 'unknown_denominator_or_rate_changed')
        if row['historical_static_selected'] == 'True':
            require(case not in choices, 'multiple_historical_choices_refused')
            choices[case] = row
        else:
            require(row['historical_static_selected'] == 'False', 'boolean_choice_required')
    expected_slots = {(a, '0', b) for a in ('roentgen_v2', 'chexgenbench_sana', 'chexgenbench_pixart')
        for b in ('maira2', 'cxrmate_single', 'llavarad', 'chexagent2')}
    require(len(index) == len(ids) == 960 and len(groups) == len(cases) == len(choices) == 80,
        'complete_eighty_case_inventory_required')
    for case, rows in groups.items():
        require(len(rows) == 12 and {(r['cxr_model_id'], r['cxr_seed'], r['report_model_id']) for r in rows} == expected_slots
            and len({(r['ehr_sha256'], r['ehr_facts_sha256']) for r in rows}) == 1,
            'fixed_ehr_model_grid_changed')
    require(len({r['cxr_candidate_id'] for r in index}) == 240
        and len({r['cxr_sha256'] for r in index}) == info['distinct_image_hashes'] == 147
        and len({r['report_sha256'] for r in index}) == info['distinct_report_hashes'] == 428,
        'duplicate_slots_hidden_or_hash_inventory_changed')
    require({r['case_id'] for r in cases} == set(groups), 'case_index_cohort_differs')
    for row in cases:
        chosen = choices[row['case_id']]
        for name, field in (('historical_selected_candidate_id', 'triple_candidate_id'),
                ('historical_selected_cxr_path', 'cxr_path'), ('historical_selected_report_path', 'report_path'),
                ('ehr_path', 'ehr_path'), ('ehr_sha256', 'ehr_sha256')):
            require(row[name] == chosen[field], 'case_choice_index_changed')
    old_scores = Path(m['source_paths']['historical_scores'])
    with old_scores.open() as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    require({r['triple_candidate_id'] for r in source if r['scoring']['selection']['selected']} ==
        {r['triple_candidate_id'] for r in choices.values()}, 'original_static_choices_changed')

    means = csv_rows('baseline_case_means.csv')
    comparison = csv_rows('baseline_comparison.csv')
    require(len(means) == 2000 and len(comparison) == 75
        and {(r['case_id'], r['method'], int(r['model_call_budget'])) for r in means} ==
        {(case, method, cap) for case in groups for method in MODES for cap in CAPS},
        'all_cases_methods_caps_required')
    fields = set(means[0]) - {'case_id', 'method', 'model_call_budget', 'ehr_scope', 'seed_replicates'}
    independent_checks = 0
    for row in comparison:
        cap, scope, method = int(row['model_call_budget']), row['ehr_scope'], row['method']
        available_cases = [r for r in means if r['method'] == method and int(r['model_call_budget']) == cap
            and (scope == 'all' or r['ehr_scope'] == scope)]
        require(int(row['fixed_ehr_cases']) == len(available_cases) ==
            {'all': 80, 'explicit_fact_proxy': 8, 'no_direct_comparable_ehr_facts': 72}[scope],
            'subgroup_or_patient_denominator_changed')
        for field in fields:
            values = [number(r[field]) for r in available_cases if r[field] != '']
            expected = sum(values) / len(values) if values else None
            require(near(row[field + '_mean'], expected)
                and int(row[field + '_available_ehr_cases']) == len(values),
                'independent_case_mean_or_missingness_mismatch')
            independent_checks += 1
    edges = csv_rows('baseline_by_edge.csv')
    require(len(edges) == 225, 'complete_three_edge_long_table_required')
    lookup = {(r['ehr_scope'], r['method'], r['model_call_budget']): r for r in comparison}
    for row in edges:
        parent = lookup[row['ehr_scope'], row['method'], row['model_call_budget']]
        for field in set(row) - {'ehr_scope', 'method', 'model_call_budget', 'fixed_ehr_cases', 'edge'}:
            require(row[field] == parent[row['edge'] + '_' + field], 'edge_long_table_changed')
    require(len(csv_rows('baseline_paired_comparison.csv')) == 60, 'paired_contrasts_missing')
    for name, label in (('baseline_comparison.csv', 'control_method_comparison.csv'),
                       ('baseline_case_means.csv', 'control_case_means.csv'),
                       ('baseline_paired_comparison.csv', 'control_paired_case_comparison.csv')):
        with Path(m['source_paths'][label]).open(newline='') as stream:
            require(csv_rows(name) == list(csv.DictReader(stream)), 'sealed_baseline_table_changed')
    require(summary['inventory'] == info and info['cases_with_direct_cached_ehr_facts'] == 8
        and info['cases_without_direct_cached_ehr_facts'] == 72
        and summary['synthetic_artifact_byte_hashes_checked'] == 1600,
        'summary_inventory_changed')
    for name in ('clinical_qualified', 'original_selection_changed', 'payload_semantics_inspected',
                 'payloads_copied_to_delivery', 'mimic_source_or_real_target_read', 'regeneration_authorized'):
        require(summary[name] is False, 'clinical_promotion_or_payload_scope_changed')
    require(summary['new_model_calls'] == summary['new_gpu_calls'] == summary['new_slurm_submissions'] == 0,
        'new_execution_scope_changed')
    archive = ROOT.with_suffix('.tar.gz')
    require(sha(archive) == ARCHIVE_PIN, 'archive_pin_changed')
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        require(len(members) == len(files) and {p.name for p in members} ==
            {ROOT.name + '/' + name for name in files}, 'archive_allowlist_changed')
        for member in members:
            name = Path(member.name).name
            require(member.isfile() and member.mode == 0o660
                and member.uid == member.gid == member.mtime == 0, 'archive_metadata_or_link_changed')
            digest = hashlib.sha256(package.extractfile(member).read()).hexdigest()
            require(digest == sha(ROOT / name), 'archive_content_differs')
    print(json.dumps({'status': 'independent_first_version_delivery_audit_passed',
        'fixed_ehr_cases': 80, 'candidate_slots': 960, 'independent_aggregate_checks': independent_checks,
        'protected_metadata_files': len(files), 'source_pins_checked': len(m['source_paths']),
        'clinical_qualified': False, 'original_selection_changed': False,
        'manifest_sha256': PIN, 'archive_sha256': ARCHIVE_PIN}, sort_keys=True))


WSPACE = WORKSPACE

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'first_version_delivery_audit_failed',
            'error_type': type(error).__name__, 'safe_reason': str(error)
            if isinstance(error, ValueError) and re.fullmatch('[a-z0-9_]+', str(error)) else 'suppressed'}))
        raise SystemExit(2) from None
