#!/usr/bin/env python3
"""Isolated encoding sensitivity over immutable opaque numeric results only."""
import argparse
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import time

PATH = Path(__file__).with_name('run_radevalx_published_benchmark_v2.py')
spec = importlib.util.spec_from_file_location('frozen_radevalx_v2_for_sensitivity', PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
base = prior.base
alignment = base.alignment
SOURCE = base.BASE/'report_metric_alignment_runs/radevalx_published_v2_12714150_001'
SOURCE_SHA = '1cd67427ef20e817103697b1883f208cded273d22725f2b54bfaad06cce4e4f5'
PROTOCOL = base.WORKSPACE/'docs/radevalx_encoding_sensitivity_protocol.md'
TEST = base.ROOT/'tests/test_radevalx_encoding_sensitivity.py'
SCHEMA = 'tricompose-radevalx-encoding-sensitivity-v1'
TARGETS = {
    'all100': {'published_radgraph_f1': (0.2844, 0.1633), 'published_radcliq': (0.3349, 0.1929)},
    'total_errors_gt3': {'published_radgraph_f1': (0.1910, 0.0091), 'published_radcliq': (0.3380, 0.0169)},
}
POLICY = {
    'exploratory_encoding_sensitivity_only': True, 'official_blank_encoding_verified': False,
    'encoding_verified': False, 'hypothetical_blank_is_zero_is_reference_truth': False,
    'clinical_qualified': False, 'local_implementation_qualified': False,
    'primary_metric_eligible': False, 'selection_changed': False,
    'regeneration_authorized': False, 'report_reference_errors_are_image_or_ehr_truth': False,
    'new_model_calls': 0, 'new_slurm_submissions': 0, 'external_api': False,
    'original_null_references_changed': False, 'bootstrap_resamples': 1000, 'seed': 0,
    'paper_targets_known_before_analysis': True, 'direction_or_threshold_fitting': False,
}


def calculate(predictions, references):
    """Keep missingness interpretation separate; never mutate input contracts."""
    definitions, original = alignment.validate(predictions, references)
    hypothetical = copy.deepcopy(original)
    cells = sum(len(v) for r in original for v in r['errors'].values())
    missing = sum(v is None for r in original for values in r['errors'].values() for v in values)
    for row in hypothetical:
        row['errors'] = {level: [0 if v is None else v for v in values]
                         for level, values in row['errors'].items()}
    scenarios = {}
    for name, rows in (('blank_is_unresolved', original), ('hypothetical_blank_is_zero', hypothetical)):
        totals = {r['item_id']: alignment.outcomes(r) for r in rows}
        groups = {'all100': rows}
        if name == 'hypothetical_blank_is_zero':
            groups['total_errors_gt3'] = [r for r in rows if totals[r['item_id']]['all_errors_total'] > 3]
        results = {}
        for cohort, subset in groups.items():
            metrics = {}
            for metric, definition in definitions.items():
                direction = 1 if definition['orientation'] == 'higher_is_better' else -1
                outcomes = {}
                for outcome in ('all_errors_total', 'clinically_significant_total'):
                    eligible = [(r['source_group_id'], direction * r['scores'][metric]['value'],
                        -totals[r['item_id']][outcome]) for r in subset
                        if r['scores'][metric]['status'] == 'complete' and totals[r['item_id']][outcome] is not None]
                    result = alignment.correlation([(quality, errors) for _, quality, errors in eligible])
                    result['attempted_pairs'] = len(subset)
                    result['paired_coverage'] = len(eligible)/len(subset) if subset else None
                    if outcome == 'clinically_significant_total' and cohort == 'all100':
                        result['conditional_encoding_group_bootstrap'] = alignment.clustered_spearman_interval(
                            eligible, POLICY['bootstrap_resamples'], POLICY['seed'])
                    outcomes[outcome] = result
                metrics[metric] = outcomes
            results[cohort] = {'attempted_pairs': len(subset), 'metrics': metrics,
                'outcome_defined_subset': cohort == 'total_errors_gt3'}
        scenarios[name] = results
    reproduction = []
    for cohort, metrics in TARGETS.items():
        for metric, expected in metrics.items():
            for outcome, target in zip(('all_errors_total', 'clinically_significant_total'), expected):
                actual = scenarios['hypothetical_blank_is_zero'][cohort]['metrics'][metric][outcome]['spearman']
                delta = actual - target if actual is not None else None
                reproduction.append({'cohort': cohort, 'metric': metric, 'outcome': outcome,
                    'paper_spearman': target, 'hypothetical_spearman': actual,
                    'signed_deviation': delta, 'matches_paper_rounding': delta is not None and abs(delta) <= 0.000050000001})
    return {'schema_version': SCHEMA, 'policy': dict(POLICY), 'attempted_pairs': len(original),
        'error_count_cells': cells, 'observed_count_cells': cells-missing, 'unresolved_count_cells': missing,
        'correlation_direction': 'oriented_quality_vs_negative_expert_error_burden',
        'scenarios': scenarios, 'paper_table4_reproduction': reproduction,
        'paper_noisy_subset_pairs': 30,
        'hypothetical_noisy_subset_pairs': scenarios['hypothetical_blank_is_zero']['total_errors_gt3']['attempted_pairs'],
        'matching_published_values': sum(r['matches_paper_rounding'] for r in reproduction),
        'paper_values_compared': len(reproduction)}


def source_pins():
    base.guard.guard()
    pins = {str(p.resolve()): base.sha256_file(p) for p in (Path(__file__), TEST, PROTOCOL, PATH)}
    manifest = base.guard.metadata(SOURCE/'manifest.json', pins, SOURCE_SHA)
    if manifest['schema_version'] != prior.SCHEMA+'-run-manifest' or any(manifest.get(k) is not False
            for k in ('clinical_qualified', 'selection_changed', 'regeneration_authorized')):
        raise ValueError('unchanged_diagnostic_source_run_required')
    base.guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    for name in ('published_predictions.json', 'references.json', 'summary.json'):
        path = SOURCE/name
        if base.sha256_file(path) != manifest['artifacts'][name]:
            raise ValueError('derived_numeric_source_hash_changed')
        pins[str(path)] = manifest['artifacts'][name]
    if base.sha256_file(base.guard.BANK/'manifest.json') != base.guard.BANK_SHA:
        raise ValueError('original_bank_changed')
    pins[str(base.guard.BANK/'manifest.json')] = base.guard.BANK_SHA
    return pins


def prepare(run_id):
    pins = source_pins()
    temporary, target = base.new_atomic_run(base.BASE/'report_metric_encoding_plans', run_id)
    try:
        base.dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'policy': POLICY,
            'source_run_manifest_sha256': SOURCE_SHA, 'paper_targets': TARGETS,
            'input_mode': 'previously_published_opaque_numeric_contracts_only'})
        base.guard.verify_pins(pins)
        base.dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {p.name: base.sha256_file(p) for p in sorted(temporary.iterdir())}})
        base.commit_atomic_run(temporary, target)
    except BaseException:
        base.discard_atomic_run(temporary)
        raise
    return target


def markdown(summary):
    lines = ['# RadEvalX blank-cell sensitivity / 空白标注解释敏感性', '',
        '**Exploratory assumption only; official blank encoding remains unverified.**', '',
        '原来的 null 结果、临床参考、score table、选优和首版交付包均未修改。',
        '下面数字假定空白类别计数为零；不是新跑模型，也不是临床评分器验证通过。', '',
        '| Released metric | Pairs | Spearman vs -significant errors | Conditional 95% CI | Spearman vs -all errors |',
        '| --- | ---: | ---: | --- | ---: |']
    render = lambda x: 'NA' if x is None else f'{x:.4f}'
    for metric, outcomes in summary['scenarios']['hypothetical_blank_is_zero']['all100']['metrics'].items():
        significant = outcomes['clinically_significant_total']
        interval = significant['conditional_encoding_group_bootstrap']['spearman_interval_95']
        ci = 'NA' if interval is None else '['+', '.join(render(v) for v in interval)+']'
        lines.append(f"| {metric} | {significant['paired_rows']} | {render(significant['spearman'])} | {ci} | "
                     f"{render(outcomes['all_errors_total']['spearman'])} |")
    lines += ['', '## Published-table reproduction / 论文数值复现', '',
        f"Matched rounding: {summary['matching_published_values']}/{summary['paper_values_compared']}; "
        f"hypothetical >3-error subgroup: {summary['hypothetical_noisy_subset_pairs']} (paper:30).", '',
        '| Cohort | Metric | Outcome | Paper | Hypothesis | Signed delta | Rounding match |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in summary['paper_table4_reproduction']:
        lines.append('| '+' | '.join(str(row[k]) if k in ('cohort', 'metric', 'outcome', 'matches_paper_rounding')
            else render(row[k]) for k in ('cohort', 'metric', 'outcome', 'paper_spearman',
                'hypothetical_spearman', 'signed_deviation', 'matches_paper_rounding'))+' |')
    lines += ['', '原始解释：所有类别必须已知才求总数，primary 有效配对仍为0/100；保留 NA。',
        'Numerical matches support compatibility, not proof of the annotation encoding. All mismatches remain visible.',
        'Readers compared reports without images; the cohort was selected using abnormality/RadCliQ-related filters.',
        'No reference-free selection, image/EHR truth, local checkpoint qualification or repair trigger is established.',
        'The >3 subgroup is outcome-defined reproduction only, not an independent test cohort.',
        'Intervals condition on the zero assumption and do not incorporate unresolved encoding uncertainty.',
        'Sources: https://physionet.org/content/rad-eval-x/1.0.0/ ; https://arxiv.org/html/2311.16764v1#S7', '']
    return '\n'.join(lines)


def evaluate(plan_root, expected, run_id, allowed):
    base.guard.guard()
    if not allowed:
        raise ValueError('explicit_exploratory_zero_assumption_required')
    pins = {}
    manifest = base.guard.metadata(Path(plan_root)/'manifest.json', pins, expected)
    if manifest['schema_version'] != SCHEMA+'-plan-manifest':
        raise ValueError('frozen_encoding_sensitivity_plan_required')
    base.guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = base.guard.metadata(Path(plan_root)/'plan.json', pins, manifest['artifacts']['plan.json'])
    targets = json.loads(json.dumps(TARGETS))
    if plan['policy'] != POLICY or plan['paper_targets'] != targets or plan['source_run_manifest_sha256'] != SOURCE_SHA:
        raise ValueError('unchanged_assumption_and_reproduction_targets_required')
    current = source_pins()
    if any(pins.get(p) != digest for p, digest in current.items()):
        raise ValueError('frozen_source_closure_changed')
    predictions = base.guard.metadata(SOURCE/'published_predictions.json', pins, current[str(SOURCE/'published_predictions.json')])
    references = base.guard.metadata(SOURCE/'references.json', pins, current[str(SOURCE/'references.json')])
    if predictions['benchmark'] != 'radevalx-1.0.0' or len(predictions['records']) != 100 or \
            predictions['metric_definitions'] != {name: {'orientation': direction, 'provenance': 'published_cached'}
                for name, direction in prior.adapter.METRICS.values()}:
        raise ValueError('same_all100_published_metric_inventory_required')
    temporary, target = base.new_atomic_run(base.BASE/'report_metric_encoding_runs', run_id)
    started = time.monotonic()
    try:
        summary = calculate(predictions, references)
        summary.update(elapsed_seconds=round(time.monotonic()-started, 6), actual_slurm_job_id=os.environ['SLURM_JOB_ID'],
            original_run_manifest_sha256=SOURCE_SHA, plan_manifest_sha256=expected,
            source_csv_rows_decoded=False, source_report_fields_accessed=False,
            original_source_bytes_rehashed_for_provenance=True, real_mimic_or_candidate_bodies_read=False)
        base.dump(temporary/'summary.json', summary)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(summary['paper_table4_reproduction'][0]))
        writer.writeheader()
        writer.writerows(summary['paper_table4_reproduction'])
        base.write_private_text(temporary/'paper_reproduction.csv', stream.getvalue())
        base.write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        base.guard.verify_pins(pins)
        base.dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-run-manifest', 'sources': pins,
            'artifacts': {p.name: base.sha256_file(p) for p in sorted(temporary.iterdir())},
            'policy': POLICY, 'source_run_manifest_sha256': SOURCE_SHA})
        base.commit_atomic_run(temporary, target)
    except BaseException:
        base.discard_atomic_run(temporary)
        raise
    return target


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('prepare', 'evaluate'))
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--plan-root', type=Path)
    cli.add_argument('--plan-manifest-sha256')
    cli.add_argument('--allow-exploratory-zero-assumption', action='store_true')
    args = cli.parse_args()
    os.umask(0o007)
    try:
        target = prepare(args.run_id) if args.action == 'prepare' else evaluate(args.plan_root,
            args.plan_manifest_sha256, args.run_id, args.allow_exploratory_zero_assumption)
    except Exception as error:
        print(json.dumps({'status': 'encoding_sensitivity_failed_closed', 'error_type': type(error).__name__,
            'source_text_exposed': False}))
        return 2
    print(json.dumps({'status': 'encoding_sensitivity_'+args.action+'_complete', 'model_calls': 0,
        'manifest_sha256': base.sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
