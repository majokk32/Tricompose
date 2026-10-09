"""Independent integrity/reward replay for the authored XL interface smoke.

No neural inference or medical adjudication. Reads authored graphs internally
and exports only hashes/counts/status, never their text or entity tokens.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
RUN = WORKSPACE / 'artifacts/protected/tricompose_v1_2/radgraph_interface_runs/native_xl_12714150_001'
OUTPUT = WORKSPACE / 'artifacts/protected/tricompose_v1_2/radgraph_interface_audits/native_xl_12714150_001'


def require(condition):
    if not condition:
        raise ValueError('radgraph_interface_integrity_check_failed')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUTPUT.exists())
    receipt = json.loads((RUN / 'run_manifest.json').read_text())
    checks = 0
    for pin in receipt['pins']:
        path = WORKSPACE / pin['path']
        require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == pin['sha256'])
        checks += 1
    for artifact in receipt['outputs']:
        path = RUN / artifact['path']
        require(path.resolve().is_relative_to(RUN) and sha256(path) == artifact['sha256'])
        checks += 1
    summary = json.loads((RUN / 'summary.json').read_text())
    table = json.loads((RUN / 'score_table.json').read_text())
    annotations = json.loads((RUN / 'native_annotations.json').read_text())
    require(summary['frozen_parameters'] is True and summary['network_disabled'] is True
            and summary['device'] == 'cpu' and summary['clinical_qualified'] is False)
    require(summary['report_inference_passes'] == 10 and table['eligible_pairs'] == 5)
    require(all(table['policy'][key] is False for key in
                ('clinical_qualified', 'image_factuality_verified', 'ehr_consistency_verified',
                 'selection_changed', 'regeneration_authorized')))
    expected_types = ('identical', 'negation', 'uncertainty', 'laterality', 'measurement', 'empty')
    require(tuple(row['authored_probe_type'] for row in table['records']) == expected_types)
    require(len(annotations['hypotheses']) == len(annotations['references']) == 5)
    spec = importlib.util.spec_from_file_location('official_radgraph_reward_replay',
                    WORKSPACE / 'RadGraph/radgraph/rewards.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    components = ('radgraph_entity_f1', 'radgraph_relation_presence_f1',
                  'radgraph_full_relation_f1')
    reward_checks = 0
    for row, hypothesis, reference in zip(table['records'], annotations['hypotheses'],
                                         annotations['references']):
        require(row['status'] == 'complete')
        native_scores = official.compute_reward(hypothesis, reference, 'all')
        for metric, score in zip(components, native_scores):
            require(abs(row['scores'][metric] - score) < 1e-12)
            reward_checks += 1
        for metadata, graph in ((row['hypothesis_graph'], hypothesis),
                                (row['reference_graph'], reference)):
            require(metadata['tokenized_text_sha256'] == hashlib.sha256(graph['text'].encode()).hexdigest())
            require(metadata['entity_count'] == len(graph['entities']))
            require(metadata['relation_count'] == sum(len(e['relations']) for e in graph['entities'].values()))
            require(sum(metadata['native_label_counts'].values()) == len(graph['entities']))
            require(metadata['nonmeasurement_observation_state_counts'] == {
                state: sum(e['label'] == label for e in graph['entities'].values())
                for state, label in [('positive', 'Observation::definitely present'),
                                     ('negative', 'Observation::definitely absent'),
                                     ('uncertain', 'Observation::uncertain')]})
            checks += 5
    require(table['records'][-1]['status'] == 'empty_input_not_eligible')
    require(all(value is None for value in table['records'][-1]['scores'].values()))
    require(all(value == 1 for value in table['records'][0]['scores'].values()))
    permission_checks = 0
    for path in (RUN, *RUN.iterdir()):
        info = path.stat()
        require(info.st_gid in (96293, 65534))
        require(info.st_mode & 0o7777 == (0o2770 if path.is_dir() else 0o660))
        permission_checks += 1
    os.umask(0o007)
    OUTPUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUTPUT.parent.chmod(0o2770)
    OUTPUT.mkdir(mode=0o2770)
    OUTPUT.chmod(0o2770)
    environment = {'python': sys.version.split()[0], 'packages': sorted([
        {'name': distribution.metadata['Name'], 'version': distribution.version}
        for distribution in importlib.metadata.distributions()], key=lambda item: item['name'].lower())}
    for filename, payload in [('environment_manifest.json', environment), ('audit.json', {
        'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'run_manifest.json'),
        'hash_and_graph_metadata_checks': checks, 'reward_components_recomputed': reward_checks,
        'protected_permission_checks': permission_checks,
        'new_model_calls': 0, 'clinical_qualified': False, 'selection_changed': False})]:
        with (OUTPUT / filename).open('x') as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write('\n')
        (OUTPUT / filename).chmod(0o660)
    print(json.dumps({'status': 'radgraph_interface_integrity_audit_passed',
                      'audit_sha256': sha256(OUTPUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
