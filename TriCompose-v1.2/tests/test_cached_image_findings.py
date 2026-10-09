"""Wholly invented metadata/mock tests; no models, pixels or patient inputs."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cached_image_worker_fixture',
    ROOT/'tools/verify_cached_image_findings.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(image_state='unknown', retained_state='negative', contract='complete'):
    rows, facts, gates, records = [], [], [], []
    images = []
    for case in range(2):
        for model in range(3):
            iid = f'invented_image_{case}_{model}'
            image_hash = m.image_interface.digest_text(iid)
            images.append({'cxr_candidate_id': iid})
            records.append({'cxr_candidate_id': iid, 'cxr_sha256': image_hash,
                'contract_status': contract,
                'states': dict.fromkeys(m.image_interface.FINDINGS, image_state) if contract == 'complete' else None})
            for report in range(4):
                cid = f'invented_triple_{case}_{model}_{report}'
                rid = f'invented_report_{case}_{model}_{report}'
                report_hash = m.image_interface.digest_text(rid)
                rows.append({'case_id': f'invented_case_{case}', 'cxr_candidate_id': iid,
                    'cxr_sha256': image_hash, 'ehr_sha256': m.image_interface.digest_text(str(case)),
                    'ehr_facts_sha256': m.image_interface.digest_text('facts'+str(case))})
                for finding in m.CHEXPERT_FINDINGS:
                    facts.append({'triple_candidate_id': cid, 'report_candidate_id': rid, 'finding': finding,
                        'cxr_candidate_id': iid, 'artifact_hashes': {'cxr_sha256': image_hash, 'report_sha256': report_hash},
                        'states': {'xrv': 'negative', 'chexbert': 'negative', 'ehr': 'unknown'}})
                    gates.append({'triple_candidate_id': cid, 'report_candidate_id': rid, 'finding': finding,
                        'report_sha256': report_hash, 'raw_chexbert_state': 'negative',
                        'scopegate_decision': 'scope_commit' if finding in ('cardiomegaly', 'consolidation',
                            'pleural_effusion', 'pneumothorax') else 'outside_verifier_scope',
                        'scopegate_retained_state': retained_state if finding in ('cardiomegaly', 'consolidation',
                            'pleural_effusion', 'pneumothorax') else None,
                        'independent_clinical_validation': False, 'regeneration_authorized': False})
    return rows, facts, gates, records, {'image_inputs': images}


def compare_fixture(parts):
    rows, facts, gates, records, plan = parts
    manifest = {'artifacts': {'candidate_gate_fact_table.jsonl': {'sha256': 'a'*64}}}
    def parent(label):
        return manifest, {'candidate_rows': rows, 'fact_rows': facts}, {}
    with patch.object(m, 'parent', side_effect=parent), \
            patch.object(m, 'sha256_file', return_value='a'*64), \
            patch.object(m.Path, 'stat', return_value=SimpleNamespace(st_size=128)), \
            patch.object(m.Path, 'read_text', return_value='\n'.join(json.dumps(r) for r in gates)):
        return m.compare_after_freeze(records, plan)


class CachedImageFindingTests(unittest.TestCase):
    def test_original_image_prompt_and_eight_heads_reused(self):
        sentinel = object()
        messages = m.image_interface.request_messages('image', image=sentinel)
        self.assertEqual(len(messages), 1)
        self.assertIs(messages[0]['content'][0]['image'], sentinel)
        text = messages[0]['content'][1]['text']
        self.assertEqual(text, m.image_interface.IMAGE_PROMPT.format(findings=', '.join(m.image_interface.FINDINGS)))
        for forbidden in ('invented_case', 'report_candidate_id', 'untrusted_report', 'score_table'):
            self.assertNotIn(forbidden, text)

    def test_image_requests_cannot_receive_report(self):
        with self.assertRaises(ValueError):
            m.image_interface.request_messages('image', image=object(), report='invented text')

    def test_valid_states_including_unknown_and_uncertain_preserved(self):
        states = dict.fromkeys(m.image_interface.FINDINGS, 'unknown')
        states['pneumonia'] = 'uncertain'
        result = m.sanitized_decode(json.dumps(states))
        self.assertEqual(result['states'], states)
        self.assertEqual(result['contract_status'], 'complete')
        self.assertFalse(result['independent_clinical_validation'])

    def test_bad_json_not_successful_unknown(self):
        for response in ('{}', 'bad json', '"positive"', '{"pneumonia":true}'):
            result = m.sanitized_decode(response)
            self.assertEqual(result['contract_status'], 'failed_unavailable')
            self.assertIsNone(result['states'])

    def test_duplicate_extra_keys_and_invalid_states_unavailable(self):
        states = dict.fromkeys(m.image_interface.FINDINGS, 'negative')
        for response in (json.dumps({**states, 'score': 1}),
                json.dumps(states)[:-1]+', "edema":"positive"}',
                json.dumps({**states, 'edema': 'absent'})):
            self.assertIsNone(m.sanitized_decode(response)['states'])

    def test_token_cap_disables_otherwise_valid_response(self):
        result = m.sanitized_decode(json.dumps(dict.fromkeys(m.image_interface.FINDINGS, 'negative')),
            token_limit_reached=True)
        self.assertIsNone(result['states'])
        self.assertEqual(result['failure_reason'], 'token_cap')

    def test_complete_six_image_inventory_without_score_selection(self):
        rows, *_ = fixture()
        before = copy.deepcopy(rows)
        images = m.expected_images(rows)
        self.assertEqual(rows, before)
        self.assertEqual(len(images), 6)
        self.assertEqual({r['case_id'] for r in images.values()}, {'invented_case_0', 'invented_case_1'})

    def test_missing_rows_or_wrong_shared_image_hash_refused(self):
        rows, *_ = fixture()
        for bad in (rows[:-1], [{**rows[0], 'cxr_sha256': 'b'*64}]+rows[1:]):
            with self.assertRaises(ValueError):
                m.expected_images(bad)

    def test_same_image_new_ehr_anchor_refused(self):
        rows, *_ = fixture()
        rows[0]['ehr_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'shared_image_or_fixed_ehr_changed'):
            m.expected_images(rows)

    def test_original_staging_header_only_inside_workspace_not_body_loading(self):
        staged = {'source_v1_staging_run': '/invented/staging', 'source_v1_run_manifest_sha256': 'a'*64}
        with patch.object(m, 'require_inside', side_effect=lambda path, root, **kw: Path(path)) as boundary, \
                patch.object(m, 'sha256_file', return_value='a'*64), \
                patch.object(m, 'bounded_json', return_value={'schema_version': 'tricompose.staging.run.v1',
                    'source_generator': 'synehrgy_gpt2_10bins'}) as read:
            path = m.synthetic_origin_manifest(staged)
            self.assertEqual(path.name, 'run_manifest.json')
            self.assertEqual(boundary.call_args_list[0].args[1], m.WORKSPACE)
            read.assert_called_once_with(path)

    def test_unknown_or_real_generator_in_header_refused(self):
        staged = {'source_v1_staging_run': '/invented/staging', 'source_v1_run_manifest_sha256': 'a'*64}
        with patch.object(m, 'require_inside', side_effect=lambda path, root, **kw: Path(path)), \
                patch.object(m, 'sha256_file', return_value='a'*64), \
                patch.object(m, 'bounded_json', return_value={'schema_version': 'tricompose.staging.run.v1',
                    'source_generator': 'real_mimic_rows'}):
            with self.assertRaisesRegex(ValueError, 'known_synthetic_ehr_generator_required'):
                m.synthetic_origin_manifest(staged)

    def test_legacy_canonical_request_hash_distinct_from_file_bytes(self):
        payload = {'invented': True, 'key': 0}
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        pretty = json.dumps(payload, indent=2)+'\n'
        self.assertEqual(m.canonical_request_sha256(payload), m.image_interface.digest_text(canonical))
        self.assertNotEqual(m.canonical_request_sha256(payload), m.image_interface.digest_text(pretty))

    def test_unknown_not_negative_or_explicit_agreement(self):
        self.assertEqual(m.relation('unknown', 'negative'), 'single_source_assertion_unqualified')
        self.assertEqual(m.relation('unknown', 'unknown'), 'both_unmentioned_or_unassessable')
        self.assertEqual(m.relation('positive', 'negative'), 'explicit_opposition_unqualified')

    def test_uncertainty_not_hard_opposition(self):
        for state in m.STATES:
            self.assertEqual(m.relation('uncertain', state), 'uncertainty_not_comparable')

    def test_comparison_preserves_all_gate_cells_and_unique_image_denominator(self):
        parts = fixture(image_state='negative')
        before = copy.deepcopy(parts)
        output, summary, _ = compare_fixture(parts)
        self.assertEqual(parts, before)
        for old, new in zip(parts[2], output):
            self.assertEqual({k: new[k] for k in old}, old)
        self.assertEqual(len(output), 336)
        self.assertEqual(summary['unique_image_finding_relation_counts']['explicit_agreement_unqualified'], 48)
        self.assertEqual(summary['unique_image_finding_relation_counts']['outside_image_verifier_scope'], 36)
        self.assertEqual(summary['retained_report_relation_counts_candidate_findings']['explicit_agreement_unqualified'], 96)
        self.assertEqual(summary['clinical_requests_resolved'], 0)
        self.assertFalse(summary['regeneration_authorized'])

    def test_unretained_report_cannot_be_used_for_agreement(self):
        parts = fixture(image_state='negative')
        for gate in parts[2]:
            if gate['finding'] == 'cardiomegaly':
                gate['scopegate_decision'] = 'abstain'
                gate['scopegate_retained_state'] = None
        _, summary, _ = compare_fixture(parts)
        self.assertEqual(summary['retained_report_relation_counts_candidate_findings']['report_assertion_not_retained'], 120)

    def test_failed_image_has_null_state_and_no_agreement(self):
        parts = fixture(contract='failed_unavailable')
        output, summary, _ = compare_fixture(parts)
        self.assertEqual(summary['failed_image_responses'], 6)
        self.assertEqual(summary['unique_image_finding_relation_counts']['image_verifier_unavailable'], 48)
        self.assertTrue(all(r['imagecheck_qwen_state'] is None for r in output))

    def test_exact_report_image_hashes_and_cached_state_required(self):
        for key, value in (('report_sha256', 'b'*64), ('report_candidate_id', 'invented_other'),
                ('raw_chexbert_state', 'positive')):
            parts = fixture()
            parts[2][0][key] = value
            with self.assertRaises(ValueError):
                compare_fixture(parts)
        parts = fixture()
        parts[3][0]['cxr_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            compare_fixture(parts)

    def test_missing_duplicate_foreign_predictions_rejected(self):
        for kind in ('missing', 'duplicate', 'foreign'):
            parts = fixture()
            if kind == 'missing':
                parts[3].pop()
            elif kind == 'duplicate':
                parts[3][-1] = copy.deepcopy(parts[3][0])
            else:
                parts[3][-1]['cxr_candidate_id'] = 'invented_other'
            with self.assertRaisesRegex(ValueError, 'all_six_image_results_required'):
                compare_fixture(parts)

    def test_missing_duplicate_gate_facts_cannot_change_denominator(self):
        for kind in ('missing', 'duplicate'):
            parts = fixture()
            if kind == 'missing':parts[2].pop()
            else:parts[2].append(copy.deepcopy(parts[2][0]))
            with self.assertRaises(ValueError):
                compare_fixture(parts)

    def test_malformed_complete_or_failed_result_not_interpreted_as_unknown(self):
        for failed, states in ((False, {}), (False, None), (False, {'edema': 'negative'}),
                (True, dict.fromkeys(m.image_interface.FINDINGS, 'unknown'))):
            parts = fixture()
            parts[3][0]['states'] = states
            if failed:parts[3][0]['contract_status'] = 'failed_unavailable'
            with self.assertRaises(ValueError):
                compare_fixture(parts)

    def test_unsupported_heads_stay_outside_scope(self):
        output, summary, _ = compare_fixture(fixture())
        outside = [r for r in output if r['finding'] == 'support_devices']
        self.assertTrue(all(r['imagecheck_qwen_state'] is None for r in outside))
        self.assertTrue(all(r['imagecheck_xrv_qwen_relation'] == 'outside_image_verifier_scope' for r in outside))

    def test_env_alone_and_missing_gpu_approval_refused(self):
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': '42'}), \
                patch.object(m.Path, 'read_text', return_value='no allocation'):
            with self.assertRaises(RuntimeError):m.require_slurm()
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': '42'}), \
                patch.object(m.Path, 'read_text', return_value='/job_42/step_0'):
            with self.assertRaises(RuntimeError):m.require_slurm(gpu=True, approved=False)

    def test_existing_run_refused_before_preparation(self):
        args = SimpleNamespace(mode='prepare', allow_image_only_verification=False,
            output_root='/invented', run_id='invented')
        with patch.object(m, 'require_slurm'), \
                patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'prepare') as prepare:
            with self.assertRaises(FileExistsError):m.execute(args)
            prepare.assert_not_called()

    def test_gpu_permission_guard_before_torch_or_plan_loading(self):
        with patch.object(m, 'require_slurm', side_effect=RuntimeError('not approved')), \
                patch.object(m, 'load_plan') as load:
            with self.assertRaisesRegex(RuntimeError, 'not approved'):
                m.run('/invented', '/invented', approved=False)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
