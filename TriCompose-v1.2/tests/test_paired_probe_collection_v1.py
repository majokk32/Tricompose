"""Invented metadata and mocked workers only; no patient/weight/body reads."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'benchmarks'))
import collect_paired_probes_v1 as paired
from test_fresh_output_acceptance import fixture


def parent():
    cases = []
    for n in range(2):
        row, ctx = fixture(seed=2)
        ctx['anchor']['case_id'] = f'fixture_case_{n}'
        anchor = paired.anchor_from_record(ctx['anchor'])
        request = {'model_id':'roentgen_v2', 'case_id':anchor.case_id, 'seed':2,
            'request_id':f'fixture_request_{n}', 'inputs':{
                'synthetic_ehr':{'path':'invented_unopened_ehr','sha256':anchor.ehr_sha256},
                'ehr_facts':{'path':'invented_unopened_facts','sha256':anchor.ehr_facts_sha256},
                'final_prompt':{'path':'invented_unopened_prompt','sha256':'a'*64}}}
        cases.append({'case_id':anchor.case_id, 'anchor':anchor.record(), 'ehr_anchor_sha256':anchor.sha256,
            'requests':[{'request':request, 'canonical_request_sha256':paired.canonical_json_sha256(request),
                'source_path':'invented_unopened_source','source_sha256':'b'*64}]})
    workers = {m:{'minimum_planning_vram_gib':24} for m in
        ('roentgen_v2','cxrmate_single','chexagent2','xrv','chexbert')}
    return {'schema_version':'tricompose-online-report-escalation-smoke-v1', 'cases':cases,
        'factory_instantiated':False, 'source_bodies_parsed':False, 'clinical_acceptance':False,
        'historical_pool_is_untouched_test':False, 'workers':workers, 'source_pins':{},
        'artifact_pins':{}, 'biovil_python':'invented_python', 'biovil_model':'invented_model',
        'biovil_asset_pins':{}, 'minimum_gpu_vram_gib':24}


def plan():
    return paired.make_plan(parent(), {}, {})


class PairedPlanTests(unittest.TestCase):
    def test_reuses_metadata_without_preflight_or_body_reads(self):
        p = plan(); paired.validate_plan(p)
        self.assertFalse(p['source_bodies_or_weights_read_during_prepare'])
        self.assertEqual(p['planned_maximum']['generation_verification_attempts'], 24)
        self.assertEqual(p['policy']['call_budget_per_case'], 12)

    def test_only_seed_and_request_identity_change(self):
        source = parent(); original = deepcopy(source)
        new = paired.make_plan(source, {}, {})
        self.assertEqual(source, original)
        for old, case in zip(source['cases'], new['cases']):
            for r, seed in zip(case['requests'], (3,4)):
                self.assertEqual(r['request']['seed'], seed)
                self.assertEqual(r['request']['inputs'], old['requests'][0]['request']['inputs'])
                self.assertEqual(case['anchor'], old['anchor'])

    def test_duplicate_or_modified_fixed_ehr_rejected(self):
        p = plan(); p['cases'][1] = deepcopy(p['cases'][0])
        with self.assertRaises(ValueError): paired.validate_plan(p)
        p = plan(); p['cases'][0]['requests'][0]['request']['inputs']['synthetic_ehr']['sha256'] = '9'*64
        with self.assertRaises(ValueError): paired.validate_plan(p)

    def test_prompt_change_is_not_a_seed_probe(self):
        p = plan(); r = p['cases'][0]['requests'][1]
        r['request']['inputs']['final_prompt']['sha256'] = '9'*64
        r['canonical_request_sha256'] = paired.canonical_json_sha256(r['request'])
        with self.assertRaises(ValueError): paired.validate_plan(p)

    def test_no_policy_or_worker_memory_silent_change(self):
        for field,value in [('controller_budget',100),('training_allowed',True),('selection_uses_secondary',True)]:
            p = plan(); p['policy'][field] = value
            with self.assertRaises(ValueError): paired.validate_plan(p)
        p = plan(); p['minimum_gpu_vram_gib'] = 16
        with self.assertRaises(ValueError): paired.validate_plan(p)

    def test_gpu_guard_precedes_plan_read_and_cuda_import(self):
        with patch.object(paired,'require_gpu_slurm',side_effect=RuntimeError('invented_guard')), \
                patch.object(paired,'require_inside') as read:
            with self.assertRaises(RuntimeError): paired.run(None)
            read.assert_not_called()

    def test_prepare_guard_precedes_any_metadata_read(self):
        with patch.object(paired,'cpu_guard',side_effect=RuntimeError('invented_guard')), \
                patch.object(paired,'checked') as read:
            with self.assertRaises(RuntimeError): paired.prepare(None)
            read.assert_not_called()

    def test_each_seed_uses_distinct_stable_branch_and_six_call_cap(self):
        p = plan(); calls = []
        def worker(case, settings, root):
            calls.append((case['requests'],settings['policy']['call_budget_per_case'],root))
            self.assertEqual(len(case['requests']),1)
            return [case['requests'][0]['request']['seed']], {'charged_model_attempts':6}, []
        with patch.object(paired,'private_directory'), patch.object(paired,'one_case',side_effect=worker):
            completed, books = paired.collect_case(p['cases'][0],p,Path('invented_not_created'))
        self.assertEqual(completed,[3,4]); self.assertEqual(sum(x['charged_model_attempts'] for x in books),12)
        self.assertEqual({str(x[2]) for x in calls},{'invented_not_created/image_branch_0','invented_not_created/image_branch_1'})
        self.assertTrue(all(x[1]==6 for x in calls))


class FreshProjectionTests(unittest.TestCase):
    def test_fresh_profile_disables_unsupported_heads_preserves_unknown(self):
        row,ctx = fixture(seed=3)
        obs = paired.fresh_snapshot(row,ctx)
        self.assertEqual(obs['states']['xrv']['support_devices'],'unknown')
        self.assertEqual(obs['states']['chexbert']['edema'],'unknown')
        self.assertEqual(obs['quality']['report_structure_quality_score_0_1'],1)
        self.assertFalse(obs['clinical_qualified'])

    def test_synthetic_structure_risks_are_explicit_gates_not_fact_truth(self):
        for key,value in [('unsupported_temporal_comparison_language',True),('generic_report',True),
                          ('repeated_sentence_count',1),('repeated_4gram_ratio',.1)]:
            row,ctx = fixture(seed=3); row['structure'][key] = value
            obs = paired.fresh_snapshot(row,ctx)
            self.assertEqual(obs['artifact_gate_failures'],1)
            self.assertEqual(obs['states']['chexbert']['edema'],'unknown')

    def test_no_staged_primary_references_or_endpoint_into_controller(self):
        row,ctx = fixture(seed=3); row['biovil_raw_cosine'] = 1
        obs = paired.fresh_snapshot(row,ctx)
        self.assertNotIn('biovil_raw_cosine',obs)

    def test_paired_matrix_keeps_both_ehrs_and_expert_image_isolation(self):
        rows = []; context = None
        for seed in (3,4):
            for model in ('cxrmate_single','chexagent2'):
                row,ctx = fixture(seed=seed,model=model,image_id=f'fixture_image_{seed}',
                    report_id=f'fixture_report_{seed}_{model}')
                rows.append(row); context = ctx
        readout = paired.paired_readout(rows,context)
        self.assertEqual(len(readout['action_credits']),8)
        self.assertEqual(len(readout['choices']),3)
        self.assertFalse(readout['choices'][1]['selection_was_online'])
        self.assertFalse(readout['choices'][1]['actual_regeneration_executed'])
        self.assertIsNone(readout['clinical_repair_success'])

    def test_missing_initial_case_retained_without_fake_success(self):
        row,ctx = fixture(seed=4)
        r = paired.paired_readout([row],ctx)
        self.assertEqual(r['status'],'initial_chain_incomplete_retained')
        self.assertEqual(r['choices'],[])

    def test_foreign_seed_or_duplicate_slot_rejected(self):
        row,ctx = fixture(seed=99)
        with self.assertRaises(ValueError): paired.paired_readout([row],ctx)
        row,ctx = fixture(seed=3)
        with self.assertRaises(ValueError): paired.paired_readout([row,row],ctx)


if __name__ == '__main__':
    unittest.main()
