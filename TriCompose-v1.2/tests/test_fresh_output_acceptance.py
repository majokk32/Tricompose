"""Invented fresh receipts, states and cost journals; no model/body/weight reads."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
import fresh_output_acceptance as gate
from tricompose_v12.execution_ledger import BoundedCallLedger, CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest, _edge
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from test_live_receipts import data, H


def fixture(*, report_state='unknown', image_state='positive', image_id='fixture_cxr',
            report_id='fixture_report', known=True, seed=0, model='cxrmate_single'):
    anchor, image, il, report, rl = data(known)
    enabled = set(gate.FINDINGS[:8])
    for name, threshold in il['thresholds'].items():
        threshold['enabled'] = name in enabled
    image.update(candidate_id=image_id, artifact={'sha256': _digest(['fixture_image', image_id])})
    image_row = il['records'][0]
    image_row.update(cxr_candidate_id=image_id, image_sha256=image['artifact']['sha256'])
    image_row['finding_states']['edema'] = image_state
    image_row['finding_probabilities']['edema'] = {
        'positive': .9, 'negative': .1, 'unknown': None, 'uncertain': .5}[image_state]
    report.update(candidate_id=report_id, parent_cxr_candidate_id=image_id,
        input_cxr={'sha256': image['artifact']['sha256']},
        artifact={'sha256': _digest(['fixture_report', report_id])})
    rl['records'][0].update(report_candidate_id=report_id, report_sha256=report['artifact']['sha256'])
    rl['records'][0]['finding_states']['edema'] = report_state
    image_labels_hash = _digest(['fixture_image_labels', image_id, image_state])
    report_labels_hash = _digest(['fixture_report_labels', report_id, report_state])
    partial = image_receipt(anchor, image, il, label_sha256=image_labels_hash,
        thresholds_sha256=H['thresholds'], checkpoint_sha256=H['xrv'])
    receipt = completed_receipt(anchor, partial, image, il, report, rl,
        image_labels_sha256=image_labels_hash, report_labels_sha256=report_labels_hash,
        thresholds_sha256=H['thresholds'], xrv_checkpoint_sha256=H['xrv'], chexbert_checkpoint_sha256=H['chexbert'])
    structure = {'case_id': anchor.case_id, 'report_candidate_id': report_id,
        'report_model_id': model, 'report_sha256': report['artifact']['sha256'],
        'image_sha256': image['artifact']['sha256'], 'parent_cxr_candidate_id': image_id,
        'source_cxr_model_id': 'roentgen_v2', 'findings_complete': True,
        'impression_complete': model == 'cxrmate_single',
        'impression_required_by_model_contract': model == 'cxrmate_single',
        'section_contract_pass': True, 'empty': False, 'generic_report': False,
        'unsupported_temporal_comparison_language': False, 'repeated_sentence_count': 0,
        'repeated_4gram_ratio': 0.0, 'normalized_report_sha256': report['artifact']['sha256']}
    row = {'case_id': anchor.case_id, 'triple_candidate_id': 'fixture_pair_' + _digest([image_id, report_id])[:24],
        'ehr_sha256': anchor.ehr_sha256, 'ehr_facts_sha256': anchor.ehr_facts_sha256,
        'cxr_candidate_id': image_id, 'cxr_sha256': image['artifact']['sha256'],
        'report_candidate_id': report_id, 'report_sha256': report['artifact']['sha256'],
        'cxr_model_id': 'roentgen_v2', 'report_model_id': model, 'seed': seed,
        'receipt': receipt, 'raw_edge_readouts': copy.deepcopy(receipt['raw_edge_readouts']), 'structure': structure}
    ctx = gate.context(anchor.record(), {'thresholds': il['thresholds'],
        'thresholds_sha256': H['thresholds'], 'checkpoint_sha256': H['xrv']}, {'checkpoint_sha256': H['chexbert']})
    return row, ctx


def reseal(row):
    r = row['receipt']; facts = r['fact_states']
    r['raw_edge_readouts'] = {edge: _edge(facts, left, right) for edge, left, right in
        (('ehr_cxr', 'ehr', 'xrv'), ('ehr_report', 'ehr', 'chexbert'), ('cxr_report', 'xrv', 'chexbert'))}
    known = r['raw_edge_readouts']['ehr_cxr']['known_reference_facts']
    joint = [f for f in facts if all(f[k] in gate.EXPLICIT for k in ('ehr', 'xrv', 'chexbert'))]
    r['known_ehr_facts'] = known
    r['all_three_supported_facts'] = sum(f['ehr'] == f['xrv'] == f['chexbert'] for f in joint)
    r['all_three_support_over_known'] = r['all_three_supported_facts'] / known if known else None
    r['verification_status'] = ('unverified_no_direct_ehr_constraints' if not known else
        'explicit_proxy_opposition_unvalidated' if any(e['proxy_opposition_facts'] for e in r['raw_edge_readouts'].values()) else
        'unverified_missing_direct_ehr_comparison' if len(joint) < known else 'direct_ehr_label_agreement_unvalidated')
    r['receipt_id'] = _digest({k: v for k, v in r.items() if k != 'receipt_id'})
    row['raw_edge_readouts'] = copy.deepcopy(r['raw_edge_readouts'])


def cached(row):
    return {'row': row, 'origin': {'kind': 'pinned_cached_reference', 'source_manifest_sha256': _digest('fixture_manifest')}}


def book_for(rs, ctx, *, failure=False, pending=False):
    anchor = gate.anchor_from_record(ctx['anchor'])
    ledger = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=4 * len(rs) + 2, max_retries=1, execution_mode='invented_fixture_no_models', sink=lambda e: None)
    for i, row in enumerate(rs):
        parent = None
        for kind in ('cxr_generator', 'xrv', 'report_generator', 'chexbert'):
            model = row['cxr_model_id'] if kind == 'cxr_generator' else row['report_model_id'] if kind == 'report_generator' else kind
            req = CallRequest(f'fixture_op_{i}_{kind}', anchor.case_id, anchor.sha256, kind, model,
                _digest(['fixture_audit', kind]), row['seed'], parent,
                None if kind == 'cxr_generator' else row['cxr_sha256'],
                row['report_sha256'] if kind == 'chexbert' else None)
            reservation = ledger.reserve(req)
            if failure and i == 0 and kind == 'cxr_generator':
                ledger.fail(reservation, error_code='timeout', retryable=True, elapsed_seconds=1.0)
                reservation = ledger.reserve(req)
            artifact = {'cxr_generator': row['cxr_sha256'], 'xrv': row['receipt']['xrv_labels_sha256'],
                'report_generator': row['report_sha256'], 'chexbert': row['receipt']['chexbert_labels_sha256']}[kind]
            receipt = row['receipt']['partial_receipt_id'] if kind == 'xrv' else row['receipt']['receipt_id'] if kind == 'chexbert' else None
            ledger.complete(reservation, CallResult(artifact, receipt), elapsed_seconds=.1)
            parent = req.operation_id
    if pending:
        ledger.reserve(CallRequest('fixture_pending', anchor.case_id, anchor.sha256, 'cxr_generator',
            'roentgen_v2', _digest('fixture_audit'), 2))
    return ledger.snapshot()


class FreshOutputAcceptanceTests(unittest.TestCase):
    def pair(self):
        base, ctx = fixture()
        alt, _ = fixture(report_state='positive', report_id='fixture_alt', model='maira2')
        return base, alt, ctx

    def assess(self, base, alt, ctx, *, observations=None, book=None):
        obs = [cached(base), cached(alt)] if observations is None else observations
        return gate.assess_fresh_output(base['triple_candidate_id'], alt['triple_candidate_id'], obs, ctx, ledger_snapshot=book)

    def test_strict_same_image_gain_passes_without_clinical_claim(self):
        base, alt, ctx = self.pair()
        result = self.assess(base, alt, ctx)
        self.assertEqual(result['selected_candidate_id'], alt['triple_candidate_id'])
        self.assertEqual(result['status'], 'proxy_preserving_report_change_unverified')
        self.assertFalse(result['clinical_acceptance']); self.assertFalse(result['clinical_repair_success'])
        self.assertFalse(result['model_execution_allowed']); self.assertIsNone(result['confirmed_faulty_modality'])

    def test_unchanged_is_unverified_not_success(self):
        base, ctx = fixture()
        result = gate.assess_fresh_output(base['triple_candidate_id'], base['triple_candidate_id'], [cached(base)], ctx)
        self.assertEqual(result['status'], 'unchanged_unverified')
        self.assertFalse(result['proposal_passes_proxy_preservation'])

    def test_cosmetic_change_and_duplicate_do_not_pass(self):
        base, _, ctx = self.pair()
        alt, _ = fixture(report_id='fixture_alt', model='maira2')
        self.assertFalse(self.assess(base, alt, ctx)['proposal_passes_proxy_preservation'])
        alt['structure']['normalized_report_sha256'] = base['structure']['normalized_report_sha256']
        self.assertIn('duplicate_not_expert_diversity', self.assess(base, alt, ctx)['rejection_reason_codes'])

    def test_conflict_cannot_be_silenced(self):
        base, ctx = fixture(report_state='negative')
        for state in ('unknown', 'uncertain'):
            alt, _ = fixture(report_state=state, report_id='fixture_alt', model='maira2')
            result = self.assess(base, alt, ctx)
            self.assertFalse(result['proposal_passes_proxy_preservation'])
            self.assertIn('conflict_silenced_not_corrected', result['rejection_reason_codes'])

    def test_underconditioned_cases_kept_and_ehr_rates_na(self):
        base, ctx = fixture(known=False)
        alt, _ = fixture(known=False, report_state='positive', report_id='fixture_alt', model='maira2')
        result = self.assess(base, alt, ctx)
        self.assertTrue(result['proposal_passes_proxy_preservation'])
        self.assertIsNone(result['selected_raw_edge_readouts']['ehr_cxr']['support_over_known'])

    def test_disabled_head_cannot_become_negative_even_resealed(self):
        base, alt, ctx = self.pair()
        alt['receipt']['fact_states'][-1]['xrv'] = 'negative'; reseal(alt)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_same_image_classifier_and_partial_identity_must_not_change(self):
        for key in ('xrv_labels_sha256', 'partial_receipt_id'):
            base, alt, ctx = self.pair(); alt['receipt'][key] = _digest('fixture_changed'); reseal(alt)
            with self.subTest(key=key), self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_ehr_and_profile_drift_are_errors_not_fallback(self):
        for key in ('ehr_anchor_sha256', 'thresholds_sha256', 'xrv_checkpoint_sha256', 'chexbert_checkpoint_sha256', 'profile'):
            base, alt, ctx = self.pair(); alt['receipt'][key] = _digest('fixture_changed'); reseal(alt)
            with self.subTest(key=key), self.assertRaises(ValueError): self.assess(base, alt, ctx)
        for key in ('ehr_sha256', 'ehr_facts_sha256', 'case_id'):
            base, alt, ctx = self.pair(); alt[key] = _digest('fixture_changed')
            with self.subTest(key=key), self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_state_drift_cannot_hide_behind_valid_digest(self):
        base, alt, ctx = self.pair(); alt['receipt']['fact_states'][3]['ehr'] = 'negative'; reseal(alt)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_raw_edge_counts_and_joint_counts_are_recomputed(self):
        for target in ('raw', 'joint', 'boolean'):
            base, alt, ctx = self.pair()
            if target == 'raw': alt['receipt']['raw_edge_readouts']['cxr_report']['comparable_facts'] += 1
            elif target == 'joint': alt['receipt']['all_three_supported_facts'] += 1
            else: alt['receipt']['all_three_supported_facts'] = True
            alt['receipt']['receipt_id'] = _digest({k: v for k, v in alt['receipt'].items() if k != 'receipt_id'})
            alt['raw_edge_readouts'] = copy.deepcopy(alt['receipt']['raw_edge_readouts'])
            with self.subTest(target=target), self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_runtime_failure_is_charged_not_clinical_opposition(self):
        base, alt, ctx = self.pair(); book = book_for([alt], ctx, failure=True)
        obs = [cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}]
        result = self.assess(base, alt, ctx, observations=obs, book=book)
        self.assertEqual(result['cost']['charged_model_attempts'], 5)
        self.assertEqual(result['cost']['failed_attempts'], 1)
        self.assertEqual(result['cost']['veto_refunds'], 0)

    def test_pending_attempt_keeps_cost_and_does_not_count_as_candidate(self):
        base, alt, ctx = self.pair(); book = book_for([alt], ctx, pending=True)
        result = self.assess(base, alt, ctx, observations=[cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}], book=book)
        self.assertEqual(result['cost']['charged_model_attempts'], 5)
        self.assertEqual(result['cost']['pending_attempts'], 1)
        self.assertEqual(len(result['observed_candidate_ids']), 2)

    def test_ledger_refund_and_chain_tampering_rejected(self):
        base, alt, ctx = self.pair(); book = book_for([alt], ctx)
        obs = [cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}]
        book['charged_model_attempts'] -= 1
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book)
        book = book_for([alt], ctx); book['events'][-1]['result']['verification_receipt_id'] = _digest('fixture_changed')
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book)

    def test_live_marker_without_completed_receipt_cannot_supply_observation(self):
        base, alt, ctx = self.pair()
        obs = [cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}]
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs)
        wrong, _ = fixture(report_id='fixture_wrong')
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book_for([wrong], ctx))

    def test_completed_model_seed_and_hash_bindings_required(self):
        for key, value in (('seed', 1), ('cxr_model_id', 'chexgenbench_sana')):
            base, alt, ctx = self.pair(); book = book_for([alt], ctx)
            alt[key] = value
            if key == 'cxr_model_id': alt['structure']['source_cxr_model_id'] = value
            obs = [cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}]
            with self.subTest(key=key), self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book)

    def test_hidden_proposal_and_duplicate_observation_rejected(self):
        base, alt, ctx = self.pair()
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=[cached(base)])
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=[cached(base), cached(alt), cached(alt)])
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=[cached(alt), cached(base)])

    def test_endpoint_fields_do_not_change_decision(self):
        base, alt, ctx = self.pair(); expected = self.assess(base, alt, ctx)
        for r in (base, alt):
            r.update(biovil_raw_cosine=1000, old_winner=True, ranking_score=-1000)
            r['structure']['endpoint_score'] = 1000
        self.assertEqual(expected, self.assess(base, alt, ctx))

    def test_deterministic_and_inputs_not_modified(self):
        base, alt, ctx = self.pair(); before = copy.deepcopy((base, alt, ctx))
        a = self.assess(base, alt, ctx); b = self.assess(base, alt, ctx)
        self.assertEqual(a, b); self.assertEqual((base, alt, ctx), before)

    def test_bad_sections_give_null_fallback_not_clinical_acceptance(self):
        base, ctx = fixture(); base['structure'].update(findings_complete=False, section_contract_pass=False)
        result = gate.assess_fresh_output(base['triple_candidate_id'], None, [cached(base)], ctx)
        self.assertIsNone(result['selected_candidate_id'])
        self.assertEqual(result['status'], 'unresolved_no_section_eligible_output')

    def test_missing_proposal_retains_fixed_reference(self):
        base, ctx = fixture()
        result = gate.assess_fresh_output(base['triple_candidate_id'], None, [cached(base)], ctx)
        self.assertEqual(result['selected_candidate_id'], base['triple_candidate_id'])
        self.assertEqual(result['status'], 'unresolved_missing_proposal_fixed_retained')

    def test_changed_image_requires_direct_ehr_branch_basis(self):
        base, ctx = fixture(known=False)
        alt, _ = fixture(known=False, image_id='fixture_new_image', report_id='fixture_new_report', report_state='positive', seed=1)
        result = self.assess(base, alt, ctx)
        self.assertFalse(result['proposal_passes_proxy_preservation'])
        self.assertIn('no_direct_ehr_image_branch_basis', result['rejection_reason_codes'])

    def test_valid_cross_image_improvement_passes_but_remains_unverified(self):
        base, ctx = fixture(image_state='negative', report_state='negative')
        alt, _ = fixture(image_id='fixture_new_image', report_id='fixture_new_report', report_state='positive', seed=1)
        result = self.assess(base, alt, ctx)
        self.assertTrue(result['proposal_passes_proxy_preservation'])
        self.assertEqual(result['status'], 'proxy_preserving_image_change_unverified')

    def test_cross_image_must_preserve_observed_report_only_reference(self):
        base, ctx = fixture(image_state='negative', report_state='unknown')
        static, _ = fixture(image_state='negative', report_state='unknown', report_id='fixture_static', model='maira2')
        for row in (base, static):
            row['receipt']['fact_states'][0]['xrv'] = 'positive'
        static['receipt']['fact_states'][0]['chexbert'] = 'positive'
        for row in (base, static): reseal(row)
        alt, _ = fixture(image_id='fixture_new_image', report_id='fixture_new_report', report_state='unknown', seed=1)
        obs = [cached(base), cached(static), cached(alt)]
        result = self.assess(base, alt, ctx, observations=obs)
        self.assertEqual(result['observed_reference_candidate_id'], static['triple_candidate_id'])
        self.assertEqual(len(result['comparisons']), 2)
        self.assertTrue(result['comparisons'][0]['exploratory_gate_pass'])
        self.assertFalse(result['comparisons'][1]['exploratory_gate_pass'])
        self.assertEqual(result['selected_candidate_id'], base['triple_candidate_id'])

    def test_unseen_stronger_report_cannot_be_a_reference(self):
        base, ctx = fixture(image_state='negative', report_state='unknown')
        alt, _ = fixture(image_id='fixture_new_image', report_id='fixture_new_report', report_state='unknown', seed=1)
        result = self.assess(base, alt, ctx)
        self.assertEqual(result['observed_reference_candidate_id'], base['triple_candidate_id'])
        self.assertEqual(len(result['comparisons']), 1)

    def test_cached_reference_cannot_be_inserted_after_live_completion(self):
        base, alt, ctx = self.pair(); third, _ = fixture(report_state='positive', report_id='fixture_third', model='llavarad')
        obs = [cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}, cached(third)]
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book_for([alt], ctx))

    def test_completed_observations_require_true_completion_order(self):
        base, alt, ctx = self.pair(); third, _ = fixture(report_state='positive', report_id='fixture_third', model='llavarad')
        obs = [cached(base), {'row': third, 'origin': {'kind': 'completed_ledger_receipt'}},
               {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}]
        with self.assertRaises(ValueError): self.assess(base, alt, ctx, observations=obs, book=book_for([alt, third], ctx))

    def test_clinical_flags_and_verified_status_cannot_be_forged(self):
        for key in ('clinical_acceptance', 'clinical_repair_success', 'model_execution_allowed_by_receipt'):
            base, alt, ctx = self.pair(); alt['receipt'][key] = True; reseal(alt)
            with self.subTest(key=key), self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_image_only_receipt_cannot_impersonate_completed_report(self):
        base, alt, ctx = self.pair()
        alt['receipt']['schema_version'] = 'tricompose-fresh-image-receipt-v1'
        alt['receipt']['report_lifecycle_status'] = 'not_generated'; reseal(alt)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_duplicate_text_does_not_become_new_diversity(self):
        base, alt, ctx = self.pair()
        alt['structure']['normalized_report_sha256'] = base['structure']['normalized_report_sha256']
        result = self.assess(base, alt, ctx)
        self.assertFalse(result['proposal_passes_proxy_preservation'])
        self.assertIn('duplicate_not_expert_diversity', result['rejection_reason_codes'])

    def test_structure_risk_cannot_worsen_despite_label_gain(self):
        for key, value in (('generic_report', True), ('unsupported_temporal_comparison_language', True),
                           ('repeated_sentence_count', 1), ('repeated_4gram_ratio', .1)):
            base, alt, ctx = self.pair(); alt['structure'][key] = value
            with self.subTest(key=key):
                self.assertFalse(self.assess(base, alt, ctx)['proposal_passes_proxy_preservation'])

    def test_source_byte_authentication_is_not_claimed_by_self_hash(self):
        base, alt, ctx = self.pair(); result = self.assess(base, alt, ctx)
        self.assertEqual(result['source_artifact_authentication'], 'required_upstream_not_proven_by_self_digest')
        self.assertFalse(result['installed_in_live_controller'])
        self.assertEqual(result['scorer_context_sha256'], _digest(ctx))

    def test_new_opposition_blocks_otherwise_strict_report_gain(self):
        base, alt, ctx = self.pair()
        for r in (base, alt):
            r['receipt']['fact_states'][0].update(xrv='positive', chexbert='positive')
        alt['receipt']['fact_states'][0]['chexbert'] = 'negative'
        for r in (base, alt): reseal(r)
        result = self.assess(base, alt, ctx)
        self.assertFalse(result['proposal_passes_proxy_preservation'])
        self.assertIn('new_explicit_proxy_opposition', result['rejection_reason_codes'])

    def test_reference_uses_expert_priority_not_observation_or_secondary(self):
        base, maira, ctx = self.pair()
        llava, _ = fixture(report_state='positive', report_id='fixture_llava', model='llavarad')
        image, _ = fixture(image_id='fixture_alt_image', report_id='fixture_alt_report', seed=1)
        result = self.assess(base, image, ctx, observations=[cached(base), cached(llava), cached(maira), cached(image)])
        self.assertEqual(result['observed_reference_candidate_id'], maira['triple_candidate_id'])

    def test_context_requires_eight_heads_and_source_categories(self):
        base, alt, ctx = self.pair(); ctx['enabled_xrv_findings'] = list(gate.FINDINGS)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)
        base, alt, ctx = self.pair(); ctx['anchor']['findings'][3]['source_categories'] = []
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_shared_image_and_report_hashes_cannot_hide_state_drift(self):
        base, alt, ctx = self.pair()
        alt['report_sha256'] = base['report_sha256']
        alt['receipt']['report_sha256'] = alt['report_sha256']
        alt['structure']['report_sha256'] = alt['report_sha256']; reseal(alt)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)
        base, ctx = fixture()
        alt, _ = fixture(image_id='fixture_new_image', report_id='fixture_alt', image_state='negative')
        alt['cxr_sha256'] = base['cxr_sha256']; alt['receipt']['cxr_sha256'] = base['cxr_sha256']
        alt['structure']['image_sha256'] = base['cxr_sha256']; reseal(alt)
        with self.assertRaises(ValueError): self.assess(base, alt, ctx)

    def test_veto_does_not_refund_completed_acquisition(self):
        base, ctx = fixture(report_state='positive')
        alt, _ = fixture(report_id='fixture_alt', report_state='unknown', model='maira2')
        book = book_for([alt], ctx)
        result = self.assess(base, alt, ctx, observations=[cached(base), {'row': alt, 'origin': {'kind': 'completed_ledger_receipt'}}], book=book)
        self.assertEqual(result['selected_candidate_id'], base['triple_candidate_id'])
        self.assertEqual(result['cost']['charged_model_attempts'], 4)
        self.assertEqual(result['cost']['ledger_snapshot_sha256'], _digest(book))

    def test_cpu_guard_precedes_metadata_load_or_creation(self):
        with patch('fresh_output_acceptance.cpu_guard', side_effect=RuntimeError('fixture_guard')), \
             patch('fresh_output_acceptance.load_archived_smoke') as load:
            with self.assertRaises(RuntimeError): gate.run_smoke('fixture_output', 'fixture_run')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
