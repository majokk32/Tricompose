"""Invented numeric controls; no prompts, pixels, patient data or inference."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_conditioning_attribution_v1 as m
from test_image_attribution_stress_v1 import packet as old_packet
from test_prompt_image_conditioning_v1 import fixture, digest


def packet(margin=.2, best=1, worst=1, report='negative'):
    value = old_packet(report=report)
    value['conditioning'] = {'status': 'scored_complete_inventory', 'matched_cosine': .5,
        'matched_minus_other_mean': margin, 'best_rank': best, 'worst_rank': worst, 'prompt_groups': 10}
    return value


def bound_fixture():
    rows = fixture()
    images = m.conditioning.images_from_index(rows, full=False)
    frames, tokens, bank, pairs = [], [], {}, []
    for image in images:
        token = digest(image['case_id'])
        frame = {**image, 'prompt_group_id': 'tokens_' + token}
        frames.append(frame)
        tokens.append({'prompt_sha256': image['prompt_sha256'], 'status': 'encoded', 'token_sequence_sha256': token})
        bank.setdefault(image['case_id'], {})[image['cxr_candidate_id']] = {
            'case_id': image['case_id'], 'lineage': {**image, 'cxr_seed': 0}}
    for image in images:
        for case in range(2):
            own = image['case_id'] == f'case_fixture_{case}'
            pairs.append({**image, 'prompt_group_id': 'tokens_' + digest(f'case_fixture_{case}'),
                'raw_cosine': '0.8' if own else '-0.2',
                'clinical_negative_validated': 'False', 'is_intended_encoder_prompt_group': str(own)})
    return frames, pairs, {'records': tokens}, bank


def bind_fixture():
    frames, pairs, tokens, bank = bound_fixture()
    with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
        return m.bind_conditioning(frames, pairs, tokens, bank, full=False)


class ConditioningAttributionTests(unittest.TestCase):
    def test_joint_high_conditioning_opposition_gives_only_report_hint(self):
        out = m.predict(packet(), 'agree_fixed_mean', 'joint_margin0')
        self.assertEqual(out['tentative_target'], 'report')
        self.assertIsNone(out['confirmed_faulty_modality'])
        self.assertIsNone(out['clinical_localization_accuracy'])
        self.assertFalse(out['regeneration_authorized'])
        self.assertFalse(out['clinical_qualified'])

    def test_joint_low_conditioning_opposition_gives_only_cxr_hint(self):
        out = m.predict(packet(margin=-.2), 'agree_fixed_mean', 'joint_margin0')
        self.assertEqual(out['tentative_target'], 'cxr')
        self.assertFalse(out['prompt_matching_is_finding_truth'])
        self.assertFalse(out['same_biovil_evidence_is_independent'])
        self.assertEqual(out['ehr_opacity_state'], 'unknown')

    def test_support_does_not_become_clean_classification(self):
        out = m.predict(packet(report='positive'), 'agree_fixed_mean', 'joint_margin0')
        self.assertIsNone(out['tentative_target'])
        self.assertIsNone(out['confirmed_faulty_modality'])

    def test_unknown_uncertain_report_blocks_joint_hint(self):
        for state in ('unknown', 'uncertain'):
            out = m.predict(packet(report=state, margin=-.2), 'agree_fixed_mean', 'joint_margin0')
            self.assertIsNone(out['tentative_target'])

    def test_conditioning_only_is_a_separate_image_control(self):
        value = packet(report='unknown', margin=-.2)
        out = m.predict(value, 'agree_fixed_mean', 'conditioning_margin_only')
        self.assertEqual(out['tentative_target'], 'cxr')
        value['image_evidence']['xrv_score'] = None
        out2 = m.predict(value, 'biovil_fixed_mean', 'conditioning_margin_only')
        self.assertEqual(out2['tentative_target'], 'cxr')

    def test_conditioning_only_high_cannot_blame_report(self):
        self.assertIsNone(m.predict(packet(), 'agree_fixed_mean', 'conditioning_margin_only')['tentative_target'])

    def test_naive_control_ignores_original_conditioning(self):
        for margin in (-.2, .2, 0):
            out = m.predict(packet(margin=margin), 'agree_fixed_mean', 'report_blame_on_opposition')
            self.assertEqual(out['tentative_target'], 'report')

    def test_exact_zero_margin_abstains(self):
        self.assertIsNone(m.predict(packet(margin=0), 'agree_fixed_mean', 'joint_margin0')['tentative_target'])

    def test_rank1_tie_is_not_high_confidence(self):
        out = m.predict(packet(best=1, worst=2), 'agree_fixed_mean', 'joint_rank1')
        self.assertEqual(out['conditioning_preference'], 'boundary_tie')
        self.assertIsNone(out['tentative_target'])

    def test_rank5_boundary_tie_abstains(self):
        out = m.predict(packet(best=5, worst=7), 'agree_fixed_mean', 'joint_rank5')
        self.assertEqual(out['conditioning_preference'], 'boundary_tie')
        self.assertIsNone(out['tentative_target'])

    def test_all_rank_rules_fixed_and_visible(self):
        value = packet(best=3, worst=3)
        self.assertEqual(m.predict(value, 'agree_fixed_mean', 'joint_rank1')['tentative_target'], 'cxr')
        self.assertEqual(m.predict(value, 'agree_fixed_mean', 'joint_rank5')['tentative_target'], 'report')

    def test_image_reader_disagreement_blocks_joint_hint(self):
        value = packet(); value['image_evidence']['xrv_score'] = .1
        self.assertIsNone(m.predict(value, 'agree_fixed_mean', 'joint_margin0')['tentative_target'])

    def test_missing_conditioning_is_not_negative(self):
        value = packet()
        value['conditioning'].update(status='incomplete_prompt_inventory_no_rank',
            matched_minus_other_mean=None, best_rank=None, worst_rank=None)
        for method in m.METHODS[1:]:
            self.assertIsNone(m.predict(value, 'agree_fixed_mean', method)['tentative_target'])

    def test_identity_and_key_cannot_enter_predictor(self):
        for key in ('case_id', 'donor_id', 'artifact_sha256', 'intervention_target', 'report_text', 'prompt_text'):
            value = packet(); value[key] = 'forbidden'
            with self.assertRaises(ValueError): m.predict(value, 'agree_fixed_mean', 'joint_margin0')

    def test_identity_never_changes_prediction_other_than_echo(self):
        a = packet(); z = deepcopy(a); z['item_id'] = 'item_0222'
        left = m.predict(a, 'agree_fixed_mean', 'joint_margin0')
        right = m.predict(z, 'agree_fixed_mean', 'joint_margin0')
        left.pop('item_id'); right.pop('item_id')
        self.assertEqual(left, right)

    def test_known_ehr_opacity_cannot_be_manufactured(self):
        value = packet(); value['ehr_opacity_state'] = 'positive'
        with self.assertRaises(ValueError): m.predict(value, 'agree_fixed_mean', 'joint_margin0')

    def test_invalid_numeric_or_missingness_rejected(self):
        for v in (True, '0.1', float('nan'), float('inf'), 3):
            value = packet(); value['conditioning']['matched_minus_other_mean'] = v
            with self.assertRaises(ValueError): m.validate_packet(value)
        value = packet(); value['conditioning']['best_rank'] = 0
        with self.assertRaises(ValueError): m.validate_packet(value)

    def test_protocol_has_no_repair_or_threshold_fit_authority(self):
        p = m.json.loads(m.PROTOCOL.read_text()); m.validate_protocol(p)
        for key in ('clinical_qualified', 'regeneration_authorized', 'threshold_fitting', 'method_selection_after_results'):
            changed = deepcopy(p); changed[key] = True
            with self.assertRaises(ValueError): m.validate_protocol(changed)

    def test_donor_image_uses_recipient_not_donor_prompt_score(self):
        frames, anchors, scores = bind_fixture()
        donor = next(f for f in frames.values() if f['case_id'] == 'case_fixture_1')
        value, original = m.recipient_conditioning('case_fixture_0', donor['cxr_candidate_id'], frames, anchors, scores)
        self.assertEqual(original['case_id'], 'case_fixture_0')
        self.assertEqual(value['matched_cosine'], -.2)
        self.assertAlmostEqual(value['matched_minus_other_mean'], -1)
        self.assertEqual(value['best_rank'], 2)
        own, _ = m.recipient_conditioning('case_fixture_1', donor['cxr_candidate_id'], frames, anchors, scores)
        self.assertEqual(own['matched_cosine'], .8)

    def test_original_pair_matrix_covers_all_images_groups(self):
        frames, anchors, scores = bind_fixture()
        self.assertEqual((len(frames), len(anchors), len(scores)), (6, 6, 6))
        self.assertTrue(all(len(v) == 2 for v in scores.values()))

    def test_matrix_missing_duplicate_or_lineage_change_rejected(self):
        for kind in ('missing', 'duplicate', 'lineage'):
            frames, pairs, tokens, bank = bound_fixture()
            if kind == 'missing': pairs.pop()
            elif kind == 'duplicate': pairs.append(deepcopy(pairs[0]))
            else: pairs[0]['ehr_sha256'] = digest('wrong')
            with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
                with self.assertRaises(ValueError): m.bind_conditioning(frames, pairs, tokens, bank, full=False)

    def test_token_group_and_fixed_ehr_lineage_checked(self):
        for key in ('prompt_group_id', 'ehr_sha256'):
            frames, pairs, tokens, bank = bound_fixture()
            frames[0][key] = digest('wrong')
            with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
                with self.assertRaises(ValueError): m.bind_conditioning(frames, pairs, tokens, bank, full=False)

    def test_cross_generator_group_never_used(self):
        frames, pairs, tokens, bank = bound_fixture()
        pairs[0]['prompt_group_id'] = 'tokens_' + digest('alien_group')
        with patch.object(m.b.policy_module, 'snapshot', side_effect=deepcopy):
            with self.assertRaises(ValueError): m.bind_conditioning(frames, pairs, tokens, bank, full=False)

    def test_abstentions_not_counted_as_correct_controls(self):
        a = m.predict(packet(report='positive'), 'agree_fixed_mean', 'joint_margin0')
        z = m.predict(packet(margin=-.2), 'agree_fixed_mean', 'joint_margin0'); z['item_id'] = 'item_0001'
        key = [{'item_id': 'item_0000', 'intervention_type': 'no_corruption', 'intervention_target': 'none'},
               {'item_id': 'item_0001', 'intervention_type': 'cxr_swap', 'intervention_target': 'cxr'}]
        resolver = [{'item_id': r['item_id'], 'benchmark_split': 'development'} for r in key]
        table = m.summarize([a, z], resolver, key)
        control = next(r for r in table if r['source_role'] == 'overall' and r['mechanical_arm'] == 'no_corruption')
        self.assertEqual(control['abstentions_not_clean_judgments'], 1)
        self.assertIsNone(control['target_recovery_over_all_available_intervened'])
        self.assertIsNone(control['conditional_target_match'])
        self.assertIsNone(control['clinical_false_repair_rate'])

    def test_unsafe_target_on_control_not_clinical_false_positive(self):
        out = m.predict(packet(), 'agree_fixed_mean', 'joint_margin0')
        key = [{'item_id': out['item_id'], 'intervention_type': 'no_corruption', 'intervention_target': 'none'}]
        rows = m.summarize([out], [{'item_id': out['item_id'], 'benchmark_split': 'development'}], key)
        for row in rows:
            self.assertEqual(row['disagrees_with_known_replacement_target'], 1)
            self.assertIsNone(row['clinical_false_repair_rate'])

    def test_guard_precedes_source_reads(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('slurm_required')), \
                patch.object(m.b, 'load_primary', side_effect=AssertionError('read too soon')):
            with self.assertRaises(RuntimeError): m.run(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
