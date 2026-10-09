"""Wholly invented lineage/numeric fixtures; no clinical bodies or inference."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import score_finding_matched_biovil_v1 as m


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(count=2):
    rows, facts = [], []
    for number in range(count):
        for model in m.lineage.MODELS:
            image = f'fixture_image_{number}_{model}'
            for expert in sorted(m.lineage.EXPERTS):
                report = f'fixture_report_{number}_{model}_{expert}'
                row = {'case_id': f'fixture_case_{number}', 'cxr_candidate_id': image,
                    'cxr_model_id': model, 'cxr_seed': '0', 'report_model_id': expert,
                    'report_candidate_id': report, 'triple_candidate_id': 'triple_' + report,
                    'cxr_sha256': sha(image), 'report_sha256': sha(report),
                    'ehr_sha256': sha(f'ehr_{number}'), 'ehr_facts_sha256': sha(f'facts_{number}'),
                    'cxr_path': str(m.PROTECTED_ROOT / f'invented/{image}.png'),
                    'prompt_path': str(m.PROTECTED_ROOT / f'invented/{image}.txt'),
                    'prompt_sha256': sha(f'prompt_{number}_{model}'), 'old_score': '0.12345'}
                rows.append(row)
                for finding in m.FINDINGS:
                    facts.append({**{k: row[k] for k in m.FACT_FIELDS}, 'finding': finding,
                        'states': {'ehr': 'unknown', 'xrv': 'unknown' if finding == 'support_devices' else 'positive',
                                   'chexbert': 'negative'}, 'ehr_source_categories': [],
                        'report_assertion_scope_qualified': False})
    images = [{k: r[k] for k in m.IMAGE_FIELDS}
              for r in m.lineage.images_from_index(rows, full=False)]
    outcomes = [{**image, 'status': 'scored', 'score_pairs': {
        finding: {family: {'positive_cosine': .7, 'negative_cosine': .2}
                  for family in m.FAMILIES} for finding in m.FINDINGS}} for image in images]
    return rows, images, facts, outcomes


def build(values):
    return m.build_tables(*values, full=False)


class FindingMatchedTests(unittest.TestCase):
    def test_protocol_is_fixed_and_unqualified(self):
        p = m.protocol()
        self.assertEqual(p['findings'], list(m.FINDINGS))
        for key in m.FALSE_FLAGS:
            self.assertIs(p[key], False)

    def test_authored_templates_do_not_use_case_specific_information(self):
        probes = m.catalog()
        self.assertEqual(len(probes), 18)
        self.assertEqual(len({(p['finding'], p['family']) for p in probes}), 18)
        for p in probes:
            self.assertIn(p['finding'].replace('_', ' '), p['positive_text'].lower())
            self.assertNotEqual(p['positive_text'], p['negative_text'])
            for forbidden in ('left', 'right', 'severe', 'prior', 'patient', 'portable', 'frontal'):
                self.assertNotIn(forbidden, p['positive_text'].lower())

    def test_no_opacity_substituted_for_other_disease(self):
        self.assertNotIn('lung_opacity', m.FINDINGS)
        self.assertNotIn('infiltration', m.FINDINGS)
        self.assertEqual(len(m.FINDINGS), 6)

    def test_full_inventory_is_80_by_3_by_4_by_6(self):
        rows, images, facts, outcomes = fixture(80)
        m.validate_inventory(rows, images, facts)
        image_rows, candidate_rows, summary = m.build_tables(rows, images, facts, outcomes)
        self.assertEqual((len(image_rows), len(candidate_rows)), (1440, 5760))
        self.assertEqual((summary['fixed_ehr_cases'], summary['candidate_triples']), (80, 960))

    def test_metadata_alignment_does_not_read_any_referenced_body(self):
        values = fixture()
        with patch.object(Path, 'read_text', side_effect=AssertionError('no body read')):
            build(values)

    def test_original_rows_facts_and_outcomes_unchanged(self):
        values = fixture(); before = deepcopy(values)
        build(values)
        self.assertEqual(values, before)

    def test_unknown_ehr_is_not_negative(self):
        _, rows, summary = build(fixture())
        self.assertTrue(all(r['ehr_state'] == 'unknown' for r in rows))
        self.assertTrue(all(r['ehr_biovil_relation'] == 'not_comparable' for r in rows))
        self.assertTrue(all(r['joint_three_modality_comparable_candidate_rows'] == 0 for r in summary['by_finding']))

    def test_device_xrv_head_stays_unavailable_despite_biovil_signal(self):
        _, rows, _ = build(fixture())
        devices = [r for r in rows if r['finding'] == 'support_devices']
        self.assertTrue(all(r['preference_state'] == 'positive' for r in devices))
        self.assertTrue(all(r['cached_xrv_state'] == 'unknown' and not r['xrv_head_available'] for r in devices))
        self.assertTrue(all(r['mean_agreement_image_state'] == 'unknown' for r in devices))

    def test_manufactured_device_head_refused(self):
        values = fixture(); values[2][5]['states']['xrv'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'device_head'):
            build(values)

    def test_missing_finding_does_not_silently_shrink_denominator(self):
        values = fixture(); values[2].pop()
        with self.assertRaisesRegex(ValueError, 'complete_six'):
            build(values)

    def test_unknown_other_field_cannot_enter_fact_projection(self):
        values = fixture(); values[2][0]['targeted_finding'] = 'edema'
        with self.assertRaisesRegex(ValueError, 'named_fact'):
            build(values)

    def test_explicit_ehr_requires_source_categories(self):
        values = fixture(); values[2][0]['states']['ehr'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'cached_source'):
            build(values)

    def test_fixed_ehr_may_not_change_between_reports(self):
        values = fixture(); values[2][6]['states']['ehr'] = 'positive'
        values[2][6]['ehr_source_categories'] = ['diagnosis']
        with self.assertRaisesRegex(ValueError, 'fixed_ehr'):
            build(values)

    def test_unqualified_report_cannot_be_marked_clinical(self):
        values = fixture(); values[2][0]['report_assertion_scope_qualified'] = True
        with self.assertRaisesRegex(ValueError, 'named_fact'):
            build(values)

    def test_shared_report_state_may_not_vary(self):
        values = fixture()
        first, second = values[0][0], values[0][1]
        second['report_sha256'] = first['report_sha256']
        for fact in values[2][6:12]:
            fact['report_sha256'] = first['report_sha256']
        values[2][6]['states']['chexbert'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'shared_artifact'):
            build(values)

    def test_image_outcome_missing_not_discarded(self):
        values = fixture(); values[3].pop()
        with self.assertRaisesRegex(ValueError, 'every_image'):
            build(values)

    def test_duplicate_outcome_refused(self):
        values = fixture(); values[3].append(deepcopy(values[3][0]))
        with self.assertRaisesRegex(ValueError, 'exact_image'):
            build(values)

    def test_changed_image_hash_refused(self):
        values = fixture(); values[3][0]['cxr_sha256'] = sha('different')
        with self.assertRaisesRegex(ValueError, 'exact_image'):
            build(values)

    def test_failed_image_keeps_all_six_unknown_readouts(self):
        values = fixture(); values[3][0].update(status='failed_without_replacement', score_pairs=None)
        image_rows, rows, summary = build(values)
        missing = [r for r in image_rows if r['cxr_candidate_id'] == values[3][0]['cxr_candidate_id']]
        self.assertEqual(len(missing), 6)
        self.assertTrue(all(r['mean_margin'] is None and r['preference_state'] == 'unknown' for r in missing))
        self.assertEqual(summary['failed_image_slots'], 1)
        self.assertEqual(len(rows), len(values[2]))

    def test_failed_outcome_cannot_have_scores(self):
        values = fixture(); values[3][0]['status'] = 'failed_without_replacement'
        with self.assertRaisesRegex(ValueError, 'exact_image'):
            build(values)

    def test_omitting_disease_score_refused(self):
        values = fixture(); values[3][0]['score_pairs'].pop('edema')
        with self.assertRaisesRegex(ValueError, 'all_six'):
            build(values)

    def test_incomplete_template_family_refused(self):
        values = fixture(); values[3][0]['score_pairs']['edema'].pop('shows_no')
        with self.assertRaisesRegex(ValueError, 'all_three'):
            build(values)

    def test_nonfinite_bool_and_out_of_range_scores_refused(self):
        for number in (math.nan, math.inf, True, 1.02, '0.5'):
            values = fixture(); values[3][0]['score_pairs']['edema']['shows_no']['positive_cosine'] = number
            with self.assertRaisesRegex(ValueError, 'cosine_pair'):
                build(values)

    def test_exact_tie_is_unknown(self):
        pairs = {f: {'positive_cosine': .4, 'negative_cosine': .4} for f in m.FAMILIES}
        result = m.score_readout(pairs)
        self.assertEqual(result['preference_state'], 'unknown')
        self.assertEqual(result['template_pattern'], 'all_tied')

    def test_mixed_template_signs_not_stable_agreement(self):
        values = fixture()
        values[3][0]['score_pairs']['edema']['shows_no'] = {'positive_cosine': .1, 'negative_cosine': .5}
        _, rows, _ = build(values)
        row = next(r for r in rows if r['cxr_candidate_id'] == values[3][0]['cxr_candidate_id'] and r['finding'] == 'edema')
        self.assertEqual(row['preference_state'], 'positive')
        self.assertEqual(row['mean_agreement_image_state'], 'positive')
        self.assertEqual(row['stable_agreement_image_state'], 'unknown')

    def test_score_reduction_invariant_to_dictionary_order(self):
        pairs = {family: {'positive_cosine': (i+1)/10, 'negative_cosine': .17} for i, family in enumerate(m.FAMILIES)}
        self.assertEqual(m.score_readout(pairs), m.score_readout(dict(reversed(list(pairs.items())))))

    def test_reader_opposition_is_not_fault_attribution(self):
        values = fixture()
        for fact in values[2]:
            if fact['finding'] != 'support_devices':
                fact['states']['xrv'] = 'negative'
        _, rows, summary = build(values)
        self.assertTrue(all(r['mean_agreement_image_state'] == 'unknown' for r in rows))
        self.assertTrue(all(r['confirmed_faulty_modality'] is None and not r['regeneration_authorized'] for r in rows))
        self.assertIsNone(summary['clinical_fault_localization_accuracy'])

    def test_report_opposition_is_not_clinical_error(self):
        _, rows, summary = build(fixture())
        self.assertTrue(all(r['report_biovil_relation'] == 'proxy_opposition' for r in rows))
        self.assertTrue(all(r['clinical_conflict_verified'] is None for r in rows))
        self.assertIs(summary['clinical_qualified'], False)

    def test_image_slots_not_report_votes(self):
        _, _, summary = build(fixture())
        for finding in summary['by_finding']:
            self.assertEqual(finding['image_slots'], 6)
            self.assertEqual(finding['candidate_triples'], 24)
            self.assertEqual(sum(finding['ehr_states_one_per_case'].values()), 2)

    def test_cpu_plan_projection_drops_generation_prompt_and_report_paths(self):
        _, images, _, _ = fixture()
        self.assertTrue(all(set(r) == set(m.IMAGE_FIELDS) for r in images))
        self.assertTrue(all('prompt_path' not in r and 'report_path' not in r for r in images))

    def test_gpu_guard_runs_before_plan_or_model_read(self):
        with patch.object(m.lineage.reuse, 'guard', side_effect=RuntimeError('no approval')) as guard:
            with patch.object(Path, 'read_text', side_effect=AssertionError('plan read')):
                with self.assertRaisesRegex(RuntimeError, 'no approval'):
                    m.evaluate(SimpleNamespace(allow_synthetic_finding_images=False))
            guard.assert_called_once_with(gpu=True, approved=False)

    def test_cpu_guard_runs_before_source_plan_reads(self):
        with patch.object(m.cached, 'cpu_guard', side_effect=RuntimeError('not cpu slurm')):
            with patch.object(m, 'protocol', side_effect=AssertionError('protocol read')):
                with self.assertRaisesRegex(RuntimeError, 'not cpu slurm'):
                    m.prepare(SimpleNamespace())

    def test_forbidden_patient_or_target_fields_do_not_enter_image_scores(self):
        values = fixture(); values[3][0]['target_state'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'exact_image'):
            build(values)

    def test_uncertain_state_preserved_and_not_comparable(self):
        values = fixture()
        for fact in values[2]:
            if fact['finding'] == 'edema':
                fact['states']['ehr'] = 'uncertain'
                fact['states']['chexbert'] = 'uncertain'
                fact['ehr_source_categories'] = ['diagnosis']
        _, rows, _ = build(values)
        for row in rows:
            if row['finding'] == 'edema':
                self.assertEqual(row['ehr_state'], 'uncertain')
                self.assertEqual(row['cached_report_state'], 'uncertain')
                self.assertEqual(row['ehr_biovil_relation'], 'not_comparable')

    def test_explicit_supported_facts_not_replaced_by_new_evidence(self):
        values = fixture()
        for fact in values[2]:
            if fact['finding'] == 'edema':
                fact['states']['ehr'] = 'positive'
                fact['ehr_source_categories'] = ['diagnosis']
                fact['states']['chexbert'] = 'positive'
        _, rows, summary = build(values)
        for row in rows:
            if row['finding'] == 'edema':
                self.assertEqual(row['ehr_biovil_relation'], 'proxy_support')
                self.assertEqual(row['cached_report_state'], 'positive')
        self.assertEqual(summary['by_finding'][0]['joint_three_modality_comparable_candidate_rows'], 24)
        self.assertIsNone(summary['clinical_repair_success'])

    def test_same_image_bytes_cannot_receive_different_scores(self):
        values = fixture()
        first_id, second_id = values[1][0]['cxr_candidate_id'], values[1][1]['cxr_candidate_id']
        first_hash = values[1][0]['cxr_sha256']
        for collection in values:
            for row in collection:
                if row.get('cxr_candidate_id') == second_id:
                    row['cxr_sha256'] = first_hash
        values[3][1]['score_pairs']['edema']['shows_no']['positive_cosine'] = .8
        with self.assertRaisesRegex(ValueError, 'same_image_hash'):
            build(values)

    def test_missing_source_category_not_replaced_by_template(self):
        values = fixture()
        values[2][0]['ehr_source_categories'] = ['invented_source']
        with self.assertRaisesRegex(ValueError, 'named_fact'):
            build(values)

    def test_fact_projection_enforces_model_lineage_without_body_reads(self):
        rows, _, facts, _ = fixture()
        observations = {}
        bank = {'fixture': {}}
        for row in rows:
            subset = [f for f in facts if f['triple_candidate_id'] == row['triple_candidate_id']]
            obs = {'case_id': row['case_id'],
                'lineage': {k: row[k] for k in (*m.HASH_FIELDS, 'cxr_candidate_id', 'report_candidate_id', 'report_model_id', 'cxr_model_id')},
                'states': {name: {f['finding']: f['states'][name] for f in subset} for name in ('ehr', 'xrv', 'chexbert')},
                'ehr_sources': {f['finding']: f['ehr_source_categories'] for f in subset}}
            obs['lineage']['cxr_seed'] = 0
            observations[row['triple_candidate_id']] = obs
            bank['fixture'][row['triple_candidate_id']] = {'score_record': {'triple_candidate_id': row['triple_candidate_id']}}
        with patch.object(m.cached.policy_module, 'snapshot', side_effect=lambda c: observations[c['score_record']['triple_candidate_id']]):
            with patch.object(Path, 'read_text', side_effect=AssertionError('no clinical body read')):
                self.assertEqual(m.project_facts(rows, bank), facts)
                observations[rows[0]['triple_candidate_id']]['lineage']['cxr_model_id'] = 'wrong_model'
                with self.assertRaisesRegex(ValueError, 'original_candidate_lineage'):
                    m.project_facts(rows, bank)

    def test_readiness_explicitly_pending_not_fake_completed_scores(self):
        rows, images, facts, _ = fixture(80)
        summary = m.cpu_plan_summary(rows, images, facts)
        self.assertEqual(summary['scoring_status'], 'pending_separate_gpu_approval')
        self.assertEqual(summary['new_model_calls'], 0)
        self.assertFalse(summary['bodies_pixels_or_weights_read'])
        self.assertEqual((summary['planned_image_finding_rows'], summary['planned_candidate_finding_rows']), (1440, 5760))


if __name__ == '__main__':
    unittest.main()
