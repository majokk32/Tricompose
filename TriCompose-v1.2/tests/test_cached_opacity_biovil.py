"""Wholly invented metadata/cosines; no model, image or patient artifacts."""
import copy
import csv
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('cached_biovil_fixture', ROOT / 'tools/score_cached_opacity_biovil.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(reports=('positive','negative','uncertain','unknown'), *, mean_positive=True, exact='positive'):
    image = {'case_id':'case_fixture', 'cxr_candidate_id':'image_fixture', 'cxr_model_id':'model_fixture',
             'cxr_sha256':'a'*64, 'ehr_sha256':'b'*64, 'ehr_facts_sha256':'c'*64}
    rows=[]
    for i,state in enumerate(reports):
        row={k:v for k,v in image.items() if k != 'cxr_model_id'}
        row.update(triple_candidate_id=f'triple_{i}',report_model_id=f'expert_{i}',
            report_sha256=f'{i:064x}',opacity_cached_ehr_state='unknown',opacity_cached_report_state=state,
            opacity_exact_state_0_5=exact,opacity_clinical_accuracy='',opacity_confirmed_faulty_modality='',
            opacity_selector_used='False')
        for j in range(66-len(row)):
            row[f'original_{j}']=f'fixture_value_{j}'
        rows.append(row)
    positive,negative=(.7,.1) if mean_positive else (.1,.7)
    outcome={k:image[k] for k in ('case_id','cxr_candidate_id','cxr_model_id','cxr_sha256')}
    outcome.update(status='scored',image_calls=1,score_pairs={f:{'positive_cosine':positive,'negative_cosine':negative}
        for f in m.frozen.reuse.FAMILIES})
    return rows,[image],[outcome]


class CachedBioViLTests(unittest.TestCase):
    def test_all_original_66_columns_and_values_preserved_and20_appended(self):
        inputs=fixture(); before=copy.deepcopy(inputs)
        out=m.overlay(*inputs)
        self.assertEqual(inputs,before)
        self.assertEqual([r['triple_candidate_id'] for r in out],[r['triple_candidate_id'] for r in inputs[0]])
        for old,new in zip(inputs[0],out):
            self.assertEqual(len(old),66); self.assertEqual(len(new),86)
            self.assertEqual({k:new[k] for k in old},old)

    def test_csv_roundtrip_preserves_all_original_cells(self):
        rows,images,outcomes=fixture()
        output=list(csv.DictReader(io.StringIO(m.opacity.csv_text(m.overlay(rows,images,outcomes)))))
        for old,new in zip(rows,output):
            self.assertEqual({k:new[k] for k in old},old)

    def test_one_image_score_shared_by_all_four_reports(self):
        inputs=fixture(); out=m.overlay(*inputs)
        values=[r[m.PREFIX+'mean_margin'] for r in out]
        self.assertEqual(values,[values[0]]*4)
        summary=m.aggregate(out,inputs[2])
        self.assertEqual((summary['image_slots'],summary['candidate_rows']),(1,4))

    def test_unknown_uncertain_reports_not_negative_or_agreement(self):
        out=m.overlay(*fixture())
        self.assertEqual([r[m.PREFIX+'report_proxy_relation'] for r in out],
            ['proxy_support','proxy_opposition','not_comparable','not_comparable'])
        self.assertEqual(out[2][m.PREFIX+'joint_proxy_pattern'],'report_evidence_unavailable')

    def test_zero_margin_preference_is_unknown(self):
        rows,images,outs=fixture()
        for pair in outs[0]['score_pairs'].values():
            pair['positive_cosine']=pair['negative_cosine']=.3
        output=m.overlay(rows,images,outs)
        self.assertTrue(all(r[m.PREFIX+'preference_state']=='unknown' for r in output))
        self.assertTrue(all(r[m.PREFIX+'template_pattern']=='all_tied' for r in output))
        self.assertTrue(all(r[m.PREFIX+'report_proxy_relation']=='not_comparable' for r in output))

    def test_negative_preference_not_always_positive(self):
        out=m.overlay(*fixture(mean_positive=False,exact='negative'))
        self.assertEqual(out[0][m.PREFIX+'preference_state'],'negative')
        self.assertEqual(out[0][m.PREFIX+'joint_proxy_pattern'],'report_proposal_opposes_two_image_sources')

    def test_all_proxy_agreement_is_not_clinical_accuracy(self):
        out=m.overlay(*fixture(reports=('positive',)))
        self.assertEqual(out[0][m.PREFIX+'joint_proxy_pattern'],'three_proxy_sources_agree')
        self.assertIsNone(out[0][m.PREFIX+'clinical_accuracy'])
        self.assertFalse(out[0][m.PREFIX+'selector_used'])

    def test_report_opposition_does_not_assign_faulty_modality(self):
        out=m.overlay(*fixture(reports=('negative',)))
        self.assertEqual(out[0][m.PREFIX+'joint_proxy_pattern'],'report_proposal_opposes_two_image_sources')
        self.assertEqual(out[0]['opacity_confirmed_faulty_modality'],'')

    def test_image_sources_disagree_even_if_report_missing(self):
        out=m.overlay(*fixture(reports=('unknown',),exact='negative'))
        self.assertEqual(out[0][m.PREFIX+'joint_proxy_pattern'],'image_sources_disagree')
        self.assertEqual(out[0][m.PREFIX+'report_proxy_relation'],'not_comparable')

    def test_missing_xrv_kept_distinct_from_opposition(self):
        out=m.overlay(*fixture(exact='unknown'))
        self.assertTrue(all(r[m.PREFIX+'exact_xrv_proxy_relation']=='not_comparable' for r in out))
        self.assertTrue(all(r[m.PREFIX+'joint_proxy_pattern']=='image_evidence_unavailable' for r in out))

    def test_failed_image_keeps_all_reports_null_new_scores_unknown(self):
        rows,images,outs=fixture()
        outs[0].update(status='failed_without_replacement',score_pairs=None)
        out=m.overlay(rows,images,outs)
        self.assertEqual(len(out),4)
        self.assertTrue(all(r[m.PREFIX+'mean_margin'] is None for r in out))
        self.assertTrue(all(r[m.PREFIX+'preference_state']=='unknown' for r in out))
        self.assertTrue(all(r[m.PREFIX+'template_pattern']=='unavailable' for r in out))
        summary=m.aggregate(out,outs)
        self.assertEqual((summary['failed_images'],summary['candidate_rows']),(1,4))
        self.assertIsNone(summary['biovil_report_proxy_relations']['agreement_over_comparable'])

    def test_failed_outcome_cannot_smuggle_scores(self):
        rows,images,outs=fixture(); outs[0]['status']='failed_without_replacement'
        with self.assertRaisesRegex(ValueError,'explicit_scored_or_failed'):
            m.overlay(rows,images,outs)

    def test_scored_outcome_cannot_hide_missing_scores(self):
        rows,images,outs=fixture(); outs[0]['score_pairs']=None
        with self.assertRaisesRegex(ValueError,'explicit_scored_or_failed'):
            m.overlay(rows,images,outs)

    def test_mean_all_templates_not_best_template(self):
        rows,images,outs=fixture()
        outs[0]['score_pairs']['shows_no']={'positive_cosine':.1,'negative_cosine':.9}
        out=m.overlay(rows,images,outs)
        self.assertAlmostEqual(out[0][m.PREFIX+'mean_margin'],(-.8+.6+.6)/3)
        self.assertEqual(out[0][m.PREFIX+'template_pattern'],'mixed_or_tied')
        self.assertAlmostEqual(out[0][m.PREFIX+'template_min_margin'],-.8)

    def test_all_negative_template_pattern(self):
        out=m.overlay(*fixture(mean_positive=False))
        self.assertEqual(out[0][m.PREFIX+'template_pattern'],'all_negative_preference')

    def test_nonfinite_or_invalid_margin_rejected(self):
        for value in (True,math.nan,math.inf,3):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite_margin_required'):
                m.preference(value)

    def test_nonfinite_or_out_of_range_cosines_rejected(self):
        for value in (True,math.nan,math.inf,2):
            rows,images,outs=fixture(); outs[0]['score_pairs']['shows_no']['positive_cosine']=value
            with self.assertRaises(ValueError):
                m.overlay(rows,images,outs)

    def test_missing_template_inventory_refused(self):
        rows,images,outs=fixture(); outs[0]['score_pairs'].pop('shows_no')
        with self.assertRaisesRegex(ValueError,'invalid_probe_inventory'):
            m.overlay(rows,images,outs)

    def test_ehr_unknown_cannot_be_enriched_from_image(self):
        rows,images,outs=fixture(); rows[0]['opacity_cached_ehr_state']='positive'
        with self.assertRaisesRegex(ValueError,'original_unqualified_states_required'):
            m.overlay(rows,images,outs)

    def test_fixed_ehr_hash_changes_refused(self):
        rows,images,outs=fixture(); rows[0]['ehr_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'same_fixed_ehr_image_lineage'):
            m.overlay(rows,images,outs)

    def test_new_score_hash_changes_refused(self):
        rows,images,outs=fixture(); outs[0]['cxr_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'scored_image_lineage_changed'):
            m.overlay(rows,images,outs)

    def test_no_missing_or_duplicate_image_outcomes(self):
        rows,images,outs=fixture()
        for bad in ([],outs*2):
            with self.assertRaisesRegex(ValueError,'all_unique_image_outcomes'):
                m.overlay(rows,images,bad)

    def test_no_duplicate_candidate_ids(self):
        rows,images,outs=fixture(); rows[1]['triple_candidate_id']=rows[0]['triple_candidate_id']
        with self.assertRaisesRegex(ValueError,'unique_nonempty_inventory'):
            m.overlay(rows,images,outs)

    def test_already_annotated_table_refused(self):
        rows,images,outs=fixture(); rows[0][m.PREFIX+'mean_margin']='old'
        with self.assertRaisesRegex(ValueError,'already_annotated_source'):
            m.overlay(rows,images,outs)

    def test_shared_report_hash_with_changed_state_refused(self):
        rows,images,outs=fixture(); rows[1]['report_sha256']=rows[0]['report_sha256']
        with self.assertRaisesRegex(ValueError,'shared_report_artifact_state_changed'):
            m.overlay(rows,images,outs)

    def test_full_grid_cannot_be_shrunk_to_favorable_fixture(self):
        rows,images,_=fixture()
        with self.assertRaisesRegex(ValueError,'complete_80_3_4_grid'):
            m.validate_inventory(rows,images)

    def test_coverage_denominator_includes_unknowns(self):
        inputs=fixture(); summary=m.aggregate(m.overlay(*inputs),inputs[2])
        result=summary['biovil_report_proxy_relations']
        self.assertEqual((result['proxy_support'],result['proxy_opposition'],result['not_comparable']),(1,1,2))
        self.assertEqual(result['comparable_coverage'],.5)
        self.assertEqual(result['agreement_over_comparable'],.5)

    def test_no_comparable_is_na_not_perfect_or_zero(self):
        inputs=fixture(reports=('unknown','uncertain')); result=m.aggregate(m.overlay(*inputs),inputs[2])['biovil_report_proxy_relations']
        self.assertEqual(result['comparable_coverage'],0)
        self.assertIsNone(result['agreement_over_comparable'])

    def test_all_four_state_patterns_have_no_clinical_action(self):
        self.assertEqual(m.joint_pattern('uncertain','positive','positive'),'image_evidence_unavailable')
        self.assertEqual(m.joint_pattern('negative','negative','uncertain'),'report_evidence_unavailable')
        with self.assertRaisesRegex(ValueError,'four_state_evidence'):
            m.joint_pattern('positive','positive','missing')
        self.assertFalse(m.POLICY['clinical_localization'])
        self.assertFalse(m.POLICY['regeneration_authorized'])
        self.assertFalse(m.POLICY['independent_model_votes'])
        self.assertFalse(m.POLICY['report_text_encoded'])

    def test_approval_guard_before_paths_and_runtime(self):
        for env,approved in (({},True),({'SLURM_JOB_ID':'123'},False)):
            with patch.dict(os.environ,env,clear=True),patch.object(m,'require_inside') as resolver, \
                    patch.object(m.frozen.reuse,'_load_runtime') as factory:
                with self.assertRaises(RuntimeError):
                    m.evaluate(SimpleNamespace(allow_synthetic_biovil=approved))
                resolver.assert_not_called(); factory.assert_not_called()

    def test_existing_output_refused_before_model_load(self):
        with patch.object(m.opacity,'guard'),patch.object(m,'require_inside',return_value=Path(__file__)), \
                patch.object(m.frozen.reuse,'_load_runtime') as factory:
            with self.assertRaises(FileExistsError):
                m.evaluate(SimpleNamespace(allow_synthetic_biovil=True,output_root='fixture',run_id='fixture'))
            factory.assert_not_called()

    def test_nonoffline_execution_refused_before_model_load(self):
        with patch.object(m.opacity,'guard'),patch.object(m,'require_inside',return_value=Path('/nonexistent_fixture')), \
                patch.dict(os.environ,{},clear=True),patch.object(m.frozen.reuse,'_load_runtime') as factory:
            with self.assertRaisesRegex(RuntimeError,'offline_frozen_scoring'):
                m.evaluate(SimpleNamespace(allow_synthetic_biovil=True,output_root='fixture',run_id='fixture'))
            factory.assert_not_called()


if __name__=='__main__':
    unittest.main()
