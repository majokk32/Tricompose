"""Invented derived metadata only; no inference, bodies or real sources."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_paired_probes_v1 as audit
from test_fresh_output_acceptance import fixture,reseal
from test_paired_probe_collection_v1 import plan


def data():
    p=plan();rows=[];triples=[]
    for case in p['cases']:
        for seed in (3,4):
            for model in ('cxrmate_single','chexagent2'):
                row,_=fixture(seed=seed,model=model,image_id=f"image_{case['case_id']}_{seed}",
                    report_id=f"report_{case['case_id']}_{seed}_{model}")
                row['case_id']=row['receipt']['case_id']=row['structure']['case_id']=case['case_id']
                row['receipt']['ehr_anchor_sha256']=case['ehr_anchor_sha256'];reseal(row)
                rows.append(row)
                triples.append({k:row[k] for k in ('case_id','cxr_model_id','seed','report_model_id',
                    'cxr_sha256','report_sha256','ehr_sha256','ehr_facts_sha256')})
    # Use fixture scorer fingerprints: plan worker descriptors are wholly invented.
    _,ctx=fixture()
    p['workers']['xrv'].update(thresholds={n:{'enabled':n in ctx['enabled_xrv_findings']} for n in audit.paired.fresh.FINDINGS},
        thresholds_sha256=ctx['thresholds_sha256'],checkpoint_sha256=ctx['xrv_checkpoint_sha256'])
    p['workers']['chexbert']['checkpoint_sha256']=ctx['chexbert_checkpoint_sha256']
    return p,rows,triples


class PairedAuditTests(unittest.TestCase):
    def test_full_unique_matrix_and_no_clinical_promotion(self):
        p,rows,triples=data();result=audit.matrix(rows,triples,p)
        self.assertEqual(result['completed_slots'],8);self.assertEqual(result['missing_slots'],0)
        self.assertEqual(result['unique_image_sha256'],4)
        self.assertIsNone(result['clinical_repair_success'])

    def test_partial_generation_is_counted_not_case_filtered(self):
        p,rows,triples=data();result=audit.matrix(rows[:-1],triples[:-1],p)
        self.assertEqual(result['missing_slots'],1)
        self.assertTrue(result['all_declared_ehrs_retained'])

    def test_duplicate_slot_cannot_pay_for_missing_case(self):
        p,rows,triples=data();rows[-1]=deepcopy(rows[0]);triples[-1]=deepcopy(triples[0])
        with self.assertRaises(ValueError):audit.matrix(rows,triples,p)

    def test_same_image_experts_must_share_exact_classifier_evidence(self):
        p,rows,triples=data();rows[1]['receipt']['fact_states'][3]['xrv']='negative';reseal(rows[1])
        with self.assertRaises(ValueError):audit.matrix(rows,triples,p)

    def test_wrong_generated_parent_binding_rejected(self):
        p,rows,triples=data();triples[0]['cxr_sha256']='9'*64
        with self.assertRaises(ValueError):audit.matrix(rows,triples,p)

    def test_unknown_remains_na_and_secondary_not_inferred(self):
        _,rows,_=data();r=audit.candidate_table(rows)[0]
        self.assertIsNone(r['biovil_raw_cosine']);self.assertIsNone(r['clinical_accuracy'])
        self.assertEqual(r['secondary_status'],'not_executed')
        self.assertEqual(r['cxr_report_comparable_facts'],0)

    def test_replay_cost_is_separate_from_paid_shared_collection(self):
        p,rows,_=data();readouts=[]
        for case in p['cases']:
            ctx=audit.paired.fresh.context(case['anchor'],p['workers']['xrv'],p['workers']['chexbert'])
            readouts.append({'case_id':case['case_id'],'readout':audit.paired.paired_readout(
                [r for r in rows if r['case_id']==case['case_id']],ctx)})
        books=[{'case_id':c['case_id'],'charged_model_attempts':6} for c in p['cases'] for _ in range(2)]
        table=audit.method_table(readouts,rows,books)
        self.assertEqual(len(table),6)
        self.assertEqual(table[0]['simulated_replay_calls'],4)
        self.assertEqual(table[0]['actual_shared_collection_attempts'],12)
        self.assertFalse(table[0]['selection_was_online'])

    def test_incomplete_case_kept_in_every_method_table(self):
        r={'case_id':'invented','readout':{'choices':[]}}
        table=audit.method_table([r],[],[{'case_id':'invented','charged_model_attempts':1}])
        self.assertEqual(len(table),3)
        self.assertTrue(all(x['selected_available'] is False for x in table))
        self.assertTrue(all(x['ehr_cxr_support_over_known'] is None for x in table))


if __name__=='__main__':unittest.main()
