"""Invented numeric evidence; mismatching a report never proves its fault."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_image_attribution_stress_v1 as m
from test_probe_repair_v1 import observation
from test_localization_observability_v1 import resolver


def packet(score=.8, margins=(.2,.2,.2), report='negative'):
    return {'item_id': 'item_0000', 'ehr_opacity_state': 'unknown', 'report_opacity_state': report,
        'image_evidence': {'xrv_score': score, 'margins': list(margins) if margins is not None else None}}


def scores(view):
    line = view['lineage']
    identity = {'case_id': view['case_id'], **{k:line[k] for k in ('cxr_candidate_id','cxr_sha256','cxr_model_id')}}
    x = {**identity, 'status':'scored', 'exact_lung_opacity_score':.2, 'infiltration_score':.9}
    b = {**identity, 'status':'scored', 'score_pairs':{f:{'positive_cosine':.3,'negative_cosine':.1} for f in m.FAMILIES}}
    return x,b


class ImageAttributionTests(unittest.TestCase):
    def test_opposition_is_only_unsafe_comparator_not_action(self):
        result=m.judge(packet(), 'agree_fixed_mean')
        self.assertEqual(result['proxy_relation'],'proxy_opposition')
        self.assertEqual(result['unsafe_report_blame_shortcut'],'report')
        self.assertIsNone(result['confirmed_faulty_modality'])
        self.assertIsNone(result['clinical_localization_accuracy'])
        self.assertFalse(result['regeneration_authorized']); self.assertFalse(result['clinical_qualified'])

    def test_support_is_not_clinical_clean_or_stop_success(self):
        result=m.judge(packet(report='positive'),'agree_fixed_mean')
        self.assertEqual(result['proxy_relation'],'proxy_support')
        self.assertEqual(result['evidence_request_reason'],'no_action_authorized_by_proxy_agreement')
        self.assertIsNone(result['unsafe_report_blame_shortcut'])

    def test_report_unknown_and_uncertain_not_negative(self):
        for state in ('unknown','uncertain'):
            result=m.judge(packet(report=state),'xrv_exact_0_5')
            self.assertEqual(result['proxy_relation'],'not_comparable_report_state')
            self.assertIsNone(result['unsafe_report_blame_shortcut'])

    def test_opacity_ehr_cannot_be_filled_or_pneumonia_promoted(self):
        for state in ('positive','negative','uncertain'):
            value=packet();value['ehr_opacity_state']=state
            with self.assertRaises(ValueError):m.judge(value,'xrv_exact_0_5')

    def test_payload_identity_key_and_secondary_score_forbidden(self):
        for field in ('report_text','patient_id','intervention_target','artifact_hash','global_biovil'):
            value=packet();value[field]='invented'
            with self.assertRaises(ValueError):m.judge(value,'agree_fixed_mean')

    def test_image_disagreement_is_missing_eligibility_not_report_fault(self):
        result=m.judge(packet(score=.1),'agree_fixed_mean')
        self.assertEqual(result['image_status'],'abstain_reader_disagreement')
        self.assertEqual(result['proxy_relation'],'not_comparable_image_abstained')
        self.assertIsNone(result['unsafe_report_blame_shortcut'])

    def test_template_stability_is_not_independent_votes(self):
        value=packet(margins=(.4,.2,-.1))
        self.assertEqual(m.judge(value,'agree_fixed_mean')['proxy_relation'],'proxy_opposition')
        result=m.judge(value,'agree_all_templates')
        self.assertEqual(result['image_status'],'abstain_template_sensitive')
        self.assertFalse(result['image_votes_are_independent'])

    def test_mean_tie_and_template_tie_abstain(self):
        self.assertEqual(m.judge(packet(margins=(.2,-.2,0)),'biovil_fixed_mean')['image_status'],'abstain_mean_tie')
        self.assertEqual(m.judge(packet(margins=(.2,.2,0)),'agree_all_templates')['image_status'],'abstain_template_tie')

    def test_unavailable_not_negative_or_success(self):
        result=m.judge(packet(score=None),'agree_fixed_mean')
        self.assertEqual(result['proxy_relation'],'not_comparable_image_unavailable')
        self.assertIsNone(result['image_state']);self.assertIsNone(result['unsafe_report_blame_shortcut'])

    def test_other_available_single_reader_does_not_hide_failure(self):
        value=packet(margins=None)
        self.assertEqual(m.judge(value,'xrv_exact_0_5')['proxy_relation'],'proxy_opposition')
        self.assertEqual(m.judge(value,'biovil_fixed_mean')['proxy_relation'],'not_comparable_image_unavailable')

    def test_frozen_boundary_and_exact_head_not_max(self):
        self.assertEqual(m.judge(packet(score=.5),'xrv_exact_0_5')['image_state'],'positive')
        self.assertEqual(m.judge(packet(score=.49),'xrv_exact_0_5')['image_state'],'negative')
        view=observation();x,b=scores(view)
        with patch.object(m.b.policy_module,'snapshot',side_effect=deepcopy):
            images,_=m.bind_image_evidence({'case_000':{0:view}},[x],[b])
        self.assertEqual(images[x['cxr_candidate_id']]['xrv_score'],.2)

    def test_complete_fixed_templates_no_favorable_subset(self):
        view=observation();_,b=scores(view)
        b['score_pairs']['present_absent']={'positive_cosine':-.3,'negative_cosine':.3}
        self.assertEqual(m.score_margins(b),[.3-.1,.3-.1,-.6])
        del b['score_pairs']['present_absent']
        with self.assertRaises(ValueError):m.score_margins(b)

    def test_failure_cannot_smuggle_score_pairs(self):
        _,b=scores(observation());b['status']='failed_without_replacement'
        with self.assertRaises(ValueError):m.score_margins(b)
        b['score_pairs']=None;self.assertIsNone(m.score_margins(b))

    def test_bad_cosines_and_raw_scores_rejected(self):
        for value in (True,'0.2',float('nan'),float('inf'),2):
            _,b=scores(observation());b['score_pairs']['shows_no']['positive_cosine']=value
            with self.assertRaises(ValueError):m.score_margins(b)
            with self.assertRaises(ValueError):m.judge(packet(score=value),'xrv_exact_0_5')

    def test_duplicate_missing_or_hash_changed_image_scores_rejected(self):
        view=observation();x,b=scores(view)
        for kind in ('duplicate','missing','hash'):
            rows=[deepcopy(x)]
            if kind=='duplicate':rows.append(deepcopy(x))
            elif kind=='missing':rows=[]
            else:rows[0]['cxr_sha256']='9'*64
            with patch.object(m.b.policy_module,'snapshot',side_effect=deepcopy):
                with self.assertRaises(ValueError):m.bind_image_evidence({'case_000':{0:view}},rows,[b])

    def test_biovil_same_parent_lineage_required(self):
        view=observation();x,b=scores(view);b['case_id']='case_other'
        with patch.object(m.b.policy_module,'snapshot',side_effect=deepcopy):
            with self.assertRaises(ValueError):m.bind_image_evidence({'case_000':{0:view}},[x],[b])

    def test_fixed_ehr_opacity_and_anchor_hash_preserved(self):
        a,z=observation(),observation(number=1)
        z['lineage']['ehr_sha256']='9'*64
        x,b=scores(a)
        with patch.object(m.b.policy_module,'snapshot',side_effect=deepcopy):
            with self.assertRaises(ValueError):m.bind_image_evidence({'case_000':{0:a,1:z}},[x],[b])
        a['states']['ehr']['lung_opacity']='positive'
        with patch.object(m.b.policy_module,'snapshot',side_effect=deepcopy):
            with self.assertRaises(ValueError):m.bind_image_evidence({'case_000':{0:a}},[x],[b])

    def test_swap_packet_uses_displayed_image_not_recipient_image(self):
        a,z=observation(),observation(model='other',number=1,image='negative')
        row=resolver(a);row['displayed_cxr_candidate_id']=z['lineage']['cxr_candidate_id'];row['displayed_cxr_sha256']=z['lineage']['cxr_sha256']
        images={z['lineage']['cxr_candidate_id']:{'cxr_sha256':z['lineage']['cxr_sha256'],'xrv_score':.1,'margins':[-.2]*3}}
        reports={a['lineage']['report_candidate_id']:{'report_sha256':a['lineage']['report_sha256'],'state':'positive'}}
        value=m.item_packets([row],images,reports)[0]
        self.assertEqual(value['image_evidence']['xrv_score'],.1)
        self.assertEqual(m.judge(value,'agree_fixed_mean')['proxy_relation'],'proxy_opposition')

    def test_swapped_report_hash_and_duplicate_item_fail_closed(self):
        a=observation();row=resolver(a)
        images={a['lineage']['cxr_candidate_id']:{'cxr_sha256':a['lineage']['cxr_sha256'],'xrv_score':.1,'margins':[-.2]*3}}
        reports={a['lineage']['report_candidate_id']:{'report_sha256':'9'*64,'state':'positive'}}
        with self.assertRaises(ValueError):m.item_packets([row],images,reports)
        reports[a['lineage']['report_candidate_id']]['report_sha256']=row['displayed_report_sha256']
        with self.assertRaises(ValueError):m.item_packets([row,row],images,reports)

    def test_source_packet_is_not_mutated(self):
        value=packet();before=deepcopy(value);m.judge(value,'agree_fixed_mean');self.assertEqual(value,before)

    def test_shortcut_can_disagree_with_mechanical_cxr_swap(self):
        out=m.judge(packet(),'agree_fixed_mean')
        row=resolver(observation());key={'item_id':out['item_id'],'intervention_type':'cxr_swap','intervention_target':'cxr'}
        summary=m.summarize([out],[row],[key])
        self.assertTrue(all(r['unsafe_shortcut_disagrees_with_mechanical_target']==1 for r in summary))
        self.assertTrue(all(r['clinical_localization_accuracy'] is None for r in summary))

    def test_empty_comparison_has_na_not_perfect_support(self):
        out=m.judge(packet(report='unknown'),'agree_fixed_mean');row=resolver(observation())
        key={'item_id':out['item_id'],'intervention_type':'no_corruption','intervention_target':'none'}
        self.assertTrue(all(r['conditional_proxy_support'] is None for r in m.summarize([out],[row],[key])))

    def test_protocol_cannot_authorize_training_or_regeneration(self):
        value=json.loads(m.PROTOCOL.read_text());m.validate_protocol(value)
        for key in ('clinical_qualified','regeneration_authorized','shortcut_is_authorized_action','synthetic_transport_validated','training_allowed'):
            bad=deepcopy(value);bad[key]=True
            with self.assertRaises(ValueError):m.validate_protocol(bad)

    def test_cpu_guard_precedes_all_inputs_and_writes(self):
        with patch.object(m.b,'cpu_guard',side_effect=RuntimeError('fixture')),patch.object(m.b,'load_primary') as read,patch.object(m,'new_atomic_run') as write:
            with self.assertRaises(RuntimeError):m.run(None)
            read.assert_not_called();write.assert_not_called()


if __name__=='__main__':unittest.main()
