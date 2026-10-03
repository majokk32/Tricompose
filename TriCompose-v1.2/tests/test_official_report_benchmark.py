"""Invented headers/CSV/report text only: no real sources or model execution."""
import csv
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'real_validation'))
import official_report_benchmark as method
import run_official_report_benchmark as runtime
from audit_official_report_gold import CONDITIONS,MANUAL_SOURCE_CONDITIONS


def reader(rows,columns):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns)
    writer.writeheader();writer.writerows(rows);stream.seek(0)
    return csv.DictReader(stream)


def gold(study='101',conditions=CONDITIONS,**states):
    return {'study_id':study,**dict.fromkeys(conditions,''),**states}


def linkage(study='101',subject='201',split='test',path='invented_report.txt'):
    return {'study_id':study,'subject_id':subject,'split':split,'report_path':path}


def collect(golds,links,conditions=CONDITIONS):
    return method.collect_sources(reader(golds,('study_id',*conditions)),reader(links,tuple(linkage())))


def prediction(item_id='item_0000',status='complete',**states):
    return {'item_id':item_id,'status':status,
        'finding_states':{**dict.fromkeys(method.HEADS,'unknown'),**states} if status=='complete' else None}


class OfficialReportBenchmarkTests(unittest.TestCase):
    def test_impression_only_excludes_history_and_finding_context(self):
        text='HISTORY: Invented suspected disease.\nFINDINGS: Invented finding.\nIMPRESSION: No pneumothorax.\nRECOMMENDATIONS: Invented plan.'
        selected,status=method.select_impression(text)
        self.assertEqual(selected,'No pneumothorax.')
        self.assertEqual(status,'explicit_impression')
        self.assertNotIn('HISTORY',selected)

    def test_no_implicit_full_report_or_findings_fallback(self):
        for text in ('Invented unsectioned report.','FINDINGS: No cardiomegaly.','history: invented\nimpression: invented'):
            self.assertEqual(method.select_impression(text),(None,'no_explicit_impression_section'))

    def test_crlf_conclusion_and_combined_section_are_supported(self):
        for name in ('IMPRESSION','CONCLUSION','FINDINGS AND IMPRESSION','FINDINGS/IMPRESSION'):
            text=f'FINDINGS: Invented context.\r\n {name}:\r\n No effusion.\r\n'
            self.assertEqual(method.select_impression(text)[0],'No effusion.')

    def test_ambiguous_empty_and_oversized_sections_abstain(self):
        examples=(('IMPRESSION: First.\nIMPRESSION: Second.','ambiguous_impression_sections'),
            ('IMPRESSION:\nHISTORY: Invented.','empty_impression_section'),
            ('IMPRESSION: '+('x'*8193),'impression_character_bound_exceeded'),
            ('','empty_report'))
        for text,reason in examples:self.assertEqual(method.select_impression(text),(None,reason))

    def test_decoder_preserves_four_states_and_binary_head_unknown(self):
        values=[0,1,2,3]+[0]*10
        decoded=method.decode(values)
        self.assertEqual(decoded['enlarged_cardiomediastinum'],'unknown')
        self.assertEqual(decoded['cardiomegaly'],'positive')
        self.assertEqual(decoded['lung_opacity'],'negative')
        self.assertEqual(decoded['lung_lesion'],'uncertain')
        self.assertEqual(decoded['no_finding'],'unknown')
        self.assertEqual(len(decoded),14)

    def test_decoder_rejects_bad_width_class_types_and_binary_head_class(self):
        for values in ([0]*13,[True]+[0]*13,[4]+[0]*13,[0]*13+[2]):
            with self.assertRaises(ValueError):method.decode(values)

    def test_normalization_matches_existing_wrapper_literal_behavior(self):
        self.assertEqual(method.input_normalization('  No\neffusion.  '),'No effusion.')
        self.assertEqual(method.input_normalization('x\\s+y'),'x y')

    def test_image_link_duplicates_count_once_and_no_case_difficulty_selection(self):
        items,_=collect([gold(Pneumonia='1')],[linkage(),linkage()])
        self.assertEqual(len(items),1)
        self.assertEqual(items[0]['item_id'],'item_0000')
        self.assertEqual(items[0]['status'],'ready_metadata')

    def test_all_unlinked_non_test_empty_and_ambiguous_entries_retained(self):
        rows=[gold(str(101+i)) for i in range(4)]
        links=[linkage('102',split='train'),linkage('103',subject='202',path=''),
            linkage('104',subject='203',path='invented_a.txt'),linkage('104',subject='203',path='invented_b.txt')]
        items,_=collect(rows,links)
        self.assertEqual([row['status'] for row in items],
            ['unlinked','linked_non_test','no_report_path_metadata','ambiguous_report_path_metadata'])

    def test_source_patient_cross_split_inconsistent_study_and_duplicate_gold_fail(self):
        for rows,links in (([gold(),gold()],[linkage()]),
            ([gold()],[linkage(),linkage('102',split='val')]),
            ([gold()],[linkage(),linkage(subject='202')])):
            with self.assertRaises(ValueError):collect(rows,links)

    def test_unknown_and_uncertain_are_not_collapsed_to_negative(self):
        stats=method.state_statistics([('unknown','positive'),('uncertain','positive'),('negative','positive')])
        self.assertEqual(stats['confusion_matrix']['unknown']['positive'],1)
        self.assertEqual(stats['confusion_matrix']['uncertain']['positive'],1)
        self.assertEqual(stats['determinate_promotions_on_uncertain_unknown'],2)
        self.assertEqual(stats['hard_polarity_flips'],1)
        self.assertEqual(stats['per_state']['negative']['reference_support'],1)

    def test_empty_or_unsupported_metric_classes_are_null_not_zero(self):
        stats=method.state_statistics([])
        self.assertIsNone(stats['exact_state_accuracy'])
        self.assertIsNone(stats['state_macro_f1_reference_supported_classes'])
        stats=method.state_statistics([('positive','negative')])
        self.assertEqual(stats['per_state']['positive']['f1'],0)
        self.assertIsNone(stats['per_state']['unknown']['f1'])

    def test_unknown_only_reference_heads_cannot_inflate_pooled_score(self):
        items,conditions=collect([gold()],[linkage()])
        summary=method.summarize(items,[prediction()],conditions)
        self.assertIsNone(summary['mean_per_head_state_macro_f1'])
        self.assertEqual(summary['aggregate_heads_with_annotated_reference'],[])
        self.assertEqual(summary['per_finding']['pneumonia']['confusion_matrix']['unknown']['unknown'],1)
        summary=method.summarize(items,[prediction(pneumonia='positive')],conditions)
        self.assertIsNone(summary['mean_per_head_state_macro_f1'])
        self.assertEqual(summary['all_eligible_heads_determinate_promotions_on_uncertain_unknown'],1)

    def test_unknown_predictions_on_annotated_refs_count_as_omissions(self):
        stats=method.state_statistics([('positive','unknown'),('uncertain','unknown'),('unknown','unknown')])
        self.assertEqual(stats['annotated_checks'],2)
        self.assertEqual(stats['omitted_annotated_assertions'],2)
        self.assertEqual(stats['annotated_state_accuracy'],0)
        self.assertAlmostEqual(stats['exact_state_accuracy'],1/3)

    def test_manual_opacity_and_binary_no_finding_excluded_without_losing_inventory(self):
        items,conditions=collect([gold(conditions=MANUAL_SOURCE_CONDITIONS,Pneumonia='1')],[linkage()],MANUAL_SOURCE_CONDITIONS)
        summary=method.summarize(items,[prediction(pneumonia='positive')],conditions)
        self.assertEqual(len(summary['evaluated_four_state_heads']),12)
        self.assertEqual(len(summary['common_eight_evaluated_heads']),7)
        self.assertNotIn('lung_opacity',summary['per_finding'])
        self.assertNotIn('no_finding',summary['per_finding'])
        self.assertEqual(summary['per_finding']['pneumonia']['exact_state_accuracy'],1)
        self.assertFalse(summary['primary_metric_eligible'])
        self.assertFalse(summary['selection_changed'])
        self.assertFalse(summary['regeneration_authorized'])

    def test_unavailable_report_not_given_fake_unknown_predictions_or_removed(self):
        items,conditions=collect([gold()],[linkage()])
        summary=method.summarize(items,[prediction(status='report_file_missing')],conditions)
        self.assertEqual(summary['annotation_inventory'],1)
        self.assertEqual(summary['completed_reports'],0)
        self.assertIsNone(summary['mean_per_head_state_macro_f1'])
        self.assertEqual(summary['execution_status_counts'],{'report_file_missing':1})

    def test_scope_commit_may_only_retain_original_state(self):
        items,conditions=collect([gold(Cardiomegaly='1')],[linkage()])
        record=prediction(cardiomegaly='positive')
        record['scope_decisions']={'cardiomegaly':{'decision':'scope_commit','state':'positive','scope_verified':True}}
        summary=method.summarize(items,[record],conditions)
        self.assertEqual(summary['frozen_four_finding_scope_diagnostic']['cardiomegaly']['conditional_state_accuracy'],1)
        record['scope_decisions']['cardiomegaly']['state']='negative'
        with self.assertRaises(ValueError):method.summarize(items,[record],conditions)

    def test_source_keys_paths_reference_rows_not_exported_by_summary(self):
        items,conditions=collect([gold(study='991001')],[linkage(study='991001',subject='992001',path='invented_private_path.txt')])
        summary=method.summarize(items,[prediction()],conditions)
        payload=json.dumps(summary)
        for private in ('991001','992001','invented_private_path.txt'):self.assertNotIn(private,payload)
        self.assertFalse(summary['model_received_gold_labels'])
        self.assertFalse(summary['independent_image_ground_truth'])

    def test_duplicate_missing_inventory_and_unknown_states_fail_closed(self):
        items,conditions=collect([gold()],[linkage()])
        for records in ([prediction(),prediction()],[],[prediction(cardiomegaly='bad')],
            [{'item_id':'item_0000','status':'report_file_missing','finding_states':{'fake':'negative'}}]):
            with self.assertRaises(ValueError):method.summarize(items,records,conditions)

    def test_real_source_access_requires_explicit_approval_and_slurm(self):
        for env,approved in (({},False),({},True),({'SLURM_JOB_ID':'12576792'},False)):
            with patch.dict(os.environ,env,clear=True),patch.object(runtime,'source_file') as access:
                with self.assertRaises(RuntimeError):runtime.run(SimpleNamespace(allow_real_report_benchmark=approved))
                access.assert_not_called()

    def test_fake_slurm_env_cannot_bypass_process_cgroup(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'999999999'},clear=True),patch.object(runtime.Path,'read_text',return_value='/unrelated/job/'):
            with self.assertRaises(RuntimeError):runtime.require_approved_slurm(SimpleNamespace(allow_real_report_benchmark=True))

    def test_error_codes_never_echo_source_paths_or_report_content(self):
        self.assertEqual(runtime.failure_code(RuntimeError('cuda_required')),'cuda_required')
        self.assertEqual(runtime.failure_code(ValueError('invented_raw_report_and_private_path')),'unclassified_failure')

    def test_missing_or_boundary_rejected_report_uses_safe_status_not_exception_text(self):
        for exception,reason in ((FileNotFoundError('invented_private_key'),'report_file_missing'),
            (PermissionError('invented_private_key'),'report_file_unreadable'),
            (ValueError('invented_private_key'),'report_path_boundary_rejected')):
            with patch.object(runtime,'source_file',side_effect=exception):
                result=runtime.read_report('invented.txt',Path('/invented'))
                self.assertEqual(result,(None,None,reason))
                self.assertNotIn('invented_private_key',json.dumps(result))


if __name__=='__main__':unittest.main()
