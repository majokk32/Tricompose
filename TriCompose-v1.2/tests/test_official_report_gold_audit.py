"""Invented CSVs only; never opens the actual gold or patient linkage files."""
import csv
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"real_validation"))
import audit_official_report_gold as audit


def reader(rows,columns):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n")
    writer.writeheader();writer.writerows(rows);stream.seek(0)
    return csv.DictReader(stream)


def gold_row(study="101",**values):
    return {"study_id":study,**dict.fromkeys(audit.CONDITIONS,""),**values}


def link(study="101",subject="201",split="test",report_path="invented_report.txt"):
    return {"study_id":study,"subject_id":subject,"split":split,"report_path":report_path}


def execute(gold,links):
    return audit.audit_streams(reader(gold,("study_id",*audit.CONDITIONS)),
        reader(links,("study_id","subject_id","split","report_path")))


class GoldCoverageTests(unittest.TestCase):
    def test_manual_airspace_source_name_is_preserved_not_clinically_aliased(self):
        row={"study_id":"101",**dict.fromkeys(audit.MANUAL_SOURCE_CONDITIONS,"")}
        row["Airspace Opacity"]="1";row["Pneumonia"]="-1"
        result=audit.audit_streams(reader([row],tuple(row)),reader([link()],tuple(link())))
        counts=result["all_annotation_label_counts"]
        self.assertEqual(len(counts),14)
        self.assertEqual(counts["Airspace Opacity"]["positive"],1)
        self.assertNotIn("Lung Opacity",counts)
        self.assertFalse(result["airspace_opacity_remapped_to_lung_opacity"])
        self.assertIsNone(result["common_eight_source_columns"]["Lung Opacity"])
        self.assertEqual(result["common_eight_source_columns"]["Pneumonia"],"Pneumonia")
        self.assertEqual(result["splits"]["test"]["study_label_counts"]["Pneumonia"]["uncertain"],1)
        self.assertFalse(result["primary_metric_eligible"])

    def test_both_opacity_names_duplicates_or_arbitrary_replacements_are_rejected(self):
        for header in (("study_id",*audit.CONDITIONS,"Airspace Opacity"),
            ("study_id",*audit.CONDITIONS[:-1],"Fracture"),
            tuple("Opacity" if name=="Lung Opacity" else name for name in ("study_id",*audit.CONDITIONS))):
            self.assertIsNone(audit.annotation_conditions(header))
        self.assertIsNone(audit.annotation_conditions(None))

    def test_header_inventory_contains_only_fixed_schema_names_or_hashes(self):
        secret="invented_private_text_not_a_column"
        inventory=audit.header_inventory(["study_id","subject_id","cardiomegaly","pleural_effusion",secret])
        names=[item["canonical_name"] for item in inventory["recognized_columns"]]
        self.assertEqual(names,["study_id","subject_id","Cardiomegaly","Pleural Effusion"])
        self.assertEqual(inventory["column_count"],5)
        self.assertEqual(len(inventory["unrecognized_columns"]),1)
        self.assertNotIn(secret,json.dumps(inventory))
        self.assertIn("Pneumonia",inventory["missing_required_names"])

    def test_header_block_is_not_claimed_as_clinical_or_label_evaluation(self):
        result=audit.blocked_schema_summary(["study_id"])
        self.assertEqual(result["status"],"blocked_annotation_schema_mismatch")
        self.assertFalse(result["annotation_rows_read"])
        self.assertFalse(result["linkage_rows_read"])
        self.assertFalse(result["primary_metric_eligible"])
        self.assertFalse(result["selection_changed"])
        self.assertFalse(result["regeneration_authorized"])
        self.assertNotIn("all_annotation_label_counts",result)
        self.assertIn("blocked",audit.markdown(result))

    def test_failure_codes_are_allowlisted_and_never_echo_arbitrary_messages(self):
        self.assertEqual(audit.safe_error_code(ValueError("unexpected published annotation class")),
            "unsupported_annotation_class")
        self.assertEqual(audit.safe_error_code(ValueError("patient crosses source splits")),
            "patient_split_overlap")
        secret="invented_patient_key_and_private_path"
        self.assertEqual(audit.safe_error_code(ValueError(secret)),"unclassified_failure")
        self.assertNotIn(secret,audit.safe_error_code(ValueError(secret)))

    def test_schema_rejection_has_diagnostic_code_without_column_or_row_values(self):
        try:
            audit.audit_streams(reader([], ("invented_unknown_column",)),reader([],tuple(link())))
        except ValueError as exc:
            self.assertEqual(audit.safe_error_code(exc),"annotation_schema_mismatch")
        else:self.fail("invalid schema must fail closed")

    def test_published_zero_is_negative_not_chexbert_class_zero_unknown(self):
        self.assertEqual(audit.published_label_state("0"),"negative")
        self.assertEqual(audit.published_label_state("0.0"),"negative")
        self.assertEqual(audit.published_label_state(""),"unknown")
        self.assertEqual(audit.published_label_state("-1.0"),"uncertain")
        self.assertEqual(audit.published_label_state("1.0"),"positive")

    def test_unexpected_annotation_classes_and_missing_cells_fail_closed(self):
        for value in (None,"NaN","2","positive","1e0"):
            with self.assertRaises(ValueError):audit.published_label_state(value)

    def test_full_14_and_common_eight_are_preserved(self):
        result=execute([gold_row(Pneumonia="1",Edema="-1",Cardiomegaly="0")],[link()])
        self.assertEqual(len(result["all_annotation_label_counts"]),14)
        self.assertEqual(len(result["common_eight_findings"]),8)
        counts=result["all_annotation_label_counts"]
        self.assertEqual(counts["Pneumonia"]["positive"],1)
        self.assertEqual(counts["Edema"]["uncertain"],1)
        self.assertEqual(counts["Cardiomegaly"]["negative"],1)
        self.assertEqual(counts["Pleural Effusion"]["unknown"],1)

    def test_duplicate_image_link_rows_do_not_inflate_study_denominator(self):
        result=execute([gold_row()],[link(),link()])
        self.assertEqual(result["linked_annotation_studies"],1)
        self.assertEqual(result["splits"]["test"]["matched_manifest_rows"],2)
        self.assertEqual(result["splits"]["test"]["unique_annotation_studies"],1)

    def test_unlinked_and_empty_report_path_are_not_silently_discarded(self):
        result=execute([gold_row(),gold_row("102")],[link(report_path="")])
        self.assertEqual(result["annotation_studies"],2)
        self.assertEqual(result["unlinked_annotation_studies"],1)
        self.assertEqual(result["splits"]["test"]["studies_with_nonempty_report_path"],0)
        self.assertFalse(result["report_file_existence_or_content_checked"])

    def test_train_linkage_is_counted_not_mislabeled_as_heldout(self):
        result=execute([gold_row()],[link(split="train")])
        self.assertEqual(result["splits"]["train"]["unique_annotation_studies"],1)
        self.assertEqual(result["splits"]["test"]["unique_annotation_studies"],0)

    def test_patient_cross_split_studies_and_duplicate_gold_are_rejected(self):
        with self.assertRaises(ValueError):execute([gold_row(),gold_row()],[link()])
        with self.assertRaises(ValueError):execute([gold_row()],
            [link(),link(study="102",split="val")])
        with self.assertRaises(ValueError):execute([gold_row()],[link(),link(subject="202")])

    def test_noncanonical_metadata_keys_or_splits_are_rejected(self):
        for row in (link(study="invalid"),link(subject=""),link(split="unknown")):
            with self.assertRaises(ValueError):execute([gold_row()],[row])

    def test_empty_gold_or_extra_text_columns_are_rejected(self):
        with self.assertRaises(ValueError):execute([],[link()])
        stream=reader([{**gold_row(),"Report Impression":"wholly invented"}],
            ("study_id",*audit.CONDITIONS,"Report Impression"))
        with self.assertRaises(ValueError):audit.audit_streams(stream,reader([link()],tuple(link())))

    def test_only_aggregate_schema_and_counts_leave_function(self):
        result=execute([gold_row(study="991001",Pneumonia="1")],
            [link(study="991001",subject="992001",report_path="invented_private_path.txt")])
        payload=json.dumps(result)
        for secret in ("991001","992001","invented_private_path.txt"):
            self.assertNotIn(secret,payload)
        self.assertFalse(result["patient_keys_written"])
        self.assertFalse(result["raw_report_text_read"])
        self.assertFalse(result["raw_ehr_fields_inspected"])
        self.assertFalse(result["independent_image_ground_truth"])

    def test_human_source_never_clears_checkpoint_or_clinical_gates(self):
        result=execute([gold_row()],[link()])
        self.assertFalse(result["primary_metric_eligible"])
        self.assertFalse(result["thresholds_fitted"])
        self.assertFalse(result["selection_changed"])
        self.assertFalse(result["regeneration_authorized"])
        self.assertEqual(result["model_calls"],0)
        self.assertIsNone(result["clinical_extraction_accuracy"])
        self.assertEqual(result["checkpoint_training_overlap_status"],"unverified_not_assessed_by_this_audit")

    def test_input_order_does_not_change_aggregate_json(self):
        gold=[gold_row("101",Pneumonia="1"),gold_row("102",Pneumonia="0")]
        links=[link(),link(study="102",subject="202",split="val")]
        result=execute(gold,links)
        self.assertEqual(result,execute(list(reversed(gold)),list(reversed(links))))
        self.assertEqual(result,json.loads(json.dumps(result)))

    def test_explicit_authorization_and_slurm_precede_every_real_source_access(self):
        for job,approval in ((None,False),(None,True),("invented",False)):
            env={} if job is None else {"SLURM_JOB_ID":job}
            with patch.dict(os.environ,env,clear=True),patch.object(audit,"source_file") as access:
                with self.assertRaisesRegex(RuntimeError,"approval"):
                    audit.run(SimpleNamespace(allow_real_gold_metadata=approval))
                access.assert_not_called()


if __name__=="__main__":
    unittest.main()
