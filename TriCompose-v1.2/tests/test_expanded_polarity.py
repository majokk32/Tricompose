"""Invented report text/metadata only; no files, models, real patients or GPUs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import prepare_expanded_polarity as builder


def registry_fixture():
    rows, roles = [], {}
    for number in range(80):
        case = f"synthetic_{number:03d}"
        roles[case] = "development" if number < 48 else "calibration" if number < 64 else "final_test"
        rows.append({"case_id": case, "triple_candidate_id": f"triple_{number}", "lineage": {
            "ehr_sha256": f"{number+1:064x}", "cxr_sha256": f"{number+101:064x}",
            "report_sha256": f"{number+1001:064x}", "cxr_model_id": "chexgenbench_sana",
            "cxr_seed": 0, "report_model_id": "chexagent2", "cxr_candidate_id": f"image_{number}",
            "report_candidate_id": f"report_{number}"}})
    return rows, roles


class ExpandedPolarityTests(unittest.TestCase):
    def test_all_named_findings_support_both_explicit_directions(self):
        for finding, (_, label, _) in builder.TARGETS.items():
            negative = f"No {label.lower()}."
            positive = f"{label} is present."
            edit, reason = builder.polarity_edit(negative, finding)
            self.assertIsNone(reason, finding)
            self.assertEqual(edit["text"], positive)
            self.assertEqual(edit["source_text_assertion"], "negative")
            edit, reason = builder.polarity_edit(positive, finding)
            self.assertIsNone(reason, finding)
            self.assertEqual(edit["text"], negative)
            self.assertEqual(edit["source_text_assertion"], "positive")

    def test_only_isolated_target_changes_and_header_survives(self):
        text = "Findings: Small bilateral pleural effusions. No pneumothorax."
        edit, _ = builder.polarity_edit(text, "pleural_effusion")
        self.assertEqual(edit["text"], "Findings: No pleural effusion. No pneumothorax.")
        start = edit["span_start"]
        self.assertEqual(edit["text"][:start] + edit["source_statement"] + edit["text"][start+len(edit["replacement_statement"]):], text)

    def test_qualified_negation_is_not_global_finding_absence(self):
        for text, finding in (("No large pleural effusion.", "pleural_effusion"),
                              ("No right pneumothorax.", "pneumothorax"),
                              ("Cardiomegaly is not severe.", "cardiomegaly"),
                              ("No focal consolidation.", "consolidation")):
            self.assertIsNone(builder.polarity_edit(text, finding)[0], text)

    def test_unknown_or_uncertainty_never_creates_an_edit(self):
        for text in ("Possible pneumonia.", "Pneumonia cannot be excluded.", "Pneumonia?", "No change in pneumonia."):
            self.assertIsNone(builder.polarity_edit(text, "pneumonia")[0])
        self.assertIsNone(builder.polarity_edit("Clear lungs.", "pneumonia")[0])
        self.assertIsNone(builder.polarity_edit("Patchy opacities.", "consolidation")[0])

    def test_mixed_findings_and_repeated_mentions_are_refused(self):
        self.assertIsNone(builder.polarity_edit("No edema or pleural effusion.", "edema")[0])
        self.assertIsNone(builder.polarity_edit("Edema and pneumonia are present.", "edema")[0])
        self.assertIsNone(builder.polarity_edit("Pneumonia. Impression: Pneumonia.", "pneumonia")[0])

    def test_measurements_and_nonpulmonary_qualifiers_are_not_invented(self):
        self.assertIsNone(builder.polarity_edit("A 3 mm pleural effusion.", "pleural_effusion")[0])
        self.assertIsNone(builder.polarity_edit("Pericardial effusion.", "pleural_effusion")[0])
        self.assertIsNone(builder.polarity_edit("Soft tissue edema.", "edema")[0])

    def test_frozen_development_cohort_excludes_all_reserved_cases(self):
        inputs = registry_fixture()
        before = copy.deepcopy(inputs)
        selected = builder.freeze_cases(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(len(selected), 48)
        self.assertTrue(all(inputs[1][row["case_id"]] == "development" for row in selected))
        self.assertEqual(builder.freeze_cases(*inputs), selected)

    def test_changed_roles_or_duplicate_ehr_hashes_are_refused(self):
        rows, roles = registry_fixture()
        roles["synthetic_000"] = "final_test"
        with self.assertRaisesRegex(ValueError, "roles changed"):
            builder.freeze_cases(rows, roles)
        rows, roles = registry_fixture()
        rows[0]["lineage"]["ehr_sha256"] = rows[1]["lineage"]["ehr_sha256"]
        with self.assertRaisesRegex(ValueError, "not unique"):
            builder.freeze_cases(rows, roles)

    def test_all_attempts_and_controls_retained_without_editability_selection(self):
        cases = [{"case_id": "synthetic_a", "ehr_sha256": "a"*64}, {"case_id": "synthetic_b", "ehr_sha256": "b"*64}]
        texts = {"synthetic_a": "No pneumothorax.", "synthetic_b": "Clear lungs."}
        records, attempts = builder.prepare_records(cases, texts)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(attempts), 16)
        self.assertEqual(sum(row["available"] for row in attempts), 1)
        self.assertEqual(sum(row["kind"] == "unchanged" for row in records), 2)
        self.assertEqual(builder.prepare_records(cases, texts), (records, attempts))

    def test_bad_reports_are_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError, "cannot be silently dropped"):
            builder.prepare_records([{"case_id": "synthetic_a", "ehr_sha256": "a"*64}], {"synthetic_a": " "})

    def test_protocol_fingerprint_binds_named_scope(self):
        self.assertEqual(set(builder.FINDINGS), set(builder.TARGETS))
        self.assertEqual(builder.protocol_hash(), builder.protocol_hash())
        self.assertEqual(len(builder.protocol_hash()), 64)


if __name__ == "__main__":
    unittest.main()
