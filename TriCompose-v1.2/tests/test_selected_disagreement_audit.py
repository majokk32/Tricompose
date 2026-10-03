"""Invented cached states only; no artifacts, models, datasets or inference."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import audit_selected_disagreements as audit
import build_finding_review as builder
from test_finding_review import review_fixture
from test_separated_qwen_review import states


def fixture():
    inputs = review_fixture(two_reports=True)
    inputs[0][0]["qwen_report_states"] = json.dumps(states(pneumonia="positive"))
    facts, _ = builder.build_rows(*inputs)
    cxrs, reports, responses, image_labels, report_labels = {}, {}, {}, {}, {}
    for row in facts:
        iid, rid, name = row["cxr_candidate_id"], row["report_candidate_id"], row["finding"]
        hashes = row["artifact_hashes"]
        cxrs[iid] = {"case_id": row["case_id"], "artifact": {"path": "invented_image", "sha256": hashes["cxr_sha256"]},
                     "ehr_sha256": hashes["ehr_sha256"], "ehr_facts_sha256": hashes["ehr_facts_sha256"]}
        reports[rid] = {"case_id": row["case_id"], "parent_cxr_candidate_id": iid,
                        "model_id": "maira2" if rid == "report_a" else "llavarad",
                        "artifact": {"path": "invented_report", "sha256": hashes["report_sha256"]}}
        for cid, kind, fingerprint, state in ((iid, "image", hashes["cxr_sha256"], row["states"]["qwen_image"]),
                                              (rid, "report", hashes["report_sha256"], row["states"]["qwen_report"])):
            responses.setdefault(cid, {"candidate_id": cid, "input_kind": kind, "artifact_sha256": fingerprint,
                "contract_status": "complete", "input_tokens": 100, "output_tokens": 85,
                "token_limit_reached": False, "states": {}})["states"][name] = state
        ilabel = image_labels.setdefault(iid, {"cxr_candidate_id": iid, "image_sha256": hashes["cxr_sha256"],
                                              "finding_probabilities": {}, "finding_states": {}})
        ilabel["finding_states"][name] = row["states"]["xrv"]
        ilabel["finding_probabilities"][name] = {"positive": .8, "negative": .2, "unknown": None}[row["states"]["xrv"]]
        report_labels.setdefault(rid, {"report_candidate_id": rid, "report_sha256": hashes["report_sha256"],
                                      "finding_states": {}})["finding_states"][name] = row["states"]["chexbert"]
    qwen = {"records": list(responses.values()), "producer": {"max_new_tokens": 384}}
    xrv = {"records": list(image_labels.values()), "thresholds": {
        name: {"enabled": True, "positive_min": .6, "negative_max": .6} for name in audit.FINDINGS}}
    return facts, qwen, cxrs, reports, xrv, {"records": list(report_labels.values())}


class SelectedDisagreementAuditTests(unittest.TestCase):
    def test_replay_keeps_inputs_and_winner_states_unchanged(self):
        inputs = fixture()
        original = copy.deepcopy(inputs)
        selected, peers = audit.audit_records(*inputs)
        self.assertEqual(inputs, original)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(peers), 2)
        self.assertEqual(selected[0]["score_minus_positive_min"], .2)
        self.assertFalse(selected[0]["distance_is_confidence_or_error_probability"])

    def test_named_lookup_is_not_positional(self):
        inputs = fixture()
        for record in inputs[1]["records"]:
            record["states"] = dict(reversed(list(record["states"].items())))
        selected, _ = audit.audit_records(*inputs)
        self.assertEqual(selected[0]["finding"], "pneumonia")

    def test_changed_candidate_artifact_hash_is_refused(self):
        inputs = fixture()
        inputs[2]["image_a"]["artifact"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            audit.audit_records(*inputs)

    def test_changed_cache_state_is_refused(self):
        inputs = fixture()
        inputs[4]["records"][0]["finding_states"]["pneumonia"] = "negative"
        with self.assertRaisesRegex(ValueError, "threshold replay mismatch"):
            audit.audit_records(*inputs)

    def test_contract_complete_does_not_authorize_clinical_attribution(self):
        selected, _ = audit.audit_records(*fixture())
        self.assertFalse(selected[0]["clinical_fault_assigned"])
        self.assertIsNone(selected[0]["confirmed_faulty_modality"])
        self.assertFalse(selected[0]["automatic_repair_eligible"])
        self.assertFalse(selected[0]["report_response"]["raw_response_reparse_available"])

    def test_token_limit_prevents_complete_contract(self):
        inputs = fixture()
        inputs[1]["records"][0].update(output_tokens=384, token_limit_reached=True)
        with self.assertRaisesRegex(ValueError, "token-limited"):
            audit.audit_records(*inputs)

    def test_unavailable_response_cannot_supply_explicit_states(self):
        inputs = fixture()
        inputs[1]["records"][0]["contract_status"] = "failed_unavailable"
        with self.assertRaisesRegex(ValueError, "unavailable response"):
            audit.audit_records(*inputs)

    def test_unknown_report_remains_unknown(self):
        inputs = fixture()
        row = next(row for row in inputs[0] if row["original_selected"] and row["finding"] == "pneumonia")
        row["states"]["chexbert"] = "unknown"
        inputs[5]["records"][0]["finding_states"]["pneumonia"] = "unknown"
        selected, _ = audit.audit_records(*inputs)
        self.assertEqual(selected[0]["states"]["chexbert"], "unknown")

    def test_oversized_or_duplicate_evidence_is_refused(self):
        inputs = fixture()
        inputs[0].append(copy.deepcopy(inputs[0][0]))
        with self.assertRaisesRegex(ValueError, "duplicate finding"):
            audit.audit_records(*inputs)
        inputs = fixture()
        with self.assertRaisesRegex(ValueError, "bounded"):
            audit.audit_records(inputs[0] * 25, *inputs[1:])


if __name__ == "__main__":
    unittest.main()
