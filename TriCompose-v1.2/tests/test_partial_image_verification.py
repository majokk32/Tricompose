"""Invented state vectors only; no clinical acceptance or image inspection."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import partial_image_verification as partial
from tricompose_v12 import invariant_verification as full
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from test_legacy_automatic_replay import invented, policy
import bind_partial_image_verification as cli

CACHE_HASH = "a" * 64


def config():
    return json.loads((ROOT / "configs/partial_image_verification_v1.json").read_text())


def fixture(case="invented_case_0", report="maira2"):
    scores, details = invented(); bank = make_legacy_bank(scores, details, policy())
    candidate = bank[case][("chexgenbench_sana", 0, report)]
    anchor = full.anchor_from_cached_candidate(candidate)
    image = partial.image_evidence_from_legacy_candidate(candidate, CACHE_HASH)
    return candidate, anchor, image


def receipts(bank):
    return [full.verify_candidate(c, full.anchor_from_cached_candidate(c))
            for grid in bank.values() for c in grid.values()]


class PartialImageTests(unittest.TestCase):
    def test_config_cannot_enable_report_votes_negative_unknown_or_execution(self):
        for key in ("uses_report_labels_scores_or_votes_for_image_receipt", "unknown_uncertain_are_negative",
                    "model_execution_allowed", "clinical_acceptance_allowed"):
            c = config(); c[key] = True
            with self.assertRaises(ValueError): partial.validate_config(c)

    def test_report_not_generated_is_distinct_from_report_unknown(self):
        candidate, anchor, image = fixture()
        r = partial.verify_image_phase(image, anchor)
        for name in ("report_candidate_id", "report_sha256", "report_label_states", "all_three_supported_facts"):
            self.assertIsNone(r[name])
        self.assertEqual(r["report_lifecycle_status"], "not_generated")
        self.assertIsNone(r["raw_edge_readouts"]["ehr_report"])
        self.assertIsNone(r["raw_edge_readouts"]["cxr_report"])
        self.assertEqual(r["raw_edge_readouts"]["ehr_cxr"]["supported_positive"], 1)
        for f in candidate["facts"]: f["states"]["chexbert"] = "unknown"
        completed = full.verify_candidate(candidate, anchor)
        self.assertIsInstance(completed["raw_edge_readouts"]["ehr_report"], dict)
        self.assertEqual(completed["raw_edge_readouts"]["ehr_report"]["missing_comparisons"], 1)

    def test_projection_works_with_every_report_field_absent(self):
        candidate, anchor, image = fixture()
        row = candidate["score_record"]
        for k in ("report_sha256", "report_candidate_id", "report_model_id"): del row["lineage"][k]
        del row["triple_candidate_id"]
        row["scoring"] = {"modality_quality": {"cxr_basic_validity_pass": True}}
        for f in candidate["facts"]:
            f["states"] = {"xrv": f["states"]["xrv"]}
            for k in ("report_candidate_id", "triple_candidate_id", "relations", "evidence_id"):
                del f[k]
            del f["artifact_hashes"]["report_sha256"]
        projected = partial.image_evidence_from_legacy_candidate(candidate, CACHE_HASH)
        self.assertEqual(projected, image)
        self.assertEqual(partial.verify_image_phase(projected, anchor), partial.verify_image_phase(image, anchor))

    def test_report_gates_labels_scores_winners_and_costs_cannot_change_image(self):
        c, a, image = fixture(); before = partial.verify_image_phase(image, a)
        c["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] = 999
        c["score_record"]["scoring"]["modality_quality"]["report_structure_quality_score_0_1"] = -1
        c["score_record"]["scoring"]["cost"] = None
        c["score_record"]["lineage"]["report_sha256"] = "invalid ignored report hash"
        for f in c["facts"]: f["states"]["chexbert"] = "invalid ignored report label"
        self.assertEqual(before, partial.verify_image_phase(partial.image_evidence_from_legacy_candidate(c, CACHE_HASH), a))

    def test_image_receipt_unchanged_across_all_four_report_paths(self):
        scores, details = invented(); grid = make_legacy_bank(scores, details, policy())["invented_case_0"]
        found = [partial.verify_image_phase(partial.image_evidence_from_legacy_candidate(c, CACHE_HASH),
                    full.anchor_from_cached_candidate(c))
                 for slot, c in grid.items() if slot[0] == "chexgenbench_sana"]
        self.assertEqual(len(found), 4)
        self.assertTrue(all(r == found[0] for r in found))

    def test_unknown_and_uncertain_ehr_do_not_become_normal_or_failure(self):
        for case in ("invented_case_1", "invented_case_2"):
            _, a, image = fixture(case); r = partial.verify_image_phase(image, a)
            self.assertEqual(r["verification_status"], "unverified_no_direct_ehr_constraints")
            self.assertIsNone(r["raw_edge_readouts"]["ehr_cxr"]["support_over_known"])
            self.assertIsNone(r["raw_edge_readouts"]["ehr_cxr"]["coverage_over_known"])

    def test_unknown_and_uncertain_image_are_missing_not_negative(self):
        _, a, image = fixture()
        for state in ("unknown", "uncertain"):
            states = tuple((n, state if n == "edema" else s) for n, s in image.finding_states)
            r = partial.verify_image_phase(replace(image, finding_states=states), a)
            edge = r["raw_edge_readouts"]["ehr_cxr"]
            self.assertEqual(edge["missing_comparisons"], 1)
            self.assertEqual(edge["proxy_opposition_facts"], 0)
            self.assertEqual(edge["supported_negative"], 0)
            self.assertEqual(r["verification_status"], "unverified_missing_image_comparison")

    def test_explicit_negative_opposition_and_support_are_separate(self):
        c, _, image = fixture()
        next(f for f in c["facts"] if f["finding"] == "edema")["states"]["ehr"] = "negative"
        a = full.anchor_from_cached_candidate(c)
        r = partial.verify_image_phase(image, a)
        self.assertEqual(r["raw_edge_readouts"]["ehr_cxr"]["proxy_opposition_facts"], 1)
        states = tuple((n, "negative" if n == "edema" else s) for n, s in image.finding_states)
        r = partial.verify_image_phase(replace(image, finding_states=states), a)
        self.assertEqual(r["raw_edge_readouts"]["ehr_cxr"]["supported_negative"], 1)
        self.assertEqual(r["verification_status"], "direct_ehr_image_label_agreement_unvalidated")
        self.assertFalse(r["clinical_acceptance"])

    def test_missing_image_validity_is_not_a_pass(self):
        _, a, image = fixture()
        r = partial.verify_image_phase(replace(image, cxr_basic_validity_pass=None), a)
        self.assertEqual(r["verification_status"], "unverified_image_validity_unavailable")
        r = partial.verify_image_phase(replace(image, cxr_basic_validity_pass=False), a)
        self.assertEqual(r["verification_status"], "artifact_metadata_invalid")

    def test_image_immutable_and_record_is_not_an_alias(self):
        _, a, image = fixture(); before = image.record()
        with self.assertRaises(FrozenInstanceError): image.cxr_sha256 = "b" * 64
        r = image.record(); r["finding_states"][0]["xrv"] = "negative"
        self.assertEqual(image.record(), before)
        with self.assertRaises(ValueError): replace(image, finding_states=list(image.finding_states))
        self.assertEqual(partial.verify_image_phase(image, a), partial.verify_image_phase(image, a))

    def test_invalid_hash_seed_inventory_or_foreign_ehr_refused(self):
        _, a, image = fixture()
        for fields in ({"cxr_sha256": "bad"}, {"seed": True}, {"cxr_basic_validity_pass": 1},
                       {"finding_states": image.finding_states[:-1]}):
            with self.assertRaises(ValueError): replace(image, **fields)
        for fields in ({"case_id": "foreign_case"}, {"ehr_sha256": "b" * 64}, {"ehr_facts_sha256": "b" * 64}):
            with self.assertRaises(ValueError): partial.verify_image_phase(replace(image, **fields), a)

    def test_projection_refuses_shared_image_lineage_and_inventory_changes(self):
        c, _, _ = fixture()
        for key in ("case_id", "cxr_candidate_id"):
            changed = copy.deepcopy(c); changed["facts"][0][key] = "different"
            with self.assertRaises(ValueError): partial.image_evidence_from_legacy_candidate(changed, CACHE_HASH)
        changed = copy.deepcopy(c); changed["facts"].append(changed["facts"][0])
        with self.assertRaises(ValueError): partial.image_evidence_from_legacy_candidate(changed, CACHE_HASH)

    def test_tampering_even_with_rehashed_payload_is_refused(self):
        _, a, image = fixture(); r = partial.verify_image_phase(image, a)
        for key, val in (("clinical_acceptance", True), ("report_sha256", "b" * 64),
                         ("all_three_supported_facts", 0), ("unexpected_score", 1)):
            changed = copy.deepcopy(r); changed[key] = val
            changed["receipt_id"] = full._digest({k:v for k,v in changed.items() if k != "receipt_id"})
            with self.assertRaises(ValueError): partial.validate_partial_receipt(changed, a)
        changed = copy.deepcopy(r); changed["raw_edge_readouts"]["ehr_cxr"]["supported_facts"] = 99
        with self.assertRaises(ValueError): partial.validate_partial_receipt(changed, a)

    def test_link_preserves_partial_and_full_receipts_and_explicit_cache_scope(self):
        c, a, image = fixture(); p = partial.verify_image_phase(image, a); f = full.verify_candidate(c, a)
        before = copy.deepcopy((p, f)); link = partial.bind_completed_report(p, f, a)
        self.assertEqual((p, f), before)
        self.assertEqual(link["partial_receipt_id"], p["receipt_id"])
        self.assertEqual(link["completed_receipt_id"], f["receipt_id"])
        self.assertEqual(link["binding_mode"], "retrospective_phase_reconstruction_not_actual_execution_order")
        self.assertFalse(link["clinical_repair_success"])

    def test_json_false_zero_or_integer_float_substitution_is_not_hash_identical(self):
        _, a, image = fixture(); r = partial.verify_image_phase(image, a)
        for key, value in (("clinical_acceptance", 0), ("known_ehr_facts", 1.0)):
            altered = copy.deepcopy(r); altered[key] = value
            self.assertEqual(altered, r)  # Python conflates these representations.
            with self.assertRaises(ValueError): partial.validate_partial_receipt(altered, a)
            altered["receipt_id"] = full._digest({k:v for k,v in altered.items() if k != "receipt_id"})
            with self.assertRaises(ValueError): partial.validate_partial_receipt(altered, a)

    def test_completed_report_cannot_change_image_hash_state_or_identity(self):
        c, a, image = fixture(); p = partial.verify_image_phase(image, a)
        for change in ("hash", "state", "identity", "validity"):
            changed = copy.deepcopy(c)
            if change in ("hash", "identity"):
                key = "cxr_sha256" if change == "hash" else "cxr_candidate_id"
                value = "b" * 64 if change == "hash" else "different_image"
                changed["score_record"]["lineage"][key] = value
                for fact in changed["facts"]:
                    if change == "hash": fact["artifact_hashes"][key] = value
                    else: fact[key] = value
            elif change == "state":
                next(f for f in changed["facts"] if f["finding"] == "edema")["states"]["xrv"] = "negative"
            else: changed["score_record"]["scoring"]["modality_quality"]["cxr_basic_validity_pass"] = False
            with self.assertRaisesRegex(ValueError, "previously verified image"):
                partial.bind_completed_report(p, full.verify_candidate(changed, a), a)

    def test_full_cohort_determinism_coverage_and_source_immutability(self):
        s, d = invented(); bank = make_legacy_bank(s, d, policy()); complete = receipts(bank)
        before = copy.deepcopy((bank, complete))
        a = partial.bind_cached_image_phases(bank, complete, CACHE_HASH, config())
        b = partial.bind_cached_image_phases(dict(reversed(list(bank.items()))), list(reversed(complete)), CACHE_HASH, config())
        self.assertEqual(a, b); self.assertEqual((bank, complete), before)
        self.assertEqual(len(a["anchors"]), 3)
        self.assertEqual(len(a["partial_receipts"]), 9)
        self.assertEqual(len(a["report_bindings"]), 36)

    def test_missing_duplicate_or_altered_completed_inventory_refused(self):
        s, d = invented(); bank = make_legacy_bank(s, d, policy()); complete = receipts(bank)
        for altered in (complete[:-1], complete + [complete[0]], []):
            with self.assertRaises(ValueError): partial.bind_cached_image_phases(bank, altered, CACHE_HASH, config())
        altered = copy.deepcopy(complete); altered[0]["receipt_id"] = "b" * 64
        with self.assertRaises(ValueError): partial.bind_cached_image_phases(bank, altered, CACHE_HASH, config())

    def test_cached_image_disagreement_across_report_paths_is_refused(self):
        s, d = invented(); bank = make_legacy_bank(s, d, policy())
        c = bank["invented_case_0"][("chexgenbench_sana", 0, "cxrmate_single")]
        next(f for f in c["facts"] if f["finding"] == "edema")["states"]["xrv"] = "negative"
        with self.assertRaisesRegex(ValueError, "differs between report paths"):
            partial.bind_cached_image_phases(bank, receipts(bank), CACHE_HASH, config())

    def test_no_raw_bodies_patient_identifiers_or_paths_in_receipts(self):
        c, a, image = fixture(); p = partial.verify_image_phase(image, a)
        text = json.dumps([p, partial.bind_completed_report(p, full.verify_candidate(c, a), a)])
        for token in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(token, text)

    def test_cli_slurm_guard_before_cache_or_config_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "load_source") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
