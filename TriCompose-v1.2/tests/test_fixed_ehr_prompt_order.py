"""Invented metadata/text fixtures only; no model, image, source or network IO."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_fixed_ehr_prompt_order as probe
from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v11.facts import extract_v11_facts


def facts_fixture(problems=None):
    canonical = canonicalize_synehrgy_case({"schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
        "case_id": "case_000", "structure": {"top_level_tokens": [], "visits": [{"covariates": [],
            "problems": problems if problems is not None else ["J18_Pneumonia", "I50_Heart failure"],
            "labs": [], "charts": [], "other_tokens": []}]}, "validation": {"strict_valid": True}},
        source_model_id="synehrgy_gpt2_10bins")
    return extract_v11_facts(canonical)


def legacy_rendering(facts, model):
    """Authored fixture, NOT reconstructed real historical prompt content."""
    rendered = probe.render_v11_prompt(facts, model)
    rendered["renderer_version"] = probe.LEGACY_RENDERERS[model]
    if rendered["included_direct_fact_ids"] and facts["direct_facts"]["congestive_heart_failure"]["state"] == "positive":
        rendered["included_direct_fact_ids"] = list(probe.LEGACY_FACTS)
        rendered["derived_rule_ids"] = list(probe.LEGACY_RULES)
        rendered["text"] += " Heart failure-related imaging prior."
        rendered["prompt_sha256"] = probe.text_hash(rendered["text"])
    return rendered


def plan_fixture():
    cases = []
    for index,case_id in enumerate(probe.CASES):
        models = {}
        for model in probe.MODELS:
            models[model] = {"proof": {"clinical_intent_sha256": f"{index+11:064x}"},
                "prompts": {arm: {"path": f"cases/{case_id}/{model}/{arm}.txt",
                    "sha256": probe.text_hash(case_id+model+arm)} for arm in probe.ARMS}}
        cases.append({"case_id": case_id, "ehr_sha256": f"{index+1:064x}",
            "ehr_facts_sha256": f"{index+21:064x}", "models": models})
    return {"cases": cases, "slots": probe.slots(cases)}


def readout_fixture():
    plan = plan_fixture(); generation = []; predictions = []
    for i,slot in enumerate(plan["slots"]):
        image = f"{i+101:064x}"
        generation.append({**slot, "status": "completed", "image_sha256": image,
            "input_ids_sha256": "a"*64 if slot["arm"] == "original" else "b"*64})
        predictions.append({"slot_id": slot["slot_id"], "image_sha256": image,
            "scores": {"pneumonia": .2 if slot["arm"] == "original" else .6,
                "consolidation": .3 if slot["arm"] == "original" else .4}})
    return plan,generation,predictions


class FixedEHRPromptOrderTests(unittest.TestCase):
    def test_exact_fixed_cartesian_product_sixteen_slots(self):
        slots = plan_fixture()["slots"]
        self.assertEqual(len(slots),16)
        self.assertEqual(len({s["slot_id"] for s in slots}),16)
        for case in probe.CASES:
            for model in probe.MODELS:
                for seed in probe.SEEDS:
                    self.assertEqual({s["arm"] for s in slots if (s["case_id"],s["model"],s["seed"]) == (case,model,seed)},set(probe.ARMS))

    def test_same_ehr_and_intent_on_each_pair(self):
        slots = plan_fixture()["slots"]
        for a,b in zip(slots[::2],slots[1::2],strict=True):
            for field in ("case_id","model","seed","ehr_sha256","ehr_facts_sha256","clinical_intent_sha256"):
                self.assertEqual(a[field],b[field])

    def test_cannot_change_or_replace_fixed_case_order(self):
        cases = plan_fixture()["cases"]
        for bad in (cases[::-1],cases[:1]):
            with self.assertRaises(ValueError): probe.slots(bad)

    def test_roentgen_moves_exact_existing_block_no_word_changes(self):
        rendered = legacy_rendering(facts_fixture(),"roentgen_v2")
        before = deepcopy(rendered); result = probe.reorder_rendering(rendered)
        self.assertEqual(result["findings_first"],result["finding_block"]+" "+result["unchanged_nonfinding_prefix"])
        self.assertEqual(Counter(result["original"].split()),Counter(result["findings_first"].split()))
        self.assertTrue(result["original"].endswith(result["finding_block"]))
        self.assertEqual(rendered,before)

    def test_sana_preserves_every_existing_clause_and_case(self):
        rendered = legacy_rendering(facts_fixture(),"chexgenbench_sana")
        result = probe.reorder_rendering(rendered)
        self.assertTrue(result["findings_first"].startswith("Radiographic findings:"))
        self.assertEqual(result["finding_block"]+" "+result["unchanged_nonfinding_prefix"],result["findings_first"])
        self.assertEqual(Counter(result["original"].split()),Counter(result["findings_first"].split()))

    def test_existing_protocol_and_context_words_not_removed(self):
        for model in probe.MODELS:
            result = probe.reorder_rendering(legacy_rendering(facts_fixture(),model))
            for phrase in ("PA chest radiograph", "heart failure"):
                self.assertIn(phrase,result["original"])
                self.assertIn(phrase,result["findings_first"])

    def test_unknown_negative_uncertain_are_not_added(self):
        facts = facts_fixture(["J18_Pneumonia","Z_No pneumothorax","Z_Possible pleural effusion"])
        for model in probe.MODELS:
            result = probe.reorder_rendering(legacy_rendering(facts,model))
            for term in ("pneumothorax","pleural effusion","consolidation","focal","device"):
                self.assertNotIn(term,result["findings_first"].lower())

    def test_no_new_attributes_or_negative_descriptions(self):
        for model in probe.MODELS:
            result = probe.reorder_rendering(legacy_rendering(facts_fixture(),model))
            for term in ("left","right","bilateral","mild","severe","small","large","no pneumonia"):
                self.assertNotIn(term,result["findings_first"].lower())

    def test_neutral_case_rejected_not_replaced_or_invented(self):
        for model in probe.MODELS:
            with self.assertRaises(ValueError):
                probe.reorder_rendering(legacy_rendering(facts_fixture([]),model))

    def test_changed_source_hash_or_finding_block_rejected(self):
        for field,value in (("prompt_sha256","c"*64),("text","Changed unrelated text.")):
            rendered = legacy_rendering(facts_fixture(),"roentgen_v2"); rendered[field]=value
            with self.assertRaises(ValueError): probe.reorder_rendering(rendered)

    def test_renderer_context_separation_and_final_input_required(self):
        for field in ("is_final_model_input","context_is_not_a_radiographic_assertion"):
            rendered = legacy_rendering(facts_fixture(),"roentgen_v2"); rendered[field]=False
            with self.assertRaises(ValueError): probe.reorder_rendering(rendered)

    def test_provenance_retains_derived_rule_as_prior(self):
        facts = facts_fixture(); rendered=legacy_rendering(facts,"roentgen_v2")
        proof=probe.grounded_proof(facts,rendered)
        self.assertEqual(proof["included_direct_fact_ids"],probe.LEGACY_FACTS)
        self.assertEqual(proof["derived_rule_ids"],rendered["derived_rule_ids"])
        self.assertTrue(proof["inherited_derived_rule_not_new_image_evidence"])
        self.assertFalse(proof["legacy_image_assertion_validity_established"])
        self.assertTrue(proof["legacy_chf_is_clinical_context_not_independent_image_evidence"])
        self.assertTrue(proof["direct_fact_evidence"]["pneumonia"]["source_fields"])

    def test_ungrounded_or_nonpositive_included_fact_rejected(self):
        rendered=legacy_rendering(facts_fixture(),"roentgen_v2")
        for field,value in (("evidence",[]),("source_fields",[]),("state","unknown")):
            facts=facts_fixture(); facts["direct_facts"]["pneumonia"][field]=value
            with self.assertRaises(ValueError): probe.grounded_proof(facts,rendered)

    def test_no_report_llm_api_training_or_retry(self):
        self.assertFalse(probe.CONFIG["report_generation"])
        self.assertFalse(probe.CONFIG["training_allowed"])
        self.assertEqual(probe.CONFIG["external_api_calls"],0)
        self.assertEqual(probe.CONFIG["model_retries"],0)
        self.assertEqual(probe.CONFIG["maximum_inference_attempts"],32)

    def test_current_renderer_not_silently_substituted(self):
        for model in probe.MODELS:
            with self.assertRaises(ValueError):
                probe.reorder_rendering(probe.render_v11_prompt(facts_fixture(),model))

    def test_duplicate_or_missing_finding_marker_rejected(self):
        for model in probe.MODELS:
            for duplicate in (False,True):
                rendered=legacy_rendering(facts_fixture(),model)
                marker="Findings:" if model=="roentgen_v2" else "Radiographic findings:"
                rendered["text"]=rendered["text"]+" "+marker if duplicate else rendered["text"].replace(marker,"Other:")
                rendered["prompt_sha256"]=probe.text_hash(rendered["text"])
                with self.assertRaises(ValueError): probe.reorder_rendering(rendered)

    def test_eight_pairs_correct_delta_direction_and_no_clinical_claim(self):
        plan,g,p=readout_fixture(); before=deepcopy((plan,g,p)); result=probe.paired_readout(plan,g,p)
        self.assertEqual(len(result),8)
        self.assertTrue(all(r["image_bytes_differ"] and r["tokenizer_ids_differ"] for r in result))
        self.assertAlmostEqual(result[0]["pneumonia_delta_findings_first_minus_original"],.4)
        self.assertTrue(all(r["clinical_accuracy"] is None and r["clinical_acceptance"] is False for r in result))
        self.assertEqual((plan,g,p),before)

    def test_identical_images_and_ids_not_hidden(self):
        plan,g,p=readout_fixture(); g[1]["image_sha256"]=g[0]["image_sha256"]
        g[1]["input_ids_sha256"]=g[0]["input_ids_sha256"]; p[1]["image_sha256"]=g[1]["image_sha256"]
        row=probe.paired_readout(plan,g,p)[0]
        self.assertFalse(row["image_bytes_differ"]); self.assertFalse(row["tokenizer_ids_differ"])

    def test_failed_slot_stays_with_null_delta(self):
        plan,g,p=readout_fixture(); g[0]={**plan["slots"][0],"status":"failed_charged"}
        result=probe.paired_readout(plan,g,p[1:])
        self.assertEqual(len(result),8)
        self.assertIsNone(result[0]["pneumonia_delta_findings_first_minus_original"])
        self.assertIsNone(result[0]["image_bytes_differ"])

    def test_missing_all_scores_is_na_not_zero(self):
        plan,g,p=readout_fixture(); result=probe.paired_readout(plan,g,[])
        self.assertTrue(all(r["pneumonia_delta_findings_first_minus_original"] is None for r in result))

    def test_unavailable_finding_stays_null(self):
        plan,g,p=readout_fixture(); p[0]["scores"]["pneumonia"]=None
        self.assertIsNone(probe.paired_readout(plan,g,p)[0]["original_pneumonia_score"])

    def test_dropped_duplicate_or_crosscase_generation_rejected(self):
        plan,g,p=readout_fixture()
        for bad in (g[:-1],[g[0]]*16):
            with self.assertRaises(ValueError): probe.paired_readout(plan,bad,p)
        for field,value in (("ehr_sha256","c"*64),("case_id",probe.CASES[1]),("seed",7),("prompt_sha256","c"*64)):
            plan,g,p=readout_fixture(); g[0][field]=value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)

    def test_cross_image_unknown_slot_or_duplicate_score_rejected(self):
        for field,value in (("image_sha256","c"*64),("slot_id","undeclared")):
            plan,g,p=readout_fixture(); p[0][field]=value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)
        plan,g,p=readout_fixture()
        with self.assertRaises(ValueError): probe.paired_readout(plan,g,p+[p[0]])

    def test_invalid_score_rejected(self):
        for value in (True,float("nan"),float("inf"),-1,2):
            plan,g,p=readout_fixture(); p[0]["scores"]["pneumonia"]=value
            with self.assertRaises(ValueError): probe.paired_readout(plan,g,p)

    def test_cpu_guard_before_source_reads(self):
        with patch.object(probe.native,"cpu_guard",side_effect=RuntimeError("blocked")),patch.object(probe,"Reader") as reader:
            with self.assertRaises(RuntimeError): probe.prepare(SimpleNamespace())
            reader.assert_not_called()

    def test_gpu_guard_before_plan_loading(self):
        with patch.object(probe.native,"gpu_guard",side_effect=RuntimeError("blocked")),patch.object(probe,"load_plan") as load:
            with self.assertRaises(RuntimeError): probe.run(SimpleNamespace())
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
