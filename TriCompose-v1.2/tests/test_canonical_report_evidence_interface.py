"""Invented fixtures only; no patient text, checkpoint, GPU or inference."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import canonical_report_evidence_interface as interface


def response(finding="pleural_effusion", polarity="positive", quotes=()):
    obj = {name: {state: [] for state in interface.POLARITIES} for name in interface.FINDINGS}
    obj[finding][polarity] = list(quotes)
    return json.dumps(obj)


def cached(text, result, limited=False):
    return {"report_sha256": interface.digest_text(text), "response": result, "token_limit_reached": limited}


IDENTITY = {"identity_sha256": "invented_frozen_run_identity"}


class CanonicalTextTests(unittest.TestCase):
    def test_horizontal_only_preserves_real_lines(self):
        text = "  Findings:\t\nNo\u00a0\u00a0pneumothorax\r\nPleural  effusion is present.  "
        obj = interface.canonicalize_report(text)
        self.assertEqual(obj["text"], " Findings: \nNo pneumothorax\r\nPleural effusion is present. ")
        self.assertEqual(obj["source_sha256"], interface.digest_text(text))
        self.assertEqual(obj["canonical_sha256"], interface.digest_text(obj["text"]))

    def test_all_vertical_and_record_separators_are_preserved(self):
        for char in interface.VERTICAL_WHITESPACE:
            with self.subTest(char=repr(char)):
                self.assertEqual(interface.canonicalize_report("X  "+char+"  Y")["text"], "X "+char+" Y")

    def test_no_stripping_and_no_flattening(self):
        self.assertEqual(interface.canonicalize_report("\n  X  \n")["text"], "\n X \n")
        self.assertNotEqual(interface.canonicalize_report("X\nY")["text"], interface.canonicalize_report("X Y")["text"])

    def test_idempotent_text_and_hash(self):
        first = interface.canonicalize_report("X\t \nY\u00a0Z")
        second = interface.canonicalize_report(first["text"])
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["canonical_sha256"], second["canonical_sha256"])

    def test_unicode_offset_mapping_reconstructs_every_character(self):
        source = "🫁\t  中文\nNo\u00a0effusion."
        obj = interface.canonicalize_report(source)
        self.assertEqual(len(obj["text"]), len(obj["offsets"]))
        for char, (left, right) in zip(obj["text"], obj["offsets"]):
            fragment = source[left:right]
            self.assertEqual(char, " " if all(interface.horizontal(c) for c in fragment) else fragment)

    def test_case_numbers_units_punctuation_negation_are_unchanged(self):
        source = "No Pneumothorax!\nTube tip is 2 cm; small effusion?"
        self.assertEqual(interface.canonicalize_report(source)["text"], source)

    def test_invalid_source_refused(self):
        for source in (None, "", "  \n", "x"*8193):
            with self.assertRaises(ValueError):
                interface.canonicalize_report(source)

    def test_same_horizontal_variants_group_but_disease_flip_does_not(self):
        values = ["No pleural effusion.", "No  pleural  effusion.", "Pleural effusion is present."]
        inputs, mappings = interface.build_input_plan({interface.digest_text(text): text for text in values})
        self.assertEqual(len(inputs), 2)
        self.assertEqual(sorted(len(row["source_sha256s"]) for row in inputs.values()), [1, 2])
        self.assertEqual(len(mappings), 3)

    def test_empty_excess_and_forged_inventory_rejected(self):
        for texts in ({}, {"wrong_hash": "Invented report."}, {str(i): "Invented report." for i in range(101)}):
            with self.assertRaises(ValueError):
                interface.build_input_plan(texts)

    def test_group_summary_has_json_stable_string_keys(self):
        inputs = {"a": {"source_sha256s": ["x", "y"]}, "b": {"source_sha256s": ["z"]}}
        result = interface.group_size_distribution(inputs)
        self.assertEqual(result, {"1": 1, "2": 1})
        self.assertEqual(result, json.loads(json.dumps(result)))


class CacheProjectionTests(unittest.TestCase):
    def test_exact_input_cache_reuse_restores_original_offsets(self):
        canonical = "Intro.\nNo pleural effusion."
        spaced = "Intro.\nNo\t  pleural  effusion."
        texts = {interface.digest_text(text): text for text in (canonical, spaced)}
        sha = interface.digest_text(canonical)
        responses = {sha: cached(canonical, response(polarity="negative", quotes=["No pleural effusion."]))}
        inputs, mappings, stages = interface.materialize_blinded(texts, responses, IDENTITY)
        self.assertEqual(len(inputs), 1)
        for fingerprint, text in texts.items():
            row = stages["canonical_input_guarded"][fingerprint]
            self.assertEqual(row["findings"]["pleural_effusion"]["state"], "negative")
            self.assertFalse(row["new_model_call"])
            self.assertEqual(row["canonical_input_sha256"], sha)
            span = row["findings"]["pleural_effusion"]["evidence"]["negative"][0]
            self.assertEqual(text[span["char_start"]:span["char_end"]], span["quote"])
            self.assertTrue(span["inverse_offset_verified"])
        self.assertEqual(stages["canonical_input_guarded"][interface.digest_text(canonical)]["cache_key_sha256"], stages["canonical_input_guarded"][interface.digest_text(spaced)]["cache_key_sha256"])

    def test_paraphrase_and_cache_miss_refused_not_fallback(self):
        text = "Pleural effusion is present."
        other = "Effusion is seen."
        with self.assertRaises(interface.CacheMissError):
            interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(other): cached(other, response())}, IDENTITY)

    def test_wrong_response_input_binding_rejected(self):
        text = "No pleural effusion."
        row = cached(text, response())
        row["report_sha256"] = "forged"
        with self.assertRaises(ValueError):
            interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(text): row}, IDENTITY)

    def test_malformed_cached_response_retains_all_unknown(self):
        text = "No pleural effusion."
        _, _, stages = interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(text): cached(text, "not json")}, IDENTITY)
        row = stages["canonical_input_guarded"][interface.digest_text(text)]
        self.assertEqual(row["contract_status"], "failed_unavailable")
        self.assertTrue(all(x["state"] == "unknown" for x in row["findings"].values()))

    def test_token_limit_failure_is_not_repaired(self):
        text = "No pleural effusion."
        _, _, stages = interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(text): cached(text, response(), True)}, IDENTITY)
        self.assertEqual(stages["canonical_input_aligned"][interface.digest_text(text)]["contract_failure_reason"], "token_limit_reached")

    def test_bad_quote_is_not_paraphrased_or_filled(self):
        text = "No pleural effusion."
        _, _, stages = interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(text): cached(text, response(quotes=["Effusion is absent."]))}, IDENTITY)
        self.assertEqual(stages["canonical_input_aligned"][interface.digest_text(text)]["contract_status"], "failed_unavailable")

    def test_forged_inverse_map_is_rejected(self):
        text = "No  pleural effusion."
        inputs, mappings = interface.build_input_plan({interface.digest_text(text): text})
        mapping = mappings[interface.digest_text(text)]
        mapping["offsets"][0][0] = 1
        with self.assertRaises(ValueError):
            interface.project_response(response(), text, next(iter(inputs.values())), mapping)

    def test_original_line_scope_and_no_polarity_flip(self):
        text = "No pneumothorax\nPleural effusion is present."
        inputs, mappings = interface.build_input_plan({interface.digest_text(text): text})
        _, guarded = interface.project_response(response(quotes=["Pleural effusion is present."]), text, next(iter(inputs.values())), mappings[interface.digest_text(text)])
        self.assertEqual(guarded["findings"]["pleural_effusion"]["state"], "positive")
        _, guarded = interface.project_response(response(finding="pneumothorax", quotes=["No pneumothorax"]), text, next(iter(inputs.values())), mappings[interface.digest_text(text)])
        self.assertEqual(guarded["findings"]["pneumothorax"]["state"], "unknown")
        self.assertEqual(guarded["findings"]["pneumothorax"]["evidence"]["negative"], [])

    def test_deterministic_and_inputs_immutable(self):
        text = "No pleural effusion."
        texts = {interface.digest_text(text): text}
        responses = {interface.digest_text(text): cached(text, response())}
        before = copy.deepcopy((texts, responses))
        self.assertEqual(interface.materialize_blinded(texts, responses, IDENTITY), interface.materialize_blinded(texts, responses, IDENTITY))
        self.assertEqual((texts, responses), before)

    def test_missing_finding_not_negative(self):
        text = "Invented report."
        _, _, stages = interface.materialize_blinded({interface.digest_text(text): text}, {interface.digest_text(text): cached(text, response())}, IDENTITY)
        self.assertTrue(all(row["state"] == "unknown" for row in stages["canonical_input_guarded"][interface.digest_text(text)]["findings"].values()))


class SeparationTests(unittest.TestCase):
    def test_key_analysis_occurs_after_blinded_materialization(self):
        visited = []
        original = interface.materialize_blinded
        text = "No pleural effusion."
        sha = interface.digest_text(text)
        raw = {sha: cached(text, response())}
        def materialize(*args):
            result = original(*args)
            visited.append("materialized")
            return result
        def analyze(args):
            self.assertEqual(visited, ["materialized"])
            raise RuntimeError("invented evaluator stop")
        with patch.object(interface, "load_blinded_cache", return_value=({sha: text}, [], {}, raw, {}, {})), patch.object(interface, "checked_cache_identity", return_value=IDENTITY), patch.object(interface, "materialize_blinded", side_effect=materialize), patch.object(interface, "analyze_strict", side_effect=analyze):
            with self.assertRaisesRegex(RuntimeError, "invented evaluator stop"):
                interface.run(SimpleNamespace())

    def test_nonoverwrite_refused_before_source_read(self):
        argv = ["interface", "--bank-run", "b", "--review-run", "r", "--context-run", "c", "--repair-run", "p", "--output-root", "o", "--run-id", "i"]
        with patch.object(sys, "argv", argv), patch.object(interface, "new_atomic_run", side_effect=FileExistsError), patch.object(interface, "run") as run, patch("builtins.print"):
            self.assertEqual(interface.main(), 1)
            run.assert_not_called()

    def test_producer_prompt_revision_and_sampling_are_pinned(self):
        producer = {"prompt_sha256": interface.digest_text(interface.PROMPT), "max_new_tokens": 512, "do_sample": False, "frozen": True}
        review = {"producer": producer, "execution": {"dtype": "invented"}}
        sources = {"review": Path("invented_review"), "review_manifest": Path("invented_manifest")}
        def read(path):
            return review if path == sources["review"] else {"program_sha256": "pinned"}
        with patch.object(interface, "read_json", side_effect=read), patch.object(interface, "sha256_file", return_value="pinned"):
            identity = interface.checked_cache_identity(SimpleNamespace(), sources)
            self.assertIn("identity_sha256", identity)
            for field, bad in (("prompt_sha256", "changed"), ("max_new_tokens", 1024), ("do_sample", True), ("frozen", False)):
                old = producer[field]
                producer[field] = bad
                with self.assertRaises(ValueError):
                    interface.checked_cache_identity(SimpleNamespace(), sources)
                producer[field] = old


if __name__ == "__main__":
    unittest.main()
