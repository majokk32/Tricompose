"""Wholly invented strings: human CSV convenience is not automatic labeling."""
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import report_review_sheets as sheets
from tricompose_v12.report_review import make_queue, annotation_template, reader_agreement
from tricompose_v12.report_assertions import digest, checked_span
import report_review_sheets as cli


def fixture(count=1, text="虚构🫁\nCardiomegaly is present."):
    texts = [text+f"\nInvented sample {i}." for i in range(count)]
    items,_ = make_queue([{"candidate_id":f"invented_{i}","report_sha256":digest(value)}
        for i,value in enumerate(texts)])
    return items,{digest(value):value for value in texts}


def human_identity(items):
    return {**sheets.identity_template(items),"reviewer_alias":"reader_a",
        "reviewer_role":"non_expert","independence_attestation":{
            "model_predictions_seen":False,"winner_flags_seen":False,"labels_generated_by_model":False}}


def rows_of(text):
    return list(csv.DictReader(io.StringIO(text,newline="")))


def csv_of(rows, fields=sheets.FIELDS):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def reviewed_sheet(items, state="positive", reason="explicit_assertion", quote="Cardiomegaly is present."):
    rows=rows_of(sheets.sheet_template(items))
    rows[0].update(status="reviewed",state=state,reason=reason,quote_1=quote)
    return rows


class SheetRoundtripTests(unittest.TestCase):
    def test_blank_48_report_template_is_exact_192_pending_rows_no_source_access(self):
        items,_=fixture(48)
        source=Mock(side_effect=AssertionError("pending must not open text"))
        text=sheets.sheet_template(items)
        result=sheets.import_sheet(items,text,sheets.identity_template(items),read_source=source)
        self.assertEqual(result,annotation_template(items));source.assert_not_called()
        self.assertEqual(len(result["records"]),192)
        self.assertEqual(text,sheets.sheet_template(items))
        for forbidden in ("invented_0","model_id","winner","positive","negative","unknown"):
            self.assertNotIn(forbidden,text)

    def test_bom_crlf_and_reordered_rows_preserve_annotation(self):
        items,_=fixture(2)
        text="\ufeff"+csv_of(list(reversed(rows_of(sheets.sheet_template(items))))).replace("\n","\r\n")
        self.assertEqual(sheets.import_sheet(items,text,sheets.identity_template(items)),annotation_template(items))

    def test_import_does_not_mutate_inputs_or_replace_human_state(self):
        items,texts=fixture();identity=human_identity(items)
        # Deliberately linguistically inconsistent human label. The importer
        # checks provenance, not clinical truth or automatic corrections.
        rows=reviewed_sheet(items,state="negative")
        before=copy.deepcopy((items,texts,identity,rows))
        result=sheets.import_sheet(items,csv_of(rows),identity,read_source=texts.__getitem__)
        self.assertEqual(result["records"][0]["state"],"negative")
        self.assertEqual((items,texts,identity,rows),before)
        self.assertEqual(result,json.loads(json.dumps(result)))

    def test_reviewed_unknown_remains_unknown_not_negative(self):
        items,texts=fixture()
        rows=reviewed_sheet(items,state="unknown",reason="not_mentioned",quote="")
        result=sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=texts.__getitem__)
        self.assertEqual(result["records"][0]["state"],"unknown")
        self.assertEqual(result["records"][0]["evidence"],[])
        self.assertTrue(all(row["state"] is None for row in result["records"][1:]))

    def test_unassessable_is_not_a_four_state_label(self):
        items,texts=fixture();rows=rows_of(sheets.sheet_template(items))
        rows[0].update(status="unassessable",reason="unassessable")
        result=sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=texts.__getitem__)
        self.assertIsNone(result["records"][0]["state"])

    def test_unknown_is_reviewed_even_without_quotes_requires_identity_and_source(self):
        items,_=fixture();rows=reviewed_sheet(items,state="unknown",reason="not_mentioned",quote="")
        with self.assertRaises(ValueError):
            sheets.import_sheet(items,csv_of(rows),sheets.identity_template(items))
        with self.assertRaisesRegex(ValueError,"source verification"):
            sheets.import_sheet(items,csv_of(rows),human_identity(items))

    def test_import_never_turns_empty_returns_into_accuracy(self):
        items,_=fixture();text=sheets.sheet_template(items)
        a=sheets.import_sheet(items,text,sheets.identity_template(items))
        result=reader_agreement(items,a,copy.deepcopy(a))
        self.assertEqual(result["both_reviewed"],0)
        self.assertIsNone(result["exact_reader_agreement"])
        self.assertIsNone(result["cohens_kappa_nominal_four_states"])
        self.assertFalse(result["adjudicated_gold_created"])

    def test_missing_duplicate_and_foreign_rows_refused(self):
        items,_=fixture();rows=rows_of(sheets.sheet_template(items))
        for bad in (rows[:-1],rows+[rows[0]], [{**row,"item_id":"foreign"} for row in rows]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                sheets.import_sheet(items,csv_of(bad),sheets.identity_template(items))

    def test_extra_duplicate_rearranged_and_missing_columns_refused(self):
        items,_=fixture();rows=rows_of(sheets.sheet_template(items))
        for fields in ((*sheets.FIELDS,"extra"),(*sheets.FIELDS,"status"),tuple(reversed(sheets.FIELDS))):
            with self.subTest(fields=fields),self.assertRaisesRegex(ValueError,"columns"):
                sheets.import_sheet(items,csv_of(rows,fields),sheets.identity_template(items))
        text=sheets.sheet_template(items).replace(",quote_2_occurrence\n","\n",1)
        with self.assertRaisesRegex(ValueError,"columns"):
            sheets.import_sheet(items,text,sheets.identity_template(items))

    def test_missing_or_extra_cells_refused(self):
        items,_=fixture();lines=sheets.sheet_template(items).splitlines()
        for line in (lines[1].rsplit(",",1)[0],lines[1]+",extra"):
            with self.assertRaisesRegex(ValueError,"malformed"):
                sheets.import_sheet(items,"\n".join([lines[0],line,*lines[2:]])+"\n",sheets.identity_template(items))

    def test_changed_source_hash_or_identity_header_refused(self):
        items,_=fixture();rows=rows_of(sheets.sheet_template(items))
        rows[0]["report_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"hash"):
            sheets.import_sheet(items,csv_of(rows),sheets.identity_template(items))
        for key in ("schema_version","inventory_sha256"):
            identity={**sheets.identity_template(items),key:"wrong"}
            with self.assertRaisesRegex(ValueError,"identity"):
                sheets.import_sheet(items,sheets.sheet_template(items),identity)
        identity={**sheets.identity_template(items),"extra":False}
        with self.assertRaisesRegex(ValueError,"identity"):
            sheets.import_sheet(items,sheets.sheet_template(items),identity)

    def test_pending_with_state_reason_or_evidence_is_refused(self):
        items,texts=fixture()
        for field,value in (("state","unknown"),("reason","not_mentioned"),("quote_1","Cardiomegaly is present.")):
            rows=rows_of(sheets.sheet_template(items));rows[0][field]=value
            with self.assertRaisesRegex(ValueError,"pending"):
                sheets.import_sheet(items,csv_of(rows),sheets.identity_template(items),read_source=texts.__getitem__)

    def test_missing_alias_nonhuman_role_or_model_attestation_refused(self):
        items,texts=fixture();text=csv_of(reviewed_sheet(items))
        for field,value in (("reviewer_alias",None),("reviewer_alias","person@example.com"),
                ("reviewer_role","LLM"),("independence_attestation",None)):
            identity={**human_identity(items),field:value}
            with self.assertRaises(ValueError):
                sheets.import_sheet(items,text,identity,read_source=texts.__getitem__)
        for key in human_identity(items)["independence_attestation"]:
            identity=human_identity(items);identity["independence_attestation"][key]=True
            with self.assertRaisesRegex(ValueError,"attestation"):
                sheets.import_sheet(items,text,identity,read_source=texts.__getitem__)


class EvidenceTests(unittest.TestCase):
    def test_unique_unicode_quote_has_exact_codepoint_offsets_hash(self):
        text="虚构🫁\nCardiomegaly is present."
        quote="Cardiomegaly is present."
        span=sheets.locate_human_quote(text,quote,"")
        self.assertEqual(span,checked_span(text,4,len(text)))
        self.assertEqual(span["offset_unit"],"unicode_codepoint")

    def test_repeated_quote_requires_human_occurrence_not_automatic_first_match(self):
        text="No effusion.\nNo effusion."
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            sheets.locate_human_quote(text,"No effusion.","")
        self.assertEqual(sheets.locate_human_quote(text,"No effusion.","1")["char_start"],13)

    def test_overlapping_occurrences_are_explicit(self):
        self.assertEqual(sheets.locate_human_quote("aaaa","aaa","1")["char_start"],1)

    def test_invalid_or_outside_occurrence_refused(self):
        for value in ("-1","1.0","01"," 0","0 ","2",True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                sheets.locate_human_quote("same same","same",value)

    def test_no_quote_with_occurrence_refused(self):
        with self.assertRaisesRegex(ValueError,"without quote"):
            sheets.locate_human_quote("invented","","0")
        self.assertIsNone(sheets.locate_human_quote("invented","",""))

    def test_paraphrases_and_whitespace_normalization_are_not_exact_evidence(self):
        for quote in ("The heart is enlarged.","No effusion.","No  effusion."):
            with self.assertRaisesRegex(ValueError,"exact source"):
                sheets.locate_human_quote("No\neffusion.",quote,"")

    def test_exact_quoted_csv_newline_comma_and_quotes_roundtrip(self):
        quote='虚构\n"Cardiomegaly", uncertain.'
        items,texts=fixture(text=quote)
        rows=reviewed_sheet(items,state="uncertain",reason="qualified_or_conflicting",quote=quote)
        result=sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=texts.__getitem__)
        self.assertEqual(result["records"][0]["evidence"][0]["quote"],quote)

    def test_import_cli_preserves_crlf_inside_exact_human_quote(self):
        quote="Cardiomegaly is present.\r\nNo effusion."
        items,texts=fixture(text=quote)
        sheet=csv_of(reviewed_sheet(items,quote=quote)).encode("utf-8")
        args=SimpleNamespace(mode="import",bundle_run="invented_bundle",
            sheet_file="invented_sheet.csv",identity_file="invented_identity.json")
        with patch.object(cli,"load_bundle",return_value=(items,texts.__getitem__,{})),\
                patch.object(cli,"require_inside",side_effect=lambda path,*a,**k:Path(path)),\
                patch.object(Path,"stat",return_value=SimpleNamespace(st_size=len(sheet))),\
                patch.object(Path,"read_bytes",return_value=sheet),\
                patch.object(cli,"read_json",return_value=human_identity(items)),\
                patch.object(cli,"write_private_json",side_effect=lambda path,value:path) as written:
            summary,_,_=cli.run(args,Path("invented_output"))
        annotation=written.call_args.args[1]
        self.assertEqual(annotation["records"][0]["evidence"][0]["quote"],quote)
        self.assertEqual(summary["human_reviewed_rows"],1)
        self.assertFalse(summary["human_states_changed_by_importer"])

    def test_missing_source_changed_source_and_duplicate_evidence_refused(self):
        items,texts=fixture();rows=reviewed_sheet(items)
        with self.assertRaisesRegex(ValueError,"source access"):
            sheets.import_sheet(items,csv_of(rows),human_identity(items))
        with self.assertRaisesRegex(ValueError,"source hash"):
            sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=lambda _:"changed")
        rows[0]["quote_2"]=rows[0]["quote_1"]
        with self.assertRaisesRegex(ValueError,"duplicate human evidence"):
            sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=texts.__getitem__)

    def test_two_distinct_quotes_preserve_human_selection(self):
        items,texts=fixture(text="Cardiomegaly is present.\nNo effusion.")
        rows=reviewed_sheet(items);rows[0]["quote_2"]="No effusion."
        result=sheets.import_sheet(items,csv_of(rows),human_identity(items),read_source=texts.__getitem__)
        self.assertEqual(len(result["records"][0]["evidence"]),2)


class RenderingSafetyTests(unittest.TestCase):
    def test_report_book_preserves_text_hashes_and_uses_opaque_ids(self):
        items,texts=fixture(2)
        book=sheets.report_book(items,texts)
        for row in items:
            self.assertIn("## "+row["item_id"],book)
            self.assertIn(texts[row["report_sha256"]],book)
        self.assertNotIn("candidate_id",book);self.assertNotIn("model_id",book)
        self.assertEqual(book,sheets.report_book(items,texts))

    def test_backtick_and_html_like_generated_text_stays_inside_longer_fence(self):
        items,texts=fixture(text="````\n<script>invented</script>\n# not a header")
        book=sheets.report_book(items,texts)
        self.assertIn("`````text\n",book)
        self.assertIn(next(iter(texts.values()))+"\n`````",book)

    def test_book_refuses_changed_text_not_normalizing_it(self):
        items,texts=fixture()
        with self.assertRaisesRegex(ValueError,"source hash"):
            sheets.report_book(items,{key:value+" " for key,value in texts.items()})

    def test_export_requires_slurm_before_loading_any_bundle(self):
        with patch.dict(os.environ,{},clear=True),patch.object(cli,"load_bundle") as access:
            with self.assertRaisesRegex(RuntimeError,"Slurm"):
                cli.run(SimpleNamespace(mode="export"),Path("not_written"))
            access.assert_not_called()

    def test_bundle_reader_requires_slurm_before_report_read_and_validates_hash(self):
        items,texts=fixture(48)
        summary={"reports":48,"synthetic_only":True,"real_inputs_read":False,
            "copied_reports_byte_identical":True,"independent_ehr_cases":2,"regeneration_authorized":False}
        artifacts={row["report_relative_path"]:{"sha256":row["report_sha256"]} for row in items}
        artifacts.update({name:{"sha256":"metadata"} for name in ("items.json","summary.json")})
        manifest={"schema_version":"tricompose-blinded-synthetic-report-copies-v1","artifacts":artifacts}
        def read(path):
            return {"manifest.json":manifest,"items.json":{"items":items},"summary.json":summary}[path.name]
        with patch.object(cli,"require_inside",side_effect=lambda path,*a,**k:Path(path)),\
                patch.object(cli,"read_json",side_effect=read),patch.object(cli,"sha256_file",return_value="metadata"):
            _,reader,_=cli.load_bundle("invented_bundle")
        h=items[0]["report_sha256"]
        with patch.dict(os.environ,{},clear=True),patch.object(Path,"read_bytes") as access:
            with self.assertRaisesRegex(RuntimeError,"Slurm"):reader(h)
            access.assert_not_called()
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}),\
                patch.object(Path,"stat",return_value=SimpleNamespace(st_size=100)),\
                patch.object(Path,"read_bytes",return_value=texts[h].encode("utf-8")):
            self.assertEqual(reader(h),texts[h])
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented"}),\
                patch.object(Path,"stat",return_value=SimpleNamespace(st_size=100)),\
                patch.object(Path,"read_bytes",return_value=b"changed"):
            with self.assertRaisesRegex(ValueError,"text/hash"):reader(h)


if __name__ == "__main__":
    unittest.main()
