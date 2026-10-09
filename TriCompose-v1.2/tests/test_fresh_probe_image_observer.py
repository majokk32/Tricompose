"""Invented metadata fixtures and callback contracts, no clinical artifacts."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"agent"))
import verify_fresh_probe_images as observer
from test_fresh_cxr_secondary import package
from tricompose_v12.invariant_verification import _digest


def sample():
    rows,_,_,_=package()
    for r in rows[:8]: r["cxr_sha256"]=_digest(["invented_old_image",r["case_id"]])
    triples=[{**{k:r[k] for k in ("case_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256",
        "cxr_model_id","report_model_id","seed")},"cxr_path":"/invented/"+r["cxr_candidate_id"]+".png"} for r in rows]
    images=observer.inventory(rows,triples)
    records=[{**{k:i[k] for k in ("cxr_candidate_id","cxr_sha256")},"contract_status":"complete",
        "states":dict.fromkeys(observer.existing.image_interface.FINDINGS,"negative")} for i in images]
    return rows,triples,records


class FreshProbeImageObserverTests(unittest.TestCase):
    def test_all_four_images_covered_deterministically_no_duplicate_report_votes(self):
        rows,triples,_=sample()
        images=observer.inventory(rows,triples)
        self.assertEqual(len(images),4)
        self.assertEqual(images,observer.inventory(list(reversed(rows)),list(reversed(triples))))
        self.assertTrue(all(set(i)=={"cxr_candidate_id","cxr_sha256","path"} for i in images))

    def test_missing_and_duplicate_lineage_rejected(self):
        rows,triples,_=sample()
        with self.assertRaises(ValueError): observer.inventory(rows[:-1],triples[:-1])
        triples[-1]=copy.deepcopy(triples[-2])
        with self.assertRaises(ValueError): observer.inventory(rows,triples)

    def test_shared_image_path_change_rejected(self):
        rows,triples,_=sample(); triples[1]["cxr_path"]="/invented/wrong.png"
        with self.assertRaises(ValueError): observer.inventory(rows,triples)

    def test_same_bytes_not_four_diverse_images(self):
        rows,triples,_=sample()
        for r,t in zip(rows,triples): r["cxr_sha256"]=t["cxr_sha256"]=_digest("invented_same_image")
        with self.assertRaises(ValueError): observer.inventory(rows,triples)

    def test_model_message_contains_image_and_original_prompt_only(self):
        sentinel=object()
        message=observer.existing.image_interface.request_messages("image",image=sentinel)
        content=message[0]["content"]
        self.assertEqual(content[0],{"type":"image","image":sentinel})
        self.assertEqual(content[1]["text"],observer.existing.image_interface.IMAGE_PROMPT.format(
            findings=", ".join(observer.existing.image_interface.FINDINGS)))
        self.assertEqual(len(content),2)

    def test_unique_image_finding_denominator_and_no_clinical_claim(self):
        rows,_,records=sample()
        result=observer.comparisons(records,rows)
        self.assertEqual(len(result),32)
        self.assertEqual(len({(r["cxr_candidate_id"],r["finding"]) for r in result}),32)
        self.assertTrue(all(r["clinical_fault_location"] is None and r["clinical_acceptance"] is False for r in result))

    def test_unavailable_kept_as_na_not_negative(self):
        rows,_,records=sample(); records[0].update(contract_status="failed_unavailable",states=None)
        result=observer.comparisons(records,rows)
        unavailable=[r for r in result if r["cxr_candidate_id"]==records[0]["cxr_candidate_id"]]
        self.assertEqual(len(unavailable),8)
        self.assertTrue(all(r["qwen_image_state"] is None and r["xrv_qwen_relation"]=="observer_unavailable" for r in unavailable))

    def test_partial_named_response_not_accepted(self):
        rows,_,records=sample(); records[0]["states"].pop("pneumonia")
        with self.assertRaises(ValueError): observer.comparisons(records,rows)

    def test_changed_image_or_shared_labels_rejected(self):
        rows,_,records=sample(); records[0]["cxr_sha256"]=_digest("invented_wrong_image")
        with self.assertRaises(ValueError): observer.comparisons(records,rows)
        rows,_,records=sample(); rows[1]["receipt"]["fact_states"][0]["xrv"]="uncertain"
        with self.assertRaises(ValueError): observer.comparisons(records,rows)

    def test_guards_precede_metadata_model_and_writes(self):
        with patch.object(observer.source.previous.gate,"cpu_guard",side_effect=RuntimeError("invented_login")), \
             patch.object(observer.source.previous.postflight,"MetadataReader") as read:
            with self.assertRaises(RuntimeError): observer.prepare(object())
            read.assert_not_called()
        with patch.object(observer,"gpu_guard",side_effect=RuntimeError("invented_cpu")), \
             patch.object(observer.source,"read_json") as read, patch.object(observer,"new_atomic_run") as output:
            with self.assertRaises(RuntimeError): observer.run(object())
            read.assert_not_called(); output.assert_not_called()


if __name__ == "__main__": unittest.main()
