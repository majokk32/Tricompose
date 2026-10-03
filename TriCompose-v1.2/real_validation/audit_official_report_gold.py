#!/usr/bin/env python3
"""Approved CPU Slurm: label/schema/link coverage only, never report text.

Study/patient keys exist only transiently for a metadata join. Outputs contain
aggregate counts and hashes, no patient keys, rows, paths or gold examples.
This is not model evaluation, training-overlap clearance, or image truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE=Path("/project2/ruishanl_1185/inference_3mod")
PROJECT=WORKSPACE.parent
sys.path.insert(0,str(WORKSPACE/"TriCompose-v1.0/eval/report_v1_1"))
from contracts import (new_atomic_run,commit_atomic_run,discard_atomic_run,
    write_private_json,write_private_text,sha256_file)

SCHEMA="tricompose-official-report-gold-coverage-audit-v1"
CONDITIONS=("Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Effusion","Pneumonia","Pneumothorax","Pleural Other","Support Devices","No Finding")
# Observed local manual-source schema. Preserve this name; do NOT silently
# relabel Airspace Opacity to the classifier's broader Lung Opacity head.
MANUAL_SOURCE_CONDITIONS=tuple("Airspace Opacity" if name=="Lung Opacity" else name for name in CONDITIONS)
STATES=("positive","negative","uncertain","unknown")
EIGHT_FINDINGS=("Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Lung Opacity","Pleural Effusion","Pneumonia","Pneumothorax")
SAFE_ERROR_CODES={
    "invalid internal metadata key":"invalid_linkage_key",
    "incomplete annotation cell":"incomplete_annotation_cell",
    "unexpected published annotation class":"unsupported_annotation_class",
    "annotation source must contain only study key and 14 labels":"annotation_schema_mismatch",
    "malformed annotation row":"malformed_annotation_row",
    "duplicate annotation study":"duplicate_annotation_study",
    "annotation audit exceeds fixed small-file bound":"annotation_row_bound_exceeded",
    "empty annotation source":"empty_annotation_source",
    "matched linkage schema incomplete or duplicated":"linkage_schema_mismatch",
    "bounded matched linkage audit required":"linkage_row_bound_exceeded",
    "malformed linkage row":"malformed_linkage_row",
    "unsupported source split":"unsupported_source_split",
    "patient crosses source splits":"patient_split_overlap",
    "annotation study has inconsistent subject or split":"inconsistent_study_linkage",
    "source outside read-only project boundary":"source_boundary_rejected",
    "source metadata exceeds predeclared bounds":"source_size_bound_exceeded",
    "read-only sources changed during audit":"source_hash_changed",
    "explicit real-label metadata approval and Slurm required":"approval_or_slurm_missing",
}


def safe_error_code(exc):
    # Never print arbitrary exception messages, which could contain input data.
    return SAFE_ERROR_CODES.get(str(exc),"unclassified_failure")


def header_inventory(header):
    """Only schema metadata: no annotation rows or arbitrary column names."""
    known=("study_id","subject_id","dicom_id","Airspace Opacity",*CONDITIONS)
    normalize=lambda name:name.strip().casefold().replace("_"," ")
    lookup={normalize(name):name for name in known}
    recognized=[];unrecognized=[]
    for index,name in enumerate(header or []):
        canonical=lookup.get(normalize(name))
        if canonical is not None:
            recognized.append({"column_index":index,"canonical_name":canonical})
        else:
            unrecognized.append({"column_index":index,
                "column_name_sha256":hashlib.sha256(name.encode("utf-8")).hexdigest()})
    observed={item["canonical_name"] for item in recognized}
    return {"column_count":len(header or []),"recognized_columns":recognized,
        "unrecognized_columns":unrecognized,
        "missing_required_names":[name for name in ("study_id",*CONDITIONS) if name not in observed]}


def blocked_schema_summary(header):
    return {"schema_version":SCHEMA,"status":"blocked_annotation_schema_mismatch",
        "error_code":"annotation_schema_mismatch","annotation_header_schema":header_inventory(header),
        "annotation_rows_read":False,"linkage_rows_read":False,
        "raw_report_text_read":False,"image_pixels_read":False,"raw_ehr_fields_inspected":False,
        "patient_keys_written":False,"model_calls":0,"thresholds_fitted":False,
        "selection_changed":False,"regeneration_authorized":False,"primary_metric_eligible":False,
        "checkpoint_training_overlap_status":"unverified_not_assessed_by_this_audit",
        "clinical_extraction_accuracy":None,"clinical_fault_localization_accuracy":None,
        "interpretation":"Blocked at annotation header. No label values or linkage rows were interpreted; no source was changed. Schema inventory contains only fixed recognized names or hashes, never arbitrary input header text."}


def source_key(value, *, study=False):
    text=value.strip() if isinstance(value,str) else ""
    if study and text.startswith("s"):text=text[1:]
    if not text.isdigit():
        raise ValueError("invalid internal metadata key")
    return str(int(text))


def published_label_state(value):
    if not isinstance(value,str):
        raise ValueError("incomplete annotation cell")
    text=value.strip()
    mapping={"":"unknown","1":"positive","1.0":"positive",
        "0":"negative","0.0":"negative","-1":"uncertain","-1.0":"uncertain"}
    if text not in mapping:
        raise ValueError("unexpected published annotation class")
    return mapping[text]


def annotation_conditions(header):
    for conditions in (CONDITIONS,MANUAL_SOURCE_CONDITIONS):
        expected={"study_id",*conditions}
        if header is not None and len(header)==len(expected) and set(header)==expected:
            return conditions
    return None


def distribution(records,conditions=CONDITIONS):
    return {name:{state:sum(row[name]==state for row in records) for state in STATES}
        for name in conditions}


def audit_streams(gold_reader, linkage_reader):
    header=gold_reader.fieldnames
    conditions=annotation_conditions(header)
    if conditions is None:
        raise ValueError("annotation source must contain only study key and 14 labels")
    expected={"study_id",*conditions}
    gold={}
    for row in gold_reader:
        if set(row)!=expected or any(value is None for value in row.values()):
            raise ValueError("malformed annotation row")
        key=source_key(row["study_id"],study=True)
        if key in gold:raise ValueError("duplicate annotation study")
        gold[key]={name:published_label_state(row[name]) for name in conditions}
        if len(gold)>10000:raise ValueError("annotation audit exceeds fixed small-file bound")
    if not gold:raise ValueError("empty annotation source")
    required={"subject_id","study_id","split","report_path"}
    columns=linkage_reader.fieldnames
    if columns is None or len(columns)!=len(set(columns)) or not required.issubset(columns):
        raise ValueError("matched linkage schema incomplete or duplicated")
    patient_splits,studies,linked,match_rows={},{},{},Counter()
    manifest_rows=0
    for row in linkage_reader:
        manifest_rows+=1
        if manifest_rows>1000000:raise ValueError("bounded matched linkage audit required")
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError("malformed linkage row")
        subject=source_key(row["subject_id"])
        study=source_key(row["study_id"],study=True)
        split=row["split"]
        if split not in {"train","val","test"}:raise ValueError("unsupported source split")
        if patient_splits.setdefault(subject,split)!=split:
            raise ValueError("patient crosses source splits")
        if study not in gold:continue
        if studies.setdefault(study,(subject,split))!=(subject,split):
            raise ValueError("annotation study has inconsistent subject or split")
        match_rows[split]+=1
        slot=linked.setdefault(study,{"subject":subject,"split":split,"report_path_nonempty":False})
        # Never resolve/open/stat this patient path: this is linkage coverage,
        # not verified report availability or a source text inspection.
        slot["report_path_nonempty"] |= bool(row["report_path"].strip())
    splits={}
    for split in ("train","val","test"):
        keys=[key for key,value in linked.items() if value["split"]==split]
        usable=[key for key in keys if linked[key]["report_path_nonempty"]]
        splits[split]={"matched_manifest_rows":match_rows[split],"unique_annotation_studies":len(keys),
            "distinct_patients":len({linked[key]["subject"] for key in keys}),
            "studies_with_nonempty_report_path":len(usable),
            "study_label_counts":distribution([gold[key] for key in keys],conditions),
            "nonempty_path_study_label_counts":distribution([gold[key] for key in usable],conditions)}
    return {"schema_version":SCHEMA,"status":"metadata_coverage_audit_only_not_model_validation",
        "annotation_studies":len(gold),"manifest_rows_scanned":manifest_rows,
        "linked_annotation_studies":len(linked),"unlinked_annotation_studies":len(gold)-len(linked),
        "annotation_columns":list(header),"all_annotation_label_counts":distribution(list(gold.values()),conditions),
        "annotation_schema_variant":"manual_airspace_opacity" if conditions==MANUAL_SOURCE_CONDITIONS else "chexpert_named_14",
        "source_label_names_preserved":True,"airspace_opacity_remapped_to_lung_opacity":False,
        "common_eight_source_columns":{name:name if name in conditions else None for name in EIGHT_FINDINGS},
        "annotation_policy_equivalence_status":"unverified_not_assessed_by_metadata_audit",
        "common_eight_findings":list(EIGHT_FINDINGS),"splits":splits,
        "patient_disjoint_linkage_splits_verified":True,"deduplicated_by_study":True,
        "report_file_existence_or_content_checked":False,"raw_ehr_fields_inspected":False,
        "raw_report_text_read":False,"image_pixels_read":False,"patient_keys_written":False,
        "model_calls":0,"thresholds_fitted":False,"selection_changed":False,"regeneration_authorized":False,
        "primary_metric_eligible":False,"independent_image_ground_truth":False,
        "checkpoint_training_overlap_status":"unverified_not_assessed_by_this_audit",
        "local_file_official_byte_authenticity_verified":False,
        "clinical_extraction_accuracy":None,"clinical_fault_localization_accuracy":None,
        "interpretation":"Published label convention: blank is unmentioned, not negative. Human report-label source metadata cannot establish image truth, gold span/scope policy equivalence, checkpoint training independence, or synthetic triple correctness. Official test labels must not fit thresholds or policies."}


def source_file(path):
    value=Path(path).resolve(strict=True)
    if not value.is_file() or not value.is_relative_to(PROJECT.resolve(strict=True)):
        raise ValueError("source outside read-only project boundary")
    return value


def run(args):
    if not args.allow_real_gold_metadata or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("explicit real-label metadata approval and Slurm required")
    golden=source_file(args.gold_labels);linkage=source_file(args.linkage_manifest)
    if golden.stat().st_size>1024*1024 or linkage.stat().st_size>512*1024*1024:
        raise ValueError("source metadata exceeds predeclared bounds")
    sources={"gold_annotation_csv":golden,"matched_linkage_manifest":linkage}
    initial={name:sha256_file(path) for name,path in sources.items()}
    previous_limit=csv.field_size_limit()
    try:
        csv.field_size_limit(16*1024*1024)
        with golden.open(encoding="utf-8-sig",newline="") as gold_stream,linkage.open(encoding="utf-8-sig",newline="") as link_stream:
            gold_reader=csv.DictReader(gold_stream)
            header=gold_reader.fieldnames
            if annotation_conditions(header) is None:
                summary=blocked_schema_summary(header)
            else:
                summary=audit_streams(gold_reader,csv.DictReader(link_stream))
    finally:csv.field_size_limit(previous_limit)
    if initial!={name:sha256_file(path) for name,path in sources.items()}:
        raise ValueError("read-only sources changed during audit")
    summary["source_sha256"]=initial
    summary["source_documentation"]={
        "annotation_release":"https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/",
        "checkpoint_training_warning":"https://github.com/stanfordmlgroup/CheXbert/blob/master/README.md"}
    return summary


def markdown(summary):
    if summary["status"]=="blocked_annotation_schema_mismatch":
        return "\n".join(["# Official label schema audit blocked / 官方标签列接口不匹配","",
            "The strict 15-column contract was not satisfied. No annotation row or linkage row was interpreted.",
            "See protected summary.json for fixed-name schema coverage and unrecognized-name hashes.",
            "This is not label coverage, inference, accuracy, selection, or repair.",""])
    lines=["# Official report-label metadata audit / 官方人工报告标签元数据审核","",
        "No report/image/EHR interpretation, model calls, thresholds or fault assignment.",
        "Human label source does NOT establish checkpoint training independence or image truth.","",
        "| Split | Linked studies | Distinct patients | Studies with nonempty report path |",
        "|---|---:|---:|---:|"]
    for split,values in summary["splits"].items():
        lines.append(f"| {split} | {values['unique_annotation_studies']} | {values['distinct_patients']} | {values['studies_with_nonempty_report_path']} |")
    lines += ["","Report path is a metadata field only: existence/content is NOT checked.",
        "Unknown/unmentioned is not negative. Repeated image/linkage rows count once per study.",
        "Original source label names are preserved. Airspace Opacity is NOT remapped to Lung Opacity.",
        "No patient keys, rows, source paths or gold report examples appear in this output.",
        "Training overlap and annotation-policy equivalence remain unverified. No primary eligibility.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("gold-labels","linkage-manifest","output-root","run-id"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--allow-real-gold-metadata",action="store_true")
    args=parser.parse_args();os.umask(0o007);temporary=None;started=time.monotonic()
    try:
        # Refuse overwrites before any real input is accessed.
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        summary=run(args)
        files=[write_private_json(temporary/"summary.json",summary),
            write_private_text(temporary/"summary.md",markdown(summary))]
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"run_id":args.run_id,
            "program_sha256":sha256_file(__file__),"source_sha256":summary["source_sha256"],
            "real_reports_images_or_ehr_opened":False,"patient_keys_written":False,"model_calls":0,
            "primary_metric_eligible":False,"selection_changed":False,"regeneration_authorized":False,
            "artifacts":{path.name:{"sha256":sha256_file(path)} for path in files}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:discard_atomic_run(temporary)
        print(json.dumps({"status":"failed","error_type":type(exc).__name__,
            "error_code":safe_error_code(exc)}))
        return 1
    status="blocked_annotation_schema_mismatch" if summary["status"]=="blocked_annotation_schema_mismatch" else "completed_official_gold_metadata_audit"
    print(json.dumps({"status":status,
        "elapsed_seconds":round(time.monotonic()-started,3),"manifest_sha256":sha256_file(target/"manifest.json")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
