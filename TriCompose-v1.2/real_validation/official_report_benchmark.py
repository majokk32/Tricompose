"""Frozen, predeclared real-report labeling diagnostic; pure helpers only.

Tests use invented CSV/text. Real inputs may be consumed only by the explicitly
approved Slurm entry point. No source key/path/text is part of output records.
"""
from __future__ import annotations

from collections import Counter
import re

from audit_official_report_gold import annotation_conditions, published_label_state, source_key

SCHEMA="tricompose-official-report-label-benchmark-v1"
METHOD="strict-impression-four-state-diagnostic-v1"
STATES=("positive","negative","uncertain","unknown")
HEADS=("enlarged_cardiomediastinum","cardiomegaly","lung_opacity","lung_lesion",
    "edema","consolidation","pneumonia","atelectasis","pneumothorax",
    "pleural_effusion","pleural_other","fracture","support_devices","no_finding")
SOURCE_NAMES={
    "enlarged_cardiomediastinum":"Enlarged Cardiomediastinum", "cardiomegaly":"Cardiomegaly",
    "lung_opacity":"Lung Opacity", "lung_lesion":"Lung Lesion", "edema":"Edema",
    "consolidation":"Consolidation", "pneumonia":"Pneumonia", "atelectasis":"Atelectasis",
    "pneumothorax":"Pneumothorax", "pleural_effusion":"Pleural Effusion",
    "pleural_other":"Pleural Other", "fracture":"Fracture", "support_devices":"Support Devices",
    "no_finding":"No Finding"}
EIGHT=("atelectasis","cardiomegaly","consolidation","edema","lung_opacity",
    "pleural_effusion","pneumonia","pneumothorax")
GATE_FINDINGS=("cardiomegaly","consolidation","pleural_effusion","pneumothorax")


def decode(values):
    if len(values)!=14:raise ValueError("prediction_width_mismatch")
    mapping={0:"unknown",1:"positive",2:"negative",3:"uncertain"}
    if any(type(value) is not int or value not in (range(2) if index==13 else range(4))
            for index,value in enumerate(values)):
        raise ValueError("prediction_class_mismatch")
    return {name:mapping[value] for name,value in zip(HEADS,values,strict=True)}


def input_normalization(text):
    # Match the unchanged local wrapper, including its literal replacements.
    return text.strip().replace("\n"," ").replace("\\s+"," ").replace("\\s+(?=[\\.,])","").strip()


def select_impression(text):
    """Conservative declared parser, NOT the full official MIMIC parser.

    Only exactly one nonempty IMPRESSION/CONCLUSION/combined conclusion section.
    No full-report/history/finding fallback, patient-ID fixups or gold-guided
    sentence selection. Nonstandard or repeated sections remain unavailable.
    """
    if not isinstance(text,str) or not text.strip():return None,"empty_report"
    header=re.compile(r"(?m)^[ \t]*([A-Z][A-Z ()/,-]{1,79}):[ \t]*")
    sections=[]
    matches=list(header.finditer(text.replace("\r\n","\n").replace("\r","\n")))
    normalized=text.replace("\r\n","\n").replace("\r","\n")
    targets={"IMPRESSION","CONCLUSION","FINDINGS AND IMPRESSION","FINDINGS/IMPRESSION","FINDINGS/ IMPRESSION"}
    for index,match in enumerate(matches):
        if match.group(1).strip() in targets:
            end=matches[index+1].start() if index+1<len(matches) else len(normalized)
            sections.append(normalized[match.end():end].strip())
    if len(sections)>1:return None,"ambiguous_impression_sections"
    if not sections:return None,"no_explicit_impression_section"
    if not sections[0]:return None,"empty_impression_section"
    if len(sections[0])>8192:return None,"impression_character_bound_exceeded"
    return sections[0],"explicit_impression"


def collect_sources(gold_reader,linkage_reader):
    """Internal metadata join. Keys/paths stay transient, never serialized."""
    conditions=annotation_conditions(gold_reader.fieldnames)
    if conditions is None:raise ValueError("annotation_schema_mismatch")
    expected={"study_id",*conditions};gold={}
    for index,row in enumerate(gold_reader):
        if set(row)!=expected or any(value is None for value in row.values()):
            raise ValueError("malformed_annotation_row")
        study=source_key(row["study_id"],study=True)
        if study in gold:raise ValueError("duplicate_annotation_study")
        gold[study]={"item_id":f"item_{index:04d}",
            "reference":{name:published_label_state(row[name]) for name in conditions}}
        if len(gold)>10000:raise ValueError("annotation_row_bound_exceeded")
    if not gold:raise ValueError("empty_annotation_source")
    required={"subject_id","study_id","split","report_path"}
    header=linkage_reader.fieldnames
    if header is None or len(header)!=len(set(header)) or not required.issubset(header):
        raise ValueError("linkage_schema_mismatch")
    patients={};linked={}
    for index,row in enumerate(linkage_reader):
        if index>=1000000:raise ValueError("linkage_row_bound_exceeded")
        if None in row or any(row.get(name) is None for name in required):
            raise ValueError("malformed_linkage_row")
        subject=source_key(row["subject_id"]);study=source_key(row["study_id"],study=True)
        split=row["split"]
        if split not in {"train","val","test"}:raise ValueError("unsupported_source_split")
        if patients.setdefault(subject,split)!=split:raise ValueError("patient_split_overlap")
        if study not in gold:continue
        slot=linked.setdefault(study,{"subject":subject,"split":split,"paths":set()})
        if (slot["subject"],slot["split"])!=(subject,split):raise ValueError("inconsistent_study_linkage")
        if row["report_path"].strip():slot["paths"].add(row["report_path"].strip())
    items=[]
    for study,entry in gold.items():
        slot=linked.get(study)
        status="unlinked" if slot is None else "linked_non_test" if slot["split"]!="test" else "ready_metadata"
        path=None
        if status=="ready_metadata":
            if not slot["paths"]:status="no_report_path_metadata"
            elif len(slot["paths"])!=1:status="ambiguous_report_path_metadata"
            else:path=next(iter(slot["paths"]))
        items.append({**entry,"status":status,"report_path":path})
    if sum(item["status"]=="ready_metadata" for item in items)>1024:
        raise ValueError("eligible_report_bound_exceeded")
    return items,conditions


def state_statistics(pairs):
    """Four text states; unknown is NOT an explicit clinical negative."""
    matrix={truth:{prediction:0 for prediction in STATES} for truth in STATES}
    for truth,prediction in pairs:
        if truth not in STATES or prediction not in STATES:raise ValueError("invalid_four_state_pair")
        matrix[truth][prediction]+=1
    count=len(pairs);per_state={}
    for state in STATES:
        support=sum(matrix[state].values());tp=matrix[state][state]
        fp=sum(matrix[other][state] for other in STATES if other!=state)
        fn=support-tp
        per_state[state]={"reference_support":support,"true_positive":tp,"false_positive":fp,"false_negative":fn,
            "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/support if support else None,
            "f1":2*tp/(2*tp+fp+fn) if support and 2*tp+fp+fn else None}
    f1s=[row["f1"] for row in per_state.values() if row["f1"] is not None]
    known=[pair for pair in pairs if pair[0]!="unknown"]
    explicit=[pair for pair in pairs if pair[0] in {"positive","negative"}]
    uncertain=[pair for pair in pairs if pair[0] in {"unknown","uncertain"}]
    return {"checks":count,"confusion_matrix":matrix,"state_order":list(STATES),"per_state":per_state,
        "exact_state_accuracy":sum(a==b for a,b in pairs)/count if count else None,
        "state_macro_f1_reference_supported_classes":sum(f1s)/len(f1s) if f1s else None,
        "annotated_checks":len(known),"annotated_state_accuracy":sum(a==b for a,b in known)/len(known) if known else None,
        "omitted_annotated_assertions":sum(b=="unknown" for _,b in known),
        "explicit_positive_negative_checks":len(explicit),
        "hard_polarity_flips":sum({a,b}=={"positive","negative"} for a,b in explicit),
        "uncertain_unknown_checks":len(uncertain),
        "determinate_promotions_on_uncertain_unknown":sum(b in {"positive","negative"} for _,b in uncertain)}


def summarize(items,records,conditions):
    if len({row["item_id"] for row in records})!=len(records):raise ValueError("duplicate_prediction_item")
    index={row["item_id"]:row for row in records}
    if set(index)!={row["item_id"] for row in items}:raise ValueError("prediction_inventory_mismatch")
    for item in items:
        row=index[item["item_id"]]
        if row["status"]=="complete":
            if item["status"]!="ready_metadata" or set(row["finding_states"])!=set(HEADS):
                raise ValueError("invalid_completed_prediction")
            if any(value not in STATES for value in row["finding_states"].values()):
                raise ValueError("invalid_prediction_state")
            for finding,decision in row.get("scope_decisions",{}).items():
                if finding not in GATE_FINDINGS or decision.get("decision") not in {"scope_commit","abstain","no_model_assertion"}:
                    raise ValueError("invalid_scope_decision")
                if decision["decision"]=="scope_commit":
                    if (decision.get("state")!=row["finding_states"][finding] or decision.get("state")=="unknown"
                            or decision.get("scope_verified") is not True):
                        raise ValueError("scope_cannot_flip_or_promote_prediction")
                elif decision.get("state")!="unknown" or decision.get("scope_verified") is not False:
                    raise ValueError("noncommit_must_remain_unknown")
        elif row.get("finding_states") is not None:raise ValueError("unavailable_item_has_prediction")
    completed=[item for item in items if index[item["item_id"]]["status"]=="complete"]
    eligible=[name for name in HEADS if name!="no_finding" and SOURCE_NAMES[name] in conditions]
    stats={name:state_statistics([(item["reference"][SOURCE_NAMES[name]],index[item["item_id"]]["finding_states"][name])
        for item in completed]) for name in eligible}
    scope={}
    for finding in GATE_FINDINGS:
        selected=[(item,index[item["item_id"]].get("scope_decisions",{}).get(finding)) for item in completed]
        commits=[(item,decision) for item,decision in selected if decision and decision["decision"]=="scope_commit"]
        scope[finding]={"full_completed_denominator":len(completed),"commit_count":len(commits),
            "commit_coverage":len(commits)/len(completed) if completed else None,
            "conditional_state_accuracy":sum(item["reference"][SOURCE_NAMES[finding]]==decision["state"] for item,decision in commits)/len(commits) if commits else None,
            "decision_counts":dict(Counter(decision["decision"] if decision else "scope_unavailable" for _,decision in selected)),
            "clinical_image_truth":False}
    # Unknown-only heads retain their confusion table but cannot inflate the
    # pooled clinical-assertion diagnostic through all-unknown matches.
    aggregate_heads=[name for name,row in stats.items() if row["annotated_checks"]>0]
    f1s=[stats[name]["state_macro_f1_reference_supported_classes"] for name in aggregate_heads
        if stats[name]["state_macro_f1_reference_supported_classes"] is not None]
    return {"schema_version":SCHEMA,"method_version":METHOD,"status":"real_report_label_diagnostic_not_independent_test",
        "annotation_inventory":len(items),"metadata_status_counts":dict(Counter(row["status"] for row in items)),
        "execution_status_counts":dict(Counter(row["status"] for row in records)),
        "completed_reports":len(completed),"evaluated_four_state_heads":eligible,
        "common_eight_evaluated_heads":[name for name in EIGHT if name in eligible],
        "excluded_heads":{"lung_opacity":"source_airspace_opacity_not_assumed_equivalent" if "Lung Opacity" not in conditions else "not_excluded",
            "no_finding":"binary_checkpoint_head_not_full_four_state_output"},
        "per_finding":stats,"mean_per_head_state_macro_f1":sum(f1s)/len(f1s) if f1s else None,
        "aggregate_heads_with_annotated_reference":aggregate_heads,
        "all_eligible_heads_determinate_promotions_on_uncertain_unknown":sum(row["determinate_promotions_on_uncertain_unknown"] for row in stats.values()),
        "all_eligible_heads_hard_polarity_flips":sum(row["hard_polarity_flips"] for row in stats.values()),
        "all_eligible_heads_omitted_annotated_assertions":sum(row["omitted_annotated_assertions"] for row in stats.values()),
        "frozen_four_finding_scope_diagnostic":scope,"scope_mask_does_not_change_raw_predictions":True,
        "patient_disjoint_source_splits_verified":True,"analysis_unit":"unique_study_report_not_independent_patient",
        "patient_cluster_bootstrap_performed":False,"model_received_gold_labels":False,
        "primary_metric_eligible":False,"checkpoint_training_overlap_status":"unverified",
        "annotation_policy_equivalence_status":"unverified_strict_impression_only",
        "independent_image_ground_truth":False,"selection_changed":False,"regeneration_authorized":False,
        "thresholds_fitted":False,"source_label_names_preserved":True,
        "interpretation":"Text-state diagnostic only. All missing/unlinked/unreadable/ambiguous/overlong cases remain in the inventory. Unknown is not negative. Source Airspace Opacity is not mapped to Lung Opacity; binary No Finding is excluded from four-state metrics. Scope commits are conditional extraction coverage, not repaired reports or independent image facts. MIMIC training overlap is unverified; no selection, fault localization, or regeneration gate is cleared."}


def markdown(summary):
    lines=["# Frozen CheXbert real-report diagnostic / 真实报告标签诊断","",
        "Not an independent held-out test, image factuality test, or repaired-triple result.",
        "Training overlap and annotation/section policy equivalence remain unverified.","",
        "| Finding | Scored checks | Four-state macro F1 | Annotated-state match | Polarity flips |",
        "|---|---:|---:|---:|---:|"]
    show=lambda value:"NA" if value is None else f"{value:.4f}"
    for name,row in summary["per_finding"].items():
        lines.append(f"| {name} | {row['checks']} | {show(row['state_macro_f1_reference_supported_classes'])} | {show(row['annotated_state_accuracy'])} | {row['hard_polarity_flips']} |")
    lines += ["","All source/missingness and confusion denominators are retained in summary.json.",
        "Airspace Opacity remains distinct; No Finding is a binary head, not a four-state head.",
        "No real report text, key, image, source path or reference row is exported.",""]
    return "\n".join(lines)
