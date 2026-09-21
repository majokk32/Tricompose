#!/bin/bash
#SBATCH --job-name=eval_ehr_rpt
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail

cd /project2/ruishanl_1185/inference_3mod/TriCompose-v1.0

runtime_xhuang90/venvs/synehrgy-v2/bin/python - <<'PY'
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ehr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/staging/bridge_gpt2_pool80_v11_20260813_003/cases")
report_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/report_candidates")
out = Path("eval/ehr_report_v1_1_pool160")
out.mkdir(parents=True, exist_ok=True)

report_runs = [
    "pool160_maira2_20260813_2f6a",
    "pool160_cxrmate_single_20260813_74d1",
    "pool160_llavarad_20260813_aa39",
    "pool160_chexagent2_fp32_20260813_51f8",
]

labels = [
    "pneumonia",
    "pleural_effusion",
    "pulmonary_edema",
    "cardiomegaly",
    "pneumothorax",
    "atelectasis",
    "consolidation",
    "lung_opacity",
    "cardiac_pacemaker",
    "congestive_heart_failure",
]

positive_patterns = {
    "pneumonia": [r"pneumonia", r"infection"],
    "pleural_effusion": [r"pleural effusion", r"effusion"],
    "pulmonary_edema": [r"pulmonary edema", r"interstitial edema", r"vascular congestion"],
    "cardiomegaly": [r"cardiomegaly", r"cardiac enlargement", r"enlarged cardiac silhouette", r"heart .* enlarged"],
    "pneumothorax": [r"pneumothorax"],
    "atelectasis": [r"atelectasis", r"bibasilar opacity"],
    "consolidation": [r"consolidation", r"consolidative"],
    "lung_opacity": [r"opacity", r"opacities", r"infiltrate"],
    "cardiac_pacemaker": [r"pacemaker", r"pacing device"],
    "congestive_heart_failure": [r"heart failure", r"pulmonary edema", r"vascular congestion", r"cardiomegaly"],
}

negative_patterns = {
    "pneumonia": [r"no .*pneumonia", r"no focal consolidation", r"no acute cardiopulmonary"],
    "pleural_effusion": [r"no .*effusion", r"without .*effusion"],
    "pulmonary_edema": [r"no .*edema", r"without .*edema", r"no pulmonary vascular congestion"],
    "cardiomegaly": [r"heart size .*normal", r"cardiomediastinal silhouette .*normal", r"no cardiomegaly"],
    "pneumothorax": [r"no .*pneumothorax", r"without .*pneumothorax"],
    "atelectasis": [r"no .*atelectasis"],
    "consolidation": [r"no .*consolidation", r"without .*consolidation"],
    "lung_opacity": [r"no .*opacity", r"no .*opacities", r"clear lungs"],
    "cardiac_pacemaker": [r"no .*pacemaker"],
    "congestive_heart_failure": [r"no .*edema", r"no pulmonary vascular congestion", r"heart size .*normal"],
}

def has_any(patterns, text):
    return any(re.search(p, text) for p in patterns)

def report_label(text, label):
    t = text.lower().replace("\n", " ")
    neg = has_any(negative_patterns.get(label, []), t)
    pos = has_any(positive_patterns.get(label, []), t)
    if neg and not pos:
        return "negative"
    if pos:
        return "positive"
    return "unknown"

def load_ehr_facts(case_id):
    p = ehr_root / case_id / "ehr_facts.json"
    x = json.loads(p.read_text())
    facts = x.get("facts", {})
    return {k: v.get("state", "unknown") for k, v in facts.items()}

rows = []
summary_by_model = defaultdict(Counter)
label_stats = defaultdict(Counter)

for run_id in report_runs:
    run = report_root / run_id
    m = json.loads((run / "manifest.json").read_text())
    model = m.get("model_id", run_id)

    for item in m.get("candidates", []):
        cand = json.loads((run / item["path"]).read_text())
        case_id = cand.get("case_id")
        ehr = load_ehr_facts(case_id)

        art = cand.get("artifact", {})
        report_path = art.get("path")
        if not report_path:
            continue

        text = Path(report_path).read_text(errors="ignore")
        for label in labels:
            ehr_state = ehr.get(label, "unknown")
            rpt_state = report_label(text, label)

            comparable = ehr_state in ("positive", "negative") and rpt_state in ("positive", "negative")
            contradiction = (
                (ehr_state == "positive" and rpt_state == "negative") or
                (ehr_state == "negative" and rpt_state == "positive")
            )

            row = {
                "case_id": case_id,
                "report_model": model,
                "report_candidate_id": cand.get("candidate_id"),
                "label": label,
                "ehr_state": ehr_state,
                "report_state": rpt_state,
                "comparable": comparable,
                "agreement": comparable and ehr_state == rpt_state,
                "contradiction": contradiction,
                "report_path": report_path,
            }
            rows.append(row)

            summary_by_model[model]["label_checks"] += 1
            if comparable:
                summary_by_model[model]["comparable"] += 1
                if row["agreement"]:
                    summary_by_model[model]["agreement"] += 1
                if contradiction:
                    summary_by_model[model]["contradiction"] += 1

            label_stats[label]["label_checks"] += 1
            if comparable:
                label_stats[label]["comparable"] += 1
                if row["agreement"]:
                    label_stats[label]["agreement"] += 1
                if contradiction:
                    label_stats[label]["contradiction"] += 1

summary = {
    "evaluation_name": "ehr_report_v1_1_pool160_rulebased_consistency",
    "ehr_facts_root": str(ehr_root),
    "report_candidates_root": str(report_root),
    "report_runs": report_runs,
    "row_count": len(rows),
    "models": {},
    "labels": {},
    "warning": "Rule-based keyword extraction. Use CheXbert/CheXpert/RadGraph later for formal report labels.",
}

for model, c in sorted(summary_by_model.items()):
    comparable = c["comparable"]
    summary["models"][model] = {
        "label_checks": c["label_checks"],
        "comparable": comparable,
        "agreement": c["agreement"],
        "contradiction": c["contradiction"],
        "agreement_rate": round(c["agreement"] / comparable, 6) if comparable else None,
        "contradiction_rate": round(c["contradiction"] / comparable, 6) if comparable else None,
    }

for label, c in sorted(label_stats.items()):
    comparable = c["comparable"]
    summary["labels"][label] = {
        "label_checks": c["label_checks"],
        "comparable": comparable,
        "agreement": c["agreement"],
        "contradiction": c["contradiction"],
        "agreement_rate": round(c["agreement"] / comparable, 6) if comparable else None,
        "contradiction_rate": round(c["contradiction"] / comparable, 6) if comparable else None,
    }

(out / "ehr_report_rulebased_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "ehr_report_rulebased_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# EHR-Report V1.1 Pool160 Rule-Based Consistency")
lines.append("")
lines.append(f"- EHR facts root: `{ehr_root}`")
lines.append(f"- Report candidates root: `{report_root}`")
lines.append(f"- Label checks: {summary['row_count']}")
lines.append("")
lines.append("## Warning")
lines.append("This is a rule-based keyword consistency check, not formal CheXbert/CheXpert/RadGraph evaluation.")
lines.append("")
lines.append("## Per Report Model")
lines.append("| Report model | Comparable | Agreement | Contradiction | Agreement rate | Contradiction rate |")
lines.append("|---|---:|---:|---:|---:|---:|")
for model, s in summary["models"].items():
    lines.append(f"| {model} | {s['comparable']} | {s['agreement']} | {s['contradiction']} | {s['agreement_rate']} | {s['contradiction_rate']} |")
lines.append("")
lines.append("## Per Label")
lines.append("| Label | Comparable | Agreement | Contradiction | Agreement rate | Contradiction rate |")
lines.append("|---|---:|---:|---:|---:|---:|")
for label, s in summary["labels"].items():
    lines.append(f"| {label} | {s['comparable']} | {s['agreement']} | {s['contradiction']} | {s['agreement_rate']} | {s['contradiction_rate']} |")

(out / "ehr_report_rulebased_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
