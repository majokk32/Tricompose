#!/bin/bash
#SBATCH --job-name=eval_ehr_rpt_pos
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
    "pneumonia": [r"\bpneumonia\b", r"\binfection\b"],
    "pleural_effusion": [r"\bpleural effusion\b", r"\beffusion\b"],
    "pulmonary_edema": [r"\bpulmonary edema\b", r"\binterstitial edema\b", r"\bvascular congestion\b", r"\bpulmonary vascular congestion\b"],
    "cardiomegaly": [r"\bcardiomegaly\b", r"\bcardiac enlargement\b", r"\benlarged cardiac silhouette\b", r"\bheart .* enlarged\b"],
    "pneumothorax": [r"\bpneumothorax\b"],
    "atelectasis": [r"\batelectasis\b"],
    "consolidation": [r"\bconsolidation\b", r"\bconsolidative\b"],
    "lung_opacity": [r"\bopacity\b", r"\bopacities\b", r"\binfiltrate\b"],
    "cardiac_pacemaker": [r"\bpacemaker\b", r"\bpacing device\b"],
    "congestive_heart_failure": [r"\bheart failure\b", r"\bpulmonary edema\b", r"\bvascular congestion\b", r"\bcardiomegaly\b"],
}

negative_patterns = {
    "pneumonia": [r"\bno .*pneumonia\b", r"\bno focal consolidation\b", r"\bno acute cardiopulmonary\b"],
    "pleural_effusion": [r"\bno .*effusion\b", r"\bwithout .*effusion\b"],
    "pulmonary_edema": [r"\bno .*edema\b", r"\bwithout .*edema\b", r"\bno pulmonary vascular congestion\b"],
    "cardiomegaly": [r"\bheart size .*normal\b", r"\bcardiomediastinal silhouette .*normal\b", r"\bno cardiomegaly\b"],
    "pneumothorax": [r"\bno .*pneumothorax\b", r"\bwithout .*pneumothorax\b"],
    "atelectasis": [r"\bno .*atelectasis\b"],
    "consolidation": [r"\bno .*consolidation\b", r"\bwithout .*consolidation\b"],
    "lung_opacity": [r"\bno .*opacity\b", r"\bno .*opacities\b", r"\bclear lungs\b"],
    "cardiac_pacemaker": [r"\bno .*pacemaker\b"],
    "congestive_heart_failure": [r"\bno .*edema\b", r"\bno pulmonary vascular congestion\b", r"\bheart size .*normal\b"],
}

def has_any(patterns, text):
    return any(re.search(p, text) for p in patterns)

def report_label(text, label):
    t = text.lower().replace("\n", " ")
    neg = has_any(negative_patterns.get(label, []), t)
    pos = has_any(positive_patterns.get(label, []), t)
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "unknown"

def load_ehr_positive_labels(case_id):
    p = ehr_root / case_id / "ehr_facts.json"
    x = json.loads(p.read_text())
    facts = x.get("facts", {})
    return {k for k, v in facts.items() if v.get("state") == "positive"}

rows = []
summary_by_model = defaultdict(Counter)
summary_by_label = defaultdict(Counter)

for run_id in report_runs:
    run = report_root / run_id
    m = json.loads((run / "manifest.json").read_text())
    model = m.get("model_id", run_id)

    for item in m.get("candidates", []):
        cand = json.loads((run / item["path"]).read_text())
        case_id = cand.get("case_id")
        ehr_pos = load_ehr_positive_labels(case_id)

        report_path = cand.get("artifact", {}).get("path")
        if not report_path:
            continue
        text = Path(report_path).read_text(errors="ignore")

        for label in sorted(ehr_pos):
            if label not in labels:
                continue

            rpt_state = report_label(text, label)
            supported = rpt_state == "positive"
            contradiction = rpt_state == "negative"
            not_mentioned = rpt_state == "unknown"

            row = {
                "case_id": case_id,
                "report_model": model,
                "report_candidate_id": cand.get("candidate_id"),
                "label": label,
                "ehr_state": "positive",
                "report_state": rpt_state,
                "supported": supported,
                "contradiction": contradiction,
                "not_mentioned": not_mentioned,
                "report_path": report_path,
            }
            rows.append(row)

            summary_by_model[model]["ehr_positive_checks"] += 1
            summary_by_model[model]["supported"] += int(supported)
            summary_by_model[model]["contradiction"] += int(contradiction)
            summary_by_model[model]["not_mentioned"] += int(not_mentioned)

            summary_by_label[label]["ehr_positive_checks"] += 1
            summary_by_label[label]["supported"] += int(supported)
            summary_by_label[label]["contradiction"] += int(contradiction)
            summary_by_label[label]["not_mentioned"] += int(not_mentioned)

summary = {
    "evaluation_name": "ehr_report_v1_1_pool160_positive_evidence_rulebased",
    "ehr_facts_root": str(ehr_root),
    "report_candidates_root": str(report_root),
    "report_runs": report_runs,
    "ehr_positive_checks": len(rows),
    "models": {},
    "labels": {},
    "warning": "Rule-based positive-evidence check. EHR unknown is not treated as negative.",
}

for model, c in sorted(summary_by_model.items()):
    n = c["ehr_positive_checks"]
    summary["models"][model] = {
        "ehr_positive_checks": n,
        "supported": c["supported"],
        "contradiction": c["contradiction"],
        "not_mentioned": c["not_mentioned"],
        "support_rate": round(c["supported"] / n, 6) if n else None,
        "contradiction_rate": round(c["contradiction"] / n, 6) if n else None,
        "not_mentioned_rate": round(c["not_mentioned"] / n, 6) if n else None,
    }

for label, c in sorted(summary_by_label.items()):
    n = c["ehr_positive_checks"]
    summary["labels"][label] = {
        "ehr_positive_checks": n,
        "supported": c["supported"],
        "contradiction": c["contradiction"],
        "not_mentioned": c["not_mentioned"],
        "support_rate": round(c["supported"] / n, 6) if n else None,
        "contradiction_rate": round(c["contradiction"] / n, 6) if n else None,
        "not_mentioned_rate": round(c["not_mentioned"] / n, 6) if n else None,
    }

(out / "ehr_report_positive_evidence_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "ehr_report_positive_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# EHR-Report V1.1 Positive-Evidence Consistency")
lines.append("")
lines.append(f"- EHR positive checks: {summary['ehr_positive_checks']}")
lines.append("")
lines.append("## Warning")
lines.append("This is a rule-based positive-evidence check. EHR unknown is not treated as negative.")
lines.append("")
lines.append("## Per Report Model")
lines.append("| Report model | EHR positive checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for model, s in summary["models"].items():
    lines.append(f"| {model} | {s['ehr_positive_checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")
lines.append("")
lines.append("## Per Label")
lines.append("| Label | EHR positive checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, s in summary["labels"].items():
    lines.append(f"| {label} | {s['ehr_positive_checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")

(out / "ehr_report_positive_evidence_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
