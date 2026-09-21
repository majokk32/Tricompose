#!/bin/bash
#SBATCH --job-name=eval_ehr_rpt_960
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
out = Path("eval/ehr_report_v1_1_full960")
out.mkdir(parents=True, exist_ok=True)

report_runs = [
    "pool160_maira2_20260813_2f6a",
    "pool160_cxrmate_single_20260813_74d1",
    "pool160_llavarad_20260813_aa39",
    "pool160_chexagent2_fp32_20260813_51f8",
    "roentgen80_maira2_20260813_a31e",
    "roentgen80_cxrmate_single_20260813_d6a4",
    "roentgen80_llavarad_20260813_b472",
    "roentgen80_chexagent2_fp32_20260813_c593",
]

direct_labels = [
    "congestive_heart_failure",
    "pneumonia",
    "cardiac_pacemaker",
    "pleural_effusion",
]

positive_patterns = {
    "congestive_heart_failure": [
        r"\bheart failure\b",
        r"\bcongestive heart failure\b",
        r"\bpulmonary edema\b",
        r"\binterstitial edema\b",
        r"\bvascular congestion\b",
        r"\bpulmonary vascular congestion\b",
        r"\bcardiomegaly\b",
        r"\bcardiac enlargement\b",
        r"\benlarged cardiac silhouette\b",
        r"\bheart .* enlarged\b",
    ],
    "pneumonia": [
        r"\bpneumonia\b",
        r"\binfection\b",
        r"\binfectious\b",
        r"\bconsolidation\b",
        r"\bconsolidative\b",
        r"\binfiltrate\b",
        r"\bairspace opacity\b",
        r"\bpatchy opacit",
    ],
    "cardiac_pacemaker": [
        r"\bpacemaker\b",
        r"\bpacing device\b",
        r"\bpacer\b",
        r"\bleft chest wall device\b",
    ],
    "pleural_effusion": [
        r"\bpleural effusion\b",
        r"\beffusion\b",
        r"\bblunting .*costophrenic\b",
    ],
}

negative_patterns = {
    "congestive_heart_failure": [
        r"\bno .*pulmonary edema\b",
        r"\bwithout .*pulmonary edema\b",
        r"\bno .*edema\b",
        r"\bwithout .*edema\b",
        r"\bno pulmonary vascular congestion\b",
        r"\bheart size .*normal\b",
        r"\bcardiomediastinal silhouette .*normal\b",
        r"\bno cardiomegaly\b",
    ],
    "pneumonia": [
        r"\bno .*pneumonia\b",
        r"\bno focal consolidation\b",
        r"\bwithout focal consolidation\b",
        r"\bno .*consolidation\b",
        r"\bno acute cardiopulmonary\b",
        r"\bclear lungs\b",
    ],
    "cardiac_pacemaker": [
        r"\bno .*pacemaker\b",
        r"\bno .*pacing device\b",
    ],
    "pleural_effusion": [
        r"\bno .*pleural effusion\b",
        r"\bno .*effusion\b",
        r"\bwithout .*effusion\b",
    ],
}

def has_any(patterns, text):
    return any(re.search(p, text) for p in patterns)

def report_state(text, label):
    t = text.lower().replace("\n", " ")
    pos = has_any(positive_patterns[label], t)
    neg = has_any(negative_patterns[label], t)

    if pos:
        return "positive"
    if neg:
        return "negative"
    return "unknown"

def load_direct_facts(case_id):
    x = json.loads((ehr_root / case_id / "ehr_facts.json").read_text())
    facts = x.get("direct_facts", {})
    out = []
    for fact_id, fact in facts.items():
        status = fact.get("status") or fact.get("state")
        if fact_id in direct_labels and status in ("documented", "positive"):
            out.append(fact_id)
    return out

rows = []
by_model = defaultdict(Counter)
by_label = defaultdict(Counter)

for run_id in report_runs:
    run = report_root / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    model = manifest.get("model_id", run_id)

    for item in manifest.get("candidates", []):
        cand = json.loads((run / item["path"]).read_text())
        case_id = cand["case_id"]
        direct_facts = load_direct_facts(case_id)

        if not direct_facts:
            continue

        report_path = cand.get("artifact", {}).get("path")
        if not report_path:
            continue

        text = Path(report_path).read_text(errors="ignore")

        for label in direct_facts:
            rstate = report_state(text, label)

            supported = rstate == "positive"
            contradiction = rstate == "negative"
            not_mentioned = rstate == "unknown"

            row = {
                "case_id": case_id,
                "report_model": model,
                "report_candidate_id": cand.get("candidate_id"),
                "parent_ids": cand.get("parent_ids"),
                "direct_fact": label,
                "ehr_state": "documented",
                "report_state": rstate,
                "supported": supported,
                "contradiction": contradiction,
                "not_mentioned": not_mentioned,
                "report_path": report_path,
            }
            rows.append(row)

            by_model[model]["checks"] += 1
            by_model[model]["supported"] += int(supported)
            by_model[model]["contradiction"] += int(contradiction)
            by_model[model]["not_mentioned"] += int(not_mentioned)

            by_label[label]["checks"] += 1
            by_label[label]["supported"] += int(supported)
            by_label[label]["contradiction"] += int(contradiction)
            by_label[label]["not_mentioned"] += int(not_mentioned)

def pack(counter):
    n = counter["checks"]
    return {
        "checks": n,
        "supported": counter["supported"],
        "contradiction": counter["contradiction"],
        "not_mentioned": counter["not_mentioned"],
        "support_rate": round(counter["supported"] / n, 6) if n else None,
        "contradiction_rate": round(counter["contradiction"] / n, 6) if n else None,
        "not_mentioned_rate": round(counter["not_mentioned"] / n, 6) if n else None,
    }

summary = {
    "evaluation_name": "ehr_report_v1_1_full960_direct_facts_rulebased",
    "ehr_facts_root": str(ehr_root),
    "report_candidates_root": str(report_root),
    "report_runs": report_runs,
    "direct_fact_labels": direct_labels,
    "total_direct_fact_checks": len(rows),
    "models": {k: pack(v) for k, v in sorted(by_model.items())},
    "labels": {k: pack(v) for k, v in sorted(by_label.items())},
    "warning": "Rule-based check over V1.1 direct_facts only. Use CheXbert/CheXpert/RadGraph for formal evaluation.",
}

(out / "ehr_report_direct_facts_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "ehr_report_direct_facts_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# EHR-Report V1.1 Direct-Facts Consistency")
lines.append("")
lines.append(f"- Total direct-fact checks: {summary['total_direct_fact_checks']}")
lines.append(f"- Direct fact labels: {', '.join(direct_labels)}")
lines.append("")
lines.append("## Warning")
lines.append("This is a rule-based check over V1.1 direct_facts only. Use CheXbert/CheXpert/RadGraph for formal evaluation.")
lines.append("")
lines.append("## Per Report Model")
lines.append("| Report model | Checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for model, s in summary["models"].items():
    lines.append(f"| {model} | {s['checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")
lines.append("")
lines.append("## Per Direct Fact")
lines.append("| Direct fact | Checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, s in summary["labels"].items():
    lines.append(f"| {label} | {s['checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")

(out / "ehr_report_direct_facts_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
