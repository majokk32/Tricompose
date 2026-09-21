#!/bin/bash
#SBATCH --job-name=eval_rpt_cxr
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail

cd /project2/ruishanl_1185/inference_3mod/TriCompose-v1.0

runtime_xhuang90/venvs/synehrgy-v2/bin/python - <<'PY'
import json, re
from pathlib import Path
from collections import Counter, defaultdict

staging_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/staging/bridge_gpt2_pool80_v11_20260813_003/cases")
report_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/report_candidates")
out = Path("eval/report_cxr_v1_1_full960")
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

labels = {
    "congestive_heart_failure": {
        "positive": [r"congestive heart failure", r"\bchf\b", r"pulmonary vascular congestion", r"pulmonary edema", r"cardiomegaly", r"vascular congestion"],
        "negative": [r"no pulmonary edema", r"without pulmonary edema", r"no cardiomegaly", r"heart size is normal", r"normal heart size"],
    },
    "pneumonia": {
        "positive": [r"pneumonia", r"infectious", r"infection", r"consolidation", r"airspace opacity"],
        "negative": [r"no pneumonia", r"no focal consolidation", r"without focal consolidation", r"no consolidation"],
    },
    "pleural_effusion": {
        "positive": [r"pleural effusion", r"effusion"],
        "negative": [r"no pleural effusion", r"without pleural effusion", r"no effusion"],
    },
    "cardiac_pacemaker": {
        "positive": [r"pacemaker", r"pacing device", r"pacer"],
        "negative": [r"no pacemaker"],
    },
}

def report_status(text, label):
    t = text.lower()
    neg = any(re.search(p, t) for p in labels[label]["negative"])
    pos = any(re.search(p, t) for p in labels[label]["positive"])
    if neg:
        return "contradiction"
    if pos:
        return "supported"
    return "not_mentioned"

def cxr_model_from_parent(parent_id):
    if "roentgen_v2" in parent_id:
        return "roentgen_v2"
    if "chexgenbench_sana" in parent_id:
        return "chexgenbench_sana"
    if "chexgenbench_pixart" in parent_id:
        return "chexgenbench_pixart"
    return "unknown"

case_direct = {}
for case_dir in sorted(staging_root.iterdir()):
    if not case_dir.is_dir():
        continue
    facts = json.loads((case_dir / "ehr_facts.json").read_text())
    pos = []
    for label, obj in facts.get("direct_facts", {}).items():
        status = obj.get("status") or obj.get("state")
        if status in ("documented", "positive") and label in labels:
            pos.append(label)
    case_direct[case_dir.name] = sorted(pos)

rows = []
by_report_model = defaultdict(Counter)
by_cxr_model = defaultdict(Counter)
by_pair = defaultdict(Counter)
by_label = defaultdict(Counter)

report_count = 0

for run_id in report_runs:
    run = report_root / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    report_model = manifest.get("model_id")

    for item in manifest.get("candidates", []):
        report_count += 1
        cand = json.loads((run / item["path"]).read_text())
        case_id = cand["case_id"]
        parent_id = cand["parent_cxr_candidate_id"]
        cxr_model = cxr_model_from_parent(parent_id)
        text = Path(cand["artifact"]["path"]).read_text(errors="ignore")

        for label in case_direct.get(case_id, []):
            status = report_status(text, label)
            row = {
                "case_id": case_id,
                "report_candidate_id": cand["candidate_id"],
                "parent_cxr_candidate_id": parent_id,
                "report_model": report_model,
                "cxr_model": cxr_model,
                "direct_fact": label,
                "status": status,
            }
            rows.append(row)

            by_report_model[report_model]["checks"] += 1
            by_report_model[report_model][status] += 1

            by_cxr_model[cxr_model]["checks"] += 1
            by_cxr_model[cxr_model][status] += 1

            by_pair[f"{cxr_model}|{report_model}"]["checks"] += 1
            by_pair[f"{cxr_model}|{report_model}"][status] += 1

            by_label[label]["checks"] += 1
            by_label[label][status] += 1

def pack(counter):
    checks = counter.get("checks", 0)
    return {
        "checks": checks,
        "supported": counter.get("supported", 0),
        "contradiction": counter.get("contradiction", 0),
        "not_mentioned": counter.get("not_mentioned", 0),
        "support_rate": round(counter.get("supported", 0) / checks, 6) if checks else None,
        "contradiction_rate": round(counter.get("contradiction", 0) / checks, 6) if checks else None,
        "not_mentioned_rate": round(counter.get("not_mentioned", 0) / checks, 6) if checks else None,
    }

summary = {
    "evaluation_name": "report_cxr_v1_1_full960_rulebased_direct_fact_consistency",
    "report_candidates": report_count,
    "direct_fact_checks": len(rows),
    "report_runs": report_runs,
    "by_report_model": {k: pack(v) for k, v in sorted(by_report_model.items())},
    "by_parent_cxr_model": {k: pack(v) for k, v in sorted(by_cxr_model.items())},
    "by_cxr_report_model_pair": {k: pack(v) for k, v in sorted(by_pair.items())},
    "by_direct_fact": {k: pack(v) for k, v in sorted(by_label.items())},
    "warning": "Rule-based report text vs parent CXR/EHR direct facts. This does not inspect image pixels.",
}

(out / "report_cxr_rulebased_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "report_cxr_rulebased_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# Report-CXR V1.1 Rule-Based Consistency")
lines.append("")
lines.append(f"- Report candidates: {summary['report_candidates']}")
lines.append(f"- Direct-fact checks: {summary['direct_fact_checks']}")
lines.append("")
lines.append("## Warning")
lines.append("This compares report text with parent CXR lineage/EHR direct facts. It does not inspect CXR pixels.")
lines.append("")
lines.append("## By Report Model")
lines.append("| Report model | Checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for model, s in summary["by_report_model"].items():
    lines.append(f"| {model} | {s['checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")

lines.append("")
lines.append("## By Parent CXR Model")
lines.append("| Parent CXR model | Checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for model, s in summary["by_parent_cxr_model"].items():
    lines.append(f"| {model} | {s['checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")

lines.append("")
lines.append("## By Direct Fact")
lines.append("| Direct fact | Checks | Supported | Contradiction | Not mentioned | Support rate | Contradiction rate | Not mentioned rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, s in summary["by_direct_fact"].items():
    lines.append(f"| {label} | {s['checks']} | {s['supported']} | {s['contradiction']} | {s['not_mentioned']} | {s['support_rate']} | {s['contradiction_rate']} | {s['not_mentioned_rate']} |")

(out / "report_cxr_rulebased_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
