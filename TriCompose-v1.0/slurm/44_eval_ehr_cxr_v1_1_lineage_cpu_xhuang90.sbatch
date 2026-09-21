#!/bin/bash
#SBATCH --job-name=eval_ehr_cxr
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
import json
from pathlib import Path
from collections import Counter, defaultdict

staging_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/staging/bridge_gpt2_pool80_v11_20260813_003/cases")
cxr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/cxr_candidates")
out = Path("eval/ehr_cxr_v1_1_pool80")
out.mkdir(parents=True, exist_ok=True)

cxr_runs = [
    "pool80_roentgen_v2_fp16_s1_20260813_0bd4",
    "pool80_chexgenbench_sana_s1_20260813_d0ce",
    "pool80_chexgenbench_pixart_s1_20260813_18b7",
]

direct_labels = [
    "congestive_heart_failure",
    "pneumonia",
    "cardiac_pacemaker",
    "pleural_effusion",
]

case_direct = {}
for case_dir in sorted(staging_root.iterdir()):
    if not case_dir.is_dir():
        continue
    facts = json.loads((case_dir / "ehr_facts.json").read_text())
    pos = []
    for label, obj in facts.get("direct_facts", {}).items():
        status = obj.get("status") or obj.get("state")
        if status in ("documented", "positive"):
            pos.append(label)
    case_direct[case_dir.name] = sorted(pos)

rows = []
by_model = defaultdict(Counter)
by_label = defaultdict(Counter)

for run_id in cxr_runs:
    run = cxr_root / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    model = manifest.get("model_id")

    for item in manifest.get("candidates", []):
        cand = json.loads((run / item["path"]).read_text())
        case_id = cand["case_id"]
        labels = case_direct.get(case_id, [])

        lineage_ok = (
            cand.get("modality") == "cxr"
            and cand.get("frozen_model") is True
            and cand.get("adapter_added_prefix") is False
            and bool(cand.get("ehr_sha256"))
            and bool(cand.get("ehr_facts_sha256"))
            and bool(cand.get("clinical_intent_sha256"))
            and bool(cand.get("prompt_sha256"))
            and Path(cand["artifact"]["path"]).exists()
        )

        row = {
            "case_id": case_id,
            "cxr_candidate_id": cand["candidate_id"],
            "cxr_model": model,
            "lineage_ok": lineage_ok,
            "direct_facts": labels,
            "direct_fact_count": len(labels),
            "image_path": cand["artifact"]["path"],
        }
        rows.append(row)

        by_model[model]["candidates"] += 1
        by_model[model]["lineage_ok"] += int(lineage_ok)
        by_model[model]["with_direct_fact"] += int(bool(labels))

        for label in labels:
            by_label[label]["cxr_candidates_conditioned"] += 1
            by_label[label][model] += 1

summary = {
    "evaluation_name": "ehr_cxr_v1_1_pool80_lineage_conditioning",
    "ehr_cases": len(case_direct),
    "cxr_candidates": len(rows),
    "cxr_runs": cxr_runs,
    "lineage_ok": sum(r["lineage_ok"] for r in rows),
    "with_direct_fact": sum(1 for r in rows if r["direct_facts"]),
    "direct_fact_labels": direct_labels,
    "by_model": {
        model: {
            "candidates": c["candidates"],
            "lineage_ok": c["lineage_ok"],
            "lineage_ok_rate": round(c["lineage_ok"] / c["candidates"], 6) if c["candidates"] else None,
            "with_direct_fact": c["with_direct_fact"],
        }
        for model, c in sorted(by_model.items())
    },
    "by_direct_fact": {
        label: dict(c)
        for label, c in sorted(by_label.items())
    },
    "warning": "This checks EHR-to-CXR lineage/conditioning only, not pixel-level clinical correctness.",
}

(out / "ehr_cxr_lineage_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "ehr_cxr_lineage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# EHR-CXR V1.1 Lineage / Conditioning Consistency")
lines.append("")
lines.append(f"- EHR cases: {summary['ehr_cases']}")
lines.append(f"- CXR candidates: {summary['cxr_candidates']}")
lines.append(f"- Lineage OK: {summary['lineage_ok']} / {summary['cxr_candidates']}")
lines.append(f"- CXR candidates with EHR direct facts: {summary['with_direct_fact']}")
lines.append("")
lines.append("## Warning")
lines.append("This checks whether CXR candidates correctly retain EHR lineage and conditioning hashes. It does not verify pixel-level clinical findings.")
lines.append("")
lines.append("## Per CXR Model")
lines.append("| CXR model | Candidates | Lineage OK | Lineage OK rate | With direct fact |")
lines.append("|---|---:|---:|---:|---:|")
for model, s in summary["by_model"].items():
    lines.append(f"| {model} | {s['candidates']} | {s['lineage_ok']} | {s['lineage_ok_rate']} | {s['with_direct_fact']} |")
lines.append("")
lines.append("## Per Direct Fact")
lines.append("| Direct fact | Conditioned CXR candidates |")
lines.append("|---|---:|")
for label, s in summary["by_direct_fact"].items():
    lines.append(f"| {label} | {s.get('cxr_candidates_conditioned', 0)} |")

(out / "ehr_cxr_lineage_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
