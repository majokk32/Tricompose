#!/bin/bash
#SBATCH --job-name=eval_cxr_div
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

runtime_xhuang90/venvs/synehrgy-v2/bin/python - <<'PY'
from pathlib import Path
from PIL import Image
import json
import itertools
import math
from collections import defaultdict

cxr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/cxr_candidates")
out = Path("eval/cxr_v1_1_pool80")
out.mkdir(parents=True, exist_ok=True)

def model_from_run(run_name):
    if "roentgen" in run_name:
        return "roentgen_v2"
    if "sana" in run_name:
        return "chexgenbench_sana"
    if "pixart" in run_name:
        return "chexgenbench_pixart"
    return "unknown"

def load_gray_vector(path, size=256):
    im = Image.open(path).convert("L").resize((size, size))
    pix = list(im.getdata())
    return pix

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def pearson_similarity(a, b):
    # Fast normalized pixel correlation, used as a lightweight duplicate/diversity proxy.
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)

    num = 0.0
    da = 0.0
    db = 0.0

    for x, y in zip(a, b):
        xa = x - ma
        yb = y - mb
        num += xa * yb
        da += xa * xa
        db += yb * yb

    if da == 0 or db == 0:
        return 1.0 if da == db else 0.0

    return num / math.sqrt(da * db)

records = []
vectors = {}

for img_path in sorted(cxr_root.glob("pool80_*/candidates/*/synthetic_cxr.png")):
    run_name = img_path.parts[-4]
    candidate_id = img_path.parts[-2]
    model = model_from_run(run_name)

    records.append({
        "model": model,
        "run_name": run_name,
        "candidate_id": candidate_id,
        "image_path": str(img_path),
    })
    vectors[candidate_id] = load_gray_vector(img_path)

pair_rows = []
overall_sims = []
by_model_sims = defaultdict(list)

for a, b in itertools.combinations(records, 2):
    sim = pearson_similarity(vectors[a["candidate_id"]], vectors[b["candidate_id"]])
    same_model = a["model"] == b["model"]

    row = {
        "candidate_id_a": a["candidate_id"],
        "candidate_id_b": b["candidate_id"],
        "model_a": a["model"],
        "model_b": b["model"],
        "same_model": same_model,
        "pixel_correlation": sim,
        "near_duplicate_0_99": sim >= 0.99,
        "near_duplicate_0_995": sim >= 0.995,
    }
    pair_rows.append(row)
    overall_sims.append(sim)

    if same_model:
        by_model_sims[a["model"]].append(sim)

summary = {
    "evaluation_name": "cxr_v1_1_pool80_diversity_pixel_correlation",
    "synthetic_cxr_root": str(cxr_root),
    "image_count": len(records),
    "pair_count": len(pair_rows),
    "metric": "downsampled_256x256_grayscale_pixel_correlation",
    "overall": {
        "mean_similarity": round(mean(overall_sims), 6),
        "min_similarity": round(min(overall_sims), 6),
        "max_similarity": round(max(overall_sims), 6),
        "near_duplicate_rate_0_99": round(sum(r["near_duplicate_0_99"] for r in pair_rows) / len(pair_rows), 6),
        "near_duplicate_rate_0_995": round(sum(r["near_duplicate_0_995"] for r in pair_rows) / len(pair_rows), 6),
    },
    "by_model": {},
}

for model, sims in sorted(by_model_sims.items()):
    model_pairs = [r for r in pair_rows if r["same_model"] and r["model_a"] == model]
    summary["by_model"][model] = {
        "pair_count": len(sims),
        "mean_similarity": round(mean(sims), 6),
        "min_similarity": round(min(sims), 6),
        "max_similarity": round(max(sims), 6),
        "near_duplicate_rate_0_99": round(sum(r["near_duplicate_0_99"] for r in model_pairs) / len(model_pairs), 6),
        "near_duplicate_rate_0_995": round(sum(r["near_duplicate_0_995"] for r in model_pairs) / len(model_pairs), 6),
    }

(out / "cxr_diversity_pair_rows.json").write_text(json.dumps(pair_rows, indent=2) + "\n")
(out / "cxr_diversity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# CXR V1.1 Pool-80 Diversity Evaluation")
lines.append("")
lines.append(f"- Synthetic CXR root: `{cxr_root}`")
lines.append(f"- Image count: {summary['image_count']}")
lines.append(f"- Pair count: {summary['pair_count']}")
lines.append(f"- Metric: {summary['metric']}")
lines.append("")
lines.append("## Overall")
for k, v in summary["overall"].items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("## Per-Model")
lines.append("| Model | Pair count | Mean similarity | Min | Max | Near-duplicate >=0.99 | Near-duplicate >=0.995 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for model, s in summary["by_model"].items():
    lines.append(
        f"| {model} | {s['pair_count']} | {s['mean_similarity']} | {s['min_similarity']} | "
        f"{s['max_similarity']} | {s['near_duplicate_rate_0_99']} | {s['near_duplicate_rate_0_995']} |"
    )
lines.append("")
lines.append("## Interpretation")
lines.append("- Lower similarity indicates higher visual diversity.")
lines.append("- Near-duplicate rate estimates how often generated CXR images are almost identical.")
lines.append("- This is a lightweight pixel-level proxy, not a clinical diversity metric.")
lines.append("- A stronger follow-up is MS-SSIM or LPIPS after installing the required packages.")

(out / "cxr_diversity_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
