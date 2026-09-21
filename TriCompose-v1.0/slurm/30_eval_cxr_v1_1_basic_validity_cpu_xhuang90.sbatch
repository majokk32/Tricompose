#!/bin/bash
#SBATCH --job-name=eval_cxr_basic
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
from PIL import Image, ImageStat
import json
from collections import Counter, defaultdict

cxr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/cxr_candidates")
out = Path("eval/cxr_v1_1_pool80")
out.mkdir(parents=True, exist_ok=True)

rows = []
by_model = defaultdict(list)

for img_path in sorted(cxr_root.glob("pool80_*/candidates/*/synthetic_cxr.png")):
    run_name = img_path.parts[-4]
    candidate_id = img_path.parts[-2]

    try:
        im = Image.open(img_path).convert("L")
        w, h = im.size
        stat = ImageStat.Stat(im)
        mean = float(stat.mean[0])
        std = float(stat.stddev[0])
        min_px, max_px = im.getextrema()

        blank = std < 1.0
        corrupted = False
    except Exception:
        w = h = mean = std = min_px = max_px = None
        blank = None
        corrupted = True

    if "roentgen" in run_name:
        model = "roentgen_v2"
    elif "sana" in run_name:
        model = "chexgenbench_sana"
    elif "pixart" in run_name:
        model = "chexgenbench_pixart"
    else:
        model = "unknown"

    row = {
        "model": model,
        "run_name": run_name,
        "candidate_id": candidate_id,
        "image_path": str(img_path),
        "width": w,
        "height": h,
        "mean_intensity": mean,
        "std_intensity": std,
        "min_intensity": min_px,
        "max_intensity": max_px,
        "blank": blank,
        "corrupted": corrupted,
    }

    rows.append(row)
    by_model[model].append(row)

summary = {
    "evaluation_name": "cxr_v1_1_pool80_basic_validity",
    "synthetic_cxr_root": str(cxr_root),
    "total_images": len(rows),
    "corrupted_images": sum(1 for r in rows if r["corrupted"]),
    "blank_images": sum(1 for r in rows if r["blank"] is True),
    "resolution_counts": {
        f"{k[0]}x{k[1]}": v
        for k, v in Counter((r["width"], r["height"]) for r in rows if not r["corrupted"]).items()
    },
    "by_model": {},
}

for model, xs in sorted(by_model.items()):
    valid = [r for r in xs if not r["corrupted"]]
    summary["by_model"][model] = {
        "count": len(xs),
        "corrupted": sum(1 for r in xs if r["corrupted"]),
        "blank": sum(1 for r in xs if r["blank"] is True),
        "resolution_counts": dict(Counter(f'{r["width"]}x{r["height"]}' for r in valid)),
        "mean_intensity_avg": round(sum(r["mean_intensity"] for r in valid) / len(valid), 4) if valid else None,
        "std_intensity_avg": round(sum(r["std_intensity"] for r in valid) / len(valid), 4) if valid else None,
        "min_intensity_min": min((r["min_intensity"] for r in valid), default=None),
        "max_intensity_max": max((r["max_intensity"] for r in valid), default=None),
    }

(out / "cxr_basic_validity_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "cxr_basic_validity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# CXR V1.1 Pool-80 Basic Validity Evaluation")
lines.append("")
lines.append(f"- Synthetic CXR root: `{cxr_root}`")
lines.append(f"- Total images: {summary['total_images']}")
lines.append(f"- Corrupted images: {summary['corrupted_images']}")
lines.append(f"- Blank images: {summary['blank_images']}")
lines.append("")
lines.append("## Resolution Counts")
for k, v in summary["resolution_counts"].items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("## Per-Model Summary")
lines.append("| Model | Count | Corrupted | Blank | Resolution | Mean intensity | Std intensity |")
lines.append("|---|---:|---:|---:|---|---:|---:|")
for model, s in summary["by_model"].items():
    res = ", ".join(f"{k}: {v}" for k, v in s["resolution_counts"].items())
    lines.append(
        f"| {model} | {s['count']} | {s['corrupted']} | {s['blank']} | {res} | "
        f"{s['mean_intensity_avg']} | {s['std_intensity_avg']} |"
    )

lines.append("")
lines.append("## Interpretation")
lines.append("- This checks whether generated CXR files are readable, non-blank, and resolution-consistent.")
lines.append("- It does not measure clinical correctness yet.")
lines.append("- Clinical accuracy requires a CXR classifier such as TorchXRayVision or CheXpert-style labeler.")
lines.append("- FID/KID requires a real CXR reference image set.")

(out / "cxr_basic_validity_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
