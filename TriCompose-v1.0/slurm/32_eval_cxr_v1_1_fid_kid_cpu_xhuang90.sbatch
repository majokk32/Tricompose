#!/bin/bash
#SBATCH --job-name=eval_cxr_fid
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import scipy.linalg
except Exception as e:
    raise RuntimeError("scipy is required for FID. Install scipy in the venv.") from e

try:
    from sklearn.metrics.pairwise import polynomial_kernel
except Exception as e:
    raise RuntimeError("scikit-learn is required for KID. Install scikit-learn in the venv.") from e

cxr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/cxr_candidates")
real_manifest = Path("/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals/manifest.csv")
out = Path("eval/cxr_v1_1_pool80")
out.mkdir(parents=True, exist_ok=True)

random.seed(4200)

def load_image_feature(path, size=128):
    # Lightweight image feature: normalized grayscale pixels + simple intensity stats.
    # This is not official Inception FID. It is a CPU-safe proxy until GPU feature extraction is set up.
    im = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    stats = np.array([
        arr.mean(),
        arr.std(),
        np.percentile(arr, 5),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 95),
    ], dtype=np.float32)
    return np.concatenate([flat, stats], axis=0)

def frechet_distance(x, y):
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y
    covmean = scipy.linalg.sqrtm(cov_x @ cov_y)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(cov_x + cov_y - 2 * covmean)
    return float(fid)

def kid_polynomial(x, y):
    # Biased polynomial-kernel MMD estimate, sufficient for quick model comparison.
    k_xx = polynomial_kernel(x, x, degree=3, gamma=1.0 / x.shape[1], coef0=1)
    k_yy = polynomial_kernel(y, y, degree=3, gamma=1.0 / y.shape[1], coef0=1)
    k_xy = polynomial_kernel(x, y, degree=3, gamma=1.0 / x.shape[1], coef0=1)
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())

synthetic = []
for img_path in sorted(cxr_root.glob("pool80_*/candidates/*/synthetic_cxr.png")):
    run_name = img_path.parts[-4]
    if "roentgen" in run_name:
        model = "roentgen_v2"
    elif "sana" in run_name:
        model = "chexgenbench_sana"
    elif "pixart" in run_name:
        model = "chexgenbench_pixart"
    else:
        model = "unknown"

    synthetic.append({
        "model": model,
        "path": str(img_path),
    })

real_paths = []
with real_manifest.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row.get("cxr_path")
        if p and Path(p).exists():
            real_paths.append(p)

if len(real_paths) < len(synthetic):
    raise RuntimeError(f"not enough real images: real={len(real_paths)}, synthetic={len(synthetic)}")

real_sample = random.sample(real_paths, len(synthetic))

print("synthetic images:", len(synthetic), flush=True)
print("real reference sample:", len(real_sample), flush=True)

real_features = np.stack([load_image_feature(p) for p in real_sample])

by_model = {}
for model in sorted(set(x["model"] for x in synthetic)):
    paths = [x["path"] for x in synthetic if x["model"] == model]
    syn_features = np.stack([load_image_feature(p) for p in paths])

    # Compare each 80-image model set against 80 sampled real images for same sample size.
    real_for_model = real_features[:len(paths)]

    by_model[model] = {
        "synthetic_count": len(paths),
        "real_reference_count": len(real_for_model),
        "proxy_fid": frechet_distance(syn_features, real_for_model),
        "proxy_kid": kid_polynomial(syn_features, real_for_model),
    }

syn_all_features = np.stack([load_image_feature(x["path"]) for x in synthetic])

summary = {
    "evaluation_name": "cxr_v1_1_pool80_realism_proxy_fid_kid",
    "synthetic_cxr_root": str(cxr_root),
    "real_reference_manifest": str(real_manifest),
    "synthetic_count": len(synthetic),
    "real_reference_count": len(real_sample),
    "feature_type": "cpu_lightweight_128x128_grayscale_pixels_plus_intensity_stats",
    "warning": "This is a proxy FID/KID, not official Inception FID/KID and not medical-domain feature FID.",
    "overall": {
        "proxy_fid": frechet_distance(syn_all_features, real_features),
        "proxy_kid": kid_polynomial(syn_all_features, real_features),
    },
    "by_model": by_model,
}

(out / "cxr_realism_proxy_fid_kid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# CXR V1.1 Pool-80 Realism Proxy FID/KID")
lines.append("")
lines.append(f"- Synthetic CXR root: `{cxr_root}`")
lines.append(f"- Real reference manifest: `{real_manifest}`")
lines.append(f"- Synthetic images: {summary['synthetic_count']}")
lines.append(f"- Real reference images: {summary['real_reference_count']}")
lines.append(f"- Feature type: `{summary['feature_type']}`")
lines.append("")
lines.append("## Warning")
lines.append("This is a CPU-safe proxy FID/KID, not official Inception FID/KID and not medical-domain feature FID.")
lines.append("")
lines.append("## Overall")
lines.append(f"- Proxy FID: {summary['overall']['proxy_fid']:.6f}")
lines.append(f"- Proxy KID: {summary['overall']['proxy_kid']:.6f}")
lines.append("")
lines.append("## Per-Model")
lines.append("| Model | Synthetic count | Real reference count | Proxy FID | Proxy KID |")
lines.append("|---|---:|---:|---:|---:|")
for model, s in sorted(summary["by_model"].items()):
    lines.append(
        f"| {model} | {s['synthetic_count']} | {s['real_reference_count']} | "
        f"{s['proxy_fid']:.6f} | {s['proxy_kid']:.6f} |"
    )
lines.append("")
lines.append("## Interpretation")
lines.append("- Lower proxy FID/KID suggests the synthetic image distribution is closer to the sampled real CXR distribution.")
lines.append("- Because this uses grayscale pixel features, it should be used only for early debugging and model comparison.")
lines.append("- Final paper/report evaluation should use Inception FID/KID and preferably TorchXRayVision feature FID.")

(out / "cxr_realism_proxy_fid_kid_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
