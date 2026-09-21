#!/bin/bash
#SBATCH --job-name=eval_cxr_fid_light
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/TriCompose-v1.0/logs/%x_%j.err

set -euo pipefail

cd /project2/ruishanl_1185/inference_3mod/TriCompose-v1.0

runtime_xhuang90/venvs/synehrgy-v2/bin/python - <<'PY'
from pathlib import Path
from PIL import Image, ImageFilter
import csv, json, random, math
import numpy as np
from scipy import linalg

cxr_root = Path("/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/cxr_candidates")
real_manifest = Path("/project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals/manifest.csv")
out = Path("eval/cxr_v1_1_pool80")
out.mkdir(parents=True, exist_ok=True)

random.seed(20260813)

def model_from_run(run_name):
    if "roentgen" in run_name:
        return "roentgen_v2"
    if "sana" in run_name:
        return "chexgenbench_sana"
    if "pixart" in run_name:
        return "chexgenbench_pixart"
    return "unknown"

def image_features(path):
    im = Image.open(path).convert("L").resize((256, 256))
    arr = np.asarray(im, dtype=np.float32) / 255.0

    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float64)

    stats = np.array([
        arr.mean(),
        arr.std(),
        arr.min(),
        arr.max(),
        np.percentile(arr, 1),
        np.percentile(arr, 5),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 95),
        np.percentile(arr, 99),
    ], dtype=np.float64)

    edges = im.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
    edge_stats = np.array([
        edge_arr.mean(),
        edge_arr.std(),
        float((edge_arr > 0.1).mean()),
    ], dtype=np.float64)

    return np.concatenate([hist, stats, edge_stats])

def frechet_distance(x, y):
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    sigma1 = np.cov(x, rowvar=False)
    sigma2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def polynomial_mmd2(x, y):
    gamma = 1.0 / x.shape[1]
    coef = 1.0
    degree = 3

    kxx = (gamma * x.dot(x.T) + coef) ** degree
    kyy = (gamma * y.dot(y.T) + coef) ** degree
    kxy = (gamma * x.dot(y.T) + coef) ** degree

    n = x.shape[0]
    m = y.shape[0]

    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)

    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())

synthetic = []
for img_path in sorted(cxr_root.glob("pool80_*/candidates/*/synthetic_cxr.png")):
    run_name = img_path.parts[-4]
    synthetic.append({
        "path": str(img_path),
        "model": model_from_run(run_name),
        "run_name": run_name,
        "candidate_id": img_path.parts[-2],
    })

real_paths = []
with real_manifest.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row.get("cxr_path")
        if p and Path(p).exists():
            real_paths.append(p)

real_sample = random.sample(real_paths, min(len(real_paths), len(synthetic)))

syn_features = []
for row in synthetic:
    syn_features.append(image_features(row["path"]))
syn_features = np.stack(syn_features)

real_features = []
for p in real_sample:
    real_features.append(image_features(p))
real_features = np.stack(real_features)

summary = {
    "evaluation_name": "cxr_v1_1_pool80_light_proxy_fid_kid",
    "synthetic_cxr_root": str(cxr_root),
    "real_reference_manifest": str(real_manifest),
    "feature_type": "64-bin grayscale histogram + intensity stats + edge stats",
    "synthetic_count": len(synthetic),
    "real_reference_count": len(real_sample),
    "overall": {
        "proxy_fid": round(frechet_distance(syn_features, real_features), 6),
        "proxy_kid": round(polynomial_mmd2(syn_features, real_features), 6),
    },
    "by_model": {},
    "warning": "CPU-safe proxy FID/KID, not official Inception FID/KID or medical-domain feature FID.",
}

for model in sorted(set(r["model"] for r in synthetic)):
    idx = [i for i, r in enumerate(synthetic) if r["model"] == model]
    x = syn_features[idx]
    y = real_features[:len(idx)]

    summary["by_model"][model] = {
        "synthetic_count": len(idx),
        "real_reference_count": len(y),
        "proxy_fid": round(frechet_distance(x, y), 6),
        "proxy_kid": round(polynomial_mmd2(x, y), 6),
    }

(out / "cxr_light_proxy_fid_kid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = []
lines.append("# CXR V1.1 Pool-80 Light Proxy FID/KID")
lines.append("")
lines.append(f"- Synthetic images: {summary['synthetic_count']}")
lines.append(f"- Real reference images: {summary['real_reference_count']}")
lines.append(f"- Feature type: `{summary['feature_type']}`")
lines.append("")
lines.append("## Warning")
lines.append("This is a CPU-safe proxy FID/KID, not official Inception FID/KID and not medical-domain feature FID.")
lines.append("")
lines.append("## Overall")
lines.append(f"- Proxy FID: {summary['overall']['proxy_fid']}")
lines.append(f"- Proxy KID: {summary['overall']['proxy_kid']}")
lines.append("")
lines.append("## Per Model")
lines.append("| Model | Synthetic count | Real reference count | Proxy FID | Proxy KID |")
lines.append("|---|---:|---:|---:|---:|")
for model, s in summary["by_model"].items():
    lines.append(f"| {model} | {s['synthetic_count']} | {s['real_reference_count']} | {s['proxy_fid']} | {s['proxy_kid']} |")
lines.append("")
lines.append("## Interpretation")
lines.append("- Lower proxy FID/KID suggests the synthetic image distribution is closer to the sampled real CXR distribution.")
lines.append("- This is useful for early debugging and model comparison.")
lines.append("- Final evaluation should use TorchXRayVision feature FID or another medical-domain image encoder.")

(out / "cxr_light_proxy_fid_kid_summary.md").write_text("\n".join(lines) + "\n")

print("saved:", out)
print(json.dumps(summary, indent=2))
PY
