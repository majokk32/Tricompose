"""Local, frozen subprocess workers. No inference happens at import/preflight.

The existing model adapters are reused unchanged. One invocation consumes one
case, one candidate and one frozen model/scorer. Native child output is private.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess

from contracts import (WORKSPACE, PROTECTED_ROOT, read_json, require_inside,
                       sha256_file, private_directory, write_private_json)
from .invariant_verification import _digest, _ID
from .runtime_dispatch import _new_private_handle, require_slurm

EVAL = WORKSPACE / "TriCompose-v1.0/eval/report_v1_1"
XRV_PYTHON = Path("/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python")
XRV_WEIGHT = (WORKSPACE / ".cache/tricompose_v1/scoring/job_10871232/xrv/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt")
CHEXBERT_WEIGHT = WORKSPACE / ".cache/tricompose_report_eval/models/chexbert/chexbert.pth"
CHEXBERT_BERT = WORKSPACE / ".cache/tricompose_report_eval/models/bert-base-uncased"


def registry():
    """Availability/VRAM are not certified by registration. All inputs are local."""
    base = "/project2/ruishanl_1185/Baselines/hf_cache/hub"
    liquid = "/project2/ruishanl_1185/SDP_for_VLM"
    rows = {
        "roentgen_v2": ("cxr_generator", "runtime/venvs/roentgen-v2-cu126-v1/bin/python",
            "runtime/models/roentgen-v2/c88d2bf041a448f505fc13dc009d6991be1c9b0f", ["--precision", "float16"], 16),
        "chexgenbench_sana": ("cxr_generator", "CheXGenBench/venv/bin/python",
            "CheXGenBench/checkpoints/sana-e20/6c8989d92e75fbcfc91a82c3570389293867ad6a", [], 16),
        "chexgenbench_pixart": ("cxr_generator", "CheXGenBench/venv/bin/python",
            "CheXGenBench/checkpoints/pixart-sigma/ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727", [], 24),
        "cxrmate_single": ("report_generator", "cxrmate/venv/bin/python",
            base+"/models--aehrc--cxrmate-single-tf/snapshots/84dfcba8125c9b9296bfc1ee317749788b6b59b4", [], 16),
        "maira2": ("report_generator", "maria-2/venv/bin/python",
            base+"/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb", [], 40),
        "llavarad": ("report_generator", liquid+"/envs/llavarad/bin/python", liquid+"/models/llava_rad",
            ["--model-base", liquid+"/models/vicuna-7b-v1.5", "--biomedbert-dir",
             "/home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d",
             "--external-root", liquid+"/SDP_for_VLM/llava-rad"], 24),
        "chexagent2": ("report_generator", "CheXagent-2/venv/bin/python",
            "CheXagent-2/checkpoints/chexagent-2-srrg-findings/9f7225fc382ddd1297ade1aa796da660237940bc",
            ["--vision-dir", str(WORKSPACE/"CheXagent-2/checkpoints/chexagent2-xraysiglip/f0edbf5d90dba44edb7f4f96d8663537cb0749bf"), "--precision", "float32"], 24),
    }
    result = {}
    for name, (kind, python, model, extra, vram) in rows.items():
        result[name] = {"model_id": name, "kind": kind, "python": str(WORKSPACE/python),
            "model_dir": str(WORKSPACE/model), "extra_args": extra,
            "minimum_planning_vram_gib": vram, "vram_estimate_not_measured": True,
            "script": str(WORKSPACE/"TriCompose-v1.1/tools"/("run_cxr.py" if kind=="cxr_generator" else "run_report.py")),
            "input_signature": "ehr_derived_radiology_text_to_cxr" if kind=="cxr_generator" else "single_current_synthetic_cxr_to_report",
            "native_structured_ehr_conditioning": False}
    for name, kind, python, weight, script in (
        ("xrv", "xrv", XRV_PYTHON, XRV_WEIGHT, "extract_cxr_labels_xrv.py"),
        ("chexbert", "chexbert", WORKSPACE/"cxrmate/venv/bin/python", CHEXBERT_WEIGHT, "extract_report_labels_chexbert.py")):
        result[name] = {"model_id": name, "kind": kind, "python": str(python),
            "checkpoint": str(weight), "script": str(EVAL/script),
            "minimum_planning_vram_gib": 4, "vram_estimate_not_measured": True}
    return result


def require_gpu_slurm():
    require_slurm()
    assigned = any(os.environ.get(k, "") not in ("", "0", "(null)")
        for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not assigned or visible in ("", "-1", "NoDevFiles", "void"):
        raise RuntimeError("explicit GPU Slurm allocation required before worker spawn")


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight_generator(spec):
    """Existing validator only; runtime factories are NEVER instantiated.

    Sana/PixArt validators hash large weights: requires CPU Slurm, not login.
    """
    require_slurm()
    for key in ("python", "model_dir", "script"):
        if not Path(spec[key]).exists(): raise FileNotFoundError("required local worker asset missing")
    if not os.access(spec["python"], os.X_OK): raise ValueError("worker Python is not executable")
    module = load_script(spec["script"], "_tricompose_preflight_"+spec["model_id"])
    extras = spec.get("extra_args", [])
    if spec["kind"] == "cxr_generator":
        precision = extras[extras.index("--precision")+1] if "--precision" in extras else None
        revision, audit, unused_factory = module._model_components(spec["model_id"], spec["model_dir"], precision)
    else:
        opts = {name: None for name in ("model_base", "biomedbert_dir", "external_root", "vision_dir", "precision")}
        opts.update(model_id=spec["model_id"], model_dir=spec["model_dir"], runtime_dir=str(PROTECTED_ROOT/"unused_preflight_runtime"))
        opts.update({extras[i][2:].replace("-", "_"): extras[i+1] for i in range(0, len(extras), 2)})
        revision, audit, unused_factory = module._model_components(argparse.Namespace(**opts))
    # Strong byte pins supplement adapters that only check sizes/small configs.
    files = {}
    root = Path(spec["model_dir"])
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".py", ".safetensors", ".bin"}:
            relative = str(path.relative_to(root))
            digest = audit.get("weight_sha256", {}).get(relative) or sha256_file(path)
            files[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return {**spec, "status": "preflighted", "model_revision": revision, "model_audit": audit,
        "asset_pins": files, "factory_instantiated": False}


def source_pins():
    # Only known source trees. Never recurse checkpoints, caches, private data or credentials.
    dirs = ("src/tricompose", "TriCompose-v1.0/src", "TriCompose-v1.1/src", "TriCompose-v1.1/tools",
        "TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.0/eval/report_v1_1",
        "experiments/roentgen_v2/src", "experiments/chexgenbench_sana/src", "experiments/chexgenbench_pixart/src",
        "experiments/cxrmate_single_report/src", "experiments/maira2_report/src", "experiments/llavarad_report/src",
        "CheXagent-2/src", "cxrmate/tools")
    return {str(p): sha256_file(p) for d in dirs for p in sorted((WORKSPACE/d).rglob("*.py")) if p.is_file()}


def check_pins(pins):
    for path, expected in pins.items():
        p = Path(path)
        wanted = expected["sha256"] if isinstance(expected, dict) else expected
        if (not p.is_file() or isinstance(expected, dict) and p.stat().st_size != expected["size_bytes"]
                or sha256_file(p) != wanted):
            raise ValueError("frozen source/asset hash changed")


def worker_environment(runtime):
    runtime = require_inside(runtime, PROTECTED_ROOT, must_exist=True)
    for name in ("cache", "tmp"):
        private_directory(runtime/name)
    env = os.environ.copy()
    # Never forward saved service credentials to a local-only worker.
    for key in list(env):
        if (key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
                or key.startswith(("AWS_", "AZURE_", "GOOGLE_APPLICATION_CREDENTIALS"))):
            env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        XDG_CACHE_HOME=str(runtime/"cache/xdg"), HF_HOME=str(runtime/"cache/huggingface"),
        HF_HUB_CACHE=str(runtime/"cache/huggingface/hub"), TRANSFORMERS_CACHE=str(runtime/"cache/huggingface/transformers"),
        TORCH_HOME=str(runtime/"cache/torch"), TORCHINDUCTOR_CACHE_DIR=str(runtime/"cache/inductor"),
        CUDA_CACHE_PATH=str(runtime/"cache/cuda"), MPLCONFIGDIR=str(runtime/"cache/matplotlib"),
        TMPDIR=str(runtime/"tmp"), TEMP=str(runtime/"tmp"), TMP=str(runtime/"tmp"),
        DIFFUSERS_VERBOSITY="error", TRANSFORMERS_VERBOSITY="error",
        PYTHONPATH=os.pathsep.join(str(WORKSPACE/p) for p in ("cxrmate", "TriCompose-v1.0/eval/report_v1_1")))
    return env


def terminate_owned(process):
    """Kill only the newly created worker process group, including grandchildren."""
    for sig, delay in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try: os.killpg(process.pid, sig)
        except ProcessLookupError: pass
        try: process.wait(timeout=delay)
        except subprocess.TimeoutExpired: continue
        # The leader may have exited before a grandchild. Finish its owned group.
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        return
    raise RuntimeError("owned worker process failed to terminate")


def run_private_process(argv, runtime, timeout):
    require_gpu_slurm()  # Before log creation, environment setup or subprocess.
    if type(timeout) is not int or timeout < 1: raise ValueError("bounded timeout required")
    env = worker_environment(runtime)
    with _new_private_handle(Path(runtime)/"native.stdout.log") as out, \
            _new_private_handle(Path(runtime)/"native.stderr.log") as err:
        child = subprocess.Popen(argv, cwd=str(WORKSPACE), env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, shell=False, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_owned(child)
            raise TimeoutError("bounded local worker timeout") from None
        except BaseException:
            terminate_owned(child)
            raise
        if code != 0: raise RuntimeError("local worker failed; see protected native log")


def argv_for(spec, *, request_run=None, cxr_run=None, report_run=None, output_root, thresholds=None):
    args = [spec["python"], spec["script"]]
    if spec["kind"] in {"cxr_generator", "report_generator"}:
        args += ["--request-run", str(request_run), "--output-root", str(output_root),
            "--output-run-id", "generated", "--model-id", spec["model_id"], "--model-dir", spec["model_dir"]]
        if spec["kind"] == "cxr_generator": args += ["--batch-size", "1"]
        args += spec.get("extra_args", [])
        if spec["model_id"] == "llavarad": args += ["--runtime-dir", str(Path(output_root)/"runtime/tmp/llavarad")]
    elif spec["kind"] == "xrv":
        weight = Path(spec["checkpoint"])
        args += ["--cxr-run", str(cxr_run), "--cache-dir", str(weight.parent), "--weight-filename", weight.name,
            "--model-name", "densenet121-res224-all", "--thresholds", str(thresholds), "--output-root", str(output_root), "--run-id", "scored"]
    elif spec["kind"] == "chexbert":
        args += ["--cxr-run", str(cxr_run), "--report-run", str(report_run), "--checkpoint", spec["checkpoint"],
            "--bert-path", str(CHEXBERT_BERT), "--batch-size", "1", "--output-root", str(output_root), "--run-id", "scored"]
    else: raise ValueError("unregistered worker kind")
    if any(v is None or str(v) == "None" for v in args): raise ValueError("missing exact worker input")
    return args


class LocalFrozenBackend:
    frozen = True
    execution_mode = "approved_slurm_backend"

    def __init__(self, spec, *, operation_root, timeout, request_run=None, cxr_run=None,
                 report_run=None, thresholds=None):
        if spec.get("status") != "preflighted": raise ValueError("worker has not passed preflight")
        self.spec = spec
        self.audit_sha256 = _digest(spec)
        self.root = Path(operation_root)
        self.timeout = timeout
        self.inputs = dict(request_run=request_run, cxr_run=cxr_run, report_run=report_run, thresholds=thresholds)

    def invoke(self, request):
        require_gpu_slurm()
        if request.model_id != self.spec["model_id"] or request.kind != self.spec["kind"]:
            raise ValueError("registered worker differs from charged operation")
        check_pins(self.spec["asset_pins"])
        private_directory(self.root/"runtime")
        args = argv_for(self.spec, output_root=self.root, **self.inputs)
        run_private_process(args, self.root/"runtime", self.timeout)
        check_pins(self.spec["asset_pins"])
        return {"output_root": str(self.root), "worker_audit_sha256": self.audit_sha256}


def single_generated(payload, spec, request_run, *, cxr_candidates=None):
    from contracts import load_cxr_candidates, load_report_candidates
    root = require_inside(payload["output_root"], PROTECTED_ROOT, must_exist=True)/"generated"
    manifest = read_json(root/"manifest.json")
    if (payload["worker_audit_sha256"] != _digest(spec) or manifest.get("frozen_model") is not True
            or manifest.get("model_id") != spec["model_id"] or manifest.get("model_revision") != spec["model_revision"]
            or manifest.get("model_audit") != spec["model_audit"] or manifest.get("candidate_count") != 1
            or manifest.get("source_request_run_manifest_sha256") != sha256_file(Path(request_run)/"manifest.json")):
        raise ValueError("single-call frozen generation audit differs")
    loaded = (load_cxr_candidates([root]) if spec["kind"]=="cxr_generator" else
        load_report_candidates([root], cxr_candidates=cxr_candidates))
    if len(loaded) != 1: raise ValueError("hidden batch generation is forbidden")
    candidate = next(iter(loaded.values()))
    if candidate["cost"].get("model_calls") != 1: raise ValueError("generation model-call count differs")
    artifact = require_inside(candidate["artifact"]["path"], root, must_exist=True)
    if spec["kind"] == "cxr_generator":
        import struct
        from tricompose_v11.tokenizer_trace import validate_tokenizer_trace
        validate_tokenizer_trace(candidate, PROTECTED_ROOT)
        with artifact.open("rb") as handle: header = handle.read(24)
        if (header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR"
                or list(struct.unpack(">II", header[16:24])) != candidate["artifact"]["dimensions"]):
            raise ValueError("PNG dimensions/header differ; no anatomy verdict implied")
    elif artifact.stat().st_size == 0:
        raise ValueError("empty report artifact")
    return root, candidate
