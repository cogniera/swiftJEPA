#Author : Paarth Sharma
#File Name : benchmark_vjepa2.py
#Project Name : swiftJEPA
#Creation Date : 16th September 2026
#Modification Date : 14th October 2026
#Description : Measure raw video to embeddings with the official input pipeline.

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

# Resolve the project and upstream packages before importing project helpers.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))
from benchmarks.pipeline import make_preprocess
from benchmarks.reporting import collect_environment, run_validity, summarize_samples

PUBLIC_CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2"
MODELS = {
    "vitl": ("vjepa2_vit_large", 256),
    "vith": ("vjepa2_vit_huge", 256),
    "vitg": ("vjepa2_vit_giant", 256),
    "vitg-384": ("vjepa2_vit_giant_384", 384),
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# Pretrained runs are the default; random weights are explicit diagnostic input.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Raw uint8 to V-JEPA 2 embeddings")
    parser.add_argument("--model", default="vitl", choices=MODELS)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", default="fp16", choices=DTYPES)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="compile the official encoder with max-autotune")
    parser.add_argument("--skip-accuracy", action="store_true", help="diagnostic run without pretrained quality validation")
    parser.add_argument("--profile", action="store_true", help="export a separate untimed pipeline trace")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--pretrained", action="store_true", dest="pretrained")
    weights.add_argument("--random-weights", action="store_false", dest="pretrained")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-id", default=None, help="shared ID for back-to-back A/B runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.frames % 2:
        parser.error("--frames must be a positive multiple of tubelet size 2")
    if min(args.batch, args.iters, args.warmup, args.cpu_threads, args.opencv_threads) <= 0:
        parser.error("batch, iterations, warmup and thread counts must be positive")
    if args.compile:
        args.warmup = max(args.warmup, 20)
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must select a CUDA device")
    args.device = f"cuda:{device.index if device.index is not None else 0}"
    return args


# Keep the encoder architecture, fp32 weights, SDPA, and forward implementation untouched.
def build_encoder(args):
    import src.hub.backbones as backbones

    backbones.VJEPA_BASE_URL = PUBLIC_CKPT_URL
    builder_name, crop_size = MODELS[args.model]
    encoder, predictor = getattr(backbones, builder_name)(
        pretrained=args.pretrained, num_frames=args.frames
    )
    del predictor
    return encoder.eval().to(args.device), crop_size


# Clip generation represents already decoded frames and stays outside inference timing.
def make_clip(batch, frames, img_size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (batch, frames, img_size, img_size, 3), dtype=np.uint8)


# Reuse initialized events so CUDA allocation and lazy initialization stay out of samples.
def make_events(stream):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for event in events:
        event.record(stream)
    events[-1].synchronize()
    return events


# Timestamp before CPU work, then corroborate the full elapsed interval with wall time.
def measure_iteration(encoder, preprocess, clip, device, dtype, events, stream, compiled=False):
    pipeline_start, gpu_start, after_h2d, end = events
    if compiled:
        torch.compiler.cudagraph_mark_step_begin()
    wall_start = time.perf_counter()
    pipeline_start.record(stream)
    pipeline_start.synchronize()
    preprocess_start = time.perf_counter()
    x_cpu = preprocess(clip).pin_memory()
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
    gpu_start.record(stream)
    x = x_cpu.to(device, non_blocking=True)
    after_h2d.record(stream)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        output = encoder(x)
    end.record(stream)
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    row = {
        "end_to_end_ms": pipeline_start.elapsed_time(end),
        "wall_ms": wall_ms,
        "preprocess_cpu_ms": preprocess_ms,
        "h2d_copy_ms": gpu_start.elapsed_time(after_h2d),
        "encoder_cuda_ms": after_h2d.elapsed_time(end),
        "gpu_total_ms": gpu_start.elapsed_time(end),
    }
    del output, x, x_cpu
    return row


# Measure the empty event window for disclosure, without subtracting it from latency.
def measure_instrumentation(events, stream):
    start, _, _, end = events
    samples = []
    for _ in range(10):
        wall_start = time.perf_counter()
        start.record(stream)
        start.synchronize()
        for event in events[1:]:
            event.record(stream)
        end.synchronize()
        samples.append({"cuda_ms": start.elapsed_time(end),
                        "wall_ms": (time.perf_counter() - wall_start) * 1000})
    return {key: summarize_samples([row[key] for row in samples]) for key in samples[0]}


# Preserve tracing counters to detect compilation or silent eager fallback in measured runs.
def compiler_snapshot():
    from torch._dynamo.utils import counters

    return {name: dict(counters[name]) for name in ("frames", "stats", "graph_break")}


# Headline timing has no hooks or profiler; each sample includes official host preparation.
def run_benchmark(args, encoder, crop_size, out_dir):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("benchmark timing requires an available CUDA device")
    preprocess = make_preprocess(crop_size)
    clip = make_clip(args.batch, args.frames, crop_size, args.seed)
    rows = []
    with torch.cuda.device(device), torch.inference_mode():
        stream = torch.cuda.current_stream(device)
        events = make_events(stream)
        instrumentation = measure_instrumentation(events, stream)
        for _ in range(args.warmup):
            measure_iteration(encoder, preprocess, clip, device, DTYPES[args.dtype], events, stream, args.compile)
        counters_before = compiler_snapshot() if args.compile else None
        for _ in range(args.iters):
            rows.append(measure_iteration(
                encoder, preprocess, clip, device, DTYPES[args.dtype], events, stream, args.compile
            ))
        counters_after = compiler_snapshot() if args.compile else None
    return {"samples": rows, "instrumentation": instrumentation,
            "compilation": {"enabled": args.compile,
                            "mode": "max-autotune" if args.compile else None,
                            "before_measured_iterations": counters_before,
                            "after_measured_iterations": counters_after}}


# Preserve per-iteration samples and enough metadata to reject incomparable or noisy runs.
def save_outputs(args, results, out_dir, environment, accuracy=None):
    rows = results["samples"]
    with (out_dir / "timings_per_iteration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", *rows[0]])
        writer.writeheader()
        for iteration, row in enumerate(rows):
            writer.writerow({"iteration": iteration, **row})
    stages = {key: summarize_samples([row[key] for row in rows]) for key in rows[0]}
    validity = run_validity([row["end_to_end_ms"] for row in rows],
                            source_dirty=environment["tracked_dirty"])
    compilation = results.get("compilation", {"enabled": False})
    if compilation["enabled"]:
        before = compilation["before_measured_iterations"]
        after = compilation["after_measured_iterations"]
        if before != after:
            validity["reasons"].append("compiler activity occurred during measured iterations")
        if after["stats"].get("unique_graphs", 0) == 0:
            validity["reasons"].append("compiled encoder captured no graphs")
        validity["passed"] = not validity["reasons"]
    accuracy = accuracy or {"passed": None, "reason": "not validated"}
    summary = {
        "schema_version": 2,
        "config": vars(args),
        "environment": environment,
        "metric": "raw pageable uint8 BTHWC to device embeddings using synchronized CUDA events",
        "preprocess": "official single view resize center crop tensor conversion normalization and pinning",
        "stages": stages,
        "validity": validity,
        "instrumentation": results["instrumentation"],
        "throughput_clips_per_s": args.batch * 1000 / stages["end_to_end_ms"]["mean_ms"],
        "compilation": compilation,
        "accuracy": accuracy,
        "acceptance_eligible": validity["passed"] and args.pretrained and accuracy["passed"] is True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


# Make thread and precision settings explicit and identical across comparator runs.
def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise SystemExit("CUDA is required for benchmark timing")
    import cv2

    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.opencv_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    args.session_id = args.session_id or uuid.uuid4().hex
    tag = f"{args.model}_{args.frames}f_b{args.batch}_{args.dtype}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir = Path(args.out) if args.out else ROOT / "benchmarks" / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    environment = collect_environment(ROOT, args.device)
    (out_dir / "run.json").write_text(json.dumps({"config": vars(args), "environment": environment},
                                                indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"building {args.model} pretrained={args.pretrained}", flush=True)
    encoder, crop_size = build_encoder(args)
    candidate = torch.compile(encoder, mode="max-autotune") if args.compile else encoder
    accuracy = None
    if args.pretrained and not args.skip_accuracy:
        from benchmarks.accuracy import validate_encoder

        preprocess = make_preprocess(crop_size)
        with torch.cuda.device(args.device):
            accuracy = validate_encoder(
                encoder, candidate, lambda clip: preprocess(clip).to(args.device),
                args.device, DTYPES[args.dtype], args.frames, args.batch, crop_size,
                seed=args.seed,
            )
        (out_dir / "accuracy.json").write_text(json.dumps(accuracy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"accuracy passed={accuracy['passed']} minimum_cosine={accuracy['minimum_cosine']}", flush=True)
        if not accuracy["passed"]:
            raise SystemExit("pretrained quality gate failed; see accuracy.json")
    results = run_benchmark(args, candidate, crop_size, out_dir)
    summary = save_outputs(args, results, out_dir, environment, accuracy)
    latency = summary["stages"]["end_to_end_ms"]
    print(f"end_to_end_ms median={latency['median_ms']:.3f} mean={latency['mean_ms']:.3f} "
          f"p95={latency['p95_ms']:.3f} valid={summary['validity']['passed']}", flush=True)
    print(f"outputs: {out_dir}", flush=True)
    if args.profile:
        from benchmarks.profiling import profile_callable

        preprocess = make_preprocess(crop_size)
        clip = make_clip(args.batch, args.frames, crop_size, args.seed)

        # Match the measured pipeline while keeping attribution outside all latency samples.
        def profile_once():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            x = preprocess(clip).pin_memory().to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=DTYPES[args.dtype], enabled=args.dtype != "fp32"):
                return candidate(x)

        with torch.cuda.device(args.device), torch.inference_mode():
            profiling = profile_callable(profile_once, out_dir)
        (out_dir / "profile.json").write_text(json.dumps(profiling, indent=2) + "\n", encoding="utf-8")
    if not summary["validity"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
