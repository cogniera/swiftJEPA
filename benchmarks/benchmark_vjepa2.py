# Author : Paarth Sharma
# File Name : benchmark_vjepa2.py
# Project Name : swiftJEPA
# Creation Date : 16th September 2026
# Modification Date : 16th September 2026
# Description : Baseline latency harness for the official V-JEPA 2 encoder.

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import src.hub.backbones as backbones

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VJEPA2_REPO = ROOT / "vjepa2"
sys.path.insert(0, str(VJEPA2_REPO))

PUBLIC_CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2"

MODELS = {
    "vitl": ("vjepa2_vit_large", 256),
    "vith": ("vjepa2_vit_huge", 256),
    "vitg": ("vjepa2_vit_giant", 256),
    "vitg-384": ("vjepa2_vit_giant_384", 384),
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    if torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"

    p = argparse.ArgumentParser(description="V-JEPA 2 inference benchmark")
    p.add_argument("--model", default="vitl", choices=MODELS.keys())
    p.add_argument("--frames", type=int, default=16, help="video frames (multiple of tubelet size 2)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="fp16", choices=DTYPES.keys())
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--pretrained", action="store_true", help="download real weights")
    p.add_argument("--device", default=default_device)
    p.add_argument("--out", default=None)
    return p.parse_args()


def build_encoder(args):
    backbones.VJEPA_BASE_URL = PUBLIC_CKPT_URL

    builder_name, img_size = MODELS[args.model]
    builder = getattr(backbones, builder_name)

    encoder, predictor = builder(pretrained=args.pretrained, num_frames=args.frames)
    del predictor

    encoder.eval()
    encoder.to(args.device)

    return encoder, img_size


class LayerTimer:

    def __init__(self, encoder, device):
        self.device = device

        self.stages = []
        self.stages.append(("patch_embed", encoder.patch_embed))

        for i in range(len(encoder.blocks)):
            name = f"block_{i:02d}"
            self.stages.append((name, encoder.blocks[i]))

        self.stages.append(("norm", encoder.norm))

        self.records = []
        self._current = None
        self._handles = []

        for name, mod in self.stages:
            pre_handle = mod.register_forward_pre_hook(self.pre(name))
            post_handle = mod.register_forward_hook(self.post(name))
            self._handles.append(pre_handle)
            self._handles.append(post_handle)

    def pre(self, name):
        # pytorch only calls a hook with the module and its input, so this makes a new hook that already knows its own stage name.
        def hook(_mod, _inp):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._current[name] = [evt, None]
        return hook

    def post(self, name):
        def hook(_mod, _inp, _out):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._current[name][1] = evt
        return hook

    def start_iter(self):
        self._current = {}

    def end_iter(self):
        self.records.append(self._current)
        self._current = None

    def elapsed_ms(self):
        out = []
        for rec in self.records:
            one_iter = {}
            for name in rec:
                start_evt, end_evt = rec[name]
                one_iter[name] = start_evt.elapsed_time(end_evt)
            out.append(one_iter)
        return out

    def remove(self):
        for handle in self._handles:
            handle.remove()


def make_clip(batch, frames, img_size):
    # this makes fake random video frames so we can time the model without needing real video.
    rng = np.random.default_rng(0)
    shape = (batch, frames, img_size, img_size, 3)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def preprocess(clip_u8):
    # these are the standard imagenet mean and std values the model was trained with.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    x = torch.from_numpy(clip_u8).float()
    x = x.div_(255.0)
    # this reorders the frame from batch,time,height,width,color to batch,color,time,height,width.
    x = x.permute(0, 4, 1, 2, 3)
    x = (x - mean) / std

    return x.contiguous()


def run_benchmark(args, encoder, img_size, out_dir):
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    # this timer only works on a gpu because it uses cuda events to measure time.
    if device.type != "cuda":
        sys.exit("CPU path not wired for CUDA-event timing, run on a CUDA device")

    timer = LayerTimer(encoder, device)
    clip = make_clip(args.batch, args.frames, img_size)

    pre_ms = []
    h2d_ms = []
    total_ms = []

    n_total = args.warmup + args.iters
    # autocast only needs to run when we are not already using full precision.
    autocast_on = dtype != torch.float32

    with torch.inference_mode():
        for i in range(n_total):
            t0 = time.perf_counter()
            x_cpu = preprocess(clip)
            x_cpu = x_cpu.pin_memory()
            t1 = time.perf_counter()

            start = torch.cuda.Event(enable_timing=True)
            after_h2d = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            timer.start_iter()
            start.record()
            x = x_cpu.to(device, non_blocking=True)
            after_h2d.record()

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_on):
                _ = encoder(x)

            end.record()
            # the gpu runs in the background so we wait here until it actually finishes.
            torch.cuda.synchronize()
            timer.end_iter()

            # the first few iterations are just warmup and get thrown away.
            if i >= args.warmup:
                pre_ms.append((t1 - t0) * 1000)
                h2d_ms.append(start.elapsed_time(after_h2d))
                total_ms.append(start.elapsed_time(end))
            else:
                timer.records.clear()

    layer_ms = timer.elapsed_ms()
    timer.remove()

    results = {
        "preprocess_ms": pre_ms,
        "h2d_copy_ms": h2d_ms,
        "gpu_total_ms": total_ms,
        "layers": layer_ms,
    }
    return results


def save_outputs(args, results, out_dir):
    layers = results["layers"]

    # every iteration times the same stages so we just read the names from the first one.
    stage_names = []
    if len(layers) > 0 and len(layers[0]) > 0:
        for name in layers[0]:
            stage_names.append(name)

    rows = []
    for name in stage_names:
        row = []
        for one_iter in layers:
            row.append(one_iter[name])
        rows.append(row)
    mat = np.array(rows)

    with open(out_dir / "timings_per_layer.csv", "w", newline="") as f:
        w = csv.writer(f)

        header = ["stage"]
        for i in range(mat.shape[1]):
            header.append("iter_" + str(i))
        header.append("mean_ms")
        header.append("p50_ms")
        header.append("p95_ms")
        w.writerow(header)

        for name, row in zip(stage_names, mat):
            line = [name]
            for v in row:
                line.append(f"{v:.4f}")
            line.append(f"{row.mean():.4f}")
            # p50 and p95 are the median time and the slowest-case time out of the iterations.
            line.append(f"{np.percentile(row, 50):.4f}")
            line.append(f"{np.percentile(row, 95):.4f}")
            w.writerow(line)

    stage_mean_ms = {}
    for name, row in zip(stage_names, mat):
        stage_mean_ms[name] = float(row.mean())

    summary = {
        # vars turns the parsed command line args into a plain dict so it can be saved as json.
        "config": vars(args),
        "preprocess_ms_mean": float(np.mean(results["preprocess_ms"])),
        "h2d_copy_ms_mean": float(np.mean(results["h2d_copy_ms"])),
        "gpu_total_ms_mean": float(np.mean(results["gpu_total_ms"])),
        "gpu_total_ms_p95": float(np.percentile(results["gpu_total_ms"], 95)),
        "throughput_clips_per_s": float(1000.0 / np.mean(results["gpu_total_ms"]) * args.batch),
        "stage_mean_ms": stage_mean_ms,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def stage_time(item):
    return item[1]


def main():
    args = parse_args()

    # this timestamp makes sure every run gets its own new folder instead of overwriting the last one.
    tag = f"{args.model}_{args.frames}f_b{args.batch}_{args.dtype}_{datetime.now():%Y%m%d_%H%M%S}"

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = HERE / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {args.device}", flush=True)
    if args.device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"building {args.model} (pretrained={args.pretrained})", flush=True)

    encoder, img_size = build_encoder(args)

    n_params = 0
    for p in encoder.parameters():
        n_params += p.numel()
    print(f"encoder params: {n_params/1e6:.1f}M, input {args.frames}x{img_size}x{img_size}")

    results = run_benchmark(args, encoder, img_size, out_dir)
    summary = save_outputs(args, results, out_dir)

    print(f"\ngpu_total_ms mean={summary['gpu_total_ms_mean']:.2f} "
          f"p95={summary['gpu_total_ms_p95']:.2f} "
          f"throughput={summary['throughput_clips_per_s']:.2f} clips/s")

    # this sorts the stages slowest first so we only print the top five.
    stage_items = list(summary["stage_mean_ms"].items())
    stage_items.sort(key=stage_time, reverse=True)
    slowest = stage_items[:5]

    slowest_text = ""
    for name, ms in slowest:
        if slowest_text != "":
            slowest_text += ", "
        slowest_text += f"{name}={ms:.2f}ms"
    print("slowest stages:", slowest_text)

    print(f"\noutputs | {out_dir}")


if __name__ == "__main__":
    main()
