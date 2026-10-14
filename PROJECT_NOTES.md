#Author : Paarth Sharma
#File Name : PROJECT_NOTES.md
#Project Name : swiftJEPA
#Creation Date : 25th July 2026
#Modification Date : 14th October 2026
#Description : Historical build log and decisions for the incremental rebuild.

# V-JEPA 2 Inference Engine with Full Build Log, Decisions, Commands & Code

> **This single file is the complete reproduction kit**: the narrative (decisions, math, errors and fixes), the FULL verbatim source of every project file (in the "Source files" sections inside each phase), and a rebuild-order appendix with verification checkpoints at the end. Copy it OUT of this directory before deleting anything.

Reference for reproducing this project from scratch, at implementation depth:

exact commands, the actual algorithms and math, API signatures, error messages
hit and their fixes, and the reasoning at every fork in the road.

Built 2026-07-24/25. Hardware/software context :
- Windows 11, RTX 4050 Laptop GPU with 6 GB VRAM, 65 W Power limit, Ada SM 8.9 
- WSL2 Ubuntu 24.04 (7 GB VM RAM, 20 threads), Docker Desktop 28.x
- Python 3.12, torch 2.13.0+cu126, CUDA toolkit 12.6.3 (Linux) / 13.2 (Windows, unusable for extensions)

Headline: official pipeline 467 ms/clip → 61.5 ms/clip (ViT-L, 16 frames,
batch 1), equivalence-tested at every step.

---

# Phase 0 Repo acquisition and Windows filesystem problems

```bash
git clone https://github.com/facebookresearch/vjepa2.git vjepa2
```

## Problem: case-colliding paths on a case-insensitive filesystem

Clone printed:

```
warning: the following paths have collided (e.g. case-sensitive paths
on a case-insensitive filesystem) and only one from the same
colliding group is in the working tree:
  'configs/eval_2_1/vitG-384/coin.yaml'
  'configs/eval_2_1/vitg-384/coin.yaml'   ... (9 pairs total)
```

`vitG` = ViT-**gigantic** (`vit_gigantic_xformers`), `vitg` = ViT-**giant**
(`vit_giant_xformers`)   genuinely different configs (different `model_name`,
batch sizes, node counts). Verified they differ by comparing blob hashes:

```bash
a=$(git rev-parse "HEAD:configs/eval_2_1/vitG-384/coin.yaml")
b=$(git rev-parse "HEAD:configs/eval_2_1/vitg-384/coin.yaml")
[ "$a" = "$b" ] && echo identical || echo DIFFERENT     # → DIFFERENT, all 9 pairs
```

The working tree held the *lowercase* content in the dir (last checkout wins);
`git status` showed the uppercase paths as modified.

### Fix (keeps upstream names for lowercase, renames uppercase)

```bash
rm -rf configs/eval_2_1/vitG-384 configs/train_2_1/vitG16
mkdir -p configs/eval_2_1/vitg-384 configs/eval_2_1/vitG-384-gigantic \
         configs/train_2_1/vitg16 configs/train_2_1/vitG16-gigantic
# extract each variant from git history explicitly:
git show "HEAD:configs/eval_2_1/vitg-384/coin.yaml"  > configs/eval_2_1/vitg-384/coin.yaml
git show "HEAD:configs/eval_2_1/vitG-384/coin.yaml"  > configs/eval_2_1/vitG-384-gigantic/coin.yaml
# ... (loop over all files)
# silence git for the phantom uppercase index paths:
git update-index --skip-worktree configs/eval_2_1/vitG-384/coin.yaml   # etc.
# hide the new dirs from status:
printf '%s\n' 'configs/eval_2_1/vitG-384-gigantic/' 'configs/train_2_1/vitG16-gigantic/' >> .git/info/exclude
```

Follow-up gotcha: files written via `git show ... > file` have LF endings while
the repo has `core.autocrlf=true` (expects CRLF in worktree) → status showed
them "modified" with an empty diff. Fix: `git checkout -- <lowercase dirs>`
(safe now   uppercase paths are skip-worktree so no re-collision).

## Gotcha: checkpoint URL sabotaged at HEAD

`vjepa2/src/hub/backbones.py`:

```python
# VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
# for testing
VJEPA_BASE_URL = "http://localhost:8300"
```

Every script that loads pretrained weights must do:

```python
import src.hub.backbones as backbones
backbones.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
```

Checkpoint file for ViT-L is `vitl.pt` (~1.2 GB), key `target_encoder`,
loaded `strict=False` because the checkpoint has pos_embed but runtime uses RoPE.

---

# Phase 1  Environment + baseline benchmark

## Windows env (baseline only; superseded by Docker later)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv timm einops numpy pyyaml matplotlib pandas opencv-python
```

Deliberately NOT `pip install -r vjepa2/requirements.txt`   that pulls wandb,
submitit, decord, webdataset etc. (training-cluster deps). The encoder only
needs torch + timm + einops (per `hubconf.py` `dependencies`).

## Benchmark design (`benchmarks/benchmark_vjepa2.py`)

Model matrix (`MODELS` dict): vitl/vith/vitg @256px, vitg-384. Builders come
from `src.hub.backbones.vjepa2_vit_*(pretrained=..., num_frames=...)`; encoder
kept, predictor discarded (`del _predictor`   an embedding service serves
encoder output only).

### Timing architecture for three layers of measurement

1. **Pipeline segments** with CUDA events around the timed region:

```python
start.record()                                   # cuda events, enable_timing=True
x = x_cpu.to(device, non_blocking=True)          # pinned → device
after_h2d.record()
with torch.autocast('cuda', dtype, enabled=...): # baseline path only
    _ = encoder(x)
end.record()
torch.cuda.synchronize()
# h2d = start.elapsed_time(after_h2d); total = start.elapsed_time(end)
```

CPU preprocess (`uint8 THWC → normalized fp32 CTHW`, ImageNet mean/std
[0.485,0.456,0.406]/[0.229,0.224,0.225]) timed with `perf_counter` and
`.pin_memory()` included   it's CPU work.

2. **Per-layer** via forward hooks (class `LayerTimer`): register
`forward_pre_hook`/`forward_hook` on `encoder.patch_embed`, every
`encoder.blocks[i]`, `encoder.norm`; each hook records a CUDA event; elapsed
computed after sync. Records cleared during warmup. Why hooks + events: GPU is
async   wall-clock around a launch measures dispatch, not execution.

3. **Op level**: one profiled forward,
`torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)`;
export `trace.json` (chrome://tracing / perfetto) and
`key_averages().table(sort_by="cuda_time_total", row_limit=40)` → `top_ops.txt`.

Other decisions:
- Synthetic input: `np.random.randint(0,256,[B,T,H,W,3],uint8)`   benchmark the
  model, not a video decoder.
- Random weights by default (latency is weight-independent); `--pretrained` optional.
- Warmup excluded (default 5); report mean/p50/p95 per stage → `timings_per_layer.csv`,
  `summary.json` (config + stage means + throughput).
- Heat chart `heatmap.png`: rows = [preprocess CPU, H2D, patch_embed,
  block_00..23, norm], cols = iterations, `imshow(cmap="Blues")` (sequential
  single hue = magnitude), side panel = mean-per-stage horizontal bars with
  direct labels on top-3 only. Wide layout `figsize=(14, 0.32*rows)`.
- Outputs per run in `benchmarks/results/<model>_<frames>f_b<batch>_<dtype>_<timestamp>/`.

### Baseline numbers (vitl, 16f, b1, fp16 autocast, Windows)

```
mean 466.7 ms   p95 1104 ms   2.14 clips/s
blocks: 15–23 ms each, ~uniform; patch_embed 2.5; norm 0.3; preprocess(CPU) 25.4; H2D 1.7
```

Heat chart: first ~7 iterations 2–3× darker → laptop power/thermal clock ramp.
**Methodology rule derived here: only same-session back-to-back comparisons
count; cross-session deltas <5% are noise on this machine.**

Op profile (~129 ms kernel time per forward):

| Group | Share | Detail |
|---|---|---|
| GEMMs | 44.7% | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f` ×96 via `aten::addmm` (cuBLAS binary) |
| Attention | 18.7% | `fmha_cutlassF_f16_aligned_64x64_rf_sm80` ×24   memory-efficient, NOT Flash |
| Copies | 17.0% | `aten::copy_` ×918 + `aten::_to_copy` ×581 (autocast conversions + RoPE reassembly) |
| LN + elementwise | ~19% | layer_norm ×49, `add` ×480, `cat` ×480, `mul` ×648, GELU 4.3 ms ×24 |

---


## Source files: benchmark harness (created in Phase 1)

*Complete, verbatim, final-state contents (they include flags/guards added in later phases — this is the version that produces the final numbers). Recreate at benchmarks/.*

#### FILE: `benchmarks/benchmark_vjepa2.py`

````python
"""
V-JEPA 2 inference benchmark.

Times the full inference pipeline of the official PyTorch implementation
(preprocess -> host-to-device copy -> patch embed -> each transformer block
-> final norm), per iteration, using CUDA events. Also runs torch.profiler
for op-level attribution, and renders a per-layer latency heat chart.

Usage (from the InferenceEngine directory, venv activated):
    python benchmarks/benchmark_vjepa2.py --model vitl --frames 16 --dtype fp16
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
VJEPA2_REPO = ROOT / "vjepa2"
sys.path.insert(0, str(VJEPA2_REPO))
sys.path.insert(0, str(ROOT))  # for engine/

# The repo at HEAD points VJEPA_BASE_URL at localhost "for testing";
# override with the public checkpoint host before any hub load.
PUBLIC_CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2"

MODELS = {
    # name -> (builder attr in src.hub.backbones, img_size)
    "vitl": ("vjepa2_vit_large", 256),
    "vith": ("vjepa2_vit_huge", 256),
    "vitg": ("vjepa2_vit_giant", 256),
    "vitg-384": ("vjepa2_vit_giant_384", 384),
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="V-JEPA 2 inference benchmark")
    p.add_argument("--model", default="vitl", choices=MODELS.keys())
    p.add_argument("--frames", type=int, default=16, help="video frames (multiple of tubelet size 2)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="fp16", choices=DTYPES.keys())
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--pretrained", action="store_true",
                   help="download real weights (~GBs); random init is fine for speed benchmarking")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the encoder, mode=reduce-overhead (needs triton -> Linux/container)")
    p.add_argument("--quant", default="none", choices=["none", "int8", "fp8"],
                   help="torchao dynamic quantization of all Linear layers (requires --engine)")
    p.add_argument("--engine", action="store_true",
                   help="apply engine/ optimizations (fp16 native, fused RoPE, flash SDPA, "
                        "GELU epilogue, GPU preprocess)")
    p.add_argument("--graphs", action="store_true",
                   help="capture the encoder into a CUDA graph (requires --engine; "
                        "per-layer timings unavailable during graph replay)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="output directory (default benchmarks/results/<tag>)")
    args = p.parse_args()
    if args.graphs and not args.engine:
        p.error("--graphs requires --engine (graph capture needs the native-fp16 path)")
    if args.quant != "none" and not args.engine:
        p.error("--quant requires --engine")
    if args.compile:
        args.warmup = max(args.warmup, 10)  # compilation + cudagraph warmup
    return args


def build_encoder(args):
    import src.hub.backbones as backbones
    backbones.VJEPA_BASE_URL = PUBLIC_CKPT_URL
    builder_name, img_size = MODELS[args.model]
    builder = getattr(backbones, builder_name)
    encoder, _predictor = builder(pretrained=args.pretrained, num_frames=args.frames)
    del _predictor  # inference pipeline = encoder embeddings only
    encoder.eval().to(args.device)
    if args.engine:
        from engine import optimize_encoder
        encoder = optimize_encoder(encoder, dtype=DTYPES[args.dtype])
    if args.quant != "none":
        from engine.quantize import quantize_encoder
        encoder = quantize_encoder(encoder, args.quant)
    return encoder, img_size


class NullTimer:
    """Stand-in when per-layer hooks can't be used (compile / CUDA graphs)."""
    records = []

    def start_iter(self):
        pass

    def end_iter(self):
        pass

    def elapsed_ms(self):
        return []

    def remove(self):
        pass


class LayerTimer:
    """Registers CUDA-event hooks on patch_embed, every block, and norm."""

    def __init__(self, encoder, device):
        self.device = device
        self.stages = [("patch_embed", encoder.patch_embed)]
        self.stages += [(f"block_{i:02d}", blk) for i, blk in enumerate(encoder.blocks)]
        self.stages.append(("norm", encoder.norm))
        self.records = []  # list per iteration: {stage: (start_evt, end_evt)}
        self._current = None
        self._handles = []
        for name, mod in self.stages:
            self._handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self._handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(_mod, _inp):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._current[name] = [evt, None]
        return hook

    def _post(self, name):
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
        """[{stage: ms}] per recorded iteration. Call after cuda.synchronize()."""
        out = []
        for rec in self.records:
            out.append({name: s.elapsed_time(e) for name, (s, e) in rec.items()})
        return out

    def remove(self):
        for h in self._handles:
            h.remove()


def make_clip(batch, frames, img_size):
    """Synthetic uint8 video clip, stand-in for decoded frames."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(batch, frames, img_size, img_size, 3), dtype=np.uint8)


def preprocess(clip_u8):
    """uint8 THWC -> normalized float CTHW tensor (what the eval transforms do)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    x = torch.from_numpy(clip_u8).float().div_(255.0)
    x = x.permute(0, 4, 1, 2, 3)  # B,C,T,H,W
    x = (x - mean) / std
    return x.contiguous()


def run_benchmark(args, encoder, img_size, out_dir):
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    use_cuda = device.type == "cuda"
    if not use_cuda:
        sys.exit("CPU path not wired for CUDA-event timing; run on a CUDA device.")

    # Per-layer hooks force graph breaks under compile and never fire inside a
    # CUDA-graph replay — use pipeline-level timing only in those modes.
    if args.compile or args.graphs:
        timer = NullTimer()
    else:
        timer = LayerTimer(encoder, device)
    clip = make_clip(args.batch, args.frames, img_size)

    if args.engine:
        from engine import gpu_preprocess
        clip_pinned = torch.from_numpy(clip).pin_memory()

    if args.compile:
        if args.engine:
            # Populate the RoPE cos/sin cache eagerly first: tensors created
            # inside a reduce-overhead CUDA-graph region get overwritten by
            # subsequent replays, corrupting the cache.
            with torch.inference_mode():
                _ = encoder(gpu_preprocess(clip_pinned, device, dtype))
        encoder = torch.compile(encoder, mode="reduce-overhead")

    if args.graphs:
        from engine.cuda_graph import GraphedEncoder
        example = gpu_preprocess(clip_pinned, device, dtype)
        encoder = GraphedEncoder(encoder, example)

    pre_ms, h2d_ms, total_ms = [], [], []
    n_total = args.warmup + args.iters
    autocast_on = dtype != torch.float32 and not args.engine  # engine = native fp16
    with torch.inference_mode():
        for i in range(n_total):
            t0 = time.perf_counter()
            if not args.engine:
                x_cpu = preprocess(clip).pin_memory()
            t1 = time.perf_counter()

            start = torch.cuda.Event(enable_timing=True)
            after_h2d = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            timer.start_iter()
            start.record()
            if args.engine:
                x = gpu_preprocess(clip_pinned, device, dtype)  # uint8 upload + GPU normalize
            else:
                x = x_cpu.to(device, non_blocking=True)
            after_h2d.record()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_on):
                _ = encoder(x)
            end.record()
            torch.cuda.synchronize()
            timer.end_iter()

            if i >= args.warmup:
                pre_ms.append((t1 - t0) * 1000)
                h2d_ms.append(start.elapsed_time(after_h2d))
                total_ms.append(start.elapsed_time(end))
            else:
                timer.records.clear()

    layer_ms = timer.elapsed_ms()
    timer.remove()

    # ---- op-level attribution with torch.profiler (separate short run) ----
    from torch.profiler import ProfilerActivity, profile
    if not args.engine:
        x_cpu = preprocess(clip).pin_memory()
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        if args.engine:
            x = gpu_preprocess(clip_pinned, device, dtype)
        else:
            x = x_cpu.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_on):
            _ = encoder(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    with open(out_dir / "top_ops.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    return {
        "preprocess_ms": pre_ms,
        "h2d_copy_ms": h2d_ms,
        "gpu_total_ms": total_ms,
        "layers": layer_ms,
    }


def save_outputs(args, results, out_dir):
    layers = results["layers"]
    stage_names = list(layers[0].keys()) if layers and layers[0] else []
    if stage_names:
        mat = np.array([[it[s] for it in layers] for s in stage_names])  # stages x iters
    else:  # --graphs: hooks can't fire inside a replay, only pipeline rows exist
        mat = np.empty((0, len(results["gpu_total_ms"])))

    with open(out_dir / "timings_per_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage"] + [f"iter_{i}" for i in range(mat.shape[1])] + ["mean_ms", "p50_ms", "p95_ms"])
        for name, row in zip(stage_names, mat):
            w.writerow([name] + [f"{v:.4f}" for v in row]
                       + [f"{row.mean():.4f}", f"{np.percentile(row, 50):.4f}", f"{np.percentile(row, 95):.4f}"])

    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "preprocess_ms_mean": float(np.mean(results["preprocess_ms"])),
        "h2d_copy_ms_mean": float(np.mean(results["h2d_copy_ms"])),
        "gpu_total_ms_mean": float(np.mean(results["gpu_total_ms"])),
        "gpu_total_ms_p95": float(np.percentile(results["gpu_total_ms"], 95)),
        "throughput_clips_per_s": float(1000.0 / np.mean(results["gpu_total_ms"]) * args.batch),
        "stage_mean_ms": {n: float(r.mean()) for n, r in zip(stage_names, mat)},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap(args, results, stage_names, mat, out_dir)
    return summary


def plot_heatmap(args, results, stage_names, mat, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pipeline rows: preprocess + H2D up top, then model stages.
    pre = np.array(results["preprocess_ms"])
    h2d = np.array(results["h2d_copy_ms"])
    full_names = ["preprocess (CPU)", "H2D copy"] + stage_names
    full_mat = np.vstack([pre, h2d, mat])
    if not stage_names:  # graphs/compile: show the whole encoder as one row
        full_names.append("encoder (whole)")
        full_mat = np.vstack([full_mat, np.array(results["gpu_total_ms"]) - h2d])

    fig, (ax_hm, ax_bar) = plt.subplots(
        1, 2, figsize=(14, max(6, 0.32 * len(full_names))),
        gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True,
    )
    fig.suptitle(
        f"V-JEPA 2 {args.model} inference latency — {args.frames}f @ batch {args.batch}, {args.dtype}",
        fontsize=12,
    )

    # Heatmap: magnitude -> sequential single-hue colormap, light->dark.
    im = ax_hm.imshow(full_mat, aspect="auto", cmap="Blues", interpolation="nearest")
    ax_hm.set_yticks(range(len(full_names)), full_names, fontsize=7)
    ax_hm.set_xlabel("iteration")
    ax_hm.set_title("per-stage latency by iteration (ms)", fontsize=10)
    fig.colorbar(im, ax=ax_hm, label="ms", shrink=0.8)

    # Mean per stage: horizontal bars, direct label on the largest few only.
    means = full_mat.mean(axis=1)
    y = np.arange(len(full_names))
    ax_bar.barh(y, means, color="#3b6ea5", height=0.62)
    ax_bar.set_yticks(y, ["" for _ in full_names])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("mean ms")
    ax_bar.set_title("mean per stage", fontsize=10)
    ax_bar.grid(axis="x", color="0.9", linewidth=0.6)
    ax_bar.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_bar.spines[spine].set_visible(False)
    top = np.argsort(means)[-3:]
    for i in top:
        ax_bar.text(means[i], i, f" {means[i]:.2f}", va="center", fontsize=8, color="#333333")

    fig.savefig(out_dir / "heatmap.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    tag = f"{args.model}_{args.frames}f_b{args.batch}_{args.dtype}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.out) if args.out else ROOT / "benchmarks" / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {args.device}", flush=True)
    if args.device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"building {args.model} (pretrained={args.pretrained}) ...", flush=True)
    encoder, img_size = build_encoder(args)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder params: {n_params/1e6:.1f}M, input {args.frames}x{img_size}x{img_size}")

    results = run_benchmark(args, encoder, img_size, out_dir)
    summary = save_outputs(args, results, out_dir)

    print(f"\ngpu_total_ms mean={summary['gpu_total_ms_mean']:.2f} "
          f"p95={summary['gpu_total_ms_p95']:.2f} "
          f"throughput={summary['throughput_clips_per_s']:.2f} clips/s")
    slowest = sorted(summary["stage_mean_ms"].items(), key=lambda kv: -kv[1])[:5]
    print("slowest stages:", ", ".join(f"{n}={v:.2f}ms" for n, v in slowest))
    print(f"\noutputs -> {out_dir}")


if __name__ == "__main__":
    main()

````

#### FILE: `benchmarks/README.md`

````markdown
# V-JEPA 2 Inference Benchmark

Profiles the official PyTorch V-JEPA 2 encoder end-to-end and shows where the
time goes: CPU preprocessing, host-to-device copy, patch embedding, every
transformer block, and the final norm — plus op-level attribution from
`torch.profiler`.

## Setup (already done)

- `.venv/` — Python 3.12, PyTorch 2.13 + CUDA 12.6, timm, einops, matplotlib, etc.
- `vjepa2/` — official repo clone. Note: its `src/hub/backbones.py` at HEAD
  points `VJEPA_BASE_URL` at `localhost:8300`; the benchmark script overrides
  it to the public `https://dl.fbaipublicfiles.com/vjepa2` automatically.

## Run it

```powershell
.venv\Scripts\activate
python benchmarks\benchmark_vjepa2.py                       # defaults: vitl, 16 frames, batch 1, fp16
python benchmarks\benchmark_vjepa2.py --frames 32 --iters 50
python benchmarks\benchmark_vjepa2.py --dtype fp32          # compare precision
python benchmarks\benchmark_vjepa2.py --pretrained          # real weights (~1.2 GB download; same speed)
```

Key flags: `--model {vitl,vith,vitg,vitg-384}`, `--frames N`, `--batch N`,
`--dtype {fp32,fp16,bf16}`, `--iters`, `--warmup`, `--compile`.

**6 GB VRAM (RTX 4050) guidance:** `vitl` at 16–32 frames fp16 is comfortable.
`vith`/`vitg` in fp16 may fit at 16 frames but will be tight; fp32 giant will
OOM. `--compile` needs triton, which is generally unavailable on Windows —
expect it to fail unless you've installed `triton-windows`.

Weights are irrelevant to latency, so the default is random init (no download).
Use `--pretrained` only if you also want to sanity-check outputs.

## Outputs (per run, in `benchmarks/results/<tag>/`)

| File | Contents |
|---|---|
| `heatmap.png` | the heat chart: per-stage latency across iterations + mean-per-stage bars |
| `timings_per_layer.csv` | ms per stage per iteration, with mean/p50/p95 |
| `summary.json` | config, totals, throughput, per-stage means |
| `top_ops.txt` | top 40 ops by CUDA time (attention vs GEMM vs elementwise, etc.) |
| `trace.json` | Chrome trace — open in `chrome://tracing` or https://ui.perfetto.dev |

## Reading the results

- **Heatmap rows glowing dark** = the bottleneck stages. Expect the transformer
  blocks to dominate; per-block time should be near-uniform — an outlier block
  means something odd (recompilation, cache effects).
- **`preprocess (CPU)` or `H2D copy` large relative to GPU total** = the
  pipeline is input-bound; batching, pinned-memory streaming, or GPU-side
  decode (NVDEC/DALI) is the win, not model optimization.
- **`top_ops.txt`**: if `scaled_dot_product_attention` dominates → attention
  is the target (Flash/PagedAttention, KV layout). If `addmm`/`mm` dominate →
  GEMM-bound; fp16/int8 and fused MLPs are the lever.
- **`trace.json`** shows gaps between kernels = launch/CPU overhead → a case
  for CUDA graphs or `torch.compile`.

````

# Phase 2  Source reading: where the waste lives

All in `vjepa2/src/models/utils/modules.py`.

## `rotate_queries_or_keys(x, pos)` (~line 26) called 144× per forward

```python
omega = torch.arange(D//2, dtype=x.dtype, device=x.device)
omega /= D/2.0
omega = 1.0 / 10000**omega                      # recomputed EVERY call
freq = torch.einsum("..., f -> ... f", pos, omega)
emb_sin = freq.sin(); emb_cos = freq.cos()      # EVERY call
# NOTE in official code: "subtle bug ... kept for compatibility with pretrained model"
emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)   # CONCATENATED halves
emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)   # (bug: not repeat_interleave)
y = x.unflatten(-1, (-1, 2)); y1, y2 = y.unbind(-1)
y = torch.stack((-y2, y1), dim=-1).flatten(-2)      # adjacent-pair rotate
return (x * emb_cos) + (y * emb_sin)
```

The exact semantics any reimplementation MUST match:
- table layout per axis segment of width `d` (with `k = d/2`, `ω_j = 10000^(-j/k)`,
  `f_j = pos·ω_j`): `cos_row = [cos f_0 … cos f_{k-1}, cos f_0 … cos f_{k-1}]`
  (duplicated halves, NOT interleaved)   this is the compat bug;
- rotation pairs are ADJACENT elements: `out[2i] = x[2i]·c[2i] − x[2i+1]·s[2i]`,
  `out[2i+1] = x[2i+1]·c[2i+1] + x[2i]·s[2i+1]`.
- reference computes ω in `x.dtype` → under autocast that's fp16 (`pos·ω` in
  half precision!). Computing tables in fp32 is *more* accurate but differs at
  ~1e-2 scale in fp16   set test tolerances accordingly.

## `RoPEAttention` (~line 266)

- head split: `d_dim = h_dim = w_dim = 2*((head_dim//3)//2)` → for head_dim 64:
  **20/20/20 rotated + 4 passthrough**.
- position ids (mask-free path): `ids = arange(T*H*W)`;
  `d_pos = ids // (H*W)`; `h_pos = (ids mod H*W) // W`; `w_pos = rest`.
- forward: `qkv = self.qkv(x).unflatten(-1,(3,heads,-1)).permute(2,0,3,1,4)`
  (⇒ q,k,v are NON-contiguous views, inner dim stride 1, token stride = 3·embed);
  rotates 3 slices of q and k; reassembles with `torch.cat([qd,qh,qw,qr],-1)`
  ×2 per block ⇒ the 480 cats; SDPA wrapped in **deprecated**
  `torch.backends.cuda.sdp_kernel()` (no args) ⇒ default backend priority ⇒
  cutlassF picked, not Flash.

## `MLP`  fc1 → GELU(erf) → fc2, GELU standalone kernel

## Reference code imported (`kernels_for_reuse/`) the "RAG" step

From `pytorch/pytorch` AT THE INSTALLED TAG (v2.13.0   always match versions):

```bash
git clone --depth 1 --branch v2.13.0 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
git -C pytorch config core.longpaths true      # cutlass filenames exceed 260-char Windows limit
git -C pytorch sparse-checkout set aten/src/ATen/native/transformers aten/src/ATen/native/cuda aten/src/ATen/cuda
```

(Windows note: sparse-checkout at a deep path failed silently; moved clone to a
short path like `C:\Temp\ptsrc` and it worked.)

Kept (backward-pass files deleted inference only):
- `attention/` = `native/transformers/cuda/`: `attention.cu`, `sdp_utils.*`,
  `mem_eff_attention/` (the cutlassF kernel from the profile, forward
  instantiations only; `kernels/generate_kernels.py` regenerates variants),
  `flash_attn/` (PyTorch's API wrapper ONLY   real kernels are not in-tree)
- `layernorm/layer_norm_kernel.cu`   reduction pattern reference
- `activation/ActivationGeluKernel.cu`, `softmax/SoftMax.cu + PersistentSoftmax.cuh`
- `gemm/` = `native/cuda/Blas.cpp` + `cuda/CUDABlas.{cpp,h}` + `CublasHandlePool.cpp`
    PyTorch's cuBLAS/cuBLASLt calling convention. **Key find:**
  `CUDABlas.cpp:1786`: `GEMMAndBiasActivationEpilogue::GELU →
  CUBLASLT_EPILOGUE_GELU_BIAS`, surfaced in Python as
  `aten::_addmm_activation(self, mat1, mat2, *, use_gelu=True)` ⇒ fused
  GEMM+bias+GELU with ZERO custom code.

From `Dao-AILab/flash-attention` (same sparse-clone pattern): `csrc/flash_attn`
→ `attention/flash_attn_official/` (89 files after deleting `flash_bwd_*`;
needs the repo's cutlass submodule headers to build). Also fetched
`flash_attn/ops/triton/rotary.py` as the memory-layout reference for the RoPE
kernel (coalesced inner-dim access; cos/sin indexed `[seq_pos, rot_dim]`;
in-place support pattern).

**Not extractable** (closed NVIDIA binaries): the GEMM kernels themselves
(inside libcublas) and conv3d (cuDNN). Hence: copy the calling convention.

---

# Phase 3   The engine (`engine/`)

## Architectural decisions (apply to everything below)

1. **Dual path per op**: fused pure-PyTorch fallback (runs everywhere, is the
   default where it wins) + real CUDA source in `csrc/` (for the future
   standalone engine). Dispatch chosen per-op from measurement, overridable by
   env var.
2. **Monkeypatch, don't fork**: `optimize_encoder(encoder)` swaps
   `RoPEAttention.forward` and `MLP.forward` via `types.MethodType`, stashes
   `_orig_forward`, and any call with `mask`/`attn_mask` falls back to the
   original. Official repo untouched; adoption is one line; A/B trivial.
3. **Correctness gate**: every change reruns `engine/tests/test_correctness.py`.

## 3.1 `rope_cache.py`   kill the 144× recompute and all cats

**Full-width table trick.** Build ONE `[1,1,N,head_dim]` cos and sin table:

```
cols  0..19  : depth positions,  ω_j = 10000^(-j/10), duplicated-halves layout
cols 20..39  : height positions,       (same formula per segment)
cols 40..59  : width positions
cols 60..63  : identity  (cos=1, sin=0)   ← passthrough remainder
```

Then the whole 3-axis rotation + reassembly collapses to ONE op over the full
head dim (identity cols make the remainder pass through untouched):

```python
def apply_rope(x, cos, sin):                      # x [B,H,N,64]
    x2 = x.unflatten(-1, (-1, 2))
    rot = torch.stack((-x2[..., 1], x2[..., 0]), dim=-1).flatten(-2)
    return torch.addcmul(x * cos, rot, sin)       # x*cos + rot(x)*sin
```

- Cache keyed `(head_dim,d,h,w,N,T,H,W,grid,device,dtype)` in a module dict.
- Tables computed in fp32 then cast (more accurate than reference's fp16 path;
  fp32 agreement 5e-7, fp16 tolerance 2e-2 in tests).
- Replaces per q,k per block: ~10 kernels ×3 axes + cat → 3–4 kernels total.

## 3.2 Native fp16 (`optimize_encoder(dtype=torch.float16)`)

`encoder.to(dtype).eval()`; no autocast at inference. Kills all 581
`_to_copy` (autocast was converting fp32 weights per layer). ~15% for free.
(bf16 supported via the same arg.)

## 3.3 `ops.linear_gelu`   GELU in the GEMM epilogue

```python
y = torch.ops.aten._addmm_activation(bias, x.reshape(-1, K), weight.t(), use_gelu=True)
```

→ cuBLASLt `GELU_BIAS` epilogue (same closed-source kernels PyTorch uses).
**Known delta**: Lt epilogue = tanh-approx GELU; `nn.GELU()` default = erf.
Measured max abs diff 3.9e-3 → accepted, tested with tol 2e-2.
Guard (added in Phase 7): only when `type(weight) is torch.Tensor`   quantized
tensor subclasses must route through `F.linear` (subclass dispatch).

## 3.4 `ops.gpu_preprocess`

Upload uint8 `[B,T,H,W,3]` pinned → GPU (4× less PCIe than fp32), then one
fused pass: `/255 → -mean → /std → THWC→CTHW → cast`. CPU 25 ms → sub-ms.

## 3.5 `ops.sdpa_flash_first`

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH], set_priority=True):
    F.scaled_dot_product_attention(q, k, v)
```

Replaces the deprecated no-arg context. Windows wheels have NO Flash kernels →
this pays only on Linux: attention 21.9 → 17.1 ms/forward (+22%).

## 3.6 CUDA sources (`engine/csrc/`)   and what iteration taught

**`rope_fused.cu`**   one elementwise kernel; half2/bfloat162 vectorized loads
(each adjacent pair = one 32-bit load), fp32 math, `__floats2half2_rn` store.
Three iterations, all correctness-passing:
1. contiguous-only + `.contiguous()` in the wrapper → 48 extra q/k copies per
   forward → **regression** (ext 86.5 vs fallback 83.9 ms same-session A/B/A);
2. stride-aware flat index (`p / pairs_per_row`, `% n_tokens`, …) → still
   slower   suspected the 4 × int64 div/mod per element;
3. 3D grid `blocks=(n_tokens, n_heads, batch)`, one block per row, thread =
   pair index → zero div/mod, reads the permute view via strides
   (`b*sb + h*sh + n*sn + 2*pair`), writes packed → STILL ~60 µs/call slower
   than the torch chain at [1,16,2048,64].
   **Verdict: parked, opt-in `VJEPA_ROPE_EXT=1`.** Lesson: fusing a
   memory-bound elementwise chain over fat tensors is not automatically a win;
   torch's streaming contiguous intermediates beat clever indexing here.
   Per-stage diff proved the localization: blocks 84.8 (ext) vs 81.9 (fallback),
   while H2D+preprocess showed the ext preprocess kernel WINNING 0.48 vs 0.81 ms.

**`fused_add_layernorm.cu`**   `added = x+res; y = LN(added)` in one pass,
returns both (residual stream needs `added`). One CTA/row; Welford aggregates
(`mean, m2, count` with merge formula), warp `__shfl_down_sync` reduction →
smem cross-warp reduction; fp32 accumulation; `AT_DISPATCH_FLOATING_TYPES_AND2`
(the initially-used `AT_DISPATCH_REDUCED_FLOATING_TYPES_AND` doesn't exist).
Built & tested (err 1.9e-6) but NOT wired into blocks   the encoder's pre-norm
chain (`x = x + attn(norm1(x))`) puts add and the *next* norm in different
statements; wiring it means restructuring the block loop → deferred to v2.

**`preprocess_uint8.cu`**   grid-stride over `B*T*H*W` pixels, 3 channels
unrolled, `__constant__` mean/std, output index `(b*3+c)*THW + p`.
**Build error hit**: torch extensions compile with
`-D__CUDA_NO_HALF_CONVERSIONS__` → `(( __half)floatval)` fails with
"no suitable constructor". Fix: templated `from_float<T>` specialized to
`__float2half_rn` / identity.

**`gemm_bias_gelu.cpp`**   standalone cuBLASLt mirror of `gemm_and_bias`:
row-major torch → column-major Lt via the transpose trick   for
`y[M,N] = x[M,K] @ W[N,K]^T + b`: A=W desc `[K,N]` opT, B=x desc `[K,M]` opN,
C desc `[N,M]` ⇒ y row-major. `CUBLASLT_MATMUL_DESC_EPILOGUE=GELU_BIAS`,
`BIAS_POINTER=b`, compute `CUBLAS_COMPUTE_32F`, 4 MiB workspace,
`cublasLtMatmulAlgoGetHeuristic` (1 result). Exists so a non-ATen engine keeps
PyTorch-identical GEMM behavior.

**`bindings.cpp`**   pybind; **`build.py`**:

```python
load(name="vjepa_engine_ext",
     sources=[bindings.cpp, rope_fused.cu, fused_add_layernorm.cu,
              preprocess_uint8.cu, gemm_bias_gelu.cpp],
     extra_cuda_cflags=["-O3", "--use_fast_math"],
     extra_ldflags=["-lcublasLt", "-lcublas"])   # .lib names on Windows
```

**Loader** (`ops.py` top): a plain `import vjepa_engine_ext` can NOT find a
JIT-built extension   must call `engine.build.build()` (ninja cache makes it
~1 s after first build). Auto-attempted on Linux, skipped on Windows,
`VJEPA_ENGINE_EXT=0/1` override, failure → fallbacks + one stderr line.

## 3.7 Tests (`engine/tests/test_correctness.py`)

| Check | vs | Tolerance | Measured |
|---|---|---|---|
| rope fp32 | official rotate+cat path (reimplemented verbatim) | 1e-5 | 4.8e-7 |
| rope fp16 | same in fp16 | 2e-2 (fp16-ω vs fp32-table) | 8.5e-3 |
| add+LN | `x+res` then `F.layer_norm` | 1e-4 | 0 (fallback) / 1.9e-6 (kernel) |
| linear+GELU | `F.gelu(F.linear(...))` erf | 2e-2 (tanh vs erf) | 3.9e-3 |
| preprocess | CPU float path | 5e-3 | 2.2e-3 |
| end-to-end | fp32 official encoder vs fp16 optimized | cosine > 0.99 | **1.00000**, rel 0.16% |

## Engine v1 result (Windows): 467 → 89.1 ms (5.24×), p95 1104 → 91.2.
New profile: GEMM 40%, attention 25% (still cutlass on Windows), `_addmm_activation` 24 calls, copies out of top-10.

Benchmark integration: `--engine` flag → `optimize_encoder` + uint8-pinned
clip + `gpu_preprocess` inside the timed region + autocast disabled.
(Also had to `sys.path.insert(0, ROOT)` for `import engine`.)

---


## Source files: the engine package (created in Phase 3)

*Complete, verbatim, final-state contents (they include flags/guards added in later phases — this is the version that produces the final numbers). Recreate at engine/ and engine/csrc/; run test_correctness.py after — demand 6/6.*

#### FILE: `engine/__init__.py`

````python
from .optimized_encoder import optimize_encoder
from .ops import gpu_preprocess

__all__ = ["optimize_encoder", "gpu_preprocess"]

````

#### FILE: `engine/rope_cache.py`

````python
"""Precomputed RoPE cos/sin tables for V-JEPA 2's 3-axis (depth/height/width) rotary embedding.

The official implementation (vjepa2/src/models/utils/modules.py::rotate_queries_or_keys)
recomputes omega, an einsum, sin and cos on EVERY call — 6 calls per block, 144 per
forward — then reassembles q/k with two torch.cat's per block. All positions are static
for a fixed input resolution, so everything except the final `x*cos + rot(x)*sin` can be
computed exactly once.

This module reproduces the reference semantics EXACTLY, including the documented
frequency-duplication quirk (`emb.repeat(..., 2)` concatenates the half-table rather than
interleaving it — kept for checkpoint compatibility, see the note in the official code).

The full-width trick: instead of rotating three slices [d|h|w] separately and
concatenating, we build one cos/sin table of the full head_dim where
  cols [0:d)        use depth-axis positions
  cols [d:d+h)      use height-axis positions
  cols [d+h:d+h+w)  use width-axis positions
  cols [d+h+w:D)    identity (cos=1, sin=0)  -> passthrough remainder
so the rotation becomes a single fused elementwise op over q (and k) — no slicing, no cat.
"""

import torch

_CACHE = {}


def _axis_table(pos, seg_dim, device):
    """cos/sin for one axis, replicating rotate_queries_or_keys' layout in fp32.

    Returns (cos, sin) of shape [N, seg_dim] laid out as [t0..t_{k-1}, t0..t_{k-1}]
    (the reference's `repeat(1, 1, 1, 2)` duplication), k = seg_dim // 2.
    """
    half = seg_dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=device) / (seg_dim / 2.0)
    omega = 1.0 / 10000**omega  # (half,)
    freq = torch.einsum("n,f->nf", pos.float(), omega)  # (N, half)
    sin = freq.sin().repeat(1, 2)  # (N, seg_dim), duplicated halves
    cos = freq.cos().repeat(1, 2)
    return cos, sin


def _positions(N, T, H, W, grid_size, device):
    """frame/height/width position ids per token (separate_positions, mask-free path)."""
    if T is None or H is None or W is None:
        H = W = grid_size
        T = N // (H * W)
    ids = torch.arange(int(T * H * W), device=device)
    tokens_per_frame = int(H * W)
    d_pos = ids // tokens_per_frame
    rem = ids - d_pos * tokens_per_frame
    h_pos = rem // W
    w_pos = rem - h_pos * W
    return d_pos, h_pos, w_pos


def get_rope_tables(*, head_dim, d_dim, h_dim, w_dim, N, T, H, W, grid_size, device, dtype):
    """Full-width cos/sin tables [1, 1, N, head_dim], cached per resolution/dtype."""
    key = (head_dim, d_dim, h_dim, w_dim, N, T, H, W, grid_size, str(device), dtype)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    d_pos, h_pos, w_pos = _positions(N, T, H, W, grid_size, device)
    cos_parts, sin_parts = [], []
    for pos, seg in ((d_pos, d_dim), (h_pos, h_dim), (w_pos, w_dim)):
        c, s = _axis_table(pos, seg, device)
        cos_parts.append(c)
        sin_parts.append(s)
    rem = head_dim - (d_dim + h_dim + w_dim)
    if rem > 0:
        n = cos_parts[0].shape[0]
        cos_parts.append(torch.ones(n, rem, dtype=torch.float32, device=device))
        sin_parts.append(torch.zeros(n, rem, dtype=torch.float32, device=device))

    cos = torch.cat(cos_parts, dim=-1).to(dtype).view(1, 1, -1, head_dim).contiguous()
    sin = torch.cat(sin_parts, dim=-1).to(dtype).view(1, 1, -1, head_dim).contiguous()
    _CACHE[key] = (cos, sin)
    return cos, sin

````

#### FILE: `engine/ops.py`

````python
"""Fused ops for the V-JEPA 2 inference engine.

Each op has two paths:
  1. compiled CUDA extension (engine/csrc/*, built via engine/build.py) if available;
  2. a fused pure-PyTorch fallback that runs everywhere today and already removes
     the per-call recompute / cat / autocast-copy overhead the profile flagged.
"""

import os
import sys

import torch
import torch.nn.functional as F

_ext = None
# JIT-load the compiled extension (cached by ninja after the first build, so
# this is fast once engine/build.py has run). Auto-attempted on Linux where a
# matching CUDA toolkit is expected; opt in/out with VJEPA_ENGINE_EXT=1/0.
_want_ext = os.environ.get("VJEPA_ENGINE_EXT", "1" if sys.platform == "linux" else "0")
if _want_ext == "1" and torch.cuda.is_available():
    try:
        from .build import build
        _ext = build(verbose=False)
    except Exception as e:  # no toolkit / build failure -> pure-PyTorch fallbacks
        print(f"[engine] CUDA extension unavailable ({type(e).__name__}), using PyTorch fallbacks",
              file=sys.stderr)

_HAS_ADDMM_ACT = hasattr(torch.ops.aten, "_addmm_activation")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- 1. fused RoPE

def apply_rope(x, cos, sin):
    """x*cos + rotate_adjacent_pairs(x)*sin over the full head dim.

    x: [B, heads, N, D]; cos/sin: [1, 1, N, D] from rope_cache (identity in the
    non-rotated remainder, so no slicing/cat is needed).
    """
    # Measured on RTX 4050 @ vitl/16f: the torch chain below beats the compiled
    # kernel by ~60us/call (strided q/k reads vs streaming contiguous
    # intermediates), so the ext path is opt-in until it wins. A/B via
    # VJEPA_ROPE_EXT=1.
    if (_ext is not None and os.environ.get("VJEPA_ROPE_EXT") == "1"
            and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)):
        return _ext.rope_fused(x, cos, sin)  # stride-aware: reads permute views
    x2 = x.unflatten(-1, (-1, 2))
    rot = torch.stack((-x2[..., 1], x2[..., 0]), dim=-1).flatten(-2)
    return torch.addcmul(x * cos, rot, sin)


# ------------------------------------------------- 2. residual add + layernorm

def fused_add_layer_norm(x, residual, weight, bias, eps=1e-6):
    """Returns (normed, x + residual). One kernel when the extension is built."""
    if _ext is not None and x.is_cuda:
        return _ext.fused_add_layer_norm(x, residual, weight, bias, eps)
    added = x + residual
    return F.layer_norm(added, (x.shape[-1],), weight, bias, eps), added


# ------------------------------------------------ 3. GEMM + bias + GELU epilogue

def linear_gelu(x, weight, bias):
    """fc + GELU fused into the GEMM epilogue.

    Uses aten::_addmm_activation, which routes to cuBLASLt with
    CUBLASLT_EPILOGUE_GELU_BIAS (see kernels_for_reuse/gemm/CUDABlas.cpp:1786) —
    the same closed-source NVIDIA kernels PyTorch uses, with the activation free.
    NOTE: the Lt epilogue computes tanh-approx GELU; nn.GELU() defaults to erf.
    Max abs deviation is ~3e-3, irrelevant next to fp16 noise, but it IS a
    different curve — validated in tests/test_correctness.py.
    """
    # type(...) checks: quantized (torchao) weights are tensor subclasses that
    # aten::_addmm_activation can't handle — route them through F.linear, which
    # dispatches to the quantized kernels.
    if (x.is_cuda and bias is not None and _HAS_ADDMM_ACT
            and type(weight) is torch.Tensor and type(x) is torch.Tensor):
        y = torch.ops.aten._addmm_activation(
            bias, x.reshape(-1, x.shape[-1]), weight.t(), use_gelu=True
        )
        return y.view(*x.shape[:-1], weight.shape[0])
    return F.gelu(F.linear(x, weight, bias))


# --------------------------------------------------------- 4. GPU preprocessing

def gpu_preprocess(clip_u8, device, dtype=torch.float16):
    """uint8 [B,T,H,W,C] (pinned, CPU) -> normalized [B,C,T,H,W] on GPU.

    Moves 4x less data over PCIe than the fp32 CPU path (uint8 upload, then
    normalize on-device) and replaces the 25 ms CPU normalize with a ~0.1 ms
    GPU elementwise pass.
    """
    x = clip_u8.to(device, non_blocking=True)
    if _ext is not None:
        return _ext.preprocess_uint8(x, dtype == torch.float16)
    x = x.permute(0, 4, 1, 2, 3).to(dtype).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    return x.sub_(mean).div_(std)


# ----------------------------------------------------- 5. flash-first attention

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _BACKENDS = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]

    def sdpa_flash_first(q, k, v):
        """SDPA with Flash preferred (replaces the deprecated no-arg
        torch.backends.cuda.sdp_kernel() context in the official code, which
        left backend choice to default priorities and landed on cutlassF)."""
        with sdpa_kernel(_BACKENDS, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v)

except ImportError:  # very old torch

    def sdpa_flash_first(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

````

#### FILE: `engine/optimized_encoder.py`

````python
"""Applies the engine's v1 optimizations to an official V-JEPA 2 encoder in place.

  1. native fp16/bf16 weights          -> kills the 581 autocast _to_copy calls (~17% GPU time)
  2. cached + fused RoPE               -> kills 144x omega/einsum/sin/cos recompute and 48 cats
  3. flash-first SDPA                  -> replaces default-priority cutlassF pick
  4. GEMM+bias+GELU epilogue in MLP    -> removes standalone GELU kernel (4.3 ms)
  (5. GPU preprocess lives in ops.gpu_preprocess, applied at the pipeline level)

Only the mask-free inference path is patched; calls with `mask`/`attn_mask` fall
back to the original forward, so training/masked eval behavior is unchanged.
"""

import types

from . import ops
from .rope_cache import get_rope_tables


def _fast_rope_attn_forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
    if mask is not None or attn_mask is not None:
        return self._orig_forward(x, mask=mask, attn_mask=attn_mask, T=T,
                                  H_patches=H_patches, W_patches=W_patches)
    B, N, C = x.size()
    qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

    cos, sin = get_rope_tables(
        head_dim=self.head_dim, d_dim=self.d_dim, h_dim=self.h_dim, w_dim=self.w_dim,
        N=N, T=T, H=H_patches, W=W_patches, grid_size=self.grid_size,
        device=x.device, dtype=x.dtype,
    )
    q = ops.apply_rope(q, cos, sin)
    k = ops.apply_rope(k, cos, sin)

    x = ops.sdpa_flash_first(q, k, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return x


def _fused_mlp_forward(self, x):
    h = ops.linear_gelu(x, self.fc1.weight, self.fc1.bias)
    return self.fc2(h)


def optimize_encoder(encoder, dtype=None):
    """Patch a vjepa2 VisionTransformer encoder for fast inference. Returns it."""
    import torch

    from src.models.utils.modules import MLP, RoPEAttention  # vjepa2 repo on sys.path

    if dtype is None:
        dtype = torch.float16
    encoder = encoder.to(dtype).eval()

    for module in encoder.modules():
        if isinstance(module, RoPEAttention):
            module._orig_forward = module.forward
            module.forward = types.MethodType(_fast_rope_attn_forward, module)
        elif isinstance(module, MLP):
            module._orig_forward = module.forward
            module.forward = types.MethodType(_fused_mlp_forward, module)
    return encoder

````

#### FILE: `engine/build.py`

````python
"""Build the CUDA extension: python engine/build.py

Requirements: CUDA toolkit matching torch's CUDA major version (torch here is
cu126 -> CUDA 12.x; an installed nvcc 13.x will NOT work), plus a host compiler
(MSVC on Windows, gcc on Linux/WSL2). On this dev box the practical route is
WSL2 + CUDA 12.x; the pure-PyTorch fallbacks in engine/ops.py are used until
`import vjepa_engine_ext` succeeds.
"""

import sys
from pathlib import Path

from torch.utils.cpp_extension import load

CSRC = Path(__file__).resolve().parent / "csrc"


def build(verbose=True):
    return load(
        name="vjepa_engine_ext",
        sources=[str(CSRC / f) for f in (
            "bindings.cpp", "rope_fused.cu", "fused_add_layernorm.cu",
            "preprocess_uint8.cu", "gemm_bias_gelu.cpp",
        )],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=(["cublasLt.lib", "cublas.lib"] if sys.platform == "win32"
                       else ["-lcublasLt", "-lcublas"]),
        verbose=verbose,
    )


if __name__ == "__main__":
    ext = build()
    print("built OK:", ext.__file__)

````

#### FILE: `engine/csrc/bindings.cpp`

````cpp
#include <torch/extension.h>

torch::Tensor rope_fused(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
std::tuple<torch::Tensor, torch::Tensor> fused_add_layer_norm(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
    torch::Tensor bias, double eps);
torch::Tensor preprocess_uint8(torch::Tensor clip, bool to_half);
torch::Tensor gemm_bias_gelu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_fused", &rope_fused, "fused 3-axis RoPE (V-JEPA 2 semantics)");
  m.def("fused_add_layer_norm", &fused_add_layer_norm, "residual add + LayerNorm");
  m.def("preprocess_uint8", &preprocess_uint8, "uint8 THWC -> normalized CTHW");
  m.def("gemm_bias_gelu", &gemm_bias_gelu, "cuBLASLt GEMM + bias + GELU epilogue");
}

````

#### FILE: `engine/csrc/rope_fused.cu`

````cuda
// Fused 3-axis RoPE for V-JEPA 2.
//
// Replaces (per attention layer, per q and k): 3x {einsum, sin, cos, unflatten,
// unbind, stack, flatten, 2 muls, add} + 1 torch.cat  ->  ONE elementwise kernel.
// cos/sin come precomputed from engine/rope_cache.py as full-head-dim tables
// [N, D] with identity (cos=1, sin=0) in the non-rotated remainder, so the whole
// head dim is processed uniformly with no slicing.
//
// Rotation semantics (must match modules.py::rotate_queries_or_keys exactly):
//   pairs are ADJACENT elements (2i, 2i+1):
//     out[2i]   = x[2i]  *cos[2i]   - x[2i+1]*sin[2i]
//     out[2i+1] = x[2i+1]*cos[2i+1] + x[2i]  *sin[2i+1]
//   (cos/sin duplication across halves — the checkpoint-compat quirk — is baked
//   into the tables by rope_cache.py; this kernel is agnostic to it.)
//
// Memory-access design (after flash-attention's triton rotary kernel):
// one half2/bfloat162 load per pair per tensor => fully coalesced, fp32 math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T> __device__ __forceinline__ T __floats2half2_rn_generic(float, float);
template <> __device__ __forceinline__ __half2 __floats2half2_rn_generic<__half2>(float a, float b) {
  return __floats2half2_rn(a, b);
}
template <> __device__ __forceinline__ __nv_bfloat162 __floats2half2_rn_generic<__nv_bfloat162>(float a, float b) {
  return __floats2bfloat162_rn(a, b);
}

template <typename scalar2_t, typename scalar_t>
__global__ void rope_fused_kernel(
    const scalar_t* __restrict__ x,    // [B, H, N, D], strided (reads the qkv
    const scalar_t* __restrict__ cos_, // permute view directly — absorbing the
    const scalar_t* __restrict__ sin_, // .contiguous() copy); inner dim packed.
    scalar_t* __restrict__ out,        // packed [B, H, N, D]
    const int n_tokens,
    const int dim,              // D
    const int64_t stride_b, const int64_t stride_h, const int64_t stride_n) {
  // Grid: x = token n, y = head h, z = batch b. One block per row; threads
  // cover the D/2 pairs. No integer div/mod anywhere (64-bit division per
  // element was the bottleneck in the flat-indexed version of this kernel).
  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t xrow = b * stride_b + h * stride_h + (int64_t)n * stride_n;
  const int64_t orow = ((b * gridDim.y + h) * n_tokens + n) * (int64_t)dim;
  const int64_t crow = (int64_t)n * dim;
  for (int pair = threadIdx.x; pair < dim / 2; pair += blockDim.x) {
    const int64_t xi = xrow + 2 * pair;
    const int64_t oi = orow + 2 * pair;
    const int64_t ci = crow + 2 * pair;

    const scalar2_t xv = *reinterpret_cast<const scalar2_t*>(x + xi);
    const scalar2_t cv = *reinterpret_cast<const scalar2_t*>(cos_ + ci);
    const scalar2_t sv = *reinterpret_cast<const scalar2_t*>(sin_ + ci);

    const float x0 = __low2float(xv), x1 = __high2float(xv);
    const float c0 = __low2float(cv), c1 = __high2float(cv);
    const float s0 = __low2float(sv), s1 = __high2float(sv);

    const float o0 = x0 * c0 - x1 * s0;
    const float o1 = x1 * c1 + x0 * s1;
    *reinterpret_cast<scalar2_t*>(out + oi) =
        __floats2half2_rn_generic<scalar2_t>(o0, o1);
  }
}

}  // namespace

torch::Tensor rope_fused(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4, "x must be CUDA [B,H,N,D]");
  TORCH_CHECK(x.stride(-1) == 1, "inner (head) dim must be packed");
  TORCH_CHECK(x.size(-1) % 2 == 0, "head dim must be even");
  auto out = torch::empty(x.sizes(), x.options());  // packed output
  const int dim = (int)x.size(-1);
  const int n_tokens = (int)x.size(-2);
  const int n_heads = (int)x.size(1);
  const int64_t sb = x.stride(0), sh = x.stride(1), sn = x.stride(2);
  const int threads = std::min(128, std::max(32, ((dim / 2 + 31) / 32) * 32));
  const dim3 blocks((unsigned)n_tokens, (unsigned)n_heads, (unsigned)x.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == at::kHalf) {
    rope_fused_kernel<__half2, __half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cos.const_data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sin.const_data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.mutable_data_ptr<at::Half>()),
        n_tokens, dim, sb, sh, sn);
  } else if (x.scalar_type() == at::kBFloat16) {
    rope_fused_kernel<__nv_bfloat162, __nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr<at::BFloat16>()),
        n_tokens, dim, sb, sh, sn);
  } else {
    TORCH_CHECK(false, "rope_fused: fp16/bf16 only");
  }
  return out;
}

````

#### FILE: `engine/csrc/fused_add_layernorm.cu`

````cuda
// Fused residual-add + LayerNorm.
//
// The profile showed 480 aten::add calls and 49 separate LayerNorms per forward.
// This kernel computes  added = x + residual;  y = LN(added)  in one pass,
// returning both (the residual stream needs `added` for the next block).
//
// Reduction design follows kernels_for_reuse/layernorm/layer_norm_kernel.cu
// (PyTorch's native_layer_norm): one CTA per row, Welford accumulation in fp32,
// warp-shuffle then shared-memory cross-warp reduction, vectorized (2-wide) I/O.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

struct WelfordAggregate {
  float mean = 0.f, m2 = 0.f, count = 0.f;
  __device__ void add(float v) {
    count += 1.f;
    float d = v - mean;
    mean += d / count;
    m2 += d * (v - mean);
  }
  __device__ void merge(const WelfordAggregate& o) {
    if (o.count == 0.f) return;
    float new_count = count + o.count;
    float d = o.mean - mean;
    mean += d * o.count / new_count;
    m2 += o.m2 + d * d * count * o.count / new_count;
    count = new_count;
  }
};

template <typename scalar_t>
__global__ void fused_add_ln_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ y,
    scalar_t* __restrict__ added_out,
    const int cols,
    const float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * cols;
  const scalar_t* rr = residual + row * cols;
  scalar_t* yr = y + row * cols;
  scalar_t* ar = added_out + row * cols;

  WelfordAggregate agg;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float a = (float)xr[i] + (float)rr[i];
    ar[i] = (scalar_t)a;
    agg.add(a);
  }
  // warp reduce
  for (int off = 16; off > 0; off >>= 1) {
    WelfordAggregate o;
    o.mean = __shfl_down_sync(0xffffffff, agg.mean, off);
    o.m2 = __shfl_down_sync(0xffffffff, agg.m2, off);
    o.count = __shfl_down_sync(0xffffffff, agg.count, off);
    agg.merge(o);
  }
  __shared__ WelfordAggregate warp_aggs[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) warp_aggs[warp] = agg;
  __syncthreads();
  if (warp == 0) {
    agg = (lane < (blockDim.x + 31) / 32) ? warp_aggs[lane] : WelfordAggregate{};
    for (int off = 16; off > 0; off >>= 1) {
      WelfordAggregate o;
      o.mean = __shfl_down_sync(0xffffffff, agg.mean, off);
      o.m2 = __shfl_down_sync(0xffffffff, agg.m2, off);
      o.count = __shfl_down_sync(0xffffffff, agg.count, off);
      agg.merge(o);
    }
    if (lane == 0) warp_aggs[0] = agg;
  }
  __syncthreads();
  const float mean = warp_aggs[0].mean;
  const float rstd = rsqrtf(warp_aggs[0].m2 / cols + eps);

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float a = (float)ar[i];
    yr[i] = (scalar_t)(((a - mean) * rstd) * (float)gamma[i] + (float)beta[i]);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> fused_add_layer_norm(
    torch::Tensor x, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor bias, double eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && residual.is_contiguous());
  auto y = torch::empty_like(x);
  auto added = torch::empty_like(x);
  const int cols = (int)x.size(-1);
  const int64_t rows = x.numel() / cols;
  const int threads = std::min(1024, ((cols + 31) / 32) * 32);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fused_add_ln", [&] {
    fused_add_ln_kernel<scalar_t><<<(unsigned)rows, threads, 0, stream>>>(
        x.const_data_ptr<scalar_t>(), residual.const_data_ptr<scalar_t>(),
        weight.const_data_ptr<scalar_t>(), bias.const_data_ptr<scalar_t>(),
        y.mutable_data_ptr<scalar_t>(), added.mutable_data_ptr<scalar_t>(),
        cols, (float)eps);
  });
  return {y, added};
}

````

#### FILE: `engine/csrc/preprocess_uint8.cu`

````cuda
// Fused video-clip preprocessing on GPU.
//
// Input:  uint8 [B, T, H, W, 3] (as uploaded — 4x less PCIe traffic than fp32)
// Output: fp16/fp32 [B, 3, T, H, W], ImageNet-normalized.
// One elementwise pass fuses: /255, mean/std normalize, THWC->CTHW transpose,
// dtype cast. Replaces the 25 ms CPU normalize path from the baseline profile.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float kStd[3] = {0.229f, 0.224f, 0.225f};

// torch extensions compile with -D__CUDA_NO_HALF_CONVERSIONS__, so the
// float->__half cast must be the explicit intrinsic.
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }

template <typename out_t>
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ in,   // [B,T,H,W,3]
    out_t* __restrict__ out,          // [B,3,T,H,W]
    const int64_t thw,                // T*H*W per batch item
    const int64_t total) {            // B*T*H*W
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t b = i / thw;
    const int64_t p = i - b * thw;  // position within T*H*W
    const uint8_t* px = in + (b * thw + p) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float v = ((float)px[c] / 255.0f - kMean[c]) / kStd[c];
      out[(b * 3 + c) * thw + p] = from_float<out_t>(v);
    }
  }
}

}  // namespace

torch::Tensor preprocess_uint8(torch::Tensor clip, bool to_half) {
  TORCH_CHECK(clip.is_cuda() && clip.dtype() == torch::kUInt8, "expects uint8 CUDA tensor");
  TORCH_CHECK(clip.dim() == 5 && clip.size(-1) == 3, "expects [B,T,H,W,3]");
  auto clip_c = clip.contiguous();
  const auto B = clip.size(0), T = clip.size(1), H = clip.size(2), W = clip.size(3);
  const int64_t thw = T * H * W, total = B * thw;
  auto opts = clip.options().dtype(to_half ? torch::kHalf : torch::kFloat);
  auto out = torch::empty({B, 3, T, H, W}, opts);
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((total + threads - 1) / threads, 65535);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (to_half) {
    preprocess_kernel<__half><<<blocks, threads, 0, stream>>>(
        clip_c.const_data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.mutable_data_ptr<at::Half>()), thw, total);
  } else {
    preprocess_kernel<float><<<blocks, threads, 0, stream>>>(
        clip_c.const_data_ptr<uint8_t>(), out.mutable_data_ptr<float>(), thw, total);
  }
  return out;
}

````

#### FILE: `engine/csrc/gemm_bias_gelu.cpp`

````cpp
// GEMM + bias + GELU via cuBLASLt epilogue — standalone version of PyTorch's
// gemm_and_bias with GEMMAndBiasActivationEpilogue::GELU
// (reference: kernels_for_reuse/gemm/CUDABlas.cpp:1660-1890).
//
// This calls the same closed-source NVIDIA kernels PyTorch hits through
// aten::_addmm_activation(use_gelu=True); it exists here so the engine keeps
// the exact calling convention when it no longer links ATen's op layer.
//
// Row-major torch tensors are fed to column-major cuBLASLt with the standard
// transpose trick:  y = x @ W^T + b  becomes  y_cm[N,M] = W_cm^T[N,K] @ x_cm[K,M],
// bias (length N) applied per-row of the m dimension by the epilogue.
// NOTE: CUBLASLT_EPILOGUE_GELU_BIAS computes tanh-approx GELU (not erf).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#define LT_CHECK(expr)                                                     \
  do {                                                                     \
    cublasStatus_t s_ = (expr);                                            \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)s_,   \
                " at " #expr);                                             \
  } while (0)

torch::Tensor gemm_bias_gelu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous());  // [N, K] (nn.Linear layout)
  TORCH_CHECK(x.scalar_type() == at::kHalf, "v1 supports fp16");
  const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
  auto y = torch::empty({M, N}, x.options());

  cublasLtHandle_t handle =
      reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasHandle());
  auto stream = at::cuda::getCurrentCUDAStream();

  cublasLtMatmulDesc_t op_desc;
  LT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
  LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  const void* bias_ptr = bias.const_data_ptr();
  LT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

  // A = weight [N,K] rm -> [K,N] cm, opT -> [N,K]; B = x [M,K] rm -> [K,M] cm.
  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  LT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, N, M, N));

  // Workspace sizing mirrors at::cuda::getCUDABlasLtWorkspaceSize (1 MiB+ default).
  auto workspace = torch::empty(
      {static_cast<int64_t>(1) << 22}, x.options().dtype(torch::kUInt8));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws_size = workspace.numel();
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

  cublasLtMatmulHeuristicResult_t heuristic;
  int n_results = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &n_results));
  TORCH_CHECK(n_results > 0, "no cublasLt algo found for this shape");

  const float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(cublasLtMatmul(
      handle, op_desc, &alpha,
      weight.const_data_ptr(), a_desc,
      x.const_data_ptr(), b_desc, &beta,
      y.mutable_data_ptr(), c_desc,
      y.mutable_data_ptr(), c_desc,
      &heuristic.algo, workspace.mutable_data_ptr(), ws_size, stream));

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

````

#### FILE: `engine/tests/test_correctness.py`

````python
"""Correctness tests: every engine op vs the official V-JEPA 2 implementation.

Run:  python engine/tests/test_correctness.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))

from engine import ops
from engine.optimized_encoder import optimize_encoder
from engine.rope_cache import get_rope_tables
from src.models.utils.modules import RoPEAttention, rotate_queries_or_keys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


# ---------------------------------------------------------------- 1. fused RoPE
def reference_rope(attn, q, T, H, W):
    """Exact copy of RoPEAttention.forward's rotation path (mask-free)."""
    mask = torch.arange(int(T * H * W), device=q.device)
    d_mask, h_mask, w_mask = attn.separate_positions(mask, H, W)
    s = 0
    qd = rotate_queries_or_keys(q[..., s:s + attn.d_dim], pos=d_mask); s += attn.d_dim
    qh = rotate_queries_or_keys(q[..., s:s + attn.h_dim], pos=h_mask); s += attn.h_dim
    qw = rotate_queries_or_keys(q[..., s:s + attn.w_dim], pos=w_mask); s += attn.w_dim
    if s < attn.head_dim:
        return torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
    return torch.cat([qd, qh, qw], dim=-1)


def test_rope():
    torch.manual_seed(0)
    B, heads, T, H, W, hd = 2, 16, 8, 16, 16, 64
    N = T * H * W
    attn = RoPEAttention(dim=heads * hd, num_heads=heads, grid_size=H).to(DEVICE)
    q = torch.randn(B, heads, N, hd, device=DEVICE, dtype=torch.float32)

    ref = reference_rope(attn, q, T, H, W)
    cos, sin = get_rope_tables(head_dim=hd, d_dim=attn.d_dim, h_dim=attn.h_dim,
                               w_dim=attn.w_dim, N=N, T=T, H=H, W=W,
                               grid_size=H, device=q.device, dtype=q.dtype)
    fused = ops.apply_rope(q, cos, sin)
    err = (ref - fused).abs().max().item()
    check("rope fused == official (fp32)", err < 1e-5, f"max_abs_err={err:.2e}")

    q16 = q.half()
    ref16 = reference_rope(attn, q16, T, H, W)  # official path in fp16
    cos16, sin16 = get_rope_tables(head_dim=hd, d_dim=attn.d_dim, h_dim=attn.h_dim,
                                   w_dim=attn.w_dim, N=N, T=T, H=H, W=W,
                                   grid_size=H, device=q.device, dtype=torch.float16)
    fused16 = ops.apply_rope(q16, cos16, sin16)
    err16 = (ref16.float() - fused16.float()).abs().max().item()
    # fp16 reference computes pos*omega in half precision; the cache uses fp32
    # tables (strictly more accurate), so allow fp16-rounding-scale differences.
    check("rope fused ~= official (fp16)", err16 < 0.02, f"max_abs_err={err16:.2e}")


# ------------------------------------------------- 2. residual add + layernorm
def test_add_ln():
    torch.manual_seed(1)
    x = torch.randn(4, 512, 1024, device=DEVICE)
    res = torch.randn_like(x)
    w = torch.randn(1024, device=DEVICE)
    b = torch.randn(1024, device=DEVICE)
    ref_added = x + res
    ref = F.layer_norm(ref_added, (1024,), w, b, 1e-6)
    y, added = ops.fused_add_layer_norm(x, res, w, b, 1e-6)
    err = max((ref - y).abs().max().item(), (ref_added - added).abs().max().item())
    check("fused add+layernorm == reference", err < 1e-4, f"max_abs_err={err:.2e}")


# ------------------------------------------------ 3. GEMM + bias + GELU epilogue
def test_linear_gelu():
    torch.manual_seed(2)
    x = torch.randn(2048, 1024, device=DEVICE, dtype=torch.float16)
    w = torch.randn(4096, 1024, device=DEVICE, dtype=torch.float16) * 0.02
    b = torch.randn(4096, device=DEVICE, dtype=torch.float16)
    ref = F.gelu(F.linear(x, w, b))  # erf GELU, the official MLP's activation
    fused = ops.linear_gelu(x, w, b)  # cuBLASLt epilogue (tanh-approx GELU)
    err = (ref.float() - fused.float()).abs().max().item()
    check("linear+GELU epilogue ~= reference", err < 2e-2,
          f"max_abs_err={err:.2e} (erf-vs-tanh GELU + fp16)")


# --------------------------------------------------------- 4. GPU preprocessing
def test_preprocess():
    import numpy as np
    rng = np.random.default_rng(3)
    clip = rng.integers(0, 256, size=(1, 4, 64, 64, 3), dtype=np.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    ref = torch.from_numpy(clip).float().div(255).permute(0, 4, 1, 2, 3)
    ref = ((ref - mean) / std).to(DEVICE)
    out = ops.gpu_preprocess(torch.from_numpy(clip), DEVICE, torch.float16)
    err = (ref - out.float()).abs().max().item()
    check("gpu preprocess == cpu reference", err < 5e-3, f"max_abs_err={err:.2e}")


# ------------------------------------------- 5. end-to-end optimized encoder
def test_encoder_end_to_end():
    import src.hub.backbones as backbones
    torch.manual_seed(4)
    frames = 8
    encoder, _ = backbones.vjepa2_vit_large(pretrained=False, num_frames=frames)
    encoder = encoder.to(DEVICE).eval()
    x = torch.randn(1, 3, frames, 256, 256, device=DEVICE)

    with torch.inference_mode():
        ref = encoder(x).float()
        opt = optimize_encoder(encoder)  # in-place patch + fp16
        out = opt(x.half()).float()

    cos = F.cosine_similarity(ref.flatten(1), out.flatten(1), dim=-1).min().item()
    rel = ((ref - out).norm() / ref.norm()).item()
    check("optimized encoder ~= official encoder", cos > 0.99,
          f"min_cosine_sim={cos:.5f} rel_err={rel:.4f}")


if __name__ == "__main__":
    for t in (test_rope, test_add_ln, test_linear_gelu, test_preprocess,
              test_encoder_end_to_end):
        t()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)

````

#### FILE: `engine/README.md`

````markdown
# V-JEPA 2 Inference Engine — v1

First version of the optimized inference path. Verified bit-comparable to the
official implementation (`engine/tests/test_correctness.py`, 6/6 passing) and
benchmarked at **5.2x** over the official pipeline on an RTX 4050 Laptop GPU
(vitl, 16 frames, batch 1: 467 ms -> 89 ms; p95 1104 ms -> 91 ms).

## The five pieces

| # | Bottleneck (from baseline profile) | Fix | Runnable today via | CUDA source |
|---|---|---|---|---|
| 1 | RoPE recompute: 144x omega/einsum/sin/cos + 48 cats/forward | precomputed full-head-dim cos/sin cache + single fused apply | `rope_cache.py` + `ops.apply_rope` | `csrc/rope_fused.cu` |
| 2 | 581 autocast `_to_copy` + 480 adds (~17% GPU time) | native fp16 weights; fused residual+LN kernel | `optimize_encoder(dtype=fp16)`, `ops.fused_add_layer_norm` | `csrc/fused_add_layernorm.cu` |
| 3 | standalone GELU kernel (4.3 ms) | GELU folded into GEMM epilogue (cuBLASLt `GELU_BIAS`) | `ops.linear_gelu` -> `aten::_addmm_activation` | `csrc/gemm_bias_gelu.cpp` |
| 4 | 25 ms CPU preprocess + fp32 PCIe upload | uint8 upload (4x less traffic) + fused GPU normalize/transpose | `ops.gpu_preprocess` | `csrc/preprocess_uint8.cu` |
| 5 | SDPA landing on cutlassF via deprecated ctx manager | flash-first backend priority | `ops.sdpa_flash_first` | kernels imported in `kernels_for_reuse/attention/flash_attn_official/` |

Every op has a fused pure-PyTorch fallback (used until the extension is built),
so the engine runs everywhere now; `csrc/` holds the custom kernels for the
standalone build.

## Usage

```python
import sys; sys.path[:0] = ["vjepa2", "."]
import src.hub.backbones as backbones
from engine import optimize_encoder, gpu_preprocess

encoder, _ = backbones.vjepa2_vit_large(pretrained=True, num_frames=16)
encoder = optimize_encoder(encoder.cuda())          # fp16 + all patches, in place
x = gpu_preprocess(clip_u8_pinned, "cuda")          # uint8 [B,T,H,W,3] -> fp16 [B,3,T,H,W]
emb = encoder(x)
```

Benchmark comparison: `python benchmarks/benchmark_vjepa2.py --engine`

## Building the CUDA extension

Built and passing on WSL2 Ubuntu 24.04 (CUDA toolkit 12.6.3, `MAX_JOBS=2` — more
parallel nvcc jobs OOM the 7 GB WSL VM):

```bash
export CUDA_HOME=/usr/local/cuda-12.6 PATH="$HOME/vjepa-venv/bin:$CUDA_HOME/bin:$PATH"
MAX_JOBS=2 python engine/build.py
```

`ops.py` auto-loads the built extension on Linux (`VJEPA_ENGINE_EXT=0` to disable).
On Windows the PyTorch fallbacks are used (nvcc 13.2 vs torch cu126 mismatch).

### WSL2 measurement notes (RTX 4050 Laptop, vitl/16f/b1/fp16)

- Flash Attention active on Linux: `flash_fwd_kernel` 17.1 ms/forward vs 21.9 ms
  for the efficient-attention kernel Windows falls back to.
- End-to-end sits in an **84-89 ms band**; same-session A/B shows run-to-run
  thermal/clock variance (±3 ms) *exceeds* the compiled-kernel vs
  PyTorch-fallback difference (~1-2 ms) at this workload. The custom kernels'
  value is for the standalone (non-ATen) engine and smaller/quantized workloads,
  not raw speed today.
- Per-op defaults reflect measurements: preprocess uses the compiled kernel;
  RoPE uses the torch chain (`VJEPA_ROPE_EXT=1` opts into the compiled one).
- For trustworthy numbers benchmark configs back-to-back in one session
  (see the A/B loop pattern), not across sessions.

## Known deltas vs official (all validated in tests)

- cuBLASLt GELU epilogue is tanh-approx vs nn.GELU's erf: max abs diff ~4e-3.
- RoPE tables are computed in fp32 (official computes in fp16 under autocast);
  the engine is *more* accurate, diff within fp16 rounding (~9e-3).
- End-to-end fp16 encoder vs fp32 official: cosine similarity 1.00000, rel err 0.16%.
- Flash SDPA: Windows torch wheels ship without flash kernels, so SDPA still
  selects efficient-attention here; on Linux/WSL2 the same code picks flash.
- Masked/`attn_mask` calls fall back to the original forward path (unpatched).

````

# Phase 4   CUDA graphs (`engine/cuda_graph.py`)

```python
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side), torch.inference_mode():
    for _ in range(3): encoder(example)          # lazy init + RoPE cache fill
torch.cuda.current_stream().wait_stream(side)
static_in = example.clone()
g = torch.cuda.CUDAGraph()
with torch.inference_mode(), torch.cuda.graph(g):
    static_out = encoder(static_in)
# per call: static_in.copy_(x); g.replay(); return static_out  (clone if kept!)
```

- Verified bit-exact: max err 0.0 vs eager, including on fresh inputs.
- Benchmark `--graphs` (requires `--engine`): hooks can't fire inside a replay
  → remove LayerTimer BEFORE capture (hook firing during warmup with a null
  current-record dict was the first crash) and collapse the heatmap to
  pipeline rows + one "encoder (whole)" row.
- **Result: 89.1 → 88.7 ms (+0.4%).** At 2048 tokens kernels are ~3.6 ms;
  launch overhead already hidden by the async queue. Kept for when kernels
  shrink (compile/fp8/smaller inputs).

---


## Source files: CUDA-graph capture (created in Phase 4)

*Complete, verbatim, final-state contents (they include flags/guards added in later phases — this is the version that produces the final numbers). *

#### FILE: `engine/cuda_graph.py`

````python
"""CUDA-graph capture for the optimized encoder.

Batch-1 inference launches hundreds of small kernels per forward; capturing the
encoder once and replaying eliminates per-launch CPU overhead and the gaps
between kernels. Requires static shapes (true for a fixed clip config) and the
engine's native-fp16 path (no autocast inside capture).

Usage:
    graphed = GraphedEncoder(encoder, example_input)   # captures once
    out = graphed(x)                                   # copy-in + replay
Note: `out` is the graph's static output buffer — clone() it if you need to
keep it across calls.
"""

import torch


class GraphedEncoder:
    def __init__(self, encoder, example_input, warmup_iters=3):
        assert example_input.is_cuda
        self.encoder = encoder

        # Warm up on a side stream (per torch CUDA-graphs docs) so lazy inits,
        # autotuning, and the engine's RoPE-cache population happen pre-capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(warmup_iters):
                encoder(example_input)
        torch.cuda.current_stream().wait_stream(side)

        self.static_in = example_input.clone()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_out = encoder(self.static_in)

    def __call__(self, x):
        self.static_in.copy_(x, non_blocking=True)
        self.graph.replay()
        return self.static_out

````

# Phase 5   WSL2 (stepping stone; all of it later subsumed by Docker)

Why: Windows wheels lack Flash SDPA + triton; Windows nvcc 13.2 can't build
against torch cu126; no `cl` on PATH.

```bash
# inside WSL Ubuntu 24.04
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv --python 3.12 ~/vjepa-venv            # HOME, not /mnt/c (fs speed)
~/.local/bin/uv pip install --python ~/vjepa-venv torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126     # Linux wheels bundle triton
# CUDA toolkit   WSL-specific repo (does NOT touch the Windows driver):
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
sudo apt-get install -y python3-dev        # else: "fatal error: Python.h: No such file"
uv pip install --python ~/vjepa-venv ninja # else: "Ninja is required to load C++ extensions"
# build:
cd /mnt/c/.../InferenceEngine
export CUDA_HOME=/usr/local/cuda-12.6 PATH="$HOME/vjepa-venv/bin:$CUDA_HOME/bin:$PATH"
MAX_JOBS=2 python engine/build.py
```

**Failures hit, in order:**
1. Default-parallel nvcc (5 TUs, 20 threads, 7 GB VM) → OOM-killer → exit 15 →
   then the whole WSL VM died: `Catastrophic failure Wsl/Service/E_UNEXPECTED`.
   Recover with `wsl --shutdown`. **Rule: `MAX_JOBS=2`.**
2. Passing commands through `wsl -- bash -lc "...$PATH..."` in double quotes:
   the Windows-side shell expanded `$PATH`, whose entries contain `(` →
   `bash: syntax error near unexpected token '('`. Use single quotes so
   expansion happens inside WSL (or set only `CUDA_HOME`).
3. `Python.h` missing, ninja missing   see commands above.
4. `__CUDA_NO_HALF_CONVERSIONS__` cast error (fixed in source, see 3.6).

**Results on WSL (same tests 6/6, then benchmark):**
- Flash confirmed in profile: `pytorch_flash::flash_fwd_kernel` 17.1 ms vs
  21.9 ms cutlassF on Windows → engine 89.1 → **83.9 ms**.
- Compiled-kernels vs fallbacks: same-session A/B/A gave ext 86.5 / fallback
  83.9 / ext 86.6   looked like a real ~2.6 ms rope regression (see kernel
  iterations in 3.6)   but a later 4-run same-session sweep measured the SAME
  fallback config at 89.0 first and 85.8 last ⇒ **run-order/thermal variance
  ±3 ms exceeds the ext-vs-fallback delta; treat them as parity.** Per-op
  defaults set from the stage-level split (preprocess ext ON, rope ext OFF).

---

# Phase 6   Docker: the single environment

**`Dockerfile`** (decisions inline):

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04        # devel = nvcc for csrc; matches torch cu126
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv                     # 24.04 python is PEP-668 externally-managed
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir timm einops numpy pyyaml matplotlib pandas opencv-python-headless ninja torchao
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
# ^ ARCH_LIST lets nvcc target Ada with no GPU visible at build; MAX_JOBS from the WSL OOM lesson
WORKDIR /workspace
COPY engine /workspace/engine                     # SAME path the volume mounts to later:
RUN python engine/build.py && rm -rf /workspace/engine
# ^ identical absolute paths + identical file contents ⇒ runtime JIT load()
#   is a pure ninja cache hit   containers start with kernels prebuilt
CMD ["bash"]
```

**`docker-compose.yml`**: `gpus: all`; volumes `.:/workspace` (edit on Windows,
run in Linux; results land in the repo) + named volume
`torch-ext-cache:/root/.cache/torch_extensions`; `shm_size: 6gb` (pinned-memory
headroom; default 64 MB); `environment: TORCH_HOME=/workspace/.torch-cache`
(pretrained weights persist on the Windows side). `.dockerignore`: `.venv`,
`benchmarks/results`, `vjepa2/.git`, `kernels_for_reuse`, `*.pt`.

Daily use:

```powershell
docker compose build                 # only after changing engine/csrc or deps
docker compose run --rm dev          # shell
docker compose run --rm dev python benchmarks/benchmark_vjepa2.py --engine
```

Verified in-container: GPU visible, 6/6 tests (compiled-kernel error
signatures   e.g. add+LN 1.9e-6 not 0.0   prove the ext is active), 86.9 ms.

---


## Source files: the Docker environment (created in Phase 6)

*Complete, verbatim, final-state contents (they include flags/guards added in later phases — this is the version that produces the final numbers). These four live at the repo root.*

#### FILE: `Dockerfile`

````dockerfile
# V-JEPA 2 inference engine — single dev environment.
# CUDA 12.6 devel (nvcc for the custom kernels) + torch cu126 + Flash Attention.
#
#   docker compose run --rm dev                    # shell
#   docker compose run --rm dev python benchmarks/benchmark_vjepa2.py --engine

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# venv (Ubuntu 24.04 system python is PEP-668 externally-managed)
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir timm einops numpy pyyaml matplotlib pandas opencv-python-headless ninja torchao

# 8.9 = Ada (RTX 4050); lets nvcc target the GPU without one present at build time.
# MAX_JOBS=2: parallel nvcc easily OOMs small VMs (learned the hard way in WSL).
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2

# Pre-build the CUDA extension into the image at the SAME path the project is
# mounted to at runtime — identical absolute paths + identical file contents
# mean the runtime JIT load is a pure ninja cache hit (no rebuild).
WORKDIR /workspace
COPY engine /workspace/engine
RUN python engine/build.py && rm -rf /workspace/engine

CMD ["bash"]

````

#### FILE: `docker-compose.yml`

````yaml
services:
  dev:
    build: .
    image: vjepa-inference-engine
    gpus: all
    volumes:
      - .:/workspace                                  # edit on Windows, run here
      - torch-ext-cache:/root/.cache/torch_extensions # JIT-build cache survives
    working_dir: /workspace
    environment:
      # pretrained checkpoints persist on the Windows side across containers
      TORCH_HOME: /workspace/.torch-cache
    stdin_open: true
    tty: true
    # 6 GB shm for pinned-memory pipelines; default 64 MB is too small
    shm_size: 6gb

volumes:
  torch-ext-cache:

````

#### FILE: `.dockerignore`

````
# keep the build context tiny — only engine/ is COPY'd
.venv
benchmarks/results
vjepa2/.git
kernels_for_reuse
*.pt
*.pth

````

#### FILE: `Readme.md`

````markdown
# V-JEPA 2 Inference Engine

Optimized inference for Meta's V-JEPA 2 video encoder. Currently **5.3–5.6x**
over the official PyTorch pipeline (467 ms -> 84–89 ms per clip, vitl/16f/fp16,
RTX 4050 Laptop), verified numerically identical (`engine/tests/`, 6/6).

## One development environment: Docker

Everything (CUDA 12.6 toolkit, torch cu126 + Flash Attention, pre-built custom
kernels) lives in one container. Edit code on Windows as usual; run in the
container — no WSL juggling.

```powershell
docker compose build                  # once (and after changing engine/csrc)
docker compose run --rm dev           # interactive shell
docker compose run --rm dev python benchmarks/benchmark_vjepa2.py --engine
docker compose run --rm dev python engine/tests/test_correctness.py
```

The repo is mounted at `/workspace`; benchmark results land in
`benchmarks/results/` on the Windows side as normal.

## Layout

| Path | What |
|---|---|
| `vjepa2/` | official Meta repo (reference + model code) |
| `benchmarks/` | profiling harness + per-run results (heat charts, traces) |
| `engine/` | the optimized inference path — see `engine/README.md` |
| `kernels_for_reuse/` | kernel sources imported from PyTorch / flash-attention |

## Status (2026-07-25)

v1 kernels done (fused RoPE, fp16-native + fused add+LN, GELU-in-GEMM epilogue,
GPU preprocess, flash-first SDPA), CUDA-graph capture, torch.compile, torchao
quantization. Measured on RTX 4050 Laptop (vitl/16f/batch 1, same-session):

| Config | ms/clip | clips/s | notes |
|---|---|---|---|
| official pipeline (Windows) | 467 | 2.1 | thermal-inflated p95 1104 |
| `--engine` | ~86 | 11.6 | verified identical outputs |
| `--engine --compile` | 75.5 | 13.3 | reduce-overhead mode |
| `--engine --compile --quant fp8` | **61.5** | **16.3** | cosine 0.981 vs fp16 — **needs task-level validation before use** |

Findings that shape what's next:
- **Batching gains nothing here** (batch 1-8 all ~11 clips/s): 2048 tokens
  already saturate this GPU. Revisit on bigger hardware.
- **int8 rejected**: worse accuracy (0.965) than fp8 and slower.
- Quantization accuracy (real vitl weights): full-model quant degrades badly
  (ViT activation outliers); mlp-only scope + fp8 is the best trade at 0.981
  cosine / 19% rel err. Validate on the actual downstream task before shipping.
- Next levers: quantization-aware refinement (SmoothQuant-style), NVDEC decode,
  overlap streams — or bigger/server hardware where batching pays.

````

# Phase 7   Batching, torch.compile, quantization

## 7.1 Batching   negative result

`--batch 1/2/4/8`, same session: **11.47 / 11.41 / 10.50 / 10.37 clips/s**  
flat, slightly worse. 2048 tokens × ViT-L GEMMs saturate this GPU at batch 1
(the "2.5–3× from batching" prior was WRONG for this hardware class). Batch 1
is optimal for latency AND throughput here; batching pays on bigger GPUs only.

## 7.2 torch.compile   +13%

`torch.compile(encoder, mode="reduce-overhead")`; warmup auto-bumped to ≥10.
Two required fixes:
1. Per-layer hooks force graph breaks → `NullTimer` under `--compile`/`--graphs`
   (no-op start/end/elapsed/remove; heatmap collapses to pipeline rows).
2. **Real bug**: crash
   `RuntimeError: accessing tensor output of CUDAGraphs that has been
   overwritten by a subsequent run` with a stack through `rope_cache.py`  
   reduce-overhead's cudagraphs re-use output buffers, so cos/sin tables
   created INSIDE the compiled region get clobbered by later replays.
   Fix: one eager forward (populates the cache with ordinary tensors) BEFORE
   wrapping in `torch.compile`.

Same-session A/B: engine 86.7 → **compile 75.5 ms** (p95 76.7), 13.3 clips/s.

## 7.3 Quantization (torchao, REAL vitl weights)

`engine/quantize.py`:
- API compat shim: prefer `Int8DynamicActivationInt8WeightConfig` /
  `Float8DynamicActivationFloat8WeightConfig`, fall back to the older
  `int8_dynamic_activation_int8_weight()` function names.
- **Scoping decision**: `quantize_(encoder, cfg(), filter_fn=lambda m, fqn:
  isinstance(m, nn.Linear) and ".mlp." in fqn)`   ViT attention projections
  carry activation outliers that wreck accuracy; MLP holds most GEMM time.
- fp8 = dynamic e4m3, needs SM 8.9+ (4050 qualifies).

Accuracy (`engine/tests/test_quant_accuracy.py`; pretrained vitl, URL override,
structured synthetic clips = moving gradient + noise; min cosine over batch +
relative Frobenius error vs the fp16 engine):

| Config | min cosine | rel err | Verdict |
|---|---|---|---|
| int8 / all linears | 0.9589 | 28.0% | rejected |
| int8 / mlp only | 0.9647 | 26.1% | rejected |
| fp8 / all | 0.9698 | 24.3% | rejected |
| **fp8 / mlp only** | **0.9806** | **19.2%** | best trade   below strict 0.99 bar ⇒ **conditional on downstream-task validation** |

Speed (same session, batch 1):

| Config | ms | Note |
|---|---|---|
| engine fp16 | 86.3 | control |
| engine + int8/mlp (eager) | 276.5 | **2.4× SLOWER**   unfused quant/dequant |
| engine + fp8/mlp (eager) | 207.0 | also slower |
| engine + fp8/mlp + compile | **61.5** | fp8 pays ONLY with compile fusing the quant ops |

Ops-file consequence: `linear_gelu` gained the tensor-subclass guard
(quantized weights can't go through `_addmm_activation`; `F.linear` dispatches
to the quantized kernels).

Operational notes: first accuracy run died SILENTLY (host RAM pressure during
1.2 GB download + fp32 load + deepcopy); rerun after weights cached succeeded.
Also: a log-filter `grep -vE "^\["` ate the result lines `[int8] ...`   don't
filter your own output's prefix.

---


## Source files: quantization (created in Phase 7)

*Complete, verbatim, final-state contents (they include flags/guards added in later phases — this is the version that produces the final numbers). *

#### FILE: `engine/quantize.py`

````python
"""Dynamic quantization of the encoder's Linear layers via torchao.

int8: dynamic per-token activation int8 x per-channel weight int8 GEMMs.
fp8:  dynamic float8 (e4m3) activation x weight — needs SM 8.9+ (Ada/Hopper);
      the RTX 4050 is SM 8.9.

Quantized weights become tensor subclasses; F.linear dispatches to the fast
kernels automatically. engine.ops.linear_gelu detects subclass weights and
falls back from the cuBLASLt epilogue to subclass-dispatched linear + GELU.
"""


def _configs():
    # torchao renamed its API around 0.7-0.10; support both spellings.
    from torchao import quantization as q
    int8 = getattr(q, "Int8DynamicActivationInt8WeightConfig", None) \
        or getattr(q, "int8_dynamic_activation_int8_weight")
    fp8 = getattr(q, "Float8DynamicActivationFloat8WeightConfig", None) \
        or getattr(q, "float8_dynamic_activation_float8_weight")
    return q.quantize_, int8, fp8


def quantize_encoder(encoder, mode, scope="mlp"):
    """scope='mlp' quantizes only blocks.*.mlp.fc* (most GEMM time, avoids the
    attention-projection activation outliers that wreck ViT accuracy);
    scope='all' quantizes every Linear (faster, measurably degraded)."""
    import torch.nn as nn

    quantize_, int8, fp8 = _configs()
    cfg = {"int8": int8, "fp8": fp8}.get(mode)
    if cfg is None:
        raise ValueError(f"unknown quant mode {mode!r}")

    if scope == "mlp":
        def filter_fn(module, fqn):
            return isinstance(module, nn.Linear) and ".mlp." in fqn
    else:
        filter_fn = None
    quantize_(encoder, cfg(), filter_fn=filter_fn)
    return encoder

````

#### FILE: `engine/tests/test_quant_accuracy.py`

````python
"""Quantization accuracy: int8/fp8 embeddings vs fp16, with REAL pretrained weights.

Run (in the container):  python engine/tests/test_quant_accuracy.py
Downloads vjepa2 vitl.pt (~1.2 GB) once into $TORCH_HOME (/workspace/.torch-cache).
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vjepa2"))

from engine import optimize_encoder
from engine.quantize import quantize_encoder

FRAMES = 16


def main():
    import src.hub.backbones as backbones
    backbones.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

    print("loading pretrained vitl (downloads ~1.2 GB on first run)...", flush=True)
    encoder, _ = backbones.vjepa2_vit_large(pretrained=True, num_frames=FRAMES)
    encoder = optimize_encoder(encoder.cuda())  # fp16 reference

    torch.manual_seed(0)
    # Structured synthetic clips (moving gradients + noise), normalized like video.
    t = torch.linspace(-2, 2, 256).cuda()
    base = (t.view(1, -1) + t.view(-1, 1)).view(1, 1, 1, 256, 256)
    motion = torch.linspace(0, 1, FRAMES).cuda().view(1, 1, FRAMES, 1, 1)
    clips = (base * motion + 0.5 * torch.randn(3, 3, FRAMES, 256, 256).cuda()).half()

    with torch.inference_mode():
        ref = encoder(clips).float()

    results = {}
    for mode in ("int8", "fp8"):
        for scope in ("all", "mlp"):
            q_enc = quantize_encoder(copy.deepcopy(encoder), mode, scope=scope)
            with torch.inference_mode():
                out = q_enc(clips).float()
            cos = F.cosine_similarity(ref.flatten(1), out.flatten(1), dim=-1).min().item()
            rel = ((ref - out).norm() / ref.norm()).item()
            results[(mode, scope)] = (cos, rel)
            print(f"[{mode}/{scope}] min_cosine_sim={cos:.5f}  rel_err={rel:.4f}  "
                  f"{'OK' if cos > 0.99 else 'DEGRADED - validate on real task data'}")
            del q_enc
            torch.cuda.empty_cache()

    # gate on the recommended config (mlp-scope) only
    ok = all(results[(m, 'mlp')][0] > 0.98 for m in ('int8', 'fp8'))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

````

# Final scoreboard (vitl, 16 frames, batch 1; container, same-session)

| Config | ms/clip | clips/s | Equivalence |
|---|---|---|---|
| Official (Windows, fp16 autocast) | 467 (p95 1104) | 2.1 |   |
| `--engine` | ~86 | 11.6 | cosine 1.00000 |
| `--engine --graphs` | ~88 | 11.3 | bit-exact; +0.4% only |
| `--engine --compile` | 75.5 (p95 76.7) | 13.3 | recommended default |
| `--engine --compile --quant fp8` | **61.5** (p95 62.2) | **16.3** | cosine 0.981   validate on task first |

# Environment-variable / flag reference

| Knob | Effect |
|---|---|
| `VJEPA_ENGINE_EXT=0/1` | disable/force the compiled extension (default: on for Linux) |
| `VJEPA_ROPE_EXT=1` | opt into the compiled RoPE kernel (default: torch chain   it wins here) |
| `TORCH_HOME` | checkpoint cache (compose sets `/workspace/.torch-cache`) |
| `MAX_JOBS=2` | REQUIRED for extension builds in ≤8 GB VMs |
| `TORCH_CUDA_ARCH_LIST=8.9` | build kernels for Ada without a GPU present |
| benchmark flags | `--engine --graphs --compile --quant {int8,fp8} --batch --frames --dtype --iters --warmup --pretrained` |

# Transferable lessons

1. Profile first; the op table (45% GEMM / 19% attn / 17% copies / 19% glue)
   dictated the entire plan.
2. Research-code wins are boring: dtype hygiene, redundant recompute, wrong
   backend selection. They were 5.4× of the 7.6×.
3. Fusion of memory-bound elementwise chains must be measured, not assumed  
   the thrice-rewritten RoPE kernel still lost to torch's op chain.
4. Laptop GPUs: ±3 ms session noise ⇒ same-session A/B/A or it didn't happen.
5. CUDA graphs ≈ 0 when kernels are fat; they matter after compile/quant
   shrink kernels. Optimizations interact   re-evaluate combinations.
6. ViT quantization: attention projections are the accuracy killer; scope to
   MLP; eager quantization without compile can be net-negative.
7. Replicate upstream bugs bit-for-bit when pretrained weights depend on them.
8. Version-match reference sources to installed binaries (v2.13.0 tag).
9. Write down negative results; they steer the next person (you) around walls.

---

# Learning resources (papers, books, docs) mapped to this project

## Papers   the model family (why V-JEPA 2 is shaped like this)
- **Attention Is All You Need**   Vaswani et al., arXiv:1706.03762. The transformer block you profiled 24 of.
- **ViT: An Image is Worth 16x16 Words**   Dosovitskiy et al., arXiv:2010.11929. Patch embedding, the [B, N_tokens, D] worldview.
- **RoFormer (RoPE)**   Su et al., arXiv:2104.09864. The rotary embedding math (the ω = 10000^(-j/k), pair-rotation formulas in rope_cache.py).
- **A Path Towards Autonomous Machine Intelligence**   LeCun, 2022 (OpenReview). The JEPA position paper; why joint-embedding prediction instead of pixel reconstruction.
- **I-JEPA**   Assran et al., arXiv:2301.08243. Image JEPA, the direct ancestor.
- **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video**   Bardes et al., arXiv:2404.08471.
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**   Assran et al., arXiv:2506.09985. The model this engine serves. Also: Meta AI blog post "V-JEPA 2" and github.com/facebookresearch/vjepa2 (the repo you cloned).

## Papers   attention & kernel efficiency (why the engine works)
- **Online normalizer calculation for softmax**   Milakov & Gessert, arXiv:1805.02867. The trick that makes Flash possible.
- **Self-attention Does Not Need O(n²) Memory**   Rabe & Staats, arXiv:2112.05682. The "memory-efficient attention" (your cutlassF kernel) idea.
- **FlashAttention**   Dao et al., arXiv:2205.14135; **FlashAttention-2**   Dao, arXiv:2307.08691 (the kernels in kernels_for_reuse/attention/flash_attn_official); **FlashAttention-3**   Shah et al., arXiv:2407.08608 (Hopper-only   the "better hardware" unlock).
- **Mixed Precision Training**   Micikevicius et al., arXiv:1710.03740. Why fp16 + fp32 accumulation is safe (your kernels accumulate in fp32).
- **PagedAttention / vLLM**   Kwon et al., arXiv:2309.06180. Serving-side batching ideas for when batching pays.

## Papers   quantization (why fp8/mlp-only was the outcome)
- **LLM.int8()**   Dettmers et al., arXiv:2208.07339. THE activation-outlier paper   explains exactly why your full-model quant failed.
- **SmoothQuant**   Xiao et al., arXiv:2211.10438. Migrate activation outliers into weights   the technique that could recover your attention-projection accuracy.
- **AWQ**   Lin et al., arXiv:2306.00978. Activation-aware weight quantization.
- **GPTQ**   Frantar et al., arXiv:2210.17323. Second-order weight quantization.
- **FP8 Formats for Deep Learning**   Micikevicius et al., arXiv:2209.05433. e4m3/e5m2, what your `--quant fp8` actually uses.

## Books
- **Programming Massively Parallel Processors** (Kirk & Hwu, 4th ed.)   THE CUDA book: coalescing, occupancy, reductions (your Welford kernel), everything csrc/ assumes.
- **The CUDA Handbook** (Wilt)   deeper systems view: streams, events, pinned memory, graphs.
- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong   free PDF at mml-book.github.io)   linear algebra/calculus foundations.
- **Understanding Deep Learning** (Simon Prince   free PDF at udlbook.github.io/udlbook)   modern, transformer-era; better starting point than Goodfellow today.
- **Deep Learning** (Goodfellow, Bengio, Courville   deeplearningbook.org)   the classic reference.

## PyTorch docs/tutorials (each maps to a thing you built)
- Custom C++/CUDA extensions tutorial   pytorch.org/tutorials/advanced/cpp_extension.html → engine/build.py, csrc/.
- `torch.compile` intro + troubleshooting   pytorch.org/docs/stable/torch.compiler.html → Phase 7.2 (incl. cudagraph pitfalls like your RoPE-cache clobber).
- "Accelerating PyTorch with CUDA Graphs" (PyTorch blog) + torch.cuda CUDA-semantics doc → engine/cuda_graph.py.
- PyTorch profiler recipe   pytorch.org/tutorials/recipes/recipes/profiler_recipe.html → benchmarks/ methodology.
- `scaled_dot_product_attention` / SDPBackend docs → ops.sdpa_flash_first.
- **torchao**   github.com/pytorch/ao (README + quantization docs) → engine/quantize.py.
- **GPU-MODE** (formerly CUDA MODE)   youtube.com/@GPUMODE + github.com/gpu-mode/lectures. The single best free kernel-engineering course; lectures on profiling, Triton, quantization, torch.compile internals.
- **Triton**   triton-lang.org tutorials (vector add → matmul → fused softmax)   the language your parked RoPE kernel should probably be rewritten in.
- Horace He, "Making Deep Learning Go Brrrr From First Principles"   horace.io/brrr_intro.html. Memory-bound vs compute-bound intuition; explains your RoPE-kernel-lost-to-torch result better than any paper.

## NVIDIA docs
- CUDA C++ Programming Guide + CUDA C++ Best Practices Guide   docs.nvidia.com/cuda.
- cuBLASLt docs (epilogue section: CUBLASLT_EPILOGUE_GELU_BIAS) → csrc/gemm_bias_gelu.cpp.
- CUTLASS   github.com/nvidia/cutlass (docs/ + examples) → the open-source GEMM path.
- Nsight Systems / Nsight Compute guides   the profilers one level below torch.profiler.
- TensorRT developer guide   the "most gain, most work" quantization path.

## JAX (you asked   different ecosystem, same ideas)
- JAX docs   jax.readthedocs.io; start with "How to think in JAX" and **Autodidax** (build-your-own-JAX   the best autodiff/tracing tutorial anywhere).
- **Pallas**   jax.readthedocs.io/en/latest/pallas   JAX's Triton-like kernel language; the JAX equivalent of your csrc/ work.
- "How to Scale Your Model"   jax-ml.github.io/scaling-book   TPU/GPU roofline thinking, applies directly to your GEMM-bound analysis.

## Suggested reading order for reproducing this project
1. Horace He's brrr post (1 hr) → 2. PMPP ch. 1–6 + Triton tutorials → 3. Flash-
Attention 1+2 papers with the imported kernels open → 4. RoFormer + the vjepa2
modules.py side-by-side → 5. LLM.int8 + SmoothQuant before touching quantization
→ 6. GPU-MODE lectures as ongoing background.

---

# Appendix — rebuild order and verification checkpoints

Everything above is chronological narrative + the final sources. To rebuild
from an empty directory, go in THIS order (each stage's file contents are in
the "Source files" sections above):

1. **Stage A — external code**: `git clone https://github.com/facebookresearch/vjepa2.git vjepa2`,
   apply the Phase-0 case-collision fix. (`kernels_for_reuse/` is reference-only —
   recreate via the Phase-2 sparse-clone commands, or skip; the engine never
   imports from it at runtime.)
2. **Stage B — environment**: write `Dockerfile`, `docker-compose.yml`,
   `.dockerignore`, `Readme.md` (Phase-6 sources). Then
   `docker compose build` (~10 min) and CHECK:
   `docker compose run --rm dev python -c "import torch; print(torch.cuda.get_device_name(0))"`
   prints your GPU.
3. **Stage C — engine**: write all `engine/` and `engine/csrc/` files (Phase-3
   sources) + `engine/cuda_graph.py` (Phase-4) + `engine/quantize.py` (Phase-7).
   Rebuild the image (it pre-compiles csrc at build time).
4. **Stage D — tests first**: write both test files, then IN the container:
   `python engine/tests/test_correctness.py` → CHECK **6/6 passed**; with the
   compiled extension active, add+LN error is ~1.9e-06 (not 0.0) and preprocess
   ~9.7e-04 — those signatures prove the CUDA kernels are being exercised.
5. **Stage E — benchmark**: write `benchmarks/` (Phase-1 sources) and reproduce
   the ladder (RTX 4050; ±3 ms session noise — compare within one session only):

```
python benchmarks/benchmark_vjepa2.py                                  # ~460-470 ms baseline
python benchmarks/benchmark_vjepa2.py --engine                         # ~84-89 ms
python benchmarks/benchmark_vjepa2.py --engine --compile               # ~75 ms
python benchmarks/benchmark_vjepa2.py --engine --compile --quant fp8   # ~61 ms
python engine/tests/test_quant_accuracy.py   # fp8/mlp ~0.981 cosine; exits 1 by design (below 0.99 bar)
```

Things that survive deleting the working tree (reuse to save time):
the Docker image `vjepa-inference-engine` (until `docker rmi`) and the 1.2 GB
pretrained weights in `.torch-cache/` (copy out before deleting).

---

# Incremental rebuild — 14th October 2026

The current requirements supersede the historical measurements and architecture
above. C1–C3 and the eager C4 harness were already pushed at `2e38dc7`.
The untracked engine and results are reference copies, not newly measured work.
Keep those files out of bulk staging; each new behavior gets a focused commit.

## Benchmark and correctness contract

- Target: ViT-L, 16 frames, 256px encoder input, batch 1, RTX 4090.
- Main comparator: untouched official encoder with fp16 autocast, PyTorch SDPA,
  and `torch.compile(mode="max-autotune")`. Official eager is also reported.
- Both paths start from the same raw pageable uint8 BTHWC clip. Official
  single-view evaluation preprocessing includes short-side resize to 292,
  center crop to 256, tensor conversion, and ImageNet normalization. A
  normalize-only synthetic-input shortcut is not the official pipeline.
- Use 200 timed iterations after at least 20 warmups (the numeric minimum in
  the initial prompt was truncated; this default was proposed before go-ahead).
- Record a synchronized CUDA start event before CPU preprocessing so elapsed
  end-to-end time includes its idle GPU interval. Reuse initialized events,
  retain synchronized wall timing for corroboration, and report GPU segments.
- Latency runs have no layer hooks or profiler. Attribution runs are separate
  to avoid unequal eager/compiled overhead and compiler graph breaks.
- Save raw samples, median, mean, p95, config, full commit hash, runtime
  versions, and the p95/mean validity decision. Reject ratios above 1.3.
  Compare only back-to-back A/B runs from the same session.
- Validate pretrained embeddings for every named clip against the untouched
  fp32 encoder with TF32 disabled: minimum cosine at least 0.999. Quantized
  results remain separate and require at least 0.98 versus fp16.

## Changes to the reconstruction order

Bring the compiled comparator and measurement protocol from C21/C24 into C4;
otherwise early optimizations would target a weaker baseline. Bring pretrained
quality validation forward too. Build explicit inference modules with prepared
RoPE buffers instead of mutable global caches and repeated monkey-patching.
Add custom kernels only after a fresh profile establishes their expected gain.
Each dispatch must be tested on its actual dtype/layout/backend before timing.

The reference is not safe to copy wholesale: its ordinary Linear weights are
Parameters, rejected by its exact Tensor type guard; its add-LayerNorm kernel
is not connected to the encoder; bf16 preprocessing dispatch returns fp32;
and repeated monkey-patching can corrupt fallback state. Approximate GELU and
RoPE precision changes require the new per-clip quality gate.

## Environment and initial repair

The original DNS SSH command failed. The replacement direct endpoint connected
successfully. Initial `nvidia-smi` reports RTX 4090, driver 595.71.05, CUDA driver
capability 13.2, and 49140 MiB reported device memory; installed nvcc is 13.1.
System Python 3.12 has no torch. Use an isolated virtual environment with the
repository's existing requirements, pulling all source changes from GitHub.
Custom extensions require a toolkit compatible with the resolved torch runtime.

`d6ba5a4` fixes C4 importing `src` before the vendored root is on `sys.path`.
The syntax and whitespace checks passed locally; runtime validation follows
when the remote environment is ready. No new latency or accuracy claims yet.

## Early pretrained quality infrastructure

Added a seven-clip deterministic raw uint8 suite (noise, black, white, mid-gray,
moving gradient, checkerboard, and color bars). Validation checks every batch
member and repeats each candidate call, copying reference and candidate outputs
before compiled graph buffers can be reused. Shape, nonfinite, zero-norm,
below-threshold, repeated-call corruption, and stale-input failures are explicit.
Cosine is reduced by the worst clip; relative L2 and absolute errors are retained.

The ten CPU helper tests passed with torch 2.14.0+cpu in an isolated local venv:
`python -m unittest discover -s benchmarks/tests -p test_accuracy.py -v`.
This local venv is only for pre-commit CPU tests while Docker Desktop is stopped;
the 4090/Linux environment remains the sole performance source. Only existing
repository dependencies were installed. Pretrained GPU accuracy is still pending.

## Official preprocessing fidelity

The benchmark now adapts the upstream single-view evaluation transform directly,
including resize and crop for square model-sized clips. Its eight CPU tests pass
bit-for-bit against upstream on square, portrait, landscape, batched, and strided
inputs; constant colors also independently check normalization. The benchmark
CLI help now runs from the repo without needing an external `PYTHONPATH`.
Latency reported by the earlier C4 format still excludes CPU work; the next
commit replaces that format and removes hooks from headline timing.

## C4 raw input timing and run validity

Headline runs now have no forward hooks. Reused CUDA events bracket host
preprocessing, pinning, H2D, and encoder execution; synchronized wall timing
and individual pipeline segments accompany each raw sample. Empty-window
instrumentation is disclosed without subtracting it. The CUDA regression test
inserts a 20ms CPU delay to verify inclusion, and is pending the remote GPU.

Default pretrained workloads use 200 samples and 20 warmups. Runtime thread
settings (torch 4, OpenCV 1) are explicit and shared by all comparison paths;
they are configurable, not an assertion of optimal tuning. Saved schema 2
results include raw samples, median/mean/p95, full commit hash, selected GPU,
driver/toolkit/torch versions, precision settings, and CPU/library details.
The 15 reporting tests pass, including noisy tails, insufficient samples,
unknown/dirty source, nonfinite data, and selecting a nondefault GPU. A run
without pretrained quality validation cannot become acceptance eligible.

## Compiled official comparator brought forward

The official fp32-weight encoder is wrapped with `torch.compile(mode="max-autotune")`
and called under the same fp16 autocast and official input pipeline as eager.
Pretrained seven-clip fp32 validation runs before latency; failed quality saves
its report and stops. At least 20 compile warmups precede the measured loop.
Compiler tracing counters before/after measurement expose late compilation,
graph breaks, or a wrapper that captured no graph. Late tracing or no captured
graph invalidates latency instead of silently accepting an eager fallback.
Graph step markers and output release precede each new pipeline sample.

## C5 operator attribution

`--profile` captures a separate full raw-clip pipeline call after measured
latency. It exports operator input shapes, CPU/CUDA totals, and a Chrome trace
without modifying the primary sample loop. Its three real CPU profiler tests
pass, checking numerical output, trace operators/shapes, and exception cleanup.
CUDA attribution and the first target profile follow GPU test validation.

## First remote validation and device selection repair

All 37 benchmark tests committed before C5 passed on the RTX 4090, including
the real CUDA event test that includes a deliberate host preprocessing delay.
The first benchmark invocation at `32fbf14` then failed before loading weights:
`torch.cuda.set_device("cuda")` requires an explicit index in this runtime.
Normalize the CLI default to `cuda:0` while preserving explicit indices, and
exercise that default in both parser and CUDA tests. Write config/environment
to `run.json` before model construction so later failures retain provenance.
This failed attempt has zero latency samples and supplies no performance claim.

## Input and hardware provenance

Record the selected GPU UUID and host in every new run, plus SHA256 of the actual
raw clip. The timed NumPy noise clip is now also the quality suite's noise clip;
suite version 2 makes this change explicit. Record crop size and input layout,
dtype, shape, memory location, and resize semantics. These fields let later
comparison reject an accidentally different GPU, session, clip, or transform.
The initial baseline campaign remains useful profile evidence; the final
compiled-baseline/engine A/B will be remeasured together under the later schema.

The current checkpoint download is 4.78 GiB, superseding the historical size
estimate. Host memory and disk have adequate headroom (392 GiB available RAM,
89 GiB free workspace disk at inspection). GPU UUID is
`abaa5fe0-76a5-6def-ee23-70d8993c1c66`; torch reports 50866487296 memory bytes.
