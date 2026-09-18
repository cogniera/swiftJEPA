# V-JEPA 2 Inference Benchmark

Times the official PyTorch V-JEPA 2 encoder end-to-end

## Setup

- `.venv/` - Python 3.12, PyTorch 2.13 + CUDA 12.6, timm, einops, numpy, etc.
- `vjepa2/` - official repo clone. Note: its `src/hub/backbones.py` at HEAD
  points `VJEPA_BASE_URL` at `localhost:8300`. The benchmark script overrides
  it to the public `https://dl.fbaipublicfiles.com/vjepa2` automatically.

## Run it

```powershell
.venv\Scripts\activate
python benchmarks\benchmark_vjepa2.py                       
python benchmarks\benchmark_vjepa2.py --frames 32 --iters 50
python benchmarks\benchmark_vjepa2.py --dtype fp32          
python benchmarks\benchmark_vjepa2.py --pretrained          
```

Key flags: `--model {vitl,vith,vitg,vitg-384}`, `--frames N`, `--batch N`, `--dtype {fp32,fp16,bf16}`, `--iters`, `--warmup`.

**6 GB VRAM guidance (RTX 4050):** `vitl` at 16–32 frames fp16 is comfortable.
`vith`/`vitg` in fp16 may fit at 16 frames but will be tight and fp32 giant will
run out of memory .

Weights are irrelevant to latency, so the default is random init.
Use `--pretrained` only if you also want to sanity-check outputs.

## Outputs 

| File | Contents |
|---|---|
| `timings_per_layer.csv` | ms per stage per iteration, metrics are mean/p50/p95 |
| `summary.json` | config, totals, throughput, per-stage means |
