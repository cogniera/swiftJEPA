#Author : Paarth Sharma
#File Name : README.md
#Project Name : swiftJEPA
#Creation Date : 16th September 2026
#Modification Date : 14th October 2026
#Description : Benchmark input contract measurement protocol and result interpretation.

# V-JEPA 2 inference benchmarks

The workload starts with a pageable NumPy uint8 clip in BTHWC layout and ends
with device embeddings. Defaults are pretrained ViT-L, 16 frames, a 256px
encoder crop, batch 1, fp16 autocast, 200 measured iterations, and 20 warmups.

The upstream single-view evaluation pipeline resizes the short side to 292,
center crops to 256, converts the clip, and applies ImageNet normalization.
Host preprocessing, pinning, transfer, and encoder execution are timed.

Run from the Docker environment:

~~~bash
docker compose run --rm dev python benchmarks/benchmark_vjepa2.py \
  --session-id baseline-session-1 --out benchmarks/results/official_eager_1
docker compose run --rm dev python benchmarks/benchmark_vjepa2.py --compile \
  --session-id baseline-session-1 --out benchmarks/results/official_compiled_1
~~~

On a disposable Linux host, install the existing requirements in an isolated
venv and use its Python interpreter. Set TORCH_HOME to an ignored checkpoint
cache. Pull source changes from GitHub before running them.

CPU-only helper tests use the same package versions, with a CPU torch build:

~~~bash
python -m unittest discover -s benchmarks/tests -v
~~~

The CUDA timing test verifies that a deliberate CPU preprocessing delay
appears in the CUDA-event end-to-end interval; it is skipped on CPU hosts.

The synchronized start event executes before host preprocessing. CUDA-event
elapsed time therefore includes the GPU idle interval while the CPU prepares
the clip. Synchronized wall time corroborates that metric. Event objects are
initialized and reused; the measured empty-window overhead is recorded and
kept in the reported timing. Attribution belongs in a separate profiling pass.

The default CPU settings are four torch intra-op threads and one OpenCV thread.
Keep these settings identical across comparisons and retain them in metadata.
Treat them as configurable runtime settings, not a claim of optimal CPU tuning.

Each run creates a new output directory:

| File | Contents |
|---|---|
| timings_per_iteration.csv | Raw end-to-end, wall, CPU preprocessing, transfer, encoder and GPU samples |
| summary.json | Median, mean, p95, source hash, runtime versions, configuration and validity |
| accuracy.json | Every pretrained clip checked against fp32, including repeat-input checks |

The compiled comparator wraps the untouched fp32-weight encoder with
torch.compile(mode="max-autotune") and uses the same fp16 autocast and official
input pipeline. At least 20 warmups precede its measured loop. Compiler counters
are saved before and after measurement; new tracing activity or no captured
graphs invalidates the run. Preserve stderr diagnostics with retained artifacts.

Use end_to_end_ms.median_ms under stages as the primary latency. GPU totals
and encoder timing explain its components. A valid latency run needs at least
200 samples, known-clean tracked source, and end-to-end p95/mean at most 1.3.
Invalid runs retain their artifacts and exit with status 2. Quality must also
pass before a run becomes eligible for a target claim.

The deterministic quality suite covers noise, constant clips, moving gradients,
checkerboards, and color bars. Each candidate output is checked per clip
against untouched fp32 output with TF32 disabled, including repeated calls and
changed input. The lossless gate is minimum cosine at least 0.999.

Pretrained accuracy runs before latency. A failed gate saves accuracy.json and
stops. --skip-accuracy is available for diagnostics and cannot establish target
acceptance. Compilation, quality comparisons, and output snapshots are outside
the latency interval.

Only back-to-back A/B comparisons sharing the same session, input seed,
preprocessing, weights, runtime, and workload support a speedup claim. Old
reference results are historical context. Random weights (--random-weights)
and short runs are useful diagnostics and cannot establish target acceptance.

Results are ignored by default. Add only reviewed run artifacts explicitly;
checkpoints, result archives, and kernels_for_reuse are never committed.
