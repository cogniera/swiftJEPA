#Author : Paarth Sharma
#File Name : profiling.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Separate untimed operator attribution for the raw clip inference pipeline.

# Keep profiler instrumentation outside the latency harness and preserve the callable's context.
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function


# Profile one completed pipeline call and export detailed traces plus an operator summary.
def profile_callable(
    run_once: Callable[[], Any], out_dir: str | Path, use_cuda: bool = True
) -> dict[str, Any]:
    if not callable(run_once):
        raise TypeError("run_once must be callable")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an available CUDA device")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=True) as recorder:
        with record_function("raw_clip_to_embeddings"):
            run_once()
        if use_cuda:
            torch.cuda.synchronize()

    # Newer PyTorch names accelerator time generically; retain the older CUDA alias if needed.
    operators = recorder.key_averages(group_by_input_shape=True)
    sort_by = "cpu_time_total"
    if use_cuda:
        sort_by = "device_time_total"
        if operators and not hasattr(operators[0], sort_by):
            sort_by = "cuda_time_total"

    trace_path = out_dir / "trace.json"
    top_ops_path = out_dir / "top_ops.txt"
    recorder.export_chrome_trace(str(trace_path))
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise RuntimeError("Profiler did not export a nonempty trace.json")
    top_ops_path.write_text(
        operators.table(sort_by=sort_by, row_limit=40) + "\n", encoding="utf-8"
    )

    return {
        "profiled_iterations": 1,
        "separate_from_latency": True,
        "activities": ["CPU", "CUDA"] if use_cuda else ["CPU"],
        "record_shapes": True,
        "sort_by": sort_by,
        "operator_count": len(operators),
        "trace": str(trace_path),
        "top_ops": str(top_ops_path),
    }
