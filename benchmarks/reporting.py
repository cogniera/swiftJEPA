#Author : Paarth Sharma
#File Name : reporting.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Latency statistics validity gates and reproducible runtime metadata

"""Reporting helpers shared by benchmark runs and their CPU tests."""

# Keep numerical and runtime dependencies lazy for command-line validation.
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
import subprocess
import sys


# Reject invalid timing data before computing any published statistics.
def summarize_samples(samples_ms):
    """Return count, mean, median, and p95 for a positive finite sample vector."""
    import numpy as np

    try:
        samples = np.asarray(samples_ms, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("samples must be a numeric vector") from error
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a nonempty vector")
    if not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("samples must be finite and strictly positive")

    with np.errstate(over="ignore", invalid="ignore"):
        statistics = (np.mean(samples), np.median(samples), np.percentile(samples, 95))
    if not all(math.isfinite(float(value)) for value in statistics):
        # Scaling prevents finite positive inputs from overflowing NumPy reductions.
        scale = float(np.max(samples))
        scaled = samples / scale
        statistics = (
            float(np.mean(scaled)) * scale,
            float(np.median(scaled)) * scale,
            float(np.percentile(scaled, 95)) * scale,
        )

    return {
        "count": int(samples.size),
        "mean_ms": float(statistics[0]),
        "median_ms": float(statistics[1]),
        "p95_ms": float(statistics[2]),
    }


# A result is usable only with enough samples, a stable tail, and clean source.
def run_validity(
    end_to_end_samples,
    minimum_iters=200,
    maximum_p95_mean=1.3,
    source_dirty=False,
):
    """Return a JSON-safe validity decision with every reason for rejection."""
    if isinstance(minimum_iters, bool) or not isinstance(minimum_iters, Integral):
        raise TypeError("minimum_iters must be an integer")
    if minimum_iters <= 0:
        raise ValueError("minimum_iters must be positive")
    if not math.isfinite(maximum_p95_mean) or maximum_p95_mean <= 0:
        raise ValueError("maximum_p95_mean must be finite and positive")

    reasons = []
    ratio = None
    try:
        samples = list(end_to_end_samples)
    except TypeError:
        samples = []
        reasons.append("samples must be an iterable of timings")
    sample_count = len(samples)
    try:
        statistics = summarize_samples(samples)
        ratio = statistics["p95_ms"] / statistics["mean_ms"]
    except ValueError as error:
        reasons.append(str(error))

    if sample_count < minimum_iters:
        reasons.append(f"requires at least {minimum_iters} measured iterations")
    if ratio is not None and ratio > maximum_p95_mean:
        reasons.append(f"p95 to mean ratio exceeds {maximum_p95_mean:g}")
    if source_dirty is True:
        reasons.append("tracked source has uncommitted changes")
    elif source_dirty is not False:
        reasons.append("tracked source cleanliness is unavailable")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "sample_count": sample_count,
        "p95_mean_ratio": ratio,
        "minimum_iters": int(minimum_iters),
        "maximum_p95_mean": float(maximum_p95_mean),
        "source_dirty": source_dirty if isinstance(source_dirty, bool) else None,
    }


# Run only explicitly named metadata commands and bound their runtime.
def _command_metadata(command, root):
    try:
        result = subprocess.run(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "reason": "unavailable",
            "error_type": type(error).__name__,
        }
    if result.returncode != 0:
        return {"available": False, "reason": "unavailable", "returncode": int(result.returncode)}
    return {"available": True, "output": result.stdout.strip()}


# Query distributions without importing the model stack or dumping environment variables.
def _package_versions():
    from importlib import metadata

    distributions = {
        "torch": ("torch",),
        "torchvision": ("torchvision",),
        "timm": ("timm",),
        "numpy": ("numpy",),
        "einops": ("einops",),
        "opencv": (
            "opencv-python-headless",
            "opencv-python",
            "opencv-contrib-python-headless",
            "opencv-contrib-python",
        ),
    }
    versions = {}
    for name, candidates in distributions.items():
        versions[name] = "unavailable"
        for distribution in candidates:
            try:
                versions[name] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


# Record the selected GPU and explicit precision/thread settings used by this process.
def collect_environment(root, device):
    """Collect source identity and runtime facts without requiring a working GPU."""
    root = Path(root).resolve()
    revision = _command_metadata(["git", "rev-parse", "HEAD"], root)
    status = _command_metadata(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    environment = {
        "commit": revision["output"] if revision["available"] else None,
        "tracked_dirty": bool(status["output"]) if status["available"] else None,
        "git": {"revision": revision, "tracked_status": status},
        "python": sys.version,
        "device": str(device),
        "packages": _package_versions(),
        "nvidia_smi": _command_metadata(["nvidia-smi"], root),
        "nvcc": _command_metadata(["nvcc", "--version"], root),
        "cpu_threads": {
            "torch": "unavailable",
            "torch_interop": "unavailable",
            "opencv": "unavailable",
        },
        "tf32": {"matmul": "unavailable", "cudnn": "unavailable"},
        "gpu": {"available": False, "reason": "unavailable"},
        "torch_cuda_version": "unavailable",
    }

    try:
        import torch

        environment["torch_runtime_version"] = str(torch.__version__)
        environment["torch_cuda_version"] = torch.version.cuda
        environment["cpu_threads"]["torch"] = int(torch.get_num_threads())
        environment["cpu_threads"]["torch_interop"] = int(torch.get_num_interop_threads())
        environment["tf32"]["matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        environment["tf32"]["cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        environment["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        environment["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        environment["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and torch.cuda.is_available():
            index = selected_device.index
            if index is None:
                index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            environment["gpu"] = {
                "available": True,
                "index": int(index),
                "name": properties.name,
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory_bytes": int(properties.total_memory),
                "multiprocessor_count": int(properties.multi_processor_count),
            }
    except Exception as error:
        environment["torch_metadata"] = {
            "available": False,
            "reason": "unavailable",
            "error_type": type(error).__name__,
        }

    try:
        import cv2

        environment["opencv_runtime_version"] = str(cv2.__version__)
        environment["cpu_threads"]["opencv"] = int(cv2.getNumThreads())
    except Exception as error:
        environment["opencv_metadata"] = {
            "available": False,
            "reason": "unavailable",
            "error_type": type(error).__name__,
        }
    return environment
