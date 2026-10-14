#Author : Paarth Sharma
#File Name : accuracy.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Deterministic raw clip suite and per clip encoder accuracy gates.

from collections.abc import Callable, Iterator

import numpy as np
import torch


# Generate clips one at a time so full resolution validation has bounded memory.
def iter_test_clips(
    frames: int = 16,
    batch: int = 1,
    crop_size: int = 256,
    seed: int = 0,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield deterministic CPU uint8 clips in contiguous BTHWC layout."""
    if min(frames, batch, crop_size) < 1:
        raise ValueError("frames, batch and crop_size must be positive")
    shape = (batch, frames, crop_size, crop_size, 3)
    yield "noise", np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    for name, value in (("black", 0), ("white", 255), ("mid_gray", 128)):
        yield name, torch.full(shape, value, dtype=torch.uint8).numpy()

    # Shift spatial patterns through time to exercise both patch and tubelet input.
    time = torch.arange(frames).view(frames, 1, 1)
    row = torch.arange(crop_size).view(1, crop_size, 1)
    column = torch.arange(crop_size).view(1, 1, crop_size)
    gradient = (column * 5 + row * 3 + time * 13) % 256
    gradient_rgb = torch.stack(tuple((gradient + phase) % 256 for phase in (0, 85, 170)), dim=-1)
    yield "moving_gradient", gradient_rgb.to(torch.uint8).unsqueeze(0).expand(shape).contiguous().numpy()
    del gradient, gradient_rgb

    tile_size = max(1, crop_size // 16)
    checker = ((column + time * 2) // tile_size + (row + time * 3) // tile_size) % 2
    checker_rgb = torch.stack((checker * 255, (1 - checker) * 255, checker * 127), dim=-1)
    yield "moving_checkerboard", checker_rgb.to(torch.uint8).unsqueeze(0).expand(shape).contiguous().numpy()
    del checker, checker_rgb

    palette = torch.tensor(
        [[255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0],
         [255, 0, 255], [255, 0, 0], [0, 0, 255], [0, 0, 0]],
        dtype=torch.uint8,
    )
    indices = ((column + time * max(1, crop_size // 16)) // max(1, crop_size // 8)) % 8
    color_bars = palette[indices.expand(frames, crop_size, crop_size)]
    yield "moving_color_bars", color_bars.unsqueeze(0).expand(shape).contiguous().numpy()


# Compare flattened embeddings per batch member so one bad clip cannot be hidden.
def compare_embeddings(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    minimum_cosine: float = 0.999,
) -> dict:
    """Return JSON serializable metrics; invalid outputs fail without NaN fields."""
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must lie between minus one and one")
    result = {
        "passed": False,
        "minimum_cosine_required": minimum_cosine,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "minimum_cosine": None,
        "per_sample_cosine": [],
        "relative_l2": None,
        "per_sample_relative_l2": [],
        "maximum_absolute_error": None,
    }
    if reference.shape != candidate.shape:
        return {**result, "failure": "shape_mismatch"}
    if reference.ndim < 1 or reference.numel() == 0:
        return {**result, "failure": "empty_embeddings"}

    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float64)
    reference_finite = bool(torch.isfinite(reference).all())
    candidate_finite = bool(torch.isfinite(candidate).all())
    result.update(reference_finite=reference_finite, candidate_finite=candidate_finite)
    if not reference_finite or not candidate_finite:
        return {**result, "failure": "nonfinite_embeddings"}

    reference = reference.reshape(reference.shape[0], -1)
    candidate = candidate.reshape(candidate.shape[0], -1)
    reference_norm = torch.linalg.vector_norm(reference, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=1)
    if bool((reference_norm == 0).any()) or bool((candidate_norm == 0).any()):
        return {**result, "failure": "zero_norm_embeddings"}
    cosine = ((reference * candidate).sum(dim=1) / (reference_norm * candidate_norm)).clamp(-1, 1)
    error = candidate - reference
    relative_l2 = torch.linalg.vector_norm(error, dim=1) / reference_norm
    result.update(
        minimum_cosine=float(cosine.min()),
        per_sample_cosine=cosine.tolist(),
        relative_l2=float(relative_l2.max()),
        per_sample_relative_l2=relative_l2.tolist(),
        maximum_absolute_error=float(error.abs().max()),
        passed=bool((cosine >= minimum_cosine).all()),
    )
    if not result["passed"]:
        result["failure"] = "cosine_below_threshold"
    return result


# Snapshot outputs before another invocation can reuse compiled CUDA graph buffers.
def _snapshot(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("encoder must return an embedding tensor")
    return output.detach().to(device="cpu", dtype=torch.float32, copy=True)


# Check repeated inputs and changed clips against the same untouched fp32 encoder.
def validate_encoder(
    reference_encoder: Callable[[torch.Tensor], torch.Tensor],
    candidate_encoder: Callable[[torch.Tensor], torch.Tensor],
    preprocess: Callable[[np.ndarray], torch.Tensor],
    device: str | torch.device,
    dtype: torch.dtype,
    frames: int = 16,
    batch: int = 1,
    crop_size: int = 256,
    minimum_cosine: float = 0.999,
    seed: int = 0,
) -> dict:
    """Validate eval-mode encoders; preprocess returns fp32 input on device.

    The caller provides pretrained fp32 weights and disables TF32 for the
    reference. No weights are converted or copied by this helper.
    """
    device = torch.device(device)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
    rows = []
    with torch.inference_mode():
        for name, clip in iter_test_clips(frames, batch, crop_size, seed):
            encoder_input = preprocess(clip)
            device_matches = encoder_input.device.type == device.type and (
                device.index is None or encoder_input.device.index == device.index
            )
            if not device_matches or encoder_input.dtype != torch.float32:
                raise ValueError("preprocess must return fp32 input on the requested device")
            with torch.autocast(device_type=device.type, enabled=False):
                reference = _snapshot(reference_encoder(encoder_input))
            measurements = []
            for _ in range(2):
                if mark_step is not None and device.type == "cuda":
                    mark_step()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                    candidate = _snapshot(candidate_encoder(encoder_input))
                measurements.append(compare_embeddings(reference, candidate, minimum_cosine))
                del candidate
            rows.append({"name": name, **measurements[0], "repeat_input": measurements[1]})
            del clip, encoder_input, reference, measurements
    valid_cosines = [metrics["minimum_cosine"] for row in rows
                     for metrics in (row, row["repeat_input"]) if metrics["minimum_cosine"] is not None]
    return {
        "passed": all(row["passed"] and row["repeat_input"]["passed"] for row in rows),
        "minimum_cosine": min(valid_cosines) if valid_cosines else None,
        "minimum_cosine_required": minimum_cosine,
        "reference_dtype": "float32",
        "candidate_autocast_dtype": str(dtype).removeprefix("torch."),
        "clip_layout": "BTHWC uint8",
        "frames": frames,
        "batch": batch,
        "crop_size": crop_size,
        "clip_seed": seed,
        "clip_suite_version": 2,
        "clips": rows,
    }
