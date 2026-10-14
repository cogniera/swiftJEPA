#Author : Paarth Sharma
#File Name : pipeline.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Batched raw video preprocessing using the official evaluation transform

"""The official single-view input pipeline for encoder benchmarks."""

# Keep model and transform dependencies unloaded until preprocessing is requested.
from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np
    import torch


# Construct the upstream transform once and adapt its single-clip API to a batch.
def make_preprocess(crop_size: int = 256) -> Callable[[np.ndarray], torch.Tensor]:
    """Return raw uint8 BTHWC -> contiguous CPU fp32 BCTHW preprocessing."""
    if isinstance(crop_size, bool) or not isinstance(crop_size, Integral):
        raise TypeError("crop_size must be an integer")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    import sys
    from pathlib import Path

    import numpy as np
    import torch

    vendor_path = str(Path(__file__).resolve().parents[1] / "vjepa2")
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    from evals.video_classification_frozen.utils import make_transforms

    transform = make_transforms(
        training=False, num_views_per_clip=1, crop_size=int(crop_size)
    )

    # Validate raw decoded frames before upstream resize, crop, and normalization.
    def preprocess(clip_u8: np.ndarray) -> torch.Tensor:
        if not isinstance(clip_u8, np.ndarray):
            raise TypeError("clip must be a numpy ndarray")
        if clip_u8.dtype != np.uint8:
            raise TypeError("clip must have uint8 dtype")
        if clip_u8.ndim != 5 or clip_u8.shape[-1] != 3:
            raise ValueError("clip must have shape [B,T,H,W,3]")
        if any(size == 0 for size in clip_u8.shape[:-1]):
            raise ValueError("batch, frames, height, and width must be nonempty")

        return torch.stack([transform(clip)[0] for clip in clip_u8]).contiguous()

    return preprocess
