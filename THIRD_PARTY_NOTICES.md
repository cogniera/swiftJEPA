# Third-Party Notices

This project is licensed under the MIT License (see [LICENSE](LICENSE)). It
redistributes and builds on third-party code that carries its own terms, listed
below. All upstream license files and per-file copyright headers must be kept
intact.

## 1. V-JEPA 2 (under 'vjepa2/')

A vendored copy of Meta's official V-JEPA 2 repository
(<https://github.com/facebookresearch/vjepa2>).

- **Copyright (c) Meta Platforms, Inc. and affiliates.** Some files carry the company's former attribution, **Copyright (c) Facebook, Inc. and its affiliates.**
- **License: MIT** (the full text at [`vjepa2/LICENSE`](vjepa2/LICENSE)); each source file carries the Meta (or the former Facebook) copyright header.
- **Exception: Apache License 2.0 for the files** (full text at [`vjepa2/APACHE-LICENSE`](vjepa2/APACHE-LICENSE)):
  - `vjepa2/src/datasets/utils/video/randaugment.py` - Copyright 2020 Ross Wightman; based on `timm`'s `data/auto_augment.py`.
  - `vjepa2/src/datasets/utils/video/randerase.py` - Copyright 2020 Ross Wightman; based on `timm`'s `data/random_erasing.py`.
  - `vjepa2/src/datasets/utils/worker_init_fn.py` - Copyright The Lightning AI team.

MIT and Apache-2.0 are both permissive and compatible with this project's MIT
license. Redistribution requires only that the notices above and the upstream
license files are preserved, which they are.

Pretrained V-JEPA 2 checkpoints (e.g. `vitl.pt`) are downloaded at runtime and
are **not** redistributed here; they are covered by Meta's terms for those
model weights.
