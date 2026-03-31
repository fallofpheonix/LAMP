"""Patch dataset utilities for training path-prior segmentation models.

Provides :class:`PatchDataset` and :func:`extract_patches` for slicing
image/mask pairs into fixed-size training patches with configurable
stride.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatchDataset:
    """Stacked image patches (``x``) and corresponding mask patches (``y``)."""

    x: np.ndarray
    y: np.ndarray


def extract_patches(image: np.ndarray, mask: np.ndarray, patch: int = 128, stride: int = 64) -> PatchDataset:
    """Extract fixed-size patches from *image* and *mask* with the given *stride*.

    Patches that do not fit exactly (border remainder) are skipped.  Returns an
    empty :class:`PatchDataset` when no complete patches can be extracted.
    """
    rows, cols = image.shape[-2], image.shape[-1]
    x_list = []
    y_list = []

    for r in range(0, max(rows - patch + 1, 1), stride):
        for c in range(0, max(cols - patch + 1, 1), stride):
            x_patch = image[..., r : r + patch, c : c + patch]
            y_patch = mask[r : r + patch, c : c + patch]
            if x_patch.shape[-2:] != (patch, patch) or y_patch.shape != (patch, patch):
                continue
            x_list.append(x_patch)
            y_list.append(y_patch)

    if not x_list:
        return PatchDataset(x=np.empty((0,)), y=np.empty((0,)))

    return PatchDataset(x=np.stack(x_list), y=np.stack(y_list))
