# ws/src/lerobot_vision/lerobot_vision/object_localizer.py
"""Fuse masks, depth and poses to obtain 3D object positions."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def localize_objects(
    masks: List[np.ndarray],
    depth: np.ndarray,
    camera_matrix: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """Compute 3D coordinates from masks and depth.

    Args:
        masks: List of binary segmentation masks.
        depth: Depth map corresponding to the masks.
        camera_matrix: Intrinsic camera matrix.

    Returns:
        List of tuples ``(label, xyz)`` representing object labels and their
        3D positions in camera coordinates.
    """
    results: List[Tuple[str, np.ndarray]] = []
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    for idx, mask in enumerate(masks):
        if mask is None or mask.size == 0:
            continue
        ys, xs = np.nonzero(mask > 0)
        if len(xs) == 0:
            continue
        z_vals = depth[ys, xs]
        z = float(np.median(z_vals))
        u = float(np.median(xs))
        v = float(np.median(ys))
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        results.append((f"obj_{idx}", np.array([x, y, z], dtype=float)))
    return results
