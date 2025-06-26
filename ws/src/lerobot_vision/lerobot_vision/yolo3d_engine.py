"""YOLOv3 3D engine wrapper."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

try:
    from openyolo3d.models.open_yolo3d import OpenYolo3D
except Exception as exc:  # pragma: no cover - external dep
    OpenYolo3D = None  # type: ignore
    logging.error("OpenYolo3D import failed: %s", exc)


class Yolo3DEngine:
    """YOLO3D wrapper class."""

    def __init__(self, checkpoint_dir: str) -> None:
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(checkpoint_dir)
        if OpenYolo3D is None:
            raise RuntimeError("OpenYolo3D unavailable")
        self.engine = OpenYolo3D(checkpoint=checkpoint_dir)

    def segment(
        self, images: List[np.ndarray], depth_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Segment objects using images and depth map."""
        dmap = depth_map
        if not isinstance(images, list) or not isinstance(dmap, np.ndarray):
            raise ValueError("Invalid inputs")
        try:
            return self.engine.segment(images, depth_map)
        except Exception as exc:  # pragma: no cover
            logging.error("Segmentation failed: %s", exc)
            raise
