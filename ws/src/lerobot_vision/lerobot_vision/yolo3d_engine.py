"""YOLO3D Engine module."""

from typing import List, Tuple
import logging
import os
import numpy as np

try:
    from openyolo3d import OpenYolo3D
except Exception as exc:  # pragma: no cover - optional dependency
    OpenYolo3D = None  # type: ignore
    logging.error("OpenYolo3D import failed: %s", exc)


class Yolo3DEngine:
    """YOLO3D engine wrapper."""

    def __init__(self, checkpoint_dir: str) -> None:
        if OpenYolo3D is None:
            raise ImportError("OpenYolo3D not available")
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError("Checkpoint directory missing")
        self.engine = OpenYolo3D(checkpoint=checkpoint_dir)

    def segment(
        self, images: List[np.ndarray], depth_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Run segmentation."""
        if not isinstance(depth_map, np.ndarray):
            logging.error("Depth map must be numpy array")
            raise TypeError("Invalid depth map")
        try:
            masks, labels = self.engine.infer(images, depth_map)
        except Exception as exc:
            logging.error("YOLO3D segmentation failed: %s", exc)
            raise
        return masks, labels
