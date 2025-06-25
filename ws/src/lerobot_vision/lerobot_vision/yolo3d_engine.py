"""ws/src/lerobot_vision/lerobot_vision/yolo3d_engine.py"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

try:
    from openyolo3d import Detector
except ImportError:  # pragma: no cover - external dependency
    Detector = None  # type: ignore

logger = logging.getLogger(__name__)


class Yolo3DEngine:
    """Perform 3D segmentation using OpenYOLO3D."""

    def __init__(self, checkpoint_dir: str) -> None:
        if Detector is None:
            raise ImportError("openyolo3d is required")
        try:
            self.detector = Detector(checkpoint_dir=checkpoint_dir)
        except Exception as exc:  # pragma: no cover - hardware
            logger.error("Failed to load Detector: %s", exc)
            raise
        logger.debug("Yolo3DEngine initialized")

    def segment(
        self, images: List[np.ndarray], depth_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Segment objects from images and depth map."""
        if not isinstance(images, list) or not isinstance(
            depth_map, np.ndarray
        ):
            raise TypeError("invalid inputs")
        try:
            masks, labels = self.detector.predict(images, depth_map)
        except Exception as exc:  # pragma: no cover - hardware
            logger.error("Segmentation failed: %s", exc)
            raise
        return masks, labels
