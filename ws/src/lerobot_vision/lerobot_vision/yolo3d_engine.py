# File: ws/src/lerobot_vision/lerobot_vision/yolo3d_engine.py
"""YOLO3D segmentation engine."""

from typing import List, Tuple
import logging

import numpy as np

try:
    from openyolo3d import Yolo3D
except ImportError:  # pragma: no cover - external dependency
    Yolo3D = None
    logging.warning("openyolo3d package not available")

logger = logging.getLogger(__name__)


class Yolo3DEngine:
    """Wrapper for the OpenYOLO3D model."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        if Yolo3D is None:
            self.model = None
            logger.warning("Yolo3D model not loaded")
        else:
            try:
                self.model = Yolo3D(checkpoint_dir=checkpoint_dir)
                logger.debug("Yolo3D initialized from %s", checkpoint_dir)
            except Exception as exc:
                logger.error("Failed to load Yolo3D: %s", exc)
                self.model = None
                raise

    def segment(
        self, images: List[np.ndarray], depth_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Segment images using the 3D model and depth map."""
        if self.model is None:
            raise RuntimeError("Yolo3D model not initialized")
        if not isinstance(images, list) or not isinstance(
            depth_map, np.ndarray
        ):
            raise TypeError("Invalid input types")
        try:
            masks, labels = self.model.segment(images, depth_map)
            return masks, labels
        except Exception as exc:
            logger.error("Segmentation failed: %s", exc)
            raise
