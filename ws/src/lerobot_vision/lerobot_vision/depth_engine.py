# File: ws/src/lerobot_vision/lerobot_vision/depth_engine.py
"""Stereo depth computation engine."""

from typing import Optional
import logging

import numpy as np

try:
    from stereoanywhere import StereoAnywhere
except ImportError:  # pragma: no cover - external dependency
    StereoAnywhere = None
    logging.warning("StereoAnywhere not available")

logger = logging.getLogger(__name__)


class DepthEngine:
    """Wrapper for StereoAnywhere depth estimation."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if StereoAnywhere is None:
            self.model = None
            logger.warning("StereoAnywhere model not loaded")
        else:
            try:
                self.model = StereoAnywhere(pretrained=True)
                logger.debug("StereoAnywhere model initialized")
            except Exception as exc:
                logger.error("Failed to load StereoAnywhere: %s", exc)
                self.model = None
                raise

    def compute_depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute depth map from stereo pair."""
        if not isinstance(left, np.ndarray) or not isinstance(
            right, np.ndarray
        ):
            raise TypeError("Inputs must be numpy arrays")
        if self.model is None:
            raise RuntimeError("Depth model not initialized")
        try:
            depth = self.model.compute(left, right)
            return depth
        except Exception as exc:
            logger.error("Depth computation failed: %s", exc)
            raise
