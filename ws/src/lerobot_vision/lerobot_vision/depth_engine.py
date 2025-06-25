"""ws/src/lerobot_vision/lerobot_vision/depth_engine.py"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from stereoanywhere import StereoAnywhere
except ImportError:  # pragma: no cover - external dependency
    StereoAnywhere = None  # type: ignore

logger = logging.getLogger(__name__)


class DepthEngine:
    """Compute depth maps using StereoAnywhere model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if StereoAnywhere is None:
            raise ImportError("stereoanywhere is required")
        try:
            self.model = StereoAnywhere(pretrained=True, model_path=model_path)
        except Exception as exc:  # pragma: no cover - hardware
            logger.error("Failed to initialize StereoAnywhere: %s", exc)
            raise
        logger.debug("DepthEngine initialized")

    def compute_depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute a depth map from stereo frames."""
        if not isinstance(left, np.ndarray) or not isinstance(
            right, np.ndarray
        ):
            raise TypeError("inputs must be numpy arrays")
        try:
            depth = self.model.compute(left, right)
        except Exception as exc:  # pragma: no cover - hardware
            logger.error("Depth computation failed: %s", exc)
            raise
        return depth
