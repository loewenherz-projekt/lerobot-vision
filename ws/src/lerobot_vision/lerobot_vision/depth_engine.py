"""Depth computation engine."""

from typing import Any
import logging
import numpy as np

try:
    from stereoanywhere import StereoAnywhere
except Exception as exc:  # pragma: no cover - optional dependency
    StereoAnywhere = None  # type: ignore
    logging.error("StereoAnywhere import failed: %s", exc)


class DepthEngine:
    """Compute depth using Stereo Anywhere."""

    def __init__(self, pretrained: bool = True) -> None:
        if StereoAnywhere is None:
            raise ImportError("StereoAnywhere not available")
        self.engine = StereoAnywhere(pretrained=pretrained)

    def compute_depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute depth map."""
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            logging.error("Inputs must be numpy arrays")
            raise TypeError("Invalid input type")
        try:
            depth = self.engine.infer(left, right)
        except Exception as exc:
            logging.error("Depth inference failed: %s", exc)
            raise
        return depth
