"""Depth engine wrapper around Stereo Anywhere."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from stereoanywhere.models.stereo_anywhere import StereoAnywhere
except Exception as exc:  # pragma: no cover - external dep may not exist
    StereoAnywhere = Any  # type: ignore
    logging.error("StereoAnywhere import failed: %s", exc)


class DepthEngine:
    """Compute depth from stereo images."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Create engine optionally loading a custom model."""
        self.model_path = model_path
        if isinstance(StereoAnywhere, type):
            try:
                self.engine = StereoAnywhere(pretrained=True)
            except Exception as exc:  # pragma: no cover - runtime path
                self.engine = None
                logging.error("StereoAnywhere init failed: %s", exc)
        else:
            self.engine = None
            logging.error("StereoAnywhere is unavailable")

    def compute_depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute depth map."""
        if (
            self.engine is None
            or not isinstance(left, np.ndarray)
            or not isinstance(right, np.ndarray)
        ):
            logging.error("Invalid input to compute_depth")
            raise ValueError("Invalid input")
        try:
            return self.engine.infer(left, right)
        except Exception as exc:  # pragma: no cover - runtime path
            logging.error("Depth computation failed: %s", exc)
            raise
