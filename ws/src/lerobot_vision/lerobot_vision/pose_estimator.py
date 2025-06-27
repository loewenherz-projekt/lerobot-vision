# ws/src/lerobot_vision/lerobot_vision/pose_estimator.py
"""Pose estimation wrapper using DOPE."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

try:
    from isaac_ros_pose_estimation.dope import (  # pragma: no cover
        DOPEEstimator,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    DOPEEstimator = None  # type: ignore
    logging.error("DOPE import failed: %s", exc)


class PoseEstimator:
    """Estimate 6-DoF poses for known objects."""

    def __init__(self) -> None:
        if DOPEEstimator is not None:
            try:
                self.estimator = DOPEEstimator()
            except Exception as exc:  # pragma: no cover - runtime path
                logging.error("DOPE init failed: %s", exc)
                self.estimator = None
        else:
            self.estimator = None

    def estimate(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Estimate object poses from an RGB image.

        Args:
            image: The input RGB image.

        Returns:
            A list of ``(position, orientation)`` tuples where ``position`` is
            a ``(x, y, z)`` array and ``orientation`` is a quaternion
            ``(x, y, z, w)``.
        """
        if self.estimator is None:
            return []
        try:
            return self.estimator.infer(image)
        except Exception as exc:  # pragma: no cover - runtime path
            logging.error("Pose estimation failed: %s", exc)
            return []
