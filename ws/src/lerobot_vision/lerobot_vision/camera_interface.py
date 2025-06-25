"""ws/src/lerobot_vision/lerobot_vision/camera_interface.py"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class StereoCamera:
    """Stereo camera using OpenCV for frame capture and undistortion."""

    def __init__(
        self,
        left_index: int = 0,
        right_index: int = 1,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        self.left_cap = cv2.VideoCapture(left_index)
        self.right_cap = cv2.VideoCapture(right_index)
        self.camera_matrix = (
            camera_matrix if camera_matrix is not None else np.eye(3)
        )
        self.dist_coeffs = (
            dist_coeffs if dist_coeffs is not None else np.zeros(5)
        )
        logger.debug("StereoCamera initialized")

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return undistorted left and right frames.

        Raises:
            RuntimeError: If either camera fails to read.
        """
        ret_l, left = self.left_cap.read()
        ret_r, right = self.right_cap.read()
        if not ret_l or not ret_r:
            logger.error("Failed to read from cameras")
            raise RuntimeError("Kamerafehler")
        left = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left, right

    def release(self) -> None:
        """Release both cameras."""
        self.left_cap.release()
        self.right_cap.release()
        logger.debug("StereoCamera released")
