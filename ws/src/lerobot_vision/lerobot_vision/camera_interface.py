# File: ws/src/lerobot_vision/lerobot_vision/camera_interface.py
"""Camera interface for stereo capture."""

from typing import Tuple
import logging

try:
    import cv2
except ImportError:  # pragma: no cover - external dependency
    cv2 = None

import numpy as np

logger = logging.getLogger(__name__)


class StereoCamera:
    """Simple stereo camera wrapper around OpenCV VideoCapture."""

    def __init__(
        self,
        left_index: int = 0,
        right_index: int = 1,
        camera_matrix=None,
        dist_coeffs=None,
    ):
        if cv2 is None:
            raise RuntimeError("cv2 not available")
        self.left_cap = cv2.VideoCapture(left_index)
        self.right_cap = cv2.VideoCapture(right_index)
        self.camera_matrix = (
            np.array(camera_matrix, dtype=float)
            if camera_matrix is not None
            else np.eye(3)
        )
        self.dist_coeffs = (
            np.array(dist_coeffs, dtype=float)
            if dist_coeffs is not None
            else np.zeros((5,))
        )
        logger.debug("StereoCamera initialized")

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return undistorted left and right frames."""
        ret_l, left = self.left_cap.read()
        ret_r, right = self.right_cap.read()
        if not ret_l or not ret_r:
            logger.error("Camera read failure")
            raise RuntimeError("Kamerafehler")
        left = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left, right

    def release(self) -> None:
        """Release both camera handles."""
        try:
            self.left_cap.release()
            self.right_cap.release()
            logger.debug("StereoCamera released")
        except Exception as exc:  # pragma: no cover - hardware release
            logger.error("Failed to release cameras: %s", exc)
            raise
