"""Camera interface module."""

from typing import Tuple
import logging
import cv2
import numpy as np


class StereoCamera:
    """Interface for stereo cameras."""

    def __init__(self, left_idx: int = 0, right_idx: int = 1) -> None:
        """Initialize camera streams."""
        self.left_cap = cv2.VideoCapture(left_idx)
        self.right_cap = cv2.VideoCapture(right_idx)
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return undistorted frames from both cameras."""
        success_left, left = self.left_cap.read()
        success_right, right = self.right_cap.read()
        if not success_left or not success_right:
            logging.error("Failed to read from cameras")
            raise RuntimeError("Kamerafehler")
        left = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left, right

    def release(self) -> None:
        """Release camera streams."""
        self.left_cap.release()
        self.right_cap.release()
