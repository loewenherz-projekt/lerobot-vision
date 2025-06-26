"""Camera interface for stereo cameras."""

from __future__ import annotations

import logging
from typing import Tuple
from pathlib import Path
import yaml

import cv2
import numpy as np


class StereoCamera:
    """Stereo camera using OpenCV VideoCapture."""

    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(5)

    def __init__(
        self,
        left_idx: int = 0,
        right_idx: int = 1,
        config_path: str | None = None,
    ) -> None:
        if config_path:
            self._load_params(config_path)
        self.left_cap = cv2.VideoCapture(left_idx)
        self.right_cap = cv2.VideoCapture(right_idx)

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return undistorted frames from both cameras."""
        ret_left, left = self.left_cap.read()
        ret_right, right = self.right_cap.read()
        if not ret_left or not ret_right:
            logging.error("Failed to read from camera")
            raise RuntimeError("Kamerafehler")
        left_ud = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right_ud = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left_ud, right_ud

    def release(self) -> None:
        """Release camera resources."""
        self.left_cap.release()
        self.right_cap.release()

    @classmethod
    def _load_params(cls, path: str) -> None:
        try:
            data = yaml.safe_load(Path(path).read_text())
            cls.camera_matrix = np.array(
                data.get("camera_matrix", cls.camera_matrix)
            )
            cls.dist_coeffs = np.array(
                data.get("dist_coeffs", cls.dist_coeffs)
            )
        except Exception as exc:  # pragma: no cover - config optional
            logging.error("Failed to load camera parameters: %s", exc)
