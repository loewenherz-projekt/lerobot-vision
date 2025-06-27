# ws/src/lerobot_vision/lerobot_vision/camera_interface.py
"""Camera interface for stereo cameras."""

from __future__ import annotations

import logging
from typing import Tuple
from pathlib import Path
import yaml

import cv2
import numpy as np


class StereoCamera:
    """Stereo camera using OpenCV ``VideoCapture``."""

    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(5)

    def __init__(
        self,
        left_idx: int = 0,
        right_idx: int = 1,
        config_path: str | None = None,
    ) -> None:
        """Initialize the stereo camera interface.

        Args:
            left_idx: Device index of the left camera.
            right_idx: Device index of the right camera.
            config_path: Optional path to a YAML file containing camera
                calibration parameters.
        """
        if config_path:
            self._load_params(config_path)
        self.left_cap = cv2.VideoCapture(left_idx)
        self.right_cap = cv2.VideoCapture(right_idx)

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve frames from both cameras.

        Returns:
            A tuple ``(left, right)`` of undistorted images captured from the
            left and right cameras respectively.
        """
        ret_left, left = self.left_cap.read()
        ret_right, right = self.right_cap.read()
        if not ret_left or not ret_right:
            logging.error("Failed to read from camera")
            raise RuntimeError("Failed to read from camera")
        left_ud = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right_ud = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left_ud, right_ud

    def release(self) -> None:
        """Release underlying camera resources."""
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


class AsyncStereoCamera(StereoCamera):  # pragma: no cover - optional helper
    """Stereo camera that captures frames on a background thread."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frame_left: np.ndarray | None = None
        self._frame_right: np.ndarray | None = None
        self._running = True
        import threading

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:  # pragma: no cover - runtime loop
        while self._running:
            try:
                self._frame_left, self._frame_right = super().get_frames()
            except Exception:
                continue

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        if self._frame_left is None or self._frame_right is None:
            return super().get_frames()
        return self._frame_left.copy(), self._frame_right.copy()

    def release(self) -> None:  # pragma: no cover
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        super().release()
