# ws/src/lerobot_vision/lerobot_vision/image_rectifier.py
"""Rectify stereo images using calibration parameters."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class ImageRectifier:
    """Apply rectification maps to image pairs."""

    def __init__(
        self,
        left_camera_matrix: np.ndarray,
        left_dist_coeffs: np.ndarray,
        right_camera_matrix: np.ndarray,
        right_dist_coeffs: np.ndarray,
        image_size: Tuple[int, int],
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> None:
        self.M1 = left_camera_matrix
        self.D1 = left_dist_coeffs
        self.M2 = right_camera_matrix
        self.D2 = right_dist_coeffs
        R = np.eye(3) if rotation is None else rotation
        T = np.array([1.0, 0.0, 0.0]) if translation is None else translation
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.M1,
            self.D1,
            self.M2,
            self.D2,
            image_size,
            R,
            T,
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.M1, self.D1, R1, P1, image_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.M2, self.D2, R2, P2, image_size, cv2.CV_32FC1
        )

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_r = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return left_r, right_r
