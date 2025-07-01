# ws/src/lerobot_vision/lerobot_vision/stereo_calibrator.py
"""Utility to calibrate a stereo camera setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


class StereoCalibrator:
    """Perform stereo calibration from image pairs."""

    def __init__(
        self, board_size: Tuple[int, int] = (7, 6), square_size: float = 1.0
    ) -> None:  # pragma: no cover - unused in tests
        self.board_size = board_size
        self.square_size = square_size
        self.objpoints: List[np.ndarray] = []
        self.left_points: List[np.ndarray] = []
        self.right_points: List[np.ndarray] = []

    def add_corners(
        self, left: np.ndarray, right: np.ndarray
    ) -> bool:  # pragma: no cover - heavy calibration
        """Detect chessboard corners and store them."""
        pattern_size = self.board_size
        ret_l, corners_l = cv2.findChessboardCorners(left, pattern_size)
        ret_r, corners_r = cv2.findChessboardCorners(right, pattern_size)
        if not ret_l or not ret_r:
            return False
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(
            cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
            corners_l,
            (11, 11),
            (-1, -1),
            term,
        )
        corners_r = cv2.cornerSubPix(
            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
            corners_r,
            (11, 11),
            (-1, -1),
            term,
        )
        objp = np.zeros((np.prod(pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        objp *= self.square_size
        self.objpoints.append(objp)
        self.left_points.append(corners_l)
        self.right_points.append(corners_r)
        return True

    def calibrate(
        self, image_size: Tuple[int, int], return_errors: bool = False
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[float, float]] | None,
    ]:
        """Run stereo calibration and return intrinsics and extrinsics."""
        if not self.objpoints:
            raise RuntimeError("No corners accumulated")

        ret_l, m1, d1, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.left_points, image_size, None, None
        )
        ret_r, m2, d2, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.right_points, image_size, None, None
        )
        if not ret_l or not ret_r:
            raise RuntimeError("Calibration failed")

        ret, _, _, r, t, _, _ = cv2.stereoCalibrate(
            # pragma: no cover - heavy calibration
            self.objpoints,
            self.left_points,
            self.right_points,
            m1,
            d1,
            m2,
            d2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        if not ret:  # pragma: no cover - runtime
            raise RuntimeError("Stereo calibration failed")

        errors = None
        if return_errors:
            errors = []
            for i, objp in enumerate(self.objpoints):
                proj_l, _ = cv2.projectPoints(
                    objp, rvecs_l[i], tvecs_l[i], m1, d1
                )
                err_l = cv2.norm(
                    self.left_points[i], proj_l, cv2.NORM_L2
                ) / len(proj_l)
                proj_r, _ = cv2.projectPoints(
                    objp, rvecs_r[i], tvecs_r[i], m2, d2
                )
                err_r = cv2.norm(
                    self.right_points[i], proj_r, cv2.NORM_L2
                ) / len(proj_r)
                errors.append((float(err_l), float(err_r)))

        return m1, d1, m2, d2, r, t, errors

    def save(
        self,
        path: str,
        m1: np.ndarray,
        d1: np.ndarray,
        m2: np.ndarray,
        d2: np.ndarray,
        r: np.ndarray | None = None,
        t: np.ndarray | None = None,
    ) -> None:  # pragma: no cover - optional saving
        data = {
            "left_camera_matrix": m1.tolist(),
            "left_dist_coeffs": d1.tolist(),
            "right_camera_matrix": m2.tolist(),
            "right_dist_coeffs": d2.tolist(),
        }
        if r is not None and t is not None:  # pragma: no cover - optional
            data["rotation"] = r.tolist()
            data["translation"] = t.tolist()
        try:
            Path(path).write_text(yaml.safe_dump(data))
        except Exception as exc:  # pragma: no cover - file IO
            logging.error("Failed to save calibration: %s", exc)
