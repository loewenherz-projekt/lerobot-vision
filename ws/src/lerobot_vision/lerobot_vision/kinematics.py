"""Simple robot kinematics utilities."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def rpy_to_matrix(angles: Sequence[float]) -> np.ndarray:
    """Convert roll, pitch, yaw to a rotation matrix."""
    roll, pitch, yaw = angles
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convert a rotation matrix to roll, pitch, yaw."""
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]
    sy = np.sqrt(r11 ** 2 + r21 ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(r32, r33)
        pitch = np.arctan2(-r31, sy)
        yaw = np.arctan2(r21, r11)
    else:
        roll = np.arctan2(-r23, r22)
        pitch = np.arctan2(-r31, sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def forward_kinematics(joints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple forward kinematics solution.

    This simplified model interprets the first three joint values as a Cartesian
    translation and the remaining three as roll, pitch and yaw angles.
    """
    pos = np.array(joints[:3], dtype=float)
    rot = rpy_to_matrix(joints[3:6])
    return pos, rot


def inverse_kinematics(position: Sequence[float], orientation: Sequence[Sequence[float]] | np.ndarray) -> list[float]:
    """Compute a simple inverse kinematics solution matching
    :func:`forward_kinematics`.
    """
    pos = [float(p) for p in position]
    rot = np.array(orientation, dtype=float).reshape(3, 3)
    rpy = matrix_to_rpy(rot)
    return pos + [float(v) for v in rpy]


__all__ = ["forward_kinematics", "inverse_kinematics", "rpy_to_matrix", "matrix_to_rpy"]
