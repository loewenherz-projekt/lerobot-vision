from __future__ import annotations

"""Simple wrapper implementing interpolated robot motion."""

import time
from dataclasses import dataclass
from typing import List, Sequence
import logging
import numpy as np

from .kinematics import forward_kinematics, inverse_kinematics

try:
    from lerobot import Robot
except Exception as exc:  # pragma: no cover - optional dependency
    Robot = None
    logging.error("LeRobot import failed: %s", exc)


@dataclass
class MoveResult:
    """Result of a movement request."""

    ok: bool
    msg: str = ""


class RobotController:
    """Robot interface with basic interpolation."""

    def __init__(self, port: str = "/dev/ttyUSB0", robot_id: int = 1) -> None:
        if Robot is None:
            raise RuntimeError("lerobot unavailable")
        self.robot = Robot(port, robot_id)
        self.max_steps = 20
        self.step_delay = 0.05
        self.range_min = -100.0
        self.range_max = 100.0

    def _validate(self, joints: Sequence[float]) -> bool:
        return all(self.range_min <= j <= self.range_max for j in joints)

    def _interpolate(
        self, start: Sequence[float], target: Sequence[float]
    ) -> None:
        diffs = [abs(t - s) for s, t in zip(start, target)]
        steps = max(1, min(self.max_steps, int(max(diffs))))
        for i in range(1, steps + 1):
            inter = [s + (t - s) * i / steps for s, t in zip(start, target)]
            self.robot.move_to_joint_positions(inter)
            time.sleep(self.step_delay)

    def move_to_joint_positions(
        self, joints: Sequence[float], use_interpolation: bool = True
    ) -> MoveResult:
        if not self._validate(joints):
            return MoveResult(False, "target out of range")
        if use_interpolation:
            try:
                start = self.robot.get_joint_positions()
            except Exception:  # pragma: no cover - runtime
                start = list(joints)
            self._interpolate(start, joints)
        else:
            self.robot.move_to_joint_positions(joints)
        return MoveResult(True, "")

    def move_linear(self, pose: Sequence[float], steps: int | None = None) -> MoveResult:
        """Move the robot linearly to a target pose.

        ``pose`` should contain ``x, y, z, roll, pitch, yaw``.
        The movement is interpolated in Cartesian space and converted
        to joint angles using the simple inverse kinematics model from
        :mod:`lerobot_vision.kinematics`.
        """

        if len(pose) != 6:
            return MoveResult(False, "invalid pose")
        try:
            current = self.robot.get_joint_positions()
        except Exception:  # pragma: no cover - runtime
            return MoveResult(False, "cannot read joints")

        start_pos, start_rot = forward_kinematics(current)
        target_pos = np.array(pose[:3], dtype=float)
        target_rot = np.array(
            forward_kinematics([0, 0, 0, *pose[3:]])[1], dtype=float
        )

        if steps is None:
            steps = self.max_steps

        for i in range(1, steps + 1):
            pos = start_pos + (target_pos - start_pos) * i / steps
            rot = start_rot + (target_rot - start_rot) * i / steps
            joints = inverse_kinematics(pos, rot)
            if not self._validate(joints):
                return MoveResult(False, "target out of range")
            self.robot.move_to_joint_positions(joints)
            time.sleep(self.step_delay)

        return MoveResult(True, "")
