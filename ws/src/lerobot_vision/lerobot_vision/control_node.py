# ws/src/lerobot_vision/lerobot_vision/control_node.py
"""Robot control node."""

from __future__ import annotations

import logging

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Pose
import numpy as np
from typing import Sequence

from .robot_controller import RobotController


class ControlNode(Node):
    """Node controlling robot via lerobot SDK."""

    def __init__(self, port: str = "/dev/ttyUSB0", robot_id: int = 1) -> None:
        """Initialize the control node.

        Args:
            port: Serial port of the robot controller.
            robot_id: Identifier of the robot.
        """
        super().__init__("control_node")
        self.declare_parameter("port", port)
        self.declare_parameter("robot_id", robot_id)
        p_port = self.get_parameter("port").get_parameter_value().string_value
        p_id = (
            self.get_parameter("robot_id").get_parameter_value().integer_value
        )
        self.controller = RobotController(p_port, p_id)
        self.sub = self.create_subscription(
            JointTrajectory,
            "trajectory",
            self._cb,
            10,
        )
        self.sub_pose = self.create_subscription(
            Pose,
            "target_pose",
            self._cb_pose,
            10,
        )

    def _cb(self, msg: JointTrajectory) -> None:
        for point in msg.points:
            try:
                self.controller.move_to_joint_positions(point.positions)
            except Exception as exc:  # pragma: no cover
                logging.error("Movement failed: %s", exc)

    def _quat_to_rpy(self, q: Sequence[float]) -> tuple[float, float, float]:
        """Convert quaternion to roll, pitch, yaw."""
        x, y, z, w = q
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return float(roll), float(pitch), float(yaw)

    def _cb_pose(self, msg: Pose) -> None:
        pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
        ]
        quat = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        pose.extend(self._quat_to_rpy(quat))
        try:
            self.controller.move_linear(pose)
        except Exception as exc:  # pragma: no cover
            logging.error("Linear movement failed: %s", exc)


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``control_node`` executable."""
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
