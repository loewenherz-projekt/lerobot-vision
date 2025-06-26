"""Robot control node."""

from __future__ import annotations

import logging

from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

try:
    from lerobot import Robot
except Exception as exc:  # pragma: no cover - optional
    Robot = None
    logging.error("LeRobot import failed: %s", exc)


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
        if Robot is None:
            raise RuntimeError("lerobot unavailable")
        p_port = self.get_parameter("port").get_parameter_value().string_value
        p_id = (
            self.get_parameter("robot_id").get_parameter_value().integer_value
        )
        self.robot = Robot(p_port, p_id)
        self.sub = self.create_subscription(
            JointTrajectory,
            "trajectory",
            self._cb,
            10,
        )

    def _cb(self, msg: JointTrajectory) -> None:
        for point in msg.points:
            try:
                self.robot.move_to_joint_positions(point.positions)
            except Exception as exc:  # pragma: no cover
                logging.error("Movement failed: %s", exc)
