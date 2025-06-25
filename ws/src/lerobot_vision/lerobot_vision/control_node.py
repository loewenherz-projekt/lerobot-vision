"""ws/src/lerobot_vision/lerobot_vision/control_node.py"""

from __future__ import annotations

import logging

from lerobot import Robot
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

logger = logging.getLogger(__name__)


class ControlNode(Node):
    """Control robot based on planned trajectories."""

    def __init__(self) -> None:
        super().__init__("control_node")
        self.robot = Robot(port="/dev/ttyUSB0", id="arm")
        self.create_subscription(
            JointTrajectory, "/arm_controller/trajectory", self.traj_cb, 1
        )
        logger.debug("ControlNode initialized")

    def traj_cb(self, msg: JointTrajectory) -> None:
        for point in msg.points:
            positions = point.positions
            try:
                self.robot.move_to_joint_positions(positions)
            except Exception as exc:  # pragma: no cover - hardware
                logger.error("Robot movement failed: %s", exc)
                raise
