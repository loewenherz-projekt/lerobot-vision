# File: ws/src/lerobot_vision/lerobot_vision/control_node.py
"""Node controlling the physical robot."""

import logging

from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

try:
    from lerobot import Robot
except ImportError:  # pragma: no cover - external dependency
    Robot = None

logger = logging.getLogger(__name__)


class ControlNode(Node):
    """Receive trajectories and command the robot."""

    def __init__(self) -> None:
        super().__init__("control_node")
        if Robot is None:
            self.robot = None
            logger.error("lerobot package not available")
        else:
            self.robot = Robot(port="/dev/ttyUSB0", id="arm")
        self.create_subscription(
            JointTrajectory, "/arm_controller/trajectory", self.traj_cb, 10
        )
        logger.debug("ControlNode initialized")

    def traj_cb(self, msg: JointTrajectory) -> None:
        if self.robot is None:
            logger.error("Robot not initialized")
            return
        for point in msg.points:
            positions = list(point.positions)
            try:
                self.robot.move_to_joint_positions(positions)
            except Exception as exc:  # pragma: no cover - hardware
                logger.error("Movement failed: %s", exc)
                raise
