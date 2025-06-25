"""Control node for robot."""

import logging

from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

try:
    from lerobot import Robot
except Exception as exc:  # pragma: no cover - optional dependency
    Robot = None  # type: ignore
    logging.error("LeRobot import failed: %s", exc)


class ControlNode(Node):
    """Node that sends trajectories to the robot."""

    def __init__(self, port: str = "/dev/ttyUSB0", robot_id: int = 1) -> None:
        super().__init__("control_node")
        if Robot is None:
            raise ImportError("LeRobot SDK not available")
        self.robot = Robot(port, robot_id)
        self.create_subscription(JointTrajectory, "trajectory", self._cb, 10)

    def _cb(self, msg: JointTrajectory) -> None:
        for point in msg.points:
            try:
                self.robot.move_to_joint_positions(point.positions)
            except Exception as exc:
                logging.error("Robot movement failed: %s", exc)
