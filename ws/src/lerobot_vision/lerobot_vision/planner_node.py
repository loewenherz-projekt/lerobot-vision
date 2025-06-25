"""Planner node using MoveIt."""

import json
import logging
from typing import Any

from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

try:
    from moveit_commander import MoveGroupCommander
except Exception as exc:  # pragma: no cover - optional dependency
    MoveGroupCommander = None
    logging.error("MoveIt import failed: %s", exc)


class PlannerNode(Node):
    """Plans actions for the robot."""

    def __init__(self) -> None:
        super().__init__("planner_node")
        self.create_subscription(String, "/robot/vision/actions", self._cb, 10)
        self.pub = self.create_publisher(JointTrajectory, "/arm_controller/trajectory", 10)
        if MoveGroupCommander:
            self.group = MoveGroupCommander("manipulator")
        else:
            self.group = None

    def _cb(self, msg: String) -> None:
        traj = self._plan_actions(msg.data)
        if traj:
            self.pub.publish(traj)

    def _plan_actions(self, actions_json: str) -> JointTrajectory:
        if self.group is None:
            raise RuntimeError("MoveIt not available")
        actions = json.loads(actions_json)
        target_pose = actions.get("pose")
        plan = self.group.plan(target_pose)
        return plan
