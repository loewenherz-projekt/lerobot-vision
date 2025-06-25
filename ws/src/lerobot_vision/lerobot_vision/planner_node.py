"""Planner node using MoveIt."""

from __future__ import annotations

import json
import logging
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

try:
    from moveit_commander import MoveGroupCommander
except Exception as exc:  # pragma: no cover - may be missing
    MoveGroupCommander = None
    logging.error("MoveIt import failed: %s", exc)


class PlannerNode(Node):
    """Plan trajectories based on actions."""

    def __init__(self) -> None:
        super().__init__("planner_node")
        self.sub = self.create_subscription(
            String, "/robot/vision/actions", self._cb, 10
        )
        self.pub = self.create_publisher(
            JointTrajectory, "/arm_controller/trajectory", 10
        )
        self.group = MoveGroupCommander("manipulator") if MoveGroupCommander else None

    def _cb(self, msg: String) -> None:
        try:
            traj = self._plan_actions(msg.data)
            self.pub.publish(traj)
        except Exception as exc:  # pragma: no cover
            logging.error("Planning failed: %s", exc)

    def _plan_actions(self, actions_json: str) -> JointTrajectory:
        """Plan trajectory."""
        if self.group is None:
            raise RuntimeError("MoveIt not available")
        actions = json.loads(actions_json)
        # Here we would create trajectory from actions
        target_pose = actions.get("target_pose")
        plan = self.group.plan(target_pose)
        if isinstance(plan, tuple):
            plan = plan[1]
        if not isinstance(plan, JointTrajectory):
            plan_msg = JointTrajectory()
        else:
            plan_msg = plan
        return plan_msg
