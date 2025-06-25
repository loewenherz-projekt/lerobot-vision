"""ws/src/lerobot_vision/lerobot_vision/planner_node.py"""

from __future__ import annotations

import json
import logging
from typing import Any

from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

logger = logging.getLogger(__name__)


class PlannerNode(Node):
    """Plan robot trajectories from actions."""

    def __init__(self) -> None:
        super().__init__("planner_node")
        self.create_subscription(
            String, "/robot/vision/actions", self.actions_cb, 1
        )
        self.pub = self.create_publisher(
            JointTrajectory, "/arm_controller/trajectory", 1
        )
        logger.debug("PlannerNode initialized")

    def actions_cb(self, msg: String) -> None:
        self._plan_actions(msg.data)
        traj = JointTrajectory()
        self.pub.publish(traj)
        logger.debug("Published trajectory")

    def _plan_actions(self, actions_json: str) -> Any:
        """Generate a collision-free trajectory using MoveIt!"""
        try:
            actions = json.loads(actions_json)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON: %s", exc)
            raise
        # Stub for MoveIt! planning
        logger.debug("Planning actions: %s", actions)
        return actions
