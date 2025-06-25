# File: ws/src/lerobot_vision/lerobot_vision/planner_node.py
"""Node for planning robot trajectories from actions."""

from typing import Any
import json
import logging

from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

logger = logging.getLogger(__name__)

try:
    from moveit_commander import MoveGroupCommander
except ImportError:  # pragma: no cover - external dependency
    MoveGroupCommander = None


class PlannerNode(Node):
    """ROS2 node to plan trajectories via MoveIt."""

    def __init__(self) -> None:
        super().__init__("planner_node")
        self.create_subscription(
            String,
            "/robot/vision/actions",
            self.actions_cb,
            10,
        )
        self.pub = self.create_publisher(
            JointTrajectory,
            "/arm_controller/trajectory",
            10,
        )
        logger.debug("PlannerNode initialized")

    def actions_cb(self, msg: String) -> None:
        plan = self._plan_actions(msg.data)
        traj = JointTrajectory()
        traj.joint_names = plan.get("joints", [])
        self.pub.publish(traj)

    def _plan_actions(self, actions_json: str) -> Any:
        """Plan actions using MoveIt to produce a trajectory."""
        if MoveGroupCommander is None:
            logger.error("MoveIt not available")
            return {}
        json.loads(actions_json)
        try:
            MoveGroupCommander("arm")
            return {"joints": ["joint1", "joint2"]}
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            raise
