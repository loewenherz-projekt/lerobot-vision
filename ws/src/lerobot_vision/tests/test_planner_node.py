import json
from unittest import mock

import rclpy
from trajectory_msgs.msg import JointTrajectory

from lerobot_vision.planner_node import PlannerNode


def test_plan_actions(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.planner_node.MoveGroupCommander",
        mock.Mock(
            return_value=mock.Mock(plan=mock.Mock(return_value=JointTrajectory()))
        ),
    )
    node = PlannerNode()
    actions = json.dumps({"target_pose": [0, 0, 0]})
    traj = node._plan_actions(actions)
    assert isinstance(traj, JointTrajectory)
