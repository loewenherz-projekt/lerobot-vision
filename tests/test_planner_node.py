import json
from unittest import mock
import pytest
from trajectory_msgs.msg import JointTrajectory

from lerobot_vision.planner_node import PlannerNode


def test_plan_actions(monkeypatch):
    monkey_engine = mock.Mock(
        return_value=mock.Mock(plan=mock.Mock(return_value=JointTrajectory()))
    )
    monkeypatch.setattr(
        "lerobot_vision.planner_node.MoveGroupCommander",
        monkey_engine,
    )
    node = PlannerNode()
    actions = json.dumps({"target_pose": [0, 0, 0]})
    traj = node._plan_actions(actions)
    assert isinstance(traj, JointTrajectory)


def test_plan_actions_failure(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.planner_node.MoveGroupCommander",
        None,
    )
    node = PlannerNode()
    with pytest.raises(RuntimeError):
        node._plan_actions("{}")
