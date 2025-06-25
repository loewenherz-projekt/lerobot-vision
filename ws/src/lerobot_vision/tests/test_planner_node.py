from unittest.mock import MagicMock

import pytest

try:
    from std_msgs.msg import String
    from rclpy.node import Node
except Exception:
    pytest.skip("ROS not available", allow_module_level=True)

from lerobot_vision.planner_node import PlannerNode


def test_plan_actions(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.planner_node.MoveGroupCommander", MagicMock()
    )
    node = PlannerNode()
    node.group.plan = MagicMock(return_value="plan")
    result = node._plan_actions('{"pose": "x"}')
    assert result == "plan"


def test_moveit_missing(monkeypatch):
    monkeypatch.setattr("lerobot_vision.planner_node.MoveGroupCommander", None)
    node = PlannerNode()
    with pytest.raises(RuntimeError):
        node._plan_actions("{}")
