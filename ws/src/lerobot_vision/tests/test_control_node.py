from unittest.mock import MagicMock

import pytest

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except Exception:
    pytest.skip("ROS messages not available", allow_module_level=True)

from lerobot_vision.control_node import ControlNode


def test_control(monkeypatch):
    monkeypatch.setattr("lerobot_vision.control_node.Robot", MagicMock())
    node = ControlNode()
    pt = JointTrajectoryPoint(positions=[1, 2, 3])
    msg = JointTrajectory(points=[pt])
    node._cb(msg)
    node.robot.move_to_joint_positions.assert_called_once_with([1, 2, 3])
