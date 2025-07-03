# tests/test_control_node.py
from unittest import mock
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose

from lerobot_vision.control_node import ControlNode


def test_control_node(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.control_node.RobotController",
        mock.Mock(return_value=mock.Mock(move_to_joint_positions=mock.Mock())),
    )
    node = ControlNode()
    msg = JointTrajectory(points=[JointTrajectoryPoint(positions=[0.1])])
    node._cb(msg)
    node.controller.move_to_joint_positions.assert_called_once_with([0.1])


def test_control_node_pose(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.control_node.RobotController",
        mock.Mock(return_value=mock.Mock(move_linear=mock.Mock())),
    )
    node = ControlNode()
    pose = Pose(
        position=mock.Mock(x=1.0, y=2.0, z=3.0),
        orientation=mock.Mock(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    node._cb_pose(pose)
    node.controller.move_linear.assert_called_once()
