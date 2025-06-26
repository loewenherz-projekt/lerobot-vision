from unittest import mock
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot_vision.control_node import ControlNode


def test_control_node(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.control_node.Robot",
        mock.Mock(return_value=mock.Mock(move_to_joint_positions=mock.Mock())),
    )
    node = ControlNode()
    msg = JointTrajectory(points=[JointTrajectoryPoint(positions=[0.1])])
    node._cb(msg)
    node.robot.move_to_joint_positions.assert_called_once_with([0.1])
