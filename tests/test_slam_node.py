import numpy as np
from unittest import mock
import rclpy

from lerobot_vision.slam_node import SlamNode


class DummyCam:
    def __init__(self, *args, **kwargs):
        pass

    def get_frames(self):
        left = np.zeros((1, 1, 3), dtype=np.uint8)
        right = np.zeros((1, 1, 3), dtype=np.uint8)
        return left, right


def test_slam_node(monkeypatch, tmp_path):
    slam_inst = mock.Mock(
        track=mock.Mock(return_value=[0, 0, 0]),
        get_map=mock.Mock(return_value=b"map"),
    )
    monkeypatch.setattr(
        "lerobot_vision.slam_node.SLAMSystem",
        mock.Mock(return_value=slam_inst),
    )
    monkeypatch.setattr("lerobot_vision.slam_node.StereoCamera", DummyCam)
    rclpy.init(args=None)
    node = SlamNode()
    node.pub_map.publish = mock.Mock()
    node.pub_pose.publish = mock.Mock()
    node.map_output = str(tmp_path / "map.npz")
    node._on_timer()
    node.pub_map.publish.assert_called_once()
    node.pub_pose.publish.assert_called_once()
    assert (tmp_path / "map.npz").exists()
    rclpy.shutdown()
