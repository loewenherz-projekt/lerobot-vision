from unittest import mock

import pytest
import rclpy
from rclpy.node import Node

from lerobot_vision.fusion import FusionModule


def test_publish_calls_methods():
    rclpy.init(args=None)
    node = Node("test")
    fusion = FusionModule(node)
    fusion._masks_to_pointcloud2 = mock.Mock(return_value=object())
    fusion._make_detections = mock.Mock(return_value=object())
    fusion.pc_pub.publish = mock.Mock()
    fusion.det_pub.publish = mock.Mock()

    fusion.publish([], [], [])

    fusion._masks_to_pointcloud2.assert_called_once()
    fusion._make_detections.assert_called_once()
    fusion.pc_pub.publish.assert_called_once()
    fusion.det_pub.publish.assert_called_once()
    rclpy.shutdown()


def test_helper_methods():
    rclpy.init(args=None)
    node = Node("test")
    fusion = FusionModule(node)

    with pytest.raises(NotImplementedError):
        fusion._masks_to_pointcloud2([1, 2])

    with pytest.raises(NotImplementedError):
        fusion._make_detections([], ["a", "b"], [])
    rclpy.shutdown()
