# tests/test_fusion.py
from unittest import mock

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

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

    masks = [np.array([[1, 0], [0, 1]], dtype=np.uint8)]
    labels = ["obj"]
    poses = [object()]

    pc = fusion._masks_to_pointcloud2(masks)
    assert isinstance(pc, PointCloud2)
    assert hasattr(pc, "points")
    assert len(pc.points) == 2

    det = fusion._make_detections(masks, labels, poses)
    assert isinstance(det, Detection3DArray)
    assert len(det.detections) == 1
    assert det.detections[0]["label"] == "obj"
    rclpy.shutdown()
