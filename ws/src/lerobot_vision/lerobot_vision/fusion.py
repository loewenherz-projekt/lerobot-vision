"""Fusion module."""

from typing import Any, List

import logging
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray


class FusionModule:
    """Fuses detection masks into ROS messages."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.pc_pub = node.create_publisher(PointCloud2, "pointcloud", 10)
        self.det_pub = node.create_publisher(Detection3DArray, "detections", 10)

    def publish(self, masks: Any, labels: Any, poses: Any) -> None:
        pc_msg = self._masks_to_pointcloud2(masks)
        det_msg = self._make_detections(masks, labels, poses)
        self.pc_pub.publish(pc_msg)
        self.det_pub.publish(det_msg)

    def _masks_to_pointcloud2(self, masks: Any) -> PointCloud2:
        raise NotImplementedError

    def _make_detections(self, masks: Any, labels: Any, poses: Any) -> Detection3DArray:
        raise NotImplementedError
