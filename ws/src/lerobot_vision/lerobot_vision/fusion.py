"""Fusion of segmentation masks into ROS messages."""

from __future__ import annotations

import logging
from typing import Any, List

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray


class FusionModule:
    """Fuse masks and poses and publish ROS messages."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.pc_pub = node.create_publisher(PointCloud2, "cloud", 10)
        self.det_pub = node.create_publisher(
            Detection3DArray,
            "detections",
            10,
        )

    def publish(self, masks: Any, labels: List[str], poses: Any) -> None:
        """Publish fused outputs."""
        pc = self._masks_to_pointcloud2(masks)
        detections = self._make_detections(masks, labels, poses)
        self.pc_pub.publish(pc)
        self.det_pub.publish(detections)

    def _masks_to_pointcloud2(self, masks: Any) -> PointCloud2:
        logging.debug("Convert masks to PointCloud2")
        raise NotImplementedError

    def _make_detections(
        self, masks: Any, labels: List[str], poses: Any
    ) -> Detection3DArray:
        logging.debug("Create Detection3DArray")
        raise NotImplementedError
