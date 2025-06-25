# File: ws/src/lerobot_vision/lerobot_vision/fusion.py
"""Fusion module for publishing vision results."""

from typing import Any
import logging

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

logger = logging.getLogger(__name__)


class FusionModule:
    """Fuse perception results and publish ROS messages."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.points_pub = node.create_publisher(
            PointCloud2, "/robot/vision/points", 10
        )
        self.dets_pub = node.create_publisher(
            Detection3DArray, "/robot/vision/detections", 10
        )
        logger.debug("FusionModule publishers created")

    def publish(self, masks: Any, labels: Any, poses: Any) -> None:
        """Publish results as PointCloud2 and Detection3DArray."""
        cloud = self._masks_to_pointcloud(masks, poses)
        detections = self._labels_to_detections(labels, poses)
        self.points_pub.publish(cloud)
        self.dets_pub.publish(detections)

    def _masks_to_pointcloud(self, masks: Any, poses: Any) -> PointCloud2:
        """Convert masks to a PointCloud2 message."""
        raise NotImplementedError

    def _labels_to_detections(
        self, labels: Any, poses: Any
    ) -> Detection3DArray:
        """Convert labels and poses to Detection3DArray."""
        raise NotImplementedError
