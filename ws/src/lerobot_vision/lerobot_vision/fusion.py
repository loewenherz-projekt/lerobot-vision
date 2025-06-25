"""ws/src/lerobot_vision/lerobot_vision/fusion.py"""

from __future__ import annotations

import logging
from typing import Any

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

logger = logging.getLogger(__name__)


class FusionModule:
    """Fuse detections and depth into ROS messages."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.points_pub = node.create_publisher(
            PointCloud2, "/robot/vision/points", 1
        )
        self.detections_pub = node.create_publisher(
            Detection3DArray, "/robot/vision/detections", 1
        )
        logger.debug("FusionModule publishers created")

    def publish(self, masks: Any, labels: Any, poses: Any) -> None:
        """Publish fused perception messages."""
        cloud = self._to_pointcloud(masks, poses)
        dets = self._to_detections(masks, labels, poses)
        self.points_pub.publish(cloud)
        self.detections_pub.publish(dets)

    def _to_pointcloud(self, masks: Any, poses: Any) -> PointCloud2:
        """Convert to PointCloud2."""
        raise NotImplementedError

    def _to_detections(
        self, masks: Any, labels: Any, poses: Any
    ) -> Detection3DArray:
        """Convert to Detection3DArray."""
        raise NotImplementedError
