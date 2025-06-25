"""Visualization node."""

import logging

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from .camera_interface import StereoCamera
from .depth_engine import DepthEngine
from .yolo3d_engine import Yolo3DEngine


class VisualizationNode(Node):
    """Publishes overlay images."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__("visualization_node")
        self.bridge = CvBridge()
        self.camera = StereoCamera()
        self.depth = DepthEngine()
        self.yolo = Yolo3DEngine(checkpoint_dir)
        self.pub = self.create_publisher(Image, "/openyolo3d/overlay", 10)
        self.create_timer(0.2, self._on_timer)

    def _on_timer(self) -> None:
        try:
            left, right = self.camera.get_frames()
            depth_map = self.depth.compute_depth(left, right)
            masks, labels = self.yolo.segment([left], depth_map)
            overlay = self._overlay(left, masks, labels)
            img_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub.publish(img_msg)
        except Exception as exc:
            logging.error("Visualization error: %s", exc)

    def _overlay(self, image: np.ndarray, masks, labels) -> np.ndarray:
        for mask, label in zip(masks, labels):
            pts = np.array(mask, dtype=np.int32)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            cv2.putText(image, label, (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image
