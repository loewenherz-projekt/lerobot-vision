# File: ws/src/lerobot_vision/lerobot_vision/visualization_node.py
"""Node publishing an overlay of detections on camera images."""

import logging
from typing import List

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .camera_interface import StereoCamera
from .yolo3d_engine import Yolo3DEngine

logger = logging.getLogger(__name__)


class VisualizationNode(Node):
    """Capture images, run detection and publish an overlay."""

    def __init__(self) -> None:
        super().__init__("visualization_node")
        self.bridge = CvBridge()
        self.camera = StereoCamera()
        self.yolo3d = Yolo3DEngine(
            checkpoint_dir="libs/OpenYOLO3D/checkpoints"
        )
        self.pub = self.create_publisher(Image, "/openyolo3d/overlay", 10)
        self.timer = self.create_timer(0.2, self.timer_cb)
        logger.debug("VisualizationNode initialized")

    def timer_cb(self) -> None:
        try:
            left, right = self.camera.get_frames()
            depth_map = np.zeros(left.shape[:2], dtype=np.float32)
            masks, labels = self.yolo3d.segment([left, right], depth_map)
            overlay = self._render_overlay(left, masks, labels)
            imgmsg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub.publish(imgmsg)
        except Exception as exc:  # pragma: no cover - hardware/network
            logger.error("Visualization failed: %s", exc)

    def _render_overlay(
        self, img: np.ndarray, masks: List[np.ndarray], labels: List[str]
    ) -> np.ndarray:
        """Render polygon masks and labels onto image."""
        out = img.copy()
        for mask, lbl in zip(masks, labels):
            pts2d, _ = cv2.projectPoints(
                mask.astype("float32"),
                (0, 0, 0),
                (0, 0, 0),
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
            )
            pts2d = pts2d.reshape(-1, 2).astype(int)
            cv2.polylines(out, [pts2d], True, (0, 255, 0), 2)
            cv2.putText(
                out,
                lbl,
                tuple(pts2d[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return out
