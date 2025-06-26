"""Visualization node creating overlays."""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from pathlib import Path

from .camera_interface import StereoCamera
from .depth_engine import DepthEngine
from .yolo3d_engine import Yolo3DEngine


class VisualizationNode(Node):
    """Publishes overlay images."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__("visualization_node")
        base = Path(__file__).resolve().parent.parent
        default_cfg = base / "config" / "camera.yaml"
        self.declare_parameter("camera_config", str(default_cfg))
        self.declare_parameter("yolo_checkpoint", checkpoint_dir)
        config_path = (
            self.get_parameter("camera_config")
            .get_parameter_value()
            .string_value
        )
        ckpt_path = (
            self.get_parameter("yolo_checkpoint")
            .get_parameter_value()
            .string_value
        )
        self.bridge = CvBridge()
        self.camera = StereoCamera(config_path=config_path)
        self.depth_engine = DepthEngine()
        self.yolo_engine = Yolo3DEngine(ckpt_path)
        self.pub = self.create_publisher(Image, "/openyolo3d/overlay", 10)
        self.create_timer(0.2, self._on_timer)

    def _on_timer(self) -> None:
        try:
            left, right = self.camera.get_frames()
            depth = self.depth_engine.compute_depth(left, right)
            masks, labels = self.yolo_engine.segment([left], depth)
            overlay = self._draw_overlay(left, masks, labels)
            msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub.publish(msg)
        except Exception as exc:  # pragma: no cover
            logging.error("Overlay generation failed: %s", exc)

    def _draw_overlay(
        self, image: np.ndarray, masks: List[np.ndarray], labels: List[str]
    ) -> np.ndarray:
        for mask, label in zip(masks, labels):
            pts = np.column_stack(np.nonzero(mask > 0)).astype(np.float32)
            if pts.size == 0:
                continue
            points_3d = np.hstack(
                (pts[:, [1, 0]], np.zeros((pts.shape[0], 1)))
            )
            proj, _ = cv2.projectPoints(
                points_3d,
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                StereoCamera.camera_matrix,
                StereoCamera.dist_coeffs,
            )
            proj = proj.reshape(-1, 1, 2).astype(int)
            cv2.polylines(image, [proj], True, (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                tuple(proj[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        return image
