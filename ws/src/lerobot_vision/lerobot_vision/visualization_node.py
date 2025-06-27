# ws/src/lerobot_vision/lerobot_vision/visualization_node.py
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
from .pose_estimator import PoseEstimator
from .object_localizer import localize_objects


class VisualizationNode(Node):
    """Publishes overlay images."""

    def __init__(self, checkpoint_dir: str) -> None:
        """Initialize the visualization node.

        Args:
            checkpoint_dir: Directory containing the YOLOv3-3D checkpoints.
        """
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
        self.pose_estimator = PoseEstimator()
        self.pub = self.create_publisher(Image, "/openyolo3d/overlay", 10)
        self.create_timer(0.2, self._on_timer)

    def _on_timer(self) -> None:
        try:
            left, right = self.camera.get_frames()
            depth = self.depth_engine.compute_depth(left, right)
            masks, labels = self.yolo_engine.segment([left], depth)
            poses = self.pose_estimator.estimate(left)
            _ = localize_objects(masks, depth, StereoCamera.camera_matrix)
            overlay = self._draw_overlay(left, masks, labels)
            msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub.publish(msg)
        except Exception as exc:  # pragma: no cover
            logging.error("Overlay generation failed: %s", exc)

    def _draw_overlay(
        self, image: np.ndarray, masks: List[np.ndarray], labels: List[str]
    ) -> np.ndarray:
        """Draw segmentation overlays on an image.

        Args:
            image: Source image on which to draw.
            masks: List of binary masks corresponding to objects.
            labels: Labels associated with each mask.

        Returns:
            The image with overlays applied.
        """
        for mask, label in zip(masks, labels):
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (int(x0), int(y0) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return image
