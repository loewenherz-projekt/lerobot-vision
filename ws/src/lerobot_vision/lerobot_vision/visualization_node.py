# ws/src/lerobot_vision/lerobot_vision/visualization_node.py
"""Visualization node creating overlays."""

from __future__ import annotations

import logging
from typing import List, Tuple

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
            _ = localize_objects(
                masks, depth, StereoCamera.camera_matrix, labels, poses
            )
            overlay = self._draw_overlay(left, masks, labels, depth, poses)
            msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub.publish(msg)
        except Exception as exc:  # pragma: no cover
            logging.error("Overlay generation failed: %s", exc)

    def _draw_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        labels: List[str],
        depth: np.ndarray,
        poses: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Draw segmentation overlays on an image.

        Args:
            image: Source image on which to draw.
            masks: List of binary masks corresponding to objects.
            labels: Labels associated with each mask.

        Returns:
            The image with overlays applied.
        """
        fx = StereoCamera.camera_matrix[0, 0]
        fy = StereoCamera.camera_matrix[1, 1]
        cx = StereoCamera.camera_matrix[0, 2]
        cy = StereoCamera.camera_matrix[1, 2]

        def _project(pt: np.ndarray) -> tuple[int, int]:
            u = int(pt[0] * fx / pt[2] + cx)
            v = int(pt[1] * fy / pt[2] + cy)
            return u, v

        for idx, (mask, label) in enumerate(zip(masks, labels)):
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            u = float(np.median(xs))
            v = float(np.median(ys))
            z = float(np.median(depth[ys, xs]))
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
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
            info = f"{z:.2f}m {x:+.2f},{y:+.2f}"
            cv2.putText(
                image,
                info,
                (int(x0), int(y1) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            if poses and idx < len(poses) and poses[idx] is not None:
                _, quat = poses[idx]
                xq, yq, zq, wq = quat
                rot = np.array(
                    [
                        [
                            1 - 2 * (yq**2 + zq**2),
                            2 * (xq * yq - zq * wq),
                            2 * (xq * zq + yq * wq),
                        ],
                        [
                            2 * (xq * yq + zq * wq),
                            1 - 2 * (xq**2 + zq**2),
                            2 * (yq * zq - xq * wq),
                        ],
                        [
                            2 * (xq * zq - yq * wq),
                            2 * (yq * zq + xq * wq),
                            1 - 2 * (xq**2 + yq**2),
                        ],
                    ]
                )
                center = np.array([x, y, z], dtype=float)
                axes = rot @ (0.05 * np.eye(3))
                for axis, color in zip(
                    axes.T,
                    [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
                ):
                    pt2 = _project(center + axis)
                    cv2.line(image, (int(u), int(v)), pt2, color, 2)
        return image
