# ws/src/lerobot_vision/lerobot_vision/visualization_node.py
"""Visualization node creating overlays."""

from __future__ import annotations

import logging
from typing import List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from pathlib import Path

from .camera_interface import StereoCamera
from .depth_engine import DepthEngine
from .yolo3d_engine import Yolo3DEngine
from .pose_estimator import PoseEstimator
from .object_localizer import localize_objects
from .image_rectifier import ImageRectifier
from .fusion import FusionModule


class TogglePublisher:
    """Service for toggling publishers."""

    @dataclass
    class Request:
        publisher: str = ""
        enable: bool = True

    @dataclass
    class Response:
        success: bool = False
        message: str = ""


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
        self.declare_parameter("publish_left_raw", False)
        self.declare_parameter("publish_right_raw", False)
        self.declare_parameter("publish_left_rectified", False)
        self.declare_parameter("publish_right_rectified", False)
        self.declare_parameter("publish_depth", False)
        self.declare_parameter("publish_overlay", True)
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
        p_left_raw = (
            self.get_parameter("publish_left_raw")
            .get_parameter_value()
            .integer_value
            == 1
        )
        p_right_raw = (
            self.get_parameter("publish_right_raw")
            .get_parameter_value()
            .integer_value
            == 1
        )
        p_left_rect = (
            self.get_parameter("publish_left_rectified")
            .get_parameter_value()
            .integer_value
            == 1
        )
        p_right_rect = (
            self.get_parameter("publish_right_rectified")
            .get_parameter_value()
            .integer_value
            == 1
        )
        p_depth = (
            self.get_parameter("publish_depth")
            .get_parameter_value()
            .integer_value
            == 1
        )
        p_overlay = (
            self.get_parameter("publish_overlay")
            .get_parameter_value()
            .integer_value
            == 1
        )
        self.bridge = CvBridge()
        self.camera = StereoCamera(config_path=config_path)
        self.depth_engine = DepthEngine()
        self.yolo_engine = Yolo3DEngine(ckpt_path)
        self.pose_estimator = PoseEstimator()
        self.fusion = FusionModule(self)
        self.pub = (
            self.create_publisher(Image, "/openyolo3d/overlay", 10)
            if p_overlay
            else None
        )
        self.pub_left_raw = (
            self.create_publisher(Image, "/stereo/left_raw", 10)
            if p_left_raw
            else None
        )
        self.pub_right_raw = (
            self.create_publisher(Image, "/stereo/right_raw", 10)
            if p_right_raw
            else None
        )
        self.pub_left_rect = (
            self.create_publisher(Image, "/stereo/left_rectified", 10)
            if p_left_rect
            else None
        )
        self.pub_right_rect = (
            self.create_publisher(Image, "/stereo/right_rectified", 10)
            if p_right_rect
            else None
        )
        self.pub_depth = (
            self.create_publisher(Image, "/stereo/depth", 10)
            if p_depth
            else None
        )
        self.rectifier: ImageRectifier | None = None
        self.create_timer(0.2, self._on_timer)
        self.toggle_srv = self.create_service(
            TogglePublisher, "toggle_publisher", self._toggle_publisher
        )

    def _on_timer(self) -> None:
        try:
            left, right = self.camera.get_frames()
            if self.pub_left_raw:
                msg = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
                self.pub_left_raw.publish(msg)
            if self.pub_right_raw:
                msg = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")
                self.pub_right_raw.publish(msg)
            if self.rectifier is None:
                h, w = left.shape[:2]
                self.rectifier = ImageRectifier(
                    StereoCamera.camera_matrix,
                    StereoCamera.dist_coeffs,
                    StereoCamera.camera_matrix,
                    StereoCamera.dist_coeffs,
                    (w, h),
                )
            left_r, right_r = self.rectifier.rectify(left, right)
            if self.pub_left_rect:
                msg = self.bridge.cv2_to_imgmsg(left_r, encoding="bgr8")
                self.pub_left_rect.publish(msg)
            if self.pub_right_rect:
                msg = self.bridge.cv2_to_imgmsg(right_r, encoding="bgr8")
                self.pub_right_rect.publish(msg)
            depth = self.depth_engine.compute_depth(left_r, right_r)
            if self.pub_depth:
                msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
                self.pub_depth.publish(msg)
            masks, labels = self.yolo_engine.segment([left_r], depth)
            poses = self.pose_estimator.estimate(left_r)
            _ = localize_objects(
                masks, depth, StereoCamera.camera_matrix, labels, poses
            )
            self.fusion.publish(masks, labels, poses)
            overlay = self._draw_overlay(left_r, masks, labels, depth, poses)
            if self.pub:
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

    def _toggle_publisher(
        self,
        request: TogglePublisher.Request,
        response: TogglePublisher.Response,
    ) -> TogglePublisher.Response:
        mapping = {
            "overlay": (
                "pub",
                (Image, "/openyolo3d/overlay", 10),
            ),
            "left_raw": (
                "pub_left_raw",
                (Image, "/stereo/left_raw", 10),
            ),
            "right_raw": (
                "pub_right_raw",
                (Image, "/stereo/right_raw", 10),
            ),
            "left_rectified": (
                "pub_left_rect",
                (Image, "/stereo/left_rectified", 10),
            ),
            "right_rectified": (
                "pub_right_rect",
                (Image, "/stereo/right_rectified", 10),
            ),
            "depth": (
                "pub_depth",
                (Image, "/stereo/depth", 10),
            ),
        }

        if request.publisher not in mapping:
            response.success = False
            response.message = "unknown publisher"
            return response

        attr, (msg_type, topic, depth) = mapping[request.publisher]
        if request.enable:
            if getattr(self, attr) is None:
                setattr(
                    self, attr, self.create_publisher(msg_type, topic, depth)
                )
        else:
            setattr(self, attr, None)
        response.success = True
        return response


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``visualization_node`` executable."""
    rclpy.init(args=args)
    node = VisualizationNode("/tmp")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
