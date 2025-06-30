# ws/src/lerobot_vision/lerobot_vision/slam_node.py
"""SLAM node using SuperPointSLAM3."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .camera_interface import StereoCamera

try:
    from SuperPointSLAM3 import System as SLAMSystem  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    SLAMSystem = None
    logging.error("SuperPointSLAM3 import failed: %s", exc)


class SlamNode(Node):
    """Node running SuperPointSLAM3."""

    def __init__(self) -> None:
        super().__init__("slam_node")
        base = Path(__file__).resolve().parent.parent
        default_cfg = base / "config" / "camera.yaml"
        self.declare_parameter("camera_config", str(default_cfg))
        self.declare_parameter("left_idx", 0)
        self.declare_parameter("right_idx", 1)
        self.declare_parameter("side_by_side", False)
        self.declare_parameter("map_output", "")
        cfg = (
            self.get_parameter("camera_config").get_parameter_value().string_value
        )
        idx_left = (
            self.get_parameter("left_idx").get_parameter_value().integer_value
        )
        idx_right = (
            self.get_parameter("right_idx").get_parameter_value().integer_value
        )
        side_by_side = (
            self.get_parameter("side_by_side")
            .get_parameter_value()
            .integer_value
            == 1
        )
        self.map_output = (
            self.get_parameter("map_output").get_parameter_value().string_value
        )
        if SLAMSystem is None:
            raise RuntimeError("SuperPointSLAM3 unavailable")
        self.slam = SLAMSystem()
        self.camera = StereoCamera(
            idx_left, idx_right, config_path=cfg, side_by_side=side_by_side
        )
        self.pub_map = self.create_publisher(PointCloud2, "/slam/map", 10)
        self.pub_pose = self.create_publisher(String, "/slam/pose", 10)
        self.create_timer(0.1, self._on_timer)

    def _on_timer(self) -> None:
        left, right = self.camera.get_frames()
        pose = self.slam.track(left, right)  # type: ignore[attr-defined]
        map_data = self.slam.get_map()  # type: ignore[attr-defined]
        self.pub_pose.publish(String(data=str(pose)))
        self.pub_map.publish(PointCloud2(data=map_data))
        if self.map_output:
            try:
                np.savez(self.map_output, map=map_data)
            except Exception as exc:  # pragma: no cover - filesystem
                logging.error("Failed to save map: %s", exc)


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``slam_node`` executable."""
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
