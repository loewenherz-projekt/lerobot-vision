# tests/test_visualization_node.py
from unittest import mock

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

from lerobot_vision import visualization_node
from lerobot_vision.visualization_node import (
    VisualizationNode,
    TogglePublisher,
)


def test_on_timer(monkeypatch):
    class DummyCam:
        camera_matrix = np.eye(3)

        def __init__(self, *args, **kwargs):
            pass

        def get_frames(self):
            return (
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera",
        DummyCam,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera.camera_matrix",
        np.eye(3),
        raising=False,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera.dist_coeffs",
        np.zeros(5),
        raising=False,
    )

    def compute_depth(*args, **kwargs):
        if kwargs.get("return_disparity"):
            depth = np.zeros((1, 1), dtype=np.float32)
            return depth, np.zeros_like(depth)
        return np.zeros((1, 1), dtype=np.float32)

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.DepthEngine",
        mock.Mock(
            return_value=mock.Mock(
                compute_depth=mock.Mock(side_effect=compute_depth)
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.Yolo3DEngine",
        mock.Mock(
            return_value=mock.Mock(
                segment=mock.Mock(
                    return_value=([np.zeros((1, 1), dtype=np.uint8)], ["obj"])
                )
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.PoseEstimator",
        mock.Mock(return_value=mock.Mock(estimate=mock.Mock(return_value=[]))),
    )
    loc_mock = mock.Mock(return_value=[])
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.localize_objects",
        loc_mock,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.CvBridge",
        mock.Mock(
            return_value=mock.Mock(
                cv2_to_imgmsg=mock.Mock(return_value=Image())
            )  # noqa: E501
        ),
    )

    class DummyRect:
        def __init__(self, *args, **kwargs):
            pass

        def rectify(self, left, right):
            return left, right

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.ImageRectifier",
        DummyRect,
    )

    class DummyFusion:
        def __init__(self, *args, **kwargs):
            self.pc_pub = mock.Mock(publish=mock.Mock())
            self.det_pub = mock.Mock(publish=mock.Mock())
            self.publish = mock.Mock(side_effect=self._do_publish)

        def _do_publish(self, masks, labels, poses):
            self.pc_pub.publish("pc")
            self.det_pub.publish("det")

    fusion_inst = DummyFusion()
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.FusionModule",
        mock.Mock(return_value=fusion_inst),
    )

    rclpy.init(args=None)
    node = VisualizationNode("/tmp")
    node.pub.publish = mock.Mock()
    node._on_timer()
    node.pub.publish.assert_called_once()
    loc_mock.assert_called_once()
    args, _ = loc_mock.call_args
    assert len(args) == 5
    fusion_inst.publish.assert_called_once()
    fusion_inst.pc_pub.publish.assert_called_once()
    fusion_inst.det_pub.publish.assert_called_once()
    rclpy.shutdown()


def test_draw_overlay(monkeypatch):
    class DummyCam:
        camera_matrix = np.eye(3)

        def __init__(self, *args, **kwargs):
            pass

        def get_frames(self):
            return (
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera",
        DummyCam,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.DepthEngine",
        mock.Mock(return_value=mock.Mock()),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.Yolo3DEngine",
        mock.Mock(return_value=mock.Mock()),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.PoseEstimator",
        mock.Mock(return_value=mock.Mock()),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.CvBridge",
        mock.Mock(
            return_value=mock.Mock(
                cv2_to_imgmsg=mock.Mock(return_value=Image())
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.FusionModule",
        mock.Mock(return_value=mock.Mock()),
    )

    monkeypatch.setattr("cv2.rectangle", mock.Mock())
    monkeypatch.setattr("cv2.putText", mock.Mock())
    monkeypatch.setattr("cv2.line", mock.Mock())

    rclpy.init(args=None)
    node = VisualizationNode("/tmp")
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=np.uint8)
    depth = np.ones((5, 5), dtype=float)
    node._draw_overlay(
        img, [mask], ["obj"], depth, [(np.zeros(3), np.array([0, 0, 0, 1]))]
    )

    cv2.rectangle.assert_called_once()
    assert cv2.putText.call_count >= 1
    assert cv2.line.call_count == 3
    rclpy.shutdown()


def test_toggle_service(monkeypatch):
    class DummyCam:
        camera_matrix = np.eye(3)

        def __init__(self, *args, **kwargs):
            pass

        def get_frames(self):
            return (
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera",
        DummyCam,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera.camera_matrix",
        np.eye(3),
        raising=False,
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera.dist_coeffs",
        np.zeros(5),
        raising=False,
    )

    def compute_depth(*args, **kwargs):
        if kwargs.get("return_disparity"):
            depth = np.zeros((1, 1), dtype=np.float32)
            return depth, np.zeros_like(depth)
        return np.zeros((1, 1), dtype=np.float32)

    monkeypatch.setattr(
        "lerobot_vision.visualization_node.DepthEngine",
        mock.Mock(
            return_value=mock.Mock(
                compute_depth=mock.Mock(side_effect=compute_depth)
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.Yolo3DEngine",
        mock.Mock(
            return_value=mock.Mock(
                segment=mock.Mock(
                    return_value=([np.zeros((1, 1), dtype=np.uint8)], ["obj"])
                )
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.PoseEstimator",
        mock.Mock(return_value=mock.Mock(estimate=mock.Mock(return_value=[]))),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.localize_objects",
        mock.Mock(return_value=[]),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.CvBridge",
        mock.Mock(
            return_value=mock.Mock(
                cv2_to_imgmsg=mock.Mock(return_value=Image())
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.ImageRectifier",
        mock.Mock(
            return_value=mock.Mock(
                rectify=mock.Mock(
                    side_effect=lambda left_frame, right_frame: (
                        left_frame,
                        right_frame,
                    )
                )
            )
        ),
    )
    # Mock for FusionModule is added here to support merged features
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.FusionModule",
        mock.Mock(return_value=mock.Mock(publish=mock.Mock())),
    )

    def create_pub(self, *args, **kwargs):
        return mock.Mock()

    monkeypatch.setattr(
        rclpy.node.Node, "create_publisher", create_pub, raising=False
    )

    rclpy.init(args=None)
    node = VisualizationNode("/tmp")
    old_pub = node.pub
    old_pub.publish = mock.Mock()
    node._on_timer()
    old_pub.publish.assert_called_once()

    req = TogglePublisher.Request(publisher="overlay", enable=False)
    # `.call()` is used here because the real service setup is mocked.
    # In a normal environment, you would invoke the callback directly.
    node.toggle_srv.call(req)
    node._on_timer()
    old_pub.publish.assert_called_once()
    assert node.pub is None

    req = TogglePublisher.Request(publisher="overlay", enable=True)
    node.toggle_srv.call(req)
    node.pub.publish = mock.Mock()
    node._on_timer()
    node.pub.publish.assert_called_once()

    # Enable disparity and masks publishers and ensure they are used
    req = TogglePublisher.Request(publisher="disparity", enable=True)
    node.toggle_srv.call(req)
    node.pub_disparity.publish = mock.Mock()

    req = TogglePublisher.Request(publisher="masks", enable=True)
    node.toggle_srv.call(req)
    node.pub_masks.publish = mock.Mock()

    node._on_timer()
    node.pub_disparity.publish.assert_called_once()
    node.pub_masks.publish.assert_called_once()
    rclpy.shutdown()


def test_main(monkeypatch):
    node = mock.Mock(destroy_node=mock.Mock())
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.VisualizationNode",
        mock.Mock(return_value=node),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.rclpy.init", mock.Mock()
    )
    spin = mock.Mock()
    monkeypatch.setattr(visualization_node.rclpy, "spin", spin, raising=False)
    shutdown = mock.Mock()
    monkeypatch.setattr(
        visualization_node.rclpy,
        "shutdown",
        shutdown,
        raising=False,
    )

    visualization_node.main([])
    spin.assert_called_once_with(node)
    node.destroy_node.assert_called_once()
    shutdown.assert_called_once()
