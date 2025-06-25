from unittest.mock import MagicMock

import pytest

try:
    from rclpy.node import Node
except Exception:
    pytest.skip("rclpy not available", allow_module_level=True)

from lerobot_vision.fusion import FusionModule


class DummyNode(Node):
    def __init__(self):
        super().__init__("dummy")


def test_publish(monkeypatch):
    node = DummyNode()
    fusion = FusionModule(node)
    pc_pub = MagicMock()
    det_pub = MagicMock()
    fusion.pc_pub = pc_pub
    fusion.det_pub = det_pub
    monkeypatch.setattr(fusion, "_masks_to_pointcloud2", lambda x: "pc")
    monkeypatch.setattr(fusion, "_make_detections", lambda m, l, p: "det")
    fusion.publish(None, None, None)
    pc_pub.publish.assert_called_once_with("pc")
    det_pub.publish.assert_called_once_with("det")
