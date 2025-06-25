from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from rclpy.node import Node
except Exception:
    pytest.skip("rclpy not available", allow_module_level=True)

from lerobot_vision.visualization_node import VisualizationNode


def test_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera", MagicMock()
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.DepthEngine", MagicMock()
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.Yolo3DEngine", MagicMock()
    )
    node = VisualizationNode(str(tmp_path))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = node._overlay(img, [np.array([[0, 0], [1, 1]])], ["l"])
    assert result is img
