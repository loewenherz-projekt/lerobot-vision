import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "ws", "src", "lerobot_vision"
    ),
)

import types  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

mod = types.ModuleType("openyolo3d")


class Detector:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def predict(self, images, depth_map):
        return [np.zeros((4, 2))], ["obj"]


mod.Detector = Detector
sys.modules["openyolo3d"] = mod

from lerobot_vision.yolo3d_engine import Yolo3DEngine  # noqa: E402


def test_segment_success():
    engine = Yolo3DEngine(checkpoint_dir="ckpt")
    masks, labels = engine.segment([np.zeros((1, 1))], np.zeros((1, 1)))
    assert labels == ["obj"]
    assert len(masks) == 1


def test_segment_type_error():
    engine = Yolo3DEngine(checkpoint_dir="ckpt")
    with pytest.raises(TypeError):
        engine.segment("not list", np.zeros((1, 1)))
