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

mod = types.ModuleType("stereoanywhere")


class StereoAnywhere:
    def __init__(self, pretrained=True, model_path=None):
        pass

    def compute(self, left, right):
        return np.ones((1, 1))


mod.StereoAnywhere = StereoAnywhere
sys.modules["stereoanywhere"] = mod

from lerobot_vision.depth_engine import DepthEngine  # noqa: E402


def test_compute_depth_success():
    engine = DepthEngine(model_path="path")
    left = np.zeros((1, 1))
    right = np.zeros((1, 1))
    depth = engine.compute_depth(left, right)
    assert np.array_equal(depth, np.ones((1, 1)))


def test_compute_depth_type_error():
    engine = DepthEngine(model_path="path")
    with pytest.raises(TypeError):
        engine.compute_depth(1, 2)
