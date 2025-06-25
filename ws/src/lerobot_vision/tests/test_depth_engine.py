import numpy as np
import pytest

from lerobot_vision.depth_engine import DepthEngine


class DummyEngine:
    def __init__(self, *args, **kwargs):
        pass

    def infer(self, left, right):
        return np.ones((1, 1))


def test_compute_depth_success(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.depth_engine.StereoAnywhere", DummyEngine
    )
    engine = DepthEngine()
    depth = engine.compute_depth(np.zeros((1, 1)), np.zeros((1, 1)))
    assert depth.shape == (1, 1)


def test_compute_depth_type_error(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.depth_engine.StereoAnywhere", DummyEngine
    )
    engine = DepthEngine()
    with pytest.raises(TypeError):
        engine.compute_depth(None, None)  # type: ignore
