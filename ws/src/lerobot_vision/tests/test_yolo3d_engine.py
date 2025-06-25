import numpy as np
import pytest

from lerobot_vision.yolo3d_engine import Yolo3DEngine


class DummyYOLO:
    def __init__(self, checkpoint):
        pass

    def infer(self, images, depth):
        return [np.zeros((1, 1))], ["label"]


def test_segment_success(monkeypatch, tmp_path):
    ckpt = tmp_path
    monkeypatch.setattr(
        "lerobot_vision.yolo3d_engine.OpenYolo3D", DummyYOLO
    )
    engine = Yolo3DEngine(str(ckpt))
    masks, labels = engine.segment([np.zeros((1, 1))], np.zeros((1, 1)))
    assert len(masks) == 1 and labels[0] == "label"


def test_missing_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.yolo3d_engine.OpenYolo3D", DummyYOLO
    )
    with pytest.raises(FileNotFoundError):
        Yolo3DEngine("/nonexistent")
