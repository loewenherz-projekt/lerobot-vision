# File: tests/test_yolo3d_engine.py
import sys
import types
import importlib
from unittest import mock

import numpy as np
import pytest


def test_segment_success(monkeypatch):
    fake_model = mock.MagicMock()
    fake_model.segment.return_value = ([np.zeros((1, 3))], ["obj"])
    yolo_stub = types.SimpleNamespace(Yolo3D=lambda checkpoint_dir: fake_model)
    monkeypatch.setitem(sys.modules, "openyolo3d", yolo_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.yolo3d_engine"
    )
    importlib.reload(module)
    Yolo3DEngine = module.Yolo3DEngine

    eng = Yolo3DEngine("chk")
    masks, labels = eng.segment([np.zeros((1, 1, 3))], np.zeros((1, 1)))
    assert labels == ["obj"]


def test_segment_no_model(monkeypatch):
    yolo_stub = types.SimpleNamespace(Yolo3D=None)
    monkeypatch.setitem(sys.modules, "openyolo3d", yolo_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.yolo3d_engine"
    )
    importlib.reload(module)
    Yolo3DEngine = module.Yolo3DEngine

    eng = Yolo3DEngine("chk")
    with pytest.raises(RuntimeError):
        eng.segment([np.zeros((1, 1, 3))], np.zeros((1, 1)))
