# File: tests/test_depth_engine.py
import sys
import types
import importlib
from unittest import mock

import numpy as np
import pytest


def test_compute_depth_success(monkeypatch):
    fake_model = mock.MagicMock()
    fake_model.compute.return_value = np.ones((2, 2))
    stereo_stub = types.SimpleNamespace(
        StereoAnywhere=lambda pretrained=True: fake_model
    )
    monkeypatch.setitem(sys.modules, "stereoanywhere", stereo_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.depth_engine"
    )
    importlib.reload(module)
    DepthEngine = module.DepthEngine

    engine = DepthEngine()
    depth = engine.compute_depth(np.zeros((2, 2)), np.zeros((2, 2)))
    assert depth.shape == (2, 2)


def test_compute_depth_no_model(monkeypatch):
    stereo_stub = types.SimpleNamespace(StereoAnywhere=None)
    monkeypatch.setitem(sys.modules, "stereoanywhere", stereo_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.depth_engine"
    )
    importlib.reload(module)
    DepthEngine = module.DepthEngine

    engine = DepthEngine()
    with pytest.raises(RuntimeError):
        engine.compute_depth(np.zeros((2, 2)), np.zeros((2, 2)))
