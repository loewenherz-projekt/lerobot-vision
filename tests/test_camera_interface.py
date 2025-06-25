# File: tests/test_camera_interface.py
import sys
import types
import importlib

import numpy as np
import pytest


def test_get_frames_success(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeCap:
        def read(self):
            return True, frame

        def release(self):
            pass

    cv2_stub = types.SimpleNamespace(
        VideoCapture=lambda idx: FakeCap(),
        undistort=lambda img, m, d: img,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.camera_interface"
    )
    importlib.reload(module)
    StereoCamera = module.StereoCamera

    cam = StereoCamera()
    left, right = cam.get_frames()
    assert np.array_equal(left, frame)
    assert np.array_equal(right, frame)


def test_get_frames_failure(monkeypatch):
    class BadCap:
        def read(self):
            return False, None

        def release(self):
            pass

    cv2_stub = types.SimpleNamespace(
        VideoCapture=lambda idx: BadCap(),
        undistort=lambda img, m, d: img,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    module = importlib.import_module(
        "ws.src.lerobot_vision.lerobot_vision.camera_interface"
    )
    importlib.reload(module)
    StereoCamera = module.StereoCamera

    cam = StereoCamera()
    with pytest.raises(RuntimeError):
        cam.get_frames()
