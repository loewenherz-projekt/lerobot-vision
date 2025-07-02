# tests/test_camera_interface.py
import numpy as np
import pytest

from lerobot_vision.camera_interface import StereoCamera


def test_get_frames_success(monkeypatch):
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    class DummyCap:
        def read(self):
            return True, img

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: DummyCap())
    cam = StereoCamera()
    left, right = cam.get_frames()
    assert np.array_equal(left, img)
    assert np.array_equal(right, img)


def test_get_frames_failure(monkeypatch):
    class BadCap:
        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: BadCap())
    cam = StereoCamera()
    with pytest.raises(RuntimeError):
        cam.get_frames()


def test_side_by_side(monkeypatch):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[:, :10] = 1
    img[:, 10:] = 2

    class DummyCap:
        def read(self):
            return True, img

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: DummyCap())
    cam = StereoCamera(side_by_side=True)
    left, right = cam.get_frames()
    assert left.shape[1] == right.shape[1] == 10
    assert np.all(left == 1)
    assert np.all(right == 2)


def test_camera_properties(monkeypatch):
    class DummyCap:
        def __init__(self):
            self.props = {}

        def read(self):
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

        def release(self):
            pass

        def set(self, prop, value):
            self.props[prop] = value

        def get(self, prop):
            return self.props.get(prop, 0)

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: DummyCap())
    cam = StereoCamera()
    cam.set_properties(width=320, height=240, fps=25)
    info = cam.get_properties()
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["fps"] == 25
