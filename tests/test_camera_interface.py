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
