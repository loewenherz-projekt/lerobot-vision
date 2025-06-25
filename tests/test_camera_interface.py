import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "ws", "src", "lerobot_vision"
    ),
)  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from lerobot_vision.camera_interface import StereoCamera  # noqa: E402


def test_get_frames_success(monkeypatch):
    class DummyCap:
        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: DummyCap())
    cam = StereoCamera()
    left, right = cam.get_frames()
    assert isinstance(left, np.ndarray)
    assert left.shape == (2, 2, 3)
    cam.release()


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
