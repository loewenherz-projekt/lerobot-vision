import cv2
import numpy as np
import pytest
from unittest import mock

from lerobot_vision.camera_interface import StereoCamera


class DummyCap:
    def __init__(self, ret=True):
        self.ret = ret

    def read(self):
        if self.ret:
            return True, np.zeros((1, 1, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


def test_get_frames_success(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda idx: DummyCap())
    cam = StereoCamera()
    left, right = cam.get_frames()
    assert left.shape == right.shape


def test_get_frames_failure(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda idx: DummyCap(ret=False))
    cam = StereoCamera()
    with pytest.raises(RuntimeError):
        cam.get_frames()
