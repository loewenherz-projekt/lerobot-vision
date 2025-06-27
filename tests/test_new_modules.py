import numpy as np
import pytest

from lerobot_vision.object_localizer import localize_objects
from lerobot_vision.pose_estimator import PoseEstimator
from lerobot_vision.stereo_calibrator import StereoCalibrator
from lerobot_vision.image_rectifier import ImageRectifier


def test_localize_objects():
    masks = [np.array([[1, 0], [0, 0]], dtype=np.uint8)]
    depth = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    cam = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    result = localize_objects(masks, depth, cam, ["cup"])
    assert len(result) == 1
    label, xyz, pose = result[0]
    assert label == "cup"
    assert pose is None
    assert xyz.shape == (3,)


def test_pose_estimator_noop(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.pose_estimator.DOPEEstimator", None, raising=False
    )
    est = PoseEstimator()
    out = est.estimate(np.zeros((1, 1, 3), dtype=np.uint8))
    assert out == []


def test_stereo_calibrator_no_corners():
    calib = StereoCalibrator()
    with pytest.raises(RuntimeError):
        calib.calibrate((10, 10))


def test_image_rectifier_identity():
    m = np.eye(3)
    d = np.zeros(5)
    rect = ImageRectifier(
        m, d, m, d, (2, 2), translation=np.array([1.0, 0, 0])
    )
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    l, r = rect.rectify(img, img)
    assert l.shape == img.shape
    assert r.shape == img.shape
