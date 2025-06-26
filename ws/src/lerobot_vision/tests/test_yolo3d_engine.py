import numpy as np
import pytest
from unittest import mock

from lerobot_vision.yolo3d_engine import Yolo3DEngine


def test_segment_invalid(tmp_path):
    ckpt = tmp_path
    ckpt.mkdir(exist_ok=True)
    with pytest.raises(RuntimeError):
        with mock.patch("lerobot_vision.yolo3d_engine.OpenYolo3D", None):
            Yolo3DEngine(str(ckpt))


def test_segment_success(tmp_path):
    ckpt_dir = tmp_path
    ckpt_dir.mkdir(exist_ok=True)
    mock_engine = mock.Mock(return_value=mock.Mock())
    with mock.patch("lerobot_vision.yolo3d_engine.OpenYolo3D", mock_engine):
        yolo = Yolo3DEngine(str(ckpt_dir))
    images = [np.zeros((1, 1, 3), dtype=np.uint8)]
    depth = np.zeros((1, 1), dtype=np.float32)
    yolo.engine = mock.Mock(return_value=(images, ["obj"]))
    with mock.patch.object(
        yolo.engine,
        "__call__",
        return_value=(images, ["obj"]),
    ):
        masks, labels = yolo.segment(images, depth)
        assert labels == ["obj"]
