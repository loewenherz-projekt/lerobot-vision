import numpy as np
import pytest
from unittest import mock

from lerobot_vision.depth_engine import DepthEngine


def test_compute_depth_invalid():
    engine = DepthEngine(pretrained=False)
    engine.engine = mock.Mock()
    with pytest.raises(ValueError):
        engine.compute_depth("bad", np.zeros((1, 1)))


def test_compute_depth_success():
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    depth = np.zeros((1, 1), dtype=np.float32)
    engine = DepthEngine(pretrained=False)
    engine.engine = mock.Mock(infer=mock.Mock(return_value=depth))
    result = engine.compute_depth(dummy, dummy)
    assert result is depth
    engine.engine.infer.assert_called_once()
