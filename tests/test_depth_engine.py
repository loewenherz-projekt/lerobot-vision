# tests/test_depth_engine.py
import numpy as np
import pytest
from unittest import mock

from lerobot_vision.depth_engine import DepthEngine


def test_compute_depth_invalid():
    engine = DepthEngine(model_path=None, use_cuda=False)
    engine.engine = mock.Mock()
    with pytest.raises(ValueError):
        engine.compute_depth("bad", np.zeros((1, 1)))


def test_compute_depth_success():
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    depth = np.zeros((1, 1), dtype=np.float32)
    engine = DepthEngine(model_path=None, use_cuda=False)
    engine.engine = mock.Mock(infer=mock.Mock(return_value=depth))
    result = engine.compute_depth(dummy, dummy)
    assert result is depth
    engine.engine.infer.assert_called_once()


def test_init_custom_model():
    class Dummy:
        def __init__(self, *args, **kwargs):
            Dummy.args = args
            Dummy.kwargs = kwargs

    with mock.patch("lerobot_vision.depth_engine.StereoAnywhere", Dummy):
        engine = DepthEngine(model_path="model.pth", use_cuda=False)
        assert isinstance(engine.engine, Dummy)
        assert Dummy.kwargs == {"model_path": "model.pth"}


def test_cuda_fallback(monkeypatch):
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    disp = np.ones((1, 1), dtype=np.float32)
    matcher = mock.Mock(compute=mock.Mock(return_value=disp))
    wls = mock.Mock(filter=mock.Mock(return_value=disp))
    import types
    import lerobot_vision.depth_engine as de

    dummy_cv2 = types.SimpleNamespace(cuda=types.SimpleNamespace())
    monkeypatch.setattr(de, "cv2", dummy_cv2)
    monkeypatch.setattr(
        de.cv2.cuda,
        "StereoSGBM_create",
        mock.Mock(return_value=matcher),
        raising=False,
    )
    monkeypatch.setattr(
        de,
        "ximgproc",
        mock.Mock(createDisparityWLSFilterGeneric=mock.Mock(return_value=wls)),
        raising=False,
    )
    engine = DepthEngine(use_cuda=True, baseline=0.5, focal=2.0)
    result = engine.compute_depth(dummy, dummy)
    expected = (engine.focal * engine.baseline) / disp
    assert np.allclose(result, expected)
    matcher.compute.assert_called_once()
    wls.filter.assert_called_once()
