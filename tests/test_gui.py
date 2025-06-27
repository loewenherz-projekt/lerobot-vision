import numpy as np
from unittest import mock

import lerobot_vision.gui as gui_mod


def setup_gui(monkeypatch):
    class DummyTk:
        def title(self, _):
            pass

        def update_idletasks(self):
            pass

        def update(self):
            pass

        def mainloop(self):
            pass

    class DummyWidget:
        def __init__(self, *a, **k):
            pass

        def pack(self, *a, **k):
            pass

        def configure(self, *a, **k):
            pass

        def destroy(self):
            pass

    class DummyVar:
        def set(self, _):
            pass

        def get(self):
            return False

    class DummyThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr(gui_mod.tk, "Tk", DummyTk)
    monkeypatch.setattr(gui_mod.tk, "Label", lambda *a, **k: DummyWidget())
    monkeypatch.setattr(gui_mod.tk, "Button", lambda *a, **k: DummyWidget())
    monkeypatch.setattr(
        gui_mod.tk, "Checkbutton", lambda *a, **k: DummyWidget()
    )
    monkeypatch.setattr(gui_mod.tk, "Toplevel", lambda *a, **k: DummyWidget())
    monkeypatch.setattr(gui_mod.tk, "StringVar", DummyVar)
    monkeypatch.setattr(gui_mod.tk, "BooleanVar", lambda *a, **k: DummyVar())
    monkeypatch.setattr(gui_mod.threading, "Thread", DummyThread)
    monkeypatch.setattr(
        gui_mod,
        "DepthEngine",
        lambda *a, **k: mock.Mock(
            compute_depth=mock.Mock(return_value=np.zeros((1, 1), dtype=float))
        ),
    )
    monkeypatch.setattr(
        gui_mod,
        "ImageRectifier",
        mock.Mock(return_value=mock.Mock(rectify=lambda l, r: (l, r))),
    )
    monkeypatch.setattr(
        gui_mod,
        "PoseEstimator",
        mock.Mock(return_value=mock.Mock(estimate=mock.Mock(return_value=[]))),
    )
    monkeypatch.setattr(
        gui_mod,
        "Yolo3DEngine",
        mock.Mock(
            return_value=mock.Mock(
                segment=mock.Mock(
                    return_value=([np.zeros((1, 1), dtype=np.uint8)], ["obj"])
                )
            )
        ),
    )
    monkeypatch.setattr(
        gui_mod, "localize_objects", mock.Mock(return_value=[])
    )

    cam = mock.Mock()
    cam.get_frames.return_value = (
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.uint8),
    )
    return cam


def test_wizard_flow(monkeypatch):
    cam = setup_gui(monkeypatch)
    gui = gui_mod.VisionGUI(cam)
    gui._review = mock.Mock()
    gui._calibrate = mock.Mock()

    gui.next_step()
    assert gui.step_idx == 1
    gui._review.assert_called_once()

    gui.next_step()
    assert gui.step_idx == 2
    gui._calibrate.assert_called_once()

    gui.prev_step()
    assert gui.step_idx == 1


def test_wizard_bounds(monkeypatch):
    cam = setup_gui(monkeypatch)
    gui = gui_mod.VisionGUI(cam)

    gui.step_idx = len(gui.steps) - 1
    gui.next_step()
    assert gui.step_idx == len(gui.steps) - 1

    gui.step_idx = 0
    gui.prev_step()
    assert gui.step_idx == 0


def test_toggle_views(monkeypatch):
    cam = setup_gui(monkeypatch)
    gui = gui_mod.VisionGUI(cam)

    gui.show_rect_var.get = lambda: True
    gui._toggle_rect()
    assert gui.rect_window is not None

    gui.show_rect_var.get = lambda: False
    gui._toggle_rect()
    assert gui.rect_window is None
