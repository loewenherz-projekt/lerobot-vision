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

    class DummyVar:
        def set(self, _):
            pass

    class DummyThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr(gui_mod.tk, "Tk", DummyTk)
    monkeypatch.setattr(gui_mod.tk, "Label", lambda *a, **k: DummyWidget())
    monkeypatch.setattr(gui_mod.tk, "Button", lambda *a, **k: DummyWidget())
    monkeypatch.setattr(gui_mod.tk, "StringVar", DummyVar)
    monkeypatch.setattr(gui_mod.threading, "Thread", DummyThread)

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
