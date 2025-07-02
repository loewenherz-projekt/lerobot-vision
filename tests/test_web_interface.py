import numpy as np
from fastapi.testclient import TestClient

from lerobot_vision import web_interface as web


def test_list_cameras(monkeypatch):
    class DummyCap:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        def isOpened(self) -> bool:
            return self.idx < 2

        def release(self) -> None:
            pass

    monkeypatch.setattr(web.cv2, "VideoCapture", lambda idx: DummyCap(idx))
    client = TestClient(web.app)
    resp = client.get("/cameras")
    assert resp.status_code == 200
    assert resp.json() == {"cameras": [0, 1]}


def test_stream_generator(monkeypatch):
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    class DummyCam:
        def __init__(self, *a, **k):
            pass

        def get_frames(self):
            return frame, frame

        def release(self):
            pass

    monkeypatch.setattr(web, "AsyncStereoCamera", DummyCam)
    web.manager.start()
    gen = web._mjpeg_generator("left")
    chunk = next(gen)
    assert b"--frame" in chunk
    web.manager.stop()
