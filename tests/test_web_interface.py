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


def test_camera_info(monkeypatch):
    class DummyCap:
        def __init__(self, *_):
            pass

        def isOpened(self) -> bool:
            return True

        def get(self, prop):
            mapping = {
                web.cv2.CAP_PROP_FRAME_WIDTH: 640,
                web.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                web.cv2.CAP_PROP_FPS: 30,
            }
            return mapping[prop]

        def release(self):
            pass

    monkeypatch.setattr(web.cv2, "VideoCapture", lambda i: DummyCap())
    client = TestClient(web.app)
    resp = client.get("/camera_info?index=0")
    assert resp.status_code == 200
    assert resp.json() == {"width": 640, "height": 480, "fps": 30}


def test_robot_move():
    client = TestClient(web.app)
    resp = client.post("/robot/move", params={"positions": "1,2"})
    assert resp.json() == {"status": "ok"}
    resp = client.get("/robot/positions")
    assert resp.json() == {"positions": [1.0, 2.0]}
