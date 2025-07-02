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


def test_ui_route(monkeypatch):
    client = TestClient(web.app)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert b"LeRobot Web Interface" in resp.content


def test_calibration_flow(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class DummyCam:
        def __init__(self, *a, **k):
            pass

        def get_frames(self):
            return frame, frame

        def release(self):
            pass

        def get_properties(self):
            return {"width": 2, "height": 2, "fps": 30}

    class DummyCal:
        def __init__(self, *a, **k):
            self.pairs = 0

        def add_corners(self, l, r):
            self.pairs += 1
            return True

        def calibrate(self, size):
            return np.eye(3), np.zeros(5), np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), None

    monkeypatch.setattr(web, "AsyncStereoCamera", DummyCam)
    monkeypatch.setattr(web, "StereoCalibrator", DummyCal)
    web.manager.start()
    client = TestClient(web.app)
    resp = client.post("/calibration/start")
    assert resp.status_code == 200
    resp = client.post("/calibration/capture")
    assert resp.json() == {"captured": True}
    resp = client.post("/calibration/finish")
    assert resp.status_code == 200
    assert "m1" in resp.json()
    web.manager.stop()


def test_camera_modes(monkeypatch):
    class DummyCap:
        def __init__(self, *_):
            self.width = 640
            self.height = 480
            self.fps = 30

        def isOpened(self):
            return True

        def set(self, prop, val):
            if prop == web.cv2.CAP_PROP_FRAME_WIDTH:
                self.width = val
            if prop == web.cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = val
            if prop == web.cv2.CAP_PROP_FPS:
                self.fps = val

        def get(self, prop):
            mapping = {
                web.cv2.CAP_PROP_FRAME_WIDTH: self.width,
                web.cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                web.cv2.CAP_PROP_FPS: self.fps,
            }
            return mapping[prop]

        def release(self):
            pass

    monkeypatch.setattr(web.cv2, "VideoCapture", lambda i: DummyCap())
    client = TestClient(web.app)
    resp = client.get("/camera_modes")
    assert resp.status_code == 200
    data = resp.json()
    assert "modes" in data


def test_models(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_text("dummy")
    monkeypatch.setattr(web, "models", web.ModelManager(str(tmp_path)))
    client = TestClient(web.app)
    resp = client.get("/models")
    assert resp.json()["models"] == ["model"]
    resp = client.post("/models/select?name=model")
    assert resp.json()["selected"] == "model"
