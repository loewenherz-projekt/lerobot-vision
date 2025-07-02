import numpy as np
import cv2
from fastapi.testclient import TestClient
import yaml

from lerobot_vision import web_interface as web
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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


def test_calibration_flow(monkeypatch, tmp_path):
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
            return (
                np.eye(3),
                np.zeros(5),
                np.eye(3),
                np.zeros(5),
                np.eye(3),
                np.zeros(3),
                None,
            )

    monkeypatch.setattr(web, "AsyncStereoCamera", DummyCam)
    monkeypatch.setattr(web, "StereoCalibrator", DummyCal)
    saved = tmp_path / "cal.yaml"

    orig_save = web.save_calibration_yaml

    def _save(data, path="calibration.yaml"):
        orig_save(data, path=str(saved))

    monkeypatch.setattr(web, "save_calibration_yaml", _save)
    web.manager.start()
    client = TestClient(web.app)
    resp = client.post("/calibration/start")
    assert resp.status_code == 200
    resp = client.post("/calibration/capture")
    assert resp.json() == {"captured": True}
    resp = client.post("/calibration/finish")
    assert resp.status_code == 200
    assert "m1" in resp.json()
    assert saved.exists()
    data = yaml.safe_load(saved.read_text())
    assert "m1" in data
    web.manager.stop()


def test_camera_modes(monkeypatch):
    class DummyCap:
        def __init__(self, *_):
            self.width = 0
            self.height = 0
            self.fps = 0
            self.fourcc = 0

        def isOpened(self):
            return True

        def set(self, prop, val):
            if prop == web.cv2.CAP_PROP_FRAME_WIDTH:
                self.width = val
            elif prop == web.cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = val
            elif prop == web.cv2.CAP_PROP_FPS:
                self.fps = val
            elif prop == web.cv2.CAP_PROP_FOURCC:
                self.fourcc = val

        def get(self, prop):
            mapping = {
                web.cv2.CAP_PROP_FRAME_WIDTH: self.width,
                web.cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                web.cv2.CAP_PROP_FPS: self.fps,
                web.cv2.CAP_PROP_FOURCC: self.fourcc,
            }
            return mapping[prop]

        def release(self):
            pass

    monkeypatch.setattr(web.cv2, "VideoCapture", lambda i: DummyCap())
    client = TestClient(web.app)
    resp = client.get("/camera_modes")
    assert resp.status_code == 200
    modes = resp.json()["modes"]
    assert len(modes) > 0
    assert all("codec" in m for m in modes)
    combos = {(m["width"], m["height"], m["fps"], m["codec"]) for m in modes}
    assert len(combos) > 1


def test_camera_settings_sets_codec(monkeypatch):
    class DummyCam:
        def __init__(self):
            self.args = None

        def set_properties(self, w, h, fps, codec):
            self.args = codec

    dummy = DummyCam()
    monkeypatch.setattr(web.manager, "camera", dummy)
    client = TestClient(web.app)
    resp = client.post("/camera_settings", params={"codec": "MJPG"})
    assert resp.status_code == 200
    assert dummy.args == "MJPG"


def test_models(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_text("dummy")
    monkeypatch.setattr(web, "models", web.ModelManager(str(tmp_path)))
    client = TestClient(web.app)
    resp = client.get("/models")
    assert resp.json()["models"] == ["model"]
    resp = client.post("/models/select?name=model")
    assert resp.json()["selected"] == "model"


def test_modules_settings():
    client = TestClient(web.app)
    resp = client.get("/modules")
    assert "yolo3d" in resp.json()
    resp = client.post(
        "/modules/select",
        params={"name": "yolo3d", "enable": "true", "score_threshold": "0.7"},
    )
    assert resp.status_code == 200
    assert web.modules["yolo3d"]["enabled"] is True
    assert web.modules["yolo3d"]["score_threshold"] == 0.7


def test_inference_test(monkeypatch):
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    called = {}

    def dummy_run(image):
        called["done"] = True
        return image

    monkeypatch.setattr(web, "run_inference", dummy_run)
    client = TestClient(web.app)
    resp = client.post(
        "/inference/test",
        files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert called.get("done") is True


def test_robot_params(monkeypatch, tmp_path):
    cfg = {"port": "/dev/ttyS1", "robot_id": 5}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))

    class DummyRobot:
        def __init__(self, port, robot_id):
            self.args = (port, robot_id)
        def move_to_joint_positions(self, pos):
            self.pos = pos
        def get_joint_positions(self):
            return [0.0] * 6
    monkeypatch.setattr(web, "Robot", DummyRobot)
    client = TestClient(web.app)
    resp = client.post("/robot/params", params={"payload": str(path)})
    assert resp.status_code == 200
    assert isinstance(web.robot.robot, DummyRobot)
    assert web.robot.robot.args == ("/dev/ttyS1", 5)


def test_fk_ik_endpoints():
    client = TestClient(web.app)
    resp = client.get("/robot/fk", params={"joints": "1,2,3,0,0,0"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["position"] == [1.0, 2.0, 3.0]
    assert data["orientation"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pose = "1,2,3,0,0,0"
    resp = client.get("/robot/ik", params={"pose": pose})
    assert resp.status_code == 200
    assert resp.json()["joints"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]


def test_nlp_planner_integration(monkeypatch):
    class DummyNlp:
        def _call_llm(self, text):
            return [{"target_pose": [0, 0, 0]}]

    class DummyPlanner:
        def __init__(self):
            pass

        def _plan_actions(self, actions_json):
            traj = JointTrajectory()
            traj.points = []
            pt = JointTrajectoryPoint()
            pt.positions = [1.0, 2.0]
            traj.points.append(pt)
            return traj

    called = {}

    def dummy_move(pos):
        called["pos"] = pos

    monkeypatch.setattr(web, "NlpNode", lambda: DummyNlp())
    monkeypatch.setattr(web, "PlannerNode", DummyPlanner)
    monkeypatch.setattr(web.robot, "move", dummy_move)

    client = TestClient(web.app)
    resp = client.post("/nlp", params={"text": "move"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert called.get("pos") == [1.0, 2.0]


def test_nlp_direct_move(monkeypatch):
    class DummyNlp:
        def _call_llm(self, text):
            return [{"joints": [3, 4]}]

    called = {}

    def dummy_move(pos):
        called["pos"] = pos

    monkeypatch.setattr(web, "NlpNode", lambda: DummyNlp())
    monkeypatch.setattr(web.robot, "move", dummy_move)

    client = TestClient(web.app)
    resp = client.post("/nlp", params={"text": "move"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert called.get("pos") == [3, 4]
