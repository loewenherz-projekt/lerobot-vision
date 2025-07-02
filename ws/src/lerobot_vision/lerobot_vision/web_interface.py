from __future__ import annotations

"""Minimal FastAPI web interface for the stereo system."""

import cv2
import numpy as np
from fastapi import FastAPI, Response, Request, WebSocket, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pathlib import Path
import asyncio
import yaml
from fastapi.templating import Jinja2Templates

from .camera_interface import AsyncStereoCamera
from .stereo_calibrator import StereoCalibrator
from .nlp_node import NlpNode
from .planner_node import PlannerNode
from .image_rectifier import ImageRectifier
from .depth_engine import DepthEngine
from .yolo3d_engine import Yolo3DEngine
from .pose_estimator import PoseEstimator
from .object_localizer import localize_objects
import json
from .kinematics import (
    forward_kinematics,
    inverse_kinematics,
    rpy_to_matrix,
)
import logging
try:
    from lerobot import Robot
except Exception as exc:  # pragma: no cover - optional
    Robot = None
    logging.error("LeRobot import failed: %s", exc)

app = FastAPI(title="LeRobot Web")
templates = Jinja2Templates(directory="webapp/templates")


def save_calibration_yaml(data: dict, path: str = "calibration.yaml") -> None:
    """Persist calibration result to a YAML file."""
    try:
        Path(path).write_text(yaml.safe_dump(data))
    except Exception as exc:  # pragma: no cover - file IO optional
        import logging

        logging.error("Failed to save calibration: %s", exc)


class CameraManager:
    """Simple manager wrapping :class:`AsyncStereoCamera`."""

    def __init__(self) -> None:
        self.camera: AsyncStereoCamera | None = None

    def start(
        self, left: int = 0, right: int = 1, side_by_side: bool = False
    ) -> None:
        if self.camera:
            self.camera.release()
        self.camera = AsyncStereoCamera(left, right, side_by_side=side_by_side)

    def frames(self) -> tuple[bytes, bytes]:
        if not self.camera:
            raise RuntimeError("camera not started")
        left, right = self.camera.get_frames()
        _, lbuf = cv2.imencode(".jpg", left)
        _, rbuf = cv2.imencode(".jpg", right)
        return lbuf.tobytes(), rbuf.tobytes()

    def stop(self) -> None:
        if self.camera:
            self.camera.release()
            self.camera = None


manager = CameraManager()


class RobotManager:
    """Simple interface to a :class:`lerobot.Robot` instance."""

    def __init__(self) -> None:
        self.positions = [0.0] * 6
        self.port = "/dev/ttyUSB0"
        self.robot_id = 1
        self.robot: Robot | None = None

    def load_params(self, payload: str) -> None:
        """Load configuration from a YAML string or file path."""
        try:
            if Path(payload).is_file():
                data = yaml.safe_load(Path(payload).read_text())
            else:
                data = yaml.safe_load(payload)
            if isinstance(data, dict):
                self.port = data.get("port", self.port)
                self.robot_id = int(data.get("robot_id", self.robot_id))
        except Exception as exc:  # pragma: no cover - file IO optional
            logging.error("Failed to load robot params: %s", exc)
            return
        if Robot is not None:
            try:
                self.robot = Robot(self.port, self.robot_id)
            except Exception as exc:  # pragma: no cover - runtime
                logging.error("Robot init failed: %s", exc)
                self.robot = None

    def move(self, positions: list[float]) -> None:
        self.positions = positions
        if self.robot is not None:
            try:
                self.robot.move_to_joint_positions(positions)
            except Exception as exc:  # pragma: no cover - runtime
                logging.error("Robot movement failed: %s", exc)

    def get_positions(self) -> list[float]:
        if self.robot is not None:
            try:
                return list(self.robot.get_joint_positions())
            except Exception:  # pragma: no cover - runtime
                pass
        return self.positions.copy()


robot = RobotManager()


class VisualizationHelper:
    """Compute rectified, depth and overlay views similarly to ``VisualizationNode``."""

    def __init__(self) -> None:
        self.rectifier: ImageRectifier | None = None
        try:
            self.depth_engine = DepthEngine(use_cuda=False)
        except Exception:  # pragma: no cover - optional deps
            self.depth_engine = None
        try:
            ckpt = Path(__file__).resolve().parent / "resources" / "checkpoint"
            self.yolo = Yolo3DEngine(str(ckpt))
        except Exception:  # pragma: no cover - optional deps
            self.yolo = None
        try:
            self.pose = PoseEstimator()
        except Exception:  # pragma: no cover - optional deps
            self.pose = None

    def _draw_overlay(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        labels: list[str],
        depth: np.ndarray,
        poses: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Draw segmentation overlays."""
        fx = manager.camera.camera_matrix[0, 0]
        fy = manager.camera.camera_matrix[1, 1]
        cx = manager.camera.camera_matrix[0, 2]
        cy = manager.camera.camera_matrix[1, 2]

        def _proj(pt: np.ndarray) -> tuple[int, int]:
            u = int(pt[0] * fx / pt[2] + cx)
            v = int(pt[1] * fy / pt[2] + cy)
            return u, v

        for idx, (mask, label) in enumerate(zip(masks, labels)):
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            u = float(np.median(xs))
            v = float(np.median(ys))
            z = float(np.median(depth[ys, xs]))
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (int(x0), int(y0) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            info = f"{z:.2f}m {x:+.2f},{y:+.2f}"
            cv2.putText(
                image,
                info,
                (int(x0), int(y1) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            if poses and idx < len(poses) and poses[idx] is not None:
                _, quat = poses[idx]
                xq, yq, zq, wq = quat
                rot = np.array(
                    [
                        [1 - 2 * (yq ** 2 + zq ** 2), 2 * (xq * yq - zq * wq), 2 * (xq * zq + yq * wq)],
                        [2 * (xq * yq + zq * wq), 1 - 2 * (xq ** 2 + zq ** 2), 2 * (yq * zq - xq * wq)],
                        [2 * (xq * zq - yq * wq), 2 * (yq * zq + xq * wq), 1 - 2 * (xq ** 2 + yq ** 2)],
                    ]
                )
                center = np.array([x, y, z], dtype=float)
                axes = rot @ (0.05 * np.eye(3))
                for axis, color in zip(axes.T, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    pt2 = _proj(center + axis)
                    cv2.line(image, (int(u), int(v)), pt2, color, 2)
        return image

    def compute(self, left: np.ndarray, right: np.ndarray):
        if self.rectifier is None:
            h, w = left.shape[:2]
            self.rectifier = ImageRectifier(
                manager.camera.camera_matrix,
                manager.camera.dist_coeffs,
                manager.camera.camera_matrix,
                manager.camera.dist_coeffs,
                (w, h),
            )
        left_r, right_r = self.rectifier.rectify(left, right)
        depth = np.zeros(left_r.shape[:2], dtype=float)
        disp = None
        if self.depth_engine:
            try:
                depth, disp = self.depth_engine.compute_depth(
                    left_r, right_r, return_disparity=True
                )
            except Exception:
                pass
        masks: list[np.ndarray] = []
        labels: list[str] = []
        if self.yolo and depth is not None:
            try:
                masks, labels = self.yolo.segment([left_r], depth)
            except Exception:
                masks = []
                labels = []
        poses = None
        if self.pose and masks:
            try:
                poses = self.pose.estimate(left_r)
            except Exception:
                poses = None
        overlay = self._draw_overlay(left_r.copy(), masks, labels, depth, poses)
        return left_r, right_r, depth, masks, overlay


vis_helper = VisualizationHelper()


class ModelManager:
    """Manage available AI model checkpoints and settings."""

    def __init__(self, root: str = "external/checkpoints") -> None:
        self.root = Path(root)
        self.selected: dict[str, str] = {}
        self.thresholds: dict[str, float] = {}

    def list_models(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.pth"))

    def select(self, module: str, name: str, score_threshold: float | None = None) -> None:
        if name in self.list_models():
            self.selected[module] = name
            if score_threshold is not None:
                self.thresholds[module] = float(score_threshold)


models = ModelManager()


class CalibrationManager:
    """Manage stereo calibration via API."""

    def __init__(self) -> None:
        self.calibrator: StereoCalibrator | None = None

    def start(
        self, board_w: int = 7, board_h: int = 6, size: float = 1.0
    ) -> None:
        self.calibrator = StereoCalibrator((board_w, board_h), size)

    def add_pair(self) -> bool:
        if not self.calibrator or not manager.camera:
            return False
        left, right = manager.camera.get_frames()
        return self.calibrator.add_corners(left, right)

    def finish(self) -> dict[str, float] | None:
        if not self.calibrator or not manager.camera:
            return None
        props = manager.camera.get_properties()
        res = self.calibrator.calibrate((props["width"], props["height"]))
        m1, d1, m2, d2, r, t, _ = res
        return {
            "m1": m1.tolist(),
            "d1": d1.tolist(),
            "m2": m2.tolist(),
            "d2": d2.tolist(),
            "r": r.tolist(),
            "t": t.tolist(),
        }


calibration = CalibrationManager()


@app.get("/")
def index() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cameras")
def list_cameras(max_index: int = 4) -> JSONResponse:
    indices: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            indices.append(idx)
        cap.release()
    return JSONResponse(content={"cameras": indices})


@app.get("/camera_info")
def camera_info(index: int = 0) -> JSONResponse:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return Response(status_code=404)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return JSONResponse(content=info)


@app.get("/camera_modes")
def camera_modes(index: int = 0) -> JSONResponse:
    """Return capture modes supported by the camera."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return Response(status_code=404)

    modes: list[dict[str, int | str]] = []

    try:  # optional dependency
        from pymediainfo import MediaInfo  # type: ignore

        info = MediaInfo.parse(f"/dev/video{index}")
        for track in info.tracks:
            if track.track_type == "Video":
                w = int(track.width or 0)
                h = int(track.height or 0)
                fps = int(float(track.frame_rate)) if track.frame_rate else 0
                codec = track.codec_id or track.format or ""
                if w and h:
                    modes.append(
                        {
                            "width": w,
                            "height": h,
                            "fps": fps,
                            "codec": codec,
                        }
                    )
    except Exception:  # pragma: no cover - MediaInfo may not be installed
        pass

    if not modes:
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        framerates = [15, 30, 60]
        codecs = ["MJPG", "YUYV"]
        for w, h in resolutions:
            for fps in framerates:
                for codec in codecs:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    cap.set(
                        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec)
                    )
                    modes.append(
                        {
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                            "codec": codec,
                        }
                    )
    cap.release()
    return JSONResponse(content={"modes": modes})


def _mjpeg_generator(side: str = "left"):
    while True:
        lbuf, rbuf = manager.frames()
        buf = lbuf if side == "left" else rbuf
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"


@app.get("/stream/{side}")
def stream(side: str) -> StreamingResponse:
    if side not in {"left", "right"}:
        return Response(status_code=404)
    return StreamingResponse(
        _mjpeg_generator(side),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start")
def start(
    left: int = 0, right: int = 1, side_by_side: bool = False
) -> dict[str, str]:
    manager.start(left, right, side_by_side)
    return {"status": "started"}


@app.post("/stop")
def stop() -> dict[str, str]:
    manager.stop()
    return {"status": "stopped"}


@app.post("/camera_settings")
def camera_settings(
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    codec: str | None = None,
) -> dict[str, str]:
    if not manager.camera:
        return {"status": "no camera"}
    manager.camera.set_properties(width, height, fps, codec)
    return {"status": "ok"}


@app.get("/robot/positions")
def robot_positions() -> JSONResponse:
    return JSONResponse(content={"positions": robot.get_positions()})


@app.post("/robot/move")
def robot_move(positions: str) -> dict[str, str]:
    try:
        vals = [float(p) for p in positions.split(",")]
    except Exception:
        return {"status": "invalid"}
    robot.move(vals)
    return {"status": "ok"}


@app.get("/robot/fk")
def robot_fk(joints: str) -> JSONResponse:
    """Return forward kinematics for a comma separated joint list."""
    try:
        vals = [float(p) for p in joints.split(",")]
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid"})
    pos, rot = forward_kinematics(vals)
    return JSONResponse(content={"position": pos.tolist(), "orientation": rot.tolist()})


@app.get("/robot/ik")
def robot_ik(pose: str) -> JSONResponse:
    """Compute inverse kinematics for a target pose."""
    try:
        vals = [float(p) for p in pose.split(",")]
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid"})
    if len(vals) != 6:
        return JSONResponse(status_code=400, content={"error": "invalid"})
    position = vals[:3]
    matrix = rpy_to_matrix(vals[3:])
    joints = inverse_kinematics(position, matrix)
    return JSONResponse(content={"joints": joints})


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request) -> HTMLResponse:
    """Serve the minimal HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/frames/{side}")
async def ws_frames(websocket: WebSocket, side: str) -> None:
    await websocket.accept()
    while True:
        lbuf, rbuf = manager.frames()
        buf = lbuf if side == "left" else rbuf
        await websocket.send_bytes(buf)


@app.websocket("/ws/calibration/{side}")
async def ws_calibration_frames(websocket: WebSocket, side: str) -> None:
    await websocket.accept()
    while True:
        lbuf, rbuf = manager.frames()
        buf = lbuf if side == "left" else rbuf
        await websocket.send_bytes(buf)


@app.websocket("/ws/rectified/{side}")
async def ws_rectified(websocket: WebSocket, side: str) -> None:
    await websocket.accept()
    while True:
        left, right = manager.frames()
        left_r, right_r, _, _, _ = vis_helper.compute(left, right)
        frame = left_r if side == "left" else right_r
        _, buf = cv2.imencode(".jpg", frame)
        await websocket.send_bytes(buf.tobytes())


@app.websocket("/ws/depth")
async def ws_depth(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        left, right = manager.frames()
        left_r, _right_r, depth, _, _ = vis_helper.compute(left, right)
        dnorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        dcol = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".jpg", dcol)
        await websocket.send_bytes(buf.tobytes())


@app.websocket("/ws/masks")
async def ws_masks(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        left, right = manager.frames()
        left_r, _right_r, _depth, masks, _overlay = vis_helper.compute(left, right)
        mask_img = np.zeros_like(left_r)
        palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for idx, mask in enumerate(masks):
            color = palette[idx % len(palette)]
            mask_img[mask > 0] = color
        _, buf = cv2.imencode(".jpg", mask_img)
        await websocket.send_bytes(buf.tobytes())


@app.websocket("/ws/overlay")
async def ws_overlay(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        left, right = manager.frames()
        _left_r, _right_r, _depth, _masks, overlay = vis_helper.compute(left, right)
        _, buf = cv2.imencode(".jpg", overlay)
        await websocket.send_bytes(buf.tobytes())


@app.websocket("/ws/robot/positions")
async def ws_robot_positions(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        await websocket.send_json({"positions": robot.get_positions()})
        await asyncio.sleep(0.1)


@app.post("/calibration/start")
def calibration_start(
    board_w: int = 7, board_h: int = 6, size: float = 1.0
) -> dict[str, str]:
    calibration.start(board_w, board_h, size)
    return {"status": "ready"}


@app.post("/calibration/capture")
def calibration_capture() -> JSONResponse:
    ok = calibration.add_pair()
    return JSONResponse(content={"captured": bool(ok)})


@app.post("/calibration/finish")
def calibration_finish() -> JSONResponse:
    result = calibration.finish()
    if result is None:
        return JSONResponse(status_code=400, content={"error": "no data"})
    save_calibration_yaml(result)
    return JSONResponse(content=result)


modules: dict[str, dict[str, float | bool]] = {
    "yolo3d": {"enabled": False, "score_threshold": 0.5},
    "dope": {"enabled": False, "score_threshold": 0.5},
    "slam": {"enabled": False, "score_threshold": 0.5},
}


def run_inference(image: np.ndarray) -> np.ndarray:
    """Run enabled modules on the given image and draw overlays."""
    output = image.copy()
    y = 15
    for name, cfg in modules.items():
        if not cfg.get("enabled"):
            continue
        text = f"{name}:{cfg.get('score_threshold', 0.0)}"
        cv2.putText(
            output,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y += 15
    return output


@app.get("/modules")
def list_modules() -> JSONResponse:
    return JSONResponse(content=modules)


@app.post("/modules/select")
def select_module(
    name: str,
    enable: bool = True,
    score_threshold: float | None = None,
) -> dict[str, str]:
    if name not in modules:
        modules[name] = {"enabled": enable, "score_threshold": 0.5}
    else:
        modules[name]["enabled"] = enable
    if score_threshold is not None:
        modules[name]["score_threshold"] = float(score_threshold)
    return {"status": "ok"}


@app.get("/models")
def list_models() -> JSONResponse:
    return JSONResponse(
        content={
            "models": models.list_models(),
            "selected": models.selected,
            "thresholds": models.thresholds,
        }
    )


@app.post("/models/select")
def select_model(name: str) -> dict[str, str]:
    models.select("default", name)
    return {"status": "ok", "selected": models.selected.get("default")}


@app.post("/robot/params")
def robot_params(payload: str) -> dict[str, str]:
    robot.load_params(payload)
    return {"status": "ok"}


@app.post("/nlp")
def nlp(text: str) -> JSONResponse:
    node = NlpNode()
    actions = node._call_llm(text)
    executed = False
    try:
        for act in actions if isinstance(actions, list) else [actions]:
            if not isinstance(act, dict):
                continue
            if "joints" in act or "positions" in act:
                robot.move(act.get("joints") or act.get("positions"))
                executed = True
            elif "target_pose" in act:
                try:
                    planner = PlannerNode()
                    traj = planner._plan_actions(json.dumps(act))
                    for point in getattr(traj, "points", []):
                        robot.move(list(point.positions))
                        executed = True
                except Exception as exc:  # pragma: no cover - planning optional
                    logging.error("Planning failed: %s", exc)
    except Exception as exc:  # pragma: no cover - runtime
        logging.error("NLP execution error: %s", exc)
    status = "ok" if executed else "error"
    return JSONResponse(content={"actions": actions, "status": status})


@app.post("/inference/test")
async def inference_test(file: UploadFile = File(...)) -> Response:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return Response(status_code=400)
    overlay = run_inference(img)
    _, buf = cv2.imencode(".jpg", overlay)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
