from __future__ import annotations

"""Minimal FastAPI web interface for the stereo system."""

import cv2
from fastapi import FastAPI, Response, Request, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pathlib import Path
import asyncio
import yaml
from fastapi.templating import Jinja2Templates

from .camera_interface import AsyncStereoCamera
from .stereo_calibrator import StereoCalibrator
from .nlp_node import NlpNode

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
    """Very small manager mimicking robot joint control."""

    def __init__(self) -> None:
        self.positions = [0.0] * 6

    def move(self, positions: list[float]) -> None:
        self.positions = positions

    def get_positions(self) -> list[float]:
        return self.positions.copy()


robot = RobotManager()


class ModelManager:
    """Manage available AI model checkpoints."""

    def __init__(self, root: str = "external/checkpoints") -> None:
        self.root = Path(root)
        self.selected: str | None = None

    def list_models(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.pth"))

    def select(self, name: str) -> None:
        if name in self.list_models():
            self.selected = name


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


modules: dict[str, bool] = {"yolo3d": False, "dope": False, "slam": False}


@app.get("/modules")
def list_modules() -> JSONResponse:
    return JSONResponse(content=modules)


@app.post("/modules/select")
def select_module(name: str, enable: bool = True) -> dict[str, str]:
    modules[name] = enable
    return {"status": "ok"}


@app.get("/models")
def list_models() -> JSONResponse:
    return JSONResponse(
        content={"models": models.list_models(), "selected": models.selected}
    )


@app.post("/models/select")
def select_model(name: str) -> dict[str, str]:
    models.select(name)
    return {"status": "ok", "selected": models.selected}


@app.post("/robot/params")
def robot_params(payload: str) -> dict[str, str]:
    # stub for robot configuration
    return {"status": "ok"}


@app.post("/nlp")
def nlp(text: str) -> JSONResponse:
    node = NlpNode()
    actions = node._call_llm(text)
    return JSONResponse(content={"actions": actions})
