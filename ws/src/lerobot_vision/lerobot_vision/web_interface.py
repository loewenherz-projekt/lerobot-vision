from __future__ import annotations

"""Minimal FastAPI web interface for the stereo system."""

import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse

from .camera_interface import AsyncStereoCamera

app = FastAPI(title="LeRobot Web")


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
