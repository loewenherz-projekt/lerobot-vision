"""Simple GUI for camera preview and calibration."""  # pragma: no cover

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path


import cv2
from PIL import Image, ImageTk
import numpy as np

from .camera_interface import AsyncStereoCamera
from .stereo_calibrator import StereoCalibrator


class VisionGUI:  # pragma: no cover - GUI helper
    """Tkinter-based GUI to preview stereo images and run calibration."""

    def __init__(self, camera: AsyncStereoCamera) -> None:  # pragma: no cover
        self.camera = camera
        self.calibrator = StereoCalibrator()
        self.root = tk.Tk()
        self.root.title("LeRobot Vision")
        self.left_label = tk.Label(self.root)
        self.left_label.pack(side=tk.LEFT)
        self.right_label = tk.Label(self.root)
        self.right_label.pack(side=tk.LEFT)
        btn = tk.Button(
            self.root,
            text="Capture Corners",
            command=self._capture,
        )
        btn.pack(fill=tk.X)
        btn2 = tk.Button(self.root, text="Calibrate", command=self._calibrate)
        btn2.pack(fill=tk.X)
        self._running = True
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self) -> None:  # pragma: no cover
        while self._running:
            try:
                left, right = self.camera.get_frames()
            except Exception:
                continue
            self._show_image(left, self.left_label)
            self._show_image(right, self.right_label)
            self.root.update_idletasks()
            self.root.update()

    def _show_image(
        self, img: np.ndarray, widget: tk.Label
    ) -> None:  # pragma: no cover
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = ImageTk.PhotoImage(Image.fromarray(rgb))
        widget.configure(image=im)
        widget.image = im

    def _capture(self) -> None:  # pragma: no cover
        try:
            left, right = self.camera.get_frames()
            self.calibrator.add_corners(left, right)
        except Exception:
            pass

    def _calibrate(self) -> None:  # pragma: no cover
        if not self.calibrator.objpoints:
            return
        h, w = self.camera.get_frames()[0].shape[:2]
        m1, d1, m2, d2, r, t = self.calibrator.calibrate((w, h))
        save_path = Path("calibration.yaml")
        self.calibrator.save(str(save_path), m1, d1, m2, d2, r, t)

    def run(self) -> None:  # pragma: no cover
        self.root.mainloop()
        self._running = False
        self.camera.release()
