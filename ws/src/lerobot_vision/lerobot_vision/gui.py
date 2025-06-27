"""Simple GUI for camera preview and calibration."""  # pragma: no cover

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path


import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

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

        self.status_var = tk.StringVar()
        self.status = tk.Label(self.root, textvariable=self.status_var)
        self.status.pack(fill=tk.X)

        capture_btn = tk.Button(
            self.root,
            text="Capture Corners",
            command=self._capture,
        )
        capture_btn.pack(fill=tk.X)

        self.prev_btn = tk.Button(
            self.root, text="Previous", command=self.prev_step
        )
        self.prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.next_btn = tk.Button(
            self.root, text="Next", command=self.next_step
        )
        self.next_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.steps = [
            "Capture corner pairs",
            "Review detected patterns",
            "Reprojection error",
        ]
        self.step_idx = 0
        self.captured_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.errors: list[tuple[float, float]] = []

        self._running = True
        self._update_status()
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
            if self.calibrator.add_corners(left, right):
                self.captured_pairs.append((left.copy(), right.copy()))
        except Exception:
            pass

    def _calibrate(self) -> None:  # pragma: no cover
        if not self.calibrator.objpoints:
            return
        h, w = self.camera.get_frames()[0].shape[:2]
        (
            m1,
            d1,
            m2,
            d2,
            r,
            t,
            self.errors,
        ) = self.calibrator.calibrate((w, h), return_errors=True)
        save_path = Path("calibration.yaml")
        self.calibrator.save(str(save_path), m1, d1, m2, d2, r, t)
        self._show_error_plot()

    def _review(self) -> None:  # pragma: no cover
        if not self.captured_pairs:
            return
        left, right = self.captured_pairs[-1]
        corners_l = self.calibrator.left_points[-1]
        corners_r = self.calibrator.right_points[-1]
        cv2.drawChessboardCorners(
            left, self.calibrator.board_size, corners_l, True
        )
        cv2.drawChessboardCorners(
            right, self.calibrator.board_size, corners_r, True
        )
        self._show_image(left, self.left_label)
        self._show_image(right, self.right_label)

    def _show_error_plot(self) -> None:  # pragma: no cover
        if not self.errors:
            return
        l_err = [e[0] for e in self.errors]
        r_err = [e[1] for e in self.errors]
        plt.figure()
        plt.plot(l_err, label="left")
        plt.plot(r_err, label="right")
        plt.xlabel("Image Pair")
        plt.ylabel("Reprojection Error")
        plt.legend()
        plt.show(block=False)
        plt.close()

    def _update_status(self) -> None:  # pragma: no cover
        step = self.step_idx + 1
        total = len(self.steps)
        text = f"Step {step}/{total}: {self.steps[self.step_idx]}"
        self.status_var.set(text)

    def next_step(self) -> None:  # pragma: no cover
        if self.step_idx >= len(self.steps) - 1:
            return
        self.step_idx += 1
        if self.step_idx == 1:
            self._review()
        elif self.step_idx == 2:
            self._calibrate()
        self._update_status()

    def prev_step(self) -> None:  # pragma: no cover
        if self.step_idx == 0:
            return
        self.step_idx -= 1
        self._update_status()

    def run(self) -> None:  # pragma: no cover
        self.root.mainloop()
        self._running = False
        self.camera.release()
