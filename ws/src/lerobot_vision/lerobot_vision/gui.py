"""Simple GUI for camera preview and calibration."""  # pragma: no cover

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path


import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

from .depth_engine import DepthEngine
from .image_rectifier import ImageRectifier
from .pose_estimator import PoseEstimator
from .yolo3d_engine import Yolo3DEngine
from .object_localizer import localize_objects

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

        # Optional views
        self.show_rect_var = tk.BooleanVar(value=False)
        self.show_depth_var = tk.BooleanVar(value=False)
        self.show_overlay_var = tk.BooleanVar(value=False)
        self.show_mask_var = tk.BooleanVar(value=False)
        self.show_disp_var = tk.BooleanVar(value=False)

        tk.Checkbutton(
            self.root,
            text="Rectified",
            variable=self.show_rect_var,
            command=self._toggle_rect,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            self.root,
            text="Depth",
            variable=self.show_depth_var,
            command=self._toggle_depth,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            self.root,
            text="Disparity",
            variable=self.show_disp_var,
            command=self._toggle_disp,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            self.root,
            text="Masks",
            variable=self.show_mask_var,
            command=self._toggle_masks,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            self.root,
            text="Overlay",
            variable=self.show_overlay_var,
            command=self._toggle_overlay,
        ).pack(side=tk.LEFT)

        self.rect_window: tk.Toplevel | None = None
        self.depth_window: tk.Toplevel | None = None
        self.overlay_window: tk.Toplevel | None = None
        self.mask_window: tk.Toplevel | None = None
        self.disp_window: tk.Toplevel | None = None

        self.rect_left_label: tk.Label | None = None
        self.rect_right_label: tk.Label | None = None
        self.depth_label: tk.Label | None = None
        self.overlay_label: tk.Label | None = None
        self.mask_label: tk.Label | None = None
        self.disp_label: tk.Label | None = None

        self.rectifier: ImageRectifier | None = None
        self.depth_engine = DepthEngine(use_cuda=False)
        try:
            ckpt = Path(__file__).resolve().parent / "resources" / "checkpoint"
            self.yolo_engine = Yolo3DEngine(str(ckpt))
        except Exception:
            self.yolo_engine = None
        try:
            self.pose_estimator = PoseEstimator()
        except Exception:
            self.pose_estimator = None

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

            compute_extra = (
                self.show_rect_var.get()
                or self.show_depth_var.get()
                or self.show_overlay_var.get()
                or self.show_mask_var.get()
                or self.show_disp_var.get()
            )
            if compute_extra:
                if self.rectifier is None:
                    h, w = left.shape[:2]
                    self.rectifier = ImageRectifier(
                        self.camera.camera_matrix,
                        self.camera.dist_coeffs,
                        self.camera.camera_matrix,
                        self.camera.dist_coeffs,
                        (w, h),
                    )
                left_r, right_r = self.rectifier.rectify(left, right)
            else:
                left_r, right_r = left, right

            if self.show_rect_var.get() and self.rect_window:
                self._show_image(left_r, self.rect_left_label)
                self._show_image(right_r, self.rect_right_label)

            if (
                self.show_depth_var.get()
                or self.show_overlay_var.get()
                or self.show_mask_var.get()
                or self.show_disp_var.get()
            ) and self.depth_engine:
                try:
                    if self.show_disp_var.get():
                        depth, disp = self.depth_engine.compute_depth(
                            left_r, right_r, return_disparity=True
                        )
                    else:
                        depth = self.depth_engine.compute_depth(left_r, right_r)
                        disp = None
                except Exception:
                    depth = np.zeros_like(left_r[:, :, 0], dtype=float)
                    disp = np.zeros_like(depth)
            else:
                depth = None
                disp = None

            if (
                self.show_depth_var.get()
                and self.depth_window
                and depth is not None
            ):
                dnorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                dcol = cv2.applyColorMap(
                    dnorm.astype(np.uint8), cv2.COLORMAP_JET
                )
                self._show_image(dcol, self.depth_label)

            masks = None
            labels = None
            if (
                (self.show_overlay_var.get() or self.show_mask_var.get())
                and depth is not None
                and self.yolo_engine
            ):
                try:
                    masks, labels = self.yolo_engine.segment([left_r], depth)
                except Exception:
                    masks = None
                    labels = None

            if (
                self.show_overlay_var.get()
                and self.overlay_window
                and depth is not None
                and masks is not None
                and labels is not None
                and self.pose_estimator
            ):
                try:
                    poses = self.pose_estimator.estimate(left_r)
                    _ = localize_objects(
                        masks, depth, self.camera.camera_matrix, labels, poses
                    )
                    overlay = self._draw_overlay(
                        left_r.copy(), masks, labels, depth, poses
                    )
                    self._show_image(overlay, self.overlay_label)
                except Exception:
                    pass

            if (
                self.show_mask_var.get()
                and self.mask_window
                and masks is not None
            ):
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
                self._show_image(mask_img, self.mask_label)

            if (
                self.show_disp_var.get()
                and self.disp_window
                and disp is not None
            ):
                dnorm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
                dcol = cv2.applyColorMap(
                    dnorm.astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                self._show_image(dcol, self.disp_label)
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

    def _toggle_rect(self) -> None:  # pragma: no cover - runtime GUI
        if self.show_rect_var.get():
            self.rect_window = tk.Toplevel(self.root)
            self.rect_left_label = tk.Label(self.rect_window)
            self.rect_left_label.pack(side=tk.LEFT)
            self.rect_right_label = tk.Label(self.rect_window)
            self.rect_right_label.pack(side=tk.LEFT)
        elif self.rect_window is not None:
            self.rect_window.destroy()
            self.rect_window = None

    def _toggle_depth(self) -> None:  # pragma: no cover - runtime GUI
        if self.show_depth_var.get():
            self.depth_window = tk.Toplevel(self.root)
            self.depth_label = tk.Label(self.depth_window)
            self.depth_label.pack()
        elif self.depth_window is not None:
            self.depth_window.destroy()
            self.depth_window = None

    def _toggle_overlay(self) -> None:  # pragma: no cover - runtime GUI
        if self.show_overlay_var.get():
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_label = tk.Label(self.overlay_window)
            self.overlay_label.pack()
        elif self.overlay_window is not None:
            self.overlay_window.destroy()
            self.overlay_window = None

    def _toggle_masks(self) -> None:  # pragma: no cover - runtime GUI
        if self.show_mask_var.get():
            self.mask_window = tk.Toplevel(self.root)
            self.mask_label = tk.Label(self.mask_window)
            self.mask_label.pack()
        elif self.mask_window is not None:
            self.mask_window.destroy()
            self.mask_window = None

    def _toggle_disp(self) -> None:  # pragma: no cover - runtime GUI
        if self.show_disp_var.get():
            self.disp_window = tk.Toplevel(self.root)
            self.disp_label = tk.Label(self.disp_window)
            self.disp_label.pack()
        elif self.disp_window is not None:
            self.disp_window.destroy()
            self.disp_window = None

    def _draw_overlay(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        labels: list[str],
        depth: np.ndarray,
        poses: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:  # pragma: no cover - runtime drawing
        fx = self.camera.camera_matrix[0, 0]
        fy = self.camera.camera_matrix[1, 1]
        cx = self.camera.camera_matrix[0, 2]
        cy = self.camera.camera_matrix[1, 2]

        def _project(pt: np.ndarray) -> tuple[int, int]:
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
                        [
                            1 - 2 * (yq**2 + zq**2),
                            2 * (xq * yq - zq * wq),
                            2 * (xq * zq + yq * wq),
                        ],
                        [
                            2 * (xq * yq + zq * wq),
                            1 - 2 * (xq**2 + zq**2),
                            2 * (yq * zq - xq * wq),
                        ],
                        [
                            2 * (xq * zq - yq * wq),
                            2 * (yq * zq + xq * wq),
                            1 - 2 * (xq**2 + yq**2),
                        ],
                    ]
                )
                center = np.array([x, y, z], dtype=float)
                axes = rot @ (0.05 * np.eye(3))
                for axis, color in zip(
                    axes.T,
                    [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
                ):
                    pt2 = _project(center + axis)
                    cv2.line(image, (int(u), int(v)), pt2, color, 2)
        return image

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


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``vision_gui`` executable."""
    parser = argparse.ArgumentParser(description="Simple camera GUI")
    parser.add_argument(
        "--config",
        help="Path to calibration YAML file",
        default=None,
    )
    parser.add_argument(
        "--left",
        type=int,
        default=0,
        help="Left camera device index",
    )
    parser.add_argument(
        "--right",
        type=int,
        default=1,
        help="Right camera device index",
    )
    opts = parser.parse_args(args)
    cam = AsyncStereoCamera(opts.left, opts.right, config_path=opts.config)
    VisionGUI(cam).run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
