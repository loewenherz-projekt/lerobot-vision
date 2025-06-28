#!/usr/bin/env python3
"""Simple command line utility to calibrate a side-by-side stereo camera."""  # pragma: no cover

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .camera_interface import StereoCamera
from .stereo_calibrator import StereoCalibrator


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Guided stereo calibration")
    parser.add_argument(
        "--device", type=int, default=0, help="Side-by-side camera device index"
    )
    parser.add_argument(
        "--board-width", type=int, default=7, help="Number of inner corners per chessboard row"
    )
    parser.add_argument(
        "--board-height", type=int, default=6, help="Number of inner corners per chessboard column"
    )
    parser.add_argument(
        "--square-size", type=float, default=1.0, help="Chessboard square size in your preferred units"
    )
    parser.add_argument(
        "--output", default="calibration.yaml", help="Destination for calibration file"
    )
    opts = parser.parse_args(args)

    cam = StereoCamera(left_idx=opts.device, side_by_side=True)
    calib = StereoCalibrator((opts.board_width, opts.board_height), opts.square_size)

    print("\nPress SPACE to capture a pair, Q to finish.\n")
    count = 0
    try:
        while True:
            left, right = cam.get_frames()
            preview = cv2.hconcat([left, right])
            cv2.imshow("stereo", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if calib.add_corners(left, right):
                    count += 1
                    print(f"Captured pair {count}")
                else:
                    print("Chessboard not detected. Try again.")
            elif key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if count == 0:
        print("No pairs captured. Exiting.")
        return

    m1, d1, m2, d2, r, t, _ = calib.calibrate(left.shape[:2][::-1], return_errors=True)
    calib.save(opts.output, m1, d1, m2, d2, r, t)
    print(f"Calibration saved to {opts.output}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
