# ws/src/lerobot_vision/lerobot_vision/depth_engine.py
"""Depth engine using CUDA acceleration with optional CPU fallback."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from stereoanywhere.models.stereo_anywhere import StereoAnywhere
except Exception as exc:  # pragma: no cover - external dep may not exist
    StereoAnywhere = Any  # type: ignore
    logging.error("StereoAnywhere import failed: %s", exc)

try:  # pragma: no cover - optional module
    import cv2
    import cv2.ximgproc as ximgproc
except Exception as exc:  # pragma: no cover - when opencv-contrib is absent
    cv2 = None  # type: ignore
    ximgproc = None  # type: ignore
    logging.error("OpenCV CUDA/ximgproc import failed: %s", exc)


class DepthEngine:
    """Compute depth from stereo images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        use_cuda: bool = True,
        baseline: float = 1.0,
        focal: float = 1.0,
    ) -> None:
        """Create the depth engine.

        Args:
            model_path: Optional path to a trained ``StereoAnywhere`` model.
            use_cuda: Use CUDA accelerated SGBM. Set ``False`` for CPU
                fallback.
        """
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.baseline = baseline
        self.focal = focal
        self.cuda_matcher = None
        self.wls_filter = None
        if not use_cuda and isinstance(StereoAnywhere, type):
            try:
                if model_path is not None:
                    self.engine = StereoAnywhere(model_path=model_path)
                else:
                    self.engine = StereoAnywhere(pretrained=True)
            except Exception as exc:  # pragma: no cover - runtime path
                self.engine = None
                logging.error("StereoAnywhere init failed: %s", exc)
        else:
            self.engine = None
            if not use_cuda:
                logging.error("StereoAnywhere is unavailable")

    def compute_depth(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        return_disparity: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
        """Compute a depth map from a pair of images.

        Args:
            left: Left image array.
            right: Right image array.

        Returns:
            The computed depth map. When ``return_disparity`` is ``True`` a
            tuple ``(depth, disparity)`` is returned. ``disparity`` will be
            ``None`` when disparity computation is unavailable.
        """
        if not isinstance(left, np.ndarray) or not isinstance(
            right, np.ndarray
        ):
            logging.error("Invalid input to compute_depth")
            raise ValueError("Invalid input")

        if self.engine is not None and not self.use_cuda:
            try:
                depth = self.engine.infer(left, right)
                return (depth, None) if return_disparity else depth
            except Exception as exc:  # pragma: no cover - runtime path
                logging.error("Depth computation failed: %s", exc)
                raise

        if cv2 is None:
            logging.error("CUDA depth computation unavailable")
            raise RuntimeError("CUDA unavailable")
        try:
            if self.cuda_matcher is None:
                self.cuda_matcher = cv2.cuda.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=16,
                    blockSize=5,
                )
                if ximgproc is not None:
                    self.wls_filter = ximgproc.createDisparityWLSFilterGeneric(
                        False
                    )
            disparity = self.cuda_matcher.compute(left, right)
            if self.wls_filter is not None:
                disparity = self.wls_filter.filter(disparity, left)
            disparity = disparity.astype(np.float32)
            with np.errstate(divide="ignore"):
                depth = (self.focal * self.baseline) / disparity
            depth[disparity == 0] = 0
            if return_disparity:
                return depth, disparity
            return depth
        except Exception as exc:  # pragma: no cover - runtime path
            logging.error("Depth computation failed: %s", exc)
            raise
