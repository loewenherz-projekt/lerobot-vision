"""LeRobot vision package."""

from .auto_deps import ensure as _ensure_deps  # noqa: E402

_ensure_deps()  # noqa: E402

from .camera_interface import StereoCamera  # noqa: E402
from .image_rectifier import ImageRectifier  # noqa: E402
from .depth_engine import DepthEngine  # noqa: E402
from .yolo3d_engine import Yolo3DEngine  # noqa: E402
from .pose_estimator import PoseEstimator  # noqa: E402
from .object_localizer import localize_objects  # noqa: E402
from .fusion import FusionModule  # noqa: E402
from .stereo_calibrator import StereoCalibrator  # noqa: E402
from .calibrate_cli import main as calibrate_cli  # noqa: E402
from .slam_node import SlamNode  # noqa: E402
from .kinematics import forward_kinematics, inverse_kinematics  # noqa: E402
from .robot_controller import RobotController  # noqa: E402

__all__ = [
    "StereoCamera",
    "ImageRectifier",
    "DepthEngine",
    "Yolo3DEngine",
    "PoseEstimator",
    "localize_objects",
    "FusionModule",
    "StereoCalibrator",
    "calibrate_cli",
    "SlamNode",
    "RobotController",
    "forward_kinematics",
    "inverse_kinematics",
]
