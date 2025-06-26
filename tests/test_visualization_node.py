from unittest import mock

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from lerobot_vision.visualization_node import VisualizationNode


def test_on_timer(monkeypatch):
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.StereoCamera",
        mock.Mock(
            return_value=mock.Mock(
                get_frames=mock.Mock(
                    return_value=(
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.DepthEngine",
        mock.Mock(
            return_value=mock.Mock(
                compute_depth=mock.Mock(
                    return_value=np.zeros((1, 1), dtype=np.float32)
                )  # noqa: E501
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.Yolo3DEngine",
        mock.Mock(
            return_value=mock.Mock(
                segment=mock.Mock(
                    return_value=([np.zeros((1, 1), dtype=np.uint8)], ["obj"])
                )
            )
        ),
    )
    monkeypatch.setattr(
        "lerobot_vision.visualization_node.CvBridge",
        mock.Mock(
            return_value=mock.Mock(
                cv2_to_imgmsg=mock.Mock(return_value=Image())
            )  # noqa: E501
        ),
    )

    rclpy.init(args=None)
    node = VisualizationNode("/tmp")
    node.pub.publish = mock.Mock()
    node._on_timer()
    node.pub.publish.assert_called_once()
    rclpy.shutdown()
