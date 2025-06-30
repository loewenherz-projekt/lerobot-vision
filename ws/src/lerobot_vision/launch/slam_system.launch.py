"""Launch demo nodes including SLAM."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    left_arg = DeclareLaunchArgument(
        "left", default_value="0", description="Left camera index"
    )
    right_arg = DeclareLaunchArgument(
        "right", default_value="1", description="Right camera index"
    )
    sbs_arg = DeclareLaunchArgument(
        "side_by_side",
        default_value="false",
        description="Single device providing side-by-side frames",
    )
    config_arg = DeclareLaunchArgument(
        "camera_config",
        default_value="$(find-pkg-share lerobot_vision)/config/camera.yaml",
        description="Path to calibration file",
    )
    map_arg = DeclareLaunchArgument(
        "map_output", default_value="", description="Optional map output"
    )
    return LaunchDescription(
        [
            left_arg,
            right_arg,
            sbs_arg,
            config_arg,
            map_arg,
            Node(
                package="stereoanywhere",
                executable="stereo_anywhere_node",
                name="stereo_anywhere",
            ),
            Node(
                package="isaac_ros_pose_estimation",
                executable="dope_pose_estimation",
                name="isaac_dope",
            ),
            Node(
                package="lerobot_vision",
                executable="visualization_node",
                name="yolo3d_viz",
                parameters=[
                    {"camera_config": LaunchConfiguration("camera_config")},
                    {"left_idx": LaunchConfiguration("left")},
                    {"right_idx": LaunchConfiguration("right")},
                    {"side_by_side": LaunchConfiguration("side_by_side")},
                ],
            ),
            Node(
                package="lerobot_vision",
                executable="nlp_node",
                name="nlp",
            ),
            Node(
                package="lerobot_vision",
                executable="planner_node",
                name="planner",
            ),
            Node(
                package="lerobot_vision",
                executable="control_node",
                name="controller",
            ),
            Node(
                package="lerobot_vision",
                executable="slam_node",
                name="slam",
                parameters=[
                    {"camera_config": LaunchConfiguration("camera_config")},
                    {"left_idx": LaunchConfiguration("left")},
                    {"right_idx": LaunchConfiguration("right")},
                    {"side_by_side": LaunchConfiguration("side_by_side")},
                    {"map_output": LaunchConfiguration("map_output")},
                ],
            ),
        ]
    )
