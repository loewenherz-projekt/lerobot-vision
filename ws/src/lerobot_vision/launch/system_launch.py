# File: ws/src/lerobot_vision/launch/system_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the system launch description."""
    return LaunchDescription(
        [
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
            ),
            Node(package="lerobot_vision", executable="nlp_node", name="nlp"),
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
        ]
    )
