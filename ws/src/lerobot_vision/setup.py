# ws/src/lerobot_vision/setup.py
from setuptools import setup

package_name = "lerobot_vision"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resources/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/system_launch.py",
                "launch/slam_system.launch.py",
                "launch/view_detections.launch.py",
            ],
        ),
        (f"share/{package_name}/config", ["config/camera.yaml"]),
        (f"share/{package_name}/rviz", ["rviz/view_detections.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maintainer",
    maintainer_email="example@example.com",
    description="LeRobot Vision package",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "visualization_node=lerobot_vision.visualization_node:main",
            "nlp_node=lerobot_vision.nlp_node:main",
            "planner_node=lerobot_vision.planner_node:main",
            "control_node=lerobot_vision.control_node:main",
            "vision_gui=lerobot_vision.gui:main",
            "slam_node=lerobot_vision.slam_node:main",
        ],
    },
)
