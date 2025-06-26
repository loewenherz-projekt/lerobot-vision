cmake_minimum_required(VERSION 3.8)
project(lerobot_vision)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    lerobot_vision/camera_interface.py
    lerobot_vision/control_node.py
    lerobot_vision/depth_engine.py
    lerobot_vision/fusion.py
    lerobot_vision/nlp_node.py
    lerobot_vision/planner_node.py
    lerobot_vision/visualization_node.py
    lerobot_vision/yolo3d_engine.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
