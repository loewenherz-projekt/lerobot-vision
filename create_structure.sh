#!/usr/bin/env bash
set -e

# Create basic workspace layout
mkdir -p ws/src/lerobot_vision/lerobot_vision
mkdir -p ws/src/lerobot_vision/launch
mkdir -p ws/src/lerobot_vision/config
mkdir -p ws/src/lerobot_vision/tests

# Touch boilerplate files
cat <<'EOPY' > ws/src/lerobot_vision/lerobot_vision/__init__.py
EOPY

cat <<'EOPY' > ws/src/lerobot_vision/launch/system_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='lerobot_vision', executable='visualization_node')
    ])
EOPY

cat <<'EOPY' > ws/src/lerobot_vision/setup.py
from setuptools import setup

setup(package_dir={'': 'lerobot_vision'})
EOPY
