#!/usr/bin/env bash
set -euo pipefail

# Umgebungen aktivieren
conda activate lerobot-vision
source /opt/ros/humble/setup.bash
source ws/install/setup.bash

# Web-Video-Server (Live-Preview aller Image-Topics)
ros2 run web_video_server web_video_server &

# System-Launch
ros2 launch lerobot_vision system_launch.py
