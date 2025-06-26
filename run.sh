#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate lerobot-vision || true

# Launch the system
ros2 launch lerobot_vision system_launch.py
