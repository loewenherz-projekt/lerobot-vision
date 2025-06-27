#!/usr/bin/env bash
set -e

# Install system dependencies for ROS2 and Python tooling
sudo apt update && sudo apt install -y git python3 python3-pip curl

# Install ROS 2 Humble
if ! command -v ros2 >/dev/null; then
  sudo apt install -y ros-humble-desktop
fi

# Initialize rosdep
sudo rosdep init 2>/dev/null || true
rosdep update

# Fetch submodules
git submodule update --init --recursive

# Warn if any submodule directories are empty
for d in external/*; do
  if [ -d "$d" ] && [ -z "$(ls -A "$d")" ]; then
    echo "Warning: submodule directory '$d' is empty. Did git fetch succeed?"
  fi
done

# Install Python requirements
pip install -r requirements.txt

# Build the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --workspace ws
