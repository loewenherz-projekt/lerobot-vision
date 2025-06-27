#!/usr/bin/env bash
set -e

# Fetch submodules before doing anything else
git submodule update --init --recursive

# Fetch pretrained checkpoints for YOLO3D and DOPE
./fetch_models.sh || true

# Exit if any external directory is empty
empty=0
for d in external/*; do
  if [ -d "$d" ] && [ -z "$(ls -A "$d")" ]; then
    echo "Error: submodule directory '$d' is empty. Did git fetch succeed?" >&2
    empty=1
  fi
done
if [ $empty -ne 0 ]; then
  echo "Submodule checkout failed. Please run 'git submodule update --init --recursive' manually." >&2
  exit 1
fi

# Install system dependencies for ROS2 and Python tooling
sudo apt update && sudo apt install -y git python3 python3-pip curl

# Install ROS 2 Humble
if ! command -v ros2 >/dev/null; then
  sudo apt install -y ros-humble-desktop
fi

# Initialize rosdep
sudo rosdep init 2>/dev/null || true
rosdep update

# Install Python requirements
pip install -r requirements.txt

# Build the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --workspace ws
