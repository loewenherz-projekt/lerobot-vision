#!/usr/bin/env bash
set -e

# Fetch submodules before doing anything else
git submodule update --init

TARGET_DIR="./external/OpenYOLO3D/models/Mask3D/third_party/MinkowskiEngine"

if [ -d "$TARGET_DIR" ]; then
    # Ordner existiert
    if [ "$(ls -A "$TARGET_DIR")" ]; then
        echo "MinkowskiEngine existiert und ist **nicht leer** – git clone wird übersprungen."
    else
        echo "MinkowskiEngine existiert, ist aber **leer** – wird gelöscht und neu gecloned."
        rm -rf "$TARGET_DIR"
        git clone https://github.com/NVIDIA/MinkowskiEngine "$TARGET_DIR"
    fi
else
    echo "MinkowskiEngine existiert **nicht** – wird neu gecloned."
    git clone https://github.com/NVIDIA/MinkowskiEngine "$TARGET_DIR"
fi

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
  echo "Submodule checkout failed. Please run 'git submodule update --init' manually." >&2
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

# Verify Python packages are available
if ! python3 - <<'EOF'
import numpy, cv2, pytest
EOF
then
  echo "Some Python packages are missing. Run 'pip install -r requirements.txt' or create the Conda environment from environment.yml." >&2
fi

# Build the workspace
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO}/setup.bash}"

if [ -f "$ROS_SETUP" ]; then
  source "$ROS_SETUP"
else
  echo "ROS 2 distribution '$ROS_DISTRO' not found at $ROS_SETUP" >&2
  echo "Please install ROS 2 or set ROS_SETUP to the correct path." >&2
  exit 1
fi
colcon build --symlink-install --workspace ws
