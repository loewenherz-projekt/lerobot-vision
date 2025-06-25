# lerobot-vision

This repository contains a ROS 2 workspace with several submodules. To
initialize all dependencies and build the workspace run:

```bash
git submodule update --init --recursive
conda env update -f environment.yml -n lerobot-vision
conda activate lerobot-vision
cd ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```
