# lerobot-vision

This repository contains a minimal ROS 2 workspace for the `lerobot_vision` package. The package demonstrates camera capture, depth computation, object segmentation and robot control. All heavy weight dependencies (YOLO3D, StereoAnywhere, MoveIt etc.) are provided as Git submodules under `external/`.

## Getting started

### 1. Install dependencies

Run `./setup.sh` once to install system requirements, initialise submodules and build the workspace. This script assumes an Ubuntu system with ROS 2 Humble available via apt.

### 2. Running the demo

Use `./run.sh` to activate the Conda environment (if present), source ROS 2 and launch `system_launch.py` which starts the example nodes.

### 3. Project structure

```
ws/
  src/
    lerobot_vision/          # ROS package
external/                    # third‑party submodules
```

The `create_structure.sh` helper can recreate the basic directory tree and some boilerplate files.

### 4. Tests

Unit tests live in `ws/src/lerobot_vision/tests`. Run them using:

```bash
pytest
```

The tests rely on lightweight stubs for ROS messages so they do not require a full ROS installation during development.

### 5. CI

The GitHub workflow at `.github/workflows/ci.yml` builds the workspace with `colcon`, runs the linters and executes the unit tests with coverage reporting via Codecov.

## Troubleshooting

* Ensure submodules are checked out (`git submodule update --init --recursive`).
* When running on a fresh system, install ROS 2 Humble before executing the scripts.

