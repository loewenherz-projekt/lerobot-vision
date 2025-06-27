# lerobot-vision

This repository contains a small ROS 2 workspace for the `lerobot_vision` package.  
The package demonstrates camera capture, depth computation and 3D object localisation.
Additional utilities implement stereo calibration, image rectification and a
lightweight wrapper around NVIDIA DOPE for pose estimation. Heavy weight
dependencies (YOLO3D, StereoAnywhere, MoveIt etc.) are provided as Git
submodules under `external/`.

An optional Tkinter GUI (`VisionGUI`) can be used to preview images and run a
simple calibration wizard.

## Getting started

### 1. Install dependencies

Run `./setup.sh` once to install system requirements, initialise submodules and build the workspace. This script assumes an Ubuntu system with ROS 2 Humble available via apt.

### 2. Create the Conda environment

Create and activate the `lerobot-vision` Conda environment:

```bash
conda env create -f environment.yml
conda activate lerobot-vision
```

If you do not use Conda, install the Python requirements manually:

```bash
pip install -r requirements.txt
```

Always activate the environment before running tests or `run.sh`.


### 3. Running the demo

Use `./run.sh` to activate the Conda environment (if present), source ROS 2 and launch `system_launch.py` which starts the example nodes.

For a quick preview of the camera feed and a simple calibration helper you can run:

```bash
python -m lerobot_vision.gui
```

### 4. Project structure

```
ws/
  src/
    lerobot_vision/          # ROS package
external/                    # third‑party submodules
```

Key modules include:

- ``StereoCalibrator`` – assists with intrinsic and extrinsic calibration
- ``ImageRectifier`` – rectifies image pairs using calibration results
- ``DepthEngine`` – uses CUDA SGBM by default with optional StereoAnywhere fallback
- ``Yolo3DEngine`` – object detection via OpenYOLO3D
- ``PoseEstimator`` – thin wrapper around NVIDIA DOPE
- ``ObjectLocalizer`` – fuses masks, depth and poses into 3D coordinates

The `create_structure.sh` helper can recreate the basic directory tree and some boilerplate files.

### 5. Tests

Unit tests live in `ws/src/lerobot_vision/tests`. Run them using:

Make sure the `lerobot-vision` Conda environment is activated before running the tests.

```bash
pytest
```

The tests rely on lightweight stubs for ROS messages so they do not require a full ROS installation during development.

### 6. CI

The GitHub workflow at `.github/workflows/ci.yml` builds the workspace with `colcon`, runs the linters and executes the unit tests with coverage reporting via Codecov.

### 7. Visualization

Image topics can be previewed with `rqt_image_view`:

```bash
ros2 run rqt_image_view rqt_image_view /openyolo3d/overlay
ros2 run rqt_image_view rqt_image_view /stereo/depth
ros2 run rqt_image_view rqt_image_view /stereo/left_raw
ros2 run rqt_image_view rqt_image_view /stereo/right_raw
ros2 run rqt_image_view rqt_image_view /stereo/left_rectified
ros2 run rqt_image_view rqt_image_view /stereo/right_rectified
```

The visualization node publishes the raw and rectified camera frames as well as
the computed depth map. These outputs can be enabled individually via the
`publish_left_raw`, `publish_right_raw`, `publish_left_rectified`,
`publish_right_rectified` and `publish_depth` ROS parameters.

You can toggle these publishers at runtime using the `toggle_publisher` service:

```bash
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'left_raw', enable: true}"
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'left_raw', enable: false}"
```

Start `web_video_server` to stream a topic in the browser:

```bash
rosrun web_video_server web_video_server
```

Then open `http://<robot_ip>:8080/stream?topic=/openyolo3d/overlay` in your browser.

## Troubleshooting

* Ensure submodules are checked out (`git submodule update --init --recursive`).
* When running on a fresh system, install ROS 2 Humble before executing the scripts.


## License

This project is licensed under the [Apache 2.0 License](LICENSE).
