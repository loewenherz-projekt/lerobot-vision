# lerobot-vision

This repository contains a small ROS 2 workspace for the `lerobot_vision` package.  
The package demonstrates camera capture, depth computation and 3D object localisation.
Additional utilities implement stereo calibration, image rectification and a
lightweight wrapper around NVIDIA DOPE for pose estimation. Heavy weight
dependencies (YOLO3D, StereoAnywhere, MoveIt etc.) are provided as Git
submodules under `external/`.

Before building the workspace you must fetch these submodules:
`git submodule update --init --recursive`.

An optional Tkinter GUI (`VisionGUI`) can be used to preview images and run a
simple calibration wizard.

An optional Tkinter GUI (`VisionGUI`) can be used to preview images, run a
simple calibration wizard and display additional views. Rectified frames,
depth maps, disparity images, segmentation masks and overlays can be toggled on
or off using the provided checkboxes.

## Getting started

### 1. Install dependencies

Before running the setup script make sure the submodules are present:

```bash
git submodule update --init --recursive
```

After fetching the submodules run `./fetch_models.sh` to download the
pretrained DOPE and YOLO3D checkpoints required by the demo.

Verify that every `external/*` directory contains files. `setup.sh` will exit if any of them are empty.

Run `./setup.sh` once to install system requirements and build the workspace. This script assumes an Ubuntu system with ROS 2 Humble available via apt. If your ROS 2 installation lives elsewhere, set the `ROS_SETUP` environment variable to the appropriate `setup.bash` before running the script.
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

Use `./run.sh` to activate the Conda environment (if present), source ROS 2 and launch `system_launch.py` which starts the example nodes. Pass `--gui` to start the GUI alongside the pipeline. Additional options allow selecting the camera indices, loading a calibration file and enabling side‑by‑side mode for single stereo devices:
```bash
./run.sh --gui --left 2 --right 3 --config path/to/camera.yaml --side-by-side
```
If ROS lives in a non‑standard location, set `ROS_SETUP` (or `ROS_DISTRO`) before running the script so it can locate the correct `setup.bash`.

CUDA acceleration is enabled by default. Disable it with:

```bash
ros2 launch lerobot_vision system_launch.py use_cuda:=false
```

Set the `OPENAI_API_KEY` environment variable to enable the NLP node.

For a quick preview of the camera feed and a simple calibration helper you can run:

```bash
vision_gui
```


#### Using the GUI

The interface offers checkboxes to toggle rectified, depth, disparity, mask and overlay windows. Screenshots and overlay recordings can be captured via the corresponding buttons. Publishers for individual image topics can be toggled at runtime through the `toggle_publisher` service. The disparity and mask views are also published via ROS on `/stereo/disparity` and `/stereo/masks`.

##### Calibration wizard

1. Place a chessboard in view of both cameras.
2. Press **Capture** to record a stereo pair with detected corners.
3. Repeat for several poses covering the full image area.
4. Click **Review** to inspect the last detection.
5. Once enough pairs are collected, press **Calibrate** to compute the parameters. A file `calibration.yaml` will be created and an error plot displayed.

Alternatively you can run the non‑GUI helper which works with side‑by‑side stereo devices. It captures chessboard images until you press ``q`` and saves the result:

```bash
python -m lerobot_vision.calibrate_cli --device 0 --output calibration.yaml
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

Unit tests live in the `tests/` directory. Before running them you must install
the Python requirements – either create the Conda environment from
`environment.yml` or run `pip install -r requirements.txt`.
Make sure the `lerobot-vision` Conda environment is activated before running
the tests. Run them using:

```bash
pytest tests
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
ros2 run rqt_image_view rqt_image_view /stereo/disparity
ros2 run rqt_image_view rqt_image_view /stereo/masks
```

To visualize the 3D output in RViz2, launch the provided configuration:

```bash
ros2 launch lerobot_vision view_detections.launch.py
```

The visualization node publishes the raw and rectified camera frames as well as
the computed depth map. These outputs can be enabled individually via the
`publish_left_raw`, `publish_right_raw`, `publish_left_rectified`,
`publish_right_rectified`, `publish_depth`, `publish_disparity` and
`publish_masks` ROS parameters.
The overlay view combines these results into an image showing bounding boxes,
labels, distances, relative positions and pose axes for each detected object.

In addition, 3D detections are published on:
* `/stereo/disparity` (`sensor_msgs/Image`)
* `/stereo/masks` (`sensor_msgs/Image`)

* ``/robot/vision/points`` (`sensor_msgs/PointCloud2`)
* ``/robot/vision/detections`` (`vision_msgs/Detection3DArray`)

You can toggle these publishers at runtime using the `toggle_publisher` service:

```bash
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'left_raw', enable: true}"
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'left_raw', enable: false}"
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'overlay', enable: false}"
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'disparity', enable: true}"
ros2 service call /visualization_node/toggle_publisher \
  lerobot_vision/srv/TogglePublisher "{publisher: 'masks', enable: false}"
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
