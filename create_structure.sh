#!/usr/bin/env bash
set -euo pipefail

# Root-Ordner
ROOT_DIR="."
WS_SRC="$ROOT_DIR/ws/src/lerobot_vision"

echo "Erstelle Projektstruktur unter $ROOT_DIR …"

# 1. Top-Level
mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

# 2. libs-Ordner (leer, zur späteren Nutzung)
mkdir -p libs

# 3. workspace-Struktur
mkdir -p ws/src/lerobot_vision/launch
mkdir -p ws/src/lerobot_vision/lerobot_vision

# 4. package.xml
cat > ws/src/lerobot_vision/package.xml << 'EOF'
<?xml version="1.0"?>
<package format="3">
  <name>lerobot_vision</name>
  <version>0.1.0</version>
  <description>Headless Roboter-Vision mit LLM-gesteuerter Pick&amp;Place</description>
  <maintainer email="you@example.com">Ihr Name</maintainer>
  <license>Apache-2.0</license>

  <!-- Build- und Laufzeit-Abhängigkeiten -->
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>vision_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_msgs</depend>
  <depend>cv_bridge</depend>
  <depend>image_transport</depend>
  <depend>ament_index_cpp</depend>

  <!-- Externe Python-Pakete -->
  <exec_depend>openyolo3d</exec_depend>
  <exec_depend>stereoanywhere</exec_depend>
  <exec_depend>lerobot</exec_depend>
  <exec_depend>isaac_ros_pose_estimation</exec_depend>
</package>
EOF

# 5. CMakeLists.txt
cat > ws/src/lerobot_vision/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.8)
project(lerobot_vision)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
EOF

# 6. launch/system_launch.py
cat > ws/src/lerobot_vision/launch/system_launch.py << 'EOF'
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='stereoanywhere', executable='stereo_anywhere_node', name='stereo_anywhere'),
        Node(package='isaac_ros_pose_estimation', executable='dope_pose_estimation', name='isaac_dope'),
        Node(package='lerobot_vision', executable='visualization_node', name='yolo3d_viz'),
        Node(package='lerobot_vision', executable='nlp_node', name='nlp'),
        Node(package='lerobot_vision', executable='planner_node', name='planner'),
        Node(package='lerobot_vision', executable='control_node', name='controller'),
    ])
EOF

# 7. Python-Module

# camera_interface.py
cat > ws/src/lerobot_vision/lerobot_vision/camera_interface.py << 'EOF'
import cv2
import numpy as np

class StereoCamera:
    def __init__(self, left_topic='/camera/left/image_raw', right_topic='/camera/right/image_raw'):
        self.left_cap = cv2.VideoCapture(0)
        self.right_cap = cv2.VideoCapture(1)
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((5,))

    def get_frames(self):
        ret_l, left = self.left_cap.read()
        ret_r, right = self.right_cap.read()
        if not ret_l or not ret_r:
            raise RuntimeError("Kamerafehler")
        left = cv2.undistort(left, self.camera_matrix, self.dist_coeffs)
        right = cv2.undistort(right, self.camera_matrix, self.dist_coeffs)
        return left, right

    def release(self):
        self.left_cap.release()
        self.right_cap.release()
EOF

# depth_engine.py
cat > ws/src/lerobot_vision/lerobot_vision/depth_engine.py << 'EOF'
import numpy as np
from stereoanywhere import StereoAnywhere

class DepthEngine:
    def __init__(self, model_path=None):
        self.engine = StereoAnywhere(pretrained=True)

    def compute_depth(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        depth_map = self.engine.infer(left_img, right_img)
        return depth_map
EOF

# yolo3d_engine.py
cat > ws/src/lerobot_vision/lerobot_vision/yolo3d_engine.py << 'EOF'
import numpy as np
from openyolo3d import OpenYolo3D

class Yolo3DEngine:
    def __init__(self, checkpoint_dir: str):
        self.model = OpenYolo3D(checkpoint=checkpoint_dir)

    def segment(self, images: list, depth_map: np.ndarray):
        masks, labels = self.model.segment(images, depth_map)
        return masks, labels
EOF

# fusion.py
cat > ws/src/lerobot_vision/lerobot_vision/fusion.py << 'EOF'
import rclpy
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

class FusionModule:
    def __init__(self, node: rclpy.node.Node):
        self.pub_pc = node.create_publisher(PointCloud2, '/robot/vision/points', 1)
        self.pub_det = node.create_publisher(Detection3DArray, '/robot/vision/detections', 1)

    def publish(self, masks, labels, poses):
        pc2 = self._masks_to_pointcloud2(masks)
        dets = self._make_detections(masks, labels, poses)
        self.pub_pc.publish(pc2)
        self.pub_det.publish(dets)

    def _masks_to_pointcloud2(self, masks):
        raise NotImplementedError

    def _make_detections(self, masks, labels, poses):
        dets = Detection3DArray()
        return dets
EOF

# nlp_node.py
cat > ws/src/lerobot_vision/lerobot_vision/nlp_node.py << 'EOF'
import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String

class NlpNode(Node):
    def __init__(self):
        super().__init__('nlp_node')
        self.pub = self.create_publisher(String, '/robot/vision/actions', 1)
        self.create_subscription(String, '/robot/vision/scene', self.scene_cb, 1)

    def scene_cb(self, msg: String):
        scene_json = msg.data
        actions = self._call_llm(scene_json)
        out = String()
        out.data = json.dumps(actions)
        self.pub.publish(out)

    def _call_llm(self, scene_json: str):
        return [
            {"action": "detect", "object": "roter Ball"},
            {"action": "move_to", "target": "roter Ball"},
            {"action": "grasp", "object": "roter Ball"},
            {"action": "move_to", "target": "grüne Box"},
            {"action": "release", "object": "roter Ball"}
        ]
EOF

# planner_node.py
cat > ws/src/lerobot_vision/lerobot_vision/planner_node.py << 'EOF'
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.create_subscription(String, '/robot/vision/actions', self.actions_cb, 1)
        self.pub = self.create_publisher(JointTrajectory, '/arm_controller/trajectory', 1)

    def actions_cb(self, msg: String):
        actions = msg.data
        plan = self._plan_actions(actions)
        traj = JointTrajectory()
        self.pub.publish(traj)

    def _plan_actions(self, actions_json: str):
        return []
EOF

# control_node.py
cat > ws/src/lerobot_vision/lerobot_vision/control_node.py << 'EOF'
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from lerobot import Robot

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.robot = Robot(port='/dev/ttyUSB0', id='arm')
        self.create_subscription(JointTrajectory, '/arm_controller/trajectory', self.traj_cb, 1)

    def traj_cb(self, msg: JointTrajectory):
        for point in msg.points:
            positions = point.positions
            self.robot.move_to_joint_positions(positions)
EOF

# visualization_node.py
cat > ws/src/lerobot_vision/lerobot_vision/visualization_node.py << 'EOF'
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from lerobot_vision.yolo3d_engine import Yolo3DEngine
from lerobot_vision.camera_interface import StereoCamera

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.bridge = CvBridge()
        self.camera = StereoCamera()
        self.yolo3d = Yolo3DEngine(checkpoint_dir='libs/OpenYOLO3D/checkpoints')
        self.pub = self.create_publisher(Image, '/openyolo3d/overlay', 1)
        self.timer = self.create_timer(0.2, self.timer_cb)

    def timer_cb(self):
        left, right = self.camera.get_frames()
        depth = None
        masks, labels = self.yolo3d.segment([left, right], depth)
        overlay = self._render_overlay(left, masks, labels)
        imgmsg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        self.pub.publish(imgmsg)

    def _render_overlay(self, img, masks, labels):
        out = img.copy()
        for mask, lbl in zip(masks, labels):
            pts2d, _ = cv2.projectPoints(mask.astype('float32'),
                                         (0,0,0), (0,0,0),
                                         self.camera.camera_matrix,
                                         self.camera.dist_coeffs)
            pts2d = pts2d.reshape(-1,2).astype(int)
            cv2.polylines(out, [pts2d], True, (0,255,0), 2)
            cv2.putText(out, lbl, tuple(pts2d[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return out
EOF

# 8. Abschluss
chmod +x ws/src/lerobot_vision/launch/system_launch.py
echo "Fertig! Ordnerstruktur und Dateien wurden erstellt in: $PWD"
