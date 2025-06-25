#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. System & ROS 2 Humble Installation ==="
sudo apt update                                                                 # 
sudo apt install -y curl gnupg2 lsb-release build-essential git unzip wget       # 
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu \
  $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo apt-key add -
sudo apt update
sudo apt install -y ros-humble-desktop                                           # 
source /opt/ros/humble/setup.bash

echo "=== 2. Conda-Umgebung erstellen ==="
conda create -y -n lerobot-vision python=3.10                                     # 
conda activate lerobot-vision
conda install -y ffmpeg -c conda-forge                                            # 

echo "=== 3. Repositories klonen ==="
mkdir -p libs && cd libs
git clone https://github.com/aminebdj/OpenYOLO3D.git                               # :contentReference[oaicite:9]{index=9}
git clone https://github.com/bartn8/stereoanywhere.git                             # :contentReference[oaicite:10]{index=10}
git clone https://github.com/huggingface/lerobot.git                               # 
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git  # 
cd ..

echo "=== 4. OpenYOLO3D installieren & Modelle herunterladen ==="
cd libs/OpenYOLO3D
conda env update -n lerobot-vision -f environment.yml                              # :contentReference[oaicite:13]{index=13}
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

# MinkowskiEngine für 3D-Masken
cd third_party
git clone --recursive https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --force_cuda --blas=openblas
cd ../../..

# Pretrained Mask3D-Checkpoint
mkdir -p data/OpenYOLO3D/models
wget -O data/OpenYOLO3D/models/mask3d_checkpoint.pth \
  https://kaldir.vc.in.tum.de/mask3d_checkpoint.pth                            # :contentReference[oaicite:14]{index=14}

# ScanNet200-Daten (Preprocessed)
mkdir -p data/scannet200
wget -P data/scannet200 \
  https://kaldir.vc.in.tum.de/scannet_benchmark/scannet200.zip
unzip data/scannet200/scannet200.zip -d data/scannet200                            # :contentReference[oaicite:15]{index=15}
cd ..

echo "=== 5. Stereo Anywhere installieren & Checkpoint holen ==="
cd libs/stereoanywhere
pip install -r requirements.txt                                                   # :contentReference[oaicite:16]{index=16}
python setup.py install

# Pretrained Sceneflow-Weights
mkdir -p ../pretrained_models/stereoanywhere
wget -O ../pretrained_models/stereoanywhere/sceneflow.pth \
  https://github.com/bartn8/stereoanywhere/releases/download/v1.0/sceneflow.pth     # :contentReference[oaicite:17]{index=17}
cd ../..

echo "=== 6. LeRobot SDK installieren ==="
cd libs/lerobot
pip install -e .[feetech]                                                         # 
cd ../..

echo "=== 7. ROS-2 Workspace & Isaac ROS Pose Estimation ==="
mkdir -p ws/src && cd ws/src
ln -s ../../libs/isaac_ros_pose_estimation .                                     # 

# Beispiel: DOPE-Modelle von NGC (via NVIDIA-Container-Registry)
mkdir -p isaac_ros_pose_estimation/models
# Nutzer muss hier seine NGC-Token eintragen oder öffentlich verfügbare Modelle nutzen
wget -O isaac_ros_pose_estimation/models/dope_model.engine \
  https://api.ngc.nvidia.com/v2/models/nvidia/isaac_ros_dope/versions/1.0/zip    # 

# Eigene ROS-2-Pakete kopieren
# cp -r ../../your_ros2_packages/lerobot_vision .

cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

echo "✅ Setup vollständig!
• conda activate lerobot-vision
• source /opt/ros/humble/setup.bash && source ws/install/setup.bash"
