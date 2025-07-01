#!/usr/bin/env bash
set -e

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO}/setup.bash}"

if [ -f "$ROS_SETUP" ]; then
    # Source the detected ROS 2 setup file
    source "$ROS_SETUP"
else
    echo "ROS 2 distribution '$ROS_DISTRO' not found at $ROS_SETUP" >&2
    echo "Please install ROS 2 or set ROS_SETUP to the correct path." >&2
    exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate lerobot-vision || true

# Also source the workspace if available
WS_SETUP="$(dirname "$0")/ws/install/setup.bash"
if [ -f "$WS_SETUP" ]; then
    source "$WS_SETUP"
else
    echo "Warning: workspace not built. Run ./setup.sh first." >&2
fi

# Launch the system and optionally the GUI
GUI=false
LEFT=0
RIGHT=1
CONFIG=""
SIDE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --gui)
            GUI=true
            shift
            ;;
        --left)
            LEFT=$2
            shift 2
            ;;
        --right)
            RIGHT=$2
            shift 2
            ;;
        --config)
            CONFIG=$2
            shift 2
            ;;
        --side-by-side)
            SIDE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done
if [[ "$GUI" == true ]]; then
    vision_gui --left "$LEFT" --right "$RIGHT" --config "$CONFIG" \
        $( [[ "$SIDE" == true ]] && echo "--side-by-side" ) &
fi
ros2 launch lerobot_vision system_launch.py left:=$LEFT right:=$RIGHT \
    camera_config:="$CONFIG" side_by_side:=$SIDE
