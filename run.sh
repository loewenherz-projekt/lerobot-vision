#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate lerobot-vision || true

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
