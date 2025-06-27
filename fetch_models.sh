#!/usr/bin/env bash
set -e

# Download DOPE and YOLO3D checkpoints into external/checkpoints
mkdir -p external/checkpoints
cd external/checkpoints

fetch() {
    url="$1"
    out="$2"
    if [ ! -f "$out" ]; then
        echo "Downloading $out..."
        curl -L -o "$out" "$url"
    fi
}

fetch "https://github.com/NVlabs/Deep_Object_Pose/releases/download/v1.0/dope.tgz" dope.tgz
fetch "https://github.com/ericwonghaha/OpenYOLO3D/releases/download/v1.0/yolo3d.tgz" yolo3d.tgz

for archive in *.tgz; do
    dir="${archive%.tgz}"
    if [ ! -d "$dir" ]; then
        tar -xzf "$archive"
    fi
done
