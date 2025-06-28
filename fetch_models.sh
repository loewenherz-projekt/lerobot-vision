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
        if ! curl -L -f -o "$out" "$url"; then
            echo "Failed to download $url"
            exit 1
        fi
    fi
}

fetch "https://github.com/NVlabs/Deep_Object_Pose/releases/download/v1.0/dope.tgz" dope.tgz
fetch "https://github.com/ericwonghaha/OpenYOLO3D/releases/download/v1.0/yolo3d.tgz" yolo3d.tgz

for archive in *.tgz; do
    dir="${archive%.tgz}"
    if [ ! -d "$dir" ]; then
        echo "Extracting $archive..."
        if ! tar -xzf "$archive"; then
            echo "Error extracting $archive"
            exit 1
        fi
    fi
done
