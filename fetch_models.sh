#!/usr/bin/env bash
set -e

# Prepare the directory for pretrained checkpoints and ensure ``gdown`` is
# available for downloading from Google Drive.
mkdir -p external/checkpoints
cd external/checkpoints

if ! command -v gdown >/dev/null; then
    pip install --no-cache-dir gdown
fi

# Download DOPE weights (YCB and HOPE) if not already present.
if [ ! -d dope ]; then
    echo "Downloading DOPE checkpoints..."
    gdown --folder --remaining-ok \
        https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg \
        -O dope_tmp
    mv dope_tmp dope
fi

# Download YOLO3D checkpoints if needed.
if [ ! -d yolo3d ]; then
    echo "Downloading YOLO3D checkpoints..."
    gdown 1FneLaYaClWDO51L9lIvlTQbheh5SfOFD -O OpenYOLO3D.zip
    unzip -q OpenYOLO3D.zip
    mv OpenYOLO3D/checkpoints yolo3d
    rm -rf OpenYOLO3D OpenYOLO3D.zip
fi
