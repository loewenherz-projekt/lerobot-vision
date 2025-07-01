#!/usr/bin/env bash
set -e

# Prepare the directory for pretrained checkpoints. The archives must be
# downloaded manually because the original URLs no longer exist.
mkdir -p external/checkpoints
cd external/checkpoints


# The original checkpoints were hosted on GitHub but have since been removed.
# If the archives are not present, instruct the user to fetch them manually from
# their current locations.
if [ ! -f dope.tgz ]; then
    echo "DOPE weights not found. Please download them from:" >&2
    echo "  https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg" >&2
    exit 1
fi
if [ ! -f yolo3d.tgz ]; then
    echo "YOLO3D weights not found. Please download them from the OpenYOLO3D release page." >&2
    exit 1
fi

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
