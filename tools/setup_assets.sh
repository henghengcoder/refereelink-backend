#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$DIR/.." && pwd )"
ASSETS_DIR="$REPO_ROOT/assets"
DATA_DIR="$ASSETS_DIR/data"
WEIGHTS_DIR="$ASSETS_DIR/weights"

cd "$REPO_ROOT"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DATA_DIR ]]; then
    mkdir -p "$DATA_DIR"
else
    echo "'$DATA_DIR' directory already exists."
fi

# Check if 'weights' directory does not exist and then create it
if [[ ! -e $WEIGHTS_DIR ]]; then
    mkdir -p "$WEIGHTS_DIR"
else
    echo "'$WEIGHTS_DIR' directory already exists."
fi

# download the models
python - <<'PY'
from ultralytics import YOLO

# Ultralytics downloads the official checkpoint when it is not present.
YOLO('yolo11s.pt')
PY
if [[ -f "yolo11s.pt" ]]; then
    mv -f "yolo11s.pt" "$WEIGHTS_DIR/yolo11s.pt"
fi
gdown -O "$WEIGHTS_DIR/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
gdown -O "$WEIGHTS_DIR/football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

# download the videos
gdown -O "$DATA_DIR/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$DATA_DIR/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
gdown -O "$DATA_DIR/08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
gdown -O "$DATA_DIR/573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
gdown -O "$DATA_DIR/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
