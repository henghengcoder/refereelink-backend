# 快速启动指南

## 1. 环境配置

项目依赖统一由 `pyproject.toml` 和 `uv.lock` 管理。首次配置运行:

```bash
uv sync --dev
```

如需使用保留的 PySide6/QML legacy 界面，再运行:

```bash
uv sync --dev --extra desktop
```

验证环境:

```bash
uv run python -c "import cv2, torch, ultralytics; print('environment ok')"
uv run pytest -q
```

Node.js: v20+ (通过 nvm 管理)

## 2. 模型权重 (需手动下载)

当前网络无法访问 Google Drive，请从以下链接手动下载模型文件，放入 `assets/weights/`。第一阶段的球员模型使用官方 Ultralytics YOLOv11 COCO 权重，只保留 `person` 类；旧的角色模型仅作为后续/兼容用途保留:

```bash
mkdir -p assets/weights
```

| 模型 | 下载链接 | 放入路径 |
|------|----------|----------|
| 官方球员检测（推荐轻量） | https://github.com/ultralytics/assets/releases | `assets/weights/yolo11n.pt` |
| 官方球员检测（推荐默认） | https://github.com/ultralytics/assets/releases | `assets/weights/yolo11s.pt` |
| 旧版角色球员检测 | https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q | `assets/weights/football-player-detection.pt` |
| 球场检测 | https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf | `assets/weights/football-pitch-detection.pt` |

可选：犯规检测模型 `mvfoul.pth.tar` (需要 fouls_far 库支持)

## 3.1 固定广角相机标定

准备同一分辨率下拍摄的棋盘格图片后运行:

```bash
python tools/calibrate_camera.py \
  --images assets/calibration_images \
  --output assets/calibration/camera.npz
```

默认棋盘格内角点为 `9 x 6`，可用 `--pattern-cols`、`--pattern-rows`
和 `--square-size` 调整。标定文件缺失时服务仍会启动，但会明确 warning
并旁路去畸变。相机分辨率改变但宽高比一致时，内参会按比例缩放；宽高比
不一致会拒绝处理。

## 4. 示例视频 (需手动下载)

| 视频 | 下载链接 | 放入路径 |
|------|----------|----------|
| sample 1 | https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF | `assets/data/0bfacc_0.mp4` |
| sample 2 | https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf | `assets/data/2e57b9_0.mp4` |
| sample 3 | https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK- | `assets/data/08fd33_0.mp4` |
| sample 4 | https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU | `assets/data/573e61_0.mp4` |
| sample 5 | https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu | `assets/data/121364_0.mp4` |

> 也可以使用任意本地 `.mp4` 文件，不需要必须下载这些示例视频。

## 5. 启动后端 (终端 1)

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

# 方式 A: 空载启动 (推荐，从前端配置视频源)
uv run python -m app.server.main --device cuda --port 8000

# 方式 B: 直接指定视频源启动
uv run python -m app.server.main --video_source assets/data/2e57b9_0.mp4 --device cuda --port 8000
```

后端启动后会监听:
- `http://localhost:8000` — FastAPI 主服务
- `http://localhost:8000/video/stream` — MJPEG 视频流
- `ws://localhost:8000/ws/state` — WebSocket 状态推送

## 6. 启动前端 (终端 2)

```bash
cd "$(git rev-parse --show-toplevel)/web"
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
npm run dev
```

打开浏览器: `http://localhost:5173`

## 7. 使用步骤

1. 浏览器打开 `http://localhost:5173`
2. 在 CONTROL 面板的 **VIDEO SOURCE** 输入框填入视频路径，例如:
   - 本地文件: `assets/data/2e57b9_0.mp4`
   - RTSP 摄像头: `rtsp://192.168.1.100:554/stream`
3. 选择 **DEVICE** (`cpu` 或 `cuda`)
4. 勾选需要的选项 (Foul Detect, Show Keypoints)
5. 点击 **START**
6. 视频开始播放，右侧显示 2D 球场投影、状态卡片、事件告警

## 8. 常见问题

**Q: 点击 START 后显示 "video source not found"**
A: 路径不对。用绝对路径，例如 `<repo-root>/assets/data/2e57b9_0.mp4`

**Q: 点击 START 后后端日志显示 YOLO 下载进度条**
A: 模型权重 `.pt` 文件缺失。请按步骤 2 下载并放入 `assets/weights/`。

**Q: RTSP 摄像头没有画面**
A: 检查 RTSP URL 是否可达，摄像头是否在线。后端会自动重连 10 次。

**Q: 如何切换轻量模型或关键点检测频率**
A: 启动时使用 `--player_model_path assets/weights/yolo11n.pt`、
`--imgsz 640` 或 `--pitch_detection_interval 5`。也可以通过
`tools/benchmark_phase1.py` 对 `yolo11n/yolo11s`、`640/960`、
`interval=1/5/10` 进行对比。

**Q: 前端显示 "WS OFF"**
A: 后端没有启动，或者端口不对。检查 `http://localhost:8000/health` 是否返回 `{"status": "ok"}`。
