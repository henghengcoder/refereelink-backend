# RefereeLink Backend Web Dashboard

Real-time computer-vision dashboard for football (soccer) matches. The
first-stage Python pipeline rectifies wide-angle frames, runs official
YOLOv11 person detection, ByteTrack tracking, low-frequency pitch keypoint
detection, RANSAC homography reuse, and bottom-center field projection. It
pushes structured state to a React web dashboard over WebSocket. Live video
is delivered as a low-latency MJPEG stream from a dedicated FastAPI endpoint.
The phase-two entity layer adds optional trajectory-level role/team semantics
without changing the phase-one UNKNOWN fallback. Ball detection and ball
projection have been removed; the pipeline tracks and projects people only.

> The original PySide6/QML dashboard is preserved under `--mode
> RADAR_DASHBOARD_LEGACY`. The web dashboard is the recommended and only
> actively maintained front-end.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    VideoSource Layer                         │
│  LocalFileSource │ RTSPSource (GStreamer + FFmpeg fallback)  │
│  - non-blocking constructor  - auto-reconnect on drop        │
│  - status reporting via StateStore                            │
└──────────────────────┬───────────────────────────────────────┘
                       │ BGR frame
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               BoundedFrameBuffer                            │
│  - realtime mode: drop oldest, deliver latest                │
│  - offline mode: block producer when full                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              InferencePipeline                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ CameraUndistorter → VisionCore                         │ │
│  │ YOLOv11 person → ByteTrack                             │ │
│  │ pitch keypoints every N frames                         │ │
│  │ RANSAC homography / history reuse                      │ │
│  │ bottom-center → field coordinates                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Foul mode reuses CameraUndistorter and keeps its own model  │
│                          ▼                                   │
│                   FrameState (Pydantic)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │ structured JSON (no image data)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     StateStore                                │
│  thread-safe shared state: latest_frame_state, metrics,       │
│  events, config, log buffer (max 200 lines), source_status    │
└──────────┬────────────────────────────┬──────────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────┐    ┌──────────────────────────┐
│ FastAPI REST     │    │ WebSocket (/ws/state)     │
│ /health          │    │  - pushes FrameState      │
│ /api/status      │    │  - pushes Metrics (1Hz)   │
│ /api/config      │    │  - receives start/stop    │
│ /api/events      │    │    and update_config      │
│ /api/pipeline/*  │    └─────────────┬────────────┘
└────────┬─────────┘                  │
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Web Dashboard (React)                     │
│  VideoPanel (MJPEG) │ Pitch2D │ StatusCards │ EventAlerts    │
│  ControlPanel       │ LogPanel                              │
└──────────────────────────────────────────────────────────────┘
```

## Communication

- **WebSocket `/ws/state`** — push structured `FrameState` and
  `MetricsSnapshot` Pydantic models; receive control commands
  (`start`, `stop`, `update_config`).
- **HTTP `GET /video/stream`** — low-latency MJPEG stream
  (`multipart/x-mixed-replace`). Not used for control data.
- **HTTP REST** — configuration, event history, health.

Base64 video over WebSocket is **never** used. Frame data only travels
over the dedicated MJPEG endpoint or future WebRTC signalling.

## Layout

```
app/
├── state/        Pydantic models, StateStore, EventBus
├── pipeline/     VideoSource, BoundedFrameBuffer, InferencePipeline, recorder
├── services/     WebSocketPublisher
├── server/       FastAPI app (REST + WebSocket + MJPEG)
├── modes/        Legacy single-purpose modes
└── runtime.py    Pitch + radar drawing helpers

web/              React + TypeScript + Vite dashboard
├── src/
│   ├── components/  VideoPanel, Pitch2D, StatusCards, EventAlerts,
│   │                ControlPanel, LogPanel
│   ├── hooks/       useWebSocket
│   ├── store/       Zustand state container
│   ├── types/       TypeScript mirrors of Pydantic models
│   └── styles/      Dark control-console theme
├── vite.config.ts   Proxies /api, /ws, /video to FastAPI in dev
├── package.json
└── tsconfig.json

tests/            104 tests
```

## Install

```bash
# 1. Create/sync the project environment from pyproject.toml and uv.lock
uv sync --dev

# Optional: enable the legacy PySide6/QML dashboard
uv sync --dev --extra desktop

# 3. Install JavaScript dependencies for the dashboard
cd web && npm install && cd ..
```

## Run

```bash
# Download model weights and sample videos (optional)
./tools/setup_assets.sh

# Start the FastAPI backend (YOLOv11 person + pitch keypoint model)
.venv/bin/python -m app.server.main \
    --video_source assets/data/smoke_test.mp4 \
    --device cpu \
    --inference_backend auto

# Optional: save the annotated pipeline output for debugging
.venv/bin/python -m app.server.main \
    --video_source assets/data/smoke_test.mp4 \
    --device cpu \
    --enable_recording \
    --target_video_path debug/recordings/smoke_test.mp4

# In another terminal: launch the React dev server
cd web && npm run dev
# Open http://localhost:5173
```

For RTSP sources:

```bash
.venv/bin/python -m app.server.main \
    --video_source rtsp://192.168.1.100:554/stream \
    --device cuda
```

`--inference_backend` accepts `auto`, `pytorch`, `onnx`, or `tensorrt`.
The backend is inferred from `.pt`, `.onnx`, and `.engine` model suffixes when
set to `auto`.

The frontend dev server proxies `/api`, `/ws`, and `/video` to the
backend on port 8000.

For a production build:

```bash
cd web && npm run build
# The compiled bundle lands in web/dist
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/health`               | Liveness probe |
| GET    | `/api/status`           | Pipeline, source, metrics snapshot |
| GET    | `/api/config`           | Current pipeline config |
| PUT    | `/api/config`           | Update config (e.g. enable foul detection) |
| GET    | `/api/events`           | Recent events (`?limit=50&offset=0`) |
| POST   | `/api/pipeline/start`   | Start inference |
| POST   | `/api/pipeline/stop`    | Stop inference |
| GET    | `/api/pipeline/recording` | Download/seek the optional annotated debug video |
| GET    | `/video/stream`         | MJPEG live video |
| WS     | `/ws/state`             | Structured state push + commands |

## Modes

| Mode | Description |
|------|-------------|
| `SERVER`                  | New web dashboard mode (recommended) |
| `RADAR_DASHBOARD_LEGACY`  | PySide6 + QML fallback |
| `RADAR`                   | Offline radar processing, writes annotated video |
| `PLAYER_DETECTION`        | YOLO player detection only |
| `PITCH_DETECTION`         | YOLO pitch keypoint detection only |
| `PLAYER_TRACKING`         | YOLO + ByteTrack |
| `TEAM_CLASSIFICATION`     | Shared YOLOv11 person + ByteTrack view; role/team remain UNKNOWN/-1 in phase 1 |
| `FOUL_DETECTION`          | MVFoul rolling window with HUD overlay |

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover camera calibration, VisionCore scheduling/projection, trajectory semantics and pipeline entity integration, Pydantic models, the bounded buffer (realtime drop
+ offline block), the video source abstraction (factory routing,
reconnect state), the StateStore and EventBus (thread safety,
bounded buffers, exception isolation), the FastAPI REST endpoints,
the WebSocket lifecycle and commands, the legacy pitch/team
core, and a synthetic-video smoke test for the full pipeline.
