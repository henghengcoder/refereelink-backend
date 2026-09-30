# Architecture

This document describes the real-time soccer analysis pipeline after the
Web dashboard migration. It is intended for engineers who need to extend
or maintain the system.

## Goals

- **Real-time, low-latency**: process the latest frame, drop older ones
  on backpressure, never let the display or the browser control the
  inference loop.
- **Modular**: a single, large generator function has been replaced by
  independent components that can be unit-tested in isolation.
- **Bounded resources**: events, log lines, and frame buffers are all
  bounded to keep memory usage flat in long-running deployments.
- **No Base64 / no in-browser YOLO**: video frames are streamed over
  the dedicated MJPEG endpoint or future WebRTC; WebSocket only carries
  structured JSON.
- **RTSP-aware**: the same pipeline can be pointed at a local file or
  a wired IP camera; auto-reconnect is built in.

## Components

### `app/state/`

| File | Purpose |
|------|---------|
| `models.py`  | Pydantic models for `FrameState`, `GameEvent`, `PlayerState`, `MetricsSnapshot`, `PipelineConfig`, `PipelineCommand` and the `HomographyStatus` / `PlayerRole` / `SourceStatus` enums. These are the wire format between the backend and the dashboard; the TypeScript declarations in `web/src/types/messages.ts` mirror them. |
| `store.py`   | Thread-safe `StateStore`. Holds the latest frame state, the current config, the bounded event log (max 500), the bounded log buffer (max 200 lines), the latest metrics snapshot, and the source/pipeline flags. |
| `events.py`  | A small synchronous `EventBus` for component-internal signalling. The dashboard does not subscribe to it directly. |

### `app/pipeline/`

| File | Purpose |
|------|---------|
| `source.py`  | `VideoSource` abstract base + `LocalFileSource` + `RTSPSource`. The RTSP source attempts a GStreamer pipeline first (`rtspsrc latency=0`), then falls back to OpenCV/FFmpeg. The constructor is non-blocking; the first `read()` triggers the actual capture. On read failure it transparently reconnects up to 10 times. |
| `buffer.py`  | `BoundedFrameBuffer` with two modes: `REALTIME` (drop oldest, never block) and `OFFLINE` (block producer when full). `PipelineMode` is passed in by the caller. |
| `engine.py`  | `InferencePipeline` delegates undistortion, official YOLOv11 person detection, ByteTrack, low-frequency pitch inference, homography reuse, and bottom-center projection to `VisionCore`. The internal thread runs `with torch.inference_mode():` and emits structured state into the `StateStore`. |
| `recorder.py`| Optional `VideoRecorder` for writing final annotated pipeline frames to an MP4 debug artifact. |

### `app/vision/core.py`

`VisionCore.process(frame, frame_index)` is the shared interface for all
player/pitch modes. Official YOLOv11 weights expose only COCO `person`, so
the output uses `role=unknown` and `team_id=-1` until custom role/team models
are introduced. `InferencePipeline` adds an optional trajectory semantic layer:
role-aware checkpoints and the NumPy HSV/Lab team classifier can update labels
at a lower frequency, with bounded temporal voting and UNKNOWN fallback. Pitch
keypoints run on frame 1 and then every configured interval (default 5); skipped
frames reuse the last homography for at most 0.5 seconds.

Ball detection and ball projection were removed; `FrameState` carries no
`ball` or `possession_track_id` fields.

### `app/events/engine.py`

Consumes `FrameState` entities and emits deduplicated, explainable event
candidates for offside geometry (attacker beyond the second-last defender). The
engine is stateful but lightweight; it records involved track IDs and evidence
fields, and deliberately does not present geometric candidates as final
referee decisions. `FoulEventAdapter` normalizes an optional MVFoul prediction
into the same event schema.

### `app/geometry/camera.py`

Loads the chessboard calibration `.npz`, scales intrinsics for proportional
resolution changes, rejects incompatible aspect ratios, and bypasses safely
when the calibration file is not present. All downstream modes consume the
same rectified frame.

### `app/services/publisher.py`

`WebSocketPublisher` is a thin async wrapper around a set of
`fastapi.WebSocket` clients. `push_loop` is started per connection and
emits each new `FrameState` at most once per connection plus a
`MetricsSnapshot` every 1 s. JPEG encoding happens once in the state store;
all MJPEG clients read the latest encoded bytes.
`handle_command_text` applies `start`, `stop`, and `update_config`
commands.

### `app/server/`

The FastAPI app is built in `app/server/main.py`. It registers
the four REST routers, the WebSocket router, and the `GET /video/stream`
MJPEG generator. The MJPEG encoder re-uses the latest raw frame held in
`StateStore._latest_raw_frame` (set by the pipeline once per processed
frame), so it does not perform any extra decoding.

## Real-time vs offline

- **Real-time mode** is the default. The buffer drops the oldest frame
  when full so the display never lags behind the camera. Phase one keeps
  role and team fields unresolved (`UNKNOWN` and `-1`).
- **Offline mode** blocks the producer when the buffer is full. Used
  when the caller needs every frame processed in order (e.g. training
  set generation).

## Timestamps

Every frame carries:

- `capture_timestamp_ms` — wall-clock time when the source handed the
  frame to the pipeline (`time.time() * 1000` at the top of
  `_run_loop`).
- `processed_timestamp_ms` — wall-clock time when the pipeline
  finished processing the frame.
- The `InferencePipeline` also tracks `inference_latency_ms` (the
  interval between model forward and result) and the running
  `end_to_end_latency_ms` (capture → processed) for the metrics
  snapshot.

Phase-one and phase-two metrics additionally include player/pitch/semantic inference latency,
pitch detection count, homography reuse ratio, and homography available
ratio, track-ID interruption count, memory, and GPU memory. `tools/benchmark_phase1.py`
evaluates YOLOv11n/s, `imgsz=640/960`, pitch intervals `1/5/10`, and an
undistortion-off ablation.

These are **software** timestamps only. For multi-camera PTP
synchronization, see the *Future Work* section.

## Communication

The dashboard never receives video over WebSocket and never performs
any inference in the browser. Two independent channels:

| Channel | Direction | Payload |
|---------|-----------|---------|
| WebSocket `/ws/state` | server → client | `FrameState`, `MetricsSnapshot` |
| WebSocket `/ws/state` | client → server | `{command, params}` (start, stop, update_config) |
| HTTP `GET /video/stream` | server → client | MJPEG (multipart/x-mixed-replace) |
| HTTP REST `/api/*` | both | JSON |

The next iteration will replace MJPEG with WebRTC. The MJPEG endpoint
is intentionally narrow and side-effect free, so swapping it does not
require any pipeline change.

## Tests

The current full suite contains 104 tests:

- 17 pre-existing tests for pitch config, projection,
  radar dashboard, runtime helpers, and the view transformer.
- 8 Pydantic model tests (serialization roundtrips, null field
  coordinates, status enums).
- 6 BoundedFrameBuffer tests (realtime drop, offline block, close
  unblocks waiters, latest-only read).
- 12 StateStore + EventBus tests (thread safety, bounded buffers,
  subscriber exception isolation, snapshot API).
- 8 video source tests (local read, factory routing, RTSP
  non-blocking constructor, status reporting).
- 7 FastAPI REST tests (health, status, config GET/PUT, events,
  pipeline start/stop, video stream endpoint).
- 4 WebSocket tests (connect, frame_state push, start command,
  update_config command).
- 3 smoke-integration tests (synthetic video, state pipeline,
  buffer pipeline).
- Phase-two semantic and entity-integration tests.
- Phase-three event candidate and foul-adapter tests.
- Phase-four backend adapter and shared JPEG-cache tests.

Run with:

```bash
uv run pytest tests/ -v
```

## Future work

- **WebRTC signalling** — replace the MJPEG endpoint with WebRTC,
  reserving `/ws/webrtc` for SDP/ICE. The pipeline does not need to
  change.
- **PTP / hardware timestamping** — add a `Clock` interface in
  `app/pipeline/source.py` and let the source override
  `capture_timestamp_ms()`. Multi-camera synchronization will go here.
- **ONNX / TensorRT** — wrap the Ultralytics `YOLO` calls in a
  `DetectorBackend` interface so FP16 and TensorRT engines can be
  selected per deployment.
- **GStreamer for offline** — the `LocalFileSource` could also
  support GStreamer pipelines for hardware-accelerated decoding on
  Jetson or other edge devices.
