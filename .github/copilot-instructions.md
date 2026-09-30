# RefereeLink Backend contribution guidance

RefereeLink is a real-time football computer-vision and assisted-officiating backend. Keep changes evidence-based and preserve the boundary between model evidence, deterministic rules, candidate events, and human review.

## Protect these contracts

- Treat `app/state/models.py` as the source of truth for `FrameState`, `MetricsSnapshot`, `GameEvent`, `PlayerState`, and pipeline configuration. When a field, enum, default, nullability, or serialized name changes, update the REST/WebSocket tests and the corresponding files under `web/src/types/`.
- Keep `/ws/state` structured JSON only. The MJPEG video stream is separate at `/video/stream`; do not put base64 video or raw frames in state messages.
- Preserve explicit `UNKNOWN`, `NONE`, `unavailable`, `stale`, and error/degradation states. Never fabricate a valid detection, team, geometry, event, or metric when the model or input is unavailable.
- Keep team identity and player role as separate semantic dimensions. Do not infer a team merely because a role or track exists.

## Protect real-time behavior

- Do not block the inference thread with UI, network, disk, or unbounded logging work.
- Keep buffers bounded and preserve the distinction between real-time frame dropping and offline blocking behavior.
- For changes involving concurrency, cancellation, reconnects, or lifecycle, check thread safety, bounded buffering, resource cleanup, and cancellation paths.
- For geometry, tracking, inference, or event changes, add focused tests and benchmark evidence where performance or accuracy is claimed.

## Keep production and experiments separate

- Do not mix experimental code from `experiments/` or `notebooks/` into `app/` without an explicit production boundary and tests.
- Do not add hard-coded local paths, IP addresses, credentials, model paths, secrets, raw videos, model weights, databases, or generated assets to the repository.
- Do not claim a performance improvement, accuracy result, or deployment validation without an executed benchmark or field-test artifact.
