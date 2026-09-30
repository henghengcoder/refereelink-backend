"""Inference backend abstractions for detector models.

The backends in this module deliberately return the model's native prediction
value.  Converting that value to ``supervision.Detections`` remains the
responsibility of the caller, which keeps this layer independent from the
player- and pitch-specific post-processing code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


BackendName = Literal["pytorch", "onnx", "tensorrt"]
Predictor = Callable[..., Any]

_BACKEND_ALIASES: dict[str, BackendName] = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "pt": "pytorch",
    "onnx": "onnx",
    "tensorrt": "tensorrt",
    "tensor_rt": "tensorrt",
    "trt": "tensorrt",
    "engine": "tensorrt",
}
_SUFFIX_BACKENDS: dict[str, BackendName] = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".engine": "tensorrt",
}


@runtime_checkable
class DetectorBackend(Protocol):
    """Common detector inference interface.

    ``predict`` intentionally has the same small argument set for every
    backend.  Device selection and model loading are backend concerns; model
    post-processing is left to the VisionCore caller.
    """

    def predict(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        half: bool = False,
        conf: float | None = None,
        iou: float | None = None,
    ) -> Any:
        """Run inference for one frame and return the native model output."""


class CallableBackend:
    """Adapt a callable fake or custom detector to :class:`DetectorBackend`.

    The wrapped callable must accept the same keyword arguments as
    ``predict``.  Keeping the adapter strict makes unsupported model APIs fail
    at the integration boundary instead of silently ignoring inference
    settings.
    """

    def __init__(self, predictor: Predictor) -> None:
        if not callable(predictor):
            raise TypeError("CallableBackend requires a callable predictor")
        self._predictor = predictor

    @property
    def predictor(self) -> Predictor:
        """Return the wrapped callable, primarily for dependency injection."""

        return self._predictor

    def predict(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        half: bool = False,
        conf: float | None = None,
        iou: float | None = None,
    ) -> Any:
        kwargs = {"imgsz": imgsz, "half": half}
        if conf is not None:
            kwargs["conf"] = conf
        if iou is not None:
            kwargs["iou"] = iou
        return self._predictor(frame, **kwargs)


def _normalize_backend(backend: str) -> BackendName:
    normalized = backend.strip().lower().replace("-", "_")
    try:
        return _BACKEND_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted({"pytorch", "onnx", "tensorrt"}))
        raise ValueError(
            f"Unsupported inference backend '{backend}'. "
            f"Supported backends: {supported}."
        ) from exc


def _backend_from_path(model_path: Path) -> BackendName:
    try:
        return _SUFFIX_BACKENDS[model_path.suffix.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUFFIX_BACKENDS))
        raise ValueError(
            f"Cannot infer inference backend from model '{model_path}'. "
            f"Supported model suffixes: {supported}; or pass backend=... explicitly."
        ) from exc


class UltralyticsBackend:
    """Run a PyTorch, ONNX, or TensorRT model through Ultralytics.

    The backend is inferred from ``.pt``/``.pth``, ``.onnx``, or ``.engine``
    when ``backend`` is omitted.  An explicit backend takes precedence over
    the suffix, which is useful when a deployment wrapper uses a non-standard
    filename.  Passing ``model`` enables dependency-free tests and custom
    model injection; otherwise Ultralytics is imported lazily and loads the
    model from disk during construction.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        backend: str | None = None,
        device: str = "cpu",
        model: object | None = None,
    ) -> None:
        self.device = device
        self.model_path = Path(model_path).expanduser() if model_path is not None else None

        if model is None:
            if self.model_path is None:
                raise FileNotFoundError(
                    "Ultralytics model file is required. Provide model_path pointing to "
                    "an existing .pt, .onnx, or .engine file."
                )
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Ultralytics model file not found: '{self.model_path}'. "
                    "Provide an existing .pt, .onnx, or .engine model file."
                )

        if backend is not None:
            self.backend: BackendName = _normalize_backend(backend)
        elif self.model_path is not None:
            self.backend = _backend_from_path(self.model_path)
        else:
            # An injected Ultralytics-compatible model has no reliable file
            # suffix.  PyTorch is the least surprising default in that case.
            self.backend = "pytorch"

        self._model = model if model is not None else self._load_model()

    @property
    def model(self) -> object:
        """Return the loaded or injected Ultralytics-compatible model."""

        return self._model

    def _load_model(self) -> object:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "UltralyticsBackend requires the 'ultralytics' package to load a model. "
                "Install the project's runtime dependencies with 'uv sync --dev'."
            ) from exc

        assert self.model_path is not None
        try:
            model = YOLO(str(self.model_path))
            # ONNX and TensorRT models manage their execution provider/device
            # internally.  ``to`` is only appropriate for PyTorch checkpoints.
            if self.backend == "pytorch" and hasattr(model, "to"):
                model = model.to(device=self.device)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.backend} Ultralytics model '{self.model_path}': {exc}"
            ) from exc

    def predict(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        half: bool = False,
        conf: float | None = None,
        iou: float | None = None,
    ) -> Any:
        """Run one inference using the shared detector call signature."""
        kwargs = {"imgsz": imgsz, "half": half, "verbose": False}
        if conf is not None:
            kwargs["conf"] = conf
        if iou is not None:
            kwargs["iou"] = iou
        try:
            return self._model(frame, **kwargs)
        except TypeError as exc:
            # Keep injected lightweight/fake models compatible when they do
            # not expose Ultralytics' optional ``verbose`` keyword.
            if not any(name in str(exc) for name in ("verbose", "conf", "iou")):
                raise
            fallback = {"imgsz": imgsz, "half": half}
            try:
                return self._model(frame, **fallback)
            except TypeError:
                return self._model(frame)


__all__ = ["BackendName", "CallableBackend", "DetectorBackend", "UltralyticsBackend"]
