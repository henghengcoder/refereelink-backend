"""Lightweight event candidate state machine for phase three.

The engine intentionally emits *candidates*.  It uses the stable entities and
field coordinates from phase two, but does not claim that a geometric rule is
the final officiating decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.state.models import FrameState, GameEvent, PlayerRole


@dataclass(frozen=True)
class EventEngineConfig:
    event_cooldown_s: float = 1.0
    offside_enabled: bool = True


class EventEngine:
    """Generate deduplicated offside candidates from projected player positions.

    Ball detection was removed from the pipeline, so ball-driven candidates
    (possession, pass, shot) are no longer produced and offside geometry only
    checks the attacker against the second-last defender.
    """

    def __init__(self, config: EventEngineConfig = EventEngineConfig()) -> None:
        self.config = config
        self._last_event_timestamp: dict[str, float] = {}

    def reset(self) -> None:
        self._last_event_timestamp.clear()

    def update(self, frame: FrameState) -> list[GameEvent]:
        timestamp = float(frame.capture_timestamp_ms) / 1000.0
        if not self.config.offside_enabled:
            return []
        return self._offside_candidates(frame, timestamp)

    def _offside_candidates(self, frame: FrameState, timestamp: float) -> list[GameEvent]:
        players = [
            player
            for player in frame.players
            if player.team_id in (0, 1)
            and player.role != PlayerRole.REFEREE
            and player.field_x is not None
        ]
        results: list[GameEvent] = []
        for attacking_team in (0, 1):
            attackers = [p for p in players if p.team_id == attacking_team]
            defenders = [p for p in players if p.team_id != attacking_team]
            if len(defenders) < 2:
                continue
            defenders.sort(key=lambda player: float(player.field_x))
            second_last = (
                defenders[-2] if attacking_team == 0 else defenders[1]
            )
            line_x = float(second_last.field_x)
            for attacker in attackers:
                attacker_x = float(attacker.field_x)
                beyond_line = attacker_x > line_x if attacking_team == 0 else attacker_x < line_x
                if not beyond_line:
                    continue
                results.extend(
                    self._emit(
                        event_type="offside_candidate",
                        confidence=0.5,
                        timestamp=timestamp,
                        frame=frame,
                        involved=[attacker.track_id, second_last.track_id],
                        field_xy=(attacker_x, attacker.field_y),
                        evidence={
                            "attacking_team": attacking_team,
                            "attacker_x": attacker_x,
                            "second_last_defender_x": line_x,
                        },
                    )
                )
        return results

    def _emit(
        self,
        *,
        event_type: str,
        confidence: float,
        timestamp: float,
        frame: FrameState,
        involved: list[int],
        field_xy: tuple[Optional[float], Optional[float]],
        evidence: dict[str, Any],
    ) -> list[GameEvent]:
        last = self._last_event_timestamp.get(event_type)
        if last is not None and timestamp - last < self.config.event_cooldown_s:
            return []
        self._last_event_timestamp[event_type] = timestamp
        return [
            GameEvent(
                event_type=event_type,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                severity="candidate",
                timestamp=timestamp,
                frame_id=frame.frame_id,
                field_x=field_xy[0],
                field_y=field_xy[1],
                involved_track_ids=involved,
                evidence=evidence,
            )
        ]


class FoulEventAdapter:
    """Convert an MVFoul-like prediction object into a wire event."""

    def __init__(self, confidence_threshold: float = 0.48) -> None:
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

    def update(
        self,
        prediction: Any,
        *,
        frame_id: int,
        timestamp: float,
        field_xy: Optional[tuple[float, float]] = None,
    ) -> Optional[GameEvent]:
        if prediction is None:
            return None
        confidence = _prediction_confidence(prediction)
        if confidence < self.confidence_threshold:
            return None
        details = _prediction_details(prediction)
        return GameEvent(
            event_type="foul_candidate",
            confidence=confidence,
            severity="candidate",
            timestamp=timestamp,
            frame_id=frame_id,
            field_x=field_xy[0] if field_xy else None,
            field_y=field_xy[1] if field_xy else None,
            foul_details=details,
            evidence={"source": "mvfoul"},
        )


def _prediction_confidence(prediction: Any) -> float:
    for name in ("confidence", "probability", "score", "offence_confidence"):
        value = getattr(prediction, name, None)
        if value is not None:
            try:
                return float(np.clip(float(value), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    return 0.0


def _prediction_details(prediction: Any) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name in ("offence", "action", "label", "severity"):
        value = getattr(prediction, name, None)
        if value is not None:
            details[name] = value.value if hasattr(value, "value") else value
    return details
