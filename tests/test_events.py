from __future__ import annotations

from types import SimpleNamespace

from app.events.engine import EventEngine, EventEngineConfig, FoulEventAdapter
from app.state.models import FrameState, PlayerRole, PlayerState


def _player(track_id: int, team_id: int, x: float, y: float = 3000.0) -> PlayerState:
    return PlayerState(
        track_id=track_id,
        role=PlayerRole.PLAYER,
        team_id=team_id,
        field_x=x,
        field_y=y,
        confidence=0.9,
        team_confidence=0.9,
        role_confidence=0.9,
        semantic_status="stable",
    )


def _frame(frame_id: int, timestamp: float) -> FrameState:
    return FrameState(
        frame_id=frame_id,
        capture_timestamp_ms=timestamp * 1000.0,
        players=[
            _player(1, 0, 1000.0),
            _player(2, 0, 11000.0),
            _player(3, 1, 6000.0),
            _player(4, 1, 7000.0),
        ],
    )


def test_event_engine_emits_offside_candidate_without_ball() -> None:
    events = EventEngine().update(_frame(1, 0.0))

    offside = next(event for event in events if event.event_type == "offside_candidate")
    assert offside.involved_track_ids == [2, 3]
    assert offside.evidence["attacking_team"] == 0
    assert "ball_x" not in offside.evidence
    assert (offside.field_x, offside.field_y) == (11000.0, 3000.0)


def test_event_engine_only_emits_offside_candidates() -> None:
    events = EventEngine().update(_frame(1, 0.0))

    assert {event.event_type for event in events} == {"offside_candidate"}


def test_event_engine_applies_offside_cooldown() -> None:
    engine = EventEngine()

    assert engine.update(_frame(1, 1.0))
    assert engine.update(_frame(2, 1.2)) == []


def test_event_engine_respects_disabled_offside() -> None:
    assert EventEngine(EventEngineConfig(offside_enabled=False)).update(_frame(1, 0.0)) == []


def test_foul_adapter_preserves_explainable_details() -> None:
    adapter = FoulEventAdapter(confidence_threshold=0.48)
    event = adapter.update(
        SimpleNamespace(confidence=0.8, offence="high", action="pushing"),
        frame_id=12,
        timestamp=2.4,
        field_xy=(4000.0, 2500.0),
    )

    assert event is not None
    assert event.event_type == "foul_candidate"
    assert event.foul_details == {"offence": "high", "action": "pushing"}
    assert event.evidence["source"] == "mvfoul"
