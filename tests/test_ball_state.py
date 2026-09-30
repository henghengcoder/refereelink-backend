from __future__ import annotations

from app.state.models import BallState, BallStatus, FrameState, PlayerRole, PlayerState


def test_ball_state_and_frame_state_serialize_optional_coordinates() -> None:
    state = BallState(status=BallStatus.UNAVAILABLE)
    frame = FrameState(
        frame_id=3,
        ball=state,
        players=[
            PlayerState(
                track_id=1,
                role=PlayerRole.UNKNOWN,
                team_id=-1,
                confidence=0.6,
            )
        ],
    )

    data = frame.model_dump(mode="json")

    assert data["ball"]["status"] == "unavailable"
    assert data["ball"]["field_x"] is None
    assert data["ball"]["field_y"] is None
    assert data["players"][0]["semantic_status"] == "unknown"
