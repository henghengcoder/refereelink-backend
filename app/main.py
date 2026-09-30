import argparse
from enum import Enum
from typing import Optional

import cv2
import supervision as sv

from app.runtime import normalize_proxy_env
from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
    ROLE_DETECTION_MODEL_PATH,
    TEAM_CLASSIFIER_PATH,
)


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    RADAR_DASHBOARD = 'RADAR_DASHBOARD'
    RADAR_DASHBOARD_LEGACY = 'RADAR_DASHBOARD_LEGACY'
    FOUL_DETECTION = 'FOUL_DETECTION'
    SERVER = 'SERVER'


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    inference_backend: str = "auto",
    foul_checkpoint_path: Optional[str] = None,
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    pitch_detection_interval: int = 5,
    imgsz: int = 640,
    role_model_path: str = ROLE_DETECTION_MODEL_PATH,
    team_classifier_path: Optional[str] = TEAM_CLASSIFIER_PATH,
    role_detection_interval: int = 3,
    team_classification_interval: int = 5,
) -> None:
    normalize_proxy_env()

    if mode == Mode.RADAR_DASHBOARD:
        from app.modes.radar_dashboard import run_radar_dashboard

        run_radar_dashboard(
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
            player_model_path=player_model_path,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
        )
        return

    if mode == Mode.RADAR_DASHBOARD_LEGACY:
        from app.modes.radar_dashboard import run_radar_dashboard

        run_radar_dashboard(
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
            player_model_path=player_model_path,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
        )
        return

    if mode == Mode.SERVER:
        from app.server.main import main as server_main
        import sys
        sys.argv = [
            sys.argv[0],
            '--video_source', source_video_path,
            '--device', device,
            '--inference_backend', inference_backend,
            '--player_model_path', player_model_path,
            '--pitch_model_path', pitch_model_path,
            '--camera_calibration_path', camera_calibration_path or '',
            '--calibration_alpha', str(calibration_alpha),
            '--pitch_detection_interval', str(pitch_detection_interval),
            '--imgsz', str(imgsz),
            '--role_model_path', role_model_path,
            '--role_detection_interval', str(role_detection_interval),
            '--team_classification_interval', str(team_classification_interval),
        ]
        if team_classifier_path:
            sys.argv += ['--team_classifier_path', team_classifier_path]
        if not enable_undistortion:
            sys.argv.append('--disable_undistortion')
        if foul_checkpoint_path:
            sys.argv += ['--foul_checkpoint_path', foul_checkpoint_path, '--enable_foul_detection']
        server_main()
        return

    if mode == Mode.PITCH_DETECTION:
        from app.modes.pitch_detection import run_pitch_detection

        frame_generator = run_pitch_detection(
            source_video_path=source_video_path,
            device=device,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
        )
    elif mode == Mode.PLAYER_DETECTION:
        from app.modes.player_detection import run_player_detection

        frame_generator = run_player_detection(
            source_video_path=source_video_path,
            device=device,
            player_model_path=player_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            imgsz=imgsz,
        )
    elif mode == Mode.PLAYER_TRACKING:
        from app.modes.player_tracking import run_player_tracking

        frame_generator = run_player_tracking(
            source_video_path=source_video_path,
            device=device,
            player_model_path=player_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            imgsz=imgsz,
        )
    elif mode == Mode.TEAM_CLASSIFICATION:
        from app.modes.team_classification import run_team_classification

        frame_generator = run_team_classification(
            source_video_path=source_video_path,
            device=device,
            player_model_path=player_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            imgsz=imgsz,
        )
    elif mode == Mode.RADAR:
        from app.modes.radar import run_radar

        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
            player_model_path=player_model_path,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
        )
    elif mode == Mode.FOUL_DETECTION:
        from app.modes.foul_detection import run_foul_detection

        if foul_checkpoint_path is None:
            from app.constants.paths import FOUL_MODEL_PATH
            foul_checkpoint_path = FOUL_MODEL_PATH

        frame_generator = run_foul_detection(
            source_video_path=source_video_path,
            device=device,
            foul_checkpoint_path=foul_checkpoint_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--inference_backend', type=str, choices=('auto', 'pytorch', 'onnx', 'tensorrt'), default='auto')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--player_model_path', type=str, default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument('--pitch_model_path', type=str, default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument('--camera_calibration_path', type=str, default=CAMERA_CALIBRATION_PATH)
    parser.add_argument('--disable_undistortion', action='store_false', dest='enable_undistortion')
    parser.set_defaults(enable_undistortion=True)
    parser.add_argument('--calibration_alpha', type=float, default=0.0)
    parser.add_argument('--pitch_detection_interval', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--role_model_path', type=str, default=ROLE_DETECTION_MODEL_PATH)
    parser.add_argument('--team_classifier_path', type=str, default=TEAM_CLASSIFIER_PATH)
    parser.add_argument('--role_detection_interval', type=int, default=3)
    parser.add_argument('--team_classification_interval', type=int, default=5)
    parser.add_argument(
        '--foul_checkpoint_path',
        type=str,
        default=None,
        help=(
            'Path to MVFoul checkpoint (.pth.tar). '
            'Required for FOUL_DETECTION mode (falls back to assets/weights/mvfoul.pth.tar). '
            'Optional for RADAR / RADAR_DASHBOARD — enables the foul HUD overlay when provided.'
        ),
    )
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        inference_backend=args.inference_backend,
        mode=args.mode,
        foul_checkpoint_path=args.foul_checkpoint_path,
        player_model_path=args.player_model_path,
        pitch_model_path=args.pitch_model_path,
        camera_calibration_path=args.camera_calibration_path,
        enable_undistortion=args.enable_undistortion,
        calibration_alpha=args.calibration_alpha,
        pitch_detection_interval=args.pitch_detection_interval,
        imgsz=args.imgsz,
        role_model_path=args.role_model_path,
        team_classifier_path=args.team_classifier_path,
        role_detection_interval=args.role_detection_interval,
        team_classification_interval=args.team_classification_interval,
    )
