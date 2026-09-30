from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = APP_DIR.parent
ASSETS_DIR = REPO_ROOT_DIR / "assets"
DATA_DIR = ASSETS_DIR / "data"
WEIGHTS_DIR = ASSETS_DIR / "weights"

# The first-stage pipeline uses the official Ultralytics person detector.  The
# previous role-aware checkpoint is kept available for explicit legacy use.
YOLO11S_PLAYER_MODEL_PATH = str(WEIGHTS_DIR / "yolo11s.pt")
YOLO11N_PLAYER_MODEL_PATH = str(WEIGHTS_DIR / "yolo11n.pt")
PLAYER_DETECTION_MODEL_PATH = YOLO11S_PLAYER_MODEL_PATH
LEGACY_PLAYER_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / "football-player-detection.pt")
PITCH_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / "football-pitch-detection.pt")
ROLE_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / "player-role-yolo11n.pt")
TEAM_CLASSIFIER_PATH = str(WEIGHTS_DIR / "team-classifier.joblib")
FOUL_MODEL_PATH = str(WEIGHTS_DIR / "mvfoul.pth.tar")
CAMERA_CALIBRATION_PATH = str(ASSETS_DIR / "calibration" / "camera.npz")
