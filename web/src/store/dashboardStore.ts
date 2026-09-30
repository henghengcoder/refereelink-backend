import { create } from 'zustand';
import type {
  FrameState,
  MetricsSnapshot,
  GameEvent,
  PipelineConfig,
  TeamCalibrationState,
} from '../types/messages';

interface DashboardState {
  frameState: FrameState | null;
  metrics: MetricsSnapshot | null;
  events: GameEvent[];
  logLines: string[];
  config: PipelineConfig;
  wsConnected: boolean;
  sourceStatus: string;
  pipelineRunning: boolean;
  teamCalibration: TeamCalibrationState;

  setFrameState: (fs: FrameState) => void;
  setMetrics: (m: MetricsSnapshot) => void;
  addEvent: (e: GameEvent) => void;
  addLog: (msg: string) => void;
  setConfig: (c: Partial<PipelineConfig>) => void;
  setWsConnected: (c: boolean) => void;
  setSourceStatus: (s: string) => void;
  setPipelineRunning: (r: boolean) => void;
  setTeamCalibration: (state: TeamCalibrationState) => void;
}

const EMPTY_PLAYERS: FrameState['players'] = [];

export const selectPitchPlayers = (state: DashboardState): FrameState['players'] =>
  state.frameState?.players ?? EMPTY_PLAYERS;

export const useDashboardStore = create<DashboardState>((set) => ({
  frameState: null,
  metrics: null,
  events: [],
  logLines: [],
  config: {
    mode: 'realtime',
    video_source: '',
    device: 'cpu',
    inference_backend: 'auto',
    enable_foul_detection: false,
    enable_recording: false,
    show_keypoints: true,
    show_tracking_boxes: true,
    show_2d_projection: true,
    foul_confidence_threshold: 0.48,
    player_model_path: 'assets/weights/yolo11s.pt',
    pitch_model_path: 'assets/weights/football-pitch-detection.pt',
    role_model_path: 'assets/weights/player-role-yolo11n.pt',
    team_classifier_path: null,
    team_calibration_path: null,
    require_team_calibration: true,
    role_detection_interval: 3,
    team_classification_interval: 5,
    track_activation_threshold: 0.25,
    track_lost_buffer: 45,
    track_matching_threshold: 0.8,
    track_minimum_consecutive_frames: 2,
    camera_calibration_path: 'assets/calibration/camera.npz',
    enable_undistortion: true,
    calibration_alpha: 0,
    pitch_detection_interval: 5,
    imgsz: 640,
    player_confidence: 0.25,
    player_iou: 0.7,
    max_prediction_gap_frames: 6,
    track_reactivation_window_frames: 12,
  },
  wsConnected: false,
  sourceStatus: 'disconnected',
  pipelineRunning: false,
  teamCalibration: {
    type: 'team_calibration',
    state: 'idle',
    match_id: '',
    camera_id: 'default',
    bundle_path: null,
    ready: false,
    goalkeeper_mapping_ready: false,
    referee_mapping_ready: false,
    observed_frames: 0,
    last_frame_index: null,
    tracks: [],
    validation_report: null,
    source_url: null,
    clip_id: null,
    clip_start_ms: null,
    clip_end_ms: null,
    clip_duration_ms: null,
    review_video_url: null,
    metadata_url: null,
    job_id: null,
    processing_progress: 0,
    processing_error: null,
  },

  setFrameState: (fs) => set({ frameState: fs }),
  setMetrics: (m) => set({ metrics: m }),
  addEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 500) })),
  addLog: (msg) =>
    set((s) => ({
      logLines: [...s.logLines, msg].slice(-200),
    })),
  setConfig: (c) => set((s) => ({ config: { ...s.config, ...c } })),
  setWsConnected: (c) => set({ wsConnected: c }),
  setSourceStatus: (s) => set({ sourceStatus: s }),
  setPipelineRunning: (r) => set({ pipelineRunning: r }),
  setTeamCalibration: (teamCalibration) => set({ teamCalibration }),
}));
