import React, { useRef, useEffect } from 'react';
import { selectPitchPlayers, useDashboardStore } from '../store/dashboardStore';
import type { BallState } from '../types/messages';

const PITCH_W = 12000;
const PITCH_H = 7000;
const PADDING = 40;
// Longest tween between two backend updates; slower streams snap instead of drifting.
const MAX_TWEEN_MS = 200;

interface Marker {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  label: string;
  color: string;
}

const TEAM_COLORS: Record<string, string> = {
  home: '#FF1493',
  away: '#00BFFF',
  none: '#FFD700',
  unknown: '#666666',
  player: '#666666',
  '0': '#FF1493',
  '1': '#00BFFF',
  '-1': '#666666',
};

function worldToCanvas(wx: number, wy: number, cw: number, ch: number): [number, number] {
  const scaleX = (cw - PADDING * 2) / PITCH_W;
  const scaleY = (ch - PADDING * 2) / PITCH_H;
  const scale = Math.min(scaleX, scaleY);
  const offsetX = (cw - PITCH_W * scale) / 2;
  const offsetY = (ch - PITCH_H * scale) / 2;
  return [wx * scale + offsetX, wy * scale + offsetY];
}

function drawPitch(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.fillStyle = '#0d3322';
  ctx.lineWidth = 1.5;

  const [px0, py0] = worldToCanvas(0, 0, w, h);
  const [px1, py1] = worldToCanvas(PITCH_W, PITCH_H, w, h);
  ctx.fillRect(px0, py0, px1 - px0, py1 - py0);
  ctx.strokeRect(px0, py0, px1 - px0, py1 - py0);

  const [cx, cy] = worldToCanvas(PITCH_W / 2, PITCH_H / 2, w, h);
  ctx.beginPath();
  ctx.arc(cx, cy, (915 / PITCH_W) * (px1 - px0), 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx, py0);
  ctx.lineTo(cx, py1);
  ctx.stroke();

  const [plx, ply] = worldToCanvas(0, (PITCH_H - 4100) / 2, w, h);
  const [prx, pry] = worldToCanvas(2015, (PITCH_H - 4100) / 2, w, h);
  ctx.strokeRect(plx, ply, prx - plx, (4100 / PITCH_H) * (py1 - py0));

  const [plx2] = worldToCanvas(PITCH_W - 2015, (PITCH_H - 4100) / 2, w, h);
  ctx.strokeRect(plx2, ply, prx - plx, (4100 / PITCH_H) * (py1 - py0));
}

const Pitch2D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const players = useDashboardStore(selectPitchPlayers);
  const ball = useDashboardStore((s) => s.frameState?.ball ?? null);
  const homographyStatus = useDashboardStore(
    (s) => s.frameState?.homography_status ?? 'unavailable'
  );
  // Markers keyed by entity, animated from their last drawn position to the
  // newest backend position over roughly one update interval.
  const markersRef = useRef<Map<number, Marker>>(new Map());
  const tweenStartRef = useRef(0);
  const tweenMsRef = useRef(0);
  const lastUpdateRef = useRef<number | null>(null);
  const ballRef = useRef<BallState | null>(null);

  useEffect(() => {
    ballRef.current = ball;
  }, [ball]);

  useEffect(() => {
    const now = performance.now();
    const t = tweenMsRef.current > 0
      ? Math.min((now - tweenStartRef.current) / tweenMsRef.current, 1)
      : 1;
    const previous = markersRef.current;
    const next = new Map<number, Marker>();
    for (const p of players) {
      if (p.field_x == null || p.field_y == null) continue;
      const key = p.entity_id ?? p.track_id;
      const old = previous.get(key);
      // Start from where the marker is currently drawn so tweens chain smoothly.
      const fromX = old ? old.fromX + (old.toX - old.fromX) * t : p.field_x;
      const fromY = old ? old.fromY + (old.toY - old.fromY) * t : p.field_y;
      next.set(key, {
        fromX,
        fromY,
        toX: p.field_x,
        toY: p.field_y,
        label: `${key}`,
        color: TEAM_COLORS[p.team] ?? TEAM_COLORS[String(p.team_id)] ?? '#888',
      });
    }
    markersRef.current = next;
    const interval = lastUpdateRef.current === null ? 0 : now - lastUpdateRef.current;
    tweenMsRef.current = interval > 0 && interval <= MAX_TWEEN_MS ? interval : 0;
    tweenStartRef.current = now;
    lastUpdateRef.current = now;
  }, [players]);

  useEffect(() => {
    let frame = 0;
    const draw = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        drawPitch(ctx, w, h);
        const tween = tweenMsRef.current;
        const t = tween > 0 ? Math.min((performance.now() - tweenStartRef.current) / tween, 1) : 1;
        for (const m of markersRef.current.values()) {
          const [sx, sy] = worldToCanvas(
            m.fromX + (m.toX - m.fromX) * t,
            m.fromY + (m.toY - m.fromY) * t,
            w,
            h
          );
          ctx.beginPath();
          ctx.arc(sx, sy, 7, 0, Math.PI * 2);
          ctx.fillStyle = m.color;
          ctx.fill();
          ctx.strokeStyle = 'rgba(0,0,0,0.6)';
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.fillStyle = '#fff';
          ctx.font = '9px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(m.label, sx, sy - 11);
        }

        const ballState = ballRef.current;
        if (ballState?.field_x != null && ballState.field_y != null) {
          const [bx, by] = worldToCanvas(ballState.field_x, ballState.field_y, w, h);
          ctx.beginPath();
          ctx.arc(bx, by, 5, 0, Math.PI * 2);
          ctx.fillStyle = ballState.status === 'fresh' ? '#fff' : '#aaa';
          ctx.fill();
          ctx.strokeStyle = '#111';
          ctx.stroke();
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <div className="panel pitch-panel">
      <div className="panel-header">
        <span>2D PITCH</span>
        <span className={`homography-badge ${homographyStatus}`}>{homographyStatus}</span>
      </div>
      <canvas
        ref={canvasRef}
        width={500}
        height={320}
        className="pitch-canvas"
      />
    </div>
  );
};

export default Pitch2D;
