import React, { useRef, useEffect } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import type { PlayerState } from '../types/messages';

const PITCH_W = 12000;
const PITCH_H = 7000;
const PADDING = 40;
// Stable fallback: zustand v5 re-renders forever if a selector returns a new [] each call.
const NO_PLAYERS: PlayerState[] = [];

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
  const players = useDashboardStore((s) => s.frameState?.players ?? NO_PLAYERS);
  const ball = useDashboardStore((s) => s.frameState?.ball ?? null);
  const homographyStatus = useDashboardStore(
    (s) => s.frameState?.homography_status ?? 'unavailable'
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    drawPitch(ctx, w, h);

    for (const p of players) {
      if (p.field_x == null || p.field_y == null) continue;
      const [sx, sy] = worldToCanvas(p.field_x, p.field_y, w, h);
      const color = TEAM_COLORS[p.team] ?? TEAM_COLORS[String(p.team_id)] ?? '#888';

      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
    ctx.fillText(`${p.entity_id ?? p.track_id}`, sx, sy - 11);
    }

    if (ball?.field_x != null && ball.field_y != null) {
      const [bx, by] = worldToCanvas(ball.field_x, ball.field_y, w, h);
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, Math.PI * 2);
      ctx.fillStyle = ball.status === 'fresh' ? '#fff' : '#aaa';
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.stroke();
    }
  }, [players, ball]);

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
