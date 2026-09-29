import React, { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '../store/dashboardStore';
import { createRefereeAlert } from '../api/refereeAlerts';
import type { RefereeAlertType } from '../types/refereeAlerts';

interface Feedback {
  kind: 'success' | 'error';
  message: string;
}

const ALERT_BUTTONS: Array<{ type: RefereeAlertType; label: string; ariaLabel: string }> = [
  {
    type: 'foul_candidate',
    label: '犯规提醒',
    ariaLabel: '手动发送犯规候选提醒（测试入口）',
  },
  {
    type: 'offside_candidate',
    label: '越位提醒',
    ariaLabel: '手动发送越位候选提醒（测试入口）',
  },
];

const EventAlerts: React.FC = () => {
  const events = useDashboardStore((s) => s.events);
  const [pendingType, setPendingType] = useState<RefereeAlertType | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const sendAlert = async (type: RefereeAlertType) => {
    if (pendingType) return;
    setPendingType(type);
    setFeedback(null);
    const label = ALERT_BUTTONS.find((b) => b.type === type)?.label ?? type;
    try {
      const alert = await createRefereeAlert(type);
      if (!mountedRef.current) return;
      setFeedback({ kind: 'success', message: `已发送${label} (${alert.event_id})` });
    } catch (e) {
      if (!mountedRef.current) return;
      const detail = e instanceof Error ? e.message : String(e);
      setFeedback({ kind: 'error', message: `${label}发送失败: ${detail}` });
    } finally {
      if (mountedRef.current) setPendingType(null);
    }
  };

  return (
    <div className="panel events-panel">
      <div className="panel-header">
        <span>EVENTS &amp; ALERTS</span>
        <span className="event-count">{events.length}</span>
      </div>
      <div className="alerts-actions">
        {ALERT_BUTTONS.map((b) => (
          <button
            key={b.type}
            type="button"
            className="btn btn-action alert-btn"
            aria-label={b.ariaLabel}
            aria-busy={pendingType === b.type}
            disabled={pendingType !== null}
            onClick={() => void sendAlert(b.type)}
          >
            {pendingType === b.type ? '...' : b.label}
          </button>
        ))}
      </div>
      {feedback && (
        <div className={`alerts-feedback ${feedback.kind}`} role="status" aria-live="polite">
          {feedback.message}
        </div>
      )}
      <div className="events-list">
        {events.length === 0 && (
          <div className="events-empty">No events detected</div>
        )}
        {events.slice(0, 20).map((e) => (
          <div key={e.id} className={`event-item severity-${e.severity}`}>
            <div className="event-header">
              <span className="event-type">{e.event_type.toUpperCase()}</span>
              <span className="event-confidence">
                {(e.confidence * 100).toFixed(0)}%
              </span>
              <span className="event-severity">{e.severity}</span>
              {e.reviewed && <span className="event-reviewed">REVIEWED</span>}
            </div>
            <div className="event-details">
              t={e.timestamp.toFixed(1)}s
              {e.field_x != null && e.field_y != null && (
                <span>
                  &nbsp;pos=({e.field_x.toFixed(0)},{e.field_y.toFixed(0)})
                </span>
              )}
              &nbsp;frame={e.frame_id}
              {e.foul_details && (
                <span>
                  &nbsp;offence={e.foul_details.offence}
                  &nbsp;action={e.foul_details.action}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EventAlerts;
