import type { RefereeAlert, RefereeAlertType } from '../types/refereeAlerts';

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = typeof payload.detail === 'string'
      ? payload.detail
      : payload.detail
        ? JSON.stringify(payload.detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function createRefereeAlert(
  type: RefereeAlertType,
): Promise<RefereeAlert> {
  return readJson(
    await fetch('/api/referee-alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type }),
    }),
  );
}
