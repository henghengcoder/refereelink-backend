export type RefereeAlertType = 'foul_candidate' | 'offside_candidate';
export type RefereeAlertSource = 'manual' | 'detector';

export interface RefereeAlert {
  event_id: string;
  type: RefereeAlertType;
  timestamp: number;
  confidence: number;
  evidence: Record<string, unknown> | null;
  case_id: string | null;
  source: RefereeAlertSource;
}
