export type FrameSource = "dataset" | "fault_transform" | "fault_timewarp";
export type PointSource = FrameSource | "sensor" | "replay" | "simulated" | "derived";

export type Clock = { as_of: string; speed: number; mission_day: number; running: boolean };
export type Spend = { total_usd: number; budget_usd: number; by_role: Record<string, number> };

export type AlertItem = {
  id: string;
  ts: string;
  tray_id: string;
  rule: string;
  severity: string;
  evidence_ids: string[];
  threshold_tag: string | null;
  assessment_id: string | null;
  detail: string;
  status: "open" | "resolved";
};

export type TrayListItem = {
  id: string;
  crop: string;
  treatment: string;
  status: string;
  fault_start: string | null;
  reference_tray_id: string | null;
  badge: string;
  open_alerts: number;
  latest_frame_id: string | null;
  shared_alerts: AlertItem[];
};

export type TrayDetail = {
  id: string;
  crop: string;
  treatment: string;
  status: string;
  fault_start: string | null;
  reference_tray_id: string | null;
  age_days: number;
  mission_day: number;
  badge: string;
  calib_flag: string;
  latest: {
    ts: string;
    frame_id: string;
    area_px: number | null;
    area_cm2: number | null;
    growth_rate: number | null;
    exg: number | null;
    chlorosis_frac: number | null;
    drift_z: number | null;
    soil_vwc: number | null;
    temp_c: number | null;
    rh_pct: number | null;
    vpd_kpa: number | null;
  } | null;
  open_alerts: { id: string; rule: string; severity: string }[];
  shared_alerts: AlertItem[];
};

export type Frame = {
  id: string;
  tray_id: string;
  ts: string;
  source: FrameSource;
  calib_flag: string;
  cm2_per_px: number;
  image_url: string;
  overlay_url: string;
};
export type Page<T> = { items: T[]; next_cursor: string | null };

export type SeriesPoint = { id: string; ts: string; value: number | null; source: PointSource };
export type Series = {
  tray_id: string;
  metric: string;
  unit: string;
  items: SeriesPoint[];
  sensor_tray?: string;
};

export type Finding = { claim: string; evidence_ids: string[]; confidence: number };
export type Cause = { cause: string; likelihood: number; evidence_ids: string[]; discriminating_check?: string };
export type RecAction = { action: string; magnitude?: string; rationale?: string; evidence_ids: string[] };
export type Assessment = {
  tray_id: string;
  status: string;
  findings: Finding[];
  probable_causes: Cause[];
  causes_considered?: string[];
  recommended_actions: RecAction[];
  setpoint_caveats: string[];
  missing_evidence?: string[];
  crew_brief: string;
};
export type ActionRow = {
  action_id: string;
  status: string;
  proposal: { tray_id?: string; action: string; magnitude?: string; rationale?: string; evidence_ids?: string[] };
};
export type AssessmentResponse = {
  assessment_id: string;
  tray_id: string;
  alert_id: string;
  assessment: Assessment | null;
  /** null while the run is still in flight */
  valid: boolean | null;
  citation_errors: string[];
  latency_ms: number | null;
  recorded: boolean;
  model: string;
  provider: string;
  actions: ActionRow[];
  cost_usd?: number | null;
};

export type StreamEvent =
  | { event: "tool_call"; data: { round: number; tool: string; args: Record<string, unknown> } }
  | { event: "tool_result"; data: { round: number; tool: string; ids: string[] } }
  | { event: "assessment"; data: { assessment_id: string; valid: boolean; latency_ms: number; recorded: boolean } }
  | { event: "error"; data: { message: string } };

export type Evidence = {
  id: string;
  kind: string;
  tray_id: string;
  ts: string | null;
  frame_id?: string;
  metric?: string;
  value?: number;
  source?: PointSource;
};

export type EvalLatest = {
  run_id: string;
  generated_at: string;
  n_scenarios: number;
  small_n: boolean;
  top1: Record<string, string>;
  citation_errors: number;
  mean_latency_ms: number | null;
  area_validation?: { dice: number | null; iou: number | null; pearson_r: number | null } | null;
  [k: string]: unknown;
};
