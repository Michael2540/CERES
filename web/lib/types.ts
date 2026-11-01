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
/** A number in free text that no tool result of the run contains. */
export type Grounding = { field: string; number: string; context: string };
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
  /** Ungrounded numbers in the free text (a note, not an error). Absent on older runs. */
  warnings?: Grounding[];
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

/** One scored scenario of the agent eval (backend/eval/score.py `score_scenario`). */
export type EvalScenario = {
  id: string;
  description: string;
  data_source: string;
  tray_id: string | null;
  alert_rule: string | null;
  as_of: string;
  expected_top_cause: string | null;
  acceptable_causes: string[];
  top_cause: string | null;
  top1_correct: boolean | null;
  alert_found: boolean;
  alert_id: string | null;
  assessment_id: string | null;
  assessed: boolean;
  valid: boolean;
  citation_errors: string[];
  schema_valid: boolean | null;
  likelihood_sum_ok: boolean | null;
  sensor_fault_considered: boolean | null;
  sensor_fault_top: boolean;
  injection_resisted: boolean | null;
  injection_note: string | null;
  note_acknowledged: boolean | null;
  false_alarm: boolean | null;
  open_alerts: string[];
  latency_ms: number | null;
  cost_usd: number;
  tool_counts: Record<string, number>;
  n_tool_calls: number;
  checks: Record<string, boolean>;
  /** null = nothing to check (an optional alert never fired). */
  passed: boolean | null;
  error: string | null;
};

/** Each value is a fraction string "k/n", never a percentage. */
export type EvalDeterministic = {
  citation_validity: string;
  schema_validity: string;
  sensor_fault_considered: string;
  sensor_fault_discrimination: string;
  injection_resistance: string;
  /** Absent on runs before numeric grounding existed. */
  numeric_grounding?: string;
  likelihood_sum_ok: string;
  false_alarm_on_nominal: string;
};

/** GET /eval/latest: the latest agent eval run (static report, not time-gated; generated_at is wall-clock UTC). */
export type EvalLatest = {
  run_id: string;
  generated_at: string;
  note: string | null;
  /** true for a scripted `--fake` run (no model was called). */
  fake: boolean;
  role: string;
  model: string;
  effort: string;
  source_db: string;
  aborted: string | null;
  ledger_usd: number | null;
  n_scenarios: number;
  n_assessed: number;
  small_n: boolean;
  passed: string;
  top1: Record<string, string>;
  top1_n: Record<string, number>;
  top1_overall: string;
  citation_errors: number;
  mean_latency_ms: number | null;
  deterministic: EvalDeterministic;
  latency_ms: { p50: number | null; p95: number | null; mean: number | null };
  cost_usd: { total: number; per_assessment: number | null };
  scenarios: EvalScenario[];
};

/** Aggregates of one slice of the area validation (null where undefined, e.g. Pearson r on one image). */
export type AreaAgg = {
  n: number;
  dice_mean?: number | null; dice_median?: number | null; dice_min?: number | null;
  iou_mean?: number | null; iou_median?: number | null; iou_min?: number | null;
  pooled_dice?: number | null; area_pearson_r?: number | null; area_r2?: number | null; area_mape?: number | null;
  pot_n?: number; pot_pearson_r?: number | null; plant_exact_frac?: number | null; plant_mae?: number | null;
  /** Per pot via the method's own instances (centroid rule, as production assigns them). */
  inst_pot_pearson_r?: number | null; inst_plant_exact_frac?: number | null; inst_plant_mae?: number | null;
};
export type AreaBreakdown = {
  overall: AreaAgg;
  by_tray: Record<string, AreaAgg>;
  by_stage: { median_gt_px: number; small: AreaAgg; large: AreaAgg };
};
/** GET /eval/area-validation: SAM 3.1 vs ALED hand annotation on raw frames (static, not time-gated). */
export type AreaValidation = {
  model: string; prompt: string; caption: string;
  n_images: number; n_gt: number; n_cached: number; n_called: number; cost_usd: number;
  partial: boolean; generated_at: string;
  sam: AreaBreakdown;
  /** SAM before the colour trim (present from the trim round on). */
  sam_untrimmed?: AreaBreakdown;
  trim?: { s_min: number; v_abs: number; v_rel: number; close_px: number };
  baseline_hsv: AreaBreakdown & { label: string };
  worst: { frame: string; dice: number; overlay: string | null }[];
  per_image: { frame: string; tray: string; dice: number; iou: number; gt_px: number; sam_px: number; gt_plants: number; sam_plants: number; inst_plants?: number }[];
};

// ---- Mars mode (M8) --------------------------------------------------------------------------
export type LinkState = "up" | "delayed" | "down";
/** GET/POST /link. The link is simulated; light_delay_note is text only. */
export type LinkInfo = {
  state: LinkState;
  delay_s: number | null;
  since: string | null;
  queue_depth: number;
  simulated: boolean;
  light_delay_note: string;
  edge_mode: string;
  /** "recorded" = a stored response is replayed; "remote_api" = a live API call. No mode runs a model on the habitat computer. */
  edge_location: "recorded" | "remote_api";
  /** The model that answers the edge triage (or the model inside the recordings); null when there are none. */
  edge_model: string | null;
  /** The Earth-side Flight Agronomist model. */
  earth_model: string;
  edge_recordings: string[];
  /** DEMO_MODE: every model result is replayed from a recording; no live call is possible. */
  demo_mode?: boolean;
  /** DEMO_MODE: alerts with a recorded Earth-side run (the others answer 409). */
  agent_recordings?: string[];
};

export type EdgeTriage = {
  tray_id: string;
  urgency: "now" | "next_check" | "log_only";
  likely_cause: string;
  evidence_ids: string[];
  crew_instruction: string;
  handoff_note_for_earth: string;
};
export type EdgeProvenance = {
  recorded: boolean;
  model: string;
  provider: string;
  edge_location?: "recorded" | "remote_api";
  badge?: string;
  recorded_as_of?: string | null;
};
export type Audit = {
  agree: boolean | null;
  urgency_delta: number | null;
  cause_match: boolean | null;
  edge_cause: string | null;
  earth_top_cause: string | null;
  edge_urgency: string | null;
  earth_status: string | null;
  notes: string;
};
export type QueueStatus = "queued" | "edge_done" | "edge_invalid" | "sent" | "audited" | "failed";
export type QueueItem = {
  id: string;
  ts: string;
  kind: string;
  alert_id: string;
  tray_id: string;
  as_of: string;
  status: QueueStatus;
  edge_json: EdgeTriage | null;
  edge_valid: boolean | null;
  edge_assessment_id: string | null;
  provenance: EdgeProvenance | null;
  citation_errors: string[];
  /** null = not evaluated (no answer, or the final reply was unparseable); [] = evaluated, nothing ungrounded. */
  grounding: Grounding[] | null;
  audit_json: Audit | null;
  earth_assessment_id: string | null;
  /** LIVE or RECORDED, and the model that answered, from the Earth-side assessments row. */
  earth_provenance?: { recorded: boolean; model: string; provider: string } | null;
  deliverable: boolean;
  deliver_in_s: number | null;
  error: string | null;
  perception: string;
};
export type DrainPlan = {
  started: boolean;
  confirm_required: boolean;
  link_state: LinkState;
  item_ids: string[];
  est_cost_usd: { low: number; high: number };
  detail: string;
};

/** GET /eval/latest?role=edge (backend/eval/edge_eval.py). */
export type EdgeEvalScenario = {
  id: string;
  data_source: string;
  tray_id: string | null;
  alert_rule: string | null;
  expected_top_cause?: string | null;
  likely_cause: string | null;
  urgency: string | null;
  top1_correct: boolean | null;
  numeric_grounding: boolean | null;
  ungrounded_numbers: Grounding[];
  provenance: { recorded: boolean; model: string; provider: string } | null;
  passed: boolean | null;
  error: string | null;
};
export type EdgeEvalLatest = Omit<EvalLatest, "deterministic" | "scenarios"> & {
  edge_mode?: string;
  deterministic: {
    citation_validity: string;
    schema_validity: string;
    numeric_grounding: string;
    injection_resistance: string;
    urgency_sanity: string;
  };
  scenarios: EdgeEvalScenario[];
};
