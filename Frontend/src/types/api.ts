export interface DatasetInfo {
  rows: number;
  columns: string[];
  dtypes: Record<string, string>;
  summary: Record<string, unknown>;
  preview: Record<string, unknown>[];
  missing_values: Record<string, { count: number; percentage: number }>;
  duplicate_rows: number;
  numeric_columns: string[];
  categorical_columns: string[];
  memory_usage_mb: number;
}

export interface CleaningReport {
  empty_rows_dropped: number;
  empty_columns_dropped: number;
  duplicate_rows_removed: number;
  numeric_nans_filled: number;
  categorical_nans_filled: number;
  columns_renamed: Record<string, string>;
  filesize_reduced_mb?: number;
}

export interface AnalysisSummary {
  [col: string]: {
    count?: number;
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    [key: string]: unknown;
  };
}

export interface StrongCorrelation {
  column_a: string;
  column_b: string;
  r_value: number;
  direction: "positive" | "negative";
  strength: "strong" | "very_strong" | "perfect";
}

export interface OutlierInfo {
  count: number;
  percentage: number;
  lower_bound: number;
  upper_bound: number;
  indices: number[];
  sample_values: number[];
}

export interface CategoricalStats {
  count: number;
  percentage: number;
}

export interface CategoricalDistribution {
  categories: Record<string, CategoricalStats>;
}

export interface AnalysisResult {
  metadata: Record<string, unknown>;
  summary: AnalysisSummary;
  correlation: Record<string, Record<string, number>>;
  strong_correlations: StrongCorrelation[];
  outliers: Record<string, OutlierInfo>;
  categorical_distribution: Record<string, CategoricalDistribution>;
  column_quality_flags: Record<string, string[]>;
  timing_ms: number;
}

export interface ChartReport {
  charts_generated: string[];
  charts_failed: Record<string, string>[];
  total_images: number;
  timing_ms: number;
}

export interface Charts {
  correlation_heatmap?: string;
  distributions?: Array<{ column: string; image: string }>;
  bar_charts?: Array<{ column: string; image: string }>;
  pie_charts?: Array<{ column: string; image: string }>;
  trend_charts?: Array<{ column: string; image: string }>;
  [key: string]: unknown;
}

export interface InsightResult {
  insights_text: string;
  model_used: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  response_time_ms: number;
  retries_attempted: number;
  success: boolean;
  fallback_reason: string;
}

export interface ApiResponse {
  filename: string;
  info: DatasetInfo;
  cleaning_report: CleaningReport;
  analysis: AnalysisResult;
  charts: Charts;
  insights: InsightResult;
}

export interface GenerateReportRequest {
  filename: string;
  analysis: AnalysisResult;
  charts: Charts;
}
}

// ─── Phase 3: Interactive Cleaning Types ───

export interface QualityIssue {
  type: "missing_values" | "type_mismatch" | "outlier";
  column: string;
  count: number;
  severity: "low" | "medium" | "high" | "none";
  suggestion: "fill_mean" | "fill_unknown" | "drop_rows" | "convert_numeric";
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  inferred_type: "numeric" | "datetime" | "string";
  missing_count: number;
  missing_percentage: number;
}

export interface InspectionReport {
  total_rows: number;
  columns: ColumnProfile[];
  issues: QualityIssue[];
}

export interface CleaningRule {
  action: "drop_rows" | "fill_mean" | "fill_value" | "none";
  value?: string | number;
}

export interface CleaningRulesMap {
  [columnName: string]: CleaningRule;
}

export interface InspectionResult {
  filename: string;
  quality_report: InspectionReport;
  preview: Record<string, unknown>[];
  raw_file_path: string;
  stage: "INSPECTION";
}
