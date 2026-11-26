import {
  Activity,
  AlignLeft,
  BarChart3,
  Filter,
  Grid,
  Layers,
  PieChart,
  Sliders,
  Table2,
  TrendingUp,
} from "lucide-react";

export interface DatasetItem {
  task_id: string;
  filename: string;
  rows: number;
  columns_count: number;
  columns: string[];
  numeric_columns?: string[];
  categorical_columns?: string[];
  created_at?: string;
}

export interface SchemaColumnMeta {
  name: string;
  type: number;
  type_name: string;
  is_numeric: boolean;
  is_temporal: boolean;
  null_count: number;
  unique_values: number;
  sample_values?: string[];
}

export interface FilterItem {
  id: string;
  column: string;
  operator: string;
  value: string;
}

export interface SupersetExploreStudioProps {
  taskId?: string;
  datasetName?: string;
  allowDatasetSwitching?: boolean;
  initialInfo?: {
    columns?: string[];
    dtypes?: Record<string, string>;
    rows?: number;
    preview?: any[];
    missing_values?: Record<string, { count: number; percentage: number }>;
    numeric_columns?: string[];
    categorical_columns?: string[];
  };
}

export const CHART_TYPES = [
  { id: "echarts_bar", label: "Column (Vert)", icon: BarChart3, desc: "Vertical categorical comparisons & counts" },
  { id: "echarts_horizontal_bar", label: "Bar (Horiz)", icon: AlignLeft, desc: "Horizontal ranking & clean labels" },
  { id: "echarts_line", label: "Line / Trend", icon: TrendingUp, desc: "Trends & series distributions" },
  { id: "echarts_combo", label: "Dual Combo", icon: TrendingUp, desc: "Bar (Left Y) + Line (Right Y) Dual Axis" },
  { id: "echarts_area", label: "Area Fill", icon: Layers, desc: "Cumulative series volume" },
  { id: "echarts_pie", label: "Pie / Donut", icon: PieChart, desc: "Part-to-whole proportions" },
  { id: "echarts_funnel", label: "Funnel Flow", icon: Filter, desc: "Stage conversion step drop-offs" },
  { id: "echarts_radar", label: "Radar Spider", icon: Activity, desc: "Multi-axis indicator radar web" },
  { id: "echarts_scatter", label: "Scatter Plot", icon: Sliders, desc: "Bivariate correlations" },
  { id: "echarts_heatmap", label: "2D Heatmap", icon: Grid, desc: "Matrix intensity (requires 2 dims)" },
  { id: "echarts_boxplot", label: "Boxplot", icon: Table2, desc: "Five-number quartile spread" },
  { id: "echarts_treemap", label: "Treemap", icon: Grid, desc: "Hierarchical rectangular area" },
];

export const AGGREGATIONS = [
  { id: "COUNT", label: "Count of Rows" },
  { id: "SUM", label: "Sum" },
  { id: "AVG", label: "Average (Mean)" },
  { id: "MIN", label: "Minimum" },
  { id: "MAX", label: "Maximum" },
  { id: "MEDIAN", label: "Median" },
  { id: "COUNT_DISTINCT", label: "Count Distinct" },
];

export const TIME_GRAINS = [
  { id: "", label: "Raw Date (No Bucketing)" },
  { id: "day", label: "Day" },
  { id: "week", label: "Week" },
  { id: "month", label: "Month" },
  { id: "quarter", label: "Quarter" },
  { id: "year", label: "Year" },
];

export const FILTER_OPERATORS = [
  { id: "eq", label: "Equals (=)" },
  { id: "neq", label: "Not Equals (!=)" },
  { id: "gt", label: "Greater Than (>)" },
  { id: "gte", label: "Greater / Equal (>=)" },
  { id: "lt", label: "Less Than (<)" },
  { id: "lte", label: "Less / Equal (<=)" },
  { id: "contains", label: "Contains Text" },
  { id: "is_null", label: "Is Null" },
  { id: "is_not_null", label: "Is Not Null" },
];

export const COLOR_PALETTES = [
  { id: "supersetColors", label: "Classic Ocean", preview: ["#1FA8C9", "#454E7C", "#5AC189", "#FF7F44"] },
  { id: "modernDark", label: "Modern Dark", preview: ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"] },
  { id: "airbnb", label: "Airbnb Coral", preview: ["#FF5A5F", "#00A699", "#7B0051", "#007A87"] },
  { id: "d3Category10", label: "D3 Categorical", preview: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] },
  { id: "emerald", label: "Emerald Clean", preview: ["#10b981", "#059669", "#34d399", "#3b82f6"] },
];
