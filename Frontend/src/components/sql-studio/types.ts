import type { DataRow, DatasetInfo, PrimitiveValue } from "@/types/api";
import type { SqlExplainNodeDto } from "@/services/api";

export interface DuckDBSqlStudioProps {
  taskId?: string | null;
  datasetName?: string;
  info?: DatasetInfo;
  allowDatasetSwitching?: boolean;
  initialSql?: string;
  externalSqlRequest?: { sql: string; nonce: number } | null;
  onTaskChange?: (taskId: string) => void;
}

export interface SqlResultState {
  columns: string[];
  column_types?: string[];
  records: DataRow[];
  total_returned: number;
  capped: boolean;
  latency_ms: number;
  executed_sql: string;
}

export interface SqlExplainState {
  sql: string;
  plan: SqlExplainNodeDto[];
  raw_plan: string;
  latency_ms: number;
}

export interface SqlWorksheetTab {
  id: string;
  title: string;
  sql: string;
  isSavedSql: boolean;
}

export interface SqlColumnProfile {
  column: string;
  column_type: string;
  total_records: number;
  non_null_count: number;
  null_count: number;
  approx_distinct: number;
  min_value: PrimitiveValue;
  max_value: PrimitiveValue;
  mean_value: number | null;
  std_deviation: number | null;
}

export interface SqlSchemaColumn {
  name: string;
  type: string;
  nullable: boolean;
  sample_values: string[];
}

export interface SqlSchemaTable {
  table_name: string;
  row_count: number;
  column_count: number;
  columns: SqlSchemaColumn[];
}

export interface SqlSchemaState {
  engine: string;
  total_tables: number;
  tables: SqlSchemaTable[];
}

export interface QueryHistoryItem {
  id: string;
  sql: string;
  executedAt: string;
  latencyMs?: number;
  rowCount?: number;
  status: "success" | "error";
}

export interface QueryRecipe {
  id: string;
  label: string;
  sql: string;
  description: string;
}

export type SchemaTypeFilter = "all" | "numeric" | "text" | "temporal";
export type InspectorTab = "schema" | "saved" | "history";
export type ResultViewMode = "table" | "chart" | "json" | "plan";
export type SqlChartType = "bar" | "horizontal_bar" | "line" | "scatter";
export type QueryRowLimit = 100 | 250 | 500 | 1000;

export const DEFAULT_STARTER_SQL = "SELECT * FROM dataset LIMIT 25;";
export const ROW_LIMIT_OPTIONS: ReadonlyArray<QueryRowLimit> = [100, 250, 500, 1000];
export const PAGE_SIZE_OPTIONS: ReadonlyArray<15 | 25 | 50> = [15, 25, 50];
