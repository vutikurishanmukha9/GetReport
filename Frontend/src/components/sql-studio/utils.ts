import type { DataRow, DatasetInfo, PrimitiveValue } from "@/types/api";
import type { QueryRecipe, SchemaTypeFilter, SqlSchemaColumn } from "./types";

export function isFiniteNumber(value: PrimitiveValue): value is number {
  return (
    value !== null &&
    value !== undefined &&
    value.constructor === Number &&
    Number.isFinite(value)
  );
}

export function classifySqlColumnType(colType: string): SchemaTypeFilter {
  const upper = (colType || "").toUpperCase();
  if (
    upper.includes("INT") ||
    upper.includes("DOUBLE") ||
    upper.includes("FLOAT") ||
    upper.includes("DECIMAL") ||
    upper.includes("NUMERIC") ||
    upper.includes("REAL")
  ) {
    return "numeric";
  }
  if (
    upper.includes("VARCHAR") ||
    upper.includes("TEXT") ||
    upper.includes("STRING") ||
    upper.includes("CHAR")
  ) {
    return "text";
  }
  if (upper.includes("DATE") || upper.includes("TIME") || upper.includes("INTERVAL")) {
    return "temporal";
  }
  return "all";
}

export function quoteSqlIdentifier(identifier: string): string {
  return `"${identifier.replace(/"/g, '""')}"`;
}

export function formatSqlLiteral(rawSample: string, colType: string): string {
  const category = classifySqlColumnType(colType);
  if (category === "numeric") {
    const parsed = Number(rawSample);
    if (rawSample.trim() !== "" && Number.isFinite(parsed)) {
      return rawSample.trim();
    }
  }
  return `'${rawSample.replace(/'/g, "''")}'`;
}

export function formatSqlStatement(rawSql: string): string {
  const trimmed = rawSql.trim();
  if (!trimmed) return "";

  // Temporarily shield single-quoted strings so whitespace/keywords inside literals stay intact
  const stringTokens: string[] = [];
  const shielded = trimmed.replace(/'(?:''|[^'])*'/g, (match) => {
    const token = `__SQL_STR_${stringTokens.length}__`;
    stringTokens.push(match);
    return token;
  });

  const normalized = shielded.replace(/\s+/g, " ");
  const majorClauses = [
    "FROM",
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "CROSS JOIN",
    "JOIN",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "UNION ALL",
    "UNION",
  ];

  let out = normalized;
  for (const clause of majorClauses) {
    const pattern = new RegExp(`\\s+(${clause})\\b`, "gi");
    out = out.replace(pattern, `\n${clause}`);
  }
  out = out.replace(/\s+\b(AND|OR)\b/gi, "\n  $1");

  return out.replace(/__SQL_STR_(\d+)__/g, (_m, idxText: string) => {
    const idx = Number(idxText);
    return stringTokens[idx] ?? "";
  });
}

export function extractSqlTemplateParams(sql: string): string[] {
  const seen = new Set<string>();
  const matches = sql.matchAll(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g);
  for (const m of matches) {
    if (m[1]) seen.add(m[1]);
  }
  return Array.from(seen);
}

export function coerceParamValue(raw: string): PrimitiveValue {
  const trimmed = raw.trim();
  if (trimmed === "") return "";
  if (trimmed.toLowerCase() === "null") return null;
  if (trimmed.toLowerCase() === "true") return true;
  if (trimmed.toLowerCase() === "false") return false;
  if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
    const num = Number(trimmed);
    if (Number.isFinite(num)) return num;
  }
  return raw;
}

export function sanitizeCsvCell(val: PrimitiveValue): string {
  if (val === null || val === undefined) return '""';
  let str = String(val);
  if (/^[=+\-@\t\r]/.test(str)) {
    str = "'" + str;
  }
  return `"${str.replace(/"/g, '""')}"`;
}

export function formatResultAsMarkdown(columns: string[], records: DataRow[], maxRows: number = 100): string {
  if (columns.length === 0) return "";
  const escapeMd = (v: PrimitiveValue) =>
    v === null || v === undefined ? "_null_" : String(v).replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
  const header = `| ${columns.map(escapeMd).join(" | ")} |`;
  const divider = `| ${columns.map(() => "---").join(" | ")} |`;
  const rows = records.slice(0, maxRows).map((r) => `| ${columns.map((c) => escapeMd(r[c])).join(" | ")} |`);
  return [header, divider, ...rows].join("\n");
}

export function getSqlErrorHint(errorMessage: string): string {
  const lower = errorMessage.toLowerCase();
  if (lower.includes("security violation") || lower.includes("not permitted")) {
    return "Only single read-only analytical statements (SELECT, WITH, DESCRIBE, EXPLAIN, SHOW) are permitted against the registered view.";
  }
  if (lower.includes("missing parameter")) {
    return "Fill in all detected {{ variable }} parameter values above the editor before executing.";
  }
  if (lower.includes("referenced column") || lower.includes("not found") || lower.includes("binder error")) {
    return 'Click any column name in the left Schema Inspector to insert exact double-quoted identifiers (e.g. "column_name").';
  }
  return "Verify DuckDB syntax, ensure double quotes around column names with spaces, or run DESCRIBE dataset; to inspect schema.";
}

export function buildSqlRecipes(
  schemaColumnsList: SqlSchemaColumn[],
  tableName: string,
  info?: DatasetInfo,
): QueryRecipe[] {
  const firstCatCol =
    schemaColumnsList.find((c) => classifySqlColumnType(c.type) === "text")?.name ||
    info?.categorical_columns?.[0];
  const firstNumCol =
    schemaColumnsList.find((c) => classifySqlColumnType(c.type) === "numeric")?.name ||
    info?.numeric_columns?.[0];

  const list: QueryRecipe[] = [
    {
      id: "preview",
      label: "Preview 25 Rows",
      sql: `SELECT *\nFROM ${tableName}\nLIMIT 25;`,
      description: "Sample top records from cleaned Arrow table",
    },
    {
      id: "count",
      label: "Row Count",
      sql: `SELECT COUNT(*) AS total_rows\nFROM ${tableName};`,
      description: "Exact O(1) Parquet metadata row count",
    },
    {
      id: "describe",
      label: "Describe Columns",
      sql: `DESCRIBE ${tableName};`,
      description: "Physical column types, nullability, and defaults",
    },
  ];

  if (firstCatCol) {
    const quoted = quoteSqlIdentifier(firstCatCol);
    list.push({
      id: "top_categories",
      label: `Top ${firstCatCol}`,
      sql: `SELECT ${quoted},\n  COUNT(*) AS row_count,\n  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_share\nFROM ${tableName}\nGROUP BY ${quoted}\nORDER BY row_count DESC\nLIMIT 15;`,
      description: `Group distribution across ${firstCatCol}`,
    });
  }

  if (firstNumCol) {
    const quoted = quoteSqlIdentifier(firstNumCol);
    list.push({
      id: "quantiles",
      label: `Stats: ${firstNumCol}`,
      sql: `SELECT\n  COUNT(${quoted}) AS valid_count,\n  ROUND(MIN(${quoted}), 2) AS min_val,\n  ROUND(QUANTILE_CONT(${quoted}, 0.25), 2) AS p25,\n  ROUND(MEDIAN(${quoted}), 2) AS median,\n  ROUND(AVG(${quoted}), 2) AS mean_val,\n  ROUND(QUANTILE_CONT(${quoted}, 0.75), 2) AS p75,\n  ROUND(MAX(${quoted}), 2) AS max_val\nFROM ${tableName};`,
      description: `Summary statistics and quantiles for ${firstNumCol}`,
    });
    list.push({
      id: "outliers",
      label: `Outliers: ${firstNumCol}`,
      sql: `WITH stats AS (\n  SELECT AVG(${quoted}) AS mu, STDDEV_SAMP(${quoted}) AS sigma\n  FROM ${tableName}\n)\nSELECT *\nFROM ${tableName}, stats\nWHERE stats.sigma > 0\n  AND ABS(${quoted} - stats.mu) > 2 * stats.sigma\nLIMIT 25;`,
      description: `Rows exceeding 2 sigma in ${firstNumCol}`,
    });
  }

  return list;
}
