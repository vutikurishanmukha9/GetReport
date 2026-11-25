import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowUpDown,
  BookMarked,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Code2,
  Copy,
  Database,
  Download,
  FileJson,
  History,
  Loader2,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RotateCcw,
  Search,
  Table2,
  Terminal,
  Trash2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, SaveKpiButton } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { api } from "@/services/api";
import type {
  DataRow,
  DatasetInfo,
  GoldenQueryItem,
  PrimitiveValue,
  RecentDatasetItem,
} from "@/types/api";

export interface DuckDBSqlStudioProps {
  taskId?: string | null;
  datasetName?: string;
  info?: DatasetInfo;
  allowDatasetSwitching?: boolean;
  initialSql?: string;
  externalSqlRequest?: { sql: string; nonce: number } | null;
  onTaskChange?: (taskId: string) => void;
}

interface SqlResultState {
  columns: string[];
  records: DataRow[];
  total_returned: number;
  capped: boolean;
  latency_ms: number;
  executed_sql: string;
}

interface SqlSchemaColumn {
  name: string;
  type: string;
  nullable: boolean;
  sample_values: string[];
}

interface SqlSchemaTable {
  table_name: string;
  row_count: number;
  column_count: number;
  columns: SqlSchemaColumn[];
}

interface SqlSchemaState {
  engine: string;
  total_tables: number;
  tables: SqlSchemaTable[];
}

interface QueryHistoryItem {
  sql: string;
  executedAt: string;
  latencyMs?: number;
  rowCount?: number;
  status: "success" | "error";
}

interface QueryRecipe {
  id: string;
  label: string;
  sql: string;
  description: string;
}

type SchemaTypeFilter = "all" | "numeric" | "text" | "temporal";
type InspectorTab = "schema" | "saved" | "history";
type ResultViewMode = "table" | "json";
type QueryRowLimit = 100 | 250 | 500 | 1000;

const DEFAULT_STARTER_SQL = "SELECT * FROM dataset LIMIT 25;";
const ROW_LIMIT_OPTIONS: ReadonlyArray<QueryRowLimit> = [100, 250, 500, 1000];
const PAGE_SIZE_OPTIONS: ReadonlyArray<15 | 25 | 50> = [15, 25, 50];

function isFiniteNumber(value: PrimitiveValue): value is number {
  return (
    value !== null &&
    value !== undefined &&
    value.constructor === Number &&
    Number.isFinite(value)
  );
}

function classifySqlColumnType(colType: string): SchemaTypeFilter {
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

function quoteSqlIdentifier(identifier: string): string {
  return `"${identifier.replace(/"/g, '""')}"`;
}

function formatSqlLiteral(rawSample: string, colType: string): string {
  const category = classifySqlColumnType(colType);
  if (category === "numeric") {
    const parsed = Number(rawSample);
    if (rawSample.trim() !== "" && Number.isFinite(parsed)) {
      return rawSample.trim();
    }
  }
  return `'${rawSample.replace(/'/g, "''")}'`;
}

function formatSqlStatement(rawSql: string): string {
  const trimmed = rawSql.trim();
  if (!trimmed) return "";

  const normalized = trimmed.replace(/\s+/g, " ");
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
  return out;
}

function sanitizeCsvCell(val: PrimitiveValue): string {
  if (val === null || val === undefined) return '""';
  let str = String(val);
  // Neutralize spreadsheet formula injection (CWE-1236)
  if (/^[=+\-@\t\r]/.test(str)) {
    str = "'" + str;
  }
  return `"${str.replace(/"/g, '""')}"`;
}

function getSqlErrorHint(errorMessage: string): string {
  const lower = errorMessage.toLowerCase();
  if (lower.includes("security violation") || lower.includes("not permitted")) {
    return "Only single read-only analytical statements (SELECT, WITH, DESCRIBE, EXPLAIN, SHOW) are permitted against the registered view.";
  }
  if (lower.includes("referenced column") || lower.includes("not found") || lower.includes("binder error")) {
    return "Click any column name in the left Schema Inspector to insert exact double-quoted identifiers (e.g. \"column_name\").";
  }
  return "Verify DuckDB syntax, ensure double quotes around column names with spaces, or run DESCRIBE dataset; to inspect schema.";
}

export const DuckDBSqlStudio: React.FC<DuckDBSqlStudioProps> = ({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  info,
  allowDatasetSwitching = false,
  initialSql,
  externalSqlRequest,
  onTaskChange,
}) => {
  const { toast } = useToast();
  const editorRef = useRef<HTMLTextAreaElement>(null);

  // Active dataset selection
  const [selectedTaskId, setSelectedTaskId] = useState<string>(initialTaskId || "");
  const [availableDatasets, setAvailableDatasets] = useState<RecentDatasetItem[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState<boolean>(false);

  // SQL Buffer & Execution State
  const [sqlQuery, setSqlQuery] = useState<string>(() => {
    const incoming = (initialSql || "").trim();
    return incoming || DEFAULT_STARTER_SQL;
  });
  const [sqlLimit, setSqlLimit] = useState<QueryRowLimit>(500);
  const [sqlRunning, setSqlRunning] = useState<boolean>(false);
  const [sqlResult, setSqlResult] = useState<SqlResultState | null>(null);
  const [sqlError, setSqlError] = useState<string | null>(null);
  const [hasCopiedSql, setHasCopiedSql] = useState<boolean>(false);
  const [hasCopiedJson, setHasCopiedJson] = useState<boolean>(false);

  // Save Golden KPI Inline State
  const [showSavePrompt, setShowSavePrompt] = useState<boolean>(false);
  const [kpiTitleInput, setKpiTitleInput] = useState<string>("");
  const [savingKpi, setSavingKpi] = useState<boolean>(false);
  const [isSavedSql, setIsSavedSql] = useState<boolean>(false);

  // Inspector & Workspace UI State
  const [showInspector, setShowInspector] = useState<boolean>(true);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("schema");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Schema Introspection State
  const [sqlSchema, setSqlSchema] = useState<SqlSchemaState | null>(null);
  const [loadingSchema, setLoadingSchema] = useState<boolean>(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [schemaSearch, setSchemaSearch] = useState<string>("");
  const [schemaTypeFilter, setSchemaTypeFilter] = useState<SchemaTypeFilter>("all");
  const [expandedSchemaCol, setExpandedSchemaCol] = useState<string | null>(null);

  // Saved Golden KPIs & Session History
  const [savedQueries, setSavedQueries] = useState<GoldenQueryItem[]>([]);
  const [loadingSaved, setLoadingSaved] = useState<boolean>(false);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([
    {
      sql: DEFAULT_STARTER_SQL,
      executedAt: "Preset",
      status: "success",
    },
  ]);

  // Results Table Interactive State
  const [resultViewMode, setResultViewMode] = useState<ResultViewMode>("table");
  const [resultsFilter, setResultsFilter] = useState<string>("");
  const [resultsSortCol, setResultsSortCol] = useState<string | null>(null);
  const [resultsSortAsc, setResultsSortAsc] = useState<boolean>(true);
  const [resultsPage, setResultsPage] = useState<number>(1);
  const [resultsPerPage, setResultsPerPage] = useState<15 | 25 | 50>(15);

  // Sync controlled taskId prop
  useEffect(() => {
    if (initialTaskId && initialTaskId !== selectedTaskId) {
      setSelectedTaskId(initialTaskId);
      setSqlSchema(null);
      setSqlResult(null);
      setSqlError(null);
    }
  }, [initialTaskId, selectedTaskId]);

  // Accept incoming SQL from sibling widgets (e.g., AI Chat "Open in SQL Studio")
  useEffect(() => {
    if (externalSqlRequest && externalSqlRequest.sql.trim()) {
      setSqlQuery(externalSqlRequest.sql.trim());
      setIsSavedSql(false);
      setSqlError(null);
      window.setTimeout(() => {
        editorRef.current?.focus();
      }, 60);
    }
  }, [externalSqlRequest]);

  // Load recent datasets when dataset switching is enabled (/analytics)
  useEffect(() => {
    if (!allowDatasetSwitching) return;
    let active = true;
    setLoadingDatasets(true);
    api
      .listRecentDatasets(20)
      .then((res) => {
        if (!active) return;
        const datasets = Array.isArray(res.datasets) ? res.datasets : [];
        setAvailableDatasets(datasets);
        if (!selectedTaskId && datasets.length > 0) {
          const firstId = datasets[0].task_id;
          setSelectedTaskId(firstId);
          onTaskChange?.(firstId);
        }
      })
      .catch(() => {
        if (active) setAvailableDatasets([]);
      })
      .finally(() => {
        if (active) setLoadingDatasets(false);
      });

    return () => {
      active = false;
    };
  }, [allowDatasetSwitching, selectedTaskId, onTaskChange]);

  // Escape key exits fullscreen mode cleanly
  useEffect(() => {
    if (!isFullscreen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsFullscreen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  // Load DuckDB schema & Saved Golden KPIs whenever active task changes
  const loadSchema = useCallback(async (targetTaskId: string) => {
    if (!targetTaskId) return;
    setLoadingSchema(true);
    setSchemaError(null);
    try {
      const data = await api.getSqlSchema(targetTaskId);
      if (data && data.success) {
        setSqlSchema({
          engine: data.engine,
          total_tables: data.total_tables,
          tables: data.tables,
        });
      } else {
        setSchemaError("Schema metadata unavailable for this job.");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Could not introspect DuckDB schema.";
      setSchemaError(msg);
    } finally {
      setLoadingSchema(false);
    }
  }, []);

  const loadSavedGoldenQueries = useCallback(async (targetTaskId: string) => {
    if (!targetTaskId) return;
    setLoadingSaved(true);
    try {
      const list = await api.getGoldenQueries(targetTaskId);
      setSavedQueries(Array.isArray(list) ? list : []);
    } catch {
      setSavedQueries([]);
    } finally {
      setLoadingSaved(false);
    }
  }, []);

  useEffect(() => {
    if (!selectedTaskId) return;
    void loadSchema(selectedTaskId);
    void loadSavedGoldenQueries(selectedTaskId);
  }, [selectedTaskId, loadSchema, loadSavedGoldenQueries]);

  // Execute DuckDB SQL Query
  const executeSql = useCallback(
    async (overrideQuery?: string) => {
      if (!selectedTaskId) {
        toast({
          title: "No Dataset Selected",
          description: "Upload or select an audited dataset before running SQL queries.",
          variant: "destructive",
        });
        return;
      }

      const query = (overrideQuery ?? sqlQuery).trim();
      if (!query) {
        setSqlError("Enter a SQL statement before executing.");
        return;
      }

      setSqlRunning(true);
      setSqlError(null);
      const start = performance.now();

      try {
        const res = await api.querySql(selectedTaskId, query, sqlLimit);
        const elapsed = res.execution_ms ?? Math.round(performance.now() - start);
        const resultCols =
          res.columns.length > 0
            ? res.columns
            : res.records.length > 0
              ? Object.keys(res.records[0])
              : [];

        setSqlResult({
          columns: resultCols,
          records: res.records,
          total_returned: res.total_returned,
          capped: Boolean(res.capped),
          latency_ms: elapsed,
          executed_sql: query,
        });
        setResultsPage(1);
        setResultsFilter("");
        setResultsSortCol(null);
        setResultsSortAsc(true);
        setIsSavedSql(false);

        const timestamp = new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
        setQueryHistory((prev) =>
          [
            {
              sql: query,
              executedAt: timestamp,
              latencyMs: elapsed,
              rowCount: res.total_returned,
              status: "success" as const,
            },
            ...prev.filter((item) => item.sql !== query),
          ].slice(0, 15),
        );

        toast({
          title: "Query Executed",
          description: `Returned ${res.total_returned.toLocaleString()} row${res.total_returned === 1 ? "" : "s"} in ${elapsed}ms`,
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Query execution failed";
        setSqlError(msg);
        const timestamp = new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
        setQueryHistory((prev) =>
          [
            {
              sql: query,
              executedAt: timestamp,
              status: "error" as const,
            },
            ...prev.filter((item) => item.sql !== query),
          ].slice(0, 15),
        );
        toast({
          title: "SQL Execution Failed",
          description: msg,
          variant: "destructive",
        });
      } finally {
        setSqlRunning(false);
      }
    },
    [selectedTaskId, sqlQuery, sqlLimit, toast],
  );

  // Cursor-aware snippet insertion into SQL editor
  const insertSnippetAtCursor = (snippet: string) => {
    const textarea = editorRef.current;
    if (!textarea) {
      setSqlQuery((prev) => {
        const trimmed = prev.trim();
        return trimmed ? `${trimmed} ${snippet}` : snippet;
      });
      return;
    }

    const start = textarea.selectionStart ?? sqlQuery.length;
    const end = textarea.selectionEnd ?? start;
    const before = sqlQuery.slice(0, start);
    const after = sqlQuery.slice(end);
    const needsLeadingSpace = before.length > 0 && !/[\s,(]$/.test(before);
    const needsTrailingSpace = after.length > 0 && !/^[\s,);]/.test(after);
    const inserted = `${needsLeadingSpace ? " " : ""}${snippet}${needsTrailingSpace ? " " : ""}`;
    const nextValue = `${before}${inserted}${after}`;

    setSqlQuery(nextValue);
    window.requestAnimationFrame(() => {
      textarea.focus();
      const cursorPos = before.length + inserted.length;
      textarea.setSelectionRange(cursorPos, cursorPos);
    });
  };

  // Save current SQL as named Golden KPI
  const handleSaveGoldenKpi = async () => {
    if (!selectedTaskId || !sqlQuery.trim()) return;
    const title = kpiTitleInput.trim() || "Verified SQL Benchmark";
    setSavingKpi(true);
    try {
      const saved = await api.saveGoldenQuery(
        selectedTaskId,
        title,
        sqlQuery.trim(),
        "Saved from DuckDB SQL Studio",
      );
      setIsSavedSql(true);
      setShowSavePrompt(false);
      setKpiTitleInput("");
      if (saved?.id) {
        setSavedQueries((prev) => [saved, ...prev.filter((item) => item.id !== saved.id)]);
      }
      toast({
        title: "Saved as Golden KPI",
        description: `"${title}" is now pinned to Saved Benchmarks.`,
      });
    } catch (err: unknown) {
      toast({
        title: "Save Failed",
        description: err instanceof Error ? err.message : "Could not save Golden KPI.",
        variant: "destructive",
      });
    } finally {
      setSavingKpi(false);
    }
  };

  const handleDeleteGoldenKpi = async (queryId: string) => {
    if (!selectedTaskId) return;
    try {
      await api.deleteGoldenQuery(selectedTaskId, queryId);
      setSavedQueries((prev) => prev.filter((item) => item.id !== queryId));
      toast({
        title: "Benchmark Removed",
        description: "Deleted Golden KPI from persisted storage.",
      });
    } catch (err: unknown) {
      toast({
        title: "Delete Failed",
        description: err instanceof Error ? err.message : "Could not remove benchmark.",
        variant: "destructive",
      });
    }
  };

  // Export query results as formula-safe CSV
  const downloadSqlCsv = () => {
    if (!sqlResult || sqlResult.records.length === 0) return;
    const cols = sqlResult.columns;
    const lines = [cols.map(sanitizeCsvCell).join(",")];
    for (const row of sqlResult.records) {
      lines.push(cols.map((c) => sanitizeCsvCell(row[c])).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `duckdb_query_result_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast({ title: "CSV Exported", description: "Downloaded DuckDB result set cleanly." });
  };

  // Active dataset/schema metadata derivation (uses backend introspection or prop info fallback)
  const activeDatasetMeta = useMemo(() => {
    return availableDatasets.find((d) => d.task_id === selectedTaskId);
  }, [availableDatasets, selectedTaskId]);

  const resolvedDatasetName = activeDatasetMeta?.filename || initialDatasetName || "dataset";
  const activeTableMeta = sqlSchema?.tables?.[0];

  const resolvedRowCount =
    activeTableMeta && Number.isFinite(activeTableMeta.row_count)
      ? activeTableMeta.row_count
      : (activeDatasetMeta?.rows ?? info?.rows ?? 0);

  const schemaColumnsList = useMemo<SqlSchemaColumn[]>(() => {
    if (activeTableMeta && Array.isArray(activeTableMeta.columns) && activeTableMeta.columns.length > 0) {
      return activeTableMeta.columns;
    }
    if (info?.columns) {
      return info.columns.map((col) => ({
        name: col,
        type: info.dtypes?.[col] || "VARCHAR",
        nullable: true,
        sample_values: [],
      }));
    }
    if (activeDatasetMeta?.columns) {
      const numericSet = new Set(activeDatasetMeta.numeric_columns || []);
      return activeDatasetMeta.columns.map((col) => ({
        name: col,
        type: numericSet.has(col) ? "DOUBLE" : "VARCHAR",
        nullable: true,
        sample_values: [],
      }));
    }
    return [];
  }, [activeTableMeta, info, activeDatasetMeta]);

  const filteredSchemaColumns = useMemo(() => {
    const searchLower = schemaSearch.toLowerCase().trim();
    return schemaColumnsList.filter((col) => {
      const matchesSearch =
        !searchLower ||
        col.name.toLowerCase().includes(searchLower) ||
        col.type.toLowerCase().includes(searchLower);
      if (!matchesSearch) return false;
      if (schemaTypeFilter === "all") return true;
      return classifySqlColumnType(col.type) === schemaTypeFilter;
    });
  }, [schemaColumnsList, schemaSearch, schemaTypeFilter]);

  // Contextual recipes derived from the real dataset schema
  const contextualRecipes = useMemo<QueryRecipe[]>(() => {
    const firstCatCol =
      schemaColumnsList.find((c) => classifySqlColumnType(c.type) === "text")?.name ||
      info?.categorical_columns?.[0];
    const firstNumCol =
      schemaColumnsList.find((c) => classifySqlColumnType(c.type) === "numeric")?.name ||
      info?.numeric_columns?.[0];
    const tableName = activeTableMeta?.table_name || "dataset";

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
  }, [schemaColumnsList, info, activeTableMeta]);

  // Identify numeric columns inside current sqlResult for right-alignment & tabular styling
  const numericResultCols = useMemo(() => {
    if (!sqlResult) return new Set<string>();
    const schemaTypes = new Map(schemaColumnsList.map((c) => [c.name, classifySqlColumnType(c.type)]));
    const result = new Set<string>();

    for (const col of sqlResult.columns) {
      if (schemaTypes.get(col) === "numeric") {
        result.add(col);
        continue;
      }
      const sampleRow = sqlResult.records.find(
        (r) => r[col] !== null && r[col] !== undefined,
      );
      if (sampleRow && isFiniteNumber(sampleRow[col])) {
        result.add(col);
      }
    }
    return result;
  }, [sqlResult, schemaColumnsList]);

  // Filter + Sort + Paginate SQL Results
  const filteredAndSortedRecords = useMemo(() => {
    if (!sqlResult) return [];
    let list = [...sqlResult.records];

    if (resultsFilter.trim()) {
      const term = resultsFilter.toLowerCase().trim();
      list = list.filter((row) =>
        sqlResult.columns.some((col) => {
          const val = row[col];
          return val !== null && val !== undefined && String(val).toLowerCase().includes(term);
        }),
      );
    }

    if (resultsSortCol) {
      list.sort((a, b) => {
        const valA = a[resultsSortCol];
        const valB = b[resultsSortCol];
        if (valA === valB) return 0;
        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;
        if (isFiniteNumber(valA) && isFiniteNumber(valB)) {
          return resultsSortAsc ? valA - valB : valB - valA;
        }
        const strA = String(valA);
        const strB = String(valB);
        return resultsSortAsc ? strA.localeCompare(strB) : strB.localeCompare(strA);
      });
    }

    return list;
  }, [sqlResult, resultsFilter, resultsSortCol, resultsSortAsc]);

  const totalResultsPages = Math.max(
    1,
    Math.ceil(filteredAndSortedRecords.length / resultsPerPage),
  );

  const paginatedRecords = useMemo(() => {
    const start = (resultsPage - 1) * resultsPerPage;
    return filteredAndSortedRecords.slice(start, start + resultsPerPage);
  }, [filteredAndSortedRecords, resultsPage, resultsPerPage]);

  const lineCount = useMemo(() => Math.max(1, sqlQuery.split("\n").length), [sqlQuery]);

  const formattedJsonPreview = useMemo(() => {
    if (!sqlResult) return "[]";
    return JSON.stringify(filteredAndSortedRecords.slice(0, 100), null, 2);
  }, [sqlResult, filteredAndSortedRecords]);

  return (
    <Card
      className={cn(
        "border border-stone-300/90 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#0B0C10] text-stone-900 dark:text-zinc-100 shadow-sm overflow-hidden transition-all duration-200",
        isFullscreen
          ? "fixed inset-3 sm:inset-5 z-50 rounded-2xl flex flex-col shadow-2xl overflow-y-auto"
          : "rounded-2xl",
      )}
    >
      {/* ─── 1. STUDIO TOP BAR ────────────────────────────────────────────────── */}
      <div className="px-4 sm:px-6 py-3.5 bg-[#EDE8DE]/80 dark:bg-[#11131A] border-b border-stone-300/80 dark:border-white/10 flex flex-col lg:flex-row lg:items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="h-9 w-9 rounded-xl bg-[#800000]/10 border border-[#800000]/25 text-[#800000] dark:bg-emerald-500/10 dark:border-emerald-500/25 dark:text-emerald-400 flex items-center justify-center shrink-0">
            <Terminal className="h-4 w-4" />
          </div>

          <div className="space-y-0.5">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-base sm:text-lg font-display font-bold tracking-tight text-stone-900 dark:text-white">
                DuckDB SQL Studio
              </h3>
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-emerald-500/15 text-emerald-800 border border-emerald-500/25 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/25">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400" />
                In-Process Arrow
              </span>
              <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-medium bg-stone-200/80 text-stone-700 border border-stone-300 dark:bg-white/[0.06] dark:text-zinc-300 dark:border-white/10">
                view: <strong>dataset</strong>
              </span>
            </div>
            <p className="text-[11px] font-sans text-stone-600 dark:text-zinc-400 tabular-nums">
              {selectedTaskId
                ? `${resolvedDatasetName} · ${resolvedRowCount.toLocaleString()} rows × ${schemaColumnsList.length} columns · Read-only sandbox`
                : "Select an audited dataset to run read-only analytical SQL queries"}
            </p>
          </div>
        </div>

        {/* Dataset Switcher + View Mode Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {allowDatasetSwitching && (
            <div className="flex items-center gap-1.5">
              <label htmlFor="duckdb-dataset-select" className="sr-only">
                Active dataset
              </label>
              <select
                id="duckdb-dataset-select"
                value={selectedTaskId}
                disabled={loadingDatasets || availableDatasets.length === 0}
                onChange={(e) => {
                  const nextId = e.target.value;
                  setSelectedTaskId(nextId);
                  setSqlSchema(null);
                  setSqlResult(null);
                  setSqlError(null);
                  onTaskChange?.(nextId);
                }}
                className="h-8 max-w-[230px] rounded-lg border border-stone-300 bg-white px-2.5 text-xs font-mono text-stone-900 dark:border-white/10 dark:bg-[#0B0C10] dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-[#800000]/40 dark:focus:ring-white/30"
              >
                {loadingDatasets ? (
                  <option value="">Loading datasets…</option>
                ) : availableDatasets.length === 0 ? (
                  <option value="">No datasets uploaded</option>
                ) : (
                  availableDatasets.map((ds) => (
                    <option key={ds.task_id} value={ds.task_id}>
                      {ds.filename} ({ds.rows.toLocaleString()} rows)
                    </option>
                  ))
                )}
              </select>
            </div>
          )}

          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setShowInspector((prev) => !prev)}
            className="h-8 rounded-lg border-stone-300 bg-white/90 hover:bg-white text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/[0.09] dark:text-zinc-200 text-xs gap-1.5 cursor-pointer"
            title={showInspector ? "Hide Inspector panel" : "Show Inspector panel"}
          >
            {showInspector ? (
              <PanelLeftClose className="h-3.5 w-3.5" />
            ) : (
              <PanelLeftOpen className="h-3.5 w-3.5" />
            )}
            <span>{showInspector ? "Hide Inspector" : "Show Inspector"}</span>
          </Button>

          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setIsFullscreen((prev) => !prev)}
            className="h-8 rounded-lg border-stone-300 bg-white/90 hover:bg-white text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/[0.09] dark:text-zinc-200 text-xs gap-1.5 cursor-pointer"
            aria-label={isFullscreen ? "Exit fullscreen focus mode" : "Enter fullscreen focus mode"}
            title={isFullscreen ? "Exit Focus Mode (Esc)" : "Expand SQL Studio"}
          >
            {isFullscreen ? (
              <Minimize2 className="h-3.5 w-3.5" />
            ) : (
              <Maximize2 className="h-3.5 w-3.5" />
            )}
            <span className="hidden sm:inline">{isFullscreen ? "Exit Focus" : "Expand"}</span>
          </Button>
        </div>
      </div>

      {/* ─── 2. MAIN WORKBENCH SPLIT ──────────────────────────────────────────── */}
      <div
        className={cn(
          "grid gap-0 flex-1 min-h-0",
          showInspector ? "lg:grid-cols-[310px_minmax(0,1fr)]" : "grid-cols-1",
        )}
      >
        {/* ──── LEFT RAIL: INSPECTOR (SCHEMA / SAVED KPIs / HISTORY) ──── */}
        {showInspector && (
          <aside className="border-b lg:border-b-0 lg:border-r border-stone-300/80 dark:border-white/10 bg-[#F3EFE6]/70 dark:bg-[#0E1016] p-3.5 flex flex-col space-y-3">
            {/* Inspector Switcher Tabs */}
            <div className="grid grid-cols-3 gap-1 p-1 rounded-xl bg-stone-200/80 dark:bg-black/40 border border-stone-300/80 dark:border-white/10 text-[11px] font-sans font-semibold">
              <button
                type="button"
                onClick={() => setInspectorTab("schema")}
                className={cn(
                  "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
                  inspectorTab === "schema"
                    ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
                    : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
                )}
              >
                <Database className="h-3 w-3 shrink-0" />
                <span>Schema</span>
              </button>
              <button
                type="button"
                onClick={() => setInspectorTab("saved")}
                className={cn(
                  "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
                  inspectorTab === "saved"
                    ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
                    : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
                )}
              >
                <BookMarked className="h-3 w-3 shrink-0" />
                <span>Saved</span>
                {savedQueries.length > 0 && (
                  <span className="px-1 rounded bg-stone-200 dark:bg-white/10 text-[10px] font-mono tabular-nums">
                    {savedQueries.length}
                  </span>
                )}
              </button>
              <button
                type="button"
                onClick={() => setInspectorTab("history")}
                className={cn(
                  "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
                  inspectorTab === "history"
                    ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
                    : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
                )}
              >
                <History className="h-3 w-3 shrink-0" />
                <span>History</span>
              </button>
            </div>

            {/* INSPECTOR TAB 1: SCHEMA TREE */}
            {inspectorTab === "schema" && (
              <div className="flex flex-col space-y-2.5 flex-1 min-h-0">
                <div className="p-2.5 rounded-xl bg-white dark:bg-[#131620] border border-stone-200 dark:border-white/10 flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-xs font-bold text-stone-900 dark:text-white">
                        {activeTableMeta?.table_name || "dataset"}
                      </span>
                      <Badge
                        variant="outline"
                        className="text-[10px] font-mono px-1.5 py-0 border-stone-300 dark:border-white/15"
                      >
                        {schemaColumnsList.length} cols
                      </Badge>
                    </div>
                    <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400 tabular-nums block mt-0.5">
                      {resolvedRowCount.toLocaleString()} rows in Arrow view
                    </span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      type="button"
                      onClick={() => {
                        const tableName = activeTableMeta?.table_name || "dataset";
                        const q = `DESCRIBE ${tableName};`;
                        setSqlQuery(q);
                        void executeSql(q);
                      }}
                      disabled={!selectedTaskId}
                      className="px-2 py-1 rounded-md text-[10px] font-mono font-semibold bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/[0.06] dark:hover:bg-white/[0.12] dark:text-zinc-200 dark:border-white/10 cursor-pointer disabled:opacity-40"
                      title="Run DESCRIBE on active table"
                    >
                      DESCRIBE
                    </button>
                    {selectedTaskId && (
                      <button
                        type="button"
                        onClick={() => void loadSchema(selectedTaskId)}
                        disabled={loadingSchema}
                        className="p-1 rounded-md text-stone-500 hover:text-stone-900 hover:bg-stone-100 dark:text-zinc-400 dark:hover:text-white dark:hover:bg-white/10 cursor-pointer"
                        title="Refresh DuckDB schema metadata"
                        aria-label="Refresh schema"
                      >
                        <RotateCcw className={cn("h-3 w-3", loadingSchema && "animate-spin")} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Search & Type Filter */}
                <div className="relative">
                  <Search className="h-3.5 w-3.5 absolute left-2.5 top-2 text-stone-400 dark:text-zinc-500" />
                  <Input
                    type="text"
                    placeholder="Filter columns..."
                    aria-label="Filter schema columns"
                    value={schemaSearch}
                    onChange={(e) => setSchemaSearch(e.target.value)}
                    className="h-7.5 pl-8 text-xs font-mono bg-white dark:bg-[#131620] border-stone-300 dark:border-white/10"
                  />
                </div>

                <div className="flex items-center gap-1 flex-wrap">
                  {(["all", "numeric", "text", "temporal"] as const).map((filterKey) => (
                    <button
                      key={filterKey}
                      type="button"
                      onClick={() => setSchemaTypeFilter(filterKey)}
                      className={cn(
                        "px-2 py-0.5 rounded-md text-[10px] font-mono font-semibold transition-colors cursor-pointer uppercase",
                        schemaTypeFilter === filterKey
                          ? "bg-[#800000] text-white dark:bg-white dark:text-zinc-950"
                          : "bg-stone-200/70 text-stone-700 hover:bg-stone-300/70 dark:bg-white/[0.05] dark:text-zinc-400 dark:hover:text-white",
                      )}
                    >
                      {filterKey}
                    </button>
                  ))}
                </div>

                {schemaError && (
                  <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-[11px] font-sans text-amber-900 dark:text-amber-300">
                    {schemaError}
                  </div>
                )}

                {/* Column Tree List */}
                <div className="overflow-y-auto space-y-1.5 pr-1 max-h-[460px]">
                  {loadingSchema ? (
                    <div className="space-y-2 py-2" aria-label="Loading schema columns">
                      {[1, 2, 3, 4, 5].map((item) => (
                        <div
                          key={item}
                          className="h-8 rounded-lg bg-stone-200/70 dark:bg-white/[0.05] animate-pulse"
                        />
                      ))}
                    </div>
                  ) : filteredSchemaColumns.length === 0 ? (
                    <div className="py-8 text-center text-xs text-stone-500 dark:text-zinc-400 font-sans">
                      {selectedTaskId ? "No columns match filter." : "No active dataset selected."}
                    </div>
                  ) : (
                    filteredSchemaColumns.map((col) => {
                      const colKind = classifySqlColumnType(col.type);
                      const isExpanded = expandedSchemaCol === col.name;

                      return (
                        <div
                          key={col.name}
                          className="rounded-xl border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#131620] hover:border-[#800000]/40 dark:hover:border-white/25 transition-colors overflow-hidden"
                        >
                          <div className="px-2.5 py-2 flex items-center justify-between gap-1.5">
                            <button
                              type="button"
                              onClick={() => insertSnippetAtCursor(quoteSqlIdentifier(col.name))}
                              className="flex items-center gap-2 min-w-0 text-left cursor-pointer bg-transparent border-0 p-0 flex-1"
                              title={`Insert ${quoteSqlIdentifier(col.name)} at cursor`}
                            >
                              <span
                                className={cn(
                                  "px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase shrink-0 border",
                                  colKind === "numeric"
                                    ? "bg-sky-500/10 text-sky-800 border-sky-500/25 dark:text-sky-300 dark:border-sky-500/30"
                                    : colKind === "text"
                                      ? "bg-emerald-500/10 text-emerald-800 border-emerald-500/25 dark:text-emerald-300 dark:border-emerald-500/30"
                                      : colKind === "temporal"
                                        ? "bg-amber-500/10 text-amber-800 border-amber-500/25 dark:text-amber-300 dark:border-amber-500/30"
                                        : "bg-stone-100 text-stone-700 border-stone-300 dark:bg-white/10 dark:text-zinc-300 dark:border-white/15",
                                )}
                              >
                                {colKind === "numeric"
                                  ? "NUM"
                                  : colKind === "text"
                                    ? "TXT"
                                    : colKind === "temporal"
                                      ? "DATE"
                                      : "COL"}
                              </span>
                              <span className="font-mono text-xs font-medium text-stone-900 dark:text-zinc-100 truncate">
                                {col.name}
                              </span>
                            </button>

                            <button
                              type="button"
                              onClick={() => setExpandedSchemaCol(isExpanded ? null : col.name)}
                              className="h-5 w-5 rounded flex items-center justify-center text-stone-500 hover:text-stone-900 hover:bg-stone-100 dark:text-zinc-400 dark:hover:text-white dark:hover:bg-white/10 cursor-pointer shrink-0"
                              aria-label={`Inspect column ${col.name}`}
                              title="Inspect type & sample values"
                            >
                              {isExpanded ? (
                                <ChevronUp className="h-3 w-3" />
                              ) : (
                                <ChevronDown className="h-3 w-3" />
                              )}
                            </button>
                          </div>

                          {isExpanded && (
                            <div className="px-2.5 pb-2.5 pt-2 border-t border-stone-200/70 dark:border-white/10 bg-[#FAF8F5] dark:bg-black/25 space-y-2 text-[11px]">
                              <div className="flex items-center justify-between font-mono text-[10px] text-stone-600 dark:text-zinc-400">
                                <span>{col.type}</span>
                                <span>{col.nullable ? "NULLABLE" : "NOT NULL"}</span>
                              </div>

                              {col.sample_values && col.sample_values.length > 0 && (
                                <div className="space-y-1">
                                  <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-sans block">
                                    Click sample to insert literal:
                                  </span>
                                  <div className="flex flex-wrap gap-1">
                                    {col.sample_values.slice(0, 3).map((val, idx) => (
                                      <button
                                        key={`${col.name}-sample-${idx}`}
                                        type="button"
                                        onClick={() =>
                                          insertSnippetAtCursor(formatSqlLiteral(val, col.type))
                                        }
                                        className="px-1.5 py-0.5 rounded bg-stone-200/70 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 font-mono text-[10px] truncate max-w-[115px] cursor-pointer"
                                        title={`Insert ${formatSqlLiteral(val, col.type)}`}
                                      >
                                        {val}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              )}

                              <div className="flex items-center gap-1 flex-wrap pt-0.5">
                                <button
                                  type="button"
                                  onClick={() => {
                                    const q = `SELECT ${quoteSqlIdentifier(col.name)},\n  COUNT(*) AS count\nFROM dataset\nGROUP BY ${quoteSqlIdentifier(col.name)}\nORDER BY count DESC\nLIMIT 15;`;
                                    setSqlQuery(q);
                                    void executeSql(q);
                                  }}
                                  className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
                                >
                                  Top Values
                                </button>
                                {colKind === "numeric" && (
                                  <button
                                    type="button"
                                    onClick={() => {
                                      const q = `SELECT\n  ROUND(MIN(${quoteSqlIdentifier(col.name)}), 2) AS min_val,\n  ROUND(AVG(${quoteSqlIdentifier(col.name)}), 2) AS avg_val,\n  ROUND(MEDIAN(${quoteSqlIdentifier(col.name)}), 2) AS median_val,\n  ROUND(MAX(${quoteSqlIdentifier(col.name)}), 2) AS max_val\nFROM dataset;`;
                                      setSqlQuery(q);
                                      void executeSql(q);
                                    }}
                                    className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
                                  >
                                    Summary Stats
                                  </button>
                                )}
                                <button
                                  type="button"
                                  onClick={() => {
                                    const q = `SELECT\n  COUNT(*) AS total_rows,\n  COUNT(${quoteSqlIdentifier(col.name)}) AS non_null,\n  COUNT(*) - COUNT(${quoteSqlIdentifier(col.name)}) AS missing\nFROM dataset;`;
                                    setSqlQuery(q);
                                    void executeSql(q);
                                  }}
                                  className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
                                >
                                  Null Check
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            )}

            {/* INSPECTOR TAB 2: SAVED GOLDEN KPIs */}
            {inspectorTab === "saved" && (
              <div className="space-y-2 overflow-y-auto max-h-[540px] pr-1">
                {loadingSaved ? (
                  <div className="py-8 text-center text-xs text-stone-500 dark:text-zinc-400">
                    <Loader2 className="h-4 w-4 animate-spin mx-auto mb-2" />
                    Loading saved KPIs…
                  </div>
                ) : savedQueries.length === 0 ? (
                  <div className="p-4 rounded-xl border border-dashed border-stone-300 dark:border-white/15 text-center space-y-1.5">
                    <BookMarked className="h-5 w-5 mx-auto text-stone-400 dark:text-zinc-500" />
                    <p className="text-xs font-medium text-stone-700 dark:text-zinc-300">
                      No Saved Golden KPIs Yet
                    </p>
                    <p className="text-[11px] text-stone-500 dark:text-zinc-400 leading-relaxed">
                      Run any analytical query and click &ldquo;Save KPI&rdquo; to pin it here.
                    </p>
                  </div>
                ) : (
                  savedQueries.map((gq) => (
                    <div
                      key={gq.id}
                      className="p-2.5 rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#131620] space-y-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-semibold text-stone-900 dark:text-white leading-snug">
                          {gq.question}
                        </span>
                        <button
                          type="button"
                          onClick={() => void handleDeleteGoldenKpi(gq.id)}
                          aria-label={`Delete benchmark ${gq.question}`}
                          title="Delete saved benchmark"
                          className="text-stone-400 hover:text-destructive p-0.5 rounded cursor-pointer shrink-0"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                      <pre className="p-2 rounded-lg bg-[#FAF8F5] dark:bg-black/30 border border-stone-200 dark:border-white/5 text-[10px] font-mono text-stone-700 dark:text-zinc-300 overflow-x-auto max-h-20 whitespace-pre-wrap break-all">
                        {gq.sql_query}
                      </pre>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => {
                            setSqlQuery(gq.sql_query);
                            setIsSavedSql(true);
                          }}
                          className="px-2 py-1 rounded-md text-[10px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 dark:bg-white/10 dark:hover:bg-white/15 dark:text-zinc-200 cursor-pointer"
                        >
                          Load
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setSqlQuery(gq.sql_query);
                            setIsSavedSql(true);
                            void executeSql(gq.sql_query);
                          }}
                          className="px-2 py-1 rounded-md text-[10px] font-semibold bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 cursor-pointer"
                        >
                          Run
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* INSPECTOR TAB 3: SESSION QUERY HISTORY */}
            {inspectorTab === "history" && (
              <div className="space-y-2 overflow-y-auto max-h-[540px] pr-1">
                {queryHistory.map((item, idx) => (
                  <div
                    key={`${item.executedAt}-${idx}-${item.sql}`}
                    className="p-2.5 rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#131620] space-y-1.5"
                  >
                    <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                      <span className="text-stone-500 dark:text-zinc-400">{item.executedAt}</span>
                      <div className="flex items-center gap-1.5 tabular-nums">
                        {item.latencyMs !== undefined && (
                          <span className="text-emerald-700 dark:text-emerald-400 font-semibold">
                            {item.latencyMs}ms
                          </span>
                        )}
                        {item.rowCount !== undefined && (
                          <span className="text-stone-500 dark:text-zinc-400">
                            {item.rowCount} rows
                          </span>
                        )}
                      </div>
                    </div>
                    <pre className="text-[11px] font-mono text-stone-800 dark:text-zinc-200 whitespace-pre-wrap break-all line-clamp-3">
                      {item.sql}
                    </pre>
                    <div className="flex items-center gap-1.5 pt-0.5">
                      <button
                        type="button"
                        onClick={() => setSqlQuery(item.sql)}
                        className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 dark:bg-white/10 dark:hover:bg-white/15 dark:text-zinc-200 cursor-pointer"
                      >
                        Load
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setSqlQuery(item.sql);
                          void executeSql(item.sql);
                        }}
                        className="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-stone-900 text-white hover:bg-stone-800 dark:bg-white dark:text-zinc-950 cursor-pointer"
                      >
                        Re-run
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>
        )}

        {/* ──── RIGHT WORKBENCH: RECIPES, EDITOR & RESULTS CONSOLE ──── */}
        <div className="p-4 sm:p-5 flex flex-col space-y-4 min-w-0">
          {/* Quick Recipes Strip */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
            <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-stone-500 dark:text-zinc-400 mr-1 shrink-0 inline-flex items-center gap-1">
              <Code2 className="h-3 w-3 text-[#800000] dark:text-emerald-400" />
              Recipes:
            </span>
            {contextualRecipes.map((recipe) => (
              <button
                key={recipe.id}
                type="button"
                title={recipe.description}
                disabled={!selectedTaskId || sqlRunning}
                onClick={() => {
                  setSqlQuery(recipe.sql);
                  void executeSql(recipe.sql);
                }}
                className="px-2.5 py-1 rounded-lg text-xs font-sans font-medium bg-[#EDE8DE]/80 hover:bg-[#E5DEC9] text-stone-800 border border-stone-300/80 dark:bg-white/[0.05] dark:hover:bg-white/[0.1] dark:text-zinc-200 dark:border-white/10 transition-colors cursor-pointer shrink-0 disabled:opacity-40"
              >
                {recipe.label}
              </button>
            ))}
          </div>

          {/* ──── THEME-NATIVE SQL EDITOR CONSOLE ──── */}
          <div className="rounded-xl border border-stone-300 dark:border-white/15 bg-white dark:bg-[#0D0F16] shadow-xs overflow-hidden">
            {/* Editor Toolbar */}
            <div className="px-3.5 py-2 bg-[#F3EFE6] dark:bg-[#141721] border-b border-stone-200 dark:border-white/10 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="px-2 py-0.5 rounded-md bg-white dark:bg-white/[0.07] border border-stone-300/80 dark:border-white/10 font-semibold text-stone-800 dark:text-zinc-200">
                  query.sql
                </span>
                <span className="text-[11px] text-stone-500 dark:text-zinc-400 hidden sm:inline tabular-nums">
                  {lineCount} line{lineCount === 1 ? "" : "s"} · {sqlQuery.length} chars
                </span>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <div className="flex items-center gap-1 mr-1">
                  <label
                    htmlFor="duckdb-row-limit"
                    className="text-[11px] font-mono text-stone-600 dark:text-zinc-400"
                  >
                    Max rows:
                  </label>
                  <select
                    id="duckdb-row-limit"
                    value={sqlLimit}
                    onChange={(e) => {
                      const parsed = Number(e.target.value);
                      if (parsed === 100 || parsed === 250 || parsed === 500 || parsed === 1000) {
                        setSqlLimit(parsed);
                      }
                    }}
                    className="h-7 rounded-md border border-stone-300 bg-white px-2 text-xs font-mono text-stone-900 dark:border-white/10 dark:bg-[#0B0C10] dark:text-zinc-200"
                  >
                    {ROW_LIMIT_OPTIONS.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                </div>

                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => setSqlQuery(formatSqlStatement(sqlQuery))}
                  className="h-7 px-2.5 text-xs gap-1 rounded-md border-stone-300 bg-white hover:bg-stone-100 text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/10 dark:text-zinc-200 cursor-pointer"
                  title="Format SQL keywords"
                >
                  <Code2 className="h-3.5 w-3.5" />
                  <span>Format</span>
                </Button>

                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    void navigator.clipboard?.writeText(sqlQuery);
                    setHasCopiedSql(true);
                    window.setTimeout(() => setHasCopiedSql(false), 1800);
                    toast({ title: "Copied", description: "SQL query copied to clipboard." });
                  }}
                  className="h-7 px-2.5 text-xs gap-1 rounded-md border-stone-300 bg-white hover:bg-stone-100 text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/10 dark:text-zinc-200 cursor-pointer"
                  title="Copy SQL query"
                >
                  {hasCopiedSql ? (
                    <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                  ) : (
                    <Copy className="h-3.5 w-3.5" />
                  )}
                  <span>{hasCopiedSql ? "Copied" : "Copy"}</span>
                </Button>

                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSqlQuery("");
                    setSqlResult(null);
                    setSqlError(null);
                    setIsSavedSql(false);
                    setShowSavePrompt(false);
                  }}
                  className="h-7 px-2.5 text-xs gap-1 rounded-md border-stone-300 bg-white hover:bg-rose-50 hover:text-destructive hover:border-destructive/30 text-stone-700 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-rose-500/10 dark:hover:text-rose-400 dark:text-zinc-300 cursor-pointer"
                  title="Clear editor and results"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>Clear</span>
                </Button>

                <Button
                  type="button"
                  size="sm"
                  disabled={sqlRunning || !selectedTaskId}
                  onClick={() => void executeSql()}
                  className="h-7 px-3 text-xs font-semibold gap-1.5 rounded-md bg-[#800000] hover:bg-[#6b0000] text-white dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-950 shadow-xs cursor-pointer"
                >
                  {sqlRunning ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Play className="h-3 w-3 fill-current" />
                  )}
                  <span>Run Query</span>
                </Button>
              </div>
            </div>

            {/* Line-Numbered Monospace Code Surface */}
            <div className="flex items-stretch font-mono text-xs leading-6 bg-white dark:bg-[#0B0D13]">
              <div
                aria-hidden="true"
                className="select-none py-3 px-2.5 text-right text-[11px] leading-6 text-stone-400 dark:text-zinc-600 bg-[#FAF8F5] dark:bg-[#0F121B] border-r border-stone-200 dark:border-white/10 tabular-nums min-w-[2.5rem]"
              >
                {Array.from({ length: lineCount }, (_, idx) => (
                  <div key={`ln-${idx + 1}`}>{idx + 1}</div>
                ))}
              </div>
              <textarea
                ref={editorRef}
                aria-label="SQL Query Input"
                value={sqlQuery}
                onChange={(e) => setSqlQuery(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                    e.preventDefault();
                    void executeSql();
                    return;
                  }
                  if (e.key === "Tab") {
                    e.preventDefault();
                    const el = e.currentTarget;
                    const start = el.selectionStart;
                    const end = el.selectionEnd;
                    const updated = `${sqlQuery.slice(0, start)}  ${sqlQuery.slice(end)}`;
                    setSqlQuery(updated);
                    window.requestAnimationFrame(() => {
                      el.setSelectionRange(start + 2, start + 2);
                    });
                  }
                }}
                rows={Math.max(5, Math.min(12, lineCount + 1))}
                spellCheck={false}
                placeholder="SELECT * FROM dataset WHERE ..."
                className="w-full py-3 px-3.5 bg-transparent text-stone-900 dark:text-emerald-300 font-mono text-xs leading-6 outline-none resize-y placeholder:text-stone-400 dark:placeholder:text-zinc-600 selection:bg-[#800000]/15 dark:selection:bg-emerald-500/20"
              />
            </div>

            {/* Editor Status Footer */}
            <div className="px-3.5 py-1.5 bg-[#FAF8F5] dark:bg-[#10131C] border-t border-stone-200 dark:border-white/10 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-stone-500 dark:text-zinc-400">
              <span>Read-only SQL: SELECT · WITH · DESCRIBE · EXPLAIN · SHOW</span>
              <span className="hidden sm:inline">Ctrl/Cmd + Enter to run · Tab inserts 2 spaces</span>
            </div>
          </div>

          {/* ──── 3. RESULTS CONSOLE ─────────────────────────────────────────── */}
          {sqlRunning ? (
            <div
              className="rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#0D0F16] p-4 space-y-3"
              aria-label="Executing DuckDB query"
            >
              <div className="flex items-center justify-between text-xs font-mono text-stone-600 dark:text-zinc-300">
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-[#800000] dark:text-emerald-400" />
                  Executing vectorized query plan in DuckDB…
                </span>
                <span>Limit {sqlLimit} rows</span>
              </div>
              <div className="space-y-2">
                {[1, 2, 3, 4, 5].map((skeletonRow) => (
                  <div
                    key={skeletonRow}
                    className="h-8 rounded-lg bg-stone-200/70 dark:bg-white/[0.05] animate-pulse"
                  />
                ))}
              </div>
            </div>
          ) : sqlError ? (
            <div className="rounded-xl border border-destructive/35 bg-destructive/10 p-4 text-xs flex items-start gap-3">
              <AlertTriangle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
              <div className="space-y-1.5 flex-1">
                <div className="font-sans font-bold text-sm text-destructive">
                  Query Evaluation Failed
                </div>
                <p className="font-mono text-destructive/95 leading-relaxed">{sqlError}</p>
                <p className="font-sans text-[11px] text-stone-700 dark:text-zinc-300">
                  {getSqlErrorHint(sqlError)}
                </p>
                {selectedTaskId && (
                  <div className="pt-1">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const q = "DESCRIBE dataset;";
                        setSqlQuery(q);
                        void executeSql(q);
                      }}
                      className="h-7 text-xs font-mono border-destructive/30 hover:bg-destructive/15 cursor-pointer"
                    >
                      Run DESCRIBE dataset
                    </Button>
                  </div>
                )}
              </div>
            </div>
          ) : !sqlResult ? (
            <div className="rounded-xl border border-dashed border-stone-300 dark:border-white/15 bg-[#F6F2EA]/70 dark:bg-[#10131C]/70 p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <h4 className="text-sm font-display font-bold text-stone-900 dark:text-white">
                  Ready to Query In-Memory Arrow View
                </h4>
                <p className="text-xs text-stone-600 dark:text-zinc-400 max-w-xl leading-relaxed">
                  Use the Schema Inspector to insert exact quoted column names, pick an analytical recipe above, or run the starter query against <code className="font-mono font-semibold">dataset</code>.
                </p>
              </div>
              <Button
                type="button"
                size="sm"
                disabled={!selectedTaskId}
                onClick={() => void executeSql(sqlQuery || DEFAULT_STARTER_SQL)}
                className="h-9 px-4 rounded-xl bg-[#800000] hover:bg-[#6b0000] text-white dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-950 font-semibold text-xs gap-1.5 shrink-0 cursor-pointer"
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Run Starter Query</span>
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Telemetry & Action Strip */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2.5 text-xs">
                <div className="flex items-center gap-2 flex-wrap font-mono">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-800 dark:text-emerald-400 border border-emerald-500/25 font-semibold tabular-nums">
                    {sqlResult.latency_ms}ms
                  </span>
                  <span className="text-stone-700 dark:text-zinc-300 tabular-nums">
                    Returned <strong>{sqlResult.total_returned.toLocaleString()}</strong> rows ×{" "}
                    <strong>{sqlResult.columns.length}</strong> cols
                  </span>
                  {sqlResult.capped && (
                    <Badge
                      variant="outline"
                      className="text-[10px] font-mono border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-300"
                    >
                      Capped at {sqlLimit}
                    </Badge>
                  )}
                  {sqlResult.records.length !== filteredAndSortedRecords.length && (
                    <span className="text-[11px] text-[#800000] dark:text-emerald-400 font-semibold tabular-nums">
                      ({filteredAndSortedRecords.length.toLocaleString()} matching filter)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  {/* Table / JSON Toggle */}
                  <div className="inline-flex items-center p-0.5 rounded-lg bg-stone-200/80 dark:bg-white/[0.06] border border-stone-300/80 dark:border-white/10">
                    <button
                      type="button"
                      onClick={() => setResultViewMode("table")}
                      className={cn(
                        "px-2 py-1 rounded-md text-[11px] font-sans font-medium flex items-center gap-1 cursor-pointer transition-colors",
                        resultViewMode === "table"
                          ? "bg-white text-stone-900 shadow-2xs dark:bg-zinc-800 dark:text-white"
                          : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white",
                      )}
                    >
                      <Table2 className="h-3 w-3" />
                      <span>Table</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setResultViewMode("json")}
                      className={cn(
                        "px-2 py-1 rounded-md text-[11px] font-sans font-medium flex items-center gap-1 cursor-pointer transition-colors",
                        resultViewMode === "json"
                          ? "bg-white text-stone-900 shadow-2xs dark:bg-zinc-800 dark:text-white"
                          : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white",
                      )}
                    >
                      <FileJson className="h-3 w-3" />
                      <span>JSON</span>
                    </button>
                  </div>

                  {resultViewMode === "table" && (
                    <div className="relative">
                      <Search className="h-3 w-3 absolute left-2.5 top-2 text-stone-400 dark:text-zinc-500" />
                      <Input
                        type="text"
                        placeholder="Filter result rows..."
                        aria-label="Filter result rows"
                        value={resultsFilter}
                        onChange={(e) => {
                          setResultsFilter(e.target.value);
                          setResultsPage(1);
                        }}
                        className="h-7 w-40 pl-7 text-[11px] font-mono bg-white dark:bg-[#131620] border-stone-300 dark:border-white/10"
                      />
                    </div>
                  )}

                  {!showSavePrompt && (
                    <SaveKpiButton
                      isSaved={isSavedSql}
                      onToggleSave={() => {
                        setKpiTitleInput(
                          sqlResult.columns.length > 0
                            ? `${resolvedDatasetName.replace(/\.[^/.]+$/, "")}: ${sqlResult.columns.slice(0, 2).join(" · ")}`
                            : "Verified SQL Benchmark",
                        );
                        setShowSavePrompt(true);
                      }}
                      savedLabel="Saved KPI"
                      defaultLabel="Save KPI"
                      className="h-7 px-2.5 text-xs rounded-lg"
                    />
                  )}

                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={downloadSqlCsv}
                    className="h-7 text-xs font-medium rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/10 dark:text-zinc-200 px-2.5 cursor-pointer gap-1.5"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export CSV</span>
                  </Button>
                </div>
              </div>

              {/* Custom Golden KPI Name Inline Bar */}
              {showSavePrompt && (
                <div className="p-2.5 rounded-xl bg-[#EDE8DE]/80 dark:bg-white/[0.04] border border-stone-300 dark:border-white/10 flex flex-wrap items-center gap-2">
                  <label
                    htmlFor="sql-golden-kpi-title"
                    className="text-xs font-sans font-semibold text-stone-800 dark:text-zinc-200"
                  >
                    Benchmark Title:
                  </label>
                  <Input
                    id="sql-golden-kpi-title"
                    type="text"
                    value={kpiTitleInput}
                    onChange={(e) => setKpiTitleInput(e.target.value)}
                    placeholder="e.g., Top performing segments"
                    className="h-7 flex-1 min-w-[200px] text-xs font-sans bg-white dark:bg-[#0B0C10]"
                  />
                  <Button
                    type="button"
                    size="sm"
                    disabled={savingKpi}
                    onClick={() => void handleSaveGoldenKpi()}
                    className="h-7 px-3 text-xs font-semibold bg-[#800000] hover:bg-[#6b0000] text-white dark:bg-white dark:text-zinc-950 cursor-pointer"
                  >
                    {savingKpi ? "Saving…" : "Pin Benchmark"}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowSavePrompt(false)}
                    className="h-7 px-2.5 text-xs cursor-pointer"
                  >
                    Cancel
                  </Button>
                </div>
              )}

              {/* Results Table vs JSON View */}
              {resultViewMode === "json" ? (
                <div className="rounded-xl border border-stone-300/80 dark:border-white/10 bg-white dark:bg-[#0B0D13] overflow-hidden">
                  <div className="px-3.5 py-2 border-b border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#12151F] flex items-center justify-between text-xs font-mono text-stone-600 dark:text-zinc-400">
                    <span>
                      JSON Preview ({Math.min(100, filteredAndSortedRecords.length)} of{" "}
                      {filteredAndSortedRecords.length} rows)
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        void navigator.clipboard?.writeText(formattedJsonPreview);
                        setHasCopiedJson(true);
                        window.setTimeout(() => setHasCopiedJson(false), 1800);
                        toast({ title: "JSON Copied", description: "Result rows copied as JSON." });
                      }}
                      className="inline-flex items-center gap-1 text-xs font-medium text-stone-800 dark:text-zinc-200 hover:underline cursor-pointer"
                    >
                      {hasCopiedJson ? (
                        <Check className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                      <span>{hasCopiedJson ? "Copied" : "Copy JSON"}</span>
                    </button>
                  </div>
                  <pre className="p-4 text-[11px] font-mono text-stone-800 dark:text-zinc-200 max-h-[380px] overflow-auto leading-relaxed">
                    {formattedJsonPreview}
                  </pre>
                </div>
              ) : (
                <div className="border border-stone-300/90 dark:border-white/10 rounded-xl overflow-hidden bg-white dark:bg-[#0D0F16]">
                  <div className="w-full overflow-auto max-h-[420px]">
                    <Table className="border-collapse">
                      <TableHeader className="bg-[#F3EFE6] dark:bg-[#141721] sticky top-0 z-10">
                        <TableRow className="border-b border-stone-300/80 dark:border-white/10">
                          <TableHead className="w-11 text-center font-mono text-[10px] font-bold text-stone-500 dark:text-zinc-400 px-2 py-2 border-r border-stone-200 dark:border-white/10 select-none">
                            #
                          </TableHead>
                          {sqlResult.columns.map((col) => {
                            const isSorted = resultsSortCol === col;
                            const isNumeric = numericResultCols.has(col);
                            return (
                              <TableHead
                                key={col}
                                onClick={() => {
                                  if (resultsSortCol === col) {
                                    setResultsSortAsc(!resultsSortAsc);
                                  } else {
                                    setResultsSortCol(col);
                                    setResultsSortAsc(true);
                                  }
                                }}
                                className={cn(
                                  "font-mono text-[11px] font-bold text-stone-900 dark:text-zinc-100 px-3 py-2 whitespace-nowrap border-r border-stone-200 dark:border-white/10 last:border-r-0 cursor-pointer hover:bg-stone-200/70 dark:hover:bg-white/[0.06] transition-colors select-none",
                                  isNumeric && "text-right",
                                )}
                                title={`Sort by ${col}`}
                              >
                                <div
                                  className={cn(
                                    "flex items-center gap-1.5",
                                    isNumeric ? "justify-end" : "justify-between",
                                  )}
                                >
                                  <span>{col}</span>
                                  <ArrowUpDown
                                    className={cn(
                                      "h-3 w-3 shrink-0",
                                      isSorted
                                        ? "text-[#800000] dark:text-emerald-400"
                                        : "text-stone-400 dark:text-zinc-600",
                                    )}
                                  />
                                </div>
                              </TableHead>
                            );
                          })}
                        </TableRow>
                      </TableHeader>
                      <TableBody className="divide-y divide-stone-200/70 dark:divide-white/[0.06] bg-white dark:bg-[#0D0F16]">
                        {paginatedRecords.length === 0 ? (
                          <TableRow>
                            <TableCell
                              colSpan={Math.max(1, sqlResult.columns.length + 1)}
                              className="text-center py-8 text-xs text-stone-500 dark:text-zinc-400 font-mono"
                            >
                              {sqlResult.records.length === 0
                                ? "Query succeeded with 0 matching rows."
                                : "No rows match the current filter."}
                            </TableCell>
                          </TableRow>
                        ) : (
                          paginatedRecords.map((row, idx) => {
                            const rowNumber = (resultsPage - 1) * resultsPerPage + idx + 1;
                            return (
                              <TableRow
                                key={`sql-row-${rowNumber}`}
                                className="hover:bg-[#FAF8F5] dark:hover:bg-white/[0.03]"
                              >
                                <TableCell className="text-center font-mono text-[10px] text-stone-400 dark:text-zinc-500 tabular-nums px-2 py-1.5 border-r border-stone-200/80 dark:border-white/[0.06] select-none">
                                  {rowNumber}
                                </TableCell>
                                {sqlResult.columns.map((col) => {
                                  const cellVal = row[col];
                                  const isNumeric = numericResultCols.has(col);
                                  return (
                                    <TableCell
                                      key={`sql-cell-${rowNumber}-${col}`}
                                      onClick={() => {
                                        if (cellVal !== null && cellVal !== undefined) {
                                          void navigator.clipboard?.writeText(String(cellVal));
                                          toast({
                                            title: "Cell Copied",
                                            description: `${col}: ${String(cellVal).slice(0, 48)}`,
                                          });
                                        }
                                      }}
                                      title="Click cell value to copy"
                                      className={cn(
                                        "font-mono text-xs text-stone-900 dark:text-zinc-200 whitespace-nowrap px-3 py-1.5 border-r border-stone-200/80 dark:border-white/[0.06] last:border-r-0 cursor-copy max-w-[320px] truncate",
                                        isNumeric && "text-right tabular-nums",
                                      )}
                                    >
                                      {cellVal === null || cellVal === undefined ? (
                                        <span className="px-1.5 py-0.5 rounded bg-stone-100 dark:bg-white/[0.06] text-stone-400 dark:text-zinc-500 italic font-sans text-[10px]">
                                          null
                                        </span>
                                      ) : isFiniteNumber(cellVal) ? (
                                        cellVal.toLocaleString(undefined, { maximumFractionDigits: 4 })
                                      ) : (
                                        String(cellVal)
                                      )}
                                    </TableCell>
                                  );
                                })}
                              </TableRow>
                            );
                          })
                        )}
                      </TableBody>
                    </Table>
                  </div>

                  {/* Pagination Footer */}
                  {filteredAndSortedRecords.length > 0 && (
                    <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 border-t border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#11131B] text-[11px] font-mono text-stone-600 dark:text-zinc-400 tabular-nums">
                      <div className="flex items-center gap-3">
                        <span>
                          Showing {((resultsPage - 1) * resultsPerPage + 1).toLocaleString()}–
                          {Math.min(
                            resultsPage * resultsPerPage,
                            filteredAndSortedRecords.length,
                          ).toLocaleString()}{" "}
                          of {filteredAndSortedRecords.length.toLocaleString()}
                        </span>
                        <div className="flex items-center gap-1">
                          <span>Rows/page:</span>
                          {PAGE_SIZE_OPTIONS.map((sizeOpt) => (
                            <button
                              key={sizeOpt}
                              type="button"
                              onClick={() => {
                                setResultsPerPage(sizeOpt);
                                setResultsPage(1);
                              }}
                              className={cn(
                                "px-1.5 py-0.5 rounded cursor-pointer",
                                resultsPerPage === sizeOpt
                                  ? "bg-stone-300/80 text-stone-900 font-bold dark:bg-white/15 dark:text-white"
                                  : "hover:text-stone-900 dark:hover:text-white",
                              )}
                            >
                              {sizeOpt}
                            </button>
                          ))}
                        </div>
                      </div>

                      {totalResultsPages > 1 && (
                        <div className="flex items-center gap-1.5">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-6 px-2 text-[11px] cursor-pointer"
                            disabled={resultsPage <= 1}
                            onClick={() => setResultsPage((prev) => Math.max(1, prev - 1))}
                          >
                            <ChevronLeft className="h-3 w-3 mr-0.5" />
                            Prev
                          </Button>
                          <span className="px-1">
                            {resultsPage} / {totalResultsPages}
                          </span>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-6 px-2 text-[11px] cursor-pointer"
                            disabled={resultsPage >= totalResultsPages}
                            onClick={() =>
                              setResultsPage((prev) => Math.min(totalResultsPages, prev + 1))
                            }
                          >
                            Next
                            <ChevronRight className="h-3 w-3 ml-0.5" />
                          </Button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default DuckDBSqlStudio;
