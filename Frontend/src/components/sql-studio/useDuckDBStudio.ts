import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { EditorView } from "@codemirror/view";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import type { DataRow, GoldenQueryItem, RecentDatasetItem } from "@/types/api";
import {
  DEFAULT_STARTER_SQL,
  type DuckDBSqlStudioProps,
  type QueryHistoryItem,
  type QueryRowLimit,
  type ResultViewMode,
  type SqlColumnProfile,
  type SqlExplainState,
  type SqlResultState,
  type SqlSchemaColumn,
  type SqlWorksheetTab,
} from "./types";
import {
  createDefaultWorksheet,
  loadPersistedHistory,
  loadPersistedWorksheets,
  savePersistedHistory,
  savePersistedWorksheets,
} from "./persistence";
import {
  buildSqlRecipes,
  classifySqlColumnType,
  coerceParamValue,
  extractSqlTemplateParams,
  isFiniteNumber,
  sanitizeCsvCell,
} from "./utils";

function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function useDuckDBStudio({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  info,
  allowDatasetSwitching = false,
  initialSql,
  externalSqlRequest,
  onTaskChange,
}: DuckDBSqlStudioProps) {
  const { toast } = useToast();
  const editorViewRef = useRef<EditorView | null>(null);
  const queryAbortRef = useRef<AbortController | null>(null);
  const historySeqRef = useRef<number>(0);
  const tabSeqRef = useRef<number>(1);
  const schemaReqSeqRef = useRef<number>(0);
  const queryReqSeqRef = useRef<number>(0);
  const handledExternalNonceRef = useRef<number | undefined>(externalSqlRequest?.nonce);

  const [taskSelection, setTaskSelection] = useState<{ propId: string | undefined; activeId: string }>({
    propId: initialTaskId,
    activeId: initialTaskId || "",
  });
  const selectedTaskId =
    initialTaskId && taskSelection.propId !== initialTaskId ? initialTaskId : taskSelection.activeId;

  const [availableDatasets, setAvailableDatasets] = useState<RecentDatasetItem[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState<boolean>(false);

  const initialTabSql = (externalSqlRequest?.sql || initialSql || "").trim() || DEFAULT_STARTER_SQL;
  const [worksheets, setWorksheets] = useState<SqlWorksheetTab[]>(() => {
    if (initialTaskId && !externalSqlRequest?.sql && !initialSql) {
      const stored = loadPersistedWorksheets(initialTaskId);
      if (stored && stored.length > 0) return stored;
    }
    return [createDefaultWorksheet(initialTabSql, 1)];
  });
  const [activeWorksheetId, setActiveWorksheetId] = useState<string>(() => worksheets[0]?.id || "tab-1");

  const activeWorksheet = useMemo(
    () => worksheets.find((w) => w.id === activeWorksheetId) ?? worksheets[0],
    [worksheets, activeWorksheetId],
  );
  const sqlQuery = activeWorksheet?.sql ?? DEFAULT_STARTER_SQL;
  const isSavedSql = activeWorksheet?.isSavedSql ?? false;

  const setSqlQuery = useCallback(
    (updater: string | ((prev: string) => string)) => {
      setWorksheets((prevTabs) =>
        prevTabs.map((tab) => {
          if (tab.id !== (activeWorksheet?.id ?? prevTabs[0]?.id)) return tab;
          const nextSql = updater instanceof Function ? updater(tab.sql) : updater;
          return { ...tab, sql: nextSql, isSavedSql: false };
        }),
      );
    },
    [activeWorksheet?.id],
  );

  const setIsSavedSql = useCallback(
    (saved: boolean) => {
      setWorksheets((prevTabs) =>
        prevTabs.map((tab) =>
          tab.id === (activeWorksheet?.id ?? prevTabs[0]?.id) ? { ...tab, isSavedSql: saved } : tab,
        ),
      );
    },
    [activeWorksheet?.id],
  );

  useEffect(() => {
    if (!externalSqlRequest || !externalSqlRequest.sql.trim()) return;
    if (handledExternalNonceRef.current === externalSqlRequest.nonce) return;
    handledExternalNonceRef.current = externalSqlRequest.nonce;
    const incomingSql = externalSqlRequest.sql.trim();
    setWorksheets((prev) =>
      prev.map((tab, idx) =>
        tab.id === activeWorksheetId || idx === 0 ? { ...tab, sql: incomingSql, isSavedSql: false } : tab,
      ),
    );
    const timer = window.setTimeout(() => editorViewRef.current?.focus(), 60);
    return () => window.clearTimeout(timer);
  }, [externalSqlRequest, activeWorksheetId]);

  useEffect(() => {
    if (!selectedTaskId) return;
    savePersistedWorksheets(selectedTaskId, worksheets);
  }, [selectedTaskId, worksheets]);

  const [selectedSql, setSelectedSql] = useState<string>("");
  const [cursorLine, setCursorLine] = useState<number>(1);
  const [cursorCol, setCursorCol] = useState<number>(1);
  const [paramValues, setParamValues] = useState<Record<string, string>>({});
  const detectedParams = useMemo(() => extractSqlTemplateParams(sqlQuery), [sqlQuery]);

  const [sqlLimit, setSqlLimit] = useState<QueryRowLimit>(500);
  const [sqlRunning, setSqlRunning] = useState<boolean>(false);
  const [resultViewMode, setResultViewMode] = useState<ResultViewMode>("table");
  const [sqlResultState, setSqlResultState] = useState<{ taskId: string; data: SqlResultState } | null>(null);
  const sqlResult = sqlResultState && sqlResultState.taskId === selectedTaskId ? sqlResultState.data : null;
  const [sqlExplainState, setSqlExplainState] = useState<{ taskId: string; data: SqlExplainState } | null>(null);
  const sqlExplain = sqlExplainState && sqlExplainState.taskId === selectedTaskId ? sqlExplainState.data : null;
  const [sqlErrorState, setSqlErrorState] = useState<{ taskId: string; message: string } | null>(null);
  const sqlError = sqlErrorState && sqlErrorState.taskId === selectedTaskId ? sqlErrorState.message : null;

  const [showSavePrompt, setShowSavePrompt] = useState<boolean>(false);
  const [kpiTitleInput, setKpiTitleInput] = useState<string>("");
  const [savingKpi, setSavingKpi] = useState<boolean>(false);

  const [sqlSchemaState, setSqlSchemaState] = useState<{ taskId: string; data: { engine: string; total_tables: number; tables: import("./types").SqlSchemaTable[] } } | null>(null);
  const sqlSchema = sqlSchemaState && sqlSchemaState.taskId === selectedTaskId ? sqlSchemaState.data : null;
  const [loadingSchema, setLoadingSchema] = useState<boolean>(false);
  const [schemaErrorState, setSchemaErrorState] = useState<{ taskId: string; message: string } | null>(null);
  const schemaError = schemaErrorState && schemaErrorState.taskId === selectedTaskId ? schemaErrorState.message : null;

  const [columnProfiles, setColumnProfiles] = useState<Record<string, SqlColumnProfile>>({});
  const [loadingProfileCol, setLoadingProfileCol] = useState<string | null>(null);

  const [savedQueries, setSavedQueries] = useState<GoldenQueryItem[]>([]);
  const [loadingSaved, setLoadingSaved] = useState<boolean>(false);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>(() => {
    if (initialTaskId) {
      const stored = loadPersistedHistory(initialTaskId);
      if (stored) return stored;
    }
    return [{ id: "preset-starter", sql: DEFAULT_STARTER_SQL, executedAt: "Preset", status: "success" }];
  });

  useEffect(() => {
    if (!selectedTaskId) return;
    savePersistedHistory(selectedTaskId, queryHistory);
  }, [selectedTaskId, queryHistory]);

  const [resultsFilter, setResultsFilter] = useState<string>("");
  const [resultsSortCol, setResultsSortCol] = useState<string | null>(null);
  const [resultsSortAsc, setResultsSortAsc] = useState<boolean>(true);
  const [resultsPage, setResultsPage] = useState<number>(1);
  const [resultsPerPage, setResultsPerPage] = useState<15 | 25 | 50>(15);

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
          setTaskSelection({ propId: initialTaskId, activeId: firstId });
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
  }, [allowDatasetSwitching, initialTaskId, selectedTaskId, onTaskChange]);

  const loadSchema = useCallback((targetTaskId: string) => {
    if (!targetTaskId) return;
    const reqId = ++schemaReqSeqRef.current;
    setLoadingSchema(true);
    api
      .getSqlSchema(targetTaskId)
      .then((data) => {
        if (schemaReqSeqRef.current !== reqId) return;
        if (data && data.success) {
          setSchemaErrorState(null);
          setSqlSchemaState({
            taskId: targetTaskId,
            data: { engine: data.engine, total_tables: data.total_tables, tables: data.tables },
          });
        } else {
          setSchemaErrorState({ taskId: targetTaskId, message: "Schema metadata unavailable for this job." });
        }
        setLoadingSchema(false);
      })
      .catch((err) => {
        if (schemaReqSeqRef.current !== reqId) return;
        const msg = err instanceof Error ? err.message : "Could not introspect DuckDB schema.";
        setSchemaErrorState({ taskId: targetTaskId, message: msg });
        setLoadingSchema(false);
      });
  }, []);

  const loadSavedGoldenQueries = useCallback((targetTaskId: string) => {
    if (!targetTaskId) return;
    setLoadingSaved(true);
    api
      .getGoldenQueries(targetTaskId)
      .then((list) => {
        setSavedQueries(Array.isArray(list) ? list : []);
      })
      .catch(() => {
        setSavedQueries([]);
      })
      .finally(() => {
        setLoadingSaved(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedTaskId) return;
    loadSchema(selectedTaskId);
    loadSavedGoldenQueries(selectedTaskId);
  }, [selectedTaskId, loadSchema, loadSavedGoldenQueries]);

  const buildBoundParams = useCallback(
    (targetSql: string): DataRow | undefined => {
      const keys = extractSqlTemplateParams(targetSql);
      if (keys.length === 0) return undefined;
      const bound: DataRow = {};
      for (const k of keys) {
        bound[k] = coerceParamValue(paramValues[k] ?? "");
      }
      return bound;
    },
    [paramValues],
  );

  const cancelQuery = useCallback(() => {
    if (queryAbortRef.current) {
      queryAbortRef.current.abort();
      queryAbortRef.current = null;
      setSqlRunning(false);
      toast({ title: "Query Cancelled", description: "Aborted in-flight DuckDB request." });
    }
  }, [toast]);

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

      const targetTaskId = selectedTaskId;
      const candidateSql = overrideQuery !== undefined ? overrideQuery : selectedSql.trim() ? selectedSql : sqlQuery;
      const query = candidateSql.trim();
      if (!query) {
        setSqlErrorState({ taskId: targetTaskId, message: "Enter a SQL statement before executing." });
        return;
      }

      queryAbortRef.current?.abort();
      const controller = new AbortController();
      queryAbortRef.current = controller;

      const reqId = ++queryReqSeqRef.current;
      setSqlRunning(true);
      const start = performance.now();

      try {
        const boundParams = buildBoundParams(query);
        const queryOpts = boundParams
          ? { params: boundParams, signal: controller.signal }
          : { signal: controller.signal };
        const res = await api.querySql(targetTaskId, query, sqlLimit, queryOpts);
        if (queryReqSeqRef.current !== reqId) return;
        const elapsed = res.execution_ms ?? Math.round(performance.now() - start);
        const resultCols =
          res.columns.length > 0 ? res.columns : res.records.length > 0 ? Object.keys(res.records[0]) : [];

        setSqlErrorState(null);
        setSqlResultState({
          taskId: targetTaskId,
          data: {
            columns: resultCols,
            column_types: res.column_types,
            records: res.records,
            total_returned: res.total_returned,
            capped: Boolean(res.capped),
            latency_ms: elapsed,
            executed_sql: query,
          },
        });
        if (resultViewMode === "plan") setResultViewMode("table");
        setResultsPage(1);
        setResultsFilter("");
        setResultsSortCol(null);
        setResultsSortAsc(true);

        const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        const historyId = `hist-${++historySeqRef.current}`;
        setQueryHistory((prev) =>
          [
            {
              id: historyId,
              sql: query,
              executedAt: timestamp,
              latencyMs: elapsed,
              rowCount: res.total_returned,
              status: "success" as const,
            },
            ...prev.filter((item) => item.sql !== query),
          ].slice(0, 30),
        );
        toast({
          title: "Query Executed",
          description: `Returned ${res.total_returned.toLocaleString()} row${res.total_returned === 1 ? "" : "s"} in ${elapsed}ms`,
        });
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        if (queryReqSeqRef.current !== reqId) return;
        const msg = err instanceof Error ? err.message : "Query execution failed";
        setSqlErrorState({ taskId: targetTaskId, message: msg });
        const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        const historyId = `hist-${++historySeqRef.current}`;
        setQueryHistory((prev) =>
          [
            { id: historyId, sql: query, executedAt: timestamp, status: "error" as const },
            ...prev.filter((item) => item.sql !== query),
          ].slice(0, 30),
        );
        toast({ title: "SQL Execution Failed", description: msg, variant: "destructive" });
      } finally {
        if (queryReqSeqRef.current === reqId) {
          queryAbortRef.current = null;
          setSqlRunning(false);
        }
      }
    },
    [selectedTaskId, selectedSql, sqlQuery, sqlLimit, buildBoundParams, resultViewMode, toast],
  );

  const runExplainSql = useCallback(async () => {
    if (!selectedTaskId) return;
    const query = (selectedSql.trim() || sqlQuery).trim();
    if (!query) return;
    setSqlRunning(true);
    const start = performance.now();
    try {
      const boundParams = buildBoundParams(query);
      const res = await api.explainSql(selectedTaskId, query, boundParams ? { params: boundParams } : undefined);
      setSqlErrorState(null);
      setSqlExplainState({
        taskId: selectedTaskId,
        data: {
          sql: res.sql,
          plan: res.plan,
          raw_plan: res.raw_plan,
          latency_ms: Math.round(performance.now() - start),
        },
      });
      setResultViewMode("plan");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to inspect query plan";
      setSqlErrorState({ taskId: selectedTaskId, message: msg });
      toast({ title: "Explain Failed", description: msg, variant: "destructive" });
    } finally {
      setSqlRunning(false);
    }
  }, [selectedTaskId, selectedSql, sqlQuery, buildBoundParams, toast]);

  const loadColumnProfile = useCallback(
    async (colName: string) => {
      if (!selectedTaskId) return;
      const cacheKey = `${selectedTaskId}:${colName}`;
      if (columnProfiles[cacheKey]) return;
      setLoadingProfileCol(colName);
      try {
        const res = await api.getSqlColumnProfile(selectedTaskId, colName);
        if (res.success) {
          setColumnProfiles((prev) => ({
            ...prev,
            [cacheKey]: {
              column: res.column,
              column_type: res.column_type,
              total_records: res.total_records,
              non_null_count: res.non_null_count,
              null_count: res.null_count,
              approx_distinct: res.approx_distinct,
              min_value: res.min_value,
              max_value: res.max_value,
              mean_value: res.mean_value,
              std_deviation: res.std_deviation,
            },
          }));
        }
      } catch {
        // Non-blocking fallback
      } finally {
        setLoadingProfileCol((curr) => (curr === colName ? null : curr));
      }
    },
    [selectedTaskId, columnProfiles],
  );

  const insertSnippetAtCursor = useCallback(
    (snippet: string) => {
      const view = editorViewRef.current;
      if (!view) {
        setSqlQuery((prev) => (prev.trim() ? `${prev.trim()} ${snippet}` : snippet));
        return;
      }
      const { from, to } = view.state.selection.main;
      const doc = view.state.doc.toString();
      const before = doc.slice(0, from);
      const after = doc.slice(to);
      const needsLeadingSpace = before.length > 0 && !/[\s,(]$/.test(before);
      const needsTrailingSpace = after.length > 0 && !/^[\s,);]/.test(after);
      const inserted = `${needsLeadingSpace ? " " : ""}${snippet}${needsTrailingSpace ? " " : ""}`;
      view.dispatch({
        changes: { from, to, insert: inserted },
        selection: { anchor: from + inserted.length },
      });
      view.focus();
    },
    [setSqlQuery],
  );

  const addWorksheet = useCallback(() => {
    const nextIdx = ++tabSeqRef.current;
    const newTab = createDefaultWorksheet(DEFAULT_STARTER_SQL, nextIdx);
    setWorksheets((prev) => [...prev, newTab]);
    setActiveWorksheetId(newTab.id);
  }, []);

  const closeWorksheet = useCallback(
    (tabId: string) => {
      if (worksheets.length <= 1) return;
      const idx = worksheets.findIndex((t) => t.id === tabId);
      const remaining = worksheets.filter((t) => t.id !== tabId);
      setWorksheets(remaining);
      if (activeWorksheetId === tabId) {
        const fallback = remaining[Math.max(0, idx - 1)] ?? remaining[0];
        if (fallback) setActiveWorksheetId(fallback.id);
      }
    },
    [worksheets, activeWorksheetId],
  );

  const handleSaveGoldenKpi = async () => {
    if (!selectedTaskId || !sqlQuery.trim()) return;
    const title = kpiTitleInput.trim() || "Verified SQL Benchmark";
    setSavingKpi(true);
    try {
      const saved = await api.saveGoldenQuery(selectedTaskId, title, sqlQuery.trim(), "Saved from DuckDB SQL Studio");
      setIsSavedSql(true);
      setShowSavePrompt(false);
      setKpiTitleInput("");
      if (saved?.id) {
        setSavedQueries((prev) => [saved, ...prev.filter((item) => item.id !== saved.id)]);
      }
      toast({ title: "Saved as Golden KPI", description: `"${title}" is now pinned to Saved Benchmarks.` });
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

  const handleUpdateGoldenKpi = async (queryId: string, question: string) => {
    if (!selectedTaskId || !question.trim()) return;
    try {
      const updated = await api.updateGoldenQuery(selectedTaskId, queryId, { question: question.trim() });
      setSavedQueries((prev) => prev.map((q) => (q.id === queryId ? updated : q)));
      toast({ title: "Benchmark Updated", description: `Renamed KPI to "${updated.question}".` });
    } catch (err: unknown) {
      toast({
        title: "Rename Failed",
        description: err instanceof Error ? err.message : "Could not update benchmark.",
        variant: "destructive",
      });
    }
  };

  const handleDeleteGoldenKpi = async (queryId: string) => {
    if (!selectedTaskId) return;
    try {
      await api.deleteGoldenQuery(selectedTaskId, queryId);
      setSavedQueries((prev) => prev.filter((item) => item.id !== queryId));
      toast({ title: "Benchmark Removed", description: "Deleted Golden KPI from persisted storage." });
    } catch (err: unknown) {
      toast({
        title: "Delete Failed",
        description: err instanceof Error ? err.message : "Could not remove benchmark.",
        variant: "destructive",
      });
    }
  };

  const downloadSqlCsv = () => {
    if (!sqlResult || sqlResult.records.length === 0) return;
    const cols = sqlResult.columns;
    const lines = [cols.map(sanitizeCsvCell).join(",")];
    for (const row of sqlResult.records) {
      lines.push(cols.map((c) => sanitizeCsvCell(row[c])).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    triggerBlobDownload(blob, `duckdb_query_result_${new Date().toISOString().slice(0, 10)}.csv`);
    toast({ title: "CSV Exported", description: "Downloaded DuckDB result preview cleanly." });
  };

  const exportFullSqlResult = async (format: "csv" | "json") => {
    if (!selectedTaskId || !sqlQuery.trim()) return;
    try {
      const targetSql = (selectedSql.trim() || sqlQuery).trim();
      const boundParams = buildBoundParams(targetSql);
      const blob = await api.downloadSqlExportBlob(selectedTaskId, targetSql, format, boundParams);
      triggerBlobDownload(blob, `duckdb_full_export_${new Date().toISOString().slice(0, 10)}.${format}`);
      toast({ title: "Full Export Downloaded", description: `Streamed query output as ${format.toUpperCase()}.` });
    } catch (err: unknown) {
      toast({
        title: "Export Failed",
        description: err instanceof Error ? err.message : "Failed to stream full SQL export.",
        variant: "destructive",
      });
    }
  };

  const activeDatasetMeta = useMemo(
    () => availableDatasets.find((d) => d.task_id === selectedTaskId),
    [availableDatasets, selectedTaskId],
  );
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

  const contextualRecipes = useMemo(
    () => buildSqlRecipes(schemaColumnsList, activeTableMeta?.table_name || "dataset", info),
    [schemaColumnsList, info, activeTableMeta],
  );

  const numericResultCols = useMemo(() => {
    if (!sqlResult) return new Set<string>();
    const schemaTypes = new Map(schemaColumnsList.map((c) => [c.name, classifySqlColumnType(c.type)]));
    const result = new Set<string>();
    for (const [idx, col] of sqlResult.columns.entries()) {
      const explicitType = sqlResult.column_types?.[idx];
      if (explicitType && classifySqlColumnType(explicitType) === "numeric") {
        result.add(col);
        continue;
      }
      if (schemaTypes.get(col) === "numeric") {
        result.add(col);
        continue;
      }
      const sampleRow = sqlResult.records.find((r) => r[col] !== null && r[col] !== undefined);
      if (sampleRow && isFiniteNumber(sampleRow[col])) result.add(col);
    }
    return result;
  }, [sqlResult, schemaColumnsList]);

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

  return {
    toast,
    editorViewRef,
    selectedTaskId,
    selectDataset: (nextId: string) => {
      setTaskSelection({ propId: initialTaskId, activeId: nextId });
      setSqlSchemaState(null);
      setSqlResultState(null);
      setSqlExplainState(null);
      setSqlErrorState(null);
      setResultViewMode("table");
      const storedTabs = loadPersistedWorksheets(nextId);
      if (storedTabs && storedTabs.length > 0) {
        setWorksheets(storedTabs);
        setActiveWorksheetId(storedTabs[0].id);
      }
      const storedHist = loadPersistedHistory(nextId);
      setQueryHistory(
        storedHist ?? [{ id: "preset-starter", sql: DEFAULT_STARTER_SQL, executedAt: "Preset", status: "success" }],
      );
      onTaskChange?.(nextId);
    },
    clearAll: () => {
      setSqlQuery("");
      setSqlResultState(null);
      setSqlExplainState(null);
      setSqlErrorState(null);
      setShowSavePrompt(false);
    },
    availableDatasets,
    loadingDatasets,
    worksheets,
    activeWorksheetId: activeWorksheet?.id ?? activeWorksheetId,
    setActiveWorksheetId,
    addWorksheet,
    closeWorksheet,
    sqlQuery,
    setSqlQuery,
    selectedSql,
    cursorLine,
    cursorCol,
    onSelectionChange: (sel: string, line: number, col: number) => {
      setSelectedSql(sel);
      setCursorLine(line);
      setCursorCol(col);
    },
    detectedParams,
    paramValues,
    setParamValue: (key: string, val: string) => setParamValues((prev) => ({ ...prev, [key]: val })),
    sqlLimit,
    setSqlLimit,
    sqlRunning,
    cancelQuery,
    resultViewMode,
    setResultViewMode,
    sqlResult,
    sqlExplain,
    sqlError,
    showSavePrompt,
    setShowSavePrompt,
    kpiTitleInput,
    setKpiTitleInput,
    savingKpi,
    isSavedSql,
    setIsSavedSql,
    loadingSchema,
    schemaError,
    columnProfiles,
    loadingProfileCol,
    loadColumnProfile,
    savedQueries,
    loadingSaved,
    queryHistory,
    resultsFilter,
    setResultsFilter,
    resultsSortCol,
    setResultsSortCol,
    resultsSortAsc,
    setResultsSortAsc,
    resultsPage,
    setResultsPage,
    resultsPerPage,
    setResultsPerPage,
    loadSchema,
    executeSql,
    runExplainSql,
    insertSnippetAtCursor,
    handleSaveGoldenKpi,
    handleUpdateGoldenKpi,
    handleDeleteGoldenKpi,
    downloadSqlCsv,
    exportFullSqlResult,
    resolvedDatasetName,
    activeTableMeta,
    resolvedRowCount,
    schemaColumnsList,
    contextualRecipes,
    numericResultCols,
    filteredAndSortedRecords,
  };
}
