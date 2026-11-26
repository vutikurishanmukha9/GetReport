import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BarChart3,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Cpu,
  Download,
  Eye,
  FileJson,
  Loader2,
  Play,
  Search,
  Table2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, SaveKpiButton } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { DataRow } from "@/types/api";
import {
  DEFAULT_STARTER_SQL,
  PAGE_SIZE_OPTIONS,
  type QueryRowLimit,
  type ResultViewMode,
  type SqlExplainState,
  type SqlResultState,
} from "./types";
import { formatResultAsMarkdown, getSqlErrorHint, isFiniteNumber } from "./utils";
import { SqlResultsChart } from "./SqlResultsChart";
import { SqlExplainTree } from "./SqlExplainTree";
import { SqlRowInspectorSheet } from "./SqlRowInspectorSheet";

interface SqlResultsPanelProps {
  selectedTaskId: string;
  sqlQuery: string;
  sqlLimit: QueryRowLimit;
  sqlRunning: boolean;
  onCancelQuery: () => void;
  sqlError: string | null;
  sqlResult: SqlResultState | null;
  sqlExplain: SqlExplainState | null;
  resultViewMode: ResultViewMode;
  onResultViewModeChange: (mode: ResultViewMode) => void;
  resolvedDatasetName: string;
  filteredAndSortedRecords: DataRow[];
  numericResultCols: Set<string>;
  resultsFilter: string;
  onResultsFilterChange: (value: string) => void;
  resultsSortCol: string | null;
  resultsSortAsc: boolean;
  onSortCol: (col: string) => void;
  resultsPage: number;
  onPageChange: (updater: number | ((prev: number) => number)) => void;
  resultsPerPage: 15 | 25 | 50;
  onPerPageChange: (size: 15 | 25 | 50) => void;
  isSavedSql: boolean;
  showSavePrompt: boolean;
  setShowSavePrompt: (show: boolean) => void;
  kpiTitleInput: string;
  setKpiTitleInput: (val: string) => void;
  savingKpi: boolean;
  onSaveGoldenKpi: () => void;
  onDownloadCsv: () => void;
  onExportFull: (fmt: "csv" | "json") => void;
  onRunSql: (sql?: string) => void;
  onSetSqlQuery: (sql: string) => void;
  onCopyText: (label: string, description: string, text: string) => void;
}

export function SqlResultsPanel({
  selectedTaskId,
  sqlQuery,
  sqlLimit,
  sqlRunning,
  onCancelQuery,
  sqlError,
  sqlResult,
  sqlExplain,
  resultViewMode,
  onResultViewModeChange,
  resolvedDatasetName,
  filteredAndSortedRecords,
  numericResultCols,
  resultsFilter,
  onResultsFilterChange,
  resultsSortCol,
  resultsSortAsc,
  onSortCol,
  resultsPage,
  onPageChange,
  resultsPerPage,
  onPerPageChange,
  isSavedSql,
  showSavePrompt,
  setShowSavePrompt,
  kpiTitleInput,
  setKpiTitleInput,
  savingKpi,
  onSaveGoldenKpi,
  onDownloadCsv,
  onExportFull,
  onRunSql,
  onSetSqlQuery,
  onCopyText,
}: SqlResultsPanelProps) {
  const [hasCopiedJson, setHasCopiedJson] = useState<boolean>(false);
  const [inspectedRow, setInspectedRow] = useState<{ index: number; row: DataRow } | null>(null);

  const totalResultsPages = Math.max(1, Math.ceil(filteredAndSortedRecords.length / resultsPerPage));
  const paginatedRecords = useMemo(() => {
    const start = (resultsPage - 1) * resultsPerPage;
    return filteredAndSortedRecords.slice(start, start + resultsPerPage);
  }, [filteredAndSortedRecords, resultsPage, resultsPerPage]);

  const numericColumnBounds = useMemo(() => {
    const bounds = new Map<string, { min: number; max: number }>();
    if (!sqlResult) return bounds;
    for (const col of numericResultCols) {
      let min = Infinity;
      let max = -Infinity;
      for (const row of filteredAndSortedRecords) {
        const v = row[col];
        if (isFiniteNumber(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
        bounds.set(col, { min, max });
      }
    }
    return bounds;
  }, [sqlResult, numericResultCols, filteredAndSortedRecords]);

  const formattedJsonPreview = useMemo(() => {
    if (!sqlResult) return "[]";
    return JSON.stringify(filteredAndSortedRecords.slice(0, 100), null, 2);
  }, [sqlResult, filteredAndSortedRecords]);

  if (sqlRunning) {
    return (
      <div
        className="rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#0D0F16] p-4 space-y-3"
        aria-label="Executing DuckDB query"
      >
        <div className="flex items-center justify-between gap-2 text-xs font-mono text-stone-600 dark:text-zinc-300">
          <span className="inline-flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-[#800000] dark:text-emerald-400" />
            Executing vectorized query plan in DuckDB…
          </span>
          <div className="flex items-center gap-2">
            <span>Limit {sqlLimit} rows</span>
            <Button
              type="button"
              size="xs"
              variant="outline"
              onClick={onCancelQuery}
              className="h-6 px-2 text-[11px] rounded-[5px] border-red-300 text-red-700 hover:bg-red-50 dark:border-red-500/30 dark:text-red-400 dark:hover:bg-red-500/10"
            >
              Cancel
            </Button>
          </div>
        </div>
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((skeletonRow) => (
            <div key={skeletonRow} className="h-8 rounded-lg bg-stone-200/70 dark:bg-white/[0.05] animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (sqlError) {
    return (
      <div className="rounded-xl border border-destructive/35 bg-destructive/10 p-4 text-xs flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
        <div className="space-y-1.5 flex-1">
          <div className="font-sans font-bold text-sm text-destructive">Query Evaluation Failed</div>
          <p className="font-mono text-destructive/95 leading-relaxed">{sqlError}</p>
          <p className="font-sans text-[11px] text-stone-700 dark:text-zinc-300">{getSqlErrorHint(sqlError)}</p>
          {selectedTaskId && (
            <div className="pt-1">
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => {
                  const q = "DESCRIBE dataset;";
                  onSetSqlQuery(q);
                  onRunSql(q);
                }}
                className="h-7 text-xs font-mono border-destructive/30 hover:bg-destructive/15 cursor-pointer"
              >
                Run DESCRIBE dataset
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (!sqlResult && resultViewMode !== "plan") {
    return (
      <div className="rounded-xl border border-dashed border-stone-300 dark:border-white/15 bg-[#F6F2EA]/70 dark:bg-[#10131C]/70 p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h4 className="text-sm font-display font-bold text-stone-900 dark:text-white">
            Ready to Query In-Memory Arrow View
          </h4>
          <p className="text-xs text-stone-600 dark:text-zinc-400 max-w-xl leading-relaxed">
            Use the Schema Inspector or <kbd className="font-mono text-[10px]">Ctrl+K</kbd> command palette to insert exact
            column identifiers, pick an analytical recipe above, or run the starter query against{" "}
            <code className="font-mono font-semibold">dataset</code>.
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          disabled={!selectedTaskId}
          onClick={() => onRunSql(sqlQuery || DEFAULT_STARTER_SQL)}
          className="h-9 px-4 rounded-xl bg-[#800000] hover:bg-[#6b0000] text-white dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-950 font-semibold text-xs gap-1.5 shrink-0 cursor-pointer"
        >
          <Play className="h-3.5 w-3.5 fill-current" />
          <span>Run Starter Query</span>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-2.5 text-xs">
        <div className="flex items-center gap-2 flex-wrap font-mono">
          {sqlResult && (
            <>
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
            </>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className="inline-flex items-center p-0.5 rounded-lg bg-stone-200/80 dark:bg-white/[0.06] border border-stone-300/80 dark:border-white/10">
            <button
              type="button"
              onClick={() => onResultViewModeChange("table")}
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
              onClick={() => onResultViewModeChange("chart")}
              className={cn(
                "px-2 py-1 rounded-md text-[11px] font-sans font-medium flex items-center gap-1 cursor-pointer transition-colors",
                resultViewMode === "chart"
                  ? "bg-white text-stone-900 shadow-2xs dark:bg-zinc-800 dark:text-white"
                  : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white",
              )}
            >
              <BarChart3 className="h-3 w-3" />
              <span>Chart</span>
            </button>
            <button
              type="button"
              onClick={() => onResultViewModeChange("json")}
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
            {sqlExplain && (
              <button
                type="button"
                onClick={() => onResultViewModeChange("plan")}
                className={cn(
                  "px-2 py-1 rounded-md text-[11px] font-sans font-medium flex items-center gap-1 cursor-pointer transition-colors",
                  resultViewMode === "plan"
                    ? "bg-white text-stone-900 shadow-2xs dark:bg-zinc-800 dark:text-white"
                    : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white",
                )}
              >
                <Cpu className="h-3 w-3" />
                <span>Plan</span>
              </button>
            )}
          </div>

          {resultViewMode === "table" && sqlResult && (
            <div className="relative">
              <Search className="h-3 w-3 absolute left-2.5 top-2 text-stone-400 dark:text-zinc-500" />
              <Input
                type="text"
                placeholder="Filter result rows..."
                aria-label="Filter result rows"
                value={resultsFilter}
                onChange={(e) => {
                  onResultsFilterChange(e.target.value);
                  onPageChange(1);
                }}
                className="h-7 w-40 pl-7 text-[11px] font-mono bg-white dark:bg-[#131620] border-stone-300 dark:border-white/10"
              />
            </div>
          )}

          {sqlResult && !showSavePrompt && (
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
              className="h-7 px-2.5 text-xs rounded-[6px]"
            />
          )}

          {sqlResult && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs font-medium rounded-[6px] border-stone-300 bg-white hover:bg-stone-100 text-stone-800 dark:border-white/10 dark:bg-white/[0.05] dark:hover:bg-white/10 dark:text-zinc-200 px-2.5 cursor-pointer gap-1.5"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Export</span>
                  <ChevronDown className="h-3 w-3 opacity-70" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[12.5rem] text-xs font-sans">
                <DropdownMenuItem onClick={onDownloadCsv} className="cursor-pointer gap-2">
                  <Download className="h-3.5 w-3.5" />
                  <span>Download Preview CSV</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onExportFull("csv")} className="cursor-pointer gap-2">
                  <Download className="h-3.5 w-3.5 text-[#800000] dark:text-emerald-400" />
                  <span>Stream Full CSV (up to 50k)</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onExportFull("json")} className="cursor-pointer gap-2">
                  <FileJson className="h-3.5 w-3.5" />
                  <span>Stream Full JSON (up to 50k)</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() =>
                    onCopyText(
                      "Markdown Copied",
                      "Result table copied as Markdown.",
                      formatResultAsMarkdown(sqlResult.columns, filteredAndSortedRecords),
                    )
                  }
                  className="cursor-pointer gap-2"
                >
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy Markdown Table</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      {showSavePrompt && (
        <div className="p-2.5 rounded-xl bg-[#EDE8DE]/80 dark:bg-white/[0.04] border border-stone-300 dark:border-white/10 flex flex-wrap items-center gap-2">
          <label htmlFor="sql-golden-kpi-title" className="text-xs font-sans font-semibold text-stone-800 dark:text-zinc-200">
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
            onClick={onSaveGoldenKpi}
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

      {resultViewMode === "plan" ? (
        <SqlExplainTree explainState={sqlExplain} />
      ) : resultViewMode === "chart" && sqlResult ? (
        <SqlResultsChart
          columns={sqlResult.columns}
          records={filteredAndSortedRecords}
          numericCols={numericResultCols}
        />
      ) : resultViewMode === "json" && sqlResult ? (
        <div className="rounded-xl border border-stone-300/80 dark:border-white/10 bg-white dark:bg-[#0B0D13] overflow-hidden">
          <div className="px-3.5 py-2 border-b border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#12151F] flex items-center justify-between text-xs font-mono text-stone-600 dark:text-zinc-400">
            <span>
              JSON Preview ({Math.min(100, filteredAndSortedRecords.length)} of {filteredAndSortedRecords.length} rows)
            </span>
            <button
              type="button"
              onClick={() => {
                onCopyText("JSON Copied", "Result rows copied as JSON.", formattedJsonPreview);
                setHasCopiedJson(true);
                window.setTimeout(() => setHasCopiedJson(false), 1800);
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
      ) : sqlResult ? (
        <div className="border border-stone-300/90 dark:border-white/10 rounded-xl overflow-hidden bg-white dark:bg-[#0D0F16]">
          <div className="w-full overflow-auto max-h-[420px]">
            <Table className="border-collapse">
              <TableHeader className="bg-[#F3EFE6] dark:bg-[#141721] sticky top-0 z-10">
                <TableRow className="border-b border-stone-300/80 dark:border-white/10">
                  <TableHead className="w-12 sticky left-0 z-20 bg-[#F3EFE6] dark:bg-[#141721] text-center font-mono text-[10px] font-bold text-stone-500 dark:text-zinc-400 px-2 py-2 border-r border-stone-200 dark:border-white/10 select-none">
                    #
                  </TableHead>
                  {sqlResult.columns.map((col, idx) => {
                    const isSorted = resultsSortCol === col;
                    const isNumeric = numericResultCols.has(col);
                    const colType = sqlResult.column_types?.[idx];
                    return (
                      <TableHead
                        key={col}
                        className={cn(
                          "font-mono text-[11px] font-bold text-stone-900 dark:text-zinc-100 px-3 py-2 whitespace-nowrap border-r border-stone-200 dark:border-white/10 last:border-r-0 hover:bg-stone-200/70 dark:hover:bg-white/[0.06] transition-colors select-none",
                          isNumeric && "text-right",
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => onSortCol(col)}
                          title={`Sort by ${col}`}
                          className={cn(
                            "w-full flex items-center gap-1.5 cursor-pointer bg-transparent border-0 p-0 text-inherit font-inherit",
                            isNumeric ? "justify-end" : "justify-between",
                          )}
                        >
                          <span className="flex flex-col items-start">
                            <span>{col}</span>
                            {colType && (
                              <span className="text-[9px] font-normal text-stone-400 dark:text-zinc-500 leading-none">
                                {colType}
                              </span>
                            )}
                          </span>
                          {isSorted ? (
                            resultsSortAsc ? (
                              <ArrowUp className="h-3 w-3 shrink-0 text-[#800000] dark:text-emerald-400" />
                            ) : (
                              <ArrowDown className="h-3 w-3 shrink-0 text-[#800000] dark:text-emerald-400" />
                            )
                          ) : (
                            <ArrowUpDown className="h-3 w-3 shrink-0 text-stone-400 dark:text-zinc-600" />
                          )}
                        </button>
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
                      <TableRow key={`sql-row-${rowNumber}`} className="group hover:bg-[#FAF8F5] dark:hover:bg-white/[0.03]">
                        <TableCell className="sticky left-0 z-10 bg-white group-hover:bg-[#FAF8F5] dark:bg-[#0D0F16] dark:group-hover:bg-[#131620] text-center font-mono text-[10px] text-stone-400 dark:text-zinc-500 tabular-nums px-2 py-1.5 border-r border-stone-200/80 dark:border-white/[0.06] select-none">
                          <button
                            type="button"
                            onClick={() => setInspectedRow({ index: rowNumber, row })}
                            title={`Inspect row #${rowNumber}`}
                            className="inline-flex items-center justify-center gap-1 w-full hover:text-[#800000] dark:hover:text-emerald-400 cursor-pointer"
                          >
                            <span>{rowNumber}</span>
                            <Eye className="h-2.5 w-2.5 opacity-0 group-hover:opacity-80" />
                          </button>
                        </TableCell>
                        {sqlResult.columns.map((col) => {
                          const cellVal = row[col];
                          const isNumeric = numericResultCols.has(col);
                          const colBound = isNumeric ? numericColumnBounds.get(col) : undefined;
                          const barPct =
                            colBound && isFiniteNumber(cellVal)
                              ? Math.round(((cellVal - colBound.min) / (colBound.max - colBound.min)) * 100)
                              : null;

                          return (
                            <TableCell
                              key={`sql-cell-${rowNumber}-${col}`}
                              onClick={() => {
                                if (cellVal !== null && cellVal !== undefined) {
                                  onCopyText("Cell Copied", `${col}: ${String(cellVal).slice(0, 48)}`, String(cellVal));
                                }
                              }}
                              title="Click cell value to copy"
                              className={cn(
                                "relative font-mono text-xs text-stone-900 dark:text-zinc-200 whitespace-nowrap px-3 py-1.5 border-r border-stone-200/80 dark:border-white/[0.06] last:border-r-0 cursor-copy max-w-[320px] truncate",
                                isNumeric && "text-right tabular-nums",
                              )}
                            >
                              {barPct !== null && barPct > 0 && (
                                <span
                                  aria-hidden="true"
                                  className="pointer-events-none absolute inset-y-1 right-1 rounded-sm bg-[#800000]/[0.06] dark:bg-emerald-400/[0.08]"
                                  style={{ width: `${Math.min(92, Math.max(4, barPct))}%` }}
                                />
                              )}
                              <span className="relative z-10">
                                {cellVal === null || cellVal === undefined ? (
                                  <span className="px-1.5 py-0.5 rounded bg-stone-100 dark:bg-white/[0.06] text-stone-400 dark:text-zinc-500 italic font-sans text-[10px]">
                                    null
                                  </span>
                                ) : isFiniteNumber(cellVal) ? (
                                  cellVal.toLocaleString(undefined, { maximumFractionDigits: 4 })
                                ) : (
                                  String(cellVal)
                                )}
                              </span>
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

          {filteredAndSortedRecords.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 border-t border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#11131B] text-[11px] font-mono text-stone-600 dark:text-zinc-400 tabular-nums">
              <div className="flex items-center gap-3">
                <span>
                  Showing {((resultsPage - 1) * resultsPerPage + 1).toLocaleString()}–
                  {Math.min(resultsPage * resultsPerPage, filteredAndSortedRecords.length).toLocaleString()} of{" "}
                  {filteredAndSortedRecords.length.toLocaleString()}
                </span>
                <div className="flex items-center gap-1">
                  <span>Rows/page:</span>
                  {PAGE_SIZE_OPTIONS.map((sizeOpt) => (
                    <button
                      key={sizeOpt}
                      type="button"
                      onClick={() => {
                        onPerPageChange(sizeOpt);
                        onPageChange(1);
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
                    onClick={() => onPageChange((prev) => Math.max(1, prev - 1))}
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
                    onClick={() => onPageChange((prev) => Math.min(totalResultsPages, prev + 1))}
                  >
                    Next
                    <ChevronRight className="h-3 w-3 ml-0.5" />
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      ) : null}

      <SqlRowInspectorSheet
        open={inspectedRow !== null}
        onOpenChange={(open) => {
          if (!open) setInspectedRow(null);
        }}
        rowIndex={inspectedRow?.index ?? null}
        rowData={inspectedRow?.row ?? null}
        columns={sqlResult?.columns ?? []}
        columnTypes={sqlResult?.column_types}
        onCopyText={onCopyText}
      />
    </div>
  );
}
