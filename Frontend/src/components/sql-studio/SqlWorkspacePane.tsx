import React from "react";
import type { EditorView } from "@codemirror/view";
import { Code2 } from "lucide-react";
import type { DataRow } from "@/types/api";
import type {
  QueryRecipe,
  QueryRowLimit,
  ResultViewMode,
  SqlExplainState,
  SqlResultState,
  SqlWorksheetTab,
} from "./types";
import { formatSqlStatement } from "./utils";
import { SqlEditorToolbar } from "./SqlEditorToolbar";
import { SqlCodeEditor } from "./SqlCodeEditor";
import { SqlResultsPanel } from "./SqlResultsPanel";

interface SqlWorkspacePaneProps {
  selectedTaskId: string;
  tableName: string;
  schemaColumnNames: string[];
  resolvedDatasetName: string;
  contextualRecipes: QueryRecipe[];
  worksheets: SqlWorksheetTab[];
  activeWorksheetId: string;
  onSelectWorksheet: (id: string) => void;
  onAddWorksheet: () => void;
  onCloseWorksheet: (id: string) => void;
  sqlQuery: string;
  onSetSqlQuery: (sql: string) => void;
  selectedSql: string;
  cursorLine: number;
  cursorCol: number;
  lineCount: number;
  editorViewRef: React.MutableRefObject<EditorView | null>;
  onSelectionChange: (selection: string, line: number, col: number) => void;
  detectedParams: string[];
  paramValues: Record<string, string>;
  onParamChange: (key: string, val: string) => void;
  sqlLimit: QueryRowLimit;
  onSqlLimitChange: (limit: QueryRowLimit) => void;
  sqlRunning: boolean;
  sqlError: string | null;
  sqlResult: SqlResultState | null;
  sqlExplain: SqlExplainState | null;
  resultViewMode: ResultViewMode;
  onResultViewModeChange: (mode: ResultViewMode) => void;
  filteredAndSortedRecords: DataRow[];
  numericResultCols: Set<string>;
  resultsFilter: string;
  onResultsFilterChange: (val: string) => void;
  resultsSortCol: string | null;
  resultsSortAsc: boolean;
  onSortCol: (col: string) => void;
  resultsPage: number;
  onPageChange: (page: number) => void;
  resultsPerPage: 15 | 25 | 50;
  onPerPageChange: (perPage: 15 | 25 | 50) => void;
  isSavedSql: boolean;
  showSavePrompt: boolean;
  setShowSavePrompt: (show: boolean) => void;
  kpiTitleInput: string;
  setKpiTitleInput: (val: string) => void;
  savingKpi: boolean;
  onSaveGoldenKpi: () => void;
  onDownloadCsv: () => void;
  onExportFull: (format: "csv" | "json") => void;
  onRunSql: (overrideSql?: string) => void;
  onCancelQuery: () => void;
  onExplain: () => void;
  onClearAll: () => void;
  onCopyText: (title: string, description: string, text: string) => void;
  onOpenCommandPalette: () => void;
}

export const SqlWorkspacePane: React.FC<SqlWorkspacePaneProps> = ({
  selectedTaskId,
  tableName,
  schemaColumnNames,
  resolvedDatasetName,
  contextualRecipes,
  worksheets,
  activeWorksheetId,
  onSelectWorksheet,
  onAddWorksheet,
  onCloseWorksheet,
  sqlQuery,
  onSetSqlQuery,
  selectedSql,
  cursorLine,
  cursorCol,
  lineCount,
  editorViewRef,
  onSelectionChange,
  detectedParams,
  paramValues,
  onParamChange,
  sqlLimit,
  onSqlLimitChange,
  sqlRunning,
  sqlError,
  sqlResult,
  sqlExplain,
  resultViewMode,
  onResultViewModeChange,
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
  onCancelQuery,
  onExplain,
  onClearAll,
  onCopyText,
  onOpenCommandPalette,
}) => {
  return (
    <div className="p-4 sm:p-5 flex flex-col space-y-4 min-w-0">
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
              onSetSqlQuery(recipe.sql);
              onRunSql(recipe.sql);
            }}
            className="px-2.5 py-1 rounded-[6px] text-xs font-sans font-medium bg-[#EDE8DE]/80 hover:bg-[#E5DEC9] text-stone-800 border border-stone-300/80 dark:bg-white/[0.05] dark:hover:bg-white/[0.1] dark:text-zinc-200 dark:border-white/10 transition-colors cursor-pointer shrink-0 disabled:opacity-40"
          >
            {recipe.label}
          </button>
        ))}
      </div>

      <div className="rounded-xl border border-stone-300 dark:border-white/15 bg-white dark:bg-[#0D0F16] shadow-xs overflow-hidden">
        <SqlEditorToolbar
          worksheets={worksheets}
          activeWorksheetId={activeWorksheetId}
          onSelectWorksheet={onSelectWorksheet}
          onAddWorksheet={onAddWorksheet}
          onCloseWorksheet={onCloseWorksheet}
          lineCount={lineCount}
          charCount={sqlQuery.length}
          sqlLimit={sqlLimit}
          onSqlLimitChange={onSqlLimitChange}
          sqlRunning={sqlRunning}
          canRun={Boolean(selectedTaskId)}
          hasSelection={Boolean(selectedSql.trim())}
          onFormat={() => onSetSqlQuery(formatSqlStatement(sqlQuery))}
          onExplain={onExplain}
          onCopy={() => onCopyText("Copied", "SQL query copied to clipboard.", sqlQuery)}
          onClear={onClearAll}
          onRun={() => onRunSql()}
          onCancel={onCancelQuery}
          onOpenCommandPalette={onOpenCommandPalette}
        />

        <SqlCodeEditor
          value={sqlQuery}
          onChange={onSetSqlQuery}
          tableName={tableName}
          schemaColumnNames={schemaColumnNames}
          editorViewRef={editorViewRef}
          onSelectionChange={onSelectionChange}
          onExecute={() => onRunSql()}
          detectedParams={detectedParams}
          paramValues={paramValues}
          onParamChange={onParamChange}
        />

        <div className="px-3.5 py-1.5 bg-[#FAF8F5] dark:bg-[#10131C] border-t border-stone-200 dark:border-white/10 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-stone-500 dark:text-zinc-400">
          <span>Read-only SQL: SELECT · WITH · DESCRIBE · EXPLAIN · SHOW</span>
          <span className="tabular-nums">
            Ln {cursorLine}, Col {cursorCol} · Ctrl+Enter run · Ctrl+K commands
          </span>
        </div>
      </div>

      <SqlResultsPanel
        selectedTaskId={selectedTaskId}
        sqlQuery={sqlQuery}
        sqlLimit={sqlLimit}
        sqlRunning={sqlRunning}
        onCancelQuery={onCancelQuery}
        sqlError={sqlError}
        sqlResult={sqlResult}
        sqlExplain={sqlExplain}
        resultViewMode={resultViewMode}
        onResultViewModeChange={onResultViewModeChange}
        resolvedDatasetName={resolvedDatasetName}
        filteredAndSortedRecords={filteredAndSortedRecords}
        numericResultCols={numericResultCols}
        resultsFilter={resultsFilter}
        onResultsFilterChange={onResultsFilterChange}
        resultsSortCol={resultsSortCol}
        resultsSortAsc={resultsSortAsc}
        onSortCol={onSortCol}
        resultsPage={resultsPage}
        onPageChange={onPageChange}
        resultsPerPage={resultsPerPage}
        onPerPageChange={onPerPageChange}
        isSavedSql={isSavedSql}
        showSavePrompt={showSavePrompt}
        setShowSavePrompt={setShowSavePrompt}
        kpiTitleInput={kpiTitleInput}
        setKpiTitleInput={setKpiTitleInput}
        savingKpi={savingKpi}
        onSaveGoldenKpi={onSaveGoldenKpi}
        onDownloadCsv={onDownloadCsv}
        onExportFull={onExportFull}
        onRunSql={onRunSql}
        onSetSqlQuery={onSetSqlQuery}
        onCopyText={onCopyText}
      />
    </div>
  );
};
