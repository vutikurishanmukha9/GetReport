import React, { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { cn } from "@/lib/utils";
import { useIsMobile } from "@/hooks/use-mobile";
import type { DuckDBSqlStudioProps } from "./sql-studio/types";
import { formatSqlStatement } from "./sql-studio/utils";
import { useDuckDBStudio } from "./sql-studio/useDuckDBStudio";
import { SqlStudioHeader } from "./sql-studio/SqlStudioHeader";
import { SqlInspectorSidebar } from "./sql-studio/SqlInspectorSidebar";
import { SqlWorkspacePane } from "./sql-studio/SqlWorkspacePane";
import { SqlCommandPalette } from "./sql-studio/SqlCommandPalette";

export type { DuckDBSqlStudioProps } from "./sql-studio/types";

export const DuckDBSqlStudio: React.FC<DuckDBSqlStudioProps> = (props) => {
  const { allowDatasetSwitching = false } = props;
  const studio = useDuckDBStudio(props);
  const isMobile = useIsMobile();

  const [showInspector, setShowInspector] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [paletteOpen, setPaletteOpen] = useState<boolean>(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  const lineCount = useMemo(() => Math.max(1, studio.sqlQuery.split("\n").length), [studio.sqlQuery]);
  const schemaColumnNames = useMemo(
    () => studio.schemaColumnsList.map((col) => col.name),
    [studio.schemaColumnsList],
  );

  const copyWithToast = (title: string, description: string, text: string) => {
    void navigator.clipboard?.writeText(text);
    studio.toast({ title, description });
  };

  const handleSortCol = (col: string) => {
    if (studio.resultsSortCol === col) {
      studio.setResultsSortAsc(!studio.resultsSortAsc);
    } else {
      studio.setResultsSortCol(col);
      studio.setResultsSortAsc(true);
    }
  };

  const inspectorNode = showInspector ? (
    <SqlInspectorSidebar
      selectedTaskId={studio.selectedTaskId}
      activeTableMeta={studio.activeTableMeta}
      schemaColumnsList={studio.schemaColumnsList}
      resolvedRowCount={studio.resolvedRowCount}
      loadingSchema={studio.loadingSchema}
      schemaError={studio.schemaError}
      columnProfiles={studio.columnProfiles}
      loadingProfileCol={studio.loadingProfileCol}
      onLoadColumnProfile={(col) => void studio.loadColumnProfile(col)}
      savedQueries={studio.savedQueries}
      loadingSaved={studio.loadingSaved}
      queryHistory={studio.queryHistory}
      onRefreshSchema={(tid) => studio.loadSchema(tid)}
      onInsertSnippet={studio.insertSnippetAtCursor}
      onLoadSql={(sql, markSaved) => {
        studio.setSqlQuery(sql);
        studio.setIsSavedSql(markSaved);
      }}
      onRunSql={(sql, markSaved) => {
        studio.setSqlQuery(sql);
        studio.setIsSavedSql(markSaved);
        void studio.executeSql(sql);
      }}
      onRenameSaved={(id, q) => void studio.handleUpdateGoldenKpi(id, q)}
      onDeleteSaved={(id) => void studio.handleDeleteGoldenKpi(id)}
    />
  ) : null;

  const workspaceNode = (
    <SqlWorkspacePane
      selectedTaskId={studio.selectedTaskId}
      tableName={studio.activeTableMeta?.table_name || "dataset"}
      schemaColumnNames={schemaColumnNames}
      resolvedDatasetName={studio.resolvedDatasetName}
      contextualRecipes={studio.contextualRecipes}
      worksheets={studio.worksheets}
      activeWorksheetId={studio.activeWorksheetId}
      onSelectWorksheet={studio.setActiveWorksheetId}
      onAddWorksheet={studio.addWorksheet}
      onCloseWorksheet={studio.closeWorksheet}
      sqlQuery={studio.sqlQuery}
      onSetSqlQuery={studio.setSqlQuery}
      selectedSql={studio.selectedSql}
      cursorLine={studio.cursorLine}
      cursorCol={studio.cursorCol}
      lineCount={lineCount}
      editorViewRef={studio.editorViewRef}
      onSelectionChange={studio.onSelectionChange}
      detectedParams={studio.detectedParams}
      paramValues={studio.paramValues}
      onParamChange={studio.setParamValue}
      sqlLimit={studio.sqlLimit}
      onSqlLimitChange={studio.setSqlLimit}
      sqlRunning={studio.sqlRunning}
      sqlError={studio.sqlError}
      sqlResult={studio.sqlResult}
      sqlExplain={studio.sqlExplain}
      resultViewMode={studio.resultViewMode}
      onResultViewModeChange={studio.setResultViewMode}
      filteredAndSortedRecords={studio.filteredAndSortedRecords}
      numericResultCols={studio.numericResultCols}
      resultsFilter={studio.resultsFilter}
      onResultsFilterChange={studio.setResultsFilter}
      resultsSortCol={studio.resultsSortCol}
      resultsSortAsc={studio.resultsSortAsc}
      onSortCol={handleSortCol}
      resultsPage={studio.resultsPage}
      onPageChange={studio.setResultsPage}
      resultsPerPage={studio.resultsPerPage}
      onPerPageChange={studio.setResultsPerPage}
      isSavedSql={studio.isSavedSql}
      showSavePrompt={studio.showSavePrompt}
      setShowSavePrompt={studio.setShowSavePrompt}
      kpiTitleInput={studio.kpiTitleInput}
      setKpiTitleInput={studio.setKpiTitleInput}
      savingKpi={studio.savingKpi}
      onSaveGoldenKpi={() => void studio.handleSaveGoldenKpi()}
      onDownloadCsv={studio.downloadSqlCsv}
      onExportFull={(fmt) => void studio.exportFullSqlResult(fmt)}
      onRunSql={(sql) => void studio.executeSql(sql)}
      onCancelQuery={studio.cancelQuery}
      onExplain={() => void studio.runExplainSql()}
      onClearAll={studio.clearAll}
      onCopyText={copyWithToast}
      onOpenCommandPalette={() => setPaletteOpen(true)}
    />
  );

  return (
    <Card
      className={cn(
        "border border-stone-300/90 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#0B0C10] text-stone-900 dark:text-zinc-100 shadow-sm overflow-hidden transition-all duration-200",
        isFullscreen ? "fixed inset-3 sm:inset-5 z-50 rounded-2xl flex flex-col shadow-2xl overflow-y-auto" : "rounded-2xl",
      )}
    >
      <SqlStudioHeader
        selectedTaskId={studio.selectedTaskId}
        activeTableName={studio.activeTableMeta?.table_name || "dataset"}
        resolvedDatasetName={studio.resolvedDatasetName}
        resolvedRowCount={studio.resolvedRowCount}
        columnCount={studio.schemaColumnsList.length}
        allowDatasetSwitching={allowDatasetSwitching}
        availableDatasets={studio.availableDatasets}
        loadingDatasets={studio.loadingDatasets}
        onSelectDataset={studio.selectDataset}
        showInspector={showInspector}
        onToggleInspector={() => setShowInspector((prev) => !prev)}
        isFullscreen={isFullscreen}
        onToggleFullscreen={() => setIsFullscreen((prev) => !prev)}
      />

      {isMobile ? (
        <div className="flex flex-col flex-1 min-h-0">
          {inspectorNode}
          {workspaceNode}
        </div>
      ) : (
        <ResizablePanelGroup direction="horizontal" className="flex-1 min-h-0">
          {showInspector && (
            <>
              <ResizablePanel defaultSize={27} minSize={18} maxSize={42}>
                {inspectorNode}
              </ResizablePanel>
              <ResizableHandle withHandle className="bg-stone-300/70 dark:bg-white/10" />
            </>
          )}
          <ResizablePanel defaultSize={showInspector ? 73 : 100} minSize={45}>
            {workspaceNode}
          </ResizablePanel>
        </ResizablePanelGroup>
      )}

      <SqlCommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        columns={studio.schemaColumnsList}
        recipes={studio.contextualRecipes}
        savedQueries={studio.savedQueries}
        history={studio.queryHistory}
        onInsertSnippet={studio.insertSnippetAtCursor}
        onLoadAndRunSql={(sql, markSaved) => {
          studio.setSqlQuery(sql);
          studio.setIsSavedSql(markSaved);
          void studio.executeSql(sql);
        }}
        onRunAction={(action) => {
          if (action === "run") void studio.executeSql();
          else if (action === "format") studio.setSqlQuery(formatSqlStatement(studio.sqlQuery));
          else if (action === "explain") void studio.runExplainSql();
          else if (action === "export") studio.downloadSqlCsv();
          else if (action === "clear") studio.clearAll();
        }}
      />
    </Card>
  );
};

export default DuckDBSqlStudio;
