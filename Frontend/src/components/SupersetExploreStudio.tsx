import React, { useMemo } from "react";
import {
  AlignLeft,
  BarChart3,
  Check,
  Copy,
  Database,
  Download,
  FileSpreadsheet,
  Loader2,
  Maximize2,
  Minimize2,
  RefreshCw,
  Sliders,
  Table2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExploreTelemetryDeck } from "./ExploreTelemetryDeck";
import type { SupersetExploreStudioProps } from "./bi-studio/constants";
import { useSupersetExplore } from "./bi-studio/useSupersetExplore";
import { NativeEChartsView } from "./bi-studio/NativeEChartsView";
import { ExploreControlsSidebar } from "./bi-studio/ExploreControlsSidebar";
import { ExploreDataTable } from "./bi-studio/ExploreDataTable";

function ExploreObservationsStrip({ insights }: { insights: any[] }) {
  return (
    <div className="rounded-xl border border-stone-200/80 bg-white/80 dark:border-white/10 dark:bg-[#111218] p-3.5 sm:p-4 space-y-2.5 shadow-2xs">
      <div className="flex items-center justify-between pb-1 border-b border-stone-100 dark:border-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-stone-400 dark:bg-zinc-500" />
          <h4 className="text-[11px] font-mono uppercase tracking-[0.08em] text-stone-600 dark:text-zinc-400 font-medium">
            Key Observations &amp; Patterns
          </h4>
        </div>
        <span className="text-[10px] font-mono text-stone-400 dark:text-zinc-500">
          {insights.length} observations
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {insights.map((ins: any) => (
          <div
            key={ins.id}
            className="group relative flex flex-col justify-between rounded-lg border border-stone-200/70 bg-stone-50/50 p-3 transition-colors hover:border-stone-300 dark:border-white/[0.05] dark:bg-white/[0.02] dark:hover:border-white/10"
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="text-[10px] font-mono font-medium uppercase tracking-wider text-stone-500 dark:text-zinc-400">
                {ins.badge || "Observation"}
              </span>
              <span className="text-[10px] font-mono text-stone-400 dark:text-zinc-600">
                {ins.type || "pattern"}
              </span>
            </div>
            <div>
              <h5 className="text-xs font-sans font-semibold text-stone-900 dark:text-white tracking-tight">
                {ins.title}
              </h5>
              <p className="text-[11.5px] leading-relaxed text-stone-600 dark:text-zinc-300 mt-1 font-sans">
                {ins.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export const SupersetExploreStudio: React.FC<SupersetExploreStudioProps> = (props) => {
  const { datasetName: initialDatasetName, allowDatasetSwitching = true } = props;
  const ex = useSupersetExplore(props);

  const totalMetricSum = useMemo(() => {
    const records = ex.queryResult?.records;
    if (!Array.isArray(records)) return 0;
    return records.reduce(
      (acc: number, r: any) =>
        acc + (r?.value !== null && r?.value !== undefined && r.value.constructor === Number && Number.isFinite(r.value) ? Math.abs(Number(r.value)) : 0),
      0,
    );
  }, [ex.queryResult?.records]);

  const totalRowCount = useMemo(() => {
    const records = ex.queryResult?.records;
    if (!Array.isArray(records)) return 0;
    return records.reduce(
      (acc: number, r: any) =>
        acc + (r?.count !== null && r?.count !== undefined && r.count.constructor === Number && Number.isFinite(r.count) ? Number(r.count) : 0),
      0,
    );
  }, [ex.queryResult?.records]);

  const filteredAndSortedRecords = useMemo(() => {
    const records = ex.queryResult?.records;
    if (!Array.isArray(records)) return [];
    let list = [...records];

    if (ex.tableSearch.trim()) {
      const q = ex.tableSearch.toLowerCase().trim();
      list = list.filter((r) =>
        Object.values(r).some((val) => val !== null && val !== undefined && String(val).toLowerCase().includes(q)),
      );
    }

    list.sort((a, b) => {
      const aVal = a[ex.tableSortCol];
      const bVal = b[ex.tableSortCol];
      if (
        aVal !== null &&
        bVal !== null &&
        aVal !== undefined &&
        bVal !== undefined &&
        aVal.constructor === Number &&
        bVal.constructor === Number &&
        Number.isFinite(aVal) &&
        Number.isFinite(bVal)
      ) {
        return ex.tableSortDir === "asc" ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
      }
      const strA = String(aVal ?? "");
      const strB = String(bVal ?? "");
      return ex.tableSortDir === "asc" ? strA.localeCompare(strB) : strB.localeCompare(strA);
    });

    return list;
  }, [ex.queryResult?.records, ex.tableSearch, ex.tableSortCol, ex.tableSortDir]);

  const totalPages = Math.max(1, Math.ceil(filteredAndSortedRecords.length / ex.tablePageSize));
  const paginatedRecords = useMemo(() => {
    const start = (ex.tablePage - 1) * ex.tablePageSize;
    return filteredAndSortedRecords.slice(start, start + ex.tablePageSize);
  }, [filteredAndSortedRecords, ex.tablePage, ex.tablePageSize]);

  const displayName = initialDatasetName || ex.activeDatasetMeta?.filename || "Active Dataset";
  const selectedColMeta = ex.schemaColumns.find((c) => c.name === ex.dimension);
  const isTemporalDimension =
    Boolean(selectedColMeta?.is_temporal) || (Array.isArray(ex.temporalCols) && ex.temporalCols.includes(ex.dimension));
  const showChartCanvas = ex.viewMode === "chart" && ex.chartType !== "table";
  const hasChartData = Boolean(ex.queryResult?.echarts_option) && (ex.queryResult?.records?.length ?? 0) > 0;
  const activeRowsCount = ex.activeDatasetMeta?.rows;

  return (
    <div
      className={`w-full rounded-2xl border border-stone-300 bg-[#FAF8F5] text-stone-900 dark:border-slate-800 dark:bg-[#07080a] dark:text-slate-100 overflow-hidden shadow-sm dark:shadow-none ${
        ex.isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#FAF8F5] dark:bg-[#07080a] overflow-y-auto" : ""
      }`}
    >
      <div className="px-3.5 sm:px-5 py-3 border-b border-stone-200 bg-[#FAF8F5] dark:border-slate-800 dark:bg-[#0c0d12] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0 w-full sm:w-auto">
          <div className="h-9 w-9 rounded-xl bg-stone-200 border border-stone-300 dark:bg-slate-800 dark:border-slate-700/80 flex items-center justify-center text-stone-700 dark:text-zinc-300 shrink-0">
            <Sliders className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold tracking-tight text-stone-900 dark:text-white truncate">
                Interactive Visual Analytics
              </span>
            </div>

            <div className="flex items-center gap-2 mt-0.5 text-xs text-stone-600 dark:text-slate-400">
              {allowDatasetSwitching && ex.availableDatasets.length > 1 ? (
                <div className="flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5 text-stone-500 dark:text-slate-400" />
                  <select
                    id="dataset-select"
                    aria-label="Select dataset to explore"
                    value={ex.selectedTaskId}
                    onChange={(e) => ex.selectDataset(e.target.value)}
                    className="bg-white border border-stone-300 text-xs rounded px-2 py-0.5 text-stone-900 focus:outline-none focus:border-[#800000]/50 dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200 dark:focus:border-white/40 font-mono max-w-[180px] xs:max-w-[240px] sm:max-w-xs truncate shadow-2xs"
                  >
                    {ex.availableDatasets.map((d) => (
                      <option key={d.task_id} value={d.task_id}>
                        {d.filename} ({d.rows !== undefined && Number.isFinite(d.rows) ? d.rows.toLocaleString() : d.rows || 0} rows)
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="flex items-center gap-2 font-mono">
                  <Database className="h-3.5 w-3.5 text-stone-500 dark:text-slate-400" />
                  <span className="text-stone-800 dark:text-slate-200 font-medium">{displayName}</span>
                  {activeRowsCount !== undefined && activeRowsCount !== null && (
                    <span>• {Number.isFinite(activeRowsCount) ? activeRowsCount.toLocaleString() : activeRowsCount} rows</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 self-end sm:self-auto">
          {ex.filters.length > 0 && (
            <Badge
              variant="outline"
              className="bg-sky-500/10 text-sky-700 border-sky-300 dark:bg-sky-500/10 dark:text-sky-400 dark:border-sky-500/30 text-xs px-2 py-0.5 rounded-lg font-mono"
            >
              {ex.filters.length} Filter{ex.filters.length > 1 ? "s" : ""}
            </Badge>
          )}

          <Button
            size="sm"
            variant="outline"
            onClick={() => void ex.executeQuery(undefined, true)}
            disabled={ex.executing || ex.loadingSchema}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${ex.executing ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">Refresh</span>
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={() => void ex.handleCopyChartImage()}
            disabled={!ex.queryResult}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
            title="Copy chart image to clipboard"
          >
            {ex.copiedChart ? <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span className="hidden sm:inline">{ex.copiedChart ? "Copied" : "Copy"}</span>
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={ex.handleExportChart}
            disabled={!ex.queryResult}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
            title="Export chart as PNG"
          >
            <Download className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">PNG</span>
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={ex.handleExportCsv}
            disabled={!ex.queryResult}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
            title="Export grouped data to CSV"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
            <span className="hidden sm:inline">CSV</span>
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => ex.setIsFullscreen(!ex.isFullscreen)}
            className="h-8 w-8 p-0 rounded-lg text-stone-600 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white cursor-pointer"
            title={ex.isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {ex.isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      <div className="lg:hidden px-3.5 py-2.5 bg-[#FAF8F5] border-b border-stone-200 dark:bg-[#0a0b10] dark:border-slate-800/80 flex items-center justify-center">
        <div className="grid grid-cols-2 p-1 rounded-xl bg-stone-200/80 border border-stone-300 dark:bg-black/50 dark:border-white/10 w-full max-w-sm">
          <button
            type="button"
            onClick={() => ex.setMobileTab("chart")}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${
              ex.mobileTab === "chart"
                ? "bg-[#800000] text-white border border-[#800000] shadow-xs font-semibold dark:bg-white dark:text-zinc-950 dark:border-white"
                : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200"
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
            <span>Chart &amp; Visuals</span>
          </button>
          <button
            type="button"
            onClick={() => ex.setMobileTab("controls")}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${
              ex.mobileTab === "controls"
                ? "bg-[#800000] text-white border border-[#800000] shadow-xs font-semibold dark:bg-white dark:text-zinc-950 dark:border-white"
                : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200"
            }`}
          >
            <Sliders className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
            <span>Query Controls</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[640px]">
        <ExploreControlsSidebar
          mobileVisible={ex.mobileTab === "controls"}
          recommendations={ex.recommendations}
          onApplyRecommendation={ex.applyRecommendation}
          chartType={ex.chartType}
          setChartType={ex.setChartType}
          dimension={ex.dimension}
          onChangeDimension={ex.setDimension}
          schemaColumns={ex.schemaColumns}
          isTemporalDimension={isTemporalDimension}
          timeGrain={ex.timeGrain}
          setTimeGrain={ex.setTimeGrain}
          secondaryMetric={ex.secondaryMetric}
          setSecondaryMetric={ex.setSecondaryMetric}
          secondaryAggregate={ex.secondaryAggregate}
          setSecondaryAggregate={ex.setSecondaryAggregate}
          numericCols={ex.numericCols}
          aggregate={ex.aggregate}
          setAggregate={ex.setAggregate}
          metric={ex.metric}
          setMetric={ex.setMetric}
          secondaryDimension={ex.secondaryDimension}
          setSecondaryDimension={ex.setSecondaryDimension}
          barMode={ex.barMode}
          setBarMode={ex.setBarMode}
          filters={ex.filters}
          onAddFilter={ex.addFilter}
          onRemoveFilter={ex.removeFilter}
          onUpdateFilter={ex.updateFilter}
          rowLimit={ex.rowLimit}
          setRowLimit={ex.setRowLimit}
          sortOrder={ex.sortOrder}
          setSortOrder={ex.setSortOrder}
          colorPalette={ex.colorPalette}
          setColorPalette={ex.setColorPalette}
          onMobileViewChart={() => ex.setMobileTab("chart")}
        />

        <div
          className={`lg:col-span-8 p-3.5 sm:p-6 flex flex-col justify-between space-y-4 sm:space-y-6 bg-stone-50/70 dark:bg-[#07080a] ${
            ex.mobileTab === "chart" ? "flex" : "hidden lg:flex"
          }`}
        >
          {ex.queryResult?.kpis && (
            <ExploreTelemetryDeck
              queryResult={ex.queryResult}
              aggregate={ex.aggregate}
              metric={ex.metric}
              dimension={ex.dimension}
            />
          )}

          {ex.queryResult?.smart_insights && ex.queryResult.smart_insights.length > 0 && (
            <ExploreObservationsStrip insights={ex.queryResult.smart_insights} />
          )}

          <div className="flex-1 min-h-[400px] sm:min-h-[480px] rounded-xl border border-stone-200/90 bg-white/95 dark:border-white/10 dark:bg-[#0c0d12] p-3.5 sm:p-5 flex flex-col relative shadow-2xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-stone-200 dark:border-white/10 mb-3">
              <div className="flex items-center gap-1 bg-stone-100 dark:bg-[#14151c] p-0.5 rounded-lg border border-stone-200/80 dark:border-white/10">
                <button
                  type="button"
                  onClick={() => ex.setViewMode("chart")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition cursor-pointer flex items-center gap-1.5 ${
                    ex.viewMode === "chart"
                      ? "bg-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:text-zinc-950"
                      : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white"
                  }`}
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>Visual Chart</span>
                </button>
                <button
                  type="button"
                  onClick={() => ex.setViewMode("data")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition cursor-pointer flex items-center gap-1.5 ${
                    ex.viewMode === "data"
                      ? "bg-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:text-zinc-950"
                      : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white"
                  }`}
                >
                  <Table2 className="h-3.5 w-3.5" />
                  <span>Aggregated Data</span>
                  <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded-full bg-black/10 dark:bg-white/15 tabular-nums">
                    {ex.queryResult?.records?.length ?? 0}
                  </span>
                </button>
              </div>

              <div className="flex items-center gap-2 flex-wrap text-xs font-sans">
                {ex.viewMode === "chart" && (ex.chartType === "echarts_bar" || ex.chartType === "echarts_horizontal_bar") && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => ex.setChartType(ex.chartType === "echarts_bar" ? "echarts_horizontal_bar" : "echarts_bar")}
                    className="h-7 px-2 rounded-lg text-[11px] gap-1 border-stone-200 bg-white hover:bg-stone-50 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
                    title="Toggle Vertical Column vs Horizontal Ranking"
                  >
                    {ex.chartType === "echarts_bar" ? <AlignLeft className="h-3 w-3" /> : <BarChart3 className="h-3 w-3" />}
                    <span>{ex.chartType === "echarts_bar" ? "Horizontal Rank" : "Vertical Column"}</span>
                  </Button>
                )}

                <div className="text-xs text-stone-600 dark:text-zinc-400 font-mono truncate hidden md:inline">
                  {ex.queryResult?.metric_label} by {ex.dimension}
                  {ex.timeGrain && <span className="text-sky-600 dark:text-sky-400 ml-1">({ex.timeGrain})</span>}
                </div>
              </div>
            </div>

            {ex.executing && (
              <div className="absolute inset-0 bg-white/80 dark:bg-[#07080a]/80 backdrop-blur-xs flex flex-col items-center justify-center z-20 gap-2 rounded-2xl">
                <Loader2 className="h-6 w-6 animate-spin text-stone-700 dark:text-zinc-300" />
                <span className="text-xs font-mono text-stone-600 dark:text-zinc-400">Computing visual analytics...</span>
              </div>
            )}

            {showChartCanvas && ex.queryResult?.echarts_option && (
              <div className="flex-1 w-full min-h-[420px]">
                <NativeEChartsView
                  option={ex.queryResult.echarts_option}
                  onChartClick={ex.handleChartClick}
                  chartRef={ex.nativeChartInstanceRef}
                  height={ex.isFullscreen ? "72vh" : "480px"}
                />
              </div>
            )}

            {showChartCanvas && !hasChartData && !ex.executing && (
              <div className="flex-1 w-full min-h-[420px] flex flex-col items-center justify-center p-8 text-center border border-dashed border-stone-300 dark:border-white/10 rounded-2xl bg-stone-100/50 dark:bg-white/[0.02]">
                <div className="h-12 w-12 rounded-2xl bg-stone-200 dark:bg-white/10 border border-stone-300 dark:border-white/10 flex items-center justify-center text-stone-500 dark:text-zinc-400 mb-3">
                  <BarChart3 className="h-6 w-6 text-stone-500 dark:text-zinc-400" />
                </div>
                <h3 className="text-sm font-semibold text-stone-900 dark:text-white">No Visual Data Available</h3>
                <p className="text-xs text-stone-600 dark:text-zinc-400 max-w-sm mt-1">
                  No data records matched the selected dimension and metric filters. Try adjusting your dimension, metric, or filters on the left panel.
                </p>
              </div>
            )}

            {(ex.viewMode === "data" || ex.chartType === "table") && ex.queryResult?.records && (
              <ExploreDataTable
                dimension={ex.dimension}
                secondaryDimension={ex.secondaryDimension}
                metricLabel={ex.queryResult.metric_label}
                colorPalette={ex.colorPalette}
                tableSearch={ex.tableSearch}
                onSearchChange={ex.setTableSearch}
                tableSortCol={ex.tableSortCol}
                onSortClick={ex.handleSortClick}
                tablePage={ex.tablePage}
                onPageChange={ex.setTablePage}
                tablePageSize={ex.tablePageSize}
                onPageSizeChange={ex.setTablePageSize}
                totalPages={totalPages}
                filteredRecordsCount={filteredAndSortedRecords.length}
                paginatedRecords={paginatedRecords}
                totalMetricSum={totalMetricSum}
                totalRowCount={totalRowCount}
                onFilterValueClick={(val) => ex.handleChartClick({ name: val })}
                onCopyRowJson={(row) => {
                  navigator.clipboard.writeText(JSON.stringify(row, null, 2));
                  ex.toast({ title: "Row Copied", description: "Row JSON copied to clipboard." });
                }}
              />
            )}
          </div>

          <div className="flex items-center justify-between text-xs text-stone-600 dark:text-slate-400 pt-2 border-t border-stone-200 dark:border-slate-800/80 font-mono">
            <span>{ex.queryResult?.total_groups !== undefined ? `${ex.queryResult.total_groups.toLocaleString()} groups computed` : "Ready"}</span>
            <span>{ex.queryResult?.records?.length ? `Displaying top ${ex.queryResult.records.length} records` : ""}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
