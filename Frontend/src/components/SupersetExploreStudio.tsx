import React, { useState, useEffect, useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import {
  BarChart3,
  TrendingUp,
  PieChart,
  Grid,
  Table2,
  Sliders,
  Database,
  RefreshCw,
  Maximize2,
  Minimize2,
  Download,
  Filter,
  Layers,
  ArrowUpDown,
  Check,
  ChevronDown,
  Info,
  Loader2,
  ChevronRight,
  FileSpreadsheet
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

interface DatasetItem {
  task_id: string;
  filename: string;
  rows: number;
  columns_count: number;
  columns: string[];
  numeric_columns?: string[];
  categorical_columns?: string[];
  created_at?: string;
}

interface SupersetExploreStudioProps {
  taskId?: string;
  datasetName?: string;
  allowDatasetSwitching?: boolean;
}

const CHART_TYPES = [
  { id: "echarts_bar", label: "Bar / Column", icon: BarChart3, desc: "Categorical comparisons & counts" },
  { id: "echarts_line", label: "Line / Trend", icon: TrendingUp, desc: "Trends & series distributions" },
  { id: "echarts_area", label: "Area Fill", icon: Layers, desc: "Cumulative series volume" },
  { id: "echarts_pie", label: "Pie / Donut", icon: PieChart, desc: "Part-to-whole proportions" },
  { id: "echarts_treemap", label: "Treemap", icon: Grid, desc: "Hierarchical rectangular area" },
  { id: "table", label: "Data Grid", icon: Table2, desc: "Tabular grouped records" },
];

const AGGREGATIONS = [
  { id: "COUNT", label: "Count of Rows" },
  { id: "SUM", label: "Sum" },
  { id: "AVG", label: "Average (Mean)" },
  { id: "MIN", label: "Minimum" },
  { id: "MAX", label: "Maximum" },
  { id: "MEDIAN", label: "Median" },
];

const COLOR_PALETTES = [
  { id: "supersetColors", label: "Classic Ocean", preview: ["#1FA8C9", "#454E7C", "#5AC189", "#FF7F44"] },
  { id: "modernDark", label: "Modern Dark", preview: ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"] },
  { id: "airbnb", label: "Airbnb Coral", preview: ["#FF5A5F", "#00A699", "#7B0051", "#007A87"] },
  { id: "d3Category10", label: "D3 Categorical", preview: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] },
  { id: "emerald", label: "Emerald Clean", preview: ["#10b981", "#059669", "#34d399", "#3b82f6"] },
];

export const SupersetExploreStudio: React.FC<SupersetExploreStudioProps> = ({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  allowDatasetSwitching = false,
}) => {
  const { toast } = useToast();
  const echartsRef = useRef<any>(null);

  // Dataset Selection State
  const [availableDatasets, setAvailableDatasets] = useState<DatasetItem[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | undefined>(initialTaskId);
  const [activeDatasetMeta, setActiveDatasetMeta] = useState<DatasetItem | null>(null);

  // Schema & Explore Options
  const [loadingSchema, setLoadingSchema] = useState<boolean>(true);
  const [schemaColumns, setSchemaColumns] = useState<Array<{
    name: string;
    type: number;
    type_name: string;
    is_numeric: boolean;
    is_temporal: boolean;
    null_count: number;
    unique_values: number;
  }>>([]);
  const [numericCols, setNumericCols] = useState<string[]>([]);
  const [categoricalCols, setCategoricalCols] = useState<string[]>([]);

  // Explore Controls State
  const [chartType, setChartType] = useState<string>("echarts_bar");
  const [dimension, setDimension] = useState<string>("");
  const [secondaryDimension, setSecondaryDimension] = useState<string>("");
  const [metric, setMetric] = useState<string>("");
  const [aggregate, setAggregate] = useState<string>("COUNT");
  const [rowLimit, setRowLimit] = useState<number>(15);
  const [sortOrder, setSortOrder] = useState<string>("desc");
  const [colorPalette, setColorPalette] = useState<string>("supersetColors");

  // Query Results State
  const [executing, setExecuting] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<"chart" | "data">("chart");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // 1. Load available datasets if switching is enabled or if initialTaskId is absent
  useEffect(() => {
    api.listRecentDatasets(15)
      .then(res => {
        if (res && res.datasets && res.datasets.length > 0) {
          setAvailableDatasets(res.datasets);
          if (!selectedTaskId) {
            setSelectedTaskId(res.datasets[0].task_id);
            setActiveDatasetMeta(res.datasets[0]);
          } else {
            const match = res.datasets.find(d => d.task_id === selectedTaskId);
            if (match) setActiveDatasetMeta(match);
          }
        }
      })
      .catch(err => {
        console.warn("Could not load recent datasets:", err);
      });
  }, [selectedTaskId]);

  // Keep selectedTaskId in sync with prop changes
  useEffect(() => {
    if (initialTaskId && initialTaskId !== selectedTaskId) {
      setSelectedTaskId(initialTaskId);
    }
  }, [initialTaskId]);

  // 2. Fetch schema whenever selectedTaskId changes
  useEffect(() => {
    if (!selectedTaskId) return;

    setLoadingSchema(true);
    api.getSupersetSchema(selectedTaskId)
      .then(res => {
        setSchemaColumns(res.columns || []);
        setNumericCols(res.numeric_columns || []);
        setCategoricalCols(res.categorical_columns || []);

        // Pick intelligent default dimension & metric
        const defaultDim = res.categorical_columns && res.categorical_columns.length > 0
          ? res.categorical_columns[0]
          : res.columns[0]?.name || "";
        setDimension(defaultDim);

        if (res.numeric_columns && res.numeric_columns.length > 0) {
          setMetric(res.numeric_columns[0]);
          setAggregate("AVG");
        } else {
          setMetric("");
          setAggregate("COUNT");
        }
      })
      .catch(err => {
        console.error("Failed to load Superset schema:", err);
        toast({
          title: "Schema Error",
          description: "Could not introspect dataset schema.",
          variant: "destructive"
        });
      })
      .finally(() => {
        setLoadingSchema(false);
      });
  }, [selectedTaskId, toast]);

  // 3. Execute Explore Query whenever controls change
  const executeQuery = async () => {
    if (!selectedTaskId || !dimension) return;

    setExecuting(true);
    try {
      const res = await api.querySupersetExplore({
        task_id: selectedTaskId,
        chart_type: chartType,
        dimension: dimension,
        secondary_dimension: secondaryDimension || undefined,
        metric: metric || undefined,
        aggregate: aggregate,
        row_limit: rowLimit,
        sort_order: sortOrder,
        color_palette: colorPalette,
      });
      setQueryResult(res);
    } catch (err: any) {
      console.error("Explore query execution failed:", err);
      toast({
        title: "Query Failed",
        description: err.message || "Could not execute aggregation.",
        variant: "destructive",
      });
    } finally {
      setExecuting(false);
    }
  };

  // Re-run query on dimension/metric/type changes
  useEffect(() => {
    if (dimension && selectedTaskId) {
      executeQuery();
    }
  }, [selectedTaskId, chartType, dimension, secondaryDimension, metric, aggregate, rowLimit, sortOrder, colorPalette]);

  // Export current chart as PNG image
  const handleExportChart = () => {
    if (!echartsRef.current) return;
    try {
      const chartInstance = echartsRef.current.getEchartsInstance();
      const base64Url = chartInstance.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: "#090d16",
      });
      const link = document.createElement("a");
      link.href = base64Url;
      link.download = `Superset_Explore_${dimension}_${new Date().toISOString().slice(0, 10)}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      toast({ title: "Chart Exported", description: "Downloaded high-res PNG image." });
    } catch (e: any) {
      toast({ title: "Export Failed", description: e.message, variant: "destructive" });
    }
  };

  // Export aggregated records to CSV
  const handleExportCsv = () => {
    if (!queryResult || !queryResult.records || queryResult.records.length === 0) return;
    const records = queryResult.records;
    const headers = Object.keys(records[0]);
    const csvContent = [
      headers.join(","),
      ...records.map((r: any) => headers.map(h => `"${r[h] ?? ""}"`).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `Superset_Aggregate_${dimension}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast({ title: "CSV Ready", description: "Exported aggregated records to CSV." });
  };

  const displayName = initialDatasetName || activeDatasetMeta?.filename || "Active Dataset";

  return (
    <div className={`w-full rounded-2xl border border-slate-800 bg-[#090d16] text-slate-100 overflow-hidden ${
      isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#090d16] overflow-y-auto" : ""
    }`}>
      
      {/* ─── Top Studio Header Bar ─── */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-[#0c0d12] flex flex-wrap items-center justify-between gap-4">
        
        {/* Left: Branding & Dataset Identity */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-slate-800 border border-slate-700/80 flex items-center justify-center text-zinc-300 shrink-0">
            <Sliders className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold tracking-tight text-white">
                Interactive Visual Analytics
              </span>
            </div>
            
            {/* Dataset selector dropdown or name badge */}
            <div className="flex items-center gap-2 mt-0.5 text-xs text-slate-400">
              {allowDatasetSwitching && availableDatasets.length > 1 ? (
                <div className="flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5 text-slate-400" />
                  <select
                    value={selectedTaskId}
                    onChange={(e) => {
                      const newId = e.target.value;
                      setSelectedTaskId(newId);
                      const match = availableDatasets.find(d => d.task_id === newId);
                      if (match) setActiveDatasetMeta(match);
                    }}
                    className="bg-[#121215] border border-slate-700 text-xs rounded px-2 py-0.5 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
                  >
                    {availableDatasets.map((d) => (
                      <option key={d.task_id} value={d.task_id}>
                        {d.filename} ({d.rows.toLocaleString()} rows)
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="flex items-center gap-2 font-mono">
                  <Database className="h-3.5 w-3.5 text-slate-400" />
                  <span className="text-slate-200 font-medium">{displayName}</span>
                  {activeDatasetMeta?.rows && (
                    <span>• {activeDatasetMeta.rows.toLocaleString()} rows</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          {/* Refresh Query */}
          <Button
            size="sm"
            variant="outline"
            onClick={executeQuery}
            disabled={executing || loadingSchema}
            className="h-8 rounded-lg border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 gap-1.5"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${executing ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">Refresh</span>
          </Button>

          {/* Export Actions */}
          <Button
            size="sm"
            variant="outline"
            onClick={handleExportChart}
            disabled={!queryResult}
            className="h-8 rounded-lg border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 gap-1.5"
            title="Export chart as PNG"
          >
            <Download className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">PNG</span>
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={handleExportCsv}
            disabled={!queryResult}
            className="h-8 rounded-lg border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 gap-1.5"
            title="Export grouped data to CSV"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" />
            <span className="hidden sm:inline">CSV</span>
          </Button>

          {/* Fullscreen Toggle */}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="h-8 w-8 p-0 rounded-lg text-slate-400 hover:text-white"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* ─── Two-Column Explore Split Layout ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[640px]">
        
        {/* ──── LEFT PANEL: Superset Explore Controls (4 cols) ──── */}
        <div className="lg:col-span-4 p-5 border-b lg:border-b-0 lg:border-r border-slate-800 bg-[#0b0f1a] space-y-6">
          
          {/* Section: Chart Type Picker */}
          <div className="space-y-2">
            <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
              Visualization Type
            </label>
            <div className="grid grid-cols-3 gap-2">
              {CHART_TYPES.map((ct) => {
                const Icon = ct.icon;
                const isActive = chartType === ct.id;
                return (
                  <button
                    key={ct.id}
                    onClick={() => setChartType(ct.id)}
                    className={`flex flex-col items-center justify-center p-2.5 rounded-xl border text-center transition ${
                      isActive
                        ? "bg-white/10 border-white/20 text-white"
                        : "bg-[#0e1017] border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                    }`}
                  >
                    <Icon className="h-4 w-4 mb-1" />
                    <span className="text-[11px] font-medium leading-tight">{ct.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section: Dimension (X-Axis / Group By) */}
          <div className="space-y-2">
            <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
              Dimension (Group By / X-Axis)
            </label>
            <select
              value={dimension}
              onChange={(e) => setDimension(e.target.value)}
              className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
            >
              {schemaColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.type_name})
                </option>
              ))}
            </select>
          </div>

          {/* Section: Metric (Y-Axis / Measure) */}
          <div className="space-y-2">
            <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
              Metric & Aggregation (Y-Axis)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {/* Aggregation Function */}
              <select
                value={aggregate}
                onChange={(e) => setAggregate(e.target.value)}
                className="bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
              >
                {AGGREGATIONS.map((agg) => (
                  <option key={agg.id} value={agg.id}>
                    {agg.label}
                  </option>
                ))}
              </select>

              {/* Column Selection (or disabled if row count) */}
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                disabled={aggregate === "COUNT" && !metric}
                className="bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono disabled:opacity-50"
              >
                <option value="">(Row Count)</option>
                {numericCols.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Section: Secondary Breakdown (Series) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                Secondary Breakdown
              </label>
              <span className="text-[10px] text-slate-400 font-mono">Optional</span>
            </div>
            <select
              value={secondaryDimension}
              onChange={(e) => setSecondaryDimension(e.target.value)}
              className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
            >
              <option value="">None (Single Series)</option>
              {schemaColumns
                .filter((c) => c.name !== dimension)
                .map((col) => (
                  <option key={col.name} value={col.name}>
                    {col.name}
                  </option>
                ))}
            </select>
          </div>

          {/* Section: Limits & Sorting */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="space-y-1.5">
              <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                Row Limit
              </label>
              <select
                value={rowLimit}
                onChange={(e) => setRowLimit(Number(e.target.value))}
                className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
              >
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
                <option value={15}>Top 15</option>
                <option value={25}>Top 25</option>
                <option value={50}>Top 50</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                Sort Order
              </label>
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value)}
                className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-white/40 font-mono"
              >
                <option value="desc">High to Low (Desc)</option>
                <option value="asc">Low to High (Asc)</option>
              </select>
            </div>
          </div>

          {/* Section: Color Palette */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
              Color Palette
            </label>
            <div className="space-y-1.5">
              {COLOR_PALETTES.map((pal) => (
                <button
                  key={pal.id}
                  onClick={() => setColorPalette(pal.id)}
                  className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg border text-xs transition ${
                    colorPalette === pal.id
                      ? "bg-slate-800 border-white/30 text-white"
                      : "bg-[#0e1017] border-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <span>{pal.label}</span>
                  <div className="flex items-center gap-1">
                    {pal.preview.map((c, i) => (
                      <span key={i} className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: c }} />
                    ))}
                  </div>
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* ──── RIGHT CANVAS: Live Visualization & KPIs (8 cols) ──── */}
        <div className="lg:col-span-8 p-5 sm:p-6 flex flex-col justify-between space-y-6 bg-[#090d16]">
          
          {/* Top: Superset BigNumber KPI Cards */}
          {queryResult?.kpis && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-3.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                  Total Records
                </span>
                <span className="text-xl sm:text-2xl font-bold text-white mt-1 block">
                  {queryResult.kpis.total_rows.toLocaleString()}
                </span>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-3.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                  {queryResult.kpis.metric_label || "Metric Value"}
                </span>
                <span className="text-xl sm:text-2xl font-bold text-emerald-400 mt-1 block">
                  {queryResult.kpis.mean !== null && queryResult.kpis.mean !== undefined
                    ? queryResult.kpis.mean.toLocaleString()
                    : queryResult.kpis.sum !== null && queryResult.kpis.sum !== undefined
                    ? queryResult.kpis.sum.toLocaleString()
                    : "—"}
                </span>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-3.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                  Peak Group
                </span>
                <span className="text-xl sm:text-2xl font-bold text-emerald-400 mt-1 block truncate">
                  {queryResult.records?.[0]?.[dimension] ?? "—"}
                </span>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0e1424] p-3.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                  Total Groups
                </span>
                <span className="text-xl sm:text-2xl font-bold text-slate-200 mt-1 block">
                  {queryResult.total_groups?.toLocaleString() ?? "—"}
                </span>
              </div>
            </div>
          )}

          {/* Main Chart or Table Canvas */}
          <div className="flex-1 min-h-[460px] rounded-xl border border-slate-800 bg-[#0b0f1a] p-4 flex flex-col relative">
            
            {/* View Switcher: Chart vs Raw Data */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-2">
              <div className="flex items-center gap-1.5 bg-[#0e1017] p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setViewMode("chart")}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                    viewMode === "chart"
                      ? "bg-slate-800 text-white shadow-xs"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Visual Chart
                </button>
                <button
                  onClick={() => setViewMode("data")}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                    viewMode === "data"
                      ? "bg-slate-800 text-white shadow-xs"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Aggregated Data ({queryResult?.records?.length ?? 0})
                </button>
              </div>

              <div className="text-xs text-slate-400 font-mono">
                {queryResult?.metric_label} by {dimension}
              </div>
            </div>

            {/* Loading Overlay */}
            {executing && (
              <div className="absolute inset-0 bg-[#090d16]/80 backdrop-blur-xs flex flex-col items-center justify-center z-20 gap-2">
                <Loader2 className="h-6 w-6 animate-spin text-zinc-300" />
                <span className="text-xs font-mono text-slate-400">Computing visual analytics...</span>
              </div>
            )}

            {/* View Mode 1: Interactive ECharts Canvas */}
            {viewMode === "chart" && chartType !== "table" && queryResult?.echarts_option && (
              <div className="flex-1 w-full min-h-[400px]">
                <ReactECharts
                  ref={echartsRef}
                  option={queryResult.echarts_option}
                  style={{ height: "420px", width: "100%" }}
                  theme="dark"
                  notMerge={true}
                  lazyUpdate={true}
                />
              </div>
            )}

            {/* View Mode 2: Aggregated Data Table */}
            {(viewMode === "data" || chartType === "table") && queryResult?.records && (
              <div className="flex-1 overflow-x-auto min-h-[400px]">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#0e1017] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3">{dimension}</th>
                      {secondaryDimension && <th className="py-2.5 px-3">{secondaryDimension}</th>}
                      <th className="py-2.5 px-3">{queryResult.metric_label || "Value"}</th>
                      <th className="py-2.5 px-3">Records</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-200">
                    {queryResult.records.map((row: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-800/30 transition">
                        <td className="py-2 px-3 text-slate-400">{idx + 1}</td>
                        <td className="py-2 px-3 font-medium text-white">{row[dimension] ?? "—"}</td>
                        {secondaryDimension && (
                          <td className="py-2 px-3 text-slate-300">{row[secondaryDimension] ?? "—"}</td>
                        )}
                        <td className="py-2 px-3 text-zinc-200 font-bold">
                          {typeof row.value === "number" ? row.value.toLocaleString() : row.value ?? "—"}
                        </td>
                        <td className="py-2 px-3 text-slate-400">{row.count?.toLocaleString() ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

          </div>

          {/* Footer Metadata */}
          <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80 font-mono">
            <span>
              {queryResult?.total_groups !== undefined ? `${queryResult.total_groups.toLocaleString()} groups computed` : "Ready"}
            </span>
            <span>
              {queryResult?.records?.length ? `Displaying top ${queryResult.records.length} records` : ""}
            </span>
          </div>

        </div>

      </div>

    </div>
  );
};
