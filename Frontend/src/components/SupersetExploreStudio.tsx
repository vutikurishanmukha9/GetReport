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
  FileSpreadsheet,
  Plus,
  Trash2,
  Server,
  Sparkles,
  Calendar,
  ScatterChart,
  Box,
  ChevronDown,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { SupersetEmbeddedDashboard } from "./SupersetEmbeddedDashboard";

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

interface FilterItem {
  id: string;
  column: string;
  operator: string;
  value: string;
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
  { id: "echarts_scatter", label: "Scatter Plot", icon: Sliders, desc: "Bivariate correlations" },
  { id: "echarts_heatmap", label: "2D Heatmap", icon: Grid, desc: "Matrix intensity (requires 2 dims)" },
  { id: "echarts_boxplot", label: "Boxplot", icon: Table2, desc: "Five-number quartile spread" },
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
  { id: "COUNT_DISTINCT", label: "Count Distinct" },
];

const TIME_GRAINS = [
  { id: "", label: "Raw Date (No Bucketing)" },
  { id: "day", label: "Day" },
  { id: "week", label: "Week" },
  { id: "month", label: "Month" },
  { id: "quarter", label: "Quarter" },
  { id: "year", label: "Year" },
];

const FILTER_OPERATORS = [
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

const COLOR_PALETTES = [
  { id: "supersetColors", label: "Classic Ocean", preview: ["#1FA8C9", "#454E7C", "#5AC189", "#FF7F44"] },
  { id: "modernDark", label: "Modern Dark", preview: ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"] },
  { id: "airbnb", label: "Airbnb Coral", preview: ["#FF5A5F", "#00A699", "#7B0051", "#007A87"] },
  { id: "d3Category10", label: "D3 Categorical", preview: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] },
  { id: "emerald", label: "Emerald Clean", preview: ["#10b981", "#059669", "#34d399", "#3b82f6"] },
];

const formatCompactNumber = (val: number | null | undefined): string => {
  if (val === null || val === undefined || isNaN(val)) return "—";
  if (Math.abs(val) >= 1_000_000_000) return (val / 1_000_000_000).toFixed(2) + "B";
  if (Math.abs(val) >= 1_000_000) return (val / 1_000_000).toFixed(2) + "M";
  if (Math.abs(val) >= 1_000) return (val / 1_000).toFixed(1) + "K";
  return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2);
};

export const SupersetExploreStudio: React.FC<SupersetExploreStudioProps> = ({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  allowDatasetSwitching = false,
}) => {
  const { toast } = useToast();
  const echartsRef = useRef<any>(null);

  // Studio Mode: Native Polars + ECharts vs Containerized Superset Companion
  const [studioMode, setStudioMode] = useState<"native" | "companion">("native");

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
    sample_values?: string[];
  }>>([]);
  const [numericCols, setNumericCols] = useState<string[]>([]);
  const [categoricalCols, setCategoricalCols] = useState<string[]>([]);
  const [temporalCols, setTemporalCols] = useState<string[]>([]);

  // Explore Controls State
  const [chartType, setChartType] = useState<string>("echarts_bar");
  const [dimension, setDimension] = useState<string>("");
  const [secondaryDimension, setSecondaryDimension] = useState<string>("");
  const [metric, setMetric] = useState<string>("");
  const [aggregate, setAggregate] = useState<string>("COUNT");
  const [timeGrain, setTimeGrain] = useState<string>("");
  const [filters, setFilters] = useState<FilterItem[]>([]);
  const [showFilterBuilder, setShowFilterBuilder] = useState<boolean>(false);
  const [rowLimit, setRowLimit] = useState<number>(25);
  const [sortOrder, setSortOrder] = useState<string>("desc");
  const [colorPalette, setColorPalette] = useState<string>("supersetColors");

  // Query Results State
  const [executing, setExecuting] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<"chart" | "data">("chart");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [mobileTab, setMobileTab] = useState<"chart" | "controls">("chart");

  // Dynamic Chart Auto-Resize
  useEffect(() => {
    const handleResize = () => {
      if (echartsRef.current) {
        try {
          echartsRef.current.getEchartsInstance?.()?.resize();
        } catch {
          // ignore
        }
      }
    };
    window.addEventListener("resize", handleResize);
    const timer = setTimeout(handleResize, 120);
    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, [mobileTab, isFullscreen, viewMode, studioMode]);

  // 1. Load available datasets
  useEffect(() => {
    api.listRecentDatasets(20)
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
        setTemporalCols(res.temporal_columns || []);

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

  // 3. Execute Explore Query
  const executeQuery = async () => {
    if (!selectedTaskId || !dimension || studioMode !== "native") return;

    setExecuting(true);
    try {
      const activeFilters = filters
        .filter(f => f.column && (f.operator.includes("null") || f.value.trim() !== ""))
        .map(f => ({
          column: f.column,
          operator: f.operator,
          value: f.operator.includes("null") ? null : f.value,
        }));

      const res = await api.querySupersetExplore({
        task_id: selectedTaskId,
        chart_type: chartType,
        dimension: dimension,
        secondary_dimension: secondaryDimension || undefined,
        metric: metric || undefined,
        aggregate: aggregate,
        time_grain: timeGrain || undefined,
        filters: activeFilters.length > 0 ? activeFilters : undefined,
        row_limit: rowLimit,
        sort_order: sortOrder,
        color_palette: colorPalette,
      });
      setQueryResult(res);
    } catch (err: any) {
      console.error("Explore query execution failed:", err);
      toast({
        title: "Query Failed",
        description: err.message || "Could not execute analytical aggregation.",
        variant: "destructive",
      });
    } finally {
      setExecuting(false);
    }
  };

  useEffect(() => {
    if (dimension && selectedTaskId && studioMode === "native") {
      executeQuery();
    }
  }, [
    selectedTaskId,
    chartType,
    dimension,
    secondaryDimension,
    metric,
    aggregate,
    timeGrain,
    filters,
    rowLimit,
    sortOrder,
    colorPalette,
    studioMode,
  ]);

  // Filter actions
  const addFilter = () => {
    if (schemaColumns.length === 0) return;
    const newFilter: FilterItem = {
      id: Math.random().toString(36).substring(7),
      column: schemaColumns[0].name,
      operator: "eq",
      value: "",
    };
    setFilters([...filters, newFilter]);
    setShowFilterBuilder(true);
  };

  const removeFilter = (id: string) => {
    setFilters(filters.filter(f => f.id !== id));
  };

  const updateFilter = (id: string, field: keyof FilterItem, val: string) => {
    setFilters(filters.map(f => (f.id === id ? { ...f, [field]: val } : f)));
  };

  // Export current chart as PNG
  const handleExportChart = () => {
    if (!echartsRef.current) return;
    try {
      const chartInstance = echartsRef.current.getEchartsInstance();
      const base64Url = chartInstance.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: "#07080a",
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
  const selectedColMeta = schemaColumns.find(c => c.name === dimension);
  const isTemporalDimension = selectedColMeta?.is_temporal || temporalCols.includes(dimension);

  return (
    <div className={`w-full rounded-2xl border border-slate-800 bg-[#07080a] text-slate-100 overflow-hidden ${
      isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#07080a] overflow-y-auto" : ""
    }`}>
      
      {/* ─── Top Studio Header Bar ─── */}
      <div className="px-3.5 sm:px-5 py-3 border-b border-slate-800 bg-[#0c0d12] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        
        {/* Left: Branding & Dataset Identity */}
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0 w-full sm:w-auto">
          <div className="h-9 w-9 rounded-xl bg-slate-800 border border-slate-700/80 flex items-center justify-center text-zinc-300 shrink-0">
            <Sliders className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold tracking-tight text-white truncate">
                Interactive Visual Analytics
              </span>
              {/* Studio Mode Toggle */}
              <div className="flex items-center bg-[#121215] border border-slate-800 rounded-lg p-0.5 ml-1">
                <button
                  type="button"
                  onClick={() => setStudioMode("native")}
                  className={`text-[11px] font-medium px-2 py-0.5 rounded-md transition ${
                    studioMode === "native"
                      ? "bg-slate-800 text-white shadow-2xs font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Native Studio
                </button>
                <button
                  type="button"
                  onClick={() => setStudioMode("companion")}
                  className={`text-[11px] font-medium px-2 py-0.5 rounded-md transition flex items-center gap-1 ${
                    studioMode === "companion"
                      ? "bg-slate-800 text-white shadow-2xs font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Server className="h-3 w-3 text-emerald-400" />
                  <span>Superset Companion</span>
                </button>
              </div>
            </div>
            
            {/* Dataset selector dropdown or name badge */}
            <div className="flex items-center gap-2 mt-0.5 text-xs text-slate-400">
              {allowDatasetSwitching && availableDatasets.length > 1 ? (
                <div className="flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5 text-slate-400" />
                  <select
                    id="dataset-select"
                    aria-label="Select dataset to explore"
                    value={selectedTaskId}
                    onChange={(e) => {
                      const newId = e.target.value;
                      setSelectedTaskId(newId);
                      const match = availableDatasets.find(d => d.task_id === newId);
                      if (match) setActiveDatasetMeta(match);
                    }}
                    className="bg-[#121215] border border-slate-700 text-xs rounded px-2 py-0.5 text-slate-200 focus:outline-none focus:border-white/40 font-mono max-w-[180px] xs:max-w-[240px] sm:max-w-xs truncate"
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
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 self-end sm:self-auto">
          {studioMode === "native" && (
            <>
              {/* Filter Count Badge */}
              {filters.length > 0 && (
                <Badge
                  variant="outline"
                  className="bg-sky-500/10 text-sky-400 border-sky-500/30 text-xs px-2 py-0.5 rounded-lg font-mono"
                >
                  {filters.length} Filter{filters.length > 1 ? "s" : ""}
                </Badge>
              )}

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
            </>
          )}

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

      {/* ─── Mode B: Apache Superset Companion Embedded View ─── */}
      {studioMode === "companion" ? (
        <SupersetEmbeddedDashboard
          taskId={selectedTaskId}
          datasetName={displayName}
          dashboardId="1"
        />
      ) : (
        /* ─── Mode A: Native In-Memory Studio ─── */
        <>
          {/* Mobile View Switcher Tab (Mobile Only) */}
          <div className="lg:hidden px-3.5 py-2.5 bg-[#0a0b10] border-b border-slate-800/80 flex items-center justify-center">
            <div className="grid grid-cols-2 p-1 rounded-xl bg-black/50 border border-white/10 w-full max-w-sm">
              <button
                type="button"
                onClick={() => setMobileTab("chart")}
                className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${
                  mobileTab === "chart"
                    ? "bg-white/10 text-white border border-white/20 shadow-xs font-semibold"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <BarChart3 className="h-3.5 w-3.5 text-emerald-400" />
                <span>Chart & Visuals</span>
              </button>
              <button
                type="button"
                onClick={() => setMobileTab("controls")}
                className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${
                  mobileTab === "controls"
                    ? "bg-white/10 text-white border border-white/20 shadow-xs font-semibold"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Sliders className="h-3.5 w-3.5 text-amber-400" />
                <span>Query Controls</span>
              </button>
            </div>
          </div>

          {/* Two-Column Explore Split Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[640px]">
            
            {/* ──── LEFT PANEL: Superset Explore Controls (4 cols) ──── */}
            <div className={`lg:col-span-4 p-4 sm:p-5 border-b lg:border-b-0 lg:border-r border-slate-800 bg-[#0c0d12] space-y-4 sm:space-y-5 ${
              mobileTab === "controls" ? "block" : "hidden lg:block"
            }`}>
              
              {/* Section: Chart Type Picker */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  Visualization Type
                </span>
                <div className="grid grid-cols-3 gap-1.5">
                  {CHART_TYPES.map((ct) => {
                    const Icon = ct.icon;
                    const isActive = chartType === ct.id;
                    return (
                      <button
                        key={ct.id}
                        onClick={() => setChartType(ct.id)}
                        className={`flex flex-col items-center justify-center p-2 rounded-xl border text-center transition ${
                          isActive
                            ? "bg-white/10 border-white/20 text-white"
                            : "bg-[#0e1017] border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                        }`}
                        title={ct.desc}
                      >
                        <Icon className="h-4 w-4 mb-1" />
                        <span className="text-[10px] font-medium leading-tight">{ct.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Section: Dimension (X-Axis / Group By) */}
              <div className="space-y-1.5">
                <label htmlFor="dimension-select" className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  Dimension (Group By / X-Axis)
                </label>
                <select
                  id="dimension-select"
                  aria-label="Dimension (Group By / X-Axis)"
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

              {/* Section: Temporal Bucketing (Time Grain) */}
              {isTemporalDimension && (
                <div className="space-y-1.5 p-2.5 rounded-xl bg-sky-950/20 border border-sky-800/40">
                  <div className="flex items-center gap-1.5 text-sky-400 text-xs font-semibold">
                    <Calendar className="h-3.5 w-3.5" />
                    <span>Time Grain (Date Truncation)</span>
                  </div>
                  <select
                    id="time-grain-select"
                    aria-label="Time Grain"
                    value={timeGrain}
                    onChange={(e) => setTimeGrain(e.target.value)}
                    className="w-full bg-[#0e1017] border border-sky-800/60 text-xs rounded-lg px-2.5 py-1.5 text-sky-200 focus:outline-none font-mono"
                  >
                    {TIME_GRAINS.map((tg) => (
                      <option key={tg.id} value={tg.id}>
                        {tg.label}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Section: Metric (Y-Axis / Measure) */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  Metric & Aggregation (Y-Axis)
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <select
                    id="aggregate-select"
                    aria-label="Aggregation Function"
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

                  <select
                    id="metric-select"
                    aria-label="Metric Column"
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
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label htmlFor="secondary-dimension-select" className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                    Secondary Breakdown
                  </label>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {chartType === "echarts_heatmap" ? "Required for Heatmap" : "Optional"}
                  </span>
                </div>
                <select
                  id="secondary-dimension-select"
                  aria-label="Secondary Breakdown"
                  value={secondaryDimension}
                  onChange={(e) => setSecondaryDimension(e.target.value)}
                  className={`w-full bg-[#0e1017] border text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none font-mono ${
                    chartType === "echarts_heatmap" && !secondaryDimension
                      ? "border-amber-500/50"
                      : "border-slate-800"
                  }`}
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

              {/* Section: Ad-Hoc Filters */}
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
                    <Filter className="h-3 w-3 text-sky-400" />
                    <span>Ad-Hoc Filters ({filters.length})</span>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={addFilter}
                    className="h-6 px-2 text-[10px] font-medium text-sky-400 hover:text-sky-300 hover:bg-sky-500/10 rounded-md gap-1"
                  >
                    <Plus className="h-3 w-3" />
                    <span>Add</span>
                  </Button>
                </div>

                {filters.length > 0 && (
                  <div className="space-y-2">
                    {filters.map((f) => (
                      <div
                        key={f.id}
                        className="p-2 rounded-xl bg-[#0e1017] border border-slate-800 space-y-1.5 text-xs"
                      >
                        <div className="flex items-center gap-1.5">
                          {/* Filter Column */}
                          <select
                            value={f.column}
                            onChange={(e) => updateFilter(f.id, "column", e.target.value)}
                            className="bg-[#121215] border border-slate-700 rounded-md px-1.5 py-1 text-slate-200 text-[11px] font-mono flex-1 truncate"
                          >
                            {schemaColumns.map((col) => (
                              <option key={col.name} value={col.name}>
                                {col.name}
                              </option>
                            ))}
                          </select>

                          {/* Filter Operator */}
                          <select
                            value={f.operator}
                            onChange={(e) => updateFilter(f.id, "operator", e.target.value)}
                            className="bg-[#121215] border border-slate-700 rounded-md px-1.5 py-1 text-slate-200 text-[11px] font-mono"
                          >
                            {FILTER_OPERATORS.map((op) => (
                              <option key={op.id} value={op.id}>
                                {op.label}
                              </option>
                            ))}
                          </select>

                          {/* Delete button */}
                          <button
                            type="button"
                            onClick={() => removeFilter(f.id)}
                            className="p-1 text-slate-400 hover:text-rose-400 transition"
                            title="Remove filter"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>

                        {/* Value Input (unless is_null / is_not_null) */}
                        {!f.operator.includes("null") && (
                          <input
                            type="text"
                            placeholder="Comparison value..."
                            value={f.value}
                            onChange={(e) => updateFilter(f.id, "value", e.target.value)}
                            className="w-full bg-[#121215] border border-slate-700 rounded-md px-2 py-1 text-slate-200 text-[11px] font-mono focus:outline-none focus:border-white/30"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Section: Limits & Sorting */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="space-y-1">
                  <label htmlFor="row-limit-select" className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                    Row Limit
                  </label>
                  <select
                    id="row-limit-select"
                    aria-label="Row Limit"
                    value={rowLimit}
                    onChange={(e) => setRowLimit(Number(e.target.value))}
                    className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none font-mono"
                  >
                    <option value={5}>Top 5</option>
                    <option value={10}>Top 10</option>
                    <option value={15}>Top 15</option>
                    <option value={25}>Top 25</option>
                    <option value={50}>Top 50</option>
                    <option value={100}>Top 100</option>
                    <option value={250}>Top 250</option>
                    <option value={500}>Top 500</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label htmlFor="sort-order-select" className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                    Sort Order
                  </label>
                  <select
                    id="sort-order-select"
                    aria-label="Sort Order"
                    value={sortOrder}
                    onChange={(e) => setSortOrder(e.target.value)}
                    className="w-full bg-[#0e1017] border border-slate-800 text-xs rounded-xl px-3 py-2 text-slate-200 focus:outline-none font-mono"
                  >
                    <option value="desc">High to Low (Desc)</option>
                    <option value="asc">Low to High (Asc)</option>
                  </select>
                </div>
              </div>

              {/* Section: Color Palette */}
              <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  Color Palette
                </span>
                <div className="space-y-1">
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

              {/* Mobile Apply CTA */}
              <div className="lg:hidden pt-3 border-t border-slate-800/80">
                <Button
                  type="button"
                  onClick={() => setMobileTab("chart")}
                  className="w-full bg-white hover:bg-zinc-200 text-black font-semibold text-xs h-9 justify-center gap-2 cursor-pointer shadow-xs"
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>Apply & View Visual Chart</span>
                </Button>
              </div>
            </div>

            {/* ──── RIGHT CANVAS: Live Visualization & KPIs (8 cols) ──── */}
            <div className={`lg:col-span-8 p-3.5 sm:p-6 flex flex-col justify-between space-y-4 sm:space-y-6 bg-[#07080a] ${
              mobileTab === "chart" ? "flex" : "hidden lg:flex"
            }`}>
              
              {/* Top: Superset BigNumber KPI Cards with Compact Number Formatting */}
              {queryResult?.kpis && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                      Total Records
                    </span>
                    <span className="text-xl sm:text-2xl font-bold text-white mt-1 block">
                      {formatCompactNumber(queryResult.kpis.total_rows)}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block truncate">
                      {queryResult.kpis.metric_label || "Metric Value"}
                    </span>
                    <span className="text-xl sm:text-2xl font-bold text-emerald-400 mt-1 block">
                      {queryResult.kpis.mean !== null && queryResult.kpis.mean !== undefined
                        ? formatCompactNumber(queryResult.kpis.mean)
                        : queryResult.kpis.sum !== null && queryResult.kpis.sum !== undefined
                        ? formatCompactNumber(queryResult.kpis.sum)
                        : "—"}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                      Peak Group
                    </span>
                    <span className="text-xl sm:text-2xl font-bold text-emerald-400 mt-1 block truncate">
                      {queryResult.records?.[0]?.[dimension] ?? "—"}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                      Total Groups
                    </span>
                    <span className="text-xl sm:text-2xl font-bold text-slate-200 mt-1 block">
                      {formatCompactNumber(queryResult.total_groups)}
                    </span>
                  </div>
                </div>
              )}

              {/* Main Chart or Table Canvas */}
              <div className="flex-1 min-h-[380px] sm:min-h-[460px] rounded-xl border border-slate-800 bg-[#0c0d12] p-3 sm:p-4 flex flex-col relative">
                
                {/* View Switcher: Chart vs Raw Data */}
                <div className="flex flex-col xs:flex-row xs:items-center justify-between gap-2 pb-3 border-b border-slate-800/80 mb-2">
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

                  <div className="text-xs text-slate-400 font-mono truncate">
                    {queryResult?.metric_label} by {dimension}
                    {timeGrain && <span className="text-sky-400 ml-1">({timeGrain})</span>}
                  </div>
                </div>

                {/* Loading Overlay */}
                {executing && (
                  <div className="absolute inset-0 bg-[#07080a]/80 backdrop-blur-xs flex flex-col items-center justify-center z-20 gap-2">
                    <Loader2 className="h-6 w-6 animate-spin text-zinc-300" />
                    <span className="text-xs font-mono text-slate-400">Computing visual analytics...</span>
                  </div>
                )}

                {/* View Mode 1: Interactive ECharts Canvas */}
                {viewMode === "chart" && chartType !== "table" && queryResult?.echarts_option && (
                  <div className="flex-1 w-full min-h-[320px] sm:min-h-[400px]">
                    <ReactECharts
                      ref={echartsRef}
                      option={queryResult.echarts_option}
                      style={{ height: "100%", minHeight: "340px", width: "100%" }}
                      theme="dark"
                      notMerge={true}
                      lazyUpdate={true}
                    />
                  </div>
                )}

                {/* View Mode 2: Aggregated Data Table */}
                {(viewMode === "data" || chartType === "table") && queryResult?.records && (
                  <div className="flex-1 overflow-x-auto min-h-[320px] sm:min-h-[400px] -mx-1 px-1 sm:mx-0 sm:px-0 scrollbar-thin">
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
        </>
      )}

    </div>
  );
};
