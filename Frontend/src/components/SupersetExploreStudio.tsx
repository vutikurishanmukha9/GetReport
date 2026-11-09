import React, { useState, useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
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
  Calendar,
  ScatterChart,
  Box,
  ChevronDown,
  Loader2,
  Copy,
  Check,
  Activity,
  Award,
} from "lucide-react";
import { Button, AnimatedDeleteButton, ShimmerActionButton } from "@/components/ui/button";
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
  { id: "echarts_combo", label: "Dual Combo", icon: TrendingUp, desc: "Bar (Left Y) + Line (Right Y) Dual Axis" },
  { id: "echarts_area", label: "Area Fill", icon: Layers, desc: "Cumulative series volume" },
  { id: "echarts_pie", label: "Pie / Donut", icon: PieChart, desc: "Part-to-whole proportions" },
  { id: "echarts_funnel", label: "Funnel Flow", icon: Filter, desc: "Stage conversion step drop-offs" },
  { id: "echarts_radar", label: "Radar Spider", icon: Activity, desc: "Multi-axis indicator radar web" },
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

const formatCompactNumber = (rawVal: unknown): string => {
  if (rawVal === null || rawVal === undefined) return "—";
  const num = typeof rawVal === "number" ? rawVal : Number(rawVal);
  if (isNaN(num)) return String(rawVal);
  if (Math.abs(num) >= 1_000_000_000) return (num / 1_000_000_000).toFixed(2) + "B";
  if (Math.abs(num) >= 1_000_000) return (num / 1_000_000).toFixed(2) + "M";
  if (Math.abs(num) >= 1_000) return (num / 1_000).toFixed(1) + "K";
  return Number.isInteger(num) ? num.toLocaleString() : num.toFixed(2);
};

// ─────────────────────────────────────────────────────────────────────────────
// NATIVE APACHE ECHARTS CANVAS (Direct DOM Mount & Zero-Latency ResizeObserver)
// ─────────────────────────────────────────────────────────────────────────────

interface NativeEChartsViewProps {
  option: Record<string, any>;
  onChartClick?: (params: any) => void;
  onInstanceReady?: (instance: echarts.ECharts) => void;
  height?: string | number;
  className?: string;
}

const sanitizeEChartsOption = (opt: any): any => {
  if (!opt || typeof opt !== 'object') return opt;
  const clone = { ...opt };

  const sanitizeAxis = (axis: any) => {
    if (!axis || typeof axis !== 'object') return axis;
    const clean = { ...axis };
    if (clean.type === 'linear') {
      clean.type = 'value';
    }
    return clean;
  };

  if (clone.yAxis) {
    if (Array.isArray(clone.yAxis)) {
      clone.yAxis = clone.yAxis.map(sanitizeAxis);
    } else {
      clone.yAxis = sanitizeAxis(clone.yAxis);
    }
  }

  if (clone.xAxis) {
    if (Array.isArray(clone.xAxis)) {
      clone.xAxis = clone.xAxis.map(sanitizeAxis);
    } else {
      clone.xAxis = sanitizeAxis(clone.xAxis);
    }
  }

  // Ensure tooltips are safely confined within canvas across all chart types
  if (clone.tooltip) {
    clone.tooltip = {
      confine: true,
      ...clone.tooltip,
    };
  }

  // Ensure legend is always scrollable and positioned at top to avoid bar/axis collision
  if (clone.legend && clone.legend.show !== false) {
    clone.legend = {
      orient: "horizontal",
      left: "center",
      right: "4%",
      pageButtonPosition: "end",
      pageIconColor: "#38bdf8",
      pageIconInactiveColor: "#475569",
      pageIconSize: 10,
      pageTextStyle: { color: "#94a3b8", fontSize: 10 },
      textStyle: { color: "#cbd5e1", fontSize: 11 },
      ...clone.legend,
      type: "scroll",
      top: "0%",
    };
    delete clone.legend.bottom;
    if (clone.grid) {
      clone.grid = {
        ...clone.grid,
        top: Math.max(typeof clone.grid.top === "number" ? clone.grid.top : parseInt(clone.grid.top || "8", 10), 45),
      };
    }
  }

  return clone;
};

const NativeEChartsView: React.FC<NativeEChartsViewProps> = ({
  option,
  onChartClick,
  onInstanceReady,
  height = "480px",
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const onChartClickRef = useRef(onChartClick);
  onChartClickRef.current = onChartClick;
  const onInstanceReadyRef = useRef(onInstanceReady);
  onInstanceReadyRef.current = onInstanceReady;

  useEffect(() => {
    if (!containerRef.current) return;

    let instance = echarts.getInstanceByDom(containerRef.current);
    if (!instance) {
      instance = echarts.init(containerRef.current, undefined, {
        renderer: "canvas",
      });
      chartInstanceRef.current = instance;
    }

    if (onInstanceReadyRef.current) {
      onInstanceReadyRef.current(instance);
    }

    const clickHandler = (params: any) => {
      if (onChartClickRef.current) {
        onChartClickRef.current(params);
      }
    };
    instance.off("click");
    instance.on("click", clickHandler);

    const resizeChart = () => {
      if (!instance || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        instance.resize({ width: rect.width, height: rect.height });
      } else {
        instance.resize();
      }
    };

    if (option && Object.keys(option).length > 0) {
      const cleanOption = sanitizeEChartsOption(option);
      try {
        // Clear previous series & coordinate systems when switching chart types to prevent layout collision
        instance.clear();
        instance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
        resizeChart();
        requestAnimationFrame(resizeChart);
        setTimeout(resizeChart, 60);
        setTimeout(resizeChart, 250);
      } catch (err) {
        console.error('ECharts setOption error, recreating instance:', err);
        try {
          instance.dispose();
          const newInstance = echarts.init(containerRef.current, undefined, { renderer: 'canvas' });
          chartInstanceRef.current = newInstance;
          if (onInstanceReadyRef.current) {
            onInstanceReadyRef.current(newInstance);
          }
          newInstance.on('click', clickHandler);
          newInstance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
          newInstance.resize();
        } catch (recoverErr) {
          console.error('Failed to recover ECharts instance:', recoverErr);
        }
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      resizeChart();
    });
    resizeObserver.observe(containerRef.current);

    window.addEventListener("resize", resizeChart);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", resizeChart);
      instance?.off("click", clickHandler);
    };
  }, [option]);

  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height, minHeight: "420px" }}
      className={`w-full relative ${className}`}
    />
  );
};

export const SupersetExploreStudio: React.FC<SupersetExploreStudioProps> = ({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  allowDatasetSwitching = false,
}) => {
  const { toast } = useToast();
  const nativeChartInstanceRef = useRef<echarts.ECharts | null>(null);


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
  const [barMode, setBarMode] = useState<"grouped" | "stacked">("grouped");
  const [dimension, setDimension] = useState<string>("");
  const [secondaryDimension, setSecondaryDimension] = useState<string>("");
  const [metric, setMetric] = useState<string>("");
  const [aggregate, setAggregate] = useState<string>("COUNT");
  const [secondaryMetric, setSecondaryMetric] = useState<string>("");
  const [secondaryAggregate, setSecondaryAggregate] = useState<string>("AVG");
  const [yAxisScale, setYAxisScale] = useState<string>("linear");
  const [timeGrain, setTimeGrain] = useState<string>("");
  const [filters, setFilters] = useState<FilterItem[]>([]);
  const [showFilterBuilder, setShowFilterBuilder] = useState<boolean>(false);
  const [rowLimit, setRowLimit] = useState<number>(25);
  const [sortOrder, setSortOrder] = useState<string>("desc");
  const [colorPalette, setColorPalette] = useState<string>("supersetColors");
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [copiedChart, setCopiedChart] = useState<boolean>(false);

  // Query Results State
  const [executing, setExecuting] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<"chart" | "data">("chart");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [mobileTab, setMobileTab] = useState<"chart" | "controls">("chart");

  // Dynamic Chart Auto-Resize
  useEffect(() => {
    const handleResize = () => {
      if (nativeChartInstanceRef.current) {
        try {
          nativeChartInstanceRef.current.resize();
        } catch {
          // ignore
        }
      }
    };
    window.addEventListener("resize", handleResize);
    const timer = setTimeout(handleResize, 100);
    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, [mobileTab, isFullscreen, viewMode]);

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
        if (!res) return;
        setSchemaColumns(Array.isArray(res.columns) ? res.columns : []);
        setNumericCols(Array.isArray(res.numeric_columns) ? res.numeric_columns : []);
        setCategoricalCols(Array.isArray(res.categorical_columns) ? res.categorical_columns : []);
        setTemporalCols(Array.isArray(res.temporal_columns) ? res.temporal_columns : []);
        setRecommendations(Array.isArray(res.recommendations) ? res.recommendations : []);

        // Pick intelligent default dimension & metric
        const defaultDim = res.categorical_columns && res.categorical_columns.length > 0
          ? res.categorical_columns[0]
          : res.columns?.[0]?.name || "";
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
    if (!selectedTaskId || !dimension) return;

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
        bar_mode: barMode,
        metric: metric || undefined,
        aggregate: aggregate,
        secondary_metric: secondaryMetric || undefined,
        secondary_aggregate: secondaryAggregate || undefined,
        y_axis_scale: yAxisScale || "linear",
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
    if (dimension && selectedTaskId) {
      executeQuery();
    }
  }, [
    selectedTaskId,
    chartType,
    barMode,
    dimension,
    secondaryDimension,
    metric,
    aggregate,
    secondaryMetric,
    secondaryAggregate,
    yAxisScale,
    timeGrain,
    filters,
    rowLimit,
    sortOrder,
    colorPalette,
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
  // Apply 1-click smart recommendation
  const applyRecommendation = (rec: any) => {
    if (rec.chart_type) setChartType(rec.chart_type);
    if (rec.dimension) setDimension(rec.dimension);
    if (rec.secondary_dimension !== undefined) setSecondaryDimension(rec.secondary_dimension || "");
    if (rec.metric !== undefined) setMetric(rec.metric || "");
    if (rec.aggregate) setAggregate(rec.aggregate);
    if (rec.secondary_metric !== undefined) setSecondaryMetric(rec.secondary_metric || "");
    if (rec.secondary_aggregate) setSecondaryAggregate(rec.secondary_aggregate);
    if (rec.time_grain !== undefined) setTimeGrain(rec.time_grain || "");
    if (rec.row_limit) setRowLimit(rec.row_limit);

    toast({
      title: "Preset Applied",
      description: `Loaded: ${rec.title}`
    });
  };

  // Copy Chart Image to Clipboard
  const handleCopyChartImage = async () => {
    if (!nativeChartInstanceRef.current) return;
    try {
      const dataUrl = nativeChartInstanceRef.current.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: "#07080a",
      });
      if (dataUrl) {
        const blob = await (await fetch(dataUrl)).blob();
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob })
        ]);
        setCopiedChart(true);
        setTimeout(() => setCopiedChart(false), 2000);
        toast({ title: "Chart Copied", description: "PNG image copied to clipboard." });
      }
    } catch (err) {
      console.warn("Clipboard write failed, downloading instead:", err);
      handleExportChart();
    }
  };

  // Click-to-filter on chart element
  const handleChartClick = React.useCallback((params: any) => {
    if (params && params.name && dimension) {
      const clickedVal = String(params.name);
      setFilters(prev => {
        const withoutDim = prev.filter(f => f.column !== dimension);
        return [...withoutDim, { id: Math.random().toString(), column: dimension, operator: "eq", value: clickedVal }];
      });
      toast({
        title: "Filter Applied",
        description: `Filtered by ${dimension} = "${clickedVal}"`
      });
    }
  }, [dimension, toast]);

  const handleInstanceReady = React.useCallback((inst: echarts.ECharts) => {
    nativeChartInstanceRef.current = inst;
  }, []);

  const handleExportChart = () => {
    if (!nativeChartInstanceRef.current) return;
    try {
      const chartInstance = nativeChartInstanceRef.current;
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
  const selectedColMeta = schemaColumns?.find(c => c.name === dimension);
  const isTemporalDimension = selectedColMeta?.is_temporal || (Array.isArray(temporalCols) && temporalCols.includes(dimension));

  return (
    <div className={`w-full rounded-2xl border border-slate-800 bg-[#07080a] text-slate-100 overflow-hidden ${
      isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#07080a] overflow-y-auto" : ""
    }`}>
      
      {/* ─── Top Studio Header Bar ─── */}
      <div className="px-3.5 sm:px-5 py-3 border-b border-slate-800 bg-card dark:bg-[#0c0d12] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        
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
                        {d.filename} ({typeof d.rows === "number" ? d.rows.toLocaleString() : d.rows || 0} rows)
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="flex items-center gap-2 font-mono">
                  <Database className="h-3.5 w-3.5 text-slate-400" />
                  <span className="text-slate-200 font-medium">{displayName}</span>
                  {activeDatasetMeta?.rows !== undefined && activeDatasetMeta?.rows !== null && (
                    <span>• {typeof activeDatasetMeta.rows === "number" ? activeDatasetMeta.rows.toLocaleString() : activeDatasetMeta.rows} rows</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 self-end sm:self-auto">
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

              {/* Copy Chart Image Action */}
              <Button
                size="sm"
                variant="outline"
                onClick={handleCopyChartImage}
                disabled={!queryResult}
                className="h-8 rounded-lg border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 gap-1.5"
                title="Copy chart image to clipboard"
              >
                {copiedChart ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                <span className="hidden sm:inline">{copiedChart ? "Copied" : "Copy"}</span>
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

      {/* ─── Visual Analytics Studio Canvas ─── */}
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
            <div className={`lg:col-span-4 p-4 sm:p-5 border-b lg:border-b-0 lg:border-r border-slate-800 bg-card dark:bg-[#0c0d12] space-y-4 sm:space-y-5 ${
              mobileTab === "controls" ? "block" : "hidden lg:block"
            }`}>
              
              {/* Section: Suggested Visual Explorations */}
              {recommendations.length > 0 && (
                <div className="space-y-2 pb-2 border-b border-slate-800/80">
                  <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400">
                    <TrendingUp className="h-3 w-3 text-emerald-400" />
                    <span>Suggested Explorations</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {recommendations.slice(0, 4).map((rec: any) => (
                      <button
                        key={rec.id}
                        type="button"
                        onClick={() => applyRecommendation(rec)}
                        className="text-left text-[11px] px-2 py-1 rounded-lg border border-white/10 bg-[#0e1017] hover:bg-white/[0.08] hover:border-emerald-500/40 text-zinc-300 hover:text-white transition group flex items-center gap-1.5 cursor-pointer max-w-full font-sans"
                        title={rec.description}
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                        <span className="truncate max-w-[130px] font-medium">{rec.title}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Section: Chart Type Picker */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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
                        className={`flex flex-col items-center justify-center p-2 rounded-xl border text-center transition font-sans ${
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
                <label htmlFor="dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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

              {/* Section: Secondary Metric for Combo Chart */}
              {chartType === "echarts_combo" && (
                <div className="space-y-2 p-3 rounded-xl bg-purple-950/20 border border-purple-800/40">
                  <div className="flex items-center gap-1.5 text-purple-400 text-xs font-semibold">
                    <TrendingUp className="h-3.5 w-3.5" />
                    <span>Secondary Metric (Line on Right Axis)</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <select
                      id="sec-metric-select"
                      aria-label="Secondary Metric"
                      value={secondaryMetric}
                      onChange={(e) => setSecondaryMetric(e.target.value)}
                      className="bg-[#0e1017] border border-purple-800/60 text-xs rounded-lg px-2.5 py-1.5 text-purple-200 font-mono"
                    >
                      <option value="">None</option>
                      {numericCols.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                    <select
                      id="sec-agg-select"
                      aria-label="Secondary Aggregate"
                      value={secondaryAggregate}
                      onChange={(e) => setSecondaryAggregate(e.target.value)}
                      className="bg-[#0e1017] border border-purple-800/60 text-xs rounded-lg px-2.5 py-1.5 text-purple-200 font-mono"
                    >
                      {AGGREGATIONS.map((a) => (
                        <option key={a.id} value={a.id}>{a.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {/* Section: Metric (Y-Axis / Measure) */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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
                  <label htmlFor="secondary-dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
                    Secondary Breakdown
                  </label>
                  <span className="text-[10px] text-zinc-400 font-sans font-medium">
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
                {secondaryDimension && chartType === "echarts_bar" && (
                  <div className="flex items-center justify-between pt-1.5 px-0.5">
                    <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400">
                      Bar Mode
                    </span>
                    <div className="flex items-center gap-1 bg-[#090b10] p-0.5 rounded-lg border border-slate-800">
                      <button
                        type="button"
                        onClick={() => setBarMode("grouped")}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                          barMode === "grouped"
                            ? "bg-primary text-primary-foreground font-bold shadow-xs"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        Grouped
                      </button>
                      <button
                        type="button"
                        onClick={() => setBarMode("stacked")}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                          barMode === "stacked"
                            ? "bg-primary text-primary-foreground font-bold shadow-xs"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        Stacked
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Section: Ad-Hoc Filters */}
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400">
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
                          <AnimatedDeleteButton
                            size="icon"
                            variant="ghost"
                            className="h-7 w-7 rounded-lg text-slate-400 hover:text-rose-400 shrink-0"
                            onDelete={() => removeFilter(f.id)}
                            title="Remove filter"
                          />
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
                  <label htmlFor="row-limit-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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
                  <label htmlFor="sort-order-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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
                <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
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
                <ShimmerActionButton
                  onClick={() => setMobileTab("chart")}
                  className="w-full h-10 text-xs justify-center"
                  icon={<BarChart3 className="h-3.5 w-3.5 text-zinc-950" />}
                >
                  <span>Apply & View Visual Chart</span>
                </ShimmerActionButton>
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
                    <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
                      Total Records
                    </span>
                    <span className="text-xl sm:text-2xl font-sans font-bold text-white mt-1 block tabular-nums">
                      {formatCompactNumber(queryResult.kpis.total_rows)}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block truncate">
                      {queryResult.kpis.metric_label || "Metric Value"}
                    </span>
                    <span className="text-xl sm:text-2xl font-sans font-bold text-emerald-400 mt-1 block tabular-nums">
                      {queryResult.kpis.mean !== null && queryResult.kpis.mean !== undefined
                        ? formatCompactNumber(queryResult.kpis.mean)
                        : queryResult.kpis.sum !== null && queryResult.kpis.sum !== undefined
                        ? formatCompactNumber(queryResult.kpis.sum)
                        : "—"}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
                      Peak Group
                    </span>
                    <span className="text-xl sm:text-2xl font-sans font-bold text-emerald-400 mt-1 block truncate">
                      {queryResult.records?.[0]?.[dimension] ?? "—"}
                    </span>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-[#121215] p-3.5">
                    <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-zinc-400 block">
                      Total Groups
                    </span>
                    <span className="text-xl sm:text-2xl font-sans font-bold text-slate-200 mt-1 block tabular-nums">
                      {formatCompactNumber(queryResult.total_groups)}
                    </span>
                  </div>
                </div>
              )}

              {/* AI/Statistical Smart Insights Strip */}
              {queryResult?.smart_insights && queryResult.smart_insights.length > 0 && (
                <div className="rounded-xl bg-gradient-to-r from-emerald-500/[0.08] via-sky-500/[0.05] to-purple-500/[0.05] border border-white/10 p-3 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-zinc-200">
                    <div className="flex items-center gap-1.5">
                      <Activity className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                      <span>Automated Data Insights</span>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400">Polars Vector Engine</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {queryResult.smart_insights.map((ins: any) => (
                      <div key={ins.id} className="flex items-start gap-2 p-2 rounded-lg bg-black/40 border border-white/5 text-xs">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold shrink-0 ${
                          ins.severity === 'positive' ? 'bg-emerald-500/20 text-emerald-300' :
                          ins.severity === 'warning' ? 'bg-amber-500/20 text-amber-300' :
                          'bg-sky-500/20 text-sky-300'
                        }`}>
                          {ins.badge || 'Insight'}
                        </span>
                        <p className="text-zinc-300 text-[11px] leading-relaxed">
                          <strong className="text-white mr-1">{ins.title}:</strong>
                          {ins.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Main Chart or Table Canvas */}
              <div className="flex-1 min-h-[380px] sm:min-h-[460px] rounded-xl border border-slate-800 bg-card dark:bg-[#0c0d12] p-3 sm:p-4 flex flex-col relative">
                
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

                {/* View Mode 1: Interactive ECharts Canvas (Native Direct Render) */}
                {viewMode === "chart" && chartType !== "table" && queryResult?.echarts_option && (
                  <div className="flex-1 w-full min-h-[420px]">
                    <NativeEChartsView
                      option={queryResult.echarts_option}
                      onChartClick={handleChartClick}
                      onInstanceReady={handleInstanceReady}
                      height={isFullscreen ? "72vh" : "480px"}
                    />
                  </div>
                )}

                {/* Empty State when no data records or query resulted in 0 rows */}
                {viewMode === "chart" && chartType !== "table" && (!queryResult || !queryResult.echarts_option || queryResult.records?.length === 0) && !executing && (
                  <div className="flex-1 w-full min-h-[420px] flex flex-col items-center justify-center p-8 text-center border border-dashed border-slate-800 rounded-xl bg-black/20">
                    <div className="h-12 w-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 mb-3">
                      <BarChart3 className="h-6 w-6 text-slate-500" />
                    </div>
                    <h3 className="text-sm font-semibold text-white">No Visual Data Available</h3>
                    <p className="text-xs text-slate-400 max-w-sm mt-1">
                      No data records matched the selected dimension and metric filters. Try adjusting your dimension, metric, or filters on the left panel.
                    </p>
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

    </div>
  );
};
