import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as echarts from "echarts";
import {
  BarChart3,
  AlignLeft,
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
  Calendar,
  Loader2,
  Copy,
  Check,
  Activity,
  Sparkles,
  AlertTriangle,
  Search,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Trophy,
  Hash,
} from "lucide-react";
import { Button, AnimatedDeleteButton, ShimmerActionButton } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ExploreTelemetryDeck } from "./ExploreTelemetryDeck";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { useTheme } from "@/hooks/useTheme";

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
  initialInfo?: {
    columns?: string[];
    dtypes?: Record<string, string>;
    rows?: number;
    preview?: any[];
    missing_values?: Record<string, { count: number; percentage: number }>;
  };
}

const CHART_TYPES = [
  { id: "echarts_bar", label: "Column (Vert)", icon: BarChart3, desc: "Vertical categorical comparisons & counts" },
  { id: "echarts_horizontal_bar", label: "Bar (Horiz)", icon: AlignLeft, desc: "Horizontal ranking & clean labels" },
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

const sanitizeEChartsOption = (opt: any, isDark: boolean = true): any => {
  if (!opt || typeof opt !== 'object') return opt;
  const clone = JSON.parse(JSON.stringify(opt));

  // High-contrast, theme-calibrated typography and hairline colors
  // Light Mode: Pure rich black/charcoal (#18181b / #292524) for 100% legibility on light canvas
  // Dark Mode: Crisp bright zinc (#f4f4f5 / #d4d4d8) for 100% legibility on dark canvas
  const primaryTextColor = isDark ? "#f4f4f5" : "#18181b";
  const secondaryTextColor = isDark ? "#a1a1aa" : "#44403c";
  const axisLabelColor = isDark ? "#d4d4d8" : "#292524";
  const axisLineColor = isDark ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.15)";
  const splitLineColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.08)";
  const labelHaloColor = isDark ? "rgba(9, 13, 22, 0.95)" : "rgba(255, 255, 255, 0.95)";

  const sanitizeAxis = (axis: any) => {
    if (!axis || typeof axis !== 'object') return axis;
    const clean = { ...axis };
    if (clean.type === 'linear') {
      clean.type = 'value';
    }
    clean.axisLabel = {
      fontSize: 11,
      fontFamily: "inherit",
      fontWeight: 500,
      ...(clean.axisLabel || {}),
      color: axisLabelColor, // Always enforce high-contrast theme color
    };
    if (clean.name) {
      clean.nameTextStyle = {
        fontSize: 11,
        fontFamily: "inherit",
        fontWeight: 600,
        ...(clean.nameTextStyle || {}),
        color: primaryTextColor,
      };
    }
    clean.axisLine = {
      ...(clean.axisLine || {}),
      lineStyle: {
        ...(clean.axisLine?.lineStyle || {}),
        color: axisLineColor,
      },
    };
    if (clean.splitLine && clean.splitLine.show !== false) {
      clean.splitLine = {
        ...(clean.splitLine || {}),
        lineStyle: {
          type: "dashed",
          ...(clean.splitLine?.lineStyle || {}),
          color: splitLineColor,
        },
      };
    }
    return clean;
  };

  if (clone.yAxis) {
    clone.yAxis = Array.isArray(clone.yAxis) ? clone.yAxis.map(sanitizeAxis) : sanitizeAxis(clone.yAxis);
  }

  if (clone.xAxis) {
    clone.xAxis = Array.isArray(clone.xAxis) ? clone.xAxis.map(sanitizeAxis) : sanitizeAxis(clone.xAxis);
  }

  // Ensure tooltips are safely confined within canvas across all chart types and styled to theme
  const origTooltip = clone.tooltip || {};
  clone.tooltip = {
    confine: true,
    ...origTooltip,
    backgroundColor: isDark ? "#18181b" : "#ffffff",
    borderColor: isDark ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.12)",
    borderWidth: 1,
    padding: [8, 12],
    textStyle: {
      fontSize: 12,
      fontWeight: 500,
      fontFamily: "inherit",
      ...(origTooltip.textStyle || {}),
      color: isDark ? "#f4f4f5" : "#18181b", // ALWAYS enforce high-contrast theme color
    },
    extraCssText: isDark
      ? "box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5); border-radius: 8px; backdrop-filter: blur(8px);"
      : "box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.12); border-radius: 8px;",
  };

  // Ensure legend is always scrollable, positioned at top, and text is 100% visible
  if (clone.legend && clone.legend.show !== false) {
    const origLegend = clone.legend;
    clone.legend = {
      ...origLegend,
      orient: "horizontal",
      left: "center",
      right: "4%",
      top: "0%",
      type: "scroll",
      pageButtonPosition: "end",
      pageIconColor: isDark ? "#38bdf8" : "#800000",
      pageIconInactiveColor: isDark ? "#52525b" : "#a8a29e",
      pageIconSize: 11,
      pageTextStyle: {
        fontSize: 10,
        fontWeight: 600,
        ...(origLegend.pageTextStyle || {}),
        color: secondaryTextColor, // Always enforce visible page text (e.g. 1/7)
      },
      textStyle: {
        fontSize: 11,
        fontWeight: 500,
        fontFamily: "inherit",
        ...(origLegend.textStyle || {}),
        color: primaryTextColor, // Always enforce visible legend item text
      },
    };
    delete clone.legend.bottom;
    if (clone.grid) {
      clone.grid = {
        ...clone.grid,
        top: Math.max(typeof clone.grid.top === "number" ? clone.grid.top : parseInt(clone.grid.top || "8", 10), 45),
      };
    }
  }

  // Ensure series labels (pie slice callouts, bar values, line markers) are high-contrast with halo outline
  if (Array.isArray(clone.series)) {
    clone.series = clone.series.map((s: any) => {
      if (!s || typeof s !== 'object') return s;
      const cleanSeries = { ...s };

      // High-visibility label formatting with 3px halo outline
      if (cleanSeries.label) {
        cleanSeries.label = {
          fontSize: 11.5,
          fontFamily: "inherit",
          fontWeight: 600,
          ...cleanSeries.label,
          color: primaryTextColor, // High-contrast solid color
          textBorderColor: labelHaloColor,
          textBorderWidth: 3, // Halo outline guarantees crisp contrast
        };
      }

      // Pie & Donut chart callout lines and slice boundaries
      if (cleanSeries.type === "pie") {
        // Defensive aggregation if pie has more than 7 slices to prevent unreadable bicycle-wheel visual slop
        if (Array.isArray(cleanSeries.data) && cleanSeries.data.length > 7) {
          const sorted = [...cleanSeries.data].sort((a: any, b: any) => (Number(b.value) || 0) - (Number(a.value) || 0));
          const top6 = sorted.slice(0, 6);
          const restVal = sorted.slice(6).reduce((acc: number, cur: any) => acc + (Number(cur.value) || 0), 0);
          const restCount = sorted.length - 6;
          cleanSeries.data = [
            ...top6,
            { name: `Other (${restCount} items)`, value: Math.round(restVal * 100) / 100 }
          ];
        }

        cleanSeries.radius = ["50%", "72%"];
        cleanSeries.avoidLabelOverlap = true;

        cleanSeries.labelLine = {
          show: true,
          length: 15,
          length2: 12,
          smooth: 0.2,
          ...(cleanSeries.labelLine || {}),
          lineStyle: {
            width: 1.2,
            ...(cleanSeries.labelLine?.lineStyle || {}),
            color: isDark ? "rgba(255, 255, 255, 0.45)" : "rgba(15, 23, 42, 0.45)",
          },
        };
        cleanSeries.itemStyle = {
          ...(cleanSeries.itemStyle || {}),
          borderRadius: 3, // Sleek, non-slop minimal radius overriding any clumsy 6px sausage corners
          borderColor: isDark ? "#07080a" : "#ffffff",
          borderWidth: 2,
        };
      }

      // Bar charts: ensure clean crisp borders without dark outlines
      if (cleanSeries.type === "bar") {
        cleanSeries.itemStyle = {
          ...(cleanSeries.itemStyle || {}),
          borderColor: "transparent",
          borderWidth: 0,
        };
      }

      return cleanSeries;
    });
  }

  // Ensure title text is high-contrast
  if (clone.title) {
    clone.title = {
      ...clone.title,
      textStyle: {
        fontSize: 13,
        fontWeight: 600,
        ...(clone.title.textStyle || {}),
        color: primaryTextColor,
      },
    };
  }

  // Ensure dataZoom slider text and borders are high-contrast
  if (Array.isArray(clone.dataZoom)) {
    clone.dataZoom = clone.dataZoom.map((dz: any) => ({
      ...dz,
      textStyle: {
        fontSize: 10,
        fontWeight: 500,
        ...(dz.textStyle || {}),
        color: secondaryTextColor,
      },
      borderColor: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.12)",
      handleStyle: {
        color: isDark ? "#38bdf8" : "#800000",
        borderColor: isDark ? "#0284c7" : "#550000",
        ...(dz.handleStyle || {}),
      },
    }));
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
  const onInstanceReadyRef = useRef(onInstanceReady);
  const { theme } = useTheme();

  // Dynamic theme state that reacts to both useTheme hook and document root classList
  const [effectiveDark, setEffectiveDark] = useState<boolean>(() => {
    if (typeof document !== "undefined") {
      return document.documentElement.classList.contains("dark");
    }
    return theme === "dark";
  });

  useEffect(() => {
    const updateThemeState = () => {
      if (typeof document !== "undefined") {
        setEffectiveDark(document.documentElement.classList.contains("dark"));
      } else {
        setEffectiveDark(theme === "dark");
      }
    };

    updateThemeState();

    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.attributeName === "class") {
          updateThemeState();
        }
      }
    });

    if (typeof document !== "undefined") {
      observer.observe(document.documentElement, { attributes: true });
    }

    return () => observer.disconnect();
  }, [theme]);

  useEffect(() => {
    onChartClickRef.current = onChartClick;
    onInstanceReadyRef.current = onInstanceReady;
  });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return () => { };
    }

    let instance = echarts.getInstanceByDom(container);
    if (!instance) {
      instance = echarts.init(container, undefined, {
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

    let timer1: ReturnType<typeof setTimeout> | undefined;
    let timer2: ReturnType<typeof setTimeout> | undefined;

    if (option && Object.keys(option).length > 0) {
      const cleanOption = sanitizeEChartsOption(option, effectiveDark);
      try {
        // Clear previous series & coordinate systems when switching chart types to prevent layout collision
        instance.clear();
        instance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
        resizeChart();
        timer1 = setTimeout(resizeChart, 60);
        timer2 = setTimeout(resizeChart, 250);
      } catch (err) {
        console.error('ECharts setOption error, recreating instance:', err);
        try {
          instance.dispose();
          instance = echarts.init(container, undefined, { renderer: 'canvas' });
          chartInstanceRef.current = instance;
          if (onInstanceReadyRef.current) {
            onInstanceReadyRef.current(instance);
          }
          instance.on('click', clickHandler);
          instance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
          instance.resize();
        } catch (recoverErr) {
          console.error('Failed to recover ECharts instance:', recoverErr);
        }
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      resizeChart();
    });
    resizeObserver.observe(container);

    window.addEventListener("resize", resizeChart);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      resizeObserver.disconnect();
      window.removeEventListener("resize", resizeChart);
      instance.off("click", clickHandler);
    };
  }, [option, effectiveDark]);

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
  initialInfo,
}) => {
  const { toast } = useToast();
  const nativeChartInstanceRef = useRef<echarts.ECharts | null>(null);
  const lastExecutedKeyRef = useRef<string>("");

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
  const [, setCategoricalCols] = useState<string[]>([]);
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
  const [yAxisScale] = useState<string>("linear");
  const [timeGrain, setTimeGrain] = useState<string>("");
  const [filters, setFilters] = useState<FilterItem[]>([]);
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

  // Interactive Data Table Exploration State
  const [tableSearch, setTableSearch] = useState<string>("");
  const [tableSortCol, setTableSortCol] = useState<string>("value");
  const [tableSortDir, setTableSortDir] = useState<"asc" | "desc">("desc");
  const [tablePage, setTablePage] = useState<number>(1);
  const [tablePageSize, setTablePageSize] = useState<number>(10);

  // Reset table pagination when search or dimension changes
  useEffect(() => {
    setTablePage(1);
  }, [tableSearch, selectedTaskId, dimension]);

  const totalMetricSum = useMemo(() => {
    if (!queryResult?.records || !Array.isArray(queryResult.records)) return 0;
    return queryResult.records.reduce(
      (acc: number, r: any) => acc + (typeof r.value === "number" ? Math.abs(r.value) : 0),
      0
    );
  }, [queryResult?.records]);

  const totalRowCount = useMemo(() => {
    if (!queryResult?.records || !Array.isArray(queryResult.records)) return 0;
    return queryResult.records.reduce(
      (acc: number, r: any) => acc + (typeof r.count === "number" ? r.count : 0),
      0
    );
  }, [queryResult?.records]);

  const filteredAndSortedRecords = useMemo(() => {
    if (!queryResult?.records || !Array.isArray(queryResult.records)) return [];
    let list = [...queryResult.records];

    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase().trim();
      list = list.filter((r) =>
        Object.values(r).some(
          (val) => val !== null && val !== undefined && String(val).toLowerCase().includes(q)
        )
      );
    }

    list.sort((a, b) => {
      const aVal = a[tableSortCol];
      const bVal = b[tableSortCol];
      if (typeof aVal === "number" && typeof bVal === "number") {
        return tableSortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      const strA = String(aVal ?? "");
      const strB = String(bVal ?? "");
      return tableSortDir === "asc" ? strA.localeCompare(strB) : strB.localeCompare(strA);
    });

    return list;
  }, [queryResult?.records, tableSearch, tableSortCol, tableSortDir]);

  const totalPages = Math.max(1, Math.ceil(filteredAndSortedRecords.length / tablePageSize));
  const paginatedRecords = useMemo(() => {
    const start = (tablePage - 1) * tablePageSize;
    return filteredAndSortedRecords.slice(start, start + tablePageSize);
  }, [filteredAndSortedRecords, tablePage, tablePageSize]);

  const handleSortClick = (col: string) => {
    if (tableSortCol === col) {
      setTableSortDir(prev => (prev === "asc" ? "desc" : "asc"));
    } else {
      setTableSortCol(col);
      setTableSortDir("desc");
    }
    setTablePage(1);
  };

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

  // 1. Load available datasets (skip if embedded with dedicated taskId)
  useEffect(() => {
    if (!allowDatasetSwitching && initialTaskId) {
      return;
    }
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
  }, [selectedTaskId, allowDatasetSwitching, initialTaskId]);

  useEffect(() => {
    if (initialTaskId && initialTaskId !== selectedTaskId) {
      setSelectedTaskId(initialTaskId);
    }
  }, [initialTaskId, selectedTaskId]);

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

        const topRec = Array.isArray(res.recommendations) && res.recommendations.length > 0
          ? res.recommendations[0]
          : null;

        const isIdCol = (name: string) => {
          const n = name.trim().toLowerCase();
          const metricKeywords = [
            "revenue", "sales", "price", "amount", "cost", "salary", "profit",
            "margin", "score", "total", "subtotal", "val", "value", "rate", "ratio",
            "qty", "quantity", "count", "units", "weight", "height", "depth", "distance",
            "duration", "balance", "budget", "fee", "tax", "discount", "spend"
          ];
          if (metricKeywords.some(kw => n.includes(kw))) return false;
          return /^(id|.*_id|id_.*|.*_no|no_.*|num|.*_num|sn|s_no|.*_sn|uuid|guid|code|.*_code|index|key|.*_key|row_.*|pk|.*_pk|fk|.*_fk|card|credit_card|creditcard|cc_num|ssn|social_security|phone|mobile|cell|telephone|fax|zip|zipcode|postal|postal_code|pin|pincode|account|acct|iban|routing|swift|cvv|cvc|passport|license|aadhaar|pan|ein|tin|tax_id|ip|ip_address|mac|mac_address|token|hash|batch|seq|sequence|version)$/i.test(n)
            || /card|ssn|phone|mobile|zipcode|postal|passport|license|iban|routing/i.test(n);
        };
        const validQuantMetric = (res.quantitative_columns && res.quantitative_columns.length > 0)
          ? res.quantitative_columns.find((c: string) => !isIdCol(c)) || ""
          : (res.numeric_columns || []).find((c: string) => !isIdCol(c)) || "";

        const targetDim = topRec?.dimension || (
          res.categorical_columns && res.categorical_columns.length > 0
            ? res.categorical_columns[0]
            : res.columns?.[0]?.name || ""
        );
        const recMetricValid = topRec?.metric && !isIdCol(topRec.metric) ? topRec.metric : "";
        const targetMetric = recMetricValid || validQuantMetric;
        const targetAgg = topRec?.aggregate || (
          targetMetric ? "AVG" : "COUNT"
        );
        const targetChartType = topRec?.chart_type || "echarts_bar";

        if (topRec?.chart_type) setChartType(topRec.chart_type);
        setDimension(targetDim);
        setMetric(targetMetric);
        setAggregate(targetAgg);
        if (topRec?.secondary_metric) setSecondaryMetric(topRec.secondary_metric);
        if (topRec?.secondary_aggregate) setSecondaryAggregate(topRec.secondary_aggregate);
        if (topRec?.time_grain) setTimeGrain(topRec.time_grain);

        // Instant first paint: if backend pre-bundled default query result, render immediately with 0 extra network calls
        if (res.default_query_result) {
          const prewarmedKey = `${selectedTaskId}|${targetChartType}|${targetDim}|${targetMetric}|${targetAgg}|${topRec?.secondary_dimension || ''}|${topRec?.secondary_metric || ''}|${topRec?.time_grain || ''}|[]|25|desc|supersetColors|grouped`;
          lastExecutedKeyRef.current = prewarmedKey;
          setQueryResult(res.default_query_result);
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
  const executeQuery = useCallback(async (signal?: AbortSignal) => {
    if (!selectedTaskId || !dimension) return;

    const activeFilters = filters
      .filter(f => f.column && (f.operator.includes("null") || f.value.trim() !== ""))
      .map(f => ({
        column: f.column,
        operator: f.operator,
        value: f.operator.includes("null") ? null : f.value,
      }));

    const currentKey = `${selectedTaskId}|${chartType}|${dimension}|${metric}|${aggregate}|${secondaryDimension || ''}|${secondaryMetric || ''}|${timeGrain || ''}|${JSON.stringify(activeFilters)}|${rowLimit}|${sortOrder}|${colorPalette}|${barMode}`;

    if (lastExecutedKeyRef.current === currentKey && queryResult) {
      // Query is identical to what is already displayed, skip redundant execution
      return;
    }
    lastExecutedKeyRef.current = currentKey;

    setExecuting(true);
    try {
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
      if (signal?.aborted) return;
      setQueryResult(res);
    } catch (err: any) {
      if (signal?.aborted) return;
      console.error("Explore query execution failed:", err);
      toast({
        title: "Query Failed",
        description: err.message || "Could not execute analytical aggregation.",
        variant: "destructive",
      });
    } finally {
      if (!signal?.aborted) {
        setExecuting(false);
      }
    }
  }, [
    selectedTaskId,
    dimension,
    filters,
    chartType,
    secondaryDimension,
    barMode,
    metric,
    aggregate,
    secondaryMetric,
    secondaryAggregate,
    yAxisScale,
    timeGrain,
    rowLimit,
    sortOrder,
    colorPalette,
    toast,
  ]);

  useEffect(() => {
    const controller = new AbortController();
    if (dimension && selectedTaskId) {
      void executeQuery(controller.signal);
    }
    return () => {
      controller.abort();
    };
  }, [
    executeQuery,
    dimension,
    selectedTaskId,
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
        const res = await fetch(dataUrl);
        if (!res.ok) throw new Error(`Failed to fetch blob: ${res.status}`);
        const blob = await res.blob();
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
    <div className={`w-full rounded-2xl border border-stone-300 bg-[#FAF8F5] text-stone-900 dark:border-slate-800 dark:bg-[#07080a] dark:text-slate-100 overflow-hidden shadow-sm dark:shadow-none ${isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#FAF8F5] dark:bg-[#07080a] overflow-y-auto" : ""
      }`}>

      {/* ─── Top Studio Header Bar ─── */}
      <div className="px-3.5 sm:px-5 py-3 border-b border-stone-200 bg-[#FAF8F5] dark:border-slate-800 dark:bg-[#0c0d12] flex flex-col sm:flex-row sm:items-center justify-between gap-3">

        {/* Left: Branding & Dataset Identity */}
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

            {/* Dataset selector dropdown or name badge */}
            <div className="flex items-center gap-2 mt-0.5 text-xs text-stone-600 dark:text-slate-400">
              {allowDatasetSwitching && availableDatasets.length > 1 ? (
                <div className="flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5 text-stone-500 dark:text-slate-400" />
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
                    className="bg-white border border-stone-300 text-xs rounded px-2 py-0.5 text-stone-900 focus:outline-none focus:border-[#800000]/50 dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200 dark:focus:border-white/40 font-mono max-w-[180px] xs:max-w-[240px] sm:max-w-xs truncate shadow-2xs"
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
                  <Database className="h-3.5 w-3.5 text-stone-500 dark:text-slate-400" />
                  <span className="text-stone-800 dark:text-slate-200 font-medium">{displayName}</span>
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
              className="bg-sky-500/10 text-sky-700 border-sky-300 dark:bg-sky-500/10 dark:text-sky-400 dark:border-sky-500/30 text-xs px-2 py-0.5 rounded-lg font-mono"
            >
              {filters.length} Filter{filters.length > 1 ? "s" : ""}
            </Badge>
          )}

          {/* Refresh Query */}
          <Button
            size="sm"
            variant="outline"
            onClick={() => void executeQuery()}
            disabled={executing || loadingSchema}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
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
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
            title="Copy chart image to clipboard"
          >
            {copiedChart ? <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            <span className="hidden sm:inline">{copiedChart ? "Copied" : "Copy"}</span>
          </Button>

          {/* Export Actions */}
          <Button
            size="sm"
            variant="outline"
            onClick={handleExportChart}
            disabled={!queryResult}
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
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
            className="h-8 rounded-lg border-stone-300 bg-white hover:bg-stone-100 text-xs text-stone-800 dark:border-slate-700 dark:bg-slate-800/80 dark:hover:bg-slate-700 dark:text-slate-200 gap-1.5 shadow-2xs cursor-pointer"
            title="Export grouped data to CSV"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
            <span className="hidden sm:inline">CSV</span>
          </Button>

          {/* Fullscreen Toggle */}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="h-8 w-8 p-0 rounded-lg text-stone-600 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white cursor-pointer"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* ─── Visual Analytics Studio Canvas ─── */}
      <>
        {/* Mobile View Switcher Tab (Mobile Only) */}
        <div className="lg:hidden px-3.5 py-2.5 bg-[#FAF8F5] border-b border-stone-200 dark:bg-[#0a0b10] dark:border-slate-800/80 flex items-center justify-center">
          <div className="grid grid-cols-2 p-1 rounded-xl bg-stone-200/80 border border-stone-300 dark:bg-black/50 dark:border-white/10 w-full max-w-sm">
            <button
              type="button"
              onClick={() => setMobileTab("chart")}
              className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${mobileTab === "chart"
                  ? "bg-[#800000] text-white border border-[#800000] shadow-xs font-semibold dark:bg-white dark:text-zinc-950 dark:border-white"
                  : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200"
                }`}
            >
              <BarChart3 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>Chart & Visuals</span>
            </button>
            <button
              type="button"
              onClick={() => setMobileTab("controls")}
              className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition cursor-pointer ${mobileTab === "controls"
                  ? "bg-[#800000] text-white border border-[#800000] shadow-xs font-semibold dark:bg-white dark:text-zinc-950 dark:border-white"
                  : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200"
                }`}
            >
              <Sliders className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
              <span>Query Controls</span>
            </button>
          </div>
        </div>

        {/* Two-Column Explore Split Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[640px]">

          {/* ──── LEFT PANEL: Superset Explore Controls (4 cols) ──── */}
          <div className={`lg:col-span-4 p-4 sm:p-5 border-b lg:border-b-0 lg:border-r border-stone-200 bg-[#F4F0EA]/70 dark:border-slate-800 dark:bg-[#0c0d12] space-y-4 sm:space-y-5 ${mobileTab === "controls" ? "block" : "hidden lg:block"
            }`}>

            {/* Section: Suggested Visual Explorations */}
            {recommendations.length > 0 && (
              <div className="space-y-2 pb-2 border-b border-stone-200 dark:border-slate-800/80">
                <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
                  <TrendingUp className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                  <span>Suggested Explorations</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {recommendations.slice(0, 4).map((rec: any) => (
                    <button
                      key={rec.id}
                      type="button"
                      onClick={() => applyRecommendation(rec)}
                      className="text-left text-[11px] px-2 py-1 rounded-lg border border-stone-200 bg-white hover:bg-stone-100 hover:border-[#800000]/40 text-stone-700 hover:text-stone-950 dark:border-white/10 dark:bg-[#0e1017] dark:hover:bg-white/[0.08] dark:hover:border-emerald-500/40 dark:text-zinc-300 dark:hover:text-white transition group flex items-center gap-1.5 cursor-pointer max-w-full font-sans shadow-2xs"
                      title={rec.description}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400 shrink-0" />
                      <span className="truncate max-w-[130px] font-medium">{rec.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Section: Chart Type Picker */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                Visualization Type
              </span>
              <div className="grid grid-cols-3 gap-1.5">
                {CHART_TYPES.map((ct) => {
                  const Icon = ct.icon;
                  const isActive = chartType === ct.id;
                  return (
                    <button
                      key={ct.id}
                      type="button"
                      onClick={() => setChartType(ct.id)}
                      className={`flex flex-col items-center justify-center p-2 rounded-xl border text-center transition font-sans cursor-pointer ${isActive
                          ? "bg-[#800000] border-[#800000] text-white shadow-sm font-semibold dark:bg-white dark:border-white dark:text-zinc-950 dark:shadow-sm"
                          : "bg-white border-stone-200 text-stone-700 hover:bg-stone-100 hover:text-stone-950 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
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
              <label htmlFor="dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                Dimension (Group By / X-Axis)
              </label>
              <select
                id="dimension-select"
                aria-label="Dimension (Group By / X-Axis)"
                value={dimension}
                onChange={(e) => setDimension(e.target.value)}
                className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono shadow-2xs"
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
              <div className="space-y-1.5 p-2.5 rounded-xl bg-sky-50 border border-sky-200 dark:bg-sky-950/20 dark:border-sky-800/40">
                <div className="flex items-center gap-1.5 text-sky-700 dark:text-sky-400 text-xs font-semibold">
                  <Calendar className="h-3.5 w-3.5" />
                  <span>Time Grain (Date Truncation)</span>
                </div>
                <select
                  id="time-grain-select"
                  aria-label="Time Grain"
                  value={timeGrain}
                  onChange={(e) => setTimeGrain(e.target.value)}
                  className="w-full bg-white border border-sky-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none font-mono dark:bg-[#0e1017] dark:border-sky-800/60 dark:text-sky-200 shadow-2xs"
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
              <div className="space-y-2 p-3 rounded-xl bg-purple-50 border border-purple-200 dark:bg-purple-950/20 dark:border-purple-800/40">
                <div className="flex items-center gap-1.5 text-purple-700 dark:text-purple-400 text-xs font-semibold">
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>Secondary Metric (Line on Right Axis)</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <select
                    id="sec-metric-select"
                    aria-label="Secondary Metric"
                    value={secondaryMetric}
                    onChange={(e) => setSecondaryMetric(e.target.value)}
                    className="bg-white border border-purple-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 font-mono dark:bg-[#0e1017] dark:border-purple-800/60 dark:text-purple-200 shadow-2xs"
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
                    className="bg-white border border-purple-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 font-mono dark:bg-[#0e1017] dark:border-purple-800/60 dark:text-purple-200 shadow-2xs"
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
              <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                Metric & Aggregation (Y-Axis)
              </span>
              <div className="grid grid-cols-2 gap-2">
                <select
                  id="aggregate-select"
                  aria-label="Aggregation Function"
                  value={aggregate}
                  onChange={(e) => setAggregate(e.target.value)}
                  className="bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono shadow-2xs"
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
                  className="bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono disabled:opacity-50 disabled:bg-stone-100 dark:disabled:bg-slate-900 shadow-2xs"
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
                <label htmlFor="secondary-dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                  Secondary Breakdown
                </label>
                <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-sans font-medium">
                  {chartType === "echarts_heatmap" ? "Required for Heatmap" : "Optional"}
                </span>
              </div>
              <select
                id="secondary-dimension-select"
                aria-label="Secondary Breakdown"
                value={secondaryDimension}
                onChange={(e) => setSecondaryDimension(e.target.value)}
                className={`w-full bg-white border text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none font-mono dark:bg-[#0e1017] dark:text-slate-200 shadow-2xs ${chartType === "echarts_heatmap" && !secondaryDimension
                    ? "border-amber-500/70"
                    : "border-stone-300 dark:border-slate-800"
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
                  <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
                    Bar Mode
                  </span>
                  <div className="flex items-center gap-1 bg-stone-200/80 dark:bg-[#090b10] p-0.5 rounded-lg border border-stone-300 dark:border-slate-800">
                    <button
                      type="button"
                      onClick={() => setBarMode("grouped")}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all cursor-pointer ${barMode === "grouped"
                          ? "bg-[#800000] text-white font-bold shadow-xs dark:bg-white dark:text-zinc-950"
                          : "text-stone-700 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white"
                        }`}
                    >
                      Grouped
                    </button>
                    <button
                      type="button"
                      onClick={() => setBarMode("stacked")}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all cursor-pointer ${barMode === "stacked"
                          ? "bg-[#800000] text-white font-bold shadow-xs dark:bg-white dark:text-zinc-950"
                          : "text-stone-700 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white"
                        }`}
                    >
                      Stacked
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Section: Ad-Hoc Filters */}
            <div className="space-y-2 pt-2 border-t border-stone-200 dark:border-slate-800/80">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
                  <Filter className="h-3 w-3 text-sky-600 dark:text-sky-400" />
                  <span>Ad-Hoc Filters ({filters.length})</span>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={addFilter}
                  className="h-6 px-2 text-[10px] font-medium text-sky-600 hover:text-sky-700 hover:bg-sky-50 dark:text-sky-400 dark:hover:text-sky-300 dark:hover:bg-sky-500/10 rounded-md gap-1 cursor-pointer"
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
                      className="p-2 rounded-xl bg-white border border-stone-200 dark:bg-[#0e1017] dark:border-slate-800 space-y-1.5 text-xs shadow-2xs"
                    >
                      <div className="flex items-center gap-1.5">
                        {/* Filter Column */}
                        <select
                          value={f.column}
                          onChange={(e) => updateFilter(f.id, "column", e.target.value)}
                          aria-label="Filter Column"
                          className="bg-stone-50 border border-stone-300 rounded-md px-1.5 py-1 text-stone-900 text-[11px] font-mono flex-1 truncate dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200"
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
                          aria-label="Filter Operator"
                          className="bg-stone-50 border border-stone-300 rounded-md px-1.5 py-1 text-stone-900 text-[11px] font-mono dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200"
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
                          className="h-7 w-7 rounded-lg text-stone-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 shrink-0 cursor-pointer"
                          onDelete={() => removeFilter(f.id)}
                          title="Remove filter"
                        />
                      </div>

                      {/* Value Input (unless is_null / is_not_null) */}
                      {!f.operator.includes("null") && (
                        <input
                          type="text"
                          placeholder="Comparison value..."
                          aria-label="Filter Comparison Value"
                          value={f.value}
                          onChange={(e) => updateFilter(f.id, "value", e.target.value)}
                          className="w-full bg-stone-50 border border-stone-300 rounded-md px-2 py-1 text-stone-900 text-[11px] font-mono focus:outline-none focus:border-[#800000]/50 dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200 dark:focus:border-white/30"
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
                <label htmlFor="row-limit-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                  Row Limit
                </label>
                <select
                  id="row-limit-select"
                  aria-label="Row Limit"
                  value={rowLimit}
                  onChange={(e) => setRowLimit(Number(e.target.value))}
                  className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 font-mono shadow-2xs"
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
                <label htmlFor="sort-order-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                  Sort Order
                </label>
                <select
                  id="sort-order-select"
                  aria-label="Sort Order"
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value)}
                  className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 font-mono shadow-2xs"
                >
                  <option value="desc">High to Low (Desc)</option>
                  <option value="asc">Low to High (Asc)</option>
                </select>
              </div>
            </div>

            {/* Section: Color Palette */}
            <div className="space-y-1.5 pt-2 border-t border-stone-200 dark:border-slate-800/80">
              <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
                Color Palette
              </span>
              <div className="space-y-1">
                {COLOR_PALETTES.map((pal) => (
                  <button
                    key={pal.id}
                    type="button"
                    onClick={() => setColorPalette(pal.id)}
                    className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg border text-xs transition cursor-pointer ${colorPalette === pal.id
                        ? "bg-[#800000] border-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:border-white dark:text-zinc-950"
                        : "bg-white border-stone-200 text-stone-700 hover:bg-stone-100 hover:text-stone-950 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
                      }`}
                  >
                    <span>{pal.label}</span>
                    <div className="flex items-center gap-1">
                      {pal.preview.map((c, i) => (
                        <span key={i} className="h-2.5 w-2.5 rounded-full ring-1 ring-black/10 dark:ring-white/20" style={{ backgroundColor: c }} />
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
                icon={<BarChart3 className="h-3.5 w-3.5 text-white dark:text-zinc-950" />}
              >
                <span>Apply & View Visual Chart</span>
              </ShimmerActionButton>
            </div>
          </div>

          {/* ──── RIGHT CANVAS: Live Visualization & KPIs (8 cols) ──── */}
          <div className={`lg:col-span-8 p-3.5 sm:p-6 flex flex-col justify-between space-y-4 sm:space-y-6 bg-stone-50/70 dark:bg-[#07080a] ${mobileTab === "chart" ? "flex" : "hidden lg:flex"
            }`}>

            {/* Top: Architectural Precision Telemetry Deck */}
            {queryResult?.kpis && (
              <ExploreTelemetryDeck
                queryResult={queryResult}
                aggregate={aggregate}
                metric={metric}
                dimension={dimension}
              />
            )}

            {/* Statistical & Observational Insights Strip */}
            {queryResult?.smart_insights && queryResult.smart_insights.length > 0 && (
              <div className="rounded-xl border border-stone-200/80 bg-white/80 dark:border-white/10 dark:bg-[#111218] p-3.5 sm:p-4 space-y-2.5 shadow-2xs">
                <div className="flex items-center justify-between pb-1 border-b border-stone-100 dark:border-white/[0.06]">
                  <div className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-stone-400 dark:bg-zinc-500" />
                    <h4 className="text-[11px] font-mono uppercase tracking-[0.08em] text-stone-600 dark:text-zinc-400 font-medium">
                      Key Observations & Patterns
                    </h4>
                  </div>
                  <span className="text-[10px] font-mono text-stone-400 dark:text-zinc-500">
                    {queryResult.smart_insights.length} observations
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {queryResult.smart_insights.map((ins: any) => (
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
            )}

            {/* Main Chart or Table Canvas */}
            <div className="flex-1 min-h-[400px] sm:min-h-[480px] rounded-xl border border-stone-200/90 bg-white/95 dark:border-white/10 dark:bg-[#0c0d12] p-3.5 sm:p-5 flex flex-col relative shadow-2xs">

              {/* View Switcher & Toolbar */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-stone-200 dark:border-white/10 mb-3">
                <div className="flex items-center gap-1 bg-stone-100 dark:bg-[#14151c] p-0.5 rounded-lg border border-stone-200/80 dark:border-white/10">
                  <button
                    type="button"
                    onClick={() => setViewMode("chart")}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition cursor-pointer flex items-center gap-1.5 ${viewMode === "chart"
                        ? "bg-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:text-zinc-950"
                        : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white"
                      }`}
                  >
                    <BarChart3 className="h-3.5 w-3.5" />
                    <span>Visual Chart</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode("data")}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition cursor-pointer flex items-center gap-1.5 ${viewMode === "data"
                        ? "bg-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:text-zinc-950"
                        : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white"
                      }`}
                  >
                    <Table2 className="h-3.5 w-3.5" />
                    <span>Aggregated Data</span>
                    <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded-full bg-black/10 dark:bg-white/15 tabular-nums">
                      {queryResult?.records?.length ?? 0}
                    </span>
                  </button>
                </div>

                {/* Canvas Toolbar Controls */}
                <div className="flex items-center gap-2 flex-wrap text-xs font-sans">
                  {/* Quick Bar Orientation Toggle */}
                  {viewMode === "chart" && (chartType === "echarts_bar" || chartType === "echarts_horizontal_bar") && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setChartType(chartType === "echarts_bar" ? "echarts_horizontal_bar" : "echarts_bar")}
                      className="h-7 px-2 rounded-lg text-[11px] gap-1 border-stone-200 bg-white hover:bg-stone-50 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
                      title="Toggle Vertical Column vs Horizontal Ranking"
                    >
                      {chartType === "echarts_bar" ? <AlignLeft className="h-3 w-3" /> : <BarChart3 className="h-3 w-3" />}
                      <span>{chartType === "echarts_bar" ? "Horizontal Rank" : "Vertical Column"}</span>
                    </Button>
                  )}

                  <div className="text-xs text-stone-600 dark:text-zinc-400 font-mono truncate hidden md:inline">
                    {queryResult?.metric_label} by {dimension}
                    {timeGrain && <span className="text-sky-600 dark:text-sky-400 ml-1">({timeGrain})</span>}
                  </div>
                </div>
              </div>

              {/* Loading Overlay */}
              {executing && (
                <div className="absolute inset-0 bg-white/80 dark:bg-[#07080a]/80 backdrop-blur-xs flex flex-col items-center justify-center z-20 gap-2 rounded-2xl">
                  <Loader2 className="h-6 w-6 animate-spin text-stone-700 dark:text-zinc-300" />
                  <span className="text-xs font-mono text-stone-600 dark:text-zinc-400">Computing visual analytics...</span>
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

              {/* View Mode 2: Enhanced Interactive Data Table */}
              {(viewMode === "data" || chartType === "table") && queryResult?.records && (
                <div className="flex-1 flex flex-col space-y-3 min-h-[420px]">

                  {/* In-Table Search Bar & Page Size Selector */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-2">
                    <div className="relative flex-1 max-w-sm">
                      <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-stone-400 dark:text-zinc-500" />
                      <input
                        type="text"
                        placeholder={`Search ${dimension} or values...`}
                        value={tableSearch}
                        onChange={(e) => setTableSearch(e.target.value)}
                        className="w-full pl-9 pr-3 py-1.5 text-xs bg-stone-50 dark:bg-[#14151c] border border-stone-200 dark:border-white/10 rounded-xl text-stone-900 dark:text-white placeholder:text-stone-400 dark:placeholder:text-zinc-500 focus:outline-none focus:border-stone-400 dark:focus:border-white/30 font-sans shadow-2xs"
                      />
                    </div>

                    <div className="flex items-center gap-2 self-end sm:self-auto">
                      <span className="text-[11px] text-stone-500 dark:text-zinc-400 font-sans">Rows per page:</span>
                      <select
                        aria-label="Rows per page"
                        value={tablePageSize}
                        onChange={(e) => {
                          setTablePageSize(Number(e.target.value));
                          setTablePage(1);
                        }}
                        className="bg-stone-50 dark:bg-[#14151c] border border-stone-200 dark:border-white/10 rounded-lg px-2 py-1 text-xs text-stone-800 dark:text-zinc-200 font-mono shadow-2xs"
                      >
                        <option value={10}>10</option>
                        <option value={25}>25</option>
                        <option value={50}>50</option>
                        <option value={100}>100</option>
                      </select>
                    </div>
                  </div>

                  {/* Table Container */}
                  <div className="flex-1 overflow-x-auto rounded-xl border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#090a0f] scrollbar-thin">
                    <table className="w-full text-left text-xs font-sans">
                      <thead className="bg-stone-50/90 text-stone-700 dark:bg-[#13141b] dark:text-zinc-300 uppercase text-[10.5px] font-semibold tracking-wider border-b border-stone-200 dark:border-white/10">
                        <tr>
                          <th className="py-2.5 px-3 w-12 text-center text-stone-400 dark:text-zinc-500">#</th>
                          <th
                            onClick={() => handleSortClick(dimension)}
                            className="py-2.5 px-3 cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                          >
                            <div className="flex items-center gap-1.5">
                              <span>{dimension}</span>
                              <ArrowUpDown className={`h-3 w-3 ${tableSortCol === dimension ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                            </div>
                          </th>
                          {secondaryDimension && (
                            <th
                              onClick={() => handleSortClick(secondaryDimension)}
                              className="py-2.5 px-3 cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                            >
                              <div className="flex items-center gap-1.5">
                                <span>{secondaryDimension}</span>
                                <ArrowUpDown className={`h-3 w-3 ${tableSortCol === secondaryDimension ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                              </div>
                            </th>
                          )}
                          <th
                            onClick={() => handleSortClick("value")}
                            className="py-2.5 px-3 min-w-[200px] cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                          >
                            <div className="flex items-center gap-1.5">
                              <span>{queryResult.metric_label || "Metric Value"} & Share</span>
                              <ArrowUpDown className={`h-3 w-3 ${tableSortCol === "value" ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                            </div>
                          </th>
                          <th
                            onClick={() => handleSortClick("count")}
                            className="py-2.5 px-3 text-right cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                          >
                            <div className="flex items-center justify-end gap-1.5">
                              <span>Records</span>
                              <ArrowUpDown className={`h-3 w-3 ${tableSortCol === "count" ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                            </div>
                          </th>
                          <th className="py-2.5 px-3 text-right w-16">Action</th>
                        </tr>
                      </thead>

                      <tbody className="divide-y divide-stone-100 dark:divide-white/[0.05] text-stone-800 dark:text-zinc-200">
                        {paginatedRecords.length > 0 ? (
                          paginatedRecords.map((row: any, idx: number) => {
                            const globalIdx = (tablePage - 1) * tablePageSize + idx + 1;
                            const rowVal = typeof row.value === "number" ? row.value : 0;
                            const pctOfTotal = totalMetricSum > 0 ? (Math.abs(rowVal) / totalMetricSum) * 100 : 0;
                            const paletteColors = COLOR_PALETTES.find(p => p.id === colorPalette)?.preview || ["#1FA8C9"];

                            return (
                              <tr key={idx} className="hover:bg-stone-50/80 dark:hover:bg-white/[0.04] transition group">
                                <td className="py-2.5 px-3 text-center text-stone-400 dark:text-zinc-500 font-mono text-[11px]">
                                  {globalIdx}
                                </td>
                                <td className="py-2.5 px-3 font-medium text-stone-900 dark:text-white">
                                  <button
                                    type="button"
                                    onClick={() => handleChartClick({ name: row[dimension] })}
                                    className="text-left hover:text-emerald-600 dark:hover:text-emerald-400 transition cursor-pointer flex items-center gap-1.5"
                                    title={`Filter by ${dimension} = "${row[dimension]}"`}
                                  >
                                    <span>{row[dimension] ?? "—"}</span>
                                  </button>
                                </td>
                                {secondaryDimension && (
                                  <td className="py-2.5 px-3 text-stone-700 dark:text-zinc-300 font-mono text-[11px]">
                                    {row[secondaryDimension] ?? "—"}
                                  </td>
                                )}
                                <td className="py-2.5 px-3">
                                  <div className="space-y-1">
                                    <div className="flex items-center justify-between gap-2">
                                      <span className="font-bold text-stone-900 dark:text-white font-mono text-[11.5px] tabular-nums">
                                        {typeof row.value === "number"
                                          ? row.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
                                          : row.value ?? "—"}
                                      </span>
                                      <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-mono font-medium">
                                        {totalMetricSum > 0 ? `${pctOfTotal.toFixed(1)}%` : ""}
                                      </span>
                                    </div>
                                    <div className="w-full h-1.5 bg-stone-200/70 dark:bg-white/10 rounded-full overflow-hidden">
                                      <div
                                        className="h-full rounded-full transition-all duration-300"
                                        style={{
                                          width: `${Math.min(100, Math.max(3, pctOfTotal))}%`,
                                          backgroundColor: paletteColors[0] || "#10b981",
                                        }}
                                      />
                                    </div>
                                  </div>
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono text-[11px] text-stone-600 dark:text-zinc-400 tabular-nums">
                                  {row.count?.toLocaleString() ?? "—"}
                                </td>
                                <td className="py-2.5 px-3 text-right">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      const text = JSON.stringify(row, null, 2);
                                      navigator.clipboard.writeText(text);
                                      toast({ title: "Row Copied", description: "Row JSON copied to clipboard." });
                                    }}
                                    className="p-1 rounded-md text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 hover:bg-stone-100 dark:hover:bg-white/10 transition cursor-pointer"
                                    title="Copy row data"
                                  >
                                    <Copy className="h-3 w-3" />
                                  </button>
                                </td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr>
                            <td colSpan={6} className="py-8 text-center text-stone-500 dark:text-zinc-400 font-sans">
                              No records matching "{tableSearch}"
                            </td>
                          </tr>
                        )}
                      </tbody>

                      {/* Summary Footer Row */}
                      {filteredAndSortedRecords.length > 0 && (
                        <tfoot className="bg-stone-50/90 dark:bg-[#13141b] border-t border-stone-200 dark:border-white/10 font-sans font-semibold text-[11px] text-stone-900 dark:text-white">
                          <tr>
                            <td className="py-2.5 px-3 text-center text-stone-500 dark:text-zinc-400">∑</td>
                            <td className="py-2.5 px-3">
                              Total ({filteredAndSortedRecords.length} categories)
                            </td>
                            {secondaryDimension && <td className="py-2.5 px-3">—</td>}
                            <td className="py-2.5 px-3 font-mono tabular-nums">
                              {totalMetricSum.toLocaleString(undefined, { maximumFractionDigits: 2 })} (100%)
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                              {totalRowCount.toLocaleString()}
                            </td>
                            <td className="py-2.5 px-3 text-right">—</td>
                          </tr>
                        </tfoot>
                      )}
                    </table>
                  </div>

                  {/* Pagination Bar */}
                  <div className="flex items-center justify-between text-xs text-stone-600 dark:text-zinc-400 pt-1 font-sans">
                    <span className="text-[11px]">
                      Showing <strong className="text-stone-900 dark:text-white font-mono">{(tablePage - 1) * tablePageSize + 1}</strong> to{" "}
                      <strong className="text-stone-900 dark:text-white font-mono">
                        {Math.min(tablePage * tablePageSize, filteredAndSortedRecords.length)}
                      </strong>{" "}
                      of <strong className="text-stone-900 dark:text-white font-mono">{filteredAndSortedRecords.length}</strong> groups
                    </span>

                    <div className="flex items-center gap-1.5">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={tablePage <= 1}
                        onClick={() => setTablePage(p => Math.max(1, p - 1))}
                        className="h-7 px-2 rounded-lg text-xs gap-1 border-stone-200 bg-white hover:bg-stone-100 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                        <span>Prev</span>
                      </Button>
                      <span className="text-[11px] px-2 font-mono">
                        {tablePage} / {totalPages}
                      </span>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={tablePage >= totalPages}
                        onClick={() => setTablePage(p => Math.min(totalPages, p + 1))}
                        className="h-7 px-2 rounded-lg text-xs gap-1 border-stone-200 bg-white hover:bg-stone-100 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
                      >
                        <span>Next</span>
                        <ChevronRight className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>

                </div>
              )}

            </div>

            {/* Footer Metadata */}
            <div className="flex items-center justify-between text-xs text-stone-600 dark:text-slate-400 pt-2 border-t border-stone-200 dark:border-slate-800/80 font-mono">
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
