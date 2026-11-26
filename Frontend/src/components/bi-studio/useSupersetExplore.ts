import { useCallback, useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import type { ChartClickPayload } from "./NativeEChartsView";
import type {
  DatasetItem,
  FilterItem,
  SchemaColumnMeta,
  SupersetExploreStudioProps,
} from "./constants";

const ID_METRIC_KEYWORDS = [
  "revenue", "sales", "price", "amount", "cost", "salary", "profit",
  "margin", "score", "total", "subtotal", "val", "value", "rate", "ratio",
  "qty", "quantity", "count", "units", "weight", "height", "depth", "distance",
  "duration", "balance", "budget", "fee", "tax", "discount", "spend",
];

function isIdentifierColumn(name: string): boolean {
  const n = name.trim().toLowerCase();
  if (ID_METRIC_KEYWORDS.some((kw) => n.includes(kw))) return false;
  return (
    /^(id|.*_id|id_.*|.*_no|no_.*|num|.*_num|sn|s_no|.*_sn|uuid|guid|code|.*_code|index|key|.*_key|row_.*|pk|.*_pk|fk|.*_fk|card|credit_card|creditcard|cc_num|ssn|social_security|phone|mobile|cell|telephone|fax|zip|zipcode|postal|postal_code|pin|pincode|account|acct|iban|routing|swift|cvv|cvc|passport|license|aadhaar|pan|ein|tin|tax_id|ip|ip_address|mac|mac_address|token|hash|batch|seq|sequence|version)$/i.test(n) ||
    /card|ssn|phone|mobile|zipcode|postal|passport|license|iban|routing/i.test(n)
  );
}

export function useSupersetExplore({
  taskId: initialTaskId,
  datasetName: initialDatasetName,
  allowDatasetSwitching = true,
  initialInfo,
}: SupersetExploreStudioProps) {
  const { toast } = useToast();
  const nativeChartInstanceRef = useRef<echarts.ECharts | null>(null);
  const lastExecutedKeyRef = useRef<string>("");
  const lastQueryResultRef = useRef<any>(null);

  const [availableDatasets, setAvailableDatasets] = useState<DatasetItem[]>([]);
  const [taskSelection, setTaskSelection] = useState<{ propId: string | undefined; activeId: string | undefined }>({
    propId: initialTaskId,
    activeId: initialTaskId,
  });
  const selectedTaskId =
    initialTaskId && !allowDatasetSwitching
      ? initialTaskId
      : initialTaskId && taskSelection.propId !== initialTaskId
        ? initialTaskId
        : taskSelection.activeId;

  const [activeDatasetMeta, setActiveDatasetMeta] = useState<DatasetItem | null>(() => {
    if (!initialInfo || !initialTaskId) return null;
    return {
      task_id: initialTaskId,
      filename: initialDatasetName || "Active Dataset",
      rows: initialInfo.rows ?? 0,
      columns_count: Array.isArray(initialInfo.columns) ? initialInfo.columns.length : 0,
      columns: initialInfo.columns || [],
      numeric_columns: initialInfo.numeric_columns || [],
      categorical_columns: initialInfo.categorical_columns || [],
      created_at: "",
    };
  });

  const [loadingSchema, setLoadingSchema] = useState<boolean>(true);
  const [schemaColumns, setSchemaColumns] = useState<SchemaColumnMeta[]>([]);
  const [numericCols, setNumericCols] = useState<string[]>([]);
  const [, setCategoricalCols] = useState<string[]>([]);
  const [temporalCols, setTemporalCols] = useState<string[]>([]);

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

  const [executing, setExecuting] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<"chart" | "data">("chart");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [mobileTab, setMobileTab] = useState<"chart" | "controls">("chart");

  const [tableSearch, setTableSearch] = useState<string>("");
  const [tableSortCol, setTableSortCol] = useState<string>("value");
  const [tableSortDir, setTableSortDir] = useState<"asc" | "desc">("desc");
  const [tablePage, setTablePage] = useState<number>(1);
  const [tablePageSize, setTablePageSize] = useState<number>(10);

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

  useEffect(() => {
    if (!allowDatasetSwitching && initialTaskId) return;
    let cancelled = false;
    api
      .listRecentDatasets(20)
      .then((res) => {
        if (cancelled) return;
        if (res && res.datasets && res.datasets.length > 0) {
          setAvailableDatasets(res.datasets);
          const targetId = initialTaskId || res.datasets[0].task_id;
          const match = res.datasets.find((d) => d.task_id === targetId);
          if (match) setActiveDatasetMeta(match);
          if (!initialTaskId && targetId) {
            setTaskSelection({ propId: initialTaskId, activeId: targetId });
          }
        }
      })
      .catch((err) => {
        console.warn("Could not load recent datasets:", err);
      });
    return () => {
      cancelled = true;
    };
  }, [allowDatasetSwitching, initialTaskId]);

  useEffect(() => {
    if (!selectedTaskId) return;
    let cancelled = false;
    setLoadingSchema(true);
    api
      .getSupersetSchema(selectedTaskId)
      .then((res) => {
        if (cancelled || !res) return;
        setSchemaColumns(Array.isArray(res.columns) ? res.columns : []);
        setNumericCols(Array.isArray(res.numeric_columns) ? res.numeric_columns : []);
        setCategoricalCols(Array.isArray(res.categorical_columns) ? res.categorical_columns : []);
        setTemporalCols(Array.isArray(res.temporal_columns) ? res.temporal_columns : []);
        setRecommendations(Array.isArray(res.recommendations) ? res.recommendations : []);

        const topRec = Array.isArray(res.recommendations) && res.recommendations.length > 0 ? res.recommendations[0] : null;
        const validQuantMetric =
          res.quantitative_columns && res.quantitative_columns.length > 0
            ? res.quantitative_columns.find((c: string) => !isIdentifierColumn(c)) || ""
            : (res.numeric_columns || []).find((c: string) => !isIdentifierColumn(c)) || "";

        const targetDim =
          topRec?.dimension ||
          (res.categorical_columns && res.categorical_columns.length > 0
            ? res.categorical_columns[0]
            : res.columns?.[0]?.name || "");
        const recMetricValid = topRec?.metric && !isIdentifierColumn(topRec.metric) ? topRec.metric : "";
        const targetMetric = recMetricValid || validQuantMetric;
        const targetAgg = topRec?.aggregate || (targetMetric ? "AVG" : "COUNT");
        const targetChartType = topRec?.chart_type || "echarts_bar";

        if (topRec?.chart_type) setChartType(topRec.chart_type);
        setDimension(targetDim);
        setMetric(targetMetric);
        setAggregate(targetAgg);
        if (topRec?.secondary_metric) setSecondaryMetric(topRec.secondary_metric);
        if (topRec?.secondary_aggregate) setSecondaryAggregate(topRec.secondary_aggregate);
        if (topRec?.time_grain) setTimeGrain(topRec.time_grain);

        if (res.default_query_result) {
          const prewarmedKey = `${selectedTaskId}|${targetChartType}|${targetDim}|${targetMetric}|${targetAgg}|${topRec?.secondary_dimension || ""}|${topRec?.secondary_metric || ""}|${topRec?.time_grain || ""}|[]|25|desc|supersetColors|grouped`;
          lastExecutedKeyRef.current = prewarmedKey;
          lastQueryResultRef.current = res.default_query_result;
          setQueryResult(res.default_query_result);
        }
        setLoadingSchema(false);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("Failed to load Superset schema:", err);
        toast({ title: "Schema Error", description: "Could not introspect dataset schema.", variant: "destructive" });
        setLoadingSchema(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTaskId, toast]);

  const executeQuery = useCallback(
    async (signal?: AbortSignal, force: boolean = false) => {
      if (!selectedTaskId || !dimension) return;

      const activeFilters = filters
        .filter((f) => f.column && (f.operator.includes("null") || f.value.trim() !== ""))
        .map((f) => ({
          column: f.column,
          operator: f.operator,
          value: f.operator.includes("null") ? null : f.value,
        }));

      const currentKey = `${selectedTaskId}|${chartType}|${dimension}|${metric}|${aggregate}|${secondaryDimension || ""}|${secondaryMetric || ""}|${timeGrain || ""}|${JSON.stringify(activeFilters)}|${rowLimit}|${sortOrder}|${colorPalette}|${barMode}`;

      if (!force && lastExecutedKeyRef.current === currentKey && lastQueryResultRef.current) return;
      lastExecutedKeyRef.current = currentKey;

      setExecuting(true);
      try {
        const res = await api.querySupersetExplore(
          {
            task_id: selectedTaskId,
            chart_type: chartType,
            dimension,
            secondary_dimension: secondaryDimension || undefined,
            bar_mode: barMode,
            metric: metric || undefined,
            aggregate,
            secondary_metric: secondaryMetric || undefined,
            secondary_aggregate: secondaryAggregate || undefined,
            y_axis_scale: yAxisScale || "linear",
            time_grain: timeGrain || undefined,
            filters: activeFilters.length > 0 ? activeFilters : undefined,
            row_limit: rowLimit,
            sort_order: sortOrder,
            color_palette: colorPalette,
          },
          signal,
        );
        if (signal?.aborted) return;
        lastQueryResultRef.current = res;
        setQueryResult(res);
      } catch (err: unknown) {
        if (signal?.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
        console.error("Explore query execution failed:", err);
        toast({
          title: "Query Failed",
          description: err instanceof Error ? err.message : "Could not execute analytical aggregation.",
          variant: "destructive",
        });
      } finally {
        if (!signal?.aborted) setExecuting(false);
      }
    },
    [
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
    ],
  );

  useEffect(() => {
    const controller = new AbortController();
    if (dimension && selectedTaskId) {
      void executeQuery(controller.signal);
    }
    return () => controller.abort();
  }, [executeQuery, dimension, selectedTaskId]);

  const addFilter = () => {
    if (schemaColumns.length === 0) return;
    setFilters([
      ...filters,
      { id: Math.random().toString(36).substring(7), column: schemaColumns[0].name, operator: "eq", value: "" },
    ]);
  };

  const removeFilter = (id: string) => setFilters(filters.filter((f) => f.id !== id));
  const updateFilter = (id: string, field: keyof FilterItem, val: string) =>
    setFilters(filters.map((f) => (f.id === id ? { ...f, [field]: val } : f)));

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
    setTablePage(1);
    toast({ title: "Preset Applied", description: `Loaded: ${rec.title}` });
  };

  const handleExportChart = () => {
    if (!nativeChartInstanceRef.current) return;
    try {
      const base64Url = nativeChartInstanceRef.current.getDataURL({
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
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        setCopiedChart(true);
        setTimeout(() => setCopiedChart(false), 2000);
        toast({ title: "Chart Copied", description: "PNG image copied to clipboard." });
      }
    } catch (err) {
      console.warn("Clipboard write failed, downloading instead:", err);
      handleExportChart();
    }
  };

  const handleChartClick = useCallback(
    (params: ChartClickPayload) => {
      if (params && params.name && dimension) {
        const clickedVal = String(params.name);
        setFilters((prev) => [
          ...prev.filter((f) => f.column !== dimension),
          { id: Math.random().toString(), column: dimension, operator: "eq", value: clickedVal },
        ]);
        toast({ title: "Filter Applied", description: `Filtered by ${dimension} = "${clickedVal}"` });
      }
    },
    [dimension, toast],
  );

  const handleExportCsv = () => {
    if (!queryResult?.records || queryResult.records.length === 0) return;
    const records = queryResult.records;
    const headers = Object.keys(records[0]);
    const csvContent = [headers.join(","), ...records.map((r: any) => headers.map((h) => `"${r[h] ?? ""}"`).join(","))].join("\n");
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

  const selectDataset = (newId: string) => {
    setTaskSelection({ propId: initialTaskId, activeId: newId });
    setTablePage(1);
    const match = availableDatasets.find((d) => d.task_id === newId);
    if (match) setActiveDatasetMeta(match);
  };

  const handleSortClick = (col: string) => {
    if (tableSortCol === col) {
      setTableSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setTableSortCol(col);
      setTableSortDir("desc");
    }
    setTablePage(1);
  };

  return {
    toast,
    nativeChartInstanceRef,
    availableDatasets,
    selectedTaskId,
    selectDataset,
    activeDatasetMeta,
    loadingSchema,
    schemaColumns,
    numericCols,
    temporalCols,
    chartType,
    setChartType,
    barMode,
    setBarMode,
    dimension,
    setDimension: (val: string) => {
      setDimension(val);
      setTablePage(1);
    },
    secondaryDimension,
    setSecondaryDimension,
    metric,
    setMetric,
    aggregate,
    setAggregate,
    secondaryMetric,
    setSecondaryMetric,
    secondaryAggregate,
    setSecondaryAggregate,
    timeGrain,
    setTimeGrain,
    filters,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    colorPalette,
    setColorPalette,
    recommendations,
    copiedChart,
    executing,
    queryResult,
    viewMode,
    setViewMode,
    isFullscreen,
    setIsFullscreen,
    mobileTab,
    setMobileTab,
    tableSearch,
    setTableSearch: (val: string) => {
      setTableSearch(val);
      setTablePage(1);
    },
    tableSortCol,
    tableSortDir,
    tablePage,
    setTablePage,
    tablePageSize,
    setTablePageSize: (size: number) => {
      setTablePageSize(size);
      setTablePage(1);
    },
    executeQuery,
    addFilter,
    removeFilter,
    updateFilter,
    applyRecommendation,
    handleCopyChartImage,
    handleChartClick,
    handleExportChart,
    handleExportCsv,
    handleSortClick,
  };
}
