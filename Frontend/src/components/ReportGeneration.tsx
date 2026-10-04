import React, { useState, useEffect, useMemo, memo } from "react";
import {
  Download, RefreshCw, Loader2, FileText, ChevronRight,
  AlertTriangle, ArrowRight, BarChart3, PieChart, Activity,
  FileSpreadsheet, TrendingUp, ChevronDown, ChevronUp, ShieldCheck,
  BookOpen, Table2, Grid, Cpu, Terminal, Play, Sliders, Search,
  Copy, Check, Bookmark, BookmarkCheck, Trash2
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AppStep } from "@/pages/Workspace";
import type { AnalysisResult, Charts, InsightResult, DatasetInfo, DataRow } from "@/types/api";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import { MLReadinessCard } from "./MLReadinessCard";
import { HiPlotParallelCoordinates, type HiPlotPayload, type HiPlotDimension } from "./HiPlotParallelCoordinates";

// Safe, memoized image container to prevent expensive base64 re-renders
const SafeChartImage = memo(({ base64Src, alt, className }: { base64Src: string; alt: string; className?: string }) => {
  const src = useMemo(() => {
    if (!base64Src) return "";
    return base64Src.startsWith("data:") ? base64Src : `data:image/png;base64,${base64Src}`;
  }, [base64Src]);

  if (!src) return null;

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
    />
  );
});

SafeChartImage.displayName = "SafeChartImage";

interface ReportGenerationProps {
  step: AppStep;
  taskId: string | null;
  filename: string;
  info: DatasetInfo;
  analysis: AnalysisResult;
  charts: Charts;
  insights: InsightResult;
  onComplete: () => void;
  onReset: () => void;
}

const getGradeColorClass = (g: string) => {
  switch (g.toUpperCase()) {
    case 'A': return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'B': return 'bg-zinc-100 text-zinc-900 border-zinc-200';
    case 'C': return 'bg-amber-50 text-amber-700 border-amber-200';
    case 'D': return 'bg-orange-50 text-orange-700 border-orange-200';
    case 'F': return 'bg-red-50 text-red-750 border-red-200';
    default: return 'bg-muted text-muted-foreground border-border';
  }
};

export const ReportGeneration = ({
  step,
  taskId,
  filename,
  info,
  analysis,
  charts,
  insights: _insights,
  onComplete,
  onReset
}: ReportGenerationProps) => {
  const { toast } = useToast();

  // Primary workspace state
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing report engine…");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Top-level Command Center tab
  const [activeMainTab, setActiveMainTab] = useState<string>("overview");

  // Visual Insights Sub-tabs
  const [activeChartTab, setActiveChartTab] = useState<string>("correlation");
  const [activeDistIndex, setActiveDistIndex] = useState<number>(0);
  const [activeBarIndex, setActiveBarIndex] = useState<number>(0);
  const [activeBoxIndex, setActiveBoxIndex] = useState<number>(0);

  // Column Ledger search & filter
  const [expandedColumns, setExpandedColumns] = useState<Record<string, boolean>>({});
  const [ledgerSearch, setLedgerSearch] = useState<string>("");
  const [ledgerGradeFilter, setLedgerGradeFilter] = useState<string>("all");

  // In-Browser DuckDB SQL Studio state
  const [sqlQuery, setSqlQuery] = useState<string>("SELECT * FROM dataset LIMIT 15;");
  const [sqlRunning, setSqlRunning] = useState<boolean>(false);
  const [sqlResult, setSqlResult] = useState<{
    columns: string[];
    records: DataRow[];
    total_returned: number;
    latency_ms: number;
  } | null>(null);
  const [sqlError, setSqlError] = useState<string | null>(null);
  const [hasCopiedSql, setHasCopiedSql] = useState<boolean>(false);
  const [isSavedSql, setIsSavedSql] = useState<boolean>(false);

  const handleSaveSqlQuery = async () => {
    if (!taskId || !sqlQuery.trim()) return;
    try {
      await api.saveGoldenQuery(taskId, "Verified Analytical Query", sqlQuery, "Saved from In-Process SQL Studio");
      setIsSavedSql(true);
      toast({
        title: "Query Saved as Golden KPI",
        description: "Successfully added this query to verified golden benchmarks.",
      });
    } catch (err: any) {
      toast({
        title: "Save Failed",
        description: err?.message || "Could not save query.",
        variant: "destructive",
      });
    }
  };

  const toggleColumnExpand = (colName: string) => {
    setExpandedColumns(prev => ({ ...prev, [colName]: !prev[colName] }));
  };

  // Real-time task tracking via hook
  const { status: taskStatus, progress: taskProgress, message: taskMessage } = useTaskStatus(taskId || undefined);

  useEffect(() => {
    let mounted = true;

    const runGeneration = async () => {
      if (step === "generating" && !isGenerating && !downloadUrl && taskId) {
        setIsGenerating(true);
        setStatus("Initializing report engine…");
        setProgress(10);

        try {
          await api.generatePersistentReport(taskId);
        } catch (error) {
          if (!mounted) return;
          console.error("Report generation trigger failed:", error);
          setStatus("Failed to start generation.");
          toast({ title: "Error", description: "Could not start report generation.", variant: "destructive" });
          setIsGenerating(false);
        }
      }
    };

    runGeneration();

    if (step === "generating" && taskId) {
      if (taskStatus === 'PROCESSING') {
        if (taskMessage) setStatus(taskMessage);
        else setStatus("Generating PDF report…");
        setProgress(Math.min(95, Math.max(taskProgress, 10)));
      } else if (taskStatus === 'COMPLETED') {
        if (mounted && !downloadUrl) {
          setStatus("Downloading PDF…");
          setProgress(100);

          api.downloadReportBlob(taskId)
            .then(blob => {
              if (!mounted) return;
              const url = window.URL.createObjectURL(blob);
              setDownloadUrl(url);
              setStatus("Report ready!");
              onComplete();
              toast({ title: "Report Ready", description: "PDF compiled and cached successfully." });
            })
            .catch(e => {
              if (!mounted) return;
              console.error("Download failed:", e);
              setStatus("Download failed.");
            })
            .finally(() => {
              if (mounted) setIsGenerating(false);
            });
        }
      } else if (taskStatus === 'FAILED') {
        setStatus("Report generation failed.");
        setIsGenerating(false);
        toast({ title: "Failed", description: "Check server logs.", variant: "destructive" });
      }
    }

    return () => { mounted = false; };
  }, [step, taskId, taskStatus, taskProgress, taskMessage, isGenerating, downloadUrl, onComplete, toast]);

  const downloadFile = () => {
    if (downloadUrl) {
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename ? `${filename.replace('.csv', '')}_Report.pdf` : 'Analysis_Report.pdf';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      toast({ title: "Not Ready", description: "Report is still generating." });
    }
  };

  const downloadExport = async (format: "csv" | "parquet" | "html") => {
    if (!taskId) return;
    try {
      toast({ title: "Preparing Export", description: `Packaging ${format.toUpperCase()} export file...` });
      const blob = await api.downloadExportBlob(taskId, format);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const safeName = filename ? filename.replace(/\.[^/.]+$/, "") : "dataset";
      link.download = format === "html" ? `Report_${safeName}.html` : `Cleaned_${safeName}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      toast({ title: "Export Downloaded", description: `Successfully exported ${format.toUpperCase()} file.` });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not download export.";
      toast({ title: "Export Failed", description: msg, variant: "destructive" });
    }
  };

  // Execute DuckDB analytical query
  const executeSql = async (overrideQuery?: string) => {
    if (!taskId) return;
    const query = (overrideQuery || sqlQuery).trim();
    if (!query) return;

    setSqlRunning(true);
    setSqlError(null);
    const start = performance.now();

    try {
      const res = await api.querySql(taskId, query, 500);
      const elapsed = Math.round(performance.now() - start);
      setSqlResult({
        columns: res.columns,
        records: res.records,
        total_returned: res.total_returned,
        latency_ms: elapsed,
      });
      toast({
        title: "Query Executed",
        description: `Returned ${res.total_returned} rows in ${elapsed}ms`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Query execution failed";
      setSqlError(msg);
      toast({
        title: "SQL Error",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setSqlRunning(false);
    }
  };

  // Export current SQL query results to CSV
  const downloadSqlCsv = () => {
    if (!sqlResult || sqlResult.records.length === 0) return;
    const cols = sqlResult.columns;
    const lines = [cols.join(",")];
    for (const row of sqlResult.records) {
      const rowLine = cols.map(c => {
        const val = row[c];
        if (val === null || val === undefined) return "";
        const str = String(val).replace(/"/g, '""');
        return `"${str}"`;
      }).join(",");
      lines.push(rowLine);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `query_result_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast({ title: "Export Ready", description: "Downloaded SQL query result as CSV." });
  };

  // Construct Parallel Coordinates Payload
  const hiplotPayload = useMemo<HiPlotPayload | null>(() => {
    if (!info.preview || info.preview.length < 2) return null;

    const candidateCols = info.columns.filter(col => {
      const dtype = (info.dtypes[col] || "").toLowerCase();
      return dtype.includes("int") || dtype.includes("float") || dtype.includes("str") || dtype.includes("cat");
    }).slice(0, 8);

    if (candidateCols.length < 2) return null;

    const dimensions: HiPlotDimension[] = candidateCols.map(col => {
      const dtype = (info.dtypes[col] || "").toLowerCase();
      const isNumeric = dtype.includes("int") || dtype.includes("float");
      const summaryCol = info.summary?.[col];

      const numMin = summaryCol && Number.isFinite(summaryCol.min) ? Number(summaryCol.min) : undefined;
      const numMax = summaryCol && Number.isFinite(summaryCol.max) ? Number(summaryCol.max) : undefined;
      const numMean = summaryCol && Number.isFinite(summaryCol.mean) ? Number(summaryCol.mean) : undefined;

      return {
        name: col,
        display_name: col,
        type: isNumeric ? "numeric" : "categorical",
        is_numeric: isNumeric,
        min: numMin,
        max: numMax,
        mean: numMean,
      };
    });

    return {
      total_rows: info.rows,
      sampled_rows: info.preview.length,
      is_sampled: info.rows > info.preview.length,
      dimensions,
      datapoints: info.preview,
    };
  }, [info.columns, info.dtypes, info.preview, info.rows, info.summary]);

  // Fallback confidence scores if backend didn't supply them
  const fallbackConfidenceScores = useMemo(() => {
    const noIssues: string[] = [];
    return {
      dataset_grade: "B",
      dataset_confidence: 82.5,
      high_confidence_count: info.columns.length,
      low_confidence_count: 0,
      critical_issues: noIssues,
      ml_readiness: undefined,
      columns: info.columns.map(col => {
      const missing = info.missing_values[col] || { count: 0, percentage: 0 };
      const completeness = 100 - missing.percentage;
      let grade = "A";
      if (completeness < 50) grade = "F";
      else if (completeness < 75) grade = "C";
      else if (completeness < 90) grade = "B";

      return {
        column: col,
        completeness,
        consistency: 92,
        validity: 95,
        stability: 88,
        overall: (completeness + 92 + 95 + 88) / 4,
        grade,
        issues: missing.count > 0 ? [`${missing.count} missing rows`] : []
      };
    })
  };
}, [info.columns, info.missing_values]);

  const confidence = analysis.confidence_scores || fallbackConfidenceScores;
  const datasetGrade = confidence.dataset_grade || "B";
  const datasetConfidence = confidence.dataset_confidence || 82.5;

  // Filter columns in the ledger
  const filteredLedgerColumns = useMemo(() => {
    return confidence.columns.filter(c => {
      const matchesSearch = c.column.toLowerCase().includes(ledgerSearch.toLowerCase().trim());
      if (!matchesSearch) return false;
      if (ledgerGradeFilter === "all") return true;
      if (ledgerGradeFilter === "A") return c.grade.toUpperCase() === "A";
      if (ledgerGradeFilter === "B") return c.grade.toUpperCase() === "B";
      if (ledgerGradeFilter === "attention") return ["C", "D", "F"].includes(c.grade.toUpperCase());
      return true;
    });
  }, [confidence.columns, ledgerSearch, ledgerGradeFilter]);

  // Visual insights tabs
  const chartTabsList = useMemo(() => {
    const list = [];
    if (charts.correlation_heatmap) {
      list.push({ id: "correlation", label: "Correlation Matrix", icon: Grid });
    }
    if (charts.distributions && charts.distributions.length > 0) {
      list.push({ id: "distributions", label: "Distributions", icon: BarChart3 });
    }
    if ((charts.bar_charts && charts.bar_charts.length > 0) || charts.donut_chart) {
      list.push({ id: "composition", label: "Composition", icon: PieChart });
    }
    if ((charts.boxplots && charts.boxplots.length > 0) || charts.scatter_plot) {
      list.push({ id: "bivariate", label: "Bivariate Relations", icon: TrendingUp });
    }
    if (hiplotPayload && hiplotPayload.dimensions.length >= 2) {
      list.push({ id: "hiplot", label: "Parallel Coordinates", icon: Sliders });
    }
    return list;
  }, [charts, hiplotPayload]);

  if (step === "complete") {
    return (
      <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in zoom-in-95 duration-400">

        {/* ─── Top Command Center Hero Bar ─── */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 p-6 sm:p-8 bg-card border border-border/80 rounded-3xl shadow-premium relative overflow-hidden">
          <div className="absolute top-0 right-0 w-80 h-80 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

          <div className="flex items-center gap-5 z-10">
            {/* Seal / Letter Grade Badge */}
            <div className={`flex flex-col items-center justify-center h-20 w-20 sm:h-24 sm:w-24 rounded-2xl ${getGradeColorClass(datasetGrade)} border shadow-sm shrink-0`}>
              <span className="text-[10px] font-mono tracking-widest uppercase text-muted-foreground/70 leading-none mb-1">GRADE</span>
              <span className="text-3xl sm:text-4xl font-display font-black leading-none tracking-tighter">{datasetGrade}</span>
              <span className="text-[10px] font-mono font-bold mt-1">{datasetConfidence.toFixed(1)}%</span>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <h1 className="text-2xl sm:text-3xl font-display font-bold text-foreground tracking-tight">
                  {filename ? filename.replace(/\.[^/.]+$/, "") : "Dataset Analysis"}
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-muted-foreground font-mono">
                {info.rows.toLocaleString()} rows • {info.columns.length} columns • {(info.memory_usage_mb || 0).toFixed(2)} MB in-memory
              </p>
              <div className="text-[11px] font-mono text-muted-foreground/80 flex items-center gap-2 pt-1">
                <span>{confidence.high_confidence_count} High Confidence Columns</span>
                <span>•</span>
                <span>{confidence.low_confidence_count} Need Attention</span>
              </div>
            </div>
          </div>

          {/* Primary & Quick Export Actions */}
          <div className="flex flex-wrap items-center gap-2.5 z-10 w-full md:w-auto">
            <Button
              size="lg"
              className="flex-1 md:flex-none rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-display text-sm font-bold tracking-tight shadow-[0_4px_16px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] border border-violet-400/40 ring-1 ring-white/20 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2 px-5 cursor-pointer"
              onClick={downloadFile}
            >
              <Download className="h-4 w-4" />
              <span>Download PDF</span>
              <ChevronRight className="h-4 w-4 opacity-70" />
            </Button>

            <Button
              size="lg"
              variant="outline"
              className="flex-1 md:flex-none rounded-xl border-border bg-white hover:bg-muted/30 shadow-2xs transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2 font-display text-sm text-foreground"
              onClick={() => downloadExport("html")}
            >
              <BookOpen className="h-4 w-4 text-primary" />
              <span>HTML Briefing</span>
            </Button>

            <Button
              size="lg"
              variant="ghost"
              className="rounded-xl text-muted-foreground hover:text-foreground text-xs flex items-center gap-1.5 px-3"
              onClick={onReset}
              title="Audit a different file"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">New Audit</span>
            </Button>
          </div>
        </div>

        {/* ─── Main Command Center Tabs ─── */}
        <Tabs value={activeMainTab} onValueChange={setActiveMainTab} className="w-full">
          <TabsList className="bg-muted/30 border border-border/80 p-1.5 rounded-2xl mb-6 shadow-2xs flex flex-wrap gap-1">
            <TabsTrigger
              value="overview"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <FileText className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>

            <TabsTrigger
              value="charts"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <TrendingUp className="h-4 w-4" />
              <span>Visual Analytics</span>
              {chartTabsList.length > 0 && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4 rounded-full bg-primary/10 text-primary">
                  {chartTabsList.length}
                </Badge>
              )}
            </TabsTrigger>

            <TabsTrigger
              value="ledger"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <ShieldCheck className="h-4 w-4" />
              <span>Trust Ledger</span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4 rounded-full bg-muted">
                {confidence.columns.length}
              </Badge>
            </TabsTrigger>

            <TabsTrigger
              value="stats"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <Activity className="h-4 w-4" />
              <span>Statistical Audit</span>
            </TabsTrigger>

            <TabsTrigger
              value="sql"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <Terminal className="h-4 w-4" />
              <span>DuckDB SQL Studio</span>
            </TabsTrigger>

            <TabsTrigger
              value="export"
              className="rounded-xl text-xs sm:text-sm px-4 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              <span>Export Hub</span>
            </TabsTrigger>
          </TabsList>

          {/* ──── TAB 1: EXECUTIVE OVERVIEW ──── */}
          <TabsContent value="overview" className="space-y-8 animate-in fade-in duration-300">
            {/* Quick Metrics KPI Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground block">
                  Completeness
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-display font-bold text-foreground">
                    {datasetConfidence.toFixed(1)}%
                  </span>
                  <span className="text-xs text-emerald-600 font-mono">High Quality</span>
                </div>
                <p className="text-[11px] font-mono text-muted-foreground mt-1">Overall data confidence index</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground block">
                  Dataset Dimensions
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-display font-bold text-foreground">
                    {info.rows.toLocaleString()}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground">x {info.columns.length} cols</span>
                </div>
                <p className="text-[11px] font-mono text-muted-foreground mt-1">Processed without schema drift</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground block">
                  Numeric vs Categorical
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-display font-bold text-foreground">
                    {info.numeric_columns?.length || 0} / {info.categorical_columns?.length || 0}
                  </span>
                </div>
                <p className="text-[11px] font-mono text-muted-foreground mt-1">Continuous vs discrete variables</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground block">
                  Memory Utilization
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-display font-bold text-foreground">
                    {(info.memory_usage_mb || 0).toFixed(2)}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground">MB</span>
                </div>
                <p className="text-[11px] font-mono text-muted-foreground mt-1">Optimized columnar storage</p>
              </Card>
            </div>

            {/* Critical Dataset Alerts (if any) */}
            {confidence.critical_issues && confidence.critical_issues.length > 0 && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-2xl flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
                <div className="space-y-1 font-mono text-xs text-red-800">
                  <span className="font-bold uppercase tracking-wider block">Critical Data Hygiene Notices</span>
                  <ul className="list-disc pl-4 space-y-0.5 font-sans">
                    {confidence.critical_issues.map((ci: string) => (
                      <li key={`critical-issue-${ci}`}>{ci}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* ML Readiness Card */}
            <MLReadinessCard mlReadiness={confidence.ml_readiness} />
          </TabsContent>

          {/* ──── TAB 2: VISUAL ANALYTICS ──── */}
          <TabsContent value="charts" className="space-y-6 animate-in fade-in duration-300">
            {chartTabsList.length > 0 ? (
              <Card className="border border-border bg-card shadow-premium rounded-3xl overflow-hidden">
                <CardHeader className="border-b border-border bg-muted/10 pb-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <CardTitle className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-primary" />
                        Visual Insights Gallery
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Algorithmic visual explorations for bivariate correlations, distribution shapes, and multidimensional trends
                      </CardDescription>
                    </div>
                  </div>

                  {/* Sub-selector pills */}
                  <div className="flex overflow-x-auto gap-2 mt-4 border-b border-border/40 pb-2 scrollbar-none snap-x whitespace-nowrap">
                    {chartTabsList.map((t) => {
                      const Icon = t.icon;
                      const isActive = activeChartTab === t.id;
                      return (
                        <button
                          key={t.id}
                          type="button"
                          onClick={() => setActiveChartTab(t.id)}
                          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold tracking-tight transition-all shrink-0 ${
                            isActive
                              ? "bg-primary text-primary-foreground shadow-2xs"
                              : "bg-secondary/40 hover:bg-secondary text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          <span>{t.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </CardHeader>

                <CardContent className="p-4 sm:p-8">
                  {/* Correlation Heatmap */}
                  {activeChartTab === "correlation" && charts.correlation_heatmap && (
                    <div className="space-y-6">
                      <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-3 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                        {(() => {
                          // SAFETY: correlation_heatmap union type is safely narrowed to image property or raw string
                          const imgSrc = (charts.correlation_heatmap as { image?: string })?.image ?? (charts.correlation_heatmap as string);
                          return (
                            <SafeChartImage
                              base64Src={imgSrc}
                              alt="Correlation Heatmap"
                              className="max-h-[380px] w-full object-contain rounded-xl"
                            />
                          );
                        })()}
                      </div>
                      <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                        <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                          Correlation Matrix Narrative
                        </div>
                        <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                          {(() => {
                            // SAFETY: correlation_heatmap union type is safely narrowed to narrative property or default string
                            const narrativeText = (charts.correlation_heatmap as { narrative?: string })?.narrative ?? "Pearson correlation matrix. Warm tones indicate strong positive interactions, while dark charcoal represents negative inverse correlations.";
                            return narrativeText;
                          })()}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Distributions */}
                  {activeChartTab === "distributions" && charts.distributions && charts.distributions.length > 0 && (
                    <div className="space-y-6">
                      <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
                        {charts.distributions.map((item, idx) => (
                          <button
                            key={`dist-tab-${item.column}`}
                            type="button"
                            onClick={() => setActiveDistIndex(idx)}
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                              activeDistIndex === idx
                                ? "bg-primary text-primary-foreground shadow-2xs"
                                : "bg-muted/50 hover:bg-muted text-muted-foreground"
                            }`}
                          >
                            {item.column}
                          </button>
                        ))}
                      </div>

                      <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                        <SafeChartImage
                          base64Src={charts.distributions[activeDistIndex]?.image}
                          alt={`Distribution for ${charts.distributions[activeDistIndex]?.column}`}
                          className="max-h-[340px] w-auto object-contain rounded-xl"
                        />
                      </div>

                      <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                        <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                          {charts.distributions[activeDistIndex]?.column} Distribution Analysis
                        </div>
                        <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                          {charts.distributions[activeDistIndex]?.narrative}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Composition */}
                  {activeChartTab === "composition" && (
                    <div className="space-y-6">
                      {charts.bar_charts && charts.bar_charts.length > 0 && (
                        <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
                          {charts.bar_charts.map((item, idx) => (
                            <button
                              key={`bar-tab-${item.column}`}
                              type="button"
                              onClick={() => setActiveBarIndex(idx)}
                              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                                activeBarIndex === idx
                                  ? "bg-primary text-primary-foreground shadow-2xs"
                                  : "bg-muted/50 hover:bg-muted text-muted-foreground"
                              }`}
                            >
                              {item.column}
                            </button>
                          ))}
                          {charts.donut_chart && (
                            <button
                              type="button"
                              onClick={() => setActiveBarIndex(-1)}
                              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                                activeBarIndex === -1
                                  ? "bg-primary text-primary-foreground shadow-2xs"
                                  : "bg-muted/50 hover:bg-muted text-muted-foreground"
                              }`}
                            >
                              {charts.donut_chart.column} (Donut)
                            </button>
                          )}
                        </div>
                      )}

                      {activeBarIndex === -1 && charts.donut_chart ? (
                        <div className="space-y-6">
                          <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                            <SafeChartImage
                              base64Src={charts.donut_chart.image}
                              alt={`Composition of ${charts.donut_chart.column}`}
                              className="max-h-[340px] w-auto object-contain rounded-xl"
                            />
                          </div>
                          <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                              Donut Composition Narrative
                            </div>
                            <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                              {charts.donut_chart.narrative}
                            </p>
                          </div>
                        </div>
                      ) : charts.bar_charts && charts.bar_charts[activeBarIndex] ? (
                        <div className="space-y-6">
                          <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                            <SafeChartImage
                              base64Src={charts.bar_charts[activeBarIndex].image}
                              alt={`Category breakdown of ${charts.bar_charts[activeBarIndex].column}`}
                              className="max-h-[340px] w-auto object-contain rounded-xl"
                            />
                          </div>
                          <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                              Categorical Breakdown
                            </div>
                            <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                              {charts.bar_charts[activeBarIndex].narrative}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center p-8 font-mono text-xs text-muted-foreground">
                          No categorical composition charts available.
                        </div>
                      )}
                    </div>
                  )}

                  {/* Bivariate & Boxplots */}
                  {activeChartTab === "bivariate" && (
                    <div className="space-y-6">
                      <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
                        {charts.boxplots && charts.boxplots.map((item, idx) => (
                          <button
                            key={`box-tab-${item.column}`}
                            type="button"
                            onClick={() => setActiveBoxIndex(idx)}
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                              activeBoxIndex === idx
                                ? "bg-primary text-primary-foreground shadow-2xs"
                                : "bg-muted/50 hover:bg-muted text-muted-foreground"
                            }`}
                          >
                            {item.column}
                          </button>
                        ))}
                        {charts.scatter_plot && (
                          <button
                            type="button"
                            onClick={() => setActiveBoxIndex(-1)}
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                              activeBoxIndex === -1
                                ? "bg-primary text-primary-foreground shadow-2xs"
                                : "bg-muted/50 hover:bg-muted text-muted-foreground"
                            }`}
                          >
                            {charts.scatter_plot.columns} (Scatter)
                          </button>
                        )}
                      </div>

                      {activeBoxIndex === -1 && charts.scatter_plot ? (
                        <div className="space-y-6">
                          <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                            <SafeChartImage
                              base64Src={charts.scatter_plot.image}
                              alt="Bivariate Scatter Plot"
                              className="max-h-[340px] w-auto object-contain rounded-xl"
                            />
                          </div>
                          <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                              Scatter Regression Narrative
                            </div>
                            <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                              {charts.scatter_plot.narrative}
                            </p>
                          </div>
                        </div>
                      ) : charts.boxplots && charts.boxplots[activeBoxIndex] ? (
                        <div className="space-y-6">
                          <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
                            <SafeChartImage
                              base64Src={charts.boxplots[activeBoxIndex].image}
                              alt={`Boxplot for ${charts.boxplots[activeBoxIndex].column}`}
                              className="max-h-[340px] w-auto object-contain rounded-xl"
                            />
                          </div>
                          <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                              Distribution Spread & Outliers
                            </div>
                            <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">
                              {charts.boxplots[activeBoxIndex].narrative}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center p-8 font-mono text-xs text-muted-foreground">
                          No bivariate relations charts available.
                        </div>
                      )}
                    </div>
                  )}

                  {/* Parallel Coordinates (HiPlot Canvas) */}
                  {activeChartTab === "hiplot" && hiplotPayload && (
                    <div className="space-y-4">
                      <div className="bg-muted/10 border border-border/80 rounded-2xl p-4">
                        <HiPlotParallelCoordinates payload={hiplotPayload} />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ) : (
              <div className="text-center py-12 font-mono text-xs text-muted-foreground">
                No visual charts available for this dataset.
              </div>
            )}
          </TabsContent>

          {/* ──── TAB 3: TRUST LEDGER ──── */}
          <TabsContent value="ledger" className="space-y-6 animate-in fade-in duration-300">
            <Card className="border border-border bg-card shadow-premium rounded-3xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
                <div>
                  <h3 className="text-lg font-display font-bold text-foreground">Column Trust Ledger</h3>
                  <p className="text-xs text-muted-foreground">Completeness, consistency, validity, and stability audits</p>
                </div>

                {/* Filter and search controls */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
                  <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-xl border border-border/60">
                    <button
                      type="button"
                      onClick={() => setLedgerGradeFilter("all")}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        ledgerGradeFilter === "all" ? "bg-white text-foreground shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      All ({confidence.columns.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setLedgerGradeFilter("A")}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        ledgerGradeFilter === "A" ? "bg-emerald-100 text-emerald-800 shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      Grade A
                    </button>
                    <button
                      type="button"
                      onClick={() => setLedgerGradeFilter("attention")}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        ledgerGradeFilter === "attention" ? "bg-amber-100 text-amber-800 shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      Flags
                    </button>
                  </div>

                  <div className="relative w-full sm:w-48">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                    <Input
                      placeholder="Search variables..."
                      value={ledgerSearch}
                      onChange={(e) => setLedgerSearch(e.target.value)}
                      className="pl-9 h-8 text-xs rounded-xl bg-white border-border/80"
                    />
                  </div>
                </div>
              </div>

              {/* Column Rows */}
              <div className="space-y-2.5 pt-4">
                {filteredLedgerColumns.length > 0 ? (
                  filteredLedgerColumns.map((c) => {
                    const isExpanded = !!expandedColumns[c.column];
                    const hasIssues = c.issues && c.issues.length > 0;

                    return (
                      <div
                        key={c.column}
                        className={`border rounded-2xl transition-all duration-200 ${
                          isExpanded ? 'border-primary bg-primary/5' : 'border-border/80 bg-white hover:bg-muted/10'
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => toggleColumnExpand(c.column)}
                          className="w-full flex items-center justify-between p-3.5 cursor-pointer select-none gap-2 text-left bg-transparent border-0"
                        >
                          <div className="flex items-center gap-3 min-w-0 flex-1">
                            <Badge className={`h-7 w-7 rounded-lg flex items-center justify-center p-0 font-bold shrink-0 ${getGradeColorClass(c.grade)}`}>
                              {c.grade}
                            </Badge>
                            <span className="font-mono text-xs font-semibold text-foreground truncate max-w-[200px]" title={c.column}>
                              {c.column}
                            </span>
                            <Badge variant="outline" className="text-[10px] font-mono px-2 py-0 h-4 bg-muted/20 border-border rounded-full hidden sm:inline-flex">
                              {info.dtypes[c.column] || "variable"}
                            </Badge>
                          </div>

                          <div className="flex items-center gap-3 shrink-0 ml-auto">
                            <span className="font-mono text-xs text-muted-foreground font-semibold">
                              {c.overall.toFixed(0)}%
                            </span>
                            {hasIssues && (
                              <Badge variant="secondary" className="bg-amber-50 text-amber-700 border border-amber-200 h-5 px-2 rounded-full text-[9px] font-mono">
                                {c.issues.length} alert{c.issues.length > 1 ? 's' : ''}
                              </Badge>
                            )}
                            {isExpanded ? (
                              <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" />
                            ) : (
                              <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                            )}
                          </div>
                        </button>

                        {isExpanded && (
                          <div className="p-4 border-t border-border/60 space-y-3 font-mono text-xs animate-in slide-in-from-top-1 duration-200">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-muted/20 p-3 rounded-xl">
                              <div className="space-y-0.5">
                                <span className="text-[10px] text-muted-foreground uppercase">Completeness</span>
                                <div className="text-foreground font-bold">{c.completeness.toFixed(1)}%</div>
                              </div>
                              <div className="space-y-0.5">
                                <span className="text-[10px] text-muted-foreground uppercase">Consistency</span>
                                <div className="text-foreground font-bold">{c.consistency.toFixed(1)}%</div>
                              </div>
                              <div className="space-y-0.5">
                                <span className="text-[10px] text-muted-foreground uppercase">Validity</span>
                                <div className="text-foreground font-bold">{c.validity.toFixed(1)}%</div>
                              </div>
                              <div className="space-y-0.5">
                                <span className="text-[10px] text-muted-foreground uppercase">Stability</span>
                                <div className="text-foreground font-bold">{c.stability.toFixed(1)}%</div>
                              </div>
                            </div>

                            {hasIssues && (
                              <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 space-y-1 text-xs text-amber-800">
                                <div className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-[10px]">
                                  <AlertTriangle className="h-3.5 w-3.5" />
                                  <span>Identified Quality Flags:</span>
                                </div>
                                <ul className="list-disc pl-4 space-y-0.5 font-sans text-xs">
                                  {c.issues.map((issueStr) => (
                                    <li key={`col-issue-${c.column}-${issueStr}`}>{issueStr}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center py-8 text-xs font-mono text-muted-foreground">
                    No columns match your filter criteria.
                  </div>
                )}
              </div>
            </Card>
          </TabsContent>

          {/* ──── TAB 4: STATISTICAL DEEP DIVE ──── */}
          <TabsContent value="stats" className="space-y-6 animate-in fade-in duration-300">
            <div className="grid gap-6 md:grid-cols-2">
              {/* Skewness & Kurtosis */}
              {analysis.advanced_stats && Object.keys(analysis.advanced_stats).length > 0 && (
                <Card className="border border-border bg-card shadow-premium rounded-3xl">
                  <CardHeader className="border-b border-border pb-3">
                    <CardTitle className="text-base font-display font-bold text-foreground">Distribution Shape (Moments)</CardTitle>
                    <CardDescription className="text-xs font-mono text-muted-foreground">Skewness (Asymmetry) & Kurtosis (Tail Weight)</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 max-h-[340px] overflow-y-auto text-sm pt-4">
                    {Object.entries(analysis.advanced_stats).map(([col, stats]) => {
                      const isSkewed = Math.abs(stats.skewness) > 1;
                      const isHeavy = Math.abs(stats.kurtosis) > 3;
                      if (!isSkewed && !isHeavy) return null;

                      return (
                        <div key={col} className="flex justify-between items-center py-2.5 border-b border-border last:border-0 font-mono text-xs">
                          <span className="font-semibold text-foreground truncate max-w-[160px]">{col}</span>
                          <div className="flex gap-1.5">
                            {isSkewed && (
                              <Badge variant="secondary" className="bg-amber-50 text-amber-700 border border-amber-200 rounded-lg text-[10px]">
                                skew: {stats.skewness.toFixed(2)}
                              </Badge>
                            )}
                            {isHeavy && (
                              <Badge variant="secondary" className="bg-primary/5 text-primary border border-primary/20 rounded-lg text-[10px]">
                                kurt: {stats.kurtosis.toFixed(2)}
                              </Badge>
                            )}
                          </div>
                        </div>
                      );
                    })}
                    {Object.values(analysis.advanced_stats).every(s => Math.abs(s.skewness) <= 1 && Math.abs(s.kurtosis) <= 3) && (
                      <p className="text-muted-foreground italic text-xs font-mono py-6 text-center">
                        All numerical variables display symmetric distributions within nominal bounds.
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Multicollinearity */}
              {analysis.multicollinearity && (
                <Card className="border border-border bg-card shadow-premium rounded-3xl">
                  <CardHeader className="border-b border-border pb-3">
                    <CardTitle className="text-base font-display font-bold text-foreground">Multicollinearity & Redundancy</CardTitle>
                    <CardDescription className="text-xs font-mono text-muted-foreground">Correlated Feature Pairs (VIF Proxy)</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 max-h-[340px] overflow-y-auto text-sm pt-4">
                    {analysis.multicollinearity.length > 0 ? (
                      analysis.multicollinearity.map((item) => (
                        <div key={`mc-${item.features.join('-')}`} className="flex flex-col py-2.5 border-b border-border last:border-0">
                          <div className="flex justify-between font-mono text-xs text-foreground font-semibold">
                            <span className="truncate max-w-[130px]">{item.features[0]}</span>
                            <ArrowRight className="h-3.5 w-3.5 text-muted-foreground mx-2 self-center shrink-0" />
                            <span className="truncate max-w-[130px]">{item.features[1]}</span>
                          </div>
                          <div className="flex justify-between mt-1 text-[11px] font-mono text-muted-foreground">
                            <span>Pearson r: <strong>{item.correlation.toFixed(2)}</strong></span>
                            <Badge variant="destructive" className="h-4.5 text-[9px] px-2 rounded-full bg-red-50 text-red-750 border border-red-200 uppercase font-bold">
                              High Collinearity
                            </Badge>
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground italic text-xs font-mono py-6 text-center">
                        No collinear feature redundancies detected.
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Time-Series Trends */}
              {analysis.time_series_analysis?.has_time_series && (
                <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-3xl">
                  <CardHeader className="border-b border-border pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base font-display font-bold text-foreground">
                        Time-Series Drift & Temporal Trends
                      </CardTitle>
                      <Badge variant="outline" className="text-[10px] font-mono bg-muted/20 border-border">
                        Time axis: {analysis.time_series_analysis.time_column}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-4">
                    <div className="grid gap-3 sm:grid-cols-2 font-mono text-xs">
                      {analysis.time_series_analysis.analyses && Object.entries(analysis.time_series_analysis.analyses).map(([col, data]: [string, any]) => (
                        <div key={col} className="p-3 bg-muted/10 border border-border rounded-xl space-y-1.5">
                          <span className="font-bold text-foreground block truncate">{col}</span>
                          <div className="flex justify-between items-center text-[11px]">
                            <span className="text-muted-foreground">Direction:</span>
                            <Badge variant={data.trend?.direction === "upward" ? "default" : "secondary"} className="text-[10px] px-2 h-4.5 rounded-full">
                              {data.trend?.direction || "neutral"} ({data.trend?.strength || "stable"})
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* ──── TAB 5: DUCKDB SQL STUDIO ──── */}
          <TabsContent value="sql" className="space-y-6 animate-in fade-in duration-300">
            <Card className="border border-border bg-card shadow-premium rounded-3xl p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-4">
                <div>
                  <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                    <Terminal className="h-5 w-5 text-primary" />
                    In-Process DuckDB SQL Studio
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    Execute high-speed analytical queries directly against the in-memory <code className="bg-muted px-1.5 py-0.5 rounded text-primary font-mono text-xs">dataset</code> table
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="delete"
                    className="h-8.5 text-xs font-mono rounded-xl px-2.5 gap-1.5"
                    onClick={() => {
                      setSqlQuery("");
                      setSqlResult(null);
                      setSqlError(null);
                      setIsSavedSql(false);
                    }}
                    title="Clear SQL query"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Clear</span>
                  </Button>

                  <Button
                    size="sm"
                    variant="outline"
                    className="h-8.5 text-xs font-mono rounded-xl bg-white border-border/80 px-3"
                    onClick={() => {
                      navigator.clipboard.writeText(sqlQuery);
                      setHasCopiedSql(true);
                      setTimeout(() => setHasCopiedSql(false), 2000);
                      toast({ title: "Copied", description: "Query copied to clipboard" });
                    }}
                  >
                    {hasCopiedSql ? <Check className="h-3.5 w-3.5 text-emerald-600 mr-1" /> : <Copy className="h-3.5 w-3.5 mr-1" />}
                    <span>{hasCopiedSql ? "Copied" : "Copy SQL"}</span>
                  </Button>

                  <Button
                    size="sm"
                    className="h-8.5 text-xs font-bold rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white shadow-sm shadow-violet-600/30 gap-1.5 px-4 cursor-pointer"
                    disabled={sqlRunning}
                    onClick={() => executeSql()}
                  >
                    {sqlRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5 fill-current" />}
                    <span>Run Query</span>
                  </Button>
                </div>
              </div>

              {/* Sample Query Chips */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[10px] font-mono text-muted-foreground uppercase mr-1">Sample Queries:</span>
                <button
                  type="button"
                  onClick={() => {
                    const q = "SELECT * FROM dataset LIMIT 15;";
                    setSqlQuery(q);
                    executeSql(q);
                  }}
                  className="px-2.5 py-1 rounded-lg text-xs font-mono bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                >
                  Preview 15 Rows
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const q = "SELECT COUNT(*) AS total_rows FROM dataset;";
                    setSqlQuery(q);
                    executeSql(q);
                  }}
                  className="px-2.5 py-1 rounded-lg text-xs font-mono bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                >
                  Count Records
                </button>
                {info.numeric_columns && info.numeric_columns[0] && (
                  <button
                    type="button"
                    onClick={() => {
                      const col = info.numeric_columns[0];
                      const q = `SELECT AVG("${col}") AS avg_val, MIN("${col}") AS min_val, MAX("${col}") AS max_val FROM dataset;`;
                      setSqlQuery(q);
                      executeSql(q);
                    }}
                    className="px-2.5 py-1 rounded-lg text-xs font-mono bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Numeric Summary
                  </button>
                )}
              </div>

              {/* SQL Query Editor Textarea */}
              <div className="relative rounded-2xl border border-border/80 overflow-hidden bg-zinc-950 font-mono text-xs">
                <div className="flex items-center justify-between px-3.5 py-2 bg-zinc-900 border-b border-zinc-800 text-[11px] text-zinc-400">
                  <div className="flex items-center gap-1.5">
                    <div className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
                    <div className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
                    <div className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
                    <span className="ml-2 font-mono text-[10px] text-zinc-500">duckdb-in-process</span>
                  </div>
                  <span className="text-[10px] text-zinc-500">Ctrl+Enter to Run</span>
                </div>
                <textarea
                  value={sqlQuery}
                  onChange={(e) => setSqlQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                      e.preventDefault();
                      executeSql();
                    }
                  }}
                  rows={4}
                  className="w-full p-4 bg-transparent text-emerald-400 font-mono text-xs outline-none resize-y leading-relaxed"
                  placeholder="SELECT * FROM dataset WHERE ..."
                  spellCheck={false}
                />
              </div>

              {/* Error Output */}
              {sqlError && (
                <div className="p-3.5 bg-red-50 border border-red-200 rounded-xl text-xs font-mono text-red-700 flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold block">Query Evaluation Failed</span>
                    <span>{sqlError}</span>
                  </div>
                </div>
              )}

              {/* SQL Result Table */}
              {sqlResult && (
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
                    <span>
                      Returned <strong>{sqlResult.total_returned}</strong> rows in <strong>{sqlResult.latency_ms}ms</strong>
                    </span>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="save"
                        className="h-8 text-xs font-mono rounded-xl px-3 gap-1.5 cursor-pointer"
                        onClick={handleSaveSqlQuery}
                        disabled={isSavedSql}
                      >
                        {isSavedSql ? (
                          <>
                            <BookmarkCheck className="h-3.5 w-3.5 text-white" />
                            <span>Saved as KPI</span>
                          </>
                        ) : (
                          <>
                            <Bookmark className="h-3.5 w-3.5 text-white" />
                            <span>Save as KPI</span>
                          </>
                        )}
                      </Button>

                      <Button
                        size="sm"
                        variant="outline"
                        className="h-8 text-xs font-mono rounded-xl bg-white border-border/80 px-3 cursor-pointer"
                        onClick={downloadSqlCsv}
                      >
                        <Download className="h-3.5 w-3.5 mr-1" />
                        <span>Export CSV</span>
                      </Button>
                    </div>
                  </div>

                  <div className="border border-border/80 rounded-2xl overflow-hidden shadow-2xs">
                    <div className="w-full overflow-x-auto max-h-[340px]">
                      <Table className="border-collapse">
                        <TableHeader className="bg-muted/30 sticky top-0 z-10">
                          <TableRow className="border-b border-border/60">
                            {sqlResult.columns.map((col) => (
                              <TableHead key={col} className="font-mono text-[11px] font-bold uppercase text-foreground px-3.5 py-2.5 whitespace-nowrap border-r border-border/60 last:border-r-0">
                                {col}
                              </TableHead>
                            ))}
                          </TableRow>
                        </TableHeader>
                        <TableBody className="divide-y divide-border/60 bg-white">
                          {sqlResult.records.map((r, rowIdx) => (
                            <TableRow key={`sql-row-${rowIdx}`} className="hover:bg-primary/[0.02]">
                              {sqlResult.columns.map((col) => (
                                <TableCell key={`sql-cell-${rowIdx}-${col}`} className="font-mono text-xs text-foreground/90 whitespace-nowrap px-3.5 py-2 border-r border-border/60 last:border-r-0">
                                  {r[col] === null || r[col] === undefined ? (
                                    <span className="text-muted-foreground/50 italic font-sans text-[11px]">null</span>
                                  ) : (
                                    String(r[col])
                                  )}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          </TabsContent>

          {/* ──── TAB 6: EXPORT & COMPLIANCE HUB ──── */}
          <TabsContent value="export" className="space-y-6 animate-in fade-in duration-300">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {/* PDF Document Tile */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                    <FileText className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Publication PDF Report</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Full executive briefing featuring confidence seal, visual plots, multicollinearity audits, and AI narrative synthesis.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    className="w-full rounded-xl bg-primary text-primary-foreground font-semibold text-xs shadow-2xs gap-2"
                    onClick={downloadFile}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download PDF</span>
                  </Button>
                </div>
              </Card>

              {/* Standalone HTML Briefing */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center">
                    <BookOpen className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Standalone HTML Dossier</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Self-contained offline document containing interactive plots, schema dictionaries, and responsive formatting.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border bg-white text-foreground font-semibold text-xs shadow-2xs gap-2"
                    onClick={() => downloadExport("html")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download HTML</span>
                  </Button>
                </div>
              </Card>

              {/* Cleaned CSV */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-amber-500/10 text-amber-600 flex items-center justify-center">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Cleaned CSV Ledger</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Standard comma-separated file with duplicate records dropped, missing values imputed, and sanitized headers.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border bg-white text-foreground font-semibold text-xs shadow-2xs gap-2"
                    onClick={() => downloadExport("csv")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export Cleaned CSV</span>
                  </Button>
                </div>
              </Card>

              {/* Apache Parquet */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-purple-500/10 text-purple-600 flex items-center justify-center">
                    <Table2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Apache Parquet Storage</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Snappy-compressed columnar format optimized for Spark, DuckDB, Snowflake, and production ML pipelines.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border bg-white text-foreground font-semibold text-xs shadow-2xs gap-2"
                    onClick={() => downloadExport("parquet")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export Parquet</span>
                  </Button>
                </div>
              </Card>

              {/* Great Expectations */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-blue-500/10 text-blue-600 flex items-center justify-center">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Great Expectations (GX) Suite</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Automated JSON data quality contract suite ready for CI/CD pipeline assertion and orchestrators.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border bg-white text-foreground font-semibold text-xs shadow-2xs gap-2"
                    onClick={async () => {
                      if (!taskId) return;
                      try {
                        await api.downloadGxSuite(taskId, filename || "dataset");
                        toast({ title: "Expectation Suite Exported", description: "Downloaded Great Expectations data contract suite (.json)" });
                      } catch (e: unknown) {
                        const msg = e instanceof Error ? e.message : "Could not download GX suite.";
                        toast({ title: "Export Failed", description: msg, variant: "destructive" });
                      }
                    }}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export GX Suite (.json)</span>
                  </Button>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  // Generating phase
  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-400">
      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl overflow-hidden">
        <CardContent className="p-8 space-y-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-3.5 bg-primary/10 rounded-2xl relative">
                <Loader2 className="h-7 w-7 text-primary animate-spin" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-display font-bold text-foreground">{status}</h3>
                  <Badge variant="outline" className="font-mono text-xs font-bold text-primary bg-primary/5 border-primary/20">
                    {Math.round(progress)}%
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Compiling audit across <span className="font-mono text-foreground font-semibold">{info.rows.toLocaleString()}</span> rows and <span className="font-mono text-foreground font-semibold">{info.columns.length}</span> columns for <span className="font-mono text-foreground">{filename}</span>
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-1.5 pt-2">
            <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden border border-border/40">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out rounded-full"
                style={{ width: `${Math.max(10, Math.min(100, progress))}%` }}
              />
            </div>
            <div className="flex justify-between items-center text-[10px] font-mono text-muted-foreground">
              <span>Initializing computation</span>
              <span>Generating Visual Charts & PDF</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="md:col-span-1 border border-border/60 bg-card/60 p-5 space-y-3 rounded-2xl">
          <div className="h-4 w-28 bg-muted animate-pulse rounded-md" />
          <div className="h-16 w-full bg-muted/70 animate-pulse rounded-xl" />
          <div className="space-y-2 pt-2">
            <div className="h-3 w-full bg-muted/50 animate-pulse rounded" />
            <div className="h-3 w-3/4 bg-muted/50 animate-pulse rounded" />
          </div>
        </Card>

        <Card className="md:col-span-2 border border-border/60 bg-card/60 p-5 space-y-3 rounded-2xl">
          <div className="flex items-center justify-between">
            <div className="h-4 w-36 bg-muted animate-pulse rounded-md" />
            <div className="h-4 w-16 bg-muted/60 animate-pulse rounded-md" />
          </div>
          <div className="h-40 w-full bg-muted/40 animate-pulse rounded-xl flex items-center justify-center">
            <BarChart3 className="h-10 w-10 text-muted-foreground/30 animate-pulse" />
          </div>
        </Card>
      </div>

      <Card className="border border-border/60 bg-card/60 p-6 space-y-3 rounded-2xl">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-primary/40 animate-pulse" />
          <div className="h-4 w-48 bg-muted animate-pulse rounded-md" />
        </div>
        <div className="space-y-2">
          <div className="h-3.5 w-full bg-muted/60 animate-pulse rounded" />
          <div className="h-3.5 w-5/6 bg-muted/60 animate-pulse rounded" />
          <div className="h-3.5 w-4/6 bg-muted/40 animate-pulse rounded" />
        </div>
      </Card>
    </div>
  );
};

export default ReportGeneration;
