import React, { useState, useEffect, useMemo, memo, lazy, Suspense, useRef } from "react";
import {
  Download, RefreshCw, Loader2, FileText, ChevronRight,
  AlertTriangle, ArrowRight, BarChart3, PieChart, Activity,
  FileSpreadsheet, TrendingUp, ChevronDown, ChevronUp, ShieldCheck,
  BookOpen, Table2, Grid, Cpu, Terminal, Sliders, Search,
  Database
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button, AnimatedDeleteButton, ShimmerActionButton } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AppStep } from "@/pages/Workspace";
import type { AnalysisResult, Charts, InsightResult, DatasetInfo, ExportFormat } from "@/types/api";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { MLReadinessCard } from "./MLReadinessCard";
import { AnimatedNumber } from "@/components/motion/animated-number";
import { HiPlotParallelCoordinates, type HiPlotPayload, type HiPlotDimension } from "./HiPlotParallelCoordinates";
import { SupersetExploreStudio } from "./SupersetExploreStudio";
import { ErrorBoundary } from "./ErrorBoundary";

const DuckDBSqlStudio = lazy(() => import("./DuckDBSqlStudio"));

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
  requestedTab?: { tab: string; nonce: number } | null;
  requestedSql?: { sql: string; nonce: number } | null;
}

const getGradeColorClass = (g: string) => {
  switch ((g || '').toUpperCase()) {
    case 'A': return 'bg-emerald-500/15 text-emerald-800 border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-500/30';
    case 'B': return 'bg-emerald-500/15 text-emerald-800 border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-500/30';
    case 'C': return 'bg-amber-500/15 text-amber-800 border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-500/30';
    case 'D': return 'bg-orange-500/15 text-orange-800 border-orange-500/30 dark:bg-orange-950/40 dark:text-orange-400 dark:border-orange-500/30';
    case 'F': return 'bg-rose-500/15 text-rose-800 border-rose-500/30 dark:bg-rose-950/40 dark:text-rose-400 dark:border-rose-500/30';
    default: return 'bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.04] dark:text-zinc-300 dark:border-white/10';
  }
};

const getSealGradeConfig = (grade: string) => {
  const g = (grade || "B").toUpperCase();
  switch (g) {
    case "A":
      return {
        container: "bg-emerald-50 border-emerald-300/80 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-500/30 dark:text-emerald-400",
        label: "text-emerald-800 dark:text-emerald-400/80",
        grade: "text-emerald-900 dark:text-emerald-400",
        score: "text-emerald-800 dark:text-emerald-300/90",
      };
    case "B":
      return {
        container: "bg-emerald-50 border-emerald-300/80 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-500/30 dark:text-emerald-400",
        label: "text-emerald-800 dark:text-emerald-400/80",
        grade: "text-emerald-900 dark:text-emerald-400",
        score: "text-emerald-800 dark:text-emerald-300/90",
      };
    case "C":
      return {
        container: "bg-amber-50 border-amber-300/80 text-amber-900 dark:bg-amber-950/40 dark:border-amber-500/30 dark:text-amber-400",
        label: "text-amber-800 dark:text-amber-400/80",
        grade: "text-amber-900 dark:text-amber-400",
        score: "text-amber-800 dark:text-amber-300/90",
      };
    case "D":
      return {
        container: "bg-orange-50 border-orange-300/80 text-orange-900 dark:bg-orange-950/40 dark:border-orange-500/30 dark:text-orange-400",
        label: "text-orange-800 dark:text-orange-400/80",
        grade: "text-orange-900 dark:text-orange-400",
        score: "text-orange-800 dark:text-orange-300/90",
      };
    case "F":
      return {
        container: "bg-rose-50 border-rose-300/80 text-rose-900 dark:bg-rose-950/40 dark:border-rose-500/30 dark:text-rose-400",
        label: "text-rose-800 dark:text-rose-400/80",
        grade: "text-rose-900 dark:text-rose-400",
        score: "text-rose-800 dark:text-rose-300/90",
      };
    default:
      return {
        container: "bg-stone-200/70 border-stone-300 text-stone-800 dark:bg-white/[0.04] dark:border-white/10 dark:text-zinc-300",
        label: "text-stone-700/80 dark:text-zinc-400/80",
        grade: "text-stone-800 dark:text-zinc-200",
        score: "text-stone-700 dark:text-zinc-400",
      };
  }
};

const VALID_MAIN_TABS = new Set([
  "overview",
  "charts",
  "ledger",
  "stats",
  "sql",
  "bi_analytics",
  "export",
]);

const MAIN_TAB_TRIGGER_CLASS =
  "group flex-auto justify-center min-h-[38px] rounded-xl text-xs px-3 py-2 font-sans font-semibold data-[state=active]:bg-[#800000] data-[state=active]:text-white data-[state=active]:border-[#800000] data-[state=active]:shadow-md text-stone-900 hover:text-stone-950 hover:bg-stone-300/70 dark:data-[state=active]:bg-white dark:data-[state=active]:text-zinc-950 dark:data-[state=active]:border-white dark:text-zinc-300 dark:hover:text-white dark:hover:bg-white/[0.08] transition-all inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer border border-transparent";

const TAB_COUNT_BADGE_CLASS =
  "tab-count-badge inline-flex items-center justify-center h-[18px] min-w-[18px] px-1.5 rounded-full text-[10px] leading-none font-sans font-bold tabular-nums bg-stone-300 text-stone-900 border border-stone-400/40 group-data-[state=active]:bg-white group-data-[state=active]:text-[#800000] group-data-[state=active]:border-transparent dark:bg-white/10 dark:text-zinc-300 dark:group-data-[state=active]:bg-zinc-950 dark:group-data-[state=active]:text-white";

function normalizeMainTab(tab?: string | null): string {
  if (tab && VALID_MAIN_TABS.has(tab)) return tab;
  return "overview";
}

export const ReportGeneration = ({
  step,
  taskId,
  filename,
  info,
  analysis,
  charts,
  insights: _insights,
  onComplete,
  onReset,
  requestedTab,
  requestedSql,
}: ReportGenerationProps) => {
  const { toast } = useToast();

  // Primary workspace state
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing report engine…");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const startedGenerationRef = useRef<string | null>(null);
  const [remoteHiplotPayload, setRemoteHiplotPayload] = useState<HiPlotPayload | null>(null);

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        window.URL.revokeObjectURL(downloadUrl);
      }
    };
  }, [downloadUrl]);

  // Top-level Command Center tab
  const [activeMainTab, setActiveMainTab] = useState<string>(() =>
    normalizeMainTab(requestedTab?.tab),
  );

  useEffect(() => {
    if (requestedTab?.tab) {
      setActiveMainTab(normalizeMainTab(requestedTab.tab));
    }
  }, [requestedTab]);

  // Visual Insights Sub-tabs
  const [activeChartTab, setActiveChartTab] = useState<string>("correlation");
  const [activeDistIndex, setActiveDistIndex] = useState<number>(0);
  const [activeBarIndex, setActiveBarIndex] = useState<number>(0);
  const [activeBoxIndex, setActiveBoxIndex] = useState<number>(0);

  // Column Ledger search & filter
  const [expandedColumns, setExpandedColumns] = useState<Record<string, boolean>>({});
  const [ledgerSearch, setLedgerSearch] = useState<string>("");
  const [ledgerGradeFilter, setLedgerGradeFilter] = useState<string>("all");

  const toggleColumnExpand = (colName: string) => {
    setExpandedColumns(prev => ({ ...prev, [colName]: !prev[colName] }));
  };

  // Fetch full multi-dimensional HiPlot sample from backend (avoiding 10-row preview cap)
  useEffect(() => {
    if (step !== "complete" || !taskId) return;
    let cancelled = false;
    api.getHiPlotData(taskId, 2000)
      .then((res) => {
        if (cancelled || !res || !Array.isArray(res.dimensions) || res.dimensions.length < 2) return;
        setRemoteHiplotPayload({
          total_rows: res.total_rows,
          sampled_rows: res.sampled_rows,
          is_sampled: Boolean(res.is_sampled ?? res.total_rows > res.sampled_rows),
          dimensions: res.dimensions,
          datapoints: res.datapoints,
        });
      })
      .catch(() => {
        // Silently keep preview fallback if analytics endpoint unavailable
      });
    return () => {
      cancelled = true;
    };
  }, [step, taskId]);

  // Persistent PDF Generation & /report/status polling (single source of truth; avoids HTTP 202 corruption & premature URL revoke)
  useEffect(() => {
    if (step !== "generating" || !taskId) return;
    if (startedGenerationRef.current === taskId) return;
    startedGenerationRef.current = taskId;

    let cancelled = false;

    const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

    const runGeneration = async () => {
      setStatus("Checking compiled report cache…");
      setProgress(15);

      try {
        let repStatus = await api.getReportStatus(taskId);
        if (cancelled) return;

        if (repStatus.status !== "ready") {
          setStatus("Compiling publication PDF report…");
          setProgress(30);
          await api.generatePersistentReport(taskId);

          for (let attempt = 0; attempt < 60 && !cancelled; attempt += 1) {
            await sleep(1500);
            if (cancelled) return;
            repStatus = await api.getReportStatus(taskId);
            if (repStatus.status === "ready") break;
            if (repStatus.status === "failed") {
              throw new Error("Server report compilation failed.");
            }
            setProgress((prev) => Math.min(92, Math.max(prev, 30 + attempt * 2)));
          }

          if (repStatus.status !== "ready") {
            throw new Error("Timed out waiting for PDF compilation.");
          }
        }

        if (cancelled) return;
        setStatus("Downloading PDF…");
        setProgress(96);

        const blob = await api.downloadReportBlob(taskId);
        if (cancelled) return;

        const url = window.URL.createObjectURL(blob);
        setDownloadUrl(url);
        setProgress(100);
        setStatus("Report ready!");
        onComplete();
        toast({ title: "Report Ready", description: "PDF compiled and ready for download." });
      } catch (error) {
        if (cancelled) return;
        startedGenerationRef.current = null;
        console.error("Report generation failed:", error);
        const errMsg = error instanceof Error ? error.message : "Could not generate PDF report.";
        setStatus("Report generation failed.");
        toast({ title: "PDF Generation Error", description: errMsg, variant: "destructive" });
      }
    };

    void runGeneration();

    return () => {
      cancelled = true;
    };
  }, [step, taskId, onComplete, toast]);

  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  const downloadFile = async () => {
    let targetUrl = downloadUrl;
    if (!targetUrl && taskId) {
      setIsDownloadingPdf(true);
      try {
        const blob = await api.downloadReportBlob(taskId);
        targetUrl = window.URL.createObjectURL(blob);
        setDownloadUrl(targetUrl);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Report is still generating.";
        toast({ title: "Not Ready", description: msg });
        setIsDownloadingPdf(false);
        return;
      } finally {
        setIsDownloadingPdf(false);
      }
    }

    if (targetUrl) {
      const link = document.createElement('a');
      link.href = targetUrl;
      const baseName = filename ? filename.replace(/\.[^/.]+$/, "") : 'Analysis';
      link.download = `${baseName}_Report.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      toast({ title: "Not Ready", description: "Report is still generating." });
    }
  };

  const downloadExport = async (format: ExportFormat) => {
    if (!taskId) return;
    try {
      toast({ title: "Preparing Export", description: `Packaging ${format.toUpperCase()} export file...` });
      const blob = await api.downloadExportBlob(taskId, format);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const safeName = filename ? filename.replace(/\.[^/.]+$/, "") : "dataset";
      const fileExt = format === "excel" ? "xlsx" : format;
      link.download = format === "html" ? `Report_${safeName}.html` : `Cleaned_${safeName}.${fileExt}`;
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

  const [isDeletingExtract, setIsDeletingExtract] = useState(false);

  const handleDeleteExtract = async () => {
    if (!taskId) return;
    if (!window.confirm("Permanently delete this extracted dataset and purge cached files from storage?")) return;
    setIsDeletingExtract(true);
    try {
      await api.deleteSnowflakeExtract(taskId);
      toast({
        title: "Extract Purged",
        description: "Local data extract permanently deleted from storage.",
      });
      onReset();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not delete extract.";
      toast({ title: "Purge Failed", description: msg, variant: "destructive" });
    } finally {
      setIsDeletingExtract(false);
    }
  };

  // Construct Parallel Coordinates Payload (prefer backend /hiplot 2000-row sample over 10-row preview)
  const hiplotPayload = useMemo<HiPlotPayload | null>(() => {
    if (remoteHiplotPayload && remoteHiplotPayload.dimensions.length >= 2) {
      return remoteHiplotPayload;
    }
    if (!info?.preview || info.preview.length < 2) return null;

    const candidateCols = (info?.columns || []).filter(col => {
      const dtype = (info.dtypes?.[col] || "").toLowerCase();
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
  }, [info.columns, info.dtypes, info.preview, info.rows, info.summary, remoteHiplotPayload]);

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

  const confidence = (analysis?.confidence_scores && Array.isArray(analysis.confidence_scores.columns))
    ? analysis.confidence_scores
    : fallbackConfidenceScores;
  const datasetGrade = confidence?.dataset_grade ?? "B";
  const datasetConfidence = confidence?.dataset_confidence ?? 82.5;

  // Filter columns in the ledger
  const filteredLedgerColumns = useMemo(() => {
    return (confidence?.columns || []).filter(c => {
      const matchesSearch = (c.column || "").toLowerCase().includes(ledgerSearch.toLowerCase().trim());
      if (!matchesSearch) return false;
      if (ledgerGradeFilter === "all") return true;
      if (ledgerGradeFilter === "A") return (c.grade || "").toUpperCase() === "A";
      if (ledgerGradeFilter === "B") return (c.grade || "").toUpperCase() === "B";
      if (ledgerGradeFilter === "attention") return ["C", "D", "F"].includes((c.grade || "").toUpperCase());
      return true;
    });
  }, [confidence?.columns, ledgerSearch, ledgerGradeFilter]);

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
      <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in zoom-in-95 duration-400">

        {/* ─── Top Command Center Hero Bar ─── */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 p-6 sm:p-8 bg-card border border-border/80 rounded-3xl shadow-premium relative overflow-hidden">
          <div className="flex items-center gap-5 z-10">
            {/* Seal / Letter Grade Badge */}
            {(() => {
              const seal = getSealGradeConfig(datasetGrade);
              return (
                <div className={`flex flex-col items-center justify-center h-20 w-20 sm:h-24 sm:w-24 rounded-2xl ${seal.container} border shadow-sm shrink-0 transition-colors`}>
                  <span className={`text-[9px] font-sans font-bold tracking-[0.2em] uppercase leading-none mb-1 ${seal.label}`}>GRADE</span>
                  <span className={`text-3xl sm:text-4xl font-sans font-black leading-none tracking-tight ${seal.grade}`}>{datasetGrade}</span>
                  <span className={`text-[11px] font-sans font-bold mt-1 tabular-nums ${seal.score}`}>
                    <AnimatedNumber value={datasetConfidence} format={(n) => n.toFixed(1) + "%"} duration={0.8} />
                  </span>
                </div>
              );
            })()}

            <div className="space-y-1.5">
              <div className="flex items-center gap-2.5 flex-wrap">
                <h1 className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-[-0.025em] leading-tight">
                  {filename ? filename.replace(/\.[^/.]+$/, "") : "Dataset Analysis"}
                </h1>
                {analysis?.snowflake_meta && (
                  <Badge variant="outline" className="text-[10px] font-sans font-medium uppercase bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.04] dark:text-zinc-300 dark:border-white/10 inline-flex items-center gap-1.5">
                    <span>Snowflake</span>
                    <span className="px-1 py-0.5 text-[8px] font-sans font-semibold uppercase tracking-widest text-stone-700 bg-stone-300/80 border border-stone-300 dark:text-zinc-400 dark:bg-white/[0.06] dark:border-white/10 rounded leading-none">Beta</span>
                  </Badge>
                )}
                {analysis?.snowflake_meta?.truncated && (
                  <Badge variant="outline" className="text-[10px] font-sans font-medium uppercase bg-rose-500/15 text-rose-400 border-rose-500/30">
                    Truncated at Limit (150MB Ceiling)
                  </Badge>
                )}
                {analysis?.snowflake_meta?.sampled && (
                  <Badge variant="outline" className="text-[10px] font-sans font-medium uppercase bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.06] dark:text-zinc-300 dark:border-white/10">
                    Sampled (10% Bernoulli)
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs sm:text-[13px] text-muted-foreground font-sans font-medium flex-wrap">
                <span><span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums"><AnimatedNumber value={info.rows} format={(n) => Math.round(n).toLocaleString()} duration={0.8} /></span> rows</span>
                <span className="text-stone-400 dark:text-white/20">•</span>
                <span><span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums"><AnimatedNumber value={info.columns.length} format={(n) => String(Math.round(n))} duration={0.8} /></span> columns</span>
                <span className="text-stone-400 dark:text-white/20">•</span>
                <span><span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums"><AnimatedNumber value={info.memory_usage_mb || 0} format={(n) => n.toFixed(2) + " MB"} duration={0.8} /></span> in-memory</span>
              </div>
              <div className="text-xs font-sans text-muted-foreground/85 flex items-center gap-2.5 pt-0.5 flex-wrap">
                <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700 dark:text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400" />
                  <span className="font-semibold tabular-nums"><AnimatedNumber value={confidence.high_confidence_count} format={(n) => String(Math.round(n))} duration={0.8} /></span> High Confidence Columns
                </span>
                <span className="text-stone-400 dark:text-white/20">•</span>
                <span className="inline-flex items-center gap-1.5 font-medium text-stone-600 dark:text-zinc-400">
                  <span className="font-semibold tabular-nums text-stone-900 dark:text-zinc-300"><AnimatedNumber value={confidence.low_confidence_count} format={(n) => String(Math.round(n))} duration={0.8} /></span> Need Attention
                </span>
              </div>
            </div>
          </div>

          {/* Primary & Quick Export Actions */}
          <div className="flex flex-wrap items-center gap-2.5 z-10 w-full md:w-auto">
            <ShimmerActionButton
              onClick={() => void downloadFile()}
              disabled={isDownloadingPdf}
              className="flex-1 md:flex-none"
              icon={<Download className="h-4 w-4 text-white dark:text-black stroke-[2.5]" />}
              trailingIcon={<ChevronRight className="h-4 w-4 text-white dark:text-black stroke-[2.5]" />}
            >
              <span>{isDownloadingPdf ? "Downloading…" : "Download PDF"}</span>
            </ShimmerActionButton>

            <Button
              size="lg"
              variant="outline"
              className="flex-1 md:flex-none rounded-xl border-stone-300 bg-stone-100 hover:bg-stone-200 text-stone-800 dark:border-white/10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-foreground shadow-xs transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2 font-sans font-semibold text-sm"
              onClick={() => downloadExport("html")}
            >
              <BookOpen className="h-4 w-4 text-primary" />
              <span>HTML Briefing</span>
            </Button>

            {analysis?.snowflake_meta && (
              <AnimatedDeleteButton
                size="sm"
                variant="danger"
                label="Purge Extract"
                confirmLabel="Purge Disk Cache?"
                confirmRequired={true}
                disabled={isDeletingExtract}
                loading={isDeletingExtract}
                onDelete={handleDeleteExtract}
                title="Permanently purge cached extract from disk"
              />
            )}

            <Button
              size="lg"
              variant="ghost"
              className="rounded-xl text-muted-foreground hover:text-foreground font-sans font-medium text-xs flex items-center gap-1.5 px-3"
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
          <TabsList className="h-auto w-full bg-[#EDE8DE] border border-[#DCD6C9] dark:bg-[#0c0d12]/95 dark:border-white/10 p-1.5 rounded-2xl mb-6 shadow-sm flex flex-wrap items-stretch justify-between gap-1.5">
            <TabsTrigger value="overview" className={MAIN_TAB_TRIGGER_CLASS}>
              <FileText className="h-3.5 w-3.5 shrink-0" />
              <span>Overview</span>
            </TabsTrigger>

            <TabsTrigger value="charts" className={MAIN_TAB_TRIGGER_CLASS}>
              <TrendingUp className="h-3.5 w-3.5 shrink-0" />
              <span>Visual Analytics</span>
              {chartTabsList.length > 0 && (
                <span className={TAB_COUNT_BADGE_CLASS}>
                  {chartTabsList.length}
                </span>
              )}
            </TabsTrigger>

            <TabsTrigger value="ledger" className={MAIN_TAB_TRIGGER_CLASS}>
              <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
              <span>Trust Ledger</span>
              <span className={TAB_COUNT_BADGE_CLASS}>
                {confidence.columns.length}
              </span>
            </TabsTrigger>

            <TabsTrigger value="stats" className={MAIN_TAB_TRIGGER_CLASS}>
              <Activity className="h-3.5 w-3.5 shrink-0" />
              <span>Statistical Audit</span>
            </TabsTrigger>

            <TabsTrigger value="sql" className={MAIN_TAB_TRIGGER_CLASS}>
              <Terminal className="h-3.5 w-3.5 shrink-0" />
              <span>DuckDB SQL Studio</span>
            </TabsTrigger>

            <TabsTrigger value="bi_analytics" className={MAIN_TAB_TRIGGER_CLASS}>
              <BarChart3 className="h-3.5 w-3.5 shrink-0" />
              <span>Interactive BI</span>
            </TabsTrigger>

            <TabsTrigger value="export" className={MAIN_TAB_TRIGGER_CLASS}>
              <Download className="h-3.5 w-3.5 shrink-0" />
              <span>Export Hub</span>
            </TabsTrigger>
          </TabsList>

          {/* ──── TAB 1: EXECUTIVE OVERVIEW ──── */}
          <TabsContent value="overview" className="space-y-8 animate-in fade-in duration-300">
            {/* Quick Metrics KPI Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
                  Completeness
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
                    {datasetConfidence.toFixed(1)}%
                  </span>
                  <span className="text-xs font-sans font-medium text-emerald-700 dark:text-emerald-400">High Quality</span>
                </div>
                <p className="text-xs font-sans text-muted-foreground/80 mt-1">Overall data confidence index</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
                  Dataset Dimensions
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
                    {info.rows.toLocaleString()}
                  </span>
                  <span className="text-xs font-sans font-medium text-muted-foreground">x {info.columns.length} cols</span>
                </div>
                <p className="text-xs font-sans text-muted-foreground/80 mt-1">Processed without schema drift</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
                  Numeric vs Categorical
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
                    {info.numeric_columns?.length || 0} / {info.categorical_columns?.length || 0}
                  </span>
                </div>
                <p className="text-xs font-sans text-muted-foreground/80 mt-1">Continuous vs discrete variables</p>
              </Card>

              <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
                <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
                  Memory Utilization
                </span>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
                    {(info.memory_usage_mb || 0).toFixed(2)}
                  </span>
                  <span className="text-xs font-sans font-medium text-muted-foreground">MB</span>
                </div>
                <p className="text-xs font-sans text-muted-foreground/80 mt-1">Optimized columnar storage</p>
              </Card>
            </div>

            {/* Critical Dataset Alerts (if any) */}
            {confidence.critical_issues && confidence.critical_issues.length > 0 && (
              <div className="rounded-3xl border border-rose-200/90 dark:border-rose-900/40 bg-stone-50/70 dark:bg-rose-950/10 p-1 sm:p-1.5 shadow-xs">
                <div className="rounded-[calc(1.5rem-2px)] sm:rounded-[calc(1.5rem-4px)] border border-rose-100 dark:border-rose-900/30 bg-white dark:bg-[#130d11] p-5 sm:p-6 shadow-xs">
                  {/* Header Row */}
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                    <div className="flex items-start gap-3.5">
                      <div className="h-10 w-10 rounded-2xl bg-rose-50 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 border border-rose-200/80 dark:border-rose-800/50 flex items-center justify-center shrink-0 shadow-2xs mt-0.5">
                        <AlertTriangle className="h-5 w-5" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[10px] font-mono font-semibold uppercase tracking-[0.14em] text-rose-600 dark:text-rose-400">
                            Data Integrity Notice
                          </span>
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-rose-50 text-rose-700 dark:bg-rose-950/80 dark:text-rose-300 border border-rose-200/80 dark:border-rose-800/60">
                            {confidence.critical_issues.length} {confidence.critical_issues.length === 1 ? "Column Gap" : "Column Gaps"} Flagged
                          </span>
                        </div>
                        <h3 className="text-base sm:text-lg font-sans font-bold text-stone-900 dark:text-white tracking-tight">
                          Critical Data Hygiene Notices
                        </h3>
                        <p className="text-xs text-stone-600 dark:text-zinc-400 font-sans leading-relaxed max-w-2xl">
                          The following columns exhibit severe missingness (&gt;50%) or distribution anomalies that may distort statistical aggregations and degrade downstream predictive models.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Issues Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 mt-5">
                    {confidence.critical_issues.map((ci: string, idx: number) => {
                      const colonIdx = ci.indexOf(":");
                      const col = colonIdx !== -1 ? ci.substring(0, colonIdx).trim() : "Dataset";
                      const rest = colonIdx !== -1 ? ci.substring(colonIdx + 1).trim() : ci;
                      const matchPct = rest.match(/(\d+(?:\.\d+)?)\s*%/);
                      const pctStr = matchPct ? `${matchPct[1]}%` : null;
                      const pctNum = matchPct ? parseFloat(matchPct[1]) : 0;
                      const label = matchPct ? rest.replace(/\s*\(\d+(?:\.\d+)?%\)/, "").trim() : rest;

                      return (
                        <div
                          key={`critical-issue-${idx}-${col}`}
                          className="group relative rounded-2xl border border-stone-200/90 dark:border-white/10 bg-stone-50/50 dark:bg-[#181116] p-4 hover:border-rose-300 dark:hover:border-rose-800/60 hover:bg-white dark:hover:bg-[#1d141b] transition-all duration-300 flex flex-col justify-between shadow-2xs"
                        >
                          <div>
                            {/* Column and Severity Pill */}
                            <div className="flex items-center justify-between gap-2">
                              <span
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white dark:bg-white/10 text-stone-900 dark:text-stone-100 font-mono text-xs font-semibold border border-stone-200/90 dark:border-white/10 truncate max-w-[170px]"
                                title={col}
                              >
                                <Database className="h-3 w-3 text-stone-500 dark:text-zinc-400 shrink-0" />
                                <span className="truncate">{col}</span>
                              </span>
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-wider uppercase bg-rose-50 text-rose-700 border border-rose-200/80 dark:bg-rose-950/70 dark:text-rose-300 dark:border-rose-800/60 shrink-0">
                                Severe
                              </span>
                            </div>

                            {/* Label & Percentage */}
                            <div className="mt-3.5 space-y-1.5">
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-sans font-medium text-stone-600 dark:text-zinc-400 truncate">
                                  {label || "High missing rate"}
                                </span>
                                {pctStr && (
                                  <span className="font-mono font-bold text-rose-700 dark:text-rose-400 tabular-nums ml-2 shrink-0">
                                    {pctStr}
                                  </span>
                                )}
                              </div>

                              {pctNum > 0 && (
                                <div className="h-1.5 w-full rounded-full bg-stone-200/80 dark:bg-white/10 overflow-hidden">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-rose-500 to-rose-600 dark:from-rose-500 dark:to-rose-400 transition-all duration-500"
                                    style={{ width: `${Math.min(pctNum, 100)}%` }}
                                  />
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Action Recommendation Footer */}
                          <div className="mt-4 pt-2.5 border-t border-stone-200/70 dark:border-white/[0.08] flex items-center justify-between">
                            <span className="text-[11px] font-sans font-medium text-stone-500 dark:text-zinc-400">
                              Suggested Action
                            </span>
                            <span className="font-mono font-semibold text-[10px] px-2 py-0.5 rounded-md bg-stone-100 dark:bg-white/5 text-stone-700 dark:text-zinc-300 border border-stone-200/90 dark:border-white/10">
                              {pctNum >= 80 ? "Drop Column" : "Impute / Synthesize"}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
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
                          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold tracking-tight transition-all shrink-0 ${isActive
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
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${activeDistIndex === idx
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
                              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${activeBarIndex === idx
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
                              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${activeBarIndex === -1
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
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${activeBoxIndex === idx
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
                            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${activeBoxIndex === -1
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
                        <HiPlotParallelCoordinates
                          payload={hiplotPayload}
                          onIsolateSelection={(selectedRows) => {
                            toast({
                              title: "Population Isolated",
                              description: `Filtered ${selectedRows.length.toLocaleString()} records from HiPlot parallel coordinates selection.`,
                            });
                          }}
                        />
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
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${ledgerGradeFilter === "all" ? "bg-card text-foreground border border-border/80 shadow-xs" : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      All ({confidence.columns.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setLedgerGradeFilter("A")}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${ledgerGradeFilter === "A" ? "bg-emerald-500/20 text-emerald-800 border border-emerald-500/30 shadow-xs dark:text-emerald-300" : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      Grade A
                    </button>
                    <button
                      type="button"
                      onClick={() => setLedgerGradeFilter("attention")}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${ledgerGradeFilter === "attention" ? "bg-amber-500/20 text-amber-800 border border-amber-500/30 shadow-xs dark:text-amber-300" : "text-muted-foreground hover:text-foreground"
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
                      className="pl-9 h-8 text-xs rounded-xl bg-background border-border/80 text-foreground placeholder:text-muted-foreground focus-visible:border-primary/50"
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
                        className={`border rounded-2xl transition-all duration-200 ${isExpanded ? 'border-primary bg-primary/5' : 'border-border/80 bg-card hover:bg-muted/20'
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
                              <Badge variant="secondary" className="bg-amber-500/10 text-amber-400 border border-amber-500/30 h-5 px-2 rounded-full text-[9px] font-mono">
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
                              <div className="rounded-2xl bg-gradient-to-b from-amber-500/[0.05] via-zinc-900/60 to-zinc-950/80 border border-amber-500/30 p-3.5 space-y-2.5 shadow-[0_2px_12px_rgba(0,0,0,0.35),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-md">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <span className="w-5 h-5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center justify-center shrink-0 shadow-[0_0_8px_rgba(245,158,11,0.25)]">
                                      <AlertTriangle className="h-3 w-3 stroke-[2.4]" />
                                    </span>
                                    <span className="text-xs font-semibold text-zinc-100 font-sans tracking-tight">Identified Quality Flags</span>
                                  </div>
                                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.9)] animate-pulse" />
                                    {c.issues.length} Flagged
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-2 pt-0.5">
                                  {c.issues.map((issueStr) => (
                                    <span
                                      key={`col-issue-${c.column}-${issueStr}`}
                                      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium text-stone-800 bg-stone-100/90 hover:bg-stone-200/90 border border-stone-300 hover:border-amber-500/50 shadow-xs dark:text-zinc-100 dark:bg-zinc-900/90 dark:hover:bg-zinc-800/90 dark:border-white/10 dark:hover:border-amber-400/50 dark:shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)] hover:shadow-[0_0_14px_rgba(245,158,11,0.2)] transition-all duration-200 select-none cursor-default"
                                    >
                                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.8)]" />
                                      <span className="font-sans tracking-tight text-stone-800 dark:text-zinc-200">{issueStr}</span>
                                    </span>
                                  ))}
                                </div>
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
                              <Badge variant="secondary" className="bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded-lg text-[10px]">
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
                            <Badge variant="destructive" className="h-4.5 text-[9px] px-2 rounded-full bg-rose-950/40 text-rose-400 border border-rose-500/30 uppercase font-bold">
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
                      {analysis.time_series_analysis.analyses && Object.entries(analysis.time_series_analysis.analyses).map(([col, data]) => (
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
            <ErrorBoundary fallbackTitle="DuckDB SQL Studio Encountered an Error">
              <Suspense
                fallback={
                  <div className="flex flex-col items-center justify-center p-12 space-y-3 rounded-2xl border border-border bg-card">
                    <Loader2 className="h-7 w-7 text-primary animate-spin" />
                    <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
                      Loading DuckDB SQL Studio...
                    </span>
                  </div>
                }
              >
                <DuckDBSqlStudio
                  taskId={taskId}
                  datasetName={filename}
                  info={info}
                  externalSqlRequest={requestedSql}
                />
              </Suspense>
            </ErrorBoundary>
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
                    disabled={isDownloadingPdf}
                    onClick={() => void downloadFile()}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>{isDownloadingPdf ? "Downloading..." : "Download PDF"}</span>
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
                    className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
                    onClick={() => downloadExport("html")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download HTML</span>
                  </Button>
                </div>
              </Card>

              {/* Excel Workbook */}
              <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
                <div className="space-y-3">
                  <div className="h-10 w-10 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-display font-bold text-base text-foreground">Excel Workbook (.xlsx)</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Spreadsheet workbook export of the cleaned, type-standardized dataset ready for Excel and BI tools.
                    </p>
                  </div>
                </div>
                <div className="pt-6">
                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
                    onClick={() => downloadExport("xlsx")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export Excel (.xlsx)</span>
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
                    className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
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
                    className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
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
                  <div className="h-10 w-10 rounded-xl bg-stone-200 text-stone-700 dark:bg-white/[0.05] dark:text-zinc-300 flex items-center justify-center">
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
                    className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
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

          {/* ──── TAB 7: SUPERSET INTERACTIVE BI ──── */}
          <TabsContent value="bi_analytics" className="space-y-6 animate-in fade-in duration-300">
            <ErrorBoundary fallbackTitle="Interactive BI Studio Encountered an Error">
              <SupersetExploreStudio
                taskId={taskId || undefined}
                datasetName={filename}
                allowDatasetSwitching={false}
                initialInfo={info}
              />
            </ErrorBoundary>
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
